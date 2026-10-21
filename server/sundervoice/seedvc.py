"""Seed-VC voices: fine-tune the singing model on one voice's lead vocals and
write a `.sundervoice` Sunder converts with on-device (VoiceKit's Seed-VC
engine). The model code is vendored in tools/convert/models/seedvc.

A Seed-VC `.sundervoice` holds, under `Voice/`:
- `profile.json`            -- VoiceKit's `VoiceProfile` with `"engine": "seedVC"`;
                               `targetEpochs`/`completedEpochs` count optimizer steps.
- `SeedVCDiT.mlpackage`     -- the voice's diffusion model (one guided Euler step).
- `SeedVCRegulator.mlpackage` -- its content/pitch conditioner.
- `reference.safetensors`   -- style vector, prompt mel and prompt condition
                               from a 25 s reference clip of the voice.
Sunder compiles the two .mlpackages when it imports the voice.
"""

import json
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .train import Stopped

SAMPLE_RATE = 44100
DEFAULT_STEPS = 1000
CHECKPOINT_INTERVAL = 100  # steps
APPLE_EPOCH = 978307200    # 2001-01-01T00:00:00Z, Foundation's reference date


def slice_clips(sources: list[Path], out_dir: Path) -> float:
    """Cuts the vocals into 2-25 s non-silent clips (Seed-VC's dataset takes
    1-30 s files), peak-normalized. Returns the seconds kept."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    for n, path in enumerate(sources):
        audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        peak = np.abs(audio).max()
        if peak < 1e-4:
            continue
        audio = audio / peak * 0.95
        merged: list[list[int]] = []
        for start, end in librosa.effects.split(audio, top_db=35, frame_length=4096, hop_length=1024):
            if merged and start - merged[-1][1] < 0.6 * SAMPLE_RATE and end - merged[-1][0] <= 20 * SAMPLE_RATE:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        pad = int(0.1 * SAMPLE_RATE)
        for i, (start, end) in enumerate(merged):
            clip = audio[max(0, start - pad):end + pad]
            pieces = []
            while len(clip) > 25 * SAMPLE_RATE:
                pieces.append(clip[:20 * SAMPLE_RATE])
                clip = clip[20 * SAMPLE_RATE:]
            pieces.append(clip)
            for j, piece in enumerate(p for p in pieces if len(p) >= 2 * SAMPLE_RATE):
                sf.write(out_dir / f"{n:03d}_{i:04d}_{j}.wav", piece, SAMPLE_RATE)
                total += len(piece) / SAMPLE_RATE
    return total


def load_clips(clips_dir: Path) -> list[np.ndarray]:
    return [sf.read(p, dtype="float32")[0] for p in sorted(clips_dir.glob("*.wav"))]


def train(job_dir: Path, sources: list[Path], steps: int, progress, should_stop):
    """Returns the fine-tuned model (eval, CPU) and the training clips' seconds.
    Resumes from `job_dir`'s checkpoint; raises `Stopped` after checkpointing."""
    from seedvc.trainer import Trainer, default_device, load_dit

    clips_dir, meta_file = job_dir / "seedvc_clips", job_dir / "seedvc.json"
    if not meta_file.exists():
        shutil.rmtree(clips_dir, ignore_errors=True)
        progress("preparing", 0, "Slicing vocals")
        seconds = slice_clips(sources, clips_dir)
        if seconds < 10:
            raise RuntimeError("The audio didn't contain enough non-silent vocals to train on.")
        meta_file.write_text(json.dumps({"training_seconds": seconds}))
    seconds = json.loads(meta_file.read_text())["training_seconds"]

    # The finished weights, saved before export so a failed export never costs the training.
    final = job_dir / "seedvc_final.pt"
    if final.exists():
        return load_dit(str(final)), seconds

    progress("preparing", 1, "Loading models")
    trainer = Trainer(str(clips_dir), default_device())
    checkpoint = job_dir / "seedvc_checkpoint.pt"
    if checkpoint.exists():
        trainer.load_checkpoint(str(checkpoint))

    def on_step(step, loss):
        progress("training", step / steps, f"Step {step} of {steps} · loss {loss:.3f}")
        if step % CHECKPOINT_INTERVAL == 0 and step < steps:
            trainer.save_checkpoint(str(checkpoint))

    if not trainer.train(steps, on_step, should_stop):
        trainer.save_checkpoint(str(checkpoint))
        raise Stopped
    trainer.save_weights(str(final))
    del trainer
    # Reloaded on the CPU, where export runs (moving the trained model off the
    # GPU would miss the attention's rotary table, a plain tensor attribute).
    return load_dit(str(final)), seconds


def write(destination: Path, name: str, model, reference_clip: np.ndarray, training_seconds: float, steps: int):
    """Writes the `.sundervoice` for `model` (see the module docstring)."""
    from safetensors.numpy import save_file
    from seedvc import export

    stage = destination.with_suffix(".staging")
    shutil.rmtree(stage, ignore_errors=True)
    voice = stage / "Voice"
    voice.mkdir(parents=True)
    export.convert_voice(model, str(voice))
    save_file(export.reference(model, reference_clip), str(voice / "reference.safetensors"))
    profile = {
        "id": str(uuid.uuid4()).upper(),  # replaced by a fresh id on import
        "name": name,
        "createdAt": time.time() - APPLE_EPOCH,
        "trainingSeconds": training_seconds,
        "targetEpochs": steps,
        "completedEpochs": steps,
        "isTrained": True,
        "engine": "seedVC",
    }
    (voice / "profile.json").write_text(json.dumps(profile))

    tmp = destination.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(voice.rglob("*")):
            if file.is_file():
                zf.write(file, str(file.relative_to(stage)))
    tmp.replace(destination)
    shutil.rmtree(stage)


def load_reference(clip: Path) -> np.ndarray:
    """The best 25 s of a clean clip of the singer, as the voice's reference."""
    from seedvc.export import pick_reference
    audio, _ = librosa.load(str(clip), sr=SAMPLE_RATE, mono=True)
    return pick_reference([audio])


def run(job_dir: Path, name: str, sources: list[Path], steps: int | None, output: Path, progress, should_stop,
        reference: Path | None = None) -> Path:
    """`reference`: a clean clip of the singer to use as the voice's reference.
    It decides how much the result sounds like them, far more than training
    does; without one, the most voiced 25 s of the training vocals is used."""
    steps = steps or DEFAULT_STEPS
    model, seconds = train(job_dir, sources, steps, progress, should_stop)
    progress("exporting", 1, "Converting to CoreML and writing .sundervoice")
    from seedvc.export import pick_reference
    clip = load_reference(reference) if reference else pick_reference(load_clips(job_dir / "seedvc_clips"))
    write(output, name, model, clip, seconds, steps)
    for path in (job_dir / "seedvc_checkpoint.pt", job_dir / "seedvc_final.pt", job_dir / "uploads"):
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    return output


def from_clip(name: str, clip: Path, output: Path) -> Path:
    """A zero-shot voice: the pretrained model with `clip` as its reference."""
    from seedvc.trainer import load_dit
    write(output, name, load_dit(None), load_reference(clip), 0, 0)
    return output
