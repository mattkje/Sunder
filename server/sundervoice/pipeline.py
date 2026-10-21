"""One voice job end to end: (songs -> lead vocals) -> dataset -> training ->
`.sundervoice`. Every stage leaves its result in the job directory, so a job
interrupted by a restart or a stop picks up where it left off."""

import gc
import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import dataset, export
from .train import Stopped, Trainer, suggested_epochs


@dataclass
class Spec:
    name: str
    inputs: list[Path]
    isolated: bool      # inputs are already clean lead vocals: skip separation
    epochs: int | None  # None = suggested from the amount of audio (Seed-VC: optimizer steps)
    engine: str = "rvc"  # "rvc" or "seedvc"
    reference: Path | None = None  # Seed-VC: a clean clip of the singer (see seedvc.run)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(job_dir: Path, spec: Spec, output: Path, progress, should_stop) -> Path:
    """`progress(stage, fraction, message)`; raises `Stopped` when asked to stop."""
    dev = device()
    sources = spec.inputs
    if not spec.isolated:
        vocals_dir = job_dir / "vocals"
        done = vocals_dir / "done"
        if not done.exists():
            shutil.rmtree(vocals_dir, ignore_errors=True)
            from .separate import isolate_lead_vocals
            isolate_lead_vocals(spec.inputs, vocals_dir, lambda f, m: progress("separating", f, m), should_stop)
            done.touch()
        sources = sorted(vocals_dir.glob("lead_*.wav"))
        _check_stop(should_stop)

    if spec.engine == "seedvc":
        from . import seedvc
        return seedvc.run(job_dir, spec.name, sources, spec.epochs, output, progress, should_stop, spec.reference)

    data_file, meta_file = job_dir / "dataset.pkl", job_dir / "dataset.json"
    if data_file.exists() and meta_file.exists():
        with data_file.open("rb") as f:
            items, frames50 = pickle.load(f)
        meta = json.loads(meta_file.read_text())
    else:
        progress("preparing", 0, "Loading feature models")
        from .features import Features
        features = Features(dev)
        audio = [dataset.load_audio(p) for p in sources]
        items, frames50 = dataset.build(audio, features, lambda f: progress("preparing", f, "Extracting features"))
        del features
        _free()
        if not items:
            raise RuntimeError("The audio didn't contain enough non-silent vocals to train on.")
        seconds = sum(it.frames for it in items) / 100
        meta = {"training_seconds": seconds, "epochs": spec.epochs or suggested_epochs(seconds)}
        tmp = data_file.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump((items, frames50), f)
        tmp.replace(data_file)
        meta_file.write_text(json.dumps(meta))

    epochs = meta["epochs"]
    trainer = Trainer(items, dev)
    checkpoint = job_dir / "checkpoint.pt"
    if checkpoint.exists():
        trainer.load_checkpoint(checkpoint)
    progress("training", trainer.completed_epochs / epochs, f"Epoch {trainer.completed_epochs} of {epochs}")
    trainer.train(epochs, checkpoint, should_stop, lambda epoch, total, step, steps, mel: progress(
        "training", ((epoch - 1) + step / steps) / total, f"Epoch {epoch} of {total} · step {step}/{steps} · mel {mel:.2f}"))

    progress("exporting", 1, "Building index and writing .sundervoice")
    centroids = export.build_index(frames50, dev)
    export.write(output, spec.name, trainer.net_g.state_dict(), centroids, meta["training_seconds"], epochs)
    del trainer
    _free()
    # Training state is large and no longer needed; the lead vocals stay so
    # they can be checked or reused.
    for path in (data_file, checkpoint, job_dir / "uploads"):
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
    return output


def _check_stop(should_stop):
    if should_stop():
        raise Stopped


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
