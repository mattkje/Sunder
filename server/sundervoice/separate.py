"""Song -> dry lead vocal, with python-audio-separator's Roformer models:
vocals/instrumental split, then a karaoke model that pulls the lead vocal away
from the backing vocals, then de-reverb (RVC trains best on dry vocals)."""

import gc
import re
from pathlib import Path

import torch

from . import assets
from .train import Stopped

# (model file, stem to keep). Stem labels come from each model's own config:
# the karaoke model's "Vocals" is the lead, its "Instrumental" is everything
# else including backing vocals; the de-reverb model's dry output is "noreverb".
STAGES = [
    ("mel_band_roformer_vocals_becruily.ckpt", "vocals"),
    ("mel_band_roformer_karaoke_becruily.ckpt", "vocals"),
    ("dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "noreverb"),
]


def isolate_lead_vocals(songs: list[Path], out_dir: Path, progress=lambda fraction, message: None,
                        should_stop=lambda: False) -> list[Path]:
    """Runs every song through each stage in turn (one model loaded at a
    time) and returns one lead-vocal WAV per song, in `out_dir`. A stop is
    honored between files -- a single file's separation can't be interrupted."""
    from audio_separator.separator import Separator

    out_dir.mkdir(parents=True, exist_ok=True)
    current = list(songs)
    for stage, (model, keep) in enumerate(STAGES):
        stage_dir = out_dir / f"stage{stage}"
        separator = Separator(model_file_dir=str(assets.separator_models()), output_dir=str(stage_dir), output_format="WAV")
        separator.load_model(model_filename=model)
        kept = []
        for i, path in enumerate(current):
            if should_stop():
                raise Stopped
            progress((stage * len(current) + i) / (len(STAGES) * len(current)), f"{_stage_name(stage)}: {path.name}")
            outputs = [stage_dir / Path(p).name for p in separator.separate(str(path))]
            kept.append(_pick(outputs, keep))
        current = kept
        del separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final = []
    for i, path in enumerate(current):
        target = out_dir / f"lead_{i:03d}.wav"
        path.replace(target)
        final.append(target)
    return final


def _pick(outputs: list[Path], stem: str) -> Path:
    for path in outputs:
        label = re.search(r"_\(([^)]+)\)", path.name)
        if label and label[1].lower() == stem:
            return path
    raise RuntimeError(f"separator produced no '{stem}' stem (got {[p.name for p in outputs]})")


def _stage_name(stage: int) -> str:
    return ["Isolating vocals", "Removing backing vocals", "Removing reverb"][stage]
