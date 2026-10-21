"""Model weights, downloaded on first use into `<models>/` with the same layout
as tools/convert/checkpoints/ (so a local checkout can point straight at it)."""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS_DIR = Path(os.environ.get("SUNDERVOICE_MODELS", "/data/models"))
RVC_DIR = MODELS_DIR / "rvc"


def rmvpe() -> Path:
    return _file("lj1995/VoiceConversionWebUI", "rmvpe.pt", RVC_DIR)


def base_generator() -> Path:
    return _file("lj1995/VoiceConversionWebUI", "pretrained_v2/f0G40k.pth", RVC_DIR)


def base_discriminator() -> Path:
    return _file("lj1995/VoiceConversionWebUI", "pretrained_v2/f0D40k.pth", RVC_DIR)


def contentvec() -> Path:
    path = RVC_DIR / "contentvec"
    if not (path / "config.json").exists():
        snapshot_download("lengyue233/content-vec-best", local_dir=path, allow_patterns=["config.json", "pytorch_model.bin"])
    return path


def separator_models() -> Path:
    path = MODELS_DIR / "separator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file(repo: str, filename: str, root: Path) -> Path:
    path = root / filename
    if not path.exists():
        hf_hub_download(repo, filename, local_dir=root)
    return path
