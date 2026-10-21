"""Writes a `.sundervoice` file: a zip of the three files Sunder's
`VoiceLibrary.importVoice` expects, in the exact formats VoiceKit loads:

- `G.safetensors`   -- inference generator (no posterior encoder `enc_q.*`),
                       float16, MLX channels-last layout.
- `index.safetensors` -- `centroids` [K <= 10000, 768] float16: k-means of the
                       training ContentVec frames (VoiceKit's VoiceIndex).
- `profile.json`    -- VoiceKit's `VoiceProfile`, as Swift's JSONEncoder
                       writes it (dates as seconds since 2001-01-01).
"""

import json
import re
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

MAX_CENTROIDS = 10000
APPLE_EPOCH = 978307200  # 2001-01-01T00:00:00Z, Foundation's reference date


def to_mlx(state: dict) -> dict:
    """PyTorch -> MLX layout, as tools/convert/export_rvc_base.py does for the
    base weights the app starts from:
    Conv1d [out, in, k] -> [out, k, in]; ConvT1d [in, out, k] -> [out, k, in];
    Conv2d [out, in, kh, kw] -> [out, kh, kw, in]; weight_norm g reshaped to
    broadcast against v; flow.flows.{2i} -> flow.flows.{i} (odd ones are Flips)."""
    out = {}
    for key, t in state.items():
        t = t.detach().float().cpu().numpy()
        transposed = key.startswith("dec.ups.")
        key = re.sub(r"^flow\.flows\.(\d+)\.", lambda m: f"flow.flows.{int(m[1]) // 2}.", key)
        if key.endswith(".weight_v") or (key.endswith(".weight") and t.ndim >= 3):
            if t.ndim == 3:
                t = t.transpose(1, 2, 0) if transposed else t.transpose(0, 2, 1)
            elif t.ndim == 4:
                t = t.transpose(0, 2, 3, 1)
        elif key.endswith(".weight_g"):
            if t.ndim == 3:
                t = t.reshape(1, 1, -1) if transposed else t.reshape(-1, 1, 1)
            elif t.ndim == 4:
                t = t.reshape(-1, 1, 1, 1)
        out[key] = np.ascontiguousarray(t)
    return out


def build_index(frames: np.ndarray, device: torch.device, iterations: int = 12) -> np.ndarray:
    """Lloyd's k-means down to 10k centroids (RVC also k-means-reduces large
    feature sets before indexing); smaller sets are used as-is."""
    x = torch.from_numpy(frames).float().to(device)
    n = x.shape[0]
    if n <= MAX_CENTROIDS:
        return frames.astype(np.float16)
    centroids = x[torch.randperm(n, device=device)[:MAX_CENTROIDS]].clone()
    x_norm = (x * x).sum(1, keepdim=True)
    for _ in range(iterations):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(MAX_CENTROIDS, 1, device=device)
        c_norm = (centroids * centroids).sum(1)[None]
        for start in range(0, n, 8192):
            chunk = x[start : start + 8192]
            nearest = (x_norm[start : start + 8192] - 2 * chunk @ centroids.T + c_norm).argmin(1)
            sums.index_add_(0, nearest, chunk)
            counts.index_add_(0, nearest, torch.ones(len(chunk), 1, device=device))
        centroids = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)
    return centroids.cpu().numpy().astype(np.float16)


def write(destination: Path, name: str, generator_state: dict, centroids: np.ndarray,
          training_seconds: float, epochs: int):
    stage = destination.with_suffix(".staging")
    stage.mkdir(parents=True, exist_ok=True)
    weights = {k: v.astype(np.float16) for k, v in to_mlx(generator_state).items() if not k.startswith("enc_q.")}
    save_file(weights, str(stage / "G.safetensors"))
    save_file({"centroids": centroids.astype(np.float16)}, str(stage / "index.safetensors"))
    profile = {
        "id": str(uuid.uuid4()).upper(),  # replaced by a fresh id on import
        "name": name,
        "createdAt": time.time() - APPLE_EPOCH,
        "trainingSeconds": training_seconds,
        "targetEpochs": epochs,
        "completedEpochs": epochs,
        "isTrained": True,
    }
    (stage / "profile.json").write_text(json.dumps(profile))

    tmp = destination.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in ("profile.json", "G.safetensors", "index.safetensors"):
            zf.write(stage / file, f"Voice/{file}")
    tmp.replace(destination)
    for file in stage.iterdir():
        file.unlink()
    stage.rmdir()
