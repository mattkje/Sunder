"""The two frozen feature extractors RVC trains on: ContentVec (HuBERT-base,
layer 12, 768-dim at 50 fps) and RMVPE pitch (Hz at 100 fps). Both take 16 kHz
mono. These are the same PyTorch models the app's CoreML versions were
converted from (tools/convert/build_coreml_rvc_features.py), so a voice
trained here sees the same features the app feeds it at conversion time."""

import numpy as np
import torch
from transformers import HubertModel

from rvc.rmvpe import RMVPE

from . import assets

SAMPLE_RATE = 16000


class Features:
    def __init__(self, device: torch.device):
        self.device = device
        self.hubert = HubertModel.from_pretrained(str(assets.contentvec())).eval().to(device)
        # Built on the CPU, then moved: the vendored class only knows how to
        # pick a CUDA device through RVC's own config module, which isn't here.
        self.rmvpe = RMVPE(str(assets.rmvpe()), is_half=False, device="cpu")
        self.rmvpe.device = device
        self.rmvpe.model.to(device)
        self.rmvpe.mel_extractor.to(device)

    @torch.no_grad()
    def content(self, audio16k: np.ndarray) -> np.ndarray:
        """[N] -> [(N - 400) // 320 + 1, 768] float32."""
        x = torch.from_numpy(audio16k).float().to(self.device)[None]
        return self.hubert(x).last_hidden_state[0].float().cpu().numpy()

    def pitch(self, audio16k: np.ndarray) -> np.ndarray:
        """[N] -> [N // 160 + 1] Hz, 0 where unvoiced."""
        return self.rmvpe.infer_from_audio(audio16k.astype(np.float32), thred=0.03).astype(np.float32)


def coarse_pitch(f0: np.ndarray) -> np.ndarray:
    """RVC's coarse pitch: mel-scaled f0 quantized to 1...255 (0 Hz -> 1)."""
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel = np.where(mel > 0, (mel - lo) * 254 / (hi - lo) + 1, mel)
    return np.clip(np.rint(mel), 1, 255).astype(np.int64)
