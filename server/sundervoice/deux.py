"""becruily's Mel-Band RoFormer "Deux" -- the vocals/instrumental model the Mac app uses by default --
run with the same chunked overlap-add as the app's Demixer (ZFTurbo's `demix()`, generic mode), so the
server's stems match the app's. audio-separator doesn't list Deux, so the vendored model code from
tools/convert/models/bs_roformer is used directly."""

import os

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

from . import assets
from .train import Stopped

REPO = "becruily/mel-band-roformer-deux"
CONFIG, CHECKPOINT = "config_deux_becruily.yaml", "becruily_deux.ckpt"
SAMPLE_RATE = 44100
NUM_OVERLAP = 2  # the config's inference.num_overlap, and what the app uses


class Deux:
    def __init__(self, device: torch.device):
        from models.bs_roformer.mel_band_roformer import MelBandRoformer

        root = assets.MODELS_DIR / "deux"
        config_path = hf_hub_download(REPO, CONFIG, local_dir=root)
        checkpoint_path = hf_hub_download(REPO, CHECKPOINT, local_dir=root)
        # unsafe_load: the config uses a `!!python/tuple` tag; it's the model's own file from its HF repo.
        cfg = yaml.unsafe_load(open(config_path))
        m = cfg["model"]
        self.model = MelBandRoformer(
            dim=m["dim"], depth=m["depth"], stereo=m["stereo"], num_stems=m["num_stems"],
            time_transformer_depth=m["time_transformer_depth"], freq_transformer_depth=m["freq_transformer_depth"],
            num_bands=m["num_bands"], dim_head=m["dim_head"], heads=m["heads"],
            attn_dropout=m["attn_dropout"], ff_dropout=m["ff_dropout"], flash_attn=False,
            dim_freqs_in=m["dim_freqs_in"], sample_rate=m["sample_rate"], stft_n_fft=m["stft_n_fft"],
            stft_hop_length=m["stft_hop_length"], stft_win_length=m["stft_win_length"],
            stft_normalized=m["stft_normalized"], mask_estimator_depth=m["mask_estimator_depth"],
            multi_stft_resolution_loss_weight=m["multi_stft_resolution_loss_weight"],
            multi_stft_resolutions_window_sizes=tuple(m["multi_stft_resolutions_window_sizes"]),
            multi_stft_hop_size=m["multi_stft_hop_size"], multi_stft_normalized=m["multi_stft_normalized"],
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False), strict=True)
        self.model = self.model.float().eval().to(device)
        self.device = device
        self.stems = cfg["training"]["instruments"]  # ["Vocals", "Instrumental"]
        self.chunk = cfg["audio"]["chunk_size"]

    @torch.inference_mode()
    def separate(self, mix: np.ndarray, progress=None, should_abort=None) -> dict[str, np.ndarray]:
        """mix: [2, N] float32 at 44.1 kHz -> {stem: [2, N]}. `progress(fraction)` after each chunk;
        raises `Stopped` between chunks once `should_abort()` turns true (a force stop)."""
        chunk, step = self.chunk, self.chunk // NUM_OVERLAP
        fade, border = chunk // 10, chunk - step
        length = mix.shape[-1]
        x = torch.from_numpy(mix).float()
        uses_border = length > 2 * border and border > 0
        if uses_border:
            x = torch.nn.functional.pad(x[None], (border, border), mode="reflect")[0]
        total = x.shape[-1]

        window = torch.ones(chunk)
        window[:fade] = torch.linspace(0, 1, fade)
        window[-fade:] = torch.linspace(1, 0, fade)
        result = torch.zeros(len(self.stems), 2, total)
        counter = torch.zeros(total)
        i = 0
        while i < total:
            if should_abort and should_abort():
                raise Stopped
            part = x[:, i : i + chunk]
            seg = part.shape[-1]
            if seg < chunk:
                mode = "reflect" if seg > chunk // 2 else "constant"
                part = torch.nn.functional.pad(part[None], (0, chunk - seg), mode=mode)[0]
            out = self.model(part[None].to(self.device))[0].cpu()  # [stems, 2, chunk]
            win = window.clone()
            if i == 0:
                win[:fade] = 1
            if i + step >= total:
                win[-fade:] = 1
            result[..., i : i + seg] += out[..., :seg] * win[:seg]
            counter[i : i + seg] += win[:seg]
            i += step
            if progress:
                progress(min(i / total, 1.0))
        result = result / counter.clamp(min=1e-8)
        if uses_border:
            result = result[..., border : border + length]
        return {name: result[n].numpy() for n, name in enumerate(self.stems)}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    torch.set_num_threads(os.cpu_count() or 1)
    return torch.device("cpu")
