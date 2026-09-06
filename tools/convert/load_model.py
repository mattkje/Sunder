"""Loads becruily_deux.ckpt into a MelBandRoformer built from its own config."""

import yaml
import torch

from models.bs_roformer.mel_band_roformer import MelBandRoformer

CONFIG_PATH = 'checkpoint/config_deux_becruily.yaml'
CKPT_PATH = 'checkpoint/becruily_deux.ckpt'


def load_config():
    # unsafe_load: this yaml uses a `!!python/tuple` tag for
    # multi_stft_resolutions_window_sizes; the file is our own download from
    # the model's official HF repo, not untrusted input.
    return yaml.unsafe_load(open(CONFIG_PATH))


def load_model(flash_attn=False):
    cfg = load_config()
    m = cfg['model']
    model = MelBandRoformer(
        dim=m['dim'], depth=m['depth'], stereo=m['stereo'], num_stems=m['num_stems'],
        time_transformer_depth=m['time_transformer_depth'], freq_transformer_depth=m['freq_transformer_depth'],
        num_bands=m['num_bands'], dim_head=m['dim_head'], heads=m['heads'],
        attn_dropout=m['attn_dropout'], ff_dropout=m['ff_dropout'], flash_attn=flash_attn,
        dim_freqs_in=m['dim_freqs_in'], sample_rate=m['sample_rate'], stft_n_fft=m['stft_n_fft'],
        stft_hop_length=m['stft_hop_length'], stft_win_length=m['stft_win_length'],
        stft_normalized=m['stft_normalized'], mask_estimator_depth=m['mask_estimator_depth'],
        multi_stft_resolution_loss_weight=m['multi_stft_resolution_loss_weight'],
        multi_stft_resolutions_window_sizes=tuple(m['multi_stft_resolutions_window_sizes']),
        multi_stft_hop_size=m['multi_stft_hop_size'], multi_stft_normalized=m['multi_stft_normalized'],
    )
    sd = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model = model.float().eval()
    return model, cfg
