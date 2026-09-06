"""Loads a plain-BSRoformer checkpoint into a BSRoformer built from its own
config. Parametrized by a checkpoint directory (containing config.yaml and
model.ckpt) so it's reusable across every plain-BS-Roformer model, not just
one -- see checkpoints/<model>/ for each model's files.
"""

import os
import yaml
import torch

from models.bs_roformer.bs_roformer import BSRoformer


def load_config(checkpoint_dir):
    return yaml.unsafe_load(open(os.path.join(checkpoint_dir, 'config.yaml')))


def load_model(checkpoint_dir, flash_attn=False):
    cfg = load_config(checkpoint_dir)
    m = cfg['model']
    model = BSRoformer(
        dim=m['dim'], depth=m['depth'], stereo=m['stereo'], num_stems=m['num_stems'],
        time_transformer_depth=m['time_transformer_depth'], freq_transformer_depth=m['freq_transformer_depth'],
        linear_transformer_depth=m.get('linear_transformer_depth', 0),
        freqs_per_bands=tuple(m['freqs_per_bands']),
        dim_head=m['dim_head'], heads=m['heads'],
        attn_dropout=m['attn_dropout'], ff_dropout=m['ff_dropout'], flash_attn=flash_attn,
        dim_freqs_in=m['dim_freqs_in'], stft_n_fft=m['stft_n_fft'],
        stft_hop_length=m['stft_hop_length'], stft_win_length=m['stft_win_length'],
        stft_normalized=m['stft_normalized'], mask_estimator_depth=m['mask_estimator_depth'],
        multi_stft_resolution_loss_weight=m['multi_stft_resolution_loss_weight'],
        multi_stft_resolutions_window_sizes=tuple(m['multi_stft_resolutions_window_sizes']),
        multi_stft_hop_size=m['multi_stft_hop_size'], multi_stft_normalized=m['multi_stft_normalized'],
        mlp_expansion_factor=m.get('mlp_expansion_factor', 4),
        skip_connection=m.get('skip_connection', False),
    )
    sd = torch.load(os.path.join(checkpoint_dir, 'model.ckpt'), map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model = model.float().eval()
    return model, cfg
