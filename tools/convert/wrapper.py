"""
Wraps a loaded MelBandRoformer to expose only the STFT-domain middle of its
forward pass: band-split -> time/freq transformer stack -> mask estimation ->
mel-band mask averaging -> complex multiply. Skips torch.stft/istft and
torch.view_as_complex/view_as_real (unreliable to convert to CoreML) in favor
of plain real-valued tensors and arithmetic throughout.

Input:  stft_repr (1, channels=2, freq=1025, time=T, 2) — real/imag STFT of
        the input chunk, computed outside this graph (Swift/vDSP mirrors it
        exactly for the app; verify.py mirrors it in Python for parity
        checking).
Output: (num_stems=2, channels=2, freq=1025, time=T, 2) — masked spectrum per
        stem (Vocals, Instrumental per becruily_deux config), ready for
        ISTFT outside this graph.

Batch is fixed to 1: the app always processes one chunk of one file at a
time, so the model's batched advanced-indexing (`stft_repr[batch_arange,
freq_indices]`) is simplified here to a plain `index_select`, which converts
far more reliably.
"""

import torch
from torch import nn
from einops import rearrange, pack, unpack, repeat


class SpectrogramMasker(nn.Module):
    def __init__(self, model):
        super().__init__()
        assert model.stereo and model.num_stems == 2, 'wrapper assumes the becruily_deux config'
        self.channels = model.audio_channels
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimators = model.mask_estimators
        self.register_buffer('freq_indices', model.freq_indices.clone())
        self.register_buffer('num_bands_per_freq', model.num_bands_per_freq.clone())

        freq_bins = model.num_bands_per_freq.shape[0]
        dc_mask = torch.ones(freq_bins)
        dc_mask[0] = 0.  # zero_dc, as in the original model
        self.register_buffer('dc_mask', dc_mask)

    def forward(self, stft_repr):
        # stft_repr: (1, s=2, f=1025, t, 2)
        s = stft_repr.shape[1]
        t = stft_repr.shape[3]

        stft_repr = rearrange(stft_repr, '1 s f t c -> (f s) t c')  # (2050, t, 2)

        x = torch.index_select(stft_repr, 0, self.freq_indices)  # (3958, t, 2)
        x = rearrange(x, 'f t c -> t (f c)').unsqueeze(0)  # (1, t, 7916)

        x = self.band_split(x)  # (1, t, num_bands, dim)

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x = time_transformer(x)
            x, = unpack(x, ps, '* t d')

            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=0)  # (n, 1, t, 3958*2)
        n = masks.shape[0]
        masks = rearrange(masks, 'n 1 t (f c) -> n f t c', c=2)  # (n, 3958, t, 2)

        full_freqs = self.num_bands_per_freq.shape[0] * s  # 2050
        scatter_indices = repeat(self.freq_indices, 'f -> n f t c', n=n, t=t, c=2)
        masks_summed = torch.zeros(n, full_freqs, t, 2, dtype=masks.dtype)
        masks_summed = masks_summed.scatter_add_(1, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1 1', r=s).clamp(min=1e-8)
        masks_averaged = masks_summed / denom  # (n, 2050, t, 2)

        sr = stft_repr[..., 0].unsqueeze(0)  # (1, 2050, t)
        si = stft_repr[..., 1].unsqueeze(0)
        mr = masks_averaged[..., 0]  # (n, 2050, t)
        mi = masks_averaged[..., 1]

        out_r = sr * mr - si * mi
        out_i = sr * mi + si * mr
        out = torch.stack([out_r, out_i], dim=-1)  # (n, 2050, t, 2)

        out = rearrange(out, 'n (f s) t c -> n s f t c', s=s)  # (n, 2, 1025, t, 2)
        out = out * self.dc_mask.view(1, 1, -1, 1, 1)

        return out
