"""
CoreML-bound wrapper for plain BSRoformer (fixed contiguous frequency
bands -- unlike MelBandRoformer's overlapping mel-filterbank bands, so no
gather/scatter-add step is needed here: band order in equals band order
out). Same STFT-domain-only strategy as wrapper.py's SpectrogramMasker --
see that file's docstring for why torch.stft/istft and complex tensors are
kept out of the traced graph.

Input:  stft_repr (1, channels=2, freq=1025, time=T, 2) -- real/imag STFT.
Output: (num_stems, channels=2, freq=1025, time=T, 2) -- masked spectrum
        per stem.

Batch is fixed to 1, matching wrapper.py's SpectrogramMasker.
"""

import torch
from torch import nn
from einops import rearrange, pack, unpack


class SpectrogramMaskerBS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.channels = model.audio_channels
        self.band_split = model.band_split
        self.layers = model.layers
        self.final_norm = model.final_norm
        self.mask_estimators = model.mask_estimators

        freqs_per_band = model.band_split.dim_inputs  # already *2*channels (complex+stereo folded)
        full_freqs = sum(freqs_per_band) // 2  # undo the complex fold to get (f * s)
        dc_mask = torch.ones(full_freqs // self.channels)
        if model.zero_dc:
            dc_mask[0] = 0.
        self.register_buffer('dc_mask', dc_mask)

    def forward(self, stft_repr):
        # stft_repr: (1, s=2, f=1025, t, 2)
        s = stft_repr.shape[1]
        t = stft_repr.shape[3]

        stft_repr = rearrange(stft_repr, '1 s f t c -> (f s) t c')  # (F_full, t, 2)
        x = rearrange(stft_repr, 'f t c -> t (f c)').unsqueeze(0)  # (1, t, F_full*2)

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

        x = self.final_norm(x)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=0)  # (n, 1, t, F_full*2)
        n = masks.shape[0]
        masks = rearrange(masks, 'n 1 t (f c) -> n f t c', c=2)  # (n, F_full, t, 2) -- already in
        # the same frequency order as stft_repr since bands are contiguous and
        # non-overlapping: no gather/scatter-average needed here (contrast wrapper.py).

        sr = stft_repr[..., 0].unsqueeze(0)  # (1, F_full, t)
        si = stft_repr[..., 1].unsqueeze(0)
        mr = masks[..., 0]  # (n, F_full, t)
        mi = masks[..., 1]

        out_r = sr * mr - si * mi
        out_i = sr * mi + si * mr
        out = torch.stack([out_r, out_i], dim=-1)  # (n, F_full, t, 2)

        out = rearrange(out, 'n (f s) t c -> n s f t c', s=s)  # (n, 2, 1025, t, 2)
        out = out * self.dc_mask.view(1, 1, -1, 1, 1)

        return out
