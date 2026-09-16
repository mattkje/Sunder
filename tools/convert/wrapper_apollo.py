"""
CoreML-bound wrapper for Apollo (restoration). Unlike wrapper.py/wrapper_bs.py
(which predict a mask and complex-multiply it against the input spectrum),
Apollo directly *reconstructs* a new spectrum per frequency band -- no mask,
no complex multiply against the input. This wrapper re-implements
Apollo.spec_band_split + feature_extractor + forward()'s per-band loop
(models/apollo/apollo.py), minus torch.stft/istft, kept out of the traced
graph for the same reason as every other wrapper here (see wrapper.py's
docstring): the Sunder app does the STFT/ISTFT itself in Swift
(DSP.swift -- for Apollo's non-power-of-two n_fft=882, via the Bluestein
path added there), matching this exactly.

Input:  stft_repr (1, channels=2, freq=enc_dim, time=T, 2) -- real/imag STFT,
        same convention as every other wrapper.
Output: (num_stems=1, channels=2, freq=enc_dim, time=T, 2) -- Apollo has no
        stem dimension (one restored stream out); the leading dim is kept
        at size 1 to match SeparatorModel.swift's generic (numStems, C, F,
        T, 2) contract, so no Swift changes are needed for this model.

Batch is fixed to 1, matching every other wrapper.
"""

import torch
from torch import nn


class ApolloWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.band_width = model.band_width  # plain python list of 80 ints -- static, trace-safe
        self.nband = model.nband
        self.eps = model.eps
        self.BN = model.BN
        self.net = model.net
        self.output = model.output

    def forward(self, stft_repr):
        # stft_repr: (1, C, F, T, 2)
        C = stft_repr.shape[1]
        T = stft_repr.shape[3]

        real = stft_repr[..., 0].reshape(C, -1, T)  # (C, F, T)
        imag = stft_repr[..., 1].reshape(C, -1, T)

        subband_feature = []
        band_idx = 0
        for i in range(self.nband):
            bw = self.band_width[i]
            r = real[:, band_idx:band_idx + bw]   # (C, bw, T)
            im = imag[:, band_idx:band_idx + bw]
            power = torch.sqrt((r * r + im * im).sum(1, keepdim=True) + self.eps)  # (C, 1, T)
            concat = torch.cat([r / power, im / power, torch.log(power)], dim=1)  # (C, 2*bw+1, T)
            subband_feature.append(self.BN[i](concat))  # (C, feature_dim, T)
            band_idx += bw
        subband_feature = torch.stack(subband_feature, dim=1)  # (C, nband, feature_dim, T)

        feature = self.net(subband_feature)  # BSNet stack, unchanged -- (C, nband, feature_dim, T)

        est_real = []
        est_imag = []
        for i in range(self.nband):
            bw = self.band_width[i]
            this_RI = self.output[i](feature[:, i]).view(C, 2, bw, T)  # (C, 2, bw, T)
            est_real.append(this_RI[:, 0])
            est_imag.append(this_RI[:, 1])
        est_real = torch.cat(est_real, dim=1)  # (C, F, T)
        est_imag = torch.cat(est_imag, dim=1)

        out = torch.stack([est_real, est_imag], dim=-1)  # (C, F, T, 2)
        return out.unsqueeze(0)  # (1, C, F, T, 2) -- numStems=1
