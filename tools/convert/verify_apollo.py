"""Parity checks for the Apollo restoration wrapper (see verify.py/verify_bs.py
for the mask-based equivalents and rationale). Parametrized by checkpoint
directory. Usage:

  .venv/bin/python verify_apollo.py wrapper <checkpoint_dir>
  .venv/bin/python verify_apollo.py coreml <checkpoint_dir> <mlpackage_path> [chunk_size]
"""

import sys
import torch

from load_model_apollo import load_model
from wrapper_apollo import ApolloWrapper

THRESHOLD = 1e-3


def make_test_audio(chunk_size):
    sr = 44100
    torch.manual_seed(0)
    t = torch.arange(chunk_size).float() / sr
    tone = 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 4000 * t)
    noise = 0.05 * torch.randn(chunk_size)
    mono = tone + noise
    stereo = torch.stack([mono, mono * 0.8], dim=0)
    return stereo.unsqueeze(0)  # (1, 2, n)


def reference_stft(model, raw_audio):
    window = torch.hann_window(model.win).to(raw_audio.device)
    b, nch, n = raw_audio.shape
    flat = raw_audio.reshape(b * nch, n)
    spec = torch.stft(flat, n_fft=model.win, hop_length=model.stride, window=window, return_complex=True)
    real = spec.real.reshape(b, nch, spec.shape[-2], spec.shape[-1])
    imag = spec.imag.reshape(b, nch, spec.shape[-2], spec.shape[-1])
    return torch.stack([real, imag], dim=-1)  # (1, nch, F, T, 2)


def _default_chunk_size(cfg):
    audio = cfg.get('audio', {}) if isinstance(cfg, dict) else {}
    return audio.get('chunk_size', int(round(3 * 44100)))


def run_wrapper_stage(checkpoint_dir):
    model, cfg = load_model(checkpoint_dir)
    wrapper = ApolloWrapper(model).eval()

    raw_audio = make_test_audio(_default_chunk_size(cfg))
    stft_repr = reference_stft(model, raw_audio)

    with torch.no_grad():
        out = wrapper(stft_repr)  # (1, C, F, T, 2)

        # Recompute via the original model's own methods (spec_band_split's
        # band-normalize math + feature_extractor + forward's per-band
        # loop), starting from the same stft_repr, as the independent
        # reference -- same role as verify_bs.py's run_wrapper_stage.
        C = stft_repr.shape[1]
        real = stft_repr[..., 0].reshape(C, -1, stft_repr.shape[3])
        imag = stft_repr[..., 1].reshape(C, -1, stft_repr.shape[3])
        spec_complex = torch.complex(real, imag)

        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(model.nband):
            bw = model.band_width[i]
            this_spec = spec_complex[:, band_idx:band_idx + bw]
            power = (this_spec.abs().pow(2).sum(1) + model.eps).sqrt().unsqueeze(1)
            subband_power.append(power)
            subband_spec_norm.append(torch.complex(this_spec.real / power, this_spec.imag / power))
            band_idx += bw
        subband_power_cat = torch.cat(subband_power, 1)

        subband_feature = []
        for i in range(model.nband):
            concat_spec = torch.cat([
                subband_spec_norm[i].real,
                subband_spec_norm[i].imag,
                torch.log(subband_power_cat[:, i].unsqueeze(1)),
            ], 1)
            subband_feature.append(model.BN[i](concat_spec))
        subband_feature = torch.stack(subband_feature, 1)

        feature = model.net(subband_feature)

        est_real = []
        est_imag = []
        band_idx = 0
        for i in range(model.nband):
            bw = model.band_width[i]
            this_RI = model.output[i](feature[:, i]).view(C, 2, bw, -1)
            est_real.append(this_RI[:, 0])
            est_imag.append(this_RI[:, 1])
        expected = torch.stack([torch.cat(est_real, 1), torch.cat(est_imag, 1)], dim=-1).unsqueeze(0)

    diff = (out - expected).abs().max().item()
    print(f'[wrapper vs original-model math] max abs diff = {diff:.3e} (threshold {THRESHOLD:.0e})')
    assert diff < THRESHOLD
    print('PASS')


def run_coreml_stage(checkpoint_dir, mlpackage_path, chunk_size=None):
    import coremltools as ct

    model, cfg = load_model(checkpoint_dir)
    wrapper = ApolloWrapper(model).eval()

    if chunk_size is None:
        chunk_size = _default_chunk_size(cfg)
    raw_audio = make_test_audio(chunk_size)
    stft_repr = reference_stft(model, raw_audio)

    with torch.no_grad():
        expected = wrapper(stft_repr).numpy()

    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    out = mlmodel.predict({'stft_repr': stft_repr.numpy().astype('float32')})
    got = out['masked_spectra']

    diff = abs(got - expected).max()
    print(f'[coreml vs wrapper] max abs diff = {diff:.3e} (threshold {THRESHOLD:.0e})')
    assert diff < THRESHOLD
    print('PASS')


if __name__ == '__main__':
    stage = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    if stage == 'wrapper':
        run_wrapper_stage(checkpoint_dir)
    elif stage == 'coreml':
        chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
        run_coreml_stage(checkpoint_dir, sys.argv[3], chunk_size=chunk_size)
    else:
        raise SystemExit(f'unknown stage {stage!r}')
