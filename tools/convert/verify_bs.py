"""Parity checks for the plain-BSRoformer wrapper (see verify.py for the
Mel-Band-Roformer equivalent and the rationale). Parametrized by checkpoint
directory. Usage:

  .venv/bin/python verify_bs.py wrapper <checkpoint_dir>
  .venv/bin/python verify_bs.py coreml <checkpoint_dir> <mlpackage_path>
"""

import sys
import torch
from einops import rearrange, pack, unpack, repeat

from load_model_bs import load_model
from wrapper_bs import SpectrogramMaskerBS

THRESHOLD = 1e-3


def make_test_audio(chunk_size, seconds=None):
    sr = 44100
    n = chunk_size if seconds is None else int(round(seconds * sr))
    torch.manual_seed(0)
    t = torch.arange(n).float() / sr
    tone = 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 4000 * t)
    noise = 0.05 * torch.randn(n)
    mono = tone + noise
    stereo = torch.stack([mono, mono * 0.8], dim=0)
    return stereo.unsqueeze(0)  # (1, 2, n)


def reference_stft(model, raw_audio):
    device = raw_audio.device
    stft_window = model.stft_window_fn(device=device)
    b, channels, _ = raw_audio.shape
    flat = rearrange(raw_audio, 'b s t -> (b s) t')
    stft_repr = torch.stft(flat, **model.stft_kwargs, window=stft_window, return_complex=True)
    stft_repr_real = torch.view_as_real(stft_repr)
    stft_repr_real = rearrange(stft_repr_real, '(b s) f t c -> b s f t c', b=b)
    return stft_repr_real


def run_wrapper_stage(checkpoint_dir):
    model, cfg = load_model(checkpoint_dir, flash_attn=False)
    wrapper = SpectrogramMaskerBS(model).eval()

    chunk_size = cfg['audio']['chunk_size']
    raw_audio = make_test_audio(chunk_size)
    stft_repr_real = reference_stft(model, raw_audio)

    with torch.no_grad():
        out = wrapper(stft_repr_real)  # (n, s, f, t, 2)

    with torch.no_grad():
        s = model.audio_channels
        stft_repr_full = rearrange(stft_repr_real, 'b s f t c -> b (f s) t c')
        x = rearrange(stft_repr_full, 'b f t c -> b t (f c)')
        x = model.band_split(x)
        for time_transformer, freq_transformer in model.layers:
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            x = time_transformer(x)
            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')
        x = model.final_norm(x)
        masks = torch.stack([fn(x) for fn in model.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)
        stft_repr_c = rearrange(stft_repr_full, 'b f t c -> b 1 f t c')
        stft_repr_c = torch.view_as_complex(stft_repr_c.contiguous())
        masks_c = torch.view_as_complex(masks.contiguous())
        expected_c = stft_repr_c * masks_c
        expected_c = rearrange(expected_c, 'b n (f s) t -> (b n s) f t', s=s)
        expected_c[:, 0] = 0.
        expected = torch.view_as_real(expected_c)
        expected = rearrange(expected, '(b n s) f t c -> b n s f t c', b=1, s=s)[0]

    diff = (out - expected).abs().max().item()
    print(f'[wrapper vs original-model math] max abs diff = {diff:.3e} (threshold {THRESHOLD:.0e})')
    assert diff < THRESHOLD
    print('PASS')


def run_coreml_stage(checkpoint_dir, mlpackage_path):
    import coremltools as ct

    model, cfg = load_model(checkpoint_dir, flash_attn=False)
    wrapper = SpectrogramMaskerBS(model).eval()

    chunk_size = cfg['audio']['chunk_size']
    raw_audio = make_test_audio(chunk_size)
    stft_repr_real = reference_stft(model, raw_audio)

    with torch.no_grad():
        expected = wrapper(stft_repr_real).numpy()

    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    out = mlmodel.predict({'stft_repr': stft_repr_real.numpy().astype('float32')})
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
        run_coreml_stage(checkpoint_dir, sys.argv[3])
    else:
        raise SystemExit(f'unknown stage {stage!r}')
