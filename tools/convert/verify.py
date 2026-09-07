"""
Parity check: does SpectrogramMasker (the CoreML-bound wrapper, and later the
actual CoreML model) reproduce the original, unmodified MelBandRoformer's
STFT-domain math exactly? Run this before trusting any CoreML output.

Two stages, both gated on a max-abs-diff threshold:
  1. wrapper (still PyTorch) vs original model's internal stft_repr/masks math
     -- catches bugs in the manual real-valued reimplementation.
  2. traced+converted CoreML model vs the same PyTorch wrapper -- catches
     anything coremltools's conversion changed (this is the actual go/no-go
     gate for Part A before any Xcode work is trusted).

Usage:
  .venv/bin/python verify.py wrapper                  # stage 1 only (no ckpt-load-of-coreml needed)
  .venv/bin/python verify.py coreml <path> [seconds]   # stage 2, against a saved .mlpackage;
                                                        # seconds defaults to 13.0 (T=1301's chunk
                                                        # length) -- pass the mobile variant's own
                                                        # chunk length in seconds to verify it, e.g.
                                                        # 6.5 for the T=651 mobile Deux package.
"""

import sys
import torch
from einops import rearrange

from load_model import load_model
from wrapper import SpectrogramMasker

THRESHOLD = 1e-3


def make_test_audio(model, seconds=13.0):
    # 13.0s @ 44100Hz == 573300 samples == the model's fixed chunk_size/T=1301 --
    # the CoreML graph (unlike the eager PyTorch wrapper) has that shape baked in.
    sr = 44100
    n = int(round(seconds * sr))
    torch.manual_seed(0)
    # a couple of sine tones plus noise, stereo -- enough to exercise all bands
    t = torch.arange(n).float() / sr
    tone = 0.3 * torch.sin(2 * torch.pi * 220 * t) + 0.2 * torch.sin(2 * torch.pi * 4000 * t)
    noise = 0.05 * torch.randn(n)
    mono = tone + noise
    stereo = torch.stack([mono, mono * 0.8], dim=0)  # (2, n)
    return stereo.unsqueeze(0)  # (1, 2, n)


def reference_stft_and_masked(model, raw_audio):
    """Recompute, from the ORIGINAL model, both the input stft_repr and the
    expected masked-spectrum output (i.e. everything between torch.stft and
    torch.istft), for comparison against the wrapper."""
    device = raw_audio.device
    stft_window = model.stft_window_fn(device=device)

    b, channels, _ = raw_audio.shape
    flat = rearrange(raw_audio, 'b s t -> (b s) t')
    stft_repr = torch.stft(flat, **model.stft_kwargs, window=stft_window, return_complex=True)
    stft_repr_real = torch.view_as_real(stft_repr)  # ((b s), f, t, 2)
    stft_repr_real = rearrange(stft_repr_real, '(b s) f t c -> b s f t c', b=b)

    with torch.no_grad():
        recon_audio = model(raw_audio)  # (b, n, s, t) -- full model incl. istft, for a sanity listen only

    return stft_repr_real, recon_audio


def run_wrapper_stage():
    model, cfg = load_model(flash_attn=False)
    wrapper = SpectrogramMasker(model).eval()

    raw_audio = make_test_audio(model)
    stft_repr_real, recon_audio = reference_stft_and_masked(model, raw_audio)

    with torch.no_grad():
        out = wrapper(stft_repr_real)  # (n, s, f, t, 2)

    # Independently recompute the same masked spectrum via the ORIGINAL
    # model's own complex-valued code path, using the model's public pieces,
    # so this is a real cross-check and not just "did the wrapper run".
    with torch.no_grad():
        b = 1
        s = model.audio_channels
        stft_repr_full = rearrange(stft_repr_real, 'b s f t c -> b (f s) t c')
        batch_arange = torch.arange(b)[..., None]
        x = stft_repr_full[batch_arange, model.freq_indices]
        x = rearrange(x, 'b f t c -> b t (f c)')
        x = model.band_split(x)
        for time_transformer, freq_transformer in model.layers:
            x = rearrange(x, 'b t f d -> b f t d')
            from einops import pack, unpack
            x, ps = pack([x], '* t d')
            x = time_transformer(x)
            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')
            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')
        masks = torch.stack([fn(x) for fn in model.mask_estimators], dim=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)
        stft_repr_c = rearrange(stft_repr_full, 'b f t c -> b 1 f t c')
        stft_repr_c = torch.view_as_complex(stft_repr_c.contiguous())
        masks_c = torch.view_as_complex(masks.contiguous()).type(stft_repr_c.dtype)
        from einops import repeat
        n = masks_c.shape[1]
        scatter_indices = repeat(model.freq_indices, 'f -> b n f t', b=b, n=n, t=stft_repr_c.shape[-1])
        stft_repr_expanded = repeat(stft_repr_c, 'b 1 ... -> b n ...', n=n)
        masks_summed = torch.zeros_like(stft_repr_expanded).scatter_add_(2, scatter_indices, masks_c)
        denom = repeat(model.num_bands_per_freq, 'f -> (f r) 1', r=s)
        masks_averaged = masks_summed / denom.clamp(min=1e-8)
        expected_c = stft_repr_c * masks_averaged
        expected_c = rearrange(expected_c, 'b n (f s) t -> (b n s) f t', s=s)
        expected_c[:, 0] = 0.
        expected = torch.view_as_real(expected_c)
        expected = rearrange(expected, '(b n s) f t c -> b n s f t c', b=b, s=s)[0]

    diff = (out - expected).abs().max().item()
    print(f'[wrapper vs original-model math] max abs diff = {diff:.3e} (threshold {THRESHOLD:.0e})')
    assert diff < THRESHOLD, 'wrapper does not match original model math'
    print('PASS')


def run_coreml_stage(path, seconds=13.0):
    import coremltools as ct

    model, cfg = load_model(flash_attn=False)
    wrapper = SpectrogramMasker(model).eval()

    raw_audio = make_test_audio(model, seconds=seconds)
    stft_repr_real, _ = reference_stft_and_masked(model, raw_audio)

    with torch.no_grad():
        expected = wrapper(stft_repr_real).numpy()

    # CPU_ONLY: this is a correctness check, not a performance one -- keeps
    # the load path simple and independent of the app's runtime compute_units.
    mlmodel = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    out = mlmodel.predict({'stft_repr': stft_repr_real.numpy().astype('float32')})
    got = out['masked_spectra']

    diff = abs(got - expected).max()
    print(f'[coreml vs wrapper] max abs diff = {diff:.3e} (threshold {THRESHOLD:.0e})')
    assert diff < THRESHOLD, 'CoreML model does not match the PyTorch wrapper'
    print('PASS')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'wrapper'
    if stage == 'wrapper':
        run_wrapper_stage()
    elif stage == 'coreml':
        path = sys.argv[2] if len(sys.argv) > 2 else 'VocalsInstrumental.mlpackage'
        seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 13.0
        run_coreml_stage(path, seconds=seconds)
    else:
        raise SystemExit(f'unknown stage {stage!r}')
