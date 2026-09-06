# Converting becruily/mel-band-roformer-deux to CoreML

One-time conversion pipeline turning the [becruily/mel-band-roformer-deux](
https://huggingface.co/becruily/mel-band-roformer-deux) PyTorch checkpoint
into the CoreML model the Sunder app bundles. Not needed to run the app --
only to reproduce or update `VocalsInstrumental.mlpackage`.

## Why this isn't a plain `coremltools.convert(torch_model)`

The model's own `forward()` does `torch.stft` -> band-split/transformer
stack -> mask estimate -> complex multiply -> `torch.istft`. `torch.stft`/
`istft` and complex tensors don't convert reliably to CoreML. So:

- `wrapper.py`'s `SpectrogramMasker` reimplements only the STFT-domain
  middle (band-split through mask-averaging and complex multiply, done with
  plain real arithmetic instead of `torch.view_as_complex`), taking an
  already-computed real/imag STFT tensor as input and returning a masked
  real/imag spectrum, with no `torch.stft`/`istft` in the traced graph.
- The Sunder app does the STFT and ISTFT itself, in Swift, via
  Accelerate/vDSP (`Sunder/Sunder/DSP.swift`), matching the model's own STFT
  params exactly (n_fft=2048, hop=441, win=2048, periodic Hann, center=True).

## Files

- `models/bs_roformer/{mel_band_roformer,attend}.py` -- vendored, unmodified
  (besides a header comment) from
  [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
  needed to reconstruct the architecture this checkpoint was trained with.
- `load_model.py` -- builds a `MelBandRoformer` from
  `checkpoint/config_deux_becruily.yaml` and loads `becruily_deux.ckpt` into
  it (a plain, unwrapped `state_dict` -- no EMA/Lightning wrapper to strip).
- `wrapper.py` -- `SpectrogramMasker`, the CoreML-bound STFT-domain-only
  module described above.
- `build_coreml.py` -- traces the wrapper and converts it to
  `VocalsInstrumental.mlpackage`.
- `verify.py` -- numeric parity checks (see below). Run these before
  trusting a (re-)conversion.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p checkpoint
curl -fL -o checkpoint/config_deux_becruily.yaml \
  https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/config_deux_becruily.yaml
curl -fL -o checkpoint/becruily_deux.ckpt \
  https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt
```

`checkpoint/` is gitignored -- the checkpoint is CC-BY-NC-4.0 and isn't
redistributed with this repo.

## Running

```
.venv/bin/python verify.py wrapper          # wrapper math vs original model, in PyTorch only
.venv/bin/python build_coreml.py            # -> VocalsInstrumental.mlpackage (several minutes: see note below)
.venv/bin/python verify.py coreml           # CoreML output vs the PyTorch wrapper
```

Both `verify.py` stages assert a max-abs-diff threshold (1e-3) and print
`PASS`/raise on failure.

## Note on conversion time and `compute_units`

`build_coreml.py` traces the model with `torch.jit.trace`, which is slow for
this graph (~15-20 minutes single-threaded CPU, mostly the depth-12
transformer stack's attention at the full T=1301 sequence length) --
expect it to look idle for a while; it isn't hung as long as `ps` shows
active CPU/thread time.

Separately, `ct.convert(..., compute_units=ct.ComputeUnit.ALL)` (the
default choice, including the ANE target) makes coremltools attempt an
on-device ANE compile as part of its automatic post-conversion model load,
and this graph's `gather`/`scatter_add` ops (from the mel-band overlap
averaging) were observed to make that step take 20+ minutes or effectively
hang. `build_coreml.py` therefore converts with `compute_units=
CPU_AND_GPU` and `skip_model_load=True`, and `verify.py coreml` loads the
saved package with `compute_units=CPU_ONLY` for the same reason. The
Sunder app matches this at runtime (`MLModelConfiguration.computeUnits =
.cpuAndGPU`, in `Sunder/Sunder/SeparatorModel.swift`) -- CPU+GPU is fast
enough for this model size; ANE was never required for acceptable
performance, just for a slow/unreliable conversion path.

`compute_precision=ct.precision.FLOAT32` is also set explicitly: the
default (float16) measured at ~1e-3 relative error against the PyTorch
wrapper (correlation 0.9999977, error concentrated on the largest-magnitude
values -- ordinary fp16 mantissa rounding, not a conversion bug), which is
probably inaudible but not worth risking for audio quality when float32
converts in the same time and this model is small enough that the ~2x
larger package (~950MB vs ~475MB) doesn't matter.

## Copying the result into the app

```
cp -R VocalsInstrumental.mlpackage ../../Sunder/Sunder/
```

Xcode's file-system-synchronized group picks it up automatically and
compiles it into the app bundle at build time -- no project-file edit
needed.
