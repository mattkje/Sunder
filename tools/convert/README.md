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

# Voice Changer models (RVC v2)

The Voice Changer (macOS only) trains a voice model on the Mac and converts
vocals with it. It's RVC v2 at 40 kHz with f0 (MIT; reference code is vendored
unmodified apart from import paths in `models/rvc/`, from
[RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)).
It is split by what needs gradients:

- **Frozen feature extractors, in CoreML:** ContentVec (HuBERT-base,
  layer 12, 768-dim) and RMVPE (pitch). Convert them with
  `build_coreml_rvc_features.py`. Both have a flexible input length and stay
  float32: in fp16, ContentVec overflows and RMVPE's voicing flips on about
  20% of frames.
- **Trainable synthesizer and discriminator, in MLX:** these live in
  `Sunder/VoiceKit`, a Swift package with an MLX port of
  `SynthesizerTrnMs768NSFsid` plus `MultiPeriodDiscriminatorV2`. The
  pretrained `f0G40k`/`f0D40k` weights are exported to MLX-layout
  safetensors by `export_rvc_base.py`.

## Setup

Weights come from `lj1995/VoiceConversionWebUI` (`rmvpe.pt`,
`pretrained_v2/f0G40k.pth`, `pretrained_v2/f0D40k.pth`) and
`lengyue233/content-vec-best`, downloaded into `checkpoints/rvc/`.

## Build and verify

```
.venv/bin/python build_coreml_rvc_features.py build_voice
.venv/bin/python verify_rvc_features.py <solo vocal file> build_voice
.venv/bin/python export_rvc_base.py build_voice --reference
cd ../../Sunder/VoiceKit && xcodebuild -scheme voicekit-cli -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .build/xcode -skipPackagePluginValidation build
.build/xcode/Build/Products/Release/voicekit-cli parity ../../tools/convert/build_voice
```

`verify_rvc_features.py` checks CoreML against PyTorch, and checks the Swift-side
pitch decoding against RVC's `RMVPE.infer_from_audio`. `voicekit-cli parity`
checks every MLX module against PyTorch fixtures: text/posterior encoders,
flow in both directions, the NSF decoder, all 9 discriminators and the mel loss.
The pass threshold is a relative error of 1e-3; the observed error is 1e-5 or better.

`voicekit-cli train`/`convert` run the full pipeline headlessly (see its
`main.swift`).

## Release asset

The app downloads everything as one zip, `VoiceChanger-v1.zip`. It contains
`ContentVec.mlpackage`, `RMVPE.mlpackage`, and the float16
`RVCBaseG/D.safetensors` from `export_rvc_base.py <dir> --half`; VoiceKit
upcasts the weights to float32 on load. It is uploaded next to the separation
models on the `models-v1` release (see
`Sunder/Sunder/VoiceAssetDownloader.swift`).

# Voice Changer models (Seed-VC)

Seed-VC voices are the higher-quality alternative to RVC: Seed-VC's singing
model (`DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2`,
44.1 kHz, f0-conditioned), from [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc).
Its code is vendored in `models/seedvc/` (GPL-3.0, see its `LICENSE`; the
model weights on Hugging Face are GPL-3.0 too). Changes from upstream: import
paths, a lean `trainer.py`, float32 fixes for Apple's GPU, and `export.py`.

Everything runs in CoreML in the app (`SeedVCConverter.swift`); training
happens on Sunder Server (or `python -m sundervoice` on a Mac), never in the app.

- **Shared, the `SeedVC-v1.zip` download:** `SeedVCWhisper.mlpackage`
  (Whisper-small encoder with its log-mel front end, float16) and
  `SeedVCVocoder.mlpackage` (BigVGAN v2 44 kHz, float32).
- **Per voice, inside its `.sundervoice`:** `SeedVCDiT.mlpackage` (float16),
  `SeedVCRegulator.mlpackage` (float32) and `reference.safetensors`. Sunder
  compiles the packages when it imports the voice.

`models/seedvc/export.py` documents each model's inputs and outputs.
Seed-VC also uses RMVPE from the RVC download.

## Build and verify

```
PYTHONPATH=models .venv/bin/python build_coreml_seedvc.py build_seedvc
cd ../../server && PYTHONPATH=../tools/convert/models:. python -m sundervoice clip <clean vocal clip> --name Test -o /tmp/test.sundervoice
cd /tmp && unzip -o test.sundervoice
cd - && PYTHONPATH=models .venv/bin/python verify_seedvc.py build_seedvc /tmp/Voice <source vocal> /tmp/out.wav
```

`verify_seedvc.py` runs the whole conversion in Python on the CoreML models;
the Swift converter follows it step for step. To check the Swift side, import
the unzipped voice with `voicekit-cli import` and run `voicekit-cli convert`
with the build folder as its last argument.

Precision: in float16 the DiT stays within 0.1% of float32, but the regulator
and BigVGAN drift 10-16%, so they stay float32.

## Release asset

`SeedVC-v1.zip` holds `SeedVCWhisper.mlpackage` and `SeedVCVocoder.mlpackage`
at its root, uploaded to the `models-v1` release (see
`Sunder/Sunder/VoiceAssetDownloader.swift`).
