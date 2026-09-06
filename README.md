# Sunder

A native macOS app that separates a song into vocals and instrumental,
using the [becruily/mel-band-roformer-deux](https://huggingface.co/becruily/mel-band-roformer-deux)
model (CC-BY-NC-4.0) converted to CoreML.

- `Sunder/` -- the Xcode app (SwiftUI). No Python or network dependency at
  runtime: STFT/ISTFT run in Swift via Accelerate/vDSP, and the separation
  network runs as a bundled CoreML model.
- `tools/convert/` -- the one-time PyTorch -> CoreML conversion pipeline
  that produces `Sunder/Sunder/VocalsInstrumental.mlpackage`. Not needed to
  run the app, only to reproduce or update that model. See
  `tools/convert/README.md`.

## Running

Open `Sunder/Sunder.xcodeproj` in Xcode and run. Drop an audio file onto
the window (or click to choose one), click Separate, then choose a folder
to save the two separated tracks into.

Output quality (Sunder → Settings…, ⌘,) is a slider from **Web** (AAC 128
kbps, the default -- small files) up to **Lossless** (16-bit WAV,
uncompressed). Files are named `<name>_Vocals.m4a`/`<name>_Instrumental.m4a`
(or `.wav` at Lossless).

## License note

The bundled model is derived from a checkpoint licensed CC-BY-NC-4.0
(non-commercial). This app is for personal/non-commercial use accordingly.
