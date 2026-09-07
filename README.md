# Sunder

A native macOS + iOS app that separates a song into vocals and/or
instrumental, entirely on-device, using CoreML source-separation models
(see **Models** below) converted from open research checkpoints.

- `Sunder/` -- the Xcode app (SwiftUI, one project/target for both
  platforms). STFT/ISTFT run in Swift via Accelerate/vDSP; each separation
  model is downloaded on demand (not bundled) and runs as a CoreML model.
- `tools/convert/` -- the PyTorch -> CoreML conversion pipeline that
  produces each model's `.mlpackage`. Not needed to run the app, only to
  reproduce or add a model. See `tools/convert/README.md`.

## Running

Open `Sunder/Sunder.xcodeproj` in Xcode and run. Drop an audio file onto
the window (or click to choose one), click Separate, then choose a folder
to save the two separated tracks into.

Output quality (Sunder → Settings…, ⌘,) is a slider from **Web** (AAC 128
kbps, the default -- small files) up to **Lossless** (16-bit WAV,
uncompressed). Files are named `<name>_Vocals.m4a`/`<name>_Instrumental.m4a`
(or `.wav` at Lossless).

## License

This repository's own code (the Sunder app, the conversion pipeline) is
licensed under the [MIT License](LICENSE). That covers the code only -- the
pre-trained model weights the app downloads at runtime are each a separate,
independently copyrighted work under their own terms, not covered by the
MIT license above:

| Model | Author | License |
|---|---|---|
| Mel-Band RoFormer (Deux) | [becruily](https://huggingface.co/becruily/mel-band-roformer-deux) | CC BY-NC 4.0 (non-commercial) |
| BS-Roformer Resurrection | [unwa](https://huggingface.co/pcunwa) | none published |
| Mel-RoFormer Gabox Fv7 | [Gabox](https://huggingface.co/GaboxR67/MelBandRoformers) | none published |
| Mel-RoFormer unwa v1e+ | [unwa](https://huggingface.co/pcunwa) | none published |

"None published" means exactly that -- no license file or tag exists for
that checkpoint anywhere it's hosted, which is a stricter, more ambiguous
default than an explicit permissive license, not a green light. Treat all
four the same way this app does: personal/non-commercial use, consistent
with the one model that does state its terms.

## Acceptable use

Sunder processes whatever audio file you give it -- that can include audio
you don't hold the rights to. Only use it on audio you own or otherwise
have the rights to process; you're responsible for that input and for how
you use the app's output. This isn't legal advice, just a plain statement
of where that responsibility sits.

## Privacy

Sunder collects nothing. No accounts, no analytics, no tracking. The only
network activity is downloading a model file from this repo's GitHub
Releases, or fetching a URL you explicitly paste in -- nothing about you or
your usage is sent anywhere.
