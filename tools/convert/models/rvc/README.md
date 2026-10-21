Vendored from RVC-Project/Retrieval-based-Voice-Conversion-WebUI (MIT, see
LICENSE), `main` branch, September 2026: `infer/rmvpe.py`,
`infer/module/{models,modules,attentions,commons,transforms}.py` and
`train/mel_processing.py`. The only changes are these:
- imports rewritten to be package-relative;
- `rmvpe.py`'s CUDA-graph helper replaced with a direct call.

They're here as the reference that the CoreML conversion
(`build_coreml_rvc_features.py`) and VoiceKit's MLX port are checked against.
