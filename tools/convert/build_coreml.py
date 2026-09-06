"""Trace SpectrogramMasker and convert it to a CoreML .mlpackage.

T (STFT time-frame count) is fixed at 1301, matching config_deux_becruily's
inference chunk_size=573300 samples at hop_length=441 -- the Swift app always
pads chunks to exactly that length before STFT (see docs/demix reference),
so a static shape is correct and safer to convert than a flexible one.
"""

import torch
import coremltools as ct

from load_model import load_model
from wrapper import SpectrogramMasker

OUT_PATH = 'VocalsInstrumental.mlpackage'
FREQ = 1025
CHANNELS = 2
T = 1301


def main():
    model, cfg = load_model(flash_attn=False)
    wrapper = SpectrogramMasker(model).eval()

    example = torch.randn(1, CHANNELS, FREQ, T, 2)
    with torch.no_grad():
        eager_out = wrapper(example)  # warm up RotaryEmbedding's internal freq cache first
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        traced_out = traced(example)

    diff = (eager_out - traced_out).abs().max().item()
    print(f'[eager vs traced] max abs diff = {diff:.3e}')
    assert diff < 1e-4, 'tracing diverged from eager wrapper output'

    # compute_units=CPU_AND_GPU (not ALL): including the ANE target makes
    # coremltools attempt an on-device ANE compile as part of convert()'s
    # default model-load/validation step for this graph's gather/scatter-add
    # ops, which was observed to take 20+ minutes or effectively hang.
    # skip_model_load=True skips that automatic load entirely -- correctness
    # is checked independently afterwards, on the saved package, in verify.py.
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='stft_repr', shape=(1, CHANNELS, FREQ, T, 2))],
        outputs=[ct.TensorType(name='masked_spectra')],
        convert_to='mlprogram',
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        skip_model_load=True,
    )
    mlmodel.author = 'converted from becruily/mel-band-roformer-deux'
    mlmodel.short_description = (
        'Mel-Band RoFormer (Vocals/Instrumental) STFT-domain masking network. '
        'Input: (1,2,1025,1301,2) real/imag STFT of a 573300-sample @44.1kHz stereo chunk. '
        'Output: (2,2,1025,1301,2) masked spectra, stem order [Vocals, Instrumental].'
    )
    mlmodel.save(OUT_PATH)
    print('saved', OUT_PATH)


if __name__ == '__main__':
    main()
