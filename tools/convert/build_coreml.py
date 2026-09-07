"""Trace SpectrogramMasker and convert it to a CoreML .mlpackage.

T (STFT time-frame count) is a CLI arg, not fixed -- the architecture
(RotaryEmbedding is relative, not fixed-length) supports any T; the Swift
app just needs its ModelSpec.chunkSize/timeFrames to match whatever T this
was converted at. Was hardcoded to 1301 (config_deux_becruily's
chunk_size=573300 @ hop_length=441) until the mobile variant needed a
second, shorter T from the same checkpoint -- generalized to match the
build_coreml_mel.py / build_coreml_bs.py CLI-arg pattern.

Usage: .venv/bin/python build_coreml.py <output.mlpackage> <T>
"""

import sys
import torch
import coremltools as ct

from load_model import load_model
from wrapper import SpectrogramMasker

FREQ = 1025
CHANNELS = 2


def main(out_path, T):
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
        f'Input: (1,2,1025,{T},2) real/imag STFT of one inference chunk '
        '(see the paired AIModel.ModelSpec in the Swift app for its exact '
        'chunk_size/hop_length). '
        f'Output: (2,2,1025,{T},2) masked spectra, stem order [Vocals, Instrumental].'
    )
    mlmodel.save(out_path)
    print('saved', out_path)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
