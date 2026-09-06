"""Trace SpectrogramMaskerBS and convert it to a CoreML .mlpackage.
Parametrized by checkpoint directory, output path, and T (STFT frame count
for that model's inference chunk_size -- see that model's config.yaml
`audio.chunk_size` / `inference.dim_t`).

Usage: .venv/bin/python build_coreml_bs.py <checkpoint_dir> <output.mlpackage> <T>
"""

import sys
import torch
import coremltools as ct

from load_model_bs import load_model
from wrapper_bs import SpectrogramMaskerBS

FREQ = 1025
CHANNELS = 2


def main(checkpoint_dir, out_path, T):
    model, cfg = load_model(checkpoint_dir, flash_attn=False)
    wrapper = SpectrogramMaskerBS(model).eval()

    example = torch.randn(1, CHANNELS, FREQ, T, 2)
    with torch.no_grad():
        eager_out = wrapper(example)  # warm up RotaryEmbedding's internal freq cache first
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        traced_out = traced(example)

    diff = (eager_out - traced_out).abs().max().item()
    print(f'[eager vs traced] max abs diff = {diff:.3e}')
    assert diff < 1e-4, 'tracing diverged from eager wrapper output'

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
    mlmodel.save(out_path)
    print('saved', out_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
