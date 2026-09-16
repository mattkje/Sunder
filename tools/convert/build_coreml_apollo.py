"""Trace ApolloWrapper and convert it to a CoreML .mlpackage.
Parametrized by checkpoint directory, output path, and T (STFT frame count
for that model's inference chunk_size -- same idea as build_coreml_bs.py).

Usage: .venv/bin/python build_coreml_apollo.py <checkpoint_dir> <output.mlpackage> <T>
"""

import sys
import torch
import coremltools as ct

from load_model_apollo import load_model
from wrapper_apollo import ApolloWrapper

CHANNELS = 2


def main(checkpoint_dir, out_path, T):
    model, cfg = load_model(checkpoint_dir)
    wrapper = ApolloWrapper(model).eval()
    freq = model.enc_dim

    example = torch.randn(1, CHANNELS, freq, T, 2)
    with torch.no_grad():
        eager_out = wrapper(example)
        traced = torch.jit.trace(wrapper, example, check_trace=False)
        traced_out = traced(example)

    diff = (eager_out - traced_out).abs().max().item()
    print(f'[eager vs traced] max abs diff = {diff:.3e}')
    assert diff < 1e-4, 'tracing diverged from eager wrapper output'

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='stft_repr', shape=(1, CHANNELS, freq, T, 2))],
        # Output name kept as 'masked_spectra' to match SeparatorModel.swift's
        # generic reader (shared by every model) -- Apollo doesn't actually
        # produce a mask, it's just the same output-tensor contract.
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
