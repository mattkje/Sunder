"""Convert the two frozen feature extractors the Voice Changer needs to CoreML.

- ContentVec (HuBERT-base, 768-dim, layer-12 output -- what RVC v2 uses):
  input `audio16k` [1, N] (1-32 s at 16 kHz; flexible so features are computed
  at the clip's real length, as RVC does -- zero padding would change them),
  output `features` [1, (N - 400) // 320 + 1, 768].
- RMVPE pitch network: input `magnitude` [1, 513, T] (T a multiple of 32, up to
  4096; linear STFT magnitude, n_fft=1024, hop=160, periodic Hann, center=True
  -- computed in Swift), output `salience` [1, T, 360]. The mel projection + log clamp live inside the
  model; salience -> Hz decoding is done in Swift (RVCPitch.swift).

Both stay float32: in float16, ContentVec's activations overflow (frame
cosine similarity drops below 0) and RMVPE's voicing decisions flip on ~20%
of frames; int8 weight quantization of ContentVec breaks some frames too.

Usage: .venv/bin/python build_coreml_rvc_features.py <out_dir>
"""

import sys
import torch
import torch.nn as nn
import coremltools as ct
from librosa.filters import mel as librosa_mel
from transformers import HubertModel

from models.rvc.rmvpe import E2E

CKPT = 'checkpoints/rvc'
CV_SAMPLES = 160000   # example/trace length
CV_MAX = 512000       # 32 s
RMVPE_FRAMES = 1024   # example/trace length
RMVPE_MAX = 4096


class ContentVec(nn.Module):
    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(f'{CKPT}/contentvec').eval()

    def forward(self, audio16k):
        return self.hubert(audio16k).last_hidden_state


class RMVPEFromMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = E2E(4, 1, (2, 2))
        self.net.load_state_dict(torch.load(f'{CKPT}/rmvpe.pt', map_location='cpu'))
        self.net.eval()
        basis = librosa_mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
        self.register_buffer('mel_basis', torch.from_numpy(basis).float())

    def forward(self, magnitude):
        mel = torch.log(torch.clamp(torch.matmul(self.mel_basis, magnitude), min=1e-5))
        return self.net(mel)


def convert(module, example, in_name, in_shape, out_name, out_path):
    with torch.no_grad():
        eager = module(example)
        traced = torch.jit.trace(module, example, check_trace=False)
        diff = (eager - traced(example)).abs().max().item()
    print(f'[{out_path}] eager vs traced max abs diff = {diff:.3e}')
    assert diff < 1e-4
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=in_name, shape=in_shape)],
        outputs=[ct.TensorType(name=out_name)],
        convert_to='mlprogram',
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        skip_model_load=True,
    )
    ml.save(out_path)
    print('saved', out_path)


if __name__ == '__main__':
    out = sys.argv[1]
    convert(ContentVec().eval(), torch.randn(1, CV_SAMPLES) * 0.1, 'audio16k',
            ct.Shape((1, ct.RangeDim(16000, CV_MAX, default=CV_SAMPLES))), 'features',
            f'{out}/ContentVec.mlpackage')
    convert(RMVPEFromMagnitude().eval(), torch.rand(1, 513, RMVPE_FRAMES), 'magnitude',
            ct.Shape((1, 513, ct.RangeDim(32, RMVPE_MAX, default=RMVPE_FRAMES))), 'salience',
            f'{out}/RMVPE.mlpackage')
