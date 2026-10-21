"""Parity checks for the Voice Changer feature extractors: CoreML vs PyTorch on
real vocal audio, plus a check that decoding the CoreML salience gives the same
f0 as RVC's own RMVPE.infer_from_audio.

Usage: .venv/bin/python verify_rvc_features.py <vocal_audio_file> <out_dir>
"""

import sys
import numpy as np
import torch
import librosa
import coremltools as ct

from build_coreml_rvc_features import ContentVec, RMVPEFromMagnitude, CV_SAMPLES, RMVPE_FRAMES, CKPT
from models.rvc.rmvpe import RMVPE

THRESHOLD = 1e-3
CONTEXT = 200  # frames at 10 ms


def rel_err(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-9)


def main(path, out):
    audio, _ = librosa.load(path, sr=16000, mono=True, offset=30, duration=12)
    audio = audio.astype(np.float32)

    x = audio[:CV_SAMPLES - 12345][None]  # off-default length exercises the flexible shape
    with torch.no_grad():
        ref = ContentVec().eval()(torch.from_numpy(x)).numpy()
    got = ct.models.MLModel(f'{out}/ContentVec.mlpackage').predict({'audio16k': x})['features']
    e = rel_err(got, ref)
    print(f'[contentvec] shape {got.shape} rel max err {e:.3e}')
    assert e < THRESHOLD

    wav = torch.from_numpy(audio)
    spec = torch.stft(wav, n_fft=1024, hop_length=160, win_length=1024,
                      window=torch.hann_window(1024), center=True, return_complex=True)
    mag = spec.abs()[None, :, :RMVPE_FRAMES]
    mag = torch.nn.functional.pad(mag, (0, RMVPE_FRAMES - mag.shape[-1]))
    with torch.no_grad():
        ref_sal = RMVPEFromMagnitude().eval()(mag).numpy()
    got_sal = ct.models.MLModel(f'{out}/RMVPE.mlpackage').predict({'magnitude': mag.numpy()})['salience']
    e = rel_err(got_sal, ref_sal)
    print(f'[rmvpe salience] shape {got_sal.shape} rel max err {e:.3e}')
    assert e < THRESHOLD

    rvc = RMVPE(f'{CKPT}/rmvpe.pt', is_half=False, device='cpu')
    n = min(RMVPE_FRAMES, spec.shape[-1])
    f0_ref = rvc.infer_from_audio(audio[: (n - 1) * 160], thred=0.03)[:n]
    f0_got = rvc.decode(got_sal[0][: len(f0_ref)], thred=0.03)
    # RMVPE's BiGRU sees ~2 s of context both ways, so the last frames differ
    # simply because the two inputs end differently (reference pads, ours has
    # more real audio). Compare the interior; the app overlaps chunks by 2 s.
    f0_ref, f0_got = f0_ref[:-CONTEXT], f0_got[:-CONTEXT]
    voiced = (f0_ref > 0) & (f0_got > 0)
    cents = 1200 * np.abs(np.log2(f0_got[voiced] / f0_ref[voiced]))
    agree = np.mean((f0_ref > 0) == (f0_got > 0))
    print(f'[rmvpe f0] voiced frames {voiced.sum()}, voicing agreement {agree:.4f}, max cents diff {cents.max():.2f}')
    assert agree > 0.99 and cents.max() < 5


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
