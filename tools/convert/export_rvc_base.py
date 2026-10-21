"""Export the RVC v2 40 kHz pretrained generator/discriminator to MLX-layout
safetensors for the Voice Changer's on-device fine-tuning (VoiceKit).

Layout changes (PyTorch -> MLX, channels-last):
- Conv1d weight   [out, in, k]        -> [out, k, in]
- ConvT1d weight  [in, out, k]        -> [out, k, in]
- Conv2d weight   [out, in, kh, kw]   -> [out, kh, kw, in]
- weight_norm g   reshaped to broadcast against the MLX-layout v
- flow.flows.{2i} -> flow.flows.{i} (the odd entries are parameterless Flips)

Also stores `mel_basis` [125, 1025] (librosa slaney, as RVC's training loss
uses) in the generator file so Swift doesn't reimplement librosa's filters.

Usage:
  .venv/bin/python export_rvc_base.py <out_dir>             # float32 weights
  .venv/bin/python export_rvc_base.py <out_dir> --reference # + parity fixtures
  .venv/bin/python export_rvc_base.py <out_dir> --half      # float16, for the app
                                                            # download (VoiceKit upcasts)
"""

import re
import sys
import numpy as np
import torch
from librosa.filters import mel as librosa_mel
from safetensors.numpy import save_file

CKPT = 'checkpoints/rvc/pretrained_v2'

CONFIG = dict(
    spec_channels=1025, segment_size=32, inter_channels=192, hidden_channels=192,
    filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, p_dropout=0,
    resblock='1', resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5]] * 3, upsample_rates=[10, 10, 2, 2],
    upsample_initial_channel=512, upsample_kernel_sizes=[16, 16, 4, 4],
    spk_embed_dim=109, gin_channels=256, sr=40000, is_half=False,
)


def to_mlx(state):
    out = {}
    for key, t in state.items():
        t = t.float().numpy()
        transposed = key.startswith('dec.ups.')
        key = re.sub(r'^flow\.flows\.(\d+)\.', lambda m: f'flow.flows.{int(m[1]) // 2}.', key)
        if key.endswith('.weight_v') or (key.endswith('.weight') and t.ndim >= 3):
            if t.ndim == 3:
                t = t.transpose(1, 2, 0) if transposed else t.transpose(0, 2, 1)
            elif t.ndim == 4:
                t = t.transpose(0, 2, 3, 1)
        elif key.endswith('.weight_g'):
            if t.ndim == 3:
                t = t.reshape(1, 1, -1) if transposed else t.reshape(-1, 1, 1)
            elif t.ndim == 4:
                t = t.reshape(-1, 1, 1, 1)
        out[key] = np.ascontiguousarray(t)
    return out


def load(name):
    return torch.load(f'{CKPT}/{name}.pth', map_location='cpu', weights_only=True)['model']


def export(out_dir, half=False):
    dtype = np.float16 if half else np.float32
    g = to_mlx(load('f0G40k'))
    g['mel_basis'] = librosa_mel(sr=40000, n_fft=2048, n_mels=125, fmin=0.0, fmax=None)
    save_file({k: v.astype(dtype) for k, v in g.items()}, f'{out_dir}/RVCBaseG.safetensors')
    save_file({k: v.astype(dtype) for k, v in to_mlx(load('f0D40k')).items()}, f'{out_dir}/RVCBaseD.safetensors')
    print('saved', out_dir)


def reference(out_dir):
    """Deterministic fixtures for VoiceKit's parity CLI: inputs + PyTorch outputs
    of the pieces that have no randomness (noise terms zeroed)."""
    from models.rvc.models import SynthesizerTrnMs768NSFsid, MultiPeriodDiscriminatorV2
    torch.manual_seed(0)
    net_g = SynthesizerTrnMs768NSFsid(**CONFIG).eval()
    net_g.load_state_dict(load('f0G40k'))
    net_d = MultiPeriodDiscriminatorV2().eval()
    net_d.load_state_dict(load('f0D40k'))

    T = 200
    phone = torch.randn(1, T, 768)
    pitchf = (110 * 2 ** (torch.linspace(0, 12, T) / 12))[None]
    pitchf[:, 150:170] = 0
    f0_mel = 1127 * torch.log(1 + pitchf / 700)
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    coarse = torch.where(f0_mel > 0, (f0_mel - lo) * 254 / (hi - lo) + 1, f0_mel)
    pitch = coarse.clamp(1, 255).round().long()
    lengths = torch.tensor([T])
    sid = torch.tensor([0])
    spec = torch.rand(1, 1025, T)

    with torch.no_grad():
        gvec = net_g.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = net_g.enc_p(phone, pitch, lengths)
        _, m_q, logs_q, _ = net_g.enc_q(spec, lengths, g=gvec)
        z_fwd = net_g.flow(m_q, x_mask, g=gvec)
        z_rev = net_g.flow(m_p, x_mask, g=gvec, reverse=True)
        # Decoder with the sine source's noise removed (randn patched to zeros)
        # so the output is deterministic; frames 130-194 span voiced+unvoiced.
        orig = torch.randn_like
        torch.randn_like = lambda x, **k: torch.zeros_like(x)
        wav = net_g.dec(z_rev[:, :, 130:194], pitchf[:, 130:194], g=gvec)
        torch.randn_like = orig
        y = wav[:, :, :12800]
        d_r, _, fmap_r, _ = net_d(y, y)
        from models.rvc.train.mel_processing import mel_spectrogram_torch
        mel = mel_spectrogram_torch(y.squeeze(1), 2048, 125, 40000, 400, 2048, 0.0, None)

    t = lambda x: np.ascontiguousarray(x.float().numpy())
    fixtures = {
        'phone': t(phone), 'pitch': pitch.int().numpy(), 'pitchf': t(pitchf), 'spec': t(spec.transpose(1, 2)),
        'm_p': t(m_p.transpose(1, 2)), 'logs_p': t(logs_p.transpose(1, 2)),
        'm_q': t(m_q.transpose(1, 2)), 'logs_q': t(logs_q.transpose(1, 2)),
        'z_fwd': t(z_fwd.transpose(1, 2)), 'z_rev': t(z_rev.transpose(1, 2)),
        'wav': t(wav.transpose(1, 2)), 'mel': t(mel.transpose(1, 2)),
    }
    for i, d in enumerate(d_r):
        fixtures[f'd{i}'] = t(d)
        f = fmap_r[i][0]
        fixtures[f'fmap{i}_0'] = t(f.transpose(1, 2) if f.ndim == 3 else f.permute(0, 2, 3, 1))
    save_file(fixtures, f'{out_dir}/rvc_reference.safetensors')
    print('saved reference fixtures')


if __name__ == '__main__':
    export(sys.argv[1], half='--half' in sys.argv)
    if '--reference' in sys.argv:
        reference(sys.argv[1])
