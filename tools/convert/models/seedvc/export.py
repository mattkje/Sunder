"""Seed-VC (singing, 44.1 kHz) -> CoreML for Sunder's VoiceKit.

Shared, in the Voice Changer's Seed-VC download (`build_coreml_seedvc.py`):
- SeedVCWhisper: `audio16k` [1, 480000] (zero-padded 30 s) -> `content` [1, 1500, 768].
  Whisper's log-mel front end lives inside the model (the DFT as a strided conv).
- SeedVCVocoder: BigVGAN v2 44k, `mel` [1, 128, T] -> `wave` [1, T * 512].

Per voice (what fine-tuning changes), in its .sundervoice:
- SeedVCRegulator: `content` [1, T, 768] (Whisper frames nearest-resampled to
  the 86.13 Hz mel rate) + `f0` [1, T] Hz (same resampling) -> `cond` [1, T, 768].
  Its GroupNorm spans the whole input, so it runs on the whole song at once.
- SeedVCDiT: one classifier-free-guided Euler velocity: `x`, `prompt_x`
  [1, 128, T], `cond` [1, T, 768], `style` [1, 192], `t` [1], `cfg` [1] -> `velocity`.
- reference.safetensors: `style` [192], `prompt_mel` [128, P], `prompt_cond` [P, 768],
  everything inference needs from the reference clip, precomputed.

Precision: the DiT is float16 (0.1% off float32); the regulator, vocoder and
Whisper's front end lose 5-16% in float16, so the regulator and vocoder stay
float32 and Whisper is float16 as upstream runs it.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from seedvc.hf_utils import load_custom_model_from_hf

SAMPLE_RATE = 44100
HOP = 512
MAX_FRAMES = 2600     # one DiT/vocoder chunk: 30 s of mel frames (2580) + slack
REG_MAX_FRAMES = 60000  # ~11.6 min: the regulator sees the whole song
REFERENCE_SECONDS = 25
WHISPER = "openai/whisper-small"
VOCODER = "nvidia/bigvgan_v2_44khz_128band_512x"


class WhisperEncoder(nn.Module):
    def __init__(self, whisper, mel_filters):
        super().__init__()
        self.encoder = whisper.encoder.float()
        n_fft = 400
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        k = torch.arange(n_fft, dtype=torch.float64)
        f = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        angle = 2 * math.pi * f[:, None] * k[None] / n_fft
        self.register_buffer("dft_re", (torch.cos(angle) * window).float()[:, None, :])
        self.register_buffer("dft_im", (-torch.sin(angle) * window).float()[:, None, :])
        self.register_buffer("mel", torch.from_numpy(mel_filters.T.copy()).float())  # [80, 201]

    def forward(self, audio16k):
        x = F.pad(audio16k[:, None, :], (200, 200), mode="reflect")
        re = F.conv1d(x, self.dft_re, stride=160)
        im = F.conv1d(x, self.dft_im, stride=160)
        power = (re * re + im * im)[:, :, :-1]  # [1, 201, 3000]
        log = torch.log10(torch.clamp(torch.matmul(self.mel, power), min=1e-10))
        log = torch.maximum(log, log.max() - 8.0)
        return self.encoder((log + 4.0) / 4.0).last_hidden_state


F0_MEL_MIN = 1127 * np.log(1 + 50.0 / 700)
F0_MEL_MAX = 1127 * np.log(1 + 1100.0 / 700)


class Regulator(nn.Module):
    """InterpolateRegulator after its nearest-neighbour resample (done by the
    caller; it commutes with content_in_proj)."""

    def __init__(self, length_regulator):
        super().__init__()
        self.lr = length_regulator

    def forward(self, content, f0):
        x = self.lr.content_in_proj(content).transpose(1, 2)
        bins = self.lr.n_f0_bins
        a = (bins - 2) / (F0_MEL_MAX - F0_MEL_MIN)
        b = F0_MEL_MIN * a - 1.0
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
        coarse = torch.round(f0_mel).clamp(1, bins - 1).long()  # f0_to_coarse, then the clamp
        x = x + self.lr.f0_embedding(coarse).transpose(1, 2)
        return self.lr.model(x).transpose(1, 2)


class DiTStep(nn.Module):
    """cfm.estimator with classifier-free guidance folded in; full (unmasked)
    attention, since every frame of a chunk is real."""

    def __init__(self, estimator):
        super().__init__()
        self.dit = estimator

    def velocity(self, x, prompt_x, t, style, cond):
        d = self.dit
        T = x.size(2)
        c = d.t_embedder(t).unsqueeze(1)
        x_in = torch.cat([x.transpose(1, 2), prompt_x.transpose(1, 2), d.cond_projection(cond),
                          style[:, None, :].expand(-1, T, -1)], dim=-1)
        h = d.cond_x_merge_linear(x_in)
        tr = d.transformer
        freqs = tr.freqs_cis[:T]
        skips = []
        for i, layer in enumerate(tr.layers):
            if i in tr.layers_receive_skip:
                h = layer.skip_in_linear(torch.cat([h, skips.pop(-1)], dim=-1))
            h = h + layer.attention(layer.attention_norm(h, c), freqs, None)
            h = h + layer.feed_forward(layer.ffn_norm(h, c))
            if i in tr.layers_emit_skip:
                skips.append(h)
        return d.final_mlp(tr.norm(h, c)).transpose(1, 2)

    def forward(self, x, prompt_x, cond, style, t, cfg):
        v = self.velocity(torch.cat([x, x]), torch.cat([prompt_x, torch.zeros_like(prompt_x)]),
                          torch.cat([t, t]), torch.cat([style, torch.zeros_like(style)]),
                          torch.cat([cond, torch.zeros_like(cond)]))
        return (1 + cfg) * v[:1] - cfg * v[1:]


# MARK: CoreML

def _convert(module, examples, inputs, output, path, fp16):
    import coremltools as ct
    module = module.eval()
    with torch.no_grad():
        eager = module(*examples)
        traced = torch.jit.trace(module, examples, check_trace=False)
        diff = (eager - traced(*examples)).abs().max().item()
    assert diff < 1e-3, f"{path}: trace differs from eager by {diff}"
    ml = ct.convert(
        traced, inputs=inputs, outputs=[ct.TensorType(name=output)], convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
        skip_model_load=True,  # also converts on Linux, where CoreML can't load models
    )
    ml.save(path)


def _frames(default, maximum=MAX_FRAMES):
    import coremltools as ct
    return ct.RangeDim(16, maximum, default=default)


def convert_shared(out_dir):
    """SeedVCWhisper + SeedVCVocoder, the parts every voice shares."""
    import coremltools as ct
    from transformers import AutoFeatureExtractor, WhisperModel
    from seedvc.bigvgan import bigvgan

    whisper = WhisperModel.from_pretrained(WHISPER).eval()
    extractor = AutoFeatureExtractor.from_pretrained(WHISPER)
    _convert(WhisperEncoder(whisper, extractor.mel_filters), (torch.randn(1, 480000) * 0.1,),
             [ct.TensorType(name="audio16k", shape=(1, 480000))], "content", f"{out_dir}/SeedVCWhisper.mlpackage", True)

    vocoder = bigvgan.BigVGAN.from_pretrained(VOCODER, use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    _convert(vocoder.eval(), (torch.randn(1, 128, 200) - 5,),
             [ct.TensorType(name="mel", shape=(1, 128, _frames(200)))], "wave", f"{out_dir}/SeedVCVocoder.mlpackage", False)


def convert_voice(model, out_dir):
    """SeedVCRegulator + SeedVCDiT from a (fine-tuned) `cfm`/`length_regulator` pair."""
    import coremltools as ct
    T = 300
    _convert(Regulator(model.length_regulator), (torch.randn(1, T, 768), torch.rand(1, T) * 400),
             [ct.TensorType(name="content", shape=(1, _frames(T, REG_MAX_FRAMES), 768)),
              ct.TensorType(name="f0", shape=(1, _frames(T, REG_MAX_FRAMES)))],
             "cond", f"{out_dir}/SeedVCRegulator.mlpackage", False)
    examples = (torch.randn(1, 128, T), torch.randn(1, 128, T), torch.randn(1, T, 768),
                torch.randn(1, 192), torch.tensor([0.5]), torch.tensor([0.7]))
    _convert(DiTStep(model.cfm.estimator), examples,
             [ct.TensorType(name="x", shape=(1, 128, _frames(T))),
              ct.TensorType(name="prompt_x", shape=(1, 128, _frames(T))),
              ct.TensorType(name="cond", shape=(1, _frames(T), 768)),
              ct.TensorType(name="style", shape=(1, 192)),
              ct.TensorType(name="t", shape=(1,)), ct.TensorType(name="cfg", shape=(1,))],
             "velocity", f"{out_dir}/SeedVCDiT.mlpackage", True)


# MARK: Reference

def pick_reference(clips44k, seconds=REFERENCE_SECONDS):
    """The `seconds`-long window with the most voiced audio across the voice's
    training clips (concatenated), as the voice's reference."""
    audio = np.concatenate(clips44k)
    n = seconds * SAMPLE_RATE
    if len(audio) <= n:
        return audio
    hop = SAMPLE_RATE // 2
    frames = len(audio) // hop
    rms = np.sqrt(np.mean(audio[:frames * hop].reshape(frames, hop) ** 2, axis=1))
    voiced = (rms > 0.02 * max(np.abs(audio).max(), 1e-6)).astype(float)
    width = n // hop
    start = int(np.argmax(np.convolve(voiced, np.ones(width), "valid"))) * hop
    return audio[start:start + n]


@torch.no_grad()
def reference(model, clip44k, whisper=None, extractor=None):
    """Everything inference needs from the reference clip, through the
    voice's own length regulator."""
    from transformers import AutoFeatureExtractor, WhisperModel
    from seedvc.audio import mel_spectrogram
    from seedvc.campplus.DTDNN import CAMPPlus
    from seedvc.rmvpe import RMVPE

    ref = torch.from_numpy(np.asarray(clip44k[:SAMPLE_RATE * REFERENCE_SECONDS], np.float32))[None]
    ref16 = torchaudio.functional.resample(ref, SAMPLE_RATE, 16000)

    whisper = whisper or WhisperModel.from_pretrained(WHISPER).eval()
    extractor = extractor or AutoFeatureExtractor.from_pretrained(WHISPER)
    feats = extractor([ref16[0].numpy()], return_tensors="pt").input_features
    S = whisper.encoder(feats.to(whisper.device)).last_hidden_state.float().cpu()[:, :ref16.size(-1) // 320 + 1]

    mel = mel_spectrogram(ref, n_fft=2048, num_mels=128, sampling_rate=SAMPLE_RATE, hop_size=HOP,
                          win_size=2048, fmin=0, fmax=None, center=False)

    fbank = torchaudio.compliance.kaldi.fbank(ref16, num_mel_bins=80, dither=0, sample_frequency=16000)
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", None),
                                        map_location="cpu"))
    style = campplus.eval()((fbank - fbank.mean(0, keepdim=True))[None])

    rmvpe = RMVPE(load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None), is_half=False, device="cpu")
    f0 = torch.from_numpy(rmvpe.infer_from_audio(ref16[0], thred=0.03)).float()[None]
    lr = model.length_regulator.cpu()
    cond = lr(S, ylens=torch.LongTensor([mel.size(2)]), n_quantizers=3, f0=f0)[0]
    return {k: np.ascontiguousarray(v.numpy(), np.float32)
            for k, v in {"style": style[0], "prompt_mel": mel[0], "prompt_cond": cond[0]}.items()}
