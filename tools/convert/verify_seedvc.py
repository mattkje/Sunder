"""End-to-end Seed-VC conversion on the CoreML models, in Python -- the
reference VoiceKit's SeedVCConverter.swift follows step for step. Listen to
the result against upstream Seed-VC's to check a build.

Usage: PYTHONPATH=models .venv/bin/python verify_seedvc.py <shared_dir> <voice_dir> <source.wav> <out.wav> [steps]
  shared_dir: SeedVCWhisper/SeedVCVocoder.mlpackage; voice_dir: an unzipped .sundervoice's Voice/"""
import sys, time, numpy as np, torch, librosa, soundfile as sf
import coremltools as ct
from safetensors.numpy import load_file
from seedvc.rmvpe import RMVPE
from seedvc.hf_utils import load_custom_model_from_hf

SR, HOP, MAX_CTX, OVERLAP = 44100, 512, 44100 // 512 * 30, 16

def load(d, name):
    return ct.models.MLModel(f'{d}/SeedVC{name}.mlpackage', compute_units=ct.ComputeUnit.CPU_AND_GPU)

def mel_frames(n):  # Seed-VC's mel: reflect pad (n_fft - hop) / 2 each side, center=False
    return (n + 2048 - HOP - 2048) // HOP + 1

def nearest(a, n):  # F.interpolate(mode='nearest') along axis 0
    return a[np.floor(np.arange(n) * len(a) / n).astype(int)]

def whisper_content(model, audio16k):
    """30 s windows with 5 s of overlap, as Seed-VC's inference does."""
    def run(chunk):
        padded = np.zeros((1, 480000), np.float32); padded[0, :len(chunk)] = chunk
        return model.predict({'audio16k': padded})['content'][0, :len(chunk) // 320 + 1]
    if len(audio16k) <= 480000:
        return run(audio16k)
    out, pos, buf = [], 0, None
    while pos < len(audio16k):
        chunk = audio16k[pos:pos + 480000] if buf is None else np.concatenate([buf, audio16k[pos:pos + 16000 * 25]])
        s = run(chunk)
        out.append(s if buf is None else s[250:])
        buf = chunk[-80000:]
        pos += 480000 if pos == 0 else len(chunk) - 80000
    return np.concatenate(out)

def crossfade(a, b, n):
    fo = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    fi = np.cos(np.linspace(np.pi / 2, 0, n)) ** 2
    b = b.copy(); m = min(n, len(b))
    b[:m] = b[:m] * fi[:m] + (a[-n:] * fo)[:m]
    return b

def convert(shared, voice, audio, steps=30, cfg=0.7, semitones=0, seed=0):
    whisper, voc = load(shared, 'Whisper'), load(shared, 'Vocoder')
    reg, dit = load(voice, 'Regulator'), load(voice, 'DiT')
    ref = load_file(f'{voice}/reference.safetensors')
    rmvpe = RMVPE(load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None), is_half=False, device='cpu')
    style, prompt_mel, prompt_cond = ref['style'][None], ref['prompt_mel'][None], ref['prompt_cond'][None]
    P = prompt_mel.shape[2]

    a16 = librosa.resample(audio, orig_sr=SR, target_sr=16000)
    content = whisper_content(whisper, a16)
    f0 = rmvpe.infer_from_audio(a16, thred=0.03).astype(np.float32)
    if semitones:
        f0 = np.where(f0 > 1, f0 * 2 ** (semitones / 12), f0)
    Tm = mel_frames(len(audio))
    cond = reg.predict({'content': nearest(content, Tm)[None], 'f0': nearest(f0, Tm)[None]})['cond']

    rng = np.random.default_rng(seed)
    window = MAX_CTX - P
    done, chunks, prev = 0, [], None
    ts = np.linspace(0, 1, steps + 1, dtype=np.float32)
    while done < Tm:
        c = np.concatenate([prompt_cond, cond[:, done:done + window]], 1)
        L = c.shape[1]
        x = rng.standard_normal((1, 128, L)).astype(np.float32)
        px = np.zeros_like(x); px[..., :P] = prompt_mel
        x[..., :P] = 0
        for i in range(steps):
            v = dit.predict({'x': x, 'prompt_x': px, 'cond': c, 'style': style,
                             't': ts[i:i + 1], 'cfg': np.array([cfg], np.float32)})['velocity']
            x = x + (ts[i + 1] - ts[i]) * v
            x[..., :P] = 0
        wave = voc.predict({'mel': x[..., P:]})['wave'].reshape(-1)
        last = done + window >= Tm
        ow = OVERLAP * HOP
        if prev is None:
            chunks.append(wave if last else wave[:-ow])
        else:
            chunks.append(crossfade(prev, wave if last else wave[:-ow], ow))
        if last:
            break
        prev = wave[-ow:]
        done += x.shape[2] - P - OVERLAP
    out = np.concatenate(chunks)
    return np.pad(out, (0, max(0, len(audio) - len(out))))[:len(audio)]

if __name__ == '__main__':
    shared, voice, src, dst = sys.argv[1:5]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    audio, _ = librosa.load(src, sr=SR)
    s = time.time()
    out = convert(shared, voice, audio, steps)
    print(f'{len(audio) / SR:.1f}s audio in {time.time() - s:.1f}s')
    sf.write(dst, out, SR)
