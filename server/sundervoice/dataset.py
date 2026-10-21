"""RVC's preprocessing (`train/preprocess.py`, `extract_f0`,
`extract_hubert_feature`), matching VoiceKit's VoiceDataset: high-pass,
silence slicing, 3.7 s segments with 0.3 s overlap, peak normalization, then
ContentVec features and RMVPE pitch per segment, aligned at 100 fps."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy import signal

from .features import SAMPLE_RATE as FEATURE_RATE, Features, coarse_pitch

SAMPLE_RATE = 40000
HOP = 400            # samples per frame at 40 kHz (100 fps)
SEGMENT_FRAMES = 32  # the decoder's training slice (12800 samples)
MAX_FRAMES = 900
SEGMENT_SECONDS = 3.7
OVERLAP_SECONDS = 0.3


@dataclass
class Item:
    wave: np.ndarray     # [frames * HOP] float32 at 40 kHz
    content: np.ndarray  # [frames, 768] float16 (50 fps features repeated x2)
    pitch: np.ndarray    # [frames] int64 coarse
    pitchf: np.ndarray   # [frames] float32 Hz

    @property
    def frames(self) -> int:
        return len(self.pitch)


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def usable_seconds(audio40k: np.ndarray) -> float:
    return sum(len(c) for c in Slicer(SAMPLE_RATE).slice(audio40k)) / SAMPLE_RATE


def build(sources: list[np.ndarray], features: Features, progress=lambda fraction: None):
    """Returns (items, content_frames [N, 768] float16 at 50 fps, for the index)."""
    b, a = signal.butter(N=5, Wn=48, btype="high", fs=SAMPLE_RATE)
    segments = []
    for source in sources:
        audio = signal.lfilter(b, a, source).astype(np.float32)
        for chunk in Slicer(SAMPLE_RATE).slice(audio):
            segments += [n for n in map(_normalize, _split(chunk)) if n is not None]

    items, content = [], []
    for i, segment in enumerate(segments):
        wave16 = librosa.resample(segment, orig_sr=SAMPLE_RATE, target_sr=FEATURE_RATE)
        feats = features.content(wave16)
        f0 = features.pitch(wave16)
        frames = min(len(feats) * 2, len(f0), len(segment) // HOP, MAX_FRAMES)
        progress((i + 1) / len(segments))
        if frames < SEGMENT_FRAMES:
            continue
        pitchf = f0[:frames]
        items.append(Item(
            wave=segment[: frames * HOP].copy(),
            content=np.repeat(feats, 2, axis=0)[:frames].astype(np.float16),
            pitch=coarse_pitch(pitchf),
            pitchf=pitchf,
        ))
        content.append(feats.astype(np.float16))
    frames50 = np.concatenate(content) if content else np.zeros((0, 768), np.float16)
    return items, frames50


def _split(chunk: np.ndarray) -> list[np.ndarray]:
    """3.7 s windows advancing by 3.4 s; the remainder becomes the last piece."""
    per = int(SEGMENT_SECONDS * SAMPLE_RATE)
    step = int((SEGMENT_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE)
    tail = int((SEGMENT_SECONDS + OVERLAP_SECONDS) * SAMPLE_RATE)
    pieces, start = [], 0
    while len(chunk) - start > tail:
        pieces.append(chunk[start : start + per])
        start += step
    pieces.append(chunk[start:])
    return pieces


def _normalize(x: np.ndarray):
    """RVC's `norm_write`: 75% peak-normalized to 0.9 + 25% original; clips
    peaking above 2.5 are rejected as broken."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak <= 0 or peak > 2.5:
        return None
    alpha = 0.75
    return (x / peak * (0.9 * alpha) + (1 - alpha) * x).astype(np.float32)


class Slicer:
    """RVC's `slicer2.Slicer` with the preprocess settings: -42 dB threshold,
    1.5 s min clip, 0.4 s min silence, 15 ms hop, 0.5 s silence kept."""

    def __init__(self, sr, threshold_db=-42, min_length_ms=1500, min_interval_ms=400, hop_ms=15, max_sil_kept_ms=500):
        min_interval = sr * min_interval_ms / 1000
        self.threshold = 10 ** (threshold_db / 20)
        self.hop = round(sr * hop_ms / 1000)
        self.window = min(round(min_interval), 4 * self.hop)
        self.min_length = round(sr * min_length_ms / 1000 / self.hop)
        self.min_interval = round(min_interval / self.hop)
        self.max_sil_kept = round(sr * max_sil_kept_ms / 1000 / self.hop)

    def _rms(self, x):
        pad = self.window // 2
        padded = np.pad(x, (pad, pad))
        frames = 1 + (len(padded) - self.window) // self.hop
        view = np.lib.stride_tricks.sliding_window_view(padded, self.window)[:: self.hop][:frames]
        return np.sqrt(np.mean(view.astype(np.float64) ** 2, axis=1))

    def slice(self, samples):
        if len(samples) <= self.min_length:
            return [samples]
        levels = self._rms(samples)

        def argmin(lo, hi):
            lo, hi = max(lo, 0), min(hi, len(levels))
            return lo + int(np.argmin(levels[lo:hi])) if hi > lo else lo

        tags, silence_start, clip_start = [], None, 0
        for i, level in enumerate(levels):
            if level < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            s = silence_start
            leading = s == 0 and i > self.max_sil_kept
            middle = i - s >= self.min_interval and i - clip_start >= self.min_length
            if not leading and not middle:
                silence_start = None
                continue
            if i - s <= self.max_sil_kept:
                pos = argmin(s, i + 1)
                tags.append((0, pos) if s == 0 else (pos, pos))
                clip_start = pos
            elif i - s <= self.max_sil_kept * 2:
                pos = argmin(i - self.max_sil_kept, s + self.max_sil_kept + 1)
                l = argmin(s, s + self.max_sil_kept + 1)
                r = argmin(i - self.max_sil_kept, i + 1)
                if s == 0:
                    tags.append((0, r))
                    clip_start = r
                else:
                    tags.append((min(l, pos), max(r, pos)))
                    clip_start = max(r, pos)
            else:
                l = argmin(s, s + self.max_sil_kept + 1)
                r = argmin(i - self.max_sil_kept, i + 1)
                tags.append((0, r) if s == 0 else (l, r))
                clip_start = r
            silence_start = None
        total = len(levels)
        if silence_start is not None and total - silence_start >= self.min_interval:
            end = min(total, silence_start + self.max_sil_kept)
            tags.append((argmin(silence_start, end + 1), total + 1))
        if not tags:
            return [samples]

        def cut(a, b):
            return samples[min(len(samples), a * self.hop) : min(len(samples), b * self.hop)]

        chunks = []
        if tags[0][0] > 0:
            chunks.append(cut(0, tags[0][0]))
        for (_, end), (start, _) in zip(tags, tags[1:]):
            chunks.append(cut(end, start))
        if tags[-1][1] < total:
            chunks.append(cut(tags[-1][1], total))
        return [c for c in chunks if len(c)]
