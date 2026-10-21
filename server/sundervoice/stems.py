"""Instrumental + vocals versions of Jellyfin tracks, written next to the original files.

Each song goes through Deux, the vocals/instrumental Roformer the Mac app uses by default (see deux.py). Both results
are encoded to FLAC with the original's tags and cover art, titled "<title> [Instrumental]" / "[Vocals]"
(see jellyfin.STEMS), then Jellyfin is asked to rescan the album so apps can find them."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from . import jellyfin
from .train import Stopped

SAMPLE_RATE = 44100


def run(job_dir: Path, tracks: list[dict], progress, should_stop, should_abort=lambda: False,
        on_track=lambda track_id, state, **info: None, log=print):
    """`tracks` are jellyfin.tracks() entries. Songs whose files already exist are skipped, so a stopped
    job resumes where it left off. Raises `Stopped` between songs when asked to stop, or mid-song (that
    song is redone on resume) when `should_abort()` turns true. `on_track` reports each song's state."""
    todo = []
    for t in tracks:
        if _done(Path(t["path"])):
            on_track(t["id"], "done")
        else:
            todo.append(t)
            on_track(t["id"], "pending")
    log(f"{len(tracks) - len(todo)} of {len(tracks)} songs already have stems; {len(todo)} to separate")
    work = job_dir / "work"
    separator = None
    refreshed = set()
    for i, track in enumerate(todo):
        if should_stop():
            raise Stopped
        name = track["name"]
        progress("separating", i / len(todo), name)
        if separator is None:
            from .deux import Deux, device
            started = time.time()
            separator = Deux(device())
            log(f"Deux loaded on {separator.device} in {time.time() - started:.0f} s")
        original = Path(track["path"])
        started = time.time()
        on_track(track["id"], "running", started=started)
        log(f"[{i + 1}/{len(todo)}] {name}: loading {original.name}")
        try:
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            mix, _ = librosa.load(str(original), sr=SAMPLE_RATE, mono=False)
            if mix.ndim == 1:
                mix = np.stack([mix, mix])
            log(f"[{i + 1}/{len(todo)}] {name}: separating {mix.shape[-1] / SAMPLE_RATE:.0f} s of audio")
            separated = time.time()
            stems = separator.separate(mix[:2], lambda f: progress("separating", (i + f) / len(todo), f"{name} · {f:.0%}"),
                                       should_abort)
            log(f"[{i + 1}/{len(todo)}] {name}: separated in {time.time() - separated:.0f} s, writing FLACs")
            for stem, audio in stems.items():
                wav = work / f"{stem}.wav"
                sf.write(wav, audio.T, SAMPLE_RATE, subtype="PCM_24")
                _encode(wav, original, jellyfin.stem_path(original, stem), f"{name} [{stem}]")
        except Stopped:
            on_track(track["id"], "pending")
            log(f"[{i + 1}/{len(todo)}] {name}: force-stopped mid-song; it is redone on resume")
            raise
        except Exception as error:
            on_track(track["id"], "failed", error=str(error) or type(error).__name__)
            raise
        on_track(track["id"], "done", seconds=round(time.time() - started))
        log(f"[{i + 1}/{len(todo)}] {name}: done in {time.time() - started:.0f} s")
        refreshed.add(track["albumId"])
    shutil.rmtree(work, ignore_errors=True)
    progress("separating", 1.0, "Updating Jellyfin")
    for album_id in refreshed:
        jellyfin.refresh(album_id)
    log(f"Asked Jellyfin to rescan {len(refreshed)} album(s)")


def _done(original: Path) -> bool:
    return all(jellyfin.stem_path(original, stem).exists() for stem in jellyfin.STEMS)


def _encode(wav: Path, original: Path, target: Path, title: str):
    """FLAC with the original's tags and cover art. Written under a hidden name first and moved into
    place, so a library scan never picks up a half-written file."""
    partial = target.with_name(f".{target.name}.partial")
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav), "-i", str(original),
               "-map", "0:a", "-map", "1:v?", "-map_metadata", "1", "-c:a", "flac", "-c:v", "copy",
               "-disposition:v", "attached_pic", "-metadata", f"title={title}", "-f", "flac", str(partial)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {original.name}: {error.stderr.strip()[-300:]}") from error
    # The service runs as root in Docker; give the new file the original's owner so the library stays consistent.
    source = original.stat()
    try:
        os.chown(partial, source.st_uid, source.st_gid)
        os.chmod(partial, source.st_mode & 0o666)
    except OSError:
        pass
    partial.replace(target)
