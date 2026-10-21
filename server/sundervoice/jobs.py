"""Persistent FIFO of jobs -- voice training and instrumental generation --
run one at a time on a background thread (separation and training both want
the whole GPU). Each job lives in
`<data>/jobs/<id>/` with its state in `job.json`; jobs that were running when
the service stopped are resumed on the next start. Each job also keeps a
timestamped `log.txt`; `status()` reports what the worker is doing."""

import json
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from . import pipeline, stems
from .train import Stopped

QUEUED, RUNNING, DONE, FAILED, STOPPED = "queued", "running", "done", "failed", "stopped"


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_requested: str | None = None
        self.force_requested: str | None = None  # stop now, even mid-song
        self.current: str | None = None          # the running job
        self.heartbeat = 0.0                     # last progress report from the running job
        self.started = time.time()
        for job in self.list():
            if job["status"] == RUNNING:
                self._update(job["id"], status=QUEUED)

    # MARK: state

    def list(self) -> list[dict]:
        jobs = []
        for path in self.root.glob("*/job.json"):
            try:
                jobs.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def get(self, job_id: str) -> dict | None:
        path = self._dir(job_id) / "job.json"
        return json.loads(path.read_text()) if path.exists() else None

    def create(self, name: str, isolated: bool, epochs: int | None, engine: str = "rvc") -> dict:
        job = {
            "id": uuid.uuid4().hex, "name": name, "isolated": isolated, "epochs": epochs, "engine": engine,
            "status": "new", "stage": None, "progress": 0.0, "message": "", "error": None,
            "created": time.time(),
        }
        self._dir(job["id"]).joinpath("uploads").mkdir(parents=True)
        self._write(job)
        return job

    def create_stems(self, name: str, tracks: "list[dict]") -> dict:  # quoted: `list` is a method in this class
        """A job that writes instrumental/vocals versions of Jellyfin tracks (see stems.py)."""
        job = {
            "id": uuid.uuid4().hex, "kind": "stems", "name": name, "tracks": tracks,
            "status": "new", "stage": None, "progress": 0.0, "message": "", "error": None,
            "created": time.time(),
        }
        self._dir(job["id"]).mkdir(parents=True)
        self._write(job)
        return job

    def reference_path(self, job_id: str, filename: str) -> Path:
        """A Seed-VC job's reference clip; kept outside `uploads/`, which the
        pipeline deletes once training is done."""
        suffix = re.sub(r"[^a-z0-9]", "", Path(filename).suffix.lower())[:5]
        return self._dir(job_id) / f"reference.{suffix or 'audio'}"

    def upload_path(self, job_id: str, index: int, filename: str) -> Path:
        suffix = re.sub(r"[^a-z0-9]", "", Path(filename).suffix.lower())[:5]
        return self._dir(job_id) / "uploads" / f"{index:03d}.{suffix or 'audio'}"

    def enqueue(self, job_id: str):
        self._update(job_id, status=QUEUED, error=None)
        self.wake.set()

    def stop(self, job_id: str, force: bool = False):
        """Stop a running job at its next safe point (stems: after the current song); `force` also aborts
        mid-song (that song is redone on resume). Queued jobs just leave the queue."""
        job = self.get(job_id)
        if job and job["status"] == RUNNING:
            self.stop_requested = job_id
            if force:
                self.force_requested = job_id
            self.log(job_id, "Force stop requested" if force else "Stop requested (finishes the current step first)")
        elif job and job["status"] == QUEUED:
            self._update(job_id, status=STOPPED)
            self.log(job_id, "Removed from the queue")

    def log(self, job_id: str, text: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text}"
        print(f"[job {job_id[:8]}] {text}", flush=True)
        try:
            with (self._dir(job_id) / "log.txt").open("a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def read_log(self, job_id: str, lines: int = 400) -> str:
        path = self._dir(job_id) / "log.txt"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.exists() else ""

    def status(self) -> dict:
        queued = [j for j in self.list() if j["status"] == QUEUED]
        return {"uptime": round(time.time() - self.started), "job": self.current,
                "sinceProgress": round(time.time() - self.heartbeat) if self.current else None,
                "queued": len(queued)}

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job["status"] == RUNNING:
            return False
        shutil.rmtree(self._dir(job_id))
        return True

    def output(self, job_id: str) -> Path:
        return self._dir(job_id) / "voice.sundervoice"

    # MARK: worker

    def run_forever(self):
        while True:
            queued = [j for j in self.list() if j["status"] == QUEUED]
            if not queued:
                self.wake.wait(timeout=5)
                self.wake.clear()
                continue
            self._run(min(queued, key=lambda j: j["created"]))

    def _run(self, job: dict):
        job_id = job["id"]
        job_dir = self._dir(job_id)
        self.stop_requested = self.force_requested = None
        self.current, self.heartbeat = job_id, time.time()
        started = time.time()
        self._update(job_id, status=RUNNING, error=None, started=started)
        self.log(job_id, f"Started ({'instrumentals' if job.get('kind') == 'stems' else 'voice training'}: {job['name']})")
        try:
            if job.get("kind") == "stems":
                self._run_stems(job)
            else:
                self._run_voice(job, job_dir)
        finally:
            final = self.get(job_id) or {}
            self.log(job_id, f"Finished as {final.get('status')} after {_duration(time.time() - started)}"
                     + (f": {final['error']}" if final.get("error") else ""))
            self.current = None

    def _progress(self, job_id: str):
        last = {"write": 0.0, "message": None}

        def progress(stage, fraction, message):
            self.heartbeat = time.time()
            if time.time() - last["write"] > 1 or fraction >= 1:  # don't rewrite job.json every step
                last["write"] = time.time()
                self._update(job_id, stage=stage, progress=round(fraction, 4), message=message)
        return progress

    def _failed(self, job_id: str, error: Exception):
        traceback.print_exc()
        self.log(job_id, "Error:\n" + traceback.format_exc().rstrip())
        self._update(job_id, status=FAILED, error=str(error) or type(error).__name__)

    def _run_voice(self, job: dict, job_dir: Path):
        job_id = job["id"]
        spec = pipeline.Spec(
            name=job["name"], inputs=sorted((job_dir / "uploads").glob("*")),
            isolated=job["isolated"], epochs=job["epochs"], engine=job.get("engine", "rvc"),
            reference=next(job_dir.glob("reference.*"), None))
        progress = self._progress(job_id)
        stage_seen = {"stage": None}

        def logged(stage, fraction, message):
            if stage != stage_seen["stage"]:
                stage_seen["stage"] = stage
                self.log(job_id, f"Stage: {stage} ({message})")
            progress(stage, fraction, message)

        try:
            pipeline.run(job_dir, spec, self.output(job_id), logged, lambda: self.stop_requested == job_id)
            self._update(job_id, status=DONE, stage="done", progress=1.0, message="Ready to download")
        except Stopped:
            self._update(job_id, status=STOPPED, message="Stopped · resume to continue from the last checkpoint")
        except Exception as error:
            self._failed(job_id, error)

    def _run_stems(self, job: dict):
        job_id = job["id"]
        progress = self._progress(job_id)
        states = dict(job.get("trackStates") or {})

        def on_track(track_id, state, **info):
            states[track_id] = {"state": state, **info}
            self._update(job_id, trackStates=states)

        count = len(job["tracks"])
        try:
            stems.run(self._dir(job_id), job["tracks"], progress, lambda: self.stop_requested == job_id,
                      lambda: self.force_requested == job_id, on_track, lambda text: self.log(job_id, text))
            self._update(job_id, status=DONE, stage="done", progress=1.0,
                         message=f"{count} {'song' if count == 1 else 'songs'} ready in Jellyfin")
        except Stopped:
            forced = self.force_requested == job_id
            self._update(job_id, status=STOPPED, message="Force-stopped · resume to redo the interrupted song and continue"
                         if forced else "Stopped · resume to continue with the remaining songs")
        except Exception as error:
            self._failed(job_id, error)

    # MARK: helpers

    def _dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def _update(self, job_id: str, **fields):
        with self.lock:
            job = self.get(job_id)
            if job:
                job.update(fields)
                self._write(job)

    def _write(self, job: dict):
        path = self._dir(job["id"]) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job))
        tmp.replace(path)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600} h {seconds % 3600 // 60} min" if seconds >= 3600 else f"{seconds // 60} min {seconds % 60} s"
