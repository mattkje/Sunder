"""HTTP API + a single-page UI for the job queue. No authentication: meant
for a home LAN, not the open internet."""

import os
import re
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from . import jellyfin
from .jobs import DONE, JobStore

DATA_DIR = Path(os.environ.get("SUNDERVOICE_DATA", "/data"))
store = JobStore(DATA_DIR / "jobs")
app = FastAPI(title="Sunder Voice Server")
STATIC = Path(__file__).parent / "static"


@app.on_event("startup")
def start_worker():
    threading.Thread(target=store.run_forever, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/logo.png")
def logo():
    return FileResponse(STATIC / "logo.png")


@app.get("/api/jobs")
def list_jobs():
    return store.list()


@app.post("/api/jobs")
def create_job(files: list[UploadFile], name: str = Form(...), isolated: bool = Form(False), epochs: int = Form(0),
               engine: str = Form("rvc"), reference: UploadFile | None = None):
    name = name.strip()[:80]
    if not name:
        raise HTTPException(400, "Name the voice.")
    if not files:
        raise HTTPException(400, "Add at least one audio file.")
    if engine not in ("rvc", "seedvc"):
        raise HTTPException(400, "Unknown engine.")
    job = store.create(name, isolated, epochs if epochs > 0 else None, engine)
    for i, upload in enumerate(files):
        with store.upload_path(job["id"], i, upload.filename or "").open("wb") as out:
            shutil.copyfileobj(upload.file, out)
    if reference and reference.filename and engine == "seedvc":
        with store.reference_path(job["id"], reference.filename).open("wb") as out:
            shutil.copyfileobj(reference.file, out)
    store.enqueue(job["id"])
    return store.get(job["id"])


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str, force: bool = False):
    _job(job_id)
    store.stop(job_id, force)
    return store.get(job_id)


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, lines: int = 400):
    _job(job_id)
    return store.read_log(job_id, max(1, min(lines, 5000)))


@app.get("/api/status")
def status():
    """What the worker is doing plus the machine's load, memory and disk, for the status panel."""
    info = store.status()
    try:
        import torch
        info["device"] = f"CUDA · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() \
            else f"CPU · {torch.get_num_threads()} threads"
    except Exception:
        info["device"] = "unknown"
    info["cpus"] = os.cpu_count()
    info["load"] = [round(x, 2) for x in os.getloadavg()]
    try:
        mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        kb = lambda key: int(mem[key].split()[0]) * 1024
        info["memory"] = {"total": kb("MemTotal"), "available": kb("MemAvailable")}
    except (OSError, KeyError, ValueError):
        info["memory"] = None
    disks = {"data": DATA_DIR}
    music = os.environ.get("JELLYFIN_PATH_MAP", "").partition("=")[2]
    if music:
        disks["music"] = Path(music)
    info["disks"] = {}
    for name, path in disks.items():
        try:
            usage = shutil.disk_usage(path)
            info["disks"][name] = {"total": usage.total, "free": usage.free}
        except OSError:
            pass
    return info


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    job = _job(job_id)
    if job["status"] not in ("stopped", "failed"):
        raise HTTPException(409, "Only stopped or failed jobs can be resumed.")
    store.enqueue(job_id)
    return store.get(job_id)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    _job(job_id)
    if not store.delete(job_id):
        raise HTTPException(409, "Stop the job before deleting it.")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = _job(job_id)
    if job["status"] != DONE:
        raise HTTPException(409, "Not finished yet.")
    filename = re.sub(r"[^\w\- ]", "", job["name"]).strip() or "voice"
    return FileResponse(store.output(job_id), filename=f"{filename}.sundervoice", media_type="application/zip")


# MARK: instrumentals for Jellyfin tracks


class TrackRef(BaseModel):
    albumId: str
    id: str


class StemsRequest(BaseModel):
    name: str
    albumIds: list[str] = []      # whole albums
    tracks: list[TrackRef] = []  # individually picked songs


@app.get("/api/jellyfin/status")
def jellyfin_status():
    return {"configured": jellyfin.configured()}


@app.get("/api/jellyfin/albums")
def jellyfin_albums(search: str = "", start: int = 0):
    _require_jellyfin()
    return jellyfin.albums(search.strip(), max(start, 0))


@app.get("/api/jellyfin/albums/{album_id}/tracks")
def jellyfin_tracks(album_id: str):
    _require_jellyfin()
    return jellyfin.tracks(_jellyfin_id(album_id))


@app.get("/api/jellyfin/albums/{album_id}/image")
def jellyfin_image(album_id: str):
    _require_jellyfin()
    try:
        return Response(jellyfin.image(_jellyfin_id(album_id)), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})
    except OSError:
        raise HTTPException(404, "No artwork.")


@app.post("/api/stems")
def create_stems(request: StemsRequest):
    """Queues instrumental generation for whole albums and/or individually picked songs. Paths come from
    Jellyfin here, never from the request, so a request can only name tracks Jellyfin knows about."""
    _require_jellyfin()
    picked: dict[str, set[str]] = {}
    for ref in request.tracks:
        picked.setdefault(_jellyfin_id(ref.albumId), set()).add(_jellyfin_id(ref.id))
    for album_id in request.albumIds:
        picked[_jellyfin_id(album_id)] = set()  # empty = every track
    tracks, seen = [], set()
    for album_id, ids in picked.items():
        for track in jellyfin.tracks(album_id):
            if (not ids or track["id"] in ids) and track["id"] not in seen:
                seen.add(track["id"])
                tracks.append(track)
    tracks = [t for t in tracks if t["found"] and not t["hasStems"]]
    if not tracks:
        raise HTTPException(400, "Nothing to generate: those songs already have instrumentals, or their files weren't found.")
    job = store.create_stems(request.name.strip()[:120] or "Instrumentals", tracks)
    store.enqueue(job["id"])
    return store.get(job["id"])


def _require_jellyfin():
    if not jellyfin.configured():
        raise HTTPException(503, "Set JELLYFIN_URL and JELLYFIN_API_KEY to browse Jellyfin.")


def _jellyfin_id(value: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F-]{32,36}", value):
        raise HTTPException(400, "Invalid Jellyfin id.")
    return value


def _job(job_id: str) -> dict:
    try:
        job = store.get(job_id)
    except KeyError:
        job = None
    if not job:
        raise HTTPException(404, "No such job.")
    return job
