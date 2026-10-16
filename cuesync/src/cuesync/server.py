"""HTTP API for the web app: submit a shot, follow its progress, fetch the results.

    cuesync serve --workflow workflows/fluid_element.api.json        # renders through COMFY_BASE_URL

POST /api/jobs (multipart): plate, voice (+ optional face, timing.json from Dub Mode), effect, x, y, seed.
GET  /api/jobs/{id}         status, stage, progress, result URLs
GET  /api/jobs/{id}/events  the same as server-sent events
GET  /api/jobs/{id}/files/{name}   final.mp4, composite.mp4, breakdown.mp4, guide.mp4, element.mp4, receipt.json
One worker renders jobs in order (one GPU).
"""

import asyncio
import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # the `server` extra
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

RESULT_FILES = ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "element.mp4", "receipt.json")
MAX_UPLOAD_MB = 300


@dataclass
class Job:
    id: str
    dir: Path
    spec: object
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for the GPU"
    error: str | None = None
    created: float = field(default_factory=time.time)
    events: list = field(default_factory=list)

    def public(self) -> dict:
        files = {n: f"/api/jobs/{self.id}/files/{n}" for n in RESULT_FILES if (self.dir / n).exists()} if self.status == "done" else {}
        receipt = None
        if self.status == "done" and (self.dir / "receipt.json").exists():
            receipt = json.loads((self.dir / "receipt.json").read_text())
        return {"id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 3),
                "message": self.message, "error": self.error, "files": files, "receipt": receipt}


class JobQueue:
    def __init__(self, root: Path, renderer):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, job: Job) -> Job:
        with self._lock:
            self.jobs[job.id] = job
        self._emit(job, "queued", 0.0, f"Waiting for the GPU ({self._q.qsize()} ahead)")
        self._q.put(job.id)
        return job

    def _emit(self, job: Job, stage: str, frac: float, msg: str) -> None:
        job.stage, job.progress, job.message = stage, frac, msg
        job.events.append({"stage": stage, "progress": round(frac, 3), "message": msg, "t": round(time.time() - job.created, 2)})

    def _work(self) -> None:
        from .pipeline import make_shot

        while True:
            job = self.jobs[self._q.get()]
            job.status = "running"
            try:
                make_shot(job.spec, self.renderer, job.dir, progress=lambda s, f, m: self._emit(job, s, f, m))
                job.status = "done"
            except Exception as exc:  # report, keep serving
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                (job.dir / "error.txt").write_text(traceback.format_exc())
                self._emit(job, "error", job.progress, job.error)


def create_app(renderer, jobs_dir: Path, allow_origins: list[str] | None = None, shot_defaults: dict | None = None):
    from .effects import _candidate_dirs, load_effect
    from .pipeline import ShotSpec

    app = FastAPI(title="Cue the Boom", version="0.1")
    if allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_methods=["*"], allow_headers=["*"])
    jq = JobQueue(jobs_dir, renderer)

    def effect_ids() -> list[str]:
        for d in _candidate_dirs(Path("effects")):
            if d.is_dir() and any(d.glob("*.json")):
                return sorted(p.stem for p in d.glob("*.json"))
        return []

    @app.get("/api/health")
    def health():
        return {"ok": True, "renderer": getattr(renderer, "name", "?"), "queued": jq._q.qsize()}

    @app.get("/api/effects")
    def effects():
        out = []
        for i in effect_ids():
            e = load_effect(i)
            out.append({"id": e.id, "fluid": e.fluid, "behaviour": e.behaviour, "stops_with_voice": e.voice_gate == "both"})
        return out

    async def save(upload: UploadFile, path: Path) -> Path:
        size = 0
        with open(path, "wb") as fh:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB << 20:
                    raise HTTPException(413, f"{upload.filename} is larger than {MAX_UPLOAD_MB} MB")
                fh.write(chunk)
        return path

    @app.post("/api/jobs")
    async def submit(plate: UploadFile = File(...), voice: UploadFile = File(...), effect: str = Form(...),
                     x: float | None = Form(None), y: float | None = Form(None), seed: int = Form(1),
                     offset_s: float | None = Form(None), timing: UploadFile | None = File(None),
                     face: UploadFile | None = File(None), label: str = Form("")):
        try:
            load_effect(effect)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, f"unknown effect {effect!r}: {exc}") from None
        job_id = uuid.uuid4().hex[:12]
        d = jq.root / job_id
        (d / "in").mkdir(parents=True)
        plate_path = await save(plate, d / "in" / f"plate{Path(plate.filename or '.mp4').suffix or '.mp4'}")
        voice_path = await save(voice, d / "in" / f"voice{Path(voice.filename or '.wav').suffix or '.wav'}")
        face_path = await save(face, d / "in" / f"face{Path(face.filename or '.webm').suffix or '.webm'}") if face else None
        t = {}
        if timing is not None:
            t = json.loads((await timing.read()).decode() or "{}")
            (d / "in" / "timing.json").write_text(json.dumps(t, indent=1))
        if offset_s is None:
            offset_s = float(t.get("noise_offset_s", 0.0)) - float(t.get("input_latency_s", 0.0))
        point = (x, y) if x is not None and y is not None else tuple(t.get("emission_point") or (0.5, 0.7))
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            raise HTTPException(400, "x and y must be in [0, 1]")
        spec = ShotSpec(plate=plate_path, voice=voice_path, effect=effect, point=(float(point[0]), float(point[1])),
                        seed=seed, offset_s=offset_s, face=face_path, face_offset_s=t.get("face_offset_s"), label=label,
                        **(shot_defaults or {}))
        return jq.submit(Job(id=job_id, dir=d, spec=spec)).public()

    def get(job_id: str) -> Job:
        job = jq.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job

    @app.get("/api/jobs/{job_id}")
    def status(job_id: str):
        return get(job_id).public()

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str):
        job = get(job_id)

        async def stream():
            sent = 0
            while True:
                while sent < len(job.events):
                    yield f"data: {json.dumps(job.events[sent])}\n\n"
                    sent += 1
                if job.status in ("done", "error"):
                    yield f"event: end\ndata: {json.dumps(job.public())}\n\n"
                    return
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/files/{name}")
    def files(job_id: str, name: str):
        job = get(job_id)
        if name not in RESULT_FILES or not (job.dir / name).exists():
            raise HTTPException(404, "no such file")
        return FileResponse(job.dir / name, media_type="application/json" if name.endswith(".json") else "video/mp4")

    return app


def serve(*, workflow: Path, jobs_dir: Path, host: str = "127.0.0.1", port: int = 8765,
          allow_origins: list[str] | None = None, timeout: float = 600.0) -> None:
    import uvicorn

    from .pipeline import ComfyRenderer

    app = create_app(ComfyRenderer(Path(workflow), timeout=timeout), jobs_dir, allow_origins)
    uvicorn.run(app, host=host, port=port, log_level="info")
