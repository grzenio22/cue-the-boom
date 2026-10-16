"""HTTP API for the web app: submit a shot, follow its progress, fetch the results.

    cuesync serve --workflow workflows/fluid_element.api.json   # full renders through COMFY_BASE_URL + instant
    cuesync serve --instant-only                                # no GPU: the pre-rendered library only

POST /api/jobs (multipart): plate (or plate_sample), voice (or voice_sample) (+ optional face, timing.json from
     Dub Mode), effect, x, y, seed, mode ("render" | "instant").
GET  /api/jobs/{id}         status, stage, progress, result URLs
GET  /api/jobs/{id}/events  the same as server-sent events
GET  /api/jobs/{id}/files/{name}   final.mp4, composite.mp4, breakdown.mp4, guide.mp4, element.mp4, receipt.json
GET  /api/samples           sample plates and noises (try it without a camera or a microphone)
POST /api/compare           two finished shots side by side, synced on the voice
Full renders go through one worker in order (one GPU); instant shots have their own worker (CPU).
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

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # the `server` extra
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

RESULT_FILES = ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "element.mp4", "receipt.json",
                "compare.mp4")
SAMPLES = next((d for d in (Path.cwd() / "samples", Path(__file__).resolve().parents[3] / "samples")
                if (d / "samples.json").exists()), Path.cwd() / "samples")
MAX_UPLOAD_MB = 300


@dataclass
class Job:
    id: str
    dir: Path
    spec: object
    mode: str = "render"  # render | instant | compare
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
        return {"id": self.id, "mode": self.mode, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 3), "message": self.message, "error": self.error, "files": files,
                "receipt": receipt}


class JobQueue:
    """One worker per mode: full renders queue for the GPU, instant shots run beside them on the CPU."""

    def __init__(self, root: Path, renderers: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.renderers = {k: v for k, v in renderers.items() if v is not None}
        self.jobs: dict[str, Job] = {}
        self._qs: dict[str, queue.Queue[str]] = {m: queue.Queue() for m in self.renderers}
        self._lock = threading.Lock()
        for mode in self.renderers:
            threading.Thread(target=self._work, args=(mode,), daemon=True).start()

    def queued(self, mode: str = "render") -> int:
        q = self._qs.get(mode)
        return q.qsize() if q else 0

    def submit(self, job: Job) -> Job:
        with self._lock:
            self.jobs[job.id] = job
        ahead = self.queued(job.mode)
        self._emit(job, "queued", 0.0, f"Waiting for the GPU ({ahead} ahead)" if job.mode == "render" else "Starting")
        self._qs[job.mode].put(job.id)
        return job

    def _emit(self, job: Job, stage: str, frac: float, msg: str) -> None:
        job.stage, job.progress, job.message = stage, frac, msg
        job.events.append({"stage": stage, "progress": round(frac, 3), "message": msg, "t": round(time.time() - job.created, 2)})

    def _work(self, mode: str) -> None:
        from .pipeline import make_shot

        while True:
            job = self.jobs[self._qs[mode].get()]
            job.status = "running"
            try:
                make_shot(job.spec, self.renderers[mode], job.dir, progress=lambda s, f, m: self._emit(job, s, f, m))
                job.status = "done"
            except Exception as exc:  # report, keep serving
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                (job.dir / "error.txt").write_text(traceback.format_exc())
                self._emit(job, "error", job.progress, job.error)


def create_app(renderer, jobs_dir: Path, allow_origins: list[str] | None = None, shot_defaults: dict | None = None,
               instant=None, samples_dir: Path = SAMPLES):
    """``renderer``: full renders (None: an instant-only server). ``instant``: the library renderer, if any."""
    from .effects import _candidate_dirs, load_effect
    from .pipeline import ShotSpec

    app = FastAPI(title="Cue the Boom", version="0.1")
    if allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_methods=["*"], allow_headers=["*"])
    jq = JobQueue(jobs_dir, {"render": renderer, "instant": instant})
    samples_dir = Path(samples_dir)

    def sample_files(kind: str) -> dict[str, Path]:
        exts = {".mp4", ".mov", ".webm"} if kind == "plates" else {".wav", ".m4a", ".mp3", ".webm", ".ogg"}
        d = samples_dir / kind
        return {p.name: p for p in sorted(d.glob("*")) if p.suffix.lower() in exts} if d.is_dir() else {}

    def effect_ids() -> list[str]:
        for d in _candidate_dirs(Path("effects")):
            if d.is_dir() and any(d.glob("*.json")):
                return sorted(p.stem for p in d.glob("*.json"))
        return []

    @app.get("/api/health")
    def health():
        return {"ok": True, "renderer": getattr(renderer, "name", None), "modes": sorted(jq.renderers),
                "queued": jq.queued("render")}

    @app.get("/api/samples")
    def samples():
        meta = {}
        if (samples_dir / "samples.json").exists():
            meta = json.loads((samples_dir / "samples.json").read_text(encoding="utf-8"))
        return {kind: [{"name": n, "url": f"/api/samples/{kind}/{n}", **meta.get(kind, {}).get(n, {})}
                       for n in sample_files(kind)] for kind in ("plates", "voices")}

    @app.get("/api/samples/{kind}/{name}")
    def sample_file(kind: str, name: str):
        f = sample_files(kind).get(name) if kind in ("plates", "voices") else None
        if f is None:
            raise HTTPException(404, "no such sample")
        return FileResponse(f)

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
    async def submit(plate: UploadFile | None = File(None), voice: UploadFile | None = File(None),
                     effect: str = Form(...), x: float | None = Form(None), y: float | None = Form(None),
                     seed: int = Form(1), offset_s: float | None = Form(None), timing: UploadFile | None = File(None),
                     face: UploadFile | None = File(None), label: str = Form(""), mode: str = Form("render"),
                     plate_sample: str | None = Form(None), voice_sample: str | None = Form(None)):
        try:
            load_effect(effect)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, f"unknown effect {effect!r}: {exc}") from None
        if mode not in jq.renderers:
            raise HTTPException(400, f"mode {mode!r} is not available here (available: {sorted(jq.renderers)})")
        if (plate is None) == (plate_sample is None) or (voice is None) == (voice_sample is None):
            raise HTTPException(400, "send a plate (or plate_sample) and a voice (or voice_sample)")
        if plate_sample is not None and plate_sample not in sample_files("plates"):
            raise HTTPException(400, f"no sample plate {plate_sample!r}")
        if voice_sample is not None and voice_sample not in sample_files("voices"):
            raise HTTPException(400, f"no sample voice {voice_sample!r}")
        job_id = uuid.uuid4().hex[:12]
        d = jq.root / job_id
        (d / "in").mkdir(parents=True)
        if plate is not None:
            plate_path = await save(plate, d / "in" / f"plate{Path(plate.filename or '.mp4').suffix or '.mp4'}")
        else:
            plate_path = sample_files("plates")[plate_sample]
        if voice is not None:
            voice_path = await save(voice, d / "in" / f"voice{Path(voice.filename or '.wav').suffix or '.wav'}")
        else:
            voice_path = sample_files("voices")[voice_sample]
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
        return jq.submit(Job(id=job_id, dir=d, spec=spec, mode=mode)).public()

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

    @app.post("/api/compare")
    def compare(a: str = Form(...), b: str = Form(...)):
        """Two finished shots side by side, both performances' onsets on the same frame."""
        from .audio import CueSheet
        from .compare import side_by_side
        from .media import read_video, write_video

        ja, jb = get(a), get(b)
        if ja.status != "done" or jb.status != "done" or ja.spec is None or jb.spec is None:
            raise HTTPException(409, "both shots must be finished")
        clips, envs, anchors, labels = [], [], [], []
        for k, j in enumerate((ja, jb), 1):
            clips.append(read_video(j.dir / "composite.mp4").frames)
            envs.append(np.asarray(CueSheet.from_json(json.loads((j.dir / "cue.json").read_text())).envelope))
            guide = json.loads((j.dir / "guide.json").read_text())
            anchors.append(int(guide["anchor_frame"]))
            labels.append(f"take {k}: {guide['effect'].replace('_', ' ')}")
        out = side_by_side(clips[0], clips[1], anchors[0], anchors[1], envs[0], envs[1], labels)
        cid = uuid.uuid4().hex[:12]
        d = jq.root / cid
        d.mkdir(parents=True)
        write_video(d / "compare.mp4", out, fps=24, crf=18, preset="veryfast")
        job = Job(id=cid, dir=d, spec=None, mode="compare", status="done", stage="done", progress=1.0, message="Done")
        jq.jobs[cid] = job
        return job.public()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def files(job_id: str, name: str):
        job = get(job_id)
        if name not in RESULT_FILES or not (job.dir / name).exists():
            raise HTTPException(404, "no such file")
        return FileResponse(job.dir / name, media_type="application/json" if name.endswith(".json") else "video/mp4")

    return app


def serve(*, workflow: Path | None, jobs_dir: Path, host: str = "127.0.0.1", port: int = 8765,
          allow_origins: list[str] | None = None, timeout: float = 600.0, library: Path | None = None) -> None:
    import uvicorn

    from .library import DEFAULT_DIR, LibraryRenderer
    from .pipeline import ComfyRenderer

    lib = Path(library) if library else DEFAULT_DIR
    instant = LibraryRenderer(lib) if (lib / "manifest.json").exists() else None
    full = ComfyRenderer(Path(workflow), timeout=timeout) if workflow else None
    if full is None and instant is None:
        raise SystemExit(f"nothing to serve: no --workflow and no element library at {lib}")
    app = create_app(full, jobs_dir, allow_origins, instant=instant)
    uvicorn.run(app, host=host, port=port, log_level="info")
