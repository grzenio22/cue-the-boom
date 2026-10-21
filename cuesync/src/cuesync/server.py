"""HTTP API for the web app: submit a shot, follow its progress, fetch the results.

    cuesync serve --workflow workflows/fluid_element.api.json   # full renders through COMFY_BASE_URL + instant
    cuesync serve --instant-only                                # no GPU: the pre-rendered library only

POST /api/jobs (multipart): plate (or plate_sample), voice (or voice_sample) (+ optional face, timing.json from
     Dub Mode), effect, x, y, seed, mode ("render" | "instant").
GET  /api/jobs/{id}         status, stage, progress, result URLs
GET  /api/jobs/{id}/events  the same as server-sent events
POST /api/hands/prepare     your tapped hand followed through your clip before a shot (its path, where it was lost)
GET  /api/jobs/{id}/files/{name}   final.mp4, composite.mp4, breakdown.mp4, guide.mp4, element.mp4, receipt.json
GET  /api/samples           sample plates and noises (try it without a camera or a microphone)
GET  /api/gallery           sample shots made ahead of time (cuesync precompute); a matching request is answered at once
GET  /api/library           the hand elements rendered ahead of time, with their poses (the studio's live preview)
POST /api/compare           two finished shots side by side, synced on the voice
POST /api/jobs/{id}/retime  move the hit / the stop of a finished shot and composite it again (no new render)
Full renders go through one worker in order (one GPU); instant shots have their own worker (CPU).
"""

import asyncio
import dataclasses
import hashlib
import json
import math
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

RESULT_FILES = ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "element.mp4", "receipt.json", "poster.jpg", "share.mp4",
                "compare.mp4", "cue.json", "guide.json", "guide_preview.mp4", "hand_track.json")
EARLY_FILES = ("guide_preview.mp4", "cue.json", "hand_track.json")  # shown while the shot is still being made
# the checkout's samples/: from the current folder or any folder above it (started from web/), or next to the source
SAMPLES = next((d / "samples" for d in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[3])
                if (d / "samples" / "samples.json").exists()), Path.cwd() / "samples")
MAX_UPLOAD_MB = 300


@dataclass
class Job:
    id: str
    dir: Path
    spec: object
    mode: str = "render"  # render | scene | instant | compare | retime
    cached: bool = False  # answered from the shots made ahead of time
    title: str = ""
    again: int | None = None
    retime: dict | None = None  # {"src": Path, "hit_shift": int, "stop": int | None}: re-time a finished shot
    replay: dict | None = None  # {"src": Path | [Path, ...]}: perform finished scene shots again, with no new render
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting for the GPU"
    error: str | None = None
    created: float = field(default_factory=time.time)
    events: list = field(default_factory=list)

    def public(self) -> dict:
        names = RESULT_FILES if self.status == "done" else EARLY_FILES
        files = {n: f"/api/jobs/{self.id}/files/{n}" for n in names if (self.dir / n).exists()}
        receipt = None
        if self.status == "done" and (self.dir / "receipt.json").exists():
            receipt = json.loads((self.dir / "receipt.json").read_text())
        spec_extra = getattr(self.spec, "extra", None) or {}
        return {"id": self.id, "mode": self.mode, "cached": self.cached, "title": self.title, "again": self.again,
                "status": self.status,
                "effect": getattr(self.spec, "effect", None) or ((receipt or {}).get("effect") or {}).get("id"),
                "gesture": spec_extra.get("gesture"),
                "heard": (spec_extra.get("auto") or {}).get("heard"),  # effect "auto": the sound's kind ("whoosh")
                "stage": self.stage,
                "progress": round(self.progress, 3), "message": self.message, "error": self.error, "files": files,
                "receipt": receipt}


ELEMENT_EFFECTS = ("fireball", "flame_plume", "smoke_puff", "steam_jet")  # tuned for element mode, in the library


def same_strokes(a, b, tol: float = 0.005) -> bool:
    """Is drawing ``b`` the drawing ``a`` (a sample's own, made ahead of time), to within ``tol`` of the frame?"""
    try:
        return len(a) == len(b) and all(
            abs(float(s["width"]) - float(t["width"])) < tol and len(s["points"]) == len(t["points"])
            and all(abs(float(p[0]) - float(q[0])) < tol and abs(float(p[1]) - float(q[1])) < tol
                    for p, q in zip(s["points"], t["points"]))
            for s, t in zip(a, b))
    except (TypeError, KeyError, ValueError, IndexError):
        return False


class JobQueue:
    """One worker per mode: full renders queue for the GPU, instant shots run beside them on the CPU."""

    def __init__(self, root: Path, renderers: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.renderers = {k: v for k, v in renderers.items() if v is not None}
        self.jobs: dict[str, Job] = {}
        modes = list(self.renderers) + ["retime"]  # re-times are compositing only: their own CPU worker
        self._qs: dict[str, queue.Queue[str]] = {m: queue.Queue() for m in modes}
        self._lock = threading.Lock()
        for mode in modes:
            threading.Thread(target=self._work, args=(mode,), daemon=True).start()

    @staticmethod
    def queue_of(mode: str) -> str:
        """Scene renders and element renders share the one GPU worker; a replay is compositing, like a re-time."""
        return "render" if mode in ("scene", "bending") else "retime" if mode in ("replay", "bending_instant", "hand_prep")             else mode

    def queued(self, mode: str = "render") -> int:
        q = self._qs.get(self.queue_of(mode))
        return q.qsize() if q else 0

    def submit(self, job: Job) -> Job:
        with self._lock:
            self.jobs[job.id] = job
        ahead = self.queued(job.mode)
        gpu = self.queue_of(job.mode) == "render"
        self._emit(job, "queued", 0.0, f"Waiting for the GPU ({ahead} ahead)" if gpu else "Starting")
        self._qs[self.queue_of(job.mode)].put(job.id)
        return job

    def _emit(self, job: Job, stage: str, frac: float, msg: str) -> None:
        job.stage, job.progress, job.message = stage, frac, msg
        job.events.append({"stage": stage, "progress": round(frac, 3), "message": msg, "t": round(time.time() - job.created, 2)})

    def _wait_for(self, job_id: str | None, timeout_s: float = 900.0) -> None:
        """Until that job (your tap being followed through the clip, on the CPU worker) has finished."""
        start = time.time()
        while job_id and job_id in self.jobs and self.jobs[job_id].status in ("queued", "running"):
            if time.time() - start > timeout_s:
                return
            time.sleep(0.25)

    def _work(self, mode: str) -> None:
        from .bending import bending_again, bending_job, prepare_take
        from .pipeline import make_shot, retime_shot
        from .replay import replay_scene_shot

        while True:
            job = self.jobs[self._qs[mode].get()]
            job.status = "running"
            try:
                say = lambda s, f, m: self._emit(job, s, f, m)  # noqa: E731
                if job.retime:
                    retime_shot(job.retime["src"], job.spec, job.dir, job.retime["hit_shift"], job.retime["stop"], say)
                elif job.replay and job.replay.get("bending"):
                    self._wait_for(job.spec.extra.pop("reuse_job", None))
                    bending_again(job.replay["src"], job.spec, job.dir, say)
                elif job.replay:
                    replay_scene_shot(job.replay["src"], job.spec, job.dir, say)
                elif job.mode == "hand_prep":  # your tap, followed through the clip before any shot is made
                    prepare_take(job.spec, job.dir, progress=say)
                elif job.mode in ("bending", "bending_instant"):  # instant: no GPU, an element rendered ahead of time
                    reuse = job.spec.extra.pop("reuse", None)
                    self._wait_for(job.spec.extra.pop("reuse_job", None))  # (its hand, still being cut out)
                    bending_job(job.spec, self.renderers.get("render") if job.mode == "bending" else None, job.dir,
                                progress=say, reuse=Path(reuse) if reuse else None)
                else:
                    make_shot(job.spec, self.renderers[mode], job.dir, progress=say)
                job.status = "done"
            except Exception as exc:  # report, keep serving
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                (job.dir / "error.txt").write_text(traceback.format_exc())
                self._emit(job, "error", job.progress, job.error)


def create_app(renderer, jobs_dir: Path, allow_origins: list[str] | None = None, shot_defaults: dict | None = None,
               instant=None, samples_dir: Path = SAMPLES, precomputed: Path | None = None, local: bool = False,
               studio: Path | None = None):
    """``renderer``: full renders (None: an instant-only server). ``instant``: the library renderer, if any.
    ``local``: served on this machine only (it may then set up the hand tracker when asked). ``studio``: the static
    studio (tools/build_static_studio.py) to serve at the root, beside the API."""
    from .effects import _candidate_dirs, load_effect
    from .pipeline import ShotSpec

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        """A browser that closes a tab or seeks a video drops its connection: on Windows each drop printed a
        ConnectionResetError traceback (harmless, and alarming in the terminal). Those alone are let go."""
        loop = asyncio.get_running_loop()
        before = loop.get_exception_handler()

        def handler(lp, context):
            if isinstance(context.get("exception"), ConnectionResetError):
                return
            if before is not None:
                before(lp, context)
            else:
                lp.default_exception_handler(context)

        loop.set_exception_handler(handler)
        yield

    app = FastAPI(title="Cue the Boom", version="0.1", lifespan=lifespan)
    if allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_methods=["*"], allow_headers=["*"])
    jq = JobQueue(jobs_dir, {"render": renderer, "instant": instant})
    samples_dir = Path(samples_dir)
    from .precompute import cache_key, copy_shot, load_cache

    cache = load_cache(precomputed)
    gallery: list[Job] = []
    rank = {"scene": 0, "bending": 0, "render": 1}  # one tile per sample pair: the scene render, else the element render
    best = min((rank.get(e["mode"], 2) for e in cache.values()), default=2)
    shown = {k: e for k, e in cache.items() if rank.get(e["mode"], 2) == best}  # scene shots only, when there are any
    # by rank where a shot has one (a hand's default is before every scene's), then real footage first, then by plate
    order = lambda kv: (kv[1].get("rank", -100 if kv[1].get("mode") == "bending" else 0),  # noqa: E731
                        not kv[1].get("real", False), kv[1].get("plate", ""), kv[1]["title"])  # your hand first
    for key, entry in sorted(shown.items(), key=order):
        same = [g for g in gallery if g.title == entry["title"]]
        if same:  # one gallery tile per sample pair: the full render when there is one
            continue
        gid = "g" + hashlib.sha1(key.encode()).hexdigest()[:11]
        job = Job(id=gid, dir=Path(entry["path"]), spec=None, mode=entry["mode"], status="done", stage="done",
                  progress=1.0, message="Made earlier", cached=True, title=entry["title"], again=entry.get("again"))
        jq.jobs[gid] = job
        gallery.append(job)
    ready = {"t": 0.0, "ok": False}

    def render_ready() -> bool:
        """Is the GPU backend actually answering? (checked at most every 15 s)"""
        if renderer is None:
            return False
        probe = getattr(renderer, "ready", None)
        if probe is None:
            return True
        if time.time() - ready["t"] > 15.0:
            ready["ok"], ready["t"] = bool(probe()), time.time()
        return ready["ok"]

    scenes = bool(getattr(renderer, "scenes", False))

    def available_modes() -> list[str]:
        modes = [m for m in jq.renderers if m != "render" or render_ready()]
        if scenes and "render" in modes:
            from .bending import helpers_ready

            modes += ["scene"] + (["bending"] if helpers_ready() else [])  # (a hand needs the tracker installed)
        if "bending" not in modes:  # no GPU: your hand still gets an element rendered ahead of time, with the tracker
            from .bending import helpers_ready, library_dir

            if helpers_ready() and library_dir() is not None:
                modes.append("bending")
        return sorted(modes)

    def sample_meta() -> dict:
        f = samples_dir / "samples.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}

    def sample_files(kind: str) -> dict[str, Path]:
        exts = {".mp4", ".mov", ".webm"} if kind == "plates" else {".wav", ".m4a", ".mp3", ".webm", ".ogg"}
        d = samples_dir / kind
        return {p.name: p for p in sorted(d.glob("*")) if p.suffix.lower() in exts} if d.is_dir() else {}

    def effect_ids() -> list[str]:
        return sorted({p.stem for d in _candidate_dirs(Path("effects")) if d.is_dir() for p in d.glob("*.json")})

    @app.get("/api/health")
    def health():
        from .bending import helpers_ready, library_dir

        return {"ok": True, "renderer": getattr(renderer, "name", None), "modes": available_modes(),
                "made_ahead": sorted({e["mode"] for e in cache.values()}), "queued": jq.queued("render"),
                "hand": {"tracker": helpers_ready(), "library": library_dir() is not None,  # what a hand of your own needs
                         "can_install": local, "installing": setup["running"], "install_error": setup["error"]}}

    setup = {"running": False, "error": None}

    @app.post("/api/preview")
    async def browser_preview(plate: UploadFile = File(...)):
        """A copy of your clip your browser can play (media.browser_copy), when it can't play the file itself."""
        import av
        from starlette.concurrency import run_in_threadpool

        from .media import browser_copy

        d = jq.root / f"preview-{uuid.uuid4().hex[:12]}"
        d.mkdir(parents=True)
        src = await save(plate, d / f"clip{Path(plate.filename or '.mp4').suffix or '.mp4'}")
        try:
            out = await run_in_threadpool(browser_copy, src, d / "preview.mp4")
        except (ValueError, OSError, av.error.FFmpegError) as exc:
            raise HTTPException(400, f"could not read that video: {exc}")
        finally:
            src.unlink(missing_ok=True)
        return FileResponse(out, media_type="video/mp4")

    @app.post("/api/hands/prepare")
    async def prepare_hand_take(plate: UploadFile = File(...), x: float = Form(...), y: float = Form(...),
                                frame: int = Form(0)):
        """Follow the hand you tapped (x, y as fractions of the picture, on frame ``frame``) through your clip before
        any shot is made: its track (hand_track.json) as soon as it is followed, so the studio draws the hand's path
        and where it lost it; then its matte. A shot sent with reuse=<this job's id> makes only the shot."""
        from .bending import helpers_ready

        if not helpers_ready():
            raise HTTPException(409, "the hand tracker isn't set up: Set up the hand tracker (cuesync hands --install)")
        if not (0 <= x <= 1 and 0 <= y <= 1) or frame < 0:
            raise HTTPException(400, "x and y must be in [0, 1] and frame >= 0")
        job_id = uuid.uuid4().hex[:12]
        d = jq.root / job_id
        (d / "in").mkdir(parents=True)
        plate_path = await save(plate, d / "in" / f"plate{Path(plate.filename or '.mp4').suffix or '.mp4'}")
        spec = ShotSpec(plate=plate_path, voice=None, effect="auto", point=(float(x), float(y)), mode="element",
                        **(shot_defaults or {}))
        spec.extra["tap_frame"] = int(frame)
        return jq.submit(Job(id=job_id, dir=d, spec=spec, mode="hand_prep", title="your hand")).public()

    @app.post("/api/hands/install")
    def install_hands():
        """Set up the hand tracker's environment (cuesync hands --install), in the background: only on this machine."""
        import subprocess
        import sys

        if not local:
            raise HTTPException(403, "the hand tracker is set up on the machine that serves the studio")
        if setup["running"]:
            return {"running": True}
        setup.update(running=True, error=None)  # (before the thread starts: a poll in between must see it running)

        def run() -> None:
            try:
                r = subprocess.run([sys.executable, "-m", "cuesync.cli", "hands", "--install"], capture_output=True,
                                   text=True, timeout=1800)
                if r.returncode != 0:
                    setup["error"] = (r.stderr.strip().splitlines() or ["the install failed"])[-1][:300]
            except Exception as exc:  # report it in health
                setup["error"] = f"{type(exc).__name__}: {exc}"[:300]
            finally:
                setup["running"] = False

        threading.Thread(target=run, daemon=True).start()
        return {"running": True}

    @app.get("/api/objects")
    def objects():
        from .gestures import OBJECTS

        return [{"id": k, "title": v["title"], "subject": v["subject"], "scene": v.get("scene", {})} for k, v in OBJECTS.items()]

    @app.get("/api/samples")
    def samples():
        meta = sample_meta()

        def ordered(kind: str) -> list[str]:  # in samples.json's order (the author's hands first), then the rest
            files = sample_files(kind)
            listed = [n for n in meta.get(kind, {}) if n in files]
            return listed + [n for n in files if n not in listed]

        return {kind: [{"name": n, "url": f"/api/samples/{kind}/{n}", **meta.get(kind, {}).get(n, {})}
                       for n in ordered(kind)] for kind in ("plates", "voices")}

    @app.get("/api/gallery")
    def gallery_list():
        return [g.public() for g in gallery]

    live: dict[str, dict] = {}

    @app.get("/api/library")
    def library_list():
        """The hand elements rendered ahead of time, for the studio's live preview to play on your hand as you
        perform: each one's pose in its render (ref: where the emitter is, which way, how big a hand), its ignition,
        where its steady burn ends (release) and its last lit frame (end), its black level, and how big it plays
        against hand for hand (fit: from a palm, from the fingertips; cuesync.bending.library_fit)."""
        from .bending import LIBRARY_EFFECTS, element_span, ignition_frame, library_dir, library_fit
        from .composite import estimate_black_level
        from .media import read_video

        lib = library_dir()
        for e in LIBRARY_EFFECTS if lib is not None else ():
            d = lib / e
            if e in live or not (d / "element.mp4").exists():
                continue
            rc = json.loads((d / "receipt.json").read_text())
            el = read_video(d / "element.mp4").frames
            floor = estimate_black_level(el)
            ign = ignition_frame(el, floor, burst=rc["effect"]["behaviour"] == "burst") or 0
            rel, end = element_span(el, floor, ign)
            live[e] = {"effect": e, "behaviour": rc["effect"]["behaviour"], "url": f"/api/library/{e}/element.mp4",
                       "size": [int(el.shape[2]), int(el.shape[1])], "fps": 24, "ignition": int(ign), "release": int(rel),
                       "end": int(end), "floor": int(floor), "ref": rc["bending"]["ref"],
                       "fit": library_fit(el, floor, ign, (np.asarray(rc["bending"]["ref"]["xy"], np.float64),
                                                           np.asarray(rc["bending"]["ref"]["dir"], np.float64),
                                                           float(rc["bending"]["ref"]["scale"])), rc["effect"]["behaviour"])}
        return list(live.values())

    @app.get("/api/library/{effect}/element.mp4")
    def library_element(effect: str):
        from .bending import LIBRARY_EFFECTS, library_dir

        lib = library_dir()
        if lib is None or effect not in LIBRARY_EFFECTS or not (lib / effect / "element.mp4").exists():
            raise HTTPException(404, "no such element")
        return FileResponse(lib / effect / "element.mp4", media_type="video/mp4")

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
            try:
                e = load_effect(i)
            except (FileNotFoundError, ValueError) as exc:  # one broken file of your own doesn't hide the rest
                print(f"skipping effect {i!r}: {exc}", flush=True)
                continue
            out.append({"id": e.id, "fluid": e.fluid, "behaviour": e.behaviour, "stops_with_voice": e.voice_gate == "both",
                        "scene": e.scene is not None, "direction": e.scene.direction if e.scene else None,
                        "element": e.id in ELEMENT_EFFECTS})
        from .bending import MADE

        for i, (module, form) in MADE.items():  # thrown from a hand, made on the CPU (no render, no GPU)
            out.append({"id": i, "fluid": module, "behaviour": form, "stops_with_voice": form in ("ball", "orb", "arc"),
                        "scene": False, "direction": None, "element": False, "made": True})
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
                     plate_sample: str | None = Form(None), voice_sample: str | None = Form(None),
                     scale: float | None = Form(None), element: str | None = Form(None),
                     direction: float | None = Form(None), subject: str | None = Form(None),
                     cue: str = Form("voice"), holds: str | None = Form(None), strokes: str | None = Form(None),
                     describe: str | None = Form(None), object: str | None = Form(None),  # noqa: A002
                     sources: str | None = Form(None), frame: int | None = Form(None), reuse: str | None = Form(None)):
        from .bending import BENDING_PROMPTS, MADE, helpers_ready  # the elements a hand can throw (rendered, or made here)
        from .cues import SOURCES
        from .scene import Strokes

        if cue not in SOURCES:
            raise HTTPException(400, f"cue must be one of {list(SOURCES)}")
        hold_list = None
        max_s = float((shot_defaults or {}).get("max_plate_s", 10.0))
        if holds:
            try:
                hold_list = [(float(a), float(b)) for a, b in json.loads(holds)]
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, f"holds must be a JSON list of [press_s, release_s]: {exc}") from None
            if not hold_list or any(not (math.isfinite(a) and math.isfinite(b)) or b <= a or a < 0 or a >= max_s
                                    for a, b in hold_list):
                raise HTTPException(400, f"each hold must be a press time before its release, within the first {max_s:.0f} s")
        stroke_data = None
        if strokes:
            try:
                stroke_data = json.loads(strokes)
                parsed = Strokes.parse(stroke_data)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise HTTPException(400, f"strokes must be a JSON list of {{points: [[x, y], ...], width}}: {exc}") from None
            if parsed is None:
                raise HTTPException(400, "strokes need at least one point")
        describe = " ".join((describe or "").split())[:200] or None
        source_list = None  # call and response: one source per hit of a rhythm, in order
        if sources:
            try:
                source_list = [(float(a), float(b)) for a, b in json.loads(sources)]
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, f"sources must be a JSON list of [x, y] points: {exc}") from None
            if not 2 <= len(source_list) <= 4 or any(not (0 <= a <= 1 and 0 <= b <= 1) for a, b in source_list):
                raise HTTPException(400, "sources: 2 to 4 points, each with x and y in [0, 1]")
            if stroke_data:
                raise HTTPException(400, "send sources or a drawing, not both")
        auto = effect == "auto"
        made = effect in MADE  # lightning, water, air, an explosion: made on the CPU from a hand, never rendered
        try:
            eff = None if made else load_effect("fireball" if auto else effect)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, f"unknown effect {effect!r}: {exc}") from None
        if made and mode != "bending":
            raise HTTPException(400, f"{effect!r} is thrown from a hand (bending mode)")
        if mode not in ("scene", "render", "instant", "bending"):
            raise HTTPException(400, f"mode {mode!r} is not available here (available: {available_modes()})")
        if mode != "scene" and (stroke_data or describe or source_list):
            raise HTTPException(400, "drawing, describing the effect and several sources need scene mode")
        if mode == "scene" and (eff is None or eff.scene is None):
            raise HTTPException(400, f"{effect!r} has no scene mode")
        if mode == "bending" and not auto and effect not in BENDING_PROMPTS and not made:
            raise HTTPException(400, f"from a hand: one of {sorted(BENDING_PROMPTS) + sorted(MADE)} (or auto)")
        if mode not in ("scene", "bending") and not auto and effect not in ELEMENT_EFFECTS:
            raise HTTPException(400, f"{effect!r} only exists in scene mode")
        if frame is not None and (mode != "bending" or not 0 <= frame < 24 * 60):
            raise HTTPException(400, "frame (the frame you tapped the hand on) is for a bending shot, from 0")

        def need_live() -> None:
            """A shot that wasn't made ahead of time needs its renderer, and the GPU answering."""
            ahead = " (the sample shots made ahead of time still work: a sample clip at its marked spot, with a sample noise)"
            if mode == "bending":  # the GPU renders it; without one, an element rendered ahead of time (the tracker needed)
                from .bending import helpers_ready, library_dir

                if helpers_ready() and ((scenes and "render" in jq.renderers and render_ready()) or made or library_dir()):
                    return
                raise HTTPException(503, "a hand of your own needs the GPU, or the hand tracker installed for an instant "
                                         "shot (see the README's From your hand)")
            if mode == "scene" and not (scenes and "render" in jq.renderers):
                raise HTTPException(400, "Into the scene needs the GPU render API, which this server doesn't have" + ahead)
            if mode != "scene" and mode not in jq.renderers:
                raise HTTPException(400, f"mode {mode!r} is not available here (available: {available_modes()})")
            if mode in ("render", "scene") and not render_ready():
                raise HTTPException(503, "the GPU is offline right now; instant mode still works" + ahead)
        if direction is not None and not -180 <= direction <= 180:
            raise HTTPException(400, "direction must be in [-180, 180] degrees")
        subject = (subject or "").strip()[:120] or None
        if (plate is None) == (plate_sample is None):
            raise HTTPException(400, "send a plate (or plate_sample)")
        if cue == "voice" and (voice is None) == (voice_sample is None):
            raise HTTPException(400, "a voice cue needs a voice (or voice_sample)")
        if cue != "voice" and (voice is not None or voice_sample is not None):
            raise HTTPException(400, f"a {cue} cue takes no voice")
        if plate_sample is not None and plate_sample not in sample_files("plates"):
            raise HTTPException(400, f"no sample plate {plate_sample!r}")
        if voice_sample is not None and voice_sample not in sample_files("voices"):
            raise HTTPException(400, f"no sample voice {voice_sample!r}")
        job_id = uuid.uuid4().hex[:12]
        d = jq.root / job_id
        meta = sample_meta()
        pm = meta.get("plates", {}).get(plate_sample, {}) if plate_sample else {}
        if object is not None and not plate_sample:  # your own clip, and what it is
            from .gestures import OBJECTS

            if object not in OBJECTS:
                raise HTTPException(400, f"object must be one of {sorted(OBJECTS)}")
            pm = OBJECTS[object]
            if mode == "scene":
                subject = subject or pm.get("subject")
        if mode == "scene" and plate_sample:  # a sample plate knows what it shows and which way its source faces
            subject = subject or pm.get("subject")
            direction = direction if direction is not None else pm.get("direction")
        # a sample noise's gesture is known: decide now (and hit the cache), where a map says what it does; with no map
        # (your hand, a prop) the sound itself decides, as it would performed live (auto_effect, below)
        if auto and voice_sample and cue == "voice" and mode != "bending" and (pm.get("scene") or mode != "scene"):
            from .gestures import DEFAULTS, effect_for

            g = meta.get("voices", {}).get(voice_sample, {}).get("gesture")
            if g:
                aim = pm | ({"direction": direction} if direction is not None else {})
                effect = effect_for(g, aim, voice_sample) if mode == "scene" else DEFAULTS[g]
                if mode == "scene" and load_effect(effect).scene is None:
                    effect = DEFAULTS[g]
                auto, auto_gesture = False, g
            else:
                auto_gesture = None
        else:
            auto_gesture = None
        if auto and cue == "clip" and mode == "scene" and pm.get("clip_cue"):  # a sample clip knows what its sound does
            effect, auto = pm["clip_cue"]["effect"], False
        sample_voice = voice_sample if cue == "voice" else ("clip" if cue == "clip" and pm.get("clip_cue") else None)
        drawn_sample = None  # the sample's own drawing, made ahead of time: it replays like any other sample shot
        if plate_sample and stroke_data and mode == "scene" and cue == "voice" and not auto:
            drawn_sample = next((e for e in pm.get("drawn", []) if e.get("voice") == voice_sample
                                 and e.get("effect") == effect and same_strokes(e.get("strokes"), stroke_data)
                                 and describe in (None, e.get("describe"))), None)
            if drawn_sample:
                describe = drawn_sample.get("describe")  # rendered live, it says what the sample said
        if (plate_sample and sample_voice and face is None and timing is None and offset_s is None and scale is None
                and element is None and x is not None and y is not None and not auto and not source_list
                and (not stroke_data or drawn_sample) and (not describe or drawn_sample)):
            hit = cache.get(cache_key(mode, plate_sample, sample_voice, effect, (x, y), seed,
                                      direction if mode == "scene" else None, subject))
            hit_rc = json.loads((Path(hit["path"]) / "receipt.json").read_text()) if hit else {}
            if hit and mode == "bending" and (
                    (frame is not None and frame != int(hit.get("tap_frame", pm.get("hand", {}).get("frame", 0))))
                    or (direction is not None and direction != hit_rc.get("bending", {}).get("aim"))):
                hit = None
            if hit:  # this exact sample shot was made ahead of time: answer at once
                copy_shot(Path(hit["path"]), d)
                spec = ShotSpec(plate=sample_files("plates")[plate_sample],
                                voice=sample_files("voices")[voice_sample] if voice_sample else None,
                                effect=effect, point=(float(x), float(y)), seed=int(hit.get("seed", seed)),
                                mode="scene" if mode == "scene" else "element", direction=direction, subject=subject,
                                cue=cue, offset_s=float(hit_rc.get("inputs", {}).get("offset_s", 0.0)), **(shot_defaults or {}))
                if mode == "bending":
                    spec.extra["tap_frame"] = int(hit.get("tap_frame", pm.get("hand", {}).get("frame", 0)))
                    spec.extra["sample_emitter"] = True
                if auto_gesture:
                    spec.extra["gesture"] = auto_gesture
                job = Job(id=job_id, dir=d, spec=spec, mode=mode, status="done", stage="done", progress=1.0,
                          message="Made earlier", cached=True, title=hit["title"])
                jq.jobs[job_id] = job
                return job.public()
        def made_ahead_scene() -> list[dict]:
            """The scene shots made ahead of time for this sample plate at its own spot (any noise)."""
            if not (plate_sample and x is not None and y is not None):
                return []
            spot = f"|{float(x):.2f},{float(y):.2f}|"
            return [e for k, e in cache.items() if e.get("mode") == "scene" and e.get("plate") == plate_sample
                    and spot in k and not e.get("drawn")]

        offline = mode == "scene" and not (scenes and "render" in jq.renderers and render_ready())
        ahead_scene = made_ahead_scene() if offline and not (stroke_data or describe or source_list or face) else []
        # a sample hand with its element made ahead: with the GPU offline, your take plays that render on the hand; an
        # element made here is made on the sample's own track and matte, GPU or not (no tracker needed)
        ahead_hand = [e for e in cache.values() if e.get("mode") == "bending" and e.get("plate") == plate_sample] \
            if mode == "bending" and plate_sample and (made or not render_ready() or not helpers_ready()) else []
        if ahead_hand:  # (a fire made ahead replays; an element made here takes a shot's track, the fire's first)
            fire_ahead = [e for e in ahead_hand if e["effect"] not in MADE]
            ahead_hand = fire_ahead if auto else [e for e in ahead_hand if e["effect"] == effect] or \
                ((fire_ahead or ahead_hand) if made else [])
        if not (ahead_scene or ahead_hand):
            need_live()
        (d / "in").mkdir(parents=True)
        if plate is not None:
            plate_path = await save(plate, d / "in" / f"plate{Path(plate.filename or '.mp4').suffix or '.mp4'}")
        else:
            plate_path = sample_files("plates")[plate_sample]
        if voice is not None:
            voice_path = await save(voice, d / "in" / f"voice{Path(voice.filename or '.wav').suffix or '.wav'}")
        else:
            voice_path = sample_files("voices")[voice_sample] if voice_sample else None
        face_path = await save(face, d / "in" / f"face{Path(face.filename or '.webm').suffix or '.webm'}") if face else None
        t = {}
        if timing is not None:
            t = json.loads((await timing.read()).decode() or "{}")
            (d / "in" / "timing.json").write_text(json.dumps(t, indent=1))
        if offset_s is None and not t and mode == "bending" and plate_sample and voice_sample and pm.get("hand"):
            from .bending import sample_noise_offset  # a sample noise on a sample hand: on the hand's own strike
            from .media import probe_video

            offset_s = sample_noise_offset(voice_path, probe_video(plate_path).duration_s,
                                           int(pm["hand"].get("strike", pm["hand"].get("frame", 0))))
        if offset_s is None:
            offset_s = float(t.get("noise_offset_s", 0.0)) - float(t.get("input_latency_s", 0.0))
        point = (x, y) if x is not None and y is not None else tuple(t.get("emission_point") or (0.5, 0.7))
        if source_list:
            point = source_list[0]
        if not (0 <= point[0] <= 1 and 0 <= point[1] <= 1):
            raise HTTPException(400, "x and y must be in [0, 1]")
        if scale is not None and not 0.4 <= scale <= 2.0:
            raise HTTPException(400, "scale must be in [0.4, 2.0]")
        spec = ShotSpec(plate=plate_path, voice=voice_path, effect=effect, point=(float(point[0]), float(point[1])),
                        seed=seed, offset_s=offset_s, face=face_path, face_offset_s=t.get("face_offset_s"), label=label,
                        scale=scale, element=element if mode == "instant" else None,
                        mode="scene" if mode == "scene" else "element", direction=direction, subject=subject,
                        cue=cue, holds=hold_list, strokes=stroke_data, describe=describe, sources=source_list,
                        **(shot_defaults or {}))
        if auto_gesture:
            spec.extra["gesture"] = auto_gesture
        if mode == "bending":  # the hand you tapped, on the frame you tapped it (its sound and gesture pick "auto")
            spec.extra["tap_frame"] = int(frame or 0)
            was = jq.jobs.get(reuse) if reuse else None  # another effect for the same take: its hand as tracked
            if was is not None and was.status == "done" and (was.dir / "hand_track.json").exists():
                spec.extra["reuse"] = str(was.dir)
            elif was is not None and was.mode == "hand_prep" and was.status != "error":  # your tap, followed already
                # (its matte may still be in the making: the shot waits for it)
                spec.extra["reuse"], spec.extra["reuse_job"] = str(was.dir), was.id
        elif auto:  # the noise decides: its gesture through the plate's own map, or (no map) what the sound is
            from .pipeline import auto_effect

            try:
                auto_effect(spec, pm, voice_sample)
            except Exception as exc:
                raise HTTPException(400, f"couldn't read the cue: {exc}") from None
        if ahead_hand:  # the GPU is offline: the sample hand's element made ahead plays this take on it (no new render)
            sample_frame = int(pm.get("hand", {}).get("frame", ahead_hand[0].get("tap_frame", 0)))
            if frame is None:
                spec.extra["tap_frame"] = sample_frame
            if spec.point == tuple(pm.get("point", ())) and spec.extra["tap_frame"] == sample_frame:
                spec.extra["sample_emitter"] = True
            if auto:  # your noise may ask for an element made here (a buzz: lightning; a long BOOM with a strike)
                from .bending import instant_effect, load_track

                try:
                    t_ = load_track(Path(ahead_hand[0]["path"]) / "hand_track.json")
                    pick, _ = instant_effect(spec, t_, len(t_["xy"]))
                except Exception:  # (a noise it can't read: the replay below says why)
                    pick = None
                if pick in MADE:
                    spec.effect = pick
            same = ahead_hand[0]["effect"] == spec.effect  # (else an element made here, on the sample's hand)
            return jq.submit(Job(id=job_id, dir=d, spec=spec, mode="replay", title=ahead_hand[0]["title"] if same else None,
                                 replay={"src": Path(ahead_hand[0]["path"]), "bending": True})).public()
        if ahead_scene:  # the GPU is offline: the sample's scene shot made ahead plays this take (no new render)
            pick = next((e for e in ahead_scene if e["effect"] == spec.effect), ahead_scene[0])
            spec.effect = pick["effect"]
            return jq.submit(Job(id=job_id, dir=d, spec=spec, mode="replay", title=pick["title"],
                                 replay={"src": Path(pick["path"])})).public()
        if mode == "bending" and (made or not helpers_ready() or not (scenes and "render" in jq.renderers and render_ready())):
            mode = "bending_instant"  # no GPU (or none needed: made here): on the CPU worker
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

        # no-transform: the studio's proxy would otherwise gzip the stream and hold every event until the job ends
        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})

    @app.post("/api/compare")
    def compare(a: str = Form(...), b: str = Form(...)):
        """Two finished shots side by side, their performances' onsets on the same frame, played twice: first
        with take 1's voice (left side marked), then with take 2's (right side marked), so it works on a
        phone speaker too."""

        from .audio import CueSheet
        from .compare import side_by_side
        from .media import read_audio, read_video, write_video

        ja, jb = get(a), get(b)
        if ja.status != "done" or jb.status != "done" or ja.spec is None or jb.spec is None:
            raise HTTPException(409, "both shots must be finished")
        clips, envs, anchors, labels, voices = [], [], [], [], []
        for k, j in enumerate((ja, jb), 1):
            clips.append(read_video(j.dir / "composite.mp4").frames)
            envs.append(np.asarray(CueSheet.from_json(json.loads((j.dir / "cue.json").read_text())).envelope))
            guide = json.loads((j.dir / "guide.json").read_text())
            anchors.append(int(guide["anchor_frame"]))
            labels.append(f"take {k}: {guide['effect'].replace('_', ' ')}")
            voices.append(read_audio(j.dir / "composite.mp4"))
        pair = side_by_side(clips[0], clips[1], anchors[0], anchors[1], envs[0], envs[1], labels)
        n, h, w2 = pair.shape[:3]
        lead = max(anchors)
        spf = 48000 // 24

        def track(v: np.ndarray, pad_frames: int) -> np.ndarray:
            out = np.zeros(n * spf, np.float32)
            body = v[: max(0, n * spf - pad_frames * spf)]
            out[pad_frames * spf: pad_frames * spf + body.size] = body
            return out

        def marked(side: int) -> np.ndarray:
            frames = pair.copy()
            x0, x1 = (0, w2 // 2) if side == 0 else (w2 // 2, w2)
            t = max(4, w2 // 240)
            frames[:, :t, x0:x1] = (255, 176, 32)
            frames[:, -t:, x0:x1] = (255, 176, 32)
            frames[:, :, x0:x0 + t] = (255, 176, 32)
            frames[:, :, x1 - t:x1] = (255, 176, 32)
            return frames

        video = np.concatenate([marked(0), marked(1)])
        audio = np.concatenate([track(voices[0], lead - anchors[0]), track(voices[1], lead - anchors[1])])
        cid = uuid.uuid4().hex[:12]
        d = jq.root / cid
        d.mkdir(parents=True)
        write_video(d / "compare.mp4", video, fps=24, audio=audio, crf=18, preset="veryfast")
        job = Job(id=cid, dir=d, spec=None, mode="compare", status="done", stage="done", progress=1.0, message="Done")
        jq.jobs[cid] = job
        return job.public()

    @app.post("/api/jobs/{job_id}/retime")
    def retime(job_id: str, shift: int = Form(0), stop: int | None = Form(None)):
        """Move every hit by ``shift`` frames and/or stop a stop-with-the-voice effect at plate frame ``stop``,
        then composite again from the same rendered elements."""
        src = get(job_id)
        if src.status != "done" or not isinstance(src.spec, ShotSpec) or src.mode == "compare":
            raise HTTPException(409, "only a finished shot can be re-timed")
        if not -48 <= shift <= 48:
            raise HTTPException(400, "shift must be within two seconds (48 frames)")
        rid = uuid.uuid4().hex[:12]
        job = Job(id=rid, dir=jq.root / rid, spec=src.spec, mode="retime", title=src.title,
                  retime={"src": src.dir, "hit_shift": int(shift), "stop": stop})
        job.dir.mkdir(parents=True)
        return jq.submit(job).public()

    @app.post("/api/jobs/{job_id}/replay")
    async def replay(job_id: str, voice: UploadFile | None = File(None), voice_sample: str | None = Form(None),
                     cue: str = Form("voice"), holds: str | None = Form(None), offset_s: float | None = Form(None),
                     timing: UploadFile | None = File(None), together: str | None = Form(None)):
        """Perform a finished scene shot again with a new take: each new hit plays one of the render's hits, with
        no new render (a still plate: a moving one would need its camera held still). ``together``, a JSON list of
        other finished shots on the same plate, plays them all as one scene: one render per source, any rhythm.
        See cuesync.replay."""

        def rendered(job: Job) -> tuple[list[Path], dict]:  # the shots with the renders it plays, and its receipt
            rc = json.loads((job.dir / "receipt.json").read_text()) if (job.dir / "receipt.json").exists() else {}
            got = job.replay["src"] if job.replay else job.dir  # a shot performed again: its renders' shots
            bases = [Path(x) for x in got] if isinstance(got, list) else [Path(got)]
            if job.status != "done" or not isinstance(job.spec, ShotSpec) or rc.get("mode") != "scene" \
                    or not all((b / "render.mp4").exists() for b in bases):
                raise HTTPException(409, "only a finished shot rendered into the scene can be performed again")
            if not rc["plate"]["still"]:
                raise HTTPException(409, "performing a shot again works on a still plate (a photo or a held frame) for now")
            return bases, rc

        src = get(job_id)
        src_rc = json.loads((src.dir / "receipt.json").read_text()) if (src.dir / "receipt.json").exists() else {}
        bending = src.status == "done" and isinstance(src.spec, ShotSpec) and src_rc.get("mode") == "bending"
        if bending:  # a hand take: its element plays the new take on the same tracked hand
            if together:
                raise HTTPException(400, "a shot from your hand is performed again on its own")
            base, others = [src.dir], []
        else:
            base, rc = rendered(src)
            try:
                others = json.loads(together or "[]")
            except ValueError as exc:
                raise HTTPException(400, f"together must be a JSON list of shot ids: {exc}") from None
            if not isinstance(others, list) or not all(isinstance(x, str) for x in others):
                raise HTTPException(400, "together must be a JSON list of shot ids")
        from .pipeline import plate_box

        def render_id(b: Path) -> tuple:  # the shot's render (a cached sample is a copy of one at another path)
            f, r_ = b / "render.mp4", json.loads((b / "receipt.json").read_text())
            with open(f, "rb") as fh:
                return (tuple(r_["inputs"]["point"]), r_["effect"]["id"], r_["inputs"].get("seed"), f.stat().st_size,
                        hashlib.sha1(fh.read(1 << 20)).hexdigest())

        box = None if bending else plate_box(src.spec.plate, tuple(rc["inputs"]["point"]), src.spec.plate_size)
        seen = set() if bending else {render_id(b) for b in base}
        for other in others:
            more, orc = rendered(get(other))
            if orc["inputs"]["plate_sha"] != rc["inputs"]["plate_sha"]:
                raise HTTPException(409, "shots performed together must be on the same plate")
            if max(abs(u - v) for u, v in zip(plate_box(src.spec.plate, tuple(orc["inputs"]["point"]), src.spec.plate_size), box)) > 0.01:
                raise HTTPException(409, "these shots were cropped differently from your clip (tapped far apart on a clip "
                                         "that doesn't fit the frame): perform them separately")
            for b in more:
                if render_id(b) not in seen:
                    seen.add(render_id(b))
                    base.append(b)
        if cue not in ("voice", "hold"):
            raise HTTPException(400, "perform it again with your voice or press-and-hold")
        if cue == "voice" and (voice is None) == (voice_sample is None):
            raise HTTPException(400, "a voice cue needs a voice (or voice_sample)")
        if voice_sample is not None and voice_sample not in sample_files("voices"):
            raise HTTPException(400, f"no sample voice {voice_sample!r}")
        hold_list = None
        if cue == "hold":
            try:
                hold_list = [(float(a), float(b)) for a, b in json.loads(holds or "[]")]
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, f"holds must be a JSON list of [press_s, release_s]: {exc}") from None
            if not hold_list:
                raise HTTPException(400, "a hold cue needs at least one press")
        rid = uuid.uuid4().hex[:12]
        d = jq.root / rid
        (d / "in").mkdir(parents=True)
        voice_path = None
        if cue == "voice":
            voice_path = (await save(voice, d / "in" / f"voice{Path(voice.filename or '.wav').suffix or '.wav'}")
                          if voice is not None else sample_files("voices")[voice_sample])
        t = json.loads((await timing.read()).decode() or "{}") if timing is not None else {}
        if offset_s is None and not t and bending and cue == "voice" and voice_sample:
            from .bending import sample_noise_offset
            from .media import probe_video

            plate_sample = next((name for name, path in sample_files("plates").items() if path == src.spec.plate), None)
            pm = sample_meta().get("plates", {}).get(plate_sample, {})
            if pm.get("hand"):
                offset_s = sample_noise_offset(voice_path, probe_video(src.spec.plate).duration_s,
                                               int(pm["hand"].get("strike", pm["hand"].get("frame", 0))))
        if offset_s is None:
            offset_s = float(t.get("noise_offset_s", 0.0)) - float(t.get("input_latency_s", 0.0))
        extra = {k: v for k, v in src.spec.extra.items() if k not in ("auto", "gesture")}  # the new take wasn't read for an effect
        spec = dataclasses.replace(src.spec, voice=voice_path, cue=cue, holds=hold_list, offset_s=offset_s, face=None, extra=extra)
        job = Job(id=rid, dir=d, spec=spec, mode="replay", title=src.title,
                  replay={"src": base if len(base) > 1 else base[0], "bending": bending})
        return jq.submit(job).public()

    @app.post("/api/jobs/{job_id}/share")
    def share(job_id: str):
        """The finished shot as a vertical reel to share (cuesync.share): made once, then served like its other files."""
        from .share import share_reel

        job = get(job_id)
        if job.status != "done" or not (job.dir / "composite.mp4").exists() or not (job.dir / "receipt.json").exists():
            raise HTTPException(409, "only a finished shot can be shared")
        if not (job.dir / "share.mp4").exists():
            share_reel(job.dir, job.dir / "share.mp4")
        return job.public()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def files(job_id: str, name: str):
        job = get(job_id)
        if name not in RESULT_FILES or not (job.dir / name).exists():
            raise HTTPException(404, "no such file")
        if job.status != "done" and name not in EARLY_FILES:
            raise HTTPException(404, "not ready yet")
        return FileResponse(job.dir / name, media_type="application/json" if name.endswith(".json") else "video/mp4")

    if studio is not None and (Path(studio) / "index.html").exists():  # the studio itself, beside the API
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(studio), html=True), name="studio")
    return app


STUDIO_STATIC = Path(__file__).resolve().parent / "studio_static"  # tools/build_static_studio.py


def serve(*, workflow: Path | None, jobs_dir: Path, host: str = "127.0.0.1", port: int = 8765,
          allow_origins: list[str] | None = None, timeout: float = 600.0, library: Path | None = None,
          precomputed: Path | None = None) -> None:
    import uvicorn

    from .library import DEFAULT_DIR, LibraryRenderer
    from .pipeline import ComfyRenderer

    lib = Path(library) if library else DEFAULT_DIR
    instant = LibraryRenderer(lib) if (lib / "manifest.json").exists() else None
    full = ComfyRenderer(Path(workflow), timeout=timeout) if workflow else None
    if full is None and instant is None:
        raise SystemExit(f"nothing to serve: no --workflow and no element library at {lib}")
    app = create_app(full, jobs_dir, allow_origins, instant=instant, precomputed=precomputed,
                     local=host in ("127.0.0.1", "localhost", "::1"), studio=STUDIO_STATIC)
    uvicorn.run(app, host=host, port=port, log_level="info")
