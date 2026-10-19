"""HTTP API for the web app: submit a shot, follow its progress, fetch the results.

    cuesync serve --workflow workflows/fluid_element.api.json   # full renders through COMFY_BASE_URL + instant
    cuesync serve --instant-only                                # no GPU: the pre-rendered library only

POST /api/jobs (multipart): plate (or plate_sample), voice (or voice_sample) (+ optional face, timing.json from
     Dub Mode), effect, x, y, seed, mode ("render" | "instant").
GET  /api/jobs/{id}         status, stage, progress, result URLs
GET  /api/jobs/{id}/events  the same as server-sent events
GET  /api/jobs/{id}/files/{name}   final.mp4, composite.mp4, breakdown.mp4, guide.mp4, element.mp4, receipt.json
GET  /api/samples           sample plates and noises (try it without a camera or a microphone)
GET  /api/gallery           sample shots made ahead of time (cuesync precompute); a matching request is answered at once
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

RESULT_FILES = ("final.mp4", "composite.mp4", "breakdown.mp4", "guide.mp4", "element.mp4", "receipt.json", "poster.jpg",
                "compare.mp4", "cue.json", "guide.json", "guide_preview.mp4")
EARLY_FILES = ("guide_preview.mp4", "cue.json")  # shown while the shot is still being made
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
        return {"id": self.id, "mode": self.mode, "cached": self.cached, "title": self.title, "status": self.status,
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
        return "render" if mode in ("scene", "bending") else "retime" if mode == "replay" else mode

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

    def _work(self, mode: str) -> None:
        from .bending import bending_again, bending_job
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
                    bending_again(job.replay["src"], job.spec, job.dir, say)
                elif job.replay:
                    replay_scene_shot(job.replay["src"], job.spec, job.dir, say)
                elif job.mode == "bending":
                    reuse = job.spec.extra.pop("reuse", None)
                    bending_job(job.spec, self.renderers[mode], job.dir, progress=say, reuse=Path(reuse) if reuse else None)
                else:
                    make_shot(job.spec, self.renderers[mode], job.dir, progress=say)
                job.status = "done"
            except Exception as exc:  # report, keep serving
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"
                (job.dir / "error.txt").write_text(traceback.format_exc())
                self._emit(job, "error", job.progress, job.error)


def create_app(renderer, jobs_dir: Path, allow_origins: list[str] | None = None, shot_defaults: dict | None = None,
               instant=None, samples_dir: Path = SAMPLES, precomputed: Path | None = None):
    """``renderer``: full renders (None: an instant-only server). ``instant``: the library renderer, if any."""
    from .effects import _candidate_dirs, load_effect
    from .pipeline import ShotSpec

    app = FastAPI(title="Cue the Boom", version="0.1")
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
    # real footage first (in the order it was picked), then by plate
    order = lambda kv: (kv[1].get("mode") != "bending", not kv[1].get("real", False), kv[1].get("rank", 0),  # noqa: E731
                        kv[1].get("plate", ""), kv[1]["title"])  # your hand first, then real footage
    for key, entry in sorted(shown.items(), key=order):
        same = [g for g in gallery if g.title == entry["title"]]
        if same:  # one gallery tile per sample pair: the full render when there is one
            continue
        gid = "g" + hashlib.sha1(key.encode()).hexdigest()[:11]
        job = Job(id=gid, dir=Path(entry["path"]), spec=None, mode=entry["mode"], status="done", stage="done",
                  progress=1.0, message="Made earlier", cached=True, title=entry["title"])
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
        return {"ok": True, "renderer": getattr(renderer, "name", None), "modes": available_modes(),
                "made_ahead": sorted({e["mode"] for e in cache.values()}), "queued": jq.queued("render")}

    @app.get("/api/objects")
    def objects():
        from .gestures import OBJECTS

        return [{"id": k, "title": v["title"], "subject": v["subject"], "scene": v.get("scene", {})} for k, v in OBJECTS.items()]

    @app.get("/api/samples")
    def samples():
        meta = sample_meta()
        return {kind: [{"name": n, "url": f"/api/samples/{kind}/{n}", **meta.get(kind, {}).get(n, {})}
                       for n in sample_files(kind)] for kind in ("plates", "voices")}

    @app.get("/api/gallery")
    def gallery_list():
        return [g.public() for g in gallery]

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
        from .bending import BENDING_PROMPTS  # the elements a hand can throw
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
        try:
            eff = load_effect("fireball" if auto else effect)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, f"unknown effect {effect!r}: {exc}") from None
        if mode not in ("scene", "render", "instant", "bending"):
            raise HTTPException(400, f"mode {mode!r} is not available here (available: {available_modes()})")
        if mode != "scene" and (stroke_data or describe or source_list):
            raise HTTPException(400, "drawing, describing the effect and several sources need scene mode")
        if mode == "scene" and eff.scene is None:
            raise HTTPException(400, f"{effect!r} has no scene mode")
        if mode == "bending" and not auto and effect not in BENDING_PROMPTS:
            raise HTTPException(400, f"from a hand: one of {sorted(BENDING_PROMPTS)} (or auto)")
        if mode not in ("scene", "bending") and not auto and effect not in ELEMENT_EFFECTS:
            raise HTTPException(400, f"{effect!r} only exists in scene mode")
        if frame is not None and (mode != "bending" or not 0 <= frame < 24 * 60):
            raise HTTPException(400, "frame (the frame you tapped the hand on) is for a bending shot, from 0")

        def need_live() -> None:
            """A shot that wasn't made ahead of time needs its renderer, and the GPU answering."""
            ahead = " (the sample shots made ahead of time still work: a sample clip at its marked spot, with a sample noise)"
            if mode in ("scene", "bending") and not (scenes and "render" in jq.renderers):
                raise HTTPException(400, "Into the scene needs the GPU render API, which this server doesn't have" + ahead)
            if mode == "bending":
                if not render_ready() and effect != "smoke_puff":  # (an air blast is made, not rendered)
                    raise HTTPException(503, "the GPU is offline right now; instant mode still works")
                return
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
            if hit:  # this exact sample shot was made ahead of time: answer at once
                copy_shot(Path(hit["path"]), d)
                spec = ShotSpec(plate=sample_files("plates")[plate_sample],
                                voice=sample_files("voices")[voice_sample] if voice_sample else None,
                                effect=effect, point=(float(x), float(y)), seed=int(hit.get("seed", seed)),
                                mode="scene" if mode == "scene" else "element", direction=direction, subject=subject,
                                cue=cue, **(shot_defaults or {}))
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
        # a sample hand with its element made ahead: with the GPU offline, your take plays that render on the hand
        ahead_hand = [e for e in cache.values() if e.get("mode") == "bending" and e.get("plate") == plate_sample] \
            if mode == "bending" and plate_sample and not render_ready() else []
        if ahead_hand and not auto:
            ahead_hand = [e for e in ahead_hand if e["effect"] == effect]
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
        elif auto:  # the noise decides: its gesture through the plate's own map, or (no map) what the sound is
            from .pipeline import auto_effect

            try:
                auto_effect(spec, pm, voice_sample)
            except Exception as exc:
                raise HTTPException(400, f"couldn't read the cue: {exc}") from None
        if ahead_hand:  # the GPU is offline: the sample hand's element made ahead plays this take on it (no new render)
            spec.point, spec.extra["tap_frame"] = tuple(pm.get("point", spec.point)), int(ahead_hand[0].get("tap_frame", 0))
            return jq.submit(Job(id=job_id, dir=d, spec=spec, mode="replay", title=ahead_hand[0]["title"],
                                 replay={"src": Path(ahead_hand[0]["path"]), "bending": True})).public()
        if ahead_scene:  # the GPU is offline: the sample's scene shot made ahead plays this take (no new render)
            pick = next((e for e in ahead_scene if e["effect"] == spec.effect), ahead_scene[0])
            spec.effect = pick["effect"]
            return jq.submit(Job(id=job_id, dir=d, spec=spec, mode="replay", title=pick["title"],
                                 replay={"src": Path(pick["path"])})).public()
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
        if offset_s is None:
            offset_s = float(t.get("noise_offset_s", 0.0)) - float(t.get("input_latency_s", 0.0))
        extra = {k: v for k, v in src.spec.extra.items() if k not in ("auto", "gesture")}  # the new take wasn't read for an effect
        spec = dataclasses.replace(src.spec, voice=voice_path, cue=cue, holds=hold_list, offset_s=offset_s, face=None, extra=extra)
        job = Job(id=rid, dir=d, spec=spec, mode="replay", title=src.title,
                  replay={"src": base if len(base) > 1 else base[0], "bending": bending})
        return jq.submit(job).public()

    @app.get("/api/jobs/{job_id}/files/{name}")
    def files(job_id: str, name: str):
        job = get(job_id)
        if name not in RESULT_FILES or not (job.dir / name).exists():
            raise HTTPException(404, "no such file")
        if job.status != "done" and name not in EARLY_FILES:
            raise HTTPException(404, "not ready yet")
        return FileResponse(job.dir / name, media_type="application/json" if name.endswith(".json") else "video/mp4")

    return app


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
    app = create_app(full, jobs_dir, allow_origins, instant=instant, precomputed=precomputed)
    uvicorn.run(app, host=host, port=port, log_level="info")
