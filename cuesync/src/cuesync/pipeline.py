"""One shot, end to end: plate + performed noise -> guide -> element (Comfy) -> composite -> deliverables.

Deliverables in the job folder:
- final.mp4: the composite with the performer's voice, a face inset (Dub Mode) and the voice ribbon
- composite.mp4: the clean composite with the voice
- breakdown.mp4: plate | guide | element | composite, so anyone can see how the shot was made
- receipt.json: every number that went into it (cue sheet, guide, render, alignment, measurements)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .align import element_index_for, plan_alignment
from .audio import CueSheet, analyze
from .compare import PLAYHEAD, RIBBON_BG, RIBBON_FILL, RIBBON_FUTURE, RIBBON_H, _font
from .composite import composite, effect_options, first_shown_frame, voice_gates
from .conform import _crop_box, clip_duration, conform, crop_point
from .effects import load_effect
from .guide import GuidePlan, paint_guide, write_guide, write_guide_preview
from .matte import grabcut_matte
from .media import probe_video, read_audio, read_video, write_video
from .score import measure_composite_end, measure_composite_onset, measure_element

FPS = 24
FINISH_PRESET = "veryfast"  # the deliverables: ~3x faster than medium at these CRFs, visually the same
SR = 48000
Progress = Callable[[str, float, str], None]


class Renderer(Protocol):
    name: str

    def __call__(self, guide_path: Path, prompt: str, seed: int, out_path: Path) -> float:
        """Render the element for ``guide_path`` to ``out_path``; returns seconds spent. A renderer that can
        render into the scene also takes ``image=`` (a plate frame to start from)."""


@dataclass
class ComfyRenderer:
    """Renders through the Comfy SDK (COMFY_BASE_URL: a self-hosted comfy-api-proxy or Comfy Cloud)."""

    workflow: Path
    timeout: float = 600.0
    name: str = "comfy"

    scenes: bool = True  # can render into the scene (the graph has the image-to-video node)

    def __call__(self, guide_path: Path, prompt: str, seed: int, out_path: Path, image: Path | None = None) -> float:
        from .runner import render_element

        return render_element(guide_path, prompt=prompt, seed=seed, workflow_path=self.workflow,
                              out_path=out_path, timeout=self.timeout, image_path=image).seconds

    def ready(self) -> bool:
        """Does the Comfy API behind COMFY_BASE_URL answer its health check?"""
        import os
        import urllib.request

        base = os.environ.get("COMFY_BASE_URL", "").rstrip("/")
        if not base:
            return True  # Comfy Cloud defaults: trust the SDK
        try:
            with urllib.request.urlopen(f"{base}/api/v2/health", timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False

    def matte(self, frame: np.ndarray, point: tuple[float, float]) -> np.ndarray | None:
        """SAM 2.1 on the server (CueSync node pack) checked against GrabCut: a single point can make SAM
        return a part of the object (one pumpkin rib, the coffee inside a mug), so its matte is used only
        when it is at least 60 % as wide under the point as GrabCut's; otherwise GrabCut's."""
        from .runner import object_matte

        sam, cut = object_matte(frame, point), grabcut_matte(frame, point)
        if sam is None or cut is None:
            return sam if cut is None else cut
        ws, wc = object_width(sam, point), object_width(cut, point)
        return sam if ws is not None and (wc is None or ws >= 0.6 * wc) else cut


@dataclass
class ShotSpec:
    plate: Path
    voice: Path | None  # None: the cue is the clip's own sound or a hold (see cue)
    effect: str
    point: tuple[float, float]
    seed: int = 1
    offset_s: float = 0.0  # plate time of the voice recording's first sample
    face: Path | None = None
    face_offset_s: float | None = None  # face clip lag behind the voice recording (Dub Mode timing.json)
    label: str = ""
    max_plate_s: float = 10.0
    plate_size: tuple[int, int] | None = None  # None: 1080x1920 for portrait plates, 1920x1080 for landscape
    element_size: tuple[int, int] | None = None  # None: 576x1024 or 1024x576 to match
    matte: Callable[[np.ndarray, tuple[float, float]], np.ndarray | None] | None = None  # None: GrabCut
    scale: float | None = None  # effect size multiplier; None: from the object's size
    element: str | None = None  # instant mode: reuse this library element (compare takes fairly)
    multi_hit: bool = True  # "pf, pf, KA-BOOM": one burst per hit (bursts and puffs)
    mode: str = "element"  # "element": rendered on black, composited; "scene": rendered into the plate
    direction: float | None = None  # scene mode: degrees clockwise from up (None: the effect's default)
    subject: str | None = None  # scene mode: what the effect happens to ("the toy dinosaur"), for the prompt
    cue: str = "voice"  # what times it: "voice" (a performance), "clip" (a sound in the plate), "hold"
    holds: list | None = None  # cue "hold": [(press_s, release_s), ...] on the plate's timeline
    strokes: list | None = None  # scene mode: the effect drawn on the plate ([{"points": [[x, y], ...], "width"}])
    describe: str | None = None  # scene mode: the performer's own words for what happens (replaces the prompt)
    extra: dict = field(default_factory=dict)


@dataclass
class ShotResult:
    out_dir: Path
    final: Path
    composite: Path
    breakdown: Path
    receipt: dict


def shot_cue(spec: "ShotSpec", duration_s: float | None = None):
    """The shot's cue (cues.Cue) from its spec: the voice, the plate's own sound, or holds."""
    from .cues import load_cue

    return load_cue(spec.cue, plate=spec.plate, voice=spec.voice, offset_s=spec.offset_s, holds=spec.holds,
                    duration_s=duration_s or spec.max_plate_s, fps=FPS, sr=SR)


def _sha(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def place_audio(samples: np.ndarray, offset_s: float, n_frames: int, fps: int = FPS, sr: int = SR) -> np.ndarray:
    """Lay the performance onto a silent track as long as the plate, starting at plate time ``offset_s``."""
    out = np.zeros(int(round(n_frames / fps * sr)), dtype=np.float32)
    start = int(round(offset_s * sr))
    src = samples
    if start < 0:
        src, start = samples[-start:], 0
    end = min(out.size, start + src.size)
    if end > start:
        out[start:end] = src[: end - start]
    return out


def _fit(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Centre-crop to w:h and resize."""
    fh, fw = frame.shape[:2]
    target = w / h
    if fw / fh > target:
        cw = int(round(fh * target))
        frame = frame[:, (fw - cw) // 2:(fw - cw) // 2 + cw]
    else:
        ch = int(round(fw / target))
        frame = frame[(fh - ch) // 2:(fh - ch) // 2 + ch]
    return np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.LANCZOS))


def _face_track(face: Path, n_frames: int, plate_offset_s: float, face_offset_s: float | None, size: int):
    """Face frames per plate frame (square ``size``), or None outside the face clip."""
    info = probe_video(face)
    scale = min(1.0, 2.0 * size / max(1, min(info.width, info.height)))  # decode small: it becomes an inset
    clip = read_video(face, size=None if scale >= 1.0 else (max(2, round(info.width * scale / 2) * 2),
                                                             max(2, round(info.height * scale / 2) * 2)),
                      max_s=n_frames / FPS + abs(plate_offset_s) + abs(face_offset_s or 0.0) + 1.0)
    times = np.asarray(clip.times_s)
    lag = face_offset_s or 0.0
    out: list[np.ndarray | None] = []
    for f in range(n_frames):
        tf = f / FPS - plate_offset_s - lag  # plate time -> voice time -> face time
        if tf < times[0] - 0.5 / FPS or tf > times[-1] + 0.5 / FPS:
            out.append(None)
            continue
        out.append(_fit(clip.frames[int(np.argmin(np.abs(times - tf)))], size, size))
    return out


def _ribbons(env: np.ndarray, n: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """The voice ribbon drawn twice, once as 'already performed' and once as 'still to come'."""
    played = Image.new("RGB", (w, RIBBON_H), RIBBON_BG)
    future = Image.new("RGB", (w, RIBBON_H), RIBBON_BG)
    dp, df = ImageDraw.Draw(played), ImageDraw.Draw(future)
    last = max(n - 1, 1)
    for i in range(n):
        v = float(env[i]) if i < env.size else 0.0
        x0, x1 = int(i / last * (w - 1)), int((i + 1) / last * (w - 1))
        bar = int(v * (RIBBON_H - 8))
        if bar > 0:
            box = [x0, RIBBON_H - 4 - bar, max(x0, x1 - 1), RIBBON_H - 4]
            dp.rectangle(box, fill=RIBBON_FILL)
            df.rectangle(box, fill=RIBBON_FUTURE)
    return np.asarray(played), np.asarray(future)


def _finalise(frames: np.ndarray, env: np.ndarray, label: str, faces) -> np.ndarray:
    """Face inset (bottom left), effect label and the live voice ribbon with its playhead."""
    n, h, w = frames.shape[:3]
    out = np.empty((n, h + RIBBON_H, w, 3), dtype=np.uint8)
    pad = w // 30
    played, future = _ribbons(env, n, w)
    font = _font(max(18, w // 14))
    stamp = Image.new("RGBA", (w, max(18, w // 14) * 2 + 16), (0, 0, 0, 0))
    ImageDraw.Draw(stamp).text((w // 2, max(18, w // 14) // 2 + 8), label, font=font, fill=(255, 255, 255, 255), anchor="mt",
                               stroke_width=max(2, w // 168), stroke_fill=(0, 0, 0, 255))
    mask_cache: dict[int, Image.Image] = {}
    for t in range(n):
        img = Image.fromarray(frames[t])
        if faces is not None and faces[t] is not None:
            s = faces[t].shape[0]
            if s not in mask_cache:
                m = Image.new("L", (s, s), 0)
                ImageDraw.Draw(m).rounded_rectangle([0, 0, s - 1, s - 1], radius=s // 6, fill=255)
                mask_cache[s] = m
            y0, x0 = h - s - pad, pad
            img.paste(Image.fromarray(faces[t]), (x0, y0), mask_cache[s])
            ImageDraw.Draw(img).rounded_rectangle([x0, y0, x0 + s - 1, y0 + s - 1], radius=s // 6,
                                                  outline=(255, 255, 255), width=max(2, w // 270))
        img.paste(stamp, (0, 0), stamp)
        out[t, :h] = np.asarray(img)
        px = int(t / max(n - 1, 1) * (w - 1))
        out[t, h:, :px + 1] = played[:, :px + 1]
        out[t, h:, px + 1:] = future[:, px + 1:]
        out[t, h:, max(0, px - 1):px + 1] = PLAYHEAD
    return out


def _breakdown(plate: np.ndarray, guide: np.ndarray, element: np.ndarray, comp: np.ndarray, align, timing,
               keyframes: list[int] | None = None, guide_label: str = "2 voice-painted guide (last keyframe)",
               index_of: Callable[[int], int | None] | None = None,
               labels: tuple[str, str] = ("3 LTX-2.5 element", "4 composite")) -> np.ndarray:
    """2x2 grid at half size: plate | guide / element | composite, each on the plate's timeline. The guide
    only exists at its keyframes, so its tile holds the latest one (the plan's list: after an mp4 round trip
    codec noise makes every frame look non-empty)."""
    n, h, w = plate.shape[:3]
    hw, hh = w // 2, h // 2
    out = np.zeros((n, h, w, 3), np.uint8)
    font = _font(max(16, w // 36))
    blank = np.zeros((hh, hw, 3), np.uint8)
    cache: dict = {}

    def small(src: np.ndarray, i: int | None, key: str) -> np.ndarray:
        if i is None:
            return blank
        k = (key, i)
        if k not in cache:
            cache[k] = np.asarray(Image.fromarray(src[i]).resize((hw, hh), Image.Resampling.BILINEAR))
        return cache[k]

    keyed = sorted(keyframes) if keyframes else [i for i in range(guide.shape[0]) if guide[i].max() > 32]

    def latest_key(i: int | None) -> int | None:
        if i is None:
            return None
        prior = [k for k in keyed if k <= i]
        return prior[-1] if prior else None

    half = lambda f: np.asarray(Image.fromarray(f).reduce(2)) if f.shape[1] // 2 == hw else np.asarray(Image.fromarray(f).resize((hw, hh)))
    for t in range(n):
        idx = index_of(t) if index_of is not None else element_index_for(align, timing, t, element.shape[0])
        g_idx = latest_key(None if idx is None else min(idx, guide.shape[0] - 1))
        tiles = [(half(plate[t]), "1 plate (untouched)"),
                 (small(guide, g_idx, "g"), guide_label),
                 (small(element, idx, "e"), labels[0]),
                 (half(comp[t]), labels[1])]
        for k, (tile, text) in enumerate(tiles):
            y0, x0 = (k // 2) * hh, (k % 2) * hw
            img = Image.fromarray(tile.copy())
            ImageDraw.Draw(img).text((8, 6), text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            out[t, y0:y0 + hh, x0:x0 + hw] = np.asarray(img)
    return out


REF_OBJECT_WIDTH = {"burst": 0.45, "puff": 0.4, "plume": 0.35, "jet": 0.3}  # object width at which an effect
                                                                          # has its designed size (frame widths)


def object_width(occluder: np.ndarray | None, point: tuple[float, float]) -> float | None:
    """Width (fraction of the frame) of the object right under the emission point: the matte's run through
    the tapped column, in a band just below the point (so a table the matte leaked into doesn't count)."""
    if occluder is None:
        return None
    h, w = occluder.shape
    x, y0 = int(min(max(point[0], 0), 1) * (w - 1)), int(point[1] * h)
    widths = []
    for row in range(max(0, y0), min(h, y0 + int(0.08 * h))):
        on = occluder[row] > 0.5
        if not on[x]:
            continue
        left = x - np.argmin(on[x::-1]) if not on[: x + 1].all() else 0
        right = x + np.argmin(on[x:]) if not on[x:].all() else w
        widths.append(right - left)
    return float(np.median(widths)) / w if widths else None


def cut_gates(gates: tuple[np.ndarray, np.ndarray], at: int, release_fire: int, release_smoke: int):
    """The (fire, smoke) gates released from plate frame ``at``: linear ramps down, never reopened."""
    gf, gs = gates[0].copy(), gates[1].copy()
    for g, rel in ((gf, release_fire), (gs, release_smoke)):
        if at < g.size:
            ramp = np.clip(1.0 - (np.arange(g.size - at) + 1) / (rel + 1), 0.0, 1.0)
            g[at:] = np.minimum(g[at:], ramp)
    return gf, gs


def auto_scale(occluder: np.ndarray | None, point: tuple[float, float], behaviour: str) -> float:
    """Effect size for the object it comes from: smaller props get smaller effects (never below 0.55),
    bigger ones a little more (up to 1.2)."""
    width = object_width(occluder, point)
    if width is None:
        return 1.0
    return float(np.clip((width / REF_OBJECT_WIDTH.get(behaviour, 0.4)) ** 0.75, 0.55, 1.2))


def load_plate(path: Path, point: tuple[float, float], max_s: float, size: tuple[int, int] | None = None,
               element_size: tuple[int, int] | None = None):
    """The plate at 24 fps, cropped to 9:16 or 16:9 (by its orientation) around the tapped point, at most
    ``max_s`` long and never padded; returns (frames, the point in the cropped frame, the element size).

    Decodes at just enough resolution to cover the crop, so a 4K phone clip stays in bounded memory."""
    info = probe_video(path)
    landscape = info.width > info.height
    size = size or ((1920, 1080) if landscape else (1080, 1920))
    element_size = element_size or ((1024, 576) if landscape else (576, 1024))
    scale = max(size[0] / info.width, size[1] / info.height)
    dec = None if scale >= 1.0 else (max(2, round(info.width * scale / 2) * 2), max(2, round(info.height * scale / 2) * 2))
    clip = read_video(path, size=dec, max_s=max_s)
    src_h, src_w = clip.frames.shape[1:3]
    duration = min(clip_duration(clip, FPS), max_s)
    frames = conform(clip, size=size, fps=FPS, duration_s=duration, focus=point)
    del clip
    box = _crop_box(src_w, src_h, size[0], size[1], point)
    px, py = crop_point(point, src_w, src_h, box)
    return frames, (min(max(px, 0.0), 1.0), min(max(py, 0.0), 1.0)), element_size


def plate_notes(path: Path, max_s: float) -> list[str]:
    """What the receipt should say about the plate: a clip longer than the shot uses is cut, not refused."""
    d = probe_video(path).duration_s
    return [f"your clip is {d:.1f} s long; the shot uses its first {max_s:g} s"] if d > max_s + 0.05 else []


def _place(eff, anchor: int, intervals: list, frames: np.ndarray, n: int, next_hit: int | None = None,
           stop: int | None = None):
    """Measure an element and put it on the plate: (timing, alignment, voice gates). ``next_hit``: a later
    hit lands then, so this one makes way; ``stop``: a stop the user moved by hand."""
    timing = measure_element(frames, kind="fire" if eff.fluid == "fire" else "any")
    align = plan_alignment(anchor, timing, mode=eff.align, lead=eff.lead_frames)
    gates = voice_gates(n, intervals, anchor, eff.voice_gate, eff.release_fire, eff.release_smoke,
                        open_from=first_shown_frame(align, timing))
    if next_hit is not None:
        gates = cut_gates(gates, next_hit + 4, release_fire=8, release_smoke=18)
    if stop is not None:
        gates = cut_gates(gates, stop, release_fire=eff.release_fire, release_smoke=eff.release_smoke)
    return timing, align, gates


def make_shot(spec: ShotSpec, renderer: Renderer, out_dir: Path, progress: Progress | None = None) -> ShotResult:
    if spec.mode == "scene":
        from .scene_shot import make_scene_shot

        return make_scene_shot(spec, renderer, out_dir, progress)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    eff = load_effect(spec.effect)

    say("reading", 0.03, "Reading your plate")
    plate, point, element_size = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size, spec.element_size)
    n = plate.shape[0]

    say("listening", 0.08, "Listening to your performance" if spec.cue == "voice" else "Reading the cue")
    c = shot_cue(spec, n / FPS)
    samples, cue = c.samples, c.cue
    (out_dir / "cue.json").write_text(json.dumps(cue.to_json() | {"source": c.source}, indent=1))

    say("painting", 0.12, "Painting your voice into a guide")
    matte_fn = spec.matte or getattr(renderer, "matte", None) or grabcut_matte
    occluder = matte_fn(plate[0], point)
    if occluder is not None:  # kept, so a re-time doesn't segment again
        Image.fromarray((np.clip(occluder, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out_dir / "matte.png")
    scale = spec.scale if spec.scale is not None else auto_scale(occluder, point, eff.behaviour)
    # One effect per hit for a rhythmic performance ("pf, pf, KA-BOOM"), each from its own sub-performance.
    hits = cue.hits() if spec.multi_hit and eff.behaviour in ("burst", "puff") else []
    multi = len(hits) >= 2
    events = [(cue, out_dir)] if not multi else [(cue.for_hit(h), out_dir / "hits" / str(k + 1)) for k, h in enumerate(hits)]
    instant = getattr(renderer, "name", "") == "library"
    # In a rhythm, a quick quiet "pf" is a small pop and the long loud "KA-BOOM" the big one: each hit's size
    # follows its loudness and how long it sounds, relative to the biggest hit.
    sizes = [1.0] * len(events)
    if multi:
        span = [max(1, sum(b - a for a, b in ec.intervals())) for ec, _ in events]
        weight = [(h.strength / max(x.strength for x in hits)) ** 2 * (d / max(span)) for h, d in zip(hits, span)]
        sizes = [float(np.clip(0.35 + 0.65 * w ** 0.5, 0.35, 1.0)) for w in weight]
    rendered = []
    for k, (ecue, folder) in enumerate(events):
        folder.mkdir(parents=True, exist_ok=True)
        if multi:
            (folder / "cue.json").write_text(json.dumps(ecue.to_json(), indent=1))
        plan_k = paint_guide(ecue, eff, point, size=element_size, scale=scale * sizes[k])
        guide_k = folder / "guide.mp4"
        write_guide(plan_k, guide_k, fps=FPS)
        if k == 0:  # something to watch while the elements are made
            write_guide_preview(plan_k, out_dir / "guide_preview.mp4", fps=FPS)
        (folder / "guide.json").write_text(json.dumps(plan_k.to_json() | ({"element": spec.element} if spec.element else {}),
                                                      indent=1))
        which = f" (hit {k + 1} of {len(events)})" if multi else ""
        say("rendering", 0.18 + 0.5 * k / len(events), (f"Choosing a pre-rendered LTX-2.5 {eff.fluid} element" if instant
            else f"Rendering the {eff.fluid} from your voice with LTX-2.5") + which)
        element_k = folder / "element.mp4"
        seconds = renderer(guide_k, eff.prompt, spec.seed + k, element_k)
        last = dict(getattr(renderer, "last", None) or {})
        if last.get("guide"):  # instant mode: keep the element's own guide for the breakdown (and re-times)
            write_video(folder / "element_guide.mp4", last["guide"]["frames"], fps=FPS, crf=18, preset=FINISH_PRESET)
            (folder / "element_guide.json").write_text(json.dumps({"keyframes": last["guide"]["keyframes"]}))
        frames_k = read_video(element_k).frames
        timing_k, align_k, gates_k = _place(eff, plan_k.anchor_frame, ecue.intervals(), frames_k, n,
                                            next_hit=hits[k + 1].frame if multi and k + 1 < len(events) else None)
        rendered.append({"cue": ecue, "plan": plan_k, "guide": guide_k, "element": frames_k, "timing": timing_k,
                         "align": align_k, "gates": gates_k, "seconds": seconds, "last": last,
                         "strength": float(np.max(ecue.envelope)) if len(ecue.envelope) else 0.0})
    matte_source = "custom" if spec.matte is not None else ("server" if hasattr(renderer, "matte") else "grabcut")
    return _finish(spec, eff, plate, cue, samples, rendered, multi, occluder, scale, renderer.name, matte_source,
                   out_dir, say, t0)


def retime_shot(src: Path, spec: ShotSpec, out_dir: Path, hit_shift: int = 0, stop_frame: int | None = None,
                progress: Progress | None = None) -> ShotResult:
    """Re-time a finished shot without rendering again: move every hit by ``hit_shift`` frames and, for an
    effect that stops with the voice, stop it at plate frame ``stop_frame``. Composites and finishes anew."""
    src, out_dir = Path(src), Path(out_dir)
    old = json.loads((src / "receipt.json").read_text())
    if old.get("mode") == "scene":
        from .scene_shot import retime_scene_shot

        return retime_scene_shot(src, spec, out_dir, hit_shift=hit_shift, progress=progress)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    eff = load_effect(spec.effect)
    say("reading", 0.05, "Reading your plate")
    plate, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size, spec.element_size)
    n = plate.shape[0]
    samples = shot_cue(spec, n / FPS).samples
    cue = CueSheet.from_json(json.loads((src / "cue.json").read_text()))
    occluder = None
    if (src / "matte.png").exists():
        occluder = np.asarray(Image.open(src / "matte.png").convert("L"), dtype=np.float32) / 255.0
        shutil.copyfile(src / "matte.png", out_dir / "matte.png")
    for name in ("cue.json", "guide_preview.mp4"):
        if (src / name).exists():
            shutil.copyfile(src / name, out_dir / name)
    folders = sorted((src / "hits").iterdir(), key=lambda d: int(d.name)) if (src / "hits").is_dir() else [src]
    multi = len(folders) > 1
    say("compositing", 0.2, "Re-timing: no new render needed")
    anchors = [int(json.loads((f / "guide.json").read_text())["anchor_frame"]) + hit_shift for f in folders]
    rendered = []
    for k, folder in enumerate(folders):
        dst = out_dir / "hits" / folder.name if multi else out_dir
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("guide.json", "guide.mp4", "element.mp4", "element_guide.mp4", "element_guide.json", "cue.json"):
            if (folder / name).exists() and folder != dst:
                shutil.copyfile(folder / name, dst / name)
        ecue = CueSheet.from_json(json.loads((folder / "cue.json").read_text())) if multi else cue
        plan_k = GuidePlan.from_json(json.loads((folder / "guide.json").read_text()), read_video(folder / "guide.mp4").frames)
        plan_k.anchor_frame = anchors[k]
        frames_k = read_video(folder / "element.mp4").frames
        intervals = [(a + hit_shift, b + hit_shift) for a, b in ecue.intervals()]
        last_one = k == len(folders) - 1
        timing_k, align_k, gates_k = _place(
            eff, anchors[k], intervals, frames_k, n, next_hit=None if last_one else anchors[k + 1],
            stop=stop_frame if (last_one and stop_frame is not None and eff.voice_gate == "both") else None)
        last = {}
        if (folder / "element_guide.mp4").exists():
            last = {"guide": {"frames": read_video(folder / "element_guide.mp4").frames,
                              "keyframes": json.loads((folder / "element_guide.json").read_text())["keyframes"]}}
        rendered.append({"cue": ecue, "plan": plan_k, "guide": dst / "guide.mp4", "element": frames_k,
                         "timing": timing_k, "align": align_k, "gates": gates_k, "seconds": 0.0, "last": last,
                         "strength": float(np.max(ecue.envelope)) if len(ecue.envelope) else 0.0})
    retimed = {"from": src.name, "hit_shift_frames": hit_shift, "stop_frame": stop_frame}
    return _finish(spec, eff, plate, cue, samples, rendered, multi, occluder, float(old.get("scale", 1.0)),
                   old["render"]["renderer"], old.get("matte", {}).get("source", "grabcut"), out_dir, say, t0,
                   retimed=retimed, render_receipt=old["render"])


def _finish(spec, eff, plate, cue, samples, rendered, multi, occluder, scale, renderer_name, matte_source, out_dir,
            say, t0, retimed=None, render_receipt=None) -> ShotResult:
    """Composite the placed elements in time order, write the deliverables, measure, write the receipt."""
    n = plate.shape[0]
    emission = (rendered[0]["plan"].point[0] * plate.shape[2], rendered[0]["plan"].point[1] * plate.shape[1])
    say("compositing", 0.70, "Compositing onto your footage")
    comp = plate
    for r in rendered:  # layered in time order: a later burst goes over an earlier one's smoke
        comp = composite(comp, r["element"], r["align"], r["timing"], gates=r["gates"], emission=emission,
                         occluder=occluder, **effect_options(eff))
    main = max(rendered, key=lambda r: r["strength"])  # the loudest hit stands for the shot
    plan, guide_path, element, timing = main["plan"], main["guide"], main["element"], main["timing"]
    align, gates, render_s = main["align"], main["gates"], sum(r["seconds"] for r in rendered)
    if multi:  # the shot's own guide/element files are the loudest hit's
        for name in ("guide.mp4", "element.mp4"):
            shutil.copyfile(main["guide"].parent / name, out_dir / name)
        (out_dir / "guide.json").write_text(json.dumps(plan.to_json(), indent=1))
        guide_path = out_dir / "guide.mp4"
    elif retimed:
        (out_dir / "guide.json").write_text(json.dumps(plan.to_json(), indent=1))
    audio = place_audio(samples, spec.offset_s if spec.cue == "voice" else 0.0, n)
    comp_path = out_dir / "composite.mp4"
    write_video(comp_path, comp, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)

    say("finishing", 0.85, "Adding your face and the voice ribbon" if spec.face else "Adding the voice ribbon")
    faces = None
    if spec.face is not None:
        try:
            faces = _face_track(spec.face, n, spec.offset_s, spec.face_offset_s, min(plate.shape[1:3]) // 3)
        except Exception:  # a broken face clip must not cost the shot
            faces = None
    env = np.zeros(n)
    m = min(n, len(cue.envelope))
    env[:m] = np.asarray(cue.envelope)[:m]
    final_path, breakdown_path = out_dir / "final.mp4", out_dir / "breakdown.mp4"

    def write_final() -> None:
        write_video(final_path, _finalise(comp, env, spec.label or eff.id.replace("_", " "), faces), fps=FPS,
                    audio=audio, crf=16, preset=FINISH_PRESET)

    ref_guide = main["last"].get("guide")  # instant mode: the element's own guide

    def write_breakdown() -> None:
        if ref_guide:
            breakdown = _breakdown(plate, ref_guide["frames"], element, comp, align, timing, ref_guide["keyframes"],
                                   guide_label="2 guide of the pre-rendered element")
        else:
            breakdown = _breakdown(plate, read_video(guide_path).frames, element, comp, align, timing, plan.keyframes)
        write_video(breakdown_path, breakdown, fps=FPS, audio=audio, crf=18, preset=FINISH_PRESET)

    with ThreadPoolExecutor(max_workers=2) as pool:  # the two encodes overlap (x264 and numpy release the GIL)
        for fut in [pool.submit(write_final), pool.submit(write_breakdown)]:
            fut.result()

    small_plate, small_comp = plate[:, ::4, ::4], comp[:, ::4, ::4]  # visibility measured on a 1/4 grid
    onset = measure_composite_onset(small_plate, small_comp)
    hit_plate = next((f for f in range(n) if gates[0][f] + gates[1][f] > 0
                      and element_index_for(align, timing, f, element.shape[0]) == timing.hit_frame), None)
    stop = measure_composite_end(small_plate, small_comp)
    voice_end = cue.voice_end_frame
    lib = {k: v for k, v in main["last"].items() if k != "guide"}
    receipt = {
        "cuesync": __version__,
        "mode": "instant" if renderer_name == "library" else "element",
        "effect": eff.to_json(),
        "inputs": {"plate_sha": _sha(spec.plate), "voice_sha": _sha(spec.voice), "point": list(spec.point),
                   "seed": spec.seed, "offset_s": spec.offset_s, "face": spec.face is not None, "cue": spec.cue},
        "voice": {"onsets_s": [round(t, 3) for t in cue.onsets_s], "hero_onset_s": round(cue.hero_onset_s, 3),
                  "active_intervals": [list(iv) for iv in cue.intervals()], "voice_end_frame": voice_end,
                  "peak_frame": cue.peak_frame},
        "guide": {k: v for k, v in plan.to_json().items() if k != "blobs"},
        "render": render_receipt or {"renderer": renderer_name, "seconds": round(render_s, 2), "prompt": eff.prompt,
                                     "model": "LTX-2.5 22B distilled (int8) + AInVFX/ainvfx-fluid IC-LoRA",
                                     **({"library": lib} if lib else {})},
        "hits": [{"frame": r["plan"].anchor_frame, "strength": round(r["strength"], 3), "length": r["plan"].frames.shape[0],
                  "size": round(r["plan"].scale, 3),
                  **({"library": r["last"].get("entry")} if r["last"].get("entry") else {"render_s": round(r["seconds"], 1)})}
                 for r in rendered] if multi else None,
        "scale": round(scale, 3),
        "matte": {"source": matte_source, "found": occluder is not None,
                  "coverage": None if occluder is None else round(float(occluder.mean()), 4)},
        "alignment": {"mode": eff.align, "offset": align.offset, "first_element_frame": align.first,
                      "visible_onset": timing.visible_onset, "hit": timing.hit_frame, "peak": timing.peak},
        "measured": {"onset_error_frames": None if onset is None else onset - plan.anchor_frame,
                     # where the hit is seen (the first plate frame showing the element's hit with a gate open)
                     # against where it should be: on the sound, or lead_frames before it (light beats sound)
                     "hit_error_frames": None if hit_plate is None else hit_plate - (plan.anchor_frame - eff.lead_frames),
                     "stops_with_voice": eff.voice_gate == "both",
                     "stopped_within_plate": stop is not None and stop < n,
                     "stop_latency_frames": None if stop is None or voice_end is None else stop - voice_end},
        "warnings": plate_notes(spec.plate, spec.max_plate_s),
        "retimed": retimed,
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=breakdown_path, receipt=receipt)
