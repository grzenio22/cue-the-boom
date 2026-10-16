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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageDraw

from . import __version__
from .align import element_index_for, plan_alignment
from .audio import analyze
from .compare import PLAYHEAD, RIBBON_BG, RIBBON_FILL, RIBBON_FUTURE, RIBBON_H, _font
from .composite import composite, effect_options, first_shown_frame, voice_gates
from .conform import _crop_box, clip_duration, conform, crop_point
from .effects import load_effect
from .guide import paint_guide, write_guide
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
        """Render the element for ``guide_path`` to ``out_path``; returns seconds spent."""


@dataclass
class ComfyRenderer:
    """Renders through the Comfy SDK (COMFY_BASE_URL: a self-hosted comfy-api-proxy or Comfy Cloud)."""

    workflow: Path
    timeout: float = 600.0
    name: str = "comfy"

    def __call__(self, guide_path: Path, prompt: str, seed: int, out_path: Path) -> float:
        from .runner import render_element

        return render_element(guide_path, prompt=prompt, seed=seed, workflow_path=self.workflow,
                              out_path=out_path, timeout=self.timeout).seconds


@dataclass
class ShotSpec:
    plate: Path
    voice: Path
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
    extra: dict = field(default_factory=dict)


@dataclass
class ShotResult:
    out_dir: Path
    final: Path
    composite: Path
    breakdown: Path
    receipt: dict


def _sha(path: Path) -> str:
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
               keyframes: list[int] | None = None) -> np.ndarray:
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
        idx = element_index_for(align, timing, t, element.shape[0])
        g_idx = latest_key(None if idx is None else min(idx, guide.shape[0] - 1))
        tiles = [(half(plate[t]), "1 plate (untouched)"),
                 (small(guide, g_idx, "g"), "2 voice-painted guide (last keyframe)"),
                 (small(element, idx, "e"), "3 LTX-2.5 element"),
                 (half(comp[t]), "4 composite")]
        for k, (tile, text) in enumerate(tiles):
            y0, x0 = (k // 2) * hh, (k % 2) * hw
            img = Image.fromarray(tile.copy())
            ImageDraw.Draw(img).text((8, 6), text, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            out[t, y0:y0 + hh, x0:x0 + hw] = np.asarray(img)
    return out


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


def make_shot(spec: ShotSpec, renderer: Renderer, out_dir: Path, progress: Progress | None = None) -> ShotResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    eff = load_effect(spec.effect)

    say("reading", 0.03, "Reading your plate")
    plate, point, element_size = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size, spec.element_size)
    n = plate.shape[0]

    say("listening", 0.08, "Listening to your performance")
    samples = read_audio(spec.voice, SR)
    cue = analyze(samples, SR, offset_s=spec.offset_s, fps=FPS)
    (out_dir / "cue.json").write_text(json.dumps(cue.to_json(), indent=1))

    say("painting", 0.12, "Painting your voice into a guide")
    plan = paint_guide(cue, eff, point, size=element_size)
    guide_path = out_dir / "guide.mp4"
    write_guide(plan, guide_path, fps=FPS)
    (out_dir / "guide.json").write_text(json.dumps(plan.to_json(), indent=1))

    say("rendering", 0.18, f"Rendering the {eff.fluid} with LTX-2.5")
    element_path = out_dir / "element.mp4"
    render_s = renderer(guide_path, eff.prompt, spec.seed, element_path)

    say("compositing", 0.70, "Compositing onto your footage")
    element = read_video(element_path).frames
    timing = measure_element(element, kind="fire" if eff.fluid == "fire" else "any")
    align = plan_alignment(plan.anchor_frame, timing, mode=eff.align, lead=eff.lead_frames)
    gates = voice_gates(n, cue.intervals(), plan.anchor_frame, eff.voice_gate, eff.release_fire, eff.release_smoke,
                        open_from=first_shown_frame(align, timing))
    emission = (point[0] * plate.shape[2], point[1] * plate.shape[1])
    occluder = spec.matte(plate[0], point) if spec.matte is not None else grabcut_matte(plate[0], point)
    comp = composite(plate, element, align, timing, gates=gates, emission=emission, occluder=occluder,
                     **effect_options(eff))
    audio = place_audio(samples, spec.offset_s, n)
    comp_path = out_dir / "composite.mp4"
    write_video(comp_path, comp, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)

    say("finishing", 0.85, "Adding your face and the voice ribbon")
    faces = None
    if spec.face is not None:
        try:
            faces = _face_track(spec.face, n, spec.offset_s, spec.face_offset_s, min(plate.shape[1:3]) // 3)
        except Exception:  # a broken face clip must not cost the shot
            faces = None
    env = np.zeros(n)
    m = min(n, len(cue.envelope))
    env[:m] = np.asarray(cue.envelope)[:m]
    final = _finalise(comp, env, spec.label or eff.id.replace("_", " "), faces)
    final_path = out_dir / "final.mp4"
    write_video(final_path, final, fps=FPS, audio=audio, crf=16, preset=FINISH_PRESET)
    del final

    guide_frames = read_video(guide_path).frames
    breakdown = _breakdown(plate, guide_frames, element, comp, align, timing, plan.keyframes)
    breakdown_path = out_dir / "breakdown.mp4"
    write_video(breakdown_path, breakdown, fps=FPS, audio=audio, crf=18, preset=FINISH_PRESET)
    del breakdown

    onset = measure_composite_onset(plate, comp)
    stop = measure_composite_end(plate, comp)
    voice_end = cue.voice_end_frame
    receipt = {
        "cuesync": __version__,
        "effect": eff.to_json(),
        "inputs": {"plate_sha": _sha(spec.plate), "voice_sha": _sha(spec.voice), "point": list(spec.point),
                   "seed": spec.seed, "offset_s": spec.offset_s, "face": spec.face is not None},
        "voice": {"onsets_s": [round(t, 3) for t in cue.onsets_s], "hero_onset_s": round(cue.hero_onset_s, 3),
                  "active_intervals": [list(iv) for iv in cue.intervals()], "voice_end_frame": voice_end,
                  "peak_frame": cue.peak_frame},
        "guide": {k: v for k, v in plan.to_json().items() if k != "blobs"},
        "render": {"renderer": renderer.name, "seconds": round(render_s, 2), "prompt": eff.prompt,
                   "model": "LTX-2.5 22B distilled (int8) + AInVFX/ainvfx-fluid IC-LoRA"},
        "matte": {"source": "custom" if spec.matte is not None else "grabcut", "found": occluder is not None,
                  "coverage": None if occluder is None else round(float(occluder.mean()), 4)},
        "alignment": {"mode": eff.align, "offset": align.offset, "first_element_frame": align.first,
                      "visible_onset": timing.visible_onset, "hit": timing.hit_frame, "peak": timing.peak},
        "measured": {"onset_error_frames": None if onset is None else onset - plan.anchor_frame,
                     "stopped_within_plate": stop is not None and stop < n,
                     "stop_latency_frames": None if stop is None or voice_end is None else stop - voice_end},
        "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=breakdown_path, receipt=receipt)
