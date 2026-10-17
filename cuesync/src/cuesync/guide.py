"""Voice-painted guide videos for the ainvfx-fluid LTX-2.5 IC-LoRA.

The control video is black except at keyframes (index 0, 1+8k, and the last frame), where flat colour
blobs mark the fluid volume. Blob size and placement follow the performer's absolute-loudness envelope,
and the whole performance shapes the effect: how long it holds, how much smoke it leaves, when it ends.

Two lessons from the real model shape the painting:
- A black frame means "unconstrained" to the model, never "gone", so an effect is ended by painting its
  end (fire collapsing into a thinning smoke wisp) and the last keyframe is never left empty. With
  ``canvas=True`` keyframes are painted on a dark canvas, as in the model author's samples, so a
  canvas-only keyframe can say "nothing here".
- Blobs smaller than one reference latent cell (64 px across) are invisible to the model, so no painted
  radius is below MIN_RADIUS_PX; a shape that would be smaller is dropped instead.
"""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from .audio import CueSheet
from .effects import EffectSpec
from .media import write_video

FIRE_OUTER = (255, 140, 0)
FIRE_INNER = (255, 220, 0)
FIRE_INNER_FRACTION = 0.55
SMOKE = (190, 190, 190)
STEAM = (255, 255, 255)
CANVAS = (18, 16, 14)  # keyframe background in the author's fire sample; black frames stay "unconstrained"
MIN_RADIUS_PX = 32  # the IC-LoRA sees the control at half resolution through a 32x VAE: 64 px is one latent cell
PLUME_FADE_FRAMES = 16  # "fade" tail: fire shrinks to nothing over this many frames after the voice ends
JET_FADE_FRAMES = 8
BURST_DECAY_FRAMES = 32  # fire shrinks to nothing this many frames after it stops holding
BURST_MAX_HOLD = 24  # a long "BOOOOM" holds the fireball at most this many frames past its second keyframe
MIN_LENGTH, MAX_LENGTH = 49, 121

Tail = Literal["wisp", "fade"]


@dataclass
class GuidePlan:
    frames: np.ndarray  # (length, H, W, 3) uint8
    anchor_frame: int  # plate frame mapped to element index 0
    keyframes: list[int]
    blobs: list[dict] = field(default_factory=list)  # per keyframe: {"e": int, "shapes": [...]}
    effect_id: str = ""
    size: tuple[int, int] = (576, 1024)
    voice_end: int | None = None  # element index where the performance ends (None: it outlasts the element)
    tail: str = "wisp"
    canvas: bool = False
    point: tuple[float, float] = (0.5, 0.7)  # emission point, normalised to the guide
    scale: float = 1.0  # size multiplier the effect was painted with

    def to_json(self) -> dict:
        return {
            "effect": self.effect_id,
            "size": list(self.size),
            "point": [round(float(self.point[0]), 4), round(float(self.point[1]), 4)],
            "scale": round(float(self.scale), 3),
            "length": int(self.frames.shape[0]),
            "anchor_frame": self.anchor_frame,
            "voice_end": self.voice_end,
            "tail": self.tail,
            "canvas": self.canvas,
            "keyframes": self.keyframes,
            "blobs": self.blobs,
        }


POP = {"burst": 0.65, "puff": 0.5}  # share of the peak size already reached at frame 1: bursts pop, they don't swell


def keyframe_indices(length: int, stride: int, with_one: bool = False) -> list[int]:
    """Valid ainvfx-fluid keyframes: 0, (1,) 1+stride*j (j>=1, up to length-8), and the last frame."""
    if length < 9 or length % 8 != 1:
        raise ValueError(f"element length must be 8k+1 (>= 9), got {length}")
    if stride <= 0 or stride % 8:
        raise ValueError(f"stride must be a positive multiple of 8, got {stride}")
    mids = []
    j = 1
    while 1 + stride * j <= length - 8:
        mids.append(1 + stride * j)
        j += 1
    return [0] + ([1] if with_one else []) + mids + [length - 1]


def anchor_frame_of(cue: CueSheet, effect: EffectSpec) -> int:
    return cue.frame_of(cue.hero_onset_s if effect.anchor == "hero" else cue.onsets_s[0])


def performance_length(cue: CueSheet, effect: EffectSpec, anchor: int | None = None) -> int:
    """Shortest valid element (8k+1) that covers the performance plus the effect's tail, within [49, 121]."""
    anchor = anchor_frame_of(cue, effect) if anchor is None else anchor
    end = cue.voice_end_frame
    if end is None:
        return MAX_LENGTH
    needed = max(0, end - anchor) + effect.tail_frames + 8
    length = 8 * int(np.ceil((needed - 1) / 8)) + 1
    return int(min(max(length, MIN_LENGTH), MAX_LENGTH))


def _smoothed(env: np.ndarray) -> np.ndarray:
    if env.size < 3:
        return env.astype(np.float64)
    padded = np.pad(env.astype(np.float64), 1, mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def _element_view(values: np.ndarray, anchor: int, length: int) -> np.ndarray:
    """values[anchor + e] for e in 0..length-1, zero outside the recording."""
    out = np.zeros(length)
    src = np.asarray(values, dtype=np.float64)
    lo, hi = max(0, anchor), min(src.size, anchor + length)
    if hi > lo:
        out[lo - anchor: hi - anchor] = src[lo:hi]
    return out


def _element_activity(cue: CueSheet, anchor: int, length: int) -> np.ndarray:
    active = np.zeros(length, dtype=bool)
    for a, b in cue.intervals():
        lo, hi = max(a - anchor, 0), min(b - anchor, length)
        if hi > lo:
            active[lo:hi] = True
    return active


def _shape(kind: str, cx: float, cy: float, rx: float, ry: float, color) -> dict:
    return {"kind": kind, "cx": round(cx, 2), "cy": round(cy, 2), "rx": round(rx, 2), "ry": round(ry, 2), "color": list(color)}


def _fluid_shapes(fluid: str, cx: float, cy: float, rx: float, ry: float, color=None) -> list[dict]:
    """Shapes for one fluid volume. Too small for the model to see (below MIN_RADIUS_PX) means no shape at all."""
    if min(rx, ry) < MIN_RADIUS_PX:
        return []
    if fluid == "fire":
        core = [_shape("fire_core", cx, cy, rx * FIRE_INNER_FRACTION, ry * FIRE_INNER_FRACTION, FIRE_INNER)]
        return [_shape("fire", cx, cy, rx, ry, FIRE_OUTER)] + (core if rx * FIRE_INNER_FRACTION >= 4 else [])
    default = SMOKE if fluid == "smoke" else STEAM
    return [_shape(fluid, cx, cy, rx, ry, color or default)]


def _thinning(base: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    """A dispersing fluid is painted darker as it thins (t: 0 dense .. 1 nearly gone)."""
    k = 1.0 - 0.65 * min(max(t, 0.0), 1.0)
    return tuple(int(round(c * k)) for c in base)


@dataclass
class _Performance:
    """What the voice did, on the element's timeline."""

    env: np.ndarray  # smoothed absolute envelope per element frame
    active: np.ndarray  # performing per element frame
    end: int | None  # element index where the performance ends
    peak_early: float  # loudness of the attack (first 1/3 s)
    active_frames: int  # how long the performer kept it up after the anchor
    energy: float  # sum of the envelope over active frames (a full-loudness second is 24)


def _performance(cue: CueSheet, anchor: int, length: int) -> _Performance:
    env = _smoothed(_element_view(cue.envelope, anchor, length))
    active = _element_activity(cue, anchor, length)
    end = cue.voice_end_frame
    end_e = None if end is None else max(1, end - anchor)
    if end_e is not None and end_e >= length:
        end_e = None
    raw = np.asarray(cue.envelope, dtype=np.float64)
    window = raw[max(0, anchor): max(0, anchor) + 8]
    return _Performance(env=env, active=active, end=end_e, peak_early=float(window.max()) if window.size else 0.0,
                        active_frames=int(active.sum()), energy=float(env[active].sum()))


def _burst_shapes(effect: EffectSpec, e: int, perf: _Performance, length: int, w: int, h: int,
                  x0: float, y0: float, keys: list[int], tail: Tail) -> list[dict]:
    r0 = max(effect.r_min * min(w, h), MIN_RADIUS_PX)  # sizes follow the short side: same look in landscape
    rpk = max((effect.r_min + (effect.r_max - effect.r_min) * perf.peak_early) * min(w, h), r0)
    rise_y = y0 - effect.rise * h * (e / (length - 1))
    later = [k for k in keys if k > 1]
    p1 = later[0] if len(later) > 1 else keys[-1]
    p2 = later[1] if len(later) > 2 else p1
    # A sustained "KA-BOOOOM" holds the fireball while the voice lasts (bounded); a clipped "pf." doesn't.
    hold = p2 + BURST_MAX_HOLD if perf.end is None else min(max(p2, perf.end), p2 + BURST_MAX_HOLD)
    shapes: list[dict] = []
    if e == 0:
        r = r0
    elif e == 1 and 1 in keys:
        r = r0 + (rpk - r0) * POP["burst"]  # the explosion has mostly happened one frame in
    elif e <= p1:
        r = r0 + (rpk - r0) * (e / p1)
    else:
        r = rpk if e <= hold else rpk * max(0.0, (hold + BURST_DECAY_FRAMES - e) / BURST_DECAY_FRAMES)
    if effect.secondary == "smoke" and e >= p2 and p2 > p1:
        t = (e - p2) / max(1, (length - 1) - p2)
        volume = 0.7 + 0.6 * min(perf.energy / 24.0, 1.0)  # more performance, more smoke
        sr_ = rpk * (0.6 + 0.7 * t) * volume
        color = _thinning(SMOKE, t) if tail == "wisp" else SMOKE
        scy = min(rise_y - 0.15 * h * t, y0 - 0.85 * sr_)
        shapes += _fluid_shapes("smoke", x0, scy, sr_, sr_, color)  # smoke under the fire
    shapes += _fluid_shapes(effect.fluid, x0, min(rise_y, y0 - 0.85 * r), r, r)
    return shapes


def _puff_shapes(effect: EffectSpec, e: int, perf: _Performance, length: int, w: int, h: int,
                 x0: float, y0: float, keys: list[int], tail: Tail, canvas: bool) -> list[dict]:
    r0 = max(effect.r_min * min(w, h), MIN_RADIUS_PX)  # sizes follow the short side: same look in landscape
    rpk = max((effect.r_min + (effect.r_max - effect.r_min) * perf.peak_early) * min(w, h), r0)
    rise_y = y0 - effect.rise * h * (e / (length - 1))
    later = [k for k in keys if k > 1]
    p1 = later[0] if len(later) > 1 else keys[-1]
    growth = 0.3 + 0.9 * min(perf.active_frames / 48.0, 1.0)  # a longer "pfffff" makes a bigger puff
    if e == 0:
        r = r0
    elif e == 1 and 1 in keys:
        r = r0 + (rpk - r0) * POP["puff"]
    elif e <= p1:
        r = r0 + (rpk - r0) * (e / p1)
    else:
        r = rpk * (1.0 + growth * (e - p1) / max(1, (length - 1) - p1))
    color = None
    if tail == "wisp":
        start = perf.end if perf.end is not None else length
        if e > start:
            t = (e - start) / max(1, (length - 1) - start)
            if canvas and t > 0.75:
                return []  # dispersed: a canvas-only keyframe
            color = _thinning(SMOKE if effect.fluid == "smoke" else STEAM, t)
    return _fluid_shapes(effect.fluid, x0, min(rise_y, y0 - 0.85 * r), r, r, color)


def _sustained_shapes(effect: EffectSpec, e: int, perf: _Performance, length: int, w: int, h: int,
                      x0: float, y0: float, tail: Tail, canvas: bool) -> list[dict]:
    """Plume (fire) and jet (steam): size follows the voice; a pause shrinks it; the end is painted."""
    span = effect.r_max - effect.r_min

    def size(i: int) -> float:
        level = perf.env[i] if perf.active[i] else 0.0
        return (effect.r_min + span * level) * (min(w, h) if effect.behaviour == "plume" else h)

    def jet(s: float, color=None) -> list[dict]:
        """A 1:8 jet whose LENGTH follows the voice; only its width is held at the visible minimum."""
        rx, ry = max(0.125 * s, MIN_RADIUS_PX), max(s / 2, MIN_RADIUS_PX)
        return _fluid_shapes(effect.fluid, x0, y0 - ry, rx, ry, color)

    def body(s: float, color=None) -> list[dict]:
        if effect.behaviour == "plume":
            r = max(s, MIN_RADIUS_PX)
            return _fluid_shapes(effect.fluid, x0, y0 - 2.5 * r, r, 2.5 * r, color)  # base stays at the emission point
        return jet(s, color)

    end = perf.end
    if end is None or e < end:
        return body(size(e))
    t_frames = e - end
    last = max(size(i) for i in range(max(0, end - 8), max(1, end)))  # what the gesture built up to, not its fade
    if tail == "fade":
        fade = PLUME_FADE_FRAMES if effect.behaviour == "plume" else JET_FADE_FRAMES
        s = last * max(0.0, 1.0 - t_frames / fade)
        if effect.behaviour == "plume":
            return _fluid_shapes(effect.fluid, x0, y0 - 2.5 * s, s, 2.5 * s)
        return _fluid_shapes(effect.fluid, x0, y0 - s / 2, 0.125 * s, s / 2) if s > 0 else []

    # "wisp": the source stops, what it already made rises and thins out.
    tail_len = max(8, (length - 1) - end)
    t = t_frames / tail_len
    if canvas and t > 0.75:
        return []  # gone: a canvas-only keyframe
    margin = MIN_RADIUS_PX  # the wisp rises, but at least one latent cell of it always stays in frame
    if effect.behaviour == "plume":
        r = max(last * 0.6, MIN_RADIUS_PX)
        if t_frames < 8:  # the first keyframe after the voice: a collapsing flame under its own smoke
            lift = min(0.12 * h * t + 2.5 * last, max(0.0, y0 - 2 * r - margin))
            return (_fluid_shapes("smoke", x0, y0 - lift - r, r, r, _thinning(SMOKE, t))
                    + _fluid_shapes("fire", x0, y0 - r, r, r))
        wisp = max(r * (1.0 - 0.5 * t), MIN_RADIUS_PX)
        lift = min(0.12 * h * t + 2.5 * last, max(0.0, y0 - wisp - margin))
        return _fluid_shapes("smoke", x0, y0 - lift - wisp, wisp, 1.4 * wisp, _thinning(SMOKE, t))
    wisp_rx = max(0.125 * last * (1.0 - 0.4 * t), MIN_RADIUS_PX)
    wisp_ry = max(0.5 * last * (1.0 - 0.4 * t), wisp_rx)
    cy = max(y0 - wisp_ry - 0.12 * h * t, margin)
    return _fluid_shapes(effect.fluid, x0, cy, wisp_rx, wisp_ry, _thinning(STEAM, t))


def _shapes_for(effect: EffectSpec, e: int, perf: _Performance, length: int, w: int, h: int,
                x0: float, y0: float, keys: list[int], tail: Tail, canvas: bool) -> list[dict]:
    if effect.behaviour == "burst":
        return _burst_shapes(effect, e, perf, length, w, h, x0, y0, keys, tail)
    if effect.behaviour == "puff":
        return _puff_shapes(effect, e, perf, length, w, h, x0, y0, keys, tail, canvas)
    if effect.behaviour in ("plume", "jet"):
        return _sustained_shapes(effect, e, perf, length, w, h, x0, y0, tail, canvas)
    raise ValueError(f"unknown behaviour {effect.behaviour!r}")


def _draw(shapes: list[dict], w: int, h: int, background=(0, 0, 0)) -> np.ndarray:
    img = Image.new("RGB", (w, h), tuple(background))
    draw = ImageDraw.Draw(img)
    for s in shapes:
        box = [s["cx"] - s["rx"], s["cy"] - s["ry"], s["cx"] + s["rx"], s["cy"] + s["ry"]]
        draw.ellipse(box, fill=tuple(s["color"]))
    return np.asarray(img, dtype=np.uint8)


def paint_guide(
    cue: CueSheet,
    effect: EffectSpec,
    point: tuple[float, float],
    *,
    size: tuple[int, int] = (576, 1024),
    length: int | None = None,
    tail: Tail = "wisp",
    canvas: bool = False,
    scale: float = 1.0,
) -> GuidePlan:
    """Paint the control video. ``point`` is the normalised emission point (x, y) in [0, 1]. ``scale``
    multiplies the effect's sizes (a small prop gets a smaller effect).

    ``length`` defaults to the performance (see performance_length). ``tail`` is how the effect ends after
    the voice stops: "wisp" paints the collapse into thinning smoke; "fade" just shrinks it (older guides).
    ``canvas`` paints keyframes on a dark canvas so an empty keyframe means "nothing here".
    """
    w, h = size
    if w % 64 or h % 64:
        raise ValueError(f"guide size must be multiples of 64, got {w}x{h}")
    if tail not in ("wisp", "fade"):
        raise ValueError(f"tail must be 'wisp' or 'fade', got {tail!r}")
    if scale != 1.0:
        effect = dataclasses.replace(effect, r_min=effect.r_min * scale, r_max=effect.r_max * scale)
    anchor = anchor_frame_of(cue, effect)
    length = performance_length(cue, effect, anchor) if length is None else length
    keys = keyframe_indices(length, effect.keyframe_stride, with_one=effect.behaviour in POP)
    perf = _performance(cue, anchor, length)
    x0, y0 = float(point[0]) * w, float(point[1]) * h

    frames = np.zeros((length, h, w, 3), dtype=np.uint8)
    blobs: list[dict] = []
    background = CANVAS if canvas else (0, 0, 0)
    for e in keys:
        shapes = _shapes_for(effect, e, perf, length, w, h, x0, y0, keys, tail, canvas)
        blobs.append({"e": e, "shapes": shapes})
        if shapes or canvas:
            frames[e] = _draw(shapes, w, h, background)
    return GuidePlan(frames=frames, anchor_frame=anchor, keyframes=keys, blobs=blobs, effect_id=effect.id,
                     size=(w, h), voice_end=perf.end, tail=tail, canvas=canvas, point=(float(point[0]), float(point[1])),
                     scale=float(scale))


def write_guide(plan: GuidePlan, path: str | Path, fps: int = 24) -> None:
    """Write the guide MP4 with a silent mono AAC track (the fluid graph decodes audio from it)."""
    n = plan.frames.shape[0]
    silence = np.zeros(int(round(n / fps * 48000)), dtype=np.float32)
    write_video(path, plan.frames, fps=fps, audio=silence, sample_rate=48000)
