"""Voice-painted guide videos for the ainvfx-fluid LTX-2.5 IC-LoRA.

The control video is black except at keyframes (index 0, 1+8k, and the last frame), where flat colour
blobs mark the fluid volume. Blob size and placement follow the performer's absolute-loudness envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
END_LEVEL = 0.05  # envelope below this for 3 frames = the noise has ended
PLUME_FADE_FRAMES = 16
JET_FADE_FRAMES = 8


@dataclass
class GuidePlan:
    frames: np.ndarray  # (length, H, W, 3) uint8
    anchor_frame: int  # plate frame mapped to element index 0
    keyframes: list[int]
    blobs: list[dict] = field(default_factory=list)  # per keyframe: {"e": int, "shapes": [...]}
    effect_id: str = ""
    size: tuple[int, int] = (576, 1024)

    def to_json(self) -> dict:
        return {
            "effect": self.effect_id,
            "size": list(self.size),
            "length": int(self.frames.shape[0]),
            "anchor_frame": self.anchor_frame,
            "keyframes": self.keyframes,
            "blobs": self.blobs,
        }


def keyframe_indices(length: int, stride: int) -> list[int]:
    """Valid ainvfx-fluid keyframes: 0, 1+stride*j (j>=1, up to length-8), and the last frame."""
    if length < 9 or length % 8 != 1:
        raise ValueError(f"element length must be 8k+1 (>= 9), got {length}")
    if stride <= 0 or stride % 8:
        raise ValueError(f"stride must be a positive multiple of 8, got {stride}")
    mids = []
    j = 1
    while 1 + stride * j <= length - 8:
        mids.append(1 + stride * j)
        j += 1
    return [0] + mids + [length - 1]


def _smoothed(env: np.ndarray) -> np.ndarray:
    if env.size < 3:
        return env.astype(np.float64)
    padded = np.pad(env.astype(np.float64), 1, mode="edge")
    return (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0


def _element_envelope(cue: CueSheet, anchor: int, length: int) -> np.ndarray:
    """Absolute envelope seen from the element's timeline (index e = plate frame anchor+e), smoothed."""
    env = np.zeros(length)
    src = np.asarray(cue.envelope, dtype=np.float64)
    for e in range(length):
        p = anchor + e
        if 0 <= p < src.size:
            env[e] = src[p]
    return _smoothed(env)


def _noise_end(env: np.ndarray) -> int | None:
    """First element index where the envelope stays below END_LEVEL for 3 frames (after it was above)."""
    seen_sound = False
    for e in range(env.size):
        if env[e] >= END_LEVEL:
            seen_sound = True
        elif seen_sound and e + 2 < env.size and (env[e:e + 3] < END_LEVEL).all():
            return e
    return None


def _shape(kind: str, cx: float, cy: float, rx: float, ry: float, color) -> dict:
    return {"kind": kind, "cx": round(cx, 2), "cy": round(cy, 2), "rx": round(rx, 2), "ry": round(ry, 2), "color": list(color)}


def _fluid_shapes(fluid: str, cx: float, cy: float, rx: float, ry: float) -> list[dict]:
    if rx < 1 or ry < 1:
        return []
    if fluid == "fire":
        return [
            _shape("fire", cx, cy, rx, ry, FIRE_OUTER),
            _shape("fire_core", cx, cy, rx * FIRE_INNER_FRACTION, ry * FIRE_INNER_FRACTION, FIRE_INNER),
        ]
    return [_shape(fluid, cx, cy, rx, ry, SMOKE if fluid == "smoke" else STEAM)]


BURST_DECAY_FRAMES = 32  # fire shrinks to nothing this many frames after its hold keyframe (49 at stride 8)


def _blobs_for(effect: EffectSpec, e: int, env: np.ndarray, peak150: float, length: int, w: int, h: int,
               x0: float, y0: float, keys: list[int]) -> list[dict]:
    rmin, rmax = effect.r_min, effect.r_max
    r0 = rmin * w
    rpk = (rmin + (rmax - rmin) * peak150) * w
    rise_y = y0 - effect.rise * h * (e / (length - 1))
    shapes: list[dict] = []

    if effect.behaviour in ("burst", "puff"):
        # Stride-aware: peak on the first keyframe after 0, hold until the next, then decay (burst) / grow (puff).
        p1 = keys[1] if len(keys) > 2 else keys[-1]
        p2 = keys[2] if len(keys) > 3 else p1
        if e == 0:
            r = r0
        elif e <= p1:
            r = r0 + (rpk - r0) * (e / p1)
        elif effect.behaviour == "burst":
            r = rpk if e <= p2 else rpk * max(0.0, (p2 + BURST_DECAY_FRAMES - e) / BURST_DECAY_FRAMES)
        else:  # puff keeps growing to 1.6x by the last frame
            r = rpk * (1.0 + 0.6 * (e - p1) / max(1, (length - 1) - p1))
        if effect.behaviour == "burst" and effect.secondary == "smoke" and e >= p2 and p2 > p1:
            t = (e - p2) / max(1, (length - 1) - p2)
            sr_ = rpk * (0.6 + 0.7 * t)
            scy = rise_y - 0.15 * h * t
            shapes += _fluid_shapes("smoke", x0, scy, sr_, sr_)  # smoke under the fire
        shapes += _fluid_shapes(effect.fluid, x0, rise_y, r, r)
        return shapes

    end = _noise_end(env)
    if effect.behaviour == "plume":
        def plume_r(i: int) -> float:
            return (rmin + (rmax - rmin) * env[i]) * w
        if end is not None and e >= end:
            r = plume_r(end) * max(0.0, 1.0 - (e - end) / PLUME_FADE_FRAMES)
        else:
            r = plume_r(e)
        ry = 2.5 * r
        return _fluid_shapes(effect.fluid, x0, y0 - ry, r, ry)  # base stays at the emission point

    if effect.behaviour == "jet":
        def jet_len(i: int) -> float:
            return (rmin + (rmax - rmin) * env[i]) * h
        if end is not None and e >= end:
            length_px = jet_len(end) * max(0.0, 1.0 - (e - end) / JET_FADE_FRAMES)
        else:
            length_px = jet_len(e)
        return _fluid_shapes(effect.fluid, x0, y0 - length_px / 2, 0.125 * length_px, length_px / 2)

    raise ValueError(f"unknown behaviour {effect.behaviour!r}")


def _draw(shapes: list[dict], w: int, h: int) -> np.ndarray:
    img = Image.new("RGB", (w, h), (0, 0, 0))
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
    length: int = 121,
) -> GuidePlan:
    """Paint the control video. ``point`` is the normalised emission point (x, y) in [0, 1]."""
    w, h = size
    if w % 64 or h % 64:
        raise ValueError(f"guide size must be multiples of 64, got {w}x{h}")
    keys = keyframe_indices(length, effect.keyframe_stride)
    anchor_t = cue.hero_onset_s if effect.anchor == "hero" else cue.onsets_s[0]
    anchor = cue.frame_of(anchor_t)
    env = _element_envelope(cue, anchor, length)
    raw = np.asarray(cue.envelope, dtype=np.float64)
    window = raw[max(0, anchor): max(0, anchor) + 5]
    peak150 = float(window.max()) if window.size else 0.0
    x0, y0 = float(point[0]) * w, float(point[1]) * h

    frames = np.zeros((length, h, w, 3), dtype=np.uint8)
    blobs: list[dict] = []
    for e in keys:
        shapes = _blobs_for(effect, e, env, peak150, length, w, h, x0, y0, keys)
        blobs.append({"e": e, "shapes": shapes})
        if shapes:
            frames[e] = _draw(shapes, w, h)
    return GuidePlan(frames=frames, anchor_frame=anchor, keyframes=keys, blobs=blobs, effect_id=effect.id, size=(w, h))


def write_guide(plan: GuidePlan, path: str | Path, fps: int = 24) -> None:
    """Write the guide MP4 with a silent mono AAC track (the fluid graph decodes audio from it)."""
    n = plan.frames.shape[0]
    silence = np.zeros(int(round(n / fps * 48000)), dtype=np.float32)
    write_video(path, plan.frames, fps=fps, audio=silence, sample_rate=48000)
