"""Element bending: a performer's hand in the shot, an element out of it on the beat.

The hand's track (``hand_track.json``, from ``cuesync.hand_track``: where the emitter is, which way it points, how big
the hand is, its gesture and its strikes, frame by frame) meets the performance here:

- **Which element** (``bending_effect``): what the sound is (``gestures.heard``: a boom, a whoosh, a hiss, a puff) and
  what the hand is doing at the hit. Two fingers or a pointing finger with a long "ffff" or "fwooom" is a flamethrower;
  an open palm hissing is steam (or a blast of air, without steam); a fist's strike with a "BOOM" is a burst, a palm's
  too (a fireball out of the palm); an unvoiced "pf" from a palm is a blast of air; a snap blows out the candle you
  tapped. A combo of booms is a burst per strike. A hand doing nothing recognisable leaves it to the sound alone.
- **When** (``snap_launch``): the element goes off on the arm's lock (the strike's extension) when the voice's hit is
  close to it, since a punch that fires before it lands reads wrong. The window is lopsided because a sound heard before
  the sight of its cause is noticed sooner than one heard after: the motion may come up to 4 frames before the voice's hit
  (the voice then lags by at most 167 ms) but only 1 frame after (it leads by at most 42 ms). Outside it, the voice wins
  and the shot says the two disagree.
- **Where and how** (``make_bending_shot``): the ainvfx-fluid IC-LoRA learned from fires and smoke that stay put, with
  guides on an 8-frame grid over black. A guide that chased a moving hand would be out of what it knows (and would only
  move every third of a second). So the element is rendered where the hand is on the strike, over black, as a static
  source; then it is carried: each frame moved, turned and scaled with the tracked emitter (its base at once, its tip a
  few frames late, the way a flame trails a moving hand), slid in time so its ignition lands on the strike, added onto
  the clip in linear light (smoke laid over), and the hand is lit from where the emitter is now, not where it was.

Every decision lands in the receipt with what it was made from.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .gestures import effect_for_kind

POINTING = ("Victory", "Pointing_Up")  # two fingers, or one: a jet from the fingertips
SNAP_BEFORE, SNAP_AFTER = 4, 1  # frames at 24 fps the strike may come before / after the voice's hit and still take it
MISMATCH_NEAR = 12  # frames: a strike this close to a hit, but outside the snap window, is the hand off the beat
COMBO_S = 1.5  # booms closer together than this are one combo: a burst per strike

BENDING_PROMPTS = {  # the element alone, over black, in the LoRA's own wording ("ainvfxfluid, ... isolated on ...")
    "fire_breath": "ainvfxfluid, a roaring jet of ragged orange fire shooting {towards}, turbulent from its very start, "
                   "licking edges and embers, isolated on a pure black background",
    "fire_burst": "ainvfxfluid, a short explosive ball of clean bright flame bursting {towards} from a point, a white-hot "
                  "core, quickly dissipating into embers, isolated on a pure black background",
    "flame_plume": "ainvfxfluid, a tall clean flame rising from a point, a bright core and licking edges, isolated on a "
                   "pure black background",
    "smoke_puff": "ainvfxfluid, a fast puff of pale grey smoke pushed {towards} from a point, wispy edges, dissipating, "
                  "isolated on a pure black background",
    "steam_jet": "ainvfxfluid, a thin hissing jet of white steam streaming fast {towards} from a small point, narrow at "
                 "its source and fanning out into translucent curling wisps, isolated on a pure black background",
}
LAG_FRAMES = {"fire_breath": 10, "fire_burst": 2, "flame_plume": 5, "smoke_puff": 6, "steam_jet": 10}  # the far end's
# age in frames: how long ago the emitter threw it (a jet crosses its reach in about 0.4 s)
LENGTH = 6.0  # the element's reach in hand sizes (wrist to middle knuckle): a jet of about six hands
SIZE = {"jet": (6.0, 0.75), "burst": (1.3, 0.3), "plume": (1.8, 0.45), "puff": (2.5, 0.4)}  # per behaviour: its reach in
# hand sizes, and at most this share of the frame's short side (a close-up hand made a fireball fill the frame)
IGNITION = 0.15  # an element has ignited once its light is this share of its brightest frame's
LOST_FADE = 4  # frames over which an element fades out once the hand has been lost for more than 8
ROOT_FADE = 0.12  # share of the element's reach over which it comes up from nothing at the fingertips: the model draws
# a jet's root as a hard bright nozzle and a flame's foot as a flat edge, and fire from a hand has neither
ROOT_FADE_HANDS = {"steam_jet": 0.3}  # steam has no nozzle to hide: over the full fade it started a hand from the palm
CANVAS = 16  # the guide's ground: a dark grey, not black, so a keyframe before the hit says "nothing here yet" (over
# pure black the model took the empty keyframes as unconstrained and had steam there from the first frame)
PREGLOW = 4  # frames of the render's build-up shown before its ignition (a spark-up, not an element that pops on);
# earlier ones are dropped (a smoke render can start drifting long before its puff)


def load_track(path: str | Path) -> dict:
    """A hand_track.json (see docs/bending_contracts.md)."""
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(t.get("version", 1)) != 1:
        raise ValueError(f"hand track version {t.get('version')} is not one this build reads")
    return t


def gesture_at(track: dict, frame: int, around: int = 3) -> str | None:
    """The hand's gesture at ``frame``: the most frequent recognised one within +-``around`` frames (a single frame's
    label flickers), or None when nothing is recognised there or the hand is lost."""
    labels = track.get("gesture") or []
    src = track.get("src") or []
    lo, hi = max(0, frame - around), min(len(labels), frame + around + 1)
    seen = [labels[i] for i in range(lo, hi) if labels[i] not in (None, "None") and (i >= len(src) or src[i] != "lost")]
    return max(set(seen), key=seen.count) if seen else None


def bending_effect(kind: str | None, gesture: str | None, hits: int = 1, span_s: float = 0.0, snap: bool = False,
                   lit: bool = False, steam: bool = True) -> tuple[str | None, str]:
    """(effect, why) for a hand in the shot. ``kind``: the take's sound (gestures.heard); ``gesture``: the hand's at
    the hit (gesture_at); ``hits``/``span_s``: how many hits the take has and how long from the first to the last;
    ``snap``: the track saw a finger snap; ``lit``: a flame burns at the tapped spot; ``steam``: steam may be picked.
    None when there is no sound to read (the caller falls back to its own default)."""
    if snap:
        return "blow_out", "a snap: the candle goes out"
    if kind is None:
        return None, "no sound to read"
    combo = kind == "boom" and hits >= 2 and span_s <= COMBO_S
    if gesture in POINTING:
        if kind in ("hiss", "whoosh"):
            return "fire_breath", f"a long {kind} from pointed fingers: a flamethrower"
        if kind == "boom":
            return ("fire_breath", "strikes from the fingertips") if combo else ("fire_burst", "a boom from the fingertips")
    if gesture == "Open_Palm":
        if kind == "hiss":
            return ("steam_jet", "an open palm hissing: steam") if steam else ("smoke_puff", "an open palm hissing: a blast of air")
        if kind == "whoosh":
            return "flame_plume", "an open palm and a swelling voice: a flame rising from it"
        if kind == "boom":
            return "fire_burst", "a palm strike with a boom: a fireball" + (" per strike" if combo else "")
        return "smoke_puff", "an open palm and a puff: a blast of air"
    if gesture == "Closed_Fist" and kind in ("boom", "puff"):
        return "fire_burst", "a fist's strike: a burst" + (" per strike" if combo else "")
    why = "the hand's gesture isn't one we read" if gesture is None else f"{gesture} with a {kind}: by the sound alone"
    return effect_for_kind(kind, True, lit, aimed=True, rhythm=hits >= 2), why


def snap_launch(cue_frame: int, events: list[dict], fps: float = 24.0, kinds: tuple = ("extension", "open")) -> dict:
    """Where the element goes off for a voice hit at ``cue_frame``: the nearest strike event (an arm's lock, a palm
    opening) when it falls from SNAP_BEFORE frames before the hit to SNAP_AFTER after (scaled to ``fps``), else the hit
    itself, flagged as a mismatch. Returns {"cue", "motion", "frame", "used", "mismatch"}."""
    k = fps / 24.0
    strikes = [int(e["frame"]) for e in events if e.get("kind") in kinds]
    m = min(strikes, key=lambda f: abs(f - cue_frame)) if strikes else None
    ok = m is not None and cue_frame - SNAP_BEFORE * k <= m <= cue_frame + SNAP_AFTER * k
    near = m is not None and abs(m - cue_frame) <= MISMATCH_NEAR * k  # a strike seconds away is another gesture
    return {"cue": int(cue_frame), "motion": m, "frame": int(m if ok else cue_frame), "used": "motion" if ok else "cue",
            "mismatch": near and not ok}


# ------------------------------------------------------------------ carrying the element on the hand

def emitter_at(track: dict, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(position px, unit direction, hand size px) of the emitter at frame ``k`` (clamped to the track)."""
    k = int(np.clip(k, 0, len(track["xy"]) - 1))
    u = np.asarray(track["dir"][k], np.float64)
    return np.asarray(track["xy"][k], np.float64), u / max(float(np.hypot(*u)), 1e-6), float(track["scale"][k])


def anchored(track: dict, k: int, tap: tuple[float, float], aim: float | None = None) -> dict:
    """The track with its emitter moved to where the performer tapped on frame ``k`` (``tap`` in px), carried with the
    hand from there: the tap kept as an offset in the hand's own frame (along its direction and across it, in hand
    sizes), so it moves, turns and scales with the hand. A tracker's palm point is the landmarks' middle, which for an
    upturned palm seen from the side lies inside the hand, not on the palm the fire should rise from. ``aim``: the
    studio's aim arrow on frame ``k`` (degrees clockwise from up); the direction turns by the same angle on every
    frame, so the aim too is carried with the hand."""
    p, u, s = emitter_at(track, k)
    d = np.asarray(tap, np.float64) - p
    a, b = float(d @ u) / max(s, 1e-6), float(d @ np.array([-u[1], u[0]])) / max(s, 1e-6)
    turn = 0.0 if aim is None else (aim - angle_of(u) + 180.0) % 360.0 - 180.0
    c, sn = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    xy, dirs = [], []
    for t in range(len(track["xy"])):
        pt, ut, st = emitter_at(track, t)
        xy.append([float(v) for v in pt + st * (a * ut + b * np.array([-ut[1], ut[0]]))])
        dirs.append([float(ut[0] * c - ut[1] * sn), float(ut[0] * sn + ut[1] * c)])  # clockwise on screen (y down)
    return track | {"xy": xy, "dir": dirs, "anchor": {"frame": int(k), "along": round(a, 3), "across": round(b, 3),
                                                      "turn": round(turn, 1)}}


def angle_of(u) -> float:
    """A unit direction as degrees clockwise from straight up (the studio's ``direction``)."""
    return math.degrees(math.atan2(float(u[0]), -float(u[1])))


def towards_words(direction: float) -> str:
    a = math.radians(direction)
    ux, uy = math.sin(a), -math.cos(a)
    return "upwards" if uy < -0.7 else "downwards" if uy > 0.7 else "to the left" if ux < 0 else "to the right"


def ignition_frame(element: np.ndarray, floor: int, burst: bool = False) -> int | None:
    """The element frame where it ignites: the first whose light above its black is IGNITION of the brightest's; for a
    burst, the frame it grows fastest (a "BOOM" is a whoomph, its beat the blast, not the first spark)."""
    lum = np.array([max(0.0, float(f[::4, ::4].astype(np.float32).mean()) - floor) for f in element])
    if lum.max() <= 0:
        return None
    if burst and len(lum) > 2:
        return int(np.argmax(np.diff(lum))) + 1
    return int(np.argmax(lum >= IGNITION * lum.max()))


def pose_forward(ref: tuple, track: dict, t: float, size: float = 1.0) -> np.ndarray:
    """2x3: the rendered element's pixel -> plate frame ``t``'s pixel. The element was rendered with the emitter at
    ``ref`` (position, unit direction, hand size); at ``t`` it is moved, turned and scaled with the emitter (a similarity
    transform), and ``size`` times bigger (a quieter hit plays it smaller). A fractional ``t`` blends its two frames."""
    a = math.floor(t)
    if t - a > 1e-3:
        return pose_forward(ref, track, a, size) * (a + 1 - t) + pose_forward(ref, track, a + 1, size) * (t - a)
    p0, u0, s0 = ref
    p1, u1, s1 = emitter_at(track, int(a))
    turn = math.atan2(u1[1], u1[0]) - math.atan2(u0[1], u0[0])
    k = size * s1 / max(s0, 1e-6)
    c, s = math.cos(turn) * k, math.sin(turn) * k
    return np.array([[c, -s, p1[0] - (c * p0[0] - s * p0[1])], [s, c, p1[1] - (s * p0[0] + c * p0[1])]], np.float64)


def pose_inverse(ref: tuple, track: dict, t: int, size: float = 1.0) -> np.ndarray:
    """2x3: plate frame ``t``'s pixel -> the rendered element's pixel (pose_forward, inverted)."""
    return cv2.invertAffineTransform(pose_forward(ref, track, t, size))


SMEAR_PX = 3.0  # each slice is smeared over its share of the history in steps the far end moves at most this far:
# on a fast turn the far end travels further than the jet is long, and slices (however thin) left gaps between them,
# regular lobes when they were half a frame apart, a dotted screen when there were hundreds
MAX_SMEAR = 48


def _strip(level0: float, g: tuple, lo: float, hi: float, box: tuple) -> tuple | None:
    """The rows and columns of ``box`` (x0, x1, y0, y1) where level0 + g . (x, y) lies in [lo, hi] (a straight strip)."""
    x0, x1, y0, y1 = box
    gx, gy = g
    if abs(gx) >= abs(gy):  # columns from the strip's two edges at the box's top and bottom
        xs = [(v - level0 - gy * y) / gx for v in (lo, hi) if np.isfinite(v) for y in (y0, y1)]
        if len(xs) < 4:  # one side open: it runs to the box's edge that way
            xs += [x0 if (gx > 0) == (not np.isfinite(lo)) else x1]
        a, b = max(x0, int(np.floor(min(xs))) - 1), min(x1, int(np.ceil(max(xs))) + 2)
        return None if a >= b else (a, b, y0, y1)
    ys = [(v - level0 - gx * x) / gy for v in (lo, hi) if np.isfinite(v) for x in (x0, x1)]
    if len(ys) < 4:
        ys += [y0 if (gy > 0) == (not np.isfinite(lo)) else y1]
    a, b = max(y0, int(np.floor(min(ys))) - 1), min(y1, int(np.ceil(max(ys))) + 2)
    return None if a >= b else (x0, x1, a, b)


def warp_element(frame: np.ndarray, track: dict, ref: tuple, t: int, lag: int, size: float = 1.0,
                 blur: bool = True, since: int | None = None) -> np.ndarray:
    """An element frame (float32, over black, at the plate's size, rendered with the emitter at ``ref``) carried onto
    plate frame ``t`` with its emission history: in thin slices along its length, its base on the emitter's pose now and
    each slice further along on the pose the emitter had when that part left the hand, up to ``lag`` frames before for
    the far end (never before ``since``, the strike). Fire already thrown keeps flying the way it was thrown while the
    hand swings on: one pose for the whole element made it a strip swung on a pivot, and one remap with a pose per pixel
    tore it apart on a fast turn. There are as many slices as it takes for the far end to move at most SLICE_PX between
    neighbours (a still hand needs few, a whip a few hundred, each warped only where it is), their weights overlap and
    sum to 1. Each slice is smeared over its own share of the history (and with ``blur`` half a frame more, as a
    180-degree shutter would), in as many steps as it takes for its far edge to move at most SMEAR_PX: a stream thrown
    on a fast turn is stretched along its path, not broken into pieces."""
    h, w = frame.shape[:2]
    p0, u0, s0 = ref
    lo_t = -np.inf if since is None else float(since)
    out = np.zeros_like(frame)
    lit = frame[::4, ::4].max(-1) > 1e-4  # where the element is, from a quarter-size look
    ys, xs = np.nonzero(lit)
    if ys.size == 0:
        return out
    box = (max(0, 4 * xs.min() - 4), min(w, 4 * xs.max() + 8), max(0, 4 * ys.min() - 4), min(h, 4 * ys.max() + 8))
    reach = max(LENGTH * s0, 1.0)
    pose = lambda c: pose_forward(ref, track, max(t - c, lo_t), size)  # noqa: E731
    n = 2 * lag + 1 if lag > 0 else 1
    tip = np.array([p0[0] + u0[0] * reach, p0[1] + u0[1] * reach, 1.0])
    g = (u0[0] * lag / reach, u0[1] * lag / reach)
    level0 = -(p0[0] * g[0] + p0[1] * g[1])  # level(x, y) = level0 + g . (x, y): 0 at the emitter, lag at the far end
    step = lag / (n - 1) if n > 1 else 1.0
    for i, c in enumerate(np.linspace(0.0, lag, n)):
        lo = -np.inf if i == 0 else c - step  # the first slice takes all behind the emitter, the last all beyond
        hi = np.inf if i == n - 1 else c + step
        b = _strip(level0, g, lo, hi, box) if n > 1 else box
        if b is None:
            continue
        x0, x1, y0, y1 = b
        part = frame[y0:y1, x0:x1]
        if n > 1:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            lev = np.clip(level0 + g[0] * xx + g[1] * yy, 0, lag)
            part = part * np.clip(1.0 - np.abs(lev - c) / step, 0.0, 1.0)[..., None]
        ph, pw = part.shape[:2]
        a, z = max(0.0, c - step / 2), min(float(lag), c + step / 2) + (0.5 if blur else 0.0)  # its share of the history
        far = np.hypot(*(pose(z) @ tip - pose(a) @ tip)) * min(1.0, (c + step) / max(lag, 1e-6))  # its far edge's path
        m = int(np.clip(np.ceil(far / SMEAR_PX), 1, MAX_SMEAR)) if z > a else 1
        times = np.linspace(a, z, m) if m > 1 else ((a + z) / 2,)
        for cc in times:
            A = pose(cc)
            A[:, 2] += A[:, 0] * x0 + A[:, 1] * y0  # from the slice's own corner
            corners = np.array([[0, 0, 1], [pw, 0, 1], [0, ph, 1], [pw, ph, 1]], np.float64) @ A.T
            dx0, dy0 = max(0, int(np.floor(corners[:, 0].min())) - 1), max(0, int(np.floor(corners[:, 1].min())) - 1)
            dx1, dy1 = min(w, int(np.ceil(corners[:, 0].max())) + 2), min(h, int(np.ceil(corners[:, 1].max())) + 2)
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            A[:, 2] -= (dx0, dy0)
            out[dy0:dy1, dx0:dx1] += cv2.warpAffine(part, A.astype(np.float32), (dx1 - dx0, dy1 - dy0),
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0) * (1.0 / len(times))
    return out


def render_spot(p: np.ndarray, u: np.ndarray, reach: float, w: int, h: int, margin: float = 0.1) -> np.ndarray:
    """Where the element is rendered: the emitter moved (along each axis, as little as it takes) so the element, from
    it to ``reach`` px along ``u``, stays ``margin`` inside the frame; centred on an axis it can't fit. Rendered at a
    hand by the frame's edge, the fire was cut along that edge, and the cut showed once it was carried away from it."""
    p = np.asarray(p, np.float64)
    tip = p + np.asarray(u, np.float64) * reach
    q = p.copy()
    for i, size in enumerate((w, h)):
        lo, hi = margin * size, (1.0 - margin) * size
        a, b = min(p[i], tip[i]), max(p[i], tip[i])
        q[i] += (lo + hi - a - b) / 2.0 if b - a > hi - lo else max(lo - a, 0.0) - max(b - hi, 0.0)
    return q


def border_fade(h: int, w: int, share: float = 0.06) -> np.ndarray:
    """(h, w) float32: 1 inside, falling to 0 over ``share`` of the short side at the render's own edges. The model
    draws a jet right up to its frame's edge; carried, that edge showed as a straight cut across the fire."""
    m = max(1.0, share * min(h, w))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
    return np.clip(d / m, 0, 1) ** 1.5


def root_fade(h: int, w: int, ref: tuple, hands: float = ROOT_FADE * LENGTH) -> np.ndarray:
    """(h, w) float32 in the element's own frame: 0 at and behind the emitter, rising to 1 ROOT_FADE of the reach along
    it, so the element comes up from nothing just past the fingertips."""
    p0, u0, s0 = ref
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    along = (xx - p0[0]) * u0[0] + (yy - p0[1]) * u0[1]
    return np.clip(along / max(hands * s0, 1.0), 0, 1).astype(np.float32)


def root_holdout(track: dict, t: int, h: int, w: int, mask: np.ndarray | None) -> np.ndarray:
    """Where the hand hides the element's root (float 0..1): within a quarter of a hand size of the emitter, the hand
    itself (its matte; without one, the side behind the fingertips, coming in over 0.3 of a hand size so the base fades
    into the hand instead of being cut off along a line), so the flame leaves the fingers rather than sitting on them."""
    p, u, s = emitter_at(track, t)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - p[0], yy - p[1]
    disc = np.clip(1.0 - (np.hypot(dx, dy) - 0.25 * s) / max(0.25 * s, 1.0), 0, 1)
    back = -(dx * u[0] + dy * u[1])  # how far behind the emitter, along its direction
    if mask is not None:  # the hand where it is behind the emitter: the fingers in front of a flame's root hide it, the
        # palm the flame rises in front of doesn't (the whole matte put fire in a cupped hand behind it)
        return disc * mask * np.clip(0.5 + back / max(0.3 * s, 1.0), 0, 1)
    return disc * np.clip(back / max(0.3 * s, 1.0), 0, 1)


def track_slice(track: dict, a: int, b: int) -> dict:
    """The track for frames a..b-1 only (the per-frame lists cut; events and sizes kept)."""
    per = ("xy", "dir", "scale", "conf", "src", "speed", "landmarks", "gesture")
    return track | {k: list(track[k][a:b]) for k in per if k in track}


def point_light(plate_lin: np.ndarray, element_lin: np.ndarray, track: dict, t: int, fluid: str,
                mask: np.ndarray | None) -> np.ndarray:
    """Until cuesync.fx_light lands: a point light half a hand ahead of the emitter, as bright and as coloured as the
    element is on this frame, strong on the hand (its matte), weak on the room; added, never darkening."""
    if fluid == "smoke":
        return plate_lin
    h, w = plate_lin.shape[:2]
    p, u, s = emitter_at(track, t)
    c = p + 0.5 * s * u
    power = float(element_lin.mean())
    if power <= 1e-5:
        return plate_lin
    colour = element_lin.reshape(-1, 3).mean(0)
    colour = colour / max(float(colour.max()), 1e-6)
    if fluid == "fire":
        colour = 0.5 * colour + 0.5 * np.array([1.0, 0.55, 0.22])
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    r = np.hypot(xx - c[0], yy - c[1])
    room = 1.0 / (1.0 + (r / (6.0 * s)) ** 2)
    near = 1.0 / (1.0 + (r / (1.5 * s)) ** 2) * (mask if mask is not None else 0.0)
    gain = min(power * 25.0, 1.5)
    return plate_lin + plate_lin * (gain * (0.6 * room + 1.4 * near))[..., None] * colour.astype(np.float32)


def air_puff(h: int, w: int, ref: tuple, seed: int = 0, frames: int = 14) -> np.ndarray:
    """The air blast's visible part, made rather than rendered (uint8 [frames, h, w, 3] over black): a puff of dust
    that leaves the palm along its aim, travels about three hands while it spreads and thins out, gone in about half a
    second. An air blast is a short event; a rendered smoke cloud parked one hand from the palm for two seconds. The
    pressure front that bends what is behind it is fx_light.air_wave."""
    p0, u0, s0 = ref
    rng = np.random.default_rng(seed)
    q = 4
    hq, wq = max(2, h // q), max(2, w // q)
    noise = cv2.GaussianBlur(rng.random((hq, wq)).astype(np.float32), (0, 0), max(1.0, 0.04 * s0 / q))
    n_ = max(3, int(0.6 * s0 / q) | 1)  # streaked along the aim: dust flung out, not a soft glow
    line = np.zeros((n_, n_), np.float32)
    cv2.line(line, (int(n_ / 2 - u0[0] * n_ / 2), int(n_ / 2 - u0[1] * n_ / 2)),
             (int(n_ / 2 + u0[0] * n_ / 2), int(n_ / 2 + u0[1] * n_ / 2)), 1.0, 1)
    noise = cv2.filter2D(noise, -1, line / max(float(line.sum()), 1.0))
    noise = np.clip((noise - noise.mean()) / max(float(noise.std()), 1e-6) * 0.5 + 0.5, 0, 1)
    yy, xx = np.mgrid[:hq, :wq].astype(np.float32) * q
    out = np.zeros((frames, h, w, 3), np.uint8)
    for k in range(frames):
        a = k / (frames - 1)
        c = p0 + u0 * (3.0 * s0) * (1 - (1 - a) ** 2)  # out fast, slowing
        r = s0 * (0.35 + 1.1 * a)
        blob = np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r * r)) * (0.25 + 0.75 * noise)
        dens = 0.55 * (1 - a) ** 1.5 * min(1.0, (k + 1) / 2)
        y = cv2.resize(np.clip(blob * dens, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out[k] = np.clip(255 * y[..., None] ** (1 / 2.2) * np.array([0.82, 0.84, 0.88]), 0, 255).astype(np.uint8)
    return out


def make_bending_shot(spec, renderer, out_dir: Path, track_path: str | Path, mask_path: str | Path | None = None,
                      progress=None, element_from: str | Path | None = None, tap_frame: int | None = None,
                      aim: float | None = None):
    """A bending shot: the plate (``spec.plate``, as cuesync.pipeline loads it), the take (``spec.voice``), the hand's
    track (``track_path``, made on that loaded plate) and optionally its matte (hand_mask.npz); ``spec.effect`` may be
    "auto" (the sound and the hand's gesture pick it). Renders one element over black at the loudest hit's strike and
    fires it on every hit of the take, carried on the hand, each as big and as bright as that hit is loud.
    ``element_from``: an earlier bending shot's folder whose element plays this take instead (perform it again: a new
    rhythm or another voice on the same hand take, no new render). ``tap_frame``: the frame ``spec.point`` was tapped
    on (default: the strike); ``aim``: the aim arrow drawn there, degrees clockwise from up (default: the hand's own)."""
    import dataclasses
    import time

    from . import __version__
    from .composite import estimate_black_level
    from .effects import load_effect
    from .gestures import heard
    from .guide import write_guide
    from .media import read_video, write_video
    from .pipeline import FINISH_PRESET, FPS, SOUND, SR, ShotResult, _finalise, load_plate, place_audio, shot_cue, soundtrack
    from .scene import paint_scene_guide, plan_window, scene_size, suited_effect
    from .transfer import gam, lin

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    t0 = time.perf_counter()
    say("reading", 0.03, "Reading your plate and your hand's track")
    plate, _, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, h, w = plate.shape[:3]
    track = load_track(track_path)
    if (int(track["W"]), int(track["H"])) != (w, h):
        raise ValueError(f"the track was made on a {track['W']}x{track['H']} plate, this one is {w}x{h}")
    mask = np.load(mask_path)["alpha"] if mask_path is not None and Path(mask_path).exists() else None
    c = shot_cue(spec, n / FPS)
    cue = c.cue
    env_all = np.asarray(cue.envelope, np.float64)
    hits = [hh.frame for hh in cue.hits()] or [cue.frame_of(cue.hero_onset_s)]
    loud = [float(env_all[f: f + 8].max()) if 0 <= f < env_all.size else 0.0 for f in hits]
    hero = hits[int(np.argmax(loud))]
    snaps = [snap_launch(f, track.get("events", [])) for f in hits]  # each hit on its own strike, when close
    snap = snaps[hits.index(hero)]
    k_launch = snap["frame"]
    tap_k = k_launch if tap_frame is None else int(np.clip(tap_frame, 0, n - 1))
    track = anchored(track, tap_k, (spec.point[0] * w, spec.point[1] * h), aim)  # where you tapped, carried on the hand
    gains = [float(np.clip((v / max(max(loud), 1e-6)) ** 0.5, 0.35, 1.0)) for v in loud]

    auto = None
    old = json.loads((Path(element_from) / "receipt.json").read_text()) if element_from is not None else None
    effect = old["effect"]["id"] if old else spec.effect
    if effect == "auto":  # the sound and the hand's gesture at the strike
        h_ = heard(cue, c.samples, SR, c.source)
        g = gesture_at(track, k_launch)
        span = (max(hits) - min(hits)) / FPS if len(hits) > 1 else 0.0
        snapped = any(e.get("kind") == "snap" and abs(int(e["frame"]) - hero) <= 4 for e in track.get("events", []))
        effect, why = bending_effect(h_["kind"], g, hits=len(hits), span_s=span, snap=snapped)
        effect = effect or "fire_burst"
        auto = h_ | {"heard": h_["kind"], "hand": g, "picked": effect, "why": why}
    eff = load_effect(effect)

    if old:  # perform it again: the element as rendered for the earlier take, where its hand was then
        ref = (np.asarray(old["bending"]["ref"]["xy"], np.float64), np.asarray(old["bending"]["ref"]["dir"], np.float64),
               float(old["bending"]["ref"]["scale"]))
        aim, prompt, scale, seconds = angle_of(ref[1]), old["render"]["prompt"], old["render"]["guide_scale"], 0.0
        element = read_video(Path(element_from) / "element.mp4").frames
    else:
        say("painting", 0.12, "Painting the element where your hand is on the strike")
        p_hit, u_hit, s_hit = emitter_at(track, k_launch)
        hands, most = SIZE.get(eff.behaviour, (LENGTH, 0.6))
        reach = min(hands * s_hit, most * min(w, h))
        ref = (render_spot(p_hit, u_hit, reach, w, h), u_hit, s_hit)  # rendered where it fits, carried onto the hand
        aim = angle_of(u_hit)
        at = (float(ref[0][0] / w), float(ref[0][1] / h))
        black = np.full((1, h, w, 3), CANVAS, np.uint8)
        eff = suited_effect(eff, "the open hand", aim, black[0], at)
        eff = dataclasses.replace(eff, scene=dataclasses.replace(eff.scene, clean=True, resting=0.0))
        start, length = plan_window(cue, eff, None, multi_hit=False)  # one element, the loudest hit's: it plays them all
        scale = float(np.clip(reach / (eff.scene.r_max * min(w, h)), 0.4, 1.6))
        sp = paint_scene_guide(cue, eff, at, black, start=start, length=length, direction=aim, scale=scale,
                               size=scene_size(w, h), multi_hit=False)
        write_guide(sp.guide, out_dir / "guide.mp4", fps=FPS)
        prompt = BENDING_PROMPTS.get(effect, eff.prompt).replace("{towards}", towards_words(aim))
        if effect == "smoke_puff":  # the air blast is made, not rendered: a puff that leaves and is gone
            element, seconds, prompt = air_puff(h, w, ref, spec.seed), 0.0, "(made, not rendered: an air blast)"
            write_video(out_dir / "element.mp4", element, fps=FPS, crf=16)
        else:
            say("rendering", 0.2, f"Rendering the {eff.fluid} over black with LTX-2.5")
            seconds = renderer(out_dir / "guide.mp4", prompt, spec.seed, out_dir / "element.mp4")
            element = read_video(out_dir / "element.mp4").frames
    if element.shape[1:3] != (h, w):
        element = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in element])
    floor = estimate_black_level(element)
    ign = ignition_frame(element, floor, burst=eff.behaviour == "burst")
    if ign is None:
        raise ValueError("the render shows no element to carry")
    # each hit fires the element: its ignition on that hit's strike, as big and as bright as the hit is loud
    plays = [(s_["frame"] - ign, g, float(np.clip(g ** 0.5, 0.6, 1.0))) for s_, g in zip(snaps, gains)]
    offset = k_launch - ign

    say("compositing", 0.7, "Carrying the element on your hand")
    from . import fx_light as fx

    lag = LAG_FRAMES.get(effect, 3)
    pre = 3 if eff.behaviour == "burst" else PREGLOW  # a burst's build-up is short
    src = list(track.get("src") or []) + ["mp"] * n
    floor_lin = float(lin(np.full((1, 1, 3), floor, np.uint8))[0, 0, 0])
    if ign >= pre + 3:  # the render's own black, pixel by pixel, from before anything shows (one value for the whole
        # frame left the painted root's lifted box behind); a render lit from its start keeps the single value
        floor_lin = np.maximum(lin(np.min(element[: ign - pre], axis=0)), floor_lin)
    lost, run = np.zeros(n, int), 0
    for t in range(n):
        run = run + 1 if src[t] == "lost" else 0
        lost[t] = run

    fade = root_fade(h, w, ref, ROOT_FADE_HANDS.get(effect, ROOT_FADE * LENGTH)) * border_fade(h, w)
    if eff.fluid == "steam":  # vapour thins out as it spreads: dense at the palm, a third as dense at its far end
        p0_, u0_, s0_ = ref
        yy_, xx_ = np.mgrid[:h, :w].astype(np.float32)
        along_ = ((xx_ - p0_[0]) * u0_[0] + (yy_ - p0_[1]) * u0_[1]) / max(0.5 * LENGTH * s0_, 1.0)
        fade = fade * np.clip(1.0 - 0.67 * along_, 0.33, 1.0)
        del yy_, xx_, along_
    fade = fade[..., None]

    def element_at(t: int) -> np.ndarray | None:
        """The element on plate frame t (linear, over black): every hit's play of it, carried on the hand, its root
        held out; None when none is on (before its build-up, after it ends)."""
        acc = None
        for off, gain, size in plays:
            e = t - off
            if not max(0, ign - pre) <= e < len(element):  # (a negative frame would wrap round to the render's end)
                continue
            E = warp_element(np.clip(lin(element[e]) - floor_lin, 0.0, None) * fade, track, ref, t, lag, size,
                             since=off + ign) * gain
            if e < ign:  # the build-up, coming up to the ignition
                E *= (e - (ign - pre) + 1) / (pre + 1)
            elif ign < pre and e < pre:  # a render lit from its first frame: bring it up rather than pop it on
                E *= (e + 1) / (pre + 1)
            acc = E if acc is None else np.maximum(acc, E)  # two plays overlapping: the brighter shows, not both added
        if acc is None:
            return None
        E = acc
        if lost[t] > 8:  # the hand is gone: let the element go rather than freeze it
            E *= max(0.0, 1.0 - (lost[t] - 8) / LOST_FADE)
        m_t = None if mask is None else mask[min(t, len(mask) - 1)].astype(np.float32) / 255.0
        return E * (1.0 - root_holdout(track, t, h, w, m_t))[..., None]

    # the light (cuesync.fx_light) needs the element's light on every frame first: a small copy is enough
    q = 8
    small = np.zeros((n, h // q, w // q, 3), np.float32)
    for t in range(n):
        E = element_at(t)
        if E is not None:
            small[t] = cv2.resize(E, (w // q, h // q), interpolation=cv2.INTER_AREA)
    white = fx.plate_white(np.stack([lin(plate[i]) for i in np.linspace(0, n - 1, 8).astype(int)]))
    noise = fx.shadow_noise(plate)
    out = np.empty_like(plate)
    CHUNK = 12  # frames lit at a time (a whole 10 s 1080p clip in float32 is 6 GB), overlapping a frame each side
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        a0, b1 = max(0, a - 1), min(n, b + 1)
        P = np.stack([lin(plate[i]) for i in range(a0, b1)])
        m = None if mask is None else mask[a0:b1].astype(np.float32) / 255.0
        fx.light_plate(P, small[a0:b1], track_slice(track, a0, b1), m, fluid=eff.fluid, white=white, out=P)
        for i in range(a, b):
            E = element_at(i)
            if E is None:
                continue
            E = fx.grain(E[None], noise, seed=spec.seed, frame0=i)[0]
            j = i - a0
            if eff.fluid == "smoke":  # smoke is laid over; fire is light, added; steam is lit vapour, screened
                al = np.clip(E.max(-1) / 0.35, 0.0, 0.9)[..., None]
                P[j] = P[j] * (1.0 - al) + np.clip(E / np.maximum(al, 1e-3), 0, 1) * al
            elif eff.fluid == "steam":  # steam scatters the room's light: laid over, never brighter than its white
                al = np.clip(fx._lum(E) / 0.6, 0.0, 0.55)[..., None]  # translucent: a dense render read as cotton wool
                P[j] = P[j] * (1.0 - al) + np.minimum(E / np.maximum(al, 1e-3), max(0.9 * white, 0.45)) * al
            else:
                P[j] = P[j] + E
        if eff.fluid == "fire":  # the hot air above and around the flame shimmers
            alpha = np.stack([cv2.resize(np.clip(fx._lum(small[i]) * 4.0, 0, 1), (w, h)) for i in range(a0, b1)])
            fx.heat_haze(P, alpha, seed=spec.seed, out=P, frame0=a0)
        fx.flash(P, [off + ign - a0 for off, _, _ in plays], [g for _, g, _ in plays], out=P)
        out[a:b] = gam(P[a - a0: b - a0])
    plate = element = small = None  # (a 10 s 1080p plate and its render are 2 GB: let them go before the finishing copies)
    if effect == "smoke_puff":  # the air blast: a pressure front out of the palm
        lo, hi = k_launch, min(n, k_launch + fx.AIR_FRAMES + 2)
        W_ = np.stack([lin(out[i]) for i in range(lo, hi)])
        pa, ua, _ = emitter_at(track, k_launch)
        fx.air_wave(W_, tuple(pa), tuple(ua), start_frame=0, seed=spec.seed, out=W_)
        out[lo:hi] = gam(W_)
    frames = fx.shake(out, [off + ign + 2 for off, _, _ in plays], [g for _, g, _ in plays], px=4.0 * h / 1080,
                      seed=spec.seed)

    say("finishing", 0.85, "Adding the sound and the cue ribbon")
    env = np.zeros(n)
    m = min(n, env_all.size)
    env[:m] = env_all[:m]
    audio = soundtrack(place_audio(c.samples, c.offset_s, n), eff, n, env, [off + ign for off, _, _ in plays],
                       [g for _, g, _ in plays], cue.voice_end_frame, spec.seed)
    comp_path, final_path = out_dir / "composite.mp4", out_dir / "final.mp4"
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    write_video(final_path, _finalise(frames, env, spec.label or effect.replace("_", " "), None), fps=FPS, audio=audio,
                crf=16, preset=FINISH_PRESET)
    receipt = {
        "cuesync": __version__, "mode": "bending", "effect": eff.to_json(),
        "inputs": {"point": list(spec.point), "seed": spec.seed, "offset_s": spec.offset_s, "cue": c.source},
        "render": {"seconds": round(seconds, 1), "prompt": prompt, "guide_scale": round(scale, 3)},
        "bending": {"track": str(track_path), "mask": None if mask is None else str(mask_path), "k_hit": int(k_launch),
                    "k_launch": int(k_launch), "ignition": int(ign), "offset": int(offset), "lag_frames": lag,
                    "snap": snap, "hits": [{"frame": int(off + ign), "gain": round(g, 3), "size": round(sz, 3),
                                            "snap": sn} for (off, g, sz), sn in zip(plays, snaps)],
                    "ref": {"xy": [round(float(v), 2) for v in ref[0]], "dir": [round(float(v), 5) for v in ref[1]],
                            "scale": round(float(ref[2]), 2)},
                    "performed_again": None if old is None else {"from": str(element_from), "no_new_render": True}},
        "auto": auto, "sound": SOUND, "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=None, receipt=receipt)


# ------------------------------------------------------------------ the studio's bending shot, from your clip

EMITTER_OF = {"fire_breath": "two_finger", "fire_burst": "palm", "flame_plume": "palm", "steam_jet": "palm",
              "smoke_puff": "palm"}  # the point on the hand cuesync.hand_track follows for each element


def hand_python(kind: str) -> str:
    """The Python that runs cuesync.hand_track (``kind`` "track") or cuesync.hand_mask ("mask"), each in its own venv
    (MediaPipe and SAM 2 clash with the core's packages): $CUESYNC_TRACK_PY / $CUESYNC_MASK_PY, else the venv the
    README sets up (cuesync/.venv-track, cuesync/.venv-mask), else this one."""
    if os.environ.get(f"CUESYNC_{kind.upper()}_PY"):
        return os.environ[f"CUESYNC_{kind.upper()}_PY"]
    root = Path(__file__).resolve().parents[2]
    for d in (root / f".venv-{kind}", root.parent / "runs" / "bending" / f"venv-{kind}"):
        exe = d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if exe.exists():
            return str(exe)
    return sys.executable


def prepare_hand(plate: Path, out_dir: Path, tap: tuple[float, float], frame: int, emitter: str) -> tuple[Path, Path | None]:
    """Track the tapped hand through ``plate`` (the clip as the pipeline loads it; ``tap`` in px on ``frame``) and matte
    it, each as a subprocess in its own venv; returns (hand_track.json, hand_mask.npz or None: without a matte the
    element's root is held out behind the fingertips instead)."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                                     os.environ.get("PYTHONPATH")]))}
    track, mask = out_dir / "hand_track.json", out_dir / "hand_mask.npz"
    r = subprocess.run([hand_python("track"), "-m", "cuesync.hand_track", str(plate), "--emitter", emitter,
                        "--seed", f"{frame},{tap[0]:.1f},{tap[1]:.1f}", "--out", str(track)],
                       env=env, capture_output=True, text=True, timeout=900)
    (out_dir / "hand_track.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not track.exists():
        last = (r.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"couldn't follow your hand through the clip ({last})")
    try:
        r = subprocess.run([hand_python("mask"), "-m", "cuesync.hand_mask", str(plate), "--track", str(track), "--out",
                            str(mask)], env=env, capture_output=True, text=True, timeout=900)
        (out_dir / "hand_mask.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        ok = r.returncode == 0 and mask.exists()
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return track, mask if ok else None


def bending_job(spec, renderer, out_dir: Path, progress=None):
    """The studio's bending shot from your clip: the plate as the pipeline loads it (kept, so performing it again reads
    the same frames), the hand you tapped (``spec.point`` on frame ``spec.extra["tap_frame"]``) tracked and matted, then
    make_bending_shot, aimed by ``spec.direction`` when you drew the arrow. ``spec`` is updated to the loaded plate."""
    from .gestures import heard
    from .media import write_video
    from .pipeline import FPS, SR, load_plate, shot_cue

    out_dir = Path(out_dir)
    (out_dir / "in").mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    say("reading", 0.02, "Reading your clip")
    frames, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, h, w = frames.shape[:3]
    loaded = out_dir / "in" / "plate24.mp4"
    write_video(loaded, frames, fps=FPS, crf=12)
    del frames
    spec.plate, spec.point = loaded, (float(point[0]), float(point[1]))
    k = int(np.clip(spec.extra.get("tap_frame") or 0, 0, n - 1))
    emitter = EMITTER_OF.get(spec.effect, "palm")
    if spec.effect == "auto":  # which point to follow, by the sound (its gesture is read from the track later)
        c = shot_cue(spec, n / FPS)
        emitter = "two_finger" if heard(c.cue, c.samples, SR, c.source)["kind"] in ("whoosh", "hiss") else "palm"
    say("tracking", 0.05, "Finding your hand and following it")
    track, mask = prepare_hand(loaded, out_dir, (spec.point[0] * w, spec.point[1] * h), k, emitter)
    return make_bending_shot(spec, renderer, out_dir, track, mask, progress, tap_frame=k, aim=spec.direction)


def bending_again(src_dir: Path, spec, out_dir: Path, progress=None):
    """Perform a finished bending shot again with a new take (``spec``): its element on the same hand take, no new
    render; a shot that was itself performed again plays its first render."""
    src_dir = Path(src_dir)
    b = json.loads((src_dir / "receipt.json").read_text())["bending"]
    first = Path(b["performed_again"]["from"]) if b.get("performed_again") else src_dir
    here = lambda name, was: str(first / name) if (first / name).exists() else was  # noqa: E731  (a shot made ahead
    # travels with its own track and matte)
    return make_bending_shot(spec, None, out_dir, here("hand_track.json", b["track"]), here("hand_mask.npz", b["mask"]),
                             progress, element_from=first, tap_frame=spec.extra.get("tap_frame"), aim=spec.direction)
