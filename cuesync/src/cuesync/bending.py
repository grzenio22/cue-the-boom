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
ROOT_FADE_SHARE = {"steam_jet": 0.05}  # steam has no nozzle to hide: over the full fade it started a hand from the palm
CANVAS = 16  # the guide's ground: a dark grey, not black, so a keyframe before the hit says "nothing here yet" (over
# pure black the model took the empty keyframes as unconstrained and had steam there from the first frame)
GROW_FROM, GROW_FRAMES = 0.35, 6  # a flame or a jet grows out of the hand over its first frames, from this share of its
# size (it was there at full height on its first lit frame: a pop, not a flame catching)
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
SLICES = 4  # history slices a frame apart along the element
FLOOR_SPAN = 16  # 8-bit: the render's black, pixel by pixel, is at most this far above its black as a whole
COVER_FLOOR = 0.5  # where less than this of the slices' weight lands, the element thins out instead of being averaged
SHOULDER = 0.7  # linear light: fire over a lit hand rolls off from here towards white instead of clipping to it
LOUD_DB = (60.0, 40.0)  # a hit this many dB under the loudest is 10x dimmer, and 10x smaller: "pf" 12 dB under "BOOM"
# plays half its size (the cue's envelope is in dB; read as amplitude, it was nine tenths of it)
HIT_SIZE = (0.45, 1.0)  # the smallest and largest a hit plays the element, against the loudest hit's
SOOT, SOOT_FULL, SOOT_GAMMA = 0.92, 0.1, 0.7  # a flame is not light alone: where it is this bright (linear) its soot
# and glare hide 92 % of what is behind it, so on a white wall it keeps its shape and colour (added alone, it went white
# on white); over a dark plate there is little behind it to hide, and it looks as added
SOOT_FROM = 0.012  # linear: fainter than this is the render's own haze, not fire, and hides nothing (it greyed a box
# on the palm before the ignition)
DAY_GAIN = 0.3  # and over a bright room it is this much brighter than rendered (the render was exposed over black; a
# flame doesn't dim with the room, and seen against a white wall at the render's own level it read as orange paint)


def shoulder(P: np.ndarray) -> None:
    """In place, linear RGB: over SHOULDER each channel rolls off towards 1 (tanh), so a lit palm goes orange, then
    gold, then white where it is brightest, as on film."""
    np.copyto(P, SHOULDER + (1.0 - SHOULDER) * np.tanh((P - SHOULDER) / (1.0 - SHOULDER)), where=P > SHOULDER)


def hit_looks(loud: list[float]) -> list[tuple[float, float]]:
    """(gain, size) for each hit from its level on the cue's envelope (54 dB over 0..1): the loudest plays the element
    as rendered, a quieter one dimmer and smaller, by its dB under the loudest (LOUD_DB, HIT_SIZE)."""
    top = max(loud, default=0.0)
    under = [54.0 * (top - v) for v in loud]
    return [(float(np.clip(10 ** (-d / LOUD_DB[0]), 0.35, 1.0)), float(np.clip(10 ** (-d / LOUD_DB[1]), *HIT_SIZE)))
            for d in under]


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


CAM_SCALE = 2  # the camera is measured on a copy this many times smaller
CAM_EXPLAIN = 0.75  # a step must line the background up this much better than standing still (the tracker's 0.75
# turned away most of a phone's small shakes over a dim room's soft texture)
CAM_STILL_STEP = 1.2  # px at the corners: a camera whose median measured step is less is locked off (a tripod
# measures about 0.5 of noise, a phone held at the eyes 3-4)


def camera_path(plate: np.ndarray, mask: np.ndarray | None = None, track: dict | None = None) -> np.ndarray | None:
    """The camera's own motion through ``plate`` [T, H, W, 3] uint8, as (T, 3, 3): frame 0's px -> frame t's px; None
    for a camera that holds still, so a tripod shot is untouched by the measurement's noise. Each step is measured on
    the background by the tracker's own guarded step (cuesync.hand_track._camera_step: corners followed there and back
    by optical flow, a similarity that most of them agree on, spread across the picture, plausible for a camera, that
    lines the background's texture up better than standing still, and that isn't just the striking arm), with the hand
    left out (its matte ``mask``, else the ``track``'s landmarks and forearm, else a disc round its emitter). A step
    the guards turn away is measured loosely (corners there and back, plausible for a camera) once the guarded steps
    have shown the clip to be handheld, and only one nothing can measure is drawn between its neighbours; a clip
    whose steps mostly fail the guards (a plain wall, a moving person on a tripod shot) holds still. First person, the phone in the other hand:
    what was already thrown then moves with the room, not with the screen."""
    from .hand_track import CAM_MIN_SHARE, _camera_step

    n, h, w = plate.shape[:3]
    if n < 2:
        return None
    q = CAM_SCALE
    hs, ws = max(16, h // q), max(16, w // q)
    S, Si = np.diag([q, q, 1.0]), np.diag([1.0 / q, 1.0 / q, 1.0])

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # a dim room's texture brought up to where the step's
    # texture check can see it (the guards still turn away grain on a plain wall)

    def gray(t: int) -> np.ndarray:  # lightly blurred: the grain out, the scene's texture kept
        g = cv2.cvtColor(cv2.resize(plate[t], (ws, hs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        return cv2.GaussianBlur(clahe.apply(g), (0, 0), 1.2)

    edge = np.zeros((hs, ws), np.uint8)
    edge[hs // 20: hs - hs // 20, ws // 20: ws - ws // 20] = 255

    def where(t: int) -> np.ndarray:  # the background on frame t: not the hand, not the frame's very edge
        m = edge.copy()
        if mask is not None:
            hand = cv2.resize(mask[min(t, len(mask) - 1)], (ws, hs), interpolation=cv2.INTER_AREA) > 32
            m[cv2.dilate(hand.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0] = 0
        elif track is not None:  # no matte: the hand's landmarks and its forearm out to the frame's edge, else a disc
            pt, _, st = emitter_at(track, t)
            lms = track.get("landmarks") or []
            lm = lms[min(t, len(lms) - 1)] if lms else None
            if lm:
                xy = np.asarray(lm, np.float64)[:, :2] / q
                cv2.fillConvexPoly(m, cv2.convexHull(xy.astype(np.int32)), 0)
                m = cv2.erode(m, np.ones((int(0.8 * st / q) | 1,) * 2, np.uint8))
                arm = xy[0] - xy[9]  # wrist, from the middle knuckle: the forearm goes on that way
                arm = arm / max(float(np.hypot(*arm)), 1e-6)
                cv2.line(m, tuple(int(v) for v in xy[0]), tuple(int(v) for v in xy[0] + arm * (hs + ws)), 0,
                         max(1, int(1.0 * st / q)))
            else:
                cv2.circle(m, (int(pt[0] / q), int(pt[1] / q)), int(1.3 * st / q), 0, -1)
        return m

    def loose(prev_: np.ndarray, cur_: np.ndarray, m: np.ndarray) -> np.ndarray | None:
        """A step the guards turned away, measured anyway: corners there and back, RANSAC, plausible for a camera.
        Used only in a clip the guarded steps already showed to be handheld (a hand filling the frame, a dark patch)."""
        pts = cv2.goodFeaturesToTrack(prev_, 400, 0.005, 6, mask=m)
        if pts is None or len(pts) < 12:
            return None
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev_, cur_, pts, None, winSize=(21, 21), maxLevel=3)
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(cur_, prev_, nxt, None, winSize=(21, 21), maxLevel=3)
        ok = ok.ravel().astype(bool) & ok2.ravel().astype(bool) & (np.linalg.norm((back - pts)[:, 0], axis=1) < 0.7)
        if ok.sum() < 12:
            return None
        A, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if A is None or inl is None or int(inl.sum()) < 12:
            return None
        zoom, turn = np.hypot(A[0, 0], A[1, 0]), np.arctan2(A[1, 0], A[0, 0])
        if np.hypot(*A[:, 2]) > 0.2 * ws or abs(turn) > np.radians(6) or abs(zoom - 1) > 0.06:
            return None
        return A

    steps = np.full((n, 4), np.nan)  # per step t-1 -> t: dx, dy (plate px), turn (rad), log zoom
    second = np.full((n, 4), np.nan)  # the steps the guards turned away, measured loosely
    prev = gray(0)
    for t in range(1, n):
        cur = gray(t)
        hands = ()
        if track is not None:  # the emitter before and after, in analysis px: a "camera" that moves it as it struck is the arm
            (p_a, _, s_a), (p_b, _, _) = emitter_at(track, t - 1), emitter_at(track, t)
            hands = ((p_a / q, p_b / q, s_a / q),)
        m = where(t - 1)
        M = _camera_step(prev, cur, m, hands, explain=CAM_EXPLAIN)
        for M_, into in ((M, steps), (None if M is not None else loose(prev, cur, m), second)):
            if M_ is not None:
                F = S @ np.vstack([M_, [0.0, 0.0, 1.0]]) @ Si
                into[t] = [F[0, 2], F[1, 2], np.arctan2(F[1, 0], F[0, 0]), np.log(np.hypot(F[0, 0], F[1, 0]))]
        prev = cur
    have = np.flatnonzero(~np.isnan(steps[1:, 0])) + 1
    if len(have) < CAM_MIN_SHARE * (n - 1):
        return None
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    filled = np.isnan(steps[:, 0]) & ~np.isnan(second[:, 0])  # handheld, so the loose measurements count
    steps[filled] = second[filled]
    known = np.flatnonzero(~np.isnan(steps[1:, 0])) + 1
    for j in range(4):  # a step nothing could measure, drawn between the measured ones either side
        steps[1:, j] = np.interp(np.arange(1, n), known, steps[known, j], left=0.0, right=0.0)
    W = np.repeat(np.eye(3)[None], n, 0)
    sizes = []
    for t in range(1, n):
        dx, dy, a, lz = steps[t]
        z = np.exp(lz)
        step = np.array([[z * np.cos(a), -z * np.sin(a), dx], [z * np.sin(a), z * np.cos(a), dy], [0.0, 0.0, 1.0]])
        W[t] = step @ W[t - 1]
        if t in have:
            sizes.append(float(np.sqrt((((step - np.eye(3)) @ corners)[:2] ** 2).sum(0).mean())))
    return None if float(np.median(sizes)) < CAM_STILL_STEP else W


def camera_between(cam: np.ndarray | None, a: float, b: float) -> np.ndarray:
    """3x3: frame ``a``'s px -> frame ``b``'s px, by the camera alone (identity with no camera path); fractional frames
    blend their neighbours."""
    if cam is None:
        return np.eye(3)

    def at(x: float) -> np.ndarray:
        x = float(np.clip(x, 0, len(cam) - 1))
        i = int(np.floor(x))
        f = x - i
        return cam[i] if f < 1e-6 or i + 1 >= len(cam) else cam[i] * (1 - f) + cam[i + 1] * f

    return at(b) @ np.linalg.inv(at(a))


CHARGE_MIN_S = 0.5  # a hum at least this long, unbroken into the loudest hit, charges it up ("mmmmmmBOOM")
EMBER_LOOP = 10  # frames of the render's first fire the ember plays back and forth while it charges
EMBER_SIZE = (0.12, 0.55)  # the ember's size against the element's, from a quiet hum to the hum's loudest


def charge_window(cue, env: np.ndarray, hero: int, fps: float = 24.0) -> tuple[int, int] | None:
    """(start, release) plate frames of a charge-up: the performer humming into the strike ("mmmmmmBOOM"). The release
    is where the loudest hit's sound surges to its peak (the foot of that last rise); a charge is the sound running
    on unbroken for at least CHARGE_MIN_S before it, voiced all along (never under a quarter of its own median) and
    clearly quieter than the strike. None for a strike out of silence ("KA-BOOM")."""
    span = next(((a, b) for a, b in cue.intervals() if a <= hero < b), None)
    if span is None:
        return None
    a, b = span
    seg = np.asarray(env[a:b], np.float64)
    if seg.size < 4 or seg.max() <= 0:
        return None
    r = a + int(np.argmax(seg >= 0.85 * seg.max()))  # where it reaches its peak...
    while r > a and env[r - 1] < env[r] - 0.02:  # ...and the foot of that rise
        r -= 1
    hum = np.asarray(env[a: max(a + 1, r - 1)], np.float64)
    if r - a < CHARGE_MIN_S * fps or hum.size < 4:
        return None
    med = float(np.median(hum))
    if med <= 0 or hum.min() < 0.25 * med or med > 0.8 * seg.max():
        return None
    return a, r


def warp_element(frame: np.ndarray, track: dict, ref: tuple, t: int, lag: int, size: float = 1.0,
                 blur: bool = True, since: int | None = None, cam: np.ndarray | None = None) -> np.ndarray:
    """An element frame (float32, over black, at the plate's size, rendered with the emitter at ``ref``) carried onto
    plate frame ``t`` with its emission history: in thin slices along its length, its base on the emitter's pose now and
    each slice further along on the pose the emitter had when that part left the hand, up to ``lag`` frames before for
    the far end (never before ``since``, the strike). Fire already thrown keeps flying the way it was thrown while the
    hand swings on: one pose for the whole element made it a strip swung on a pivot, and one remap with a pose per pixel
    tore it apart on a fast turn. The slices are a quarter of a frame of history apart (half without ``blur``), each
    warped only where it is, their weights overlapping and summing to 1, and each smeared over its own share of the
    history (the shutter's own smear) in as many steps as it takes for its far edge to move at most SMEAR_PX: a stream
    thrown on a fast turn is stretched along its path, not broken into pieces. ``cam`` (camera_path): what was thrown
    then has moved with the room since, as the camera saw it."""
    h, w = frame.shape[:2]
    p0, u0, s0 = ref
    lo_t = -np.inf if since is None else float(since)
    out = np.zeros_like(frame)
    cover = np.zeros(frame.shape[:2], np.float32)  # the slices' weights, carried as they are: what each pixel got
    lit = frame[::4, ::4].max(-1) > 1e-4  # where the element is, from a quarter-size look
    ys, xs = np.nonzero(lit)
    if ys.size == 0:
        return out
    box = (max(0, 4 * xs.min() - 4), min(w, 4 * xs.max() + 8), max(0, 4 * ys.min() - 4), min(h, 4 * ys.max() + 8))
    reach = max(LENGTH * s0, 1.0)
    def pose(c: float) -> np.ndarray:  # the pose a slice c frames old was thrown with, carried by the camera since
        then = min(max(t - c, lo_t), float(t))  # before the strike nothing has been thrown: all on the hand now
        A = pose_forward(ref, track, then, size)
        return A if cam is None else (camera_between(cam, then, t) @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
    n = (SLICES if blur else SLICES // 2) * lag + 1 if lag > 0 else 1  # SLICES a frame (half without blur)
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
        wgt = np.ones(part.shape[:2], np.float32)
        if n > 1:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            lev = np.clip(level0 + g[0] * xx + g[1] * yy, 0, lag)
            wgt = np.clip(1.0 - np.abs(lev - c) / step, 0.0, 1.0).astype(np.float32)
            part = part * wgt[..., None]
        ph, pw = part.shape[:2]
        a, z = max(0.0, c - step / 2), min(float(lag), c + step / 2)  # its share of the history: no more (a whole frame
        # each smeared the far end of a fast swing into a featureless brushstroke), no less (gaps between slices)
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
            A32 = A.astype(np.float32)
            out[dy0:dy1, dx0:dx1] += cv2.warpAffine(part, A32, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR,
                                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) * (1.0 / len(times))
            cover[dy0:dy1, dx0:dx1] += cv2.warpAffine(wgt, A32, (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_LINEAR,
                                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) * (1.0 / len(times))
    if n > 1:  # neighbouring slices carried to different poses no longer sum to one where they meet: a fast sweep
        # came out as a venetian blind of blobs. Each pixel is the average of what reached it, and only where too
        # little reached it (the slices really flew apart) does it fade
        out /= np.maximum(cover, COVER_FLOOR)[..., None]
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


def root_fade(h: int, w: int, ref: tuple, length: float) -> np.ndarray:
    """(h, w) float32 in the element's own frame: 0 at and behind the emitter, rising to 1 ROOT_FADE of the reach along
    it, so the element comes up from nothing just past the fingertips."""
    p0, u0, s0 = ref
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    along = (xx - p0[0]) * u0[0] + (yy - p0[1]) * u0[1]
    return np.clip(along / max(length, 1.0), 0, 1).astype(np.float32)


def fit_element(element: np.ndarray, ref: tuple, h: int, w: int) -> tuple[np.ndarray, tuple]:
    """A render of another frame size on an (h, w) plate: a window of the plate's size, centred on where the render is
    lit (over all its frames), its emitter pose moved with it. Never stretched: a landscape render on a portrait
    clip keeps its fire's shape (what doesn't fit the window is left out). Returns (frames, ref)."""
    n, eh, ew = element.shape[:3]
    lit = element[:, ::4, ::4].max(axis=(0, 3)) > 24
    ys, xs = np.nonzero(lit)
    cy, cx = (4 * (ys.min() + ys.max()) / 2, 4 * (xs.min() + xs.max()) / 2) if ys.size else (eh / 2, ew / 2)
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    out = np.zeros((n, h, w, 3), element.dtype)
    sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(eh, y0 + h), min(ew, x0 + w)
    if sy1 > sy0 and sx1 > sx0:
        out[:, sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] = element[:, sy0:sy1, sx0:sx1]
    p0, u0, s0 = ref
    return out, (np.asarray(p0, np.float64) - (x0, y0), u0, s0)


STILL_DIFF = 6.0  # mean grey levels: the picture round a lost hand changing less than this since it was seen = held still


def still_there(plate: np.ndarray, track: dict, seen: int, t: int) -> bool:
    """Is the hand last seen on frame ``seen`` still where it was on frame ``t``? The picture within a hand of its last
    position barely changed (a quarter-size look): the tracker lost a still hand (cut by the frame's edge, blurred),
    the hand didn't go anywhere."""
    p, _, s = emitter_at(track, seen)
    h, w = plate.shape[1:3]
    r = max(8.0, 1.0 * s)
    y0, y1, x0, x1 = max(0, int(p[1] - r)), min(h, int(p[1] + r)), max(0, int(p[0] - r)), min(w, int(p[0] + r))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return False
    a = plate[seen, y0:y1:4, x0:x1:4].astype(np.float32).mean(-1)
    b = plate[t, y0:y1:4, x0:x1:4].astype(np.float32).mean(-1)
    return float(np.abs(a - b).mean()) < STILL_DIFF


def holdout_box(track: dict, t: int, h: int, w: int) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): the only part of the frame root_holdout can touch (half a hand round the emitter)."""
    p, _, s = emitter_at(track, t)
    r = 0.5 * s + 2
    return (max(0, int(p[1] - r)), min(h, int(p[1] + r) + 1), max(0, int(p[0] - r)), min(w, int(p[0] + r) + 1))


def root_holdout(track: dict, t: int, h: int, w: int, mask: np.ndarray | None,
                 box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Where the hand hides the element's root (float 0..1): within a quarter of a hand size of the emitter, the hand
    itself (its matte; without one, the side behind the fingertips, coming in over 0.3 of a hand size so the base fades
    into the hand instead of being cut off along a line), so the flame leaves the fingers rather than sitting on them.
    With ``box`` (holdout_box) only that part of the frame, as a patch (``mask`` is the whole frame's either way)."""
    p, u, s = emitter_at(track, t)
    y0, y1, x0, x1 = box or (0, h, 0, w)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    mask = None if mask is None else mask[y0:y1, x0:x1]
    dx, dy = xx - p[0], yy - p[1]
    disc = np.clip(1.0 - (np.hypot(dx, dy) - 0.25 * s) / max(0.25 * s, 1.0), 0, 1)
    back = -(dx * u[0] + dy * u[1])  # how far behind the emitter, along its direction
    if mask is not None:  # the hand where it is behind the emitter: the fingers in front of a flame's root hide it, the
        # palm the flame rises in front of doesn't (the whole matte put fire in a cupped hand behind it)
        return disc * mask * np.clip(0.5 + back / max(0.3 * s, 1.0), 0, 1)
    return disc * np.clip(back / max(0.3 * s, 1.0), 0, 1)


def left_frame(track: dict, w: int, h: int) -> tuple[dict, np.ndarray]:
    """(track, gone): where the emitter has gone more than half a hand past the frame's edge, the hand has left the
    shot and the trackers only guess at it (their guesses swung a pointing hand round as it went out, and the fire
    with it): its pose is held where it went out, and ``gone`` counts the frames since (0 while it is in)."""
    n = len(track["xy"])
    xy, d, sc = [list(map(float, v)) for v in track["xy"]], [list(map(float, v)) for v in track["dir"]],         [float(v) for v in track["scale"]]
    gone, last = np.zeros(n, int), None
    for t in range(n):
        m = 0.5 * sc[t]
        out = not (-m <= xy[t][0] <= w + m and -m <= xy[t][1] <= h + m)
        if out and last is not None:
            xy[t], d[t], sc[t] = xy[last], d[last], sc[last]
            gone[t] = gone[t - 1] + 1 if t else 1
        elif not out:
            last = t
    return track | {"xy": xy, "dir": d, "scale": sc}, gone


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
                      aim: float | None = None, instant: bool = False, picked: dict | None = None,
                      only_hits: list[int] | None = None):
    """A bending shot: the plate (``spec.plate``, as cuesync.pipeline loads it), the take (``spec.voice``), the hand's
    track (``track_path``, made on that loaded plate) and optionally its matte (hand_mask.npz); ``spec.effect`` may be
    "auto" (the sound and the hand's gesture pick it). Renders one element over black at the loudest hit's strike and
    fires it on every hit of the take, carried on the hand, each as big and as bright as that hit is loud.
    ``element_from``: an earlier bending shot's folder whose element plays this take instead (perform it again: a new
    rhythm or another voice on the same hand take, no new render). ``tap_frame``: the frame ``spec.point`` was tapped
    on (default: the strike); ``aim``: the aim arrow drawn there, degrees clockwise from up (default: the hand's own).
    ``only_hits``: which of the take's hits (their order in time) this hand fires, the rest left to another hand (two
    hands answering each other: run it once per hand, each on the other's composite); every hit keeps its Foley."""
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
    if only_hits:  # this hand's hits: its render posed on the loudest of them
        mine = [i for i in only_hits if 0 <= i < len(hits)]
        hero = hits[max(mine, key=lambda i: loud[i])] if mine else hero
    charge = charge_window(cue, env_all, hero, FPS)
    if charge:  # the hum launches nothing: it charges the strike, which goes off where the sound surges
        hits = sorted({f for f in hits if not charge[0] - 2 <= f < charge[1] + 3} | {charge[1]})
        loud = [float(env_all[f: f + 8].max()) if 0 <= f < env_all.size else 0.0 for f in hits]
        hero = charge[1]
    snaps = [snap_launch(f, track.get("events", [])) for f in hits]  # each hit on its own strike, when close
    snap = snaps[hits.index(hero)]
    k_launch = snap["frame"]
    tap_k = k_launch if tap_frame is None else int(np.clip(tap_frame, 0, n - 1))
    arrow = aim  # the aim arrow you drew, if you did (kept in the receipt: performing again keeps it)
    track = anchored(track, tap_k, (spec.point[0] * w, spec.point[1] * h), aim)  # where you tapped, carried on the hand
    looks = hit_looks(loud)

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
        # a burst is rendered on a landscape canvas whatever the clip's shape (on a portrait one the model drew a narrow
        # fountain of sparks, seed after seed), in its middle, and placed on the hand by its pose (fit_element)
        rw, rh = (max(w, h), min(w, h)) if eff.behaviour == "burst" else (w, h)
        start_at = p_hit if (rw, rh) == (w, h) else np.array([rw / 2, rh / 2]) - u_hit * reach / 2
        ref = (render_spot(start_at, u_hit, reach, rw, rh), u_hit, s_hit)  # rendered where it fits, carried onto the hand
        aim = angle_of(u_hit)
        at = (float(ref[0][0] / rw), float(ref[0][1] / rh))
        black = np.full((1, rh, rw, 3), CANVAS, np.uint8)
        eff = suited_effect(eff, "the open hand", aim, black[0], at)
        eff = dataclasses.replace(eff, scene=dataclasses.replace(eff.scene, clean=True, resting=0.0))
        start, length = plan_window(cue, eff, None, multi_hit=False)  # one element, the loudest hit's: it plays them all
        scale = float(np.clip(reach / (eff.scene.r_max * min(w, h)), 0.4, 1.6))
        sp = paint_scene_guide(cue, eff, at, black, start=start, length=length, direction=aim, scale=scale,
                               size=scene_size(rw, rh), multi_hit=False)
        write_guide(sp.guide, out_dir / "guide.mp4", fps=FPS)
        prompt = BENDING_PROMPTS.get(effect, eff.prompt).replace("{towards}", towards_words(aim))
        if effect == "smoke_puff":  # the air blast is made, not rendered: a puff that leaves and is gone
            element, seconds, prompt = air_puff(h, w, ref, spec.seed), 0.0, "(made, not rendered: an air blast)"
            write_video(out_dir / "element.mp4", element, fps=FPS, crf=16)
        else:
            say("rendering", 0.2, f"Rendering the {eff.fluid} over black with LTX-2.5")
            seconds = renderer(out_dir / "guide.mp4", prompt, spec.seed, out_dir / "element.mp4")
            element = read_video(out_dir / "element.mp4").frames
    ref_saved = ref  # the emitter's pose in element.mp4 as saved (the receipt's: performing again fits it again)
    if element.shape[1:3] != (h, w):  # (a render of another size: placed by its pose, never stretched)
        element, ref = fit_element(element, ref, h, w)
    floor = estimate_black_level(element)
    ign = ignition_frame(element, floor, burst=eff.behaviour == "burst")
    if ign is None:
        raise ValueError("the render shows no element to carry")
    # each hit fires the element: its ignition on that hit's strike, as big and as bright as the hit is loud
    plays = [(s_["frame"] - ign, g, sz) for s_, (g, sz) in zip(snaps, looks)]
    every = plays  # (the Foley sounds on every hit of the take, whichever hand fires it)
    if only_hits:
        plays = [plays[i] for i in only_hits if 0 <= i < len(plays)]
        snaps = [snaps[i] for i in only_hits if 0 <= i < len(snaps)]
    offset = k_launch - ign

    say("compositing", 0.7, "Carrying the element on your hand")
    from . import fx_light as fx

    lag = LAG_FRAMES.get(effect, 3)
    pre = 3 if eff.behaviour == "burst" else PREGLOW  # a burst's build-up is short
    src = list(track.get("src") or []) + ["mp"] * n
    floor_lin = float(lin(np.full((1, 1, 3), floor, np.uint8))[0, 0, 0])
    if ign >= pre + 3:  # the render's own black, pixel by pixel, from before anything shows (one value for the whole
        # frame left the painted root's lifted box behind); a render lit from its start keeps the single value
        mins = np.min(element[: ign - pre], axis=0)  # opened: a speck lit before the ignition (in all those frames) left
        # its floor so high that subtracting it cut a dark hole in the fire that later burned over it
        mins = cv2.morphologyEx(mins, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        # grey, and never far above the render's black: a root already glowing before the ignition is fire, not black,
        # and its orange taken off as "black" left the flame's root green
        mins = np.minimum(mins.min(-1, keepdims=True), np.uint8(min(255, int(floor) + FLOOR_SPAN))).repeat(3, -1)
        floor_lin = np.maximum(lin(mins), floor_lin)
    first = next((t for t in range(n) if src[t] not in ("hold", "lost")), n)
    for t in range(first):  # the tracker holds its first sight back to the clip's start: a hand that came in later
        if not still_there(plate, track, first, t):  # wasn't there yet (a punch's fire hung in the air before the fist)
            src[t] = "before"
    track, gone = left_frame(track, w, h)
    lost, run, seen_at = np.zeros(n, int), 0, 0
    for t in range(n):
        if src[t] == "lost" and run >= 8 and still_there(plate, track, seen_at, t):
            src[t] = "hold"  # the tracker lost it, but nothing moved where it was: a still hand, still there
        run = run + 1 if src[t] == "lost" else 0
        seen_at = t if src[t] != "lost" else seen_at
        lost[t] = run
        if gone[t]:  # it left the frame: the fire lets go at once, where the hand went out
            lost[t] = max(lost[t], 8 + gone[t])

    hands_, most_ = SIZE.get(eff.behaviour, (LENGTH, 0.6))
    reach_px = min(hands_ * ref[2], most_ * min(w, h))  # the element's own reach: its root fades over a share of it (a
    # share of six hands faded a palm-sized fireball out to its top, and it hung above the hand)
    fade = root_fade(h, w, ref, ROOT_FADE_SHARE.get(effect, ROOT_FADE) * reach_px) * border_fade(h, w)
    if eff.fluid == "steam":  # vapour thins out as it spreads: dense at the palm, a third as dense at its far end
        p0_, u0_, s0_ = ref
        yy_, xx_ = np.mgrid[:h, :w].astype(np.float32)
        along_ = ((xx_ - p0_[0]) * u0_[0] + (yy_ - p0_[1]) * u0_[1]) / max(0.5 * LENGTH * s0_, 1.0)
        fade = fade * np.clip(1.0 - 0.67 * along_, 0.33, 1.0)
        del yy_, xx_, along_
    fade = fade[..., None]
    cam = camera_path(plate, mask, track)  # a handheld camera: what was thrown moves with the room

    if charge:  # the hum's energy: quick to rise, slow to let go (the ember keeps a little between breaths)
        a_, r_ = charge
        raw = np.asarray(env_all[a_:r_ + 3], np.float64) if r_ + 3 <= env_all.size else np.zeros(r_ + 3 - a_)
        lo_, hi_ = float(np.percentile(raw[: r_ - a_], 10)), float(raw[: r_ - a_].max())  # (the envelope is in dB: a
        raw = np.clip((raw - lo_) / max(hi_ - lo_, 1e-6), 0.0, 1.0)  # swell to four times as loud is only +12 dB of it)
        energy = np.zeros_like(raw)
        for i, v in enumerate(raw):
            energy[i] = v if i == 0 or v > energy[i - 1] else energy[i - 1] * 0.85 + v * 0.15
        energy = np.clip(energy, 0.0, 1.0)
        flicker = 1.0 + 0.12 * np.random.default_rng(spec.seed).standard_normal(energy.size)

    def ember_at(t: int) -> np.ndarray | None:
        """The charging ember on plate frame t: the render's first fire, small and looped, on the emitter, as big and
        as bright as the hum is loud; it fades as the strike's own fire takes over."""
        i = t - charge[0]
        if not 0 <= i < energy.size:
            return None
        k = int(np.clip(ign + abs((i % (2 * EMBER_LOOP)) - EMBER_LOOP), 0, len(element) - 1))
        en = float(energy[min(i, charge[1] - charge[0] - 1)])
        size = EMBER_SIZE[0] + (EMBER_SIZE[1] - EMBER_SIZE[0]) * en ** 0.7  # (grows early: the hum is heard building)
        gain = (0.45 + 1.0 * en) * float(flicker[i])  # a hot core, brighter than the fire it becomes * (1.0 if t < charge[1] else (charge[1] + 3 - t) / 4)
        E = warp_element(source(k), track, ref, t, 0, size, blur=False)
        return E * gain

    lit_box: dict[int, tuple[int, int, int, int] | None] = {}

    def source(e: int) -> np.ndarray:
        """Render frame e, linear, its black taken off and its root faded: worked out only where it is lit (most of a
        render over black is black, and this ran on every pixel of every frame for every hit)."""
        if e not in lit_box:
            look = element[e][::4, ::4].max(-1) > floor + 2
            ys, xs = np.nonzero(look)
            lit_box[e] = None if ys.size == 0 else (max(0, 4 * ys.min() - 4), min(h, 4 * ys.max() + 8),
                                                     max(0, 4 * xs.min() - 4), min(w, 4 * xs.max() + 8))
        out_ = np.zeros((h, w, 3), np.float32)
        if lit_box[e] is not None:
            y0, y1, x0, x1 = lit_box[e]
            fl = floor_lin[y0:y1, x0:x1] if np.ndim(floor_lin) else floor_lin
            out_[y0:y1, x0:x1] = np.clip(lin(element[e][y0:y1, x0:x1]) - fl, 0.0, None) * fade[y0:y1, x0:x1]
        return out_

    def element_at(t: int) -> np.ndarray | None:
        """The element on plate frame t (linear, over black): every hit's play of it, carried on the hand, its root
        held out; None when none is on (before its build-up, after it ends)."""
        acc = None
        for off, gain, size in plays:
            e = t - off
            if eff.behaviour == "burst" and e > ign:  # a smaller blast is over sooner (a fireball lasts as long as it
                e = ign + int(round((e - ign) / size))  # is big): a "pf" puffs and is gone, a "BOOM" rolls on
            if not max(0, ign - pre) <= e < len(element):  # (a negative frame would wrap round to the render's end)
                continue
            grow = min(1.0, GROW_FROM + (1.0 - GROW_FROM) * max(0.0, e - ign + pre) / (pre + GROW_FRAMES))
            if eff.behaviour == "burst":  # (a burst is the blast itself: no growing into it)
                grow = 1.0
            E = warp_element(source(e), track, ref, t, lag, size * grow, since=off + ign, cam=cam) * gain
            if e < ign:  # the build-up, coming up to the ignition
                E *= (e - (ign - pre) + 1) / (pre + 1)
            elif ign < pre and e < pre:  # a render lit from its first frame: bring it up rather than pop it on
                E *= (e + 1) / (pre + 1)
            acc = E if acc is None else np.maximum(acc, E)  # two plays overlapping: the brighter shows, not both added
        if charge and charge[0] <= t < charge[1] + 3:  # the ember the hum charges, lighting the fingers
            ember = ember_at(t)
            if ember is not None:
                acc = ember if acc is None else np.maximum(acc, ember)
        if acc is None:
            return None
        E = acc
        if src[t] == "before":  # the hand hasn't come in yet
            return None
        if lost[t] > 8:  # the hand is gone: let the element go rather than freeze it
            E *= max(0.0, 1.0 - (lost[t] - 8) / LOST_FADE)
        m_t = None if mask is None else mask[min(t, len(mask) - 1)]
        y0, y1, x0, x1 = box = holdout_box(track, t, h, w)
        if y1 > y0 and x1 > x0:
            hold = root_holdout(track, t, h, w, None if m_t is None else m_t.astype(np.float32) / 255.0, box)
            E[y0:y1, x0:x1] *= (1.0 - hold)[..., None]
        return E

    q = 8  # the light (cuesync.fx_light) needs only a small copy of each frame's element
    # the camera's exposure jumps by what the fire adds to the room's light: a stop or more in the dark, nothing to
    # speak of in a sunlit room (where the flash washed the whole frame out)
    k_full = int(np.clip(ign + 4, 0, len(element) - 1))
    fire_mean = float(fx._lum(np.clip(lin(element[k_full][::4, ::4]) - (floor_lin[::4, ::4] if np.ndim(floor_lin)
                                                                         else floor_lin), 0, None)).mean())
    flash_k = float(min(1.0, np.log2(1.0 + fire_mean / max(float(fx._lum(lin(plate[n // 2][::4, ::4])).mean()), 1e-4))))
    some = np.stack([lin(plate[i]) for i in np.linspace(0, n - 1, 8).astype(int)])
    white, tint = fx.plate_white(some), fx.plate_tint(some)
    del some
    noise = fx.shadow_noise(plate)
    out = np.empty_like(plate)
    CHUNK = 12  # frames lit at a time (a whole 10 s 1080p clip in float32 is 6 GB), overlapping a frame each side
    made: dict[int, np.ndarray | None] = {}  # each frame's element, carried once (it was carried twice: small, then full)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        a0, b1 = max(0, a - 1), min(n, b + 1)
        for i in range(a0, b1):
            if i not in made:
                made[i] = element_at(i)
        small = np.zeros((b1 - a0, h // q, w // q, 3), np.float32)
        for i in range(a0, b1):
            if made[i] is not None:
                small[i - a0] = cv2.resize(made[i], (w // q, h // q), interpolation=cv2.INTER_AREA)
        P = np.stack([lin(plate[i]) for i in range(a0, b1)])
        m = None if mask is None else mask[a0:b1].astype(np.float32) / 255.0
        fx.light_plate(P, small, track_slice(track, a0, b1), m, fluid=eff.fluid, white=white, tint=tint, out=P)
        for i in range(a, b):
            E = made[i]
            if E is None:
                continue
            look = E[::4, ::4].max(-1) > 1e-4  # grain only where the element is
            ys_, xs_ = np.nonzero(look)
            if ys_.size:
                y0, y1 = max(0, 4 * ys_.min() - 4), min(h, 4 * ys_.max() + 8)
                x0, x1 = max(0, 4 * xs_.min() - 4), min(w, 4 * xs_.max() + 8)
                E[y0:y1, x0:x1] = fx.grain(E[None, y0:y1, x0:x1], noise, seed=spec.seed, frame0=i)[0]
            j = i - a0
            if eff.fluid == "smoke":  # smoke is laid over; fire is light, added over its soot; steam is lit vapour
                al = np.clip(E.max(-1) / 0.35, 0.0, 0.9)[..., None]
                P[j] = P[j] * (1.0 - al) + np.clip(E / np.maximum(al, 1e-3), 0, 1) * al
            elif eff.fluid == "steam":  # steam scatters the room's light: laid over, never brighter than its white
                al = np.clip(fx._lum(E) / 0.6, 0.0, 0.55)[..., None]  # translucent: a dense render read as cotton wool
                P[j] = P[j] * (1.0 - al) + np.minimum(E / np.maximum(al, 1e-3), max(0.9 * white, 0.45)) * al
            else:  # fire is light, added, and its soot and glare hide some of what is behind it
                a_ = np.clip((fx._lum(E) - SOOT_FROM) / SOOT_FULL, 0.0, 1.0) ** SOOT_GAMMA
                cover = float(a_.sum())
                if cover > 0:  # how bright the room is behind it
                    behind = float((fx._lum(P[j]) * a_).sum()) / cover
                    E = E * np.float32(1.0 + DAY_GAIN * min(1.0, behind / fx.EXPOSED_WHITE))
                P[j] = P[j] * (1.0 - SOOT * a_)[..., None] + E
        if eff.fluid == "fire":  # the hot air above and around the flame shimmers
            alpha = np.stack([cv2.resize(np.clip(fx._lum(small[i - a0]) * 4.0, 0, 1), (w, h)) for i in range(a0, b1)])
            fx.heat_haze(P, alpha, seed=spec.seed, out=P, frame0=a0)
        fx.flash(P, [off + ign - a0 for off, _, _ in plays], [g * flash_k for _, g, _ in plays], out=P)
        if eff.fluid == "fire":  # a shoulder, not a clip: the root over a lit palm went flat white
            shoulder(P)
        out[a:b] = gam(P[a - a0: b - a0])
        for i in [k for k in made if k < b - 1]:  # the next chunk starts a frame back: keep that one and the next
            del made[i]
    plate = element = small = made = None  # (a 10 s 1080p plate and its render are 2 GB: let them go before the finishing copies)
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
    audio = soundtrack(place_audio(c.samples, c.offset_s, n), eff, n, env, [off + ign for off, _, _ in every],
                       [g for _, g, _ in every], cue.voice_end_frame, spec.seed)
    comp_path, final_path = out_dir / "composite.mp4", out_dir / "final.mp4"
    k_poster = int(np.clip(k_launch + 8, 0, len(frames) - 1))  # the player's still: the fire at its fullest, not a dark frame
    cv2.imwrite(str(out_dir / "poster.jpg"), cv2.cvtColor(np.ascontiguousarray(frames[k_poster]), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    write_video(final_path, _finalise(frames, env, spec.label or effect.replace("_", " "), None), fps=FPS, audio=audio,
                crf=16, preset=FINISH_PRESET)
    receipt = {
        "cuesync": __version__, "mode": "bending", "effect": eff.to_json(),
        "inputs": {"point": list(spec.point), "seed": spec.seed, "offset_s": spec.offset_s, "cue": c.source},
        "render": {"seconds": round(seconds, 1), "prompt": prompt, "guide_scale": round(scale, 3)},
        "bending": {"track": str(track_path), "mask": None if mask is None else str(mask_path), "k_hit": int(k_launch),
                    "tap_frame": int(tap_k), "aim": None if arrow is None else round(float(arrow), 2),
                    "k_launch": int(k_launch), "ignition": int(ign), "offset": int(offset), "lag_frames": lag,
                    "snap": snap, "hits": [{"frame": int(off + ign), "gain": round(g, 3), "size": round(sz, 3),
                                            "snap": sn} for (off, g, sz), sn in zip(plays, snaps)],
                    "ref": {"xy": [round(float(v), 2) for v in ref_saved[0]],
                            "dir": [round(float(v), 5) for v in ref_saved[1]], "scale": round(float(ref_saved[2]), 2)},
                    "performed_again": None if old is None or instant else {"from": str(element_from), "no_new_render": True},
                    "instant": None if not instant else {"library": str(element_from), "no_gpu": True},
                    "camera": "locked" if cam is None else "handheld: thrown fire moves with the room",
                    "charge": None if not charge else {"from": int(charge[0]), "release": int(charge[1]),
                                                       "seconds": round((charge[1] - charge[0]) / FPS, 2)}},
        "auto": auto or picked, "sound": SOUND, "seconds_total": round(time.perf_counter() - t0, 1),
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
    # the checkout: beside this source (an editable install), else the folder the server runs in or one above it (a
    # regular install puts this file in site-packages, far from the checkout's helper environments)
    roots = [Path(__file__).resolve().parents[2].parent, Path.cwd(), *Path.cwd().parents]
    dirs = [d for r in roots for d in (r / "cuesync" / f".venv-{kind}", r / "runs" / "bending" / f"venv-{kind}")]
    for d in dirs:
        exe = d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if exe.exists():
            return str(exe)
    return sys.executable


READY_MARK = ".cuesync-ready"  # written into a helper environment once `cuesync hands --install` has checked it works
_ready: dict[str, tuple[bool, float]] = {}


def helpers_ready() -> bool:
    """Can this machine follow a hand: a tracker environment that works (cuesync/.venv-track with its ready mark, or one
    set up by hand or $CUESYNC_TRACK_PY that imports MediaPipe), or MediaPipe in this one? An environment still being
    installed is not ready (its Python exists long before its packages do). Without it the studio doesn't offer a hand
    of your own (the sample hand still plays)."""
    import importlib.util
    import time

    py = hand_python("track")
    if py == sys.executable:
        return all(importlib.util.find_spec(m) is not None for m in ("mediapipe", "onnxruntime", "av"))
    venv = Path(py).parent.parent
    if (venv / READY_MARK).exists():
        return True
    if venv.name == ".venv-track" and venv.parent.name == "cuesync":  # the installer's own: its mark or nothing (it
        return False  # exists, half installed, while `cuesync hands --install` runs)
    ok, when = _ready.get(py, (False, 0.0))
    if not ok and time.time() - when > 30:  # set up by hand: asked at most every 30 s while it isn't ready
        try:
            ok = subprocess.run([py, "-c", "import mediapipe, onnxruntime, av"], capture_output=True,
                                timeout=120).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        _ready[py] = (ok, time.time())
    return ok


def package_root() -> Path:
    """The folder holding this ``cuesync`` package for a helper's PYTHONPATH: the checkout's cuesync/src when there is
    one (never a whole site-packages: the helper would take the core's opencv-python with it, which MediaPipe's
    opencv-contrib clashes with), else this package's own folder's parent."""
    here = Path(__file__).resolve().parents[1]
    if here.name != "site-packages":
        return here
    for r in (Path.cwd(), *Path.cwd().parents):
        if (r / "cuesync" / "src" / "cuesync" / "bending.py").exists():
            return r / "cuesync" / "src"
    return here


def prepare_hand(plate: Path, out_dir: Path, tap: tuple[float, float], frame: int, emitter: str,
                 say=None) -> tuple[Path, Path | None]:
    """Track the tapped hand through ``plate`` (the clip as the pipeline loads it; ``tap`` in px on ``frame``) and matte
    it, each as a subprocess in its own venv; returns (hand_track.json, hand_mask.npz or None: without a matte the
    element's root is held out behind the fingertips instead)."""
    env = os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(package_root()), os.environ.get("PYTHONPATH")]))}
    track, mask = out_dir / "hand_track.json", out_dir / "hand_mask.npz"
    r = subprocess.run([hand_python("track"), "-m", "cuesync.hand_track", str(plate), "--emitter", emitter,
                        "--seed", f"{frame},{tap[0]:.1f},{tap[1]:.1f}", "--out", str(track)],
                       env=env, capture_output=True, text=True, timeout=900)
    (out_dir / "hand_track.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not track.exists():
        last = (r.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"couldn't follow your hand through the clip ({last})")
    if say:
        say("matting", 0.09, "Cutting your hand out, so your fingers can pass in front of the fire")
    try:
        r = subprocess.run([hand_python("mask"), "-m", "cuesync.hand_mask", str(plate), "--track", str(track), "--out",
                            str(mask)], env=env, capture_output=True, text=True, timeout=900)
        (out_dir / "hand_mask.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        ok = r.returncode == 0 and mask.exists()
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    return track, mask if ok else None


LIBRARY_EFFECTS = ("fire_breath", "fire_burst", "flame_plume", "steam_jet")  # rendered ahead, for a hand with no GPU


def library_dir() -> Path | None:
    """The hand elements rendered ahead of time (samples/library/bending/<effect>/: element.mp4 + receipt.json), in the
    checkout the server runs in or one above it; None when there are none."""
    for r in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[3]):
        d = r / "samples" / "library" / "bending"
        if d.is_dir() and any((d / e / "element.mp4").exists() for e in LIBRARY_EFFECTS):
            return d
    return None


def instant_effect(spec, track: dict, n: int) -> tuple[str, dict | None]:
    """The effect for a hand with no GPU: the one asked for, else what the take's sound and the hand's gesture pick
    (as make_bending_shot does), held to what the library has (the air blast is made, not rendered: always there)."""
    from .gestures import heard
    from .pipeline import FPS, SR, shot_cue

    effect, picked = spec.effect, None
    if effect == "auto":
        c = shot_cue(spec, n / FPS)
        hits = [hh.frame for hh in c.cue.hits()] or [c.cue.frame_of(c.cue.hero_onset_s)]
        env = np.asarray(c.cue.envelope, np.float64)
        hero = max(hits, key=lambda f: float(env[f: f + 8].max()) if 0 <= f < env.size else 0.0)
        span = (max(hits) - min(hits)) / FPS if len(hits) > 1 else 0.0
        h_, g = heard(c.cue, c.samples, SR, c.source), gesture_at(track, hero)
        effect, why = bending_effect(h_["kind"], g, hits=len(hits), span_s=span)
        effect = effect or "fire_burst"
        picked = h_ | {"heard": h_["kind"], "hand": g, "picked": effect, "why": why}
    lib = library_dir()
    if effect != "smoke_puff" and (lib is None or not (lib / str(effect) / "element.mp4").exists()):
        effect = "fire_burst"
        picked = None if picked is None else picked | {"picked": effect, "why": picked["why"] + " (the library's nearest)"}
    return effect, picked


def bending_job(spec, renderer, out_dir: Path, progress=None, reuse: Path | None = None):
    """The studio's bending shot from your clip: the plate as the pipeline loads it (kept, so performing it again reads
    the same frames), the hand you tapped (``spec.point`` on frame ``spec.extra["tap_frame"]``) tracked and matted, then
    make_bending_shot, aimed by ``spec.direction`` when you drew the arrow. ``spec`` is updated to the loaded plate.
    ``reuse``: an earlier bending shot of the same clip and tap (another effect picked on the badge): its loaded plate,
    track and matte are taken as they are, so only the new element is rendered."""
    import shutil
    from .gestures import heard
    from .media import write_video
    from .pipeline import FPS, SR, load_plate, shot_cue

    out_dir = Path(out_dir)
    (out_dir / "in").mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    if reuse is not None and (Path(reuse) / "hand_track.json").exists() and (Path(reuse) / "in" / "plate24.mp4").exists():
        reuse = Path(reuse)
        rc = json.loads((reuse / "receipt.json").read_text())
        shutil.copyfile(reuse / "in" / "plate24.mp4", out_dir / "in" / "plate24.mp4")
        for name in ("hand_track.json", "hand_mask.npz"):
            if (reuse / name).exists():
                shutil.copyfile(reuse / name, out_dir / name)
        spec.plate, spec.point = out_dir / "in" / "plate24.mp4", tuple(rc["inputs"]["point"])
        spec.extra["tap_frame"] = rc["bending"].get("tap_frame", spec.extra.get("tap_frame"))
        mask = out_dir / "hand_mask.npz"
        say("reading", 0.05, "Your hand as it was tracked for the last shot")
        return carry(spec, renderer, out_dir, out_dir / "hand_track.json", mask if mask.exists() else None, progress,
                     spec.extra.get("tap_frame"))
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
    track, mask = prepare_hand(loaded, out_dir, (spec.point[0] * w, spec.point[1] * h), k, emitter, say)
    return carry(spec, renderer, out_dir, track, mask, progress, k)


def carry(spec, renderer, out_dir: Path, track: Path, mask: Path | None, progress, tap_frame: int | None):
    """make_bending_shot for the studio: rendered on the GPU (``renderer``), or with none an element rendered ahead of
    time from the library, carried on your hand (an instant shot; the air blast is made, not rendered)."""
    if renderer is not None:
        return make_bending_shot(spec, renderer, out_dir, track, mask, progress, tap_frame=tap_frame, aim=spec.direction)
    t = load_track(track)
    spec.effect, picked = instant_effect(spec, t, len(t["xy"]))
    lib = None if spec.effect == "smoke_puff" else library_dir() / spec.effect
    return make_bending_shot(spec, None, out_dir, track, mask, progress, element_from=lib, tap_frame=tap_frame,
                             aim=spec.direction, instant=lib is not None, picked=picked)


def bending_again(src_dir: Path, spec, out_dir: Path, progress=None):
    """Perform a finished bending shot again with a new take (``spec``): its element on the same hand take, no new
    render; a shot that was itself performed again plays its first render."""
    src_dir = Path(src_dir)
    b = json.loads((src_dir / "receipt.json").read_text())["bending"]
    first = Path(b["performed_again"]["from"]) if b.get("performed_again") else \
        Path(b["instant"]["library"]) if b.get("instant") else src_dir
    here = lambda name, was: str(first / name) if (first / name).exists() else was  # noqa: E731  (a shot made ahead
    # travels with its own track and matte)
    return make_bending_shot(spec, None, out_dir, here("hand_track.json", b["track"]), here("hand_mask.npz", b["mask"]),
                             progress, element_from=first, tap_frame=spec.extra.get("tap_frame"),
                             aim=spec.direction if spec.direction is not None else b.get("aim"))
