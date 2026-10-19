"""Lightning made from the hand (cuesync.elements): arcs struck from the fingertips, or a crackling ball of them over
an open palm, drawn frame by frame from the hand's track and lit onto the plate.

- **Channels.** A bolt is a jagged line (recursive midpoint displacement: each midpoint pushed across its segment by a
  share of that segment's length, gently at the coarse levels so the channel runs on, sharply at the fine ones so it
  kinks) with forks that leave it more often towards its far end, each dimmer and thinner, as a streamer branches when
  it runs out.
- **Strikes.** A channel is struck and then re-struck down the same path for 1-3 frames (real lightning's return
  strokes), flashing or going dark from frame to frame, before a new path is found. While it lives, its root rides the
  fingertip (or the palm) and its far end stays where it struck (carried by the camera on a handheld shot): a path
  drawn anew every frame jumped about like static on a screen, and one drawn once and carried whole with the hand
  swung its far end through the room. A new path leaves near its stream's last aim, which sways slowly, so the
  streams sweep rather than jump.
- **The ball.** Filaments from its middle out to its skin (in 3D, turning slowly), arcs crawling over its skin and
  arcs jumping from it to a fingertip, round a body of glowing blue air that is softest at its edge: no rim is drawn,
  so it is no disc pasted on the palm.
- **The voice.** The louder the voice, the more streams leave each finger, the further they reach and the thicker
  they are; a hit strikes every stream afresh at once, a little further. Between strikes, and for a moment after the
  voice stops, a faint crackle stays at the fingertip. The ball grows while the voice holds and shrinks when it stops.
- **The look.** The channel itself is far over white (a phone clips it to white with a blue rim), the air round it
  glows blue, then violet further out; the glow of many channels piled up is held under GLOW_CAP, so a ball of them
  is blue light over the palm, not a white disc. A palm is often already at the top of red in the picture, where
  added blue light can only turn it pink-white, so over the ball the glow and the body hide a share of the palm
  behind them (GLOW_VEIL, BALL_VEIL), which then reads blue-violet with its lines still in it. The fingertip burns
  white-blue; the ball's middle is a small white-violet spark.
- **The light it throws** (cuesync.fx_light's physics): every lit pixel of the channels lights the hand close in front
  of it and the room further behind, shaded on the round hand, times each surface's own colour, so skin keeps its
  texture under it. The large-area light does not strobe with the channels (a photosensitivity limit: at most about 3
  big changes a second): its strength holds a few frames at its peak and lets go slowly, and its shape is pooled over
  the last few frames' channels; only the channels and their glow flicker frame to frame. On the hand it is capped
  (HAND_CAP), so the palm under a ball keeps its own detail.

Everything is numpy and OpenCV; the random parts come from ``seed`` (the strikes, in one pass over the frames, then
each frame's own draws from the seed and its index), so the same inputs give the same bytes on any number of threads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from . import fx_light as fx
from .elements import FORMS, Timeline
from .hand_track import REACH_OUT, _pointing, finger_reach
from .transfer import gam, lin

_STRIKES, _CRACKLE, _FRAME, _SPARK = 11, 12, 13, 14  # mixed into each seed: each random part its own stream

# ---- the voice
GATE = (0.15, 0.25)  # energy: under the first a sound is a breath or a click, not the voice (it lit stray sparks on a
# silent hand); over the second it plays in full, faded in between
CRACKLE_LEFT = 0.72  # the crackle at the fingertip outlasts the voice, dying by this share a frame (about 0.3 s)
CRACKLE_MIN = 0.06  # below this the crackle is gone

# ---- arcs from the fingertips
ARC_REACH = (1.4, 5.0)  # hand sizes: how far a stream reaches, from the quietest voice to the loudest
ARC_WIDTH = (1.3, 3.6)  # px for a hand HAND_PX across: the main channel's width from the quietest voice to the loudest
HAND_PX = 220.0  # widths are drawn for a hand this many px across (the dev takes') and scale with it, within WIDTH_K
WIDTH_K = (0.6, 1.8)
ARC_STREAMS = 3  # at most this many streams per finger (one for a quiet voice)
ARC_FAN = 18.0  # degrees between the outer streams' aims (at 22 an outer one read as aimed off the finger)
ARC_WANDER, ARC_WANDER_HZ = 5.0, (0.6, 1.7)  # degrees and Hz: each stream's aim sways slowly (a new path anywhere in
# the fan read as random flashes, not one storm)
ARC_JITTER = 4.0  # degrees: each new path's own turn off its stream's aim
LIFE, LIFE_P = (1, 2, 3), (0.25, 0.45, 0.3)  # frames a path is kept, and how often each
RESTROKE = (0.45, 0.5)  # a re-stroke flashes with this chance, plus this times the level; else the channel is dark
HIT_REACH = 0.3  # a hit's strike reaches this much further at full size
ROUGH, FORK = 0.2, (0.07, 0.08)  # a stream's jaggedness (share of a segment), its forking (at the quietest, more when
# louder)
FINGER_NEAR = 0.6  # hand sizes: the tap names the fingertip within this of the emitter; a second finger beside it
# (index and middle) arcs too when it is out straight (REACH_OUT) and its tip within FINGER_PAIR of the first's
FINGER_PAIR = 0.8

# ---- the ball over the palm
BALL_R = (0.15, 0.5)  # hand sizes: its radius, from a voice just begun to one held long and loud
BALL_GROW, BALL_SHRINK = 10.0, 3.0  # frames: it grows towards the voice's level this slowly, and shrinks this fast
# (at 5 a ball hung on, flickering, through the second's pause between two sounds)
BALL_FILAMENTS = (5, 14)  # filaments from its middle to its skin, small to full grown
BALL_SKIN, BALL_GROUND = 4, 2  # arcs crawling over its skin, and arcs jumping from it to a fingertip or knuckle
GROUND_P = (0.25, 0.45)  # a jumping arc strikes with this chance, plus this times the level (more often, and thicker,
# they outshone the ball)
BALL_ROUGH = 0.22  # its filaments' jaggedness (at 0.3 they tangled and sprawled over the palm: no longer a ball)
BALL_BLOOM = 0.12  # its glow in the air round it, at the loudest, fading over its radius: it shines, not only crackles
BALL_TURN = 0.05  # radians a frame: it turns slowly (its filaments' far ends move, but never jump)
BALL_PULSE = 0.12  # a hit swells it by this share, gone over 3 frames
BALL_CORE = 0.05  # radii: its white-violet middle (0.07 made it a white blob over the palm)
BALL_BODY = 0.4  # the glowing air inside it at the loudest (blue, under GLOW_CAP), as deep as the ball is there (a
# sphere): a tangle of filaments alone didn't read as a ball
BALL_VEIL = 0.3  # its body hides this much of the palm behind it at its middle, at the loudest: light added to a palm
# already at the top of red can only push it towards pink-white, and the ball read pink, never blue
BALL_SOFT = 0.25  # radii: its body's edge is softened over this (a sharp edge, and a bright rim drawn on it, made it a
# disc pasted on the palm); the arcs crawling over its skin draw its outline
CONTACT = 0.025  # hand sizes: the spark where a filament meets the ball's skin, facing us
TARGETS = (4, 8, 12, 16, 20, 5, 9, 13)  # the landmarks a grounding arc may jump to: fingertips and knuckles

# ---- the look (linear light)
CORE_RGB = np.array([0.92, 0.96, 1.0], np.float32)  # the channel itself: white, the faintest blue
CORE_GAIN = 3.0  # the channel is far over white; the shoulder takes it to a clipped white with a blue rim
GLOWS = (  # (sigma in hand sizes, peak over a 1 px line at core 1, linear rgb): the air round the channel glows blue,
    (0.008, 0.55, (0.25, 0.50, 1.00)),  # then violet further out (saturated: a paler blue over skin read pink-white)
    (0.025, 0.22, (0.12, 0.30, 1.00)),
    (0.08, 0.08, (0.12, 0.22, 1.00)),
    (0.25, 0.03, (0.25, 0.15, 1.00)),
)
GLOW_CAP = {"arc": 0.6, "ball": 0.3}  # linear luminance: the glow is held under this, softly. A ball's filaments'
# glows piled up to a white disc that hid the palm; arcs over a wall held to 0.15 read as white paint
GLOW_VEIL = {"arc": 0.0, "ball": 0.5}  # where the glow is at its cap it hides this much of what is behind it (over
# the palm, as BALL_VEIL: the ball's glow round its filaments read pink without it); the arcs' glow only adds
DEFOCUS = 0.7  # px at 1080: no phone films a channel as crisply as it is drawn
SPARK_RGB = np.array([0.95, 0.86, 1.0], np.float32)  # the burning fingertip and the ball's middle: white-violet
HAZE_RGB = np.array([0.04, 0.15, 1.0], np.float32)  # the ball's body and the halo round a spark: blue (saturated: under
# the cap, a paler blue only whitened the palm)
CORONA = 0.04  # hand sizes: the burning fingertip's spot

# ---- the light it throws
LIGHT_RGB = np.array([0.3, 0.45, 1.0], np.float32)
LIGHT_RGB = LIGHT_RGB / float(LIGHT_RGB @ fx._LUMA)  # blue, a little violet, at luminance 1 (a bluish white lit skin
# and the room as a plain lamp would)
LIGHT = {"arc": (0.2, 2.5, 5.0), "ball": (0.45, 0.4, 4.0)}  # (how far in front of the hand the channels are, in hand
# sizes; gain on the hand; gain on the room). A thin channel has no radius to stand off by; at fx_light's own
# HAND_DEPTH of it the palm under a ball went white
HAND_CAP = 1.0  # the light may brighten the hand by at most this (+1 stop), softly: at fx_light's 4 a palm under the
# ball went white and lost its lines
LIGHT_HOLD, LIGHT_LETGO = 4, 0.85  # the light's strength holds its peak over this many frames, then lets go by this
# share a frame: following each stroke, the room strobed at 12 Hz
LIGHT_FLICKER = 0.06  # share of the strokes' own flicker kept in the light (under the 10 % a flash is counted from)
LIGHT_POOL = (1.0, 0.7, 0.45, 0.25)  # the light's shape: this frame's channels and the three before, weighted (a
# stream swinging across the fan swung the room's light with it)
FIELD_Q = 6  # the light is worked out on a copy this many times smaller (it is smooth)
LOST_FADE = 4  # frames: a hand the tracker lost lets its lightning go over these
CHUNK = 8  # frames made at a time (their layers are kept until lit)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _rot(u: np.ndarray, deg: float) -> np.ndarray:
    """``u`` turned ``deg`` degrees clockwise on screen (y down)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([u[0] * c - u[1] * s, u[0] * s + u[1] * c])


# ---- channels ------------------------------------------------------------------------------------------------------

def displace(a, b, rng: np.random.Generator, rough: float = ROUGH, min_seg: float = 3.0,
             curve: float = 0.0) -> np.ndarray:
    """A jagged channel from ``a`` to ``b`` (2D or 3D): recursive midpoint displacement down to segments of about
    ``min_seg``, each midpoint pushed across its segment by a Gaussian of ``rough`` times the segment's length (and a
    little along it: kinks, not only sideways). ``curve`` first bows the whole path by that share of its length (2D),
    so a long arc swings rather than runs straight."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return np.stack([a, b])
    pts = np.stack([a, b])
    if curve and a.size == 2:
        n = np.array([-d[1], d[0]]) / length
        pts = np.stack([a, (a + b) / 2 + n * curve * length, b])
    levels = int(np.clip(math.ceil(math.log2(max(length / (len(pts) - 1) / min_seg, 1.0))), 0, 10))
    for lev in range(levels):  # every segment halved at each level: a fractal, no long straight pieces left
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        r = rough * (0.45 + 0.55 * min(1.0, lev / 3.0))  # its course gentle (a channel runs on), its kinks sharp
        dirs = np.diff(pts, axis=0) / np.maximum(seg, 1e-9)[:, None]
        if pts.shape[1] == 2:
            off = np.stack([-dirs[:, 1], dirs[:, 0]], 1) * (rng.normal(0, r, len(seg)) * seg)[:, None]
            off += dirs * (rng.normal(0, 0.2 * r, len(seg)) * seg)[:, None]
        else:  # 3D: any direction across the segment
            off = rng.normal(0, 1, (len(seg), 3))
            off -= (off * dirs).sum(1, keepdims=True) * dirs
            off *= (r * seg)[:, None]
        out = np.empty((2 * len(pts) - 1, pts.shape[1]))
        out[0::2], out[1::2] = pts, (pts[:-1] + pts[1:]) / 2 + off
        pts = out
    return pts


def bolt_tree(p0, u, length: float, rng: np.random.Generator, width: float, rough: float = ROUGH,
              fork: float = FORK[0], min_seg: float = 3.0, inten: float = 1.0, depth: int = 0, max_depth: int = 2,
              out: list | None = None) -> list[tuple[np.ndarray, float, float]]:
    """A bolt from ``p0`` heading ``u`` for ``length`` px with its forks: [(points, width px, intensity)], the main
    channel first. Forks leave at its points, more often towards its far end, 15-50 degrees off its local heading, a
    quarter to two thirds of what is left of it long, dimmer and thinner; they fork once more themselves."""
    out = [] if out is None else out
    p0, u = np.asarray(p0, np.float64), np.asarray(u, np.float64)
    pts = displace(p0, p0 + u * length, rng, rough, min_seg, curve=rng.normal(0, 0.07))
    out.append((pts, width, inten))
    if depth >= max_depth:
        return out
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    for along in np.linspace(0.1, 0.9, 16):  # sixteen places a fork may leave from
        if rng.random() >= fork * (0.3 + 1.4 * along) * (0.5 if depth else 1.0):
            continue
        i = int(np.searchsorted(cum, along * cum[-1]))
        j = min(i + max(2, len(pts) // 40), len(pts) - 1)
        v = _rot(_unit(pts[j] - pts[i]), rng.choice([-1, 1]) * rng.uniform(15, 50))
        bolt_tree(pts[i], v, (cum[-1] - cum[i]) * rng.uniform(0.25, 0.65), rng, width * 0.6, rough, fork, min_seg,
                  inten * rng.uniform(0.35, 0.6), depth + 1, max_depth, out)
    return out


def _between(rng: np.random.Generator, width: float, inten: float, rough: float, min_seg: float, bow: float = 0.25):
    """A short arc from (0, 0) to (1, 0), to be carried onto its two ends (_carry): finger to finger, ball to
    fingertip, a crackle at the tip."""
    return [(displace((0.0, 0.0), (1.0, 0.0), rng, rough, min_seg, curve=rng.normal(0, bow)), width, inten)]


def _carry(tree: list, a0: np.ndarray, b0: np.ndarray, a1: np.ndarray, b1: np.ndarray) -> list:
    """The tree with the similarity that takes (a0, b0) onto (a1, b1) applied to its points: its root moved to where
    the hand is now, its far end to where it struck, the shape between kept."""
    d0, d1 = b0 - a0, b1 - a1
    l0 = float(np.hypot(*d0))
    if l0 < 1e-9:
        return [(p - a0 + a1, w, i) for p, w, i in tree]
    k = float(np.hypot(*d1)) / l0
    turn = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    c, s = math.cos(turn) * k, math.sin(turn) * k
    M = np.array([[c, s], [-s, c]])
    return [((p - a0) @ M + a1, w, i) for p, w, i in tree]


@dataclass
class _Path:
    """A channel kept for its life: struck on ``born`` with its root at ``a`` and its far end at ``b``; ``to``: the
    landmark a grounding arc jumps to (-1: it didn't strike this time)."""

    born: int
    life: int
    a: np.ndarray
    b: np.ndarray
    tree: list
    to: int = -1


# ---- the hand ------------------------------------------------------------------------------------------------------

@dataclass
class _Hand:
    """The track in the plate's pixels: the emitter (n, 2), its direction (n, 2), the hand's size (n,), the landmarks
    (n, 21, 3: MediaPipe's z in px; a frame without them holds the last ones seen) or None, and how much of the
    element shows on each frame (n,) for where the hand is."""

    xy: np.ndarray
    u: np.ndarray
    s: np.ndarray
    lm3: np.ndarray | None
    presence: np.ndarray
    size: tuple[int, int]  # the plate's (w, h)

    @property
    def lm(self) -> np.ndarray | None:
        return None if self.lm3 is None else self.lm3[..., :2]


def _hand(plate: np.ndarray, track: dict) -> _Hand:
    """The track read onto the plate (_Hand). The lightning shows where the tracker saw the hand: not before the hand
    came into the picture (the tracker holds its first sight back to the clip's start), and let go over LOST_FADE
    frames when the tracker lost a hand that moved on (a still one it lost keeps it)."""
    from .bending import still_there

    n, h, w = plate.shape[:3]
    xy, u, s = fx.track_arrays(track, n, w, h)
    kx, ky = w / float(track.get("W", w)), h / float(track.get("H", h))
    lms = (track.get("landmarks") or [])[:n]
    lm3 = None
    if any(v is not None for v in lms):  # (cuesync.bending.filled_landmarks, with the depth kept)
        rows = [None if v is None else np.asarray(v, np.float64).reshape(21, -1) for v in lms]
        rows = [None if r is None else np.c_[r[:, :2], r[:, 2] if r.shape[1] > 2 else np.zeros(21)] for r in rows]
        last = next(r for r in rows if r is not None)
        lm3 = np.zeros((n, 21, 3))
        for t in range(n):
            last = rows[t] if t < len(rows) and rows[t] is not None else last
            lm3[t] = last * (kx, ky, math.sqrt(kx * ky))
    here = track | {"xy": xy.tolist(), "dir": u.tolist(), "scale": s.tolist()}
    src = list(track.get("src") or [])[:n] + ["mp"] * n
    presence = np.ones(n)
    first = next((t for t in range(n) if src[t] not in ("hold", "lost")), None)
    if first is not None:  # the tracker holds its first sight back to the clip's start: a hand that came in later
        for t in range(first):  # wasn't there yet
            if not still_there(plate, here, first, t):
                presence[t] = 0.0
        run, seen = 0, first
        for t in range(first, n):  # lost: let it go rather than freeze it (unless nothing moved where it was)
            gone = src[t] == "lost" and not still_there(plate, here, seen, t)
            run = run + 1 if gone else 0
            seen = t if src[t] not in ("lost", "hold") else seen
            presence[t] = min(presence[t], max(0.0, 1.0 - run / LOST_FADE))
    return _Hand(xy, u, s, lm3, presence, (w, h))


def _smooth(v: np.ndarray, k: int = 2) -> np.ndarray:
    """Zero-phase smoothing over +-k frames (triangular weights) along axis 0."""
    wts = np.concatenate([np.arange(1, k + 2), np.arange(k, 0, -1)]).astype(np.float64)
    p = np.pad(v, [(k, k)] + [(0, 0)] * (v.ndim - 1), mode="edge")
    return sum(wt * p[i: i + len(v)] for i, wt in enumerate(wts)) / wts.sum()


def _fingers(hand: _Hand, tap: int) -> list[int]:
    """The fingertip landmarks the arcs leave from: the one the emitter was carried onto at the tap (the nearest
    within FINGER_NEAR hands of it), and the index or middle finger beside it when that one is out straight next to
    it (two fingers together). [] with no landmarks, or a tap off the fingertips: the emitter alone."""
    if hand.lm is None:
        return []
    k = int(np.clip(tap, 0, len(hand.xy) - 1))
    lm, s = hand.lm[k], hand.s[k]
    tips = (4, 8, 12, 16, 20)
    d = {f: float(np.linalg.norm(lm[f] - hand.xy[k])) for f in tips}
    first = min(d, key=d.get)
    if d[first] > FINGER_NEAR * s:
        return []
    pair = {8: 12, 12: 8}.get(first)
    if pair is not None and finger_reach(hand.lm3[k])[(pair - 8) // 4] > REACH_OUT and \
            np.linalg.norm(lm[pair] - lm[first]) < FINGER_PAIR * s:
        return [first, pair]
    return [first]


def _aims(hand: _Hand, timeline: Timeline, finger: int | None, frame: tuple[int, int]) -> np.ndarray:
    """(n, 2) unit: which way the arcs leave on each frame. The studio's aim carried with the hand (the track's
    direction turned so it points the aim on the tap frame, which it already does when cuesync.bending.anchored turned
    it); without one, the finger's own, from its middle joint to its tip, in 3D (cuesync.hand_track's: a finger
    pointing into the scene points at its vanishing point), smoothed over a few frames (the landmarks jitter). From the
    knuckle, a finger crooked as in a finger gun aimed 20 degrees off its last joint, which is what the eye follows.
    Where the finger points at the camera and is too short in the picture to read, the track's own direction."""
    n = len(hand.xy)
    if timeline.aim is not None or finger is None:
        u = hand.u
        if timeline.aim is not None:
            k = int(np.clip(timeline.tap_frame, 0, n - 1))
            now = math.degrees(math.atan2(u[k, 0], -u[k, 1]))
            u = np.stack([_rot(v, (timeline.aim - now + 180.0) % 360.0 - 180.0) for v in u])
        return _unit(u)
    d = _pointing(hand.lm3[:, finger], hand.lm3[:, finger - 2], frame)
    short = np.linalg.norm(hand.lm[:, finger] - hand.lm[:, finger - 2], axis=-1) / hand.s
    sure = np.clip((short - 0.1) / 0.15, 0, 1)[:, None]
    return _unit(_smooth(d * sure + hand.u * (1 - sure)))


# ---- the voice -----------------------------------------------------------------------------------------------------

def levels(timeline: Timeline) -> np.ndarray:
    """(n,) the voice as the lightning plays it: the energy, faded out under GATE (a breath is not the voice)."""
    e = timeline.energy.astype(np.float64)
    return e * np.clip((e - GATE[0]) / (GATE[1] - GATE[0]), 0.0, 1.0)


def _follow(x: np.ndarray, up: float, down: float) -> np.ndarray:
    """``x`` followed with a time constant of ``up`` frames rising, ``down`` falling."""
    out, c = np.zeros_like(x), 0.0
    for t, v in enumerate(x):
        c += (v - c) * (1.0 - math.exp(-1.0 / (up if v > c else down)))
        out[t] = c
    return out


# ---- the strikes ---------------------------------------------------------------------------------------------------

def _width_k(s: float) -> float:
    return float(np.clip(s / HAND_PX, *WIDTH_K))


def arc_frames(hand: _Hand, timeline: Timeline, seed: int = 0) -> tuple[list[list], list[list]]:
    """Per frame, the arcs from the fingertips: ([(points, width px, intensity)], [(x, y, sigma, amplitude)] glowing
    spots). Each finger's streams are struck, re-struck and let go as the module docstring says; ``Timeline.camera``
    carries a far end with the room on a handheld shot."""
    from .bending import camera_between

    n = timeline.n
    L = levels(timeline)
    crackle = np.zeros(n)
    for t in range(n):
        crackle[t] = max(L[t], crackle[t - 1] * CRACKLE_LEFT if t else 0.0)
    fingers = _fingers(hand, timeline.tap_frame)
    aims = _aims(hand, timeline, fingers[0] if fingers else None, hand.size)
    hits = dict(zip(timeline.hits, timeline.sizes))
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _STRIKES])
    sway = rng.uniform(0, 2 * np.pi, (max(1, len(fingers)), ARC_STREAMS, 2))
    paths: dict[tuple[int, int], _Path] = {}
    bolts, spots = [], []
    for t in range(n):
        s, lv, pres = float(hand.s[t]), float(L[t]), float(hand.presence[t])
        if pres <= 0 or (lv <= 0 and crackle[t] < CRACKLE_MIN):
            paths.clear()
            bolts.append([])
            spots.append([])
            continue
        kw = _width_k(s)
        roots = [hand.xy[t]] if not fingers else \
            [hand.xy[t] + (hand.lm[t, f] - hand.lm[t, fingers[0]]) for f in fingers]  # (the emitter rides the tip)
        streams = 0 if lv <= 0 else 1 + int(round((ARC_STREAMS - 1) * lv))
        hit = hits.get(t)
        lit, out = 0.0, []
        for fi, root in enumerate(roots):
            for j in range(ARC_STREAMS):
                key = (fi, j)
                if j >= streams:
                    paths.pop(key, None)
                    continue
                p = paths.get(key)
                if p is None or t - p.born >= p.life or hit is not None:
                    prng = np.random.default_rng(int(rng.integers(1 << 62)))
                    sec = 2 * math.pi * t / fx.FPS
                    wander = ARC_WANDER * (0.6 * math.sin(ARC_WANDER_HZ[0] * sec + sway[fi, j, 0])
                                           + 0.4 * math.sin(ARC_WANDER_HZ[1] * sec + sway[fi, j, 1]))
                    fan = 0.0 if streams == 1 else ARC_FAN * (j / (streams - 1) - 0.5)
                    v = _rot(aims[t], fan + wander + prng.normal(0, ARC_JITTER))
                    reach = s * (ARC_REACH[0] + (ARC_REACH[1] - ARC_REACH[0]) * lv) * prng.uniform(0.8, 1.05)
                    width = kw * (ARC_WIDTH[0] + (ARC_WIDTH[1] - ARC_WIDTH[0]) * lv) * (1.0 if j == 0 else 0.75)
                    if hit is not None:
                        reach *= 1.0 + HIT_REACH * hit
                    tree = bolt_tree(root, v, reach, prng, width, ROUGH, FORK[0] + FORK[1] * lv, max(2.0, 0.008 * s))
                    p = paths[key] = _Path(t, int(rng.choice(LIFE, p=LIFE_P)), root.copy(), tree[0][0][-1].copy(), tree)
                    flash = 1.0  # a new path: its first, brightest stroke
                else:
                    on = rng.random() < RESTROKE[0] + RESTROKE[1] * lv
                    flash = float(rng.uniform(0.55, 0.9)) if on else float(rng.uniform(0.0, 0.1))
                end = p.b
                if timeline.camera is not None and t > p.born:  # the far end struck the room: it moves with it
                    end = (camera_between(timeline.camera, p.born, t) @ np.r_[p.b, 1.0])[:2]
                gain = flash * (0.55 + 0.45 * lv) * pres
                if gain > 0.02:
                    out += [(q, wd, it * gain) for q, wd, it in _carry(p.tree, p.a, p.b, root, end)]
                lit = max(lit, gain)
        # the crackle at the fingertips: small arcs out of the tip and crawling up the finger, struck every frame
        c = float(crackle[t])
        crng = np.random.default_rng([seed & 0xFFFFFFFF, _CRACKLE, t])
        sp = []
        for fi, root in enumerate(roots):
            back = hand.lm[t, fingers[fi] - 2] if fingers else root - aims[t] * 0.5 * s  # the finger's middle joint
            for _ in range(2 + int(round(2 * c))):
                tip = root + _rot(aims[t], crng.uniform(-110, 110)) * s * crng.uniform(0.08, 0.28)
                out += _carry(_between(crng, 0.9 * kw, 0.45 * (0.4 + 0.6 * c) * crng.uniform(0.5, 1.0) * pres, 0.35,
                                       0.04), np.zeros(2), np.array([1.0, 0.0]), root, tip)
            if crng.random() < 0.5 + 0.4 * c:
                a = back + crng.uniform(-0.12, 0.12, 2) * s
                out += _carry(_between(crng, 0.9 * kw, 0.4 * (0.4 + lit) * pres, 0.35, 0.03), np.zeros(2),
                              np.array([1.0, 0.0]), a, root)
            if fi == 1 and crng.random() < 0.6:  # two fingers: arcs jump between their tips
                out += _carry(_between(crng, kw, 0.55 * (0.4 + lit) * pres, 0.4, 0.03), np.zeros(2),
                              np.array([1.0, 0.0]), roots[0], root)
            sp.append((root[0], root[1], CORONA * s, (0.15 * c + 0.4 * lit * (0.5 + lv)) * pres))
        bolts.append(out)
        spots.append(sp)
    return bolts, spots


def _filament(rng: np.random.Generator) -> list:
    """A ball's filament in the unit ball (3D): from near its middle out to its skin, forking there."""
    d = _unit(rng.normal(0, 1, 3))
    b = d * rng.uniform(0.92, 1.05)
    tree = [(displace(rng.normal(0, 0.05, 3), b, rng, BALL_ROUGH, 0.02), 1.0, 1.0)]
    for _ in range(rng.integers(1, 3) if rng.random() < 0.7 else 0):
        tree.append((displace(b, _unit(d + rng.normal(0, 0.35, 3)), rng, BALL_ROUGH, 0.02), 0.6, 0.5))
    return tree


def _skin(rng: np.random.Generator) -> list:
    """An arc crawling over a ball's skin (3D, the unit ball, as seen: z towards the camera): a jagged stretch of great
    circle, half of them along its outline, which is what draws its edge."""
    a3, b3 = rng.normal(0, 1, (2, 3)) * ((1.0, 1.0, 0.25) if rng.random() < 0.5 else 1.0)
    a3, b3 = _unit(a3), _unit(b3)
    arc = _unit(np.stack([a3 * (1 - x) + b3 * x for x in np.linspace(0, 1, 9)])) * 1.02
    return [(np.concatenate([displace(arc[j], arc[j + 1], rng, 0.25, 0.02)[:-1] for j in range(8)] + [arc[-1:]]),
             0.9, 0.55)]


def ball_frames(hand: _Hand, timeline: Timeline, seed: int = 0) -> tuple[list[list], list[list], list[tuple | None]]:
    """Per frame, the ball over the palm: ([(points, width px, intensity)], [(x, y, sigma, amplitude)] sparks,
    (x, y, radius, level, charge, flicker) or None: the voice's level now, and the charge it has built up, which grows
    and shrinks smoothly and sizes the ball and its body). Filaments from its middle out to its skin (in 3D, turning
    slowly, projected: those at the back dimmer, brightest where they meet the skin, with a spark there), arcs
    crawling over its skin, and arcs jumping from it to a fingertip or knuckle (it is held, not pasted on). Each is
    struck and re-struck for 1-3 frames as a channel is, in the ball's own frame, so it rides the palm."""
    n = timeline.n
    L = levels(timeline)
    charge = _follow(L, BALL_GROW, BALL_SHRINK)  # it grows while the voice holds, and shrinks when it stops
    hits = dict(zip(timeline.hits, timeline.sizes))
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _STRIKES])
    paths: dict[tuple[str, int], _Path] = {}
    pulse = 0.0
    bolts, sparks, balls = [], [], []

    def strike(key: tuple[str, int], make, t: int, lv: float, hit: float | None) -> tuple[_Path, float]:
        """The path in ``key``'s slot on frame t, struck anew when its life is over (or on a hit), and how bright."""
        p = paths.get(key)
        if p is None or t - p.born >= p.life or hit is not None:
            prng = np.random.default_rng(int(rng.integers(1 << 62)))
            to = int(prng.integers(len(TARGETS))) if prng.random() < GROUND_P[0] + GROUND_P[1] * lv else -1
            p = paths[key] = _Path(t, int(rng.choice(LIFE, p=LIFE_P)), np.zeros(2), np.array([1.0, 0.0]),
                                   make(prng), to)
            return p, 1.0
        on = rng.random() < RESTROKE[0] + RESTROKE[1] * lv
        return p, float(rng.uniform(0.55, 0.9)) if on else float(rng.uniform(0.0, 0.1))

    for t in range(n):
        s, lv, pres = float(hand.s[t]), max(float(L[t]), float(charge[t])), float(hand.presence[t])
        hit = hits.get(t)
        pulse = BALL_PULSE * hit if hit is not None else pulse * 0.6
        if pres <= 0 or lv < 0.03:
            paths.clear()
            bolts.append([])
            sparks.append([])
            balls.append(None)
            continue
        kw = _width_k(s)
        c = hand.xy[t]
        R = s * (BALL_R[0] + (BALL_R[1] - BALL_R[0]) * charge[t]) * (1.0 + pulse)
        ang, tilt = BALL_TURN * t, 0.35
        turn = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]) @ \
            np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]])
        out, sp = [], []
        K = int(round(BALL_FILAMENTS[0] + (BALL_FILAMENTS[1] - BALL_FILAMENTS[0]) * charge[t]))
        for key, count, make, wd in (("fil", K, _filament, 0.8 + 0.6 * lv), ("skin", BALL_SKIN, _skin, 1.0)):
            for i in range(count):
                p, flash = strike((key, i), make, t, lv, hit)
                gain = flash * (0.4 + 0.6 * lv) * pres
                if gain <= 0.02:
                    continue
                for q, w_, it in p.tree:
                    q3 = q @ turn.T if key == "fil" else q  # (an arc on its skin lives too briefly to turn)
                    front = 0.55 + 0.45 * float(np.clip(q3[-1, 2], -1, 1))  # those at the back dimmer
                    q2 = c + R * q3[:, :2]
                    if key == "fil" and w_ == 1.0:  # dim inside (the middle is the spark's), brightest at the skin:
                        k_ = int(0.45 * len(q2))  # a ball, not a white star of glows piled up in the middle
                        out += [(q2[: k_ + 1], wd * kw * 0.7, 0.4 * gain * front), (q2[k_:], wd * kw, gain * front)]
                        if q3[-1, 2] > 0.2:
                            sp.append((q2[-1, 0], q2[-1, 1], CONTACT * s, 0.3 * gain * front))
                    else:
                        out.append((q2, w_ * wd * kw, it * gain * front))
        for i in range(BALL_GROUND if hand.lm is not None else 0):  # grounded on the hand
            p, flash = strike(("ground", i), lambda prng: _between(prng, 1.2, 0.75, 0.2, 0.01, 0.1), t, lv, hit)
            tgt = None if p.to < 0 else hand.lm[t, TARGETS[p.to]]
            if tgt is None or flash <= 0.05 or np.linalg.norm(tgt - c) < 1.2 * R:  # (not to a knuckle under the ball)
                continue
            out += [(q, w_ * kw * (0.7 + 0.5 * lv), it * flash * pres) for q, w_, it in
                    _carry(p.tree, p.a, p.b, c + _unit(tgt - c) * R, tgt)]
        bolts.append(out)
        sparks.append(sp)
        flicker = float(np.random.default_rng([seed & 0xFFFFFFFF, _SPARK, t]).uniform(0.7, 1.0))
        balls.append((float(c[0]), float(c[1]), float(R), lv * pres, float(charge[t]) * pres, flicker))
    return bolts, sparks, balls


# ---- drawing -------------------------------------------------------------------------------------------------------

def _box(bolts: list, spots: list, s: float, h: int, w: int) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1): where the channels and spots are, with room for their widest glow."""
    xs, ys = [], []
    for p, _, it in bolts:
        if it > 0 and len(p) >= 2:
            xs += [p[:, 0].min(), p[:, 0].max()]
            ys += [p[:, 1].min(), p[:, 1].max()]
    for x, y, r, a in spots:
        if a > 0:
            xs += [x - 3 * r, x + 3 * r]
            ys += [y - 3 * r, y + 3 * r]
    if not xs:
        return None
    pad = 3 * GLOWS[-1][0] * s + 8
    y0, y1 = max(0, int(min(ys) - pad)), min(h, int(max(ys) + pad) + 1)
    x0, x1 = max(0, int(min(xs) - pad)), min(w, int(max(xs) + pad) + 1)
    return None if y1 <= y0 or x1 <= x0 else (y0, y1, x0, x1)


def draw(bolts: list, box: tuple[int, int, int, int], seed: int = 0, frame: int = 0) -> np.ndarray:
    """The channels' core in ``box`` ((y0, y1, x0, x1) of the frame), (bh, bw) float32 linear: each polyline drawn
    anti-aliased at sub-pixel precision, thinning towards its end, in brighter and dimmer stretches; where they cross,
    the brighter shows."""
    y0, y1, x0, x1 = box
    core = np.zeros((y1 - y0, x1 - x0), np.float32)
    tmp = np.zeros_like(core, np.uint8)
    rng = np.random.default_rng([seed & 0xFFFFFFFF, _FRAME, frame])
    for pts, width, inten in bolts:
        if len(pts) < 2 or inten <= 0:
            continue
        p = pts[:, :2] - (x0, y0)
        a0 = np.floor(p.min(0)).astype(int) - int(width) - 3
        a1 = np.ceil(p.max(0)).astype(int) + int(width) + 3
        bx0, by0 = max(0, a0[0]), max(0, a0[1])
        bx1, by1 = min(core.shape[1], a1[0]), min(core.shape[0], a1[1])
        if bx1 <= bx0 or by1 <= by0:
            continue
        sub = tmp[by0:by1, bx0:bx1]
        sub[:] = 0
        chunks = 8
        bead = rng.uniform(0.6, 1.0, chunks)
        for c in range(chunks):
            a, b = c * (len(p) - 1) // chunks, (c + 1) * (len(p) - 1) // chunks + 1
            if b - a < 2:
                continue
            wd = width * (1.0 - 0.6 * c / chunks)
            th = max(1, int(round(wd)))
            lv = int(np.clip(255 * bead[c] * min(1.0, wd / th), 30, 255))
            q = np.round((p[a:b] - (bx0, by0)) * 4).astype(np.int32)  # (2 fractional bits: sub-pixel)
            cv2.polylines(sub, [q.reshape(-1, 1, 2)], False, lv, th, cv2.LINE_AA, shift=2)
        reg = core[by0:by1, bx0:bx1]
        np.maximum(reg, sub.astype(np.float32) * np.float32(inten / 255.0), out=reg)
    return core


def _spot(out: np.ndarray, box: tuple, x: float, y: float, sigma: float, amp: float) -> None:
    """Adds a Gaussian spot to ``out`` (a layer over ``box``)."""
    y0, _, x0, _ = box
    r = int(4 * sigma) + 2
    cx, cy = x - x0, y - y0
    a0, a1 = max(0, int(cx) - r), min(out.shape[1], int(cx) + r)
    b0, b1 = max(0, int(cy) - r), min(out.shape[0], int(cy) + r)
    if a1 <= a0 or b1 <= b0 or amp <= 0:
        return
    xx = (np.arange(a0, a1, dtype=np.float32) - cx) ** 2
    yy = (np.arange(b0, b1, dtype=np.float32)[:, None] - cy) ** 2
    out[b0:b1, a0:a1] += np.float32(amp) * np.exp(-(xx + yy) / np.float32(2 * sigma * sigma))


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur, done small when it is wide (it is smooth)."""
    h, w = x.shape[:2]
    q = int(np.clip(sigma / 3, 1, 16))
    if q == 1 or min(h, w) < 4 * q:
        return cv2.GaussianBlur(x, (0, 0), max(sigma, 0.3))
    small = cv2.resize(x, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / q), (w, h), interpolation=cv2.INTER_LINEAR)


def layer(bolts: list, spots: list, ball: tuple | None, s: float, box: tuple, side: int, cap: float, hide: float,
          seed: int = 0, frame: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Over ``box``: linear RGB (bh, bw, 3), the channels' white-hot core and the blue-violet glow round them, the
    burning spots (white-violet, a blue halo) and the ball's middle and body; and (bh, bw) how much of the plate behind
    them is hidden (the glow at its cap hides ``hide`` of it, the ball's body BALL_VEIL). The glow is held under
    ``cap`` (GLOW_CAP)."""
    core = draw(bolts, box, seed, frame)
    if DEFOCUS > 0:
        core = cv2.GaussianBlur(core, (0, 0), DEFOCUS * side / 1080)
    out = core[..., None] * (CORE_RGB * np.float32(CORE_GAIN))
    glow = np.zeros_like(out)
    for sig_h, g, rgb in GLOWS:  # (a blur spreads a line's energy: scaled so its peak over a 1 px line is g)
        sig = sig_h * s
        glow += _blur(core, sig)[..., None] * (np.float32(g * sig * math.sqrt(2 * math.pi)) * np.float32(rgb))
    spark = np.zeros(core.shape, np.float32)
    for x, y, r, a in spots:
        _spot(spark, box, x, y, r, a)
    haze = np.zeros(core.shape, np.float32)
    veil = np.zeros(core.shape, np.float32)
    if ball is not None:
        x, y, R, lv, full, flick = ball
        _spot(spark, box, x, y, BALL_CORE * R, (0.6 + 0.6 * lv) * flick)
        y0, _, x0, _ = box
        r = int(2.5 * R) + 2
        a0, a1 = max(0, int(x - x0) - r), min(core.shape[1], int(x - x0) + r)
        b0, b1 = max(0, int(y - y0) - r), min(core.shape[0], int(y - y0) + r)
        if a1 > a0 and b1 > b0:
            d = np.hypot(np.arange(a0, a1, dtype=np.float32) - (x - x0), np.arange(b0, b1, dtype=np.float32)[:, None]
                         - (y - y0))
            deep = np.sqrt(np.clip(1.0 - (d / R) ** 2, 0.0, 1.0))  # the glowing air, as deep as the ball is there
            deep = _blur(deep, BALL_SOFT * R)
            bloom = np.exp(-0.5 * (d / R) ** 2)  # its glow in the air round it
            haze[b0:b1, a0:a1] = (BALL_BODY * (0.3 + 0.7 * full)) * deep + (BALL_BLOOM * full) * bloom
            veil[b0:b1, a0:a1] = (BALL_VEIL * full) * deep
    if spark.any():
        out += spark[..., None] * SPARK_RGB
        haze += 0.6 * _blur(spark, 0.12 * s)
    glow += haze[..., None] * HAZE_RGB
    y = fx._lum(glow)
    held = np.tanh(y / cap)
    glow *= (cap * held / np.maximum(y, 1e-6))[..., None]
    veil += hide * held
    return out + glow, veil


# ---- the light -----------------------------------------------------------------------------------------------------

def light_power(power: np.ndarray) -> np.ndarray:
    """(n,) the strength of the light the channels throw on the hand and the room: their own power (luminance summed)
    held at its peak over LIGHT_HOLD frames and let go by LIGHT_LETGO a frame, with LIGHT_FLICKER of the strokes' own
    flicker kept. So it comes up on the first stroke and stays while they strike, instead of strobing with them."""
    out, held = np.zeros_like(power), 0.0
    for t in range(len(power)):
        held = max(float(power[max(0, t - LIGHT_HOLD + 1): t + 1].max()), held * LIGHT_LETGO)
        out[t] = held * (1.0 - LIGHT_FLICKER * (1.0 - power[t] / held)) if held > 0 else 0.0
    return out


def _soft_cap(x: np.ndarray, cap: float) -> np.ndarray:
    return np.float32(cap) * (1.0 - np.exp(-x / np.float32(cap)))


# ---- the shot ------------------------------------------------------------------------------------------------------

def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline: Timeline, form: str, seed: int = 0,
           progress=None) -> np.ndarray:
    """The shot with lightning from the hand: uint8 RGB [T, H, W, 3] (cuesync.elements' contract). ``form``: "arc"
    (bolts from the fingertip the emitter was carried onto, or two fingers together, along the aim) or "ball" (a
    crackling ball over the palm where the emitter is). ``progress(stage, frac, msg)``, as cuesync.bending's, hears
    how far it is. Frames with no lightning and no light on them are the plate's own bytes."""
    from .bending import shoulder

    if form not in FORMS["lightning"]:
        raise ValueError(f"unknown lightning form {form!r}; one of {FORMS['lightning']}")
    n, h, w = plate.shape[:3]
    if timeline.n != n:
        raise ValueError(f"the timeline has {timeline.n} frames, the plate {n}")
    if mask is not None and mask.shape[:3] != (n, h, w):
        raise ValueError(f"the mask is {mask.shape[:3]}, the plate {(n, h, w)}")
    say = progress or (lambda stage, frac, msg: None)
    say("lightning", 0.0, "Striking the lightning")
    hand = _hand(plate, track)
    if form == "arc":
        bolts, spots = arc_frames(hand, timeline, seed)
        balls = [None] * n
    else:
        bolts, spots, balls = ball_frames(hand, timeline, seed)
    depth, hand_gain, room_gain = LIGHT[form]
    side = min(h, w)
    some = np.stack([lin(plate[i][::2, ::2]) for i in np.linspace(0, n - 1, 8).astype(int)])
    white, tint = fx.plate_white(some), fx.plate_tint(some)
    del some
    noise = fx.shadow_noise(plate)
    hf, wf = max(1, h // FIELD_Q), max(1, w // FIELD_Q)
    out = np.empty_like(plate)

    def made(t: int):
        """Frame t's lightning: (box, linear RGB layer over it, its power in plate px), or None."""
        ball = balls[t]
        box = _box(bolts[t], spots[t], hand.s[t], h, w)
        if ball is not None:
            x, y, R = ball[:3]
            bb = _box([], [(x, y, 0.5 * R, 1.0)], hand.s[t], h, w)
            box = bb if box is None else \
                (min(box[0], bb[0]), max(box[1], bb[1]), min(box[2], bb[2]), max(box[3], bb[3]))
        if box is None:
            return None
        E, veil = layer(bolts[t], spots[t], ball, float(hand.s[t]), box, side, GLOW_CAP[form], GLOW_VEIL[form], seed, t)
        return box, E, float(fx._lum(E).sum(dtype=np.float64)), veil

    layers: dict[int, tuple | None] = {}
    smalls: list[np.ndarray | None] = []  # each frame's channels, small, at unit power: the light's shape
    power = np.zeros(n)
    fields: dict[int, tuple] = {}
    last = None
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        got: list = [None] * (b - a)
        fx._each(b - a, lambda i: got.__setitem__(i, made(a + i)))
        for t in range(a, b):
            layers[t] = got[t - a]
            small = None
            if layers[t] is not None and layers[t][2] > 0:
                (y0, y1, x0, x1), E, power[t], _ = layers[t]
                small = np.zeros((hf, wf), np.float32)
                sy0, sx0 = y0 // FIELD_Q, x0 // FIELD_Q
                sy1, sx1 = max(sy0 + 1, min(hf, y1 // FIELD_Q)), max(sx0 + 1, min(wf, x1 // FIELD_Q))
                small[sy0:sy1, sx0:sx1] = cv2.resize(fx._lum(E), (sx1 - sx0, sy1 - sy0), interpolation=cv2.INTER_AREA)
                small /= max(float(small.sum(dtype=np.float64)), 1e-12)
                last = small
            smalls.append(small)
        strength = light_power(power[:b])  # (causal: frames already made don't change)
        for t in range(a, b):
            pool = [(wt, smalls[t - k]) for k, wt in enumerate(LIGHT_POOL) if t - k >= 0 and smalls[t - k] is not None]
            if strength[t] <= 1e-9:
                fields[t] = None
                continue
            src = sum(wt * sm for wt, sm in pool) / sum(wt for wt, _ in pool) if pool else last
            if src is None:
                fields[t] = None
                continue
            y = src * np.float32(strength[t] / (FIELD_Q * FIELD_Q))  # (per small px: the mean of the px it covers)
            s_f = hand.s[t] / FIELD_Q
            near, far = fx.spread(y, [max(depth * s_f, 1.0), fx.FAR * s_f])
            fields[t] = (near * np.float32(hand_gain), far * np.float32(room_gain), y)

        def finish(i: int) -> None:
            t = a + i
            if layers[t] is None and fields[t] is None:
                out[t] = plate[t]
                return
            P = lin(plate[t])
            if fields[t] is not None:
                near, far, y = fields[t]
                light = cv2.resize(far, (w, h), interpolation=cv2.INTER_LINEAR)
                m = None if mask is None else mask[t]
                if m is not None and not m[::4, ::4].any():
                    m = None
                if m is None and mask is None:
                    m = fx.arm_matte(tuple(hand.xy[t]), tuple(hand.u[t]), float(hand.s[t]), w, h)
                elif m is not None:
                    m = m.astype(np.float32) * np.float32(1 / 255)
                    m = np.minimum(m, cv2.GaussianBlur(m, (0, 0), max(0.5, fx.MATTE_SOFT * side)))
                if m is not None:
                    on_hand = cv2.resize(near, (w, h), interpolation=cv2.INTER_LINEAR) * \
                        fx.hand_shading(m, y, float(hand.s[t]))
                    light += m * _soft_cap(on_hand, HAND_CAP)
                P += fx.albedo(P, white, tint) * (light[..., None] * LIGHT_RGB)
            if layers[t] is not None:
                (y0, y1, x0, x1), E, _, veil = layers[t]
                P[y0:y1, x0:x1] *= 1.0 - veil[..., None]
                E = fx.grain(E[None], noise, seed=seed, frame0=t)[0]
                P[y0:y1, x0:x1] += E
            shoulder(P)  # (the channels' cores clip to white softly)
            out[t] = gam(P)

        fx._each(b - a, finish)
        for t in range(a, b):
            layers.pop(t, None)
            fields.pop(t, None)
        say("lightning", b / n, f"Striking the lightning: frame {b} of {n}")
    return out
