"""Water bent from an open palm, made rather than rendered (cuesync.elements): what is behind it, seen through it.

Water throws no light, so nothing here is added the way fire is: it is a lens and a mirror over the plate. A frame's
water is a set of Gaussian metaballs (a lone one is an exact sphere of radius R, two near ones bridge the way water
does), read as a half-thickness map. Each pixel's camera ray is traced through it: bent in at the front surface, across,
out at the back (Snell both ways, with a little dispersion), then on to what is behind: the hand just behind it, at
the depth its landmarks give it, or the room far behind, where the plate is read, averaged over as much of it as the
water squeezes into the pixel. Over that: the room reflected (Fresnel), the key light's glints, aeration where the water
is thin or fast, the plate's own grain. Under it, on the hand: the light the water focuses, inside the shadow it casts.

What makes it read as water and not glass is that it never holds still: its outline wobbles in its low modes (driven
by the hand's motion and the voice), small ripples travel over it, the water circulates inside it (seen as the room
through it swirling, and bubbles carried round), it drips from its underside, and it froths where it is thin and fast.

Two forms (cuesync.elements.FORMS["water"]):

- ``orb``: water gathers out of the air into a ball floating over the open palm, sized to the hand, wobbling and
  circulating, carried on the hand with the lag a mass of water has. It is held while the voice holds, and each hit in
  it jolts it. When the voice ends with a hit, or the hand thrusts, it is thrown along the aim and bursts into a splash
  and drops; a voice that just stops lets it fall onto the palm and splash off it.
- ``whip``: a stream of water out of the palm that swirls round the hand and trails it like a ribbon when it moves
  (each part of it held to the swirl by a spring, stiff at the root and soft at the tail). It pours out with the
  voice, each hit lashes it out along the aim, shedding drops, and when the voice ends it breaks into beads that fall.

The orb floats over the palm, not in it: it sits along the palm's normal, which is taken to face the camera (the
tracker's handedness, and with it the normal's sign, is unreliable for an upturned palm), and the hand hides water only
where the hand is in front of it, by depth (the palm's plane through its landmarks, the fingers off it), not wherever
its matte is.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from . import fx_light as fx
from .bending import emitter_at
from .elements import FORMS, Timeline
from .transfer import gam, lin

FPS = 24
_ORB, _WHIP, _NOISE, _GRAIN = 1, 2, 3, 4  # mixed into each seed, so one seed gives each its own randomness

# ---- the optics -----------------------------------------------------------------------------------------------------
ETA = (1.331, 1.336, 1.342)  # water's index for R, G, B: the room through it fringes faintly (a little exaggerated: at
# the true spread the fringe is under a pixel)
F0 = 0.02  # water's reflectance head on (Fresnel, Schlick)
TAU = 0.5  # the water's surface is where the blobs' summed field is this
SIG_K = 1.0 / math.sqrt(2.0 * math.log(1.0 / TAU))  # a blob's sigma per unit radius: a lone blob is a sphere of radius R
REACH = 2.4  # sigmas: a blob's field is laid this far out, tapering to nothing over its last fifth (laid to 3.2 sigmas
# the orb alone cost 60 ms a frame; cut off without the taper, a neighbour's surface stepped where it crossed the edge)
ABSORB = np.array([0.30, 0.10, 0.07], np.float32)  # per hand size of path: water is faintly cyan where it is thick
LENS = 0.5  # how far a ray turns per unit of the surface's slope, through both faces (a ball lens: 2 (n - 1) / n);
# it says how much of the plate one pixel of water sees
ROOM_DEPTH = 0.55  # shares of the frame's long side: the room behind the water (a ray turned by an angle sees about
# what the camera sees in that direction)
NEAR = (0.08, 1.5)  # hand sizes: the hand behind the water is taken at least / at most this far behind its back...
NEAR_OPEN = 0.6  # ... and this far where no hand is right behind it (a ray turned onto the hand from beside it)
DEFOCUS = 0.5  # a magnified patch of the plate is blurred by this share of its magnification: a ball lens is never sharp
# on what is right behind it (read point by point it showed the video's blocks, magnified)
BASE_BLUR = 0.04  # of the water's local radius: the least the plate is blurred through it
LOD_MAX = 4  # pyramid levels the plate is read through (16 times smaller: the widest squeeze the water makes)
RIM_DARK = 0.3  # grazing rays see the room far round the side, through little water: the rim is this much darker
# (at the prototype's 0.55 the outline read as a glass marble's)
TIR_DIM = 0.5  # where a ray is turned back inside the water (total internal reflection), what it shows is this dim
TIR_SOFT = 0.08  # Snell's discriminant over which a ray goes from leaving the water to being turned back in it
# (switched at once, the line between the two was a jagged stair inside the rim)
GLINT = (2.6, 0.07)  # the key light's mirror image on the surface: a sharp peak and a soft sheen...
GLINT_POW = (500.0, 25.0)  # ... and how tight each is
CRESCENT = 0.45  # the key light gathered by the water onto its far inside edge, opposite the light
INNER_GLOW = 0.06  # light bent and gathered inside the body: a faint moving net of brighter lines where it is thick
KEY_ELEV = 0.9  # the key light's direction out of the picture, towards the camera, per unit across it
KEY_UP = 0.35  # the key light is taken this much from above (a room's lights are mostly over the camera) and the rest
# from where the plate is brightest (read alone, a window behind the hand lit the orb from below)
KEY_SHARE = 0.015  # the brightest share of the room read as its key light...
KEY_SMOOTH = 6.0  # ... held steady over this many frames (sigma): read on each frame alone, a glint jumped when a lamp
# came into view on a walk
ENV_Q = 16  # the room the water reflects is the plate, this many times smaller and blurred
DEPTH_SOFT = 0.08  # hand sizes: where the water passes behind the hand, it goes from seen to hidden over this depth
EDGE_SOFT = 0.5  # px: the camera's own softness on the water's outline (a cut-out edge was one pixel sharp)
FOAM_ALBEDO = 0.75  # froth is white matter lit by the room: this share of its key light comes back
FOAM_MAX = 0.8  # froth hides at most this much of what is seen through the water

# ---- motion and imperfection ----------------------------------------------------------------------------------------
WAVE = (0.025, 0.04)  # log-field units: the surface's small waves at rest, and more at the voice's loudest (about 2 %
# and 5 % of the radius at the outline); without them the outline was a compass circle and the orb read as glass, at
# twice this what was seen through it folded like crumpled cellophane
WAVE_SIZE = 0.2  # hand sizes: the size of those waves
SPARKLE = 0.22  # slope: finer, faster waves that break the key light's one glint into many, as water does
SPARKLE_SIZE = 0.035  # hand sizes
SPARKLE_BENDS = 0.15  # the share of that slope that bends what is seen through the water (all of it crumpled the room
# seen through the orb like cellophane: its glints break up, the lens stays whole)
INNER = 0.015  # slope: the water circulating inside turns what is seen through it this much, where it is thick
# (the room seen through it, a frame's height behind, moves 15 px for it: at 0.035 it swirled into mush)
INNER_SIZE = 0.15  # hand sizes
FOAM_SIZE = 0.022  # hand sizes: the bubbles froth is made of
FLOW_CYCLE = 12  # frames: the water's patterns drift with its flow this long, then cross-fade into fresh ones (a pattern
# carried round for good winds itself into streaks)
SHUTTER = 0.24  # frames: the exposure (1/100 s), over which a moving drop smears into a streak
G = 60.0  # hand sizes a second squared: gravity (a hand is about 9 cm: true gravity is 110; at that a splash left the
# frame before it read)
DRAG = 0.985  # per frame: what a flying drop keeps of its speed
ZCAM = 4.5  # hand sizes: how far the camera is from the hand (a first-person shot); a thrown orb shrinks as it goes
NOISE_SIZE, NOISE_CELL = 256, 4.0  # px: the tiling noise textures, and their features' sigma

# ---- the orb --------------------------------------------------------------------------------------------------------
ORB_R = 0.36  # hand sizes: its radius (a ball 6 cm across over a palm 9 cm long)
ORB_GAP = 0.25  # hand sizes between its underside and the palm: it floats (at 0.1 its shadow and its focus sat under
# its own rim, and it read as set into the palm)
ORB_SPRING = (2.4, 0.45)  # Hz, damping: it follows the palm as a mass of water held on a spring would, trailing and
# overshooting
ORB_BOB = (0.025, 0.8)  # hand sizes, Hz: its slow float up and down
GATHER, GATHER_MIN, GATHER_LEAD = 12, 4, 2  # frames: gathering takes this long, at least this long squeezed in front of
# an early throw, and is done this long before a throw it was gathered for
GATHER_DROPS = 22  # drops that come together out of the air into it
GATHER_FROM = (1.6, 3.0)  # radii: how far out they start
MODE_HZ = {2: 1.6, 3: 3.1, 4: 4.8}  # its wobbles: two, three and four lobes round it (a drop's ratios, 1 : 1.9 : 3)
MODE_AMP = {3: 0.03, 4: 0.022}  # of the radius, at rest the third of it and at the voice's loudest all of it
MODE_REST = 0.35
MODE2_DRIVE, MODE2_DAMP, MODE2_MAX = 0.006, 0.25, 0.35  # the two-lobe stretch, driven by its acceleration (inertia
# squashes it), its damping, and the most it stretches
CIRCULATE = (0.9, 1.8)  # rad a second: it turns over inside this fast at rest, and more at the voice's loudest
BUBBLES = 11  # air caught inside, carried round by the circulation
BUBBLE_R = (0.010, 0.032)  # hand sizes
DRIP_EVERY = (26, 60)  # frames between drips off its underside
DRIP_FORM, DRIP_R = 7, 0.17  # frames a drip takes to swell and let go, and its radius in the orb's
JOLT = (0.18, 4)  # a hit while it is held: how hard it stretches it, and how many drops it spits per unit of the hit
RING = (0.04, 0.35, 0.22, 6.0)  # the ripple a hit sends over it: log-field height, wavelength and speed (radii, a
# frame) and how long it lasts (frames)
THROW_SPEED = 18.0  # hand sizes a second, at the loudest hit (0.6 of it at the quietest): at 11 gravity took back
# most of its rise and it burst a hand over the fingers
TAKEOFF = 2  # frames it takes to get up to speed (it jumped off the palm in one)
RECEDE = 0.5  # of the throw's speed: thrown along the aim it also goes away from the camera, into the room
BURST_AFTER = 5  # frames in the air before it bursts
THRUST_SIZE = 0.8  # how hard a thrust with no hit throws it
DROP_SIZE = 0.35  # how hard it splashes when let fall onto the palm
LINGER = 6  # frames it is held on after the voice stops before it falls
SPLASH = (170, 45, 30)  # drops, sheet pieces and spray a burst at full strength makes
SPLASH_SPEED = (3.0, 12.0)  # hand sizes a second: how fast its drops leave (slower, the burst was a puff a hand across)
SHEET_LIFE, SPRAY_LIFE = 5, 7  # frames the torn sheet and the fine spray last
BURST_CORE = 3  # frames the ball's body lasts, shrinking, as it bursts, its skin torn off its outer part
SHEET_FROM = (0.7, 1.0)  # radii: where in the ball the torn skin starts out (from all through it, the small pieces
# overlapping its middle made it a thin sheet, and the burst's first frame was a flat splat stuck on the room)

# ---- the whip -------------------------------------------------------------------------------------------------------
WHIP_NODES = 40  # the stream's nodes, root to tail
WHIP_ORBIT = 1.1  # hand sizes: the stream swirls round the hand this far out...
WHIP_SWEEP = (0.55, 0.8)  # ... wrapping this much of a turn round it, quiet and at the voice's loudest
WHIP_RISE = 0.3  # of its length: it rises out of the palm over this much of it before it swings out to that circle
WHIP_FACE = 0.7  # its circle faces the camera this much and the palm the rest (round the palm's own normal, an
# upturned palm's circle was seen edge on, a slug on the palm)
WHIP_TILT = 35.0  # degrees its circle is tipped about the hand's length, so one side passes behind the hand
WHIP_SPIN = (0.45, 0.5)  # turns a second it swirls at, quiet, and more at the voice's loudest
WHIP_WOBBLE = (0.14, 3.0, 1.3)  # its swirl's lobes: how far in and out (of its circle), how many, turns a second
WHIP_HZ = (3.0, 0.9)  # how stiffly it holds its swirl at the root and at the tail: the tail trails a moving hand,
# as a ribbon does (one stiffness throughout and the stream moved with the hand as one piece, a hoop)
WHIP_DAMP = 0.55  # their damping ratio
WHIP_R = (0.1, 0.028)  # hand sizes: the stream's radius at its root and at its tail
WHIP_BULB = 0.15  # hand sizes: the water over the palm it pours out of
WHIP_LIFT = 0.2  # hand sizes it starts over the palm
WHIP_CRACK = (8, 3.6, 0.45, 2.5)  # a hit cracks it: frames, how far out along the aim (hand sizes), the kink that runs
# down it (hand sizes across), and how much stiffer it is while it lashes out
WHIP_SHED = 9.0  # hand sizes a second: a tail faster than this sheds drops
WHIP_SAG = 0.12  # of gravity, while it is held up
WHIP_WAVES = (0.22, 5.0, 1.6)  # bulges running down the stream: their depth, how many to its length, how many a second
WHIP_FLOW = 2.0  # hand sizes a second: the water flows out along it (its inside pattern drifts that way)
WHIP_GROW, WHIP_BREAK = 10, 5  # frames it takes to pour out, and to neck into beads when the voice stops
WHIP_LEAD = 6  # frames it is out before a hit that opens the voice

# ---- the voice ------------------------------------------------------------------------------------------------------
VOICE_ON = 0.1  # the voice sounds where its energy is over this (a breath, the room's hum is not the voice)
VOICE_GAP = 6  # frames: silences this short inside the voice are bridged ("ka-BOOM" is one sound)
VOICE_SMOOTH = 2.0  # frames: how the energy is smoothed before it drives the water (it shivered with every syllable)

PALM_LM = (0, 1, 5, 9, 13, 17)  # the landmarks the palm's plane goes through: the wrist, the thumb's root, the knuckles
TRACK_SMOOTH = 1.5  # frames: the hand's size and the palm's normal are smoothed over this (MediaPipe's jitter made a
# held orb breathe)
NO_LANDMARKS_TILT = 0.6  # with no landmarks the palm is taken to face this much along the emitter's direction, the
# rest towards the camera
FINGER_NEAR, FINGER_FAR = 0.15, 0.5  # hand sizes: a landmark's own depth holds this close to it and fades into the
# palm's plane by this far


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, *salt])


def _ss(a: float, b: float, x):
    t = np.clip((np.asarray(x, np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _tsmooth(x, sigma: float) -> np.ndarray:
    """``x`` (n, ...) smoothed along its first axis by a Gaussian of ``sigma`` frames, its ends held."""
    x = np.asarray(x, np.float64)
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    return sum(k[i] * pad[i:i + len(x)] for i in range(2 * r + 1))


def _remap(img: np.ndarray, x: np.ndarray, y: np.ndarray, border: int = cv2.BORDER_REFLECT) -> np.ndarray:
    """``img`` read (bilinear) at float px ``x``, ``y`` of any shape: cv2.remap's maps, laid out in rows of 1024 (it
    takes at most 32767 a side)."""
    shape = np.shape(x)
    x = np.asarray(x, np.float32).reshape(-1)
    y = np.asarray(y, np.float32).reshape(-1)
    n = x.size
    if n == 0:
        return np.zeros(shape + img.shape[2:], np.float32)
    cols = 1024
    rows = -(-n // cols)
    pad = rows * cols - n
    if pad:
        x, y = np.concatenate([x, np.zeros(pad, np.float32)]), np.concatenate([y, np.zeros(pad, np.float32)])
    out = cv2.remap(img, x.reshape(rows, cols), y.reshape(rows, cols), cv2.INTER_LINEAR, borderMode=border)
    return out.reshape((rows * cols,) + img.shape[2:])[:n].reshape(shape + img.shape[2:])


# ---- the hand -------------------------------------------------------------------------------------------------------

@dataclass
class Hand:
    """The open palm frame by frame, in the picture's px, with depth z in px towards the camera."""

    p: np.ndarray  # (n, 2): where the water is held (the tracked emitter: the performer's tap, carried with the hand)
    nrm: np.ndarray  # (n, 3): unit normal out of the palm, on the side facing the camera
    along: np.ndarray  # (n, 3): unit, in the palm's plane, from the wrist towards the fingers
    s: np.ndarray  # (n,): hand size px
    plane: np.ndarray  # (n, 3): the palm's depth, z = a x + b y + c
    off: np.ndarray | None  # (n, 21, 3): each landmark's x, y and depth off that plane (None: no landmarks)

    def depth(self, k: int, x: np.ndarray, y: np.ndarray, fingers: bool = True) -> np.ndarray:
        """The hand's depth at px (x, y) on frame ``k``: its palm's plane, pulled to each landmark's own depth near it
        (a finger curled up out of the plane stands in front of what the plane is behind); with ``fingers`` False,
        the plane alone (where light lands on it: through the landmarks' bumps a focus broke into a string of them)."""
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        a, b, c = self.plane[k]
        z = np.float32(a) * x + np.float32(b) * y + np.float32(c)
        if self.off is None or not fingers:
            return z
        o = self.off[k].astype(np.float32)
        s = float(self.s[k])
        d2 = (x[..., None] - o[:, 0]) ** 2 + (y[..., None] - o[:, 1]) ** 2
        w = 1.0 / (d2 + np.float32((FINGER_NEAR * s) ** 2))
        return z + (w * o[:, 2]).sum(-1) / (w.sum(-1) + np.float32(1.0 / (FINGER_FAR * s) ** 2))


def hand_of(track: dict, n: int) -> Hand:
    """The palm from the track: its point (the emitter), size (smoothed), normal and depth (from its landmarks; without
    them, a palm facing the camera tipped towards the emitter's direction, at depth 0)."""
    em = [emitter_at(track, t) for t in range(n)]
    p = np.array([e[0] for e in em])
    d = np.array([e[1] for e in em])
    s = _tsmooth(np.array([e[2] for e in em]), TRACK_SMOOTH)
    lms = track.get("landmarks") or []
    seen = [t for t in range(min(n, len(lms))) if lms[t] is not None]
    if not seen:
        nrm = _unit(np.concatenate([NO_LANDMARKS_TILT * d, np.full((n, 1), math.sqrt(1 - NO_LANDMARKS_TILT ** 2))], 1))
        d3 = np.concatenate([d, np.zeros((n, 1))], 1)
        along = _unit(d3 - (d3 * nrm).sum(1, keepdims=True) * nrm)
        return Hand(p, nrm, along, s, np.zeros((n, 3)), None)
    L, last = [], lms[seen[0]]
    for t in range(n):
        last = lms[t] if t < len(lms) and lms[t] is not None else last
        L.append(np.asarray(last, np.float64)[:, :3])
    L = np.stack(L) * np.array([1.0, 1.0, -1.0])  # MediaPipe's z grows away from the camera
    raw = _unit(np.cross(L[:, 5] - L[:, 0], L[:, 17] - L[:, 0]))
    # one side of the palm all through, each frame agreeing with the one before, from the frame it most clearly faces
    # the camera on (that side): which hand MediaPipe says it is, and so which side its cross product is, flips for an
    # upturned palm
    ref = int(np.argmax(np.abs(raw[:, 2])))
    sign = np.ones(n)
    sign[ref] = 1.0 if raw[ref, 2] >= 0 else -1.0
    for t in list(range(ref + 1, n)) + list(range(ref - 1, -1, -1)):
        u = t - 1 if t > ref else t + 1
        sign[t] = sign[u] if raw[t] @ raw[u] >= 0 else -sign[u]
    nrm = _unit(_tsmooth(raw * sign[:, None], TRACK_SMOOTH))
    a3 = L[:, 9] - L[:, 0]
    along = _unit(a3 - (a3 * nrm).sum(1, keepdims=True) * nrm)
    plane = np.zeros((n, 3))
    off = L.copy()
    for t in range(n):
        A = np.c_[L[t, PALM_LM, :2], np.ones(len(PALM_LM))]
        plane[t] = np.linalg.lstsq(A, L[t, PALM_LM, 2], rcond=None)[0]
        off[t, :, 2] -= np.c_[L[t, :, :2], np.ones(21)] @ plane[t]
    return Hand(p, nrm, along, s, plane, off)


def _aim(tl: Timeline, hand: Hand, k: int) -> np.ndarray:
    """The unit direction in the picture a throw or a crack goes: the studio's arrow, else out of the palm as it shows
    in the picture, else (a palm facing the camera) the way the fingers point."""
    if tl.aim is not None:
        a = math.radians(tl.aim)
        return np.array([math.sin(a), -math.cos(a)])
    out = hand.nrm[k, :2]
    if np.hypot(*out) > 0.35:
        return out / np.hypot(*out)
    return _unit(hand.along[k, :2] + 1e-9)


# ---- the voice ------------------------------------------------------------------------------------------------------

def sounding(tl: Timeline) -> np.ndarray:
    """(n,) bool: the frames the voice sounds on (or holds a sound), short silences inside it bridged."""
    on = (tl.energy > VOICE_ON) | np.array([tl.held_at(t) for t in range(tl.n)], bool)
    idx = np.flatnonzero(on)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= VOICE_GAP + 1:
            on[a:b] = True
    return on


def _thrusts(track: dict) -> list[int]:
    return sorted(int(e["frame"]) for e in track.get("events") or [] if e.get("kind") == "extension")


@dataclass
class OrbPlan:
    """When the orb does what, in frames."""

    gather: int  # it starts to come together out of the air...
    formed: int  # ... and is whole
    release: int | None  # thrown or let fall (None: held to the end)
    throw: bool  # thrown along the aim (a hit, a thrust), else let fall onto the palm (the voice just stopped)
    size: float  # how hard: the throwing hit's size
    jolts: list[tuple[int, float]]  # the hits while it is held, with their sizes


def orb_plan(tl: Timeline, track: dict) -> OrbPlan:
    """The orb gathers when the voice first sounds and is thrown on its last hit, or on a thrust before that; a voice
    that opens with the very hit that throws it ("BOOM") has it gathered just before. With no hit nor thrust it is let
    fall LINGER frames after the voice stops; with no voice at all it gathers on the tap and is held."""
    heard = np.flatnonzero(sounding(tl))
    hits = sorted(zip(tl.hits, tl.sizes))
    start = int(heard[0]) if heard.size else int(tl.tap_frame)
    if hits:
        start = min(start, int(hits[0][0]))
    throw_at = next((f for f in _thrusts(track) if f >= start + GATHER_MIN), None)
    size = THRUST_SIZE
    if hits and (throw_at is None or hits[-1][0] < throw_at):
        throw_at, size = int(hits[-1][0]), float(hits[-1][1])
    if throw_at is not None:
        g0 = max(0, min(start, throw_at - GATHER - GATHER_LEAD))
        g1 = max(g0 + 1, min(g0 + GATHER, throw_at - GATHER_LEAD))
        release, throw = throw_at, True
    else:
        g0, g1, throw, size = start, start + GATHER, False, DROP_SIZE
        release = max(int(heard[-1]) + 1 + LINGER, g1 + 1) if heard.size else None
    jolts = [(int(f), float(z)) for f, z in hits if g1 <= f and (release is None or f < release)]
    return OrbPlan(g0, g1, release, throw, size, jolts)


@dataclass
class WhipPlan:
    start: int  # it pours out of the palm...
    end: int | None  # ... and lets go here (None: held to the end)
    cracks: list[tuple[int, float]]  # the hits and thrusts that crack it, with how hard


def whip_plan(tl: Timeline, track: dict) -> WhipPlan:
    """The stream pours out when the voice first sounds (WHIP_LEAD frames before a hit that opens it, so the hit has
    a stream to crack), cracks on every hit and thrust, and lets go LINGER frames after the voice stops."""
    heard = np.flatnonzero(sounding(tl))
    hits = sorted(zip(tl.hits, tl.sizes))
    start = int(heard[0]) if heard.size else int(tl.tap_frame)
    if hits and hits[0][0] <= start + 1:
        start = max(0, int(hits[0][0]) - WHIP_LEAD)
    end = int(heard[-1]) + 1 + LINGER if heard.size else None
    cracks = [(int(f), float(z)) for f, z in hits] + [(f, THRUST_SIZE) for f in _thrusts(track)]
    cracks = sorted((f, z) for f, z in cracks if f >= start and (end is None or f < end))
    return WhipPlan(start, end, cracks)


# ---- what a frame's water is made of --------------------------------------------------------------------------------

class Blob(NamedTuple):
    x: float
    y: float
    z: float  # depth px, towards the camera
    r: float  # radius px
    shape: tuple = (1.0, 0.0, 0.0, 1.0)  # 2x2 row by row: takes a circle to the blob's outline (stretched, smeared)
    vx: float = 0.0  # px a frame: how the water moves here (what is seen inside it drifts with it)
    vy: float = 0.0
    spin: float = 0.0  # rad a frame: the water turning round the blob's centre
    foam: float = 0.0  # 0..1: how aerated it is
    modes: tuple = ()  # ((lobes, share of the radius, phase), ...): its outline's wobble


@dataclass
class Scene:
    """One frame's water."""

    blobs: list = field(default_factory=list)
    bubbles: list = field(default_factory=list)  # (x, y, r, seen): air in the orb, and how clearly it is seen (0..1)
    rings: list = field(default_factory=list)  # ripples running over it: (x, y, age frames, height, wavelength, speed)
    wave: float = WAVE[0]  # how high its small waves are
    held: tuple | None = None  # (x, y, z, R): the orb over the palm, which focuses light onto the hand and shades it


def _shape(u: np.ndarray, k: float, across: float = 1.0) -> tuple:
    """The shape matrix that stretches a circle k times along unit ``u`` and ``across`` times across it."""
    ux, uy = float(u[0]), float(u[1])
    return (k * ux * ux + across * uy * uy, (k - across) * ux * uy, (k - across) * ux * uy, k * uy * uy + across * ux * ux)


def _smeared(r: float, v: np.ndarray, k: float = 1.0) -> tuple:
    """A blob of radius ``r`` moving ``v`` px a frame: smeared along it over the shutter, ``k`` times longer besides."""
    sp = float(np.hypot(*v))
    if sp < 1e-6:
        return _shape(np.array([1.0, 0.0]), k, 1.0 / math.sqrt(k))
    smear = 1.0 + min(6.0, sp * SHUTTER / max(2.0 * r, 1.0))
    return _shape(v / sp, k * smear, 1.0 / math.sqrt(k))


class Field:
    """The summed blobs on a box of the frame: f = sum of w g, and the g-weighted local sigma^2, depth, flow and foam."""

    def __init__(self, blobs: list[Blob], h: int, w: int):
        ext = []
        for b in blobs:
            if b.r < 0.5:
                continue
            sig = b.r * SIG_K * (1.0 + sum(abs(a) for _, a, _ in b.modes))
            S = np.asarray(b.shape, np.float64).reshape(2, 2)
            C = S @ S.T
            ex, ey = REACH * sig * math.sqrt(C[0, 0]), REACH * sig * math.sqrt(C[1, 1])
            ext.append((b, (b.y - ey, b.y + ey + 1, b.x - ex, b.x + ex + 1)))
        boxes = [(max(0, int(e[0])), min(h, int(e[1]) + 1), max(0, int(e[2])), min(w, int(e[3]) + 1)) for _, e in ext]
        boxes = [bx for bx in boxes if bx[1] > bx[0] and bx[3] > bx[2]]
        if not boxes:
            self.box = None
            return
        self.box = (min(b[0] for b in boxes), max(b[1] for b in boxes), min(b[2] for b in boxes),
                    max(b[3] for b in boxes))
        y0, y1, x0, x1 = self.box
        self.f, self.s2, self.zc, self.vx, self.vy, self.fo = (np.zeros((y1 - y0, x1 - x0), np.float32)
                                                               for _ in range(6))
        for b, e in ext:
            self._splat(b, e)

    def _splat(self, b: Blob, e: tuple) -> None:
        y0, y1, x0, x1 = self.box
        ya, yb = max(y0, int(e[0])), min(y1, int(e[1]) + 1)
        xa, xb = max(x0, int(e[2])), min(x1, int(e[3]) + 1)
        if yb <= ya or xb <= xa:
            return
        sig = b.r * SIG_K
        dx = (np.arange(xa, xb, dtype=np.float32) - np.float32(b.x))[None, :]
        dy = (np.arange(ya, yb, dtype=np.float32) - np.float32(b.y))[:, None]
        Si = np.linalg.inv(np.asarray(b.shape, np.float64).reshape(2, 2)).astype(np.float32)
        u, v = Si[0, 0] * dx + Si[0, 1] * dy, Si[1, 0] * dx + Si[1, 1] * dy  # in the blob's own frame: a circle
        q = (u * u + v * v) * np.float32(1.0 / (sig * sig))
        if b.modes:
            th = np.arctan2(v, u)
            m = 1.0 + sum(np.float32(a) * np.cos(np.float32(l) * th - np.float32(ph)) for l, a, ph in b.modes)
            q = q / (m * m)
        r2 = REACH * REACH
        t = np.clip((r2 - q) * np.float32(1.0 / (r2 - 0.64 * r2)), 0.0, 1.0)
        g = np.exp(-0.5 * q) * (t * t * (3.0 - 2.0 * t))
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        self.f[sl] += g
        self.s2[sl] += g * np.float32(sig * sig)
        self.zc[sl] += g * np.float32(b.z)
        self.vx[sl] += g * (np.float32(b.vx) - np.float32(b.spin) * dy)
        self.vy[sl] += g * (np.float32(b.vy) + np.float32(b.spin) * dx)
        if b.foam:
            self.fo[sl] += g * np.float32(b.foam)


# ---- the room and the noise -----------------------------------------------------------------------------------------

class Room:
    """The room's light, frame by frame: what the water reflects (the plate, small and blurred) and its key light."""

    def __init__(self, plate: np.ndarray):
        n, H, W = plate.shape[:3]
        q = ENV_Q
        self.env = [cv2.GaussianBlur(lin(cv2.resize(f, (max(1, W // q), max(1, H // q)), interpolation=cv2.INTER_AREA)),
                                     (0, 0), 3.0) for f in plate]
        at, colour, white = [], [], []
        for e in self.env:
            y = fx._lum(e)
            ys, xs = np.nonzero(y >= np.quantile(y, 1.0 - KEY_SHARE))
            at.append(((xs.mean() + 0.5) * q, (ys.mean() + 0.5) * q))
            c = e[ys, xs].mean(0)
            colour.append(c / max(float(c.max()), 1e-6))
            white.append(float(fx._lum(e[ys, xs]).mean()))
        self.at = _tsmooth(at, KEY_SMOOTH)
        self.colour = _tsmooth(colour, KEY_SMOOTH).astype(np.float32)
        self.white = _tsmooth(white, KEY_SMOOTH)
        self.q = q

    def key(self, k: int, c) -> np.ndarray:
        """Unit vector from px ``c`` towards the key light on frame ``k`` (z towards the camera)."""
        d = self.at[k] - np.asarray(c, np.float64)
        d = (1.0 - KEY_UP) * d / max(float(np.hypot(*d)), 1e-6) + KEY_UP * np.array([0.0, -1.0])
        d = d / max(float(np.hypot(*d)), 1e-6)
        v = np.array([d[0], d[1], KEY_ELEV])
        return (v / np.linalg.norm(v)).astype(np.float32)

    def lit(self, k: int) -> np.ndarray:
        """The key light's colour and strength (linear)."""
        return self.colour[k] * np.float32(max(self.white[k], 0.3))


class Noise:
    """Tiling noise textures (zero mean, unit spread), read anywhere at fractional px, drifting with the water."""

    def __init__(self, seed: int, count: int = 6):
        rng = _rng(seed, _NOISE)
        N = NOISE_SIZE
        fy, fxq = np.fft.fftfreq(N)[:, None], np.fft.rfftfreq(N)[None, :]
        low = np.exp(-2.0 * math.pi ** 2 * NOISE_CELL ** 2 * (fy ** 2 + fxq ** 2))  # a Gaussian blur, periodic
        self.tex = []
        for _ in range(count):
            z = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((N, N))) * low, (N, N))
            self.tex.append(((z - z.mean()) / max(float(z.std()), 1e-9)).astype(np.float32))

    def flow(self, i: int, x, y, vx, vy, k: int, size: float) -> np.ndarray:
        """Texture ``i`` at px (x, y) on frame ``k``, its features ``size`` px, carried by the flow (vx, vy) px a frame:
        two copies a half cycle apart, each drifting with the flow for FLOW_CYCLE frames and then taken up afresh
        elsewhere in the texture while the other is at its full weight (a flow map)."""
        N = NOISE_SIZE
        per = 2.0 * NOISE_CELL / max(size, 1e-3)  # texture px per frame px
        acc, w2 = 0.0, 0.0
        for ph in (0.0, 0.5):
            u = k / FLOW_CYCLE + ph
            cyc = math.floor(u)
            age = (u - cyc) * FLOW_CYCLE
            wgt = 1.0 - abs(2.0 * (u - cyc) - 1.0)
            ox, oy = (cyc * 97.31 + ph * 41.7) % N, (cyc * 61.13 + ph * 13.9) % N
            tx = np.mod((x - vx * age) * per + ox, N)
            ty = np.mod((y - vy * age) * per + oy, N)
            acc = acc + wgt * _remap(self.tex[i], tx, ty, cv2.BORDER_WRAP)
            w2 += wgt * wgt
        return acc / math.sqrt(w2)


# ---- shading --------------------------------------------------------------------------------------------------------

def _pyramid(P: np.ndarray) -> list[list[np.ndarray]]:
    """The frame at LOD_MAX + 1 sizes, each split into its channels."""
    lv = [P]
    for _ in range(LOD_MAX):
        lv.append(cv2.pyrDown(lv[-1]))
    return [list(cv2.split(v)) for v in lv]


def _lod(pyr: list, c: int, x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Channel ``c`` of the frame at px (x, y), averaged over ``sigma`` px (between pyramid levels)."""
    lev = np.clip(np.log2(1.0 + np.maximum(sigma, 0.0)), 0.0, LOD_MAX)
    lo = np.minimum(lev.astype(np.int32), LOD_MAX - 1)
    fr = (lev - lo).astype(np.float32)
    out = np.empty(x.shape, np.float32)
    for L in range(LOD_MAX):
        sel = np.flatnonzero(lo == L)
        if not sel.size:
            continue
        xs, ys = x[sel], y[sel]
        a = _remap(pyr[L][c], (xs + 0.5) / 2 ** L - 0.5, (ys + 0.5) / 2 ** L - 0.5)
        b = _remap(pyr[L + 1][c], (xs + 0.5) / 2 ** (L + 1) - 0.5, (ys + 0.5) / 2 ** (L + 1) - 0.5)
        out[sel] = a + (b - a) * fr[sel]
    return out


def _footprint(d: np.ndarray, hxx, hxy, hyy) -> np.ndarray:
    """How many px of what is ``d`` px behind the water one pixel of it sees (sigma px): the map from the water to
    there is about 1 + d LENS times the surface's curvature. Where it squeezes the plate it is averaged; where it
    magnifies, defocused (DEFOCUS)."""
    j00, j01, j11 = 1.0 + d * LENS * hxx, d * LENS * hxy, 1.0 + d * LENS * hyy
    det = np.abs(j00 * j11 - j01 * j01)
    span = np.sqrt(np.maximum(j00 * j00 + j01 * j01, j01 * j01 + j11 * j11))
    mag = 1.0 / np.sqrt(np.maximum(det, 1e-6))
    return 0.5 * np.maximum(span - 1.0, 0.0) + DEFOCUS * np.maximum(mag - 1.0, 0.0)


@dataclass
class Layer:
    """The water over a box of the frame, to lay over the plate: premultiplied colour and coverage (linear)."""

    box: tuple  # (y0, y1, x0, x1)
    colour: np.ndarray  # (h, w, 3)
    alpha: np.ndarray  # (h, w)
    h: np.ndarray  # (h, w): its half thickness px...
    z: np.ndarray  # (h, w): ... and its middle's depth (for the light it focuses onto the hand behind it)


def _refract(ix, iy, iz, nx, ny, nz, eta):
    """Snell: unit incident ray i, unit normal n facing against it, ratio n1 / n2; (refracted ray, how far from
    being turned back it is: the discriminant, under 0 where it is)."""
    cosi = -(ix * nx + iy * ny + iz * nz)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    a = eta * cosi - np.sqrt(np.maximum(k, 0.0))
    return (eta * ix + a * nx, eta * iy + a * ny, eta * iz + a * nz), k


def shade(fld: Field, sc: Scene, k: int, hand: Hand, room: Room, noise: Noise, pyr: list, matte: np.ndarray | None,
          code: float, seed: int) -> Layer | None:
    """The water of ``fld`` as seen on frame ``k``: traced through to the plate (``pyr``, the frame's pyramid) and the
    hand (``matte`` uint8, or None), reflecting the room, glinting, froth and grain; hidden where the hand is in front
    of it. Pixels only where it covers some of the picture."""
    H, W = pyr[0][0].shape
    y0, y1, x0, x1 = fld.box
    s = float(hand.s[k])
    f = fld.f
    has = f > 1e-6
    fs = np.maximum(f, 1e-6)
    lf = np.where(has, np.log(fs / TAU), -10.0).astype(np.float32)
    sig = np.sqrt(fld.s2 / fs)
    vx, vy = fld.vx / fs, fld.vy / fs
    ys, xs = np.nonzero(lf > -0.7)  # as far out as the waves can move the surface
    if not ys.size:
        return None
    X, Y = (xs + x0).astype(np.float32), (ys + y0).astype(np.float32)
    bump = sc.wave * noise.flow(0, X, Y, vx[ys, xs], vy[ys, xs], k, WAVE_SIZE * s)
    for rx, ry, age, amp, lam, speed in sc.rings:  # a ripple from a hit, running out from where it struck
        d = np.hypot(X - rx, Y - ry) - speed * age
        bump += amp * np.exp(-age / RING[3]) * np.sin(2 * math.pi * d / lam) * np.exp(-(d / (1.5 * lam)) ** 2)
    lf[ys, xs] += bump.astype(np.float32)
    sd = np.where(has, sig * np.float32(SIG_K) * lf, -1e3)  # signed distance to the surface, px (+ inside), and far
    # outside where no blob reaches (taken as 0 there, every px of the box was half covered by water)
    cover = np.clip(0.5 + sd, 0.0, 1.0)
    act = cover > 0
    rows, cols = np.flatnonzero(act.any(1)), np.flatnonzero(act.any(0))
    if not rows.size:
        return None
    ra, rb = max(0, rows[0] - 3), min(act.shape[0], rows[-1] + 4)
    ca, cb = max(0, cols[0] - 3), min(act.shape[1], cols[-1] + 4)
    crop = (slice(ra, rb), slice(ca, cb))
    by0, bx0 = y0 + ra, x0 + ca
    h2 = np.sqrt(np.maximum(2.0 * sig[crop] ** 2 * lf[crop], 0.0)).astype(np.float32)
    hb = cv2.GaussianBlur(h2, (0, 0), 0.8)
    gy, gx = np.gradient(hb)
    gx, gy = np.clip(gx, -6, 6), np.clip(gy, -6, 6)
    hxx = cv2.GaussianBlur(np.gradient(gx, axis=1), (0, 0), 1.5)
    hyy = cv2.GaussianBlur(np.gradient(gy, axis=0), (0, 0), 1.5)
    hxy = cv2.GaussianBlur(np.gradient(gx, axis=0), (0, 0), 1.5)
    nd = np.sqrt(gx * gx + gy * gy + 1.0)
    dnx, dny, dnz = -gx / nd, -gy / nd, 1.0 / nd
    a_ = act[crop]
    iy, ix = np.nonzero(a_)
    X, Y = (ix + bx0).astype(np.float32), (iy + by0).astype(np.float32)
    h = h2[iy, ix]
    R = sig[crop][iy, ix] / np.float32(SIG_K)
    zc = (fld.zc / fs)[crop][iy, ix]
    fvx, fvy = vx[crop][iy, ix], vy[crop][iy, ix]
    foam = (fld.fo / fs)[crop][iy, ix]
    thick = np.clip(h / np.maximum(0.2 * R, 1.0), 0.0, 1.0)
    sx = SPARKLE * thick * noise.flow(3, X, Y, fvx, fvy, k, SPARKLE_SIZE * s)
    sy = SPARKLE * thick * noise.flow(4, X, Y, fvx, fvy, k, SPARKLE_SIZE * s)
    gxa, gya = gx[iy, ix] + sx, gy[iy, ix] + sy  # the surface the light glints off, its finest waves and all...
    na = np.sqrt(gxa * gxa + gya * gya + 1.0)
    nx, ny, nz = -gxa / na, -gya / na, 1.0 / na
    gxr, gyr = gx[iy, ix] + SPARKLE_BENDS * sx, gy[iy, ix] + SPARKLE_BENDS * sy  # ... and the one seen through
    nr = np.sqrt(gxr * gxr + gyr * gyr + 1.0)

    # in through the front, across, out through the back (whose normal is the front's mirrored), on to what is behind
    eta = ETA[1]
    (tx, ty, tz), _ = _refract(0.0, 0.0, -1.0, -gxr / nr, -gyr / nr, 1.0 / nr, 1.0 / eta)
    path = 2.0 * h / np.maximum(-tz, 0.25)
    ex, ey = X + tx * path, Y + ty * path
    bx = _remap(dnx, ex - bx0, ey - by0)
    byy = _remap(dny, ex - bx0, ey - by0)
    bz = _remap(dnz, ex - bx0, ey - by0)
    (ox, oy, oz), kk = _refract(tx, ty, tz, -bx, -byy, bz, eta)
    out = _ss(0.0, TIR_SOFT, kk).astype(np.float32)  # 1: out through the back; 0: turned back in, on as it was going
    ox, oy, oz = out * ox + (1 - out) * tx, out * oy + (1 - out) * ty, out * oz - (1 - out) * np.abs(tz)
    inner = noise.flow(1, X, Y, fvx, fvy, k, INNER_SIZE * s)
    deep = np.clip(h / np.maximum(R, 1.0), 0.0, 1.0)
    tanx = np.clip(ox / np.maximum(-oz, 0.3), -3, 3) + INNER * deep * inner
    tany = np.clip(oy / np.maximum(-oz, 0.3), -3, 3) + INNER * deep * noise.flow(2, X, Y, fvx, fvy, k, INNER_SIZE * s)

    vis = np.ones(X.shape, np.float32)
    dn = np.full(X.shape, NEAR_OPEN * s, np.float32)
    if matte is not None:
        mh = _remap(matte, X, Y).astype(np.float32) / 255.0
        on = np.flatnonzero(mh > 0.01)
        if on.size:
            zh = hand.depth(k, X[on], Y[on])
            zf, zb, m_ = zc[on] + h[on], zc[on] - h[on], mh[on]
            vis[on] = 1.0 - m_ * np.clip((zh - zf) / (DEPTH_SOFT * s) + 0.5, 0.0, 1.0)  # the hand in front hides it
            cut = np.clip((zf - zh) / np.maximum(2.0 * h[on], 1e-3), 0.0, 1.0)  # the hand through it: only the water
            path[on] *= 1.0 - m_ * (1.0 - cut)  # in front of it is crossed
            dn[on] = m_ * np.clip(zb - zh, NEAR[0] * s, NEAR[1] * s) + (1.0 - m_) * dn[on]  # (switched where the
            # matte crossed a half, the hand's outline behind the water showed through it as a jagged stair)
    dfar = ROOM_DEPTH * max(H, W)
    hxx_, hxy_, hyy_ = hxx[iy, ix], hxy[iy, ix], hyy[iy, ix]
    sig_n = _footprint(dn, hxx_, hxy_, hyy_) + BASE_BLUR * R
    sig_f = _footprint(dfar, hxx_, hxy_, hyy_)
    qx, qy = ex + tanx * dn, ey + tany * dn
    near = np.stack([_lod(pyr, c, qx, qy, sig_n) for c in range(3)], -1)
    far = np.stack([_lod(pyr, c, ex + tanx * dfar * (ETA[c] - 1) / (eta - 1), ey + tany * dfar * (ETA[c] - 1) / (eta - 1),
                         sig_f) for c in range(3)], -1)
    if matte is not None:  # which of the two a ray lands on, softened (a thin stream flicked between them pixel by pixel)
        qm = 4
        msel = cv2.GaussianBlur(cv2.resize(matte, (W // qm, H // qm), interpolation=cv2.INTER_AREA).astype(np.float32)
                                / 255.0, (0, 0), max(1.0, 0.04 * s / qm))
        m = _remap(msel, (qx + 0.5) / qm - 0.5, (qy + 0.5) / qm - 0.5)[:, None]
        col = m * near + (1.0 - m) * far
    else:
        col = far
    col *= np.exp(-ABSORB[None, :] * (path / max(s, 1.0))[:, None])
    rr = np.clip(1.0 - nz, 0.0, 1.0)
    col *= (1.0 - RIM_DARK * _ss(0.35, 0.95, rr) ** 1.5).astype(np.float32)[:, None]
    col *= (TIR_DIM + (1.0 - TIR_DIM) * out)[:, None]
    # the room reflected (behind the camera: taken as the room itself, mirrored and blurred)
    F = (F0 + (1.0 - F0) * (1.0 - nz) ** 5).astype(np.float32)
    env = room.env[k]
    rq = room.q
    ev = _remap(env, (X - 2.0 * nz * nx * 0.5 * W) / rq - 0.5, (Y - 2.0 * nz * ny * 0.5 * H) / rq - 0.5)
    col = col * (1.0 - F)[:, None] + ev * F[:, None]
    # the key light's glints and sheen, the light it gathers onto its far inside edge, and inside it a faint net
    Lk = room.key(k, ((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    hv = Lk + np.array([0.0, 0.0, 1.0], np.float32)
    hv /= np.linalg.norm(hv)
    ndh = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, 1.0)
    spec = GLINT[0] * ndh ** GLINT_POW[0] + GLINT[1] * ndh ** GLINT_POW[1]
    lxy = math.hypot(float(Lk[0]), float(Lk[1])) + 1e-6
    away = -(nx * Lk[0] + ny * Lk[1]) / lxy / np.maximum(np.hypot(nx, ny), 1e-3)
    cres = CRESCENT * _ss(0.25, 0.85, rr) * (1.0 - _ss(0.9, 1.0, rr)) * _ss(0.2, 0.95, away)
    net = INNER_GLOW * np.exp(-(inner / 0.22) ** 2) * deep * (1.0 - F)
    lit = room.lit(k)
    col += (spec + cres + net).astype(np.float32)[:, None] * lit[None, :]
    fm = np.clip(foam * (0.55 + 0.45 * noise.flow(5, X, Y, fvx, fvy, k, FOAM_SIZE * s)), 0.0, 1.0) * FOAM_MAX
    col = col * (1.0 - fm)[:, None] + (fm * (0.6 + 0.4 * nz))[:, None] * (FOAM_ALBEDO * lit)[None, :]
    if code > 0:  # the plate's grain: what is seen through the water was read smoothed, its glints are clean
        y = np.maximum(col @ fx._LUMA, 1e-6)
        step = (2.2 * code) * y ** np.float32(1.2 / 2.2) / y
        g = _rng(seed, _GRAIN, k).standard_normal(y.size).astype(np.float32)
        col = np.maximum(col * (1.0 + g * step)[:, None], 0.0)
    alpha = (a_ * 0.0).astype(np.float32)
    alpha[iy, ix] = cover[crop][iy, ix] * vis
    C = np.zeros(alpha.shape + (3,), np.float32)
    C[iy, ix] = col * alpha[iy, ix, None]
    C = cv2.GaussianBlur(C, (0, 0), EDGE_SOFT)
    alpha = cv2.GaussianBlur(alpha, (0, 0), EDGE_SOFT)
    return Layer((by0, by0 + C.shape[0], bx0, bx0 + C.shape[1]), C, alpha, h2, (fld.zc / fs)[crop])


CAUSTIC_GAIN, CAUSTIC_KEEP = 1.0, 0.4  # the focus's strength, and the key light's share of what lights the hand (the
# rest is the room's soft light, which the water does not focus)
CAUSTIC_SOFT = (0.08, 0.3)  # of the distance it falls: how soft the focus and the shadow are (a room's key light is
# wide: a shadow blurred as little as its focus showed the orb's outline drawn on the palm, a hard-edged disc)
CAUSTIC_ELEV = 1.8  # the light it focuses comes more from the camera's side than its glint does (from the glint's
# direction the focus landed a hand off, on the wall)
SHADE_AO = 0.14  # the palm right under it loses this much of the room's light


def caustic(P: np.ndarray, layer: Layer, k: int, hand: Hand, room: Room, matte: np.ndarray,
            held: tuple | None = None) -> tuple:
    """The water's light on the hand behind it, in place: each cell of water in front of the hand sends its share of
    the key light along the ray bent through it to the hand, at the hand's own depth there, and lights it there; the
    light the hand had there without the water is taken away (energy kept: a bright focus inside a soft shadow).
    ``held``, the orb over the palm (x, y, z, R): its bulk also takes some of the room's soft light off the palm right
    under it. Returns the box it changed."""
    s = float(hand.s[k])
    ly0, ly1, lx0, lx1 = layer.box
    L = room.key(k, ((lx0 + lx1) / 2.0, (ly0 + ly1) / 2.0)) * np.float32([1.0, 1.0, CAUSTIC_ELEV / KEY_ELEV])
    L /= np.linalg.norm(L)
    q = 2
    hw = (max(1, (lx1 - lx0) // q), max(1, (ly1 - ly0) // q))
    hq = cv2.resize(layer.h, hw, interpolation=cv2.INTER_AREA)
    zq = cv2.resize(layer.z, hw, interpolation=cv2.INTER_AREA)
    iy, ix = np.nonzero(hq > 0.6)
    if not iy.size:
        return (ly0, ly0, lx0, lx0)
    zh = hand.depth(k, (ix + 0.5) * q - 0.5 + lx0, (iy + 0.5) * q - 0.5 + ly0, fingers=False)
    front = np.clip((zq[iy, ix] - zh) / (DEPTH_SOFT * s) + 0.5, 0.0, 1.0)  # water behind the hand lights none of it
    d = np.clip(zq[iy, ix] - hq[iy, ix] - zh, NEAR[0] * s, NEAR[1] * s)  # px from the water's back to the hand
    ao_r = 0.0
    if held is not None:  # the orb's light lands as one: taken cell by cell on the tilted palm, its focus was drawn
        cx, cy, cz, R = held  # out into a faint streak
        dh = float(np.clip(cz - R - hand.depth(k, np.float32([cx]), np.float32([cy]), fingers=False)[0],
                           NEAR[0] * s, NEAR[1] * s))
        d = np.where(np.hypot((ix + 0.5) * q + lx0 - cx, (iy + 0.5) * q + ly0 - cy) < 1.2 * R, dh, d)
        ao_r = R + 0.5 * dh
    dmax = float(d.max())
    pad = int(dmax * (abs(L[0]) + abs(L[1])) / L[2] + 3.0 * CAUSTIC_SOFT[1] * dmax + 3.0 * ao_r + 4)
    H, W = P.shape[:2]
    y0, y1, x0, x1 = max(0, ly0 - pad), min(H, ly1 + pad), max(0, lx0 - pad), min(W, lx1 + pad)
    if y1 - y0 < 2 * q or x1 - x0 < 2 * q:
        return (y0, y0, x0, x0)
    gh, gw = (y1 - y0) // q, (x1 - x0) // q
    gy, gx = np.gradient(cv2.GaussianBlur(hq / q, (0, 0), 0.8))
    gx, gy = gx[iy, ix], gy[iy, ix]
    nd = np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx / nd, -gy / nd, 1.0 / nd
    (tx, ty, tz), _ = _refract(-L[0], -L[1], -L[2], nx, ny, nz, 1.0 / ETA[1])
    hc = hq[iy, ix] / q
    sp = 2.0 * hc / np.maximum(-tz, 0.25)
    (ox, oy, oz), kk = _refract(tx, ty, tz, -nx, -ny, nz, ETA[1])
    out = _ss(0.0, TIR_SOFT, kk)
    ox, oy, oz = out * ox + (1 - out) * tx, out * oy + (1 - out) * ty, out * oz + (1 - out) * tz
    cx_, cy_ = ix + (lx0 - x0) / q, iy + (ly0 - y0) / q
    dq = d / q
    wx, wy = cx_ + tx * sp + ox / np.maximum(-oz, 0.3) * dq, cy_ + ty * sp + oy / np.maximum(-oz, 0.3) * dq
    sx, sy = cx_ - L[0] / L[2] * (dq + 2 * hc), cy_ - L[1] / L[2] * (dq + 2 * hc)
    T = (1.0 - (F0 + (1 - F0) * (1 - nz) ** 5)) * front
    acc, rem = np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32)
    for Xs, Ys, wgt, tgt in ((wx, wy, T, acc), (sx, sy, front, rem)):
        xi, yi = np.round(Xs).astype(int), np.round(Ys).astype(int)
        on = (xi >= 0) & (xi < gw) & (yi >= 0) & (yi < gh)  # light landing off the box is not piled onto its edge
        np.add.at(tgt, (yi[on], xi[on]), wgt[on].astype(np.float32))  # (that drew the box's outline)
    dm = float(np.median(dq))
    g = 1.0 + CAUSTIC_GAIN * (cv2.GaussianBlur(acc, (0, 0), max(1.2, CAUSTIC_SOFT[0] * dm))  # the room's light is no
                              - cv2.GaussianBlur(rem, (0, 0), max(2.0, CAUSTIC_SOFT[1] * dm)))  # point: soft focus,
    g = cv2.resize(np.clip(g, 0.55, 3.0), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)  # softer shadow
    m = matte[y0:y1, x0:x1].astype(np.float32) / 255.0
    lift = CAUSTIC_KEEP * (g - 1.0)
    if held is not None:
        foot = np.array([cx, cy]) - hand.nrm[k, :2] * (R + ORB_GAP * s)  # the palm under it
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        q2 = ((xx - foot[0]) ** 2 + (yy - foot[1]) ** 2) / ao_r ** 2
        ao = SHADE_AO * np.exp(-0.5 * q2) * np.clip(1.0 - q2 / 9.0, 0.0, 1.0) ** 2  # nothing left of it by 3 sigmas
        lift = lift * (1.0 - ao) - ao
    edge = _box_fade(y1 - y0, x1 - x0, max(4, min(y1 - y0, x1 - x0) // 8))  # what reaches the box's edge fades by it
    P[y0:y1, x0:x1] *= (1.0 + lift * m * edge)[..., None]
    return (y0, y1, x0, x1)


def _box_fade(h: int, w: int, px: int) -> np.ndarray:
    """(h, w): 1 inside, falling smoothly to 0 over ``px`` at the edges."""
    fy = _ss(0.0, px, np.minimum(np.arange(h), np.arange(h)[::-1]) + 0.5)
    fx_ = _ss(0.0, px, np.minimum(np.arange(w), np.arange(w)[::-1]) + 0.5)
    return np.outer(fy, fx_).astype(np.float32)


def bubbles(P: np.ndarray, items: list, layer: Layer, lit: np.ndarray, L: np.ndarray) -> None:
    """Air caught in the water, over it, in place: each (x, y, r, seen) a small sphere of air, its rim silvered (light
    turned back at it) inside a faint dark edge, its middle a touch brighter, a pin of glint towards the light; at
    ``seen`` of its strength, and only where the water covers the picture."""
    H, W = P.shape[:2]
    by0, by1, bx0, bx1 = layer.box
    for x, y, r, seen in items:
        if r < 0.8 or not (by0 <= y < by1 and bx0 <= x < bx1):
            continue
        wgt = float(layer.alpha[int(y) - by0, int(x) - bx0]) * seen
        rr = int(2 * r + 3)
        ya, yb, xa, xb = max(0, int(y) - rr), min(H, int(y) + rr + 1), max(0, int(x) - rr), min(W, int(x) + rr + 1)
        if yb <= ya or xb <= xa or wgt <= 0.01:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        d = np.hypot(xx - x, yy - y)
        ring = np.exp(-((d - r) / (0.22 * r + 0.6)) ** 2)
        core = np.clip(1 - d / r, 0, 1)
        gx, gy = x + 0.45 * r * L[0], y + 0.45 * r * L[1]
        pin = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.18 * r + 0.4) ** 2))
        patch = P[ya:yb, xa:xb]
        edge = np.exp(-((d - 1.2 * r) / (0.15 * r + 0.5)) ** 2)
        patch *= (1.0 + 0.10 * wgt * core - 0.25 * wgt * edge)[..., None]
        patch += (wgt * (0.3 * ring + 0.8 * pin))[..., None] * lit[None, None, :]


# ---- free water: drops, the burst's sheet, spray --------------------------------------------------------------------

class Drops:
    """Water flying free of the hand, each piece on its own: under gravity and air drag, in the room's coordinates (the
    first frame's px) when the camera moves, so what was already thrown moves with the room, not with the screen, and
    seen in perspective from where it left, so it shrinks going away. A piece with a ``life`` shrinks away over its last
    ``fade`` frames (a torn sheet thinning out, spray, a drip soaking into the palm)."""

    def __init__(self, cam: np.ndarray | None, size: tuple[int, int], s: float):
        self.cam, self.size, self.zcam = cam, size, ZCAM * s
        self.x = np.zeros((0, 3))  # room px and depth towards the camera
        self.v = np.zeros((0, 3))  # px a frame
        self.z0 = np.zeros(0)  # the depth it left at (its perspective is from there)
        self.info = np.zeros((0, 7))  # r, foam, born, life, fade, stretch, gravity (px a frame squared)

    def add(self, k: int, x, v, r, foam=0.0, life=np.inf, fade=1.0, stretch=1.0, g=None) -> None:
        """Pieces leaving frame ``k`` at px and depth ``x`` (m, 3), ``v`` px a frame (m, 3)."""
        x, v = np.atleast_2d(np.asarray(x, np.float64)).copy(), np.atleast_2d(np.asarray(v, np.float64)).copy()
        m = len(x)
        if self.cam is not None:  # into the room's coordinates
            M = np.linalg.inv(self.cam[k])
            x[:, :2] = x[:, :2] @ M[:2, :2].T + M[:2, 2]
            v[:, :2] = v[:, :2] @ M[:2, :2].T
        g = G * self.zcam / ZCAM / FPS ** 2 if g is None else g
        info = np.stack(np.broadcast_arrays(r, foam, k, life, fade, stretch, g), -1).reshape(m, 7).astype(np.float64)
        self.x, self.v = np.concatenate([self.x, x]), np.concatenate([self.v, v])
        self.z0 = np.concatenate([self.z0, x[:, 2]])
        self.info = np.concatenate([self.info, info])

    def step(self) -> None:
        self.v[:, 1] += self.info[:, 6]
        self.v *= DRAG
        self.x += self.v

    def blobs(self, k: int) -> list[Blob]:
        """The pieces as frame ``k`` sees them; the ones gone (spent, out of the picture) are dropped."""
        if not len(self.x):
            return []
        H, W = self.size
        xy, vxy = self.x[:, :2], self.v[:, :2]
        if self.cam is not None:
            M = self.cam[k]
            xy, vxy = xy @ M[:2, :2].T + M[:2, 2], vxy @ M[:2, :2].T
        kz = self.zcam / np.maximum(self.zcam - (self.x[:, 2] - self.z0), 0.2 * self.zcam)
        ctr = np.array([W / 2.0, H / 2.0])
        xy = ctr + (xy - ctr) * kz[:, None]
        r0, foam, born, life, fade, stretch, _ = self.info.T
        age = k - born
        r = r0 * kz * np.clip((life - age) / fade, 0.0, 1.0) ** 0.8
        vxy = vxy * kz[:, None]
        z = self.x[:, 2]
        out = [Blob(xy[j, 0], xy[j, 1], z[j], r[j], _smeared(r[j], vxy[j], stretch[j]), vxy[j, 0], vxy[j, 1], 0.0,
                    foam[j]) for j in np.flatnonzero((age < life) & (r >= 0.5))]
        keep = (age < life) & (xy[:, 1] < H + 4 * r + 8) & (xy[:, 0] > -4 * r - 8) & (xy[:, 0] < W + 4 * r + 8)
        self.x, self.v, self.z0, self.info = self.x[keep], self.v[keep], self.z0[keep], self.info[keep]
        return out


def splash(drops: Drops, k: int, c: np.ndarray, R: float, v: np.ndarray, size: float, s: float, rng,
           flat: np.ndarray | None = None) -> None:
    """A ball of water of radius ``R`` at ``c`` (px, depth) moving ``v`` bursting on frame ``k``: what is left of the
    ball shrinking away over BURST_CORE frames as its skin is torn off it into sheet pieces flung out and thinning,
    drops of many sizes, and a fine spray. ``flat``: the normal of a surface it burst on (the palm), along which
    nothing goes in and along it only a little goes out."""
    def dirs(m: int) -> np.ndarray:
        d = _unit(rng.normal(size=(m, 3)))
        if flat is not None:
            d = _unit(d - (d @ flat)[:, None] * flat + 0.45 * np.abs(d @ flat)[:, None] * flat)
        return d

    pace = s / FPS * (0.6 + 0.4 * size)
    n_drop, n_sheet, n_spray = (max(1, int(a * (0.4 + 0.6 * size))) for a in SPLASH)
    drops.add(k, c, 0.5 * v, R, life=BURST_CORE, fade=BURST_CORE)
    d = dirs(n_sheet)
    drops.add(k, c + d * R * rng.uniform(*SHEET_FROM, (n_sheet, 1)),
              0.2 * v + d * rng.uniform(3.0, 5.0, (n_sheet, 1)) * pace, 0.22 * R * rng.uniform(0.6, 1.0, n_sheet),
              foam=0.3, life=SHEET_LIFE, fade=SHEET_LIFE, stretch=1.5)
    d = dirs(n_drop)
    r = R * np.clip(rng.lognormal(-2.4, 0.6, n_drop), 0.025, 0.35)
    drops.add(k, c + d * R * rng.uniform(0.5, 0.95, (n_drop, 1)),
              0.3 * v + d * rng.uniform(*SPLASH_SPEED, (n_drop, 1)) * pace, r, foam=np.where(r < 0.06 * R, 0.3, 0.0))
    d = dirs(n_spray)
    drops.add(k, c + d * 0.7 * R, 0.3 * v + d * rng.uniform(4.0, 10.0, (n_spray, 1)) * pace,
              R * rng.uniform(0.02, 0.05, n_spray), foam=1.0, life=rng.uniform(3, SPRAY_LIFE, n_spray), fade=SPRAY_LIFE)


# ---- the orb --------------------------------------------------------------------------------------------------------

class _Spring:
    """A second-order follower: the water's centre trails the hand and overshoots, as a mass of water would."""

    def __init__(self, x: np.ndarray, hz: float, zeta: float):
        self.x, self.v = np.asarray(x, np.float64).copy(), np.zeros(len(x))
        self.w, self.z = 2 * math.pi * hz, zeta

    def step(self, target: np.ndarray) -> np.ndarray:
        """One frame towards ``target``; returns the acceleration (px a second squared)."""
        h = 1.0 / FPS / 4
        for _ in range(4):
            a = self.w ** 2 * (target - self.x) - 2 * self.z * self.w * self.v
            self.v += a * h
            self.x += self.v * h
        return a


def _rotate(u: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Points ``u`` (m, 3) turned ``ang`` rad round unit ``axis`` (Rodrigues)."""
    c, s = math.cos(ang), math.sin(ang)
    return u * c + np.cross(axis, u) * s + np.outer(u @ axis, axis) * (1 - c)


def orb_scenes(tl: Timeline, track: dict, hand: Hand, size: tuple[int, int], seed: int,
               cam: np.ndarray | None) -> list[Scene]:
    """The orb, frame by frame (see the module's docstring and orb_plan)."""
    n = tl.n
    H, W = size
    scenes = [Scene() for _ in range(n)]
    plan = orb_plan(tl, track)
    g0, g1, rel = plan.gather, plan.formed, plan.release
    if g0 >= n:
        return scenes
    rng = _rng(seed, _ORB)
    e = np.clip(_tsmooth(tl.energy, VOICE_SMOOTH), 0, 1)
    drops = Drops(cam, size, float(np.median(hand.s)))
    gather = [(rng.uniform(0, 2 * math.pi), rng.uniform(*GATHER_FROM), rng.uniform(0.08, 0.18), rng.uniform(-0.8, 0.8),
               rng.uniform(0.0, 0.35), rng.uniform(g0 + 0.3 * (g1 - g0), g1), rng.choice([-1.0, 1.0]))
              for _ in range(GATHER_DROPS)]
    bub = _unit(rng.normal(size=(BUBBLES, 3))) * rng.uniform(0.15, 0.75, (BUBBLES, 1)) ** (1 / 3)
    bub_r = rng.uniform(*BUBBLE_R, BUBBLES)
    axis = _unit(np.array([rng.uniform(-1, 1), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 1.0)]))
    phase = {l: (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(-0.6, 0.6)) for l in (3, 4)}
    jolts = dict(plan.jolts)
    jolt_at: list[tuple[int, float]] = []  # (frame, size): for the lobes' ringing
    rings: list[tuple] = []  # (frame, unit offset from the centre, size)
    spring = None
    m2, m2v = np.zeros(2), np.zeros(2)
    circ = 0.0
    next_drip = g1 + int(rng.integers(*DRIP_EVERY))
    fly = None  # the orb in the air: (frame, centre, velocity, radius, thrown)
    for k in range(g0, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        R = ORB_R * s
        t = k / FPS
        sc.wave = WAVE[0] + WAVE[1] * float(e[k])
        if rel is None or k < rel:
            zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
            bob = ORB_BOB[0] * s * math.sin(2 * math.pi * ORB_BOB[1] * t)
            target = np.array([p[0], p[1], zp]) + nrm * (R + ORB_GAP * s + bob)
            if spring is None:
                spring = _Spring(target, *ORB_SPRING)
            acc = spring.step(target)
            c = spring.x
            if k in jolts:  # a hit while it is held jolts it, sends a ripple over it and spits drops
                z = jolts[k]
                ang = rng.uniform(0, 2 * math.pi)
                m2v += JOLT[0] * z * 2 * math.pi * MODE_HZ[2] * np.array([math.cos(ang), math.sin(ang)])
                jolt_at.append((k, z))
                rings.append((k, _unit(rng.normal(size=2)) * rng.uniform(0.1, 0.4), z))
                m = max(1, int(round(JOLT[1] * z)))
                d = _unit(rng.normal(size=(m, 3)) + 0.6 * nrm)
                drops.add(k, c + d * R * 0.95, spring.v / FPS + d * rng.uniform(2.5, 4.5, (m, 1)) * s / FPS,
                          R * rng.uniform(0.05, 0.09, m))
            w2 = 2 * math.pi * MODE_HZ[2]
            drive = -acc[:2] / max(s, 1.0) * MODE2_DRIVE
            m2v += (-(w2 ** 2) * m2 - 2 * MODE2_DAMP * w2 * m2v + drive * w2 ** 2) / FPS
            m2 += m2v / FPS
            circ += (CIRCULATE[0] + CIRCULATE[1] * float(e[k])) / FPS
            vol = float(_ss(g0 + 0.15 * (g1 - g0), g1, k)) if k < g1 else 1.0
            r = R * vol ** (1 / 3)
            st = float(np.hypot(*m2))
            u2 = m2 / st if st > 1e-6 else np.array([1.0, 0.0])
            k2 = 1.0 + min(MODE2_MAX, st)
            modes = []
            for l in (3, 4):
                ph0, psi, drift = phase[l]
                wl = 2 * math.pi * MODE_HZ[l]
                amp = MODE_AMP[l] * (MODE_REST + (1 - MODE_REST) * float(e[k])) * math.sin(wl * t + psi)
                amp += sum(1.5 * MODE_AMP[l] * z * math.exp(-(k - kj) / 8.0) * math.sin(wl * (k - kj) / FPS)
                           for kj, z in jolt_at)
                modes.append((l, amp, ph0 + drift * t))
            v_c = spring.v / FPS
            if r > 0.5:
                sc.blobs.append(Blob(c[0], c[1], c[2], r, _shape(u2, k2, 1.0 / math.sqrt(k2)), v_c[0], v_c[1],
                                     (CIRCULATE[0] + CIRCULATE[1] * float(e[k])) / FPS, 0.0, tuple(modes)))
            if r > 0.3 * R:
                sc.held = (c[0], c[1], c[2], r)
            for ang, d0, rs, z0, ts, ta, spin in gather:  # drops coming together out of the air, spiralling in
                ts_ = g0 + ts * (g1 - g0)
                if not ts_ <= k < ta:
                    continue
                u = (k - ts_) / max(ta - ts_, 1.0)
                rad = d0 * (1 - u) ** 1.6 * R
                a = ang + spin * 1.3 * u
                dd = np.array([math.cos(a), math.sin(a)])
                pos = c[:2] + rad * dd
                vel = -dd * 1.6 * d0 * R * (1 - u) ** 0.6 / max(ta - ts_, 1.0) + v_c[:2]
                rr = rs * R * float(_ss(0.0, 0.3, u))
                if rad > 0.55 * r and rr > 0.5:
                    sc.blobs.append(Blob(pos[0], pos[1], c[2] + z0 * R * (1 - u), rr, _smeared(rr, vel, 1.3),
                                         vel[0], vel[1]))
            if k >= g1:  # air inside, carried round as it circulates; then a drip off its underside, now and then
                rot = _rotate(bub, axis, circ)
                for (bxu, byu, bzu), br in zip(rot, bub_r):
                    grow = 1.0 + 0.12 * (1.0 - bzu)  # the far ones, seen through the water, a little magnified...
                    out = math.hypot(bxu, byu) * grow  # ... and those seen near its rim, through its grazing surface,
                    seen = (0.75 + 0.25 * bzu) * (1.0 - float(_ss(0.55, 0.85, out)))  # gone (they sat on its outline
                    sc.bubbles.append((c[0] + r * bxu * grow, c[1] + r * byu * grow, br * s, seen))  # like pearls)
                if k >= next_drip:
                    u = (k - next_drip) / DRIP_FORM
                    down = -nrm
                    if u < 1.0:
                        sc.blobs.append(Blob(*(c + down * (0.72 + 0.38 * u) * r), DRIP_R * r * (0.5 + 0.5 * u),
                                             vx=v_c[0], vy=v_c[1]))
                    else:  # it lets go and falls onto the palm under it
                        fall = (ORB_GAP * s + 0.1 * r) * float(np.hypot(*down[:2]))
                        life = max(2.0, FPS * math.sqrt(2 * max(fall, 1.0) / (G * s)))
                        drops.add(k, c + down * 1.1 * r, v_c + down * 0.4 * s / FPS, DRIP_R * r, life=life, fade=2)
                        next_drip = k + DRIP_FORM + int(rng.integers(*DRIP_EVERY))
            for kj, off, z in rings:
                if k - kj < 3 * RING[3]:
                    sc.rings.append((c[0] + off[0] * r, c[1] + off[1] * r, k - kj, RING[0] * z, RING[1] * r,
                                     RING[2] * r))
        elif fly is None or fly[1] is not None:  # let go and not yet burst
            if fly is None:  # let go: thrown along the aim, or dropped onto the palm
                c = spring.x if spring is not None else np.array([p[0], p[1], 0.0]) + nrm * (R + ORB_GAP * s)
                v0 = spring.v / FPS if spring is not None else np.zeros(3)
                if plan.throw:
                    aim = _aim(tl, hand, k)
                    sp = THROW_SPEED * s / FPS * (0.6 + 0.4 * plan.size)
                    v = 0.5 * v0 + np.array([aim[0] * sp, aim[1] * sp, -RECEDE * sp])
                else:
                    v = v0
                fly = (k, c.copy(), v, R, plan.throw, nrm.copy())
            k0, c0, v, R0, thrown, nrm0 = fly
            tau = k - k0
            if thrown:
                go = tau * tau / (2 * TAKEOFF) if tau < TAKEOFF else tau - TAKEOFF / 2
                c = c0 + v * go + np.array([0.0, 0.5 * G * s / FPS ** 2 * tau * tau, 0.0])
                vel = v * min(1.0, tau / TAKEOFF) + np.array([0.0, G * s / FPS ** 2 * tau, 0.0])  # (at full speed
                # from the frame it was let go on, it was drawn stretched into a long tear lying over the palm)
                kz = ZCAM * s / max(ZCAM * s - (c[2] - c0[2]), 0.2 * ZCAM * s)
                xy = np.array([W / 2.0, H / 2.0]) + (c[:2] - np.array([W / 2.0, H / 2.0])) * kz
                edge = min(xy[0], W - xy[0], xy[1], H - xy[1]) < 0.6 * R0 * kz
                done = tau >= BURST_AFTER or edge
            else:  # falling onto the palm under it
                c = c0 + v * tau - nrm0 * 0.5 * G * s / FPS ** 2 * tau * tau
                vel = v - nrm0 * G * s / FPS ** 2 * tau
                kz, xy = 1.0, c[:2]
                done = float((c0 - c) @ nrm0) >= ORB_GAP * s
            if done:  # burst: from here on only its drops are left
                splash(drops, k, np.array([xy[0], xy[1], c[2]]), R0 * kz, vel, plan.size, s, rng,
                       None if thrown else nrm0)
                fly = (k0, None, v, R0, thrown, nrm0)
            else:  # in the air: stretched along its flight, a tail pulled out behind it
                sp = float(np.hypot(*vel[:2])) * kz
                u = vel[:2] / max(float(np.hypot(*vel[:2])), 1e-6)
                stretch = 1.0 + min(0.8, sp / (2.5 * R0)) if thrown else 1.0
                sc.blobs.append(Blob(xy[0], xy[1], c[2], R0 * kz, _smeared(R0 * kz, vel[:2] * kz, stretch),
                                     vel[0] * kz, vel[1] * kz, CIRCULATE[0] / FPS))
                if thrown:
                    for j in range(1, 5):
                        tp = xy - u * sp * 0.35 * j
                        sc.blobs.append(Blob(tp[0], tp[1], c[2], R0 * kz * (0.5 - 0.08 * j), vx=vel[0] * kz,
                                             vy=vel[1] * kz, foam=0.15 * j))
        sc.blobs += drops.blobs(k)
        drops.step()
    return scenes


# ---- the whip -------------------------------------------------------------------------------------------------------

def _chain(pts: np.ndarray, per: int) -> np.ndarray:
    """A Catmull-Rom curve through ``pts`` (m, d), ``per`` points a span."""
    P = np.concatenate([pts[:1], pts, pts[-1:]])
    t = np.arange(per)[:, None] / per
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(pts[-1:])
    return np.concatenate(out)


def _swirl(hand: Hand, k: int, root: np.ndarray, u: np.ndarray, phase: float, sweep: float, t: float) -> np.ndarray:
    """(m, 3): the shape the stream holds on frame ``k`` at shares ``u`` of its length: out of the palm at ``root``,
    swung out over WHIP_RISE of it to a circle round the hand (facing the camera, tipped so one side goes behind the
    hand), wrapped ``sweep`` turns round it from ``phase`` rad, its lobes running round it (``t`` seconds in)."""
    s = float(hand.s[k])
    along = hand.along[k]
    tilt = math.radians(WHIP_TILT)
    axis = _unit(WHIP_FACE * np.array([0.0, 0.0, 1.0]) + (1.0 - WHIP_FACE) * hand.nrm[k])
    axis = _unit(axis * math.cos(tilt) + along * math.sin(tilt))
    e1 = _unit(along - (along @ axis) * axis)
    e2 = np.cross(axis, e1)
    th = phase + 2.0 * math.pi * sweep * u
    rho = WHIP_ORBIT * s * _ss(0.0, WHIP_RISE, u)
    rho = rho * (1.0 + WHIP_WOBBLE[0] * np.sin(WHIP_WOBBLE[1] * th - 2.0 * math.pi * WHIP_WOBBLE[2] * t))
    return root + rho[:, None] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)


def _lash(root: np.ndarray, aim: np.ndarray, u: np.ndarray, s: float, age: float) -> np.ndarray:
    """(m, 3): the stream lashed out straight along ``aim`` (unit, 3) by a hit ``age`` frames ago, a kink running down
    it from the root to the tail as a whip's loop does."""
    side = _unit(np.cross(aim, np.array([0.0, 0.0, 1.0])) + 1e-9)
    at = min(1.0, age / (0.6 * WHIP_CRACK[0]))
    kink = WHIP_CRACK[2] * s * np.exp(-((u - at) / 0.15) ** 2) * (1.0 - at)
    return root + np.outer(u * WHIP_CRACK[1] * s, aim) + np.outer(kink, side)


def whip_scenes(tl: Timeline, track: dict, hand: Hand, size: tuple[int, int], seed: int,
                cam: np.ndarray | None) -> list[Scene]:
    """The whip, frame by frame (see the module's docstring and whip_plan): each node of the stream is held to the
    swirl it makes round the hand (_swirl) by a spring, stiff at the root and soft at the tail, so when the hand
    moves the stream trails it like a ribbon; a hit lashes it out along the aim (_lash). It is worked in the
    picture's px (the hand and a first-person camera move together); the drops it sheds fly in the room's."""
    n = tl.n
    scenes = [Scene() for _ in range(n)]
    plan = whip_plan(tl, track)
    if plan.start >= n:
        return scenes
    rng = _rng(seed, _WHIP)
    e = np.clip(_tsmooth(tl.energy, VOICE_SMOOTH), 0, 1)
    drops = Drops(cam, size, float(np.median(hand.s)))
    N = WHIP_NODES
    u_n = np.arange(N) / (N - 1)
    sub = 4
    dt = 1.0 / FPS / sub
    x = v = root0 = None
    phase = rng.uniform(0.0, 2.0 * math.pi)
    for k in range(plan.start, n):
        sc = scenes[k]
        p, nrm, s = hand.p[k], hand.nrm[k], float(hand.s[k])
        zp = float(hand.depth(k, np.float32([p[0]]), np.float32([p[1]]))[0])
        root = np.array([p[0], p[1], zp]) + nrm * WHIP_LIFT * s
        alive = plan.end is None or k < plan.end
        broke = 0 if alive else k - plan.end  # frames since it let go
        grow = float(_ss(plan.start, plan.start + WHIP_GROW, k))
        t = k / FPS
        sweep = WHIP_SWEEP[0] + (WHIP_SWEEP[1] - WHIP_SWEEP[0]) * float(e[k])
        phase += 2.0 * math.pi * (WHIP_SPIN[0] + WHIP_SPIN[1] * float(e[k])) / FPS
        u_t = u_n * max(grow, 0.02)  # poured out: the nodes lie along the part of the swirl out so far
        target = _swirl(hand, k, root, u_t, phase, sweep, t)
        crack = 0.0
        for kc, z in plan.cracks:
            age = k - kc
            if 0 <= age < WHIP_CRACK[0]:
                w = min(1.0, (age + 1) / 2.0) * float(_ss(WHIP_CRACK[0], 0.5 * WHIP_CRACK[0], age)) * min(1.0, z)
                if w > crack:
                    crack, crack_age = w, age
        if crack > 0:
            aim = _aim(tl, hand, k)
            aim3 = _unit(np.array([aim[0], aim[1], -0.2]))
            target = (1.0 - crack) * target + crack * _lash(root, aim3, u_t, s, crack_age)
        if x is None:
            x, v, root0 = target.copy(), np.zeros((N, 3)), root
        sc.wave = WAVE[0] + WAVE[1] * float(e[k])
        if broke <= WHIP_BREAK:
            om = 2.0 * math.pi * (WHIP_HZ[0] + (WHIP_HZ[1] - WHIP_HZ[0]) * u_n ** 0.7) * (1.0 + (WHIP_CRACK[3] - 1.0)
                                                                                         * crack)
            for j in range(sub):
                if alive:
                    a = om[:, None] ** 2 * (target - x) - 2.0 * WHIP_DAMP * om[:, None] * v
                    a[:, 1] += WHIP_SAG * G * s
                else:
                    a = np.zeros((N, 3))
                    a[:, 1] += G * s
                v = v + a * dt
                x = x + v * dt
                if alive:  # the root rides the palm (between this frame's place and the last's)
                    x[0], v[0] = root0 + (root - root0) * (j + 1) / sub, (root - root0) * FPS
            root0 = root
            tail = v[-1, :2]
            if alive and np.hypot(*tail) > WHIP_SHED * s and rng.random() < 0.7:
                m = int(rng.integers(1, 3))
                drops.add(k, np.repeat(x[-1:], m, 0), np.repeat(v[-1:] * 0.8 / FPS, m, 0)
                          + rng.normal(size=(m, 3)) * 0.04 * s, s * rng.uniform(0.015, 0.035, m))
            pts = _chain(x, 6)
            vel = _chain(v / FPS, 6)
            arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            u = arc / max(arc[-1], 1e-6)
            rad = s * (WHIP_R[0] * (1 - u) ** 0.9 + WHIP_R[1]) * grow ** 0.5
            rad *= 1 + WHIP_WAVES[0] * np.sin(2 * math.pi * (WHIP_WAVES[1] * u - WHIP_WAVES[2] * t))
            lam = max(9.0 * float(np.mean(rad)), 1.0)  # the beads it necks into are this far apart
            neck = broke / WHIP_BREAK if not alive else 0.0
            rad *= 1.0 - 0.85 * neck * (0.5 - 0.5 * np.cos(2 * math.pi * arc / lam))
            if not alive and broke == WHIP_BREAK:  # necked through: beads, falling
                beads = np.flatnonzero(np.diff(np.floor(arc / lam + 0.5)) > 0)
                rb = (0.75 * rad[beads] ** 2 * lam) ** (1 / 3)  # a bead holds its length of the stream
                drops.add(k, pts[beads], vel[beads] + rng.normal(size=(len(beads), 3)) * 0.02 * s, rb)
                drops.add(k, x[0], v[0] / FPS, WHIP_BULB * s * 0.8)
                x = x[:0]
            else:
                tang = _unit(np.gradient(pts, axis=0))
                speed = np.linalg.norm(vel[:, :2], axis=1) * FPS / max(s, 1.0)
                foam = np.clip((u - 0.7) / 0.3, 0, 1) * 0.35 + np.clip((speed - 5.0) / 6.0, 0, 1) * 0.3
                fl = vel[:, :2] + tang[:, :2] * WHIP_FLOW * s / FPS
                step = 0
                for i in range(len(pts)):  # blobs at most 0.4 of their radius apart
                    if i and arc[i] - arc[step] < 0.4 * rad[i]:
                        continue
                    step = i
                    sc.blobs.append(Blob(pts[i, 0], pts[i, 1], pts[i, 2], rad[i], vx=fl[i, 0], vy=fl[i, 1],
                                         foam=float(foam[i])))
                if alive:
                    sc.blobs.append(Blob(x[0, 0], x[0, 1], x[0, 2], WHIP_BULB * s * max(grow, 0.3), vx=v[0, 0] / FPS,
                                         vy=v[0, 1] / FPS, spin=CIRCULATE[0] / FPS,
                                         modes=((3, 0.05 * math.sin(2 * math.pi * MODE_HZ[3] * t), k * 0.07),)))
        sc.blobs += drops.blobs(k)
        drops.step()
        if not len(x) and not len(drops.x):
            break
    return scenes


# ---- the shot -------------------------------------------------------------------------------------------------------

GRAIN_UNIT = 0.943  # what fx_light's grain reading reads off unit white noise: the plate's grain is laid at its strength


def frame(k: int, plate_k: np.ndarray, matte: np.ndarray | None, sc: Scene, hand: Hand, room: Room, noise: Noise,
          code: float, seed: int) -> np.ndarray:
    """Frame ``k`` with its water: uint8 RGB, the plate's own bytes wherever there is none."""
    out = plate_k.copy()
    if not sc.blobs:
        return out
    H, W = plate_k.shape[:2]
    fld = Field(sc.blobs, H, W)
    if fld.box is None:
        return out
    P = lin(plate_k)
    layer = shade(fld, sc, k, hand, room, noise, _pyramid(P), matte, code, seed)
    if layer is None:
        return out
    boxes = [layer.box]
    if matte is not None:
        boxes.append(caustic(P, layer, k, hand, room, matte, sc.held))
    y0, y1, x0, x1 = layer.box
    P[y0:y1, x0:x1] = P[y0:y1, x0:x1] * (1.0 - layer.alpha[..., None]) + layer.colour
    if sc.bubbles:
        bubbles(P, sc.bubbles, layer, room.lit(k), room.key(k, (sc.held or (W / 2, H / 2))[:2]))
        bx = np.array([[b[0], b[1], b[2]] for b in sc.bubbles])
        rr = int(2 * bx[:, 2].max() + 4)
        boxes.append((max(0, int(bx[:, 1].min()) - rr), min(H, int(bx[:, 1].max()) + rr + 1),
                      max(0, int(bx[:, 0].min()) - rr), min(W, int(bx[:, 0].max()) + rr + 1)))
    for y0, y1, x0, x1 in boxes:
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = gam(P[y0:y1, x0:x1])
    return out


def render(plate: np.ndarray, track: dict, mask: np.ndarray | None, timeline: Timeline, form: str, seed: int = 0,
           progress=None) -> np.ndarray:
    """The shot's frames with the water in them, uint8 RGB [T, H, W, 3] (cuesync.elements' contract): ``plate`` uint8
    RGB [T, H, W, 3], ``track`` the hand's (its emitter carried to the tap), ``mask`` the hand's matte [T, H, W] uint8
    (or None: then nothing hides the water), ``form`` "orb" or "whip". The same inputs and seed give the same bytes.
    ``progress(stage, share, message)``, if given, hears how far it has got."""
    if form not in FORMS["water"]:
        raise ValueError(f"water has no form {form!r}; one of {FORMS['water']}")
    plate = np.asarray(plate)
    n, H, W = plate.shape[:3]
    if timeline.n != n or (mask is not None and len(mask) != n):
        raise ValueError(f"the plate has {n} frames, the timeline {timeline.n}, the mask "
                         f"{None if mask is None else len(mask)}")
    say = progress or (lambda stage, share, msg: None)
    say("water", 0.0, "Working out where the water goes")
    hand = hand_of(track, n)
    cam = None if timeline.camera is None else np.asarray(timeline.camera, np.float64)
    scenes = (orb_scenes if form == "orb" else whip_scenes)(timeline, track, hand, (H, W), seed, cam)
    room = Room(plate)
    noise = Noise(seed)
    wet = [k for k in range(n) if scenes[k].blobs]
    code = fx.shadow_noise(plate) / GRAIN_UNIT / 255.0 if wet else 0.0
    out = np.empty_like(plate)
    done = [0]

    def one(k: int) -> None:
        out[k] = frame(k, plate[k], None if mask is None else mask[k], scenes[k], hand, room, noise, code, seed)
        done[0] += 1
        if done[0] % 8 == 0:
            say("water", done[0] / n, f"Shaping the water: frame {done[0]} of {n}")

    with ThreadPoolExecutor(fx.THREADS) as ex:
        list(ex.map(one, range(n)))
    say("water", 1.0, "The water is in")
    return out
