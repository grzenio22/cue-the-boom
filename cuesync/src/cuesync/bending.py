"""Element bending: a performer's hand in the shot, an element out of it on the beat.

The hand's track (``hand_track.json``, from ``cuesync.hand_track``: where the emitter is, which way it points, how big
the hand is, its gesture and its strikes, frame by frame) meets the performance here:

- **Which element** (``bending_effect``): what the sound is (``gestures.heard``: a boom, a whoosh, a hiss, a puff) and
  what the hand is doing at the hit. Two fingers or a pointing finger with a long "ffff" or "fwooom" is a flamethrower;
  an open palm with a held "ffff" is a flame burning in it; a fist's strike with a "BOOM" is a burst, a palm's
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
from . import fx_light as fx
from . import fire_relight as fire_light
from . import hand_surface
from .fx_volume import volume_element

POINTING = ("Victory", "Pointing_Up")  # two fingers, or one: a jet from the fingertips
SNAP_BEFORE, SNAP_AFTER = 4, 1  # frames at 24 fps the strike may come before / after the voice's hit and still take it
MISMATCH_NEAR = 12  # frames: a strike this close to a hit, but outside the snap window, is the hand off the beat
COMBO_S = 1.5  # booms closer together than this are one combo: a burst per strike
EXPLODE_S = 0.4  # a boom held this long unbroken, with a fist or a palm strike, is an explosion (a short one a fireball)

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
SIZE = {"jet": (6.0, 0.75), "burst": (2.0, 0.45), "plume": (1.8, 0.45), "puff": (2.5, 0.4)}  # per behaviour: its reach in
# hand sizes, and at most this share of the frame's short side (a close-up hand made a fireball fill the frame)
FIT = {"plume": 2.8, "burst": 2.2}  # how big a flame and a fireball play at their fullest (the loudest hit), in hands of
# this hand, measured on what the render draws: a palm's flame this tall, a fireball this wide (its lit area's
# equivalent diameter). The model paints well inside its guide: a render guided to SIZE's reach drew a palm-high
# flame and a fist-sized fireball, small against the hand that threw it
FIT_TIP = 1.1  # a fireball from the fingertips plays smaller than a fist's: a shot, not a blast (at 1.6 a finger
# gun's ball overwhelmed the gesture and read as an explosion)
FIT_MOST = 3.0  # and never more than this many times as big as rendered
FIT_LIT = 0.04  # linear: what of a render counts as drawn (its haze is fainter)
FIT_ROOM = (0.06, 0.03)  # shares of the frame: a flame's top stays this far under the frame's top, and a fireball
# this far inside its edges (grown on a hand near the edge, it was cut by it)
THROW, THROW_FRAMES, RELEASE_AFTER = 1.0, 7.0, 1  # a fireball leaves the fist a frame after the strike and flies on
# along it towards THROW hands, its speed dying away over THROW_FRAMES (stopped dead, it hung in the air)
RISE = 0.025  # hands a frame: and it rises as it burns out, hot gas
IGNITION = 0.15  # an element has ignited once its light is this share of its brightest frame's
LOST_GRACE, LOST_FADE = 0, 4  # the hand lost (and the picture where it was changed): the element fades out over these
# frames from where it was last seen (after 8 frames' grace it burned a third of a second beside a swung fist; the
# track bridges any shorter gap itself, so a lost frame is never a flicker)
ROOT_FADE = 0.12  # share of the element's reach over which it comes up from nothing at the fingertips: the model draws
# a jet's root as a hard bright nozzle and a flame's foot as a flat edge, and fire from a hand has neither
ROOT_ROUND = 0.15  # how far the root's fade curves back at the sides, in its own length over its width squared
ROOT_FADE_SHARE = {"steam_jet": 0.05, "fire_breath": 0.06}  # steam has no nozzle to hide (over the full fade it
# started a hand from the palm); a jet six hands long left a gap at the finger (a palm's flame: FOOT)
FOOT, FOOT_WOBBLE, FOOT_RISE = 0.25, 0.35, 0.08  # a flame standing on the palm comes up out of it over this many
# hands from the emitter, the line it comes up along broken by noise a third of that deep, rising this many hands a
# frame with the flame (its foot as rendered, a clean arc on the palm, read as a cut-out pasted on; faded over half a
# hand, it floated above the skin)
CONTACT = (0.35, 1.6)  # the pool of light the foot throws on the palm it stands in: its radius in hands (wider across
# than along the flame), and its brightness against the foot's own (the room light sees the flame from half its
# radius away; its foot is on the skin)
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
                   lit: bool = False, longest_s: float = 0.0) -> tuple[str | None, str]:
    """(effect, why) for a hand in the shot. ``kind``: the take's sound (gestures.heard); ``gesture``: the hand's at
    the hit (gesture_at); ``hits``/``span_s``: how many hits the take has and how long from the first to the last;
    ``snap``: the track saw a finger snap; ``lit``: a flame burns at the tapped spot; ``longest_s``: the voice's
    longest unbroken sound (a rolling "BOOOOM" with a strike is an explosion).
    None when there is no sound to read (the caller falls back to its own default)."""
    if snap:
        return "blow_out", "a snap: the candle goes out"
    if kind is None:
        return None, "no sound to read"
    combo = kind == "boom" and hits >= 2 and span_s <= COMBO_S
    if kind == "buzz":  # a voiced "zzzzz": lightning
        if gesture == "Open_Palm":
            return "lightning_ball", "a buzz over an open palm: ball lightning"
        return "lightning_arc", "a buzz from the hand: lightning" + (" from the fingertips" if gesture in POINTING else "")
    if kind == "boom" and gesture in ("Closed_Fist", "Open_Palm") and longest_s >= EXPLODE_S:
        return "explosion", "a long rolling BOOM with a strike: an explosion"
    if gesture in POINTING:
        if kind in ("hiss", "whoosh"):
            return "fire_breath", f"a long {kind} from pointed fingers: a flamethrower"
        if kind == "boom":
            return ("fire_breath", "strikes from the fingertips") if combo else ("fire_burst", "a boom from the fingertips")
    if gesture == "Open_Palm":
        if kind == "hiss":  # a held "fffff" over a palm is a flame in it (as Try it live shows it; steam is a tap on the
            # badge away)
            return "flame_plume", "an open palm and a held fffff: a flame burning in it"
        if kind == "whoosh":
            return "flame_plume", "an open palm and a swelling voice: a flame rising from it"
        if kind == "boom":
            return "fire_burst", "a palm strike with a boom: a fireball" + (" per strike" if combo else "")
        return "fire_burst", "an open palm and a pf: a pop of fire, as big as it is loud"  # (a hand throws fire: the
        # air blast and steam are a tap on the badge away)
    if gesture == "Closed_Fist" and kind in ("boom", "puff"):
        return "fire_burst", "a fist's strike: a burst" + (" per strike" if combo else "")
    why = "the hand's gesture isn't one we read" if gesture is None else "by the sound alone"
    if kind == "puff" and not lit:  # a hand's pf, whatever the hand is doing: a pop of fire
        return "fire_burst", why + ": a pf, a pop of fire"
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


CARRY_NEAREST = 3  # the tap rides the hand's landmarks nearest it (the nearest counting most), not the tracker's
# emitter: a fingertip tapped, kept as an offset from the palm point (1.4 hands across), went a hand off the finger
# into mid-air as it curled into a punch


def filled_landmarks(track: dict) -> np.ndarray | None:
    """[T, 21, 2]: each frame's landmarks (px), a frame without them holding the last ones before (the first found,
    before any), as the track holds its emitter. None when the track has none."""
    lms = track.get("landmarks") or []
    found = [np.asarray(lm, np.float64)[:, :2] for lm in lms if lm is not None]
    if not found:
        return None
    out, last = [], found[0]
    for t in range(len(track["xy"])):
        lm = lms[t] if t < len(lms) else None
        last = last if lm is None else np.asarray(lm, np.float64)[:, :2]
        out.append(last)
    return np.stack(out)


def anchored(track: dict, k: int, tap: tuple[float, float], aim: float | None = None) -> dict:
    """The track with its emitter moved to where the performer tapped on frame ``k`` (``tap`` in px), carried with the
    hand from there: on the landmarks nearest the tap (a fingertip tapped stays on that fingertip as it curls), the
    tap kept as an offset from them in the hand's own frame (along its direction and across it, in hand sizes), so
    it moves, turns and scales with the hand; on the tracker's emitter when the track has no landmarks. A tracker's
    palm point is the landmarks' middle, which for an upturned palm seen from the side lies inside the hand, not on
    the palm the fire should rise from. A tap on a frame the hand wasn't seen on (blurred by a finger gun's recoil,
    say) goes on the part of the hand nearest it on the nearest frame it was seen: measured from the hand where it
    was before or after, it landed 0.7 hands off the fingertip. ``aim``: the studio's aim arrow on frame ``k`` (degrees
    clockwise from up); the direction turns by the same angle on every frame, so the aim too is carried with the
    hand."""
    k = int(np.clip(k, 0, len(track["xy"]) - 1))
    p, u, s = emitter_at(track, k)
    tap_ = np.asarray(tap, np.float64)
    L = filled_landmarks(track)
    lms = track.get("landmarks") or []
    seen = k < len(lms) and lms[k] is not None
    if L is not None:
        k_ = k if seen else min((t for t, lm in enumerate(lms) if lm is not None), key=lambda t: abs(t - k))
        dist = np.linalg.norm(L[k_] - tap_, axis=-1)
        near = np.argsort(dist)[:CARRY_NEAREST if seen else 1]
        wts = 1.0 / (dist[near] + 0.05 * max(s, 1e-6)) ** 2
        wts = wts / wts.sum()
        base = [wts @ L[t][near] for t in range(len(L))]
    else:
        near, base = [], [emitter_at(track, t)[0] for t in range(len(track["xy"]))]
    d = tap_ - base[k] if seen or L is None else np.zeros(2)
    a, b = float(d @ u) / max(s, 1e-6), float(d @ np.array([-u[1], u[0]])) / max(s, 1e-6)
    turn = 0.0 if aim is None else (aim - angle_of(u) + 180.0) % 360.0 - 180.0
    c, sn = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    xy, dirs = [], []
    for t in range(len(track["xy"])):
        _, ut, st = emitter_at(track, t)
        xy.append([float(v) for v in base[t] + st * (a * ut + b * np.array([-ut[1], ut[0]]))])
        dirs.append([float(ut[0] * c - ut[1] * sn), float(ut[0] * sn + ut[1] * c)])  # clockwise on screen (y down)
    return track | {"xy": xy, "dir": dirs, "anchor": {"frame": k, "along": round(a, 3), "across": round(b, 3),
                                                      "turn": round(turn, 1), "on": [int(j) for j in near]}}


def angle_of(u) -> float:
    """A unit direction as degrees clockwise from straight up (the studio's ``direction``)."""
    return math.degrees(math.atan2(float(u[0]), -float(u[1])))


def towards_words(direction: float) -> str:
    a = math.radians(direction)
    ux, uy = math.sin(a), -math.cos(a)
    return "upwards" if uy < -0.7 else "downwards" if uy > 0.7 else "to the left" if ux < 0 else "to the right"


def ignition_frame(element: np.ndarray, floor: int, burst: bool = False) -> int | None:
    """The element frame where it ignites: the first whose light above its black is IGNITION of the brightest's; for a
    burst, the frame it grows fastest (a "BOOM" is a whoomph, its beat the blast, not the first spark). An element too
    small to lift a frame's mean over its black (a palm's flame on a landscape canvas, a hundredth of it, read as an
    empty render) is measured by its light where it is, pixels over the black."""
    lum = np.array([max(0.0, float(f[::4, ::4].astype(np.float32).mean()) - floor) for f in element])
    if lum.max() <= 0:
        lum = np.array([float(np.clip(f[::4, ::4].astype(np.float32) - floor - FLOOR_SPAN, 0.0, None).mean())
                        for f in element])  # (clearly over the black: an empty render's noise is not an element)
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
SOOT, SOOT_DARK = 0.9, 0.02  # a flame is not light alone: where it outshines what is behind it (its luminance over
# that's, plus SOOT_DARK) its glare and soot hide up to 90 % of it, so on a white wall it keeps its shape; where it is
# dimmer than the wall it only adds (hiding it there drew a rusty outline round the flame and turned sparks to dirt)
SHUTTER, DEFOCUS = 0.5, 0.8  # the fire is smeared for half a frame of the hand's motion (a phone's shutter), and softened
# this many px at 1080 (no phone films as crisply as the render draws): a crisp flame on soft footage read as pasted on
SOOT_FROM = 0.012  # linear: fainter than this is the render's own haze, not fire
SOOT_BODY, SOOT_EDGE = 0.011, 0.018  # shares of the short side: the flame hides by its body this smooth, and the cap on
# how much it hides (by what is there) this smooth
DAY_CORE = (0.12, 0.45, 0.04, 0.18)  # over a room this bright behind it (linear, from the first to the second), the
# flame's dense body (its own light from the third to the fourth) covers the room whatever it outshines: on a white
# wall it was a pale see-through veil, the wall showing through its body
FIRE_PEAK = 0.85  # linear luminance of a render's hottest fire (its 99.5th percentile is 0.82-0.97): its heat's 1
DAY = (0.03, 0.3)  # linear: the room behind a flame from dark to lit. A camera exposed for a lit room sees a flame's
CORE_GAIN, CORE_FROM = 2.5, 0.65  # core blown to white: up to this many times as bright as rendered where its heat
# is over CORE_FROM, in a lit room (a dark room's exposure already shows the render's own white core). Brightening the
# whole flame instead (twice over a white wall) only took it orange, then yellow, then lemon: the render's red is
# clipped already, so more gain can't reach white, and its tips, body and core all went up alike: a flat sticker
BODY_GAIN = 0.25  # and the whole flame up to this much brighter over a bright room: the render was exposed over black,
# and a flame doesn't dim with the room. Its light on the hand is worked out from the flame as rendered
LIGHT_CAP = 0.65  # the fire lifts no surface past this share of its own body's luminance (skin's albedo): uncapped,
# a lit fist by a fireball went a blown peach as bright as the fire, and the fire read as a sprite on a tinted hand (at
# 0.5 the light undersold it)
FIRE_LIGHT = 2.0  # the fire's light on the room against fx_light's physical light from it: a fireball by a lit wall
# lifted it 12 % (a phone filming a real one sees the wall flare)
RIM = 0.5  # how much the flame's dim outer edge is kept from hiding more of the room than it gives back (a darker
# outline round it, the room's luminance halved just outside the flame); all of it let the wall through the fireball
BLOOM = ((0.004, 0.06), (0.018, 0.035), (0.07, 0.02))  # the lens's glow round the fire: (sigma, share of the short
BLOOM_DARK = 0.5  # side; weight), half as strong again in a dark room. (fx_light's glow is light on surfaces: over a lit
# wall it can't be seen)
BLEED, BLEED_TINT = 0.5, (1.0, 0.96, 0.88)  # where the fire overexposes a channel, half of that spills into the others
# through a warm white: orange, then cream, then white, as on a sensor, not orange, yellow, lemon


def _smooth(a: float, b: float, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fire_hdr(E0: np.ndarray, behind: float) -> np.ndarray:
    """The rendered fire ``E0`` (linear, over black) as a camera exposed for the room behind it (``behind``,
    room_behind) sees it: its core brighter in a lit room (CORE_GAIN), the whole of it a little (BODY_GAIN), and its
    colour kept to what its heat allows: no greener than the renders' own 99th percentile at that heat (a dim speck
    yellower than its brightness read khaki, or green over an olive wall)."""
    from . import fx_light as fx

    Y = fx._lum(E0)
    h = np.clip(Y / FIRE_PEAK, 0.0, 1.5)
    day = float(_smooth(DAY[0], DAY[1], behind))
    g = (1.0 + BODY_GAIN * min(1.0, behind / fx.EXPOSED_WHITE)) * (1.0 + (CORE_GAIN - 1.0) * day
                                                                   * _smooth(CORE_FROM, 1.0, h))
    c = E0 / np.maximum(Y, 1e-6)[..., None]  # its colour at luminance 1
    hh = np.minimum(h, 1.0)
    g2 = np.minimum(c[..., 1], c[..., 0] * np.minimum(1.0, 0.18 + 0.85 * hh))
    b2 = np.minimum(c[..., 2], g2 * (0.1 + 0.9 * _smooth(0.6, 1.0, hh)))
    c = np.stack([c[..., 0], g2, b2], -1)
    c /= np.maximum(fx._lum(c), 1e-6)[..., None]
    return (c * (Y * g)[..., None]).astype(np.float32)


def cap_light(P: np.ndarray, plate: np.ndarray, E: np.ndarray) -> None:
    """In place: the fire's light on the plate (``P`` lit, ``plate`` before) lifting no surface much past LIGHT_CAP of
    the flame's own body (``E``'s luminance, each pixel weighted by its own), rolled off towards it (tanh) rather than
    cut at it: cut, all the lit skin came to the same brightness, an even orange cast over the arm with no falloff."""
    from . import fx_light as fx

    Y = fx._lum(E[::4, ::4])
    wgt = Y * (Y > 0.02)
    if wgt.sum() <= 0:
        return
    body = float((Y * wgt).sum() / wgt.sum())
    D = P - plate
    L = np.maximum(D, 0.0)
    lL = fx._lum(L)
    allowed = np.maximum(LIGHT_CAP * body - fx._lum(plate), 0.15 * LIGHT_CAP * body)  # (a little always gets through)
    s = allowed * np.tanh(lL / allowed) / np.maximum(lL, 1e-6)
    P[...] = plate + (D - L) + L * s[..., None]


def contact_light(P: np.ndarray, E: np.ndarray, p, u, s: float, m: np.ndarray | None) -> None:
    """In place: the pool of light a flame's foot throws on the palm it stands in (emitter ``p``, the flame along
    ``u``, a hand ``s`` px; ``m`` the hand's matte or None): an ellipse CONTACT[0] hands round, flattened along the
    flame, as bright as CONTACT[1] times the flame's foot (``E`` near the emitter), times each pixel's own colour."""
    from . import fx_light as fx

    h, w = P.shape[:2]
    r = max(CONTACT[0] * s, 2.0)
    R = int(2.5 * r)
    y0, y1, x0, x1 = max(0, int(p[1]) - R), min(h, int(p[1]) + R), max(0, int(p[0]) - R), min(w, int(p[0]) + R)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - float(p[0]), yy - float(p[1])
    along = dx * float(u[0]) + dy * float(u[1])
    across = dx * float(u[1]) - dy * float(u[0])
    pool = np.exp(-0.5 * ((across / (1.3 * r)) ** 2 + ((along - 0.1 * s) / (0.7 * r)) ** 2)).astype(np.float32)
    look = np.exp(-0.5 * (across ** 2 + (along - 0.3 * s) ** 2) / (0.4 * s) ** 2).astype(np.float32)  # the foot
    foot = (E[y0:y1, x0:x1] * look[..., None]).reshape(-1, 3).sum(0) / max(float(look.sum()), 1e-6)
    if float(foot.max()) <= 1e-4:
        return
    Pb = np.maximum(P[y0:y1, x0:x1], 0.0)
    under = float(fx._lum(Pb)[pool > 0.5].mean()) if (pool > 0.5).any() else 0.0
    albedo = Pb / max(under, 1e-3) * 0.6  # (each pixel's own colour: the skin keeps its texture under the light)
    on = pool if m is None else pool * cv2.GaussianBlur(m[y0:y1, x0:x1].astype(np.float32), (0, 0), 3.0)
    P[y0:y1, x0:x1] += albedo * (CONTACT[1] * on)[..., None] * foot[None, None, :]


def bloom(E: np.ndarray, dark: float) -> np.ndarray:
    """The lens's glow round the fire ``E`` (BLOOM), stronger in a dark room (``dark`` 1)."""
    h, w = E.shape[:2]
    side, q = min(h, w), 4
    out = np.zeros_like(E)
    small = None
    for sg, wt in BLOOM:
        sig = sg * side
        if sig < 6:
            out += wt * cv2.GaussianBlur(E, (0, 0), sig)
        else:
            small = cv2.resize(E, (w // q, h // q), interpolation=cv2.INTER_AREA) if small is None else small
            out += wt * cv2.resize(cv2.GaussianBlur(small, (0, 0), sig / q), (w, h), interpolation=cv2.INTER_LINEAR)
    return out * np.float32(1.0 + BLOOM_DARK * dark)


def bleed(P: np.ndarray, E: np.ndarray) -> None:
    """In place: the fire's own overexposure (``E``'s share of each pixel of ``P``) spilt through a warm white."""
    from . import fx_light as fx

    over = np.maximum(P - 1.0, 0.0).max(-1)
    share = np.clip(fx._lum(E) / np.maximum(fx._lum(P), 1e-6), 0.0, 1.0)
    tint = np.asarray(BLEED_TINT, np.float32)
    P += (BLEED * over * share)[..., None] * (tint / float(fx._lum(tint)))


def shoulder(P: np.ndarray) -> None:
    """In place, linear RGB: over SHOULDER each channel rolls off towards 1 (tanh), so a lit palm goes orange, then
    gold, then white where it is brightest, as on film."""
    np.copyto(P, SHOULDER + (1.0 - SHOULDER) * np.tanh((P - SHOULDER) / (1.0 - SHOULDER)), where=P > SHOULDER)


LOUD_SLACK_DB = 6.0  # performing again, a take within this many dB of the one the render was made for plays it full
PUFF_LOOK = (0.7, 0.5)  # performing again, a lone "pf" plays a render at most this bright and this big: a pop, however
# loud (a sample "pf." sat inside the slack of the BOOM it was played on, and threw the whole fireball)
SUSTAIN_XFADE = 6  # frames a held flame's loop, and its letting go, cross-fade over
LETGO = 10  # frames a held flame dies down over when its render burns to the end (it vanished in a frame)


def hit_looks(loud: list[float], ref: float | None = None) -> list[tuple[float, float]]:
    """(gain, size) for each hit from its level on the cue's envelope (54 dB over 0..1): the loudest plays the element
    as rendered, a quieter one dimmer and smaller, by its dB under the loudest (LOUD_DB, HIT_SIZE). ``ref``: performing
    again, the loudest hit of the take the element was rendered for (the envelope is absolute dBFS): a new take more
    than LOUD_SLACK_DB quieter plays it smaller (a lone quiet "pf" on a BOOM's render is a pop, not the BOOM again), a
    louder one no bigger than rendered."""
    top = max(loud, default=0.0)
    if ref is not None:
        top = max(top, ref - LOUD_SLACK_DB / 54.0)
    under = [54.0 * (top - v) for v in loud]
    return [(float(np.clip(10 ** (-d / LOUD_DB[0]), 0.35, 1.0)), float(np.clip(10 ** (-d / LOUD_DB[1]), *HIT_SIZE)))
            for d in under]


def element_span(element: np.ndarray, floor: float, ign: int) -> tuple[int, int]:
    """(release, end) of a held element (a jet, a plume): the frame after its last one still burning at full (it lets
    go from there) and the frame after its last lit one."""
    lit = np.array([float(np.clip(f[::8, ::8].max(-1).astype(np.float32) - floor - 2, 0, None).mean()) for f in element])
    body = lit[ign:]
    top = float(np.percentile(body, 75)) if body.size else 0.0
    if top <= 0:
        return len(element), len(element)
    on, seen = np.nonzero(body >= 0.6 * top)[0], np.nonzero(body > 0.05 * top)[0]
    rel = ign + int(on[-1]) + 1 if on.size else len(element)
    return rel, max(rel, ign + int(seen[-1]) + 1 if seen.size else len(element))


GROWN_SHARE, GROW_FAST = 0.6, 6  # a held render played again is grown once it burns at this share of its steady body,
# and gets there in at most this many frames: a palm's flame that took 1.4 s to catch in its own take was still a
# candle when a short new fffff ended


def grown_frame(element: np.ndarray, floor: float, ign: int, rel: int) -> int:
    """The first frame from the ignition where a held render burns at GROWN_SHARE of its steady body (element_span's
    measure); the ignition itself when it never gets there."""
    lit = np.array([float(np.clip(f[::8, ::8].max(-1).astype(np.float32) - floor - 2, 0, None).mean())
                    for f in element[ign:rel]])
    if not lit.size:
        return ign
    top = float(np.percentile(lit, 75))
    on = np.nonzero(lit >= GROWN_SHARE * top)[0] if top > 0 else []
    return ign + int(on[0]) if len(on) else ign


def held_frames(d: int, ign: int, rel: int, end: int, hold: int, x: int = SUSTAIN_XFADE,
                grown: int | None = None) -> list[tuple[int, float]]:
    """The render's frames, with their weights, that a held hit plays ``d`` frames after its ignition, for a voice
    that holds ``hold`` frames: its body as rendered; looped while the voice holds on past it (the loop's end
    cross-faded into its start); and from wherever the voice stops, its letting go ([] once that has burned out).
    A take as long as the render's plays it as rendered. ``grown`` (grown_frame): a render slower than GROW_FAST to
    catch runs through its catching in GROW_FAST frames, then on from there."""
    body = rel - ign
    stop = max(int(hold), 1)
    g = 0 if grown is None else int(np.clip(grown - ign, 0, max(body - 1, 0)))
    ahead = g - GROW_FAST if g > GROW_FAST else 0  # the frames of its catching skipped

    def at(k: int) -> int:
        """Body frame k of the voice's hold, on the render's own time."""
        return k if not ahead or k < 0 else (k * g) // GROW_FAST if k < GROW_FAST else k + ahead

    looping = at(stop - 1) + 1 > body and body > 2 * x + 2  # (too short a body to loop: it holds its last frame)

    def burning(k: int) -> list[tuple[int, float]]:
        """Body frame k. Looping, the steady burn after the catching runs from A = ign + lead to rel and round again:
        the last x frames before each wrap fade into the x frames before A, so the wrap lands on A from A - 1 (the
        first wrap too: it had jumped from the body's last frame to A unblended)."""
        if not looping or k < body - x:
            return [(ign + min(k, body - 1), 1.0)]
        lead = min(x, body // 3)
        span = body - lead
        m = (k - lead) % span
        if m >= span - x:
            w = (m - (span - x) + 1) / (x + 1)
            return [(ign + lead + m, 1.0 - w), (max(ign, ign + lead + m - span), w)]
        return [(ign + lead + m, 1.0)]

    if d < 0:
        return [(ign + d, 1.0)]
    if d < stop:
        return burning(at(d))
    r = d - stop
    if end - rel <= x:  # a render that burns to its last frame has no letting go: it dies down from where it was
        return [] if r >= LETGO else [(f, wt * (1.0 - r / LETGO) ** 2) for f, wt in burning(at(stop - 1 + r))]
        # (on a curve: faded in a straight line, its last frames were a translucent yellow-green sheet over the palm)
    if at(stop - 1) + 1 == body:  # as long as rendered: its own letting go, uncut
        return [(rel + r, 1.0)] if rel + r < end else []
    if rel + r >= end:
        return []
    if r >= x:
        return [(rel + r, 1.0)]
    w = (r + 1) / (x + 1)
    return [(f, wt * (1.0 - w)) for f, wt in burning(at(stop - 1))] + [(rel + r, w)]


def hold_of(frame: int, sounding: list[tuple[int, int]]) -> int | None:
    """Frames the voice holds on after a hit's strike at ``frame``: to the end of the sounding stretch it starts (None
    when it starts none)."""
    for a, b in sounding:
        if a - 3 <= frame < b:
            return max(1, b - frame)
    return None


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


def thrown_pose(ref: tuple, track: dict, since: int, t: int, size: float = 1.0,
                cam: np.ndarray | None = None) -> np.ndarray:
    """2x3: where a fireball thrown on the strike (frame ``since``) is on plate frame ``t``. It leaves the fist
    RELEASE_AFTER frames after the strike, in the pose the fist gave it then, and flies on along the strike's direction,
    slowing, and rises (RISE); the camera carries it with the room since. Carried on the fist after the strike, it
    was a ball stuck to a glove, pulled back with the arm."""
    k = since + RELEASE_AFTER
    A = pose_forward(ref, track, k, size)
    _, u, s = emitter_at(track, since)
    age = max(0.0, t - k)
    A[:, 2] += (np.asarray(u, np.float64) * THROW * (1.0 - np.exp(-age / THROW_FRAMES)) + (0.0, -RISE * age)) * s
    return A if cam is None else (camera_between(cam, k, t) @ np.vstack([A, [0.0, 0.0, 1.0]]))[:2]


def warp_element(frame: np.ndarray, track: dict, ref: tuple, t: int, lag: int, size: float = 1.0,
                 blur: bool = True, since: int | None = None, cam: np.ndarray | None = None,
                 thrown: bool = False) -> np.ndarray:
    """An element frame (float32, over black, at the plate's size, rendered with the emitter at ``ref``) carried onto
    plate frame ``t`` with its emission history: in thin slices along its length, its base on the emitter's pose now and
    each slice further along on the pose the emitter had when that part left the hand, up to ``lag`` frames before for
    the far end (never before ``since``, the strike). Fire already thrown keeps flying the way it was thrown while the
    hand swings on: one pose for the whole element made it a strip swung on a pivot, and one remap with a pose per pixel
    tore it apart on a fast turn. The slices are a quarter of a frame of history apart (half without ``blur``), each
    warped only where it is, their weights overlapping and summing to 1, and each smeared over its own share of the
    history (the shutter's own smear) in as many steps as it takes for its far edge to move at most SMEAR_PX: a stream
    thrown on a fast turn is stretched along its path, not broken into pieces. ``cam`` (camera_path): what was thrown
    then has moved with the room since, as the camera saw it. ``thrown`` (a fireball): once it has left the fist
    (thrown_pose), the whole element on that one pose, the fist no longer moving it."""
    h, w = frame.shape[:2]
    if thrown and since is not None and t > since + RELEASE_AFTER:
        return cv2.warpAffine(frame, thrown_pose(ref, track, since, t, size, cam).astype(np.float32), (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
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


def root_fade(h: int, w: int, ref: tuple, length: float, rounded: float = ROOT_ROUND) -> np.ndarray:
    """(h, w) float32 in the element's own frame: 0 at and behind the emitter, rising to 1 ROOT_FADE of the reach along
    it, so the element comes up from nothing just past the fingertips. The edge curves back at the sides (ROOT_ROUND):
    a straight one cut a wide fireball's base flat along a line."""
    p0, u0, s0 = ref
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    along = (xx - p0[0]) * u0[0] + (yy - p0[1]) * u0[1]
    across = (xx - p0[0]) * u0[1] - (yy - p0[1]) * u0[0]
    length = max(length, 1.0)
    return np.clip((along - rounded * across * across / length) / length, 0, 1).astype(np.float32)


def source_profile(value: dict | None = None) -> dict[str, float]:
    """Validate optional authored source calibration; absent fields preserve the original playback."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - {"scale", "exposure"}:
        raise ValueError("source_profile must be an object with only scale and exposure")
    profile = {}
    for key, low, high in (("scale", .25, 2.), ("exposure", .25, 4.)):
        v = value.get(key, 1.)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not low <= v <= high or not math.isfinite(v):
            raise ValueError(f"source_profile.{key} must be a finite number between {low} and {high}")
        profile[key] = float(v)
    return profile


def library_size(behaviour: str, s_ref: float, short_ref: int, s_hand: float, short: int) -> float:
    """How much bigger to play an element made ahead for another hand (the library's: a hand ``s_ref`` px in a frame
    ``short_ref`` px on its short side) than its hand-for-hand scale, so it is as big on this hand (``s_hand`` px, in a
    frame ``short`` px) as a render for it would be: each was painted SIZE's reach of its own hand, capped at a share of
    its frame. The library's close-up hands were capped at 0.7 of a hand, and carried hand for hand onto a smaller one
    a fireball was a pilot light (a render for that hand reaches 1.8)."""
    hands, most = SIZE.get(behaviour, (LENGTH, 0.6))
    made = min(hands * s_ref, most * short_ref)
    wanted = min(hands * s_hand, most * short)
    return float(wanted * s_ref / max(made * s_hand, 1e-6))


_FOOT_NOISE: dict[int, np.ndarray] = {}


def foot_fade(box: tuple[int, int, int, int], p, u, s: float, t: int, seed: int = 0) -> np.ndarray:
    """(y1 - y0, x1 - x0) float32 over ``box`` (y0, y1, x0, x1): a flame's foot coming up out of the palm, 0 at the
    emitter ``p`` and 1 FOOT hands (``s`` px) along ``u``, the line between broken by smooth noise that moves with the
    hand and rises with the flame on frame ``t``."""
    if seed not in _FOOT_NOISE:  # a 64x64 tile of smooth noise that wraps round, unit spread
        f = np.fft.fft2(np.random.default_rng(seed).standard_normal((64, 64)))
        k = np.fft.fftfreq(64)
        n = np.real(np.fft.ifft2(f * np.exp(-0.5 * (k[:, None] ** 2 + k[None, :] ** 2) / 0.06 ** 2))).astype(np.float32)
        _FOOT_NOISE[seed] = n / max(float(n.std()), 1e-6)
    y0, y1, x0, x1 = box
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - float(p[0]), yy - float(p[1])
    along = dx * float(u[0]) + dy * float(u[1])
    across = dx * float(u[1]) - dy * float(u[0])
    cell = max(0.05 * s, 1.0)  # (a tile cell is this many px: the wobble's eddies about a seventh of a hand)
    nz = cv2.remap(_FOOT_NOISE[seed], np.mod(across / cell, 64.0).astype(np.float32),
                   np.mod((along - FOOT_RISE * s * t) / cell, 64.0).astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_WRAP)
    length = max(FOOT * s, 1.0)
    r = np.clip((along + FOOT_WOBBLE * length * nz * np.clip(2.0 * along / length, 0.0, 1.0)) / length, 0.0, 1.0)
    # (the noise comes in over the foot's first half: nothing shows at or below the emitter)
    return r * r * (3.0 - 2.0 * r)


def drawn_extent(frames, ref: tuple) -> tuple[float, float, float]:
    """What a render draws, in hands of its emitter's pose ``ref``, at its fullest (the 90th percentile over ``frames``,
    each linear over black at the plate's size): (its height along its direction, its lit area's equivalent diameter,
    its far reach along its direction from the emitter). Frames with next to nothing lit are left out."""
    p0, u0, s0 = ref
    s0 = max(float(s0), 1e-6)
    hgt, diam, far = [], [], []
    for E in frames:
        L = 0.2126 * E[::2, ::2, 0] + 0.7152 * E[::2, ::2, 1] + 0.0722 * E[::2, ::2, 2]
        ys, xs = np.nonzero(L > FIT_LIT)
        if ys.size < 50:
            continue
        along = (2.0 * xs - p0[0]) * u0[0] + (2.0 * ys - p0[1]) * u0[1]
        hgt.append(float(np.percentile(along, 98) - np.percentile(along, 2)) / s0)
        diam.append(float(np.sqrt(16.0 * ys.size / np.pi)) / s0)  # (each lit sample stands for 4 pixels)
        far.append(float(np.percentile(along, 98)) / s0)
    if not hgt:
        return 0.0, 0.0, 0.0
    return float(np.percentile(hgt, 90)), float(np.percentile(diam, 90)), float(max(far))


def room_along(p, u, w: int, h: int, margin: float) -> float:
    """px from ``p`` along ``u`` to the frame's edge, ``margin`` of the frame inside it (0 when already past it)."""
    ts = []
    for i, size in enumerate((w, h)):
        if abs(u[i]) > 1e-6:
            lim = size * (1.0 - margin) if u[i] > 0 else size * margin
            ts.append((lim - p[i]) / u[i])
    return max(0.0, min(ts)) if ts else float("inf")


def fit_play(behaviour: str, drawn: tuple[float, float, float], track: dict, k: int, w: int, h: int,
             least: float = 1.0) -> tuple[float, dict]:
    """How many times bigger than hand for hand to play a flame or a fireball (in space only: each hit's own size comes
    on top), so that at its fullest it is FIT's size on this hand: what the render draws (drawn_extent) brought to it,
    within the frame's room (FIT_ROOM) for the flame's top or the thrown fireball's far edge, never smaller than
    ``least`` nor more than FIT_MOST. Returns (that factor, how it was fitted: the receipt's)."""
    hgt, diam, far = drawn
    span = range(k, min(len(track["xy"]), k + 48))
    s = float(np.median([float(track["scale"][t]) for t in span]))
    if behaviour == "plume":
        key, have, target = "plume", hgt, FIT["plume"]
        y = float(np.median([float(track["xy"][t][1]) for t in span]))
        room = (y - FIT_ROOM[0] * h) / max(s, 1e-6) / max(hgt, 1e-6)
    else:
        key = "burst_tip" if gesture_at(track, k) in POINTING else "burst"
        have, target = diam, FIT_TIP if key == "burst_tip" else FIT["burst"]
        p, u, s_k = emitter_at(track, k)
        room = (room_along(p, u, w, h, FIT_ROOM[1]) / max(s_k, 1e-6) - 0.5 * THROW) / max(far, 1e-6)  # (at its
        # biggest it has flown about half its way)
    play = float(min(FIT_MOST, max(least, min(target / max(have, 1e-6), room))))
    return play, {"as": key, "drawn": round(have, 2), "target": target, "room": round(float(min(room, 99.0)), 2),
                  "play": round(play, 2)}


def library_fit(element: np.ndarray, floor: float, ign: int, ref: tuple, behaviour: str) -> dict | None:
    """The live preview's sizes for an element rendered ahead (fit_play without the frame's room, which it can't
    know): how many times bigger than hand for hand it plays from a palm, and (a fireball) from the fingertips. None
    for an element FIT doesn't size."""
    from .transfer import lin

    if behaviour not in FIT:
        return None
    fl = float(lin(np.full((1, 1, 3), int(floor), np.uint8))[0, 0, 0])
    hgt, diam, _ = drawn_extent((np.clip(lin(element[e]) - fl, 0.0, None) for e in range(ign, len(element), 2)), ref)
    have = max(hgt if behaviour == "plume" else diam, 1e-6)
    return {"palm": round(float(min(FIT_MOST, max(1.0, FIT[behaviour] / have))), 3),
            "tip": round(float(min(FIT_MOST, max(1.0, (FIT_TIP if behaviour == "burst" else FIT[behaviour]) / have))), 3)}


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


def motion_blur(E: np.ndarray, step: np.ndarray) -> np.ndarray:
    """``E`` smeared along ``step`` (px the hand moved since the last frame) for SHUTTER of it, as the camera smeared the
    hand: a crisp flame on a blurred hand read as pasted on. Only where the element is; still hands leave it as it is."""
    length = SHUTTER * float(np.hypot(*step))
    if length < 1.5:
        return E
    look = E[::4, ::4].max(-1) > 1e-4
    ys, xs = np.nonzero(look)
    if ys.size == 0:
        return E
    pad = int(length) + 4
    h, w = E.shape[:2]
    y0, y1 = max(0, 4 * ys.min() - pad), min(h, 4 * ys.max() + pad)
    x0, x1 = max(0, 4 * xs.min() - pad), min(w, 4 * xs.max() + pad)
    k = int(np.ceil(length)) | 1
    kernel = np.zeros((k, k), np.float32)
    c, u = k // 2, np.asarray(step, np.float64) / max(float(np.hypot(*step)), 1e-6)
    for s in np.linspace(-length / 2, length / 2, 2 * k):
        kernel[int(round(c + s * u[1])), int(round(c + s * u[0]))] += 1.0
    out = E.copy()
    out[y0:y1, x0:x1] = cv2.filter2D(E[y0:y1, x0:x1], -1, kernel / kernel.sum(), borderType=cv2.BORDER_CONSTANT)
    return out


def room_behind(E: np.ndarray, P: np.ndarray) -> float:
    """How bright the room is behind a flame (linear luminance): the plate's mean under it, weighted by the flame."""
    from . import fx_light as fx

    on = np.clip((fx._lum(E) - SOOT_FROM) / 0.35, 0.0, 1.0)
    cover = float(on.sum())
    return float((fx._lum(P) * on).sum()) / cover if cover > 0 else 0.0


def matte_box(m: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): where the hand matte (uint8 or float) is, from a quarter-size look; empty when it is nowhere."""
    ys, xs = np.nonzero(m[::4, ::4] > (8 if m.dtype == np.uint8 else 0.03))
    if ys.size == 0:
        return 0, 0, 0, 0
    h, w = m.shape[:2]
    return max(0, 4 * ys.min() - 4), min(h, 4 * ys.max() + 8), max(0, 4 * xs.min() - 4), min(w, 4 * xs.max() + 8)


def root_holdout(track: dict, t: int, h: int, w: int, mask: np.ndarray | None,
                 box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Where the hand hides the element (float 0..1): its matte, wherever it is behind the emitter; without one, within
    a quarter of a hand size of the emitter, the side behind the fingertips, coming in over 0.3 of a hand size so the
    base fades into the hand instead of being cut off along a line. The flame leaves the fingers, not sits on them.
    With ``box`` (holdout_box) only that part of the frame, as a patch (``mask`` is the whole frame's either way)."""
    p, u, s = emitter_at(track, t)
    y0, y1, x0, x1 = box or (0, h, 0, w)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    mask = None if mask is None else mask[y0:y1, x0:x1]
    dx, dy = xx - p[0], yy - p[1]
    disc = np.clip(1.0 - (np.hypot(dx, dy) - 0.25 * s) / max(0.25 * s, 1.0), 0, 1)
    back = -(dx * u[0] + dy * u[1])  # how far behind the emitter, along its direction
    if mask is not None:  # the hand and arm wherever they are behind the emitter, nearer the camera than the fire it
        # threw: the fingers in front of a flame's root hide it, the palm the flame rises in front of doesn't (the
        # whole matte put fire in a cupped hand behind it), and a first-person arm hides the flame swung across it
        return mask * np.clip(0.5 + back / max(0.3 * s, 1.0), 0, 1)
    return disc * np.clip(back / max(0.3 * s, 1.0), 0, 1)


FIRE_VOLUME = dict(slabs=9, thickness=.9, parallax=.60, flow=.040, optical_depth=.85, tau_cap=1.25,
                   density_contrast=.34, foreground_cooling=.34, filament_relief=.32, energy=.95)


def fire_landmarks(track: dict, t: int) -> np.ndarray | None:
    """A measured 3D hand for this frame; do not invent depth across tracking gaps."""
    frames = track.get("landmarks") or []
    if t >= len(frames) or frames[t] is None:
        return None
    lm = np.asarray(frames[t], np.float32)
    sources = track.get("src") or []
    if lm.shape != (21, 3) or not np.isfinite(lm).all() or (t < len(sources) and sources[t] == "rtm"):
        return None
    return lm


def palm_views(track: dict, reference: int) -> np.ndarray:
    """Small relative view changes from the palm normal, zero on missing-depth frames."""
    n = len(track.get("landmarks") or track.get("xy", []))
    angles = np.full(n, np.nan, np.float32)
    for t in range(n):
        lm = fire_landmarks(track, t)
        if lm is None:
            continue
        normal = np.cross(lm[5] - lm[0], lm[17] - lm[0])
        if np.linalg.norm(normal) < 1e-5:
            continue
        if normal[2] > 0:
            normal *= -1
        angles[t] = np.arctan2(normal[0], -normal[2])
    known = np.flatnonzero(np.isfinite(angles))
    result = np.zeros(n, np.float32)
    if not len(known):
        return result
    # Smooth only uninterrupted measured spans: a missing frame remains a true fallback.
    for span in np.split(known, np.flatnonzero(np.diff(known) > 1) + 1):
        values = np.unwrap(angles[span])
        angles[span] = np.convolve(np.pad(values, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    baseline = angles[known[np.argmin(abs(known - reference))]]
    result[known] = np.clip(.07 + .4 * (angles[known] - baseline), -.17, .17)
    return result


def hand_motion_px(track: dict, t: int) -> float:
    """Screen-space shutter motion, robust to one mistracked fingertip.

    Camera motion belongs here: the photograph blurs from both camera and hand movement.
    """
    if t <= 0:
        return 0.
    frames = track.get("landmarks") or []
    if t < len(frames) and frames[t] is not None and frames[t - 1] is not None:
        now, before = np.asarray(frames[t], np.float32), np.asarray(frames[t - 1], np.float32)
        # A 2D fallback can measure photographic motion without supplying fictitious surface depth.
        if (now.shape in ((21, 2), (21, 3)) and before.shape in ((21, 2), (21, 3))
                and np.isfinite(now[:, :2]).all() and np.isfinite(before[:, :2]).all()):
            return float(np.median(np.linalg.norm(now[:, :2] - before[:, :2], axis=1)))
    return float(np.linalg.norm(emitter_at(track, t)[0] - emitter_at(track, t - 1)[0]))


def fire_visibility(track: dict, t: int, matte: np.ndarray | None, geo: dict | None, z0: float,
                    contact: bool = True, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Camera visibility of the approximate root surface, in plate pixel coordinates.

    Smaller landmark z is nearer the camera. This is a 2.5D hand proxy, not recovered scene depth.
    With ``box=(y0,y1,x0,x1)``, return only that patch, evaluating the same surface on its overlap.
    """
    h, w = matte.shape if matte is not None else (int(track["H"]), int(track["W"]))
    by0, by1, bx0, bx1 = box or (0, h, 0, w)
    visibility = np.ones((by1-by0, bx1-bx0), np.float32)
    y0, y1, x0, x1 = matte_box(matte) if matte is not None else holdout_box(track, t, h, w)
    y0, y1, x0, x1 = max(y0, by0), min(y1, by1), max(x0, bx0), min(x1, bx1)
    if y1 <= y0 or x1 <= x0:
        return visibility
    patch = (slice(y0-by0, y1-by0), slice(x0-bx0, x1-bx0))
    if geo is None:
        visibility[patch] -= root_holdout(track, t, h, w, matte, (y0, y1, x0, x1))
        return visibility
    p, u, s = emitter_at(track, t)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    along = (xx - p[0]) * u[0] + (yy - p[1]) * u[1]
    fire_z = z0 - .055 * s - .24 * np.maximum(along, 0.)
    hide = geo["valid"][y0:y1, x0:x1] * _smooth(-.07*s, .07*s, fire_z - geo["depth"][y0:y1, x0:x1])
    bury = matte[y0:y1, x0:x1] * _smooth(-.015*s, .045*s, -along)
    visible = 1. - np.maximum(hide, bury)
    if contact:
        # Only the contact surface softens the root; no full-frame half-plane clips a thrown fireball.
        visible *= 1. - matte[y0:y1, x0:x1] * (1. - _smooth(-.015*s, .10*s, along))
    visibility[patch] = visible
    return visibility


def _fire_over_floor(plate: np.ndarray, emission: np.ndarray,
                     opacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optical over and its permitted background luminance floor, on the supplied patch."""
    lum, plate_lum = fx._lum(emission), fx._lum(plate)
    alpha = np.clip(opacity, 0., 1.)
    body = _smooth(.04, .28, lum)*_smooth(.12, .45, alpha)
    floor = plate_lum*(1.-.28*body)
    np.minimum(alpha, (.98*lum+plate_lum-floor)/np.maximum(plate_lum, 1e-5), out=alpha)
    result = plate*(1.-alpha[..., None])
    result += emission
    return result, floor


def fire_over(plate: np.ndarray, emission: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """Dense gas can attenuate a wall by at most 28%; faint edges cannot darken it."""
    return _fire_over_floor(plate, emission, opacity)[0]


def finish_fire(plate: np.ndarray, emission: np.ndarray, opacity: np.ndarray,
                out: np.ndarray | None = None) -> np.ndarray:
    """Optical over and bloom with a bounded, plate-anchored camera response in each channel.

    Compress the whole luminous result, not a mixture with an unbounded linear result: the latter clipped a
    broad white core after display conversion. Separate channel shoulders let hot filaments approach warm
    yellow above their orange body. Each knee stays above its photographed channel; a calibrated luminance
    floor permits depth in dense gas while preventing faint edges from cutting a dark rim into a wall.
    Quarter-resolution bloom retains its original full-canvas sampling grid. On the usual multiple-of-four
    plate dimensions, resizing a grid-aligned crop is exactly the same interpolation as resizing the canvas.
    """
    h, w = plate.shape[:2]
    if out is None:
        out = plate.copy()
    elif out is not plate:
        out[...] = plate
    low = cv2.resize(emission, (max(1, w//4), max(1, h//4)), interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(low, (0, 0), max(.0035*min(h, w), .5))
    x, y, bw, bh = cv2.boundingRect(np.any(low != 0., axis=-1).astype(np.uint8))
    if bw == 0 or bh == 0:
        return out
    if h % 4 == 0 and w % 4 == 0:
        lx0, ly0 = max(0, x-1), max(0, y-1)
        lx1, ly1 = min(low.shape[1], x+bw+1), min(low.shape[0], y+bh+1)
        x0, y0, x1, y1 = 4*lx0, 4*ly0, 4*lx1, 4*ly1
        glow = cv2.resize(low[ly0:ly1, lx0:lx1], (x1-x0, y1-y0))
    else:
        x0, y0, x1, y1 = 0, 0, w, h
        glow = cv2.resize(low, (w, h))
    sl = (slice(y0, y1), slice(x0, x1))
    room, E = plate[sl], emission[sl]
    result, floor = _fire_over_floor(room, E, opacity[sl])
    result += glow*.035
    knee = np.maximum(.78, room)
    ceiling = np.maximum(1., room)
    span = np.maximum(ceiling-knee, 1e-6)
    mapped = np.minimum(ceiling, knee+span*(-np.expm1(-np.maximum(result-knee, 0.)/span)))
    np.copyto(result, mapped, where=result > knee)
    need = np.maximum(floor-fx._lum(result), 0.)
    headroom = np.maximum(ceiling-result, 0.)
    fraction = np.minimum(need/np.maximum(fx._lum(headroom), 1e-6), 1.)
    result += headroom*fraction[..., None]
    out[sl] = result
    return out


def blend_element_frames(source, picks: list[tuple[int, float]], boxes: dict,
                         shape: tuple[int, int]) -> np.ndarray:
    """Blend a held source in the original weight order, without multiplying its black canvas."""
    if len(picks) == 1 and picks[0][1] == 1.:
        return source(picks[0][0])
    out = np.zeros((*shape, 3), np.float32)
    for k, weight in picks:
        frame = source(k)  # populates its cached lit box on first access
        if boxes[k] is not None:
            y0, y1, x0, x1 = boxes[k]
            out[y0:y1, x0:x1] += weight*frame[y0:y1, x0:x1]
    return out


def left_frame(track: dict, w: int, h: int) -> tuple[dict, np.ndarray]:
    """(track, gone): where the emitter has gone more than half a hand past the frame's edge, the hand has left the
    shot and the trackers only guess at it (their guesses swung a pointing hand round as it went out, and the fire
    with it): its pose is held where it went out, and ``gone`` counts the frames since (0 while it is in)."""
    n = len(track["xy"])
    xy = [list(map(float, v)) for v in track["xy"]]
    d = [list(map(float, v)) for v in track["dir"]]
    sc = [float(v) for v in track["scale"]]
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
    on (default: the strike); ``aim``: the aim arrow drawn there, degrees clockwise from up. Without an arrow,
    plumes and non-pointing fireballs rise upward; directional jets keep the tracked hand's aim.
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
    auto = None
    old = json.loads((Path(element_from) / "receipt.json").read_text()) if element_from is not None else None
    profile = source_profile(old.get("source_profile") if old is not None else None)
    # performing your own take again: as big as it is loud against the take it was rendered for (not a library's)
    loud_ref = old["bending"].get("loud_ref") if old is not None and not instant else None
    looks = hit_looks(loud, loud_ref)
    if old is not None and len(hits) == 1 and not charge and heard(cue, c.samples, SR, c.source)["kind"] == "puff":
        looks = [(min(g_, PUFF_LOOK[0]), min(s_, PUFF_LOOK[1])) for g_, s_ in looks]
    effect = old["effect"]["id"] if old else spec.effect
    if effect == "auto":  # the sound and the hand's gesture at the strike
        h_ = heard(cue, c.samples, SR, c.source)
        g = gesture_at(track, k_launch)
        span = (max(hits) - min(hits)) / FPS if len(hits) > 1 else 0.0
        snapped = any(e.get("kind") == "snap" and abs(int(e["frame"]) - hero) <= 4 for e in track.get("events", []))
        effect, why = bending_effect(h_["kind"], g, hits=len(hits), span_s=span, snap=snapped,
                                     longest_s=h_.get("longest_s", 0.0))
        effect = effect or "fire_burst"
        auto = h_ | {"heard": h_["kind"], "hand": g, "picked": effect, "why": why}
    arrow = aim if aim is not None else spec.direction
    aim_mode = "explicit" if arrow is not None else "tracked"
    if arrow is None and (effect == "flame_plume" or
                          (effect == "fire_burst" and gesture_at(track, k_launch) not in POINTING)):
        # A side-on palm can project its wrist-to-knuckle axis downward. That axis
        # carries the tapped point, but does not give a rising flame its direction.
        arrow, aim_mode = 0.0, "upward_default"
    if effect == "earth_rock" or effect.startswith("water_"):
        from .motion import bridge_short_gaps
        track = bridge_short_gaps(track, plate, mask)
    track = anchored(track, tap_k, (spec.point[0] * w, spec.point[1] * h), arrow)
    track["aim_mode"] = aim_mode
    if aim_mode == "upward_default":
        track["dir"] = [[0.0, -1.0]] * len(track["dir"])
    if effect in MADE:  # lightning, water, air, an explosion: made here, not rendered over black
        return made_shot(spec, out_dir, effect, plate, track, mask, c, env_all, snaps, looks, k_launch, tap_k, arrow,
                         auto or picked, old, element_from, instant, track_path, mask_path, say, t0)
    eff = load_effect(effect)
    lib_k = 1.0  # an element made ahead for another hand, played as big as a render for this one (library_size)
    if old:  # perform it again: the element as rendered for the earlier take, where its hand was then
        ref = (np.asarray(old["bending"]["ref"]["xy"], np.float64), np.asarray(old["bending"]["ref"]["dir"], np.float64),
               float(old["bending"]["ref"]["scale"]))
        aim, prompt, scale, seconds = angle_of(ref[1]), old["render"]["prompt"], old["render"]["guide_scale"], 0.0
        element = read_video(Path(element_from) / "element.mp4").frames
        if instant:  # (only how big: a hit's own size also sets how long a burst lasts, and as that a library
            # fireball played three times as slow)
            lib_k = library_size(eff.behaviour, ref[2], min(element.shape[1:3]), emitter_at(track, k_launch)[2], min(w, h))
    else:
        say("painting", 0.12, "Painting the element where your hand is on the strike")
        p_hit, u_hit, s_hit = emitter_at(track, k_launch)
        hands, most = SIZE.get(eff.behaviour, (LENGTH, 0.6))
        reach = min(hands * s_hit, most * min(w, h))
        # a burst or a flame is rendered on a landscape canvas whatever the clip's shape (on a portrait one the model drew
        # a narrow fountain of sparks for a fireball, seed after seed, and a thin tan wisp for a palm's flame), in its
        # middle, and placed on the hand by its pose (fit_element)
        rw, rh = (max(w, h), min(w, h)) if eff.behaviour in ("burst", "plume") else (w, h)
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
    floor = estimate_black_level(element)  # (on the render as made: fitted to a portrait plate, its window's padding is
    ign = ignition_frame(element, floor, burst=eff.behaviour == "burst")  # black 0, and a black read off that border
    # put a punch's blast half a second after its BOOM and a palm's flame on the render's dim first frames)
    if element.shape[1:3] != (h, w):  # (a render of another size: placed by its pose, never stretched)
        element, ref = fit_element(element, ref, h, w)
    if ign is None:
        raise ValueError("the render shows no element to carry")
    # each hit fires the element: its ignition on that hit's strike, as big and as bright as the hit is loud
    plays = [(s_["frame"] - ign, g, sz) for s_, (g, sz) in zip(snaps, looks)]
    every = plays  # (the Foley sounds on every hit of the take, whichever hand fires it)
    # a render played again (your take again, or one made ahead) holds a jet or a plume as long as this voice holds
    # it, and lets it go when the voice stops: as rendered it lasted as long as the take it was made for
    held = old is not None and eff.behaviour in ("jet", "plume")
    rel = end_ = grown = ign
    holds: list[int] = []
    if held:
        rel, end_ = element_span(element, floor, ign)
        grown = grown_frame(element, floor, ign, rel)
        sounding = cue.intervals()
        holds = [hold_of(s_["frame"], sounding) or (rel - ign) for s_ in snaps]
    if only_hits:
        plays = [plays[i] for i in only_hits if 0 <= i < len(plays)]
        snaps = [snaps[i] for i in only_hits if 0 <= i < len(snaps)]
        holds = [holds[i] for i in only_hits if 0 <= i < len(holds)]
    offset = k_launch - ign

    say("compositing", 0.7, "Carrying the element on your hand")
    from . import fx_light as fx

    lag = LAG_FRAMES.get(effect, 3)
    pre = 0 if eff.behaviour == "burst" else PREGLOW  # a burst is the blast itself: it is there at once (the model's frames
    # before it caught were a grey-white square on the palm)
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
        if src[t] == "lost" and still_there(plate, track, seen_at, t):
            src[t] = "hold"  # the tracker lost it, but nothing moved where it was last seen: a still hand, still there
        run = run + 1 if src[t] == "lost" else 0
        seen_at = t if src[t] not in ("lost", "hold") else seen_at  # (checked on every frame, so against a frame it
        # was seen on: against the last one held, a slow drift would never be more than a frame's change)
        lost[t] = run
        if gone[t]:  # it left the frame: the fire lets go at once, where the hand went out
            lost[t] = max(lost[t], LOST_GRACE + gone[t])

    hands_, most_ = SIZE.get(eff.behaviour, (LENGTH, 0.6))
    reach_px = min(hands_ * ref[2], most_ * min(w, h))  # the element's own reach: its root fades over a share of it (a
    # share of six hands faded a palm-sized fireball out to its top, and it hung above the hand)
    fade = (root_fade(h, w, ref, 1.0, rounded=0.0) if eff.behaviour == "plume"  # (a flame's foot: foot_fade, on the
            # hand; this one's rounded edge, a few px long, cut a steep U into the flame's base on the palm)
            else root_fade(h, w, ref, ROOT_FADE_SHARE.get(effect, ROOT_FADE) * reach_px)) * border_fade(h, w)
    if eff.fluid == "steam":  # vapour thins out as it spreads: dense at the palm, a third as dense at its far end
        p0_, u0_, s0_ = ref
        yy_, xx_ = np.mgrid[:h, :w].astype(np.float32)
        along_ = ((xx_ - p0_[0]) * u0_[0] + (yy_ - p0_[1]) * u0_[1]) / max(0.5 * LENGTH * s0_, 1.0)
        fade = fade * np.clip(1.0 - 0.67 * along_, 0.33, 1.0)
        del yy_, xx_, along_
    fade = fade[..., None]
    cam = camera_path(plate, mask, track)  # a handheld camera: what was thrown moves with the room
    views = palm_views(track, k_launch) if eff.fluid == "fire" else None

    def volume_source(S: np.ndarray, t: int, source_frames: list[int], source_weight: float = 1.) -> np.ndarray:
        if eff.fluid != "fire":
            return S if profile["exposure"] == 1. else S * profile["exposure"]
        # Carry RGB and extinction through exactly the same camera/hand transform.
        boxes = [lit_box[f] for f in source_frames if lit_box[f] is not None]
        bounds = ((min(b[2] for b in boxes), min(b[0] for b in boxes),
                   max(b[3] for b in boxes), max(b[1] for b in boxes)) if boxes else (0, 0, 0, 0))
        E, T, info = volume_element(S, t / FPS, float(views[t]) if t < len(views) else 0., FIRE_VOLUME,
                                     active_bounds=bounds, crop_output=True)
        packed = np.zeros((h, w, 4), np.float32)
        x0, y0, x1, y1 = info["crop"]
        # Calibrate radiance and optical depth together, once per source frame.
        # Relighting receives this same calibrated emission after placement.
        exposure = profile["exposure"]
        packed[y0:y1, x0:x1, :3] = E if exposure == 1. else E * exposure
        # RGB already contains the held-release fade. Relative density reconstruction
        # normalizes that fade away, so carry it explicitly into optical depth only.
        optical_scale = exposure * max(0., min(float(source_weight), 1.))
        packed[y0:y1, x0:x1, 3] = 1. - T if optical_scale == 1. else 1. - np.clip(T, 0., 1.) ** optical_scale
        return packed

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
        gain = (0.45 + 1.0 * en) * float(flicker[i])  # a hot core, brighter than the fire it becomes
        gain *= 1.0 if t < charge[1] else (charge[1] + 3 - t) / 4
        E = warp_element(volume_source(source(k), t, [k]), track, ref, t, 0, size * play, blur=False)
        E *= gain
        return E

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
            if DEFOCUS > 0:  # no phone films as crisply as the render draws
                out_[y0:y1, x0:x1] = cv2.GaussianBlur(out_[y0:y1, x0:x1], (0, 0), DEFOCUS * min(h, w) / 1080)
        return out_

    play, fit = lib_k, None  # how big it plays against hand for hand: a flame or a fireball as big as FIT on this hand
    if eff.fluid == "fire" and eff.behaviour in FIT:  # (a render made for another hand, a library's, may play smaller)
        drawn = drawn_extent((source(e) for e in range(ign, len(element), 2)), ref)
        play, fit = fit_play(eff.behaviour, drawn, track, k_launch, w, h, least=0.3 if instant else 1.0)
    base_play = play
    play *= profile["scale"]
    if fit is not None:
        fit = fit | {"base_play": round(base_play, 3), "source_scale": profile["scale"], "play": round(play, 3)}
    thrown = eff.behaviour == "burst"  # a fireball leaves the fist at the strike (thrown_pose)
    if thrown and old is not None and not instant and "thrown" in old["bending"]:
        # Reperform the saved motion intent; a library's source pose does not decide a new hand's release.
        # An absent legacy field is not an explicit anchored choice.
        thrown = old["bending"]["thrown"] is not None

    faded: dict[int, float] = {}  # plate frame -> its element's strength as played (a held flame dying down: under 1)

    def element_at(t: int) -> np.ndarray | None:
        """The element on plate frame t (linear, over black): every hit's play of it, carried on the hand, its root
        held out; None when none is on (before its build-up, after it ends)."""
        acc, free, strength = None, None, 0.0  # (free: fireballs already thrown, no longer on the hand)
        for k_, (off, gain, size) in enumerate(plays):
            e = t - off
            if held and e >= ign:  # held as long as the voice holds it, then let go
                picks = held_frames(e - ign, ign, rel, end_, holds[k_], grown=grown)
                if not picks or max(f for f, _ in picks) >= len(element):
                    continue
            else:
                if eff.behaviour == "burst" and e > ign:  # a smaller blast is over sooner (a fireball lasts as long as
                    e = ign + int(round((e - ign) / size))  # it is big): a "pf" puffs and is gone, a "BOOM" rolls on
                if not max(0, ign - pre) <= e < len(element):  # (a negative frame would wrap round to the render's end)
                    continue
                picks = [(e, 1.0)]
            source_weight = min(1.0, sum(wt for _, wt in picks))
            strength = max(strength, source_weight)
            grow = min(1.0, GROW_FROM + (1.0 - GROW_FROM) * max(0.0, e - ign + pre) / (pre + GROW_FRAMES))
            if eff.behaviour == "burst":  # (a burst is the blast itself: no growing into it)
                grow = 1.0
            S = blend_element_frames(source, picks, lit_box, (h, w))
            S = volume_source(S, t, [f for f, _ in picks], source_weight)
            E = warp_element(S, track, ref, t, lag, size * grow * play, since=off + ign, cam=cam, thrown=thrown)
            E *= gain
            if thrown and t > off + ign + RELEASE_AFTER:  # in flight: smeared along its path over the shutter (on the
                # fist, the fist's smear; thrown, it was crisp at its fastest)
                a_pose = thrown_pose(ref, track, off + ign, t, size * play, cam)
                b_pose = thrown_pose(ref, track, off + ign, t - 1, size * play, cam)
                E = motion_blur(E, a_pose[:, 2] - b_pose[:, 2])
                free = E if free is None else np.maximum(free, E)
                continue
            if e < ign:  # the build-up, coming up to the ignition: its fire only, orange or white-hot (the model's
                # dim grey haze before the flame catches showed as a grey blob on the palm)
                warm = np.clip(3.0 * (E[..., 0] - E[..., 2]) / np.maximum(E[..., 0], 1e-4) - 1.0, 0.0, 1.0)
                E *= ((e - (ign - pre) + 1) / (pre + 1)
                      * np.maximum(warm, np.clip((fx._lum(E[..., :3]) - 0.35) / 0.3, 0.0, 1.0)))[..., None]
            elif ign < pre and e < pre:  # a render lit from its first frame: bring it up rather than pop it on
                E *= (e + 1) / (pre + 1)
            acc = E if acc is None else np.maximum(acc, E)  # two plays overlapping: the brighter shows, not both added
        if charge and charge[0] <= t < charge[1] + 3:  # the ember the hum charges, lighting the fingers
            ember = ember_at(t)
            if ember is not None:
                acc = ember if acc is None else np.maximum(acc, ember)
                strength = 1.0  # (the ember sets its own strength by its gain)
        if acc is None and free is None:
            return None
        faded[t] = strength
        if src[t] == "before":  # the hand hasn't come in yet
            return None
        if acc is not None and lost[t] > LOST_GRACE:  # the hand is gone: let what it carries go rather than freeze it
            acc *= max(0.0, 1.0 - (lost[t] - LOST_GRACE) / LOST_FADE)  # (a fireball it threw flies on)
        if acc is not None and eff.behaviour == "plume":  # it comes up out of the palm
            ys_, xs_ = np.nonzero(acc[::4, ::4].max(-1) > 1e-4)
            if ys_.size:
                fb = (max(0, 4 * ys_.min() - 4), min(h, 4 * ys_.max() + 8), max(0, 4 * xs_.min() - 4),
                      min(w, 4 * xs_.max() + 8))
                p_, u_, s_ = emitter_at(track, t)
                acc[fb[0]:fb[1], fb[2]:fb[3]] *= foot_fade(fb, p_, u_, s_, t, spec.seed)[..., None]
        m_t = None if mask is None else mask[min(t, len(mask) - 1)]
        y0, y1, x0, x1 = box = holdout_box(track, t, h, w) if m_t is None else matte_box(m_t)
        hold = None
        if eff.fluid != "fire" and y1 > y0 and x1 > x0:
            hold = (1.0 - root_holdout(track, t, h, w, None if m_t is None else m_t.astype(np.float32) / 255.0,
                                       box))[..., None]
        for E in (acc, free):
            if E is not None and hold is not None:
                E[y0:y1, x0:x1] *= hold
        if acc is not None and t > 0:  # the camera's shutter smears the fire as it smears the hand carrying it (not
            # a fireball it has thrown)
            acc = motion_blur(acc, emitter_at(track, t)[0] - emitter_at(track, t - 1)[0])
        return free if acc is None else acc if free is None else np.maximum(acc, free)

    q = 8  # the light (cuesync.fx_light) needs only a small copy of each frame's element
    # the camera's exposure jumps by what the fire adds to the room's light: a stop or more in the dark, nothing to
    # speak of in a sunlit room (where the flash washed the whole frame out)
    if eff.fluid != "fire":
        k_full = int(np.clip(ign + 4, 0, len(element) - 1))
        fire_mean = float(fx._lum(np.clip(lin(element[k_full][::4, ::4]) - (floor_lin[::4, ::4] if np.ndim(floor_lin)
                                                                             else floor_lin), 0, None)).mean())
        flash_k = float(min(1.0, np.log2(1.0 + fire_mean / max(float(fx._lum(lin(plate[n // 2][::4, ::4])).mean()), 1e-4))))
        some = np.stack([lin(plate[i]) for i in np.linspace(0, n - 1, 8).astype(int)])
        white, tint = fx.plate_white(some), fx.plate_tint(some)
        del some
    noise = fx.shadow_noise(plate)
    # Robust clip exposure prevents auto gains from pumping as the emitting hand moves.
    ambient_hand, ambient_room = [], []
    if eff.fluid == "fire":
        for i in np.linspace(0, n - 1, min(n, 12)).astype(int):
            y = fx._lum(lin(plate[i][::8, ::8]))
            p_, _, s_ = emitter_at(track, i)
            yy, xx = np.mgrid[:y.shape[0], :y.shape[1]]
            near = ((8 * xx - p_[0])**2 + (8 * yy - p_[1])**2) < (2.5 * s_)**2
            mm = np.zeros_like(y) if mask is None else mask[i][::8, ::8] / 255.
            for values, keep in ((ambient_hand, (mm > .65) & near & (y > .0015)),
                                 (ambient_room, (mm < .05) & near & (y > .002))):
                if keep.sum() > 20:
                    values.append(float(np.percentile(y[keep], 60)))
    light_cfg = {}
    if ambient_hand:
        light_cfg["hand_ambient"] = float(np.median(ambient_hand))
    if ambient_room:
        light_cfg["room_ambient"] = float(np.median(ambient_room))
    out = np.empty_like(plate)
    k_poster, poster_energy = int(np.clip(k_launch + 8, 0, n - 1)), 0.
    CHUNK = 1 if eff.fluid == "fire" else 12  # depth and lighting are per-frame; never cache a chunk of hand surfaces
    made: dict[int, np.ndarray | None] = {}  # each frame's element, carried once (it was carried twice: small, then full)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        a0, b1 = (a, b) if eff.fluid == "fire" else (max(0, a - 1), min(n, b + 1))
        for i in range(a0, b1):
            if i not in made:
                made[i] = element_at(i)
        if eff.fluid == "fire":
            E0 = made[a]
            x0, y0, bw, bh = ((0, 0, 0, 0) if E0 is None else
                              cv2.boundingRect(cv2.bitwise_not(cv2.inRange(E0, (0.,)*4, (0.,)*4))))
            if bw == 0 or bh == 0:
                out[a] = plate[a]  # lin/gam is identity for the uint8 plate; no fire means no work or change
                made.clear()
                say("compositing", 0.7 + 0.14*b/n, f"Carrying the element on your hand: frame {b} of {n}")
                continue
            fire_box = (y0, y0+bh, x0, x0+bw)
            P = lin(plate[a])[None]
            small = None
        else:
            P = np.stack([lin(plate[i]) for i in range(a0, b1)])
            small = np.zeros((b1 - a0, h // q, w // q, 3), np.float32)
            for i in range(a0, b1):
                if made[i] is not None:
                    small[i - a0] = cv2.resize(made[i], (w // q, h // q), interpolation=cv2.INTER_AREA)
        m = None if mask is None else mask[a0:b1].astype(np.float32) / 255.0
        if eff.fluid != "fire":
            fx.light_plate(P, small, track_slice(track, a0, b1), m, fluid=eff.fluid, white=white, tint=tint, out=P)
        for i in range(a, b):
            if made[i] is None:
                continue
            E0 = made[i]  # as rendered: its light on the hand was worked out from this
            E = E0
            j = i - a0
            if eff.fluid == "fire":
                p_, u_, s_ = emitter_at(track, i)
                lm = fire_landmarks(track, i)
                matte = None if m is None else m[j]
                geo = hand_surface.surface(lm, matte, s_) if lm is not None and matte is not None else None
                z0 = hand_surface.emission_depth(lm, p_) if lm is not None else 0.
                # Camera occlusion does not extinguish the source lighting the hand.
                P[j], _, _ = fire_light.relight(P[j], E0[..., :3], matte, lm, p_, s_, geo, z0,
                                               light_cfg | {"direction": u_, "hand_motion_px": hand_motion_px(track, i)})
                visibility = fire_visibility(track, i, matte, geo, z0, contact=eff.behaviour == "plume", box=fire_box)
                y0, y1, x0, x1 = fire_box
                E0[y0:y1, x0:x1] *= visibility[..., None]
                E, opacity = E0[..., :3], E0[..., 3]
                # Rank visible emission, not the unoccluded source or its reflected light.
                # A strict improvement keeps the first frame when multiple peaks are equal.
                visible_energy = float(fx._lum(E[y0:y1:4, x0:x1:4]).sum(dtype=np.float64))
                if visible_energy > poster_energy:
                    k_poster, poster_energy = i, visible_energy
                # Heat distorts the room before emission is added; preserve foreground skin after field smoothing.
                alpha = np.clip(fx._lum(E) * 4., 0., 1.)[None]
                fx.heat_haze(P[j:j+1], alpha, seed=spec.seed, out=P[j:j+1], frame0=i,
                             foreground=None if matte is None else matte[None])
            look = E[::4, ::4].max(-1) > 1e-4  # grain only where the element is
            ys_, xs_ = np.nonzero(look)
            if ys_.size:
                y0, y1 = max(0, 4 * ys_.min() - 4), min(h, 4 * ys_.max() + 8)
                x0, x1 = max(0, 4 * xs_.min() - 4), min(w, 4 * xs_.max() + 8)
                E[y0:y1, x0:x1] = fx.grain(E[None, y0:y1, x0:x1], noise, seed=spec.seed, frame0=i)[0]
            if eff.fluid == "smoke":  # smoke is laid over; fire is light, added over its soot; steam is lit vapour
                al = np.clip(E.max(-1) / 0.35, 0.0, 0.9)[..., None]
                P[j] = P[j] * (1.0 - al) + np.clip(E / np.maximum(al, 1e-3), 0, 1) * al
            elif eff.fluid == "steam":  # steam scatters the room's light: laid over, never brighter than its white
                al = np.clip(fx._lum(E) / 0.6, 0.0, 0.55)[..., None]  # translucent: a dense render read as cotton wool
                P[j] = P[j] * (1.0 - al) + np.minimum(E / np.maximum(al, 1e-3), max(0.9 * white, 0.45)) * al
            else:
                finish_fire(P[j], E, opacity, out=P[j])
        if eff.fluid != "fire":
            fx.flash(P, [off + ign - a0 for off, _, _ in plays], [g * flash_k for _, g, _ in plays], out=P)
        out[a:b] = gam(P[a - a0: b - a0])
        if eff.fluid == "fire":
            made.clear()  # each fire frame is consumed once; the next frame never needs its occluded emission
        for i in [k for k in made if k < b - 1]:  # the next chunk starts a frame back: keep that one and the next
            del made[i]
        say("compositing", 0.7 + 0.14 * b / n, f"Carrying the element on your hand: frame {b} of {n}")
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
    cv2.imwrite(str(out_dir / "poster.jpg"), cv2.cvtColor(np.ascontiguousarray(frames[k_poster]), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    write_video(final_path, _finalise(frames, env, spec.label or effect.replace("_", " "), None), fps=FPS, audio=audio,
                crf=16, preset=FINISH_PRESET)
    receipt = {
        "cuesync": __version__, "mode": "bending", "effect": eff.to_json(),
        "source_profile": profile,
        "inputs": {"point": list(spec.point), "seed": spec.seed, "offset_s": spec.offset_s, "cue": c.source},
        "render": {"seconds": round(seconds, 1), "prompt": prompt, "guide_scale": round(scale, 3)},
        "bending": {"track": str(track_path), "mask": None if mask is None else str(mask_path), "k_hit": int(k_launch),
                    "poster_frame": int(k_poster),
                    "tap_frame": int(tap_k), "aim": None if arrow is None else round(float(arrow), 2),
                    "aim_mode": track.get("aim_mode", "explicit" if arrow is not None else "tracked"),
                    "k_launch": int(k_launch), "ignition": int(ign), "offset": int(offset), "lag_frames": lag,
                    "snap": snap, "hits": [{"frame": int(off + ign), "gain": round(g, 3), "size": round(sz, 3),
                                            "snap": sn} for (off, g, sz), sn in zip(plays, snaps)],
                    "ref": {"xy": [round(float(v), 2) for v in ref_saved[0]],
                            "dir": [round(float(v), 5) for v in ref_saved[1]], "scale": round(float(ref_saved[2]), 2)},
                    "fit": fit, "thrown": None if not thrown else {"after": RELEASE_AFTER, "hands": THROW},
                    "loud_ref": round(float(loud_ref if loud_ref is not None else max(loud, default=0.0)), 4),
                    "held": None if not held else {"release": int(rel), "end": int(end_), "holds": [int(v) for v in holds]},
                    "performed_again": None if old is None or instant else {"from": str(element_from), "no_new_render": True},
                    "instant": None if not instant else {"library": str(element_from), "no_gpu": True},
                    "camera": "locked" if cam is None else "handheld: thrown fire moves with the room",
                    "fire_look": None if eff.fluid != "fire" else {
                        "method": "image-based 2.5D: coherent depth slabs and landmark hand surface",
                        "volume": FIRE_VOLUME, "lighting": "exposure-aware three-band local relight",
                        "ambient": light_cfg, "foreground_safe_heat": True},
                    "charge": None if not charge else {"from": int(charge[0]), "release": int(charge[1]),
                                                       "seconds": round((charge[1] - charge[0]) / FPS, 2)}},
        "auto": auto or picked, "sound": SOUND, "seconds_total": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=None, receipt=receipt)


MADE = {  # effect -> (its module, its form): elements made from the plate, the hand's track and the voice
    # (cuesync.elements' contract), not rendered over black by LTX-2.5
    "lightning_arc": ("lightning", "arc"), "lightning_ball": ("lightning", "ball"),
    "water_orb": ("water", "orb"), "water_whip": ("water", "whip"), "water_surge": ("water", "surge"),
    "water_ribbon": ("water", "ribbon"), "water_shield": ("water", "shield"), "water_jet": ("water", "jet"),
    "air_ball": ("air", "ball"), "air_blast": ("air", "blast"),
    "earth_rock": ("earth", "rock"), "earth_shatter": ("earth", "shatter"),
    "earth_volley": ("earth", "volley"), "earth_lance": ("earth", "lance"),
    "explosion": ("explosion", "fist"),  # (a palm's when the hand is open on its strike)
}
MADE_FOLEY = {"explosion": ("fire", "burst"), "lightning_arc": ("electric", "arc"), "lightning_ball": ("electric", "ball"),
              "water_orb": ("water", "orb"), "water_whip": ("water", "whip"), "water_surge": ("water", "whip"), "air_ball": ("air", "ball"),
              "water_ribbon": ("water", "whip"), "water_shield": ("water", "orb"), "water_jet": ("water", "whip"),
              "air_blast": ("air", "blast"), "earth_rock": ("earth", "rock"),
              "earth_shatter": ("earth", "shatter"), "earth_volley": ("earth", "shatter"),
              "earth_lance": ("earth", "shatter")}  # each one's Foley


def made_shot(spec, out_dir: Path, effect: str, plate: np.ndarray, track: dict, mask: np.ndarray | None, c, env_all,
              snaps: list[dict], looks: list[tuple[float, float]], k_launch: int, tap_k: int, arrow: float | None,
              auto: dict | None, old: dict | None, element_from, instant: bool, track_path, mask_path, say, t0: float):
    """A shot of an element made rather than rendered (MADE): the voice's timeline (each hit on its strike, as big as
    it is loud; where the voice holds) handed with the plate, the hand's track and its matte to the element's module,
    then the Foley under it, the shot and its receipt written as for a rendered element."""
    import importlib
    import time
    import types

    from . import __version__
    from .elements import Timeline
    from .media import write_video
    from .pipeline import FINISH_PRESET, FPS, SOUND, ShotResult, _finalise, mix_foley, place_audio, foley

    n = len(plate)
    module, form = MADE[effect]
    if effect == "explosion" and gesture_at(track, k_launch) == "Open_Palm":
        form = "palm"
    env = np.zeros(n)
    m = min(n, np.asarray(env_all).size)
    env[:m] = np.asarray(env_all, np.float64)[:m]
    cam = camera_path(plate, mask, track)  # a handheld camera: what the element leaves in the room moves with it
    timeline = Timeline(n=n, energy=env, hits=[int(s_["frame"]) for s_ in snaps], sizes=[float(sz) for _, sz in looks],
                        holds=[(int(a), int(b)) for a, b in c.cue.intervals()], tap_frame=int(tap_k), aim=arrow,
                        camera=cam)
    say("compositing", 0.3, f"Making the {module} on your hand")
    coverage = np.zeros(plate.shape[:3], np.uint8) if module == "earth" else None
    frames = importlib.import_module(f".{module}", __package__).render(
        plate, track, mask, timeline, form, seed=spec.seed, progress=say,
        **({"coverage": coverage} if coverage is not None else {}))
    if coverage is not None:
        np.savez_compressed(Path(out_dir) / "effect_mask.npz", alpha=coverage, fps=FPS, version=1)
    say("finishing", 0.85, "Adding the sound and the cue ribbon")
    audio = place_audio(c.samples, c.offset_s, n)
    if effect in MADE_FOLEY:
        fluid, behaviour = MADE_FOLEY[effect]
        eff = types.SimpleNamespace(id=effect, fluid=fluid, behaviour=behaviour)
        sounding = [int(np.argmax(timeline.sizes))] if effect == "explosion" and timeline.hits else range(len(looks))
        foley_env = np.zeros_like(env) if effect == "explosion" else env
        audio = mix_foley(audio, [foley(eff, n, foley_env, [timeline.hits[i] for i in sounding],
                                        [looks[i][0] for i in sounding],
                                        c.cue.voice_end_frame,
                                        spec.seed)])
    if module in ("earth", "water"):
        from .earth import choreography as earth_choreography
        from .water_sheets import choreography as water_choreography
        choreography = earth_choreography if module == "earth" else water_choreography
        actions=choreography(track,timeline,form)
        # The rock's physical releases get their own fracture accents. Keep the
        # original voice timing and avoid doubling accents within two frames.
        releases=[a for a in actions if a["action"]!="raise_stack"
                  and all(abs(a["frame"]-f)>2 for f in timeline.hits)]
        if releases:
            eff=types.SimpleNamespace(id=effect,fluid=module,behaviour="shatter" if module=="earth" else "whip")
            audio=mix_foley(audio,[foley(eff,n,np.zeros(n),[a["frame"] for a in releases],
                                        [.65*a["strength"] for a in releases],n-1,spec.seed+31)])
    out_dir = Path(out_dir)
    comp_path, final_path = out_dir / "composite.mp4", out_dir / "final.mp4"
    k_poster = int(np.clip(k_launch + 8, 0, n - 1))
    cv2.imwrite(str(out_dir / "poster.jpg"), cv2.cvtColor(np.ascontiguousarray(frames[k_poster]), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    write_video(comp_path, frames, fps=FPS, audio=audio, crf=14, preset=FINISH_PRESET)
    write_video(final_path, _finalise(frames, env, spec.label or effect.replace("_", " "), None), fps=FPS, audio=audio,
                crf=16, preset=FINISH_PRESET)
    receipt = {
        "cuesync": __version__, "mode": "bending",
        "effect": {"id": effect, "made": module, "form": form, "fluid": module, "behaviour": form},
        "inputs": {"point": list(spec.point), "seed": spec.seed, "offset_s": spec.offset_s, "cue": c.source},
        "render": {"seconds": 0.0, "prompt": f"(made, not rendered: cuesync.{module})", "guide_scale": 1.0},
        "bending": {"track": str(track_path), "mask": None if mask is None else str(mask_path), "k_hit": int(k_launch),
                    "tap_frame": int(tap_k), "aim": None if arrow is None else round(float(arrow), 2),
                    "aim_mode": track.get("aim_mode", "explicit" if arrow is not None else "tracked"),
                    "k_launch": int(k_launch),
                    "snap": snaps[int(np.argmax([sz for _, sz in looks]))] if snaps else {"used": "cue", "mismatch": False},
                    "hits": [{"frame": int(f), "gain": round(g, 3), "size": round(sz, 3),
                                                         "snap": sn} for f, (g, sz), sn in
                                                        zip(timeline.hits, looks, snaps)],
                    "holds": [list(v) for v in timeline.holds],
                    "performed_again": None if old is None or instant else {"from": str(element_from),
                                                                            "no_new_render": True},
                    "camera": "locked" if cam is None else "handheld: what it leaves moves with the room"},
        "auto": auto, "sound": SOUND, "seconds_total": round(time.perf_counter() - t0, 1),
    }
    if module in ("earth", "water"):
        from .earth import choreography as earth_choreography
        from .water_sheets import choreography as water_choreography
        choreography = earth_choreography if module == "earth" else water_choreography
        from .motion import analyze
        receipt["bending"]["movement"] = {"method": "camera-compensated 2D strokes; confidence and continuity gated",
                                          "flow_bridge_frames": track.get("flow_bridge_frames",[]),
                                          "observed": analyze(track,cam).receipt(),
                                          "actions": choreography(track,timeline,form)}
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    say("done", 1.0, "Done")
    return ShotResult(out_dir=out_dir, final=final_path, composite=comp_path, breakdown=None, receipt=receipt)


# ------------------------------------------------------------------ the studio's bending shot, from your clip

EMITTER_OF = {"fire_breath": "two_finger", "fire_burst": "palm", "flame_plume": "palm", "steam_jet": "palm",
              "smoke_puff": "palm", "lightning_arc": "two_finger", "lightning_ball": "palm", "water_orb": "palm",
              "water_whip": "palm", "water_surge": "palm", "air_ball": "palm", "air_blast": "palm", "explosion": "palm",
              "water_ribbon": "palm", "water_shield": "palm", "water_jet": "palm",
              "earth_rock": "palm", "earth_shatter": "palm", "earth_volley": "palm", "earth_lance": "palm"}  # the point on the
# hand cuesync.hand_track follows for each element


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
    if kind == "mask":  # no SAM set up: the tracker's venv has MediaPipe, whose skin segmenter mattes the hand on the
        # CPU (in this one the matte was the landmarks' hull on every frame)
        return hand_python("track")
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


def _helper_env() -> dict:
    return os.environ | {"PYTHONPATH": os.pathsep.join(filter(None, [str(package_root()), os.environ.get("PYTHONPATH")]))}


def track_hand(plate: Path, out_dir: Path, tap: tuple[float, float], frame: int, emitter: str) -> Path:
    """Follow the tapped hand through ``plate`` (the clip as the pipeline loads it; ``tap`` in px on ``frame``) with the
    tracker, a subprocess in its own venv: out_dir/hand_track.json."""
    track = out_dir / "hand_track.json"
    r = subprocess.run([hand_python("track"), "-m", "cuesync.hand_track", str(plate), "--emitter", emitter,
                        "--seed", f"{frame},{tap[0]:.1f},{tap[1]:.1f}", "--out", str(track)],
                       env=_helper_env(), capture_output=True, text=True, timeout=900)
    (out_dir / "hand_track.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0 or not track.exists():
        last = (r.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"couldn't follow your hand through the clip ({last})")
    return track


LOST_RETRY, LOST_NONE = 0.25, 0.9  # a tap that lost the hand on more of the clip than this is followed again from a
# clearer frame; on this much, there was no hand where it was tapped


def _lost_share(t: dict) -> float:
    return sum(s == "lost" for s in t["src"]) / max(1, len(t["src"]))


def follow_tap(plate: Path, out_dir: Path, tap: tuple[float, float], frame: int, emitter: str, say=None) -> Path:
    """track_hand, checked. No hand where it was tapped is said (it was followed as a still point and called ready).
    A tap that lost the hand on more than LOST_RETRY of the clip is followed again from the frame the hand was seen
    most surely, the tap carried onto it, and the better track kept, with that frame and point as its "retap"."""
    track = track_hand(plate, out_dir, tap, frame, emitter)
    t = load_track(track)
    lost = _lost_share(t)
    if lost >= LOST_NONE:
        raise RuntimeError("couldn't find a hand where you tapped: tap your palm or fingertips on a frame where your "
                           "hand is clear")
    seen = [i for i, src in enumerate(t["src"]) if src == "mp"]
    if lost > LOST_RETRY and seen:
        k2 = max(seen, key=lambda i: (float(t["conf"][i]), -abs(i - frame)))
        p2 = tuple(float(v) for v in anchored(t, frame, tap)["xy"][k2])
        if say:
            say("tracking", 0.3, "Following your hand again, from the frame it shows most clearly")
        again = out_dir / "follow_again"
        again.mkdir(exist_ok=True)
        try:
            t2 = load_track(track_hand(plate, again, p2, k2, emitter))
        except RuntimeError:
            return track
        if _lost_share(t2) < lost:
            raw = json.loads((again / "hand_track.json").read_text())
            raw["retap"] = {"frame": int(k2), "x": round(p2[0] / raw["W"], 4), "y": round(p2[1] / raw["H"], 4),
                            "lost_before": round(lost, 3)}
            track.write_text(json.dumps(raw), encoding="utf-8")
    return track


def retapped(track: Path, frame: int, point: tuple[float, float]) -> tuple[int, tuple[float, float]]:
    """The frame and point (fractions) a followed tap is anchored on: the tap, or where follow_tap moved it."""
    r = json.loads(Path(track).read_text()).get("retap") if Path(track).exists() else None
    return (int(r["frame"]), (float(r["x"]), float(r["y"]))) if r else (frame, point)


def prepare_hand(plate: Path, out_dir: Path, tap: tuple[float, float], frame: int, emitter: str,
                 say=None) -> tuple[Path, Path | None]:
    """Track the tapped hand through ``plate`` (the clip as the pipeline loads it; ``tap`` in px on ``frame``) and matte
    it, each as a subprocess in its own venv; returns (hand_track.json, hand_mask.npz or None: without a matte the
    element's root is held out behind the fingertips instead)."""
    env = _helper_env()
    track, mask = follow_tap(plate, out_dir, tap, frame, emitter, say), out_dir / "hand_mask.npz"
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
    (as make_bending_shot does), held to what the library has (the air blast and the elements made here, MADE, are
    made, not rendered: always there)."""
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
        effect, why = bending_effect(h_["kind"], g, hits=len(hits), span_s=span, longest_s=h_.get("longest_s", 0.0))
        effect = effect or "fire_burst"
        picked = h_ | {"heard": h_["kind"], "hand": g, "picked": effect, "why": why}
    lib = library_dir()
    if effect != "smoke_puff" and effect not in MADE and (lib is None or not (lib / str(effect) / "element.mp4").exists()):
        effect = "fire_burst"
        picked = None if picked is None else picked | {"picked": effect, "why": picked["why"] + " (the library's nearest)"}
    return effect, picked


def load_take(spec, out_dir: Path, say) -> tuple[Path, int, int, int, int]:
    """The studio's clip as the pipeline loads it, kept (in/plate24.mp4, its own sound with it: the shot and performing
    it again read the same frames); ``spec`` is updated to it. (loaded, frames, height, width, the tapped frame)."""
    from .media import probe_video, read_audio, write_video
    from .pipeline import FPS, SR, load_plate, place_audio, shot_cue

    say("reading", 0.02, "Reading your clip")
    if spec.voice is not None or spec.cue == "clip":  # (a cue it can't use, a silent clip on "its own sound", is said
        # now, not after the hand has been tracked)
        shot_cue(spec, min(probe_video(spec.plate).duration_s or spec.max_plate_s, spec.max_plate_s))
    frames, point, _ = load_plate(spec.plate, spec.point, spec.max_plate_s, spec.plate_size)
    n, h, w = frames.shape[:3]
    loaded = out_dir / "in" / "plate24.mp4"
    try:  # the clip's own sound goes with it: the shot (and performing it again) may be cued by it
        sound = place_audio(read_audio(spec.plate, SR), 0.0, n)
    except (ValueError, OSError):
        sound = None
    write_video(loaded, frames, fps=FPS, audio=sound, crf=12)
    del frames
    spec.plate, spec.point = loaded, (float(point[0]), float(point[1]))
    return loaded, n, h, w, int(np.clip(spec.extra.get("tap_frame") or 0, 0, n - 1))


def note_crop(track_path: Path, plate, point, size) -> None:
    """Write into a track where its frame sits in the clip as uploaded (pipeline.plate_box, fractions of that frame),
    for the studio to draw it on that clip: a 4:3 webcam take is cropped to 16:9 around the tap, and the track's px
    are the crop's (the preview ring sat off the hand)."""
    from .pipeline import plate_box

    try:
        box = [round(float(v), 5) for v in plate_box(Path(plate), tuple(point), size)]
        t = json.loads(Path(track_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    t["box"] = box
    Path(track_path).write_text(json.dumps(t), encoding="utf-8")


def prepare_take(spec, out_dir: Path, progress=None, emitter: str = "palm") -> Path:
    """Your tap followed through the clip before any shot is made (the studio draws the hand's path under your aim, and
    says where it lost it), then matted; a shot on the same tap reuses it (bending_job's ``reuse``)."""
    out_dir = Path(out_dir)
    (out_dir / "in").mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    clip, tapped = spec.plate, tuple(spec.point)  # (as uploaded: load_take moves the spec onto the loaded plate)
    loaded, n, h, w, k = load_take(spec, out_dir, say)
    say("tracking", 0.1, "Finding your hand and following it")
    track = follow_tap(loaded, out_dir, (spec.point[0] * w, spec.point[1] * h), k, emitter, say)
    k, point = retapped(track, k, tuple(spec.point))
    note_crop(track, clip, tapped, spec.plate_size)
    receipt = {"mode": "hand_prep", "inputs": {"point": list(point)}, "bending": {"tap_frame": k, "emitter": emitter}}
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    say("tracked", 0.5, "Your hand, followed through the clip")
    try:
        r = subprocess.run([hand_python("mask"), "-m", "cuesync.hand_mask", str(loaded), "--track",
                            str(out_dir / "hand_track.json"), "--out", str(out_dir / "hand_mask.npz")],
                           env=_helper_env(), capture_output=True, text=True, timeout=900)
        (out_dir / "hand_mask.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as exc:
        (out_dir / "hand_mask.log").write_text(f"{type(exc).__name__}: {exc}" + chr(10), encoding="utf-8")
    say("done", 1.0, "Your hand is ready")
    return out_dir / "hand_track.json"


def take_emitter(spec, n: int) -> str:
    """Which point of the hand a shot of ``n`` frames follows: the effect's own, or, when the sound decides, the fingers
    for a hiss or a whoosh (a flamethrower), else the palm."""
    from .gestures import heard
    from .pipeline import FPS, SR, shot_cue

    if spec.effect != "auto":
        return EMITTER_OF.get(spec.effect, "palm")
    c = shot_cue(spec, n / FPS)
    return "two_finger" if heard(c.cue, c.samples, SR, c.source)["kind"] in ("whoosh", "hiss") else "palm"


def bending_job(spec, renderer, out_dir: Path, progress=None, reuse: Path | None = None):
    """The studio's bending shot from your clip: the plate as the pipeline loads it (kept, so performing it again reads
    the same frames), the hand you tapped (``spec.point`` on frame ``spec.extra["tap_frame"]``) tracked and matted, then
    make_bending_shot, aimed by ``spec.direction`` when you drew the arrow. ``spec`` is updated to the loaded plate.
    ``reuse``: the same clip and tap already loaded, tracked and matted (the studio's preview of your tap, or an earlier
    shot when another effect is picked on the badge): taken as it is, so only the shot is made (the aim you drew turns
    the fire, whichever point of the hand was followed)."""
    import shutil

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
        track, mask = out_dir / "hand_track.json", out_dir / "hand_mask.npz"
        say("reading", 0.05, "Your hand as it was followed for your tap")
        return carry(spec, renderer, out_dir, track, mask if mask.exists() else None, progress, spec.extra.get("tap_frame"))
    clip, tapped = spec.plate, tuple(spec.point)
    loaded, n, h, w, k = load_take(spec, out_dir, say)
    emitter = take_emitter(spec, n)
    say("tracking", 0.05, "Finding your hand and following it")
    track, mask = prepare_hand(loaded, out_dir, (spec.point[0] * w, spec.point[1] * h), k, emitter, say)
    k, spec.point = retapped(track, k, tuple(spec.point))  # (followed again from a clearer frame, the tap carried there)
    note_crop(track, clip, tapped, spec.plate_size)
    spec.extra["tap_frame"] = k  # (with its frame: performing again copies this spec, and the point on the old frame
    # put the fire 1.7 hands off the palm)
    return carry(spec, renderer, out_dir, track, mask, progress, k)


def carry(spec, renderer, out_dir: Path, track: Path, mask: Path | None, progress, tap_frame: int | None):
    """make_bending_shot for the studio: rendered on the GPU (``renderer``), or with none an element rendered ahead of
    time from the library, carried on your hand (an instant shot; the air blast is made, not rendered)."""
    if renderer is not None or spec.effect in MADE:  # (an element made here needs neither a GPU nor the library)
        return make_bending_shot(spec, renderer, out_dir, track, mask, progress, tap_frame=tap_frame, aim=spec.direction)
    t = load_track(track)
    spec.effect, picked = instant_effect(spec, t, len(t["xy"]))
    lib = None if spec.effect == "smoke_puff" or spec.effect in MADE else library_dir() / spec.effect
    return make_bending_shot(spec, None, out_dir, track, mask, progress, element_from=lib, tap_frame=tap_frame,
                             aim=spec.direction, instant=lib is not None, picked=picked)


def track_for(track: dict, emitter: str) -> dict:
    """``track``, a hand followed for one point on it (its "emitter"), moved onto ``emitter``: read again off its
    landmarks where it has them (hand_track.emitter_points), and elsewhere moved as the nearest such frame's point
    moved. (A sample hand followed for its fireball's palm struck a fingertip's lightning out of the palm, down the
    arm.)"""
    from .hand_track import emitter_points

    lms = track.get("landmarks") or []
    have = np.array([i for i, lm in enumerate(lms) if lm])
    if track.get("emitter") == emitter or have.size == 0:
        return track
    sign = -1.0 if track.get("hand") == "left" else 1.0
    xy, d = np.asarray(track["xy"], np.float64), np.asarray(track["dir"], np.float64)
    new_xy, new_d = xy.copy(), d.copy()
    for i in have:
        p, u, _ = emitter_points(np.asarray(lms[i], np.float64), emitter, sign, frame=(track["W"], track["H"]))
        new_xy[i], new_d[i] = p, u
    for i in range(len(xy)):
        if i >= len(lms) or not lms[i]:
            j = int(have[np.argmin(np.abs(have - i))])
            new_xy[i], new_d[i] = xy[i] + (new_xy[j] - xy[j]), new_d[j]
    return {**track, "xy": new_xy.round(2).tolist(), "dir": new_d.round(4).tolist(), "emitter": emitter}


SAMPLE_TAIL_S = 0.4  # a sample noise's last sound lands at least this long before its clip's end...
SAMPLE_HELD_S = 1.2  # ...and a noise held this long lets go there too...
SAMPLE_LEAD_S = 0.3  # ...but a held noise moves at most this much earlier, unless its loudest hit would miss the clip


def sample_noise_offset(voice: Path, plate_s: float, frame: int, *, fit_held: bool = False) -> float:
    """Where a sample noise starts on a hand's clip (s): its loudest hit on ``frame``, the hand's strike,
    moved earlier as far as it takes for its last sound to land in the clip,
    and a held noise (a sound over SAMPLE_HELD_S) to let go there too, though by no more than SAMPLE_LEAD_S; a hit's
    tail may run off the end, but its loudest hit stays inside the clip."""
    from .audio import analyze
    from .media import read_audio
    from .pipeline import FPS, SR

    cue = analyze(read_audio(voice, SR), SR, fps=FPS)
    spans = cue.intervals()
    held = max([b_ - a_ for a_, b_ in spans] or [0]) / FPS >= SAMPLE_HELD_S
    end = max(cue.onsets_s[-1], max(b_ for _, b_ in spans) / FPS if held else 0.0)
    on = frame / FPS - cue.hero_onset_s
    off = max(0.0, on - SAMPLE_LEAD_S, min(on, plate_s - SAMPLE_TAIL_S - end))
    if held and fit_held:
        # A held procedural element needs time to form and release. Its renderer
        # gates on observed hand presence; a late marked strike need not truncate it.
        off = min(off, max(0.0, plate_s - SAMPLE_TAIL_S - end))
    return float(max(0.0, min(off, plate_s - SAMPLE_TAIL_S - cue.hero_onset_s)))


def bending_again(src_dir: Path, spec, out_dir: Path, progress=None):
    """Perform a finished bending shot again with a new take (``spec``): its element on the same hand take, no new
    render; a shot that was itself performed again plays its first render. An element made here (MADE) is made anew on
    that take's own track and matte (a sample hand's, shipped with it: no tracker needed)."""
    src_dir = Path(src_dir)
    b = json.loads((src_dir / "receipt.json").read_text())["bending"]
    first = Path(b["performed_again"]["from"]) if b.get("performed_again") else \
        Path(b["instant"]["library"]) if b.get("instant") else src_dir
    hand = Path(spec.extra.get("reuse") or first)
    here = lambda name, was: str(hand / name) if (hand / name).exists() else was  # noqa: E731  (a shot made ahead
    # travels with its own track and matte)
    if spec.effect in MADE:  # (followed for its fire's point on the hand: moved onto this element's)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        track = Path(out_dir) / "hand_track.json"
        moved = track_for(load_track(Path(here("hand_track.json", b["track"]))), EMITTER_OF.get(spec.effect, "palm"))
        if spec.extra.get("sample_emitter"):
            k = int(np.clip(spec.extra.get("tap_frame") or 0, 0, len(moved["xy"]) - 1))
            spec.point = (moved["xy"][k][0] / moved["W"], moved["xy"][k][1] / moved["H"])
        track.write_text(json.dumps(moved))
        return make_bending_shot(spec, None, out_dir, track, here("hand_mask.npz", b["mask"]), progress,
                                 tap_frame=spec.extra.get("tap_frame"), aim=spec.direction)
    saved_aim = None if b.get("aim_mode") == "upward_default" else b.get("aim")
    return make_bending_shot(spec, None, out_dir, here("hand_track.json", b["track"]), here("hand_mask.npz", b["mask"]),
                             progress, element_from=first, tap_frame=spec.extra.get("tap_frame"),
                             aim=spec.direction if spec.direction is not None else saved_aim,
                             instant=bool(b.get("instant")))
