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
SIZE = {"jet": (6.0, 0.75), "burst": (2.0, 0.45), "plume": (1.8, 0.45), "puff": (2.5, 0.4)}  # per behaviour: its reach in
# hand sizes, and at most this share of the frame's short side (a close-up hand made a fireball fill the frame)
IGNITION = 0.15  # an element has ignited once its light is this share of its brightest frame's
LOST_GRACE, LOST_FADE = 0, 4  # the hand lost (and the picture where it was changed): the element fades out over these
# frames from where it was last seen (after 8 frames' grace it burned a third of a second beside a swung fist; the
# track bridges any shorter gap itself, so a lost frame is never a flicker)
ROOT_FADE = 0.12  # share of the element's reach over which it comes up from nothing at the fingertips: the model draws
# a jet's root as a hard bright nozzle and a flame's foot as a flat edge, and fire from a hand has neither
ROOT_ROUND = 0.15  # how far the root's fade curves back at the sides, in its own length over its width squared
ROOT_FADE_SHARE = {"steam_jet": 0.05, "fire_breath": 0.06, "flame_plume": 0.04}  # steam has no nozzle to hide (over
# the full fade it started a hand from the palm); a jet six hands long or a lighter's flame left a gap at the finger
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
                   lit: bool = False) -> tuple[str | None, str]:
    """(effect, why) for a hand in the shot. ``kind``: the take's sound (gestures.heard); ``gesture``: the hand's at
    the hit (gesture_at); ``hits``/``span_s``: how many hits the take has and how long from the first to the last;
    ``snap``: the track saw a finger snap; ``lit``: a flame burns at the tapped spot.
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
        lum = np.array([float(np.clip(f[::4, ::4].astype(np.float32) - floor, 0.0, None).mean()) for f in element])
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
DAY_GAIN = 1.0  # and over a bright room it is up to this much brighter than rendered (twice over a white wall): the
# render was exposed over black, and a flame doesn't dim with the room; at the render's own level it read as orange
# paint, at 2.6 times its cores went to featureless lemon. Its light on the hand stays as rendered


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


def held_frames(d: int, ign: int, rel: int, end: int, hold: int, x: int = SUSTAIN_XFADE) -> list[tuple[int, float]]:
    """The render's frames, with their weights, that a held hit plays ``d`` frames after its ignition, for a voice
    that holds ``hold`` frames: its body as rendered; looped while the voice holds on past it (the loop's end
    cross-faded into its start); and from wherever the voice stops, its letting go ([] once that has burned out).
    A take as long as the render's plays it as rendered."""
    body = rel - ign
    stop = max(int(hold), 1)
    looping = stop > body and body > 2 * x + 2  # (too short a body to loop: it holds its last frame)

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
        return burning(d)
    r = d - stop
    if end - rel <= x:  # a render that burns to its last frame has no letting go: it dies down from where it was
        return [] if r >= LETGO else [(f, wt * (1.0 - (r + 1) / (LETGO + 1))) for f, wt in burning(stop - 1 + r)]
    if stop == body:  # as long as rendered: its own letting go, uncut
        return [(rel + r, 1.0)] if rel + r < end else []
    if rel + r >= end:
        return []
    if r >= x:
        return [(rel + r, 1.0)]
    w = (r + 1) / (x + 1)
    return [(f, wt * (1.0 - w)) for f, wt in burning(stop - 1)] + [(rel + r, w)]


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
    it, so the element comes up from nothing just past the fingertips. The edge curves back at the sides (ROOT_ROUND):
    a straight one cut a wide fireball's base flat along a line."""
    p0, u0, s0 = ref
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    along = (xx - p0[0]) * u0[0] + (yy - p0[1]) * u0[1]
    across = (xx - p0[0]) * u0[1] - (yy - p0[1]) * u0[0]
    length = max(length, 1.0)
    return np.clip((along - ROOT_ROUND * across * across / length) / length, 0, 1).astype(np.float32)


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

    auto = None
    old = json.loads((Path(element_from) / "receipt.json").read_text()) if element_from is not None else None
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
        effect, why = bending_effect(h_["kind"], g, hits=len(hits), span_s=span, snap=snapped)
        effect = effect or "fire_burst"
        auto = h_ | {"heard": h_["kind"], "hand": g, "picked": effect, "why": why}
    eff = load_effect(effect)
    if eff.behaviour == "plume":  # a flame standing on the hand burns straight up, however the hand turns (it tipped over
        # sideways on a turning fist, like a lighter's never does)
        track = track | {"dir": [[0.0, -1.0]] * len(track["dir"])}

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
    if element.shape[1:3] != (h, w):  # (a render of another size: placed by its pose, never stretched)
        element, ref = fit_element(element, ref, h, w)
    floor = estimate_black_level(element)
    ign = ignition_frame(element, floor, burst=eff.behaviour == "burst")
    if ign is None:
        raise ValueError("the render shows no element to carry")
    # each hit fires the element: its ignition on that hit's strike, as big and as bright as the hit is loud
    plays = [(s_["frame"] - ign, g, sz) for s_, (g, sz) in zip(snaps, looks)]
    every = plays  # (the Foley sounds on every hit of the take, whichever hand fires it)
    # a render played again (your take again, or one made ahead) holds a jet or a plume as long as this voice holds
    # it, and lets it go when the voice stops: as rendered it lasted as long as the take it was made for
    held = old is not None and eff.behaviour in ("jet", "plume")
    rel = end_ = ign
    holds: list[int] = []
    if held:
        rel, end_ = element_span(element, floor, ign)
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
            if DEFOCUS > 0:  # no phone films as crisply as the render draws
                out_[y0:y1, x0:x1] = cv2.GaussianBlur(out_[y0:y1, x0:x1], (0, 0), DEFOCUS * min(h, w) / 1080)
        return out_

    def element_at(t: int) -> np.ndarray | None:
        """The element on plate frame t (linear, over black): every hit's play of it, carried on the hand, its root
        held out; None when none is on (before its build-up, after it ends)."""
        acc = None
        for k_, (off, gain, size) in enumerate(plays):
            e = t - off
            if held and e >= ign:  # held as long as the voice holds it, then let go
                picks = held_frames(e - ign, ign, rel, end_, holds[k_])
                if not picks or max(f for f, _ in picks) >= len(element):
                    continue
            else:
                if eff.behaviour == "burst" and e > ign:  # a smaller blast is over sooner (a fireball lasts as long as
                    e = ign + int(round((e - ign) / size))  # it is big): a "pf" puffs and is gone, a "BOOM" rolls on
                if not max(0, ign - pre) <= e < len(element):  # (a negative frame would wrap round to the render's end)
                    continue
                picks = [(e, 1.0)]
            grow = min(1.0, GROW_FROM + (1.0 - GROW_FROM) * max(0.0, e - ign + pre) / (pre + GROW_FRAMES))
            if eff.behaviour == "burst":  # (a burst is the blast itself: no growing into it)
                grow = 1.0
            S = source(picks[0][0]) if picks == [(picks[0][0], 1.0)] else sum(wt * source(f) for f, wt in picks)
            E = warp_element(S, track, ref, t, lag, size * grow, since=off + ign, cam=cam) * gain
            if e < ign:  # the build-up, coming up to the ignition: its fire only, orange or white-hot (the model's
                # dim grey haze before the flame catches showed as a grey blob on the palm)
                warm = np.clip(3.0 * (E[..., 0] - E[..., 2]) / np.maximum(E[..., 0], 1e-4) - 1.0, 0.0, 1.0)
                E *= ((e - (ign - pre) + 1) / (pre + 1)
                      * np.maximum(warm, np.clip((fx._lum(E) - 0.35) / 0.3, 0.0, 1.0)))[..., None]
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
        if lost[t] > LOST_GRACE:  # the hand is gone: let the element go rather than freeze it
            E *= max(0.0, 1.0 - (lost[t] - LOST_GRACE) / LOST_FADE)
        m_t = None if mask is None else mask[min(t, len(mask) - 1)]
        y0, y1, x0, x1 = box = holdout_box(track, t, h, w) if m_t is None else matte_box(m_t)
        if y1 > y0 and x1 > x0:
            hold = root_holdout(track, t, h, w, None if m_t is None else m_t.astype(np.float32) / 255.0, box)
            E[y0:y1, x0:x1] *= (1.0 - hold)[..., None]
        if t > 0:  # the camera's shutter smears the fire as it smears the hand carrying it
            E = motion_blur(E, emitter_at(track, t)[0] - emitter_at(track, t - 1)[0])
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
    behind: dict[int, float] = {}  # each frame's room behind its fire (room_behind), and its fire brightened by it once
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        a0, b1 = max(0, a - 1), min(n, b + 1)
        for i in range(a0, b1):
            if i not in made:
                made[i] = element_at(i)
        P = np.stack([lin(plate[i]) for i in range(a0, b1)])
        if eff.fluid == "fire":  # over a bright room the flame is brighter than rendered (its light on the skin is not:
            # boosted, it blew the skin out in a sunlit room)
            for i in range(a0, b1):
                if made[i] is not None and i not in behind:
                    behind[i] = room_behind(made[i], P[i - a0])
        small = np.zeros((b1 - a0, h // q, w // q, 3), np.float32)
        for i in range(a0, b1):
            if made[i] is not None:
                small[i - a0] = cv2.resize(made[i], (w // q, h // q), interpolation=cv2.INTER_AREA)
        m = None if mask is None else mask[a0:b1].astype(np.float32) / 255.0
        fx.light_plate(P, small, track_slice(track, a0, b1), m, fluid=eff.fluid, white=white, tint=tint, out=P)
        for i in range(a, b):
            if made[i] is None:
                continue
            E0 = made[i]  # as rendered: its light on the hand was worked out from this
            E = E0
            if eff.fluid == "fire":
                E = E0 * np.float32(1.0 + DAY_GAIN * min(1.0, behind.get(i, 0.0) / fx.EXPOSED_WHITE))
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
            else:  # fire is light, added, and where it outshines the room behind it, it hides it: measured against
                # the room as a whole (pixel by pixel, a jet changed from pale to orange where it crossed a skirting)
                side = min(h, w)  # by the flame's body, not its strands (pixel by pixel, the wall showed through the
                # dim gaps between bright strands: a jet like orange lace)
                body = cv2.GaussianBlur(fx._lum(E), (0, 0), SOOT_BODY * side)
                a_ = np.clip(body / (behind.get(i, 0.0) + SOOT_DARK), 0.0, 1.0)
                # ... and never more than it outshines what is there: a dim edge of the flame over a lit palm in a dark
                # room hid the palm, a dark rim round the fireball's foot
                there = cv2.GaussianBlur(fx._lum(np.maximum(P[j], 0.0)), (0, 0), SOOT_EDGE * side)
                a_ = np.minimum(a_, np.clip(cv2.GaussianBlur(fx._lum(E), (0, 0), SOOT_EDGE * side) / (there + SOOT_DARK),
                                            0.0, 1.0))
                a_ *= cv2.GaussianBlur(np.clip(fx._lum(E) / 0.02, 0.0, 1.0), (0, 0), 3.0 * side / 1080)  # where flame is
                lo, hi, c0, c1 = DAY_CORE
                day = float(np.clip((behind.get(i, 0.0) - lo) / (hi - lo), 0.0, 1.0))
                if day > 0:  # a bright room: the flame's dense body covers it (its edges stay light)
                    core = np.clip((cv2.GaussianBlur(fx._lum(E0), (0, 0), h / 1080) - c0) / (c1 - c0), 0.0, 1.0)
                    core = cv2.erode(core * core * (3.0 - 2.0 * core), np.ones((3, 3), np.uint8))
                    a_ = a_ + day * day * (3.0 - 2.0 * day) * core * (1.0 - a_)
                a_ = cv2.GaussianBlur(a_, (0, 0), 1.5 * h / 1080)
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
                    "loud_ref": round(float(loud_ref if loud_ref is not None else max(loud, default=0.0)), 4),
                    "held": None if not held else {"release": int(rel), "end": int(end_), "holds": [int(v) for v in holds]},
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


def prepare_take(spec, out_dir: Path, progress=None, emitter: str = "palm") -> Path:
    """Your tap followed through the clip before any shot is made (the studio draws the hand's path under your aim, and
    says where it lost it), then matted; a shot on the same tap reuses it (bending_job's ``reuse``)."""
    out_dir = Path(out_dir)
    (out_dir / "in").mkdir(parents=True, exist_ok=True)
    say = progress or (lambda stage, frac, msg: None)
    loaded, n, h, w, k = load_take(spec, out_dir, say)
    say("tracking", 0.1, "Finding your hand and following it")
    track = follow_tap(loaded, out_dir, (spec.point[0] * w, spec.point[1] * h), k, emitter, say)
    k, point = retapped(track, k, tuple(spec.point))
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
    loaded, n, h, w, k = load_take(spec, out_dir, say)
    emitter = take_emitter(spec, n)
    say("tracking", 0.05, "Finding your hand and following it")
    track, mask = prepare_hand(loaded, out_dir, (spec.point[0] * w, spec.point[1] * h), k, emitter, say)
    k, spec.point = retapped(track, k, tuple(spec.point))  # (followed again from a clearer frame, the tap carried there)
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
