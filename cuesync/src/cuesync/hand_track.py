"""The performer's hand, frame by frame: where the element comes out of it, which way it points, how big the hand is,
what it is doing, and when it strikes (``hand_track.json``, see docs/bending_contracts.md).

The element is rendered once, standing still over black, where the hand is on the strike; the compositor then carries
it on this track, frame by frame. So the track has to be right on every frame the hand is in, steady while the hand
holds, and exact about *when* a strike locks, the frame the element goes off on; whether the camera stands on a
tripod facing the performer or is the performer's own eyes (a phone held in the other hand, or on the head), with the
hand reaching in from the bottom of the picture, big, the wrist cut off, and the picture shaking. In order:

1. **Find the hand.** MediaPipe's GestureRecognizer (HandLandmarker's 21 landmarks plus a canned gesture such as
   Open_Palm or Victory; Apache-2.0) in VIDEO mode, which carries each hand's crop on from the frame before, on a copy
   with the long side at 1280 px, scaled back to the clip. Up to two hands a frame; the one followed is the hand nearest
   the Studio's click (``seed``), else the chosen hand, else the hand most in view (how much of it is in the picture
   times its size, over the clip: in a first-person shot the casting hand, not a thumb of the hand holding the phone).
   From there it is followed frame by frame to whichever detection is where it was heading, the same size give or
   take (a fragment of the hand that MediaPipe reports as a hand is not).
   **And the camera.** The camera's own motion is measured on the picture around the hands (camera_motion) and taken
   out: following, doubting, finding strikes and smoothing all happen in the scene's coordinates, so a shaking camera
   is neither a hand jumping nor a hand moving, and the result is put back into each frame's pixels, so the element
   still rides the hand wherever the shake puts it. A plain backdrop gives nothing to measure; the camera then counts
   as still, and the shake stays in the hand's path (followed, as the hand is, and rounded off where it is fast).
2. **Doubt it.** A frame counts as good only when MediaPipe calls it the same hand (left or right) as the rest of the
   track, and it is on the path its neighbours trace: its emitter within 0.6 x the hand's size of the median of the
   five frames around it, its landmarks on average within 0.3 sizes of theirs, its size within a half of theirs. A
   median keeps any steady motion, however fast, so only a jump off the path fails.
3. **Look again** where it failed: RTMPose-m hand (rtmlib, ONNX; Apache-2.0) on the crop where the good frames either
   side put the hand, kept when it is confident and lands where the hand should be. It is a different network, trained
   on other data, and often sees the fingers MediaPipe lost to motion blur. The two place the knuckles and wrist up to
   a quarter of a hand apart, so each re-detection is moved by how far RTMPose sits from MediaPipe on the good frames
   around it. At either end of the track it is chained outward from the last frame it found, so a hand MediaPipe only
   picks up late is followed back to where it entered.
4. **Fill what is still missing.** Up to 3 frames: optical flow (pyramidal Lucas-Kanade) of every landmark, forward
   from the good frame before and backward from the one after, blended by how close each is. 4 to 8 frames: drawn
   straight between the two (``interp``; a TAPIR point track, google-deepmind/tapnet, Apache-2.0, would go here, and
   isn't wired in yet). Longer: ``lost``, holding the last good emitter with confidence 0 (the compositor fades the
   element out); before the hand is first found: ``hold``, holding the first good one.
5. **Find the strikes** on the filled but unsmoothed path: an arm locks (``extension``) on the first frame after a peak
   of speed where the speed is under 0.3 x the peak, within 6 frames; a palm opens (``open``) where a fist's fingers
   spread out (measured in 3D, or MediaPipe's gesture going from Closed_Fist to Open_Palm). Smoothing first would smear
   both.
6. **Smooth without delay.** Savitzky-Golay (quadratic) over the emitter, the hand's size, its direction as an
   unwrapped angle and the landmarks: over 7 frames (at 24 fps) while the hand moves, over 15 while it holds, blended by
   its speed. A symmetric filter has no lag, unlike the causal filters live previews use, which would delay the very
   peak the shot is synced to; the long one stills the landmarks' shake on a held pose (several px on a big hand); and
   both are cut at each strike, reaching at most one frame across, so a lock stays on its frame.

The hand's size (``scale``) is measured in 3D (MediaPipe's z), so a hand turning or reaching into the scene keeps its
size, and from the fingers and knuckles when the wrist is cut off by the frame (hand_size). A finger pointing into the
scene points at its vanishing point (_pointing).

Handedness is the hand as it looks in the clip: MediaPipe's left/right, which is the performer's own for a normal camera
and swapped in a mirrored selfie recording (``mirrored=True`` swaps it back).

Run from its own venv (MediaPipe brings its own OpenCV, which clashes with the core's), as the pipeline does
(cuesync.bending.prepare_hand, with ``cuesync/src`` on PYTHONPATH):
``python -m cuesync.hand_track plate.mp4 --emitter two_finger --seed frame,x,y --out run/hand_track.json``.
Setting that venv up (Python 3.12; from the repo root; Windows paths, ``bin/python`` elsewhere)::

    uv venv cuesync/.venv-track --python 3.12
    uv pip install --python cuesync/.venv-track/Scripts/python.exe mediapipe==1.1.1 onnxruntime==1.31.0 av==19.0.1
    # and for its tests: pytest

The two models (MediaPipe's gesture recognizer, RTMPose-m hand; both Apache-2.0, ~70 MB) download on first use into
``~/.cache/cuesync`` (or ``$CUESYNC_HAND_MODELS``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

VERSION = 1
EMITTERS = ("two_finger", "index_tip", "palm", "fist")
GESTURES = ("Open_Palm", "Closed_Fist", "Pointing_Up", "Victory", "Thumb_Up", "Thumb_Down", "ILoveYou", "None")

PROC_LONG_SIDE = 1280  # MediaPipe and the flow run on a copy this big; landmarks are scaled back to the clip
MP_CONF = 0.5  # MediaPipe's detection, presence and tracking thresholds (its defaults)
NUM_HANDS = 2
ASSOC = 1.2  # x hand size: how far from where it was heading a detection may be and still be the same hand
MAX_JUMP = 0.6  # x hand size: a frame whose emitter is further than this off its neighbours' path is an outlier
SHAPE_JUMP = 0.3  # x hand size: ...or whose landmarks are on average further than this off theirs
FLOW_MAX = 3  # frames: gaps up to this long are filled by optical flow
INTERP_MAX = 8  # frames: longer gaps (after flow and re-detection) are lost
RTM_MIN = 0.4  # RTMPose's mean keypoint score for a hand to count: hands score 0.45-0.6, empty boxes 0.1-0.34
RTM_CHAIN_MISSES = 2  # at the track's ends, stop chaining outward after this many frames in a row with no hand
SEED_RADIUS = 0.12  # x short side: a detection with a landmark this close to the Studio's click is the clicked hand
SEED_FRAMES = 12  # ...on the clicked frame or this many either side of it
SEED_WINDOW = 0.9  # x short side: the window of the second look around the click
CAM_MIN_POINTS = 20  # background corners that must agree before the camera counts as having moved
CAM_MIN_SHARE = 0.35  # a clip's camera counts as moving only when at least this share of its steps were measured
CAM_STRIKE = 0.25  # hand sizes a frame: faster than any handheld shake; a step matching such a hand is its arm
PUSH = 0.15  # x hand size: two-finger and fist emitters sit this far out past the fingers / knuckles
PALM_FACE = 0.2  # x hand size: from the knuckle row's centre out to the palm's face, along its normal
FINGER_LEN = (1.15, 1.25)  # index, middle finger (knuckle to tip, 3D) in wrist-to-middle-knuckle lengths
FOCAL = 0.7  # x the picture's long side: the focal length a phone's main camera has, for fingers into the scene
KNUCKLE_ROW = 0.85  # index to little knuckle (3D), same units
SG_WINDOW_24 = 7  # Savitzky-Golay window at 24 fps (scaled with fps, kept odd: 9 at 30)
SG_ORDER = 2
HOLD_WINDOW_24 = 15  # ...and while the hand holds still (0.6 s: longer starts to round off a slow drift)
FALLBACK_TURN = math.pi  # radians a second: guessed landmarks cannot turn as freely as a measured hand
STILL, MOVING = 0.5, 1.5  # hand sizes a second: slower holds (the long window), faster moves (the 7-frame one)
EXT_DROP = 0.3  # an arm has locked once its speed falls under this share of the peak...
EXT_WITHIN = 6  # ...within this many frames of it (at 24 fps)
PEAK_MIN = 3.0  # x hand size per second: a slower peak is a drift, not a strike
OPEN_FIST, OPEN_PALM = 0.8, 1.1  # fingertip spread (x hand size, in 3D) of a closed hand / an open one
REACH_OUT, REACH_IN = 1.4, 1.05  # a finger's tip over its middle joint, from the palm (3D): out past, in under

CONF = {"flow": 0.6, "tapir": 0.5, "interp": 0.3, "hold": 0.0, "lost": 0.0}  # mp and rtm carry their own scores

WRIST, PALM, MCP, TIPS = 0, (0, 5, 9, 13, 17), (5, 9, 13, 17), (8, 12, 16, 20)

MODELS_ENV = "CUESYNC_HAND_MODELS"
MODEL_FILES = {
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                               "gesture_recognizer/float16/latest/gesture_recognizer.task",
    "rtmpose-m_hand5_256x256.onnx": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                                    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip",
}


# ------------------------------------------------------------------ geometry (pure numpy: usable without MediaPipe)

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def hand_scale(lm: np.ndarray) -> np.ndarray:
    """Wrist (landmark 0) to the middle finger's knuckle (9) in the picture, px: the brief's measure, which shrinks as
    the hand turns away; used only to compare two estimates of the same frame (see hand_size for the hand's size)."""
    lm = np.asarray(lm, np.float64)
    return np.linalg.norm(lm[..., WRIST, :2] - lm[..., 9, :2], axis=-1)


def hand_size(lm: np.ndarray, frame: tuple[int, int] | None = None) -> np.ndarray:
    """The hand's size in px, in wrist-to-middle-knuckle lengths, measured in 3D (MediaPipe's z, in px like x) so that
    a hand turning, or pointing into the scene, doesn't shrink it: the median of four estimates, that length itself,
    the index and middle fingers' lengths (knuckle to tip, so curling them doesn't change it either) and the knuckle
    row's width, each over its usual share of it (FINGER_LEN, KNUCKLE_ROW; measured on the dev plates). With the
    clip's ``frame`` (W, H) given, the wrist's own estimate fades out as the wrist nears the picture's edge (from 0.25
    hand sizes inside it to 0.05) and is left out past that, cut off as in a first-person shot, where MediaPipe only
    guesses where it is; fading, not switching, so the size doesn't step as the wrist crosses."""
    lm = np.asarray(lm, np.float64)

    def seg(a, b):
        return np.linalg.norm(lm[..., a, :] - lm[..., b, :], axis=-1)

    est = np.stack([seg(WRIST, 9),
                    (seg(5, 6) + seg(6, 7) + seg(7, 8)) / FINGER_LEN[0],
                    (seg(9, 10) + seg(10, 11) + seg(11, 12)) / FINGER_LEN[1],
                    seg(5, 17) / KNUCKLE_ROW], -1)
    with_wrist = np.median(est, axis=-1)
    if frame is None:
        return with_wrist
    without = np.median(est[..., 1:], axis=-1)
    W, H = frame
    x, y = lm[..., WRIST, 0], lm[..., WRIST, 1]
    inside = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))  # px from the nearest edge, negative past it
    w = np.clip((inside - 0.05 * without) / (0.2 * np.maximum(without, 1e-9)), 0.0, 1.0)
    return w * with_wrist + (1 - w) * without


def inside_share(lm: np.ndarray, frame: tuple[int, int]) -> float:
    """The share of a hand's 21 landmarks inside the picture."""
    lm = np.asarray(lm, np.float64)
    W, H = frame
    return float(np.mean((lm[:, 0] >= 0) & (lm[:, 0] < W) & (lm[:, 1] >= 0) & (lm[:, 1] < H)))


def palm_centre(lm: np.ndarray) -> np.ndarray:
    return np.asarray(lm, np.float64)[..., PALM, :2].mean(-2)


def _pointing(tip: np.ndarray, knuckle: np.ndarray, frame: tuple[int, int] | None = None) -> np.ndarray:
    """Which way a finger points in the picture, from knuckle to tip in 3D (``[..., 3]``, MediaPipe's z in px). Across
    the picture that is just knuckle to tip. A finger pointing into the scene, away from a first-person camera, gets
    short in the picture and its direction there is noise, but it points at its vanishing point: through a camera of
    focal length f (FOCAL x the long side, a phone's main camera) that is f (dx, dy) / dz from the picture's centre,
    so the direction from the tip is f (dx, dy) + dz (centre - tip), which is knuckle-to-tip again when dz is 0. Only
    for a finger pointing away (dz > 0): one pointing toward the camera keeps the picture's own direction, as
    MediaPipe's z toward the camera is too unsure to bend it by (on A's plate it would tip a finger pointing left 36
    degrees up). Without ``frame`` (the clip's W, H) the picture's own direction is used."""
    d = tip - knuckle
    if frame is None or d.shape[-1] < 3:
        return _unit(d[..., :2])
    W, H = frame
    f = FOCAL * max(W, H)
    centre = np.array([W / 2.0, H / 2.0])
    return _unit(f * d[..., :2] + np.maximum(d[..., 2:3], 0.0) * (centre - tip[..., :2]))


def emitter_points(lm: np.ndarray, emitter: str, palm_sign: float = 1.0, frame: tuple[int, int] | None = None,
                   finger: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xy, dir, scale) of the emitter from landmarks ``[..., 21, 3]`` (px; z in px with the wrist at 0). ``scale`` is
    hand_size (``frame``, the clip's (W, H), tells it when the wrist is cut off). ``finger`` (knuckle, tip landmarks)
    puts a fingertip emitter on that finger instead (see clicked_finger).

    - ``index_tip``: the index fingertip (8), pointing from its knuckle (5) to the tip (_pointing).
    - ``two_finger``: between the index and middle fingertips (8, 12), pointing from their knuckles (5, 9), pushed
      0.15 x the hand's size out past the tips, where a jet would leave them.
    - ``fist``: the knuckles (5, 9, 13, 17), pointing from the wrist through them, pushed out the same way.
    - ``palm``: on the palm's face: the knuckle row's centre (5, 9, 13, 17), out along the palm's normal by PALM_FACE
      hand sizes as far as the palm faces across the picture. (The brief's mean of the wrist and the knuckles is a
      point inside the hand: seen edge-on, as a palm pushed sideways is, it lands on the heel side of the hand, not on
      the face the push comes out of.) It points out of the palm: the palm's normal as it shows in the picture, which
      leans with the hand (a palm pushed left with its fingers tipped back faces a little down). A palm facing the
      camera has no direction in the picture to speak of, so as it turns to face the camera the direction blends over
      to the fingers' (wrist to knuckles). ``palm_sign`` is +1 for a hand MediaPipe calls right, -1 for a left one
      (the two are mirror images, so the same cross product points the other way).

    The fingers point from the knuckle rather than the middle joint (the brief's): a longer, steadier baseline that
    agrees with it for a straight finger, and with perspective for one pointing into the scene (_pointing)."""
    lm = np.asarray(lm, np.float64)
    p2 = lm[..., :2]
    scale = hand_size(lm, frame)
    if finger is not None and emitter in ("index_tip", "two_finger"):
        knuckle, tip = finger
        return p2[..., tip, :], _pointing(lm[..., tip, :], lm[..., knuckle, :], frame), scale
    if emitter == "index_tip":
        d = _pointing(lm[..., 8, :], lm[..., 5, :], frame)
        xy = p2[..., 8, :]
    elif emitter == "two_finger":
        tip = (lm[..., 8, :] + lm[..., 12, :]) / 2
        d = _pointing(tip, (lm[..., 5, :] + lm[..., 9, :]) / 2, frame)
        xy = tip[..., :2] + PUSH * scale[..., None] * d
    elif emitter == "fist":
        knuckles = p2[..., MCP, :].mean(-2)
        d = _unit(knuckles - p2[..., WRIST, :])
        xy = knuckles + PUSH * scale[..., None] * d
    elif emitter == "palm":
        n = palm_sign * np.cross(lm[..., 5, :] - lm[..., WRIST, :], lm[..., 17, :] - lm[..., WRIST, :])
        n = _unit(n)
        side = np.linalg.norm(n[..., :2], axis=-1, keepdims=True)  # 1: the palm faces across the picture, 0: at us
        xy = p2[..., MCP, :].mean(-2) + PALM_FACE * scale[..., None] * n[..., :2]
        w = np.clip((side - 0.25) / 0.25, 0.0, 1.0)
        fingers = _unit(p2[..., MCP, :].mean(-2) - p2[..., WRIST, :])
        d = _unit(w * _unit(n[..., :2]) + (1 - w) * fingers)
    else:
        raise ValueError(f"unknown emitter {emitter!r}; one of {EMITTERS}")
    return xy, d, scale


def finger_spread(lm: np.ndarray) -> np.ndarray:
    """How far the fingertips are from the palm's centre, in hand sizes, in 3D (MediaPipe's z): about 0.6 for a fist,
    1.1-1.4 for an open hand. In the picture alone, fingers pointing at the camera would read as curled."""
    lm = np.asarray(lm, np.float64)
    c = lm[..., PALM, :].mean(-2)
    return np.linalg.norm(lm[..., TIPS, :] - c[..., None, :], axis=-1).mean(-1) / np.maximum(hand_size(lm), 1e-9)


FINGERS = {4: (2, 4), 8: (5, 8), 12: (9, 12), 16: (13, 16), 20: (17, 20)}  # tip: (knuckle, tip)


def clicked_finger(lm: np.ndarray, click, emitter: str) -> tuple[int, int] | None:
    """The finger a Studio click names, for a fingertip emitter: when the click on the hand (``lm``, the frame it was
    matched on) is on another fingertip than the emitter's own (within 0.35 hand sizes of it, and well nearer than
    the emitter's own tip), (knuckle, tip) of that finger; else None. MediaPipe doesn't always name the fingers of an
    odd pose the way a person would (a digit pointing out sideways read as the thumb, say), and the click is the
    performer's own say on which digit the element comes out of."""
    if emitter not in ("index_tip", "two_finger"):
        return None
    lm = np.asarray(lm, np.float64)
    c = np.asarray(click[-2:], np.float64)
    d = {k: float(np.linalg.norm(lm[k, :2] - c)) for k in FINGERS}
    own = [8] if emitter == "index_tip" else [8, 12]
    best = min(d, key=d.get)
    if best in own or d[best] > 0.35 * float(hand_size(lm)) or d[best] > 0.6 * min(d[o] for o in own):
        return None
    return FINGERS[best]


def finger_reach(lm: np.ndarray) -> np.ndarray:
    """``[..., 4]``: how much farther each finger's tip (index, middle, ring, little) is from the palm's centre than
    its middle joint, in 3D: about 1.5-1.9 straight out, under 1 curled into the palm."""
    lm = np.asarray(lm, np.float64)
    c = lm[..., PALM, :].mean(-2)[..., None, :]
    tips, pips = lm[..., [8, 12, 16, 20], :], lm[..., [6, 10, 14, 18], :]
    return np.linalg.norm(tips - c, axis=-1) / np.maximum(np.linalg.norm(pips - c, axis=-1), 1e-9)


def landmark_gesture(lm: np.ndarray) -> str:
    """MediaPipe's gesture vocabulary read off the landmarks, for the many frames its own classifier calls "None"
    (it was trained on hands facing the camera): Open_Palm with all four fingers out, Closed_Fist with all four in,
    Victory with the index and middle out and the others in, Pointing_Up with only the index out (whichever way it
    points). Anything less clear stays "None": a finger counts as out past REACH_OUT, in under REACH_IN, nothing in
    between, since MediaPipe's own finger landmarks are often wrong on the poses its classifier gives up on."""
    r = finger_reach(lm)
    out, tucked = r > REACH_OUT, r < REACH_IN
    if out.all():
        return "Open_Palm"
    if tucked.all():
        return "Closed_Fist"
    if out[0] and out[1] and tucked[2] and tucked[3]:
        return "Victory"
    if out[0] and tucked[1] and tucked[2] and tucked[3]:
        return "Pointing_Up"
    return "None"


def _out(lm: np.ndarray, tip: int) -> bool:
    """A digit straight out: its tip REACH_OUT times farther from the palm's centre than its joint two back."""
    c = lm[list(PALM), :].mean(0)
    return np.linalg.norm(lm[tip] - c) > REACH_OUT * max(np.linalg.norm(lm[tip - 2] - c), 1e-9)


def pointing_gesture(lm: np.ndarray, emitter: str, finger: tuple[int, int] | None = None) -> str | None:
    """For a fingertip emitter: Pointing_Up when the digit the element leaves (the emitter's index, or the digit the
    click named) is straight out, Victory when a two-finger emitter's index and middle both are; else None. The shot
    has said which digit fires and the hand holds it out: it is pointing it, whatever the other fingers do."""
    lm = np.asarray(lm, np.float64)
    if emitter == "two_finger" and finger is None:
        if _out(lm, 8) and _out(lm, 12):
            return "Victory"
        return "Pointing_Up" if _out(lm, 8) else None
    if emitter in ("index_tip", "two_finger"):
        return "Pointing_Up" if _out(lm, finger[1] if finger else 8) else None
    return None


# ------------------------------------------------------------------ zero-phase smoothing

def savgol(x: np.ndarray, window: int, order: int = SG_ORDER) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0: each sample replaced by a least-squares polynomial of ``order`` through
    the ``window`` samples centred on it (at the ends, through the first / last ``window``), so nothing is delayed.
    Shortened (kept odd) for a short run; a run shorter than ``order`` + 2 is returned as it is."""
    x = np.asarray(x, np.float64)
    n = len(x)
    window = min(int(window) | 1, n if n % 2 else n - 1)
    if window <= order + 1:
        return x.copy()
    h = window // 2
    a = np.vander(np.arange(-h, h + 1, dtype=np.float64), order + 1, increasing=True)  # [window, order+1]
    fit = np.linalg.pinv(a)  # polynomial coefficients from a window of samples
    out = np.empty_like(x)
    windows = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)  # [n-window+1, ..., window]
    out[h:n - h] = np.tensordot(windows, fit[0], axes=([-1], [0]))
    out[:h] = np.tensordot(a[:h] @ fit, x[:window], axes=([1], [0]))
    out[n - h:] = np.tensordot(a[h + 1:] @ fit, x[n - window:], axes=([1], [0]))
    return out


def smooth_pieces(valid: np.ndarray, cuts: list[int] | tuple = ()) -> list[tuple[int, int, int, int]]:
    """How the track is smoothed: runs of valid frames, cut at each strike. Returns (fit_lo, fit_hi, out_lo, out_hi):
    each piece is fitted on ``[fit_lo, fit_hi)`` and written to ``[out_lo, out_hi)``; a piece reaches one frame past a
    cut (never more), and nothing reaches past an invalid frame."""
    valid = np.asarray(valid, bool)
    pieces = []
    t, n = 0, len(valid)
    while t < n:
        if not valid[t]:
            t += 1
            continue
        a = t
        while t < n and valid[t]:
            t += 1
        b = t
        inner = sorted(c for c in set(int(c) for c in cuts) if a < c < b)
        edges = [a, *inner, b]
        for lo, hi in zip(edges[:-1], edges[1:]):
            pieces.append((lo - 1 if lo in inner else lo, hi + 1 if hi in inner else hi, lo, hi))
    return pieces


def smooth_track(x: np.ndarray, valid: np.ndarray, cuts=(), window: int = SG_WINDOW_24) -> np.ndarray:
    """``x`` (``[T, ...]``) smoothed piece by piece (see smooth_pieces); invalid frames are left as they are."""
    x = np.asarray(x, np.float64)
    out = x.copy()
    for fl, fh, ol, oh in smooth_pieces(valid, cuts):
        out[ol:oh] = savgol(x[fl:fh], window)[ol - fl:oh - fl]
    return out


def sg_window(fps: float, at_24: int = SG_WINDOW_24) -> int:
    return max(5, int(round(at_24 * fps / 24.0)) | 1)


def stillness(xy: np.ndarray, scale: np.ndarray, valid: np.ndarray, cuts, fps: float) -> np.ndarray:
    """0..1 per frame: 1 where the hand moves slower than STILL hand sizes a second, 0 faster than MOVING. What
    counts is the fastest it moves within reach of the hold window (on a 13-frame smoothed path, so the shake of the
    landmarks doesn't read as motion): a hold's heavy smoothing then never reaches into a move's start or end."""
    T = len(xy)
    path = smooth_track(xy, valid, cuts, sg_window(fps, 13))
    v = np.zeros(T)
    for fl, fh, ol, oh in smooth_pieces(valid, cuts):
        if oh - ol > 1:
            v[ol:oh] = np.linalg.norm(np.gradient(path[ol:oh], axis=0), axis=-1) * fps / np.maximum(scale[ol:oh], 1.0)
    reach = sg_window(fps, HOLD_WINDOW_24) // 2
    fastest = np.array([v[max(0, t - reach):t + reach + 1].max() for t in range(T)]) if T else v
    return np.clip((MOVING - fastest) / (MOVING - STILL), 0.0, 1.0)


def adaptive_smooth(x: np.ndarray, still: np.ndarray, valid: np.ndarray, cuts, fps: float) -> np.ndarray:
    """``x`` smoothed over 7 frames where the hand moves and over HOLD_WINDOW_24 (about a second) where it holds,
    blended by ``still`` (see stillness). Both are symmetric, so neither delays anything; the short one keeps up with a
    strike, the long one stills the landmarks' shake on a held pose (several px on a big hand)."""
    x = np.asarray(x, np.float64)
    short = smooth_track(x, valid, cuts, sg_window(fps))
    long = smooth_track(x, valid, cuts, sg_window(fps, HOLD_WINDOW_24))
    w = np.asarray(still, np.float64).reshape((-1,) + (1,) * (x.ndim - 1))
    return w * long + (1 - w) * short


def steady_direction(angle: np.ndarray, src: list[str], fps: float) -> np.ndarray:
    """Limit fallback turns from the last measured direction, in scene radians. RTMPose and flow can place the
    emitter well while blurring the knuckle-to-tip vector beyond use; after smoothing, allow at most half a turn a
    second until MediaPipe sees it again. Its measured turns stay untouched. Before the first measurement, work
    backward from it; never carry a direction across a lost stretch or constrain a run with no measured direction."""
    out = np.asarray(angle, np.float64).copy()
    valid = np.array([s not in ("hold", "lost") for s in src])
    for lo, hi, _, _ in smooth_pieces(valid):
        measured = [t for t in range(lo, hi) if src[t] == "mp"]
        if not measured:
            continue
        first = measured[0]
        for order, step in ((range(first - 1, lo - 1, -1), -1), (range(first + 1, hi), 1)):
            for t in order:
                if src[t] == "mp":
                    continue
                prev = out[t - step]
                delta = (out[t] - prev + np.pi) % (2 * np.pi) - np.pi
                out[t] = prev + np.clip(delta, -FALLBACK_TURN / fps, FALLBACK_TURN / fps)
    return out


# ------------------------------------------------------------------ strikes

def frame_speed(xy: np.ndarray, valid: np.ndarray, fps: float) -> np.ndarray:
    """px/s with which the emitter arrived at each frame (from the frame before); 0 where either frame is invalid."""
    xy = np.asarray(xy, np.float64)
    v = np.zeros(len(xy))
    ok = np.asarray(valid, bool)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps
    both = ok[1:] & ok[:-1]
    v[1:][both] = step[both]
    return v


def find_extensions(speed: np.ndarray, scale: float, fps: float, valid=None) -> list[dict]:
    """Arms locking: a peak of speed (the fastest within +-3 frames, and at least PEAK_MIN hand sizes a second),
    then the first frame within EXT_WITHIN frames where the speed has fallen under EXT_DROP x the peak. The frame
    found is the first one with the arm still: the one before it is where the strike lands. Only tracked frames
    (``valid``) can be that frame: a hand that is lost hasn't stopped."""
    speed = np.asarray(speed, np.float64)
    ok = np.ones(len(speed), bool) if valid is None else np.asarray(valid, bool)
    ok = ok & np.r_[False, ok[:-1]]  # its speed came from two tracked frames
    within = max(2, int(round(EXT_WITHIN * fps / 24.0)))
    reach = max(1, int(round(3 * fps / 24.0)))
    floor = PEAK_MIN * scale
    events, t, n = [], 1, len(speed)
    while t < n:
        lo, hi = max(0, t - reach), min(n, t + reach + 1)
        if speed[t] >= floor and speed[t] == speed[lo:hi].max():
            peak = float(speed[t])
            stop = next((k for k in range(t + 1, min(n, t + within + 1)) if ok[k] and speed[k] < EXT_DROP * peak),
                        None)
            if stop is not None:
                events.append({"frame": int(stop), "kind": "extension", "peak_speed": round(peak, 1)})
                t = stop + 1
                continue
        t += 1
    return events


def find_opens(spread: np.ndarray, gestures: list[str], valid: np.ndarray, speed: np.ndarray, fps: float) -> list[dict]:
    """Palms opening: the fingertips' spread rising from a fist's (under OPEN_FIST hand sizes, within the 6 frames
    before) to an open hand's (over OPEN_PALM within the 4 after), at the frame it crosses halfway; or MediaPipe's
    gesture going from Closed_Fist to Open_Palm in as short a time. ``peak_speed`` is the fastest the hand moved
    within 3 frames of it."""
    k = fps / 24.0
    before, after, near = max(2, round(6 * k)), max(2, round(4 * k)), max(1, round(3 * k))
    spread = np.where(np.asarray(valid, bool), np.asarray(spread, np.float64), np.nan)
    mid = (OPEN_FIST + OPEN_PALM) / 2
    n = len(spread)
    found = []
    for t in range(1, n):
        if spread[t] >= mid > spread[t - 1] and all(t - f > before for f in found):  # one opening, not its wobble
            lo, hi = spread[max(0, t - before):t], spread[t:min(n, t + after + 1)]
            if np.nanmin(lo, initial=np.inf) < OPEN_FIST and np.nanmax(hi, initial=-np.inf) > OPEN_PALM:
                found.append(t)
    ok = np.asarray(valid, bool)
    for t in range(1, n):  # a fist held for 2 of the frames before, an open palm for 3 of the 4 after
        if ok[t] and gestures[t] == "Open_Palm" and gestures[t - 1] != "Open_Palm" and \
                gestures[max(0, t - before):t].count("Closed_Fist") >= 2 and \
                gestures[t:t + 4].count("Open_Palm") >= 3:
            if all(abs(t - f) > near for f in found):
                found.append(t)
    events = []
    for t in sorted(found):
        v = float(np.max(speed[max(0, t - near):min(n, t + near + 1)]))
        events.append({"frame": int(t), "kind": "open", "peak_speed": round(v, 1)})
    return events


# ------------------------------------------------------------------ models

def models_dir() -> Path:
    """$CUESYNC_HAND_MODELS, else ~/.cache/cuesync (where cuesync.hand_mask keeps its MediaPipe model too)."""
    return Path(os.environ.get(MODELS_ENV) or Path.home() / ".cache" / "cuesync")


def model_path(name: str) -> Path:
    """A model file, downloaded on first use (MediaPipe's from Google, RTMPose's from OpenMMLab; both Apache-2.0)."""
    path = models_dir() / name
    if path.exists():
        return path
    import io
    import urllib.request
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_FILES[name]
    print(f"hand_track: downloading {url}", file=sys.stderr)
    data = urllib.request.urlopen(url, timeout=120).read()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith("end2end.onnx")))
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


# ------------------------------------------------------------------ the clip

def _proc_size(w: int, h: int) -> tuple[int, int]:
    s = min(1.0, PROC_LONG_SIDE / max(w, h))
    return max(16, int(round(w * s))), max(16, int(round(h * s)))


class Clip:
    """The clip as the tracker reads it: its size and rate, and each frame at the working size (long side
    PROC_LONG_SIDE). From a file, frames are decoded one at a time, scaled to that size and kept JPEG-packed, so a
    10 s 1080p clip costs about 40 MB instead of the 1.5 GB it decodes to; they are unpacked again only where
    re-detection or flow needs them. Given frames, it reads them where they are."""

    def __init__(self, video, fps: float | None = None):
        if isinstance(video, (str, Path)):
            from .media import probe_video

            info = probe_video(video)
            self.path, self.arr = Path(video), None
            self.W, self.H = info.width, info.height
            self.fps = float(fps or info.fps or 24.0)
        else:
            self.path, self.arr = None, np.asarray(video)
            if self.arr.ndim != 4 or self.arr.shape[-1] != 3 or self.arr.dtype != np.uint8:
                raise ValueError("frames must be RGB uint8 [T, H, W, 3]")
            self.H, self.W = self.arr.shape[1:3]
            self.fps = float(fps or 24.0)
        self.pw, self.ph = _proc_size(self.W, self.H)
        self.s = np.array([self.pw / self.W, self.ph / self.H])  # full-size px -> working px
        self._packed: list[bytes] = []
        self._gray: dict[int, np.ndarray] = {}

    @property
    def n(self) -> int:
        if self.arr is not None:
            return len(self.arr)
        if not self._packed:
            self.pack()
        return len(self._packed)

    def frames(self):
        """Every frame at the working size, in order (RGB uint8)."""
        if self.arr is not None:
            for t in range(len(self.arr)):
                yield self.small(t)
        elif self._packed:
            for t in range(len(self._packed)):
                yield self.small(t)
        else:
            yield from self._decode()

    def pack(self) -> None:
        for _ in self.frames():
            pass

    def _decode(self):
        from concurrent.futures import ThreadPoolExecutor

        import av
        import cv2

        from .media import _hdr_kind, _rotation_k, hdr_to_sdr

        def pack(img):
            return cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

        # Packing runs on a worker thread (OpenCV lets go of the GIL), alongside whatever consumes the frames.
        with av.open(str(self.path)) as container, ThreadPoolExecutor(1) as packer:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            hdr, pending = None, []
            for i, frame in enumerate(container.decode(stream)):
                if i == 0:
                    hdr = _hdr_kind(frame, stream)
                k = _rotation_k(frame)
                # Full size first, then down with OpenCV: scaled inside the decoder the colours come out a shade
                # different (another YUV matrix), and that was enough for MediaPipe to lose a dark hand at the
                # picture's edge on every frame.
                img = frame.to_ndarray(format="rgb48le" if hdr else "rgb24")
                if hdr:
                    img = hdr_to_sdr(img, hdr)
                if k:
                    img = np.ascontiguousarray(np.rot90(img, k))
                if img.shape[1::-1] != (self.pw, self.ph):
                    img = cv2.resize(img, (self.pw, self.ph), interpolation=cv2.INTER_AREA)
                pending.append(packer.submit(pack, img))
                yield img
            self._packed = [f.result() for f in pending]

    def small(self, t: int) -> np.ndarray:
        import cv2

        if self.arr is not None:
            f = self.arr[t]
            return f if f.shape[1::-1] == (self.pw, self.ph) else cv2.resize(f, (self.pw, self.ph),
                                                                            interpolation=cv2.INTER_AREA)
        if not self._packed:
            self.pack()
        return np.ascontiguousarray(cv2.imdecode(np.frombuffer(self._packed[t], np.uint8), cv2.IMREAD_COLOR)[..., ::-1])

    def gray(self, t: int) -> np.ndarray:
        import cv2

        if t not in self._gray:
            self._gray[t] = cv2.cvtColor(self.small(t), cv2.COLOR_RGB2GRAY)
        return self._gray[t]


# ------------------------------------------------------------------ detection

def mediapipe_hands(clip: Clip) -> list[list[dict]]:
    """Per frame, every hand MediaPipe sees: {"lm": [21, 3] px, "label": "Left" | "Right", "score", "gesture"}.
    VIDEO mode, so each hand's crop is carried on from the frame before; up to NUM_HANDS hands; run on the working
    size. Landmarks are in the clip's px; z is in px like x, relative to the wrist (smaller = nearer the camera)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not isinstance(clip, Clip):
        clip = Clip(clip)
    W, H = clip.W, clip.H
    opts = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path("gesture_recognizer.task"))),
        running_mode=vision.RunningMode.VIDEO, num_hands=NUM_HANDS, min_hand_detection_confidence=MP_CONF,
        min_hand_presence_confidence=MP_CONF, min_tracking_confidence=MP_CONF)
    out = []
    with vision.GestureRecognizer.create_from_options(opts) as rec:
        for t, small in enumerate(clip.frames()):
            r = rec.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small)),
                                        int(round(t * 1000.0 / clip.fps)))
            hands = []
            for k, lms in enumerate(r.hand_landmarks):
                lm = np.array([[p.x * W, p.y * H, p.z * W] for p in lms], np.float64)
                cat = r.handedness[k][0]
                g = r.gestures[k][0].category_name if r.gestures and r.gestures[k] else "None"
                hands.append({"lm": lm, "label": cat.category_name, "score": float(cat.score),
                              "gesture": g if g in GESTURES else "None"})
            out.append(_dedupe(hands))
    return out


def seeded_look(clip: Clip, seed: tuple[int, float, float]) -> list[list[dict]]:
    """MediaPipe again, for the hand the Studio's click points at, on a window around it (SEED_WINDOW x the short
    side, at the working size, black past the picture's edge), the window following the hand forward from the clicked
    frame and back from it. A hand cut by the frame's edge, or small and dark, is often missed by the full-frame pass
    (its palm detector wants the whole palm in the picture) and found like this. Per frame, the one hand nearest the
    window's centre if any, in the clip's px."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    T, (sx_, sy_) = clip.n, clip.s
    f0, x, y = seed
    f0 = int(np.clip(f0, 0, T - 1))
    side = int(SEED_WINDOW * min(clip.pw, clip.ph))  # working px
    out: list[list[dict]] = [[] for _ in range(T)]
    opts = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path("gesture_recognizer.task"))),
        running_mode=vision.RunningMode.VIDEO, num_hands=NUM_HANDS, min_hand_detection_confidence=MP_CONF,
        min_hand_presence_confidence=MP_CONF, min_tracking_confidence=MP_CONF)
    for order in (range(f0, T), range(f0, -1, -1)):
        centre = np.array([x * sx_, y * sy_])  # working px
        misses = 0
        with vision.GestureRecognizer.create_from_options(opts) as rec:
            for k, t in enumerate(order):
                x0, y0 = centre - side / 2
                win = cv2.warpAffine(clip.small(t), np.float32([[1, 0, -x0], [0, 1, -y0]]), (side, side),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                r = rec.recognize_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(win)),
                                            int(round(k * 1000.0 / clip.fps)))
                best = None
                for j, lms in enumerate(r.hand_landmarks):
                    lm = np.array([[(p.x * side + x0) / sx_, (p.y * side + y0) / sy_, p.z * side / sx_] for p in lms])
                    d = np.linalg.norm(palm_centre(lm) * clip.s - centre)
                    if d < 0.5 * side and (best is None or d < best[0]):
                        cat = r.handedness[j][0]
                        g = r.gestures[j][0].category_name if r.gestures and r.gestures[j] else "None"
                        best = (d, {"lm": lm, "label": cat.category_name, "score": float(cat.score),
                                    "gesture": g if g in GESTURES else "None"})
                if best is not None:
                    out[t] = [best[1]]
                    centre = 0.5 * centre + 0.5 * palm_centre(best[1]["lm"]) * clip.s
                    misses = 0
                else:
                    misses += 1
                    if misses > 2 * SEED_FRAMES:  # gone: the window stops chasing it
                        break
    return out


def _dedupe(hands: list[dict]) -> list[dict]:
    """MediaPipe sometimes reports one hand twice: of two the same size with their palms on top of each other, keep
    the more confident. (A smaller, partial copy is left in: following the hand by size and place sorts it out.)"""
    keep = []
    for h in sorted(hands, key=lambda h: -h["score"]):
        c, s = palm_centre(h["lm"]), hand_scale(h["lm"])
        if all(np.linalg.norm(c - palm_centre(k["lm"])) > 0.25 * max(s, hand_scale(k["lm"]))
               or not 0.8 < s / max(hand_scale(k["lm"]), 1e-9) < 1.25 for k in keep):
            keep.append(h)
    return keep


class _RTMPose:
    """RTMPose-m hand (OpenMMLab's ONNX export, Apache-2.0), run with onnxruntime on the CPU: the 21 hand keypoints, in
    MediaPipe's order, inside a box. The model sees the square around the box's centre, 1.25 x its longer side, at
    256 x 256 (RGB, as it was trained: its pipeline turns its BGR reads to RGB), and answers each keypoint's x and y as
    two 1D distributions at half-pixel steps (SimCC); a keypoint's score is the mean of their peaks. This is what
    rtmlib does, without rtmlib's download and progress-bar dependencies."""

    SIZE, PAD, SPLIT = 256, 1.25, 2.0
    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path("rtmpose-m_hand5_256x256.onnx")),
                                            providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, clip: Clip, t: int, box) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (clip px) and their scores in ``box`` (x0, y0, x1, y1, clip px) on frame ``t``."""
        import cv2

        b = np.asarray(box, np.float64) * np.tile(clip.s, 2)  # working px
        c = (b[:2] + b[2:]) / 2
        k = self.SIZE / max(self.PAD * max(b[2] - b[0], b[3] - b[1]), 1.0)  # model px per working px
        warp = np.array([[k, 0, self.SIZE / 2 - k * c[0]], [0, k, self.SIZE / 2 - k * c[1]]], np.float32)
        crop = cv2.warpAffine(clip.small(t), warp, (self.SIZE, self.SIZE), flags=cv2.INTER_LINEAR)
        x = ((crop.astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        sx, sy = self.session.run(None, {self.input: np.ascontiguousarray(x)})
        loc = np.stack([sx[0].argmax(-1), sy[0].argmax(-1)], -1) / self.SPLIT
        score = (sx[0].max(-1) + sy[0].max(-1)) / 2
        kp = (loc - self.SIZE / 2) / k + c
        return kp / clip.s, score.astype(np.float64)


def _box(lm: np.ndarray, pad: float = 0.15) -> np.ndarray:
    lo, hi = lm[:, :2].min(0), lm[:, :2].max(0)
    m = pad * max(hi - lo)
    return np.array([lo[0] - m, lo[1] - m, hi[0] + m, hi[1] + m])


# ------------------------------------------------------------------ the camera's own motion

class Camera:
    """The camera's own motion through a clip: for each frame, the similarity (shift, turn, zoom) taking the scene's
    coordinates (frame 0's pixels) to that frame's pixels. All identity for a camera that holds still."""

    def __init__(self, P: np.ndarray):
        self.P = np.asarray(P, np.float64)  # [T, 2, 3]
        A = self.P[:, :, :2]
        inv = np.linalg.inv(A)
        self.Q = np.concatenate([inv, -(inv @ self.P[:, :, 2:])], -1)  # frame px -> scene

    @classmethod
    def still(cls, T: int) -> "Camera":
        return cls(np.tile(np.eye(2, 3), (T, 1, 1)))

    def zoom(self) -> np.ndarray:
        return np.hypot(self.P[:, 0, 0], self.P[:, 1, 0])

    def turn(self) -> np.ndarray:
        """Radians the frame is turned from the scene (positive: clockwise on screen, as y points down)."""
        return np.arctan2(self.P[:, 1, 0], self.P[:, 0, 0])

    def to_frame(self, pts, t: int | None = None) -> np.ndarray:
        """Scene points into frame px: ``pts`` ``[T, ..., 2 or 3]`` (every frame), or ``[..., 2 or 3]`` on frame
        ``t``. A third coordinate (MediaPipe's z, in px) scales with the zoom."""
        return self._apply(self.P, pts, t)

    def to_scene(self, pts, t: int | None = None) -> np.ndarray:
        return self._apply(self.Q, pts, t)

    @staticmethod
    def _apply(M: np.ndarray, pts, t) -> np.ndarray:
        pts = np.asarray(pts, np.float64)
        M = M[t] if t is not None else M.reshape(M.shape[:1] + (1,) * (pts.ndim - 2) + (2, 3))
        out = pts.copy()
        out[..., :2] = np.einsum("...ij,...j->...i", M[..., :2], pts[..., :2]) + M[..., 2]
        if pts.shape[-1] > 2:
            out[..., 2] = pts[..., 2] * np.hypot(M[..., 0, 0], M[..., 1, 0])
        return out


def _hand_mask(dets: list[list[dict]], t: int, w: int, h: int, k: float) -> np.ndarray:
    """255 where the background is, at the analysis size: every hand of frames t-1 and t (or the nearest frames that
    saw one, within 4) masked off, its outline grown by 0.25 sizes, and a band out past the wrist for 4 sizes (the
    forearm and the arm after it: they move with the hand, and would pass for the camera moving)."""
    import cv2

    hands = [hd for f in (t - 1, t) if 0 <= f < len(dets) for hd in dets[f]]
    for off in range(1, 5):
        if hands:
            break
        hands = [hd for f in (t - 1 - off, t + off) if 0 <= f < len(dets) for hd in dets[f]]
    blocked = np.zeros((h, w), np.uint8)
    for hd in hands:
        lm = hd["lm"][:, :2] * k
        size = float(hand_size(hd["lm"])) * k
        u = _unit(lm[WRIST] - lm[9])
        n = np.array([-u[1], u[0]])
        arm = [lm[WRIST] + 0.6 * size * n, lm[WRIST] - 0.6 * size * n,
               lm[WRIST] + 4 * size * u - 0.9 * size * n, lm[WRIST] + 4 * size * u + 0.9 * size * n]
        one = np.zeros_like(blocked)
        hull = cv2.convexHull(np.concatenate([lm, np.array(arm)]).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(one, hull, 255)
        r = max(1, int(0.25 * size))
        blocked |= cv2.dilate(one, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    mask = 255 - blocked
    mask[:4], mask[-4:], mask[:, :4], mask[:, -4:] = 0, 0, 0, 0
    return mask


def _hand_steps(dets: list[list[dict]], t: int, k: float) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Each hand seen on frames t-1 and t: (palm centre before, after, size), at the analysis scale ``k``."""
    out = []
    for h1 in dets[t] if t < len(dets) else []:
        c1, s1 = palm_centre(h1["lm"]), float(hand_size(h1["lm"]))
        near = [(np.linalg.norm(palm_centre(h0["lm"]) - c1), h0) for h0 in dets[t - 1]]
        if near:
            d, h0 = min(near, key=lambda x: x[0])
            if d < s1:
                out.append((palm_centre(h0["lm"]) * k, c1 * k, s1 * k))
    return out


def _camera_step(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, hands=(), explain: float = 0.75) -> np.ndarray | None:
    """The similarity (2x3, analysis px) carrying frame ``prev`` onto ``cur``, from corners on the background; None
    when the background doesn't say: too few corners agree, or they bunch in a sliver of the picture; the step is
    implausible for a camera; warping by it doesn't line the background up better than standing still does; or,
    while a hand strikes (moves more than CAM_STRIKE sizes in the frame; ``hands``: (palm before, palm after, size),
    analysis px), it moves the hand just as the hand moved: that is the arm swinging with it, and counted as the
    camera it would hide the strike. (A hand held still in the room under a shaking camera moves exactly like the
    camera too, but slowly; that one is the camera, and the case the compensation is for.) ``explain``: how much
    better than standing still the warped frame must match the next (its error at most this share of standing still's)."""
    import cv2

    h, w = prev.shape
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=7, mask=mask, blockSize=7)
    if pts is None or len(pts) < CAM_MIN_POINTS:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **lk)
    ok = st.ravel().astype(bool) & st2.ravel().astype(bool) & (np.linalg.norm((back - pts)[:, 0], axis=1) < 0.7)
    if ok.sum() < CAM_MIN_POINTS:
        return None
    M, inl = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                         maxIters=2000, confidence=0.995)
    if M is None or inl is None or inl.sum() < CAM_MIN_POINTS or inl.mean() < 0.5:
        return None
    agree = pts[ok][inl.ravel().astype(bool), 0]
    lo, hi = np.percentile(agree, [5, 95], axis=0)
    if hi[0] - lo[0] < 0.2 * w or hi[1] - lo[1] < 0.15 * h:
        return None
    for before, after, size in hands:
        moved_by = np.linalg.norm(after - before)
        if moved_by > CAM_STRIKE * size and np.linalg.norm(M[:, :2] @ before + M[:, 2] - after) < 0.3 * moved_by:
            return None
    zoom, turn = np.hypot(M[0, 0], M[1, 0]), np.arctan2(M[1, 0], M[0, 0])
    if np.hypot(*M[:, 2]) > 0.2 * prev.shape[1] or abs(turn) > np.radians(6) or abs(zoom - 1) > 0.06:
        return None
    if np.hypot(*M[:, 2]) < 0.15 and abs(turn) < 3e-4 and abs(zoom - 1) < 3e-4:
        return None  # the corners' own shake, not the camera's
    # The step has to explain the background's texture: blurred past the sensor's grain (on a plain backdrop the
    # best "corners" are grain, and any warp smooths grain), the frame warped by it must match the next one clearly
    # better than the frame left where it was.
    p2, c2 = cv2.GaussianBlur(prev, (0, 0), 2.0), cv2.GaussianBlur(cur, (0, 0), 2.0)
    edges = (cv2.morphologyEx(p2, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 6) & (mask > 0)
    if edges.sum() < 300:
        return None
    moved = cv2.warpAffine(p2, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    still = np.abs(c2.astype(np.int16) - p2)[edges].mean()
    warped = np.abs(c2.astype(np.int16) - moved)[edges].mean()
    return M if warped < explain * still else None


def camera_motion(clip: Clip, dets: list[list[dict]]) -> Camera:
    """How the camera itself moved (a Camera): corners found on everything but the hands (each hand's landmarks, and
    its forearm out past the wrist, masked off; at half the working size, lightly blurred), followed to the next
    frame by optical flow
    and back again (a corner that doesn't come back to where it started is dropped), and a similarity fitted to them
    with RANSAC. A step where fewer than CAM_MIN_POINTS corners agree, or that would be implausible for a camera (a
    fifth of the frame, 6 degrees or 6% of zoom in one frame), counts as the camera holding still: a plain backdrop
    can't tell. And when fewer than CAM_MIN_SHARE of a clip's steps could be measured at all, the camera is taken to
    have held still throughout: a camera seen to move only now and then is likelier an arm or a body moving. In a
    clip whose camera does move, a step that couldn't be measured is drawn between the measured ones either side."""
    import cv2

    T = clip.n
    w, h = max(16, clip.pw // 2), max(16, clip.ph // 2)
    k = w / clip.W  # clip px -> analysis px
    steps = np.full((T, 4), np.nan)  # per step t-1 -> t: dx, dy (clip px), turn (rad), log zoom
    prev = None
    for t in range(T):
        g = cv2.resize(cv2.cvtColor(clip.small(t), cv2.COLOR_RGB2GRAY), (w, h), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), 1.2)  # the grain out, the scene's texture kept
        if prev is not None:
            M = _camera_step(prev, g, _hand_mask(dets, t, w, h, k), _hand_steps(dets, t, k))
            if M is not None:
                steps[t] = [*(M[:, 2] / k), np.arctan2(M[1, 0], M[0, 0]), np.log(np.hypot(M[0, 0], M[1, 0]))]
        prev = g
    have = np.flatnonzero(~np.isnan(steps[:, 0]))
    if T < 2 or len(have) < CAM_MIN_SHARE * (T - 1):
        return Camera.still(T)
    # A step that couldn't be measured (no hand seen to mask off, say) is drawn between the measured ones either
    # side; before the first and after the last, the camera holds still.
    for j in range(4):
        steps[1:, j] = np.interp(np.arange(1, T), have, steps[have, j], left=0.0, right=0.0)
    P = np.tile(np.eye(3), (T, 1, 1))
    for t in range(1, T):
        dx, dy, a, lz = steps[t]
        z = np.exp(lz)
        step = np.array([[z * np.cos(a), -z * np.sin(a), 0], [z * np.sin(a), z * np.cos(a), 0], [0, 0, 1]])
        step[:2, 2] = [dx, dy]
        P[t] = step @ P[t - 1]
    return Camera(P[:, :2, :])


# ------------------------------------------------------------------ following one hand

def _seed_px(seed, W: int, H: int) -> tuple[int, float, float] | None:
    if seed is None:
        return None
    f, x, y = seed
    if 0 <= x <= 1 and 0 <= y <= 1:  # fractions of the frame, as the Studio's aim gives them
        x, y = x * W, y * H
    return int(f), float(x), float(y)


def _pick_anchor(dets: list[list[dict]], hand: str, seed, mirrored: bool,
                 frame: tuple[int, int] | None = None) -> tuple[int, int] | None:
    """(frame, index) of the detection the track starts from: with a seed, a hand within SEED_RADIUS of the click
    within SEED_FRAMES of its frame (else None: no hand there). With no seed and no hand chosen, the hand most in view
    over the clip: how much of it is in the picture, times its size, summed over the frames it is seen in. In a
    first-person shot that is the hand doing the casting, not the one holding the phone (out of the picture, or a
    thumb at its edge)."""
    T = len(dets)
    if seed is not None:  # a hand near the click, on the clicked frame or the nearest frame that has one
        f0, sx, sy = seed
        reach = SEED_RADIUS * min(frame) if frame else np.inf
        for off in range(0, SEED_FRAMES + 1):
            for f in (f0 - off, f0 + off):
                if 0 <= f < T and dets[f]:
                    d = [np.min(np.hypot(h["lm"][:, 0] - sx, h["lm"][:, 1] - sy)) for h in dets[f]]
                    if min(d) < reach:
                        return f, int(np.argmin(d))
        return None
    if hand in ("left", "right"):
        want = {"left": "Left", "right": "Right"}[hand]
        if mirrored:
            want = "Left" if want == "Right" else "Right"
        labels = [want]
    else:
        def seen(h):
            return (inside_share(h["lm"], frame) if frame else 1.0) * float(hand_size(h["lm"]))

        score = {lab: sum(seen(h) for d in dets for h in d if h["label"] == lab) for lab in ("Left", "Right")}
        labels = sorted(score, key=lambda lab: -score[lab])
    for lab in labels:
        have = np.array([any(h["label"] == lab for h in d) for d in dets])
        if not have.any():
            continue
        best, run, start = (0, 0), 0, 0  # the middle of the longest run of frames seeing it
        for t in range(T + 1):
            if t < T and have[t]:
                run, start = run + 1, start if run else t
            else:
                if run > best[1]:
                    best = (start, run)
                run = 0
        f = best[0] + best[1] // 2
        k = max((i for i, h in enumerate(dets[f]) if h["label"] == lab),
                key=lambda i: (inside_share(dets[f][i]["lm"], frame) if frame else 1.0) * hand_size(dets[f][i]["lm"]))
        return f, k
    return None


def follow_hand(dets: list[list[dict]], anchor: tuple[int, int]) -> list[dict | None]:
    """From the anchor, forward then backward: each frame takes the detection nearest where the hand was heading, if
    it is within ASSOC hand sizes of it, preferring the same handedness and a size within a third of the recent one,
    and never one 1.8x bigger or smaller (a false detection, often a fragment of the hand). Where it is heading: on at
    the speed between the last two frames it was seen in a row, for at most two frames; while it goes unseen the
    allowance grows. Sizes are the median of the last five, so one bad detection can't shift the yardstick."""
    T = len(dets)
    f0, k0 = anchor
    chosen: list[dict | None] = [None] * T
    chosen[f0] = dets[f0][k0]
    label = dets[f0][k0]["label"]
    for step in (1, -1):
        seen = [(f0, palm_centre(chosen[f0]["lm"]), hand_size(chosen[f0]["lm"]))]
        t = f0 + step
        while 0 <= t < T:
            tl, cl, _ = seen[-1]
            size = float(np.median([s for _, _, s in seen[-5:]]))
            pred = cl
            if len(seen) > 1 and abs(tl - seen[-2][0]) == 1:
                vel = cl - seen[-2][1]
                vel *= min(1.0, 1.5 * size / max(np.linalg.norm(vel), 1e-9))  # at most 1.5 sizes a frame
                pred = cl + vel * min(abs(t - tl), 2)
            allow = ASSOC * (1 + 0.5 * min(abs(t - tl) - 1, 4))
            best, cost = None, np.inf
            for h in dets[t]:
                grew = abs(math.log(max(hand_size(h["lm"]), 1e-6) / size))
                if grew > math.log(1.8):  # a hand doesn't double or halve between sightings; a false detection does
                    continue
                c = (np.linalg.norm(palm_centre(h["lm"]) - pred) / size + (0.5 if h["label"] != label else 0.0)
                     + 2.0 * max(0.0, grew - math.log(1.35)))
                if c < cost:
                    best, cost = h, c
            if best is not None and cost < allow:
                chosen[t] = best
                seen.append((t, palm_centre(best["lm"]), hand_size(best["lm"])))
            t += step
    return chosen


def gate(chosen: list[dict | None], emitter: str, finger: tuple[int, int] | None = None) -> tuple[np.ndarray, str]:
    """Which followed frames are good (see the module's step 2), and the track's handedness (MediaPipe's label)."""
    T = len(chosen)
    labels = [c["label"] for c in chosen if c is not None]
    label = max(set(labels), key=labels.count) if labels else "Right"
    # The other hand's label is kept where it is the same hand followed on: in a run of frames followed one after
    # another that also has frames with this label (MediaPipe calls an upturned palm the other hand; dropped for that,
    # a palm carried through a house was lost for 66 of its 161 frames). A run with only the other label is the other
    # hand, or a false find taken up across a gap (one 41 frames off, kept, put a finger gun's track on the floor).
    good = np.array([c is not None and c["label"] == label for c in chosen])
    t = 0
    while t < T:
        if chosen[t] is None:
            t += 1
            continue
        e = t
        while e < T and chosen[e] is not None:
            e += 1
        if good[t:e].any():
            good[t:e] = True
        t = e
    sign = 1.0 if label == "Right" else -1.0
    pts = np.full((T, 2, 2), np.nan)
    scale = np.full(T, np.nan)
    for t in np.flatnonzero(good):
        lm = chosen[t]["lm"]
        pts[t] = [emitter_points(lm, emitter, sign, finger=finger)[0], palm_centre(lm)]
        # jumps are in the picture, so the yardstick is the hand's size there; floored at half its true size so a
        # hand turned edge-on doesn't make every frame look like a jump
        scale[t] = max(float(hand_scale(lm)), 0.5 * float(hand_size(lm)))
    lms = np.stack([c["lm"][:, :2] if c is not None else np.full((21, 2), np.nan) for c in chosen])
    bad = []
    for t in np.flatnonzero(good):
        near = [k for k in range(max(0, t - 2), min(T, t + 3)) if good[k]]
        if len(near) < 3:
            continue
        med = np.median(pts[near], axis=0)
        ref = np.median(scale[near])
        # and the whole hand: its landmarks on average within SHAPE_JUMP sizes of their medians (a first detection
        # placed off the hand moves every landmark at once; a big hand's 0.6 sizes is a lot of pixels)
        shape = np.linalg.norm(lms[t] - np.median(lms[near], axis=0), axis=-1).mean()
        if np.any(np.linalg.norm(pts[t] - med, axis=-1) > MAX_JUMP * ref) or not 0.67 < scale[t] / ref < 1.5 \
                or shape > SHAPE_JUMP * ref:
            bad.append(t)
    good[bad] = False
    return good, label


# ------------------------------------------------------------------ re-detecting and filling

def _interp_rows(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return (1 - w) * a + w * b


def _accept_rtm(kp: np.ndarray, sc: np.ndarray, expect_c: np.ndarray, expect_s: float, W: int, H: int) -> bool:
    if sc.mean() < RTM_MIN:
        return False
    lm = np.concatenate([kp, np.zeros((21, 1))], 1)
    c, s = palm_centre(lm), hand_scale(lm)
    inside = -0.25 * expect_s < c[0] < W + 0.25 * expect_s and -0.25 * expect_s < c[1] < H + 0.25 * expect_s
    return inside and np.linalg.norm(c - expect_c) < MAX_JUMP * expect_s and 0.67 < s / expect_s < 1.5


def redetect(clip: Clip, L: np.ndarray, src: list[str], conf: np.ndarray, rtm=None,
             cam: Camera | None = None) -> int:
    """RTMPose on the frames MediaPipe got wrong or missed (see the module's step 3), in place. Returns how many it
    found. ``rtm`` is the RTMPose wrapper (made here when None and there is anything to look at); ``cam``, the
    camera's motion, which carries the hand from the frames where it was seen to the frame it is looked for in."""
    T, H, W = len(src), clip.H, clip.W
    cam = cam or Camera.still(T)
    good = np.array([s == "mp" for s in src])
    if not good.any() or good.all():
        return 0
    found, model = 0, [rtm]

    def look(t, box):
        """RTMPose's keypoints and scores in ``box`` on frame ``t``; scores of 0 when rtmlib isn't installed (the
        core venv), so nothing is re-detected and the gaps are filled instead."""
        if model[0] is None:
            try:
                model[0] = _RTMPose()
            except ImportError:
                model[0] = lambda clip_, t_, box_: (np.zeros((21, 2)), np.zeros(21))
        return model[0](clip, t, box)

    offsets: dict[int, np.ndarray] = {}

    def offset(t):
        """Where RTMPose puts each landmark relative to MediaPipe on good frame ``t``: the two networks place the
        knuckles and wrist differently (0.1-0.25 hand sizes apart), and a re-detected frame is moved by the offsets
        of the good frames around it so it doesn't step off the track."""
        if t not in offsets:
            kp, sc = look(t, _box(L[t]))
            offsets[t] = kp - L[t, :, :2] if sc.mean() >= RTM_MIN else np.zeros((21, 2))
        return offsets[t]

    def put(t, kp, sc, z):
        nonlocal found
        L[t, :, :2], L[t, :, 2] = kp, z
        src[t], conf[t] = "rtm", float(np.clip(sc.mean(), 0, 1))
        found += 1

    idx = np.flatnonzero(good)
    first, last = idx[0], idx[-1]
    # Inside the track: the box drawn between the good frames either side.
    for t in range(first + 1, last):
        if src[t] != "none":
            continue
        a, b = idx[idx < t][-1], idx[idx > t][0]
        w = (t - a) / (b - a)
        guess = between(cam, L, a, b, t)
        box, exp_c, exp_s = _box(guess), palm_centre(guess), float(hand_scale(guess))
        kp, sc = look(t, box)
        kp = kp - _interp_rows(offset(a), offset(b), w)
        if _accept_rtm(kp, sc, exp_c, exp_s, W, H):
            put(t, kp, sc, _interp_rows(L[a, :, 2], L[b, :, 2], w))
    # At the ends: chained outward from the last frame found, moving on at the hand's speed there.
    for start, step in ((first, -1), (last, 1)):
        ref, prev, misses, t = start, None, 0, start + step
        while 0 <= t < T and misses < RTM_CHAIN_MISSES:
            if src[t] != "none":
                break
            here = cam.to_scene(L[ref], ref)
            if prev is not None:  # moving on through the scene at the hand's last speed
                here[:, :2] += (palm_centre(here) - palm_centre(cam.to_scene(L[prev], prev))) / (ref - prev) * (t - ref)
            guess = cam.to_frame(here, t)
            kp, sc = look(t, _box(guess))
            kp = kp - offset(start)
            ok = _accept_rtm(kp, sc, palm_centre(guess), float(hand_scale(guess)), W, H)
            if not ok and prev is not None:  # a strike stops dead where the arm locks: also where the hand last was
                # (carried on at its speed, the look overshot a thrown fist and lost it at the hit)
                guess = cam.to_frame(cam.to_scene(L[ref], ref), t)
                kp, sc = look(t, _box(guess))
                kp = kp - offset(start)
                ok = _accept_rtm(kp, sc, palm_centre(guess), float(hand_scale(guess)), W, H)
            if ok:
                put(t, kp, sc, L[ref, :, 2])
                prev, ref, misses = ref, t, 0
            else:
                misses += 1
            t += step
    return found


def between(cam: Camera, L: np.ndarray, a: int, b: int, t: int) -> np.ndarray:
    """The hand on frame ``t``, drawn straight through the scene from frame ``a`` to frame ``b`` (in frame t's px)."""
    w = (t - a) / (b - a)
    return cam.to_frame(_interp_rows(cam.to_scene(L[a], a), cam.to_scene(L[b], b), w), t)


def _lk(g0: np.ndarray, g1: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts.astype(np.float32).reshape(-1, 1, 2), None, winSize=(31, 31),
                                         maxLevel=3,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    return p1.reshape(-1, 2).astype(np.float64), st.reshape(-1).astype(bool)


def fill_gaps(clip: Clip, L: np.ndarray, src: list[str], conf: np.ndarray, cam: Camera | None = None) -> None:
    """Optical flow through gaps up to FLOW_MAX frames, straight lines (through the scene, so a moving camera still
    carries the hand) up to INTERP_MAX, lost beyond; ``hold`` before the first frame found and ``lost`` after the last
    (see the module's step 4). In place."""
    T, s, g = len(src), clip.s, clip.gray
    cam = cam or Camera.still(T)
    have = np.array([x in ("mp", "rtm") for x in src])
    if not have.any():
        for t in range(T):
            src[t], conf[t] = "lost", 0.0
        return
    idx = np.flatnonzero(have)
    for t in range(idx[0]):
        src[t] = "hold"
    for t in range(idx[-1] + 1, T):
        src[t] = "lost"
    for a, b in zip(idx[:-1], idx[1:]):
        n = b - a - 1
        if n == 0:
            continue
        gap = range(a + 1, b)
        if n > INTERP_MAX:
            for t in gap:
                src[t] = "lost"
            continue
        for t in gap:  # straight between the two; the flow below replaces it where it can
            L[t] = between(cam, L, a, b, t)
            src[t], conf[t] = "interp", CONF["interp"] * min(conf[a], conf[b])
        if n > FLOW_MAX:
            continue
        fwd, okf = {}, {}
        p, ok = L[a, :, :2] * s, np.ones(21, bool)
        for t in gap:
            p, st = _lk(g(t - 1), g(t), p)
            ok &= st
            fwd[t], okf[t] = p / s, ok.copy()
        bwd, okb = {}, {}
        p, ok = L[b, :, :2] * s, np.ones(21, bool)
        for t in reversed(gap):
            p, st = _lk(g(t + 1), g(t), p)
            ok &= st
            bwd[t], okb[t] = p / s, ok.copy()
        for t in gap:
            w = (t - a) / (b - a)
            lin = L[t, :, :2].copy()
            both = okf[t] & okb[t]
            est = np.where(both[:, None], _interp_rows(fwd[t], bwd[t], w),
                           np.where(okf[t][:, None], fwd[t], np.where(okb[t][:, None], bwd[t], lin)))
            L[t, :, :2] = est
            src[t], conf[t] = "flow", CONF["flow"] * min(conf[a], conf[b])


# ------------------------------------------------------------------ the track

def track_emitter(video_path_or_frames, fps: float | None = None, emitter: str = "two_finger", hand: str = "auto",
                  seed=None, out: str | Path | None = None, mirrored: bool = False, detections=None,
                  log=None) -> dict:
    """The emitter's track through a clip (the hand_track.json dict; see the module's docstring and
    docs/bending_contracts.md), also written to ``out`` when given.

    ``video_path_or_frames``: a video file, RGB uint8 frames ``[T, H, W, 3]``, or a Clip. ``fps``: the clip's (read
    from the file when None). ``emitter``: two_finger | index_tip | palm | fist. ``hand``: auto | left | right.
    ``seed``: (frame, x, y) of the Studio's click on the hand to follow, x and y in px or as fractions of the frame
    (both <= 1).
    ``mirrored``: the clip is a mirrored selfie recording (swaps left and right). ``detections``: MediaPipe's per-frame
    hands, when already run (mediapipe_hands); ``log``: a callable for one-line progress notes."""
    if emitter not in EMITTERS:
        raise ValueError(f"unknown emitter {emitter!r}; one of {EMITTERS}")
    if hand not in ("auto", "left", "right"):
        raise ValueError("hand is auto, left or right")
    clock = time.perf_counter()
    note = log or (lambda m: None)
    clip = video_path_or_frames if isinstance(video_path_or_frames, Clip) else Clip(video_path_or_frames, fps)
    fps, H, W = clip.fps, clip.H, clip.W
    dets = detections if detections is not None else mediapipe_hands(clip)
    T = clip.n
    if len(dets) != T:
        raise ValueError(f"{len(dets)} frames of detections for a {T}-frame clip")
    note(f"{T} frames {W}x{H} @ {fps:g} fps, mediapipe: hands in {sum(bool(d) for d in dets)} "
         f"({time.perf_counter() - clock:.1f}s)")
    # The camera's own motion. Following, doubting, finding strikes and smoothing all happen in the scene's
    # coordinates, so a shaking camera is neither a hand jumping nor a hand moving; the result is put back into each
    # frame's pixels, so the element still rides the hand wherever the camera puts it.
    cam = camera_motion(clip, dets)
    moved = np.hypot(*(cam.P[-1, :, 2] if T else (0, 0)))
    note(f"camera: {sum(not np.allclose(cam.P[t], cam.P[t - 1]) for t in range(1, T))} of {max(T - 1, 0)} steps "
         f"measured, {moved:.0f}px off where it started ({time.perf_counter() - clock:.1f}s)")
    seed = _seed_px(seed, W, H)
    if seed is not None and detections is None:
        # A second look around the click, for the hand it points at: often one the full-frame pass misses (cut by
        # the picture's edge, small, dark). Its finds join that frame's detections.
        extra = seeded_look(clip, seed)
        dets = [_dedupe(list(dets[t]) + extra[t]) for t in range(T)]
        note(f"second look around the click: the hand on {sum(bool(e) for e in extra)} frames "
             f"({time.perf_counter() - clock:.1f}s)")
    anchor = _pick_anchor(dets, hand, seed, mirrored, (W, H))
    if seed is not None and anchor is None and detections is None:
        # A blurred click can precede the first clear hand by more than SEED_FRAMES. Restart the crop pass on the
        # nearest detection within twice that time, so its backward pass can see the clicked hand. That detection
        # is only a starting place: accept the result only if it supplies a hand near the ORIGINAL click within
        # the original radius and time window. A hand elsewhere in the clip is never enough on its own.
        candidates = [(abs(t - seed[0]), float(np.linalg.norm(palm_centre(h["lm"]) - seed[1:])), t, k)
                      for t, hands in enumerate(dets) if abs(t - seed[0]) <= 2 * SEED_FRAMES
                      for k, h in enumerate(hands)]
        if candidates:
            _, _, t, k = min(candidates)
            retry = seeded_look(clip, (t, *palm_centre(dets[t][k]["lm"])))
            merged = [_dedupe(list(dets[f]) + retry[f]) for f in range(T)]
            recovered = _pick_anchor(merged, hand, seed, mirrored, (W, H))
            if recovered is not None:
                dets, anchor = merged, recovered
                note(f"click recovered by looking back from frame {t}: anchored on frame {anchor[0]} "
                     f"({time.perf_counter() - clock:.1f}s)")
    if seed is not None and anchor is None:
        note(f"no hand near the click ({seed[1]:.0f}, {seed[2]:.0f}) within {SEED_FRAMES} frames of frame {seed[0]}: "
             f"MediaPipe found none there, even looking again; the track holds the click")
    scene = [[dict(h, lm=cam.to_scene(h["lm"], t), frame_lm=h["lm"]) for h in dets[t]] for t in range(T)]

    L = np.full((T, 21, 3), np.nan)
    src, conf = ["none"] * T, np.zeros(T)
    gesture_mp = ["None"] * T
    label = "Right"
    finger, on_tip = None, False  # on_tip: the click is on a fingertip (the element comes out of a digit)
    if anchor is not None:
        chosen = follow_hand(scene, anchor)
        if seed is not None:
            # Which digit the click names: matched on the frame (within SEED_FRAMES of the click's) where the followed
            # hand has a fingertip nearest it, since a click can come a moment before or after the pose it means.
            near = [t for t in range(max(0, seed[0] - SEED_FRAMES), min(T, seed[0] + SEED_FRAMES + 1))
                    if chosen[t] is not None]
            if near:
                tips = [4, 8, 12, 16, 20]
                best = min(near, key=lambda t: np.min(np.linalg.norm(chosen[t]["frame_lm"][tips, :2] - seed[1:],
                                                                     axis=-1)))
                finger = clicked_finger(chosen[best]["frame_lm"], seed, emitter)
                lm_ = chosen[best]["frame_lm"]
                on_tip = bool(np.min(np.linalg.norm(lm_[tips, :2] - seed[1:], axis=-1)) < 0.35 * float(hand_size(lm_)))
                if finger is not None:
                    note(f"the click is on landmark {finger[1]} (MediaPipe's name for that digit, frame {best}), not "
                         f"the {emitter} tip: the emitter rides that finger")
        good, label = gate(chosen, emitter, finger)
        for t in np.flatnonzero(good):
            L[t] = chosen[t]["frame_lm"]
            src[t], conf[t], gesture_mp[t] = "mp", chosen[t]["score"], chosen[t]["gesture"]
        n_rtm = redetect(clip, L, src, conf, cam=cam) if good.any() else 0
        note(f"gated: {int(good.sum())} good, rtmpose found {n_rtm} more ({time.perf_counter() - clock:.1f}s)")
    fill_gaps(clip, L, src, conf, cam=cam)
    sign = 1.0 if label == "Right" else -1.0

    # The emitter in each frame's px, then in the scene's.
    tracked = np.array([s not in ("hold", "lost") for s in src])
    xy_f, dir_f, scale_f = emitter_points(np.nan_to_num(L), emitter, sign, (W, H), finger)
    zoom, turn = cam.zoom(), cam.turn()
    xy_raw = cam.to_scene(xy_f)
    scale_raw = scale_f / zoom
    angle_raw = np.arctan2(dir_f[:, 1], dir_f[:, 0]) - turn
    lms_raw = cam.to_scene(np.nan_to_num(L))

    # Strikes, on the filled but unsmoothed path.
    speed_raw = frame_speed(xy_raw, tracked, fps)
    typical = float(np.median(scale_raw[tracked])) if tracked.any() else 1.0
    events = find_extensions(speed_raw, typical, fps, tracked)
    gesture = _spread_gestures(gesture_mp, src, np.nan_to_num(L))
    for t in range(T):  # a fingertip emitter's digit held out: pointing (where MediaPipe names nothing); an open palm
        # only when the click was on a fingertip (a hiss follows the fingertips, and every open palm hissing at the
        # camera became two fingers pointing, and a flamethrower)
        if tracked[t] and gesture_mp[t] == "None" and (gesture[t] == "None" or gesture[t] == "Open_Palm" and on_tip):
            g = pointing_gesture(np.nan_to_num(L[t]), emitter, finger)
            gesture[t] = g or gesture[t]
    in_view = tracked & np.array([tracked[t] and inside_share(L[t], (W, H)) >= 0.8 for t in range(T)])
    events += find_opens(finger_spread(np.nan_to_num(L)), gesture, in_view, speed_raw, fps)  # tips guessed off-frame
    events.sort(key=lambda e: e["frame"])
    cuts = [e["frame"] for e in events]

    # Smoothing, piece by piece, lighter while the hand moves than while it holds.
    still = stillness(xy_raw, scale_raw, tracked, cuts, fps)
    xy = adaptive_smooth(xy_raw, still, tracked, cuts, fps)
    scale = adaptive_smooth(scale_raw, still, tracked, cuts, fps)
    ang = angle_raw.copy()
    for fl, fh, ol, oh in smooth_pieces(tracked, cuts):
        seg = np.unwrap(angle_raw[fl:fh])  # each piece unwrapped on its own, so none averages across +-pi
        ang[ol:oh] = adaptive_smooth(seg, still[fl:fh], np.ones(fh - fl, bool), [], fps)[ol - fl:oh - fl]
    ang = steady_direction(ang, src, fps)
    lms = adaptive_smooth(lms_raw.reshape(T, -1), still, tracked, cuts, fps).reshape(T, 21, 3)
    # ... and back into each frame's px.
    xy, scale, lms = cam.to_frame(xy), scale * zoom, cam.to_frame(lms)
    ang = ang + turn
    direction = np.stack([np.cos(ang), np.sin(ang)], -1)
    speed = np.zeros(T)
    for fl, fh, ol, oh in smooth_pieces(tracked):
        if oh - ol > 1:
            speed[ol:oh] = np.linalg.norm(np.gradient(xy[ol:oh], axis=0), axis=-1) * fps

    # Frames without a hand hold the nearest one before (lost) or the first one found (hold).
    if tracked.any():
        idx = np.flatnonzero(tracked)
        for t in range(T):
            if not tracked[t]:
                k = idx[idx < t][-1] if (idx < t).any() else idx[0]
                xy[t], direction[t], scale[t] = xy[k], direction[k], scale[k]
    else:  # no hand at all: hold the click if there was one, else the picture's centre
        xy[:] = (seed[1], seed[2]) if seed is not None else (W / 2, H / 2)
        direction[:] = (0.0, -1.0)
        scale[:] = 0.1 * min(W, H)
    conf[~tracked] = 0.0

    if mirrored:
        label = "Left" if label == "Right" else "Right"
    track = {
        "version": VERSION, "fps": fps, "W": int(W), "H": int(H), "emitter": emitter, "hand": label.lower(),
        "xy": np.round(xy, 2).tolist(),
        "dir": np.round(_unit(direction), 4).tolist(),
        "scale": np.round(scale, 2).tolist(),
        "conf": np.round(np.clip(conf, 0, 1), 3).tolist(),
        "src": src,
        "speed": np.round(speed, 1).tolist(),
        "landmarks": [np.round(lms[t], 1).tolist() if tracked[t] else None for t in range(T)],
        "gesture": gesture,
        "events": events,
    }
    counts = {k: src.count(k) for k in ("mp", "rtm", "flow", "tapir", "interp", "hold", "lost") if k in src}
    note(f"done in {time.perf_counter() - clock:.1f}s: {counts}, events {[(e['kind'], e['frame']) for e in events]}")
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(track, separators=(",", ":")), encoding="utf-8")
    return track


def _spread_gestures(gesture_mp: list[str], src: list[str], L: np.ndarray | None = None) -> list[str]:
    """MediaPipe's gesture where it saw the hand and named one, unless the landmarks clearly read all four fingers out
    or all four in (Open_Palm, Closed_Fist): its classifier called an upturned open palm "Victory" or "Thumb_Up", and a
    flamethrower came out of it; else the gesture read off the landmarks (landmark_gesture); across a filled gap, the
    nearest named one within 3 frames. "None" where the hand is lost."""
    T = len(src)
    out = list(gesture_mp)
    for t in range(T):
        if src[t] == "mp" and L is not None:
            g = landmark_gesture(L[t])
            if out[t] == "None" or g in ("Open_Palm", "Closed_Fist"):
                out[t] = g if g != "None" else out[t]
    named = [t for t in range(T) if src[t] == "mp" and out[t] != "None"]
    for t in range(T):
        if src[t] in ("rtm", "flow", "tapir", "interp"):
            k = min(named, key=lambda s: abs(s - t)) if named else None
            out[t] = out[k] if k is not None and abs(k - t) <= 3 else (landmark_gesture(L[t]) if L is not None
                                                                         else "None")
        elif src[t] != "mp":
            out[t] = "None"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cuesync.hand_track", description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--emitter", default="two_finger", choices=EMITTERS)
    ap.add_argument("--hand", default="auto", choices=("auto", "left", "right"))
    ap.add_argument("--seed", help="frame,x,y of a click on the hand (px, or fractions of the frame)")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--mirrored", action="store_true", help="a mirrored selfie recording")
    ap.add_argument("--out", help="where to write hand_track.json (default: beside the video)")
    a = ap.parse_args(argv)
    seed = tuple(float(v) for v in a.seed.split(",")) if a.seed else None
    out = Path(a.out) if a.out else Path(a.video).with_name("hand_track.json")
    track_emitter(a.video, a.fps, a.emitter, a.hand, seed, out=out, mirrored=a.mirrored,
                  log=lambda m: print(f"hand_track: {m}", file=sys.stderr))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
