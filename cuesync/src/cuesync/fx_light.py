"""Light, flash, shake, heat haze, air blast and grain for an element bent from a moving hand (element bending, WP5).

In element bending the effect is rendered over black with its emitter held still, then warped onto the tracked hand
(WP3). The light the model baked into that render was thrown from where the hand was when it rendered, so none of it
is kept: the shot is lit again here, deterministically, from the warped element itself. Every lit pixel of the flame
is a little lamp: a surface at distance r in the picture and d in front of it receives its luminance over
pi (r^2 + d^2), summed over the whole flame (a convolution of the flame's own image). So the light moves where the
flame is (a fireball lifting off the palm leaves it dimmer), falls off from its root, flickers exactly as the flame
does, and two flames light what is near each. The hand and arm (the hand matte, WP2) sit HAND_DEPTH of the flame's
radius from it and take it strongly; everything else stands FAR behind it, as a wall a few hands back would, and takes it weakly
and widely. Its colour is the element's, held to what a flame (2000-2700 K) or steam (a cool white) can throw.
Nothing is lit behind the hand: past the hand, in the half-space the emitter faces away from, the light is 0. It is
added in linear light as the plate's own colour times the light (a surface reflects what falls on it), so the plate
keeps its texture under it and nothing ever gets darker. Smoke throws no light; it only veils a little.

Then the camera's side of a strike: an exposure flash on the launch frame, a shake a few frames after the peak, heat
haze over fire, the travelling ripple that is an air blast, and the plate's own grain on the clean rendered layers.

Everything is numpy and OpenCV, and every random part is drawn from ``seed`` and the frame index, so the same inputs
give the same bytes, on any number of threads. Frames are linear RGB float32 [T, H, W, 3] (``transfer.lin``/``gam``
convert), 24 fps; points and directions are pixels and unit vectors, origin top-left, y down
(docs/bending_contracts.md). The order the compositor applies them in:

    plate -> light_plate -> heat_haze / air_wave -> + element (grain on it first; Add for fire and steam, Over for
    smoke) -> flash -> gam -> shake
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import cv2
import numpy as np

FPS = 24
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # Rec.709 luminance of linear RGB
THREADS = min(4, os.cpu_count() or 1)  # frames are done independently: this changes the speed (and the scratch
# memory, ~150 MB a thread at 1080p), never the bytes
_GRAIN, _HAZE, _AIR, _SHAKE = 1, 2, 3, 4  # mixed into each seed, so one seed gives each effect its own noise

# The light. Distances are in hand scales (the track's |wrist - middle knuckle|), so they grow as the hand nears the
# camera. A surface of the plate's colour c gains c / white * gain * sum_p Y(p) / (pi (r^2 + d^2)) over the flame's
# pixels p at distance r: d is how far in front of the flame it stands.
HAND_DEPTH, FAR = 0.5, 2.5  # the hand: half the flame's own radius from it (the palm right under a fireball takes about
# twice its brightness whatever its size; a candle lights only the skin next to it); the room: FAR hand scales behind
# the flame (a wall about 60 cm behind a hand), so weak but reaching across the frame
HAND_GAIN, ROOM_GAIN = 2.0, 2.0  # over that physical light (fitted by eye on the cupped-palm and two-palm plates)
EXPOSED_WHITE = 0.8  # linear: where a camera puts a white surface in a lit room (and cuesync.media an HDR clip's
# diffuse white): a plate's white is never taken to be brighter than this, nor its brightest surface for white paint
TOP_ALBEDO, WHITE_FLOOR = 0.8, 0.05  # in a dim room nothing reaches that white: its brightest surface (a bath, a wall)
# is taken to reflect TOP_ALBEDO, so a phone exposing a dark room up shows the fire lighting skin as it does; taken
# at 0.8 there, skin read as near black and the fire lit nothing. Never under WHITE_FLOOR (a black plate)
FIELD_MAX = 480  # px: the light is worked out on a copy of the element at most this long (it is smooth)
KEY_WB = 0.3  # the plate's own key light taken this far out of a surface's colour before the fire's light falls on it:
# skin under a warm key read as red paint, and fire on it went salmon instead of gold
SHADOW_ALBEDO = 0.5  # a surface in shadow reflects at least this much of how bright it reads in the picture (gamma):
# a black suit off the key light is not black paint, and lit by a fireball its folds show
FEATHER = 2.0  # behind the emitter, off the hand, the light fades out over this distance (soft, so the edge of the
# hand's shadow is no line across the wall); past it nothing is lit
ARM_HALF = 0.5  # with no matte: the hand and forearm are taken to be a band this wide either side of the line back
MAX_LIGHT = 4.0  # the light may brighten a surface this much at most (+2.3 stops: a hand inside a fireball)
ALBEDO_FLOOR = 0.005  # a pixel too dark to read a colour from still reflects this much (a shadow is not black paint)
GLOW, GLOW_RADIUS = 0.035, 1.0  # the flame's own light scattered in the air and lens: a wide halo, visible on black (the
# critiques: without it the flame sat on the plate like a sticker; no lens flares), blurred by this many of its radii
# What each fluid throws: the colour temperatures its light is held to, how bright it is against fire (steam only
# scatters light, it makes none) and how much it glows. Smoke throws none and only veils (SMOKE_*).
FLUIDS = {"fire": {"kelvin": (2000.0, 2700.0), "gain": 1.0, "glow": 1.0},
          "steam": {"kelvin": (7000.0, 10000.0), "gain": 0.0, "glow": 0.0}}  # steam scatters light, it makes none
SMOKE_VEIL, SMOKE_FULL = 0.12, 0.2  # thick smoke (this linear luminance over black) veils a tenth of what is behind it

# The camera.
FLASH_STOPS = 0.6  # the exposure jumps this much on a launch at full strength...
FLASH_DECAY = (1.0, 0.5, 0.2)  # ... and falls back over three frames
IRIS_STOPS, IRIS_BIG = -0.12, 0.8  # a burst this strong makes the auto-exposure overshoot on the frame after the flash
SHAKE_AFTER, SHAKE_FRAMES = 2, 8  # the shock reaches the camera 2 frames after the peak, and it settles in 8
SHAKE_HZ = (8.0, 11.0)  # a phone in the hand wobbles at about this rate once knocked

# Heat haze over fire: sizes are shares of the frame's short side, displacements px at 1080p.
HAZE_PX, HAZE_MAX = 2.0, 3.0  # typical and largest shift of what is seen through the hot air (the brief: <= 3 px)
HAZE_RISE, HAZE_AROUND = 0.12, 0.02  # the hot air shimmers this far above the flame and this far around it
HAZE_CELLS = ((0.035, 0.8), (0.014, 0.5))  # two sizes of ripple and their weights
HAZE_SPEED = 0.006  # the ripples rise this far per frame (about 150 px/s at 1080p)

# Air blast: a pressure front travelling out from the palm, bending what is seen through it.
AIR_FRAMES = 8  # it crosses its reach in a third of a second (the brief: 6-10 frames)
AIR_FROM, AIR_TO = 0.04, 0.55  # radius of the front on its first and last frame, shares of the short side
AIR_WIDTH = 0.03  # share of the short side: the thickness of the front
AIR_PX = 7.0  # px at 1080p: how far the front bends what is behind it, at its strongest
AIR_CA = 0.3  # the red channel bends this much more and blue this much less: a slight colour fringe on the front
AIR_CONE = 2.0  # power of the cosine away from the blast's direction: strongest straight ahead, nothing behind
AIR_SMOKE = 0.22  # opacity of the thin dust caught on the leading edge (smoke only there, never filling the blast); it
# catches the light the room's lit surfaces do, so it shows even where the ripple has no texture to bend (black)

GRAIN_SOFT = 0.6  # px: phone noise is a little blotchy (demosaicing, denoising), not per-pixel white noise


def _each(n: int, fn: Callable[[int], None]) -> None:
    """fn(f) for every frame f, on THREADS threads (each call writes only its own frame)."""
    if n <= 1 or THREADS == 1:
        for f in range(n):
            fn(f)
        return
    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(fn, range(n)))


def _rng(seed: int, *salt: int) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, *salt])


def _out(frames: np.ndarray, out: np.ndarray | None) -> np.ndarray:
    """Where to write: ``out`` (pass the input itself to work in place: a 10 s 1080p clip is 6 GB of float32), else a
    new float32 array."""
    return np.empty(frames.shape, np.float32) if out is None else out


def _lum(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


# ---- colour ---------------------------------------------------------------------------------------------------------

def _black_body() -> tuple[np.ndarray, np.ndarray]:
    """Linear sRGB colour of a black body from 1900 K to 12000 K, each scaled to luminance 1: Planck's law integrated
    against the CIE 1931 colour matching functions (in the analytic fit of Wyman, Sloan and Shirley, JCGT 2013)."""
    lam = np.arange(380.0, 781.0, 2.0)

    def lobe(mu: float, s1: float, s2: float) -> np.ndarray:
        return np.exp(-0.5 * ((lam - mu) / np.where(lam < mu, s1, s2)) ** 2)

    cmf = np.stack([1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2),
                    0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1),
                    1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)])
    xyz_to_rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    kelvin = np.arange(1900.0, 12001.0, 25.0)
    m = lam * 1e-9
    planck = 1.0 / (m ** 5 * np.expm1(1.4388e-2 / (m[None, :] * kelvin[:, None])))  # second radiation constant hc/k
    rgb = np.maximum(planck @ cmf.T @ xyz_to_rgb.T, 0.0)
    return kelvin, (rgb / (rgb @ _LUMA.astype(np.float64))[:, None]).astype(np.float32)


_KELVIN, _KELVIN_RGB = _black_body()
_KELVIN_CHROMA = _KELVIN_RGB / _KELVIN_RGB.sum(1, keepdims=True)


def kelvin_of(rgb: np.ndarray) -> float:
    """The temperature of the black body whose colour is nearest linear ``rgb`` (in r:g:b proportions, i.e. its
    correlated colour temperature). A flame's colour sits off the black-body line (a rendered fire has almost no
    blue), so its nearest point is read, not its blue-to-red ratio alone, which read any orange as below 2000 K."""
    c = np.maximum(np.asarray(rgb, np.float64), 0)
    c = c / max(float(c.sum()), 1e-12)
    return float(_KELVIN[np.argmin(((_KELVIN_CHROMA - c) ** 2).sum(1))])


def kelvin_rgb(kelvin: float | np.ndarray) -> np.ndarray:
    """Linear sRGB colour of a black body at ``kelvin``, scaled to luminance 1."""
    return np.stack([np.interp(kelvin, _KELVIN, _KELVIN_RGB[:, c]) for c in range(3)], -1).astype(np.float32)


# ---- the track ------------------------------------------------------------------------------------------------------

def track_arrays(track: dict, n: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The emitter point (n, 2), unit direction (n, 2) and hand scale (n,) from a hand_track.json dict, in the pixels
    of a w x h frame (a track made on another size of the same video is scaled onto it)."""
    xy = np.asarray(track["xy"], np.float64).reshape(-1, 2)
    u = np.asarray(track["dir"], np.float64).reshape(-1, 2)
    s = np.asarray(track["scale"], np.float64).reshape(-1)
    if not len(xy) == len(u) == len(s) == n:
        raise ValueError(f"the track has {len(xy)} frames, the clip {n}")
    kx, ky = w / float(track.get("W", w)), h / float(track.get("H", h))
    xy, u, s = xy * (kx, ky), u * (kx, ky), s * np.sqrt(kx * ky)
    norm = np.hypot(u[:, 0], u[:, 1])
    u = np.where(norm[:, None] > 1e-9, u / np.maximum(norm, 1e-9)[:, None], (0.0, -1.0))  # no direction: up
    return xy, u, np.maximum(s, 1.0)


def arm_matte(xy: tuple[float, float], u: tuple[float, float], scale: float, w: int, h: int) -> np.ndarray:
    """(h, w) float32: the hand and forearm when there is no matte (WP2's half-space fallback): a band ARM_HALF hand
    scales either side of the line from the emitter back, against its direction, to the edge of the frame."""
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)[:, None]
    dx, dy = xs - np.float32(xy[0]), ys - np.float32(xy[1])
    behind = -(dx * np.float32(u[0]) + dy * np.float32(u[1]))
    across = np.abs(dx * np.float32(u[1]) - dy * np.float32(u[0]))
    s = np.float32(scale)
    return (np.clip(1 + behind / (0.25 * s), 0, 1) * np.clip((ARM_HALF * s - across) / (0.2 * s), 0, 1)).astype(np.float32)


def _matte(mask: np.ndarray, f: int) -> np.ndarray:
    m = mask[f]
    return m.astype(np.float32) * np.float32(1 / 255) if m.dtype == np.uint8 else m.astype(np.float32, copy=False)


# ---- the light ------------------------------------------------------------------------------------------------------

def _smooth2(x: np.ndarray) -> np.ndarray:
    """Zero-phase smoothing over 2 frames (1/4, 1/2, 1/4): no lag, so the light peaks on the flame's own peak."""
    p = np.pad(np.asarray(x, np.float64), [(1, 1)] + [(0, 0)] * (np.ndim(x) - 1), mode="edge")
    return 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]


def element_light(element_lin: np.ndarray, fluid: str = "fire",
                  size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame, the light the element throws: its power (n,), its radius (n,, px) and its colour (n, 3).

    The power P is the element's luminance summed over the frame (in the plate's pixels: ``size`` = (h, w) of the
    plate when the element is at another size), so the light flickers with the flame and nothing else. The radius is
    that of a disc as bright as the element's average glow and as bright in sum (its participation area, (sum Y)^2 /
    sum Y^2). The colour is the element's mean colour read as a black-body temperature and held to the fluid's range
    (a yellow-white core gives 2700 K, a dying red fireball 2000 K), at luminance 1. All three are smoothed over 2
    frames, zero-phase, so light and flame breathe together and the light never lags."""
    n, he, we = element_lin.shape[:3]
    area = 1.0 if size is None else size[0] * size[1] / float(he * we)
    sums, sq = np.zeros((n, 3)), np.zeros(n)

    def one(f: int) -> None:
        sums[f] = element_lin[f].reshape(-1, 3).sum(0, dtype=np.float64)
        y = _lum(element_lin[f]).ravel()
        sq[f] = float(np.dot(y, y))

    _each(n, one)
    sums = np.maximum(sums, 0.0) * area
    power = sums @ _LUMA.astype(np.float64)
    radius = np.sqrt(power ** 2 / np.maximum(sq * area, 1e-12) / np.pi)
    lo, hi = FLUIDS.get(fluid, FLUIDS["fire"])["kelvin"]
    seen = np.nonzero(power > 1e-6)[0]
    if seen.size:  # a frame with no element takes its neighbours' size and colour (it throws no light anyway)
        t = np.interp(np.arange(n), seen, [kelvin_of(sums[f]) for f in seen])
        radius = np.interp(np.arange(n), seen, radius[seen])
    else:
        t = np.full(n, 0.5 * (lo + hi))
    return (_smooth2(power).astype(np.float32), _smooth2(radius).astype(np.float32),
            kelvin_rgb(_smooth2(np.clip(t, lo, hi))))


def plate_white(plate_lin: np.ndarray, frames: int = 8) -> float:
    """The plate's white, in linear light: what a white surface under the scene's own light reads. Its brightest
    surface (the 99.9th percentile of luminance over a few frames at a quarter size) over TOP_ALBEDO, held between
    WHITE_FLOOR and EXPOSED_WHITE (a lit room's white is where the camera puts it). The plate over it is read as each
    surface's colour, which is what the light multiplies."""
    n, h, w = plate_lin.shape[:3]
    picks = np.unique(np.linspace(0, n - 1, min(n, frames)).round().astype(int))
    tops = [np.percentile(_lum(cv2.resize(plate_lin[f], (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)),
                          99.9) for f in picks]
    return float(np.clip(np.median(tops) / TOP_ALBEDO, WHITE_FLOOR, EXPOSED_WHITE))


def plate_tint(plate_lin: np.ndarray, frames: int = 8) -> np.ndarray:
    """The colour of the plate's own key light, at luminance 1: the mean colour of its brightest surfaces (the top
    half percent of luminance over a few frames at a quarter size). White for a plate with no light to read."""
    n, h, w = plate_lin.shape[:3]
    picks = np.unique(np.linspace(0, n - 1, min(n, frames)).round().astype(int))
    tops = []
    for f in picks:
        px = cv2.resize(plate_lin[f], (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA).reshape(-1, 3)
        y = _lum(px)
        tops.append(px[y >= np.percentile(y, 99.5)].mean(0))
    c = np.maximum(np.median(tops, axis=0), 0).astype(np.float32)
    y = float(c @ _LUMA)
    return c / y if y > 1e-6 else np.ones(3, np.float32)


def albedo(plate_lin: np.ndarray, white: float, tint: np.ndarray) -> np.ndarray:
    """Each surface's colour as the light multiplies it: the plate over its white, at least SHADOW_ALBEDO of how bright
    it reads in the picture (a surface in shadow is not black paint), the key light's tint partly taken out."""
    p = np.maximum(plate_lin, 0)
    a = np.maximum(p * np.float32(1 / white), np.float32(SHADOW_ALBEDO) * np.minimum(p, 1) ** np.float32(1 / 2.2))
    return a / (np.maximum(tint, 1e-3) ** np.float32(KEY_WB)).astype(np.float32) + np.float32(ALBEDO_FLOOR)


def spread(y: np.ndarray, depths: list[float]) -> list[np.ndarray]:
    """For each depth d (px of ``y``'s grid): at every pixel x, the light the glowing image ``y`` (h, w, luminance)
    throws onto a surface d in front of it, sum_p y(p) / (pi (|x - p|^2 + d^2)). A convolution, by FFT, padded so
    nothing wraps round the frame's edge."""
    h, w = y.shape
    hh, ww = 2 * h, 2 * w
    dy = np.minimum(np.arange(hh), hh - np.arange(hh)).astype(np.float64)[:, None] ** 2
    dx = np.minimum(np.arange(ww), ww - np.arange(ww)).astype(np.float64)[None, :] ** 2
    f = np.fft.rfft2(y.astype(np.float64), (hh, ww))
    return [np.maximum(np.fft.irfft2(f * np.fft.rfft2(1.0 / (np.pi * (dx + dy + d * d))), (hh, ww))[:h, :w], 0)
            .astype(np.float32) for d in depths]


def light_fields(element_lin: np.ndarray, s: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    """Per frame, the element's light on the hand (HAND_DEPTH of its radius from it, at most a hand) and on the room
    (FAR hands behind it), before gain: two (n, hf, wf) fields on a grid at most FIELD_MAX long, and that grid's px per
    plate px. ``s`` is the hand scale per frame in plate px, ``size`` the plate's (h, w). Smoothed over 2 frames, zero-phase, as the flame's colour is,
    so light and flame breathe together and the light never lags."""
    n, he, we = element_lin.shape[:3]
    k = min(1.0, FIELD_MAX / float(max(he, we)))
    hf, wf = max(1, int(round(he * k))), max(1, int(round(we * k)))
    per = np.sqrt(hf * wf / float(size[0] * size[1]))  # field px per plate px
    hand, room = np.zeros((n, hf, wf), np.float32), np.zeros((n, hf, wf), np.float32)

    def one(f: int) -> None:
        y = _lum(element_lin[f])  # (each pixel's mean luminance: the sum in the grid's own px is the same light)
        y = y if (hf, wf) == (he, we) else cv2.resize(y, (wf, hf), interpolation=cv2.INTER_AREA)
        total = float(y.sum(dtype=np.float64))
        if total <= 0:
            return
        r = np.sqrt(total ** 2 / max(float(np.dot(y.ravel(), y.ravel())), 1e-12) / np.pi)  # its radius (element_light's)
        hand[f], room[f] = spread(y, [max(HAND_DEPTH * min(r, s[f] * per), 1.0), FAR * s[f] * per])

    _each(n, one)
    return _smooth2(hand).astype(np.float32), _smooth2(room).astype(np.float32), per


def light_plate(plate_lin: np.ndarray, element_lin: np.ndarray, track: dict, mask: np.ndarray | None = None,
                fluid: str = "fire", seed: int = 0, out: np.ndarray | None = None,
                white: float | None = None, tint: np.ndarray | None = None) -> np.ndarray:
    """The plate lit by the element: float32 [T, H, W, 3], linear. Never darker than ``plate_lin`` anywhere (fire,
    steam); for smoke, a slight veil where it is thick and no light.

    ``element_lin`` is the warped element over black [T, h, w, 3] (any size of the same frame: only its light is used),
    ``track`` the hand_track.json dict, ``mask`` the hand matte [T, H, W] (float 0..1 or uint8) or None (then a band
    back from the emitter stands in for the arm). Every pixel of the flame lights what is near it (``spread``): the
    matte strongly (HAND_DEPTH of the flame's radius from it), the whole shot weakly (FAR hands behind it); off the matte, behind the
    emitter's plane, it fades out over FEATHER hand scales and is exactly 0 beyond (the hand's far side). ``seed`` is
    unused: the light has no random part, its flicker is the element's own. ``out`` may be ``plate_lin`` itself, to
    light it in place. A clip lit in chunks: pass each the whole plate's ``plate_white`` and ``plate_tint`` as ``white``
    and ``tint`` and overlap the chunks by a frame (the light is smoothed over its neighbours), and each frame is lit as
    in the whole clip."""
    n, h, w = plate_lin.shape[:3]
    if len(element_lin) != n or (mask is not None and len(mask) != n):
        raise ValueError(f"the plate has {n} frames, the element {len(element_lin)}, the mask "
                         f"{None if mask is None else len(mask)}")
    xy, u, s = track_arrays(track, n, w, h)
    out = _out(plate_lin, out)
    if fluid not in FLUIDS:
        return _veil(plate_lin, element_lin, s, out)
    look = FLUIDS[fluid]
    power, radius, colour = element_light(element_lin, fluid, size=(h, w))
    hand_f, room_f, _ = light_fields(element_lin, s, (h, w)) if look["gain"] > 0 else (None, None, 1.0)
    white = plate_white(plate_lin) if white is None else float(white)
    tint = plate_tint(plate_lin) if tint is None else np.asarray(tint, np.float32)
    xs, ys = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)[:, None]

    def one(f: int) -> None:
        if power[f] <= 0 or look["gain"] <= 0 or float(element_lin[f].max()) <= 0:  # (smoothed over its neighbours,
            out[f] = plate_lin[f]  # the light came on a frame before its flame: a palm glowing before the fire)
            return
        (x0, y0), (ux, uy), sf = xy[f].astype(np.float32), u[f].astype(np.float32), np.float32(s[f])
        room = cv2.resize(room_f[f], (w, h), interpolation=cv2.INTER_LINEAR) * np.float32(ROOM_GAIN * look["gain"])
        hand = cv2.resize(hand_f[f], (w, h), interpolation=cv2.INTER_LINEAR) * np.float32(HAND_GAIN * look["gain"])
        seen = np.clip(1 + ((xs - x0) * ux + (ys - y0) * uy) * (1 / (FEATHER * sf)), 0, 1)  # 0: past the hand
        m = _matte(mask, f) if mask is not None else arm_matte((x0, y0), (ux, uy), sf, w, h)
        reach = seen + m * (1 - seen)  # where light can get to: in front of the hand, and the hand itself
        light = np.minimum(room * reach + hand * m, MAX_LIGHT)
        k = light[..., None] * colour[f]  # the light's colour and strength, per pixel
        lit = plate_lin[f] + albedo(plate_lin[f], white, tint) * k
        if GLOW > 0:  # the faint halo the flame casts in the air around it, kept off the hand's far side too
            q = 8
            small = cv2.resize(element_lin[f], (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
            halo = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.5, GLOW_RADIUS * float(radius[f]) / q)), (w, h),
                              interpolation=cv2.INTER_LINEAR)
            lit += np.maximum(halo, 0) * (reach * (GLOW * look["glow"]))[..., None]
        out[f] = lit

    _each(n, one)
    return out


def _veil(plate_lin: np.ndarray, element_lin: np.ndarray, s: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Smoke throws no light: where it is thick it only veils the plate a little towards the room's own colour (a
    slight darkening of what's bright, a haze over what's dark)."""
    n, h, w = plate_lin.shape[:3]

    def one(f: int) -> None:
        q = 4
        small = cv2.resize(_lum(element_lin[f]), (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
        dense = cv2.resize(cv2.GaussianBlur(small, (0, 0), max(0.5, 0.25 * float(s[f]) / q)), (w, h),
                           interpolation=cv2.INTER_LINEAR)
        veil = (SMOKE_VEIL * np.clip(dense / SMOKE_FULL, 0, 1))[..., None]
        room = plate_lin[f].reshape(-1, 3).mean(0, dtype=np.float64).astype(np.float32)
        out[f] = plate_lin[f] + veil * (room - plate_lin[f])

    _each(n, one)
    return out


# ---- the camera -----------------------------------------------------------------------------------------------------

def flash_stops(n: int, launch_frames: list[int], strengths: list[float], iris: bool = True) -> np.ndarray:
    """(n,) exposure change in stops: +FLASH_STOPS x strength on each launch frame, falling back over 3 frames, then
    (``iris``, a burst of at least IRIS_BIG) one frame of the auto-exposure overshooting the other way."""
    stops = np.zeros(n)
    for f0, st in zip(launch_frames, strengths):
        st = max(0.0, float(st))
        for k, share in enumerate(FLASH_DECAY):
            if 0 <= f0 + k < n:
                stops[f0 + k] += FLASH_STOPS * st * share
        if iris and st >= IRIS_BIG and 0 <= f0 + len(FLASH_DECAY) < n:
            stops[f0 + len(FLASH_DECAY)] += IRIS_STOPS * st
    return stops


def flash(frames_lin: np.ndarray, launch_frames: list[int], strengths: list[float], iris: bool = True,
          out: np.ndarray | None = None) -> np.ndarray:
    """The hit flash: a global exposure lift of +0.6 stop at each launch (at strength 1), gone 3 frames later, with
    the iris dip after a big burst. Frames it doesn't touch are copied unchanged. ``out`` may be ``frames_lin``."""
    stops = flash_stops(len(frames_lin), launch_frames, strengths, iris)
    out = _out(frames_lin, out)
    for f in range(len(frames_lin)):
        if stops[f]:
            out[f] = frames_lin[f] * np.float32(2.0 ** stops[f])
        elif out is not frames_lin:
            out[f] = frames_lin[f]
    return out


def shake_offsets(n: int, peak_frames: list[int], strengths: list[float], px: float = 4.0, seed: int = 0,
                  short_side: int = 1080) -> np.ndarray:
    """(n, 2) camera offset in px, per frame: one knock per strike, SHAKE_AFTER frames after its peak, of ``px`` x
    strength at 1080p (scaled to ``short_side``) in a seeded direction, then a wobble at 8-11 Hz dying out over
    SHAKE_FRAMES frames. Never more than ``px`` x strength from a single strike."""
    d = np.zeros((n, 2))
    amp = px * short_side / 1080.0
    for k, (p, st) in enumerate(zip(peak_frames, strengths)):
        rng = _rng(seed, _SHAKE, k)
        a = rng.uniform(0, 2 * np.pi)
        hz = rng.uniform(*SHAKE_HZ, size=2)
        for t in range(SHAKE_FRAMES):
            f = p + SHAKE_AFTER + t
            if 0 <= f < n:
                env = amp * max(0.0, float(st)) * (1 - t / SHAKE_FRAMES) ** 2
                d[f] += env * np.array([np.cos(a) * np.cos(2 * np.pi * hz[0] * t / FPS),
                                        np.sin(a) * np.cos(2 * np.pi * hz[1] * t / FPS)])
    return d


def shake(frames_u8: np.ndarray, peak_frames: list[int], strengths: list[float], px: float = 4.0,
          seed: int = 0) -> np.ndarray:
    """The camera knocked by each strike (``shake_offsets``): translation only, the edge it uncovers filled by
    extending the frame's own edge (a few px, never seen). Frames it doesn't move are returned byte for byte."""
    n, h, w = frames_u8.shape[:3]
    d = shake_offsets(n, peak_frames, strengths, px, seed, min(h, w))
    out = np.array(frames_u8)
    for f in np.nonzero(np.hypot(d[:, 0], d[:, 1]) >= 0.05)[0]:
        M = np.float32([[1, 0, d[f, 0]], [0, 1, d[f, 1]]])
        out[f] = cv2.warpAffine(frames_u8[f], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


# ---- heat haze and the air blast --------------------------------------------------------------------------------------

def _alpha(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) * np.float32(1 / 255) if a.dtype == np.uint8 else a.astype(np.float32, copy=False)


def haze_field(alpha: np.ndarray, t: int, grids: list[np.ndarray], side: int, q: int = 4) -> np.ndarray | None:
    """(h, w, 2) float32 shift, in px, of what is seen through the hot air over a flame of coverage ``alpha`` (h, w)
    on frame ``t``, or None where there is no flame. The hot air reaches HAZE_RISE of the short side above the flame
    (rising) and HAZE_AROUND around it; inside it, ripples (``grids``: seeded noise, two sizes) rise at HAZE_SPEED.
    The shift is HAZE_PX typically and never more than HAZE_MAX (px at 1080p)."""
    h, w = alpha.shape
    if float(alpha.max()) < 1e-3:
        return None
    hq, wq = max(1, h // q), max(1, w // q)
    a = cv2.resize(alpha, (wq, hq), interpolation=cv2.INTER_AREA)
    keep = np.float32(np.exp(-q / (HAZE_RISE * side)))  # per low-res row: hot air carried up from the flame fades out
    for y in range(hq - 2, -1, -1):
        np.maximum(a[y], a[y + 1] * keep, out=a[y])
    hot = np.clip(1.5 * cv2.GaussianBlur(a, (0, 0), max(0.5, HAZE_AROUND * side / q)), 0, 1)
    hot[hot < 0.01] = 0  # the blur's far tail would shift whole frames by a thousandth of a pixel
    gx = (np.arange(wq, dtype=np.float32) + 0.5) * q - 0.5
    gy = (np.arange(hq, dtype=np.float32)[:, None] + 0.5) * q - 0.5
    ripple = np.zeros((hq, wq, 2), np.float32)
    for grid, (cell, weight) in zip(grids, HAZE_CELLS):
        c = cell * side
        mx = np.broadcast_to(gx / c + 1, (hq, wq)).astype(np.float32)
        my = np.broadcast_to((gy + HAZE_SPEED * side * t) / c + 1, (hq, wq)).astype(np.float32)
        ripple += weight * cv2.remap(grid, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    k = side / 1080.0
    shift = ripple * (hot * (HAZE_PX * k))[..., None]
    mag = np.hypot(shift[..., 0], shift[..., 1])
    shift *= np.minimum(1.0, HAZE_MAX * k / np.maximum(mag, 1e-6))[..., None]
    return cv2.resize(shift, (w, h), interpolation=cv2.INTER_LINEAR)  # a blend of capped shifts stays capped


def _haze_grids(frames: int, h: int, w: int, seed: int) -> list[np.ndarray]:
    """The ripples' seeded noise, one grid per size, tall enough for ``frames`` frames of rising. Each grid is drawn
    from its own stream, row by row, so a taller grid starts with the same rows: a clip done in chunks sees the
    same ripples as done whole."""
    side = min(h, w)
    return [_rng(seed, _HAZE, k).standard_normal((int((h + HAZE_SPEED * side * frames) / (cell * side)) + 4,
                                                  int(w / (cell * side)) + 4, 2)).astype(np.float32)
            for k, (cell, _) in enumerate(HAZE_CELLS)]


def _remap_box(frame: np.ndarray, shift: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """``frame`` resampled at (x + shift) inside ``box`` = (y0, y1, x0, x1); ``shift`` is the box's own (bh, bw, 2).
    Past the frame's edge it reads the edge."""
    y0, y1, x0, x1 = box
    mx = np.arange(x0, x1, dtype=np.float32) + shift[..., 0]
    my = np.arange(y0, y1, dtype=np.float32)[:, None] + shift[..., 1]
    return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _box(weight: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(weight)
    if ys.size == 0:
        return None
    h, w = weight.shape[:2]
    return max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)


def heat_haze(frames_lin: np.ndarray, element_alpha: np.ndarray, seed: int = 0, out: np.ndarray | None = None,
              frame0: int = 0) -> np.ndarray:
    """Fire's heat haze: what is seen through the hot air above and around the flame shimmers, rising, by at most
    3 px (``haze_field``). ``element_alpha`` is the warped element's coverage [T, H, W] (float 0..1 or uint8).
    Frames without a flame are copied unchanged. ``out`` may be ``frames_lin``. ``frame0``: the clip's frame number
    of the first frame given, for a clip done in chunks (each chunk then shimmers as the whole clip would)."""
    n, h, w = frames_lin.shape[:3]
    side = min(h, w)
    grids = _haze_grids(frame0 + n, h, w, seed)
    out = _out(frames_lin, out)

    def one(f: int) -> None:
        shift = haze_field(_alpha(element_alpha[f]), frame0 + f, grids, side)
        box = None if shift is None else _box(np.abs(shift).max(-1) > 1e-3, 4)
        if box is None:
            if out is not frames_lin:
                out[f] = frames_lin[f]
            return
        y0, y1, x0, x1 = box
        moved = _remap_box(frames_lin[f], shift[y0:y1, x0:x1], box)
        if out is not frames_lin:
            out[f] = frames_lin[f]
        out[f, y0:y1, x0:x1] = moved

    _each(n, one)
    return out


def _direction(direction) -> tuple[float, float]:
    """A unit vector from [ux, uy], or from an angle in degrees clockwise from up (the studio's)."""
    if np.ndim(direction) == 0:
        a = np.radians(float(direction))
        return float(np.sin(a)), float(-np.cos(a))
    ux, uy = (float(v) for v in direction)
    norm = np.hypot(ux, uy)
    return (ux / norm, uy / norm) if norm > 1e-9 else (0.0, -1.0)


def _lit_tone(frame: np.ndarray) -> np.ndarray:
    """The colour of the room's lit surfaces (the mean of its brightest tenth, at a quarter size): what dust in the
    air, catching the same light, looks like."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    y = _lum(small)
    return small[y >= np.percentile(y, 90)].mean(0, dtype=np.float64).astype(np.float32)


def air_front(k: int, side: int) -> tuple[float, float]:
    """The blast's front on its k-th frame: (radius in px, strength in px). It slows as it spreads, and fades."""
    u = k / AIR_FRAMES
    radius = side * (AIR_FROM + (AIR_TO - AIR_FROM) * (1 - (1 - u) ** 2))
    return radius, AIR_PX * side / 1080.0 * min(1.0, (k + 1) / 2) * (1 - u) ** 1.2


def air_wave(frames_lin: np.ndarray, origin_xy, direction, start_frame: int, seed: int = 0,
             out: np.ndarray | None = None) -> np.ndarray:
    """The air blast (smoke_puff): a pressure front travels out from the palm at ``origin_xy`` over AIR_FRAMES frames
    from ``start_frame``, strongest along ``direction`` (a unit vector, or degrees clockwise from up). It bends what is
    seen through it, outward and back, with the red and blue channels bent a little apart (a slight colour fringe),
    and a thin breath of dust rides its leading edge. Frames outside the blast are copied unchanged."""
    n, h, w = frames_lin.shape[:3]
    side = min(h, w)
    ux, uy = _direction(direction)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    rng = _rng(seed, _AIR)
    lobes = np.arange(3, 8)
    phase, amp = rng.uniform(0, 2 * np.pi, lobes.size), rng.uniform(0.5, 1.0, lobes.size) / lobes
    dust = np.clip(cv2.resize(rng.random((max(2, h // 24), max(2, w // 24))).astype(np.float32), (w, h),
                              interpolation=cv2.INTER_CUBIC), 0, 1)  # patches of dust, so the edge breaks up
    out = _out(frames_lin, out)
    if out is not frames_lin:
        out[:] = frames_lin
    xs, ys = np.arange(w, dtype=np.float32) - ox, np.arange(h, dtype=np.float32)[:, None] - oy
    dist = np.sqrt(xs ** 2 + ys ** 2) + 1e-3
    nx, ny = xs / dist, ys / dist
    cone = np.clip(nx * ux + ny * uy, 0, 1) ** AIR_CONE
    angle = np.arctan2(ys, xs)
    wobble = 1 + 0.05 * (amp[:, None, None] * np.sin(lobes[:, None, None] * angle + phase[:, None, None])).sum(0)
    width = AIR_WIDTH * side / 2
    for k in range(AIR_FRAMES):
        f = start_frame + k
        if not 0 <= f < n:
            continue
        radius, strength = air_front(k, side)
        z = (dist - radius * wobble) / width
        bend = (strength * cone * z * np.exp(0.5 * (1 - z * z))).astype(np.float32)  # out ahead, back behind, <= 1
        box = _box(np.abs(bend) > 0.02, 4)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        frame = frames_lin[f]
        room = _lit_tone(frame)
        sx, sy, b = -nx[y0:y1, x0:x1], -ny[y0:y1, x0:x1], bend[y0:y1, x0:x1]
        for c, more in enumerate((1 + AIR_CA, 1.0, 1 - AIR_CA)):  # each channel copied out before it is written
            shift = np.stack([sx * b * more, sy * b * more], -1)
            out[f, y0:y1, x0:x1, c] = _remap_box(np.ascontiguousarray(frame[..., c]), shift, box)
        edge = AIR_SMOKE * (1 - k / AIR_FRAMES) * cone * np.exp(-0.5 * ((z - 1.0) / 0.8) ** 2) * dust
        out[f] += edge[..., None] * (room - out[f])
    return out


# ---- grain ------------------------------------------------------------------------------------------------------------

def _residual_sigma(y: np.ndarray, keep: np.ndarray | None = None) -> float | None:
    """Robust standard deviation (1.4826 x MAD) of an image's 3x3 high-pass, over ``keep``: its fine noise."""
    r = y - cv2.blur(y, (3, 3))
    v = r[keep] if keep is not None else r.ravel()
    if v.size < 500:
        return None
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def _unit_grain_sigma() -> float:
    """What ``_residual_sigma`` reads off grain made at strength 1 (a fixed field): to scale grain to a plate."""
    return _residual_sigma(_soft_noise(_rng(0, _GRAIN, 99999), 256, 256)) or 1.0


def _soft_noise(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    return cv2.GaussianBlur(rng.standard_normal((h, w), dtype=np.float32), (0, 0), GRAIN_SOFT)


def shadow_noise(plate_u8: np.ndarray, frames: int = 6) -> float:
    """The plate's noise in its shadows, in 8-bit code values of luma (its fine high-pass, in pixels between 6 and 64
    whose surroundings are that dark too, median of a few frames). 2.0 when the plate has no shadows to read."""
    clip = plate_u8 if plate_u8.ndim == 4 else plate_u8[None]
    picks = np.unique(np.linspace(0, len(clip) - 1, min(len(clip), frames)).round().astype(int))
    reads = []
    for f in picks:
        y = clip[f].astype(np.float32) @ _LUMA
        base = cv2.blur(y, (9, 9))
        r = _residual_sigma(y, (base >= 6) & (base <= 64))
        if r is not None:
            reads.append(r)
    return float(np.median(reads)) if reads else 2.0


def grain(layer_lin: np.ndarray, plate_u8: np.ndarray | float, seed: int = 0, out: np.ndarray | None = None,
          frame0: int = 0) -> np.ndarray:
    """Phone-like grain on a clean rendered layer (the element, or the light), matched to the plate: luminance noise,
    slightly blotchy (GRAIN_SOFT), as strong in the 8-bit picture as the plate's own shadow noise wherever the layer
    is (none where it is black), different every frame and the same for the same seed. Never negative.
    ``plate_u8`` is the plate's frames, or its ``shadow_noise`` already measured; with ``frame0`` (the clip's frame
    number of the first frame given) a clip done in chunks gets the same grain as done whole."""
    n, h, w = layer_lin.shape[:3]
    sigma = float(plate_u8) if np.ndim(plate_u8) == 0 else shadow_noise(plate_u8)
    code = sigma / _unit_grain_sigma() / 255.0  # per unit noise, in 8-bit gamma steps
    out = _out(layer_lin, out)

    def one(f: int) -> None:
        layer = layer_lin[f]
        y = np.maximum(_lum(layer), 0)
        # one gamma step at luminance y is 2.2 y^(1.2/2.2) in linear light; the noise keeps the layer's own colour
        step = (2.2 * code) * y ** np.float32(1.2 / 2.2) / np.maximum(y, 1e-6)
        noise = _soft_noise(_rng(seed, _GRAIN, frame0 + f), h, w)
        out[f] = np.maximum(layer * (1 + noise * step)[..., None], 0)

    _each(n, one)
    return out
