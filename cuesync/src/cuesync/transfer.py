"""From an in-scene render back to the shot: timing, and (for real video plates) delta transfer.

An in-scene render starts from a plate frame, so it is a new video of the plate's scene. For a still plate
that is the shot: CueSync only re-times it. For a moving plate the real plate must stay what it is, so only
what the render changed is put onto it, in linear light:

    out = lerp(P * L, gain * G', M),   L = blur(gain * G') / blur(gain * R)

G' is the render frame warped onto the plate frame, R the render's own settled scene before the effect (warped
there too), ``gain`` the colour difference between R and the plate. The render follows the plate's camera, so
both videos are tracked onto the start frame (affine ECC fits on the region the effect never reaches, each
starting from its neighbour's) and the render is resampled once, straight onto its plate frame. The render's
exposure (a model exposing for a fireball darkens the whole room) is measured far from the effect and taken out
before comparing. The change carries the light the effect throws on the scene (the table, the wall, the
prop) as a gain L on the plate (light multiplies what it falls on), so the plate keeps its own texture and
colour under it while the render's brightening, darkening and tint carry over; M is where the STRUCTURE changed (the fire's detail, smoke,
debris: a high-pass difference of the logs, which a smooth change of light does not produce), and there the
render replaces the plate.

The transfer is checked, not trusted: the structural change where the effect never reaches (texture that
moved or was re-drawn, not light) is the residual; above ``MAX_RESIDUAL`` (the render drifted or re-shaped the
scene), or when the effect covers nearly the whole frame, the transfer is rejected and the caller falls back.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from .scene import draw

_LIN8 = ((np.arange(256, dtype=np.float64) / 255.0) ** 2.2).astype(np.float32)
STILL_TOL = 1.0  # mean absolute frame difference (0..255) below which a plate is a still
STILL_BLOCK_TOL = 4.0  # ... and no 1/32-frame block may change more than this (a small moving thing)
MAX_RESIDUAL = 0.06  # mean structural change outside the effect (log high-pass units) a transfer may show
MIN_STATIC = 0.05  # share of the frame the effect must leave untouched to measure the transfer against


def lin(x: np.ndarray) -> np.ndarray:
    return _LIN8[x]


def gam(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def is_still(frames: np.ndarray, tol: float = STILL_TOL, block_tol: float = STILL_BLOCK_TOL) -> bool:
    """Is the plate a held still (a photo turned into a clip)? Every frame is compared with the first, as a
    whole (sensor noise averages out) and block by block (a small thing moving in a corner still counts)."""
    if len(frames) < 2:
        return True
    h, w = frames.shape[1:3]
    blocks = (max(1, w // 32), max(1, h // 32))
    ref = frames[0][::4, ::4].astype(np.float32)
    ref_b = cv2.resize(frames[0], blocks, interpolation=cv2.INTER_AREA).astype(np.float32)
    for f in frames[1:]:
        if float(np.abs(f[::4, ::4].astype(np.float32) - ref).mean()) >= tol:
            return False
        if float(np.abs(cv2.resize(f, blocks, interpolation=cv2.INTER_AREA).astype(np.float32) - ref_b).max()) >= block_tol:
            return False
    return True


def effect_region(blobs: list[dict], size: tuple[int, int], out_size: tuple[int, int] | None = None,
                  grow: float = 0.06) -> np.ndarray:
    """Everywhere the guide ever paints the effect, grown by ``grow`` of the short side (bool, out_size)."""
    w, h = size
    mask = np.zeros((h, w, 3), np.uint8)
    for b in blobs:
        shapes = [dict(s, color=[255, 255, 255]) for s in b["shapes"]]
        mask = np.maximum(mask, draw(shapes, np.zeros((h, w, 3), np.uint8)))
    m = mask[..., 0] > 0
    k = max(3, int(grow * min(w, h)) | 1)
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    if out_size is not None and out_size != size:
        m = cv2.resize(m.astype(np.uint8), out_size, interpolation=cv2.INTER_NEAREST) > 0
    return m


def change_energy(render: np.ndarray, region: np.ndarray, ref_frames: int) -> np.ndarray:
    """Per render frame: mean change inside the effect region against the settled scene (frames before
    ``ref_frames``), measured on a quarter-size grid."""
    small = render[:, ::4, ::4].astype(np.float32)
    reg = region[::4, ::4]
    ref = np.median(small[: max(1, ref_frames)], axis=0)
    if not reg.any():
        reg = np.ones_like(reg)
    return np.array([float(np.abs(f - ref).max(-1)[reg].mean()) for f in small])


def visible_onset(energy: np.ndarray, ref_frames: int, share: float = 0.2) -> int | None:
    """First render frame where the change rises ``share`` of the way from the settled level to its peak."""
    base = float(np.median(energy[: max(1, ref_frames)]))
    peak = float(energy.max())
    if peak - base < 2.0:  # nothing visibly happened
        return None
    above = np.nonzero(energy - base > share * (peak - base))[0]
    above = above[above >= 1]
    return int(above[0]) if above.size else None


VISIBLE = 30  # 8-bit, largest channel: a pixel the effect has visibly changed
FIRST_RISE = 0.06  # an effect has started once its changed share rises this far towards its peak (fitted by eye)
MERGED = 0.15  # a later hit of a rhythm counts only if the change climbs again by this share of the whole
DIED = 0.3  # ... and only if the hit before it had died back by this much first


def changed_share(frames: np.ndarray, region: np.ndarray, ref_frames: int, level: int = VISIBLE,
                  flame: bool = False) -> np.ndarray:
    """Per frame: the share of the effect region's pixels that visibly changed against the settled scene
    (the median of the frames before ``ref_frames``), on a half-size grid. ``flame``: only the changed pixels
    that are flame-coloured (flame dies back between the bursts of a rhythm; the smoke it leaves doesn't)."""
    small = frames[:, ::2, ::2].astype(np.int16)
    reg = region[::2, ::2]
    if not reg.any():
        reg = np.ones_like(reg)
    ref = np.median(small[: max(1, ref_frames)], axis=0)
    out = []
    for f in small:
        m = np.abs(f - ref).max(-1) > level
        if flame:
            m &= (f[..., 0] > 170) & (f[..., 0] - f[..., 2] > 70)
        out.append(float(m[reg].mean()))
    return np.array(out)


def first_visible(share: np.ndarray, ref_frames: int, rise: float = FIRST_RISE) -> int | None:
    """The first frame where the effect is visibly there: its changed share has risen ``rise`` of the way from
    the settled scene to its peak (the first frame a viewer sees it, not the middle of its growth)."""
    base = float(np.median(share[: max(1, ref_frames)]))
    peak = float(share.max())
    if peak - base < 0.01:  # nothing visibly happened
        return None
    above = np.nonzero(share[1:] > base + rise * (peak - base))[0]
    return int(above[0]) + 1 if above.size else None


STEP = 0.05  # a share of the scene outside the effect changing at once, and staying changed: a scene step


def onset_past_steps(frames: np.ndarray, region: np.ndarray, ref_frames: int, share: np.ndarray) -> int | None:
    """``first_visible``, unless at that moment the scene outside the effect changed at least as much as the
    effect region (LTX-2.5 can re-shape the whole scene at a latent block boundary, on a real plate): then the
    onset is measured against the frames just after that step. A fire that lights the room changes the room
    less than the fire's own region, so it is not taken for a step."""
    on = first_visible(share, ref_frames)
    if on is None:
        return None
    out = changed_share(frames, ~region, ref_frames)
    base = float(np.median(out[: max(1, ref_frames)]))
    step = next((f for f in range(max(1, ref_frames), on + 1) if out[f] - out[f - 1] > STEP
                 and out[f: f + 3].min() - base > STEP and out[f] >= share[f]), None)
    if step is None:
        return on
    after = first_visible(changed_share(frames[step:], region, 2), 2)
    return None if after is None else step + after


def hit_onsets(share: np.ndarray, wants: list[int], ref_frames: int) -> list[int | None]:
    """Where each hit of a rhythm shows on the render (``wants``: the render frame each should show on). The
    first: from the settled scene, against its own peak (a small "pf" before a big BOOM). Each later one: the
    hit before it must have died back, then the change climbs clearly again near its own cue. A hit the model
    merged into its neighbour has neither, and is None: it is not guessed."""
    out: list[int | None] = [None] * len(wants)
    base = float(np.median(share[: max(1, ref_frames)]))
    span = float(share.max()) - base
    if span < 0.01 or not wants:
        return out
    end = wants[1] - 2 if len(wants) > 1 else len(share) - 1
    out[0] = first_visible(share[: end + 1], ref_frames)
    prev = out[0]
    for k in range(1, len(wants)):
        lo = max(wants[k - 1] + 2, wants[k] - 10)
        hi = min(wants[k + 1] - 2 if k + 1 < len(wants) else len(share) - 1, wants[k] + 8)
        if hi - lo < 3:
            continue
        m = lo + int(np.argmin(share[lo: min(hi, wants[k] + 2) + 1]))
        before = float(share[min(prev if prev is not None else wants[k - 1], m): m + 1].max())
        top = float(share[m: hi + 1].max())
        if share[m] > (1 - DIED) * before or top - share[m] < MERGED * span:
            continue
        up = np.nonzero(share[m: hi + 1] > share[m] + 2 * FIRST_RISE * (top - share[m]))[0]
        if up.size:
            out[k] = prev = m + int(up[0])
    return out


def steepest_rise(energy: np.ndarray, around: int, before: int = 12, after: int = 16) -> int | None:
    """The frame where the change jumps the most within [around - before, around + after]: the hit of the
    loudest event in a rhythm (earlier events' smoke lifts the level, so a share-of-peak threshold would
    find the first small pop instead)."""
    e = np.convolve(np.pad(energy, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    d = np.diff(e)
    lo, hi = max(0, around - before), min(d.size, around + after)
    if hi <= lo or d[lo:hi].max() < 0.5:
        return None
    return int(lo + np.argmax(d[lo:hi]) + 1)


MAX_WARP = 0.04  # a registration that moves the frame more than this share of its width is not trusted


def _ecc(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, s: int = 4) -> np.ndarray:
    """Affine warp (full resolution) with ref(x) ~ src(W x), fitted where ``mask`` is set, on the frames'
    structure (high-pass log luminance), so that light thrown on part of the scene doesn't pull the fit.
    An implausible fit (a large move or zoom) falls back to no warp. Each frame is fitted on its own, from no warp:
    the onset measure's thresholds were fitted by eye on renders registered this way (``track`` follows a camera
    across a whole clip, for the transfer)."""
    h, w = ref.shape[:2]

    def g(im):
        small = cv2.resize(im, (w // s, h // s), interpolation=cv2.INTER_AREA)
        y = np.log(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 + 0.02)
        return (y - cv2.GaussianBlur(y, (0, 0), 2.0)).astype(np.float32)

    m = cv2.resize(mask.astype(np.uint8), (w // s, h // s), interpolation=cv2.INTER_NEAREST)
    gr, gs = g(ref), g(src)
    ident = np.eye(2, 3, dtype=np.float32)
    warp = ident.copy()
    try:
        _, warp = cv2.findTransformECC(gr, gs, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5), m, 5)
    except cv2.error:
        return ident
    if np.abs(warp[:, :2] - np.eye(2)).max() > MAX_WARP or np.abs(warp[:, 2]).max() * s > MAX_WARP * w:
        return ident
    # keep the warp only if it really aligns the structure better than leaving the frame where it is
    sel = m > 0
    moved = cv2.warpAffine(gs, warp, (w // s, h // s), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)
    if not sel.any() or float(np.abs(moved - gr)[sel].mean()) > 0.9 * float(np.abs(gs - gr)[sel].mean()):
        return ident
    warp[:, 2] *= s
    return warp


TRACK_MAX = 0.25  # a tracked camera may move this share of the width (or scale/turn this much) before a fit is distrusted
COLOUR_MIN, COLOUR_SPAN = 0.08, 0.20  # linear light: near the effect, a change this big replaces the plate
FAR = 0.12  # share of the short side: further than this from the effect, a change of brightness is the camera's exposure
CRUSHED = 0.004  # linear luminance: an 8-bit frame this dark has no texture left to compare, only quantisation
UNSEEN = (0.002, 0.004)  # linear luminance: where the render and the scene are both darker than this, a change of
# structure is noise in the black, not content (it swapped a dark background for the render's slightly bluer one)
LIGHT_EPS = 0.004  # linear light added to both sides of a light ratio, so dark noise can't make a gain
MAX_GAIN = 8.0  # light from the render may brighten the clip this much at most (and darken it to 1/32)
# how much of the render's change of colour carries over, and at most how far per channel: a warmer cast (a fire's
# light) more fully than a cooler one (a warm glow going out, which in full turned white wax mint green)
TINT, MAX_TINT = {"warm": 0.85, "cool": 0.5}, {"warm": 2.0, "cool": 1.4}
FIRE_WARMTH, FIRE_BRIGHT = 0.1, 0.2  # linear light: a flame is warmer than what was there before, and bright
REACH = 0.15  # share of the short side around the effect within which a flame is replaced, not carried as light
# a performer's hand in the shot (fire from the fingertips): behind the source, along the arm, the clip's own pixels
# stay, lit by the effect but never replaced; this far behind the source it starts, this wide either side of the arm
BEHIND, ARM = 0.02, 0.15  # shares of the short side
HAND_GROW, HAND_AREA = 0.03, (0.002, 0.6)  # the performer's matte grown by this share of the short side (the hand
# moves a little as it holds the pose); a matte smaller or bigger than this share of the frame is no performer (SAM may
# give the whole person, hands and chest, which is kept too: but never where the effect was painted)


def behind(point: tuple[float, float], direction: float, w: int, h: int) -> np.ndarray:
    """(h, w) float: the performer's hand and arm, taken to be behind the source (fingertips at ``point``, the
    effect going ``direction``, degrees clockwise from up): from just behind the point back to the frame's edge,
    ARM of the short side either side of that line, softly edged."""
    side = min(w, h)
    a = np.radians(direction)
    ux, uy = np.sin(a), -np.cos(a)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - point[0] * w, yy - point[1] * h
    back = -(dx * ux + dy * uy)  # how far behind the source, along the aim
    across = np.abs(dx * uy - dy * ux)
    m = np.clip((back - BEHIND * side) / (0.03 * side), 0, 1) * np.clip(1 - (across - ARM * side) / (0.05 * side), 0, 1)
    return m.astype(np.float32)


def hand_matte(matte_fn, frame: np.ndarray, point: tuple[float, float], direction: float) -> np.ndarray | None:
    """The performer's hand (or the whole performer) in ``frame`` (the render's start frame, where the pose is
    held), segmented a little behind the source (a tap on the fingertips can land just off them, on the wall), or
    None when what comes back is too small or too big to be one."""
    h, w = frame.shape[:2]
    a = np.radians(direction)
    back = 0.04 * min(w, h)
    at = (float(np.clip(point[0] - back * np.sin(a) / w, 0.0, 1.0)), float(np.clip(point[1] + back * np.cos(a) / h, 0.0, 1.0)))
    m = matte_fn(frame, at)
    if m is None or not HAND_AREA[0] <= float(np.mean(m > 0.5)) <= HAND_AREA[1]:
        return None
    return m.astype(np.float32)


def grown(mask: np.ndarray) -> np.ndarray:
    """A matte grown by HAND_GROW of the short side and softly edged (it covers a hand that sways a little)."""
    h, w = mask.shape[:2]
    r = max(1, int(HAND_GROW * min(w, h)))
    big = cv2.dilate((mask > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return np.clip(cv2.GaussianBlur(big.astype(np.float32), (0, 0), r / 3), 0, 1)


def _hp(im: np.ndarray, s: int) -> np.ndarray:
    """High-pass log luminance at 1/s size: structure, blind to a smooth change of light or exposure."""
    h, w = im.shape[:2]
    small = cv2.resize(im, (w // s, h // s), interpolation=cv2.INTER_AREA)
    y = np.log(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 + 0.02)
    return (y - cv2.GaussianBlur(y, (0, 0), 2.0)).astype(np.float32)


def _m3(A: np.ndarray) -> np.ndarray:
    return np.vstack([A, [0.0, 0.0, 1.0]]).astype(np.float64)


def _scaled(A: np.ndarray, k: float) -> np.ndarray:
    """A warp in pixels of one size, in pixels of a size 1/k of it."""
    B = np.array(A, np.float32)
    B[:, 2] /= k
    return B


def _err(gr: np.ndarray, gs: np.ndarray, sel: np.ndarray, A: np.ndarray) -> float:
    moved = cv2.warpAffine(gs, A, gr.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    return float(np.abs(moved - gr)[sel].mean())


def _fit(gr: np.ndarray, gs: np.ndarray, m: np.ndarray, init: np.ndarray) -> np.ndarray | None:
    """ECC from ``init`` (a warp at this size); None where it does not converge."""
    try:
        _, A = cv2.findTransformECC(gr, gs, np.array(init, np.float32), cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5), m, 5)
    except cv2.error:
        return None
    return A


def track(frames: np.ndarray, ref: np.ndarray, mask: np.ndarray, anchor: int = 0, s: int = 4) -> np.ndarray:
    """Affine warps (n, 2, 3) at full size with ref(x) ~ frames[i](W_i x), fitted on structure where ``mask`` is
    set. A camera that moves far over a clip is followed step by step: each fit starts from its neighbour's, coarse
    to fine, and if it still doesn't converge the frame is fitted to its neighbour and the two warps are chained.
    (Every frame fitted from no warp at all stopped converging once the camera had drifted a few pixels, and then
    fell back to no warp: the render was compared with a scene it was no longer aligned to.) A fit that aligns
    worse than where it started, or moves implausibly far, keeps its neighbour's warp."""
    n = len(frames)
    h, w = ref.shape[:2]
    W = np.repeat(np.eye(2, 3, dtype=np.float32)[None], n, 0)
    m4 = cv2.resize(mask.astype(np.uint8), (w // s, h // s), interpolation=cv2.INTER_NEAREST)
    m8 = cv2.resize(mask.astype(np.uint8), (w // (2 * s), h // (2 * s)), interpolation=cv2.INTER_NEAREST)
    sel = m4 > 0
    if n == 0 or not sel.any() or not m8.any():
        return W
    r4, r8 = _hp(ref, s), _hp(ref, 2 * s)
    g4: dict[int, np.ndarray] = {}

    def hp4(i: int) -> np.ndarray:
        if i not in g4:
            g4[i] = _hp(frames[i], s)
        return g4[i]

    def step(i: int, init: np.ndarray, prev: int | None) -> None:
        start = _scaled(init, s)
        coarse = _fit(r8, _hp(frames[i], 2 * s), m8, _scaled(init, 2 * s))
        A = _fit(r4, hp4(i), m4, start if coarse is None else _scaled(_scaled(coarse, 0.5 / s), s))
        if A is None and prev is not None:  # a small step: the frame onto its neighbour, chained to the neighbour's warp
            D = _fit(hp4(prev), hp4(i), m4, np.eye(2, 3, dtype=np.float32))
            if D is not None:
                A = (_m3(D) @ _m3(start))[:2].astype(np.float32)
        W[i] = init
        if A is None:
            return
        full = _scaled(A, 1.0 / s)
        if (np.abs(full[:, :2] - np.eye(2)).max() <= TRACK_MAX and np.abs(full[:, 2]).max() <= TRACK_MAX * w
                and _err(r4, hp4(i), sel, A) <= _err(r4, hp4(i), sel, start)):
            W[i] = full

    a = int(np.clip(anchor, 0, n - 1))
    step(a, np.eye(2, 3, dtype=np.float32), None)
    for i in range(a + 1, n):
        step(i, W[i - 1], i - 1)
        g4.pop(i - 1, None)
    g4.clear()
    for i in range(a - 1, -1, -1):
        step(i, W[i + 1], i + 1)
        g4.pop(i + 1, None)
    return W


@dataclass
class TransferStats:
    residual: float | None  # mean structural change outside the effect region (lower is better)
    inside: float | None  # ... and inside it, for comparison
    gain: list[float]
    plate_moves: bool
    accepted: bool  # False: the caller must not use this transfer (see reason)
    reason: str = ""
    exposure: float | None = None  # the render's darkest exposure against its settled scene (1 = unchanged)


def _structure(lin_rgb: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """High-pass of the log luminance: edges and texture, blind to a smooth change of light."""
    y = np.log(lin_rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32) + 2e-3)
    return y - cv2.GaussianBlur(y, (0, 0), sigma)


_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, done at a reduced size for a wide sigma (same result to the eye, a fraction of the cost)."""
    h, w = x.shape[:2]
    q = int(max(1, min(8, sigma // 4)))
    if q == 1:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    small = cv2.resize(x, (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), sigma / q), (w, h), interpolation=cv2.INTER_LINEAR)


def light_gain(gl: np.ndarray, rl: np.ndarray, sigma: float, weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """How the render's light changed, as a per-channel gain for the clip, and how much trusted support it had.
    The clip keeps its own texture and colour under the gain, while the render's brightening, darkening and tint
    carry over. (Added and subtracted differences washed texture out under strong light, drove channels below
    zero where light was taken away, and turned a room lit only by a fire grey instead of warm.) The per-pixel
    ratio is averaged in log over ``sigma``: a change of light or exposure scales every surface alike, so the
    ratio is smooth across an object's edge where a ratio of two blurs is not (that drew a halo round
    everything). Pixels ``weight`` doesn't trust are filled in from around them (normalised convolution). The
    brightness change carries over whole, its colour partly (``TINT``)."""
    r = np.log((gl + LIGHT_EPS) / (rl + LIGHT_EPS))
    w = np.ones(gl.shape[:2], np.float32) if weight is None else weight.astype(np.float32)
    support = _smooth(w, sigma)
    g = _smooth(r * w[..., None], sigma) / np.maximum(support, 1e-4)[..., None]
    lum = (g @ _LUMA)[..., None]
    c = g - lum
    warm = (c[..., 0] - c[..., 2])[..., None] > 0
    cap = np.where(warm, np.log(MAX_TINT["warm"]), np.log(MAX_TINT["cool"]))
    g = lum + np.clip(np.where(warm, TINT["warm"], TINT["cool"]) * c, -cap, cap)
    return np.exp(np.clip(g, -np.log(32.0), np.log(MAX_GAIN))).astype(np.float32), support


def render_exposure(render: np.ndarray, R8: np.ndarray, Ar: np.ndarray, far: np.ndarray, q: int = 4) -> np.ndarray:
    """Per render frame: its brightness far from the effect against its settled scene (a model that exposes for a
    fireball darkens the whole room, by up to 3 stops). Measured where the frame has texture and isn't clipped,
    and smoothed over 5 frames."""
    h, w = R8.shape[:2]
    ref = cv2.resize(lin(R8), (w // q, h // q), interpolation=cv2.INTER_AREA) @ _LUMA
    fq = cv2.resize(far.astype(np.uint8), (w // q, h // q), interpolation=cv2.INTER_NEAREST) > 0
    out = np.ones(len(render), np.float32)
    for e, f in enumerate(render):
        g = cv2.warpAffine(cv2.resize(f, (w // q, h // q), interpolation=cv2.INTER_AREA), _scaled(Ar[e], q), (w // q, h // q),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        y = lin(g) @ _LUMA
        ok = fq & (ref > 0.01) & (ref < 0.8) & (y < 0.8)
        if ok.sum() > 50:
            out[e] = float(np.clip(np.median(y[ok] / ref[ok]), 1 / 16, 4.0))
    pad = np.pad(out, 2, mode="edge")
    return np.array([np.median(pad[i: i + 5]) for i in range(len(out))], np.float32)


def transfer(plate: np.ndarray, render: np.ndarray, index: list[int | None], region: np.ndarray, ref_frames: int,
             start: int, threads: int = 8, keep: np.ndarray | None = None) -> tuple[np.ndarray, TransferStats]:
    """Delta-transfer ``render`` (already at the plate's size) onto ``plate``. ``index[f]`` is the render
    frame for plate frame f (None: before the render starts, the plate is left alone). After the render
    ends, its last frame's change stays (what the effect left behind stays). Check ``stats.accepted``.

    The render follows the plate's camera (its scene is painted from the plate's frames), so both are tracked
    onto the start frame, and each render frame is warped once, straight onto its plate frame, with the settled
    scene it is compared against warped there too. The render's own exposure dip is measured and taken out
    before anything is compared (structure, change), but the shot keeps it: a camera exposing for a fireball
    darkens the room, and taking the dip out of the picture too amplified the render's crushed colour into a cast.

    ``keep`` (start-frame layout, see ``behind``): where the clip's own pixels must stay, lit by the effect but
    never replaced (a performer's hand, which moves on its own and which the render re-draws as it likes); it is
    also left out of the camera tracking and of the residual."""
    n, h, w = plate.shape[:3]
    static = ~region
    hand = np.zeros((h, w), np.float32) if keep is None else np.clip(keep, 0, 1).astype(np.float32)
    track_on = static & (hand < 0.5)  # a moving hand would drag the fit of the camera with it
    moves = not is_still(plate)
    if static.mean() < MIN_STATIC:
        return plate.copy(), TransferStats(residual=None, inside=None, gain=[1.0, 1.0, 1.0], plate_moves=moves, accepted=False,
                                           reason="the effect covers nearly the whole frame")
    R8 = np.median(render[: max(1, ref_frames)], axis=0).astype(np.uint8)
    R = lin(R8)
    a = min(max(start, 0), n - 1)
    p0 = plate[a]
    gain = (np.median(lin(p0)[static] + 1e-3, axis=0) / np.median(R[static] + 1e-3, axis=0)).astype(np.float32)
    gain = np.where(np.isfinite(gain), np.clip(gain, 0.25, 4.0), 1.0).astype(np.float32)
    rl = R * gain
    r_struct = _structure(rl)
    if track_on.mean() < MIN_STATIC:
        track_on = static
    Ar = track(render, R8, track_on)  # settled render -> render frame e
    Ap = track(plate, p0, track_on, anchor=a) if moves else np.repeat(np.eye(2, 3, dtype=np.float32)[None], n, 0)  # start -> f
    far = cv2.distanceTransform(static.astype(np.uint8), cv2.DIST_L2, 5) > FAR * min(w, h)
    far &= hand < 0.5  # a performer moves on their own: their hands and clothes are no measure of the exposure
    if far.mean() < MIN_STATIC:
        far = static
    k_exp = render_exposure(render, R8, Ar, far)
    # the render may replace the plate only around the effect; elsewhere only its smooth light reaches the
    # plate, and not where the render's own content moved (else anything moving in the plate would ghost)
    feather = cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(2.0, 0.02 * min(w, h)))
    feather = np.clip(feather * 2.0, 0.0, 1.0)
    hand = hand * (1.0 - feather)  # where the effect was painted it goes in front of the performer (fire up past the chest)
    near = np.clip(cv2.GaussianBlur(region.astype(np.float32), (0, 0), max(2.0, 0.05 * min(w, h))) * 3.0, 0.0, 1.0)
    # the render's flame can grow past where it was painted (taller, over a bright window); a flame is replaced, not
    # carried as light, this far around the effect (as light alone, its top half went see-through over the window)
    away = cv2.distanceTransform((~region).astype(np.uint8), cv2.DIST_L2, 5) / min(w, h)  # share of the short side
    reach = np.clip(1.0 - (away - REACH) / 0.05, 0.0, 1.0).astype(np.float32)
    ones = np.ones((h, w), np.float32)
    ident = np.eye(2, 3, dtype=np.float32)

    out = np.empty_like(plate)
    res_in: list[float] = []
    res_out: list[float] = []

    def one(f: int) -> None:
        e = index[f]
        if e is None:
            out[f] = plate[f]
            return
        P = Ap[f]
        same = np.allclose(P, ident, atol=1e-4)

        def onto(x: np.ndarray, border: int = cv2.BORDER_CONSTANT) -> np.ndarray:  # start-frame layout -> plate frame f
            return x if same else cv2.warpAffine(x, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)

        # plate frame f -> start frame -> render frame e: one warp, so the render is resampled only once
        M = (_m3(Ar[e]) @ _m3(cv2.invertAffineTransform(P)))[:2].astype(np.float32)
        G = cv2.warpAffine(render[e], M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        pl = lin(plate[f])
        seen = onto(ones)  # what the start frame saw (a moving camera reveals new ground at the edges)
        valid = seen > 0.99
        if same:
            rlf, rsf = rl, r_struct
        else:  # where the start frame never looked, the clip itself is the settled scene (cut off there, the fire
            # left a hard-edged box where new ground came into view)
            rlf = onto(rl, cv2.BORDER_REFLECT) * seen[..., None] + pl * (1.0 - seen)[..., None]
            rsf = _structure(rlf)
        rep_ = cv2.BORDER_REPLICATE
        feat, nearf, reachf = onto(feather, rep_), onto(near, rep_), onto(reach, rep_)
        reg_f = onto(region.astype(np.float32), rep_) > 0.5
        hand_f = onto(hand, rep_) if keep is not None else None
        stat_f = ~reg_f & valid & (hand_f < 0.5 if hand_f is not None else True)
        k = float(k_exp[e])
        gl = lin(G) * gain
        norm = gl / k  # at the settled scene's exposure, for comparing only
        sdiff = np.abs(_structure(norm) - rsf)
        y_g, y_r = norm.max(-1), rlf.max(-1)
        # clipped light is light, and a frame the model darkened to a few 8-bit levels has no texture to compare
        trusted = (y_g < 0.85) & (y_r < 0.85) & (y_r > 2e-3) & ((gl @ _LUMA) > CRUSHED)
        changed = np.clip((cv2.GaussianBlur(sdiff, (0, 0), 2.0) - 0.06) / 0.18, 0.0, 1.0)
        seen_y = cv2.GaussianBlur(np.maximum(norm @ _LUMA, rlf @ _LUMA), (0, 0), 2.0)
        changed = changed * np.clip((seen_y - UNSEEN[0]) / UNSEEN[1], 0.0, 1.0)  # black on black stays the clip's
        # near the effect a smooth, bright change (a fireball, a plume of smoke) has little structure, and taken as
        # light it was added to the plate as a see-through ghost; so there plain change of colour counts too
        colour = np.clip((cv2.GaussianBlur(np.abs(norm - rlf).max(-1), (0, 0), 2.0) - COLOUR_MIN) / COLOUR_SPAN, 0.0, 1.0)
        changed = np.maximum(changed, colour * nearf)
        Mm = cv2.GaussianBlur(changed, (0, 0), 3.0) * feat
        warmer = (norm[..., 0] - norm[..., 2]) - (rlf[..., 0] - rlf[..., 2])
        fire = np.clip((warmer - FIRE_WARMTH) / 0.2, 0.0, 1.0) * np.clip((norm @ _LUMA - FIRE_BRIGHT) / 0.2, 0.0, 1.0)
        Mm = np.maximum(Mm, cv2.GaussianBlur(fire, (0, 0), 3.0) * reachf)
        # near the effect its light in some detail; elsewhere smooth light, taken only where the render's own content
        # didn't move (else anything moving in the render would print onto the clip) and filled in from around it
        quiet = np.clip(1.0 - cv2.dilate(cv2.GaussianBlur(changed, (0, 0), 4.0), np.ones((9, 9), np.uint8)), 0.0, 1.0)
        good = quiet * valid * trusted
        if hand_f is not None:  # the render re-draws a hand as it likes (a green hand is not green light): the light
            good = good * (1.0 - hand_f)  # on it is filled in from around it
        room, _ = light_gain(gl, rlf, 0.12 * min(w, h), good + 1e-3)
        local, support = light_gain(gl, rlf, 16.0, good)
        a3 = np.clip(support / 0.2, 0.0, 1.0)[..., None]
        far_light = local * a3 + room * (1.0 - a3)
        detail = feat if hand_f is None else feat * (1.0 - hand_f)  # on the hand only smooth light: its detail is the render's
        L = light_gain(gl, rlf, 4.0, valid * trusted + 1e-3)[0] * detail[..., None] + far_light * (1.0 - detail)[..., None]
        if hand_f is not None:
            Mm = Mm * (1.0 - hand_f)
        M3 = Mm[..., None]
        out[f] = gam(pl * L * (1.0 - M3) + gl * M3)
        res_in.append(float(sdiff[reg_f].mean()) if reg_f.any() else 0.0)
        ok = stat_f & trusted
        res_out.append(float(sdiff[ok].mean()) if ok.any() else 0.0)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, range(n)))
    residual = float(np.mean(res_out)) if res_out else 0.0
    inside = float(np.mean(res_in)) if res_in else 0.0
    ok = residual <= MAX_RESIDUAL
    used = [k_exp[e] for e in index if e is not None]
    return out, TransferStats(residual=round(residual, 4), inside=round(inside, 4), gain=[round(float(g), 3) for g in gain],
                              plate_moves=moves, accepted=ok, exposure=round(float(min(used)), 3) if used else None,
                              reason="" if ok else
                              f"the render moved or re-shaped the scene outside the effect (residual {residual:.2f})")
