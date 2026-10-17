"""From an in-scene render back to the shot: timing, and (for real video plates) delta transfer.

An in-scene render starts from a plate frame, so it is a new video of the plate's scene. For a still plate
that is the shot: CueSync only re-times it. For a moving plate the real plate must stay what it is, so only
what the render changed is put onto it, in linear light:

    out = lerp(P + gain * (G' - R), gain * G', M)

G' is the render frame registered onto R (an affine ECC fit on the region the effect never reaches), R the
render's own settled scene before the effect, ``gain`` the colour difference between R and the plate. The
difference carries the light the effect throws on the scene (the table, the wall, the prop) and keeps the
plate's own texture under it; M is where the STRUCTURE changed (the fire's detail, smoke, debris: a high-pass
difference of the logs, which a smooth change of light does not produce), and there the render replaces the
plate. If the plate moves, these layers follow it (a second ECC fit, plate frame onto the start frame).

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
    An implausible fit (a large move or zoom) falls back to no warp."""
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


@dataclass
class TransferStats:
    residual: float | None  # mean structural change outside the effect region (lower is better)
    inside: float | None  # ... and inside it, for comparison
    gain: list[float]
    plate_moves: bool
    accepted: bool  # False: the caller must not use this transfer (see reason)
    reason: str = ""


def _structure(lin_rgb: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """High-pass of the log luminance: edges and texture, blind to a smooth change of light."""
    y = np.log(lin_rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32) + 2e-3)
    return y - cv2.GaussianBlur(y, (0, 0), sigma)


def transfer(plate: np.ndarray, render: np.ndarray, index: list[int | None], region: np.ndarray, ref_frames: int,
             start: int, threads: int = 8) -> tuple[np.ndarray, TransferStats]:
    """Delta-transfer ``render`` (already at the plate's size) onto ``plate``. ``index[f]`` is the render
    frame for plate frame f (None: before the render starts, the plate is left alone). After the render
    ends, its last frame's change stays (what the effect left behind stays). Check ``stats.accepted``."""
    n, h, w = plate.shape[:3]
    static = ~region
    moves = not is_still(plate)
    if static.mean() < MIN_STATIC:
        return plate.copy(), TransferStats(residual=None, inside=None, gain=[1.0, 1.0, 1.0], plate_moves=moves, accepted=False,
                                           reason="the effect covers nearly the whole frame")
    R8 = np.median(render[: max(1, ref_frames)], axis=0).astype(np.uint8)
    R = lin(R8)
    p0 = plate[min(max(start, 0), n - 1)]
    gain = (np.median(lin(p0)[static] + 1e-3, axis=0) / np.median(R[static] + 1e-3, axis=0)).astype(np.float32)
    gain = np.where(np.isfinite(gain), np.clip(gain, 0.25, 4.0), 1.0).astype(np.float32)
    rl = R * gain
    r_struct = _structure(rl)
    reg_cache: dict[int, np.ndarray] = {}

    def layers(e: int):
        if e not in reg_cache:
            reg_cache[e] = _ecc(render[e], R8, static)
        Gw = cv2.warpAffine(render[e], reg_cache[e], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REFLECT)
        gl = lin(Gw) * gain
        sdiff = np.abs(_structure(gl) - r_struct)
        y_g, y_r = gl.max(-1), rl.max(-1)
        sdiff_valid = np.where((y_g < 0.85) & (y_r < 0.85) & (y_r > 2e-3), sdiff, 0.0)  # clipped light is light
        M = cv2.GaussianBlur(np.clip((cv2.GaussianBlur(sdiff, (0, 0), 2.0) - 0.06) / 0.18, 0.0, 1.0), (0, 0), 3.0)
        return gl, gl - rl, M, sdiff_valid

    out = np.empty_like(plate)
    res_in, res_out = [], []

    def one(f: int) -> None:
        e = index[f]
        if e is None:
            out[f] = plate[f]
            return
        gl, delta, M, sdiff = layers(e)
        if moves and f != start:  # the layers live in the start frame's coordinates: follow the plate
            A = _ecc(plate[f], p0, static)
            gl = cv2.warpAffine(gl, A, (w, h), borderMode=cv2.BORDER_REFLECT)
            delta = cv2.warpAffine(delta, A, (w, h), borderMode=cv2.BORDER_CONSTANT)
            M = cv2.warpAffine(M, A, (w, h), borderMode=cv2.BORDER_CONSTANT)
        pl = lin(plate[f])
        M3 = M[..., None]
        out[f] = gam(np.maximum(pl + delta, 0.0) * (1.0 - M3) + gl * M3)
        res_in.append(float(sdiff[region].mean()) if region.any() else 0.0)
        res_out.append(float(sdiff[static].mean()))

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, range(n)))
    residual = float(np.mean(res_out)) if res_out else 0.0
    inside = float(np.mean(res_in)) if res_in else 0.0
    ok = residual <= MAX_RESIDUAL
    return out, TransferStats(residual=round(residual, 4), inside=round(inside, 4), gain=[round(float(g), 3) for g in gain],
                              plate_moves=moves,
                              accepted=ok, reason="" if ok else
                              f"the render moved or re-shaped the scene outside the effect (residual {residual:.2f})")
