"""Composite an over-black fluid element onto untouched footage.

An element rendered over black is a premultiplied image without an alpha channel: one RGB value can't say
how much is density and how much is light, so each material gets its own model. Blending happens in linear
light; the material split and the shoulder work on display values.
- Fire (warm, saturated, bright pixels, plus white-hot cores grown from them) is emission with an opaque
  core: out = plate*(1-a) + E, where a may reach ~1 in the core (the element's own orange structure shows,
  nothing washes out) but never exceeds what keeps the pixel at least as bright as the plate behind it, so
  fire never darkens what it passes in front of (no dark fringes on bright plates).
- Smoke and steam get a density alpha (luma relative to the densest smoke, plus an opaque interior measured
  from the cloud's soft coverage) and are relit: their colour is the plate's local ambient light times the
  material's albedo, modulated by the render's own shading, plus some of the render's light. White smoke
  stays white and billowy on a bright wall, dims to the room in a dark one, and soot (low albedo) reads as
  dense dark smoke anywhere.
Fire also lights the plate (a soft multiplicative pool with a finite reach, stronger on the object it comes
from) and blooms.

Voice gates scale what belongs to the flame (any warm pixel, however dim) with the fire gate and the rest
with the smoke gate, so a flame goes out as a whole instead of leaving a ghost. Fire dims like fire (red
lasts longest, blue goes first) and a sustained flame lifts off its base as it goes out. The object the
effect comes from, when its matte is known, stays in front of everything below the emission point;
without a matte a soft horizontal holdout stands in. Plate-matched grain goes on wherever the element
contributes. Pixels nothing touches are copied from the plate exactly.
"""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .align import Alignment, element_index_for
from .score import ElementTiming

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class CompositeParams:
    # material split
    sat_lo: float = 0.25
    sat_hi: float = 0.5
    warm_hue_max_deg: float = 70.0
    fire_bright_lo: float = 0.35  # dark warm pixels (fire-lit smoke) are smoke for alpha, but go out with the fire
    fire_bright_hi: float = 0.6
    core_lo: float = 0.7  # white-hot pixels next to warm fire are fire too
    core_hi: float = 0.9
    core_reach_frac: float = 0.06  # how far (fraction of width) a white-hot core may extend from warm fire
    # fire: emission with an opaque core, never darker than the plate behind it
    fire_occlusion: float = 0.97
    fire_alpha_lo: float = 0.2  # on luma: a flame's dim red periphery is mostly light, its body and core opaque
    fire_alpha_hi: float = 0.6
    fire_gain: float = 1.0
    fire_lift_frac: float = 0.5  # a sustained flame going out lifts off its base this far (fraction of height)
    fire_darken: float = 0.2  # how much darker than the plate a flame's body may get (0: pure light, never darker)
    fringe_glow: tuple[float, float, float] = (1.0, 0.45, 0.08)  # a flame's dim skirt glows orange, not deep red
    # smoke and steam: density alpha, relit by the room
    smoke_opacity: float = 1.0  # dense cores: in linear light even a few % transmission shows a bright plate
    smoke_alpha_gamma: float = 0.8  # thin smoke is a little denser than its luma says
    smoke_mask_level: float = 0.06  # luma where a pixel starts to belong to the cloud
    interior_blur_frac: float = 0.04
    interior_lo: float = 0.45
    interior_hi: float = 0.9
    ambient_blur_frac: float = 0.15  # the room light reaching the smoke: the plate blurred this wide
    shading_lo: float = 0.35  # the render's shading is kept as a factor in [lo, hi] around 1
    shading_hi: float = 1.6
    shading_gamma: float = 0.7  # < 1 softens it: dense white smoke stays white, just shaded
    smoke_desat: float = 0.6  # the model lights smoke like a sky (blue-grey); smoke itself is near neutral
    holdout_soft_frac: float = 0.1  # without an object matte, the cut below the emission point fades over this
    object_light_boost: float = 0.6  # the object the fire comes from catches extra light...
    object_light_soft_frac: float = 0.015  # ...through a softened matte, so a rough matte never draws an edge
    # light on the plate, and bloom
    light_gain: float = 0.6
    light_radius: float = 1.5  # x the fire's radius
    light_reach: float = 6.0  # x light_radius: beyond it the plate is untouched
    light_tint: tuple[float, float, float] = (1.0, 0.55, 0.22)
    bloom_gain: float = 0.5
    bloom_sigma_frac: float = 0.02
    bloom_threshold: float = 0.6
    # finishing
    shoulder: float = 0.85  # what is added rolls off softly above this instead of clipping (hue kept)
    grain: bool = True
    black_floor: int | None = None  # 8-bit levels of codec noise / model haze; None = measured per element
    end_fade_frames: int = 20  # the element's own end: fire fades over this many frames...
    smoke_end_frames: int = 36  # ...and smoke dissipates over ~1.5 s instead of vanishing


# ------------------------------------------------------------------ helpers

def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lin(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0.0, None), 2.2, dtype=np.float32)


def _gam(x: np.ndarray) -> np.ndarray:
    return np.power(np.clip(x, 0.0, None), 1.0 / 2.2, dtype=np.float32)


_LIN8 = ((np.arange(256, dtype=np.float64) / 255.0) ** 2.2).astype(np.float32)
_GAM16 = ((np.arange(65536, dtype=np.float64) / 65535.0) ** (1.0 / 2.2)).astype(np.float32)


def _gam_fast(x: np.ndarray) -> np.ndarray:
    """_gam through a 16-bit table (values above 1 computed exactly: the shoulder needs them)."""
    g = _GAM16[np.clip(x * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)]
    big = x > 1.0
    if big.any():
        g[big] = np.power(x[big], 1.0 / 2.2, dtype=np.float32)
    return g


_LIGHT_STEPS, _LIGHT_N = 2048, 8192  # light gains up to 4, in steps of 1/2048
_light_tables: dict = {}


def _light_table(tint: tuple[float, float, float]) -> np.ndarray:
    """Display-value gain per quantised light level: (1 + 2.2 L tint) ** (1/2.2), per channel."""
    if tint not in _light_tables:
        lv = np.arange(_LIGHT_N, dtype=np.float64)[:, None] / _LIGHT_STEPS
        _light_tables[tint] = ((1.0 + 2.2 * lv * np.asarray(tint)) ** (1.0 / 2.2)).astype(np.float32)
    return _light_tables[tint]


def _floor_tables(floor: int) -> tuple[np.ndarray, np.ndarray]:
    """uint8 element value -> (display value, linear value) with its black level removed."""
    v = np.clip((np.arange(256, dtype=np.float64) - floor) / (255 - floor), 0.0, 1.0) if floor > 0 \
        else np.arange(256, dtype=np.float64) / 255.0
    return v.astype(np.float32), (v ** 2.2).astype(np.float32)


def _dissipate(a: np.ndarray, d: float) -> np.ndarray:
    """Gain for smoke of density ``a`` that is ``d`` (0..1) of the way to gone: what is thinner than the
    receding threshold vanishes, the rest thins, and everything is gone at d = 1."""
    if d <= 0.0:
        return np.ones_like(a)
    if d >= 1.0:
        return np.zeros_like(a)
    return _smoothstep(0.6 * d - 0.15, 0.6 * d + 0.35, a) * (1.0 - d) ** 1.2


def _blackbody(e: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Fire dimmed by k in [0, 1] the way fire dims: red lasts, green follows, blue goes first."""
    k = np.clip(k, 0.0, 1.0)[..., None]
    return e * np.concatenate([k, k ** 1.7, k ** 3], axis=-1)


def ambient_map(plate_frame: np.ndarray, params: CompositeParams) -> np.ndarray:
    """The room light around every pixel (linear RGB, plate size): the plate blurred wide, from its first
    frame (plates are locked off, and one map keeps the smoke from flickering with the footage)."""
    h, w = plate_frame.shape[:2]
    s = 8
    small = _lin(np.asarray(Image.fromarray(plate_frame).resize((max(1, w // s), max(1, h // s)), Image.Resampling.BOX),
                            dtype=np.float32) / 255.0)
    blurred = np.stack([_blur(small[..., c], params.ambient_blur_frac * w / s) for c in range(3)], -1)
    return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(blurred[..., c]), "F").resize((w, h), Image.Resampling.BILINEAR),
                                dtype=np.float32) for c in range(3)], -1)


def smoke_shading_ref(element: np.ndarray, params: CompositeParams, floor: int, fire_effect: bool, ref: float) -> float:
    """Median un-premultiplied linear luma of the element's smoke (one value per shot, so shading never pumps)."""
    sample = element[:: max(1, element.shape[0] // 16), ::4, ::4]
    vals = []
    for f in sample:
        e = _black_floor_f(f, floor)
        y = e @ _LUMA
        a = np.clip(y / ref, 0.0, 1.0) ** params.smoke_alpha_gamma
        m = (a > 0.3) & (fire_weight_map(e, params, f.shape[1], fire_effect) < 0.1)
        if m.any():
            vals.append((_lin(e[m]) / np.maximum(a[m], 0.05)[:, None]) @ _LUMA)
    return float(np.median(np.concatenate(vals))) if vals else 0.25


def _peak(rgb: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])


def _hue_sat(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = e[..., 0], e[..., 1], e[..., 2]
    mx, mn = _peak(e), np.minimum(np.minimum(r, g), b)
    d = mx - mn
    sat = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    hue = np.zeros_like(mx)
    m = d > 0
    rm = m & (mx == r)
    gm = m & (mx == g) & ~rm
    bm = m & ~rm & ~gm
    safe = np.where(m, d, 1.0)
    hue = np.where(rm, ((g - b) / safe) % 6.0, hue)
    hue = np.where(gm, (b - r) / safe + 2.0, hue)
    hue = np.where(bm, (r - g) / safe + 4.0, hue)
    return hue * 60.0, sat


def _box1d(x: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return x
    pad = [(0, 0)] * x.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(x, pad, mode="edge"), axis=axis, dtype=np.float64)
    n = x.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def _blur(x: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur of a float map (three box passes), done at reduced resolution when it is wide."""
    if sigma_px < 0.5:
        return x.astype(np.float32)
    h, w = x.shape
    scale = max(1, int(sigma_px // 3))
    small = x.astype(np.float32)
    if scale > 1:
        small = np.asarray(Image.fromarray(small, "F").resize((max(1, w // scale), max(1, h // scale)),
                                                             Image.Resampling.BOX), dtype=np.float32)
    s = sigma_px / scale
    r = max(1, int(round((np.sqrt(4.0 * s * s + 1.0) - 1.0) / 2.0)))  # three passes of width 2r+1 ~ sigma s
    for _ in range(3):
        small = _box1d(_box1d(small, r, 0), r, 1)
    if scale > 1:
        small = np.asarray(Image.fromarray(small, "F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    return small


def estimate_black_level(element: np.ndarray) -> int:
    """The element's 'black': per-channel median of a border band over sampled frames, plus 2 levels."""
    s = element[:: max(1, element.shape[0] // 12)]
    b = max(2, min(s.shape[1], s.shape[2]) // 32)
    band = np.concatenate([s[:, :b].reshape(-1, 3), s[:, -b:].reshape(-1, 3),
                           s[:, :, :b].reshape(-1, 3), s[:, :, -b:].reshape(-1, 3)])
    return int(min(max(np.median(band, axis=0).max() + 2, 2), 24))


def _apply_black_floor(elem8: np.ndarray, floor: int) -> np.ndarray:
    if floor <= 0:
        return elem8
    e = elem8.astype(np.int16) - floor
    np.clip(e, 0, 255 - floor, out=e)
    return ((e.astype(np.float32) * (255.0 / (255 - floor))) + 0.5).astype(np.uint8)


def _black_floor_f(elem8: np.ndarray, floor: int) -> np.ndarray:
    """The element in [0, 1] with its black level removed, kept in float (no second 8-bit rounding)."""
    e = elem8.astype(np.float32)
    if floor > 0:
        e = np.clip((e - floor) * (1.0 / (255 - floor)), 0.0, 1.0)
    else:
        e *= 1.0 / 255.0
    return e


def _warmth(e: np.ndarray, params: CompositeParams) -> tuple[np.ndarray, np.ndarray]:
    """(warm and saturated, regardless of brightness; the same and bright enough to be flame)."""
    hue, sat = _hue_sat(e)
    warm = _smoothstep(params.sat_lo, params.sat_hi, sat) * ((hue <= params.warm_hue_max_deg) | (hue >= 345.0))
    return warm, warm * _smoothstep(params.fire_bright_lo, params.fire_bright_hi, _peak(e))


def fire_weight_map(e: np.ndarray, params: CompositeParams, full_width: int, fire_effect: bool = True,
                    base: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel fire weight: warm saturated bright pixels, plus white-hot cores grown from them."""
    if base is None:
        _, base = _warmth(e, params)
    if not fire_effect:
        return base
    near = _smoothstep(0.01, 0.1, _blur(base, params.core_reach_frac * full_width))
    redness = _smoothstep(-0.04, 0.0, e[..., 0] - e[..., 2])  # codec noise can push a white core's blue past red
    core = _smoothstep(params.core_lo, params.core_hi, e @ _LUMA) * near * redness
    return np.maximum(base, core)


def smoke_reference_luma(element: np.ndarray, params: CompositeParams, floor: int, fire_effect: bool) -> float:
    """Luma of the densest smoke in the element (fire and its white-hot cores excluded): luma relative to it is
    a luminous fluid's alpha. The brightest pixel whose 3x3 neighbourhood agrees with it: an isolated stray
    pixel can't set it, and a smooth peak keeps its true value."""
    sample = element[:: max(1, element.shape[0] // 24), ::2, ::2]
    e = _apply_black_floor(sample, floor).astype(np.float32) / 255.0
    w = sample.shape[2]
    ref = 0.0
    for f in e:
        y = (f @ _LUMA) * (fire_weight_map(f, params, w, fire_effect) < 0.1)
        supported = _box1d(_box1d(y, 1, 0), 1, 1) >= 0.7 * y
        if supported.any():
            ref = max(ref, float(y[supported].max()))
    return min(max(ref if ref > 0.02 else 0.62, 0.3), 1.0)


def plate_grain_sigma(plate: np.ndarray, max_frames: int = 7) -> float:
    """Sensor/codec noise of a locked-off plate (8-bit levels): robust spread of frame minus temporal median."""
    if plate.shape[0] < 3:
        return 0.0
    idx = np.linspace(0, plate.shape[0] - 1, min(max_frames, plate.shape[0])).astype(int)
    h, w = plate.shape[1:3]
    crop = plate[idx, h // 4: 3 * h // 4: 2, w // 4: 3 * w // 4: 2].astype(np.float32)
    med = np.median(crop, axis=0)
    resid = crop[len(idx) // 2] - med
    return float(1.4826 * np.median(np.abs(resid - np.median(resid))))


def voice_gates(n_frames: int, intervals: list[tuple[int, int]], anchor: int, mode: str,
                release_fire: int, release_smoke: int, open_from: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per plate frame (fire_gain, smoke_gain) in [0, 1].

    ``open_from``: the first plate frame the element is shown (a hit-aligned burst shows its build-up before
    the sound); the gates open there, never later than the anchor.
    "attack": open from then on (the effect's own render decides how it ends).
    "both": open while the performer is making the noise, then a finite linear release (fire collapses in
    ``release_fire`` frames, smoke disperses in ``release_smoke``); a new phrase re-attacks.
    """
    gf, gs = np.zeros(n_frames), np.zeros(n_frames)
    start = max(0, min(anchor, anchor if open_from is None else open_from))
    if start >= n_frames:
        return gf, gs
    if mode == "attack":
        gf[start:], gs[start:] = 1.0, 1.0
        return gf, gs
    if mode != "both":
        raise ValueError(f"unknown voice gate {mode!r}")
    a0 = max(anchor, 0)
    active = np.zeros(n_frames, dtype=bool)
    for a, b in intervals:
        if b > a0:  # phrases from the anchor on; an earlier blip doesn't hold the gate
            active[max(a, a0): min(b, n_frames)] = True
    active[start:a0] = True  # the shown build-up before the sound
    # the onset can fire a frame or two before the level rises: bridge that, never a pause to a later phrase
    nxt = min((a for a, b in intervals if b > a0), default=None)
    if nxt is not None and a0 < nxt <= a0 + 2:
        active[a0:nxt] = True
    if not active[a0:].any() and a0 < n_frames:
        active[a0] = True
    last_on = None
    for p in range(start, n_frames):
        if active[p]:
            gf[p] = gs[p] = 1.0
            last_on = p
        elif last_on is not None:
            k = p - last_on
            gf[p] = max(0.0, 1.0 - k / (release_fire + 1)) if release_fire > 0 else 0.0
            gs[p] = max(0.0, 1.0 - k / (release_smoke + 1)) if release_smoke > 0 else 0.0
    return gf, gs


def first_shown_frame(alignment: Alignment, timing: ElementTiming) -> int:
    return alignment.offset + (timing.visible_onset if alignment.first is None else alignment.first)


def environment_colour(plate_frame: np.ndarray, element: np.ndarray, floor: int, params: CompositeParams) -> np.ndarray:
    """The room's colour around the effect: the plate under the union of the element's cloud, plus a margin
    (one value per shot, so the smoke never changes colour as its crop changes)."""
    h, w = plate_frame.shape[:2]
    sample = element[:: max(1, element.shape[0] // 16)]
    eh, ew = sample.shape[1:3]
    y = (_apply_black_floor(sample, floor).astype(np.float32) / 255.0) @ _LUMA
    mask = (y > params.smoke_mask_level).any(axis=0)
    if not mask.any():
        return plate_frame.reshape(-1, 3).astype(np.float32).mean(0) / 255.0
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    m = 0.1
    y0, y1 = int(max(0, (rows[0] / eh - m) * h)), int(min(h, (rows[-1] / eh + m) * h)) + 1
    x0, x1 = int(max(0, (cols[0] / ew - m) * w)), int(min(w, (cols[-1] / ew + m) * w)) + 1
    return plate_frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32).mean(0) / 255.0


# ------------------------------------------------------------------ per frame

def composite_frame(
    plate8: np.ndarray,
    elem8: np.ndarray,
    params: CompositeParams = CompositeParams(),
    *,
    g_fire: float = 1.0,
    g_smoke: float = 1.0,
    holdout_row: int | None = None,
    emission: tuple[float, float] | None = None,
    fire_energy_ref: float | None = None,
    plate_key: float | None = None,
    grain_sigma: float = 0.0,
    black_floor: int | None = None,
    fire_effect: bool = True,
    dense_smoke: bool = True,
    smoke_ref: float | None = None,
    env_rgb: np.ndarray | None = None,
    seed: int = 0,
    ambient: np.ndarray | None = None,
    shading_ref: float | None = None,
    smoke_albedo: float = 0.85,
    occluder: np.ndarray | None = None,
    lift: bool = False,
    light_matte: np.ndarray | None = None,
    thin_soot: bool = False,
) -> np.ndarray:
    """Composite one element frame (already at plate size) onto one plate frame.

    ``holdout_row``: without an ``occluder``, rows below it get nothing (with a soft edge). ``occluder``: the
    object's matte (plate size, 0..1); it stays in front of the effect below the emission point.
    ``fire_energy_ref`` normalises the light the fire casts (the element's brightest fire frame).
    ``ambient``: the room light (ambient_map); ``env_rgb`` (a flat colour) stands in when it is missing.
    ``lift``: a sustained flame going out (g_fire < 1) lifts off its base instead of dimming in place.
    ``thin_soot``: its thin smoke is soot too, darkening what is behind it (by default thin smoke scatters light on,
    which round a black cloud drew a pale halo on the wall).
    Only the element's bounding box plus a margin is processed for the element itself; the light pool may
    reach further.
    """
    h, w = plate8.shape[:2]
    if black_floor is None:
        black_floor = params.black_floor if params.black_floor is not None else estimate_black_level(elem8[None])
    gf, gs = float(np.clip(g_fire, 0, 1)), float(np.clip(g_smoke, 0, 1))
    k = 4  # find the element's extent on a 1/4 grid: cheap, and the margin covers the rounding
    nz = _peak(elem8[::k, ::k]).astype(np.int16) > black_floor + 2  # ignore speckle when sizing the crop
    if holdout_row is not None:
        nz[min(nz.shape[0], (holdout_row + int(0.02 * h)) // k + 1):] = False
    rows, cols = nz.any(axis=1), nz.any(axis=0)
    if not rows.any() or (gf <= 0 and gs <= 0):
        return plate8.copy()

    margin = int(3 * max(params.bloom_sigma_frac, params.interior_blur_frac) * w) + 4 + k
    y0, y1 = max(0, int(np.argmax(rows)) * k - margin), min(h, (len(rows) - int(np.argmax(rows[::-1]))) * k + margin)
    x0, x1 = max(0, int(np.argmax(cols)) * k - margin), min(w, (len(cols) - int(np.argmax(cols[::-1]))) * k + margin)
    plate_c8, elem_c8 = plate8[y0:y1, x0:x1], elem8[y0:y1, x0:x1]
    p = plate_c8.astype(np.float32) * (1.0 / 255.0)
    t_g, t_l = _floor_tables(black_floor)
    e, e_lin = t_g[elem_c8], t_l[elem_c8]
    p_lin = _LIN8[plate_c8]

    warm, base_fire = _warmth(e, params)
    fire_w = fire_weight_map(e, params, w, fire_effect, base=base_fire)
    # (fire or smoke, read pixel by pixel off a compressed dark-orange render, flipped from one codec block to the
    # next: a fireball's sooty middle came out as grey squares. Smoothed over a couple of blocks' width.)
    fire_w = np.clip(_blur(fire_w, 0.004 * w), 0.0, 1.0)
    if fire_effect and not dense_smoke:  # pure fire (a plume): its dim red periphery is glow, not soot
        fire_w = np.maximum(fire_w, warm)
    smoke_w = 1.0 - fire_w
    flame = np.maximum(fire_w, warm) if fire_effect else fire_w  # everything that goes out with the fire
    y = e @ _LUMA
    yy = np.arange(y0, y1, dtype=np.float32)[:, None]

    # What stays visible: the object in front below the emission point, else a soft horizontal holdout.
    if occluder is not None and emission is not None:
        occ = np.clip(occluder[y0:y1, x0:x1].astype(np.float32), 0.0, 1.0)
        hold = 1.0 - occ * _smoothstep(emission[1] - 0.02 * h, emission[1] + 0.03 * h, yy)
        if holdout_row is not None:  # here: the surface the object stands on; nothing reaches below it
            hold = hold * _smoothstep(holdout_row + 0.03 * h, holdout_row - 0.005 * h, yy)
    elif holdout_row is not None:
        hold = np.broadcast_to(_smoothstep(holdout_row + params.holdout_soft_frac * h, holdout_row - 0.01 * h, yy),
                               y.shape)
    else:
        hold = np.ones_like(y)

    # The fire's gain per pixel: a sustained flame going out loses its base first and rises away.
    kf = np.full_like(y, gf)
    if lift and gf < 1.0 and emission is not None:
        cut = emission[1] - (1.0 - gf) * params.fire_lift_frac * h
        ragged = yy - (_peak(e) - 0.5) * 0.25 * h  # the cut follows the flame's own structure, not a line
        kf = _smoothstep(cut + 0.04 * h, cut - 0.02 * h, ragged) * (0.3 + 0.7 * gf)
    kf = kf * hold

    # fire: an opaque core that never darkens the plate
    a_fire = params.fire_occlusion * _smoothstep(params.fire_alpha_lo, params.fire_alpha_hi, y)
    # only the bright core may darken what is behind it (for saturation); dim edges are pure light, so no
    # dark outline ever traces a flame on a bright wall
    core = _smoothstep(0.45, 0.75, y)
    ratio = e_lin / np.maximum(p_lin, 1e-3)
    lum_cap = (e_lin @ _LUMA) / np.maximum(p_lin @ _LUMA, 1e-3)  # edges: never darker than the wall (in luma)
    cap = lum_cap + core * (_peak(ratio) + params.fire_darken - lum_cap)
    a_fire = np.minimum(a_fire, cap)
    af = fire_w * a_fire * kf
    # A flame's dim skirt, added over a bright wall, turns deep red into salmon; real skirts glow orange.
    # Shift the dimmest fire towards an orange glow of the same luminance.
    glow = np.asarray(params.fringe_glow, np.float32)
    skirt = (1.0 - _smoothstep(0.1, 0.3, y))[..., None] * fire_w[..., None]
    e_fire = e_lin + skirt * ((e_lin @ _LUMA)[..., None] * (glow / float(glow @ _LUMA)) - e_lin)
    emit = _blackbody(e_fire, kf) * (fire_w * params.fire_gain)[..., None]

    # smoke and steam: density alpha, relit by the room
    ref = smoke_ref if smoke_ref else max(float(y.max()), 0.3)
    a_lum = np.clip(y / ref, 0.0, 1.0) ** params.smoke_alpha_gamma
    a_body = a_lum
    if dense_smoke:
        # Deep inside a cloud = dense, however dark it renders (self-shadowed). Depth is a blur of the cloud's
        # soft coverage, so it never follows the render's luma contours (a luma test there drew bands).
        cover = _smoothstep(0.5 * params.smoke_mask_level, 2.0 * params.smoke_mask_level, y)
        depth = _blur(cover, params.interior_blur_frac * w)
        a_body = np.maximum(a_lum, params.smoke_opacity * _smoothstep(params.interior_lo, params.interior_hi, depth) * cover)
    # Smoke going away (its gate releasing, or the element's own end) dissipates: thin parts vanish first and
    # the dense core last, instead of the whole cloud turning uniformly see-through.
    g_smoke_px = flame * kf + (1.0 - flame) * hold * _dissipate(a_body, 1.0 - gs)
    asm = smoke_w * a_body * g_smoke_px
    if dense_smoke:  # a fireball's flame burns inside its own soot: the cloud's body is dense behind the fire too
        body = params.smoke_opacity * _smoothstep(params.interior_lo, params.interior_hi, depth) * cover
        soot = np.clip(body - a_fire, 0.0, 1.0) * fire_w * kf
        af = af + soot
    if ambient is not None:
        amb = ambient[y0:y1, x0:x1]
    else:
        env = p.reshape(-1, 3).mean(0) if env_rgb is None else np.asarray(env_rgb, np.float32)
        amb = np.broadcast_to(_lin(env), e.shape)
    c = e_lin / np.maximum(a_body, 0.05)[..., None]  # the render's light per unit of smoke
    cy_l = c @ _LUMA
    shade = np.clip(cy_l / max(shading_ref or 0.25, 1e-3), params.shading_lo, params.shading_hi) ** params.shading_gamma
    # smoke inside the fire is lit by it: it keeps the render's own fire-lit colour (lit by the room, a fireball's
    # sooty middle went grey); the room's light only shapes smoke outside the flame
    inside = _smoothstep(0.25, 0.6, _blur(fire_w, params.core_reach_frac * w))[..., None] if fire_effect else 0.0
    c = c + params.smoke_desat * (cy_l[..., None] - c) * smoke_w[..., None] * (1.0 - inside)  # fire light keeps its colour
    key = 0.5 - 0.25 * float(_smoothstep(0.2, 0.6, np.asarray(0.5 if plate_key is None else plate_key)))
    # Thin smoke passes on what is behind it (forward scattering) plus some of the render's light, so wisps
    # never darken a bright wall. Dense smoke shows its own albedo: the room's light, shaped by the render's
    # shading, plus some of the render's own light (fire-lit smoke stays fire-lit).
    # (luminous fluids, steam, are translucent throughout: no dense regime)
    dens = _smoothstep(0.25, 0.85, a_body)[..., None] if dense_smoke else np.zeros_like(a_body)[..., None]
    thin_col = amb + key * c
    key_d = key * min(1.0, (smoke_albedo / 0.85) ** 1.5)  # soot shows little of the render's light-grey smoke
    dense_col = (1.0 - key_d) * smoke_albedo * amb * shade[..., None] + key_d * c
    if thin_soot:
        thin_col = dense_col
    smoke_col = thin_col + dens * (dense_col - thin_col)
    smoke_col = smoke_col + inside * (c - smoke_col)
    emit = emit + smoke_col * asm[..., None]
    if dense_smoke:
        emit = emit + (smoke_albedo * (1.0 - key)) * amb * soot[..., None]
    alpha = np.minimum(af + asm, 1.0)[..., None]  # fire_w + smoke_w = 1: complementary fractions add

    # light: a warm pool around the fire, stronger on dark plates and on the object; bloom off the hottest fire
    fire_l = y * fire_w * kf
    light = None
    if params.light_gain > 0 and fire_l.sum() > 0 and fire_energy_ref:
        energy = min(float(fire_l.mean()) * (y1 - y0) * (x1 - x0) / (h * w) / fire_energy_ref, 1.0)
        ys, xs = np.nonzero(fire_l > 0.05)
        if energy > 0 and ys.size:
            wts = fire_l[ys, xs]
            cy, cx = float((ys * wts).sum() / wts.sum()) + y0, float((xs * wts).sum() / wts.sum()) + x0
            if emission is not None:  # the light comes from where the fire starts as much as from its middle
                cx, cy = 0.5 * (cx + emission[0]), 0.5 * (cy + emission[1])
            radius = params.light_radius * max(np.sqrt(ys.size / np.pi), 0.02 * w)
            key_l = 0.5 if plate_key is None else plate_key
            g = 8
            gy, gx = np.mgrid[0:h:g, 0:w:g].astype(np.float32)
            d = np.sqrt(((gx - cx) ** 2 + (gy - cy) ** 2) / radius ** 2)
            small = params.light_gain * (1.0 - 0.7 * key_l) * energy / (1.0 + d * d)
            small = (small * _smoothstep(params.light_reach, 0.5 * params.light_reach, d)).astype(np.float32)
            on = small > 1e-4  # finite pool: only its bounding box is touched
            if on.any():
                r_, c_ = np.nonzero(on.any(1))[0], np.nonzero(on.any(0))[0]
                gy0, gy1, gx0, gx1 = max(0, r_[0] - 1), min(small.shape[0], r_[-1] + 2), max(0, c_[0] - 1), min(small.shape[1], c_[-1] + 2)
                ly0, ly1, lx0, lx1 = gy0 * g, min(h, gy1 * g), gx0 * g, min(w, gx1 * g)
                patch = Image.fromarray(np.ascontiguousarray(small[gy0:gy1, gx0:gx1]), "F")
                L = np.asarray(patch.resize((lx1 - lx0, ly1 - ly0), Image.Resampling.BILINEAR), dtype=np.float32)
                if light_matte is not None and params.object_light_boost > 0:
                    L = L * (1.0 + params.object_light_boost * light_matte[ly0:ly1, lx0:lx1])
                light = (ly0, ly1, lx0, lx1, L)  # linear-light gain: plate_lin * (1 + 2.2 * L * tint)
    bloom = params.bloom_gain * _blur(np.clip(fire_l - params.bloom_threshold, 0, None), params.bloom_sigma_frac * w)

    # Work only inside the region anything touches: the element's crop plus the light pool. Outside the crop
    # the light is a gain applied to display values (the same as the linear gain, without converting the
    # whole frame); inside it everything is blended in linear light.
    ry0, ry1, rx0, rx1 = y0, y1, x0, x1
    if light is not None:
        ry0, ry1, rx0, rx1 = min(ry0, light[0]), max(ry1, light[1]), min(rx0, light[2]), max(rx1, light[3])
    plate_r = plate8[ry0:ry1, rx0:rx1]
    base = plate_r.astype(np.float32) * (1.0 / 255.0)
    out = base.copy()
    lit = np.zeros(out.shape[:2], bool)
    tint = np.asarray(params.light_tint, np.float32)
    cy_ = slice(y0 - ry0, y1 - ry0)
    cx_ = slice(x0 - rx0, x1 - rx0)
    gain_lin = None  # the light's linear gain over the crop
    if light is not None:
        ly0, ly1, lx0, lx1, L = light
        sl = (slice(ly0 - ry0, ly1 - ry0), slice(lx0 - rx0, lx1 - rx0))
        lit[sl] = L > 1e-4
        out[sl] *= _light_table(tuple(params.light_tint))[np.clip(L * _LIGHT_STEPS, 0, _LIGHT_N - 1).astype(np.int32)]
        lmap = np.zeros(out.shape[:2], np.float32)
        lmap[sl] = L
        gain_lin = 1.0 + 2.2 * lmap[cy_, cx_][..., None] * tint
    # the fire's light on the plate, but none of it leaking through an opaque element
    region_lin = p_lin if gain_lin is None else p_lin * (1.0 + (gain_lin - 1.0) * (1.0 - alpha))
    region = _gam_fast(region_lin * (1.0 - alpha) + emit)
    region = 1.0 - (1.0 - region) * (1.0 - bloom[..., None] * tint)
    emit_peak = _peak(emit)
    if params.grain and grain_sigma > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, grain_sigma / 255.0, region.shape[:2] + (1,)).astype(np.float32)
        region += noise * np.clip(alpha + (emit_peak[..., None] > 0.002), 0, 1)
    rng_d = np.random.default_rng(seed + 7919)  # one-level triangular dither: smooth ramps stay smooth in 8 bits
    region += ((rng_d.random(region.shape[:2], np.float32) - rng_d.random(region.shape[:2], np.float32))
               * (1.0 / 255.0))[..., None]
    out[cy_, cx_] = region
    if params.shoulder < 1.0:  # roll off what was ADDED, keeping hue: compress the max channel, scale all three
        limit = np.maximum(_peak(base), params.shoulder)
        mx = _peak(out)
        over = mx > limit
        if over.any():
            room = np.maximum(1.0 - limit[over], 1e-4)
            new = limit[over] + room * (1.0 - np.exp(-(mx[over] - limit[over]) / room))
            out[over] *= (new / mx[over])[:, None]
    region8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    untouched = ~lit
    untouched[cy_, cx_] &= (alpha[..., 0] <= 0) & (emit_peak <= 0) & (bloom <= 0)
    region8[untouched] = plate_r[untouched]  # bit-exact where nothing contributes
    out8 = plate8.copy()
    out8[ry0:ry1, rx0:rx1] = region8
    return out8


# ------------------------------------------------------------------ whole shot

def object_base_row(occluder: np.ndarray, emission: tuple[float, float]) -> int | None:
    """The row where the object meets what it stands on: the bottom of its matte under the emission point."""
    h, w = occluder.shape
    cx = int(min(max(emission[0], 0), w - 1))
    band = occluder[:, max(0, cx - w // 20): cx + w // 20 + 1].max(axis=1) > 0.5
    rows = np.nonzero(band[int(min(max(emission[1], 0), h - 1)):])[0]
    return None if rows.size == 0 else int(emission[1]) + int(rows[-1])


def effect_options(effect) -> dict:
    """The composite() options an effect implies (an EffectSpec from cuesync.effects)."""
    return {"holdout_below": effect.holdout_below, "fire_effect": effect.fluid == "fire",
            "dense_smoke": effect.fluid == "smoke" or effect.secondary == "smoke",
            "smoke_albedo": effect.smoke_albedo, "lift": effect.fluid == "fire" and effect.voice_gate == "both",
            # a fireball's pale, half-saturated interior is fire-lit smoke (relit as soot), not flame
            "fire_sat": (0.4, 0.65) if effect.secondary == "smoke" else None}


def composite(
    plate: np.ndarray,
    element: np.ndarray,
    alignment: Alignment,
    timing: ElementTiming,
    params: CompositeParams = CompositeParams(),
    *,
    gates: tuple[np.ndarray, np.ndarray] | None = None,
    emission: tuple[float, float] | None = None,
    holdout_below: float | None = None,
    fire_effect: bool = True,
    dense_smoke: bool = True,
    smoke_albedo: float = 0.85,
    occluder: np.ndarray | None = None,
    lift: bool = False,
    fire_sat: tuple[float, float] | None = None,
) -> np.ndarray:
    """Composite ``element`` onto every plate frame it covers; returns frames with the plate's shape.

    ``gates``: per plate frame (fire, smoke) gains from voice_gates (default: always 1).
    ``emission``: emission point in plate pixels. ``occluder``: the object's matte at plate size (it stays
    in front below the emission point); without one, ``holdout_below`` (fraction of the plate height) cuts the
    element that far below the emission point. ``smoke_albedo``: 0.3 soot .. 0.95 steam. ``lift``: sustained
    fire lifts off its base as the fire gate closes.
    """
    n, h, w, _ = plate.shape
    out = plate.copy()
    elen = element.shape[0]
    if fire_sat is not None:
        params = dataclasses.replace(params, sat_lo=fire_sat[0], sat_hi=fire_sat[1])
    floor = params.black_floor if params.black_floor is not None else estimate_black_level(element)
    fire_ref = 0.0
    for f in element[:: max(1, elen // 16)]:  # the brightest fire frame normalises the light the fire casts
        e = _apply_black_floor(f[::4, ::4], floor).astype(np.float32) / 255.0
        fire_ref = max(fire_ref, float(((e @ _LUMA) * fire_weight_map(e, params, f.shape[1] // 4, fire_effect)).mean()))
    plate_key = float((plate[0, ::8, ::8].astype(np.float32) / 255.0 @ _LUMA).mean())
    smoke_ref = smoke_reference_luma(element, params, floor, fire_effect)
    env_rgb = environment_colour(plate[0], element, floor, params)
    amb = ambient_map(plate[0], params)
    light_matte = None
    if occluder is not None and params.object_light_boost > 0:
        light_matte = np.clip(_blur(occluder.astype(np.float32), params.object_light_soft_frac * w), 0.0, 1.0)
    shading_ref = smoke_shading_ref(element, params, floor, fire_effect, smoke_ref)
    grain = plate_grain_sigma(plate) if params.grain else 0.0
    holdout_row = None
    if emission is not None and occluder is not None:
        holdout_row = object_base_row(occluder, emission)
    elif emission is not None and holdout_below is not None:
        holdout_row = int(round(emission[1] + holdout_below * h))
    gf_all = gates[0] if gates is not None else np.ones(n)
    gs_all = gates[1] if gates is not None else np.ones(n)
    window = min(params.end_fade_frames, elen // 4)
    smoke_window = min(params.smoke_end_frames, elen // 3)

    def one(f: int) -> None:
        idx = element_index_for(alignment, timing, f, elen)
        if idx is None:
            return
        src = element[idx]
        if src.shape[:2] != (h, w):
            src = np.asarray(Image.fromarray(src).resize((w, h), Image.Resampling.LANCZOS))
        fade = min(1.0, (elen - 1 - idx) / window) if window > 0 else 1.0
        smoke_fade = min(1.0, (elen - 1 - idx) / smoke_window) if smoke_window > 0 else 1.0
        g_fire = float(gf_all[f]) * fade if f < len(gf_all) else 0.0
        g_smoke = float(gs_all[f]) * smoke_fade if f < len(gs_all) else 0.0
        out[f] = composite_frame(plate[f], src, params, g_fire=g_fire, g_smoke=g_smoke, holdout_row=holdout_row,
                                 emission=emission, fire_energy_ref=fire_ref or None, plate_key=plate_key,
                                 grain_sigma=grain, black_floor=floor, fire_effect=fire_effect, dense_smoke=dense_smoke,
                                 smoke_ref=smoke_ref, env_rgb=env_rgb, seed=f, ambient=amb, shading_ref=shading_ref,
                                 smoke_albedo=smoke_albedo, occluder=occluder, lift=lift, light_matte=light_matte)

    # Frames are independent and numpy releases the GIL in the heavy loops, so threads share the work.
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=max(1, min(16, cpus - 1))) as pool:
        list(pool.map(one, range(n)))
    return out
