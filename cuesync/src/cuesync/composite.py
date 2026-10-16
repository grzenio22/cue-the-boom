"""Composite an over-black fluid element onto untouched footage.

An element rendered over black is a premultiplied image without an alpha channel, so the alpha is built
per material:
- Fire (warm, saturated, bright pixels, plus white-hot cores grown from them) partly occludes the plate in
  proportion to its own peak channel and adds its light: out = plate*(1-a) + E. Pure screen can never
  darken a bright plate, so fire over daylight went pale; a partly opaque body keeps it orange.
- Smoke and steam: a luminous fluid's alpha is its luma relative to its densest luma, so thin edges stay
  thin. Dense smoke is also self-shadowed, so its interior (by a soft, render-following measure) is made
  opaque even where it renders dark; otherwise the core would be the most see-through part. Smoke leans
  toward the room's colour, and the room's light lifts its self-shadowed parts (a render on black has none).
Fire also lights the plate (a soft multiplicative pool with a finite reach) and blooms.

Voice gates scale what belongs to the flame (any warm pixel, however dim) with the fire gate and the rest
with the smoke gate, so a flame goes out as a whole instead of leaving a ghost. A holdout cuts everything
below the emission point, which also hides any nozzle or ground the model invents. Plate-matched grain
goes on wherever the element contributes. Pixels nothing touches are copied from the plate exactly.
"""

from __future__ import annotations

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
    # fire: occlusion in proportion to the flame's own peak channel, emission everywhere
    fire_occlusion: float = 0.6
    fire_alpha_lo: float = 0.3
    fire_alpha_hi: float = 0.8
    fire_gain: float = 1.0
    # smoke and steam
    smoke_opacity: float = 0.92
    smoke_mask_level: float = 0.06  # luma where a pixel starts to belong to the cloud
    interior_blur_frac: float = 0.04
    interior_lo: float = 0.5
    interior_hi: float = 0.95
    smoke_tint: float = 0.35  # how far smoke colour leans toward the room's (luminance kept)
    smoke_ambient: float = 0.35  # the room's light on dense smoke's self-shadowed parts
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
    end_fade_frames: int = 12  # safety fade at the element's own end


# ------------------------------------------------------------------ helpers

def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hue_sat(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = e[..., 0], e[..., 1], e[..., 2]
    mx, mn = e.max(-1), e.min(-1)
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


def _warmth(e: np.ndarray, params: CompositeParams) -> tuple[np.ndarray, np.ndarray]:
    """(warm and saturated, regardless of brightness; the same and bright enough to be flame)."""
    hue, sat = _hue_sat(e)
    warm = _smoothstep(params.sat_lo, params.sat_hi, sat) * ((hue <= params.warm_hue_max_deg) | (hue >= 345.0))
    return warm, warm * _smoothstep(params.fire_bright_lo, params.fire_bright_hi, e.max(-1))


def fire_weight_map(e: np.ndarray, params: CompositeParams, full_width: int, fire_effect: bool = True) -> np.ndarray:
    """Per-pixel fire weight: warm saturated bright pixels, plus white-hot cores grown from them."""
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
) -> np.ndarray:
    """Composite one element frame (already at plate size) onto one plate frame.

    ``holdout_row``: rows below it get nothing (with a soft edge). ``fire_energy_ref`` normalises the light
    the fire casts (the element's brightest fire frame). ``env_rgb``: the room's colour (one per shot).
    Only the element's bounding box plus a margin is processed for the element itself; the light pool may
    reach further.
    """
    h, w = plate8.shape[:2]
    if black_floor is None:
        black_floor = params.black_floor if params.black_floor is not None else estimate_black_level(elem8[None])
    elem8 = _apply_black_floor(elem8, black_floor)
    gf, gs = float(np.clip(g_fire, 0, 1)), float(np.clip(g_smoke, 0, 1))
    nz = elem8.max(axis=-1) > 2  # ignore single-level speckle when sizing the crop
    if holdout_row is not None:
        nz[min(h, holdout_row + int(0.02 * h)):] = False
    rows, cols = nz.any(axis=1), nz.any(axis=0)
    if not rows.any() or (gf <= 0 and gs <= 0):
        return plate8.copy()

    margin = int(3 * max(params.bloom_sigma_frac, params.interior_blur_frac) * w) + 4
    y0, y1 = max(0, int(np.argmax(rows)) - margin), min(h, h - int(np.argmax(rows[::-1])) + margin)
    x0, x1 = max(0, int(np.argmax(cols)) - margin), min(w, w - int(np.argmax(cols[::-1])) + margin)
    p = plate8[y0:y1, x0:x1].astype(np.float32) / 255.0
    e = elem8[y0:y1, x0:x1].astype(np.float32) / 255.0

    warm, _ = _warmth(e, params)
    fire_w = fire_weight_map(e, params, w, fire_effect)
    smoke_w = 1.0 - fire_w
    flame = np.maximum(fire_w, warm) if fire_effect else fire_w  # everything that goes out with the fire
    g_smoke_px = flame * gf + (1.0 - flame) * gs
    y = e @ _LUMA
    if holdout_row is not None:
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        hold = _smoothstep(holdout_row + 0.04 * h, holdout_row - 0.01 * h, yy)  # 1 above, 0 below the cut
    else:
        hold = np.ones((y1 - y0, 1), np.float32)

    peak = e.max(-1)
    a_fire = params.fire_occlusion * _smoothstep(params.fire_alpha_lo, params.fire_alpha_hi, peak)
    ref = smoke_ref if smoke_ref else max(float(y.max()), 0.3)
    a_lum = np.clip(y / ref, 0.0, 1.0)
    interior = np.zeros_like(y)
    if dense_smoke:
        # Deep inside a cloud AND darker than the cloud's own rim = self-shadowed, so dense, however dark it
        # renders. A centre-lit cloud is brighter inside than at its rim and keeps its luminous alpha.
        cloud = (y > params.smoke_mask_level).astype(np.float32)
        depth = _blur(cloud, params.interior_blur_frac * w)
        rim = (depth > 0.25) & (depth < 0.6) & (cloud > 0)
        rim_luma = float(np.median(y[rim])) if rim.any() else ref
        shadowed = _smoothstep(0.0, 0.15, rim_luma - y)
        interior = _smoothstep(params.interior_lo, params.interior_hi, depth) * shadowed
    a_body = np.maximum(a_lum, params.smoke_opacity * interior)
    af, asm = fire_w * a_fire * gf * hold, smoke_w * a_body * g_smoke_px * hold
    alpha = np.maximum(af, asm)[..., None]
    env = p.reshape(-1, 3).mean(0) if env_rgb is None else np.asarray(env_rgb, np.float32)
    tint = (1.0 - params.smoke_tint) + params.smoke_tint * env / max(float(env @ _LUMA), 1e-3)
    emit = e * (fire_w * gf * params.fire_gain * hold)[..., None] + e * tint * (smoke_w * g_smoke_px * hold)[..., None]
    if dense_smoke:  # the room lights the dense, self-shadowed parts only, so thin rims are never lifted
        shadow = np.clip(1.0 - y / ref, 0.0, 1.0)
        dense = params.smoke_opacity * interior * shadow * smoke_w * g_smoke_px * hold
        emit = emit + (params.smoke_ambient * env)[None, None, :] * dense[..., None]

    # light: a warm pool around the fire, stronger on dark plates; plus bloom off the hottest fire
    fire_l = y * fire_w * gf * hold
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
            key = 0.5 if plate_key is None else plate_key
            gy, gx = np.mgrid[0:h:8, 0:w:8].astype(np.float32)
            d = np.sqrt(((gx - cx) ** 2 + (gy - cy) ** 2) / radius ** 2)
            small = params.light_gain * (1.0 - 0.7 * key) * energy / (1.0 + d * d)
            small *= _smoothstep(params.light_reach, 0.5 * params.light_reach, d)  # finite: far pixels untouched
            if small.max() > 0:
                light = np.asarray(Image.fromarray(small.astype(np.float32), "F").resize((w, h), Image.Resampling.BILINEAR),
                                   dtype=np.float32)
    bloom = params.bloom_gain * _blur(np.clip(fire_l - params.bloom_threshold, 0, None), params.bloom_sigma_frac * w)

    out = plate8.astype(np.float32) / 255.0
    lit = light > 1e-4 if light is not None else np.zeros((h, w), bool)
    if lit.any():
        out[lit] *= 1.0 + light[lit][:, None] * np.asarray(params.light_tint, np.float32)
    region = out[y0:y1, x0:x1] * (1.0 - alpha) + emit
    region = 1.0 - (1.0 - region) * (1.0 - bloom[..., None] * np.asarray(params.light_tint, np.float32))
    if params.grain and grain_sigma > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, grain_sigma / 255.0, region.shape[:2] + (1,)).astype(np.float32)
        region += noise * np.clip(alpha + (emit.max(-1, keepdims=True) > 0.02), 0, 1)
    out[y0:y1, x0:x1] = region
    if params.shoulder < 1.0:  # roll off what was ADDED, keeping hue: compress the max channel, scale all three
        base = plate8.astype(np.float32) / 255.0
        limit = np.maximum(base.max(-1), params.shoulder)
        mx = out.max(-1)
        over = mx > limit
        if over.any():
            room = np.maximum(1.0 - limit[over], 1e-4)
            new = limit[over] + room * (1.0 - np.exp(-(mx[over] - limit[over]) / room))
            out[over] *= (new / mx[over])[:, None]
    out8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    same = np.abs(out8.astype(np.int16) - plate8.astype(np.int16)).max(-1) <= 0
    untouched = ~lit
    untouched[y0:y1, x0:x1] &= (alpha[..., 0] <= 0) & (emit.max(-1) <= 0) & (bloom <= 0)
    out8[untouched | same] = plate8[untouched | same]  # bit-exact where nothing contributes
    return out8


# ------------------------------------------------------------------ whole shot

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
) -> np.ndarray:
    """Composite ``element`` onto every plate frame it covers; returns frames with the plate's shape.

    ``gates``: per plate frame (fire, smoke) gains from voice_gates (default: always 1).
    ``emission``: emission point in plate pixels; with ``holdout_below`` (fraction of the plate height) the
    element is cut below it.
    """
    n, h, w, _ = plate.shape
    out = plate.copy()
    elen = element.shape[0]
    floor = params.black_floor if params.black_floor is not None else estimate_black_level(element)
    fire_ref = 0.0
    for f in element[:: max(1, elen // 16)]:  # the brightest fire frame normalises the light the fire casts
        e = _apply_black_floor(f[::4, ::4], floor).astype(np.float32) / 255.0
        fire_ref = max(fire_ref, float(((e @ _LUMA) * fire_weight_map(e, params, f.shape[1] // 4, fire_effect)).mean()))
    plate_key = float((plate[0, ::8, ::8].astype(np.float32) / 255.0 @ _LUMA).mean())
    smoke_ref = smoke_reference_luma(element, params, floor, fire_effect)
    env_rgb = environment_colour(plate[0], element, floor, params)
    grain = plate_grain_sigma(plate) if params.grain else 0.0
    holdout_row = None
    if emission is not None and holdout_below is not None:
        holdout_row = int(round(emission[1] + holdout_below * h))
    gf_all = gates[0] if gates is not None else np.ones(n)
    gs_all = gates[1] if gates is not None else np.ones(n)
    window = min(params.end_fade_frames, elen // 4)

    last_idx, last_src = None, None
    for f in range(n):
        idx = element_index_for(alignment, timing, f, elen)
        if idx is None:
            continue
        if idx != last_idx:
            src = element[idx]
            if src.shape[:2] != (h, w):
                src = np.asarray(Image.fromarray(src).resize((w, h), Image.Resampling.LANCZOS))
            last_idx, last_src = idx, src
        fade = min(1.0, (elen - 1 - idx) / window) if window > 0 else 1.0
        g_fire = float(gf_all[f]) * fade if f < len(gf_all) else 0.0
        g_smoke = float(gs_all[f]) * fade if f < len(gs_all) else 0.0
        out[f] = composite_frame(plate[f], last_src, params, g_fire=g_fire, g_smoke=g_smoke, holdout_row=holdout_row,
                                 emission=emission, fire_energy_ref=fire_ref or None, plate_key=plate_key,
                                 grain_sigma=grain, black_floor=floor, fire_effect=fire_effect, dense_smoke=dense_smoke,
                                 smoke_ref=smoke_ref, env_rgb=env_rgb, seed=f)
    return out
