"""Composite an over-black fluid element onto untouched footage.

Black is not an alpha channel, so the matte is split by colour: warm saturated pixels (fire) are
screened on; grey/white pixels (smoke, steam) get a luminance alpha and a premultiplied "over" with
slight darkening so thick smoke covers bright backgrounds. A blurred glow from the fire lights the plate.
Pixels the element doesn't touch are copied from the plate bit-exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .align import Alignment, element_index_for
from .score import ElementTiming

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class CompositeParams:
    fire_gain: float = 1.0
    smoke_gamma: float = 1.4  # kept for compatibility; smoke alpha now uses the element's own reference luma
    smoke_darken: float = 0.15
    glow_gain: float = 0.35
    glow_sigma_frac: float = 0.08
    glow_tint: tuple[float, float, float] = (1.0, 0.6, 0.25)
    sat_lo: float = 0.25
    sat_hi: float = 0.5
    warm_hue_max_deg: float = 70.0
    fire_bright_lo: float = 0.35  # dark warm pixels (fire-lit smoke) take the smoke path, not the screen path
    fire_bright_hi: float = 0.6
    black_floor: int = 4  # 8-bit levels of codec noise / model haze treated as true black
    smoke_ref_min: float = 0.62  # lower bound for the smoke reference luma
    end_fade_frames: int = 12  # fade the element's last frames so smoke doesn't pop off


def _apply_black_floor(elem8: np.ndarray, floor: int) -> np.ndarray:
    if floor <= 0:
        return elem8
    e = elem8.astype(np.int16) - floor
    np.clip(e, 0, 255 - floor, out=e)
    return ((e.astype(np.float32) * (255.0 / (255 - floor))) + 0.5).astype(np.uint8)


def smoke_reference_luma(element: np.ndarray, params: CompositeParams = CompositeParams()) -> float:
    """Maximum smoke luma of the element (lightly subsampled): the luma of 'fully dense' smoke.

    Any reference below the densest smoke makes the core opaque too early and leaves a dark ring on bright
    plates (out is then non-monotonic), so the max is used; fire pixels are excluded, so fire sparkle can't
    inflate it. One value per element keeps the smoke alpha stable from frame to frame (no pumping)."""
    sample = element[:: max(1, element.shape[0] // 24), ::2, ::2]
    e = _apply_black_floor(sample, params.black_floor).astype(np.float32) / 255.0
    hue, sat = _hue_sat(e)
    warm = (hue <= params.warm_hue_max_deg) | (hue >= 345.0)
    fire_w = _smoothstep(params.sat_lo, params.sat_hi, sat) * warm * _smoothstep(params.fire_bright_lo, params.fire_bright_hi, e.max(-1))
    y = e @ _LUMA
    smoke = y[(fire_w < 0.5) & (y > 0.02)]
    ref = float(smoke.max()) if smoke.size else params.smoke_ref_min
    return max(params.smoke_ref_min, min(ref, 1.0))


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


def _glow_map(fire_luma: np.ndarray, sigma_frac: float, full_width: int) -> np.ndarray:
    """Gaussian-blurred fire luminance at 1/4 resolution, upsampled; float in [0,1], exactly 0 far away.

    The blur radius scales with the *full* frame width so cropped regions glow like the whole frame.
    """
    h, w = fire_luma.shape
    small = Image.fromarray(np.clip(fire_luma * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")
    small = small.resize((max(1, w // 4), max(1, h // 4)), Image.Resampling.BOX)
    radius = sigma_frac * full_width / 4.0
    blurred = small.filter(ImageFilter.GaussianBlur(radius=radius)).resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(blurred, dtype=np.float32) / 255.0


def composite_frame(
    plate8: np.ndarray,
    elem8: np.ndarray,
    params: CompositeParams = CompositeParams(),
    smoke_ref: float | None = None,
    gain: float = 1.0,
) -> np.ndarray:
    """Composite one element frame (already at plate size) onto one plate frame.

    ``gain`` scales the (premultiplied) element, e.g. for the end fade. Only the element's bounding box plus a
    4-sigma glow margin is processed; the rest is copied verbatim.
    """
    elem8 = _apply_black_floor(elem8, params.black_floor)
    if gain < 1.0:
        elem8 = (elem8.astype(np.float32) * max(gain, 0.0) + 0.5).astype(np.uint8)
    nz = elem8.any(axis=-1)
    rows, cols = nz.any(axis=1), nz.any(axis=0)
    if not rows.any():
        return plate8.copy()
    if smoke_ref is None:
        smoke_ref = smoke_reference_luma(elem8[None], params)
    h, w = nz.shape
    margin = int(4 * params.glow_sigma_frac * w) + 4
    y0, y1 = max(0, int(np.argmax(rows)) - margin), min(h, h - int(np.argmax(rows[::-1])) + margin)
    x0, x1 = max(0, int(np.argmax(cols)) - margin), min(w, w - int(np.argmax(cols[::-1])) + margin)
    out = plate8.copy()
    out[y0:y1, x0:x1] = _composite_region(plate8[y0:y1, x0:x1], elem8[y0:y1, x0:x1], params, full_width=w, smoke_ref=smoke_ref)
    return out


def _composite_region(plate8: np.ndarray, elem8: np.ndarray, params: CompositeParams, full_width: int, smoke_ref: float) -> np.ndarray:
    p = plate8.astype(np.float32) / 255.0
    e = elem8.astype(np.float32) / 255.0
    hue, sat = _hue_sat(e)
    warm = (hue <= params.warm_hue_max_deg) | (hue >= 345.0)
    bright = _smoothstep(params.fire_bright_lo, params.fire_bright_hi, e.max(-1))
    fire_w = _smoothstep(params.sat_lo, params.sat_hi, sat) * warm * bright
    smoke_w = 1.0 - fire_w
    y = e @ _LUMA
    # Physically consistent "over" for an element rendered over black (= premultiplied): alpha is the
    # pixel's luma relative to fully dense smoke, so out moves monotonically from the plate to the smoke
    # colour as density rises. No dark ring on bright plates (the old min() saturated before the core).
    a = (np.clip(y / smoke_ref, 0.0, 1.0) * smoke_w)[..., None]
    p1 = p * (1.0 - a) * (1.0 - params.smoke_darken * a) + e * smoke_w[..., None]
    fire = np.clip(e * fire_w[..., None] * params.fire_gain, 0.0, 1.0)
    p2 = 1.0 - (1.0 - p1) * (1.0 - fire)
    g = _glow_map((e * fire_w[..., None]) @ _LUMA, params.glow_sigma_frac, full_width)
    glow = params.glow_gain * g[..., None] * np.asarray(params.glow_tint, dtype=np.float32)
    out = 1.0 - (1.0 - np.clip(p2, 0.0, 1.0)) * (1.0 - glow)
    out8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    untouched = (elem8.max(-1) == 0) & (g == 0)
    out8[untouched] = plate8[untouched]  # bit-exact where the element contributes nothing
    return out8


def composite(
    plate: np.ndarray,
    element: np.ndarray,
    alignment: Alignment,
    timing: ElementTiming,
    params: CompositeParams = CompositeParams(),
) -> np.ndarray:
    """Composite ``element`` onto every plate frame it covers; returns frames with the plate's shape."""
    n, h, w, _ = plate.shape
    out = plate.copy()
    elen = element.shape[0]
    smoke_ref = smoke_reference_luma(element, params)
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
        window = min(params.end_fade_frames, elen // 4)
        fade = min(1.0, (elen - 1 - idx) / window) if window > 0 else 1.0
        out[f] = composite_frame(plate[f], last_src, params, smoke_ref=smoke_ref, gain=fade)
    return out
