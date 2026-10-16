"""Measure when a rendered element actually becomes visible, when it hits and when it peaks.

Placement is not appearance: the guide says where fluid should be, the model decides when it reads.
For fire the energy is fire-weighted (warm, saturated, bright pixels and their white-hot cores), so the
late white smoke of a fireball can't pose as its peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class ElementTiming:
    visible_onset: int
    peak: int
    curve: np.ndarray
    hit: int | None = None  # steepest rise before the peak: when the effect reads as "happening"
    extra: dict = field(default_factory=dict)

    @property
    def hit_frame(self) -> int:
        return self.visible_onset if self.hit is None else self.hit


def _median3(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x.copy()
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def fire_weight(e: np.ndarray, sat_lo: float = 0.25, sat_hi: float = 0.5, warm_hue_max_deg: float = 70.0,
                bright_lo: float = 0.35, bright_hi: float = 0.6) -> np.ndarray:
    """Per-pixel 'this is fire' weight in [0, 1] for float RGB in [0, 1]: warm, saturated and bright."""
    from .composite import _hue_sat, _smoothstep  # one definition of fire, shared with the compositor

    hue, sat = _hue_sat(e)
    warm = (hue <= warm_hue_max_deg) | (hue >= 345.0)
    return _smoothstep(sat_lo, sat_hi, sat) * warm * _smoothstep(bright_lo, bright_hi, e.max(-1))


def measure_element(frames: np.ndarray, black_level: float = 0.06, kind: str = "any") -> ElementTiming:
    """Effect energy per frame (3-frame median): mean luma of pixels above ``black_level``, fire-weighted
    when ``kind == "fire"``.

    peak = argmax; visible_onset = first frame reaching 10% of the max *before* the peak; hit = the
    steepest rise between the onset and the peak.
    """
    curve = np.empty(frames.shape[0], dtype=np.float64)
    for i, f in enumerate(frames):  # per frame keeps memory bounded for 121 x 1024 x 576 elements
        e = f[::2, ::2].astype(np.float32) / 255.0
        y = e @ _LUMA
        w = (y > black_level).astype(np.float32)
        if kind == "fire":
            w *= fire_weight(e)
        curve[i] = float((y * w).mean())
    curve = _median3(curve)
    top = curve.max()
    if top <= 0:
        raise ValueError("element is black: nothing visible to measure")
    peak = int(np.argmax(curve))
    onset = int(np.argmax(curve[: peak + 1] >= 0.1 * top))
    rise = np.diff(curve[onset: peak + 1]) if peak > onset else np.zeros(0)
    hit = onset + 1 + int(np.argmax(rise)) if rise.size and rise.max() > 0 else onset
    return ElementTiming(visible_onset=onset, peak=peak, curve=curve, hit=hit)


def measure_composite_onset(
    plate: np.ndarray, composite: np.ndarray, pixel_delta: int = 8, min_fraction: float = 0.002
) -> int | None:
    """First frame where more than ``min_fraction`` of pixels differ from the plate by > ``pixel_delta`` (max channel)."""
    n = min(len(plate), len(composite))
    for i in range(n):
        diff = np.abs(composite[i].astype(np.int16) - plate[i].astype(np.int16)).max(axis=-1)
        if (diff > pixel_delta).mean() > min_fraction:
            return i
    return None


def measure_composite_end(
    plate: np.ndarray, composite: np.ndarray, pixel_delta: int = 8, min_fraction: float = 0.002
) -> int | None:
    """First frame after which the composite no longer differs visibly from the plate; len(plate) when it is
    still changed on the last frame (the stop is then at least that late); None when nothing ever changed.
    The 'it stops when I do' measurement."""
    n = min(len(plate), len(composite))
    last = None
    for i in range(n):
        diff = np.abs(composite[i].astype(np.int16) - plate[i].astype(np.int16)).max(axis=-1)
        if (diff > pixel_delta).mean() > min_fraction:
            last = i
    return None if last is None else last + 1
