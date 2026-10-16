"""Measure when a rendered element actually becomes visible and when it peaks (placement != appearance)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class ElementTiming:
    visible_onset: int
    peak: int
    curve: np.ndarray


def _median3(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x.copy()
    p = np.pad(x, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def measure_element(frames: np.ndarray, black_level: float = 0.06) -> ElementTiming:
    """Effect energy per frame = mean of luma over pixels brighter than ``black_level`` (3-frame median).

    visible_onset = first frame reaching 10% of the max; peak = argmax.
    """
    curve = np.empty(frames.shape[0], dtype=np.float64)
    for i, f in enumerate(frames):  # per frame keeps memory bounded for 121 x 1024 x 576 elements
        y = (f.astype(np.float32) / 255.0) @ _LUMA
        curve[i] = float((y * (y > black_level)).mean())
    curve = _median3(curve)
    top = curve.max()
    if top <= 0:
        raise ValueError("element is black: nothing visible to measure")
    onset = int(np.argmax(curve >= 0.1 * top))
    return ElementTiming(visible_onset=onset, peak=int(np.argmax(curve)), curve=curve)


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
