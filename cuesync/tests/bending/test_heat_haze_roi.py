"""Heat-haze optimization must preserve the original full-frame calculation."""

import numpy as np
import pytest

from cuesync import fx_light as fx


def reference_heat_haze(frames, alpha, seed=0, frame0=0, foreground=None):
    n, h, w = frames.shape[:3]
    grids = fx._haze_grids(frame0 + n, h, w, seed)
    result = frames.copy()
    for f in range(n):
        shift = fx.haze_field(fx._alpha(alpha[f]), frame0 + f, grids, min(h, w))
        if shift is not None and foreground is not None:
            shift *= (1 - np.clip(fx._alpha(foreground[f]), 0, 1))[..., None]
        box = None if shift is None else fx._box(np.abs(shift).max(-1) > 1e-3, 4)
        if box is not None:
            y0, y1, x0, x1 = box
            result[f, y0:y1, x0:x1] = fx._remap_box(frames[f], shift[y0:y1, x0:x1], box)
    return result


@pytest.mark.parametrize("shape", [(144, 256), (257, 143)])
@pytest.mark.parametrize("foreground_kind", ["none", "float", "uint8", "opaque"])
@pytest.mark.parametrize("inplace", [False, True])
def test_heat_haze_matches_full_frame_reference(shape, foreground_kind, inplace):
    h, w = shape
    rng = np.random.default_rng(15)
    frames = rng.random((4, h, w, 3), dtype=np.float32)
    alpha = np.zeros((4, h, w), np.float32)
    alpha[1, h // 2:3 * h // 4, w // 2:3 * w // 4] = .8
    alpha[2, :h // 3, :w // 3] = 1  # support touches top/left borders
    alpha[3, 2 * h // 3:, 2 * w // 3:] = 1  # bottom/right borders
    foreground = None
    if foreground_kind != "none":
        foreground = rng.uniform(-.2, 1.2, alpha.shape).astype(np.float32)
        if foreground_kind == "uint8":
            foreground = (np.clip(foreground, 0, 1) * 255).astype(np.uint8)
            alpha = (alpha * 255).astype(np.uint8)
        elif foreground_kind == "opaque":
            foreground[:] = 1
    expected = reference_heat_haze(frames, alpha, 7, 19, foreground)
    result = fx.heat_haze(frames, alpha, 7, frames if inplace else None, 19, foreground)
    np.testing.assert_array_equal(result, expected)


def test_foreground_conversion_is_limited_to_active_heat_region(monkeypatch):
    h, w = 540, 960
    frames = np.ones((1, h, w, 3), np.float32)
    alpha = np.zeros((1, h, w), np.float32)
    alpha[0, 300:330, 450:480] = 1
    foreground = np.zeros_like(alpha, dtype=np.uint8)
    converted = []
    original = fx._alpha

    def record(a):
        if a.dtype == np.uint8:
            converted.append(a.size)
        return original(a)

    monkeypatch.setattr(fx, "_alpha", record)
    fx.heat_haze(frames, alpha, foreground=foreground)
    assert converted and max(converted) < h * w // 2


def test_heat_haze_keeps_exact_support_threshold_and_padding(monkeypatch):
    rng = np.random.default_rng(4)
    frames = rng.random((1, 40, 60, 3), dtype=np.float32)
    alpha = np.ones((1, 40, 60), np.float32)
    shift = np.zeros((40, 60, 2), np.float32)
    epsilon = np.float32(1e-3)
    shift[5, 5] = epsilon  # excluded by the strictly-greater threshold
    shift[30, 50] = -epsilon
    shift[20, 30, 0] = np.nextafter(epsilon, np.float32(1))
    shift[21, 31, 1] = -np.nextafter(epsilon, np.float32(1))
    shift[22, 32] = (.4, -.7)
    monkeypatch.setattr(fx, "haze_field", lambda *args: shift.copy())
    expected = reference_heat_haze(frames, alpha)
    np.testing.assert_array_equal(fx.heat_haze(frames, alpha), expected)
