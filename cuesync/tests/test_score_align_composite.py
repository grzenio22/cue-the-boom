import numpy as np
import pytest

from cuesync.align import ElementTiming, element_index_for, plan_alignment
from cuesync.composite import composite
from cuesync.score import measure_composite_onset, measure_element


# ---------------------------------------------------------------- Task 6
def ramp_element(n=60, start=10, peak=30):
    f = np.zeros((n, 32, 32, 3), np.uint8)
    for i in range(start, n):
        v = min(1, (i-start)/(peak-start)) if i <= peak else max(0, 1-(i-peak)/20)
        f[i, 8:24, 8:24] = int(255*v)
    return f


def test_onset_and_peak():
    t = measure_element(ramp_element())
    assert 10 <= t.visible_onset <= 13 and 29 <= t.peak <= 31


def test_black_element_raises():
    with pytest.raises(ValueError):
        measure_element(np.zeros((10, 8, 8, 3), np.uint8))


def test_composite_onset():
    plate = np.full((20, 8, 8, 3), 100, np.uint8); comp = plate.copy(); comp[7:] += 40
    assert measure_composite_onset(plate, comp) == 7


# ---------------------------------------------------------------- Task 7
T = ElementTiming(visible_onset=6, peak=20, curve=np.zeros(1))


def test_onset_lands_on_anchor():
    a = plan_alignment(40, T)
    assert a.offset == 34 and element_index_for(a, T, 40, 121) == 6 and element_index_for(a, T, 39, 121) is None


def test_peak_remap_clipped():
    a = plan_alignment(40, T, plate_peak_frame=47)     # wants speed 2.0
    assert a.speed == 1.25


def test_index_none_past_end():
    a = plan_alignment(0, ElementTiming(0, 5, np.zeros(1)))
    assert element_index_for(a, ElementTiming(0, 5, np.zeros(1)), 200, 121) is None


# ---------------------------------------------------------------- Task 8
def setup(plate_val, elem_rgb, n=10):
    plate = np.full((n, 64, 64, 3), plate_val, np.uint8)
    el = np.zeros((n, 32, 32, 3), np.uint8); el[:, 8:24, 8:24] = elem_rgb
    T = ElementTiming(0, 5, np.zeros(1))
    return plate, el, T


def test_black_element_and_pre_anchor_untouched():
    plate, el, T = setup(120, (0, 0, 0))
    out = composite(plate, el, plan_alignment(3, T), T)
    assert np.array_equal(out, plate)


def test_frames_before_anchor_bit_exact():
    plate, el, T = setup(120, (255, 140, 0))
    out = composite(plate, el, plan_alignment(4, T), T)
    assert np.array_equal(out[:4], plate[:4]) and out[6].mean() > plate[6].mean()


def test_smoke_darkens_bright_plate_not_whitewash():
    plate, el, T = setup(250, (190, 190, 190))
    out = composite(plate, el, plan_alignment(0, T), T)
    assert out[2, 32, 32].mean() < 245          # grey smoke over white must read darker


def test_fire_screens_and_glows_outside_blob():
    plate, el, T = setup(60, (255, 140, 0))
    out = composite(plate, el, plan_alignment(0, T), T)
    assert out[2, 32, 32, 0] > 200               # inside the fire
    assert out[2, 32, 12, 0] > 62                # glow spills ~4 px beyond the element edge (blob spans 16..48)


def test_element_longer_than_plate_truncates():
    plate, el, T = setup(100, (255, 140, 0), n=5)
    el = np.repeat(el[:1], 50, 0)
    assert composite(plate, el, plan_alignment(2, T), T).shape == plate.shape


def test_far_pixels_bit_exact_even_with_fire():
    plate = np.random.default_rng(0).integers(0, 255, (3, 256, 256, 3), dtype=np.uint8)
    el = np.zeros((3, 256, 256, 3), np.uint8); el[:, 120:136, 120:136] = (255, 140, 0)
    T0 = ElementTiming(0, 1, np.zeros(1))
    out = composite(plate, el, plan_alignment(0, T0), T0)
    assert np.array_equal(out[:, :40, :40], plate[:, :40, :40])   # far corner untouched despite glow
