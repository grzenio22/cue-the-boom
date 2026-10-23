"""Visible contracts for finite-thickness reconstruction, independent of its slabs."""
import numpy as np
import pytest

from cuesync.fx_volume import volume_element


LUMA = np.array([.2126, .7152, .0722], np.float64)


@pytest.fixture
def volume():
    return volume_element


def fire():
    yy, xx = np.mgrid[:112, :96].astype(np.float32)
    shape = np.exp(-((xx-48)/14)**2-((yy-55)/35)**4)
    filaments = .6 + .4*np.sin(xx*.48+yy*.17)**2
    rgb = (shape*filaments)[..., None]*np.array([1., .38, .035], np.float32)
    rgb[shape < .002] = 0.
    return rgb


def test_empty_fire_leaves_plate_untouched(volume):
    source = np.zeros((32, 48, 3), np.float32)
    emission, trans, _ = volume(source, 0.)
    np.testing.assert_array_equal(emission, source)
    np.testing.assert_array_equal(trans, np.ones((32, 48), np.float32))


@pytest.mark.parametrize("size", [(1, 1), (1, 3), (4, 1), (16, 24)])
def test_faint_and_tiny_fire_does_not_pop_to_black_or_nan(volume, size):
    source = np.full((*size, 3), 1e-6, np.float32)
    emission, trans, _ = volume(source, .7)
    assert emission.shape == source.shape and trans.shape == size
    assert np.isfinite(emission).all() and np.isfinite(trans).all()
    assert np.min(emission) >= 0 and 0 <= trans.min() <= trans.max() <= 1
    assert (emission@LUMA).sum() == pytest.approx(.95*(source@LUMA).sum(), rel=2e-5)
    # Absolute faintness must not become an opaque volume after normalization.
    np.testing.assert_array_equal(trans, np.ones(size, np.float32))


@pytest.mark.parametrize("config", [None, {"energy": 8.}, {"energy": .4}])
def test_volume_partitions_energy_instead_of_stacking_brighter_copies(volume, config):
    source = fire()
    original = source.copy()
    emission, _, _ = volume(source, 1.3, -.15, config)
    ratio = (emission@LUMA).sum()/(source@LUMA).sum()
    expected = .95 if config is None else .4 if config["energy"] == .4 else 1.
    assert ratio == pytest.approx(expected, rel=2e-5)
    np.testing.assert_array_equal(source, original)


def test_luminous_body_keeps_extinction_until_the_background_is_known(volume):
    source = fire()
    emission, trans, _ = volume(source, 1.3, .12)
    body = (source@LUMA) > .15
    # A mid-bright wall may be chromatically attenuated by a dense flame body.
    # The earlier assumed-white-background cap discarded this optical depth.
    assert np.percentile((1-trans)[body], 90) > .4
    assert np.mean((1-trans)[body]) > .25
    assert (emission@LUMA).sum() == pytest.approx(.95*(source@LUMA).sum(), rel=2e-5)


@pytest.mark.parametrize("background", [.65, .98])
def test_faint_fire_cannot_draw_a_dark_rim_on_bright_wall(volume, background):
    from cuesync.bending import finish_fire, fire_over

    source = fire()*.01
    emission, trans, _ = volume(source, 2., .15, {"optical_depth": 2.})
    opacity = 1.-trans
    plate = np.full_like(emission, background)
    # Transmission is intrinsic to the source; actual plate-aware composition
    # provides the faint-edge guard, including after its display shoulder.
    for result in (fire_over(plate, emission, opacity), finish_fire(plate, emission, opacity)):
        assert ((result-plate)@LUMA).min() >= -2e-7
        assert np.isfinite(result).all()


def test_very_faint_gas_has_negligible_intrinsic_extinction(volume):
    source = fire()*.001
    emission, trans, _ = volume(source, 1.7)
    assert (1-trans).max() < .01
    assert emission.max() < .002


@pytest.mark.parametrize("energy", [0., 1e-6])
def test_attenuated_emission_cannot_leave_invisible_opaque_gas(volume, energy):
    emission, trans, _ = volume(fire(), 1.7, config={"energy": energy})
    assert emission.max() < 2e-6
    np.testing.assert_array_equal(trans, np.ones_like(trans))


def test_view_changes_depth_continuously_without_a_global_shift(volume):
    source = fire()
    left = volume(source, 1., -.15)[0]
    right = volume(source, 1., .15)[0]
    close = volume(source, 1., -.149)[0]
    change = np.linalg.norm(left-right)
    assert change > np.linalg.norm(source)*.001
    assert np.linalg.norm(left-close) < change*.1
    assert change < np.linalg.norm(source)*.15
    # Distant black background cannot become a halo from spreading the volume.
    # The source reaches x=82; leave room for its legitimate subpixel edge shift.
    assert not np.any(left[:, :8]) and not np.any(right[:, -8:])


def test_advected_structure_is_reproducible_and_has_no_frame_noise(volume):
    source = fire()
    now = volume(source, 1.2, .12)[0]
    repeat = volume(source, 1.2, .12)[0]
    next_frame = volume(source, 1.2+1/24, .12)[0]
    np.testing.assert_array_equal(now, repeat)
    change = np.linalg.norm(now-next_frame)/np.linalg.norm(now)
    assert 0 < change < .03


@pytest.mark.parametrize("time", [0., 1.3, 13/24])
def test_cached_depth_field_preserves_float32_phase_and_advection(time):
    from cuesync.fx_volume import _prepare_noise

    # Reference the unhoisted field arithmetic: even a small phase change can
    # move a remapped filament across a subpixel interpolation boundary.
    yy, xx = np.mgrid[:17, :23].astype(np.float32)
    x, y = (xx-11)/7., yy/17.*3.
    field = _prepare_noise(x, y, time)
    for z in (-1.8, -.37, 0., .82, 1.8):
        advected_y = y+time*.40
        expected = (np.sin(x*5.3+advected_y*3.6+z*3.1)*.25
                    + np.sin(x*10.1-advected_y*7.9+z*5.7+time*.24)*.18
                    + np.sin(x*18.4+advected_y*11.2-z*8.8-time*.19)*.11
                    + np.sin(x*34.7-advected_y*20.5+z*13.4)*.055)
        np.testing.assert_array_equal(field(z), expected)


def test_thermal_core_does_not_fill_large_turbulent_gaps(volume):
    from cuesync.bending import finish_fire, fire_over

    source = fire()
    source[40:72, 40:56] = 0.
    emission, trans, _ = volume(source, 1., .15)
    assert np.max(emission[45:67, 45:51]) < 1e-7
    assert np.min(trans[45:67, 45:51]) == 1.
    assert emission.sum() > 0
    plate = np.full_like(emission, .85)
    gap = (slice(45, 67), slice(45, 51))
    np.testing.assert_array_equal(fire_over(plate, emission, 1.-trans)[gap], plate[gap])
    # Optical bloom may reach a real dark gap, but neither extinction nor the
    # camera response can turn that absent source into a dark stripe.
    finished = finish_fire(plate, emission, 1.-trans)
    assert ((finished[gap]-plate[gap])@LUMA).min() >= -2e-7


def test_invalid_pixels_do_not_poison_the_rest_of_the_fire(volume):
    source = fire()
    source[30, 20] = [np.nan, np.inf, -np.inf]
    source[0, 0] = [-1., -2., -3.]
    emission, trans, _ = volume(source, 1.)
    assert np.isfinite(emission).all() and np.isfinite(trans).all()
    assert emission.min() >= 0 and 0 <= trans.min() <= trans.max() <= 1
    assert emission.max() < 3.


@pytest.mark.parametrize("offset", [(40, 60), (0, 0), (160, 110)])
def test_bounded_volume_preserves_original_canvas_parallax_and_padding(volume, offset):
    source = np.zeros((224, 256, 3), np.float32)
    x, y = offset
    patch = fire()[:min(112, 224-y), :min(96, 256-x)]
    source[y:y+patch.shape[0], x:x+patch.shape[1]] = patch
    bounds = (x, y, x+patch.shape[1], y+patch.shape[0])
    expected_E, expected_T, _ = volume(source, 1.8, -.15)
    crop_E, crop_T, info = volume(source, 1.8, -.15, active_bounds=bounds, crop_output=True)
    x0, y0, x1, y1 = info["crop"]
    actual_E = np.zeros_like(source)
    actual_T = np.ones(source.shape[:2], np.float32)
    actual_E[y0:y1, x0:x1] = crop_E
    actual_T[y0:y1, x0:x1] = crop_T
    np.testing.assert_allclose(actual_E, expected_E, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual_T, expected_T)
    assert crop_E.shape[0]*crop_E.shape[1] < source.shape[0]*source.shape[1]


def test_empty_bounded_volume_returns_an_empty_crop_without_allocating_a_plate(volume):
    source = np.zeros((128, 160, 3), np.float32)
    emission, trans, info = volume(source, 0., active_bounds=(10, 20, 40, 50), crop_output=True)
    assert emission.shape == (0, 0, 3) and trans.shape == (0, 0)
    assert info["crop"] == [0, 0, 0, 0]
