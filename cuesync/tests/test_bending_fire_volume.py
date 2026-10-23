"""Production boundaries for finite-thickness fire and foreground interaction."""
import numpy as np
import cv2
import pytest

from cuesync import bending as b
from cuesync import fx_light as fx


def test_faint_fire_cannot_cut_a_dark_rim_into_a_bright_plate():
    plate = np.full((12, 16, 3), 2.0, np.float32)
    fire = np.zeros_like(plate)
    fire[3:9, 5:11] = [.03, .007, .001]
    result = b.fire_over(plate, fire, np.full((12, 16), .9, np.float32))
    assert np.all(fx._lum(result) >= fx._lum(plate))
    assert np.array_equal(result[0], plate[0])


def test_depth_overlap_hides_a_near_finger_but_keeps_fire_over_a_far_palm():
    shape = (40, 40)
    matte = np.ones(shape, np.float32)
    geo = dict(valid=matte.copy(), depth=np.full(shape, 8., np.float32))
    geo['depth'][18:20, 16:24] = -12.
    tr = dict(xy=[[20., 24.]], dir=[[0., -1.]], scale=[20.])
    visibility = b.fire_visibility(tr, 0, matte, geo, 0.)
    assert visibility[18, 20] < .01  # near finger crosses the fire
    assert visibility[16, 20] > .99  # far palm does not hide it
    assert visibility[30, 20] < .01  # the root is buried below the surface
    assert 0 < visibility[23, 20] < 1  # soft contact, no cut line


def test_missing_depth_uses_the_existing_directional_holdout():
    matte = np.zeros((40, 40), np.float32)
    matte[20:] = 1.
    tr = dict(xy=[[20., 24.]], dir=[[0., -1.]], scale=[20.])
    actual = b.fire_visibility(tr, 0, matte, None, 0.)
    expected = 1. - b.root_holdout(tr, 0, 40, 40, matte)
    assert np.allclose(actual, expected, atol=1e-7)


def test_palm_parallax_falls_back_per_frame_without_inventing_depth():
    lm = np.zeros((21, 3), np.float32)
    lm[5] = [10, -10, -5]
    lm[17] = [-10, -10, 5]
    changed = lm.copy()
    changed[5, 2] = -10
    changed[17, 2] = 10
    invalid = lm.copy()
    invalid[8, 2] = np.nan
    tr = dict(landmarks=[lm.tolist(), None, changed.tolist(), invalid.tolist()])
    views = b.palm_views(tr, 0)
    assert views[1] == views[3] == 0.
    assert np.isfinite(views).all()
    assert abs(views[2] - views[0]) > .01


def test_heat_haze_preserves_foreground_pixels_after_its_spatial_blur():
    rng = np.random.default_rng(4)
    plate = rng.random((1, 240, 240, 3), dtype=np.float32)
    alpha = np.ones((1, 240, 240), np.float32)
    matte = np.zeros_like(alpha)
    matte[:, 90:150, 100:140] = 1.
    result = fx.heat_haze(plate, alpha, foreground=matte)
    assert np.array_equal(result[:, 90:150, 100:140], plate[:, 90:150, 100:140])
    assert np.abs(result[:, :80] - plate[:, :80]).max() > .01


def test_hand_motion_uses_robust_landmarks_then_screen_track_when_depth_is_missing():
    lm = np.zeros((21, 3), np.float32)
    moved = lm.copy()
    moved[:, 0] = 10.
    moved[8, 0] = 1000.  # one mistracked fingertip cannot smear the whole hand
    tr = dict(landmarks=[lm.tolist(), moved.tolist(), None], src=["mp", "mp", "flow"],
              xy=[[0., 0.], [40., 0.], [43., 4.]], dir=[[0., -1.]]*3, scale=[20.]*3)
    assert b.hand_motion_px(tr, 0) == 0.
    assert b.hand_motion_px(tr, 1) == 10.
    assert b.hand_motion_px(tr, 2) == 5.


def test_hand_motion_can_use_2d_tracking_without_claiming_surface_depth():
    before = np.zeros((21, 2), np.float32)
    now = before + [3., 4.]
    tr = dict(landmarks=[before.tolist(), now.tolist()], src=["rtm", "rtm"],
              xy=[[0., 0.], [40., 0.]], dir=[[0., -1.]]*2, scale=[20.]*2)
    assert b.fire_landmarks(tr, 1) is None
    assert b.hand_motion_px(tr, 1) == 5.


@pytest.mark.parametrize("shape,corner", [((192, 128), False), ((194, 130), True), ((192, 128), True)])
@pytest.mark.parametrize("opacity_level", [.12, .6])
def test_cropped_fire_finish_matches_full_canvas_optics(shape, corner, opacity_level):
    rng = np.random.default_rng(12)
    h, w = shape
    plate = rng.uniform(.03, 1.1, (h, w, 3)).astype(np.float32)
    before = plate.copy()
    emission = np.zeros_like(plate)
    y, x = (0, w-17) if corner else (h//2, w//2)
    emission[y:y+21, x:x+17] = rng.uniform(0., 1.5, (21, 17, 3))
    opacity = np.full((h, w), opacity_level, np.float32)
    # Independent full-canvas optics: calibrated dense gas may attenuate a bright background;
    # faint edges keep the plate luminance, and the entire added highlight goes through the camera response.
    lum, plate_lum = fx._lum(emission), fx._lum(plate)
    smooth = lambda lo, hi, v: (lambda u: u*u*(3.-2.*u))(np.clip((v-lo)/(hi-lo), 0., 1.))
    alpha = np.clip(opacity, 0., 1.)
    body = smooth(.04, .28, lum)*smooth(.12, .45, alpha)
    floor = plate_lum*(1.-.28*body)
    alpha = np.minimum(alpha, (.98*lum+plate_lum-floor)/np.maximum(plate_lum, 1e-5))
    result = plate*(1.-alpha[..., None])+emission
    low = cv2.resize(emission, (max(1, w//4), max(1, h//4)), interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(low, (0, 0), max(.0035*min(h, w), .5))
    result += cv2.resize(low, (w, h))*.035
    knee = np.maximum(.78, plate)
    ceiling = np.maximum(1., plate)
    span = np.maximum(ceiling-knee, 1e-6)
    mapped = np.minimum(ceiling, knee+span*(-np.expm1(-np.maximum(result-knee, 0.)/span)))
    result = np.where(result > knee, mapped, result)
    need = np.maximum(floor-fx._lum(result), 0.)
    headroom = np.maximum(ceiling-result, 0.)
    result += headroom*np.minimum(need/np.maximum(fx._lum(headroom), 1e-6), 1.)[..., None]
    expected = result
    actual = b.finish_fire(plate, emission, opacity)
    assert np.array_equal(plate, before)
    assert np.allclose(actual, expected, rtol=0, atol=2e-7)
    assert b.finish_fire(plate, emission, opacity, out=plate) is plate
    assert np.allclose(plate, expected, rtol=0, atol=2e-7)


@pytest.mark.parametrize("level", [0., 1e-8, .6, 1., 2.])
@pytest.mark.parametrize("shape", [(16, 20), (17, 21)])
def test_fire_finish_without_emission_is_exact_identity(level, shape):
    plate = np.full((*shape, 3), level, np.float32)
    emission = np.zeros_like(plate)
    opacity = np.ones(shape, np.float32)
    assert np.array_equal(b.finish_fire(plate, emission, opacity), plate)
    target = np.empty_like(plate)
    assert b.finish_fire(plate, emission, opacity, out=target) is target
    assert np.array_equal(target, plate)
    before = plate.copy()
    assert b.finish_fire(plate, emission, opacity, out=plate) is plate
    assert np.array_equal(plate, before)


def test_fire_finish_bounds_finite_hdr_emission_over_an_ldr_plate():
    rng = np.random.default_rng(42)
    plate = rng.uniform(0., 1., (32, 40, 3)).astype(np.float32)
    plate[:8] = 1.  # a white wall has no display headroom
    emission = np.exp(rng.uniform(-10., 10., plate.shape)).astype(np.float32)
    opacity = rng.uniform(-.2, 1.2, plate.shape[:2]).astype(np.float32)
    result = b.finish_fire(plate, emission, opacity)
    assert np.isfinite(result).all()
    assert result.min() >= 0.
    assert result.max() <= 1.


def test_fire_finish_keeps_hot_core_detail_instead_of_clipping_it_white():
    plate = np.full((16, 48, 3), .35, np.float32)
    levels = np.linspace(.5, 2.5, 48, dtype=np.float32)
    emission = np.broadcast_to(levels[None, :, None]*np.array([1., .88, .65], np.float32), plate.shape)
    result = b.finish_fire(plate, emission, np.zeros(plate.shape[:2], np.float32))
    assert not np.any(np.all(result >= 1., axis=-1))
    # The practical fire range must remain ordered and resolved after the shoulder.
    peak = result[8].max(-1)
    assert np.all(np.diff(peak) > 0.)
    assert np.ptp(peak) > .10
    assert np.all(result[..., 0] > result[..., 1])
    assert np.all(result[..., 1] > result[..., 2])


@pytest.mark.parametrize("plate_colour", [[.9, .9, .9], [.95, .7, .3], [1., .9, .8], [2., 2., 2.], [2., 1.3, .7]])
def test_fire_finish_cannot_turn_faint_emission_into_a_dark_rim(plate_colour):
    plate = np.empty((32, 40, 3), np.float32)
    plate[...] = plate_colour
    emission = np.zeros_like(plate)
    emission[12:20, 12:28] = [.03, .007, .001]
    result = b.finish_fire(plate, emission, np.ones(plate.shape[:2], np.float32))
    assert np.all(fx._lum(result) >= fx._lum(plate)-1e-7)
    assert np.all(result <= np.maximum(1., plate))


def test_fire_finish_keeps_hot_filaments_warm_yellow_above_the_orange_body():
    plate = np.empty((16, 20, 3), np.float32)
    plate[...] = [.06, .03, .02]
    emission = np.empty_like(plate)
    emission[...] = [2., .7, .1]
    opacity = np.full(plate.shape[:2], .4, np.float32)
    result = b.finish_fire(plate, emission, opacity)
    assert np.all(fx._lum(result) > fx._lum(plate))
    assert np.all(result[..., 0] > .98)
    assert np.all((result[..., 1] > .7) & (result[..., 1] < result[..., 0]))
    assert np.all(result[..., 2] < .2)
    # Compressing every channel by the red peak would suppress this resolved hot green filament.
    assert np.all(result[..., 1]/result[..., 0] > .7)


@pytest.mark.parametrize("opacity", [.12, .28, .6, 1.])
@pytest.mark.parametrize("plate_level", [.9, 3.])
def test_dense_fire_can_attenuate_a_wall_only_within_its_calibrated_body(opacity, plate_level):
    plate = np.full((16, 20, 3), plate_level, np.float32)
    emission = np.full_like(plate, .15)
    alpha = np.full(plate.shape[:2], opacity, np.float32)
    lum = fx._lum(emission)
    u = np.clip((lum-.04)/(.28-.04), 0., 1.)
    v = np.clip((alpha-.12)/(.45-.12), 0., 1.)
    floor = fx._lum(plate)*(1.-.28*(u*u*(3.-2.*u))*(v*v*(3.-2.*v)))
    for result in (b.fire_over(plate, emission, alpha), b.finish_fire(plate, emission, alpha)):
        assert np.all(fx._lum(result) >= floor-3e-7)
        if opacity == .12:
            assert np.all(fx._lum(result) >= fx._lum(plate)-3e-7)
        else:
            assert np.all(fx._lum(result) < fx._lum(plate)-.01)


def test_visibility_patch_matches_full_surface_where_emission_can_reach():
    matte = np.zeros((80, 100), np.float32)
    matte[30:75, 25:80] = 1.
    geo = dict(valid=matte.copy(), depth=np.full(matte.shape, 8., np.float32))
    geo['depth'][38:42, 35:45] = -12.
    tr = dict(xy=[[40., 48.]], dir=[[0., -1.]], scale=[20.])
    box = (24, 55, 20, 65)
    full = b.fire_visibility(tr, 0, matte, geo, 0.)
    cropped = b.fire_visibility(tr, 0, matte, geo, 0., box=box)
    assert cropped.shape == (31, 45)
    assert np.array_equal(cropped, full[24:55, 20:65])


def test_sparse_held_frame_blending_matches_full_frame_weight_order():
    rng = np.random.default_rng(17)
    frames = [np.zeros((80, 120, 3), np.float32) for _ in range(3)]
    boxes = [(5, 45, 20, 50), (30, 60, 40, 65), None]
    for frame, box in zip(frames, boxes):
        if box is not None:
            y0, y1, x0, x1 = box
            frame[y0:y1, x0:x1] = rng.random((y1-y0, x1-x0, 3), dtype=np.float32)
    picks = [(0, .2), (1, .7), (2, .1)]
    expected = sum(weight*frames[k] for k, weight in picks)
    actual = b.blend_element_frames(lambda k: frames[k], picks, dict(enumerate(boxes)), (80, 120))
    assert np.array_equal(actual, expected)
