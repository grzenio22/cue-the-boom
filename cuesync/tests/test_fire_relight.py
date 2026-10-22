"""Perceptual contracts for local fire light, independent of model assets."""
import importlib
import tracemalloc

import numpy as np
import pytest
import cv2


def _scene(value=.03):
    h, w = 128, 96
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    plate = np.full((h, w, 3), value, np.float32)
    matte = np.zeros((h, w), np.float32)
    matte[78:119, 29:69] = 1
    flame = np.exp(-.5*(((xx-48)/7)**2+((yy-60)/14)**2))
    emission = flame[..., None]*np.array([2., 1.3, .5], np.float32)
    normal = np.zeros_like(plate); normal[..., 2] = -1
    geo = dict(normals=normal, depth=np.zeros_like(matte), valid=matte, light_valid=matte)
    return plate, emission, matte, None, np.array([48, 83], np.float32), 25., geo, 0.


def _relight(*args, **kwargs):
    return importlib.import_module('cuesync.fire_relight').relight(*args, **kwargs)


def test_no_emission_preserves_plate_exactly_without_geometry():
    P, E, matte, lm, p, scale, geo, z = _scene()
    original = P.copy()
    lit, added, metrics = _relight(P, E*0, None, None, p, scale, None, None, {})
    np.testing.assert_array_equal(lit, original)
    np.testing.assert_array_equal(P, original)
    assert np.count_nonzero(added) == 0
    assert metrics['source_power'] == 0


def test_dark_skin_and_room_reflect_new_light_without_global_exposure():
    dark = _scene(.025)
    bright = _scene(.45)
    dark_lit, dark_add, dark_m = _relight(*dark, {})
    _, bright_add, bright_m = _relight(*bright, {})
    palm = (slice(81, 92), slice(38, 58))
    assert float(dark_add[palm].mean()) > .012
    assert float(dark_add[palm].mean())/.025 > float(bright_add[palm].mean())/.45*3
    assert dark_m['room_mean_added'] > bright_m['room_mean_added']*3
    assert dark_lit[2, 2].mean()-dark[0][2, 2].mean() < .005


def test_black_background_cannot_borrow_skin_color_at_matte_edges():
    args = list(_scene(0.))
    args[0][args[2] > .5] = [.5, .3, .2]
    _, added, _ = _relight(*args, dict(room_light=2., dark_gain=2., max_work_size=64))
    assert float(added[args[2] == 0].max(initial=0)) < 1e-7


def test_lighting_feathers_inside_sharp_segmentation_edges():
    # Camera blur can leave a soft photographed hand under a hard SAM outline.
    # A new light must not paint a sharp bright polygon along that outline.
    args = list(_scene(.03))
    _, added, _ = _relight(*args, dict(room_light=0, room_occlusion=0, contact=0))
    edge_step = np.abs(added[87, 29]-added[87, 28]).mean()
    interior = added[87, 35].mean()
    assert float(edge_step) < float(interior)*.3


def test_hand_motion_softens_light_boundary_without_lighting_background_gaps():
    args = _scene(.03)
    cfg = dict(room_light=0, room_occlusion=0, contact=0)
    _, steady, _ = _relight(*args, cfg)
    _, moving, _ = _relight(*args, dict(cfg, hand_motion_px=12))
    assert float(moving[94, 30].mean()) < float(steady[94, 30].mean())*.6
    assert float(moving[99, 48].mean()) > float(steady[99, 48].mean())*.9
    np.testing.assert_allclose(moving[args[2] == 0], steady[args[2] == 0])


def test_motion_transition_has_complementary_room_light_without_a_dark_seam():
    args = _scene(.03)
    _, added, _ = _relight(*args, dict(light=0, contact=0, room_light=1,
                                      room_occlusion=0, hand_motion_px=12))
    # At a blurred hand edge, inward fading hand light must not leave a zero-
    # coverage strip beside the room. Background-only room samples fill it.
    assert float(added[84, 29].mean()) > float(added[84, 28].mean())*.65


@pytest.mark.parametrize('touches_edge', [False, True])
def test_cropped_hand_feather_is_equivalent_at_soft_and_frame_edges(touches_edge):
    matte = np.zeros((120, 180), np.float32)
    left = 0 if touches_edge else 30
    matte[20:100, left:left+60] = 1
    matte[30:70, left+45:left+55] = 0  # A finger gap must stay a gap.
    matte[20:100, left+60:left+61] = .03
    binary = cv2.copyMakeBorder((matte > .05).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)[1:-1, 1:-1]
    x = np.clip((distance-.5)/7., 0, 1)
    expected = matte*x*x*(3-2*x)
    module = importlib.import_module('cuesync.fire_relight')
    region, feather = module._hand_feather(matte, 7.)
    actual = np.zeros_like(matte); actual[region] = feather
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert feather.size < matte.size//2


def test_extreme_emission_only_adds_finite_light_within_display_range():
    args = list(_scene(.97)); args[1] *= 1000
    lit, added, _ = _relight(*args, dict(light=100, contact=100, room_light=100))
    assert lit.dtype == np.float32 and added.dtype == np.float32
    assert np.isfinite(lit).all()
    assert np.all(lit >= args[0])
    assert float(lit.max()) <= 1
    np.testing.assert_allclose(lit-args[0], added, atol=1e-7)


def test_missing_hand_data_uses_room_only_instead_of_painting_a_hand():
    P, E, _, _, p, scale, _, _ = _scene()
    lit, added, metrics = _relight(P, E, None, None, p, scale, None, None, {})
    assert np.isfinite(lit).all() and np.all(added >= 0)
    assert metrics['palm_mean_added'] == 0
    assert metrics['geometry_valid'] is False


def test_losing_landmarks_does_not_extinguish_light_on_segmented_fingers():
    args = list(_scene(.03))
    cfg = dict(contact=0, room_light=0, room_occlusion=0)
    _, tracked, _ = _relight(*args, cfg)
    args[6] = None
    _, fallback, _ = _relight(*args, cfg)
    fingers = (slice(79, 87), slice(31, 43))
    assert float(fallback[fingers].mean()) > float(tracked[fingers].mean())*.55
    assert float(fallback[fingers].mean()) < float(tracked[fingers].mean())*1.2
    assert float(fallback[args[2] == 0].max()) < .01


def test_partial_surface_coverage_cannot_leave_half_the_palm_unlit():
    args = list(_scene(.03))
    args[6]['light_valid'] = args[6]['light_valid'].copy()
    args[6]['light_valid'][:, 48:] = 0
    _, added, _ = _relight(*args, dict(contact=0, room_light=0, room_occlusion=0))
    known = added[79:87, 34:43].mean()
    unsupported = added[79:87, 53:62].mean()
    assert float(unsupported) > float(known)*.55


def test_room_light_follows_a_thrown_flame_instead_of_remaining_at_the_palm():
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    P = np.full((256, 256, 3), .03, np.float32)
    peaks = []
    for height in (165, 65):
        flame = np.exp(-.5*(((xx-128)/8)**2+((yy-height)/10)**2))
        E = flame[..., None]*np.array([2, 1, .3], np.float32)
        _, added, _ = _relight(P,E,None,None,[128,200],35,None,None,{})
        peaks.append(np.unravel_index(np.argmax(added[..., 0]), added.shape[:2])[0])
    assert peaks[0]-peaks[1] > 70


def test_contact_highlight_releases_when_flame_leaves_the_hand():
    P,E,matte,lm,p,scale,geo,z0 = _scene(.03)
    yy, xx = np.mgrid[:128, :96].astype(np.float32)
    contact = []
    for offset in (.1, 2.5):
        f = np.exp(-.5*(((xx-p[0])/5)**2+((yy-(p[1]-offset*scale))/7)**2))
        _, added, _ = _relight(P,f[...,None]*np.array([2,1,.3],np.float32),matte,lm,p,scale,geo,z0,
                              dict(light=0,contact=1,room_light=0,room_occlusion=0))
        contact.append(float(added[81:84,46:50].mean()))
    assert contact[1] < contact[0]*.1


def test_soft_hand_blocker_never_darkens_ambient_or_increases_added_light():
    args = _scene(.04)
    plain, _, _ = _relight(*args, dict(room_occlusion=0))
    blocked, _, _ = _relight(*args, dict(room_occlusion=.6))
    assert np.all(blocked >= args[0])
    assert np.all(blocked <= plain+1e-7)
    assert float(np.max(plain-blocked)) > .0001


def test_light_work_is_cropped_and_keeps_distant_pixels_exact():
    P = np.full((720, 1280, 3), .04, np.float32)
    E = np.zeros_like(P); E[330:350, 630:650] = [1, .6, .2]
    lit, added, metrics = _relight(P, E, None, None, [640, 365], 35, None, None, {})
    np.testing.assert_array_equal(lit[:100], P[:100])
    assert np.count_nonzero(added[:100]) == 0
    assert metrics['work_pixels'] < 720*1280//3
    assert metrics['roi'] != [0, 0, 1280, 720]


def test_full_hd_relighting_has_bounded_temporary_memory_and_keeps_inputs():
    # A wide illumination region used to allocate several simultaneous RGB
    # multiplication temporaries, making a short phone clip miss its CPU budget.
    P = np.full((1080, 1920, 3), .03, np.float32)
    E = np.zeros_like(P); E[450:550, 800:1000] = [1, .5, .2]
    matte = np.zeros(P.shape[:2], np.float32); matte[550:880, 820:1100] = 1
    before = (P.copy(), E.copy(), matte.copy())
    tracemalloc.start()
    try:
        lit, added, metrics = _relight(P, E, matte, None, [960, 600], 220, None, None, {})
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 190*1024**2
    for original, preserved in zip((P, E, matte), before):
        np.testing.assert_array_equal(original, preserved)
    assert not np.shares_memory(lit, P) and not np.shares_memory(added, E)
    assert metrics['roi'] == [0, 0, 1920, 1080]
    np.testing.assert_allclose(lit-P, added, atol=1e-7)


def test_horizontal_fire_illuminates_palm_like_equal_energy_upward_fire():
    yy, xx = np.mgrid[:128, :128].astype(np.float32)
    P = np.full((128, 128, 3), .08, np.float32)
    p = np.array([63.5, 63.5], np.float32)
    matte = (((xx-p[0])**2+(yy-p[1])**2) < 6**2).astype(np.float32)
    normal = np.zeros_like(P); normal[..., 2] = -1
    geo = dict(normals=normal, depth=np.zeros_like(matte), valid=matte, light_valid=matte)
    plume = np.exp(-.5*(((xx-p[0])/5)**2+((yy-(p[1]-29))/8)**2))
    vertical = plume[..., None]*np.array([2, 1.4, .5], np.float32)
    horizontal = np.ascontiguousarray(np.rot90(vertical, -1))
    cfg = dict(contact=0, room_light=0, room_occlusion=0, light=1)
    _, up, _ = _relight(P, vertical, matte, None, p, 20, geo, 0, dict(cfg, direction=[0, -1]))
    _, right, _ = _relight(P, horizontal, matte, None, p, 20, geo, 0, dict(cfg, direction=[1, 0]))
    assert float(right[62:66, 62:66].mean()) == pytest.approx(float(up[62:66, 62:66].mean()), rel=.08)


@pytest.mark.parametrize('field', ['emission', 'matte', 'point', 'scale'])
def test_invalid_shapes_and_coordinates_fail_clearly(field):
    args = list(_scene())
    if field == 'emission': args[1] = args[1][:-1]
    if field == 'matte': args[2] = args[2][:-1]
    if field == 'point': args[4] = [np.nan, 2]
    if field == 'scale': args[5] = 0
    with pytest.raises(ValueError):
        _relight(*args, {})
