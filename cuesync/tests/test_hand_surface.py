"""Hand geometry used for fire depth ordering and directional illumination."""

import numpy as np
import pytest


def hand(z=12.0):
    """A flat open palm with extended fingers, in image pixels."""
    xy = [(64, 104), (39, 84), (25, 74), (17, 62), (9, 52),
          (42, 58), (40, 40), (39, 25), (38, 12),
          (61, 52), (61, 33), (61, 18), (61, 5),
          (80, 57), (81, 40), (82, 26), (83, 14),
          (96, 65), (100, 52), (103, 40), (105, 30)]
    return np.column_stack((xy, np.full(21, z))).astype(np.float32)


def test_segmentation_gaps_never_occlude_or_receive_hand_light():
    # A geometric palm fan must not fill a real hole in SAM coverage.
    from cuesync.hand_surface import surface

    matte = np.ones((128, 128), np.float32)
    matte[70:76, 60:66] = 0
    geo = surface(hand(), matte, 50)
    assert geo["valid"][80, 64] > 0.9
    assert not geo["valid"][70:76, 60:66].any()
    assert not geo["light_valid"][70:76, 60:66].any()


def test_flat_palm_retains_depth_and_camera_facing_unit_normals():
    from cuesync.hand_surface import surface

    geo = surface(hand(), np.ones((128, 128), np.float32), 50)
    assert geo["depth"][80, 64] == pytest.approx(12, abs=0.1)
    supported = geo["valid"] > 0.1
    assert np.allclose(np.linalg.norm(geo["normals"][supported], axis=-1), 1, atol=1e-5)
    assert np.all(geo["normals"][supported, 2] < 0)
    assert np.allclose(geo["normals"][80, 64], [0, 0, -1], atol=0.02)


def test_near_thumb_surface_wins_over_far_palm():
    from cuesync.hand_surface import surface

    lm = hand()
    lm[1:5] = [(39, 84, 12), (48, 78, -8), (61, 72, -20), (75, 67, -25)]
    geo = surface(lm, np.ones((128, 128), np.float32), 50)
    assert geo["depth"][72, 61] < -15
    assert geo["depth"][90, 64] > 8


def test_emitter_interpolates_a_tilted_palm_plane():
    from cuesync.hand_surface import emitter_depth

    lm = hand()
    lm[:, 2] = 0.1 * lm[:, 0] + 0.2 * lm[:, 1] - 4
    assert emitter_depth(lm, [64, 80]) == pytest.approx(18.4, abs=1e-4)


def test_fingertip_source_uses_local_depth_instead_of_distant_palm():
    from cuesync.hand_surface import emission_depth, emitter_depth

    lm = hand()
    lm[8, 2] = -30
    assert emitter_depth(lm, [38, 12]) == pytest.approx(12)
    assert emission_depth(lm, [38, 12]) == pytest.approx(-30)
    assert emission_depth(lm, [64, 80]) == pytest.approx(12)


@pytest.mark.parametrize("lm", [None, np.full((21, 3), np.nan), np.full((21, 3), np.inf)])
def test_lost_tracking_returns_zero_geometry_and_finite_emitter_depth(lm):
    from cuesync.hand_surface import emission_depth, emitter_depth, surface

    geo = surface(lm, np.ones((64, 64), np.uint8) * 255, 30)
    for key in ("depth", "normals", "valid", "light_valid"):
        assert np.isfinite(geo[key]).all()
        assert not geo[key].any()
    assert emitter_depth(lm, [20, 30]) == 0
    assert emission_depth(lm, [20, 30]) == 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 128), (128, 1), (0, 0)])
def test_tiny_and_empty_images_are_safe(shape):
    from cuesync.hand_surface import surface

    geo = surface(hand(), np.ones(shape, np.float32), 50)
    assert geo["depth"].shape == shape
    assert geo["normals"].shape == (*shape, 3)
    assert all(np.isfinite(geo[k]).all() for k in ("depth", "normals", "valid", "light_valid"))


def test_offscreen_hand_has_no_valid_surface_and_never_wraps():
    from cuesync.hand_surface import surface

    lm = hand()
    lm[:, :2] -= 1000
    geo = surface(lm, np.ones((128, 128), np.float32), 50)
    assert not geo["valid"].any()
    assert not geo["light_valid"].any()


def test_byte_and_normalized_mattes_produce_equal_coverage():
    from cuesync.hand_surface import surface

    mask = np.zeros((128, 128), np.uint8)
    mask[20:110, 20:110] = 255
    a, b = surface(hand(), mask, 50), surface(hand(), mask / 255, 50)
    assert np.array_equal(a["valid"], b["valid"])
    assert np.array_equal(a["depth"], b["depth"])


def test_large_hand_downsampling_cannot_fill_full_resolution_sam_gaps():
    from cuesync.hand_surface import surface

    matte = np.zeros((1080, 1920), np.float32)
    matte[20:620, 50:650] = 1
    matte[370:390, 320] = 0
    geo = surface(hand() * 5, matte, 250)
    assert not geo["valid"][370:390, 320].any()
    assert not geo["light_valid"][370:390, 320].any()
    assert geo["depth"][400, 320] == pytest.approx(60, abs=0.5)
    assert geo["bbox"][2] < 1000
    assert geo["bbox"][3] < 800


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_scale_is_rejected(scale):
    from cuesync.hand_surface import surface

    with pytest.raises(ValueError, match="scale"):
        surface(hand(), np.ones((128, 128)), scale)


@pytest.mark.parametrize("matte", [np.zeros((4, 4, 3)), np.full((4, 4), np.nan)])
def test_malformed_matte_is_rejected(matte):
    from cuesync.hand_surface import surface

    with pytest.raises(ValueError, match="matte"):
        surface(hand(), matte, 50)


def test_malformed_landmark_shape_is_rejected():
    from cuesync.hand_surface import surface

    with pytest.raises(ValueError, match="21"):
        surface(np.zeros((20, 3)), np.ones((128, 128)), 50)


@pytest.mark.parametrize("point", [[np.nan, 0], [0, np.inf], None])
def test_unavailable_emitter_position_returns_finite_fallback(point):
    from cuesync.hand_surface import emission_depth, emitter_depth

    assert emitter_depth(hand(), point) == 0
    assert emission_depth(hand(), point) == 0
