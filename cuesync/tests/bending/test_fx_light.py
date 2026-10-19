"""WP5, cuesync.fx_light: the light from a tracked emitter, the camera's flash and shake, heat haze, the air blast and
grain. Built on a synthetic shot: a textured wall, a hand matte (palm and forearm) that moves and turns, and a
flickering flame over black at the fingertips."""

import hashlib
import importlib.util
import subprocess
import sys

import cv2
import numpy as np
import pytest

from cuesync import fx_light as fx
from cuesync.transfer import gam, lin

H, W, T = 144, 256, 12


def scene(n: int = T, h: int = H, w: int = W, fluid_rgb=(1.0, 0.42, 0.08), turn: float = 0.0):
    """(plate_lin, element_lin, track, matte) for a synthetic shot. The emitter moves right and turns from up towards
    up-left; the matte is the palm behind it and the forearm going back to the frame's edge; the flame sits in front
    of the emitter, grows and flickers. ``turn`` (degrees) turns the whole hand."""
    rng = np.random.default_rng(5)
    wall = cv2.GaussianBlur(rng.uniform(30, 200, (h, w)).astype(np.float32), (0, 0), 3)
    plate_u8 = np.clip(wall[..., None] * np.float32([1.0, 0.9, 0.8]) + rng.normal(0, 2, (h, w, 3)), 0, 255)
    plate = np.repeat(lin(plate_u8.astype(np.uint8))[None], n, 0)
    s = 22.0
    xy = np.stack([np.linspace(0.40 * w, 0.55 * w, n), np.linspace(0.55 * h, 0.50 * h, n)], -1)
    ang = np.radians(np.linspace(0, -40, n) + turn)  # degrees clockwise from up
    u = np.stack([np.sin(ang), -np.cos(ang)], -1)
    track = {"version": 1, "fps": 24, "W": w, "H": h, "xy": xy.tolist(), "dir": u.tolist(), "scale": [s] * n,
             "conf": [1.0] * n, "src": ["mp"] * n}
    matte = np.zeros((n, h, w), np.float32)
    element = np.zeros((n, h, w, 3), np.float32)
    for f in range(n):
        m = np.zeros((h, w), np.uint8)
        palm = xy[f] - 0.6 * s * u[f]
        cv2.circle(m, tuple(int(v) for v in palm.round()), int(0.6 * s), 255, -1)
        cv2.line(m, tuple(int(v) for v in palm.round()), tuple(int(v) for v in (palm - 6 * s * u[f]).round()), 255,
                 int(0.7 * s))
        matte[f] = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0)
        blob = np.zeros((h, w), np.float32)
        tip = xy[f] + 0.9 * s * u[f]
        cv2.circle(blob, tuple(int(v) for v in tip.round()), int((0.5 + 0.04 * f) * s), 1.0, -1)
        flicker = 0.8 + 0.2 * np.sin(1.7 * f)
        element[f] = cv2.GaussianBlur(blob, (0, 0), 3)[..., None] * np.float32(fluid_rgb) * np.float32(flicker)
    return plate, element, track, matte


def _ahead(track, f, h=H, w=W):
    (x0, y0), (ux, uy) = track["xy"][f], track["dir"][f]
    xs, ys = np.arange(w)[None, :], np.arange(h)[:, None]
    return (xs - x0) * ux + (ys - y0) * uy


# ---- the light ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fluid", ["fire", "steam"])
@pytest.mark.parametrize("with_matte", [True, False])
def test_light_never_darkens(fluid, with_matte):
    plate, element, track, matte = scene()
    out = fx.light_plate(plate, element, track, matte if with_matte else None, fluid)
    assert out.dtype == np.float32 and out.shape == plate.shape
    assert float((plate - out).max()) <= 0.0  # max darkening = 0, exactly, in linear light...
    assert int((gam(plate).astype(int) - gam(out)).max()) <= 0  # ... and in the 8-bit picture
    if fluid == "steam":  # steam scatters the room's light and makes none (it lit the nearest fingertip pink-white)
        assert float((out - plate).max()) == 0.0
    else:
        assert float((out - plate).max()) > 0.05  # fire does light something


def test_light_never_darkens_an_hdr_or_negative_plate():
    plate, element, track, matte = scene(n=3)
    plate = plate * 3.0 - 0.01  # values over 1 and a little under 0, as a resampled linear plate can have
    out = fx.light_plate(plate.astype(np.float32), element, track, matte)
    assert float((plate - out).max()) <= 0.0


@pytest.mark.parametrize("turn", [0.0, 90.0, -135.0])
def test_nothing_is_lit_past_the_hands_far_side(turn):
    plate, element, track, matte = scene(turn=turn)
    added = fx.light_plate(plate, element, track, matte) - plate
    for f in range(T):
        s = track["scale"][f]
        far_side = (_ahead(track, f) < -fx.FEATHER * s) & (matte[f] == 0)
        assert far_side.sum() > 1000
        assert float(np.abs(added[f][far_side]).max()) == 0.0
        lit_hand = (_ahead(track, f) < -fx.FEATHER * s) & (matte[f] > 0.99)
        assert float(added[f][lit_hand].mean()) > 0  # the hand and arm behind the emitter are still lit


def test_light_is_strong_on_the_hand_and_weak_elsewhere():
    plate, element, track, matte = scene(n=1)
    plate[:] = 0.2  # a flat grey wall: only the light differs
    added = (fx.light_plate(plate, element, track, matte) - plate)[0] @ fx._LUMA
    s = track["scale"][0]
    y = element[0] @ fx._LUMA  # round the flame's own middle
    py, px = (np.indices(y.shape) * y).reshape(2, -1).sum(1) / y.sum()
    r = np.hypot(np.arange(W)[None, :] - px, np.arange(H)[:, None] - py)
    ring = (r > 1.0 * s) & (r < 1.5 * s) & (_ahead(track, 0) > -fx.FEATHER * s)
    on, off = ring & (matte[0] > 0.99), ring & (matte[0] == 0)
    assert on.sum() > 20 and off.sum() > 20
    assert added[on].mean() > 4 * added[off].mean() > 0
    far = (r > 5 * s) & (_ahead(track, 0) > 0)
    assert 0 < added[far].max() < 0.25 * added[on].mean()  # light still reaches far, faintly


def test_the_light_comes_from_where_the_flame_is():
    """A fireball lifting off the palm leaves it dimmer; a flame off to one side lights what is under it, not what is
    as far from the hand on the other side (a light at the hand lit both alike)."""
    plate, element, track, matte = scene(n=1)
    plate[:] = 0.2
    s, (x0, y0), (ux, uy) = track["scale"][0], track["xy"][0], track["dir"][0]
    palm = (int(x0 - 0.6 * s * ux), int(y0 - 0.6 * s * uy))

    def lit_at(centres, where):
        el = np.zeros_like(element)
        for c in centres:
            cv2.circle(el[0], tuple(int(v) for v in c), int(0.5 * s), (0.9, 0.5, 0.1), -1)
        return float(((fx.light_plate(plate, el, track, matte) - plate)[0] @ fx._LUMA)[where[1], where[0]])

    near = lit_at([(x0 + 0.8 * s * ux, y0 + 0.8 * s * uy)], palm)
    lifted = lit_at([(x0 + 2.5 * s * ux, y0 + 2.5 * s * uy)], palm)
    assert near > 2.5 * lifted > 0
    a, b = (0.15 * W, 0.3 * H), (0.85 * W, 0.3 * H)  # the flame at a; b mirrors it across the hand
    under_a, under_b = (int(a[0]), int(a[1] + s)), (int(b[0]), int(b[1] + s))
    track_up = dict(track, xy=[[W / 2, 0.9 * H]], dir=[[0.0, -1.0]])  # (the hand below both: nothing is past it)
    side = np.zeros_like(element)
    cv2.circle(side[0], tuple(int(v) for v in a), int(0.5 * s), (0.9, 0.5, 0.1), -1)
    lit = (fx.light_plate(plate, side, track_up, None) - plate)[0] @ fx._LUMA
    assert lit[under_a[1], under_a[0]] > 4 * lit[under_b[1], under_b[0]]


def test_a_plates_white_follows_how_the_room_is_exposed():
    plate = np.full((2, 90, 160, 3), 0.002, np.float32)  # a dark room, its brightest surface at 0.065 (a bath)
    plate[:, 30:60, 60:100] = 0.065
    assert fx.plate_white(plate) == pytest.approx(0.065 / fx.TOP_ALBEDO, rel=0.02)  # not 0.8: skin there is not black
    assert fx.plate_white(np.zeros_like(plate)) == pytest.approx(fx.WHITE_FLOOR)
    plate[:, :20] = 1.3  # a lit room with a white wall: where the camera puts white, at most
    assert fx.plate_white(plate) == pytest.approx(fx.EXPOSED_WHITE)


def test_a_warm_key_light_is_partly_taken_out_of_what_the_fire_lights():
    """Skin under a warm key reads as red paint; the fire's light on it is kept from going salmon (KEY_WB), and a
    surface in shadow still shows its texture when lit (SHADOW_ALBEDO)."""
    plate = np.full((2, 90, 160, 3), 0.002, np.float32)
    plate[:, 30:60, 60:100] = (0.31, 0.10, 0.066)  # the two-palm plate's skin
    tint = fx.plate_tint(plate)
    assert float(tint @ fx._LUMA) == pytest.approx(1.0, abs=1e-4) and tint[0] > tint[1] > tint[2]
    assert np.allclose(fx.plate_tint(np.zeros_like(plate)), 1.0)
    skin = fx.albedo(plate[0, 40, 80], fx.plate_white(plate), tint)
    raw = plate[0, 40, 80] / fx.plate_white(plate)
    assert skin[1] / skin[0] > raw[1] / raw[0]  # less red than the key light painted it
    dark = np.float32([[0.002, 0.002, 0.002], [0.004, 0.004, 0.004]])
    a = fx.albedo(dark, fx.plate_white(plate), np.ones(3, np.float32))
    assert a[1, 0] > 1.3 * a[0, 0] > 0  # a fold a little lighter in the shadow stays lighter under the fire


def test_a_round_hand_is_lit_on_the_side_facing_the_flame():
    """The hand is shaded as if round: its edge toward the flame takes more of the light than its far edge."""
    m = np.zeros((200, 200), np.float32)
    cv2.circle(m, (100, 100), 50, 1.0, -1)
    flame = np.zeros((50, 50), np.float32)
    cv2.circle(flame, (5, 25), 4, 1.0, -1)  # to the left of the hand
    sh = fx.hand_shading(m, flame, 200.0)
    assert sh[100, 55] > sh[100, 145] + 0.3  # the near edge lit, the far edge in the shading's floor
    assert fx.SHADE_AMB - 1e-3 <= sh[m > 0].min() and sh.max() <= 1.0 + 1e-6


def test_light_breathes_with_the_element_and_has_its_colour():
    plate, element, track, matte = scene()
    power, radius, colour = fx.element_light(element, "fire")
    sums = np.array([float(e.reshape(-1, 3).sum(0) @ fx._LUMA) for e in element])
    padded = np.pad(sums, 1, mode="edge")
    np.testing.assert_allclose(power, 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:], rtol=1e-5)
    power2, radius2, _ = fx.element_light(element * 2, "fire")
    np.testing.assert_allclose(power2, 2 * power, rtol=1e-5)  # twice the flame, twice the light, no wiggle of its own
    np.testing.assert_allclose(radius2, radius, rtol=1e-5)  # ... from a ball the same size
    disc = np.zeros((1, H, W, 3), np.float32)
    cv2.circle(disc[0], (128, 72), 30, (0.9, 0.5, 0.1), -1)
    assert fx.element_light(disc)[1][0] == pytest.approx(30, rel=0.03)  # a disc's radius is its own
    for c in colour:
        assert 2000 - 1 <= fx.kelvin_of(c) <= 2700 + 1 and c[0] > c[1] > c[2]  # a flame's orange
        assert abs(float(c @ fx._LUMA) - 1.0) < 1e-3
    white_hot = np.ones_like(element) * element[..., :1]  # a white element still throws a flame's light
    assert fx.kelvin_of(fx.element_light(white_hot, "fire")[2][0]) == pytest.approx(2700, abs=1)
    _, _, cool = fx.element_light(white_hot, "steam")
    assert all(c[2] > c[0] for c in cool)  # steam: a cool white


def test_track_on_another_size_is_scaled_onto_the_plate():
    plate, element, track, matte = scene(n=2)
    half = dict(track, W=W // 2, H=H // 2, xy=(np.asarray(track["xy"]) / 2).tolist(),
                scale=(np.asarray(track["scale"]) / 2).tolist())
    np.testing.assert_allclose(fx.light_plate(plate, element, half, matte), fx.light_plate(plate, element, track, matte),
                               atol=2e-3)


def test_smoke_throws_no_light_only_a_slight_veil():
    plate, element, track, matte = scene(n=3, fluid_rgb=(0.3, 0.3, 0.3))
    out = fx.light_plate(plate, element, track, matte, "smoke")
    room = plate[0].reshape(-1, 3).mean(0)
    change = out - plate
    assert np.all(np.abs(change) <= fx.SMOKE_VEIL * np.abs(room - plate) + 1e-6)
    far = cv2.dilate((element.max(-1) > 1e-4).astype(np.uint8)[0], np.ones((61, 61), np.uint8)) == 0
    assert float(np.abs(change[0][far]).max()) < 1e-4


def test_light_in_place_matches():
    plate, element, track, matte = scene(n=4)
    fresh = fx.light_plate(plate, element, track, matte)
    same = plate.copy()
    assert fx.light_plate(same, element, track, matte, out=same) is same
    np.testing.assert_array_equal(same, fresh)


def test_a_clip_done_in_chunks_matches_it_done_whole():
    plate, element, track, matte = scene()
    white, tint = fx.plate_white(plate), fx.plate_tint(plate)
    whole = fx.light_plate(plate, element, track, matte, white=white, tint=tint)
    for a in (1, 5):  # chunks overlapping their neighbours by a frame: the frames inside them match exactly
        part = fx.light_plate(plate[a - 1:a + 5], element[a - 1:a + 5], {k: v[a - 1:a + 5] if isinstance(v, list) else v
                                                                          for k, v in track.items()}, matte[a - 1:a + 5],
                              white=white, tint=tint)
        np.testing.assert_array_equal(part[1:-1], whole[a:a + 4])
    alpha = np.clip(element.max(-1) / 0.3, 0, 1)
    np.testing.assert_array_equal(np.concatenate([fx.heat_haze(plate[a:a + 5], alpha[a:a + 5], 3, frame0=a)
                                                  for a in range(0, T, 5)]), fx.heat_haze(plate, alpha, 3))
    sigma = fx.shadow_noise(gam(plate))
    np.testing.assert_array_equal(np.concatenate([fx.grain(element[a:a + 5], sigma, 3, frame0=a) for a in range(0, T, 5)]),
                                  fx.grain(element, sigma, 3))


# ---- flash and shake ------------------------------------------------------------------------------------------------

def test_flash_lands_on_the_launch_and_decays_over_three_frames():
    frames = np.full((16, 4, 4, 3), 0.1, np.float32)
    out = fx.flash(frames, [3, 9], [1.0, 0.5])
    stops = np.log2(out[:, 0, 0, 0] / 0.1)
    assert stops[2] == 0 and stops[3] == pytest.approx(0.6, abs=1e-5)  # +0.6 stop on the launch, nothing before
    assert stops[3] > stops[4] > stops[5] > 0  # falling back over 3 frames
    assert stops[6] < 0  # the iris dip after a big burst...
    assert stops[7] == stops[8] == 0
    assert stops[9] == pytest.approx(0.3, abs=1e-5) and stops[12] == 0  # ... but not after a small one
    np.testing.assert_array_equal(fx.flash(frames, [3], [1.0], iris=False)[6], frames[6])


def test_shake_starts_two_frames_after_the_peak_and_settles_in_eight():
    d = fx.shake_offsets(40, [10, 25], [1.0, 0.6], px=4.0, seed=3)
    mag = np.hypot(d[:, 0], d[:, 1])
    assert np.all(mag[:12] == 0) and mag[12] == pytest.approx(4.0)  # the knock, 2 frames after the peak
    assert 3.0 <= mag.max() <= 6.0  # 3-6 px at 1080p
    assert np.all(mag[20:27] == 0)  # settled within 8 frames, before the next strike
    assert mag[27] == pytest.approx(0.6 * 4.0)  # one knock per strike, scaled by its strength
    assert np.all(mag[35:] == 0)
    assert not np.allclose(d, fx.shake_offsets(40, [10, 25], [1.0, 0.6], px=4.0, seed=4))  # seeded


def test_shake_moves_only_its_frames_by_translation():
    rng = np.random.default_rng(1)
    still = cv2.GaussianBlur(rng.uniform(0, 255, (540, 960)).astype(np.float32), (0, 0), 2)  # half size: half the px
    frames = np.repeat(np.repeat(np.clip(still, 0, 255).astype(np.uint8)[None, ..., None], 3, -1), 13, 0)
    out = fx.shake(frames, [2], [1.0], px=4.0, seed=3)
    d = fx.shake_offsets(13, [2], [1.0], px=4.0, seed=3, short_side=540)
    window = cv2.createHanningWindow((960, 540), cv2.CV_32F)
    for f in range(13):
        if f < 4 or f >= 12:
            np.testing.assert_array_equal(out[f], frames[f])  # untouched, byte for byte
        else:
            (sx, sy), _ = cv2.phaseCorrelate(frames[f, ..., 0].astype(np.float32), out[f, ..., 0].astype(np.float32),
                                             window)
            assert sx == pytest.approx(d[f, 0], abs=0.25) and sy == pytest.approx(d[f, 1], abs=0.25)
    assert out[4, :, :2].std() > 0  # the uncovered edge is the frame's own edge, extended


# ---- heat haze and the air blast ------------------------------------------------------------------------------------

def test_heat_haze_is_above_and_around_the_flame_and_at_most_3px():
    h, w = 1080, 1920
    alpha = np.zeros((h, w), np.float32)
    cv2.circle(alpha, (960, 700), 120, 1.0, -1)
    grids = fx._haze_grids(24, h, w, seed=2)
    mags = []
    for t in (0, 6, 12):
        shift = fx.haze_field(alpha, t, grids, min(h, w))
        mag = np.hypot(shift[..., 0], shift[..., 1])
        mags.append(mag)
        assert mag.max() <= 3.0 + 1e-4
        assert mag[480:560, 900:1020].mean() > 0.5  # shimmering above the flame
        assert mag[900:, :].max() == 0 and mag[:, :700].max() == 0  # nothing well below or beside it
    assert not np.allclose(mags[0], mags[1])  # the ripples move
    assert fx.haze_field(np.zeros((h, w), np.float32), 0, grids, 1080) is None


def test_heat_haze_leaves_frames_without_flame_alone():
    plate, element, track, matte = scene(n=4)
    alpha = np.clip(element.max(-1) / 0.3, 0, 1)
    alpha[1] = 0
    out = fx.heat_haze(plate, alpha, seed=1)
    np.testing.assert_array_equal(out[1], plate[1])
    assert np.abs(out[0] - plate[0]).max() > 0


def test_air_wave_travels_out_from_the_palm_with_a_colour_fringe():
    radii = [fx.air_front(k, 1080) for k in range(fx.AIR_FRAMES)]
    assert 6 <= fx.AIR_FRAMES <= 10
    assert all(b[0] > a[0] for a, b in zip(radii, radii[1:]))  # outward, every frame
    assert radii[-1][1] < 0.2 * max(r[1] for r in radii)  # and dying out
    n, h, w = 14, 360, 640
    rng = np.random.default_rng(4)
    tex = lin(np.clip(cv2.GaussianBlur(rng.uniform(0, 255, (h, w, 3)).astype(np.float32), (0, 0), 2), 0, 255)
              .astype(np.uint8))
    frames = np.repeat(tex[None], n, 0)
    origin, start = (200.0, 180.0), 3
    out = fx.air_wave(frames, origin, (1.0, 0.0), start, seed=5)
    for f in list(range(start)) + list(range(start + fx.AIR_FRAMES, n)):
        np.testing.assert_array_equal(out[f], frames[f])
    k = 3
    radius, _ = fx.air_front(k, min(h, w))
    diff = np.abs(out[start + k] - frames[start + k]).max(-1)
    ys, xs = np.nonzero(diff > 1e-3)
    dist = np.hypot(xs - origin[0], ys - origin[1])
    assert np.median(dist) == pytest.approx(radius, rel=0.25)  # the front is where it should be by now
    assert np.all(xs >= origin[0] - 2)  # and only ahead of the palm
    red, blue = (np.abs(out[start + k, ..., c] - frames[start + k, ..., c]).sum() for c in (0, 2))
    assert red > 1.2 * blue  # red bends further than blue: the fringe


# ---- grain ----------------------------------------------------------------------------------------------------------

def test_grain_matches_the_plates_shadow_noise():
    rng = np.random.default_rng(8)
    plate_u8 = np.clip(40 + rng.normal(0, 3.0, (2, 270, 480, 1)), 0, 255).repeat(3, -1).astype(np.uint8)
    assert fx.shadow_noise(plate_u8) == pytest.approx(3.0 * 0.943, rel=0.15)  # what the 3x3 high-pass sees of 3 codes
    layer = np.zeros((2, 270, 480, 3), np.float32)
    layer[:, :, 240:] = lin(np.uint8(40))  # a flat element as dark as the plate's shadows, black on the left
    out = fx.grain(layer, plate_u8, seed=6)
    assert np.all(out >= 0) and np.all(out[:, :, :240] == 0)  # no grain where the layer is black
    got = fx._residual_sigma(gam(out[0, :, 250:470]).astype(np.float32) @ fx._LUMA)
    assert got == pytest.approx(fx.shadow_noise(plate_u8), rel=0.3)  # as noisy as the plate's shadows
    np.testing.assert_array_equal(out, fx.grain(layer, plate_u8, seed=6))
    assert not np.array_equal(out[0], out[1])  # a new grain every frame
    assert not np.array_equal(out, fx.grain(layer, plate_u8, seed=7))


# ---- determinism ----------------------------------------------------------------------------------------------------

def chain_hash(seed: int = 7) -> str:
    """Every step of the compositor's chain on the synthetic shot, hashed (the final 8-bit frames and the lit plate)."""
    plate, element, track, matte = scene()
    lit = fx.light_plate(plate, element, track, matte, "fire", seed)
    lit = fx.heat_haze(lit, np.clip(element.max(-1) / 0.3, 0, 1), seed, out=lit)
    lit = fx.air_wave(lit, track["xy"][2], track["dir"][2], 2, seed, out=lit)
    comp = lit + fx.grain(element, gam(plate), seed)
    comp = fx.flash(comp, [3], [1.0], out=comp)
    final = fx.shake(gam(comp), [5], [1.0], 4.0, seed)
    return hashlib.sha256(final.tobytes() + lit.tobytes()).hexdigest()


def test_output_is_bit_identical_across_runs(monkeypatch):
    first = chain_hash()
    assert chain_hash() == first
    monkeypatch.setattr(fx, "THREADS", 1)
    assert chain_hash() == first  # the thread count changes the speed, never the bytes
    code = (f"import importlib.util as u; s = u.spec_from_file_location('t', {__file__!r}); m = u.module_from_spec(s); "
            "s.loader.exec_module(m); print(m.chain_hash())")
    fresh = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert fresh == first  # and in a fresh process
    assert chain_hash(seed=8) != first
    print(f"fx_light chain sha256 {first}")
