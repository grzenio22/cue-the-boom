"""Air, made rather than rendered (cuesync.air): a ball of air over the palm sized to the hand and held by the voice,
a push of air whose front runs out across the room and whose dust stays in the room on a handheld shot, the same
bytes for the same seed, and the plate untouched where the air never reaches."""

import time

import numpy as np
import pytest

from cuesync import air
from cuesync.elements import Timeline


def track_of(n, w, h, xy, d=(0.0, -1.0), scale=24.0, landmarks=None):
    """A hand_track.json dict: the emitter (the palm) at xy(t), pointing d, hand size ``scale``."""
    return {"version": 1, "fps": 24, "W": w, "H": h, "emitter": "palm", "hand": "right",
            "xy": [list(map(float, xy(t))) for t in range(n)], "dir": [list(d)] * n, "scale": [scale] * n,
            "conf": [1.0] * n, "src": ["mp"] * n, "speed": [0.0] * n, "landmarks": landmarks or [None] * n,
            "gesture": ["Open_Palm"] * n, "events": []}


def wall(n, w, h, seed=0):
    """A textured wall (what the air bends has to show) with a little noise, the same on every frame."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 120 + 45 * np.sin(x / 5.0) * np.sin(y / 7.0)
    f = np.stack([base, base * 0.9, base * 0.75], -1) + rng.normal(0, 2, (h, w, 1))
    return np.repeat(np.clip(f, 0, 255).astype(np.uint8)[None], n, 0)


def held(n, a, b, level=0.8, hits=()):
    """A voice holding a sound over frames [a, b), silent around it."""
    e = np.zeros(n)
    e[a:b] = level
    return Timeline(n=n, energy=e, hits=list(hits), sizes=[1.0] * len(hits), holds=[(a, b)], tap_frame=a)


def change(plate, out, t):
    """How much frame ``t`` changed, per pixel (0..255)."""
    return np.abs(out[t].astype(np.int16) - plate[t].astype(np.int16)).max(-1)


def spread(ch, thr=6):
    """(centroid x, y, radius holding 80 % of the changed pixels) of a change map."""
    ys, xs = np.nonzero(ch > thr)
    cx, cy = xs.mean(), ys.mean()
    return cx, cy, float(np.percentile(np.hypot(xs - cx, ys - cy), 80))


W, H = 216, 384  # a phone's 9:16, small


def test_an_unknown_form_or_a_timeline_of_another_length_is_refused():
    plate = wall(10, W, H)
    tr = track_of(10, W, H, lambda t: (108, 260))
    with pytest.raises(ValueError):
        air.render(plate, tr, None, held(10, 2, 8), "tornado")
    with pytest.raises(ValueError):
        air.render(plate, tr, None, held(12, 2, 8), "ball")


def test_frames_the_air_never_reaches_are_the_plate_byte_for_byte():
    n = 70
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (108, 260))
    out = air.render(plate, tr, None, held(n, 20, 30), "ball", seed=3)
    assert out.dtype == np.uint8 and out.shape == plate.shape
    assert np.array_equal(out[:20], plate[:20])  # before the voice
    assert np.array_equal(out[-10:], plate[-10:])  # after the ball has burst and gone
    assert all(change(plate, out, t).max() > 20 for t in range(28, 34))


@pytest.mark.parametrize("form", ["ball", "blast"])
def test_the_same_seed_gives_the_same_bytes_and_another_seed_other_air(form):
    n = 36
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (100 + t, 260))
    tl = held(n, 6, 20, hits=[6])
    a = air.render(plate, tr, None, tl, form, seed=5)
    b = air.render(plate, tr, None, tl, form, seed=5)
    c = air.render(plate, tr, None, tl, form, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_ball_stands_over_the_palm_and_is_sized_to_the_hand():
    n = 40
    plate = wall(n, W, H)
    looks = []
    for scale in (20.0, 40.0):
        tr = track_of(n, W, H, lambda t: (108, 280), scale=scale)  # a palm facing up and at the lens
        out = air.render(plate, tr, None, held(n, 4, 40), "ball", seed=1)
        looks.append(spread(change(plate, out, 30)))
    (x1, y1, r1), (x2, y2, r2) = looks
    assert abs(x1 - 108) < 0.3 * r1 and y1 < 280 and y2 < y1  # over the palm, further off it for a bigger hand
    assert 1.5 < r2 / r1 < 2.6  # sized to the hand


def test_a_held_voice_grows_the_ball():
    n = 100
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (108, 280), scale=22.0)
    out = air.render(plate, tr, None, held(n, 4, 100), "ball", seed=2)
    early, late = spread(change(plate, out, 4 + air.BALL_FORM + 2))[2], spread(change(plate, out, 95))[2]
    assert late > 1.12 * early


def test_a_breath_keeps_the_ball_and_a_long_silence_lets_it_go():
    def lives(gap):
        n = 120
        e = np.zeros(n)
        e[10:30] = 0.8
        e[30 + gap: 50 + gap] = 0.8
        tl = Timeline(n=n, energy=e, hits=[], sizes=[], holds=[(10, 30), (30 + gap, 50 + gap)], tap_frame=10)
        return air.ball_lives(tl, air.voice_level(tl))

    assert len(lives(air.BALL_BRIDGE // 2)) == 1  # a breath between two "ffff"s
    assert len(lives(air.BALL_BRIDGE + air.BALL_LINGER + air.BALL_RELEASE + 10)) == 2
    quiet = Timeline(n=60, energy=np.zeros(60), hits=[], sizes=[], tap_frame=12)
    assert air.ball_lives(quiet, air.voice_level(quiet)) == [(12, 12 + air.BALL_MIN)]  # no voice: one on the tap


def test_what_is_behind_the_palm_is_behind_the_hand():
    # a palm facing up and at the lens: above it in the picture, and nearer the lens, is in front of it (over the
    # fingers); below it, and further off, behind it (hidden by the hand's matte). By depth alone, the dust over a
    # tilted palm's fingers was hidden and the hand read as cut out of the air
    b = air._Blast((100, 200), (0.0, -1.0), 0.6, (0.0, -0.6, 0.8), 24.0, W, H, 0, 1.0, 0, 0)
    pts = np.array([[100, 170, 1.0], [100, 230, 1.0], [100, 200, 1 - 30 / b.f], [100, 200, 1 + 30 / b.f]])
    assert b.behind(pts).tolist() == [False, True, False, True]


def test_the_blast_front_runs_out_from_the_palm_bending_the_wall():
    n = 40
    plate = wall(n, W, H)
    o = (108, 300)
    tr = track_of(n, W, H, lambda t: o, scale=24.0)
    tl = Timeline(n=n, energy=np.zeros(n), hits=[8], sizes=[1.0], tap_frame=8)
    out = air.render(plate, tr, None, tl, "blast", seed=1)
    assert np.array_equal(out[:8], plate[:8])

    def front(t):  # how far from the palm the frame has changed (the front leads the dust it throws)
        d = np.hypot(*(np.mgrid[0:H, 0:W][::-1] - np.reshape(o, (2, 1, 1)))).astype(int).ravel()
        prof = np.bincount(d, change(plate, out, t).astype(float).ravel()) / np.maximum(np.bincount(d), 1)
        return int(np.flatnonzero(prof > 2).max())

    runs = [front(8 + kf) for kf in (2, 4, 6)]
    assert runs[0] < runs[1] < runs[2] and runs[0] > 3 * 24  # out past the hand within two frames
    assert change(plate, out, 8 + 4).max() > 40  # the wall is bent, not tinted


def test_what_the_blast_throws_stays_in_the_room_on_a_handheld_shot():
    n = 40
    plate = wall(n, W, H)
    plate[:] = plate[0, :, :, :].mean((0, 1)).astype(np.uint8)  # a plain wall: only the thrown dust shows
    tr = track_of(n, W, H, lambda t: (108, 300), scale=24.0)
    pan = 2.0  # px a frame: the phone turns, the room slides left... here right, by this much a frame
    cam = np.stack([np.array([[1, 0, pan * t], [0, 1, 0], [0, 0, 1]], float) for t in range(n)])

    def layer(camera):
        tl = Timeline(n=n, energy=np.zeros(n), hits=[6], sizes=[1.0], tap_frame=6, camera=camera)
        out = air.render(plate, tr, None, tl, "blast", seed=2)
        return change(plate, out, 30).astype(float)

    locked, handheld = layer(None), layer(cam)
    assert locked.max() > 10
    fits = [np.sum(handheld[:, s:] * locked[:, :W - s]) / np.sqrt(np.sum(handheld[:, s:] ** 2) * np.sum(
        locked[:, :W - s] ** 2) + 1e-9) for s in range(0, 80)]
    assert abs(int(np.argmax(fits)) - pan * (30 - 6)) <= 2  # the dust has moved with the room, not with the screen


def test_the_push_goes_where_the_arrow_points():
    n = 10
    tr = track_of(n, W, H, lambda t: (108, 300), d=(0.0, -1.0))
    normal = np.array([0.0, 0.0, 1.0])  # a palm flat to the lens
    u, q = air.push_direction(tr, 5, normal, 90.0, 0, n, W, H)  # the arrow drawn to the right
    assert np.allclose(u, (1.0, 0.0), atol=1e-6) and q >= air.AIM_ACROSS
    u, q = air.push_direction(tr, 5, normal, None, 0, n, W, H)
    assert np.allclose(u, (0.0, -1.0), atol=1e-6) and q < 0.05  # no arrow: the hand's way, straight at the lens


def test_the_palm_faces_the_lens_whichever_way_the_landmarks_wind():
    n = 6
    L = np.zeros((21, 3))
    L[0], L[5], L[17] = (100, 300, 0), (90, 270, -5), (120, 272, 5)  # wrist, index and little finger knuckles (px)
    swapped = L.copy()
    swapped[[5, 17]] = L[[17, 5]]  # the same plane wound the other way (a left hand, or MediaPipe's flip)
    for pts in (L, swapped):
        tr = track_of(n, W, H, lambda t: (108, 290), landmarks=[pts.tolist()] * n)
        nm = air.palm_normals(tr, n, W, H)
        assert np.all(nm[:, 2] > 0.9) and np.allclose(np.linalg.norm(nm, axis=1), 1)


@pytest.mark.parametrize("form", ["ball", "blast"])
def test_slow_a_phone_frame_takes_about_half_a_second(form):
    n = 30
    w, h = 1080, 1920
    plate = wall(n, w, h)
    tr = track_of(n, w, h, lambda t: (540 + 3 * t, 1300), scale=280.0)
    tl = held(n, 2, 30, hits=[4])
    t0 = time.perf_counter()
    out = air.render(plate, tr, None, tl, form, seed=1)
    touched = sum(not np.array_equal(out[t], plate[t]) for t in range(n))
    assert (time.perf_counter() - t0) / touched < 1.0
