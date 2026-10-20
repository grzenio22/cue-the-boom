"""Air, made rather than rendered (cuesync.air): a ball of air over the palm sized to the hand and held by the voice,
which leaves the hand seen through it whole, bends what is behind it without a tear or a colour fringe on the hand,
pops and gathers tight again on a comeback hit, and bursts into a push of air when the voice lets go; a push of air
whose front runs out across the room, whose ring of dusty air rolls away from the palm (at the lens, ringing the hand
and swelling past the frame), shows its lumps and never a fog or a grey smudge, whose dust stays in the room on a
handheld shot and is one pool the next push throws again, placed on the open hand, and which knocks the phone when it
comes at the lens; the same bytes for the same seed, the plate untouched where the air never reaches, and a frame
within its time."""

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
    n = 140
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (108, 260))
    out = air.render(plate, tr, None, held(n, 20, 30), "ball", seed=3)
    assert out.dtype == np.uint8 and out.shape == plate.shape
    assert np.array_equal(out[:20], plate[:20])  # before the voice
    assert np.array_equal(out[-10:], plate[-10:])  # after the ball has burst and the air it let go has settled
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


def test_the_hand_seen_through_the_ball_keeps_its_shape():
    # bent as much as the wall, a palm under the ball melted into a disc and the fingers' outlines broke
    n = 30
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (108, 280), scale=30.0)
    mask = np.zeros((n, H, W), np.uint8)
    mask[:, 170:330, 70:146] = 255  # a hand under the ball and reaching into it
    tl = held(n, 2, 30)
    bare = change(plate, air.render(plate, tr, None, tl, "ball", seed=2), 24)
    over = change(plate, air.render(plate, tr, mask, tl, "ball", seed=2), 24)
    inside = mask[24] > 0
    assert over[inside].mean() < 0.7 * bare[inside].mean()


def test_letting_go_of_the_ball_throws_a_push_of_air():
    n = 100
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (108, 280), scale=22.0)
    tl = held(n, 4, 30)
    (_, t_rel), = air.ball_lives(tl, air.voice_level(tl))
    out = air.render(plate, tr, None, tl, "ball", seed=4)
    ball = spread(change(plate, out, t_rel - 1))[2]
    gone = t_rel + air.BALL_RELEASE + 1  # the ball itself has burst and thinned out: what shows is the push
    assert change(plate, out, gone).max() > 20
    assert spread(change(plate, out, t_rel + 4))[2] > 1.5 * ball  # out past where the ball was


def test_the_ring_of_dusty_air_rolls_out_along_the_push_and_widens():
    b = air._Blast((108, 300), (0.0, -1.0), 0.8, (0.0, -0.8, 0.6), 24.0, W, H, 0, 1.0, 0, 0)

    def ring(kf):  # (middle, median radius) of the ring's parcels as the camera sees them, its wake left out
        xy = air._project(b.ring(kf)[~b.c_wake], b.pp)
        c = xy.mean(0)
        return c, float(np.median(np.hypot(*(xy - c).T)))

    (c0, r0), (c8, r8) = ring(0), ring(8)
    assert c8[1] < c0[1] - 24 and abs(c8[0] - c0[0]) < 12  # up the picture, the way it was pushed, over a hand
    assert r8 > 1.3 * r0


def test_a_push_on_a_fist_opening_comes_from_the_open_hand():
    n = 12

    def hand(o):  # landmarks: the wrist, and each finger's knuckle with its tip reaching o palms past it
        L = np.zeros((21, 3))
        L[0] = (100, 300, 0)
        for j, x in zip((5, 9, 13, 17), (85, 95, 105, 115), strict=True):
            L[j], L[j + 3] = (x, 260, 0), (x, 260 - 40 * o, 0)
        return L.tolist()

    opens = [0.15] * 6 + [0.5, 0.95, 1.0, 1.0, 1.0, 1.0]
    opened = air.openness(track_of(n, W, H, lambda t: (100, 290), landmarks=[hand(o) for o in opens]), n, W, H)
    assert air._open_from(opened, 5) == 7  # a hit on the fist: the push comes from the first open frame
    assert air._open_from(opened, 9) == 9  # a hit on the open hand: its own frame


def test_the_rooms_dust_is_one_pool_the_next_push_throws_again():
    # each push brought its own room dust and it built up: the dust of one still hung through the next
    b1 = air._Blast((108, 300), (0.0, -1.0), 0.6, (0.0, -0.6, 0.8), 24.0, W, H, 0, 1.0, 0, 0)
    b2 = air._Blast((108, 300), (0.0, -1.0), 0.6, (0.0, -0.6, 0.8), 24.0, W, H, 30, 1.0, 0, 1)
    room = air._Room(b1, 0)
    for kf in range(air.BLAST_FRAMES + 1):
        room.throw(b1, kf, kf, np.eye(3))
    first = room.kick >= 0
    assert first.mean() > 0.5
    for kf in range(air.BLAST_FRAMES + 1):
        room.throw(b2, kf, 30 + kf, np.eye(3))
    assert room.kick.size == air.ROOM_N and (room.kick[first] >= 30).mean() > 0.5  # the same dust, thrown again


def test_a_push_at_the_lens_knocks_the_phone():
    w, h = 540, 960
    frames = np.zeros((8, h, w, 3), np.uint8)
    frames[:, 440:520, 230:310] = 200  # a bright patch round the swell's middle: only the knock moves it
    knocked = frames.copy()
    air.buffet(knocked, 2, (w / 2, h / 2), 1.0, (1.0, 0.0))
    assert np.array_equal(knocked[:2], frames[:2])

    def middle(f):
        y, x = np.mgrid[0:h, 0:w]
        g = f[..., 1].astype(np.float64)
        return (g * x).sum() / g.sum(), (g * y).sum() / g.sum()

    hit = knocked[3]
    (x0, y0), (x1, y1) = middle(frames[3]), middle(hit)
    assert x1 - x0 > 0.5 * air.BUFFET_SHAKE * w / 1080 and abs(y1 - y0) < 0.2  # knocked along the push
    assert sum(not np.array_equal(knocked[t], frames[t]) for t in range(2, 8)) >= 4  # spread over frames, not one


def test_the_knock_never_flickers_the_exposure():
    # a dip on one frame read as an exposure flicker: the knock swells and shakes the picture, it never dims it
    grey = np.full((6, 96, 54, 3), 120, np.uint8)
    knocked = grey.copy()
    air.buffet(knocked, 0, (27, 48), 1.0, (0.0, -1.0))
    assert np.abs(knocked.astype(int) - 120).max() <= 1


def test_a_push_at_the_lens_knocks_the_phone_soon_after_the_voice():
    # waiting for the front to pass the frame's corners, the knock came 0.4 s after the voice
    at_lens = air._Blast((540, 1500), (0.0, -1.0), 0.1, (0.0, -0.1, 0.99), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    assert at_lens.knock() <= air.BUFFET_SOON + 1


def test_what_is_behind_the_palm_is_behind_the_hand():
    # a palm facing up and at the lens: above it in the picture, and nearer the lens, is in front of it (over the
    # fingers); below it, and further off, behind it (hidden by the hand's matte). By depth alone, the dust over a
    # tilted palm's fingers was hidden and the hand read as cut out of the air
    b = air._Blast((100, 200), (0.0, -1.0), 0.6, (0.0, -0.6, 0.8), 24.0, W, H, 0, 1.0, 0, 0)
    pts = np.array([[100, 170, 1.0], [100, 230, 1.0], [100, 200, 1 - 30 / b.f], [100, 200, 1 + 30 / b.f]])
    assert b.behind(pts).tolist() == [False, True, False, True]


def test_dusty_air_in_front_of_the_palm_shows_over_the_hand_once_it_has_left_it():
    # thinned over the whole matte, the ring rose from behind the fingers instead of off the palm
    b = air._Blast((100, 200), (0.0, -1.0), 0.6, (0.0, -0.6, 0.8), 24.0, W, H, 0, 1.0, 0, 0)
    pts = np.array([[100, 170, 1.0], [100, 230, 1.0]])  # one in front of the palm's plane, one behind it
    hand = np.ones((H, W), np.float32)
    early, late = b.seen(pts, pts[:, :2], hand, 0.0), b.seen(pts, pts[:, :2], hand, 1.0)
    assert early.tolist() == [0.0, 0.0] and late.tolist() == [1.0, 0.0]


def test_a_push_at_the_lens_sends_a_ring_at_it_that_swells_past_the_frame():
    # pushed at the lens, the ring hung over the hand as grey puffs between the fingers; centred on a palm low in the
    # frame, only its top arch was on screen, and read as smoke rising behind the hand
    w, h, size = 1080, 1920, 300.0
    b = air._Blast((540, 1500), (0.0, -1.0), 0.1, (0.0, -0.1, 0.99), size, w, h, 0, 1.0, 0, 0)

    def ring(kf):  # (middle, median radius) of the ring as the camera sees it
        xy = air._project(b.ring(kf)[~b.c_wake], b.pp)
        c = np.median(xy, 0)
        return c, float(np.median(np.hypot(*(xy - c).T)))

    (c0, r0), (c4, r4), (c8, r8) = ring(0), ring(4), ring(8)
    assert r0 < 1.2 * size and r8 > 2.0 * size  # rushing out past the frame's edges ...
    assert np.hypot(*(c8 - c0)) < 0.5 * r8  # ... round the hand, not sliding off sideways
    assert c0[1] < 1500 - 0.4 * size and c0[1] + r0 > 1500  # its middle above the palm, its lower arc across it
    assert abs(c8[1] - h / 2) < 0.5 * abs(c0[1] - h / 2)  # drifting to the frame's middle as it swells ...
    assert c4[1] - r4 > 0 and c4[1] + r4 < h  # ... so the whole ring is still on screen half way
    assert air.RING_WAKE * (1 - b.at) < 0.05 and b.c_wake.mean() < 0.05  # no stem filling its hole


def test_the_dusty_air_shows_its_lumps_not_a_fog_over_the_frame():
    # a ring swelling past the frame laid a fog over all of it: the whole picture lifted by 8 grey levels
    P0 = np.full((960, 540, 3), 0.15, np.float32)
    y, x = np.mgrid[0:240, 0:135]
    fog = np.full((240, 135), 20.0, np.float32)
    lump = (40 * np.exp(-((x - 67) ** 2 + (y - 120) ** 2) / 30.0)).astype(np.float32)
    light, tone = np.array([0.0, -1.0]), np.array([0.6, 0.55, 0.5])
    lifts = []
    for dens in (fog, fog + lump):
        P = P0.copy()
        air._lay_cloud(P, dens, dens, np.zeros_like(dens), tone, light, 3.0, None, (0, 0), 0.0)
        lifts.append(P - P0)
    assert np.abs(lifts[0]).mean() < 0.1 * np.abs(lifts[1]).max()  # the even fog hardly shows ...
    assert lifts[1][480, 270].mean() > 0.05  # ... the lump does


def test_dusty_air_is_no_darker_than_a_bright_floor_it_covers():
    # over a bright floor, a ring forming read as a grey smudge
    P = np.full((960, 540, 3), 0.8, np.float32)
    y, x = np.mgrid[0:240, 0:135]
    dens = (40 * np.exp(-((x - 67) ** 2 + (y - 120) ** 2) / 30.0)).astype(np.float32)
    air._lay_cloud(P, dens, dens, np.zeros_like(dens), np.array([0.5, 0.45, 0.4]), np.array([0.0, -1.0]), 3.0, None,
                   (0, 0), 0.0)
    assert P.mean(-1).min() > 0.85 * 0.8


def test_each_push_rolls_its_own_ring():
    # three pushes off one palm drew one ring three times: a replay
    rings = [air._Blast((108, 300), (0.0, -1.0), 0.8, (0.0, -0.8, 0.6), 24.0, W, H, 0, 1.0, 0, j) for j in range(3)]
    assert len({round(b.ring0, 3) for b in rings}) == 3
    leans = [float(np.arccos(np.clip(b.axis @ b.n3, -1, 1))) for b in rings]  # each tilted off the push its own way
    assert min(leans) > 0.25 * air.RING_VARY[0] and len({round(v, 3) for v in leans}) == 3


def test_the_dusty_air_is_lit_on_the_side_towards_the_light():
    # in one flat tone the cloud read as a glow: a lump is brighter towards the room's light, darker away from it
    P = np.full((1080, 1080, 3), 0.2, np.float32)
    y, x = np.mgrid[0:270, 0:270]
    dens = (30 * np.exp(-((x - 135) ** 2 + (y - 135) ** 2) / 50.0)).astype(np.float32)
    air._lay_cloud(P, dens, dens, np.zeros_like(dens), np.array([0.6, 0.6, 0.6]), np.array([0.0, -1.0]), 4.0, None,
                   (0, 0), 0.0)
    top, bottom = P[540 - 4 * 4, 540].mean(), P[540 + 4 * 4, 540].mean()
    assert top > bottom + 0.01


def test_the_balls_streaks_share_one_spin_axis():
    # leaning every which way, the streaks read as a ball of yarn with no axis
    ball = air._Ball(0, 0)
    bands = ~ball.inflow & ~ball.wisp
    assert bands.sum() >= 0.55 * air.BALL_STREAKS
    assert np.all(ball.w[bands, 2] > np.cos(0.2)) and np.all(ball.speed[~ball.inflow] > 0)  # round one axis, one way
    lean = air.spin_axis(np.array([0.0, -0.2, 0.98]))
    assert abs(np.linalg.norm(lean) - 1) < 1e-9 and lean[2] <= air.BALL_AXIS_Z + 1e-9 and lean[1] < 0


def test_the_balls_bend_is_continuous_across_its_rim():
    # switched on outside the rim only, its drag tore a straight edge behind the ball; unsmoothed, the shell's chord
    # kinked one by 9 px over 2
    ball = air._Ball(0, 0)
    R = 500.0
    g = np.arange(-1.7 * R, 1.7 * R, air.FIELD_Q, dtype=np.float32)
    for spin in np.linspace(0, 6, 7):
        ball.spin = spin
        dx, dy = ball.field(g[None, :], g[:, None], R, 1.0, 0, 1.0, None, 1.0)[:2]
        step = max(float(np.abs(np.diff(f, axis=ax)).max()) for f in (dx, dy) for ax in (0, 1))
        assert step / air.FIELD_Q < 1.0  # under a px of shift per px of picture


def test_the_balls_bend_draws_no_colour_fringe_over_the_hand():
    # with a fringe over the matte, a thin cyan line ran along the thumb
    P = np.zeros((40, 40, 3), np.float32)
    P[:, 14:] = 0.8  # an edge both sides of the matte's (x 10) read from, 6 px on
    dx = np.full((40, 40), 6.0, np.float32)
    ca = np.zeros((40, 40), np.float32)
    ca[:, :10] = air.BALL_CA  # off the hand
    air._remap_rgb(P, dx, np.zeros_like(dx), (0, 40, 0, 40), ca)
    assert np.abs(P[:, 10:, 0] - P[:, 10:, 2]).max() < 1e-6 and np.abs(P[:, :, 0] - P[:, :, 2]).max() > 0


def test_a_comeback_hit_pops_the_ball_and_it_gathers_tight_again():
    # thinned out as it swelled, the ball blinked out on the frame after the hit, then hung on as a huge faint swirl
    f = 52
    assert air.ball_pulse(f - 1, [f]) == (0.0, 0.0)
    burst, flash = air.ball_pulse(f + 1, [f])
    assert burst == pytest.approx(air.PULSE_BURST) and flash == 1.0 and air.ball_pulse(f, [f])[1] == 1.0
    assert air.ball_pulse(f + 8, [f])[0] < 0.1 * air.PULSE_BURST and air.ball_pulse(f + 4, [f])[1] < 0.1
    n = 70
    e = np.zeros(n)
    e[4:40] = 0.8
    e[f:] = 0.8
    tl = Timeline(n=n, energy=e, hits=[4, f], sizes=[1.0, 1.0], holds=[(4, 40), (f, n)], tap_frame=4)
    (life,) = air.ball_lives(tl, air.voice_level(tl))
    assert air.ball_pulses(tl, air.voice_level(tl), life) == [f]
    plate = wall(n, W, H)
    out = air.render(plate, track_of(n, W, H, lambda t: (108, 280), scale=22.0), None, tl, "ball", seed=3)
    assert change(plate, out, f + 1).astype(float).sum() > 1.3 * change(plate, out, f - 1).astype(float).sum()


def test_a_hit_after_a_breath_bursts_the_ball_and_it_gathers_again():
    # held through a take, the ball never let go and its shot had no payoff
    n = 120
    e = np.zeros(n)
    e[10:40] = 0.8
    e[60:100] = 0.8
    tl = Timeline(n=n, energy=e, hits=[10, 41, 60], sizes=[1.0] * 3, holds=[(10, 40), (60, 100)], tap_frame=10)
    level = air.voice_level(tl)
    (life,) = air.ball_lives(tl, level)
    assert air.ball_pulses(tl, level, life) == [60]  # not the first hit, nor one the voice never dipped before


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
def test_a_quarter_size_frame_takes_a_quarter_of_the_time(form):
    # the fast suite's guard on speed: a quarter of a phone frame's pixels in a quarter of its second
    n, w, h = 16, 540, 960
    plate = wall(n, w, h)
    tr = track_of(n, w, h, lambda t: (270 + 2 * t, 650), scale=140.0)
    tl = held(n, 2, 16, hits=[3])
    t0 = time.perf_counter()
    out = air.render(plate, tr, None, tl, form, seed=1)
    touched = sum(not np.array_equal(out[t], plate[t]) for t in range(n))
    assert (time.perf_counter() - t0) / touched < 0.25


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
