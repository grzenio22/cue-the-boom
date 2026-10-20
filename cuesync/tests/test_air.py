"""Air, made rather than rendered (cuesync.air): a ball of air over the palm sized to the hand, never wider than about
one and a half spans of it, and held by the voice, its streaks fragments of arcs no longer than 60 degrees on orbits
leaning their own ways round one spin axis and a swirling puff of air, seen by the dust it whirls round (specks lighter
and darker than the wall, which read at phone size, never white paint or a veil) and by what it bends (a straight
edge behind it wobbles by px, never folded and never stepped), its sheen as dark as it is light and never more than
a few grey levels either way, its far side behind the hand, which leaves the hand seen through it whole and
undarkened, the fingers rippling with its spin rather than swollen by a steady lens, bends what is behind it without
a tear, an elbow or a colour fringe on the hand, spins twice as fast after each of the voice's hits, pops and gathers
tight again on a comeback hit without flashing the whole picture, and bursts into a push of air when the voice lets
go; a push of air whose narrow front leads from the palm on the push's frame and runs out across the room at about
1500 px a second, slowing, without sinking what follows it under the wall, whose thin ring of dusty air and burst of
dust follow it at about half its pace (at the lens, the ring ringing the hand and swelling as it comes; out of a
ball, keeping its band), bends what is seen through it, shows its lumps and never a fog, a cream veil, a grey smudge,
a stain on a white switch plate or a hole under the wall, is lit a little brighter than what it covers, scatters what
is seen through it, and over the palm shows in soft faint puffs, not a film or a worm (a ball's burst softened);
whose dust drifts down and settles over 20-30 frames, stays in the room on a handheld shot and is one pool the next
push throws again, placed on the open hand, and which knocks the phone 10-15 px as a damped spring (and swells the
picture when it comes at the lens); the hand's edge takes the air's change on its share of what is behind it; the
same bytes for the same seed, the plate untouched where the air never reaches, and a frame within its time."""

import math
import time

import cv2
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


def test_the_ball_stays_within_one_and_a_half_spans_of_the_hand():
    # grown and popped, it was wider than the frame by the take's last hit
    n = 60

    def hand(t):  # the wrist, the knuckles, and the fingertips: thumb to little finger about 56 px
        L = np.zeros((21, 3))
        L[0] = (108, 300, 0)
        for j, x in zip((5, 9, 13, 17), (96, 104, 112, 120), strict=True):
            L[j], L[j + 3] = (x, 270, 0), (x, 240, 0)
        L[4], L[20] = (80, 268, 0), (132, 247, 0)
        return L.tolist()

    plate = wall(n, W, H)
    tl = held(n, 4, n)
    looks = []
    for landmarks in ([hand(t) for t in range(n)], None):
        tr = track_of(n, W, H, lambda t: (108, 280), scale=60.0, landmarks=landmarks)
        looks.append(spread(change(plate, air.render(plate, tr, None, tl, "ball", seed=1), 50))[2])
    span = air.hand_span(track_of(n, W, H, lambda t: (108, 280), landmarks=[hand(t) for t in range(n)]), n, W, H)[0]
    assert looks[0] <= 0.75 * span  # within one and a half spans across ...
    assert looks[1] > 1.3 * looks[0]  # ... where a hand as big but its span unknown grows a bigger one


def test_the_ball_never_darkens_or_veils_the_hand_seen_through_it():
    # in a dark room its pink-grey coils over the palm read as a wig, and the palm went darker under them
    skin = np.array([0.08, 0.04, 0.035], np.float32)  # a palm in a dark bathroom
    P0 = np.broadcast_to(skin, (600, 600, 3)).copy()
    ball = air._Ball(2, 0)
    a = air.spin_axis(np.array([0.0, -0.78, 0.62]))
    for t in range(12):
        P = P0.copy()
        bent = ball.draw(P, (0, 0), np.array([300.0, 300.0]), 200.0, a, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.6, t,
                         np.ones((600, 600), np.float32), None, 0.0, 1.0)
    lum, lum0 = P @ air.fx._LUMA, P0 @ air.fx._LUMA
    assert lum.mean() >= 0.995 * lum0.mean() and np.percentile(lum, 1) >= 0.9 * lum0.min()  # never darker under it
    # (but for a speck of grit here and there)
    paint = (P - bent) @ air.fx._LUMA
    assert abs(paint.mean()) < 0.02 * lum0.mean()  # and what it carries is barely there over the skin: no milk, no
    # coils


def test_a_held_voice_grows_the_ball():
    n = 100
    plate = wall(n, W, H)
    tr = track_of(n, W, H, lambda t: (108, 280), scale=22.0)
    out = air.render(plate, tr, None, held(n, 4, 100), "ball", seed=2)
    early, late = ((change(plate, out, t) > 6).sum() for t in (4 + air.BALL_FORM + 2, 95))
    assert late > 1.2 * early  # it covers a fifth more of the picture


def test_the_balls_sheen_is_broken_into_arcs_on_orbits_of_their_own():
    # arcs of up to 170 degrees on shared bands of latitude overlapped into closed wound ellipses: a Slinky
    ball = air._Ball(0, 0)
    assert ball.span.max() <= math.radians(60)  # fragments of arcs, never a loop ...
    lean = np.degrees(np.arccos(np.clip(ball.w @ ball.w.T, -1, 1)))
    assert np.median(lean[np.triu_indices(len(lean), 1)]) > 10  # ... each on an orbit leaning its own way ...
    assert np.abs(np.sin(ball.lat)).max() > 0.8  # ... spread over the sphere


def test_the_balls_middle_is_a_swirling_puff_not_more_stripes():
    # streaks on its inner shells were more stripes stacked inside it; with none there, its inside was clean wall
    ball = air._Ball(0, 0)
    assert ball.rho.min() >= 0.8 > air.CORE_PUFF[0]  # the streaks keep to its outer shells ...
    g = np.linspace(-1, 1, 81, dtype=np.float32)
    core = ball._core(g[None, :], g[:, None], ball.turn_axes(np.array([0.0, -0.6, 0.8])))
    r = np.hypot(g[None, :], g[:, None])
    assert core[r < 0.3].mean() > 0.2 and core[r > air.CORE_PUFF[0]].max() == 0  # ... its middle is a puff of air
    # ...
    ball.spin += 1.0
    moved = ball._core(g[None, :], g[:, None], ball.axes)
    assert np.abs(moved - core)[r < 0.5].mean() > 0.1 * core[r < 0.5].mean()  # ... turning with the spin


def test_the_balls_sheen_is_as_dark_as_it_is_light_and_faint():
    # lit, its streaks traced lighter wound ellipses on the wall
    P0 = np.full((700, 700, 3), 0.25, np.float32)  # a plain wall: what shows of the bend is its light
    ball = air._Ball(3, 0)
    a = air.spin_axis(np.array([0.0, -0.78, 0.62]))
    lifts = []
    for t in range(6):
        bent = ball.draw(P0.copy(), (0, 0), np.array([350.0, 350.0]), 240.0, a, 1.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.6, t,
                         None, None, 0.0, 1.0)
        lifts.append(grey_lift(bent, P0))
    lift = np.array(lifts)
    assert lift.max() <= air.SHEEN[2] + 0.6 and lift.min() >= -air.SHEEN[2] - 0.6  # a few grey levels either way
    shows = np.abs(lift) > 1.5
    assert 0.3 < (lift[shows] > 0).mean() < 0.7  # as much darker as lighter


def test_the_balls_dust_is_what_shows_of_it_at_phone_size():
    # 45 specks at a third of an opacity of 0.35, lit a little, changed the wall by 9-10 grey levels at 360x640: the
    # ball nearly vanished
    n = 40
    w, h = 540, 960
    plate = np.full((n, h, w, 3), (120, 118, 96), np.uint8)  # a plain wall: nothing for the bend to bend
    tr = track_of(n, w, h, lambda t: (270, 700), scale=150.0)
    out = air.render(plate, tr, None, held(n, 4, n), "ball", seed=2)
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (w * 2 // 3, h * 2 // 3), interpolation=cv2.INTER_AREA)
             .astype(np.float32) for f in (plate[30], out[30])]  # (as a 1080 frame at 360x640)
    d = small[1] - small[0]
    assert np.percentile(np.abs(d), 99.9) >= 18  # specks that read at phone size ...
    assert (d > 8).sum() > 0.3 * (np.abs(d) > 8).sum() and (d < -8).sum() > 0.3 * (np.abs(d) > 8).sum()  # ... lint
    # lighter than the wall and grit darker ...
    assert np.abs(d).max() <= air.SHELL_LEVELS + air.SHEEN[2] + 1  # ... never white paint


def test_a_hit_whips_the_ball_round_twice_as_fast():
    f = 30
    assert air.hit_spin(f - 1, [f]) == 1.0
    assert air.hit_spin(f, [f]) == air.hit_spin(f + air.HIT_SPIN[1], [f]) == 1 + air.HIT_SPIN[0] == 2.0
    assert 1.0 < air.hit_spin(f + air.HIT_SPIN[1] + 2, [f]) < 2.0 and air.hit_spin(f + 13, [f]) == 1.0


def test_the_fingers_seen_through_the_ball_ripple_and_never_swell():
    # bent by a steady lens, the fingers inside the ball looked fatter than in the plate
    S, R = 600, 200.0
    hand = np.zeros((S, S), np.float32)
    hand[200:600, 220:380] = 1  # a hand reaching up into the ball
    inside = np.zeros((S, S), bool)
    inside[260:560, 250:350] = True  # well inside its matte
    ball = air._Ball(4, 0)
    a = air.spin_axis(np.array([0.0, -0.78, 0.62]))
    got = []
    keep = air._remap_rgb
    air._remap_rgb = lambda P, dx, dy, box, ca: got.append(np.stack([dx, dy])) if box == (0, S, 0, S) else None
    try:
        for t in range(8):
            ball.draw(np.full((S, S, 3), 0.3, np.float32), (0, 0), np.array([300.0, 300.0]), R, a, 1.0, 0.0, 0.0, 0.0,
                      1.0, 0.25, 0.6, t, hand.copy(), None, 0.0, 1.0)
    finally:
        air._remap_rgb = keep
    D = np.array(got)[:, :, inside]  # (frames, 2, px)
    assert np.hypot(D[:, 0], D[:, 1]).max() <= air.HAND_BEND + 0.5  # a few px at the most ...
    assert np.hypot(*D.mean(0)).mean() < 1.5  # ... with no steady part (a lens would swell them) ...
    assert np.hypot(*(D[1:] - D[:-1]).transpose(1, 0, 2)).mean() > 0.3  # ... rippling as it spins


def test_a_bend_is_never_steeper_than_a_quarter_px_per_px():
    # a ball's pulse bent a switch plate's divider by 8-10 px over a few: a tear, not a wobble
    y, x = np.mgrid[0:200, 0:200].astype(np.float32)
    fx_ = np.where(x > 100, 10.0, 0.0).astype(np.float32) * (y > 50)  # a step of 10 px, both ways
    fy_ = (6 * np.sin(x / 3.0)).astype(np.float32)  # and a ripple far too steep
    for cell in (1, 2, 4):
        gx, gy = air._tame(fx_, fy_, cell)
        steep = max(float(np.abs(np.diff(f, axis=ax)).max()) for f in (gx, gy) for ax in (0, 1)) / cell
        assert steep <= 1.02 * air.BEND_SLOPE
    ball = air._Ball(1, 0)  # and the ball's bend is gone well inside the crop round it (BALL_PAD), burst and flashing
    R = 200.0
    P0 = np.full((int(2 * air.BALL_PAD * R), int(2 * air.BALL_PAD * R), 3), 0.3, np.float32)
    got = []
    keep = air._remap_rgb
    air._remap_rgb = lambda P, dx, dy, box, ca: got.append((dx, dy, box))
    try:
        ball.draw(P0.copy(), (0, 0), np.array([air.BALL_PAD * R] * 2), R, air.spin_axis(np.array([0.0, -0.78, 0.62])),
                  1.0, 0.0, air.PULSE_BURST, 1.0, 1.0, 0.25, 0.6, 3, None, None, 0.0, 1.0)
    finally:
        air._remap_rgb = keep
    dx, dy, box = got[0]
    yy, xx = np.mgrid[box[0]:box[1], box[2]:box[3]]
    far = np.hypot(xx - air.BALL_PAD * R, yy - air.BALL_PAD * R) > air.BALL_REACH[1] * R
    assert np.hypot(dx, dy)[far].max() < 0.05  # nothing left to step at the crop's edge


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


def test_a_true_push_at_the_lens_is_tuned_apart_from_one_partly_at_it():
    # one tuning for both, retuned for the true push, halved a ball's release ring and broke it up
    true = air._Blast((540, 1500), (0.0, -1.0), 0.1, (0.0, -0.1, 0.99), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    part = air._Blast((540, 1500), (0.0, -1.0), 0.55, (0.0, -0.55, 0.83), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    ball = air._Blast((540, 1200), (0.0, -1.0), 0.45, (0.0, -0.45, 0.89), 300.0, 1080, 1920, 0, 1.0, 0, 0, ring=260.0)
    assert true.lens == 1.0 and part.lens == 0.0 and ball.lens == 0.0 and ball.whole == 1.0 and part.whole == 0.0
    assert true.core0 / true.ring0 > air.RING_CORE > part.core0 / part.ring0  # a doughnut, and a thin band
    assert ball.core0 / ball.ring0 < air.RING_CORE


def test_the_ring_of_dusty_air_rolls_out_along_the_push_and_widens():
    b = air._Blast((108, 300), (0.0, -1.0), 0.8, (0.0, -0.8, 0.6), 24.0, W, H, 0, 1.0, 0, 0)

    def ring(kf):  # (middle, median radius) of the ring's parcels as the camera sees them, its wake left out
        xy = air._project(b.ring(kf)[~b.c_wake], b.pp)
        c = xy.mean(0)
        return c, float(np.median(np.hypot(*(xy - c).T)))

    (c0, r0), (c8, r8) = ring(0), ring(8)
    assert c8[1] < c0[1] - 12 and abs(c8[0] - c0[0]) < 12  # up the picture, the way it was pushed, over half a hand
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


def middle(f):
    """(x, y) of the bright patch's middle in a frame (its green's weight)."""
    h, w = f.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    g = f[..., 1].astype(np.float64)
    return np.array([(g * x).sum() / g.sum(), (g * y).sum() / g.sum()])


def test_every_push_knocks_the_phone_and_it_settles_as_a_damped_spring():
    # 3 px, and only for a push at the lens, was no knock at all; off on three frames and back at once, it snapped
    # back by 10 px in one frame
    w, h = 540, 960
    n = 14
    frames = np.zeros((n, h, w, 3), np.uint8)
    frames[:, 440:520, 230:310] = 200  # a bright patch round the swell's middle: only the knock moves it
    knock = air.KNOCK_PX * w / 1080
    for swell in (0.0, 1.0):  # a push across the picture, and one at the lens
        knocked = frames.copy()
        air.buffet(knocked, 2, (w / 2, h / 2), swell, knock, (1.0, 0.0))
        assert np.array_equal(knocked[:2], frames[:2])
        moved = np.array([middle(knocked[t]) - middle(frames[t]) for t in range(2, n)])
        dx = moved[:, 0]
        assert np.abs(moved[:, 1]).max() < 0.2 and int(np.argmax(dx)) == 1  # along the push, out in two frames ...
        assert 0.85 * knock < dx.max() < 1.1 * knock  # ... by 10-15 px at 1080 ...
        assert np.abs(np.diff(dx[1:])).max() < 0.55 * knock  # ... back by under half of it a frame ...
        assert -0.35 * knock < dx.min() < -0.12 * knock  # ... and past where it was by about 3 px at 1080 ...
        assert np.abs(dx[-4:]).max() < 0.05 * knock  # ... settled
        if swell == 0:
            assert np.array_equal(knocked[2 + len(air.KNOCK_SHAPE) + 1:], frames[2 + len(air.KNOCK_SHAPE) + 1:])
        else:
            assert sum(not np.array_equal(knocked[t], frames[t]) for t in range(2, n)) >= 6  # the swell too
    assert 10 <= air.KNOCK_PX <= 15


def test_the_knock_never_flickers_the_exposure():
    # a dip on one frame read as an exposure flicker: the knock swells and shakes the picture, it never dims it
    grey = np.full((6, 96, 54, 3), 120, np.uint8)
    knocked = grey.copy()
    air.buffet(knocked, 0, (27, 48), 1.0, air.KNOCK_PX * 54 / 1080, (0.0, -1.0))
    assert np.abs(knocked.astype(int) - 120).max() <= 1


def test_a_push_knocks_the_phone_on_the_strike_and_one_at_the_lens_as_its_gust_arrives():
    # waiting for the front to pass the frame's corners, the knock came 0.4 s after the voice
    at_lens = air._Blast((540, 1500), (0.0, -1.0), 0.1, (0.0, -0.1, 0.99), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    across = air._Blast((540, 1500), (0.0, -1.0), 1.0, (0.0, -1.0, 0.0), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    assert across.knock() == air.KNOCK_AT[0] < at_lens.knock() <= air.KNOCK_AT[1]


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


def test_a_push_at_the_lens_sends_a_ring_at_it_round_the_hand_swelling_as_it_comes():
    # pushed at the lens, the ring hung over the hand as grey puffs between the fingers; centred on a palm low in the
    # frame, only its top arch was on screen, and read as smoke rising behind the hand
    w, h, size = 1080, 1920, 300.0
    b = air._Blast((540, 1500), (0.0, -1.0), 0.1, (0.0, -0.1, 0.99), size, w, h, 0, 1.0, 0, 0)

    def ring(kf):  # (middle, median radius) of the ring as the camera sees it
        xy = air._project(b.ring(kf)[~b.c_wake], b.pp)
        c = np.median(xy, 0)
        return c, float(np.median(np.hypot(*(xy - c).T)))

    (c0, r0), (c8, r8), (_, r16) = ring(0), ring(8), ring(16)
    assert r0 < 0.6 * size and r8 > 2 * r0 and r16 > 1.1 * r8  # swelling as it comes ...
    assert abs(c8[0] - c0[0]) < 0.25 * r8 and c8[1] < c0[1] and c8[1] + r8 > 1500 - 0.25 * size  # ... drifting up
    # to the frame's middle, still round the hand, not sliding off sideways
    assert c0[1] < 1500 and c0[1] + r0 > 1500  # its middle above the palm, its lower arc across it
    assert air.RING_WAKE * (1 - b.at) < 0.05 and b.c_wake.mean() < 0.05  # no stem filling its hole


def test_the_front_leads_the_ring_and_the_dust_it_throws():
    # the puff led and the shock trailed it: the dusty ring was out at 500 px on the push's first frame, the front
    # still on the fingers
    w, h, size = 540, 960, 150.0
    for q, normal in ((0.45, (0.0, -0.45, 0.89)), (0.9, (0.0, -0.9, 0.44))):  # one at the lens, one across it
        b = air._Blast((245, 775), (-0.1, -1.0), q, normal, size, w, h, 0, 1.0, 0, 0)
        lit, drk = air._speck_bufs(h, w, 4)
        coarse = [np.zeros((h // air.CLOUD_Q, w // air.CLOUD_Q), np.float32) for _ in range(3)]
        front, rings, dust = [], [], []
        for kf in range(13):
            b.dust(kf, np.eye(3), np.eye(3), lit, drk, 4, *coarse, None)
            ring = air._project(b.p[b.cloud][~b.c_wake], b.pp)
            thrown = air._project(b.p[~b.cloud & b.alive], b.pp)
            front.append(b.front(kf)[0])
            rings.append(np.percentile(np.hypot(*(ring - b.o).T), 95))
            dust.append(np.percentile(np.hypot(*(thrown - b.o).T), 95))
        front, rings, dust = np.array(front), np.array(rings), np.array(dust)
        assert np.all(front[3:] > rings[3:]) and np.all(front[3:] > dust[3:])  # the front ahead of both ...
        assert 0.25 < (rings[12] - rings[4]) / (front[12] - front[4]) < 0.7  # ... the ring following at about half
        # its pace ...
        assert np.all(np.diff(front) > 0) and front[2] - front[1] > 1.4 * (front[12] - front[11])  # ... which slows


def haze(dens, front=None, broad=None, whole=None, burst=None, puff=None):
    """The coarse maps _lay_cloud lays: all of ``dens`` showing."""
    z = np.zeros_like(dens)
    return air._Haze(dens, dens, *(z if m is None else m for m in (front, broad, whole, burst, puff)))


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
        air._lay_cloud(P, haze(dens, broad=dens), tone, light, 3.0, None, (0, 0), 0.0)  # (swelling past the frame)
        lifts.append(P - P0)
    assert np.abs(lifts[0]).mean() < 0.1 * np.abs(lifts[1]).max()  # the even fog hardly shows ...
    assert lifts[1][480, 270].mean() > 0.04 * 0.15  # ... the lump does, a little over what it covers


def ring_dens(r=60.0, width=4.0, peak=30.0, lumps=0.0):
    """A thin ring of dusty air on the coarse grid (270 x 150), lumpy along it by ``lumps``."""
    y, x = np.mgrid[0:270, 0:150].astype(np.float32)
    d = np.hypot(x - 75, y - 135)
    ang = np.arctan2(y - 135, x - 75)
    return (peak * np.exp(-0.5 * ((d - r) / width) ** 2) * (1 + lumps * np.sin(9 * ang))).astype(np.float32)


def test_a_small_ring_keeps_its_band_and_never_sinks_under_the_wall():
    # its broad mean dropped, its shadow half sent to the wall's tone and its core darkened all round, a ball's
    # release ring was half as strong and broke into a string of lumps with grey holes
    P0 = np.full((1080, 600, 3), 0.15, np.float32)
    tone = np.array([0.6, 0.55, 0.5])
    P = P0.copy()
    air._lay_cloud(P, haze(ring_dens()), tone, np.array([0.0, -1.0]), 3.0, None, (0, 0), 0.0)
    lift = (P - P0) @ air.fx._LUMA
    band = lift[np.abs(np.hypot(*np.mgrid[0:1080, 0:600][::-1] - np.reshape((302, 542), (2, 1, 1))) - 242) < 6]
    assert np.percentile(band, 10) > 0  # all round
    assert lift.min() > -1e-4  # nowhere a shade under the wall


def test_dusty_air_is_a_faint_haze_in_the_colour_of_what_it_covers():
    # in the room's lit tone going towards white and half opaque, a ring was a cream torus stamped on a beige wall,
    # and a milky veil over the hand
    wall_ = np.array([0.22, 0.2, 0.12], np.float32)  # a warm wall, darker than the room's lit tone
    P0 = np.broadcast_to(wall_, (1080, 600, 3)).copy()
    P = P0.copy()
    air._lay_cloud(P, haze(ring_dens(width=6.0, peak=60.0)), np.array([0.7, 0.68, 0.66]), np.array([0.0, -1.0]), 3.0,
                   None, (0, 0), 0.0)
    lum = (P @ air.fx._LUMA) / (P0 @ air.fx._LUMA)
    assert lum.max() < 1 + air.CLOUD_ALPHA * (air.CLOUD_LIFT[0] + air.CLOUD_LIFT[1] - 1) + 0.02  # lit a little
    ratio = P / P0
    assert np.abs(ratio[..., 0] - ratio[..., 2]).max() < 0.15  # in the wall's own colour, not a white or grey one
    hand = np.ones((1080, 600), np.float32)
    P = P0.copy()
    dens = ring_dens(width=6.0, peak=60.0)
    air._lay_cloud(P, haze(dens, front=dens, whole=dens), np.array([0.7, 0.68, 0.66]), np.array([0.0, -1.0]), 3.0, hand,
                   (0, 0), 0.0)
    lum = (P @ air.fx._LUMA) / (P0 @ air.fx._LUMA)
    assert lum.max() < 1 + air.CLOUD_ON_HAND[1] * (air.CLOUD_LIFT[0] + air.CLOUD_LIFT[1] - 1) + 0.02  # over the
    # hand, at its thickest and well in front of it, barely there


def test_dusty_air_in_front_of_the_hand_is_lit_brighter_than_the_skin():
    # in the room's lit tone, about as bright as a lit palm, a ring in front of the hand only took the skin's colour
    # out: it read as behind the hand
    skin = np.array([0.75, 0.42, 0.36], np.float32)
    P0 = np.broadcast_to(skin, (1080, 600, 3)).copy()
    dens = ring_dens()
    P = P0.copy()
    air._lay_cloud(P, haze(dens, front=dens, whole=dens), np.array([0.5, 0.4, 0.3]), np.array([0.0, -1.0]), 3.0,
                   np.ones((1080, 600), np.float32), (0, 0), 0.0)  # (a true push at the lens, well in front of the hand)
    lift = (P - P0) @ air.fx._LUMA
    assert lift.max() > 0.02 and lift.min() > -0.005  # brighter on its lit side, never darker than the skin


def test_dusty_air_scatters_what_is_seen_through_it():
    # a white switch plate's crisp edge cut every lump that crossed it: the ring read as behind the plate
    P0 = np.full((1080, 600, 3), 0.1, np.float32)
    P0[:, 300:] = 0.8  # a sharp edge under the ring's band
    dens = ring_dens(r=56.0, width=8.0, peak=40.0)
    steps = []
    for scatter in (0.0, 1.0):
        P = P0.copy()
        keep, air.SCATTER = air.SCATTER, (air.SCATTER[0], scatter, scatter, air.SCATTER[3])
        try:
            air._lay_cloud(P, haze(dens), np.array([0.6, 0.55, 0.5]), np.array([0.0, -1.0]), 3.0, None, (0, 0), 0.0)
        finally:
            air.SCATTER = keep
        row = P[540 - 4 * 56, :, 1]  # where the band crosses the edge
        steps.append(float(np.abs(np.diff(row[290:310])).max()))
    assert steps[1] < 0.75 * steps[0]


def puffs_of(dens):
    """A ring's density as _Blast.dust splats it into ``puffs``: broken along the ring by a smooth mask."""
    y, x = np.mgrid[0:dens.shape[0], 0:dens.shape[1]].astype(np.float32)
    ang = np.arctan2(y - dens.shape[0] / 2, x - dens.shape[1] / 2)
    return (dens * air._smooth((0.5 + 0.3 * 1.4 * np.sin(5 * ang + 0.5) - air.HAND_PUFF[0])
                               / (air.HAND_PUFF[1] - air.HAND_PUFF[0]))).astype(np.float32)


def grey_lift(P, P0):
    """How much ``P`` (linear) is brighter than ``P0``, in 8-bit grey levels as displayed."""
    def grey(x):
        return 255 * np.clip(x @ air.fx._LUMA, 0, 1) ** (1 / 2.2)
    return grey(P) - grey(P0)


def test_over_the_palm_a_ring_leaving_it_shows_in_soft_faint_puffs():
    # thresholding the air's own lumps kept a ribbon along the band's crest: an opaque cream worm with a hard edge on
    # the palm, the skin blurred inside it and crisp outside (toothpaste); the film between lumps read as a pale band
    # across the base of the fingers. A ring crossing the hand whole (one at the lens, a ball's burst) keeps its band
    skin, wall_ = np.array([0.75, 0.42, 0.36], np.float32), np.array([0.2, 0.17, 0.12], np.float32)
    P0 = np.empty((1080, 600, 3), np.float32)
    P0[:, :300], P0[:, 300:] = wall_, skin
    hand = np.zeros((1080, 600), np.float32)
    hand[:, 300:] = 1  # the right half is the hand, the left the room
    dens = ring_dens(width=6.0, peak=40.0, lumps=0.3)
    tone, light = np.array([0.6, 0.55, 0.5]), np.array([0.0, -1.0])
    on_ring = np.abs(np.hypot(*np.mgrid[0:1080, 0:600][::-1] - np.reshape((302, 542), (2, 1, 1))) - 242) < 4
    lifts, gaps = [], []
    for whole in (False, True):
        P = P0.copy()
        h_ = haze(dens, whole=dens) if whole else haze(dens, puff=puffs_of(dens))
        air._lay_cloud(P, h_, tone, light, 3.0, hand, (0, 0), 0.0)
        lift = grey_lift(P, P0)
        lifts.append(lift)
        band = lift[on_ring & (hand > 0)]
        gaps.append(float((band < 0.1 * band.max()).mean()))
    assert gaps[0] > 0.25 and gaps[0] > gaps[1] + 0.2  # in puffs, where a whole ring keeps its band
    lift = lifts[0]
    over, room = lift[:, 310:], lift[:, :290]
    assert np.percentile(over, 99.5) <= 0.6 * np.percentile(room, 99.5)  # faint over the skin ...
    gy, gx = np.gradient(cv2.GaussianBlur(over, (0, 0), 1.0))
    assert np.hypot(gx, gy).max() * air.HAND_NEAR[2] < over.max()  # ... soft: it rises over more than a lump ...
    n, _, st, _ = cv2.connectedComponentsWithStats((over > 10).astype(np.uint8))
    for i in range(1, n):  # ... and no ribbon lifted by 10 grey levels
        assert max(st[i, 2], st[i, 3]) <= 3 * min(st[i, 2], st[i, 3])


def test_over_the_palm_a_balls_burst_is_softened():
    # its small lumps, laid with the skin scattered under them, were pale polygonal blotches: a rash, or soap foam
    skin = np.array([0.75, 0.42, 0.36], np.float32)
    P0 = np.broadcast_to(skin, (1080, 600, 3)).copy()
    hand = np.ones((1080, 600), np.float32)
    dens = ring_dens(width=3.0, peak=40.0, lumps=0.8)
    steep = []
    for burst in (None, dens):
        P = P0.copy()
        air._lay_cloud(P, haze(dens, whole=dens, burst=burst), np.array([0.6, 0.55, 0.5]), np.array([0.0, -1.0]), 3.0,
                       hand, (0, 0), 0.0)
        gy, gx = np.gradient(grey_lift(P, P0))
        steep.append(float(np.hypot(gx, gy).max()))
    assert steep[1] < 0.6 * steep[0]


def test_dusty_air_is_no_darker_than_a_bright_floor_it_covers():
    # over a bright floor, a ring forming read as a grey smudge
    P = np.full((960, 540, 3), 0.8, np.float32)
    y, x = np.mgrid[0:240, 0:135]
    dens = (40 * np.exp(-((x - 67) ** 2 + (y - 120) ** 2) / 30.0)).astype(np.float32)
    air._lay_cloud(P, haze(dens), np.array([0.5, 0.45, 0.4]), np.array([0.0, -1.0]), 3.0, None, (0, 0), 0.0)
    assert P.mean(-1).min() > 0.85 * 0.8


def test_dusty_air_over_a_white_switch_plate_is_a_veil_not_a_stain():
    # a lump crossing the plate darkened it from 242 to 226 with the plate's edge crisp under it: a grey stain
    P0 = np.full((960, 540, 3), 0.25, np.float32)
    P0[400:, 100:440] = 0.85  # the plate, its top edge across the lump's lit side
    y, x = np.mgrid[0:240, 0:135]
    dens = (60 * np.exp(-((x - 67) ** 2 + (y - 120) ** 2) / 400.0)).astype(np.float32)
    P = P0.copy()
    air._lay_cloud(P, haze(dens), np.array([0.6, 0.55, 0.5]), np.array([0.0, -1.0]), 14.0, None, (0, 0), 0.0)
    lift = grey_lift(P, P0)
    plate = lift[410:640, 150:390]
    assert plate.min() > -4  # hardly darker ...
    assert plate.max() > 1.5  # ... its side towards the light a faint sheen
    assert lift[300:390, 200:340].mean() > 2  # (and a faint haze over the wall)
    step = np.abs(np.diff(P[390:410, 268, 1])).max()  # the plate's edge under the lump ...
    assert step < 0.8 * np.abs(np.diff(P0[390:410, 268, 1])).max()  # ... softened, as dust in front of it does


def test_the_front_never_sinks_the_ring_following_it_under_the_wall():
    # its trailing dark band, as dark as the one ahead of it, sank the inner edge of the ring following it 5 grey levels
    # under the wall
    b = air._Blast((540, 1500), (0.0, -1.0), 0.8, (0.0, -0.8, 0.6), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    for kf in (3, 6, 9, 12):
        P0 = np.full((1920, 1080, 3), 0.2, np.float32)
        P = P0.copy()
        b.bend(P, kf, b.o, b.u, 1.0)
        radius, strength, _ = b.front(kf)
        d = np.hypot(*(np.mgrid[0:1920, 0:1080][::-1] - b.o.reshape(2, 1, 1)))
        lift = grey_lift(P, P0)
        assert lift[d < 0.85 * radius].min() > -2  # behind it
        assert lift[(d > radius) & (d < 1.3 * radius)].min() < -2  # (ahead of it, its dark band)


def edge_rows(P, cols, lo, hi, level):
    """Where a straight dark-over-bright edge crosses each of ``cols`` (rows in [lo, hi)), every crossing."""
    g = cv2.GaussianBlur(P @ air.fx._LUMA, (0, 0), 1.5)
    return [np.flatnonzero(np.diff((g[lo:hi, c] > level).astype(np.int8))) + lo for c in cols]


def test_the_balls_bend_wobbles_a_straight_edge_behind_it_and_never_folds_it():
    # with milky streaks and a smooth lens, the switch plate behind the ball never moved: the air showed only as paint
    S, R = 700, 240.0
    P0 = np.full((S, S, 3), 0.05, np.float32)
    P0[470:] = 0.6  # a straight edge across the ball, half a radius under its middle
    ball = air._Ball(1, 0)
    a = air.spin_axis(np.array([0.0, -0.78, 0.62]))
    shifts = []
    for t in range(16):
        P = P0.copy()
        ball.draw(P, (0, 0), np.array([350.0, 350.0]), R, a, 1.0, 0.0, 0.0, 0.0, 1.0, 0.25, 0.6, t, None, None, 0.0,
                  1.0)
        rows = edge_rows(P, range(180, 520, 5), 380, 560, 0.33)
        assert all(len(r) == 1 for r in rows)  # one crossing in every column: bent, never folded over itself
        shifts.append(np.array([r[0] + 0.5 - 470 for r in rows]))
    dev = np.abs(np.array(shifts))
    assert 6 <= np.percentile(dev.max(1), 50) and dev.max() <= 30  # it wobbles by px, not tens of them ...
    assert np.abs(np.diff(shifts, axis=0)).max() > 1.0  # ... and moves as the ball spins


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
    air._lay_cloud(P, haze(dens), np.array([0.6, 0.6, 0.6]), np.array([0.0, -1.0]), 4.0, None, (0, 0), 0.0)
    top, bottom = P[540 - 4 * 4, 540].mean(), P[540 + 4 * 4, 540].mean()
    assert top > bottom + 0.01


def test_the_balls_streaks_share_one_spin_axis():
    # leaning every which way, the streaks read as a ball of yarn with no axis
    ball = air._Ball(0, 0)
    assert np.all(ball.w[:, 2] >= np.cos(air.FRAG_TILT) - 1e-9) and np.all(ball.speed > 0)  # round one axis, one way
    assert np.all(ball.d_turn > 0)  # (its dust too)
    lean = air.spin_axis(np.array([0.0, -0.2, 0.98]))
    assert abs(np.linalg.norm(lean) - 1) < 1e-9 and lean[2] <= air.BALL_AXIS_Z + 1e-9 and lean[1] < 0


def test_the_balls_bend_is_continuous_across_its_rim():
    # switched on outside the rim only, its drag tore a straight edge behind the ball; unsmoothed, the shell's chord
    # kinked one by 9 px over 2, and eased over 0.05 radii a switch plate's straight edge still had an elbow
    ball = air._Ball(0, 0)
    for R in (500.0, 750.0):  # (and swelled by a pulse)
        g = np.arange(-1.7 * R, 1.7 * R, air.FIELD_Q, dtype=np.float32)
        for spin in np.linspace(0, 6, 7):
            ball.spin = spin
            dx, dy = ball.field(g[None, :], g[:, None], R, 1.0, 0, 1.0, None, 1.0)[:2]
            step = max(float(np.abs(np.diff(f, axis=ax)).max()) for f in (dx, dy) for ax in (0, 1))
            assert step / air.FIELD_Q < 0.4  # under 0.4 px of shift per px of picture


def test_the_hands_edge_takes_the_airs_change_on_its_share_of_the_wall():
    # the plate's edge pixels carry the wall the fingers stood against; lifted less than the wall round them, the
    # fingertips got a dark outline
    h, w = 120, 120
    d = np.hypot(*np.mgrid[0:h, 0:w][::-1] - 60.0) - 30  # a round hand of radius 30, its edge 3 px soft
    cover = np.clip(0.5 - d / 3, 0, 1)[..., None]
    wall_, skin = np.array([0.12, 0.11, 0.08], np.float32), np.array([0.6, 0.3, 0.25], np.float32)
    P0 = (cover * skin + (1 - cover) * wall_).astype(np.float32)
    matte = (d < 0).astype(np.float32)
    lift = np.float32(0.3) * ((1 - cv2.GaussianBlur(matte, (0, 0), 6.0)) * (1 - matte))[..., None]  # the wall's,
    # eased off near the hand
    P = P0 + lift
    ideal = P0 + (1 - cover) * 0.3
    band = np.abs(d) < 4
    before = float(np.abs(P - ideal)[band].max())
    air._reedge(P, P0, matte, 1.0)
    assert float(np.abs(P - ideal)[band].max()) < 0.4 * before


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
    shift = [float(out[t].astype(float).mean() - plate[t].astype(float).mean()) for t in (f - 1, f, f + 1)]
    assert max(shift[1:]) - shift[0] < air.PULSE_RISE + 1.0  # and the punch stays round it: no exposure flash


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
    keep = air.KNOCK_PX, air.BUFFET
    air.KNOCK_PX, air.BUFFET = 0.0, 0.0  # (the phone held still: what moves is the front)
    try:
        out = air.render(plate, tr, None, tl, "blast", seed=1)
    finally:
        air.KNOCK_PX, air.BUFFET = keep
    assert np.array_equal(out[:8], plate[:8])

    def front(t):  # how far from the palm the frame has changed (the front leads the dust it throws)
        d = np.hypot(*(np.mgrid[0:H, 0:W][::-1] - np.reshape(o, (2, 1, 1)))).astype(int).ravel()
        prof = np.bincount(d, change(plate, out, t).astype(float).ravel()) / np.maximum(np.bincount(d), 1)
        return int(np.flatnonzero(prof > 2).max())

    runs = [front(8 + kf) for kf in (1, 3, 5, 7)]
    assert runs[0] > 24 and runs[1] > 2.5 * 24  # off the palm on the push's first frame, out past the hand by its
    # third ...
    assert runs[0] < runs[1] < runs[2] < runs[3]  # ... and on
    assert change(plate, out, 8 + 5).max() > 40  # the wall is bent, not tinted


def test_the_front_is_a_narrow_band_running_out_at_about_1500_px_a_second_and_slowing():
    # a 200 px band that crossed the frame in eight frames, gone before the eye found it
    b = air._Blast((540, 1500), (0.0, -1.0), 1.0, (0.0, -1.0, 0.0), 300.0, 1080, 1920, 0, 1.0, 0, 0)
    speed = np.diff([b.front(kf)[0] for kf in range(air.BLAST_FRAMES + 1)]) * air.fx.FPS
    assert 1300 < speed.mean() < 1900 and np.all(np.diff(speed) < 0) and speed[0] > 1.5 * speed[-1]
    got = {}
    keep = air._remap_rgb
    air._remap_rgb = lambda P, dx, dy, box, ca: got.update(dx=dx, dy=dy, box=box)
    try:
        for kf in (6, 12):
            b.bend(np.full((1920, 1080, 3), 0.2, np.float32), kf, b.o, b.u, 1.0)
            y0, y1, x0, x1 = got["box"]
            col = np.abs(got["dy"][:1500 - y0, 540 - x0])  # straight up the push, from the palm
            wide = np.flatnonzero(col > 0.5 * col.max())
            assert 40 <= wide.max() - wide.min() <= 65  # the band at half its bend
    finally:
        air._remap_rgb = keep


def test_the_blasts_dust_settles_over_twenty_to_thirty_frames():
    # it hung for two seconds, too faint and fine to see once it had slowed; gone 12 frames after the push, the room
    # was clean again before the eye found it
    n = 70
    plate = wall(n, W, H)
    plate[:] = plate[0].mean((0, 1)).astype(np.uint8)  # a plain wall: only the dust shows
    tr = track_of(n, W, H, lambda t: (108, 300), scale=24.0)
    out = air.render(plate, tr, None, Timeline(n=n, energy=np.zeros(n), hits=[6], sizes=[1.0], tap_frame=6), "blast",
                     seed=3)
    specks = [int((change(plate, out, 6 + kf) > 4).sum()) for kf in (10, 14, 20, 26, 30)]
    assert min(specks[:4]) > 0  # still settling 20 frames on ...
    assert specks[4] < specks[1]  # ... thinning out ...
    end = 6 + air.BLAST_FRAMES + sum(air.DUST_LIFE)
    assert np.array_equal(out[end + 1:], plate[end + 1:])  # ... and gone


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
