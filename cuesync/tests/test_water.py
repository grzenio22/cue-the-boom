"""cuesync.water: water bent from an open palm, made from the plate (the orb and the whip). Built on a synthetic shot:
a textured wall, a palm with no landmarks (taken to face the camera, tipped up), still or moving, its matte, and a
voice's timeline."""

import cv2
import numpy as np
import pytest

from cuesync import water
from cuesync.elements import Timeline
from cuesync.transfer import lin

H, W = 192, 128
S = 40.0  # the hand's size px


def wall(n: int, seed: int = 3) -> np.ndarray:
    """A textured wall, the same on every frame, uint8 [n, H, W, 3]."""
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.uniform(40, 220, (H, W)).astype(np.float32), (0, 0), 2.5)
    stripes = 30.0 * np.sin(np.arange(W) / 3.0)[None, :]
    f = np.clip((g + stripes)[..., None] * np.float32([1.0, 0.92, 0.82]), 0, 255).astype(np.uint8)
    return np.repeat(f[None], n, 0)


def track_of(n: int, xy=lambda t: (64.0, 130.0), d=(0.0, -1.0), events=None) -> dict:
    """A hand_track.json dict: the palm's point at xy(t), its direction d (up: the palm faces the camera, tipped up)."""
    return {"version": 1, "fps": 24, "W": W, "H": H, "emitter": "palm", "hand": "right",
            "xy": [list(map(float, xy(t))) for t in range(n)], "dir": [list(d)] * n, "scale": [S] * n,
            "conf": [1.0] * n, "src": ["mp"] * n, "speed": [0.0] * n, "landmarks": [None] * n,
            "gesture": ["Open_Palm"] * n, "events": events or []}


def matte_of(track: dict) -> np.ndarray:
    """The palm (a disc round its point) and the forearm down to the frame's edge, uint8 [n, H, W]."""
    out = np.zeros((len(track["xy"]), H, W), np.uint8)
    for t, (x, y) in enumerate(track["xy"]):
        cv2.circle(out[t], (int(x), int(y)), int(0.55 * S), 255, -1)
        cv2.line(out[t], (int(x), int(y)), (int(x), H + 10), 255, int(0.6 * S))
    return out


def voice(n: int, hum=(4, 40), hits=((44, 1.0),), aim=None) -> Timeline:
    """A hum over ``hum`` frames and hits (frame, size), each a burst of loudness dying away."""
    e = np.zeros(n)
    e[hum[0]:hum[1]] = 0.35
    t = np.arange(n)
    for f, z in hits:
        e = np.maximum(e, np.where(t >= f, z * np.exp(-(t - f) / 4.0), 0.0))
    return Timeline(n, e, [f for f, _ in hits], [z for _, z in hits], [hum], tap_frame=2, aim=aim)


def scenes(form: str, tl: Timeline, track: dict, seed: int = 0):
    hand = water.hand_of(track, tl.n)
    fn = water.orb_scenes if form == "orb" else water.whip_scenes
    return hand, fn(tl, track, hand, (H, W), seed, None)


# ---- the contract ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["orb", "whip"])
def test_render_keeps_the_contract_and_the_same_inputs_give_the_same_bytes(form):
    n = 60
    plate, track = wall(n), track_of(n)
    tl = voice(n)
    out = water.render(plate, track, matte_of(track), tl, form, seed=4)
    assert out.dtype == np.uint8 and out.shape == plate.shape
    assert np.array_equal(out[:3], plate[:3])  # nothing before the voice
    assert all(not np.array_equal(out[k], plate[k]) for k in (20, 30, 40))  # the water through the voice
    assert np.array_equal(out, water.render(plate, track, matte_of(track), tl, form, seed=4))
    assert not np.array_equal(out, water.render(plate, track, matte_of(track), tl, form, seed=5))
    assert np.array_equal(water.render(plate, track, None, tl, form, seed=4)[:3], plate[:3])  # no matte: still renders


def test_render_refuses_what_it_cannot_play():
    n = 10
    plate, track = wall(n), track_of(n)
    with pytest.raises(ValueError):
        water.render(plate, track, None, voice(n, hum=(1, 5), hits=()), "arc")
    with pytest.raises(ValueError):
        water.render(plate, track, None, voice(n + 1, hum=(1, 5), hits=()), "orb")


# ---- when it does what ----------------------------------------------------------------------------------------------

def test_the_orb_gathers_with_the_voice_and_is_thrown_on_its_last_hit():
    n = 80
    p = water.orb_plan(voice(n, hum=(6, 40), hits=((20, 0.5), (44, 1.0))), track_of(n))
    assert (p.gather, p.formed) == (6, 6 + water.GATHER)
    assert (p.release, p.throw, p.size) == (44, True, 1.0)
    assert p.jolts == [(20, 0.5)]  # the hit while it is held jolts it


def test_a_voice_opening_on_its_hit_gathers_the_orb_just_before_it():
    p = water.orb_plan(voice(60, hum=(30, 31), hits=((30, 1.0),)), track_of(60))
    assert p.release == 30 and p.throw
    assert p.formed == 30 - water.GATHER_LEAD and p.gather == p.formed - water.GATHER


def test_a_thrust_throws_the_orb_before_the_last_hit():
    n = 80
    track = track_of(n, events=[{"frame": 30, "kind": "extension"}])
    p = water.orb_plan(voice(n, hum=(6, 50), hits=((60, 1.0),)), track)
    assert (p.release, p.throw, p.size) == (30, True, water.THRUST_SIZE)


def test_a_hit_the_voice_holds_on_long_after_jolts_the_orb_instead_of_throwing_it():
    n = 90
    p = water.orb_plan(voice(n, hum=(6, 70), hits=((20, 1.0),)), track_of(n))
    assert not p.throw and p.jolts == [(20, 1.0)]
    assert p.release == 70 + water.LINGER  # let fall when the voice stops


def test_a_voice_that_just_stops_lets_the_orb_fall_and_no_voice_holds_it():
    n = 80
    p = water.orb_plan(voice(n, hum=(6, 40), hits=()), track_of(n))
    assert (p.release, p.throw) == (40 + water.LINGER, False)
    p = water.orb_plan(Timeline(n, np.zeros(n), [], [], tap_frame=9), track_of(n))
    assert (p.gather, p.release) == (9, None)


def test_the_whip_pours_out_with_the_voice_cracks_on_each_hit_and_lets_go_after_it():
    n = 80
    tl = voice(n, hum=(6, 40), hits=((20, 0.6), (35, 1.0)))
    p = water.whip_plan(tl, track_of(n))
    last = int(np.flatnonzero(water.sounding(tl))[-1])  # the last hit dying away
    assert last > 40 and (p.start, p.end) == (6, last + 1 + water.LINGER)
    assert p.cracks == [(20, 0.6), (35, 1.0)]
    p = water.whip_plan(voice(n, hum=(30, 31), hits=((30, 1.0),)), track_of(n))
    assert p.start == 30 - water.WHIP_LEAD  # a stream out before the hit that opens the voice


# ---- where it is ----------------------------------------------------------------------------------------------------

def test_the_orb_floats_over_the_palm_sized_to_the_hand_and_holds_steady():
    n = 70
    track = track_of(n)
    hand, sc = scenes("orb", voice(n, hum=(4, 60), hits=()), track)
    held = [s_.held for s_ in sc[30:60]]
    assert all(h is not None for h in held)
    x, y, z, r, w = np.array(held).T
    assert np.allclose(r, water.ORB_R * S, rtol=0.02) and np.all(w == 1.0)
    p = np.array(track["xy"][40])
    assert y.mean() < p[1] - 0.3 * S and abs(x.mean() - p[0]) < 0.1 * S  # over the palm (tipped up: above it)...
    assert (z - r).min() > hand.depth(40, np.float32([p[0]]), np.float32([p[1]]))[0]  # ... and in front of it
    step = np.hypot(np.diff(x), np.diff(y))
    assert step.max() < 0.02 * S  # a still hand's orb only bobs: no jitter


def test_the_orb_rides_a_moving_hand_with_the_lag_of_a_mass_of_water():
    n = 60
    track = track_of(n, xy=lambda t: (30.0 + 1.5 * t, 130.0))
    hand, sc = scenes("orb", voice(n, hum=(4, 58), hits=()), track)
    lag = [track["xy"][k][0] - sc[k].held[0] for k in range(30, 50)]
    assert 0.5 < np.mean(lag) < 0.3 * S  # it trails the hand a little, and stays over it


def test_a_thrown_orb_flies_into_the_room_and_bursts_flat_on_its_wall(monkeypatch):
    monkeypatch.setattr(water, "ROOM_DEPTH", 0.6)  # the wall three hands behind the palm, inside this small frame
    n = 70
    tl = voice(n, hum=(4, 40), hits=((40, 1.0),), aim=0.0)  # thrown up, and away into the room
    hand, sc = scenes("orb", tl, track_of(n))
    k = next(k for k in range(40, n) if any(b.flat < 1.0 for b in sc[k].blobs))  # it hits the wall
    assert 40 + 2 < k < 40 + 12
    ball = [max(sc[j].blobs, key=lambda b: b.r) for j in range(40, k)]
    assert ball[-1].y < ball[0].y - 0.5 * S and ball[-1].r < 0.8 * ball[0].r  # up along the aim, and away
    assert all(b.move for b in ball[1:])  # blurred by its flight
    splat = max(sc[k].blobs, key=lambda b: b.r)
    assert splat.r > ball[-1].r and splat.flat < 0.5  # flattened against the wall...
    burst = [b for b in sc[k + 1].blobs if b.free and np.hypot(b.x - splat.x, b.y - splat.y) < 3 * splat.r]
    r = np.array([b.r for b in burst])
    assert len(burst) > 20 and r.max() > 2.5 * np.percentile(r, 10)  # ... flying apart in pieces of many sizes...
    assert np.mean([abs(b.z - splat.z) < 0.2 * S for b in burst]) > 0.9  # ... out along the wall
    assert any(not w[6] for w in sc[k + 3].wet)  # it leaves a wet patch on the wall, and drips run down it
    drips = [b for b in sc[k + 8].blobs if b.free and b.flat < 1.0]
    assert drips and all(b.vy > 0 for b in drips)


def test_an_orb_thrown_out_of_the_picture_does_not_burst_in_the_air():
    n = 60
    tl = voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0)  # to the right, out of this narrow frame
    hand, sc = scenes("orb", tl, track_of(n))
    xs = [max(b.x for b in sc[k].blobs) for k in range(40, 44)]
    assert xs[-1] - xs[0] > 1.0 * S  # away along the aim...
    assert not any(b.flat < 1.0 or 5 in [l for l, _, _ in b.modes] for k in range(40, n) for b in sc[k].blobs)
    assert max(len(sc[k].blobs) for k in range(42, n)) < 10  # ... and gone, with no splash in mid-air


def test_a_splash_flies_on_along_the_motion_in_drops_of_many_sizes_and_falls():
    drops = water.Drops(None, (2000, 2000), S)
    rng = np.random.default_rng(1)
    v = np.array([12.0, 0.0, 0.0])
    water.splash(drops, 0, np.array([1000.0, 1000.0, 0.0]), 10 * S, v, 1.0, 10 * S, rng)
    d = drops.v[:, :2] - water.SPLASH_CARRY * v[:2]
    assert (d @ v[:2] / np.linalg.norm(v)).mean() > 0.3 * np.linalg.norm(d, axis=1).mean()  # on along it, no star
    r = drops.info[:, 0]
    assert r.max() > 15 * r.min() and np.median(r) < 0.1 * r.max()  # many small, a few big
    vy0 = drops.v[:, 1].copy()
    drops.step()
    assert np.all(drops.v[:, 1] > vy0 * water.DRAG)  # gravity from the first frame


def test_the_orbs_shade_comes_on_as_it_gathers_and_goes_as_it_is_thrown():
    n = 70
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), track_of(n))
    w = np.array([0.0 if s_.held is None else s_.held[4] for s_ in sc])
    plan = water.orb_plan(voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), track_of(n))
    assert w[plan.gather] < 0.1 and w[plan.formed] == 1.0 and w[40 - 1] == 1.0
    assert np.abs(np.diff(w)).max() < 0.4  # never all at once (it popped on with a few drops in the air)
    assert 0.0 < w[41] < 1.0 and w[40 + water.TAKEOFF + 2] == 0.0  # let go over a few frames, not in one


def test_the_light_on_the_palm_does_not_blink_when_the_orb_is_thrown():
    n = 60
    plate, track = wall(n), track_of(n)
    matte = matte_of(track)
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), track)
    room, noise = water.Room(plate), water.Noise(0)
    gain = []
    for k in range(34, 48):
        P = lin(plate[k])
        Q = P.copy()
        if sc[k].blobs:
            layer = water.shade(water.Field(sc[k].blobs, H, W), sc[k], k, hand, room, noise, water._pyramid(P),
                                matte[k], 0.0, 0)
            if layer is not None:
                water.caustic(Q, layer, k, hand, room, matte[k], sc[k].held)
        m = matte[k] > 0
        gain.append(float((Q[..., 1][m] / np.maximum(P[..., 1][m], 1e-4)).mean()))
    gain = np.array(gain)
    held = gain[:6]  # held, the orb shades the palm...
    assert np.ptp(held) < 0.01 and held.mean() < 0.995
    step = np.diff(gain[5:])  # ... and once thrown its shade leaves the palm over a few frames, with no on and off
    assert step.min() > -0.003 and step.max() < 0.6 * (1.0 - held.mean()) and gain[-1] > 0.998


def test_an_orb_let_fall_splashes_off_the_palm():
    n = 80
    hand, sc = scenes("orb", voice(n, hum=(4, 30), hits=()), track_of(n))
    rel = 30 + water.LINGER
    assert sc[rel - 1].held is not None
    burst = next(k for k in range(rel, n) if len(sc[k].blobs) > 20)
    assert burst - rel < 12  # it falls the gap onto the palm and splashes


def test_the_whip_trails_a_moving_hand_like_a_ribbon():
    n = 50
    tl = voice(n, hum=(2, 49), hits=())

    def spread(track):
        _, sc = scenes("whip", tl, track)
        return np.mean([[b.x for b in sc[k].blobs if b.w < 1.0][-1] - track["xy"][k][0] for k in range(30, 45)])

    still = spread(track_of(n))
    moving = spread(track_of(n, xy=lambda t: (10.0 + 4.0 * t, 130.0)))  # 4 px a frame to the right
    assert moving < still - 0.25 * S  # the stream's tail lags behind a hand that moves


def test_the_whip_is_a_stream_round_the_hand_sized_to_it():
    n = 40
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    b = sc[30].blobs
    xy = np.array([[q.x, q.y] for q in b])
    reach = np.hypot(*(xy - track["xy"][30]).T)
    assert len(b) > 30 and 0.8 * S < reach.max() < 2.0 * S  # out round the hand, about a hand from it
    assert max(q.r for q in b) < 0.2 * S  # a stream, not a ball


def test_the_whip_thins_to_beads_at_its_tail_and_sheds_drops_when_it_lashes():
    n = 40
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=((20, 1.0),)), track)
    body = [b for b in sc[16].blobs if not b.free]
    tube = [b for b in body if b.w < 1.0]  # the stream's blobs, each weighted by its share of it
    assert len(tube) > 20 and max(b.r for b in tube[-8:]) < 0.5 * max(b.r for b in tube[:8])  # it tapers
    beads = [b for b in body if b.w == 1.0 and b.stream]
    assert beads and max(b.r for b in beads) < water.WHIP_R[0] * S  # its tail necks into beads
    assert any(b.free for k in range(21, 30) for b in sc[k].blobs)  # the lash flings drops off it
    assert max(np.hypot(*b.move) / (2.0 * b.r) for k in range(21, 26) for b in sc[k].blobs if b.move) > 0.7  # blurred


def test_the_whip_rises_out_of_a_frothing_pool_that_wets_the_palm():
    n = 40
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    pool = [b for b in sc[30].blobs if b.flat < 1.0]
    p = np.array(track["xy"][30])
    assert len(pool) == 1 and pool[0].r > 1.5 * water.WHIP_R[0] * S and pool[0].foam > 0.3  # wide, flat, churning...
    assert np.hypot(pool[0].x - p[0], pool[0].y - p[1]) < 0.2 * S  # ... on the palm (no ball over it: a test tube)
    assert any(w[6] for w in sc[30].wet)  # the palm round it wet
    arc = [b.stream[0] for b in sc[30].blobs if b.stream and b.w < 1.0]
    assert len(arc) > 20 and np.all(np.diff(arc) > 0)  # the stream knows where along itself each part is


# ---- what a frame's water is made of --------------------------------------------------------------------------------

def test_blobs_weighted_along_a_stream_sum_to_a_tube_of_their_radius():
    r = 8.0
    gap = water.WHIP_SPACING * r
    w = water._tube_weight((1.0, 0.0, 0.0, 1.0), r * water.SIG_K, np.array([1.0, 0.0]), gap)

    def half_width(weight):
        fld = water.Field([water.Blob(x, 96.0, 0.0, r, w=weight) for x in np.arange(10.0, 118.0, gap)], H, W)
        return (fld.f[:, 64 - fld.box[2]] > water.TAU).sum() / 2.0

    assert abs(half_width(w) - r) < 0.15 * r
    assert half_width(1.0) > 1.5 * r  # unweighted, a hose half as thick again


def test_free_pieces_do_not_fuse_with_each_other():
    r = 10.0
    pair = [(64.0 - 1.15 * r, 96.0), (64.0 + 1.15 * r, 96.0)]  # their outlines 0.3 r apart
    for free, bridged in ((False, True), (True, False)):
        fld = water.Field([water.Blob(x, y, 0.0, r, free=free) for x, y in pair], H, W)
        assert (fld.f[96 - fld.box[0], 64 - fld.box[2]] > water.TAU) == bridged


def test_a_fast_drop_is_blurred_along_its_motion_into_a_see_through_streak():
    r = 4.0
    shape, move = water._moving(r, np.array([40.0, 0.0]))
    assert np.allclose(shape, (1.0, 0.0, 0.0, 1.0)) and move == (min(water.SMEAR_MAX * 2 * r, 40 * water.SHUTTER), 0.0)
    plate = wall(1)

    def layer(mv):
        fld = water.Field([water.Blob(64.0, 96.0, 0.0, r, free=True, move=mv)], H, W)
        return water.shade(fld, water.Scene(wave=0.0), 0, _hand_at(-2.0 * S), water.Room(plate), water.Noise(0),
                           water._pyramid(lin(plate[0])), None, 0.0, 0)

    still, fast = layer(()), layer(move)
    assert abs(fast.alpha.sum() - still.alpha.sum()) < 0.03 * still.alpha.sum()  # as much water, spread out...
    assert fast.alpha.max() < 0.6 * still.alpha.max()  # ... seen through...
    cols = np.arange(fast.box[2], fast.box[3])[fast.alpha.max(0) > 0.02]
    rows = np.arange(fast.box[0], fast.box[1])[fast.alpha.max(1) > 0.02]
    assert np.ptp(cols) > 2 * r + 0.8 * move[0] and np.ptp(rows) < 2 * r + 4  # ... along its motion only


def test_the_room_is_mirrored_by_direction_alone():
    plate = wall(1)
    plate[0, : int(water.ENV_TOP * H)] = 128  # the upper room, plain
    room = water.Room(plate)
    up, down, level = room.reflect(0, np.array([0.0, 0.0, 0.6]), np.array([-0.9, 0.9, 0.0]))
    assert np.allclose(up, level, rtol=0.05) and np.all(down < 0.6 * up)  # what is below level, darker


# ---- how it looks ---------------------------------------------------------------------------------------------------

def _hand_at(depth: float, n: int = 1) -> water.Hand:
    """A palm facing the camera, its plane at ``depth`` px (towards the camera)."""
    return water.Hand(np.array([[64.0, 100.0]] * n), np.array([[0.0, 0.0, 1.0]] * n), np.array([[0.0, -1.0, 0.0]] * n),
                      np.full(n, S), np.array([[0.0, 0.0, depth]] * n), None)


def _one_drop(hand: water.Hand, matte, r: float = 0.4 * S) -> np.ndarray:
    plate = wall(1)
    sc = water.Scene([water.Blob(64.0, 100.0, 0.0, r)])
    return water.frame(0, plate[0], matte, sc, hand, water.Room(plate), water.Noise(0), 0.0, 0), plate[0]


def test_the_hand_hides_the_water_only_where_it_is_in_front_of_it():
    matte = np.full((H, W), 255, np.uint8)
    out, plate = _one_drop(_hand_at(+2.0 * S), matte)  # the hand nearer the camera than the water
    inside = (slice(90, 110), slice(54, 74))
    assert np.abs(out[inside].astype(int) - plate[inside]).max() <= 2
    out, plate = _one_drop(_hand_at(-2.0 * S), matte)  # the hand behind it
    assert np.abs(out[inside].astype(int) - plate[inside]).mean() > 8


def test_water_bends_the_plate_behind_it_with_no_cut_out_edge():
    out, plate = _one_drop(_hand_at(-2.0 * S), None)
    diff = np.abs(out.astype(int) - plate).max(-1)
    assert diff[95:105, 59:69].mean() > 10  # what is behind it, bent
    yy, xx = np.mgrid[0:H, 0:W]
    assert diff[np.hypot(xx - 64, yy - 100) > 1.3 * 0.4 * S].max() == 0  # nothing off it (no box round it)
    assert abs(out[85:115, 49:79].mean() - plate[85:115, 49:79].mean()) < 25  # clear: no darker or brighter as a whole
    fld = water.Field([water.Blob(64.0, 100.0, 0.0, 0.4 * S)], H, W)
    plate = wall(1)
    layer = water.shade(fld, water.Scene(wave=0.0), 0, _hand_at(-2.0 * S), water.Room(plate), water.Noise(0),
                        water._pyramid(lin(plate[0])), None, 0.0, 0)
    row = layer.alpha[100 - layer.box[0]]
    assert ((row > 0.05) & (row < 0.95)).sum() >= 4  # the outline is soft over pixels on each side, not stepped


def test_the_orb_lights_the_palm_under_it():
    n = 50
    plate, track = wall(n), track_of(n)
    matte = matte_of(track)
    out = water.render(plate, track, matte, voice(n, hum=(4, 49), hits=()), "orb")
    k = 35
    on = (matte[k] > 0) & (np.abs(out[k].astype(int) - plate[k]).max(-1) > 0)
    hand, sc = scenes("orb", voice(n, hum=(4, 49), hits=()), track)
    x, y, _, r, _ = sc[k].held
    yy, xx = np.mgrid[0:H, 0:W]
    off_orb = np.hypot(xx - x, yy - y) > 1.3 * r
    assert (on & off_orb).sum() > 50  # its focus and shade on the palm, outside the orb itself
    wall_far = (matte[k] == 0) & (np.hypot(xx - x, yy - y) > 2.0 * r)
    assert np.array_equal(out[k][wall_far], plate[k][wall_far])  # and none on the wall off it


def test_a_small_drop_is_clear_with_a_bright_glint_not_a_grey_bead():
    plate = wall(1)
    big = 400.0  # a hand this size, so a small drop is a few px across
    hand = water.Hand(np.array([[64.0, 100.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, -1.0, 0.0]]),
                      np.array([big]), np.array([[0.0, 0.0, -2.0 * big]]), None)
    r = 0.015 * big
    sc = water.Scene([water.Blob(64.0, 100.0, 0.0, r, free=True)], wave=0.0)
    out = water.frame(0, plate[0], None, sc, hand, water.Room(plate), water.Noise(0), 0.0, 0).astype(float)
    yy, xx = np.mgrid[0:H, 0:W]
    on = np.hypot(xx - 64, yy - 100) < r
    p = plate[0].astype(float)
    assert abs(out[on].mean() - p[on].mean()) < 0.2 * p[on].mean()  # what is behind it seen through it, not grey
    assert out[on].max() > p[on].max() + 25  # and a glint


def test_the_light_the_orb_focuses_wobbles_with_it_in_a_shifting_net():
    q, R, dh = 2, 40.0, 20.0
    L = np.array([0.0, -0.5, 0.86]) / np.linalg.norm([0.0, -0.5, 0.86])
    plain = water._spot(80, 80, q, 0, 0, 80.0, 80.0, R, dh, L)
    core = plain > 0.5 * plain.max()
    noise = water.Noise(0)
    nets = [water._spot(80, 80, q, 0, 0, 80.0, 80.0, R, dh, L, (), noise, k) for k in (0, 1)]
    assert abs(nets[0][core].mean() - plain[core].mean()) < 0.15 * plain[core].mean()  # as much light...
    assert np.std(nets[0][core] / plain[core]) > 0.05  # ... in a net of brighter lines...
    assert np.abs(nets[1] - nets[0])[core].mean() > 0.005 * plain.max()  # ... that shifts from frame to frame
    lobed = water._spot(80, 80, q, 0, 0, 80.0, 80.0, R, dh, L, ((3, 0.05, 0.0),))
    assert np.abs(lobed - plain)[core].max() > 0.05 * plain.max()  # and wobbling with the orb
