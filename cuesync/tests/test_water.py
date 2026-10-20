"""cuesync.water: water bent from an open palm, made from the plate (the orb and the whip). Built on a synthetic shot:
a textured wall, a palm with no landmarks (taken to face the camera, tipped up), still or moving, its matte, and a
voice's timeline."""

import math

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


def test_a_thrown_orb_flies_into_the_room_and_bursts_on_its_wall_where_the_water_stays(monkeypatch):
    monkeypatch.setattr(water, "ROOM_DEPTH", 0.6)  # the wall three hands behind the palm, inside this small frame
    n = 70
    tl = voice(n, hum=(4, 40), hits=((40, 1.0),), aim=0.0)  # thrown up, and away into the room
    hand, sc = scenes("orb", tl, track_of(n))
    k = next(k for k in range(40, n) if sc[k].film)  # it hits the wall
    assert 40 + 2 < k < 40 + 12
    ball = [max(sc[j].blobs, key=lambda b: b.r) for j in range(40, k)]
    assert ball[-1].y < ball[0].y - 0.5 * S and ball[-1].r < 0.8 * ball[0].r  # up along the aim, and away
    assert all(b.move for b in ball[1:])  # blurred by its flight
    sheets = [next(it for it in sc[j].film if isinstance(it, water.Sheet)) for j in range(k, n)]
    r = np.array([q.r for q in sheets])
    assert r[0] > 1.2 * ball[-1].r and np.all(np.diff(r) >= 0) and r[8] > 1.25 * r[0]  # it lands squashed and
    # spreads over the wall...
    look, wet = np.array([q.look for q in sheets]), np.array([q.wet for q in sheets])
    assert look[0] == 1.0 and look[-1] == 0.0 and np.abs(np.diff(look)).max() < 0.4  # ... draining over frames...
    assert np.all(np.diff(wet) >= 0) and 0.0 < wet[0] <= 0.5 and wet[-1] == 1.0  # ... into the wet patch it leaves
    froth = np.array([q.froth for q in sheets])
    assert sheets[0].dome == 1.0 and froth[0] > 0.4  # it lands as a churned body...
    assert froth[1] <= 0.3 * froth[0] and froth[2] == 0.0  # ... whose froth is gone in two frames (a milky fog)
    sag = np.array([q.sag for q in sheets])
    assert sag[0] == 0.0 and sag[-1] == 1.0 and np.all(np.diff(sag) >= 0)  # the patch's lower edge runs down...
    assert sheets[-1].runs and all(math.sin(a) > 0.2 for a, _, _ in sheets[-1].runs)  # ... in runs along its bottom
    fingers = sheets[water.FINGER_DRAIN[0]].fingers  # at their longest
    lengths = np.array([ln for _, ln, _, _ in fingers])
    down = np.array([math.sin(a) > -0.3 for a, *_ in fingers])
    assert len(fingers) >= water.WALL_FINGERS[0] and (lengths[down] >= 0.25).sum() >= 2  # jets long enough to see,
    # down and out sideways (most too short to see, the splash was a disc)...
    assert lengths.max() > 2.0 * lengths.min()  # ... of different lengths (alike, a starburst)...
    assert all(ln <= water.FINGER_UP + 1e-9 for a, ln, _, _ in fingers if -math.sin(a) > 0.99)  # ... none far up...
    assert all(water.FINGER_BEAD[0] <= bd <= water.FINGER_BEAD[1] for *_, bd in fingers)  # ... each ending in a bead
    assert all(ln <= water.FINGER_DRAIN[1] + 1e-6 for _, ln, _, _ in sheets[-1].fingers)  # ... and drained into it, so
    # no one finger is left long on the stain (a trunk)
    beads = [[it for it in sc[j].film if isinstance(it, water.Bead)] for j in range(k, n)]
    rb = np.array([b.r for b in beads[3]])
    assert len(rb) > 10 and rb.max() > 2.5 * np.percentile(rb, 10)  # drops of many sizes lie on the wall round it
    held = np.array([sum(b.a for b in bs) for bs in beads])
    assert np.abs(np.diff(held[3:])).max() < 0.25 * held.max()  # flung out at the hit, they go a few at a time
    big = sorted(beads[-1], key=lambda b: -b.r)[:3]
    assert all(b.vy > 0 for b in big)  # the big ones run down the wall...
    trails = [it for it in sc[n - 1].film if isinstance(it, water.Trail)]
    assert trails and all(t.y1 > t.y0 for t in trails)  # ... leaving wet trails
    assert not any(b.free and b.r > 0.05 * S for j in range(k, k + 4) for b in sc[j].blobs)  # no lumps in the air


def test_the_water_on_the_wall_darkens_it_and_its_drops_glint_but_none_of_it_is_on_the_hand():
    plate = np.full((1, H, W, 3), 150, np.uint8)
    room, noise = water.Room(plate), water.Noise(0)
    sheet = water.Sheet(64.0, 70.0, 25.0, ((0.5, 0.3, 0.05, 1.4), (2.5, 0.2, 0.05, 1.5)), ((2, 0.05, 0.0),), 3.0, 0.0,
                        0.0, 1.0)
    bead = water.Bead(64.0, 130.0, 6.0, 0.0, 0.0, 1.0)
    matte = np.zeros((H, W), np.uint8)
    matte[60:80, :20] = 255  # the hand, in front of the wall, over the sheet's left edge
    sc = water.Scene(film=[sheet, bead])
    out = water.frame(0, plate[0], matte, sc, _hand_at(0.0), room, noise, 0.0, 0).astype(int)
    assert out[70, 64].max() < 140  # the wet patch, darker
    assert out[70, 40:44].mean() < out[70, 52:56].mean()  # darker again at its rim
    assert np.array_equal(out[60:80, :8], plate[0, 60:80, :8])  # nothing on the hand
    assert np.array_equal(out[:30, :30], plate[0, :30, :30])  # nothing off it
    edge = out[70, 30:50, 1]
    assert ((edge > edge.min() + 2) & (edge < edge.max() - 2)).sum() >= 3  # its outline soft, not a stain's line
    drop = out[118:142, 52:76]
    assert drop.max() > 200 and drop.min() < 140  # a drop: a glint on it and a dark line along its foot
    assert out[134:136, 60:68].mean() < out[124:126, 60:68].mean() + 60  # the wall seen through it, not a white bead
    sagged = sheet._replace(fingers=(), sag=1.0, runs=((0.5 * math.pi, 0.1, 0.2),))  # a run down from its bottom
    out = water.frame(0, plate[0], None, water.Scene(film=[sagged]), _hand_at(0.0), room, noise, 0.0, 0).astype(int)
    assert out[98, 64].mean() < out[98, 54].mean() - 3  # its lower edge sags into it


def test_the_burst_on_the_wall_froths_out_along_its_jets_not_evenly_round():
    plate = np.full((1, H, W, 3), 120, np.uint8)
    room, noise = water.Room(plate), water.Noise(0)
    jets = ((0.5 * math.pi, 0.8, 0.08, 1.4), (0.5 * math.pi + 0.5, 0.5, 0.06, 1.4))  # both down
    sheet = water.Sheet(64.0, 70.0, 25.0, jets, ((2, 0.0, 0.0),), 6.0, 1.0, 1.0, 0.5, froth=0.8)
    out = water.frame(0, plate[0], None, water.Scene(film=[sheet]), _hand_at(0.0), room, noise, 0.0, 0).astype(int)
    down, up = out[88:94, 62:67].mean(), out[46:52, 62:67].mean()
    assert down > up + 5  # white out along its jets, not all round (a starburst)


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
    assert w[(plan.gather + plan.formed) // 2] < 0.3  # its light comes in as it forms (white hot half formed)
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
    b = [q for q in sc[30].blobs if not q.free]  # (not the drips falling off it)
    xy = np.array([[q.x, q.y] for q in b])
    reach = np.hypot(*(xy - track["xy"][30]).T)
    assert len(b) > 30 and 0.8 * S < reach.max() < 2.0 * S  # out round the hand, about a hand from it
    assert max(q.r for q in b) < 0.25 * S  # a stream, not a ball (flared a little where it rises out of its pool)


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


def test_the_whip_rises_churning_out_of_a_smooth_pool_that_wets_the_palm():
    n = 40
    track = track_of(n)
    hand, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    pool = sorted([b for b in sc[30].blobs if b.flat < 1.0 and not b.stream], key=lambda b: -b.r)
    main, parts = pool[0], pool[1:]
    p = np.array(track["xy"][30])
    assert main.r > 1.2 * water.WHIP_R[0] * S and main.flat <= 0.25  # a wide, low dome...
    assert len(parts) == water.WHIP_POOL_PARTS[0] and all(
        b.flat == main.flat and b.r < main.r and np.hypot(b.x - main.x, b.y - main.y) < 1.2 * main.r
        for b in parts)  # ...
    # an irregular puddle, flat pools run into it...
    assert main.foam < 0.2  # ... smooth (frothing all over, crumpled cellophane)...
    assert np.hypot(main.x - p[0], main.y - p[1]) < 0.2 * S  # ... on the palm (no ball over it: a test tube)
    tube = [b for b in sc[30].blobs if b.stream and b.w < 1.0]
    assert tube[0].foam > 0.4 > tube[len(tube) // 2].foam  # the stream churns where it rises out of it...
    assert tube[0].flat < 0.5 and tube[len(tube) // 2].flat == 1.0  # ... as flat as the pool there, not a bulb...
    nrm = hand.nrm[30]  # (on the palm itself: seen from the camera, it is squashed as the palm turns from it)
    squash = math.sqrt(nrm[2] ** 2 + water.WHIP_POOL[1] ** 2 * (1.0 - nrm[2] ** 2))
    disc = np.reshape(water._shape(nrm[:2] / np.hypot(*nrm[:2]), squash, 1.0), (2, 2))
    sv = np.linalg.svd(np.linalg.inv(disc) @ np.reshape(main.shape, (2, 2)), compute_uv=False)
    assert sv[0] < 1.6 * sv[1]  # ... and no pointed head on it (drawn out along the stream, a snake's)
    assert sc[30].wet  # the palm round it wet
    arc = [b.stream[0] for b in tube]
    assert len(arc) > 20 and np.all(np.diff(arc) > 0)  # the stream knows where along itself each part is


def test_the_pool_moves_with_the_palm_and_does_not_turn():
    n = 30
    track = track_of(n, xy=lambda t: (30.0 + 2.0 * t, 130.0))
    _, sc = scenes("whip", voice(n, hum=(2, 29), hits=()), track)
    pool = [b for b in sc[20].blobs if b.flat < 1.0 and not b.stream]
    assert pool and all(abs(b.vx - 2.0) < 1e-6 and b.vy == 0.0 and b.spin == 0.0 for b in pool)  # (still, while
    # the palm moved under it, or turning, the patterns its flow carries were wound into rings: a whorl)


def test_the_stream_fades_where_it_hangs_off_the_hand_but_not_while_it_lashes():
    n = 40
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=((30, 1.0),)), track)
    p = np.array(track["xy"][25])
    tube = [b for b in sc[25].blobs if b.stream and b.w < 1.0]
    off = [b.seen for b in tube if np.hypot(b.x - p[0], b.y - p[1]) > water.OFF_HAND[1][1] * S]
    on = [b.seen for b in tube if np.hypot(b.x - p[0], b.y - p[1]) < water.OFF_HAND[1][0] * S]
    assert off and on and max(off) < 0.5 and min(on) == 1.0  # off the hand, faint (a handle, a bangle)...
    lash = [b.seen for b in sc[32].blobs if b.stream and b.w < 1.0]
    assert min(lash) > 0.9  # ... flung out, whole


def test_the_wet_skin_round_the_stream_does_not_flicker():
    n = 60
    plate, track = wall(n), track_of(n)
    matte = matte_of(track)
    out = water.render(plate, track, matte, voice(n, hum=(2, 59), hits=()), "whip", seed=1)
    d = (out[20:58].astype(float) - plate[20:58]).mean(-1)
    every_other = np.abs((d * np.where(np.arange(len(d)) % 2 == 0, 1.0, -1.0)[:, None, None]).mean(0))
    assert every_other[matte[0] > 0].mean() < 0.4  # no swing from frame to frame over the hand (wetted on every
    # other frame, its skin buzzed: 0.52 levels here)


def test_the_streams_bulges_are_irregular_not_ribs():
    n = 40
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    gaps, swell = [], []
    for k in (20, 26, 32, 38):
        tube = [b for b in sc[k].blobs if b.stream and b.w < 1.0]
        arc = np.array([b.stream[0] for b in tube])
        r = np.array([b.r for b in tube])
        mid = slice(len(r) // 6, len(r) * 2 // 3)  # past its root, short of its fraying tail
        arc, r = arc[mid], r[mid]
        d = r / np.polyval(np.polyfit(arc, r, 2), arc) - 1.0  # its taper taken out
        swell.append(np.std(d))
        peaks = arc[1:-1][(d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > 0.0)]
        gaps += list(np.diff(peaks) / np.median(r))
    assert 0.05 < np.mean(swell) < 0.4  # it swells and thins along its length...
    assert np.std(gaps) > 0.25 * np.mean(gaps)  # ... at no set spacing...
    assert np.mean(gaps) > 4.0  # ... far apart: bulges, not ribs


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


def test_a_fast_drop_is_blurred_along_its_motion_into_a_see_through_streak(monkeypatch):
    r = 4.0
    shape, move = water._moving(r, np.array([40.0, 0.0]))
    assert np.allclose(shape, (1.0, 0.0, 0.0, 1.0)) and move == (min(water.SMEAR_MAX * 2 * r, 40 * water.SHUTTER), 0.0)
    plate = wall(1)

    def layer(mv):
        fld = water.Field([water.Blob(64.0, 96.0, 0.0, r, free=True, move=mv)], H, W)
        return water.shade(fld, water.Scene(wave=0.0), 0, _hand_at(-2.0 * S), water.Room(plate), water.Noise(0),
                           water._pyramid(lin(plate[0])), None, 0.0, 0)

    kept = layer(move)
    monkeypatch.setattr(water, "BLUR_KEEP", (0.0, 0.0, 0.0))  # spread evenly over the exposure
    still, fast = layer(()), layer(move)
    assert abs(fast.alpha.sum() - still.alpha.sum()) < 0.03 * still.alpha.sum()  # as much water, spread out...
    assert fast.alpha.max() < 0.6 * still.alpha.max()  # ... seen through...
    cols = np.arange(fast.box[2], fast.box[3])[fast.alpha.max(0) > 0.02]
    rows = np.arange(fast.box[0], fast.box[1])[fast.alpha.max(1) > 0.02]
    assert np.ptp(cols) > 2 * r + 0.6 * move[0] and np.ptp(rows) < 2 * r + 4  # ... along its motion only (its ends
    # fading as the shutter opens and closes)
    assert kept.box == fast.box
    lum = (kept.colour - fast.colour).max(-1).max(0)  # down each column, the most light kept over the even spread
    on = np.flatnonzero(fast.alpha.max(0) > 0.05)
    third = len(on) // 3
    front, mid, back = lum[on[-third:]].mean(), lum[on[third:-third]].mean(), lum[on[:third]].mean()
    assert mid > 0.02 and max(front, back) < mid  # its own light, froth and glint, keeps along its smear, evenly,
    # fading at both ends as the shutter opens and closes (kept most at its front, a lash's streaks ended in a comb)


def _periodic(g: np.ndarray) -> float:
    """The strongest regular pattern finer than a few px in ``g`` (2D): the peak of the windowed spectrum of what a
    2 px blur takes out of it, as its sinusoid's amplitude against the mean."""
    h = g - cv2.GaussianBlur(g, (0, 0), 2.0)
    F = np.abs(np.fft.fft2(h * np.outer(np.hanning(h.shape[0]), np.hanning(h.shape[1]))))
    F[0, 0] = 0.0
    return float(2.0 * F.max() / (0.25 * h.size) / max(float(g.mean()), 1e-9))


@pytest.mark.parametrize("move", [(2.3, 1.4), (0.9, -2.7), (90.0, 40.0)])
def test_the_motion_blur_draws_no_regular_pattern(move):
    """Water moving alike is blurred alike: a plain sheet drifting (the held orb) or flung (a lash) stays plain where
    its whole smear covers it, with no peak in its spectrum (its taps shifted pixel by pixel by a patterned share of a
    step drew a screen-door hatch on slow water and evenly spaced bars across a lash)."""
    n = 240
    C, A = np.full((n, n, 3), 0.5, np.float32), np.ones((n, n), np.float32)
    mx, my = np.full((n, n), move[0], np.float32), np.full((n, n), move[1], np.float32)
    box, C, A, _ = water._motion_blur(C, A, mx, my, (100, 100 + n, 100, 100 + n), 2000, 2000)
    y0, x0 = 100 - box[0], 100 - box[2]
    hy, hx = int(abs(move[1]) / 2 + 6), int(abs(move[0]) / 2 + 6)
    for img in (A, C[..., 1]):
        g = img[y0 + hy:y0 + n - hy, x0 + hx:x0 + n - hx]
        assert _periodic(g) < 1e-3 and g.std() < 1e-3 * g.mean()


def _smooth_plate(n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(3)
    g = cv2.GaussianBlur(rng.uniform(60, 200, (n, n)).astype(np.float32), (0, 0), 6.0)
    return np.clip(np.repeat(g[..., None], 3, -1) * np.float32([1.0, 0.92, 0.82]), 0, 255).astype(np.uint8)[None]


def _big_hand(n: int = 1, s: float = 160.0) -> water.Hand:
    return water.Hand(np.array([[128.0, 128.0]] * n), np.array([[0.0, 0.0, 1.0]] * n),
                      np.array([[0.0, -1.0, 0.0]] * n), np.full(n, s), np.array([[0.0, 0.0, -2.0 * s]] * n), None)


def test_a_drifting_orb_shows_no_screen_door_hatch():
    plate = _smooth_plate()
    hand, room, noise = _big_hand(), water.Room(plate), water.Noise(0)
    r = 0.45 * 160.0

    def pattern(move):
        sc = water.Scene([water.Blob(128.0, 128.0, 0.0, r, move=move)], wave=water.WAVE[0])
        out = water.frame(0, plate[0], None, sc, hand, room, noise, 0.0, 0).astype(np.float32)
        e = int(0.5 * r)
        return _periodic(cv2.cvtColor(out[128 - e:128 + e, 128 - e:128 + e], cv2.COLOR_RGB2GRAY))

    assert pattern((2.3, 1.4)) < 1.3 * pattern(())  # no finer regular pattern than the same orb still shows


def test_a_lash_tears_into_streams_of_their_own_lengths_and_sweeps_a_sheet():
    w, d = 240, 90.0  # a strip of water flung down across its length
    C, A = np.full((30, w, 3), 0.3, np.float32), np.ones((30, w), np.float32)
    own = np.zeros((30, w, 4), np.float32)
    own[..., 0], own[..., 2], own[..., 3] = 0.5, 1.0, 6.0  # its own light, its sheet, and that times its thickness
    mx, my = np.zeros((30, w), np.float32), np.full((30, w), d, np.float32)

    def ends(rag):  # how far down the frame each column of its smear reaches, and the sheet it sweeps
        box, _, a, sheet = water._motion_blur(C, A, mx, my, (300, 330, 300, 300 + w), 1000, 1000, own,
                                              np.ones(3, np.float32), rag, 0)
        cols = range(300 + 10 - box[2], 300 + w - 10 - box[2])
        return np.array([box[0] + np.flatnonzero(a[:, j] > 0.05).max() for j in cols]), sheet

    alike, _ = ends(None)
    torn, sheet = ends(np.ones((30, w), np.float32))
    assert np.ptp(alike) <= 1  # moving alike, its streams end in a line...
    assert np.std(torn) > 0.03 * d  # ... torn, each its own length...
    assert np.abs(np.diff(torn)).mean() < 0.15 * np.std(torn)  # ... neighbours alike: a ragged curve (each streak its
    # own length, a row of teeth)
    on = sheet[..., 0] > 0.5
    assert sheet is not None and on.mean() > 0.5  # and it sweeps a sheet, as thick as the water that passed
    assert abs(np.median(sheet[..., 1][on]) - 6.0) < 0.1


def test_the_sheet_a_lash_sweeps_bends_what_is_behind_it_and_catches_the_light_along_its_edges():
    sheet = np.zeros((120, 120, 2), np.float32)
    sheet[20:100, 20:100] = (0.85, 8.0)
    layer = water.Layer((40, 160, 40, 160), np.zeros((120, 120, 3), np.float32), np.zeros((120, 120), np.float32),
                        np.zeros((120, 120), np.float32), np.zeros((120, 120), np.float32),
                        np.zeros((120, 120), np.float32), sheet)
    matte = np.zeros((200, 200), np.uint8)
    matte[:, 100:] = 255  # the hand behind its right half
    room = water.Room(np.full((1, 200, 200, 3), 160, np.uint8))

    def change(plate):
        P = lin(plate)
        Q = P.copy()
        water._sheet(Q, layer, matte, room, water.Noise(0), 0)
        return np.abs(Q - P).max(-1)

    dark = change(np.full((200, 200, 3), 20, np.uint8))
    assert dark[:40].max() == 0 and dark[:, :40].max() == 0  # nothing off it
    assert dark[62:140, 60:64].mean() > 3.0 * dark[90:110, 76:84].mean()  # its edges catch the light, its body clear
    assert dark[62:140, 136:140].mean() < 0.6 * dark[62:140, 60:64].mean()  # over the hand less: the skin keeps its
    # colour
    barred = np.full((200, 200, 3), 20, np.uint8)
    barred[:, ::8] = 200
    bent = change(barred)
    assert bent[62:140, 62:72].mean() > 3.0 * dark[62:140, 62:72].mean()  # and it bends what is behind it


def test_the_held_orbs_highlight_is_a_window_that_holds_its_shape():
    n = 8
    plate = np.repeat(_smooth_plate(), n, 0)
    hand, room, noise = _big_hand(n), water.Room(plate), water.Noise(0)
    lit = []
    for k in range(n):  # its waves running over it at the voice's loudest, frame by frame
        sc = water.Scene([water.Blob(128.0, 140.0, 0.0, 72.0, modes=((3, 0.03, 0.4), (4, 0.012, -0.3)))],
                         wave=water.WAVE[0] + water.WAVE[1])
        lit.append(water.frame(k, plate[k], None, sc, hand, room, noise, 0.0, 0).min(-1) >= 245)
    assert lit[0].sum() > 8
    assert all((m & lit[0]).sum() > 0.8 * (m | lit[0]).sum() for m in lit)  # the same window on every frame (moved
    # by the waves, a cloud)


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


def test_the_light_the_orb_focuses_wobbles_with_it_in_the_net_its_moving_surface_draws():
    q, R, dh = 2, 40.0, 20.0
    L = np.array([0.0, -0.5, 0.86]) / np.linalg.norm([0.0, -0.5, 0.86])
    plain, none = water._spot(80, 80, q, 0, 0, 80.0, 80.0, R, dh, L)
    assert none is None
    core = plain > 0.5 * plain.max()

    def full(m):  # a mask on the grid of q px cells, at the picture's px (where the net is)
        return cv2.resize(m.astype(np.uint8), (80 * q, 80 * q), interpolation=cv2.INTER_NEAREST) > 0

    plate = wall(1)
    noise = water.Noise(0)
    rays = []
    for k, ph in ((0, 0.0), (1, 0.6)):  # its wobble turning from frame to frame
        fld = water.Field([water.Blob(80.0, 80.0, 0.0, R, modes=((3, 0.03, ph),))], H, W)
        layer = water.shade(fld, water.Scene(), k, _hand_at(-2.0 * S, 2), water.Room(np.repeat(plate, 2, 0)), noise,
                            water._pyramid(lin(plate[0])), None, 0.0, 0)
        rays.append(water._rays(layer, 80.0, 80.0, R, dh, L, noise, k, S))
    nets = [water._spot(80, 80, q, 0, 0, 80.0, 80.0, R, dh, L, (), r)[1] for r in rays]
    assert nets[0].shape == (80 * q, 80 * q)  # at the picture's px
    c = nets[0][full(core)]
    assert abs(c.mean() - 1.0) < 0.15  # as much light...
    assert np.percentile(c, 99) > 1.5 and np.percentile(c, 10) < 0.9  # ... gathered into bright lines, dim between...
    assert np.all(nets[0][full(plain < 0.05 * plain.max())] == 1.0)  # ... in its core only...
    assert np.abs(nets[1] - nets[0])[full(core)].mean() > 0.03  # ... that shift with the surface from frame to frame
    lobed, _ = water._spot(80, 80, q, 0, 0, 80.0, 80.0, R, dh, L, ((3, 0.05, 0.0),))
    assert np.abs(lobed - plain)[core].max() > 0.05 * plain.max()  # and wobbling with the orb


def test_a_drip_off_the_orb_lands_on_the_palm_and_leaves_a_wet_dot_that_dries():
    n = 150
    _, sc = scenes("orb", voice(n, hum=(4, 145), hits=()), track_of(n))
    wet = [k for k in range(n) if sc[k].wet]
    assert wet and wet[0] + water.DRIP_WET[1] <= n
    strength = [sc[k].wet[0][5] for k in range(wet[0], wet[0] + water.DRIP_WET[1])]
    assert np.all(np.diff(strength) < 0)  # drying...
    assert strength[24] > 0.4 * strength[0]  # ... slowly: half of it still there a second on (gone in a third of a
    # second, it was barely seen)
    x, y, rx, ry, ang, st, phase, pin, soft = sc[wet[0]].wet[0]
    assert soft >= 0.5 and ry <= rx and pin > 0  # round on the palm, feathered, a glint on it (small, hard and uneven,
    # a mole)
