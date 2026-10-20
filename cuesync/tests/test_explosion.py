"""The explosion (cuesync.explosion): where it goes off, what it measures on any render, and a whole shot on a tiny
plate with a made-up render: the blast lights the frame on its strike and leaves the plate untouched before it."""

import numpy as np

from cuesync import explosion as X
from cuesync.elements import Timeline


def hand(n, w, h, at=(60.0, 60.0), scale=20.0, events=None):
    lm = [[at[0] + dx, at[1] + dy, 0.0] for dx, dy in
          [(0, 20)] + [(0, 5 - 3 * i) for i in range(20)]]  # wrist below, the knuckles and fingers above it: aim up
    return {"xy": [list(at)] * n, "dir": [[0.0, -1.0]] * n, "scale": [scale] * n, "landmarks": [lm] * n,
            "events": events or [], "src": ["mp"] * n}


def test_a_punch_throws_it_ahead_of_the_fist_and_harder_further():
    tr = hand(10, 200, 200, events=[{"kind": "extension", "frame": 5, "peak_speed": 2000.0}])
    p, aim, s = X.blast_point(tr, 5, "fist")
    lead = 60.0 + 5 - 3 * 19  # the topmost landmark
    assert np.allclose(aim, (0, -1)) and s == 20.0
    assert p[1] == lead - X.THROW[1] * 20.0  # (2000 px/s over a 20 px hand: the throw's upper bound)
    soft = hand(10, 200, 200, events=[{"kind": "extension", "frame": 5, "peak_speed": 100.0}])
    assert X.blast_point(soft, 5, "fist")[0][1] == lead - X.THROW[0] * 20.0
    palm = X.blast_point(tr, 5, "palm")[0]
    assert palm[1] > X.blast_point(tr, 5, "fist")[0][1]  # a palm's blooms just past the palm
    assert np.allclose(X.inside(np.array([-50.0, 500.0]), 200, 100), (20.0, 90.0))


def test_the_renders_blast_is_measured_not_assumed():
    el = np.full((30, 120, 160, 3), 5, np.uint8)
    for k in range(10, 30):
        r = 2 + 3 * (k - 10)
        yy, xx = np.mgrid[:120, :160]
        el[k][(xx - 100) ** 2 + (yy - 40) ** 2 <= r * r] = 220
    k, centre, r8 = X.render_blast(el, 5)
    assert k == 12 and np.allclose(centre, (100, 40), atol=4)  # (frames 10, 11: specks too small to count)
    assert abs(r8 - (2 + 3 * (k + X.GROW_FRAMES - 10))) < 4  # its radius GROW_FRAMES after it shows
    el[:10, 36:44, 96:104] = 240  # a spark the render holds where the blast will be, before it goes off
    assert X.render_blast(el, 5)[0] == k  # (measured from it, the ball played at 8 times its size)


def test_an_explosion_goes_off_on_its_hit_and_not_before():
    n, h, w = 24, 96, 128
    plate = np.full((n, h, w, 3), 90, np.uint8)
    el = np.full((40, 96, 128, 3), 4, np.uint8)
    yy, xx = np.mgrid[:96, :128]
    for k in range(5, 40):
        r = min(30, 4 + 4 * (k - 5))
        el[k][(xx - 64) ** 2 + (yy - 48) ** 2 <= r * r] = (255, 160, 40) if k < 25 else (40, 38, 36)
    tr = hand(n, w, h, at=(64.0, 70.0), scale=12.0)
    tl = Timeline(n=n, energy=np.zeros(n), hits=[8], sizes=[1.0], tap_frame=8)
    out = X.render(plate, tr, None, tl, "fist", seed=3, element=el)
    assert out.shape == plate.shape and out.dtype == np.uint8
    assert np.array_equal(out[:8], plate[:8])  # nothing before the strike
    assert out[8:10].astype(int).mean() > plate[8].mean() + 20  # the flash and the core on the strike
    assert np.array_equal(X.render(plate, tr, None, tl, "fist", seed=3, element=el), out)  # the same bytes again


def test_the_fireballs_light_is_its_own_image_spread_and_falling_off_round_it():
    h, w, r = 120, 200, 10
    yy, xx = np.mgrid[:h, :w]
    y = ((xx - 100) ** 2 + (yy - 60) ** 2 <= r * r).astype(np.float32)
    spread, norm = X.falloff((h, w), X.LIGHT_FALLOFF[0] * r, r, X.LIGHT_FALLOFF[1])
    L = X.blast_light(y, spread, norm)
    assert abs(L[60, 100] - 1.0) < 0.15 and 0.5 < L[60, 100 + r] < 0.8  # a ball of luminance 1 lights its middle 1
    past = L[60, 100 + 2 * r]
    assert 0.15 < past < 0.4 and abs(L[60, 100 - 2 * r] - past) < 0.02  # a wall a radius past its edge a quarter...
    assert 0.2 < L[60, 100 + 4 * r] / past < 0.45  # ...and twice as far a third of that, and so on (a pool of light)
    assert L.min() > 0.0
    hot, dim = X.light_colour(np.array([[0.9, 0.7, 0.3]])), X.light_colour(np.array([[0.2, 0.06, 0.01]]))
    assert hot[0] > hot[1] > hot[2] and dim[1] / dim[0] < hot[1] / hot[0]  # warm; a cooling ball deeper orange
    assert abs(float(hot @ np.array([0.2126, 0.7152, 0.0722])) - 1.0) < 1e-3


def disc(h, w, r, colour, floor=4):
    yy, xx = np.mgrid[:h, :w]
    E = np.full((h, w, 3), floor, np.uint8)
    E[(xx - w // 2) ** 2 + (yy - h // 2) ** 2 <= r * r] = colour
    return E, np.hypot(xx - w // 2, yy - h // 2)


def test_the_fireballs_rim_breaks_into_tongues_and_its_smokes_outer_third_is_eaten_into_crisp_wisps():
    E, d = disc(240, 240, 70, (255, 160, 40))
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    top = np.sin(ang) < 0

    def ring(img, rr):
        return img[(120 + rr * np.sin(ang)).round().astype(int), (120 + rr * np.cos(ang)).round().astype(int)] \
            .max(-1).astype(int)

    assert ring(E, 71).max() == 4 and ring(E, 68).min() == 255  # a hard round rim
    ball = E.copy()
    o = X.Outline(240, 240, 240, seed=1)
    maps = o.warp(E, 4, (0, 240, 0, 240), 10, 0.0, [])
    past = ring(E, 74) > 30
    assert 0.05 < past.mean() < 0.5  # tongues past it, here and there...
    assert (ring(E, 82)[top] > 30).any() and not (ring(E, 82)[~top] > 30).any()  # ...a fifth of its radius out on top
    assert np.array_equal(E[d < 56], ball[d < 56])  # (the ball inside them untouched)
    assert ring(E, 68).min() < 0.75 * ring(E, 68).max()  # the rim's band thinned between them: see-through
    near, desat = maps[:2]
    assert near.min() >= 0 and X.Outline.density(maps, (240, 240), 240)[d < 63].min() > 0.99  # the fire whole...
    assert desat.max() > 0.15  # ...its rim less saturated
    S, _ = disc(240, 240, 70, (40, 40, 40))
    q = X.Outline(240, 240, 240, seed=1)
    maps = q.warp(S, 4, (0, 240, 0, 240), 10, 0.0, [])
    assert maps[0].max() < 0.1 and maps[1].max() < 0.05  # smoke: no tongues, its colour kept
    dens = X.Outline.density(maps, (240, 240), 240)
    outer, inner = (d > 70 * 0.7) & (d < 70 * 0.98), d < 70 * 0.5
    assert dens[inner].min() > 0.95 and 0.4 < dens[outer].mean() < 0.85 and dens[outer].std() > 0.25  # eaten away...
    assert ((dens[outer] > 0.15) & (dens[outer] < 0.85)).mean() < 0.3  # ...into wisps cut sharp, not feathered
    rows = np.arange(240)[:, None]
    crown, base = (d < 66) & (rows < 120 - 35), (d < 66) & (rows > 120 + 35)
    later = {t: X.Outline.density(q.warp(S.copy(), 4, (0, 240, 0, 240), t, 0.0, []), (240, 240), 240) for t in (30, 60)}
    assert later[60][crown].mean() < 0.75 * dens[crown].mean()  # it thins from its crown as it climbs...
    assert later[30][base].mean() < 0.7 * dens[base].mean() < later[30][crown].mean()  # ...and its underside draws in
    # sooner (its lobes hung below it like ink drips)


def test_the_soot_fills_in_over_frames_and_keeps_its_billows():
    el = np.full((16, 96, 96, 3), 4, np.uint8)
    yy, xx = np.mgrid[:96, :96]
    d = np.hypot(xx - 48, yy - 48)
    for k in range(16):  # a lit ring of smoke round a dark middle, open to the outside until frame 6
        el[k][(d > 22) & (d < 34)] = 120
        if k < 6:
            el[k][(np.abs(xx - 48) < 5) & (yy > 48)] = 4
    raw = [X.soot_shape(f[::2, ::2], 4, 3, 1.0)[24, 24] for f in el]
    assert raw[5] < 0.2 and raw[6] > 0.8  # measured frame by frame, its middle filled at once when the gap closed...
    shapes = X.SootShapes(el, 4, 0, 1.0, 96)
    seen = [float(shapes.at(k)[24, 24]) for k in range(16)]
    steps = np.diff(seen)
    assert steps.max() <= X.SOOT_RAMP + 1e-6 and seen[-1] > 0.8  # ...it fills in over frames
    rng = np.random.default_rng(0)
    E = np.full((80, 80, 3), 4, np.uint8)
    E[10:70, 10:70] = (4 + rng.integers(0, 20, (60, 60, 1))).astype(np.uint8)  # a dark cloud with folds in it
    sil = np.zeros((80, 80), np.float32)
    sil[10:70, 10:70] = 1
    F = X.fill_soot(E, 4, (0, 80, 0, 80), sil, np.ones((80, 80), np.float32))[20:60, 20:60].astype(int)
    assert F.min() >= 4 + X.SOOT_FILL[1] * 251 - 1 and F.max(-1).std() > 3  # dense, its folds kept


def test_the_hand_keeps_its_skin_and_is_lit_on_the_rim_facing_the_fire():
    m = np.zeros((200, 200), np.float32)
    m[60:140, 60:140] = 1.0  # an arm's matte; the fire to its left
    rim = X.rim_light(m, (0.0, 100.0))
    assert rim[100, 60:70].max() > 0.3 and rim[100, 130:140].max() < 0.05 and rim[100, 90:110].max() < 0.05  # its
    # edge facing the fire, not its far side, nor its face
    from cuesync.transfer import gam, lin

    skin = np.full((200, 200, 3), (246, 162, 149), np.uint8)
    P = lin(skin).astype(np.float32)
    lit = P + np.float32(0.6) * np.array([2.8, 0.57, 0.02], np.float32)  # the fire's orange light, hard on it
    out = gam(X.warm_light(P, lit, m, rim)).astype(int)
    front, edge, wall = out[100, 100], out[100, 60 + int(np.argmax(rim[100, 60:70]))], out[100, 20]
    assert edge[1] > 162 + 10 and edge[1] <= X.SKIN_GR * edge[0] + 1 and edge[2] <= 149 + 6  # an orange-pink rim...
    assert (front - (246, 162, 149)) @ np.array([0.2126, 0.7152, 0.0722]) <= X.SKIN_LIFT[0] + 1  # ...its face hardly lit
    assert wall[1] - 162 <= X.LIGHT_WARM * (wall[0] - 246) + 2 or wall[0] >= 254  # the room's light kept warm


def test_the_fireball_rolls_up_lights_the_room_while_it_burns_and_not_once_it_is_soot():
    n, h, w = 70, 192, 192
    plate = np.full((n, h, w, 3), 120, np.uint8)
    el = np.full((70, 96, 128, 3), 4, np.uint8)
    yy, xx = np.mgrid[:96, :128]
    for k in range(5, 70):
        el[k][(xx - 64) ** 2 + (yy - 48) ** 2 <= min(30, 4 + 4 * (k - 5)) ** 2] = (255, 160, 40) if k < 22 else (40, 38, 36)
    tr = hand(n, w, h, at=(96.0, 160.0), scale=12.0)
    tl = Timeline(n=n, energy=np.zeros(n), hits=[10], sizes=[1.0], tap_frame=10)
    out = X.render(plate, tr, None, tl, "fist", seed=3, element=el).astype(int)
    o = X.inside(X.blast_point(tr, 10, "fist")[0], w, h)
    side = out[:, int(o[1]) - 6:int(o[1]) + 6, 4:14].mean((1, 2)) - 120  # the wall well beside the ball
    lit = side[15]  # (the strike + 5)
    assert lit.mean() > 10 and lit[0] > lit[1] > lit[2]  # lit, warm
    assert np.abs(side[60]).max() < 3  # the fire burnt out to soot: no light
    dark = np.clip(120 - out.mean(-1), 0, None)  # the smoke, where it darkens the wall
    rows = np.arange(h)[:, None]

    def middle(t):
        return float((dark[t] * rows).sum() / dark[t].sum())

    assert middle(56) < middle(41) - 2 < middle(26) - 4  # its smoke rises, and keeps rising
