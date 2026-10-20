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


def test_the_fireballs_rim_breaks_into_tongues_and_its_smokes_outer_third_into_wisps():
    E, d = disc(240, 240, 70, (255, 160, 40))
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)

    def ring(img, rr):
        return img[(120 + rr * np.sin(ang)).round().astype(int), (120 + rr * np.cos(ang)).round().astype(int)]             .max(-1).astype(int)

    assert ring(E, 71).max() == 4 and ring(E, 68).min() == 255  # a hard round rim
    o = X.Outline(240, 240, 240, seed=1)
    near, desat, dens, _ = o.warp(E, 4, (0, 240, 0, 240), 10, 0.0, [])
    assert 0.1 < (ring(E, 71) > 30).mean() < 0.5 and (ring(E, 72) > 30).any()  # tongues past it, here and there
    assert ring(E, 68).min() < 0.75 * ring(E, 68).max()  # the rim's band thinned between them: see-through
    dq = d[2::4, 2::4]  # (the maps are at a quarter size)
    assert near[dq < 63].min() > 0.9 and dens[dq < 63].min() > 0.99 and desat.max() > 0.25  # its rim less saturated
    S, _ = disc(240, 240, 70, (40, 40, 40))
    q = X.Outline(240, 240, 240, seed=1)
    near, desat, dens, outer3 = q.warp(S, 4, (0, 240, 0, 240), 10, 0.0, [])
    outer, inner = (dq > 70 * 0.7) & (dq < 70 * 0.95), dq < 70 * 0.4
    assert near.max() < 0.1 and desat.max() < 0.05  # smoke: no tongues, its colour kept
    assert dens[inner].min() > 0.95 and 0.35 < dens[outer].mean() < 0.65 and dens[outer].std() > 0.12  # wisps
    late = q.warp(S.copy(), 4, (0, 240, 0, 240), 60, 0.0, [])[2]
    rows = np.arange(2, 240, 4)[:, None]
    crown, base = (dq < 60) & (rows < 120 - 35), (dq < 60) & (rows > 120 + 35)
    assert late[crown].mean() < 0.75 * dens[crown].mean() and np.allclose(late[base], dens[base], atol=0.02)  # it
    # thins from its crown as it climbs


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

    assert middle(46) < middle(36) - 2 < middle(26) - 4  # its smoke rises, and keeps rising
