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


def fire_weight(E):
    from cuesync import composite as C

    return np.clip(C.fire_weight_map(C._black_floor_f(E, 4), X.PARAMS, E.shape[1], True), 0.0, 1.0).astype(np.float32)


def shown(a):
    """Display values (0-255) of linear ``a``."""
    return np.clip(a, 0, 1) ** (1 / 2.2) * 255


def burning(size=240, r=70, colour=(255, 160, 40), wall=0.35, since=10, burn=1.0, seed=1):
    """A ball of render fire on a lit wall through Fire.layer: (the result, linear; the composite it was drawn over;
    the room; distance from the ball's middle)."""
    E, d = disc(size, size, r, colour)
    wf = fire_weight(E)
    room = np.full((size, size, 3), wall, np.float32)
    Q = room.copy()
    Q[d < r + 6] *= 0.5  # (the composite's dull ball and a dark band round it: the sticker and its shadow)
    under = Q.copy()
    fire = X.Fire(size, size, size, seed)
    box = (0, size, 0, size)
    fire.layer(Q, room, E, wf, box, 4, burn, 0.0, since, fire.ball(wf, burn) if burn > 0 else None)
    return Q, under, room, d


def test_the_burning_ball_is_a_light_with_one_edge_and_tongues_licking_off_its_top():
    Q, under, room, d = burning()
    shown = (np.clip(Q, 0, 1) ** (1 / 2.2)) @ np.array([0.299, 0.587, 0.114])
    wall = float((room[0, 0] ** (1 / 2.2)) @ np.array([0.299, 0.587, 0.114]))
    assert shown[d < 50].mean() > max(0.8, wall + 0.1)  # brighter than the wall it lights, near white...
    assert shown[d < 50].mean() > ((np.clip(under, 0, 1) ** (1 / 2.2)) @ np.array([0.299, 0.587, 0.114]))[d < 50].mean()
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    top = np.sin(ang) < -0.3

    def ring(rr):
        return shown[(120 + rr * np.sin(ang)).round().astype(int), (120 + rr * np.cos(ang)).round().astype(int)]

    assert ring(66).min() > wall  # ...its limb too (no dark rim, no pale line: one edge)
    assert (shown[d > 72] >= wall - 1.0 / 255).all()  # nothing round it darker than the wall (no drop shadow)
    lick = ring(72) > wall + 0.03
    assert 0.1 < lick[top].mean() < 0.9 and not lick[np.sin(ang) > 0.3].any()  # tongues off its top, here and there
    assert not (ring(70 + 0.35 * 70) > wall + 0.03).any()  # (under a third of its radius long)
    later = burning(since=13)[0]
    assert np.abs(later - Q)[(d > 72) & (d < 88)].max() > 0.05  # they come and go
    dead = burning(burn=0.0, colour=(170, 60, 10))[0]  # (its fire dim)
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert (dead @ lum)[(d > 72) & (d < 75)].mean() < 0.3 < 0.35 <= (Q @ lum)[(d > 72) & (d < 75)].mean()  # as it
    # dies the composite's soot round it comes back (a little lit by its glow)


def test_the_tongues_rise_off_the_ball_never_apart_from_it_and_are_its_own_fire():
    import cv2

    E, d = disc(240, 240, 70, (255, 160, 40))
    rng = np.random.default_rng(0)
    E[d <= 70] = np.clip(E[d <= 70] * rng.uniform(0.7, 1.0, (int((d <= 70).sum()), 1)), 0, 255).astype(np.uint8)
    wf = fire_weight(E)
    fire = X.Fire(240, 240, 1080, 1)  # (its px at 1080)
    A, _, mid, rf = fire.ball(wf)
    u = fire.exposed(E, 4)[1]
    tall = []
    for since in range(8, 20):
        a, col, at = fire.tongue_layer(A, u, mid, rf, (0, 0), since)
        drawn = np.zeros((240, 240), np.uint8)
        drawn[at] = a > 0.3
        assert drawn[121:].sum() == 0 and drawn.sum() > 30  # off its upper half
        n, _ = cv2.connectedComponents(drawn | (A > 0.5).astype(np.uint8))
        assert n == 2  # every tongue one piece with the ball: none floats free of it
        y = (col @ np.array([0.2126, 0.7152, 0.0722], np.float32))[a > 0.3]
        assert y.std() > 0.03  # its own fire, not one flat colour...
        rows = np.broadcast_to(np.arange(at[0].start, at[0].stop)[:, None], a.shape)[a > 0.3]
        warm = (col[..., 1] / np.maximum(col[..., 0], 1e-6))[a > 0.3]
        tips, roots = rows < np.percentile(rows, 30), rows > np.percentile(rows, 70)
        assert warm[tips].mean() < warm[roots].mean()  # ...cooling to their tips
        tall.append(drawn.sum(0))
    change = [np.abs(tall[i + 1].astype(int) - tall[i]).sum() for i in range(len(tall) - 1)]
    assert min(change) > 0  # they grow, thin and go, frame by frame
    yy, xx = np.mgrid[:240, :240]
    E[np.hypot(xx - 185, yy - 75) < 28] = (255, 160, 40)  # a lobe bulging out over the ball's side
    A, _, mid, rf = fire.ball(fire_weight(E))
    under = (np.maximum.accumulate(A > 0.5, axis=0)) & (A < 0.05)  # (below the ball in its column, outside it)
    for since in range(8, 20):
        t = fire.tongue_layer(A, fire.exposed(E, 4)[1], mid, rf, (0, 0), since)
        drawn = np.zeros((240, 240), np.float32)
        drawn[t[2]] = t[0]
        assert drawn[under].max() < 0.01  # nothing hangs under the lobe


def test_the_balls_glow_is_its_highlights_spread_wide_beside_white_hot_fire_none_beside_its_folds():
    yy, xx = np.mgrid[:240, :240]
    d = np.hypot(xx - 120, yy - 120)
    E = np.full((240, 240, 3), 4, np.uint8)
    E[(d <= 70) & (xx < 120)] = (255, 225, 120)  # white-hot on its left
    E[(d <= 70) & (xx >= 120)] = (200, 90, 20)  # a deep orange fold on its right
    wf = fire_weight(E)
    room = np.full((240, 240, 3), 0.35, np.float32)
    Q = room.copy()
    fire = X.Fire(240, 240, 1080, 1)
    fire.layer(Q, room, E, wf, (0, 240, 0, 240), 4, 1.0, 0.0, 10, fire.ball(wf, 1.0))
    s, wall = shown(Q) @ np.array([0.299, 0.587, 0.114]), float(shown(room[0, 0]) @ np.array([0.299, 0.587, 0.114]))
    band = (np.abs(yy - 120) < 30) & (d > 72) & (d < 76)
    left, right = s[band & (xx < 120)].mean() - wall, s[band & (xx > 120)].mean() - wall
    assert left > 3 and abs(right) < 0.5  # a glow beside its white-hot side, none beside its fold
    far = (np.abs(yy - 120) < 30) & (d > 100)
    assert np.abs(s[far] - wall).max() < 0.5  # (and it dies away within a few tens of px)


def test_the_dying_ball_is_its_fire_over_brown_black_soot_not_the_composites_relit_smoke():
    yy, xx = np.mgrid[:240, :240]
    d = np.hypot(xx - 120, yy - 120)
    E = np.full((240, 240, 3), 4, np.uint8)
    E[d <= 70] = (200, 80, 15)  # a dimming fire...
    folds = (d <= 70) & ((np.hypot(xx - 100, yy - 110) < 14) | (np.hypot(xx - 140, yy - 135) < 12))
    E[folds] = (60, 14, 6)  # ...with dark folds in it
    wf = fire_weight(E)
    room = np.full((240, 240, 3), 0.35, np.float32)
    Q = room.copy()
    Q[d <= 74] = (0.25, 0.03, 0.03)  # the composite's raspberry relit smoke
    fire = X.Fire(240, 240, 1080, 1)
    ball = fire.ball(wf, 0.0, E.max(-1) > 20)
    fire.layer(Q, room, E, wf, (0, 240, 0, 240), 4, 0.0, 1.0, 20, ball)
    f = shown(Q[folds & (d < 60)])
    assert (f[:, 2] - f[:, 1]).mean() < -15 and (f[:, 1] / f[:, 0]).mean() < 0.65  # its folds red-brown...
    assert (f @ np.array([0.299, 0.587, 0.114])).mean() < 120  # ...and dark
    body = shown(Q[(d < 60) & ~folds])
    assert body[:, 0].mean() > 200 and body.std(0).max() < 12  # its fire as the render has it, no contour bands


def test_an_embers_glow_in_the_soot_goes_orange_to_brown_to_grey_never_pink():
    E = np.full((160, 160, 3), 4, np.uint8)
    d = np.hypot(*(np.mgrid[:160, :160] - 80))
    E[d < 10] = (255, 170, 60)
    wf = fire_weight(E)
    room = np.full((160, 160, 3), 0.4, np.float32)
    Q = np.full((160, 160, 3), 0.08, np.float32)  # grey soot
    fire = X.Fire(160, 160, 1080, 1)
    fire.layer(Q, room, E, wf, (0, 160, 0, 160), 4, 0.0, 0.0, 30, None)
    s = shown(Q)
    rings = [s[(d >= r) & (d < r + 2)].mean(0) for r in (11, 14, 18, 24, 30, 40, 70)]
    luma = [float(v @ np.array([0.299, 0.587, 0.114])) for v in rings]
    assert all(a > b for a, b in zip(luma, luma[1:]))  # it dies away from the ember...
    assert all(v[2] - v[1] <= -10 for v in rings[:4]) and all(v[2] <= v[1] + 0.5 for v in rings)  # ...never pink
    assert np.abs(rings[-1] - shown(Q[0, 0])).max() < 1.0 and rings[-1][2] >= rings[-1][1] - 1  # (to the soot's grey)


def test_the_soot_hides_a_white_plate_as_it_hides_the_wall():
    P = np.full((200, 300, 3), 0.25, np.float32)
    P[80:120, 70:230] = 0.8  # a white switch plate on the wall
    Q = P * 0.4  # the composite's soot over it, a pure multiply...
    Q[:, :20] = P[:, :20]  # (none at its left)
    before = Q.copy()
    X.scatter(Q, P, (0, 200, 0, 300), 300)
    s, b = shown(Q) @ np.array([0.299, 0.587, 0.114]), shown(before) @ np.array([0.299, 0.587, 0.114])
    assert s[100, 150] - s[30, 150] < 0.7 * (b[100, 150] - b[30, 150])  # ...the plate goes under the cloud
    assert abs(s[30, 150] - b[30, 150]) < 1.5 and abs(s[190, 280] - b[190, 280]) < 1.5  # the smoked wall as it was
    sharp = [np.abs(np.diff(a[74:87, 150])).max() / (a[86, 150] - a[74, 150]) for a in (b, s)]
    assert sharp[1] < 0.7 * sharp[0]  # its edge softened
    assert np.array_equal(Q[:, :10], P[:, :10])  # where there is no soot, nothing
    lit = P * 1.2
    X.scatter(lit, P, (0, 200, 0, 300), 300)
    assert np.allclose(lit, P * 1.2)  # (nor where the fire lights it)


def test_the_smokes_outer_third_is_eaten_into_crisp_wisps_and_the_fire_kept_whole():
    E, d = disc(240, 240, 70, (255, 160, 40))
    o = X.Outline(240, 240, 240, seed=1)
    maps = o.maps(E, 4, (0, 240, 0, 240), 10, 0.0)
    assert maps[0].min() >= 0 and X.Outline.density(maps, (240, 240), 240)[d < 63].min() > 0.99  # the fire whole
    S, _ = disc(240, 240, 70, (40, 40, 40))
    q = X.Outline(240, 240, 240, seed=1)
    maps = q.maps(S, 4, (0, 240, 0, 240), 10, 0.0)
    assert maps[0].max() < 0.1  # smoke
    dens = X.Outline.density(maps, (240, 240), 240)
    outer, inner = (d > 70 * 0.7) & (d < 70 * 0.98), d < 70 * 0.5
    assert dens[inner].min() > 0.95 and 0.4 < dens[outer].mean() < 0.85 and dens[outer].std() > 0.25  # eaten away...
    assert ((dens[outer] > 0.15) & (dens[outer] < 0.85)).mean() < 0.3  # ...into wisps cut sharp, not feathered
    rows = np.arange(240)[:, None]
    crown, base = (d < 66) & (rows < 120 - 35), (d < 66) & (rows > 120 + 35)
    later = {t: X.Outline.density(q.maps(S, 4, (0, 240, 0, 240), t, 0.0), (240, 240), 240) for t in (30, 60)}
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
    seen = [float(shapes.at(k)[12, 12]) for k in range(16)]  # (at a quarter of the render's size)
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
    assert wall[1] - 162 <= X.LIGHT_WARM * (wall[0] - 246) + 2 or wall[0] >= 254  # the room's light kept warm...
    assert wall[2] <= 149 + 1  # ...blue not rising
    hot = gam(X.warm_light(P, lit, m, rim, burn=1.0)).astype(int)  # while the ball burns: a hot rim, the face as it was
    luma = np.array([0.2126, 0.7152, 0.0722])
    i = 60 + int(np.argmax(rim[100, 60:70]))
    assert (hot[100, i] - edge) @ luma > 5 and np.abs(hot[100, 100] - front).max() <= 1


def test_the_shoulder_looked_up_is_the_shoulder():
    from cuesync.bending import SHOULDER as S
    from cuesync.transfer import lin

    plate = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, -1)
    room = lin(plate).astype(np.float64)
    P = room * np.linspace(0.5, 3.0, 256).reshape(16, 16, 1)

    def curve(x):
        return np.where(x > S, S + (1.0 - S) * np.tanh((x - S) / (1.0 - S)), x)

    exact = np.clip(np.where(P > room, room + curve(P) - curve(room), P), 0, 1) ** (1 / 2.2) * 255
    assert np.abs(X.shoulder_added(P.astype(np.float32), plate).astype(float) - exact).max() <= 1.0


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
