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
    assert k == 11 and np.allclose(centre, (100, 40), atol=4)  # (frame 10: a speck too small to count)
    assert abs(r8 - (2 + 3 * (k + X.GROW_FRAMES - 10))) < 4  # its radius GROW_FRAMES after it shows


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
