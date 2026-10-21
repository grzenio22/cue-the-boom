"""cuesync.water: water bent from an open palm, made from the plate (the orb and the whip). Built on a synthetic shot:
a textured wall, a palm with no landmarks (taken to face the camera, tipped up), still or moving, its matte, and a
voice's timeline."""

import math

import cv2
import numpy as np
import pytest

from cuesync import water
from cuesync.elements import Timeline
from cuesync.transfer import gam, lin

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


def thrust_track(n, frame=40):
    """A measured extension: the tracker reports a palm impulse of three hand lengths/s."""
    t = track_of(n, events=[{"frame": frame, "kind": "extension"}])
    t["speed"][max(0, frame - 1):frame + 2] = [3.0 * S] * len(t["speed"][max(0, frame - 1):frame + 2])
    return t


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




def test_a_stationary_final_hit_does_not_send_water_up_the_wall():
    n = 95
    track = track_of(n)
    _, sc = scenes("orb", voice(n, hum=(4, 40), hits=((44, 1.0),)), track)
    assert not water.orb_plan(voice(n), track).throw
    assert not any(s.film or s.cast for s in sc)  # no wall impact without a hand impulse
    assert not any(b.lead for s in sc[44:] for b in s.blobs)
    assert any(len(s.blobs) > 30 for s in sc[44:56])  # the cue still makes a local splash


def test_a_stationary_final_hit_keeps_a_spreading_sheet_on_the_palm_over_several_frames():
    n, release = 130, 44
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    consecutive = longest = 0
    for k in range(release, release + 30):
        on_palm = [b for b in sc[k].blobs if not b.free and b.flat < 0.5 and b.r > 0.2 * S
                   and np.linalg.norm(np.array([b.x, b.y]) - hand.p[k]) < 0.35 * S]
        consecutive = consecutive + 1 if on_palm else 0
        longest = max(longest, consecutive)
    assert longest >= 10  # the volume spreads and drains visibly instead of disappearing after one splash frame


def test_a_stationary_final_hit_empties_the_raised_body_before_its_sheet_drains():
    n, release = 100, 44
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    body = []
    for k in range(release, release + water.COLLAPSE[1]):
        connected = [b for b in sc[k].blobs if not b.free and b.r > 0.2 * S]
        if connected:
            b = max(connected, key=lambda b: b.r * b.flat)
            height = hand.p[k, 1] - b.y + b.r * np.linalg.norm(np.reshape(b.shape, (2, 2))[1])
            body.append((height, b.r * b.flat))
    assert len(body) >= 6  # several connected intermediate shapes bridge the sphere to the palm sheet
    heights, thickness = np.asarray(body).T
    assert np.count_nonzero(np.abs(np.diff(heights)) > 0.015 * S) >= 3
    assert heights[-1] < heights[0] - 0.15 * S and thickness[-1] < 0.55 * thickness[0]


def test_a_sploosh_keeps_a_visible_connected_ribbon_from_the_raised_water_to_the_palm():
    n, release = 85, 44
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    for k in range(release + 3, release + 8):
        fld = water.Field(sc[k].blobs, H, W)
        y0, _, x0, _ = fld.box
        attached = (fld.f - fld.fm > water.TAU).astype(np.uint8)
        _, labels = cv2.connectedComponents(attached, connectivity=8)
        px, py = np.rint(hand.p[k] - [x0, y0]).astype(int)
        palm_label = labels[py, px]
        assert palm_label != 0
        connected = labels == palm_label
        # The whole path from the descending reservoir to the skin must survive phone scaling until it flattens.
        first, last = int(round(sc[k].held[1] - y0)), int(round(py - 0.05 * S))
        widths = connected[first:last + 1].sum(axis=1)
        assert len(widths) == last - first + 1 and widths.min() >= 0.075 * S, k - release


def test_the_receiving_mass_flattens_while_its_liquid_front_still_crosses_the_palm():
    n, release = 85, 44
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    k = release + 11
    x, y, z, _, _ = sc[k].held
    skin = float(hand.depth(k, hand.p[k, 0:1], hand.p[k, 1:2])[0])
    above = np.dot(np.array([x, y, z]) - np.r_[hand.p[k], skin], hand.nrm[k])
    assert above < 0.10 * S  # the reservoir becomes a film rather than retaining a raised glass stem
    early, late = sc[k].coat[2], sc[release + 22].coat[2]
    assert early < 0.42 * S and late > early + 0.10 * S


def test_the_scene_spreads_its_main_wet_front_across_the_palm_in_three_to_six_exposures():
    n, release = 100, 44
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    samples = [(k, sc[k].coat[2] / hand.s[k]) for k in range(release + 2, release + 26)
               if sc[k].coat is not None]
    frames, radii = np.asarray(samples).T
    gain = radii[-1] - radii[0]
    assert gain > 0.10  # the timed reflection must travel across fresh skin, rather than remain a fixed spot
    early = frames[np.flatnonzero(radii >= radii[0] + 0.20 * gain)[0]]
    late = frames[np.flatnonzero(radii >= radii[0] + 0.80 * gain)[0]]
    # Exclude first contact and the last draining fringe: the main spreading sheet reads over several exposures.
    assert 3 <= late - early + 1 <= 6


def test_the_pouring_reservoir_reaches_the_palms_depth_at_its_foot():
    n, release = 85, 44
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    k = release + 4
    incoming = [b for b in sc[k].blobs if b.stream]
    assert incoming  # the low receiving sheet cannot establish that the raised water reaches the palm
    fld = water.Field(incoming, H, W)
    y0, _, x0, _ = fld.box
    px, py = np.rint(hand.p[k] - [x0, y0]).astype(int)
    palm_z = float(hand.depth(k, hand.p[k, 0:1], hand.p[k, 1:2])[0])
    assert fld.f[py, px] > water.TAU
    assert abs(float(fld.zc[py, px]) - palm_z) < 0.1 * S
    assert fld.zc[fld.f > water.TAU].max() > palm_z + 0.3 * S


def test_a_runoff_root_follows_palm_depth_while_its_airborne_bead_keeps_its_birth_depth():
    n, release, k = 85, 44, 56
    track = track_of(n)
    tl = voice(n, hum=(4, 40), hits=((release, 1.0),))
    hands = [water.hand_of(track, n) for _ in range(2)]
    hands[1].plane[release + 11:, 2] += S
    sc = [water.orb_scenes(tl, track, hand, (H, W), 4, None) for hand in hands]
    root = hands[0].p[k] + np.array([-0.27, -0.48]) * S
    roots, beads = [], []
    for sequence in sc:
        streams = [b for b in sequence[k].blobs if b.stream]
        roots.append(min(streams, key=lambda b: np.linalg.norm(np.array(b[:2]) - root)))
        airborne = [b for b in sequence[k].blobs if b.free and not b.stream and b.flat == 1.0]
        beads.append(min(airborne, key=lambda b: np.linalg.norm(np.array(b[:2]) - root)))
    assert np.allclose(roots[0][:2], root) and np.allclose(roots[1][:2], root)
    assert roots[1].z - roots[0].z == pytest.approx(S)
    assert np.allclose(beads[0][:3], beads[1][:3])


def test_airborne_palm_spray_does_not_follow_a_later_hand_translation():
    n, release = 90, 44
    jump = release + 8
    tl = voice(n, hum=(4, 40), hits=((release, 1.0),))
    stationary = track_of(n)
    moving = track_of(n, xy=lambda k: (64.0 + (S if k >= jump else 0.0), 130.0))
    _, a = scenes("orb", tl, stationary)
    _, b = scenes("orb", tl, moving)

    def airborne(sc):
        return sorted([drop for drop in sc[jump].blobs if drop.free and drop.flat == 1.0
                       and 0.014 * S <= drop.r <= 0.029 * S], key=lambda drop: drop.r)

    before, after = airborne(a), airborne(b)
    assert before and len(before) == len(after)
    assert np.allclose([drop.x for drop in before], [drop.x for drop in after], atol=1e-6)


def test_a_quiet_final_hit_sends_palm_spray_less_high_than_a_strong_hit():
    n, release = 90, 44
    track = track_of(n)
    heights = []
    for strength in (0.2, 1.0):
        tl = voice(n, hum=(4, 40), hits=((release, strength),))
        hand, sc = scenes("orb", tl, track)
        assert water.orb_plan(tl, track).release == release
        spray = [b for b in sc[release + 12].blobs if b.free and b.flat == 1.0
                 and 0.014 * S <= b.r <= 0.029 * S]
        assert spray
        heights.append(max(hand.p[release + 12, 1] - b.y for b in spray))
    assert heights[0] + 0.04 * S < heights[1] < 2.0 * heights[0]


def test_the_held_orb_continuously_grows_attached_lobes_and_releases_small_drops():
    n = 100
    _, sc = scenes("orb", voice(n, hum=(4, 90), hits=()), track_of(n))
    lobed, detached = [], []
    for k in range(20, 70):
        orb = max(sc[k].blobs, key=lambda b: b.r)
        beyond = lambda b: math.hypot(b.x - orb.x, b.y - orb.y)
        fld = water.Field(sc[k].blobs, H, W)
        y0, y1, x0, x1 = fld.box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        connected = fld.f - fld.fm > water.TAU
        lobed.append(bool(np.any(connected & (np.hypot(xx - orb.x, yy - orb.y) > 1.18 * orb.r))))
        detached.append(any(b.free and b.r < 0.4 * orb.r and beyond(b) > 1.1 * orb.r for b in sc[k].blobs))
    assert sum(lobed) >= 25  # outward moving volumes change the boundary throughout the hold, beyond rounded wobble
    assert sum(detached) >= 4  # necks release water into separate falling drops


def test_a_broad_attached_fold_keeps_the_optical_depth_of_the_connected_water():
    n, k = 80, 23
    _, sc = scenes("orb", voice(n, hum=(4, 78), hits=()), track_of(n), seed=1)
    orb = max(sc[k].blobs, key=lambda b: b.r)
    fld = water.Field(sc[k].blobs, H, W)
    y0, y1, x0, x1 = fld.box
    yy, xx = np.mgrid[y0:y1, x0:x1]
    outer = ((fld.f - fld.fm > water.TAU) & (fld.fm < 0.05)
             & (np.hypot(xx - orb.x, yy - orb.y) > 1.1 * orb.r))
    assert outer.any()
    optical_radius = np.sqrt(fld.s2[outer]) / water.SIG_K
    assert np.median(optical_radius) > 0.5 * orb.r  # the flowing boundary shares the connected body's optical depth


def test_the_local_collapse_wets_the_palm_progressively_then_the_palm_dries():
    n, release = 140, 44
    _, sc = scenes("orb", voice(n, hum=(4, 40), hits=((release, 1.0),)), track_of(n))
    wet = np.array([max((w[5] for w in s.wet if w[2] > 0.2 * S), default=0.0) for s in sc])
    assert wet[:release].max() == 0.0  # occasional drips do not count as the broad wet palm
    peak = release + int(np.argmax(wet[release:release + 30]))
    assert peak >= release + 4 and wet[peak] > 2.0 * wet[release + 1]
    assert wet[release + 75] < 0.4 * wet[peak]  # a brief glistening palm visibly dries after the water runs away


def test_a_cue_on_the_first_frame_can_release_water_without_a_prior_held_pose():
    n = 3
    tl, track = voice(n, hum=(0, 1), hits=((0, 1.0),)), track_of(n)
    _, sc = scenes("orb", tl, track)
    assert water.orb_plan(tl, track).release == 0
    assert len(sc) == n and all(s.blobs for s in sc)


def test_a_wet_skin_coat_alone_glints_only_inside_the_hand_and_loses_gloss_as_it_dries():
    plate, track = wall(1), track_of(1)
    matte = matte_of(track)[0]
    p = np.asarray(track["xy"])
    room = water.Room(plate)
    normal = water._unit(room.key(0, p[0]) + np.array([0.0, 0.0, 1.0]))
    hand = water.Hand(p, normal[None], np.array([[0.0, -1.0, 0.0]]), np.array([S]),
                      np.zeros((1, 3)), None)

    def draw(wet):
        return water.frame(0, plate[0], matte, water.Scene(coat=(*p[0], 0.7 * S, wet, 0.3)), hand, room,
                           water.Noise(4), 0.0, 4)

    full, drying = draw(1.0), draw(0.15)
    delta = full.astype(float) - plate[0]
    assert np.count_nonzero(delta.max(-1) > 1) > 20  # a skin film catches a visible, uneven reflection of the key
    assert np.std(delta[matte > 0]) > 0.1 * delta[matte > 0].mean()
    assert np.array_equal(full[matte == 0], plate[0][matte == 0])
    assert np.array_equal(full, draw(1.0))
    assert 0.0 < (drying.astype(float) - plate[0]).sum() < 0.4 * delta.sum()


def test_the_post_collapse_palm_reflects_the_light_beyond_plain_wet_skin():
    n, release = 120, 44
    track, plate = track_of(n), wall(n)
    matte = matte_of(track)
    room = water.Room(plate)
    normal = water._unit(room.key(0, track["xy"][0]) + np.array([0.0, 0.0, 1.0]))
    hand = water.Hand(np.asarray(track["xy"]), np.repeat(normal[None], n, 0),
                      np.tile([0.0, -1.0, 0.0], (n, 1)), np.full(n, S), np.zeros((n, 3)), None)
    tl = voice(n, hum=(4, 40), hits=((release, 1.0),))
    sc = water.orb_scenes(tl, track, hand, (H, W), 4, None)
    responses = []
    for k in (release + 23, release + 60):
        assert sc[k].coat is not None
        plain = water.Scene(wet=sc[k].wet, band=sc[k].band, coat=(*sc[k].coat[:3], 0.0, 0.0))
        glossy = water.Scene(wet=sc[k].wet, band=sc[k].band, coat=sc[k].coat)
        a, b = [water.frame(k, plate[k], matte[k], scene, hand, room, water.Noise(4), 0.0, 4)
                for scene in (plain, glossy)]
        delta = b.astype(float) - a
        assert np.array_equal(b[matte[k] == 0], plate[k][matte[k] == 0])
        responses.append(delta.sum())
    assert responses[0] > 20 and 0.0 <= responses[1] < 0.4 * responses[0]


def test_an_oblique_palm_keeps_broken_wet_highlights_for_a_second_after_the_sheet_drains():
    n, release = 105, 44
    track, plate = track_of(n), wall(n)
    plate[:, :60, :32] = [238, 245, 250]  # a visible room window away from the palm's mirror direction
    plate[:, :75, 90:] = [38, 35, 30]
    matte, hand = matte_of(track), water.hand_of(track, n)
    hand.nrm[:] = water._unit(np.array([-0.05, -0.79, 0.61]))
    room = water.Room(plate)
    sc = water.orb_scenes(voice(n, hum=(4, 40), hits=((release, 1.0),)), track, hand, (H, W), 4, None)
    yy, xx = np.mgrid[:H, :W]
    palm = np.hypot(xx - 64, yy - 130) < 0.5 * S
    for k in (release + 26, release + 38, release + 50):
        assert sc[k].coat is not None
        plain = water.Scene(wet=sc[k].wet, band=sc[k].band, coat=(*sc[k].coat[:3], 0.0, 0.0))
        glossy = water.Scene(wet=sc[k].wet, band=sc[k].band, coat=sc[k].coat)
        a, b = [water.frame(k, plate[k], matte[k], scene, hand, room, water.Noise(4), 0.0, 4)
                for scene in (plain, glossy)]
        delta = b.astype(float) - a
        highlights = delta.max(-1)[palm]
        assert 0.04 < np.mean(highlights > 2) < 0.65, k - release
        assert highlights.max() < 90  # a transparent film leaves the skin visible under the window's reflection
        assert np.corrcoef(a[palm].ravel(), b[palm].ravel())[0, 1] > 0.9
        assert np.array_equal(b[matte[k] == 0], plate[k][matte[k] == 0])


def test_settled_palm_beads_drain_and_shrink_instead_of_freezing_as_glitter():
    n = 130
    track = track_of(n)
    _, sc = scenes("orb", voice(n, hum=(4, 40), hits=((44, 1.0),)), track)
    beads = lambda k: [b for b in sc[k].blobs if b.free and b.flat < 1.0 and b.r < 0.08 * S]
    early, late = beads(60), beads(100)
    assert early and late and len(early) == len(late)
    assert all(b.glint <= 0.4 for b in early)
    assert sum(b.r for b in late) < 0.85 * sum(b.r for b in early)
    assert max(b.y - a.y for a, b in zip(early, late)) > 0.03 * S


def test_a_drained_film_catches_sharp_light_along_the_existing_skin_creases():
    size = 256
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    distance = yy - (128.0 + 0.16 * (xx - 128.0))
    skin = np.full((size, size, 3), [186, 117, 101], np.float32)
    skin *= (1.0 - 0.18 * np.exp(-(distance / 1.5) ** 2))[..., None]
    plate = np.uint8(skin)[None]
    room = water.Room(plate)
    # The lower crease bank turns into the light while the smooth palm faces slightly away from its mirror peak.
    normal = np.array([0.0, -0.45, 0.89])
    hand = water.Hand(np.array([[128.0, 128.0]]), normal[None], np.array([[0.0, -1.0, 0.0]]),
                      np.array([100.0]), np.zeros((1, 3)), None)
    out = water.frame(0, plate[0], np.full((size, size), 255, np.uint8),
                      water.Scene(coat=(128.0, 128.0, 68.0, 1.0, 0.0)), hand, room, water.Noise(4), 0.0, 4)
    added = np.maximum(out.astype(float) - plate[0], 0).mean(-1)
    center = (np.abs(xx - 128) < 40) & (np.abs(yy - 128) < 28)
    banks = center & (np.abs(distance) > 1) & (np.abs(distance) < 4)
    flat = center & (np.abs(distance) > 10)
    assert added[banks].mean() > 2.0 * added[flat].mean()
    assert np.mean(added[flat] > 8) < 0.15  # wet skin retains clear skin between its narrow reflections


def test_a_spreading_palm_film_has_an_advancing_broken_liquid_edge():
    size = 256
    plate = np.full((1, size, size, 3), [186, 117, 101], np.uint8)
    room = water.Room(plate)
    hand = water.Hand(np.array([[128.0, 128.0]]), np.array([[0.0, -0.45, 0.89]]),
                      np.array([[0.0, -1.0, 0.0]]), np.array([100.0]), np.zeros((1, 3)), None)
    yy, xx = np.mgrid[:size, :size]
    peaks = []
    for radius in (36.0, 65.0):
        out = water.frame(0, plate[0], np.full((size, size), 255, np.uint8),
                          water.Scene(coat=(128.0, 128.0, radius, 1.0, 1.0)), hand, room,
                          water.Noise(4), 0.0, 4)
        added = np.maximum(out.astype(float) - plate[0], 0).mean(-1)
        radial = np.hypot(xx - 128, (yy - 128) / 0.8)
        boundary = (radial > 0.72 * radius) & (radial < 1.30 * radius)
        assert added[boundary].max() > 10
        assert added[boundary].sum() > 0.30 * added.sum()
        peaks.append(np.average(radial[boundary], weights=added[boundary] + 1e-6))
        # The irregular front may reach beyond the mean radius, but it remains a local palm film.
        assert np.array_equal(out[radial > 1.32 * radius], plate[0][radial > 1.32 * radius])
    assert peaks[1] > peaks[0] + 20  # the reflection travels with the wet front rather than lighting one fixed patch


def _phone_palm_film(radius: float, flowing: float, creased: bool = False):
    """A smooth palm crop at capture resolution, then reduced by three as in the phone gallery."""
    size, center, scale = 384, 192.0, 180.0
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    dx = xx - center
    skin = np.full((size, size, 3), [186, 117, 101], np.float32)
    if creased:
        first = yy - (center + 0.14 * dx + 0.0018 * dx * dx)
        second = yy - (center - 20.0 - 0.25 * dx + 0.0008 * dx * dx)
        skin *= (1.0 - 0.18 * np.exp(-(first / 2.3) ** 2)
                 - 0.12 * np.exp(-(second / 2.2) ** 2))[..., None]
    plate = skin.astype(np.uint8)[None]
    room = water.Room(plate)
    normal = np.array([0.0, -0.45, 0.89])
    hand = water.Hand(np.array([[center, center]]), normal[None], np.array([[0.0, -1.0, 0.0]]),
                      np.array([scale]), np.zeros((1, 3)), None)
    out = water.frame(0, plate[0], np.full((size, size), 255, np.uint8),
                      water.Scene(coat=(center, center, radius, 1.0, flowing)), hand, room,
                      water.Noise(4), 0.0, 4)
    added = np.maximum(out.astype(np.float32) - plate[0], 0.0).mean(-1)
    return cv2.resize(added, (128, 128), interpolation=cv2.INTER_AREA), plate[0], out


def _phone_reflections(added: np.ndarray):
    """Resolve glints bright enough to survive a small phone image, ignoring isolated subpixel sparks."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(added > 5.0), connectivity=8)
    components = [j for j in range(1, len(stats)) if stats[j, cv2.CC_STAT_AREA] >= 2]
    return labels, stats, components


def test_the_spreading_rim_stays_connected_at_phone_size_over_five_exposures():
    radii = (64.0, 76.0, 88.0, 100.0, 112.0)
    travel = []
    yy, xx = np.mgrid[:128, :128]
    for radius in radii:
        added, _, _ = _phone_palm_film(radius, 1.0)
        labels, stats, components = _phone_reflections(added)
        assert components, radius
        largest = max(components, key=lambda j: stats[j, cv2.CC_STAT_AREA])
        w, h = stats[largest, cv2.CC_STAT_WIDTH:cv2.CC_STAT_HEIGHT + 1]
        # A narrow but continuous arc reads as the edge of a sheet, unlike a necklace of flashing dots.
        assert np.hypot(w - 1, h - 1) > 0.65 * radius / 3.0, radius
        on_rim = labels == largest
        travel.append(np.average(np.hypot(xx[on_rim] - 64, (yy[on_rim] - 64) / 0.8),
                                 weights=added[on_rim]))
    assert np.all(np.diff(travel) > 1.0)  # the connected reflection visibly crosses fresh palm skin each exposure
    assert travel[-1] - travel[0] > 10.0


def test_a_drained_film_consolidates_reflections_along_curved_skin_creases_at_phone_size():
    radius = 100.0
    added, plate, out = _phone_palm_film(radius, 0.0, creased=True)
    _, stats, components = _phone_reflections(added)
    assert 1 <= len(components) <= 4  # two broad palm folds catch a few connected glints, rather than glitter
    spans = [np.hypot(stats[j, cv2.CC_STAT_WIDTH] - 1, stats[j, cv2.CC_STAT_HEIGHT] - 1)
             for j in components]
    assert max(spans) > 0.35 * radius / 3.0
    yy, xx = np.mgrid[:384, :384]
    center = np.hypot(xx - 192, yy - 192) < 0.65 * radius
    assert np.corrcoef(plate[center].ravel(), out[center].ravel())[0, 1] > 0.9
    assert np.max(out.astype(float) - plate) < 90  # clear skin stays visible through its wet reflection


def test_a_resolved_wet_skin_footprint_has_a_visibly_irregular_boundary():
    skin = lin(np.full((256, 256, 3), [186, 117, 101], np.uint8))
    wet = skin.copy()
    water._wet(wet, [(128.0, 128.0, 68.0, 54.4, 0.0, 0.14, 2.7, 0.0, 0.15)],
               np.full((256, 256), 255, np.uint8), np.ones(3), np.array([0.0, -1.0]), sheen=False)
    darkened = (skin - wet).mean(-1) > 0.07 * skin.mean(-1)
    reach = []
    for angle in np.linspace(-math.pi, math.pi, 180, endpoint=False):
        distances = np.linspace(0.5, 1.35, 200)
        xx = np.rint(128.0 + 68.0 * distances * math.cos(angle)).astype(int)
        yy = np.rint(128.0 + 54.4 * distances * math.sin(angle)).astype(int)
        wet_samples = distances[darkened[yy, xx]]
        assert wet_samples.size
        reach.append(wet_samples.max())
    # Remove the palm's elliptical foreshortening before measuring an uneven spreading front.
    assert np.percentile(reach, 95) - np.percentile(reach, 5) > 0.22
    assert np.std(reach) > 0.07


def test_a_sideways_palm_keeps_the_whole_moving_film_boundary():
    plate = np.full((1, 256, 256, 3), [186, 117, 101], np.uint8)
    hand = water.Hand(np.array([[128.0, 128.0]]), np.array([[-0.8, 0.0, 0.6]]),
                      np.array([[0.0, -1.0, 0.0]]), np.array([100.0]), np.zeros((1, 3)), None)
    out = water.frame(0, plate[0], np.full((256, 256), 255, np.uint8),
                      water.Scene(coat=(128.0, 128.0, 60.0, 1.0, 1.0)), hand, water.Room(plate),
                      water.Noise(4), 0.0, 4)
    delta = np.abs(out.astype(float) - plate[0])
    assert np.count_nonzero(delta[190:196] > 1) > 20  # a rotated lobe extends past the unrotated radius
    assert not delta[210:].any()


def test_the_flat_splash_is_on_the_palm_not_at_the_spheres_centre():
    n = 70
    hand, sc = scenes("orb", voice(n, hum=(4, 30), hits=()), track_of(n))
    flat = []
    for k in range(36, 50):
        sheets = [b for b in sc[k].blobs if not b.free and b.flat < 0.15 and b.r > 0.3 * S]
        if sheets:
            flat.append((k, max(sheets, key=lambda b: b.r)))  # the central sheet; its tongues run beyond the palm
    assert flat
    assert all(np.linalg.norm(np.array([b.x, b.y]) - hand.p[k]) < 0.15 * S for k, b in flat)


def test_a_thin_pool_conforms_to_a_tilted_palm_without_a_straight_depth_cut():
    plate = np.full((1, 256, 256, 3), 150, np.uint8)
    hand = water.Hand(np.array([[128.0, 128.0]]), np.array([[0.0, -0.45, 0.89]]),
                      np.array([[0.0, -1.0, 0.0]]), np.array([100.0]), np.array([[0.0, 0.5, -64.0]]), None)
    matte = np.full((256, 256), 255, np.uint8)
    sc = water.Scene([water.Blob(128.0, 128.0, 0.0, 30.0, flat=0.04)])
    layer = water.shade(water.Field(sc.blobs, 256, 256), sc, 0, hand, water.Room(plate), water.Noise(0),
                        water._pyramid(lin(plate[0])), matte, 0.0, 0)
    y0, _, x0, _ = layer.box
    upper, lower = layer.alpha[116 - y0, 128 - x0], layer.alpha[140 - y0, 128 - x0]
    assert min(upper, lower) > 0.1 and max(upper, lower) / min(upper, lower) < 1.5
    assert layer.z[140 - y0, 128 - x0] - layer.z[116 - y0, 128 - x0] == pytest.approx(12.0, abs=1.0)


def test_a_distant_flat_splash_is_visible_away_from_the_hand():
    plate = np.full((1, H, W, 3), 150, np.uint8)
    hand = water.hand_of(track_of(1), 1)
    sc = water.Scene([water.Blob(64.0, 60.0, -150.0, 16.0, flat=0.12)])
    layer = water.shade(water.Field(sc.blobs, H, W), sc, 0, hand, water.Room(plate), water.Noise(0),
                        water._pyramid(lin(plate[0])), np.zeros((H, W), np.uint8), 0.0, 0)
    assert layer.alpha.max() > 0.5
    assert layer.z[layer.alpha > 0.5].max() < -100.0


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

def test_the_orb_gathers_with_the_voice_and_drops_on_a_final_hit_without_a_thrust():
    n = 80
    p = water.orb_plan(voice(n, hum=(6, 40), hits=((30, 0.5), (44, 1.0))), track_of(n))
    assert (p.gather, p.formed) == (6, 6 + water.GATHER)
    assert (p.release, p.throw, p.size) == (44, False, 1.0)
    assert p.jolts == [(30, 0.5)]  # the hit while it is held jolts it


def test_a_voice_opening_on_its_hit_gathers_the_orb_just_before_it():
    p = water.orb_plan(voice(60, hum=(30, 31), hits=((30, 1.0),)), track_of(60))
    assert p.release == 30 and not p.throw
    assert p.formed == 30 - water.GATHER_LEAD and p.gather == p.formed - water.GATHER


def test_a_thrust_throws_the_orb_before_the_last_hit():
    n = 80
    track = thrust_track(n, 30)
    p = water.orb_plan(voice(n, hum=(6, 50), hits=((60, 1.0),)), track)
    assert (p.release, p.throw, p.size) == (30, True, water.THRUST_SIZE)


def test_an_extension_keeps_the_impulse_that_preceded_the_stopped_hand():
    n = 80
    track = track_of(n, events=[{"frame": 41, "kind": "extension", "peak_speed": 3.2 * S}])
    track["speed"][35:42] = [v * S for v in (3.2, 2.1, 1.5, 1.3, 1.2, 1.0, 0.5)]
    plan = water.orb_plan(voice(n, hum=(6, 70), hits=()), track)
    assert (plan.release, plan.throw) == (41, True)  # tracker labels the stop up to six frames after the impulse
    track["speed"] = [0.0] * n
    assert not water.orb_plan(voice(n, hum=(6, 70), hits=()), track).throw  # a stale label alone is insufficient


def test_a_hit_the_voice_holds_on_long_after_jolts_the_orb_instead_of_throwing_it():
    n = 90
    p = water.orb_plan(voice(n, hum=(6, 70), hits=((30, 1.0),)), track_of(n))
    assert not p.throw and p.jolts == [(30, 1.0)]
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
    hand, sc = scenes("orb", tl, thrust_track(n))
    k = next(k for k in range(40, n) if sc[k].film)  # it hits the wall
    assert 40 + 2 < k < 40 + 12
    ball = [max(sc[j].blobs, key=lambda b: b.r) for j in range(40, k)]
    assert ball[-1].y < ball[0].y - 0.5 * S and ball[-1].r < 0.8 * ball[0].r  # up along the aim, and away
    assert all(b.move for b in ball[1:])  # blurred by its flight
    sheets = [next(it for it in sc[j].film if isinstance(it, water.Sheet)) for j in range(k, n)]
    r = np.array([q.r for q in sheets])
    assert r[0] > 1.2 * ball[-1].r and np.all(np.diff(r) >= 0) and r[8] > 1.25 * r[0]  # it lands squashed and
    # spreads over the wall...
    assert r[0] < 0.7 * r[2] and r[1] > 0.8 * r[2]  # ... growing over its first frames, not there whole in one
    fa = -0.5 * math.pi  # thrown up: it lies drawn out up and down, its middle moving on up
    stretch = [(a, ph) for l, a, ph in sheets[-1].wobble if l == 2]
    assert stretch and stretch[0][0] >= 0.15 and math.cos(stretch[0][1] - 2.0 * fa) > 0.99
    assert sheets[-1].y < sheets[0].y - 0.05 * r[-1]
    look, wet = np.array([q.look for q in sheets]), np.array([q.wet for q in sheets])
    assert look[0] == 1.0 and look[-1] == 0.0 and np.abs(np.diff(look)).max() < 0.4  # ... draining over frames...
    assert np.all(np.diff(wet) >= 0) and 0.0 < wet[0] <= 0.5 and wet[-1] > 0.85  # ... into the wet patch it leaves,
    # soaking in (half as dark at first: at its full darkness from its sixth frame, a brown decal stamped on the wall)
    froth = np.array([q.froth for q in sheets])
    assert sheets[0].dome == 1.0 and 0.0 < froth[0] <= 0.35  # it lands flashing white for a moment...
    assert froth[1] <= 0.3 * froth[0] and froth[2] == 0.0  # ... gone in two frames (a milky core, a cotton ball)
    sag = np.array([q.sag for q in sheets])
    assert sag[0] == 0.0 and np.all(np.diff(sag) >= 0) and sag[-1] > 0.4 * water.WALL_CREEP[0]  # the patch's lower edge
    assert sag[8] < 0.5 * sag[-1]  # creeps down, slowly (sagged in six frames and held, it was a stamp)...
    runs = np.array([[ln for _, _, ln in q.runs] for q in sheets])
    assert water.WALL_SAG[1][0] <= runs.shape[1] <= water.WALL_SAG[1][1]  # ... and runs down in narrow runs...
    assert all(math.sin(a) > 0.2 and hw < 0.05 for a, hw, _ in sheets[-1].runs)  # ... along its bottom...
    assert runs[-1].max() > 2.0 * runs[-1].min() and np.all(np.diff(runs, axis=0) >= 0)  # ... of their own lengths,
    assert (runs[8] < 0.9 * runs[-1]).sum() >= runs.shape[1] // 2  # creeping on (at once, alike: a row of legs)
    fingers = sheets[water.FINGER_DRAIN[0]].fingers  # at their longest
    lengths = np.array([ln for _, ln, _, _ in fingers])
    along = np.array([math.cos(a - fa) for a, *_ in fingers])
    assert len(fingers) >= water.WALL_FINGERS[0] and along.mean() > 0.3  # tongues flung out round it, mostly the way
    # it was thrown (mostly down whatever the throw, a jellyfish's legs)...
    assert along[np.argmax(lengths)] > 0.5 and lengths.max() > 2.0 * lengths.min()  # ... the longest that way, of
    # different lengths (alike, a starburst)...
    across = np.abs(along) < 0.3
    assert np.all(lengths[across] <= 0.5 * lengths.max())  # ... short across it (long out to the sides, arms)...
    assert all(water.FINGER_BEAD[0] <= bd <= water.FINGER_BEAD[1] for *_, bd in fingers)  # ... each ending in a bead
    assert all(ln <= water.FINGER_DRAIN[1] + 1e-6 for _, ln, _, _ in sheets[-1].fingers)  # ... and drained into it, so
    # no one finger is left long on the stain (a trunk)
    trails = [[it for it in sc[j].film if isinstance(it, water.Trail)] for j in range(k, n)]
    assert not any(trails[:water.WALL_DRIPS[4]])  # nothing runs down it in the first frames (legs one frame on)...
    ball_r = sheets[-1].r / (water.WALL_SPREAD[1] * (1.0 + water.WALL_SOAK[0]))
    drips = [[t for t in ts if t.r > water.WALL_BEADS[1][1] * ball_r] for ts in trails]  # (bigger than what it flings)
    first = [next(j for j, ts in enumerate(drips) if len(ts) > i) for i in range(len(drips[-1]))]
    assert len(first) >= 2 and np.diff(first).min() >= water.WALL_DRIPS[5][0] - 1  # ... then drips, one after another
    runs = sorted(t.y1 - t.y0 for t in drips[-1])
    assert np.diff(runs).min() > 1.0  # ... each its own length (all at once, as long, a row of legs)
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
    assert out[70, 64].max() < 143  # the wet patch, darker
    assert np.array_equal(out[60:80, :8], plate[0, 60:80, :8])  # nothing on the hand
    assert np.array_equal(out[:30, :30], plate[0, :30, :30])  # nothing off it
    edge = out[70, 30:50, 1]
    assert ((edge > edge.min() + 2) & (edge < edge.max() - 2)).sum() >= 3  # its outline soft, not a stain's line
    drop = out[118:142, 52:76]
    assert drop.max() > 200 and drop.min() < 140  # a drop: a glint on it and a dark line along its foot
    assert out[134:136, 60:68].mean() < out[124:126, 60:68].mean() + 60  # the wall seen through it, not a white bead
    run = sheet._replace(fingers=(), runs=((0.5 * math.pi, 0.15, 0.6),))  # a run down from its bottom
    out = water.frame(0, plate[0], None, water.Scene(film=[run]), _hand_at(0.0), room, noise, 0.0, 0).astype(int)
    assert out[100, 64].mean() < out[100, 54].mean() - 3  # its lower edge runs down in it, narrow
    crept = run._replace(sag=0.4)
    low = water.frame(0, plate[0], None, water.Scene(film=[crept]), _hand_at(0.0), room, noise, 0.0, 0).astype(int)
    assert low[98, 56].mean() < out[98, 56].mean() - 3  # and creeps down along it


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
    hand, sc = scenes("orb", tl, thrust_track(n))
    xs = [max(b.x for b in sc[k].blobs) for k in range(40, 44)]
    assert xs[-1] - xs[0] > 1.0 * S  # away along the aim...
    burst = {(l, a) for l, a in water.BURST_SWELL[2]}
    assert not any(b.flat < 1.0 or {(l, a) for l, a, _ in b.modes} == burst for k in range(40, n)
                   for b in sc[k].blobs if b.r > 0.05 * S)  # (the drops it leaves on the palm aside)
    assert max(len([b for b in sc[k].blobs if b.r > 0.05 * S]) for k in range(42, n)) < 6  # ... and gone, with no
    # splash in mid-air (the drops it trails and leaves on the palm aside)


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


def test_the_orb_gathers_over_sixteen_frames_out_of_drops_drawn_in_from_far():
    n = 40
    tl, track = voice(n, hum=(4, 39), hits=()), track_of(n)
    hand, sc = scenes("orb", tl, track)
    plan = water.orb_plan(tl, track)
    assert plan.formed - plan.gather == water.GATHER >= 14  # (in 12, whole by its eighth frame: a poof)
    body = [max((b for b in sc[k].blobs if not b.free), key=lambda b: b.r, default=None) for k in range(n)]
    r = np.array([0.0 if b is None else b.r for b in body])
    R = water.ORB_R * S
    mid = (plan.gather + plan.formed) // 2
    assert r[plan.gather + 2] < 0.25 * R and 0.3 * R < r[mid] < 0.8 * R  # a seed, growing evenly...
    assert r[plan.formed] == pytest.approx(R)  # ... to its size
    drops = [[b for b in sc[k].blobs if b.free] for k in range(plan.gather, plan.formed)]
    assert max(len(d) for d in drops) >= 10  # out of drops drawn in...
    c = np.array(track["xy"][plan.gather + 4])
    assert max(np.hypot(b.x - c[0], b.y - c[1]) for d in drops[:6] for b in d) > 2.0 * R  # ... from far round it


def test_a_hit_squashes_the_held_orb_along_the_aim_flings_drops_and_flares_its_light():
    n = 60
    hand, sc = scenes("orb", voice(n, hum=(4, 58), hits=((30, 1.0),), aim=0.0), track_of(n))  # (up)

    def along_over_across(k):  # its outline's reach up over its reach across (its shape matrix's support)
        S2 = np.reshape(max((q for q in sc[k].blobs if not q.free), key=lambda q: q.r).shape, (2, 2))
        return np.linalg.norm(S2.T @ [0.0, 1.0]) / np.linalg.norm(S2.T @ [1.0, 0.0])

    before = along_over_across(29)
    q = 1.0 - water.HIT_SQUASH[0][1]
    for k in (30, 31):  # squashed along the aim by a third for two frames (a marble that sat still on the beat)...
        assert 0.9 * q ** 1.5 < along_over_across(k) / before < 0.85
    assert max(along_over_across(k) for k in range(32, 38)) > 1.05 * before  # ... springing back through a stretch...
    assert abs(along_over_across(50) / before - 1.0) < 0.1  # ... and settling
    free = [sum(b.free for b in sc[k].blobs) for k in (29, 31)]
    assert free[1] - free[0] >= 10  # drops flung out round it
    assert sc[30].flare > 1.3 and 1.0 < sc[31].flare < sc[30].flare and sc[33].flare == 1.0  # its light flares


def test_a_thrown_orb_stays_clear_smeared_along_its_flight_and_trails_drops():
    n = 60
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), thrust_track(n))  # (to the right)
    ball = [max(sc[k].blobs, key=lambda b: b.r) for k in (42, 43)]
    assert 0.0 < ball[1].foam <= 0.18  # most of the moving body transmits the room
    travel = math.hypot(ball[1].x - ball[0].x, ball[1].y - ball[0].y)
    smear = np.hypot(*ball[1].move)
    way = np.array(ball[1].move) / smear
    long = 2.0 * ball[1].r * np.linalg.norm(np.reshape(ball[1].shape, (2, 2)) @ way) + smear
    assert 0.0 < smear <= water.THROWN_SHUTTER * travel + 1e-6  # ... smeared along its flight...
    assert long <= water.THROWN_LONG * 2.0 * ball[1].r + 1e-6  # ... a ball, not much longer than wide (smeared over
    # all its travel, a tall see-through capsule)...
    trail = [b for b in sc[43].blobs if b.free and b.r < 0.5 * ball[1].r]
    assert len(trail) >= 2 and all(b.x < ball[1].x for b in trail)  # ... drops torn off its back, left behind it


def test_the_palm_where_the_orb_sat_is_left_wet_with_drops_on_it():
    n = 70
    track = thrust_track(n)
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), track)
    p = np.array(track["xy"][50])
    wet = [w for w in sc[50].wet if w[5] > 0.5 * water.PALM_WET[0]]
    assert wet and math.hypot(wet[0][0] - p[0], wet[0][1] - p[1]) < 0.5 * S  # wet where it sat...
    assert sc[n - 1].wet[0][5] > 0.6 * wet[0][5]  # ... drying slowly (thrown off a palm left dry, it was never on it)
    drops = [b for b in sc[50].blobs if b.flat < 1.0 and math.hypot(b.x - p[0], b.y - p[1]) < 0.6 * S]
    assert water.PALM_DROPS[0][0] <= len(drops) <= water.PALM_DROPS[0][1]  # ... with a few drops left on it


def test_the_orbs_rim_is_lit_towards_the_key_light_and_gathers_it_inside_its_far_edge():
    plate = _smooth_plate()
    plate[0, :20] = 255  # the room's light over it
    hand, room, noise = _big_hand(), water.Room(plate), water.Noise(0)
    r = 72.0
    sc = water.Scene([water.Blob(128.0, 140.0, 0.0, r)], wave=0.0)
    out = lin(water.frame(0, plate[0], None, sc, hand, room, noise, 0.0, 0)) @ np.float32([0.2126, 0.7152, 0.0722])

    def ring(lo, hi, a0, a1):  # its mean brightness over a band of its radius, between angles (rad, 0 right, y down)
        th = np.linspace(a0, a1, 24)
        q = np.linspace(lo, hi, 4)
        return np.mean([out[int(140 + r * f * math.sin(t)), int(128 + r * f * math.cos(t))] for t in th for f in q])

    up, down = -0.5 * math.pi, 0.5 * math.pi
    assert ring(0.92, 0.98, up - 0.8, up + 0.8) > 2.0 * ring(0.92, 0.98, down - 0.8, down + 0.8)  # bright towards
    # the light, dark away from it (dark evenly all round, a glass marble's outline)...
    side = 0.5 * (ring(0.76, 0.88, -0.4, 0.4) + ring(0.76, 0.88, math.pi - 0.4, math.pi + 0.4))
    assert ring(0.76, 0.88, down - 0.4, down + 0.4) > 1.15 * side  # ... and its light gathered inside its far edge


def test_the_light_the_orb_focuses_is_rolled_off_below_white():
    R = 40.0
    P = np.ones((200, 200, 3), np.float32) * np.float32([0.9, 0.45, 0.4])  # lit skin, its red nearly clipped
    plate = (np.clip(P, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)[None]
    hand = water.Hand(np.array([[100.0, 100.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, -1.0, 0.0]]),
                      np.array([100.0]), np.array([[0.0, 0.0, 0.0]]), None)
    room = water.Room(plate)
    fld = water.Field([water.Blob(100.0, 100.0, R + 25.0, R)], 200, 200)
    layer = water.shade(fld, water.Scene(), 0, hand, room, water.Noise(0), water._pyramid(P.copy()), None, 0.0, 0)
    Q = P.copy()
    water.caustic(Q, layer, 0, hand, room, np.full((200, 200), 255, np.uint8), (100.0, 100.0, R + 25.0, R, 1.0),
                  flare=1.5)
    raised = Q > P + 1e-4
    assert raised.any() and Q[raised].max() <= water.CAUSTIC_KNEE[1] + 1e-4  # brighter, but no channel to white...
    assert Q.min(-1).max() < 0.8  # ... (lifted straight, a white lamp glued to the palm)


def test_the_orbs_shade_comes_on_as_it_gathers_and_goes_as_it_is_thrown():
    n = 70
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), thrust_track(n))
    w = np.array([0.0 if s_.held is None else s_.held[4] for s_ in sc])
    plan = water.orb_plan(voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), thrust_track(n))
    assert w[plan.gather] < 0.1 and w[plan.formed] == 1.0 and w[40 - 1] == 1.0
    assert w[(plan.gather + plan.formed) // 2] < 0.3  # its light comes in as it forms (white hot half formed)
    assert np.abs(np.diff(w)).max() < 0.4  # never all at once (it popped on with a few drops in the air)
    assert 0.0 < w[41] < 1.0 and w[40 + water.TAKEOFF + 2] == 0.0  # let go over a few frames, not in one


def test_the_light_on_the_palm_does_not_blink_when_the_orb_is_thrown():
    n = 60
    plate, track = wall(n), thrust_track(n)
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


def test_the_whip_is_a_stream_sized_to_the_hand():
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


def test_a_hit_flings_the_stream_along_the_aim_tapering_and_breaking_up_from_its_tail(monkeypatch):
    monkeypatch.setattr(water, "ROOM_DEPTH", 0.6)  # the wall three hands behind the palm, inside this small frame
    n = 60
    track = track_of(n)
    hand, sc = scenes("whip", voice(n, hum=(2, 58), hits=((30, 1.0),), aim=0.0), track)  # (up)
    p = np.array(track["xy"][30])

    def strand(k):
        return sorted((b for b in sc[k].blobs if b.stream and b.w < 1.0), key=lambda b: b.stream[0])

    out = strand(31)
    assert min(b.y for b in out) < p[1] - 1.5 * S  # flung out along the aim in two frames...
    assert out[0].r > 4.0 * out[-1].r  # ... one stream tapering to a fine tail (a rod, as thick all along)...
    late = strand(33)
    assert late[-1].stream[0] < 0.9 * out[-1].stream[0]  # ... breaking up from its tail (held whole for six frames,
    assert min(b.y for b in late) > min(b.y for b in out) + 0.25 * S  # a rod), falling...
    assert sum(b.free for b in sc[33].blobs) > 2 * sum(b.free for b in sc[29].blobs) + 10  # ... into drops...
    beads = [b for k in range(31, 35) for b in sc[k].blobs if b.free and b.r > 0.02 * S]
    long = np.array([np.divide(*np.linalg.svd(np.reshape(b.shape, (2, 2)), compute_uv=False)) for b in beads])
    assert len(beads) >= 4 and np.mean(long < 2.0) > 0.7  # ... short drops and a few surviving ligaments; long drawn
    # shapes across the arc looked like zipper teeth. Motion still smears each drop along its own flight.
    assert any(b.move for b in beads)
    assert np.ptp([b.r for b in beads]) > 0.5 * np.median([b.r for b in beads])  # ... of uneven sizes (round, alike
    # and evenly spaced, a string of pearls)
    assert not strand(30 + water.LASH_BACK[1] - 1)  # all of it flung, until it pours out again
    k = next((k for k in range(30, n) if sc[k].film), None)
    assert k is not None and k < 30 + 12  # its drops strike the wall and wet it, as the orb's splash does
    sheet = next(it for it in sc[n - 1].film if isinstance(it, water.Sheet))
    assert sheet.wobble[0][0] == 2 and sheet.wobble[0][1] >= water.LASH_WALL[1]  # a streak, drawn out the way it went


def test_the_whip_rises_churning_out_of_a_smooth_pool_that_wets_the_palm():
    n = 40
    track = track_of(n)
    hand, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    pool = sorted([b for b in sc[30].blobs if b.flat < 1.0 and not b.stream], key=lambda b: -b.r)
    main, parts = pool[0], pool[1:]
    p = np.array(track["xy"][30])
    assert main.r > 1.2 * water.WHIP_R[0] * S and main.flat <= 0.05  # a wide, thin film (a domed gel lens with a
    # raised rim)...
    assert len(parts) == water.WHIP_POOL_PARTS[0] and all(
        b.flat == main.flat and b.r < main.r and np.hypot(b.x - main.x, b.y - main.y) < 1.2 * main.r
        for b in parts)  # ...
    # an irregular puddle, flat pools run into it...
    assert main.foam < 0.2  # ... smooth (frothing all over, crumpled cellophane)...
    assert np.hypot(main.x - p[0], main.y - p[1]) < 0.2 * S  # ... on the palm (no ball over it: a test tube)
    tube = [b for b in sc[30].blobs if b.stream and b.w < 1.0]
    assert tube[0].foam > tube[len(tube) // 2].foam + 0.2  # the stream churns where it rises out of it...
    assert tube[0].flat < 0.5 and tube[len(tube) // 2].flat == 1.0  # ... as flat as the pool there, not a bulb...
    nrm = hand.nrm[30]  # (on the palm itself: seen from the camera, it is squashed as the palm turns from it)
    squash = math.sqrt(nrm[2] ** 2 + water.WHIP_POOL[1] ** 2 * (1.0 - nrm[2] ** 2))
    disc = np.reshape(water._shape(nrm[:2] / np.hypot(*nrm[:2]), squash, 1.0), (2, 2))
    sv = np.linalg.svd(np.linalg.inv(disc) @ np.reshape(main.shape, (2, 2)), compute_uv=False)
    assert sv[0] < 1.6 * sv[1]  # ... and no pointed head on it (drawn out along the stream, a snake's)
    x, y, rx, ry, _, strength, *_ = sc[30].wet[0]
    assert rx > 1.5 * main.r and strength >= 0.15  # the palm round it wet (as dry as ever, a sticker on it)
    arc = [b.stream[0] for b in tube]
    assert len(arc) > 20 and np.all(np.diff(arc) > 0)  # the stream knows where along itself each part is


def test_the_pool_moves_with_the_palm_and_does_not_turn():
    n = 30
    track = track_of(n, xy=lambda t: (30.0 + 2.0 * t, 130.0))
    _, sc = scenes("whip", voice(n, hum=(2, 29), hits=()), track)
    pool = [b for b in sc[20].blobs if b.flat < 1.0 and not b.stream]
    assert pool and all(abs(b.vx - 2.0) < 1e-6 and b.vy == 0.0 and b.spin == 0.0 for b in pool)  # (still, while
    # the palm moved under it, or turning, the patterns its flow carries were wound into rings: a whorl)


def test_the_held_stream_is_one_connected_stream_rising_out_of_the_palm():
    n = 40
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    p = np.array(track["xy"][30])
    tube = [b for b in sc[30].blobs if b.stream and b.w < 1.0]
    arc = np.array([b.stream[0] for b in tube])
    gaps = np.diff(arc) / np.array([b.r for b in tube[1:]])
    assert gaps.max() < 1.0  # one stream, no gaps along it (faded off the hand, a handle)...
    assert 1.5 * S < arc[-1] < 3.0 * S  # ... one and a half to two palms long...
    assert min(b.y for b in tube) < p[1] - 1.0 * S  # ... standing up out of the palm (round it, a ring lying on it)
    assert all(b.flat == 1.0 for b in tube[len(tube) // 3:])  # round, held up in the air (flattened, a plastic ribbon)


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
    gaps, swell, most = [], [], []
    for k in range(12, 40):
        tube = [b for b in sc[k].blobs if b.stream and b.w < 1.0]
        arc = np.array([b.stream[0] for b in tube])
        r = np.array([b.r for b in tube])
        mid = slice(len(r) // 6, len(r) * 2 // 3)  # past its root, short of its fraying tail
        arc, r = arc[mid], r[mid]
        d = r / np.polyval(np.polyfit(arc, r, 2), arc) - 1.0  # its taper taken out
        swell.append(np.std(d))
        most.append(np.percentile(np.abs(d), 90))  # (a drip swelling where it hangs off the hand aside)
        peaks = arc[1:-1][(d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > 0.0)]
        gaps += list(np.diff(peaks) / np.median(r))
    assert 0.03 < np.mean(swell) < 0.15 and max(most) < 0.35  # broad moving pulses make the flow visible while small
    # irregular bulges remain; deep repeating necks would turn it into ribs or a twisting ribbon
    assert np.std(gaps) > 0.2 * np.mean(gaps)  # ... at no set spacing (ribs: under a twentieth)...
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


def test_a_flung_drop_smears_clear_its_glint_a_thin_bright_line():
    plate = np.full((1, 256, 256, 3), 120, np.uint8)
    drop = water.Blob(128.0, 128.0, 0.0, 3.0, free=True, move=(24.0, 0.0))  # (a small drop, for a hand 160 px)
    sc = water.Scene([drop], wave=0.0)
    out = water.frame(0, plate[0], None, sc, _big_hand(), water.Room(plate), water.Noise(0), 0.0, 0)
    g = out.astype(float).mean(-1)[118:139, 122:135]  # the middle of its smear, across it
    lit = g.mean(1) > 1.04 * 120
    assert g.max() > 1.1 * 120 and 1 <= lit.sum() <= 6  # its glint drawn into a thin bright line...
    assert np.abs(g[~lit].mean(1) - 120).max() < 0.05 * 120  # ... the rest of it clear (with its body's light kept
    # along its smear too, a soft white blob beside its glint's line: sleet)


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
    box, C, A = water._motion_blur(C, A, mx, my, (100, 100 + n, 100, 100 + n), 2000, 2000)
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


def test_torn_water_smears_into_streams_of_their_own_lengths():
    w, d = 240, 90.0  # a strip of water flung down across its length
    C, A = np.full((30, w, 3), 0.3, np.float32), np.ones((30, w), np.float32)
    own = np.zeros((30, w, 2), np.float32)
    own[..., 0] = 0.5  # its own light
    mx, my = np.zeros((30, w), np.float32), np.full((30, w), d, np.float32)

    def ends(rag):  # how far down the frame each column of its smear reaches
        box, _, a = water._motion_blur(C, A, mx, my, (300, 330, 300, 300 + w), 1000, 1000, own,
                                       np.ones(3, np.float32), rag, 0)
        cols = range(300 + 10 - box[2], 300 + w - 10 - box[2])
        return np.array([box[0] + np.flatnonzero(a[:, j] > 0.05).max() for j in cols])

    alike = ends(None)
    torn = ends(np.ones((30, w), np.float32))
    assert np.ptp(alike) <= 1  # moving alike, its streams end in a line...
    assert np.std(torn) > 0.03 * d  # ... torn, each its own length...
    assert np.abs(np.diff(torn)).mean() < 0.15 * np.std(torn)  # ... neighbours alike: a ragged curve (each streak its
    # own length, a row of teeth)


def test_the_held_orbs_highlight_is_broken_by_its_waves_round_where_the_window_is():
    n = 8
    plate = np.repeat(_smooth_plate(), n, 0)
    hand, room, noise = _big_hand(n), water.Room(plate), water.Noise(0)
    lit = []
    for k in range(n):  # its waves running over it at the voice's loudest, frame by frame
        sc = water.Scene([water.Blob(128.0, 140.0, 0.0, 72.0, modes=((3, 0.03, 0.4), (4, 0.012, -0.3)))],
                         wave=water.WAVE[0] + water.WAVE[1])
        lit.append(water.frame(k, plate[k], None, sc, hand, room, noise, 0.0, 0).min(-1) >= 245)
    at = np.array([np.argwhere(m).mean(0) for m in lit])
    assert all(m.sum() > 4 for m in lit)
    assert np.abs(at - at[0]).max() < 0.2 * 72.0  # round where the window is...
    assert min((m & lit[0]).sum() / (m | lit[0]).sum() for m in lit) < 0.6  # ... moving with the waves from frame to
    # frame (the same px on every frame, a glass marble's)


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
    for b in sc[k].blobs:
        if b.free:
            box = water.Field([b], H, W).box
            if box is not None:
                y0, y1, x0, x1 = box
                pad = 3 + int(math.ceil(math.hypot(*b.move))) if b.move else 3
                wall_far[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)] = False
    # Detached water has its own image footprint; the orb's light still belongs only on the palm.
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
    assert np.percentile(c, 99) > 1.3 and np.percentile(c, 10) < 0.9  # ... gathered into bright patches, dim
    # between...
    assert np.all(nets[0][full(plain < 0.05 * plain.max())] == 1.0)  # ... in its core only...
    assert np.abs(nets[1] - nets[0])[full(core)].mean() > 0.015  # ... that shift with the surface from frame to
    # frame
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


# ---- the wet wall, the plain wall -----------------------------------------------------------------------------------

def _patch(sheet: water.Sheet, size: int = 3):
    """The wet patch ``sheet`` leaves (water._soaked), over a grid round its middle: (dx, dy, wetness)."""
    e = int(size * sheet.r)
    yy, xx = np.mgrid[-e:e + 1, -e:e + 1].astype(np.float32)
    din, re = water._inside(xx, yy, sheet.r, sheet.fingers, sheet.wobble)
    wm = water._soaked(sheet, xx, yy, re - np.hypot(xx, yy), water.Noise(0))[0]
    return xx, yy, wm


def test_the_wet_patch_keeps_the_crowns_tongues_and_sags_only_where_it_faces_down():
    r = 40.0
    up = -0.5 * math.pi
    sheet = water.Sheet(0.0, 0.0, r, ((up, 0.1, 0.06, 1.4),), (), 1.0, 0.0, 0.0, 1.0, reach=(0.9,))  # a tongue drained
    xx, yy, wm = _patch(sheet)  # back into its tip after running out 0.9 of the radius
    col = wm[:, xx.shape[1] // 2]
    assert col[np.abs(yy[:, 0] + 1.6 * r) < 1].max() > 0.5  # the wall it ran over stays wet (it drained and the stain
    # lost its tongues: a tombstone)
    sagged = sheet._replace(sag=0.3)
    _, _, low = _patch(sagged)
    row, colm = yy[:, 0], xx.shape[1] // 2

    def reach(m, axis_line, along):  # how far out the patch is wet along a line through its middle
        on = along[m[axis_line] > 0.5]
        return on.max()

    side = reach(wm, (yy == 0), xx[yy == 0]), reach(low, (yy == 0), xx[yy == 0])
    foot = reach(wm, (xx == 0), yy[xx == 0]), reach(low, (xx == 0), yy[xx == 0])
    assert abs(side[1] - side[0]) < 0.06 * r  # its sides stay where they were (pulled down column by column, they
    # were drawn into straight lines: a rectangle)...
    assert foot[1] > foot[0] + 0.2 * r  # ... its foot creeps down
    edge = np.array([reach(wm, np.abs(np.arctan2(yy, xx) - a) < 0.02, np.hypot(xx, yy)[np.abs(np.arctan2(yy, xx) - a)
                                                                                      < 0.02])
                     for a in np.linspace(-0.3 * math.pi, 0.3 * math.pi, 24)])
    assert np.std(edge) > 0.03 * r  # its edge broken, not a compass circle


def test_the_wet_patchs_tongues_narrow_to_points_with_drops_past_their_tips():
    r = 160.0
    sheet = water.Sheet(0.0, 0.0, r, ((0.0, 0.7, 0.05, 1.4),), (), 1.0, 0.0, 0.0, 1.0, dots=((0, 0.15, 0.0, 0.03),))
    xx, yy, wm = _patch(sheet)
    row = wm[np.abs(yy[:, 0]) < 0.5][0]  # along the tongue, through its middle
    x = xx[0]

    def wide(a0, a1):  # how wide it is wet across the tongue from ``a0`` to ``a1`` radii out, on average
        return np.mean([(wm[:, j] > 0.5).sum() for j in np.flatnonzero((x > a0 * r) & (x < a1 * r))])

    assert wide(1.1, 1.25) > 1.4 * wide(1.5, 1.6) > 0  # narrowing to a point (widened, round-tipped: a cartoon's
    # ears)...
    tip = x[(row > 0.5) & (x > 0) & (x < 1.8 * r)].max()
    assert tip < 1.75 * r  # ... as far as it ran...
    dot = (x > 1.8 * r) & (x < 1.92 * r)
    assert row[dot].max() > 0.5 and row[(x > tip + 1) & (x < 1.8 * r - 2)].min() < 0.5  # ... and a drop past its tip


def test_the_wet_patch_is_mottled_and_darker_along_its_edge():
    plate = np.full((1, 256, 256, 3), 150, np.uint8)
    room, noise = water.Room(plate), water.Noise(0)
    sheet = water.Sheet(128.0, 128.0, 70.0, (), (), 3.0, 0.0, 0.0, 1.0)
    out = water.frame(0, plate[0], None, water.Scene(film=[sheet]), _big_hand(), room, noise, 0.0, 0)
    g = out.astype(float).mean(-1)
    inside = cv2.distanceTransform((g < 148).astype(np.uint8), cv2.DIST_L2, 5)  # px in from its outline
    middle = g[inside > 15]
    assert middle.std() > 0.01 * middle.mean()  # mottled (an even tint: a shadow, ink)...
    assert g[(inside >= 4) & (inside <= 9)].mean() < g[(inside >= 20) & (inside <= 30)].mean() - 1  # ... and darker
    # in a band along its edge, just inside where it feathers out, where it dries last


def test_the_wet_patch_soaks_only_into_the_wall_it_landed_on():
    plate = np.full((1, H, W, 3), 140, np.uint8)
    plate[0, 60:72, 70:100] = 245  # a light switch on the wall, inside where it landed
    room, noise = water.Room(plate), water.Noise(0)
    sheet = water.Sheet(64.0, 70.0, 30.0, (), ((2, 0.05, 0.0),), 3.0, 0.0, 0.0, 1.0)
    out = water.frame(0, plate[0], None, water.Scene(film=[sheet]), _hand_at(0.0), room, noise, 0.0, 0).astype(int)
    assert out[70, 50].max() < 135  # the wall round it, wet
    assert np.abs(out[62:70, 78:96] - 245).max() <= 1  # the switch, as it was (stained grey, a dirty smear)


def test_strikes_on_the_same_place_are_one_wet_patch():
    plate = np.full((1, H, W, 3), 150, np.uint8)
    room, noise = water.Room(plate), water.Noise(0)
    sheet = water.Sheet(64.0, 70.0, 25.0, (), ((2, 0.05, 0.0),), 3.0, 0.0, 0.0, 1.0)

    def dark(film):
        out = water.frame(0, plate[0], None, water.Scene(film=film), _hand_at(0.0), room, noise, 0.0, 0)
        return 150.0 - out[66:75, 60:69].astype(float).mean()

    one, three = dark([sheet]), dark([sheet, sheet._replace(x=66.0), sheet._replace(x=62.0)])
    assert 0.0 < one < three < 1.4 * one  # a little darker, not three stains stacked on each other


def test_the_water_stays_only_on_a_plain_wall():
    flat = np.full((H, W, 3), 160, np.uint8)
    assert water._plain(flat, 64.0, 80.0, 15.0) and water._plain(None, 64.0, 80.0, 15.0)
    assert not water._plain(wall(1)[0], 64.0, 80.0, 15.0)  # (coats, a shoe shelf)
    n = 70
    tl = voice(n, hum=(4, 40), hits=((40, 1.0),), aim=0.0)
    track = thrust_track(n)
    hand = water.hand_of(track, n)
    for plate, stays in ((np.repeat(flat[None], n, 0), True), (wall(n), False)):
        water.ROOM_DEPTH, depth = 0.6, water.ROOM_DEPTH
        try:
            sc = water.orb_scenes(tl, track, hand, (H, W), 0, None, plate)
        finally:
            water.ROOM_DEPTH = depth
        assert any(sc[k].film for k in range(n)) == stays  # wet on a plain wall; on anything else, it bursts...
        if not stays:
            assert max(sum(b.free for b in sc[k].blobs) for k in range(40, n)) > 20  # ... and its drops fall


# ---- the held orb ---------------------------------------------------------------------------------------------------

def test_the_held_orbs_rim_leaves_clear_gaps_instead_of_a_broad_white_collar():
    plate = np.full((1, 256, 256, 3), 90, np.uint8)
    plate[:, :20] = 255  # one light over a plain, dark room makes the rim's width readable
    hand, room, noise = _big_hand(), water.Room(plate), water.Noise(0)
    r = 72.0
    sc = water.Scene([water.Blob(128.0, 140.0, 0.0, r)], held=(128.0, 140.0, 0.0, r, 1.0))
    out = lin(water.frame(0, plate[0], None, sc, hand, room, noise, 0.0, 0)).mean(-1)
    angles = np.linspace(-0.5 * math.pi - 0.75, -0.5 * math.pi + 0.75, 80)
    broad = np.array([np.mean([out[int(140 + r * f * math.sin(a)), int(128 + r * f * math.cos(a))]
                              for f in np.linspace(0.92, 0.96, 4)]) for a in angles])
    outer = [out[int(140 + 0.985 * r * math.sin(a)), int(128 + 0.985 * r * math.cos(a))] for a in angles]
    assert np.mean(broad > 0.22) < 1.0 / 3.0  # most of the inner lip stays clear between the short grazing arcs
    assert max(outer) > 0.25  # a few grazing facets still catch the light; the rim has not simply disappeared


def test_the_held_orb_ripples_sparkles_wobbles_and_carries_its_air_round_a_loop():
    n = 60
    tl, track = voice(n, hum=(4, 58), hits=()), track_of(n)
    hand, sc = scenes("orb", tl, track)
    k = 40
    orb = max(sc[k].blobs, key=lambda b: b.r)
    assert len(sc[k].trains) == water.RIPPLES[0]  # small waves running over it in trains...
    lam = [tr[7] / orb.r for tr in sc[k].trains]
    assert all(water.RIPPLES[1][0] <= v <= water.RIPPLES[1][1] for v in lam)
    ran = [sc[j].trains[0][8] for j in (k, k + 1)]
    assert ran[1] > ran[0]  # ... running on from frame to frame
    assert len(sc[k].sparks) == water.SPARKS[0] <= 12  # a few points of light (two dozen were glitter)...
    assert all(-1.0 <= sp[3] <= 1.0 and water.SPARKS[4][0] <= sp[4] <= water.SPARKS[4][1] for sp in sc[k].sparks)
    lit = [{(round(sp[3], 3), round(sp[4], 3)) for sp in sc[j].sparks if sp[6] > 0.3} for j in range(k, k + 4)]
    assert all(lit) and all(a != b for a, b in zip(lit, lit[1:]))  # ... flashing and gone, frame to frame...
    strength = np.array([[sp[6] for sp in sc[j].sparks] for j in range(k, k + 12)])
    assert all(np.diff(np.flatnonzero(s_ > 0.5 * s_.max())).max(initial=0) <= 3 for s_ in strength.T)  # ... each
    # flashing every two or three frames
    assert sc[k].sweep is not None and sc[k + 3].sweep[4] > sc[k].sweep[4]  # a highlight carried round it
    lobes = {l: (a, ph) for l, a, ph in orb.modes}
    assert set(lobes) == {3, 4, 5} and 0.03 < sum(abs(a) for a, _ in lobes.values()) < 0.16  # broad asymmetric lobes
    # to 9 % of its radius...
    nxt = {l: ph for l, _, ph in max(sc[k + 1].blobs, key=lambda b: b.r).modes}
    assert all(abs(nxt[l] - lobes[l][1]) > 0.05 for l in lobes)  # ... its lobes running round it
    c = np.array([orb.x, orb.y])
    ang = []
    for j in range(k, k + 6):
        o = max(sc[j].blobs, key=lambda b: b.r)
        b = sc[j].bubbles[0]
        ang.append(math.atan2(b[1] - o.y, b[0] - o.x))
    turn = np.angle(np.exp(1j * np.diff(ang)))
    assert np.all(turn * np.sign(turn.sum()) > 0)  # its air carried one way round a loop inside it
    assert all(water.BUBBLE_R[0] * S <= b[2] <= water.BUBBLE_R[1] * S for b in sc[k].bubbles)  # small
    assert np.hypot(*(np.array(sc[k].bubbles[0][:2]) - c)) < orb.r


def test_the_orbs_points_of_light_flash_on_its_crests_on_its_side_towards_the_light():
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    crest = np.where(np.abs(np.hypot(xx - 100, yy - 100) - 40) < 3, 1.0, -1.0).astype(np.float32)  # one crest, 40 px
    zero = np.zeros((n, n), np.float32)  # out
    layer = water.Layer((0, n, 0, n), np.zeros((n, n, 3), np.float32), np.ones((n, n), np.float32), zero, zero, zero,
                        crest)
    L = np.array([-0.6, -0.6, 0.53]) / np.linalg.norm([-0.6, -0.6, 0.53])  # up and to the left
    P = np.zeros((n, n, 3), np.float32)
    water.glitter(P, [(100.0, 100.0, 70.0, t, 0.6, 1.0, 1.0) for t in (-1.0, 0.0, 1.0)], layer, np.ones(3, np.float32),
                  L)
    ys, xs = np.nonzero(P[..., 0] > 0.5)
    d = np.hypot(xs - 100, ys - 100)
    assert xs.size and np.all(np.abs(d - 40) < 5)  # on the crest (all over its face, glitter in a snow globe)...
    assert np.all(((xs - 100) * L[0] + (ys - 100) * L[1]) / np.hypot(L[0], L[1]) > 0.6 * d)  # ... towards the light


def test_the_held_orbs_outline_wobbles_in_its_lobes_alone():
    plate = _smooth_plate()
    hand, room, noise = _big_hand(), water.Room(plate), water.Noise(0)
    r = 72.0
    trains = [(128.0, 128.0, r, *water._unit(np.array([math.cos(a), math.sin(a), 0.1])), 0.05, 0.3 * r, 0.0)
              for a in (0.3, 2.4, 4.4)]  # short ripples, at their tallest
    sc = water.Scene([water.Blob(128.0, 128.0, 0.0, r)], trains=trains, wave=water.WAVE[0] + water.WAVE[1])
    fld = water.Field(sc.blobs, 256, 256)
    layer = water.shade(fld, sc, 0, hand, room, noise, water._pyramid(lin(plate[0])), None, 0.0, 0)
    a = np.pad(layer.alpha, 60)
    edge = []
    for t in np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False):  # how far out its outline is, all round
        rr = np.arange(0.8 * r, 1.2 * r, 0.25)
        v = a[(128 + rr * math.sin(t)).astype(int) - layer.box[0] + 60,
              (128 + rr * math.cos(t)).astype(int) - layer.box[2] + 60]
        edge.append(rr[np.argmax(v < 0.5)])
    edge = np.array(edge) - np.mean(edge)
    fine = edge - np.real(np.fft.ifft(np.fft.fft(edge) * (np.abs(np.fft.fftfreq(720, 1 / 720)) <= 6)))
    assert np.std(fine) < 0.012 * r  # no bumps finer than its lobes (a score of them crinkled it: beaten glass)


# ---- drops, the throw -----------------------------------------------------------------------------------------------

def test_a_drop_beside_a_body_of_water_keeps_its_own_size_and_motion():
    big = water.Blob(64.0, 100.0, 0.0, 40.0)
    drop = water.Blob(64.0 + 48.0, 100.0, 0.0, 3.0, free=True, move=(20.0, 0.0))
    fld = water.Field([big, drop], H, W)
    y, x = 100 - fld.box[0], 112 - fld.box[2]
    assert np.sqrt(fld.s2[y, x]) / water.SIG_K < 2.0 * drop.r  # shaded as the drop it is (as the orb's rim: a ring)
    assert fld.mx[y, x] > 0.7 * drop.move[0]  # smeared by its own motion (by the orb's stillness: a round bubble)


def test_drops_flung_off_the_orb_streak_over_a_whole_frame():
    n = 60
    hand, sc = scenes("orb", voice(n, hum=(4, 58), hits=((30, 1.0),), aim=0.0), track_of(n))
    flung = [b for b in sc[31].blobs if b.free and b.move]
    speed = [np.hypot(b.vx, b.vy) for b in flung]
    smear = [np.hypot(*b.move) for b in flung]
    assert len(flung) >= 10 and np.median(np.array(smear) / np.maximum(speed, 1e-6)) > 0.9  # a dash its whole
    # travel long (over half of it, a round ring: a soap bubble)


def test_a_thrown_orb_is_sharp_at_its_front_clear_in_its_core_and_torn_by_the_air():
    n = 60
    hand, sc = scenes("orb", voice(n, hum=(4, 40), hits=((40, 1.0),), aim=90.0), thrust_track(n))
    ball = max(sc[43].blobs, key=lambda b: b.r)
    assert ball.lead == 1.0 and ball.foam <= 0.18  # sharp front, clear body
    before = {(round(b.x), round(b.y)) for b in sc[42].blobs if b.free}
    new = [b for b in sc[43].blobs if b.free and b.r < 0.5 * ball.r and (round(b.x), round(b.y)) not in before]
    assert len(new) >= 3 and all(b.x < ball.x for b in new)  # ... visible drops torn off its back; smaller trail
    # fragments fall below half a pixel on this 128-pixel-wide fixture
    assert np.ptp([b.y for b in new]) > 0.3 * ball.r  # in a cone (in a line, a comet's tail)
    plate = np.full((n, H, W, 3), 128, np.uint8)
    fld = water.Field([ball], H, W)
    layer = water.shade(fld, sc[43], 43, hand, water.Room(plate), water.Noise(0), water._pyramid(lin(plate[43])),
                        None, 0.0, 0)
    P = lin(plate[43])
    y0, y1, x0, x1 = layer.box
    P[y0:y1, x0:x1] = P[y0:y1, x0:x1] * (1.0 - layer.alpha[..., None]) + layer.colour
    out = gam(P).astype(float).mean(-1)
    yy, xx = np.mgrid[0:H, 0:W]
    mid = out[np.hypot(xx - ball.x, yy - ball.y) < 0.4 * ball.r]
    assert mid.mean() < 1.12 * 128  # no uniform milky veil over a featureless wall


def test_the_thrown_orb_casts_its_shadow_on_the_wall_before_it_strikes():
    n = 70
    water.ROOM_DEPTH, depth = 0.6, water.ROOM_DEPTH
    try:
        tl = voice(n, hum=(4, 40), hits=((40, 1.0),), aim=0.0)
        hand, sc = scenes("orb", tl, thrust_track(n))
    finally:
        water.ROOM_DEPTH = depth
    k = next(k for k in range(40, n) if sc[k].film)  # it strikes the wall
    cast = [sc[j].cast for j in range(k - water.WALL_CAST[0], k)]
    assert all(cast) and not sc[k].cast  # over its last frames in the air, not once it has struck
    dark = [c[0][4] for c in cast]
    ahead = [c[0][2] for c in cast]
    assert np.all(np.diff(dark) > 0) and np.all(np.diff(ahead) < 0) and ahead[-1] == 1  # darker as it nears
    sheet = next(it for it in sc[k].film if isinstance(it, water.Sheet))
    assert np.hypot(cast[-1][0][0] - sheet.x, cast[-1][0][1] - sheet.y) < sheet.r  # where it strikes


# ---- the whip, held -------------------------------------------------------------------------------------------------

def test_the_held_stream_spills_drops_that_fall_back_into_its_pool(monkeypatch):
    monkeypatch.setattr(water, "WHIP_SPILL", water.WHIP_SPILL[:1] + ((0.03, 0.05),) + water.WHIP_SPILL[2:])  # (seen on
    n = 40  # a hand this small)
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 39), hits=()), track)
    tube = [b for b in sc[30].blobs if b.stream and b.w < 1.0]
    assert tube[0].r < 1.6 * water.WHIP_R[0] * S * (1.0 + water.WHIP_FLARE[0])  # a slender stream (a sculpture)
    falling = [[b for b in sc[k].blobs if b.free and b.r >= 0.03 * S and b.move] for k in range(24, 34)]
    assert np.mean([len(f) for f in falling]) >= water.WHIP_SPILL[0][0]  # drops spilling off it, falling...
    assert any(sc[k].plops for k in range(20, n))  # ... and rings where they land in the pool


def test_the_pool_drains_after_a_lash_and_leaves_the_palm_wet_with_drops():
    n = 60
    track = track_of(n)
    _, sc = scenes("whip", voice(n, hum=(2, 30), hits=((24, 1.0),)), track)

    def pool(k):
        return [b for b in sc[k].blobs if b.flat < 0.1 and not b.stream and not b.free]

    assert pool(23)  # a pool while it pours...
    k = next(k for k in range(24, n) if not pool(k))
    assert k - 24 <= math.ceil(water.WHIP_DRAIN) + 1  # ... drained to a film within a few frames of the lash (a clear
    # disc, then a ring, left on the palm)
    late = 30 + water.LINGER + 8
    assert not pool(late) and sc[late].wet and sc[late].wet[0][5] > 0.5 * water.WHIP_WET[1]  # the skin left wet...
    left = [b for b in sc[late].blobs if b.free and b.flat < 1.0]
    assert water.WHIP_LEFT[0][0] <= len(left) <= water.WHIP_LEFT[0][1]  # ... with a few drops on it
    assert all(b.glint <= 0.3 for b in left)


def test_wall_tongues_keep_a_broad_dark_root_after_the_body_starts_drying():
    n = 90
    sc = [water.Scene() for _ in range(n)]
    drops = water.Drops(None, (1200, 1200), 100.0)
    water._wall(sc, drops, (0, 600.0, 600.0, -100.0, 60.0, 100.0, np.array([0.0, -1.0]), 1.0),
                n, None, np.random.default_rng(8))
    sheet = next(it for it in sc[65].film if isinstance(it, water.Sheet))
    broad = [(a, ln, hw) for (a, _, hw, _), ln in zip(sheet.fingers, sheet.reach) if hw >= 0.075 and ln >= 0.45]
    assert len(broad) >= 4  # a few substantial tongues carry the late silhouette, rather than a square with pin rays
    r = 100.0
    plain = water.Sheet(0.0, 0.0, r, ((0.0, 0.1, 0.1, 1.4),), (), 1.0, 0.0, 0.0, 1.0,
                        reach=(0.8,), dry=1.0)
    xx, yy, wet = _patch(plain)
    fresh = _patch(plain._replace(dry=0.0))[2]
    body = (np.hypot(xx, yy) < 0.5 * r)
    root = (xx > 1.05 * r) & (xx < 1.35 * r) & (np.abs(yy) < 0.02 * r)
    assert wet[root].mean() > 0.7 * wet[body].mean()
    assert (wet[root] / fresh[root]).mean() >= (wet[body] / fresh[body]).mean()


def test_a_stream_root_feathers_into_its_pool_without_a_solid_bottom_edge():
    y = np.linspace(0.0, 180.0, 181)
    pts = np.c_[np.zeros_like(y), -y, np.zeros_like(y)]
    zeros = np.zeros_like(pts)
    blobs = water._stream(pts, zeros, np.full(len(y), 12.0), np.zeros(len(y)), zeros[:, :2],
                          y, np.ones(len(y)), root_feather=24.0)
    assert blobs[0].w < 0.1 * blobs[5].w  # no opaque straight cut across the root on the palm
    assert blobs[1].w < blobs[2].w < blobs[3].w  # the contact becomes a stream over at least 20 picture pixels
    assert blobs[5].w == pytest.approx(blobs[6].w)


def test_a_lash_breaks_into_scattered_short_drops_instead_of_a_zipper():
    n = 24
    hand = water.hand_of(track_of(n), n)
    root = np.array([64.0, 130.0, 0.0])
    u = np.linspace(0.0, 1.0, water.WHIP_NODES)
    x = root + np.c_[12.0 * np.sin(u * 3.0), -100.0 * u, np.zeros(len(u))]
    flung = water._fling(0, x, np.zeros_like(x), np.array([0.0, -1.0]), 1.0, S, -1500.0, (H, W))
    drops = water.Drops(None, (H, W), S)
    sc = [water.Scene() for _ in range(n)]
    rng = np.random.default_rng(4)
    for k in range(3):
        water._lash(flung, k, root, S, (H, W), drops, sc, n, None, rng, np.zeros(65), 0.13)
    assert len(drops.x) > 10
    assert drops.info[:, 5].max() <= 2.0  # a few short ligaments survive; no long barbs across the broken arc
    side_speed = drops.v[:, 0]
    assert np.std(side_speed) > 0.03 * S  # drops separate sideways instead of retaining a single dotted path
