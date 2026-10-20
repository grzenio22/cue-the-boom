"""Lightning made from the hand (cuesync.lightning): arcs from a fingertip and a ball over a palm, on small synthetic
plates, tracks and mattes."""

import cv2
import numpy as np
import pytest

from cuesync import lightning as L
from cuesync.elements import Timeline
from cuesync.transfer import lin

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
W, H = 270, 480  # a quarter of a phone's 1080x1920


def flashes(frames: np.ndarray, under: np.ndarray | None = None, fps: int = 24) -> float:
    """The most flashes in any second over a large area of the picture (WCAG's general flash, for a phone held at
    reading distance): a flash is a pair of opposing changes in relative luminance of 0.1 or more where the darker is
    under 0.8, counted per pixel between its turning points; a large area is a quarter of a square 0.75 of the
    picture's short side across (a quarter of a 10 degree field). Half the most changes that a quarter of some such
    square makes within a second. With ``under`` (the plate), only the changes the effect made: its luminance over the
    plate's (a hand swung across a white wall flashes on its own)."""
    q = max(1, min(frames.shape[1:3]) // 120)
    Yabs = lin(frames[:, ::q, ::q]) @ LUMA
    Y = Yabs if under is None else Yabs - lin(under[:, ::q, ::q]) @ LUMA
    n, h, w = Y.shape
    ev = np.zeros(Y.shape, bool)
    low, low_abs = Y[0].copy(), Yabs[0].copy()  # the last turning point at the bottom (and how bright it was), or the
    top = Y[0].copy()  # lowest yet; the last at the top, or the highest yet
    way = np.zeros((h, w), np.int8)  # 0 undecided, 1 rising, -1 falling
    for t in range(1, n):
        y, ya = Y[t], Yabs[t]
        lower = (way <= 0) & (y < low)
        low, low_abs = np.where(lower, y, low), np.where(lower, ya, low_abs)
        top = np.where((way >= 0) & (y > top), y, top)
        down = (way >= 0) & (top - y >= 0.1) & (ya < 0.8)
        up = (way <= 0) & (y - low >= 0.1) & (low_abs < 0.8) & ~down
        ev[t] = up | down
        way[down], way[up] = -1, 1
        low, low_abs = np.where(down, y, low), np.where(down, ya, low_abs)
        top = np.where(up, y, top)
    side = max(1, int(0.75 * min(h, w)))
    c = np.concatenate([np.zeros((1, h, w), np.int32), np.cumsum(ev, 0, dtype=np.int32)])
    most = 0
    for t in range(max(1, n - fps + 1)):
        count = c[min(n, t + fps)] - c[t]
        k = most + 1
        while k <= count.max():
            share = cv2.blur((count >= k).astype(np.float32), (side, side))
            inside = share[side // 2: h - (side - 1) // 2, side // 2: w - (side - 1) // 2]
            if inside.size == 0 or inside.max() < 0.25:
                break
            most, k = k, k + 1
    return most / 2.0


# ---------------------------------------------------------------- a synthetic hand

def landmarks(cx: float, cy: float, s: float, middle_out: bool = False) -> np.ndarray:
    """21 MediaPipe landmarks (x, y, z px) of a right hand held up, the palm to the camera, its size ``s`` (wrist to
    middle knuckle), the middle knuckle at (cx, cy): the index out straight, the middle finger out beside it or curled
    into the palm (a finger gun), ring and little fingers out."""
    lm = np.zeros((21, 3))
    lm[0] = cx, cy + s, 0
    lm[1:5] = [(cx - 0.35 * s, cy + 0.75 * s, 0), (cx - 0.5 * s, cy + 0.55 * s, 0), (cx - 0.62 * s, cy + 0.4 * s, 0),
               (cx - 0.72 * s, cy + 0.25 * s, 0)]
    for k, (dx, out) in enumerate(((-0.3, True), (0.0, middle_out), (0.25, True), (0.45, True))):
        b = 5 + 4 * k
        x = cx + dx * s
        if out:
            lm[b: b + 4] = [(x, cy + 0.05 * s, 0), (x, cy - 0.35 * s, 0), (x, cy - 0.6 * s, 0), (x, cy - 0.8 * s, 0)]
        else:  # curled: the tip back down by the knuckle
            lm[b: b + 4] = [(x, cy + 0.05 * s, 0), (x, cy - 0.25 * s, 0), (x, cy - 0.1 * s, 0), (x, cy + 0.1 * s, 0)]
    return lm


def take(n: int, at, s: float = 60.0, emitter: str = "index", middle_out: bool = False, palm=(225, 150, 130),
         wall=(150, 142, 128), texture: float = 18.0):
    """(plate uint8 [n, H, W, 3], track dict, matte uint8 [n, H, W]): a wall and a hand held up, both with ``texture``
    (code values), whose middle knuckle is at ``at(t)``, the emitter on its index fingertip or on its palm."""
    rng = np.random.default_rng(5)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.5) * texture
    plate = np.zeros((n, H, W, 3), np.uint8)
    mask = np.zeros((n, H, W), np.uint8)
    xy, lms = [], []
    for t in range(n):
        lm = landmarks(*at(t), s, middle_out)
        lms.append(lm.tolist())
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(lm[:, :2].astype(np.int32)), 255)
        for f in (8, 12, 16, 20, 4):
            if f != 12 or middle_out:
                cv2.line(m, tuple(int(v) for v in lm[f - 3, :2]), tuple(int(v) for v in lm[f, :2]), 255, int(0.18 * s))
        cv2.line(m, tuple(int(v) for v in lm[0, :2]), (int(lm[0, 0]), H), 255, int(0.7 * s))  # the forearm
        mask[t] = m
        frame = np.empty((H, W, 3), np.float32)
        frame[:] = wall
        frame[m > 0] = palm
        plate[t] = np.clip(frame + tex[..., None], 0, 255).astype(np.uint8)
        xy.append(lm[8, :2] if emitter == "index" else lm[[0, 5, 9, 13, 17], :2].mean(0))
    track = {"version": 1, "fps": 24, "W": W, "H": H, "emitter": "palm", "hand": "right",
             "xy": [list(map(float, p)) for p in xy], "dir": [[0.0, -1.0]] * n, "scale": [s] * n, "conf": [1.0] * n,
             "src": ["mp"] * n, "speed": [0.0] * n, "landmarks": lms, "gesture": ["None"] * n, "events": []}
    return plate, track, mask


def voice(energy, hits=(), sizes=None, **kw) -> Timeline:
    e = np.asarray(energy, np.float32)
    return Timeline(n=len(e), energy=e, hits=list(hits), sizes=list(sizes or [1.0] * len(hits)), **kw)


def mains(bolts: list, root: np.ndarray, s: float) -> list[np.ndarray]:
    """The far ends of the streams struck from ``root`` on a frame (the channels leaving it more than a hand long: a
    crackle at the tip is a fraction of one)."""
    return [p[-1] for p, _, it in bolts if np.allclose(p[0], root, atol=1e-6) and np.linalg.norm(p[-1] - root) > s]


# ---------------------------------------------------------------- arcs

def test_arcs_are_rooted_on_the_moving_fingertip_and_their_far_ends_hold_a_frame_or_two():
    n, s = 30, 60.0
    plate, track, mask = take(n, lambda t: (110 + 3 * t, 330), s)
    hand = L._hand(plate, track)
    bolts, _ = L.arc_frames(hand, voice(np.full(n, 0.8)), seed=3)
    ends = []
    for t in range(n):
        tip = np.asarray(track["landmarks"][t][8][:2])
        e = mains(bolts[t], tip, s)
        assert e, f"no stream from the fingertip on frame {t}"
        if t:  # none left behind where the fingertip was a frame ago
            assert not mains(bolts[t], np.asarray(track["landmarks"][t - 1][8][:2]), s)
        ends.append(e)
    held = [sum(any(np.allclose(a, b) for b in ends[t + 1]) for a in ends[t]) / len(ends[t]) for t in range(n - 1)]
    assert np.mean(held) > 0.3  # most streams are re-struck down the same path on the next frame...
    for t in range(n - 3):  # ...but none for longer than 3 frames
        assert not any(all(any(np.allclose(a, b) for b in ends[t + k]) for k in range(1, 4)) for a in ends[t])
    for t in range(n - 2):  # and a new path is found at least every 3 frames
        assert any(not any(np.allclose(a, b) for b in ends[t]) for k in (1, 2) for a in ends[t + k])


def test_a_handheld_camera_carries_a_held_far_end_with_the_room():
    n, s = 24, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    cam = np.repeat(np.eye(3)[None], n, 0)
    cam[:, 0, 2] = 2.0 * np.arange(n)  # the room drifts 2 px right a frame
    bolts, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 0.8), camera=cam), seed=4)
    tip = np.asarray(track["landmarks"][0][8][:2])
    ends = [mains(b, tip, s) for b in bolts]
    moved = sum(any(np.allclose(a + (2.0, 0.0), b) for b in ends[t + 1]) for t in range(n - 1) for a in ends[t])
    stayed = sum(any(np.allclose(a, b) for b in ends[t + 1]) for t in range(n - 1) for a in ends[t])
    assert moved > 5 and stayed == 0


def test_a_louder_voice_strikes_more_streams_further():
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])

    def strikes(level):
        bolts, _ = L.arc_frames(hand, voice(np.full(n, level)), seed=6)
        ends = [e for e in (mains(b, tip, s) for b in bolts) if e]  # (on a frame all its streams may be dark)
        return np.mean([len(e) for e in ends]), np.mean([max(np.linalg.norm(a - tip) for a in e) for e in ends]) / s

    (few, near), (many, far) = strikes(0.35), strikes(1.0)
    assert many >= few + 1 and far > 1.7 * near  # (the main channel: the streamers beside it are shorter)


def test_a_strike_is_one_main_channel_with_dim_thin_branches():
    """A loud voice is one channel with branches, not a rope of equal wires: the streamers beside a finger's main
    channel are at most half as thick, a bolt's forks are much dimmer and thinner and leave along its way, and drawn,
    only the main channel clips to white."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    bolts, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 1.0)), seed=6)
    tip = np.asarray(track["landmarks"][0][8][:2])
    main_w, seen = L._width_k(s) * L.ARC_WIDTH[1], 0
    for frame in bolts:
        widths = [wd for p, wd, _ in frame if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
        main = sum(np.isclose(wd, main_w) for wd in widths)
        assert main <= 1 and all(np.isclose(wd, main_w) or wd <= L.SIDE_WIDTH * main_w + 1e-9 for wd in widths)
        seen += main
    assert seen > 0.8 * n
    tree = L.bolt_tree((0.0, 0.0), (0.0, -1.0), 400.0, np.random.default_rng(3), 4.0, fork=0.3)
    assert len(tree) > 3
    for p, wd, it in tree[1:]:
        assert it <= L.FORK_INTEN[1] + 1e-9 and wd <= L.FORK_WIDTH * 4.0 + 1e-9
        d = p[-1] - p[0]
        assert np.degrees(np.arccos(-d[1] / np.linalg.norm(d))) < 90  # (none heads back towards the hand)
    main = np.stack([np.full(60, 50.0), np.linspace(10, 90, 60)], 1)
    fork = np.stack([np.full(60, 120.0), np.linspace(10, 90, 60)], 1)
    E, _ = L.layer([(main, 4.0, 1.0), (fork, 4.0 * L.FORK_WIDTH, L.FORK_INTEN[1])], [], None, 60.0, (0, 100, 0, 170),
                   L.GLOW_CAP["arc"], 0.0, np.zeros(2), [])
    assert E[50, 50, 0] > 1.0 > E[50, 120, 0]  # (red: the glow round both is blue, a white needs red as well)


def test_streamers_leave_well_off_the_main_channel_on_either_side_and_three_only_at_the_peak():
    """A streamer leaves its finger at least SIDE_GAP's first off its main channel's heading, the two on opposite
    sides (bunched within a few degrees, they ran beside it as a rope); under the voice's peak there is one at most."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    main_w = L._width_k(s) * L.ARC_WIDTH[1]
    bolts, _ = L.arc_frames(hand, voice(np.full(n, 1.0)), seed=6)
    for frame in bolts:
        main = [p[-1] - tip for p, wd, _ in frame if np.allclose(p[0], tip) and np.isclose(wd, main_w)]
        sides = [p[-1] - tip for p, wd, _ in frame if np.allclose(p[0], tip) and not np.isclose(wd, main_w)
                 and np.linalg.norm(p[-1] - tip) > s]
        assert len(main) == 1 and len(sides) <= 2
        m = main[0]
        off = [np.degrees(np.arctan2(m[0] * d[1] - m[1] * d[0], m @ d)) for d in sides]
        assert all(abs(a) > L.SIDE_GAP[0] - 1.0 for a in off)
        assert len(off) < 2 or off[0] * off[1] < 0
    bolts, _ = L.arc_frames(hand, voice(np.full(n, 0.6)), seed=6)
    assert max(len(mains(b, tip, s)) for b in bolts) == 2


def test_the_crawling_arc_waits_for_a_steady_finger():
    """An arc crawls up the finger to its tip while the finger is steady, none while it moves faster than
    CRAWL_STILL (smeared with it, the arc was a pale band across the finger)."""
    n, s = 20, 60.0

    def crawls(speed):
        plate, track, _ = take(n, lambda t: (100 + speed * t, 330), s)
        bolts, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 0.9)), seed=2)
        tips = [np.asarray(track["landmarks"][t][8][:2]) for t in range(n)]
        return sum(np.allclose(p[-1], tips[t]) and not np.allclose(p[0], tips[t]) for t in range(n)
                   for p, _, _ in bolts[t])

    assert crawls(0.0) > n // 3 and crawls(8.0) == 0  # (CRAWL_STILL is 3.75 px a frame at 270 wide)


def test_a_fast_finger_keeps_its_bolt_joined_to_the_tip_and_a_small_bolt_glows_hot():
    """Smeared along a fast finger's motion, a channel is smeared at most SMEAR_MAX and stays at least WELD_KEEP of
    itself at the tip; a thin channel struck hot glows as if GLOW_FLOOR of a loud one's width, a dim one as it is."""
    s, box, cap = 60.0, (0, 120, 0, 120), L.GLOW_CAP["arc"]
    bolt = np.stack([np.full(50, 60.0), np.linspace(100.0, 20.0, 50)], 1)  # straight up from (60, 100)
    core = L.draw([(bolt, 2.0, 1.0)], box)[..., 0]
    fast = L._smear(core[..., None].copy(), box, np.array([40.0, 0.0]), [(60.0, 100.0)], s, L.SMEAR_MAX)[..., 0]
    assert fast[97:100].max() >= L.WELD_KEEP * core[97:100].max() - 1e-6  # (welded at the tip)
    assert (fast[80] > 0.01).sum() <= L.SMEAR_MAX + 6  # (a 2 px line smeared 12 px, not 40)
    floor = L.GLOW_FLOOR * L.ARC_WIDTH[1]
    for inten, more in ((0.9, 1.5), (0.3, 1.0)):
        thin = [(bolt, 1.3, inten)]
        plain, _ = L.layer(thin, [], None, s, box, cap, 0.0, np.zeros(2), [])
        hot, _ = L.layer(thin, [], None, s, box, cap, 0.0, np.zeros(2), [], floor=floor)
        assert hot[60, 68, 2] >= more * plain[60, 68, 2] - 1e-6  # (its glow, 8 px off it)
        if more == 1.0:
            assert np.allclose(hot, plain)


def test_the_arcs_light_the_room_blue_white_and_the_hand_violet(monkeypatch):
    """On a grey wall the arcs' light is a cool blue-white, with more green than red in it (in the hand's violet, a
    wall went mauve); on the hand it is that violet (a greener blue on skin at the top of red went cyan)."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0))
    n = 8
    plate, track, mask = take(n, lambda t: (120, 330), wall=(150, 150, 150), texture=0.0)
    out = L.render(plate, track, mask, voice(np.full(n, 0.9), hits=[0]), "arc", seed=2)
    d = (lin(out[4:]) - lin(plate[4:])).mean(0)
    wall = np.zeros((H, W), bool)
    wall[400:, 210:] = True  # (the wall below and right of the hand, away from the channels' own glow)
    wall &= mask[0] == 0
    hand = (mask[0] > 0) & (np.indices((H, W))[0] < 380)
    dw, dh = d[wall].mean(0), d[hand].mean(0)
    assert dw[2] > dw[1] > dw[0] > 0
    assert dh[2] > dh[0] > dh[1]


def up_of(ends: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Degrees clockwise from straight up, from ``tip`` to each of ``ends``."""
    return np.degrees(np.arctan2(ends[:, 0] - tip[0], tip[1] - ends[:, 1]))


def test_the_arcs_leave_along_the_finger_and_from_two_fingers_held_together(monkeypatch):
    """Two fingers out together (the index left of the middle): each strikes its main channel along the fingers, the
    two turned apart so they separate; the second strikes no streamers of its own. They light the room as one finger
    does, not twice as much."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0))  # (the light alone)
    n, s = 12, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True)
    hand = L._hand(plate, track)
    assert L._fingers(hand, 0) == [8, 12]
    tl = voice(np.full(n, 0.9))
    bolts, _ = L.arc_frames(hand, tl, seed=1)
    main_w = L._width_k(s) * (L.ARC_WIDTH[0] + (L.ARC_WIDTH[1] - L.ARC_WIDTH[0]) * 0.9)
    lean = {}
    for f in (8, 12):
        tip = np.asarray(track["landmarks"][0][f][:2])
        ends = np.array([p[-1] for b in bolts for p, wd, _ in b if np.allclose(p[0], tip) and np.isclose(wd, main_w)])
        assert len(ends) >= n - 1
        up = up_of(ends, tip)
        assert abs(np.median(up)) < 15 and np.mean(np.abs(up) < 25) > 0.9  # the fingers point straight up
        lean[f] = np.median(up)
    assert lean[8] < lean[12] - 5  # turned apart: the index's to the left, the middle's to the right
    tip12 = np.asarray(track["landmarks"][0][12][:2])
    assert all(len(mains(b, tip12, s)) <= 1 for b in bolts)  # (its main channel only)
    _, track1, _ = take(n, lambda t: (135, 330), s)
    assert L._fingers(L._hand(plate, track1), 0) == [8]  # a finger gun: the middle finger is curled
    room = mask[0] == 0
    room[:, 180:] = False  # (the wall well away from the channels' own glow)
    room[:200] = False

    def lit(tr):
        out = L.render(plate, tr, mask, tl, "arc", seed=1)
        return (lin(out[6:]) @ LUMA - lin(plate[6:]) @ LUMA)[:, room].mean()

    assert lit(track) < 1.3 * lit(track1)


def test_a_crackle_stays_at_the_fingertip_a_moment_after_the_voice_stops():
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    e = np.zeros(n)
    e[:8] = 0.9
    bolts, spots = L.arc_frames(L._hand(plate, track), voice(e), seed=2)
    tip = np.asarray(track["landmarks"][0][8][:2])
    assert all(mains(bolts[t], tip, s) for t in range(8))
    for t in (8, 9, 10):  # the voice has stopped: no stream, a crackle a fraction of a hand long, the tip burning
        assert bolts[t] and not mains(bolts[t], tip, s)
        assert max(np.linalg.norm(p - tip, axis=1).max() for p, _, _ in bolts[t]) < 0.8 * s
        assert spots[t] and spots[t][0][3] > 0
    assert not any(bolts[t] for t in range(20, n))


# ---------------------------------------------------------------- the shot

def test_same_inputs_give_the_same_bytes_and_silence_leaves_the_plate_alone():
    n = 12
    plate, track, mask = take(n, lambda t: (110 + 2 * t, 330))
    e = np.r_[np.zeros(3), np.full(6, 0.9), np.zeros(3)]
    a = L.render(plate, track, mask, voice(e, hits=[3]), "arc", seed=7)
    assert a.shape == plate.shape and a.dtype == np.uint8
    assert np.array_equal(a, L.render(plate, track, mask, voice(e, hits=[3]), "arc", seed=7))
    assert not np.array_equal(a, L.render(plate, track, mask, voice(e, hits=[3]), "arc", seed=8))
    assert np.array_equal(a[:3], plate[:3])  # before the voice: the plate's own bytes
    assert not np.array_equal(a[4], plate[4])
    assert np.array_equal(L.render(plate, track, mask, voice(np.zeros(n)), "ball"), plate)


def test_a_stuttering_voice_flickers_the_arcs_but_not_the_room():
    """A voice that comes and goes six times a second ("ka-ka-ka"): every onset strikes afresh, the arcs flash and go
    dark with it, but the light they throw on the hand and the room changes at most 3 times a second."""
    n = 48
    plate, track, mask = take(n, lambda t: (120 + t, 330), wall=(200, 196, 188))
    e = np.where(np.arange(n) % 4 < 2, 1.0, 0.0)
    out = L.render(plate, track, mask, voice(e, hits=list(range(0, n, 4))), "arc", seed=1)
    assert flashes(out, plate) <= 3.0
    assert np.abs(out.astype(int) - plate).max() > 100  # (and the arcs themselves are there)
    strobe = plate.copy()
    strobe[::2] = 250  # (the meter sees a strobe)
    assert flashes(strobe, plate) >= 10.0


def test_the_light_dies_with_the_channels():
    """The light on the hand and the room follows what can be seen: a few frames after the voice stops (the crackle at
    the tip left), it is a fraction of what it was, and once that is gone and the camera has come back the frames are
    the plate's own."""
    n = 40
    plate, track, mask = take(n, lambda t: (120, 330), wall=(170, 160, 150))
    e = np.r_[np.full(10, 0.9), np.zeros(n - 10)]
    out = L.render(plate, track, mask, voice(e, hits=[0]), "arc", seed=3)
    lit = (lin(out) @ LUMA - lin(plate) @ LUMA).clip(0, None)
    wall = mask[0] == 0
    assert lit[15][wall].mean() < 0.25 * lit[8][wall].mean()
    assert np.array_equal(out[32:], plate[32:])


def test_the_light_only_adds_to_the_plate(monkeypatch):
    """With the camera's exposure held, no pixel is darker than the plate's: the roll-off to white that softens the
    channels never dims the plate's own highlights."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0))
    n = 8
    plate, track, mask = take(n, lambda t: (120, 330), wall=(250, 248, 245))
    plate[:, :60] = 255  # a window blown out to white
    out = L.render(plate, track, mask, voice(np.full(n, 0.9), hits=[0]), "arc", seed=2)
    assert (out.astype(int) >= plate).all() and (out[:, :60] == 255).all()


def test_the_ball_shows_from_a_soft_start_without_sputtering_and_hovers_off_the_palm():
    """A voice that begins softly and dips for a frame before it swells shows the ball on every frame from its start,
    fading in; a palm seen from above (its fingers further from the camera than its wrist) holds the ball further up
    the picture than the palm's middle, over it."""
    n, s = 20, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    for lm in track["landmarks"]:
        for p in lm:
            p[2] = 1.2 * (300 + s - p[1])  # (the higher in the picture, the further away)
    e = np.r_[np.zeros(3), 0.21, 0.13, 0.2, np.full(n - 6, 0.9)]
    hand = L._hand(plate, track)
    _, sparks, balls = L.ball_frames(hand, voice(e), seed=1)
    assert all(b is None for b in balls[:3]) and all(b is not None for b in balls[3:])
    assert balls[3][3] < balls[5][3] < balls[10][3]  # faded in
    for t in range(3, n):
        assert track["xy"][t][1] - balls[t][1] > 0.1 * s and abs(balls[t][0] - track["xy"][t][0]) < 0.05 * s


def test_the_ball_grows_with_the_held_voice_and_shrinks_when_it_stops():
    n, s = 40, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.r_[np.zeros(4), np.full(24, 0.9), np.zeros(12)]
    _, _, balls = L.ball_frames(L._hand(plate, track), voice(e), seed=0)
    assert all(b is None for b in balls[:4])
    r = [b[2] / s for b in balls[4:28]]
    lo, hi = L.BALL_R
    assert r[0] < lo + 0.05 and r[-1] > lo + 0.6 * (hi - lo) and np.all(np.diff(r) > 0)
    assert balls[31] is None or balls[31][2] < 0.7 * balls[27][2]


def test_the_ball_is_a_blue_light_over_the_palm_that_keeps_its_lines():
    """A palm already at the top of red in the picture: the ball is the brightest thing over it and reads blue, not a
    white disc; the palm's lines show faintly through its dense body and in full just past its outline, which is lit,
    not veiled; the camera exposes down for it, and the wall round it is lit blue against the room further off."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    flat, _, _ = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), texture=0.0)
    out_flat = L.render(flat, track, mask, tl, "ball", seed=0)  # the same lightning over a palm with no lines
    t = n - 1
    c = np.asarray(track["xy"][t])
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - c[0], yy - c[1]) / s
    on = mask[t] > 0
    under, round_it = (d < 0.35) & on, (d > 0.7) & (d < 1.0) & on
    o, p = out[t].astype(np.float32), plate[t].astype(np.float32)
    lum = lin(out[t]) @ LUMA
    assert lum[(d < 0.25) & on].mean() > 1.25 * lum[round_it].mean()  # a light: the brightest thing there (its
    # saturated blue has little luminance of its own)
    white = (o > 245).all(-1)
    assert white[under].mean() < 0.1  # its white-hot middle and channels, not a white disc
    assert (o[under, 2] - o[under, 0]).mean() > (p[under, 2] - p[under, 0]).mean() + 40  # blue
    lines = (o - out_flat[t]).mean(-1), (p - flat[t]).mean(-1)  # the palm's lines, as they come through
    assert np.corrcoef(lines[0][under], lines[1][under])[0, 1] > 0.8
    assert 0.1 < np.std(lines[0][under]) / np.std(lines[1][under]) < 1.2
    R = L.ball_frames(L._hand(plate, track), tl, 0)[2][t][2] / s
    round_it = (d > 1.15 * R) & (d < 1.6 * R) & on  # the palm just past its outline: lit, not veiled or milky
    assert lum[round_it].mean() > (lin(plate[t]) @ LUMA)[round_it].mean()
    assert np.std(lines[0][round_it]) / np.std(lines[1][round_it]) > 0.7
    near, far = (d > 0.6) & (d < 1.0) & ~on, (d > 1.8) & ~on
    near, far = (o - p)[near], (o - p)[far]
    assert far.mean() < -5  # the camera exposed down for it...
    assert (near[:, 2] - near[:, 0]).mean() > (far[:, 2] - far[:, 0]).mean() + 4  # ...and it lights the wall blue


def test_the_ball_settles_the_exposure_once_for_a_phrase(monkeypatch):
    """Three sounds with a second's gap between each: the camera exposes down once, by at most EXPOSURE's stops, and
    holds there through the gaps instead of breathing with each sound (by 0.6 stops three times in 4 s, it pumped
    the whole room)."""
    depth, hand_gain, _, near_gain = L.LIGHT["ball"]
    monkeypatch.setitem(L.LIGHT, "ball", (depth, hand_gain, 0.0, near_gain))  # (the camera alone, not the ball's light)
    n, s = 80, 80.0
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.zeros(n)
    e[4:20], e[34:48], e[60:76] = 0.9, 0.9, 0.9
    out = L.render(plate, track, mask, voice(e, hits=[4, 34, 60]), "ball", seed=0)
    far = np.zeros((H, W), bool)
    far[:70] = True  # (the wall at the top, far from the ball)
    stops = np.log2((lin(out) @ LUMA)[:, far].mean(1) / (lin(plate) @ LUMA)[:, far].mean(1))
    assert -L.EXPOSURE["ball"][0] - 0.03 < stops.min() < -0.2
    assert np.ptp(stops[20:78]) < 0.05  # (through both gaps)


def test_the_balls_glare_is_the_same_light_over_skin_and_wall():
    """The glare is screened over the picture as shown: either side of an outline, at the same distance from the
    ball, it takes the same share of what is left to white, and that share is bluest."""
    frame = np.zeros((101, 101, 3), np.uint8)
    frame[:, :50], frame[:, 50:] = (225, 150, 130), (110, 104, 90)  # a palm, an olive wall
    lit = frame.copy()
    L.glare(lit, (50.0, 50.0, 20.0, 1.0, 1.0, 1.0))
    share = (lit.astype(float) - frame) / (255.0 - frame)
    assert np.allclose(share[50, 40], share[50, 60], atol=0.025)  # (code values: a share of 30 is 1/60 a step)
    assert share[50, 40, 2] > share[50, 40, 1] > share[50, 40, 0] > 0


def test_the_balls_light_wraps_the_edges_of_the_fingers_facing_it():
    """A finger left of the ball catches its light on its right edge, facing the ball, not on its left one."""
    m = np.zeros((120, 160), np.float32)
    m[:, 40:60] = 1.0  # a finger; the ball's middle at (100, 60)
    box, wl = L.wrap_light(m, (100.0, 60.0, 20.0, 1.0, 1.0, 1.0), 270)
    lit = np.zeros_like(m)
    lit[box[0]:box[1], box[2]:box[3]] = wl
    assert lit[60, 56:60].max() > 0.2 and lit[60, 40:45].max() < 0.02
    assert not lit[:, :40].any() and not lit[:, 60:].any()  # (on the hand only)


def test_a_knotted_filament_is_drawn_again():
    t = np.linspace(0, 4 * np.pi, 200)
    assert L.knotted(np.stack([np.cos(t) * t, np.sin(t) * t], 1))  # a spiral
    assert not L.knotted(L.displace((0.0, 0.0), (1.0, 0.0), np.random.default_rng(1), 0.1, 0.02))
    assert sum(L.knotted(L._filament(np.random.default_rng(i))[0][0]) for i in range(200)) <= 4


def test_bad_inputs_are_refused():
    plate, track, mask = take(4, lambda t: (120, 330))
    with pytest.raises(ValueError, match="form"):
        L.render(plate, track, mask, voice(np.ones(4)), "orb")
    with pytest.raises(ValueError, match="timeline"):
        L.render(plate, track, mask, voice(np.ones(5)), "arc")
    with pytest.raises(ValueError, match="mask"):
        L.render(plate, track, mask[:, ::2], voice(np.ones(4)), "arc")
