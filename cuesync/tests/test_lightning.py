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
        ends = [mains(b, tip, s) for b in bolts]
        return np.mean([len(e) for e in ends]), np.mean([np.linalg.norm(a - tip) for e in ends for a in e]) / s

    (few, near), (many, far) = strikes(0.35), strikes(1.0)
    assert many >= few + 1 and far > 1.8 * near


def test_the_arcs_leave_along_the_finger_and_from_two_fingers_held_together():
    n, s = 12, 60.0
    plate, track, _ = take(n, lambda t: (135, 330), s, middle_out=True)
    hand = L._hand(plate, track)
    assert L._fingers(hand, 0) == [8, 12]
    bolts, _ = L.arc_frames(hand, voice(np.full(n, 0.9)), seed=1)
    for f in (8, 12):
        tip = np.asarray(track["landmarks"][0][f][:2])
        ends = np.array([a for b in bolts for a in mains(b, tip, s)])
        assert len(ends) > n // 2
        up = np.degrees(np.arctan2(ends[:, 0] - tip[0], tip[1] - ends[:, 1]))  # clockwise from straight up
        assert abs(np.median(up)) < 15 and np.mean(np.abs(up) < 20) > 0.9  # the fingers point straight up
    _, track1, _ = take(n, lambda t: (135, 330), s)
    assert L._fingers(L._hand(plate, track1), 0) == [8]  # a finger gun: the middle finger is curled


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


def test_the_ball_grows_with_the_held_voice_and_shrinks_when_it_stops():
    n, s = 40, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.r_[np.zeros(4), np.full(24, 0.9), np.zeros(12)]
    _, _, balls = L.ball_frames(L._hand(plate, track), voice(e), seed=0)
    assert all(b is None for b in balls[:4])
    r = [b[2] / s for b in balls[4:28]]
    assert r[0] < 0.2 and r[-1] > 0.33 and np.all(np.diff(r) > 0)
    assert balls[31] is None or balls[31][2] < 0.7 * balls[27][2]


def test_the_ball_keeps_the_palm_its_detail_and_glows_blue_over_it():
    """A palm already at the top of red in the picture: under the ball it keeps its lines, does not go white, and
    turns blue-violet; the wall round it is lit blue."""
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
    under = (d < 0.35) & (mask[t] > 0)
    o, p = out[t].astype(np.float32), plate[t].astype(np.float32)
    white = (o > 245).all(-1)
    assert white[under].mean() < 0.1  # the channels' white cores, not a white disc
    lines = (o - out_flat[t]).mean(-1), (p - flat[t]).mean(-1)  # the palm's lines, as they come through
    assert np.corrcoef(lines[0][under], lines[1][under])[0, 1] > 0.8
    assert 0.4 < np.std(lines[0][under]) / np.std(lines[1][under]) < 1.2
    assert (o[under, 2] - o[under, 0]).mean() > (p[under, 2] - p[under, 0]).mean() + 40  # bluer
    wall = (d > 0.6) & (d < 1.0) & (mask[t] == 0)
    gain = (o[wall] - p[wall]).mean(0)
    assert gain[2] > gain[1] > gain[0] and gain[2] > 4  # the room round it lit blue


def test_bad_inputs_are_refused():
    plate, track, mask = take(4, lambda t: (120, 330))
    with pytest.raises(ValueError, match="form"):
        L.render(plate, track, mask, voice(np.ones(4)), "orb")
    with pytest.raises(ValueError, match="timeline"):
        L.render(plate, track, mask, voice(np.ones(5)), "arc")
    with pytest.raises(ValueError, match="mask"):
        L.render(plate, track, mask[:, ::2], voice(np.ones(4)), "arc")
