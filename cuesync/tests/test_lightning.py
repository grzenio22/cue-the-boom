"""Lightning made from the hand (cuesync.lightning): arcs from a fingertip and a ball over a palm, on small synthetic
plates, tracks and mattes."""

import colorsys

import cv2
import numpy as np
import pytest

from cuesync import lightning as L
from cuesync.elements import Timeline
from cuesync.transfer import gam, lin

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
         wall=(150, 142, 128), texture: float = 18.0, size: tuple[int, int] = (W, H)):
    """(plate uint8 [n, H, W, 3], track dict, matte uint8 [n, H, W]): a wall and a hand held up, both with ``texture``
    (code values), whose middle knuckle is at ``at(t)``, the emitter on its index fingertip or on its palm; the picture
    ``size`` (w, h) px."""
    W, H = size
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
    channel are at most SIDE_WIDTH as thick and SIDE_INTEN as bright on every frame (re-struck brighter than it, one
    came close to it), a bolt's forks are much dimmer and thinner and leave along its way, and drawn, only the main
    channel clips to white."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    bolts, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 1.0)), seed=6)
    tip = np.asarray(track["landmarks"][0][8][:2])
    main_w, seen = L._width_k(s) * L.ARC_WIDTH[1], 0
    for frame in bolts:
        streams = [(wd, it) for p, wd, it in frame if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
        main = [it for wd, it in streams if np.isclose(wd, main_w)]
        assert len(main) <= 1 and all(np.isclose(wd, main_w) or wd <= L.SIDE_WIDTH * main_w + 1e-9 for wd, _ in streams)
        assert all(it <= L.SIDE_INTEN * main[0] + 1e-9 for wd, it in streams if not np.isclose(wd, main_w))
        seen += len(main)
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


def course(p: np.ndarray, pieces: int) -> np.ndarray:
    """A channel resampled to ``pieces`` equal lengths and smoothed over a few of them: its course, kinks dropped."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    at = np.linspace(0.0, cum[-1], pieces + 1)
    q = np.stack([np.interp(at, cum, p[:, 0]), np.interp(at, cum, p[:, 1])], 1)
    for _ in range(2):
        q[1:-1] = (q[:-2] + q[1:-1] + q[2:]) / 3.0
    return q


def test_streamers_keep_well_round_from_the_main_channel_on_either_side_and_three_only_at_the_peak():
    """Seen from the fingertip, a streamer's course past the first fifth of its length keeps at least SIDE_APART's
    angle round from the main channel where that is as far out (leaving 18 degrees off its heading, one bent back and
    ran 100-150 px beside the main channel for its whole length, a faint twin), the two on opposite sides; under the
    voice's peak there is one at most."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    main_w = L._width_k(s) * L.ARC_WIDTH[1]
    seen = 0
    for seed in (6, 7, 8):
        bolts, _ = L.arc_frames(hand, voice(np.full(n, 1.0)), seed=seed)
        for frame in bolts:
            main = [p for p, wd, _ in frame if np.allclose(p[0], tip) and np.isclose(wd, main_w)]
            sides = [p for p, wd, _ in frame if np.allclose(p[0], tip) and not np.isclose(wd, main_w)
                     and np.linalg.norm(p[-1] - tip) > s]
            assert len(main) == 1 and len(sides) <= 2
            m = course(main[0], 64) - tip
            out, ang = np.maximum.accumulate(np.hypot(*m.T)), np.arctan2(m[:, 1], m[:, 0])
            way = []
            for p in sides:
                q = course(p, 24)[5:] - tip  # (past the first fifth)
                at = np.minimum(np.searchsorted(out, np.hypot(*q.T)), len(out) - 1)
                off = np.degrees((np.arctan2(q[:, 1], q[:, 0]) - ang[at] + np.pi) % (2 * np.pi) - np.pi)
                assert np.abs(off).min() >= L.SIDE_APART[0] - 0.5
                way.append(np.sign(off.mean()))
                seen += 1
            assert len(way) < 2 or way[0] != way[1]
    assert seen > n
    bolts, _ = L.arc_frames(hand, voice(np.full(n, 0.6)), seed=6)
    assert max(len(mains(b, tip, s)) for b in bolts) == 2


def crossings(chans: list[np.ndarray], free: list[float]) -> int:
    """How many times two of ``chans`` (polylines) cross in the picture, not counting where either is within its
    ``free`` px of where it leaves (a fork leaves on its channel, a streamer from its channel's fingertip)."""
    count = 0
    for i in range(len(chans)):
        for j in range(i + 1, len(chans)):
            a, b = chans[i][:, :2], chans[j][:, :2]
            if (a.min(0) > b.max(0)).any() or (b.min(0) > a.max(0)).any():
                continue
            p, r = a[:-1, None], (a[1:] - a[:-1])[:, None]
            q, u = b[None, :-1], (b[1:] - b[:-1])[None]
            den = r[..., 0] * u[..., 1] - r[..., 1] * u[..., 0]
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = ((q - p)[..., 0] * u[..., 1] - (q - p)[..., 1] * u[..., 0]) / den
                tb = ((q - p)[..., 0] * r[..., 1] - (q - p)[..., 1] * r[..., 0]) / den
            hit = (den != 0) & (ta >= 0) & (ta <= 1) & (tb >= 0) & (tb <= 1)
            at = (p + np.where(hit, ta, 0.0)[..., None] * r)[hit]
            far = (np.linalg.norm(at - a[0], axis=1) > free[i]) & (np.linalg.norm(at - b[0], axis=1) > free[j])
            seen = (at[:, 0] >= 0) & (at[:, 0] < W) & (at[:, 1] >= 0) & (at[:, 1] < H)
            count += int((far & seen).sum())
    return count


def test_no_fork_or_streamer_crosses_or_rejoins_another_channel():
    """However loud, with one finger or two, held still or swept: no fork, streamer or second finger's channel
    crosses another channel or rejoins its own away from where it leaves (a closed loop is the most obviously fake
    shape lightning can take), the crackle at the fingertips aside."""
    n, s = 24, 60.0
    for middle_out, speed, level in ((False, 0.0, 1.0), (False, 4.0, 0.9), (True, 0.0, 0.95), (True, 4.0, 1.0)):
        plate, track, _ = take(n, lambda t: (135 + speed * t, 330), s, middle_out=middle_out)
        hand = L._hand(plate, track)
        for seed in range(3):
            bolts, _ = L.arc_frames(hand, voice(np.full(n, level), hits=[0]), seed=seed)
            for t, frame in enumerate(bolts):
                tips = [np.asarray(track["landmarks"][t][f][:2]) for f in (8, 12)]
                chans = [p for p, _, _ in frame  # (the crackle: short arcs out of, up to and between the tips)
                         if not (min(np.linalg.norm(p[[0, -1], None] - tips, axis=-1).min(1)) < 1e-6
                                 and np.linalg.norm(p - p[0], axis=1).max() < 0.35 * s
                                 or min(np.linalg.norm(p[-1] - tp) for tp in tips) < 1e-6)]
                lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in chans]
                free = [max(0.16 * s, (L.ROOM_SKIP + 0.02) * ln) for ln in lengths]
                assert crossings(chans, free) == 0, (middle_out, speed, seed, t)


def test_no_fork_or_streamer_runs_alongside_another_channel_and_streamers_are_short():
    """A fork or a streamer lies within ALONG's distance of another channel for at most ALONG's share of its length
    past where it leaves (a fork ran 30 px beside its channel for 150 px: two wires side by side), and a streamer
    reaches at most SIDE_REACH's half of its main channel's reach (at 60-70 % they read as wires)."""
    n, s = 24, 60.0
    plate, track, _ = take(n, lambda t: (135, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    near = 0.8 * L.ALONG[0] * s  # (a little inside the rule's own distance: it is drawn on whole pixels)
    for seed in range(3):
        bolts, _ = L.arc_frames(hand, voice(np.full(n, 1.0), hits=[0]), seed=seed)
        for frame in bolts:
            streams = [p for p, _, _ in frame if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
            reach = sorted(np.linalg.norm(p[-1] - tip) for p in streams)
            assert len(reach) < 2 or reach[-2] <= L.SIDE_REACH[1] * reach[-1] + 1e-6  # (the main channel the longest)
            chans = [p for p, _, _ in frame  # (the crackle: short arcs out of and up to the tip)
                     if not (np.allclose(p[0], tip) and np.linalg.norm(p - p[0], axis=1).max() < 0.35 * s
                             or np.allclose(p[-1], tip))]
            for i, c in enumerate(chans[1:], 1):
                seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
                cum = np.concatenate([[0.0], np.cumsum(seg)])
                past = (cum > 0.2 * cum[-1]) & (np.linalg.norm(c - c[0], axis=1) > 0.2 * s)
                seen = (c[:, 0] >= 0) & (c[:, 0] < W) & (c[:, 1] >= 0) & (c[:, 1] < H)
                others = [o for j, o in enumerate(chans)  # (not its own forks, which leave from it)
                          if j != i and np.linalg.norm(c[1:] - o[0], axis=1).min() > 1e-6]
                if not others:
                    continue
                gap = np.min(np.linalg.norm(c[:, None] - np.concatenate(others)[None], axis=-1), axis=1)
                by = past & seen & (gap < near)
                assert seg[by[1:] & by[:-1]].sum() <= (L.ALONG[2] + 0.05) * cum[-1]


def test_a_branch_fades_along_its_length_and_a_main_channel_does_not():
    """Drawn with a brighter channel, a dim one (a streamer or a fork) fades towards its end as TAPER says, to about a
    third of itself; the brightest drawn keeps its brightness end to end."""
    line = np.stack([np.full(80, 30.0), np.linspace(10, 150, 80)], 1)
    other = line + (60.0, 0.0)
    box = (0, 160, 0, 120)
    main = L.draw([(line, 2.0, 1.0)], box)[..., 0]
    branch = L.draw([(line, 2.0, 0.3), (other, 2.0, 1.0)], box)[..., 0]  # (the same stretches as the main: drawn first)
    start, end = slice(12, 25), slice(135, 148)  # (the first and last tenth of it)
    ratio = (branch[end, 26:35].max(1) / main[end, 26:35].max(1)).mean() / \
        (branch[start, 26:35].max(1) / main[start, 26:35].max(1)).mean()
    assert ratio < 0.5
    assert main[end, 26:35].max() > 0.5 * main[start, 26:35].max()
    wide = L.draw([(line, 3.0, 0.3), (other, 3.0, 1.0)], box)[..., 0]  # (and it thins: a wedge, not a wire)
    assert (wide[start, 22:39] > 0.02).sum(1).mean() >= 2.0 * (wide[end, 22:39] > 0.02).sum(1).mean()


def test_a_long_finger_keeps_its_own_way_and_a_short_one_points_into_the_scene():
    """A finger pointing away from the camera but long and clear in the picture keeps its direction there (bent
    towards the picture's middle by its vanishing point, the arcs left a finger pointing up and left straight up); one
    pointing away and short in the picture points at its vanishing point."""
    n, s = 6, 60.0
    plate, track, _ = take(n, lambda t: (60, 400), s)  # (low on the left: the picture's middle is up and right)
    up_left = np.array([-np.sin(np.radians(30)), -np.cos(np.radians(30))])
    track["dir"] = [up_left.tolist()] * n  # (the track's own direction, where the finger is too short to read, agrees)
    hand = L._hand(plate, track)
    for reach, own in ((0.45, True), (0.16, False)):  # (hand sizes, its middle joint to its tip in the picture)
        hand.lm3[:, 8, :2] = hand.lm3[:, 6, :2] + up_left * reach * s
        hand.lm3[:, 8, 2] = hand.lm3[:, 6, 2] + 2.0 * s  # (its tip far further from the camera than its middle joint)
        aim = L._aims(hand, voice(np.ones(n)), 8, hand.size)[n // 2]
        off = np.degrees(np.arccos(np.clip(aim @ up_left, -1, 1)))
        assert off < 3.0 if own else off > 15.0


def test_the_main_channel_leaves_along_the_finger():
    """A main channel's first ARC_LEAD runs along its finger's aim, and it heads no further than ARC_OFF off it, so
    the bolt comes out of the finger however far its end strays."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    tip = np.asarray(track["landmarks"][0][8][:2])
    main_w = L._width_k(s) * L.ARC_WIDTH[1]
    for seed in range(4):
        bolts, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 1.0), hits=[0, 12]), seed=seed)
        for frame in bolts:
            for p, wd, _ in frame:
                if np.allclose(p[0], tip) and np.isclose(wd, main_w):
                    lead = tip + np.array([0.0, -1.0]) * L.ARC_LEAD * np.linalg.norm(p[-1] - tip)  # (straight up)
                    assert np.linalg.norm(p - lead, axis=1).min() < 1e-6
                    assert abs(up_of(p[-1:], tip)[0]) < L.ARC_OFF + 1e-6


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
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))
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


def test_the_room_light_lands_as_the_same_blue_on_a_tinted_wall_and_lifts_a_dark_room_half_a_stop_at_most(
        monkeypatch):
    """On a wall already lavender in the plate the arcs' light is no more violet than on a grey one (its blue-white
    graded a lavender room violet), and on a dark plate the room far from the channels comes up by at most about half
    a stop (it came up three quarters of one)."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))
    n = 8
    far = np.zeros((H, W), bool)
    far[400:, 210:] = True  # (the wall below and right of the hand, away from the channels' own glow)

    def light(wall, palm=(225, 150, 130)):
        plate, track, mask = take(n, lambda t: (120, 330), wall=wall, palm=palm, texture=0.0)
        out = L.render(plate, track, mask, voice(np.full(n, 1.0), hits=[0]), "arc", seed=2)
        here = far & (mask[0] == 0)
        stops = np.log2((lin(out[4:]) @ LUMA)[:, here].mean() / (lin(plate[4:]) @ LUMA)[:, here].mean())
        return (lin(out[4:]) - lin(plate[4:]))[:, here].mean((0, 1)), stops

    (grey, _), (lavender, _) = light((150, 150, 150)), light((150, 128, 148))
    assert lavender[2] / lavender[1] < 1.05 * grey[2] / grey[1] and lavender[0] / lavender[1] < 1.05 * grey[0] / grey[1]
    _, stops = light((40, 34, 40), palm=(70, 50, 45))
    assert 0.1 < stops < 0.55


def test_two_fingers_at_the_voices_peak_strike_a_channel_each_not_a_thicket():
    """Two fingers over a dark wall at the voice's peak: a main channel each, the first finger's with one streamer
    at most and the second's alone, kept PAIR_LIFE frames and re-struck every frame rather than struck anew or left
    dark, so the room flashes less than twice a second for any seed (new channels on and off every frame flashed it
    3 times a second, the photosensitivity limit)."""
    n, s = 36, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True, wall=(142, 122, 139), palm=(200, 140, 125))
    e = np.clip(0.93 + 0.07 * np.sin(np.arange(n) / 3.0), 0, 1)
    tl = voice(e, hits=[0, 20])
    hand = L._hand(plate, track)
    tips = [np.asarray(track["landmarks"][0][f][:2]) for f in (8, 12)]
    thick = 0.9 * L._width_k(s) * (L.ARC_WIDTH[0] + (L.ARC_WIDTH[1] - L.ARC_WIDTH[0]) * e.min())  # (main channels')
    for seed in range(5):
        bolts, _ = L.arc_frames(hand, tl, seed=seed)
        for b in bolts:
            assert len(mains(b, tips[0], s)) <= L.PAIR_STREAMS and len(mains(b, tips[1], s)) <= 1
        main = [[(p[-1], it) for p, wd, it in b if np.allclose(p[0], tips[0]) and wd > thick] for b in bolts]
        assert all(len(m) == 1 and m[0][1] > 0.5 for m in main)  # (on every frame, never left dark)
        ends = {tuple(np.round(m[0][0], 6)) for m in main}
        assert len(ends) <= n // L.PAIR_LIFE + 2  # (each path kept PAIR_LIFE frames, a hit striking anew)
        out = L.render(plate, track, mask, tl, "arc", seed=seed)
        assert flashes(out, plate) < 2.0


def struck(frame: list, tip: np.ndarray, s: float) -> bool:
    """Whether a main channel leaves ``tip`` on this frame, struck or re-struck (not left dark): the widest stream from
    it, at least a quiet main channel's width, and bright."""
    streams = [(wd, it) for p, wd, it in frame if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
    widest = max(streams, default=(0.0, 0.0))
    return widest[0] >= 0.99 * L._width_k(s) * L.ARC_WIDTH[0] and widest[1] > 0.3


def test_two_fingers_one_behind_the_other_along_the_aim_both_strike_at_the_peak():
    """Fingertips lined up along the aim (the second behind the first): over the voice's peak both strike a main
    channel on at least 70 % of frames, the second's leaving past the first's tip (dropped on its first failure to
    keep clear, it struck on few frames and its tip fed only a jumper into the first's), and the room flashes less than
    twice a second."""
    n, s = 36, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True, wall=(142, 122, 139), palm=(200, 140, 125))
    e = np.clip(0.93 + 0.07 * np.sin(np.arange(n) / 3.0), 0, 1)
    tl = voice(e, hits=[0, 20], aim=270.0)  # (pointing left: the index's tip in front of the middle finger's)
    hand = L._hand(plate, track)
    tips = [np.asarray(track["landmarks"][0][f][:2]) for f in (8, 12)]
    assert abs(tips[1][1] - tips[0][1]) < 0.1 * s and tips[1][0] - tips[0][0] > 0.2 * s  # (lined up, the middle behind)
    for seed in range(3):
        bolts, _ = L.arc_frames(hand, tl, seed=seed)
        assert np.mean([struck(b, tips[0], s) and struck(b, tips[1], s) for b in bolts]) >= 0.7
        assert flashes(L.render(plate, track, mask, tl, "arc", seed=seed), plate) < 2.0


def test_two_fingers_at_the_peak_at_full_size_are_not_a_blaze_of_white():
    """Two fingers at the voice's peak at a phone's 1080x1920: the white-hot core (luminance over 0.85, 0.3 over the
    plate) covers at most 1.5 % of the frame on any frame (the test's quarter-size picture draws the channels thicker
    against it)."""
    n, s, k = 12, 240.0, 4
    plate, track, mask = take(n, lambda t: (135 * k, 330 * k), s, middle_out=True, wall=(142, 122, 139),
                             palm=(200, 140, 125), size=(W * k, H * k))
    e = np.clip(0.93 + 0.07 * np.sin(np.arange(n) / 3.0), 0, 1)
    out = L.render(plate, track, mask, voice(e, hits=[0]), "arc", seed=0)
    Y, P = lin(out) @ LUMA, lin(plate) @ LUMA
    assert ((Y > 0.85) & (Y - P > 0.3)).mean((1, 2)).max() < 0.015


def test_a_curled_finger_strikes_nothing_whatever_its_depth_says():
    """A middle finger MediaPipe's depth calls out straight (its reach over REACH_OUT) while the picture shows it curled
    into the hand, or straight but short with its tip inside the silhouette well behind the index's (as on IMG_6653,
    where a second bolt erupted from the middle of the hand and crossed the index finger): only the index strikes and
    only its tip burns. Held out beside the index, both strike."""
    n, s = 24, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s)  # (the middle finger curled: its tip back by its knuckle)
    for lm in track["landmarks"]:
        lm[12][1] = lm[9][1] - 0.2 * s  # (its tip folded back down level with its knuckle, inside the hand)
        lm[12][2] = -2.0 * s  # (and far nearer the camera than its knuckle: out straight, to the depth)
    assert L._fingers(L._hand(plate, track), 0) == [8, 12]
    short = [[list(p) for p in lm] for lm in track["landmarks"]]
    for lm in short:  # (straight and out from the palm, but its tip 0.45 hand sizes behind the index's)
        x, cy = lm[9][0], lm[9][1] - 0.05 * s
        lm[10:13] = [[x, cy - 0.05 * s, 0.0], [x, cy - 0.2 * s, 0.0], [x, cy - 0.35 * s, -2.0 * s]]
    for tr in (track, dict(track, landmarks=short)):
        hand = L._hand(plate, tr)
        assert not L._pair_out(hand, [8, 12], mask).any()
        for seed in range(3):
            bolts, spots = L.arc_frames(hand, voice(np.full(n, 0.95), hits=[0]), seed=seed, matte=mask)
            tip12 = hand.lm[0, 12]
            assert all(len(sp) == 1 for sp in spots)
            assert not any(np.allclose(p[0], tip12) or np.allclose(p[-1], tip12) for b in bolts for p, _, _ in b)
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True)
    assert L._pair_out(L._hand(plate, track), [8, 12], mask).all()


def test_the_bolts_go_behind_the_hand_not_across_it():
    """Struck back down across the hand (the index up, the studio's arrow pointing down over the palm and the forearm):
    no fork or streamer lies on the hand's matte for more than ON_HAND's share of its length, and drawn, nothing of the
    bolts shows over the hand further than HIDE_NEAR from the fingertip: the hand hides them."""
    n, s = 12, 60.0
    plate, track, mask = take(n, lambda t: (135, 230), s, texture=0.0)
    tl = voice(np.full(n, 1.0), hits=[0], aim=180.0)
    hand = L._hand(plate, track)
    bolts, _ = L.arc_frames(hand, tl, seed=1, matte=mask)
    main_w = L._width_k(s) * L.ARC_WIDTH[1]
    for t in range(n):
        tip = hand.lm[t, 8]
        for p, wd, _ in bolts[t]:
            if np.allclose(p[0], tip) and np.isclose(wd, main_w) or np.allclose(p[-1], tip) or \
                    np.linalg.norm(p - tip, axis=1).max() < 0.6 * s:
                continue  # (the main channel is kept wherever it strikes; the crackle lives on the finger)
            q = np.round(p[:, :2]).astype(int)
            inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
            on = np.zeros(len(p), bool)
            on[inside] = mask[t][q[inside, 1], q[inside, 0]] >= 128
            on &= np.linalg.norm(p - p[0], axis=1) > L.ON_HAND[0] * s
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            assert seg[on[1:] & on[:-1]].sum() <= (L.ON_HAND[1] + 0.02) * seg.sum()
    out = L.render(plate, track, mask, tl, "arc", seed=1)
    yy, xx = np.indices((H, W))
    for t in range(n):
        tip = hand.lm[t, 8]
        inside = cv2.erode(mask[t], np.ones((5, 5), np.uint8)) > 0  # (the channels are as soft as the plate: one
        # passing just outside the hand's outline blurs a px or two over it)
        far = inside & (np.hypot(xx - tip[0], yy - tip[1]) > L.HIDE_NEAR[1] * s)
        assert ((out[t].min(-1) >= 235) & far).sum() == 0  # (no white core over the hand)


def turning(o: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which way round ``o`` -> ``x`` -> ``y`` turns (the sign of their cross product), elementwise."""
    return (x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1]) - (x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0])


def self_crossed(p: np.ndarray, step: float = 4.0) -> bool:
    """Whether a polyline crosses itself, cut into pieces about ``step`` px long (pieces side by side not counted)."""
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    at = np.linspace(0.0, cum[-1], max(2, int(cum[-1] / step) + 1))
    q = np.stack([np.interp(at, cum, p[:, 0]), np.interp(at, cum, p[:, 1])], 1)
    for i in range(len(q) - 3):
        a, b, c, d = q[i], q[i + 1], q[i + 2:-1], q[i + 3:]
        if ((turning(a, b, c) * turning(a, b, d) < 0) & (turning(c, d, a) * turning(c, d, b) < 0)).any():
            return True
    return False


def test_no_channel_crosses_itself_and_no_fork_runs_back_to_its_channel():
    """However loud, from one finger or two: no channel crosses itself, and a fork ends no nearer the channel it leaves
    than its middle is (one curled back to end a few px from its channel, and a second finger's channel carried a
    closed 'flag' loop: the most obviously fake shapes lightning can take)."""
    n, s = 16, 60.0
    forks = 0
    for middle_out in (False, True):
        plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=middle_out)
        hand = L._hand(plate, track)
        for seed in range(3):
            bolts, _ = L.arc_frames(hand, voice(np.full(n, 1.0), hits=[0, 8]), seed=seed, matte=mask)
            for t, frame in enumerate(bolts):
                tips = hand.lm[t, [8, 12]]
                chans = [p[:, :2] for p, _, _ in frame]
                for i, p in enumerate(chans):
                    if np.linalg.norm(p[:, None] - tips, axis=-1).min(1).max() < 0.6 * s:
                        continue  # (the crackle at a tip)
                    assert not self_crossed(p)
                    parents = [c for j, c in enumerate(chans)  # (a channel it leaves from part way along)
                               if j != i and 0 < int(np.argmin(np.linalg.norm(c - p[0], axis=1))) < len(c) - 1
                               and np.linalg.norm(c - p[0], axis=1).min() < 1e-6]
                    for c in parents:
                        forks += 1
                        gap = lambda x: np.linalg.norm(c - x, axis=1).min()  # noqa: E731
                        assert gap(p[-1]) >= gap(p[len(p) // 2]) - 2.0
    assert forks > 20


def test_a_lead_turned_far_from_its_bolt_bends_rather_than_elbows():
    """Where a channel's first stretch turns more than LEAD_TURN to its far end (a second finger's, led out past the
    first's tip), the turn is spread over pieces of at most LEAD_TURN, not one right angle at an elbow."""
    rng = np.random.default_rng(2)
    p0, k, end = np.zeros(2), np.array([0.0, -60.0]), np.array([240.0, -120.0])  # (a 76 degree turn at k)
    pts = L._course(p0, k, end, rng, 0.0, 3.0)  # (no jaggedness: the course alone, bowed a little at its end)
    d = np.diff(pts, axis=0)
    d = d[np.linalg.norm(d, axis=1) > 1e-6]
    turn = np.degrees(np.abs(np.arctan2(d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0], (d[:-1] * d[1:]).sum(1))))
    assert turn.max() <= L.LEAD_TURN + 12.0 and turn.sum() > 60.0
    assert np.allclose(pts[0], p0) and np.allclose(pts[-1], end)


def test_the_room_is_never_darker_than_the_plate_while_bolts_strike():
    """Over a dark, a mid and a light lavender wall, with the camera exposing down for the arcs, the room away from them
    nets no darker than the plate on any frame they strike at the voice's peak, with one finger or two (the exposure's
    0.3 stops outweighed the capped room light: at the loudest, two fingers over a mid wall dimmed the room they lit
    by 0.12 stops)."""
    n, s = 16, 60.0
    far = np.zeros((H, W), bool)
    far[400:, 210:] = True  # (the wall below and right of the hand, away from the channels' own glow)
    e = np.clip(0.93 + 0.07 * np.sin(np.arange(n) / 3.0), 0, 1)
    for wall, palm in (((40, 34, 40), (70, 50, 45)), ((142, 122, 139), (200, 140, 125)), ((200, 190, 200), (220, 160,
                                                                                                          140))):
        for middle_out in (False, True):
            plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=middle_out, wall=wall, palm=palm,
                                      texture=0.0)
            out = L.render(plate, track, mask, voice(e, hits=[0]), "arc", seed=2)
            here = far & (mask[0] == 0)
            stops = np.log2((lin(out) @ LUMA)[:, here].mean(1) / (lin(plate) @ LUMA)[:, here].mean(1))
            assert stops.min() > 0.0, (wall, middle_out)


def up_of(ends: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Degrees clockwise from straight up, from ``tip`` to each of ``ends``."""
    return np.degrees(np.arctan2(ends[:, 0] - tip[0], tip[1] - ends[:, 1]))


def test_the_arcs_leave_along_the_finger_and_from_two_fingers_held_together(monkeypatch):
    """Two fingers out together (the index left of the middle): each strikes its main channel along the fingers, the
    two turned apart so they separate; the second strikes no streamers of its own. They light the room as one finger
    does, not twice as much."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))  # (the light alone)
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
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))
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
    """It grows from a point while the voice holds, over about 5 frames rather than popping into being (its body on
    its first frame at most 0.35 of its size 6 frames on: sized from its charge, it was 60-65 % of its full size), and
    shrinks when the voice stops, its body as dense as ever while it is a ball at all (it shrinks rather than fading
    out); a ball barely charged has no body, only its filaments' crackle, reaching past where its skin will be."""
    n, s = 40, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.r_[np.zeros(4), np.full(24, 0.9), np.zeros(12)]
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(e), seed=0)
    assert all(b is None for b in balls[:4])
    r = [b[2] / s for b in balls[4:28]]
    lo, hi = L.BALL_R
    assert r[-1] > lo + 0.6 * (hi - lo) and np.all(np.diff(r) > 0)
    first = next(t for t in range(4, n) if L._dense(balls[t][3], balls[t][4]) > 0)
    assert balls[first][2] <= 0.35 * balls[first + 6][2]
    assert balls[31] is None or balls[31][2] < 0.7 * balls[27][2]
    assert L._dense(balls[30][3], balls[30][4]) == 1.0  # (shrinking, still dense)
    assert L._dense(0.05, 0.01) == 0.0
    glow, out = np.zeros((600, 600, 3), np.float32), np.zeros((600, 600, 3), np.float32)
    veil = np.zeros((600, 600), np.float32)
    L._sphere(glow, out, veil, (0, 600, 0, 600), (300.0, 300.0, 40.0, 0.05, 0.01, 1.0), 2.5, 0, 0)  # (barely charged)
    assert not glow.any() and not out.any() and not veil.any()
    c, lm = np.asarray(balls[4][:2]), np.asarray(track["landmarks"][4])[:, :2]
    own = [p for p, _, _ in bolts[4] if np.linalg.norm(lm - p[-1, :2], axis=1).min() > 1e-6]  # (none to the hand)
    assert max(np.linalg.norm(p[:, :2] - c, axis=1).max() for p in own) > 1.2 * balls[4][2]


def test_the_ball_is_a_blue_light_over_the_palm_that_keeps_its_lines():
    """A palm already at the top of red in the picture: the ball is the brightest thing over it and reads blue, not a
    white disc: its body well over the luminance of the palm round it, and never darker towards its outline than the
    skin just outside it (a dark limb made it a glass marble); the palm just past its outline is lit and keeps its
    lines; the camera exposes down for it, and the wall round it is lit blue against the room further off."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    flat, _, _ = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), texture=0.0)
    out_flat = L.render(flat, track, mask, tl, "ball", seed=0)  # the same lightning over a palm with no lines
    t = n - 1
    x, y, R = L.ball_frames(L._hand(plate, track), tl, 0)[2][t][:3]
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - x, yy - y) / R
    on = mask[t] > 0
    under = (d < 0.85) & on
    o, p = out[t].astype(np.float32), plate[t].astype(np.float32)
    lum = lin(out[t]) @ LUMA
    assert lum[(d < 0.6) & on].mean() > 1.5 * lum[(d > 1.75) & (d < 2.5) & on].mean()  # a light: the brightest
    # thing there (its saturated blue has little luminance of its own)

    def ring(r: float) -> float:
        return float(lum[(np.abs(d - r) < 0.03) & on].mean())

    assert ring(0.97) > ring(0.8) - 0.02 and ring(0.97) > ring(1.05)  # (no dark limb)
    white = (o > 245).all(-1)
    assert white[under].mean() < 0.1  # its white-hot middle and channels, not a white disc
    assert (o[under, 2] - o[under, 0]).mean() > (p[under, 2] - p[under, 0]).mean() + 40  # blue
    lines = (o - out_flat[t]).mean(-1), (p - flat[t]).mean(-1)  # the palm's lines, as they come through
    round_it = (d > 1.15) & (d < 1.6) & on  # the palm just past its outline: lit, not veiled or milky
    assert lum[round_it].mean() > (lin(plate[t]) @ LUMA)[round_it].mean()
    assert np.std(lines[0][round_it]) / np.std(lines[1][round_it]) > 0.7
    d *= R / s
    near, far = (d > 0.6) & (d < 1.0) & ~on, (d > 1.8) & ~on
    near, far = (o - p)[near], (o - p)[far]
    assert far.mean() < -5  # the camera exposed down for it...
    assert (near[:, 2] - near[:, 0]).mean() > (far[:, 2] - far[:, 0]).mean() + 4  # ...and it lights the wall blue


def hue(rgb) -> float:
    """Degrees: the hue of a colour as shown (code values)."""
    r, g, b = (float(v) / 255.0 for v in rgb)
    return 360.0 * colorsys.rgb_to_hsv(r, g, b)[0]


def hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_the_balls_body_is_white_in_its_middle_then_cyan_then_blue_not_a_clipped_disc():
    """The body rolls off to white with its hue kept: white-hot in its middle (saturation at most 0.1 within 0.15 of
    its radius) and a saturated blue towards its limb (at least 0.35 at 0.7-0.85), at 0.15 of its radius at least 1.3
    times as bright as at 0.75, with a cyan ring at its limb (hue 185-215), its colour the arcs' own blue (within 25
    degrees of their glow's hue: in a greener blue it read aqua beside them). Each channel clipped on its own, it was a
    flat pale-aqua disc at (205-215, 247-254, 253); under a glare, a pale glass sphere (saturation 0.17-0.25 at every
    radius, its luma 0.86 to 0.70 from its middle out, a near-white lavender ring at its limb)."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    t = n - 1
    x, y, R = L.ball_frames(L._hand(plate, track), tl, 0)[2][t][:3]
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - x, yy - y) / R
    o = out[t].astype(np.float32)
    lum = lin(out[t]) @ LUMA

    def at(r: float) -> np.ndarray:
        return np.abs(d - r) < 0.05

    assert lum[at(0.15)].mean() >= 1.3 * lum[at(0.75)].mean()
    body = (d > 0.4) & (d < 0.85)

    def sat(sel: np.ndarray) -> float:
        return colorsys.rgb_to_hsv(*(o[sel].mean(0) / 255.0))[1]

    assert sat(d < 0.15) <= 0.1 and sat((d > 0.7) & (d < 0.85)) >= 0.35  # (white in its middle, a blue round it)
    assert 185.0 <= hue(o[np.abs(d - 0.95) < 0.03].mean(0)) <= 215.0  # (its ring cyan)
    assert ((o[body, 1] >= 250) & (o[body, 2] >= 250)).mean() < 0.1  # (not two channels clipped: pale aqua)
    line = np.stack([np.full(60, 50.0), np.linspace(10, 90, 60)], 1)
    E, _ = L.layer([(line, 4.0, 1.0)], [], None, 60.0, (0, 100, 0, 100), L.GLOW_CAP["arc"], 0.0, np.zeros(2), [])
    glow = E[30:70, 52:62].reshape(-1, 3).mean(0)  # (the arcs' glow, 2-12 px off a channel)
    assert hue_gap(hue(o[body].mean(0)), hue(gam(np.ascontiguousarray(glow[None, None] / glow.max()))[0, 0])) < 25.0


BALL_PROFILE = ((0.0, 246, 252, 255), (0.15, 224, 242, 255), (0.3, 191, 225, 255), (0.45, 177, 208, 255),
                (0.6, 172, 202, 255), (0.75, 156, 194, 255), (0.9, 172, 214, 255))  # (radii, r, g, b): the ball's body
# as shown, which web/lib/lightning.test.ts holds Try it live's body (its BODY_STOPS) to as well


def test_the_balls_body_is_the_profile_the_live_preview_draws():
    """At a phone's 1080x1920, at full charge over a palm, the median colour of each thin ring of the ball's body is
    within 8 levels of BALL_PROFILE, which the live preview's body is held to too: the two draw the same ball (the
    live one was a saturated cyan disc while the shot's was a pale one)."""
    k, n = 4, 24
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135 * k, 300 * k), 80.0 * k, emitter="palm", palm=(240, 165, 148),
                              size=(W * k, H * k))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    balls = L.ball_frames(L._hand(plate, track), tl, 0)[2]
    yy, xx = np.indices(out.shape[1:3])
    rings = []
    for t in (n - 6, n - 1):
        x, y, R = balls[t][:3]
        d = np.hypot(xx - x, yy - y) / R
        rings.append([np.median(out[t][np.abs(d - r) < 0.03], axis=0) for r, *_ in BALL_PROFILE])
    assert np.abs(np.mean(rings, 0) - np.array([rgb for _, *rgb in BALL_PROFILE])).max() <= 8.0


def test_the_hue_keeping_roll_off_keeps_a_little_over_exposed_blue_blue_and_takes_a_lot_to_white():
    blue = np.array([0.2, 0.5, 1.0], np.float32)
    P = np.stack([blue * k for k in (0.6, 1.3, 5.0)])
    Q = L.hue_shoulder(P)
    assert np.allclose(Q[0], P[0]) and (Q <= 1.0).all()
    assert hue_gap(hue(gam(Q[1:2])[0]), hue(gam(blue[None])[0])) < 8.0 and Q[1, 0] / Q[1, 2] < 0.4
    assert Q[2].min() / Q[2].max() > 0.85  # (white)


def test_a_gap_between_the_fingers_does_not_show_through_the_ball():
    """The wall in a gap between two fingers behind the ball's top is scattered in its glowing body: a soft shade at
    most, not a sharp dark window through it (seen sharp, a glass marble)."""
    n, s = 12, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), texture=0.0)
    t = n - 1
    x, y, R = L.ball_frames(L._hand(plate, track), tl, 0)[2][t][:3]
    gap, gmask = plate.copy(), mask.copy()
    x0, x1 = int(x - 0.12 * R), int(x + 0.12 * R)
    gap[:, : int(y - 0.2 * R), x0:x1] = (150, 142, 128)  # (the wall, between two fingers)
    gmask[:, : int(y - 0.2 * R), x0:x1] = 0
    a, b = (lin(L.render(pl, track, mk, tl, "ball", seed=0)[t]) @ LUMA for pl, mk in ((plate, mask), (gap, gmask)))
    row = int(y - 0.6 * R)
    seen = (b - a)[row, int(x - 0.5 * R): int(x + 0.5 * R)]  # what the gap does to the ball, across it
    step = abs(float((lin(gap[t]) @ LUMA - lin(plate[t]) @ LUMA)[row, int(x)]))
    assert np.abs(seen).max() < 0.2 * step and np.abs(np.diff(seen)).max() < 0.1 * step  # (seen sharp: 0.26, 0.22)


def test_the_ball_settles_the_exposure_once_for_a_phrase(monkeypatch):
    """Three sounds with a second's gap between each: the camera exposes down once, by at most EXPOSURE's stops, and
    holds there through the gaps instead of breathing with each sound (by 0.6 stops three times in 4 s, it pumped
    the whole room). After one sound and silence it holds while the ball fades, and comes back once it has gone
    (held a second and a half, the room stayed dark with no ball in it)."""
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
    e[34:] = 0.0  # one sound, then silence: the camera holds while the ball fades, then comes back once it has gone
    out = L.render(plate, track, mask, voice(e, hits=[4]), "ball", seed=0)
    stops = np.log2((lin(out) @ LUMA)[:, far].mean(1) / (lin(plate) @ LUMA)[:, far].mean(1))
    balls = L.ball_frames(L._hand(plate, track), voice(e, hits=[4]), 0)[2]
    gone = next(t for t in range(20, n) if balls[t] is None)
    assert stops[gone - 1] < -0.25 and stops[-1] > -0.03 and np.all(np.diff(stops[gone:]) > -0.005)


def test_the_camera_holds_through_a_gap_between_phrases_and_comes_back_after_a_long_one():
    """The render looks ahead: the ball out for longer than it takes the camera to let go, but back within the frames
    it holds, the exposure holds through the gap (no rise over 0.05 stops: let go, it rose by 0.2 and fell again); out
    for longer than that, it comes back in between."""
    bright = np.zeros(150)
    bright[0:20], bright[44:64], bright[120:140] = 0.9, 0.9, 0.9  # (gaps of 24 and 56 frames)
    dip = L.exposure_dip(bright, *L.EXPOSURE["ball"])
    assert np.ptp(dip[16:64]) < 0.05
    assert dip[110] < dip[64] - 0.2


def test_the_balls_light_leaves_the_skin_round_it_lit_violet_not_veiled():
    """Round the ball's outline (1.05-1.3 of its radius) a palm at the top of red is lit, not veiled: its texture and
    the edges of the fingers at least 0.9 of the plate's, still saturated (at least 0.2), violet (hue 250-320) and at
    least 1.3 times as bright as the plate exposed down, while an olive wall there goes blue (hue 200-250). (Taken
    towards a cool white by the glare, the skin there kept saturation 0.12, texture 0.8 and edges 0.84: frosted
    glass.)"""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), wall=(110, 104, 90))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    t = n - 1
    x, y, R = L.ball_frames(L._hand(plate, track), tl, 0)[2][t][:3]
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - x, yy - y) / R
    o = out[t].astype(np.float32)
    Y, P = lin(out[t]) @ LUMA, lin(plate[t]) @ LUMA
    far = (yy < 70) & (mask[t] == 0)
    exposed = Y[far].mean() / P[far].mean()  # (the camera's dip, read off the wall far from the ball)
    on, ring = mask[t] > 0, (d > 1.05) & (d < 1.3)
    skin, wall = ring & on, ring & ~on
    g_o, g_p = (cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in (out[t], plate[t]))

    def fine(g):
        return g - cv2.GaussianBlur(g, (0, 0), 2.0)

    def grad(g):
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    edge = (cv2.morphologyEx(mask[t], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & ring
    assert edge.sum() > 20
    assert fine(g_o)[skin].std() >= 0.9 * fine(g_p)[skin].std()
    assert grad(g_o)[edge].mean() >= 0.9 * grad(g_p)[edge].mean()
    h_skin, s_skin, _ = colorsys.rgb_to_hsv(*(o[skin].mean(0) / 255.0))
    assert s_skin >= 0.2 and 250.0 <= 360.0 * h_skin <= 320.0
    assert 200.0 <= hue(o[wall].mean(0)) <= 250.0
    assert Y[skin].mean() >= 1.3 * P[skin].mean() * exposed


def test_the_balls_glare_is_a_narrow_band_of_blue_the_same_over_skin_and_wall():
    """The lens's glare adds the same narrow band of blue over a palm and over an olive wall just past the ball's
    outline (blue over green over red), never over its body and gone a third of a radius out (screened wide over the
    picture, it washed the body and the fingers round it to one pale ice-blue)."""
    frame = np.zeros((161, 161, 3), np.uint8)
    frame[:, :80], frame[:, 80:] = (200, 140, 125), (110, 104, 90)  # a palm, an olive wall
    lit = frame.copy()
    L.glare(lit, (80.0, 80.0, 40.0, 1.0, 1.0, 1.0))
    added = lit.astype(int) - frame
    for r in (1.02, 1.1):
        skin, wall = added[80, int(80 - r * 40)], added[80, int(80 + r * 40)]
        assert np.abs(skin - wall).max() <= 1 and skin[2] > skin[1] > skin[0] >= 0 and skin[2] >= 10
    assert added[80, int(80 + 1.05 * 40), 2] <= 255 * L.GLARE_RIM[0] + 1
    assert not added[80, 80: int(80 + 0.85 * 40)].any()  # (its body)
    assert not added[80, int(80 + 1.45 * 40):].any()


def test_the_balls_arcs_over_the_skin_are_electric_blue_white():
    """An arc jumping from the ball to a fingertip lies over the skin in blue-white: the brightest 40 of its pixels on
    the skin more than 1.3 radii out have blue over green over red (red at most 10 levels over green) and a hue of
    190-230 (added over skin at the top of red, it clipped red, left green behind and read as magenta-white scratches,
    (245, 197, 245))."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0, 4, 8])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    hand = L._hand(plate, track)
    bolts, _, balls = L.ball_frames(hand, tl, 0)
    picked = []
    for t in range(n):
        if balls[t] is None:
            continue
        x, y, R = balls[t][:3]
        for p, _, _ in bolts[t]:
            if np.linalg.norm(hand.lm[t] - p[-1, :2], axis=1).min() > 1e-6:
                continue  # (not an arc to a fingertip or knuckle)
            q = np.unique(np.round(p[:, :2]).astype(int), axis=0)
            keep = (np.hypot(q[:, 0] - x, q[:, 1] - y) > 1.3 * R) & (mask[t][q[:, 1], q[:, 0]] > 0)
            picked += [out[t][b, a] for a, b in q[keep]]
    picked = np.array(picked, float)
    assert len(picked) >= 100
    top = picked[np.argsort(picked @ LUMA)[-40:]].mean(0)
    assert top[2] >= top[1] >= top[0] - 10 and 190.0 <= hue(top) <= 230.0


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
