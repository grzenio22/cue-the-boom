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

def test_arcs_are_rooted_on_the_moving_fingertip_and_struck_anew_every_frame_or_two():
    """A channel's root rides the fingertip and its far end holds where it struck while it lives, re-struck down the
    same path the next frame at most: a new path every frame or two (held 3-4 frames, the channels hung in the air)."""
    n, s = 30, 60.0
    plate, track, mask = take(n, lambda t: (110 + 3 * t, 330), s)
    hand = L._hand(plate, track)
    bolts, _, _ = L.arc_frames(hand, voice(np.full(n, 0.8)), seed=3)
    ends = []
    for t in range(n):
        tip = np.asarray(track["landmarks"][t][8][:2])
        e = mains(bolts[t], tip, s)
        assert e, f"no stream from the fingertip on frame {t}"
        if t:  # none left behind where the fingertip was a frame ago
            assert not mains(bolts[t], np.asarray(track["landmarks"][t - 1][8][:2]), s)
        ends.append(e)
    held = [sum(any(np.allclose(a, b) for b in ends[t + 1]) for a in ends[t]) / len(ends[t]) for t in range(n - 1)]
    assert np.mean(held) > 0.25  # some are re-struck down the same path on the next frame...
    for t in range(n - 2):  # ...but none on the frame after, and a new path is found at least every other frame
        assert not any(all(any(np.allclose(a, b) for b in ends[t + k]) for k in (1, 2)) for a in ends[t])
        assert any(not any(np.allclose(a, b) for b in ends[t]) for k in (1, 2) for a in ends[t + k])


def test_a_handheld_camera_carries_a_held_far_end_with_the_room():
    n, s = 24, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    cam = np.repeat(np.eye(3)[None], n, 0)
    cam[:, 0, 2] = 2.0 * np.arange(n)  # the room drifts 2 px right a frame
    bolts, _, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 0.8), camera=cam), seed=4)
    tip = np.asarray(track["landmarks"][0][8][:2])
    ends = [mains(b, tip, s) for b in bolts]
    moved = sum(any(np.allclose(a + (2.0, 0.0), b) for b in ends[t + 1]) for t in range(n - 1) for a in ends[t])
    stayed = sum(any(np.allclose(a, b) for b in ends[t + 1]) for t in range(n - 1) for a in ends[t])
    assert moved > 5 and stayed == 0


def test_a_louder_voice_strikes_further_thicker_and_forks_more():
    """One channel from the finger however loud; the louder, the further it reaches, the thicker it leaves the finger
    and the more it forks."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])

    def strikes(level):
        bolts, _, _ = L.arc_frames(hand, voice(np.full(n, level)), seed=6)
        reach, width, forks = [], [], []
        for b in bolts:
            main = [(p, wd) for p, wd, _ in b if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
            if main:  # (on a frame its stroke may be dark)
                reach.append(np.linalg.norm(main[0][0][-1] - tip) / s)
                width.append(main[0][1])
                forks.append(sum(not np.allclose(p[0], tip) and not np.allclose(p[-1], tip) for p, _, _ in b))
        return np.mean(reach), np.mean(width), np.mean(forks)

    (near, thin, few), (far, thick, many) = strikes(0.35), strikes(1.0)
    assert far > 1.5 * near and thick > 1.5 * thin and many > few


def test_one_main_channel_leaves_each_fingertip_with_dim_thin_forks():
    """However loud: one main channel leaves the fingertip (two or three streams side by side from one finger read as
    parallel cables), struck on nearly every frame; a bolt's forks are much dimmer and thinner and leave along its way;
    and drawn, only the main channel clips to white."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    tip = np.asarray(track["landmarks"][0][8][:2])
    hand = L._hand(plate, track)
    for level in (0.3, 0.6, 1.0):
        for seed in (6, 7):
            bolts, _, _ = L.arc_frames(hand, voice(np.full(n, level)), seed=seed)
            assert all(len(mains(b, tip, s)) <= 1 for b in bolts)
            assert sum(len(mains(b, tip, s)) for b in bolts) > 0.8 * n
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
    assert E[30, 50, 0] > 1.0 > E[30, 120, 0]  # (red: the glow round both is blue, a white needs red as well)


def test_a_main_channel_tapers_to_its_end_round_a_white_core_in_a_narrow_blue_rim():
    """Drawn alone, a main channel thins from where it leaves to a fraction of that at its far end (MAIN_TAPER: thick
    to the end, a loud one read as neon tubing); its core is white and the blue round it a narrow rim, the broad halo
    a third under what it was: at a phone's size, 15 and 40 px out from a loud channel its blue is under 0.5 and 0.25
    (it was 0.65 and 0.32)."""
    line = np.stack([np.full(120, 60.0), np.linspace(10, 230, 120)], 1)
    box = (0, 240, 0, 120)
    core = L.draw([(line, 6.0, 1.0)], box)[..., 0]
    wide = lambda rows: (core[rows, 40:80] > 0.1).sum(1).mean()  # noqa: E731
    assert wide(slice(14, 30)) >= 2.5 * wide(slice(210, 226))
    E, _ = L.layer([(line, 6.0, 1.0)], [], None, 300.0, box, L.GLOW_CAP["arc"], 0.0, np.zeros(2), [])
    glow = E[60:100, :, 2].mean(0)
    assert E[60:100, 60].min() > 1.0 and glow[75] < 0.5 and glow[100] < 0.25


def test_a_bolt_strikes_a_surface_in_the_frame_and_splashes_there_for_two_frames():
    """A main channel ends inside the frame, TARGET_PAD in from its edges, rather than running off it, and on what
    the plate shows in its way: with a floor's line across the picture above the fingertip, most strikes land on it.
    Where it strikes there is a white-hot spot while it shows, and short forks splash out of it."""
    n, s = 40, 60.0
    plate, track, mask = take(n, lambda t: (120, 330), s, wall=(150, 142, 128), texture=4.0)
    line = 120
    plate[:, :line] = (60, 50, 45)  # (a dark floor meeting a light wall, a hand and a half over the fingertip)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    bolts, spots, struck = L.arc_frames(hand, voice(np.full(n, 1.0)), seed=2, matte=mask, plate=plate)
    pad = L.TARGET_PAD * W
    ends, on_line = [], 0
    for t in range(n):
        main = [p for p, _, _ in bolts[t] if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
        for p in main:
            e = p[-1]
            assert pad - 1e-6 <= e[0] <= W - 1 - pad + 1e-6 and pad - 1e-6 <= e[1] <= H - 1 - pad + 1e-6
            on_line += abs(e[1] - line) < 0.15 * s
            ends.append(tuple(np.round(e, 6)))
            assert any(np.hypot(x - e[0], y - e[1]) < 1e-6 for x, y, _, _ in struck[t])  # (its white-hot spot)
    assert on_line > 0.6 * len(ends)
    splash = [p for t in range(n) for p, _, _ in bolts[t] if not np.allclose(p[0], tip) and
              any(np.allclose(p[0], e) for e in ends)]  # (out of where a channel strikes)
    assert splash and all(np.linalg.norm(p - p[0], axis=1).max() < 1.3 * L.SPLASH[1][1] * s for p in splash)


def test_a_tap_on_the_palm_strikes_from_a_fingertip():
    """With the emitter on the palm (no tip within FINGER_NEAR of it), the arcs leave the index or the middle
    fingertip, whichever reaches further along the aim, never the middle of the hand (from there a bolt ran up the
    fingers and a fork down the palm, drawn on the skin)."""
    n, s = 12, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, emitter="palm", middle_out=True)
    hand = L._hand(plate, track)
    assert L._fingers(hand, 0) in ([12, 8], [8, 12])
    bolts, spots, _ = L.arc_frames(hand, voice(np.full(n, 0.9), hits=[0]), seed=1, matte=mask)
    for t in range(n):
        tips = hand.lm[t, [8, 12]]
        assert all(np.linalg.norm(tips - (x, y), axis=1).min() < L.TIP_OUT * s + 1.0 for x, y, _, _ in spots[t])
        assert not mains(bolts[t], hand.xy[t], s)
    assert L._fingers(hand, 0, aim=315.0)[0] == 8  # (aimed up and left, the index's tip reaches furthest)


def crossings(chans: list[np.ndarray], free: list[float]) -> int:
    """How many times two of ``chans`` (polylines) cross in the picture, not counting where either is within its
    ``free`` px of where it leaves (a fork leaves on its channel, a splash where its channel strikes)."""
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


def test_no_fork_or_splash_crosses_or_rejoins_another_channel():
    """However loud, with one finger or two, held still or swept: no fork, splash or second finger's channel
    crosses another channel or rejoins its own away from where it leaves (a closed loop is the most obviously fake
    shape lightning can take), the crackle at the fingertips aside."""
    n, s = 24, 60.0
    for middle_out, speed, level in ((False, 0.0, 1.0), (False, 4.0, 0.9), (True, 0.0, 0.95), (True, 4.0, 1.0)):
        plate, track, _ = take(n, lambda t: (135 + speed * t, 330), s, middle_out=middle_out)
        hand = L._hand(plate, track)
        for seed in range(3):
            bolts, _, _ = L.arc_frames(hand, voice(np.full(n, level), hits=[0]), seed=seed)
            for t, frame in enumerate(bolts):
                tips = [np.asarray(track["landmarks"][t][f][:2]) for f in (8, 12)]
                chans = [p for p, _, _ in frame  # (the crackle: short arcs out of, up to and between the tips)
                         if not (min(np.linalg.norm(p[[0, -1], None] - tips, axis=-1).min(1)) < 1e-6
                                 and np.linalg.norm(p - p[0], axis=1).max() < 0.35 * s
                                 or min(np.linalg.norm(p[-1] - tp) for tp in tips) < 1e-6)]
                lengths = [np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in chans]
                free = [max(0.16 * s, (L.ROOM_SKIP + 0.02) * ln) for ln in lengths]
                assert crossings(chans, free) == 0, (middle_out, speed, seed, t)


def test_no_fork_or_splash_runs_alongside_another_channel():
    """A fork or a splash lies within ALONG's distance of another channel for at most ALONG's share of its length
    past where it leaves (a fork ran 30 px beside its channel for 150 px: two wires side by side)."""
    n, s = 24, 60.0
    plate, track, _ = take(n, lambda t: (135, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    near = 0.8 * L.ALONG[0] * s  # (a little inside the rule's own distance: it is drawn on whole pixels)
    for seed in range(3):
        bolts, _, _ = L.arc_frames(hand, voice(np.full(n, 1.0), hits=[0]), seed=seed)
        for frame in bolts:
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
    """Drawn with a brighter channel, a dim one (a fork) fades towards its end as TAPER says, to about a third of
    itself; the brightest drawn keeps its brightness end to end."""
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
        bolts, _, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 1.0), hits=[0, 12]), seed=seed)
        for frame in bolts:
            for p, wd, _ in frame:
                if np.allclose(p[0], tip) and np.isclose(wd, main_w):
                    lead = tip + np.array([0.0, -1.0]) * L.ARC_LEAD * np.linalg.norm(p[-1] - tip)  # (straight up)
                    assert np.linalg.norm(p - lead, axis=1).min() < 1e-6
                    assert abs(up_of(p[-1:], tip)[0]) < L.ARC_OFF + 1e-6


def test_the_crackle_waits_for_a_steady_finger_and_leaves_into_the_room():
    """Short arcs crackle out of a steady fingertip, none while it moves faster than CRAWL_STILL (smeared with it,
    they were broken hooks), each leaving within CRACKLE_FAN of the finger's aim, into the room rather than back over
    the finger; none crawls up the finger (drawn on it, an arc read as a drawing on the skin)."""
    n, s = 20, 60.0

    def crackle(speed):
        plate, track, _ = take(n, lambda t: (100 + speed * t, 330), s)
        bolts, _, _ = L.arc_frames(L._hand(plate, track), voice(np.full(n, 0.9)), seed=2)
        tips = [np.asarray(track["landmarks"][t][8][:2]) for t in range(n)]
        assert not any(np.allclose(p[-1], tips[t]) for t in range(n) for p, _, _ in bolts[t])  # (nothing up to it)
        return [p - tips[t] for t in range(n) for p, _, _ in bolts[t]
                if np.allclose(p[0], tips[t]) and np.linalg.norm(p - p[0], axis=1).max() < 0.35 * s]

    still = crackle(0.0)
    assert len(still) > n and not crackle(8.0)  # (CRAWL_STILL is 3.75 px a frame at 270 wide)
    assert all(abs(up_of(p[-1:], np.zeros(2))[0]) <= L.CRACKLE_FAN + 1e-6 for p in still)


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


def test_the_arcs_light_the_room_blue_white_and_the_hand_a_cool_white(monkeypatch):
    """On a grey wall the arcs' light is a cool blue-white, with more green than red in it (in the hand's violet, a
    wall went mauve); on the hand it is a cool white: every channel of the skin rises, blue the most against what the
    skin had of it (in a violet, the hand under a white bolt went purple)."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))
    n = 8
    plate, track, mask = take(n, lambda t: (120, 330), wall=(150, 150, 150), texture=0.0)
    out = L.render(plate, track, mask, voice(np.full(n, 0.9), hits=[0]), "arc", seed=2)
    d = (lin(out[4:]) - lin(plate[4:])).mean(0)
    wall = np.zeros((H, W), bool)
    wall[400:, 210:] = True  # (the wall below and right of the hand, away from the channels' own glow)
    wall &= mask[0] == 0
    hand = (mask[0] > 0) & (np.indices((H, W))[0] > 300)  # (the palm, away from the fingertip's own spot)
    dw, dh, ph = d[wall].mean(0), d[hand].mean(0), lin(plate[0])[hand].mean(0)
    assert dw[2] > dw[1] > dw[0] > 0
    assert dh.min() > 0 and dh[2] / ph[2] > max(dh[0] / ph[0], dh[1] / ph[1])


def test_the_room_light_lands_as_the_same_blue_on_a_tinted_wall(monkeypatch):
    """On a wall already lavender in the plate the arcs' light is no more violet than on a grey one (its blue-white
    graded a lavender room violet)."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))
    n = 8
    far = np.zeros((H, W), bool)
    far[400:, 210:] = True  # (the wall below and right of the hand, away from the channels' own glow)

    def light(wall):
        plate, track, mask = take(n, lambda t: (120, 330), wall=wall, texture=0.0)
        out = L.render(plate, track, mask, voice(np.full(n, 1.0), hits=[0]), "arc", seed=2)
        return (lin(out[4:]) - lin(plate[4:]))[:, far & (mask[0] == 0)].mean((0, 1))

    grey, lavender = light((150, 150, 150)), light((150, 128, 148))
    assert lavender[2] / lavender[1] < 1.05 * grey[2] / grey[1] and lavender[0] / lavender[1] < 1.05 * grey[0] / grey[1]


def test_in_a_dark_room_the_bolts_light_the_hand_and_the_room_from_their_own_image():
    """In a dark bathroom the brightest thing in the frame lit nothing (the palm 94 -> 97, the tub's rim 37 -> 35).
    Over a dark plate the bolts' own image, blurred wide (KEY), lights the hand from about 55 to well over 100 (code
    values), a cool white, its blue rising at least 50; the room round them comes up, more near them than far off; and
    the light flickers with the strikes, under the flash limit."""
    n = 24
    plate, track, mask = take(n, lambda t: (120, 330), wall=(30, 28, 30), palm=(70, 50, 45), texture=0.0)
    out = L.render(plate, track, mask, voice(np.full(n, 0.9), hits=[0, 12]), "arc", seed=2)
    hand = (mask[0] > 0) & (np.indices((H, W))[0] > 300)
    g_out = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in out[4:]]).astype(np.float32)
    g_plate = cv2.cvtColor(plate[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    assert g_plate[hand].mean() < 60 and g_out[:, hand].mean() > 100
    rise = (out[4:].astype(np.float32) - plate[4:].astype(np.float32))[:, hand].mean((0, 1))
    assert rise[2] >= 50 and rise[2] > rise[0] > 0
    yy, xx = np.indices((H, W))
    near, far = (mask[0] == 0) & (yy < 200), (mask[0] == 0) & (yy > 430) & (xx > 200)
    lift = (g_out - g_plate[None]).mean(0)
    assert lift[near].mean() > 1.5 * lift[far].mean() > 0
    assert np.ptp((lin(out[4:]) @ LUMA)[:, hand].mean(1)) > 0.01  # (it flickers with the strikes...)
    assert flashes(out, plate) < 2.0  # (...but no more than a flash or so a second)


def test_two_fingers_at_the_voices_peak_strike_a_channel_each_not_a_thicket():
    """Two fingers over a dark wall at the voice's peak: one main channel each, kept PAIR_LIFE frames and re-struck
    on each rather than left dark, so the room flashes less than twice a second for any seed (new channels on and off
    every frame flashed it 3 times a second, the photosensitivity limit)."""
    n, s = 36, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True, wall=(142, 122, 139), palm=(200, 140, 125))
    e = np.clip(0.93 + 0.07 * np.sin(np.arange(n) / 3.0), 0, 1)
    tl = voice(e, hits=[0, 20])
    hand = L._hand(plate, track)
    tips = [np.asarray(track["landmarks"][0][f][:2]) for f in (8, 12)]
    thick = 0.9 * L._width_k(s) * (L.ARC_WIDTH[0] + (L.ARC_WIDTH[1] - L.ARC_WIDTH[0]) * e.min())  # (main channels')
    for seed in range(5):
        bolts, _, _ = L.arc_frames(hand, tl, seed=seed)
        for b in bolts:
            assert len(mains(b, tips[0], s)) <= 1 and len(mains(b, tips[1], s)) <= 1
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
        bolts, _, _ = L.arc_frames(hand, tl, seed=seed)
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
            bolts, spots, _ = L.arc_frames(hand, voice(np.full(n, 0.95), hits=[0]), seed=seed, matte=mask)
            tip12 = hand.lm[0, 12]
            assert all(len(sp) == 1 for sp in spots)
            assert not any(np.allclose(p[0], tip12) or np.allclose(p[-1], tip12) for b in bolts for p, _, _ in b)
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True)
    assert L._pair_out(L._hand(plate, track), [8, 12], mask).all()


def test_the_bolts_go_behind_the_hand_not_across_it():
    """Struck back down across the hand (the index up, the studio's arrow pointing down over the palm and the forearm):
    no fork lies on the hand's matte for more than ON_HAND's share of its length, and drawn, nothing of the bolts shows
    over the hand further than HIDE_NEAR from the fingertip (left on top out to half a hand, a channel ran down the
    palm like a drawing on the skin), nor where they strike behind it: the hand hides them."""
    n, s = 12, 60.0
    plate, track, mask = take(n, lambda t: (135, 230), s, texture=0.0)
    tl = voice(np.full(n, 1.0), hits=[0], aim=180.0)
    hand = L._hand(plate, track)
    bolts, _, _ = L.arc_frames(hand, tl, seed=1, matte=mask)
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
        far = inside & (np.hypot(xx - tip[0], yy - tip[1]) > max(L.HIDE_NEAR[1], 3.0 * L.CORONA) * s)  # (the tip's
        # own burning spot aside)
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
            bolts, _, _ = L.arc_frames(hand, voice(np.full(n, 1.0), hits=[0, 8]), seed=seed, matte=mask)
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
    two turned apart so they separate, one channel each. They light the room as one finger
    does, not twice as much."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))  # (the light alone)
    n, s = 12, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True)
    hand = L._hand(plate, track)
    assert L._fingers(hand, 0) == [8, 12]
    tl = voice(np.full(n, 0.9))
    bolts, _, _ = L.arc_frames(hand, tl, seed=1)
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
    bolts, spots, _ = L.arc_frames(L._hand(plate, track), voice(e), seed=2)
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


def test_the_arcs_do_not_depend_on_the_balls_look(monkeypatch):
    """The arcs are drawn and lit with none of the ball's look: with every constant of the ball's body, its light and
    its tone set to something else, the arcs' frames are the same bytes (the ball can be reworked without touching
    them)."""
    n = 10
    plate, track, mask = take(n, lambda t: (110 + 2 * t, 330))
    tl = voice(np.r_[np.zeros(2), np.full(8, 0.9)], hits=[2])
    before = L.render(plate, track, mask, tl, "arc", seed=4)
    rgb = np.array([1.0, 0.1, 0.5], np.float32)
    for name, value in (("PLASMA_RGB", rgb), ("BODY_RGB", rgb), ("HALO_RGB", rgb), ("BALL_GLOW_RGB", rgb),
                        ("BALL_HOT", (9.0, 0.6)), ("BODY", (5.0, 0.5)), ("HALO", (2.0, 2.0)), ("BALL_TAU", (9.0, 1.0)),
                        ("HALO_VOICE", (2.0, 3.0)), ("BRIGHT_TONE", (0.3, 0.5)), ("SCATTER", 1.0), ("WRAP_GAIN", 20.0),
                        ("SPILL", (3.0, 5.0)), ("SHELL", (30, 40)), ("GROUND", (1, 2)), ("LIFT_PX", (0.0, 1.0))):
        monkeypatch.setattr(L, name, value)
    monkeypatch.setitem(L.LIGHT, "ball", (0.1, 20.0, 20.0, 5.0))
    monkeypatch.setitem(L.LIGHT_RGB, "ball", (rgb, rgb))
    monkeypatch.setitem(L.KEY, "ball", (50.0, 50.0, 1.0))
    monkeypatch.setitem(L.KEY_RGB, "ball", (rgb, rgb))
    assert np.array_equal(L.render(plate, track, mask, tl, "arc", seed=4), before)


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
    fading in; a palm seen from above (its fingers further from the camera than its wrist) holds the ball well up the
    picture from the palm's point, LIFT_PX off it (at its point it was a flare stamped on the palm), wandering a few px
    round there (WANDER: held dead still it was pasted on), the palm's point kept as where its light pools."""
    n, s = 20, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    for lm in track["landmarks"]:
        for p in lm:
            p[2] = 1.2 * (300 + s - p[1])  # (the higher in the picture, the further away)
    e = np.r_[np.zeros(3), 0.21, 0.13, 0.2, np.full(n - 6, 0.9)]
    hand = L._hand(plate, track)
    _, sparks, balls = L.ball_frames(hand, voice(e), seed=1)
    assert all(b is None for b in balls[:3]) and all(b is not None for b in balls[3:])
    assert balls[3].level < balls[5].level < balls[10].level  # faded in
    k = W / 1080.0
    wander = L.WANDER[0] * k
    off = np.array([(b.px - b.x, b.py - b.y) for b in balls[3:]])
    assert np.allclose(off + [(b.x, b.y) for b in balls[3:]], np.asarray(track["xy"][3:]))  # (its palm point)
    assert np.all(off[:, 1] > L.LIFT_PX[0] * k - wander) and np.all(off[:, 1] < L.LIFT_PX[1] * k + wander)
    assert np.all(np.abs(off[:, 0]) <= wander + 1e-6)
    assert 0.3 * wander < np.ptp(off[:, 0]) and 0.3 * wander < np.ptp(off[:, 1])  # (it wanders)


def test_the_ball_grows_with_the_held_voice_and_shrinks_when_it_stops():
    """It grows from a point while the voice holds, over about 5 frames rather than popping into being (its body on
    its first frame at most 0.35 of its size 6 frames on: sized from its charge, it was 60-65 % of its full size), and
    shrinks when the voice stops, its body as dense as ever while it is a ball at all (it shrinks rather than fading
    out); a ball barely charged has no body, only its shell's crackle, reaching past where its skin will be."""
    n, s = 40, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.r_[np.zeros(4), np.full(24, 0.9), np.zeros(12)]
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(e), seed=0)
    assert all(b is None for b in balls[:4])
    r = [b.R / s for b in balls[4:28]]
    lo, hi = L.BALL_R
    assert r[-1] > lo + 0.6 * (hi - lo) and np.all(np.diff(r) > 0)
    first = next(t for t in range(4, n) if L._dense(balls[t].level, balls[t].charge) > 0)
    assert balls[first].R <= 0.35 * balls[first + 6].R
    assert balls[31] is None or balls[31].R < 0.7 * balls[27].R
    assert L._dense(balls[30].level, balls[30].charge) == 1.0  # (shrinking, still dense)
    assert L._dense(0.05, 0.01) == 0.0
    out, veil = np.zeros((600, 600, 3), np.float32), np.zeros((600, 600), np.float32)
    L._sphere(out, veil, (0, 600, 0, 600), L.Ball(300.0, 300.0, 40.0, 0.05, 0.01, 1.0))  # (barely charged)
    assert not out.any() and not veil.any()
    c, lm = np.asarray(balls[4][:2]), np.asarray(track["landmarks"][4])[:, :2]
    own = [p for p, _, _ in bolts[4] if np.linalg.norm(lm - p[-1, :2], axis=1).min() > 1e-6]  # (none to the hand)
    assert max(np.linalg.norm(p[:, :2] - c, axis=1).max() for p in own) > 1.2 * balls[4].R


def test_the_ball_is_ringed_by_a_shell_of_arcs_and_jumps_to_the_fingers():
    """Its arcs crawl over its skin brightest where they run along its outline, which draws it (filaments from its
    middle out were a radial starburst: a flare), and GROUND's arcs jump from its skin to the fingertips and the
    fingers' bases, spread over them (two at most, now and then, read as stray sparks)."""
    n, s = 24, 80.0
    plate, track, _ = take(n, lambda t: (135, 300), s, emitter="palm")
    hand = L._hand(plate, track)
    tl = voice(np.full(n, 0.95), hits=[0])
    bolts, _, balls = L.ball_frames(hand, tl, seed=0)
    rim, face, jumps = [], [], []
    for t in range(8, n):
        b, lm = balls[t], hand.lm[t]
        to = [p for p, _, _ in bolts[t] if np.linalg.norm(lm - p[-1], axis=1).min() < 1e-6]
        ends = {int(np.argmin(np.linalg.norm(lm - p[-1], axis=1))) for p in to}
        assert len(to) >= L.GROUND[0] - 2 and len(ends) >= min(len(to), 4)  # (to the fingers, spread over them)
        assert ends <= set(L.TARGETS)
        jumps.append(len(to))
        for p, _, it in bolts[t]:
            if any(p is q for q in to):
                continue
            r = np.linalg.norm(p - (b.x, b.y), axis=1).mean() / b.R
            (rim if r > 0.9 else face).append(it)
    assert np.mean(jumps) >= L.GROUND[0]
    assert len(rim) > 3 * (n - 8) and np.mean(rim) > np.mean(face)


def test_the_ball_is_a_blue_light_over_the_palm_that_keeps_its_lines(monkeypatch):
    """A palm already at the top of red in the picture: the ball is the brightest thing over it and reads blue, not a
    white disc: its white-hot middle well over the luminance of the palm round it, and its body never darker towards
    its edge than the skin just outside it (a dark limb made it a glass marble). It hides the palm behind its dense
    middle, but its outer part is light over the palm, whose lines show through it; the palm just past its edge is lit
    and keeps its lines; the camera exposes down for it, and the wall round it is lit, a cool white, against the room
    further off."""
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
    assert lum[(d < 0.2) & on].mean() > 1.5 * lum[(d > 1.75) & (d < 2.5) & on].mean()  # a light: the brightest
    # thing there (its saturated blue has little luminance of its own)

    for name in ("SHELL", "GROUND"):  # (its body alone, the arcs crossing its edge aside)
        monkeypatch.setattr(L, name, (0, 0))
    body = lin(L.render(plate, track, mask, tl, "ball", seed=0)[t]) @ LUMA
    monkeypatch.undo()

    def ring(r: float) -> float:
        return float(body[(np.abs(d - r) < 0.03) & on].mean())

    assert ring(0.97) > ring(0.8) - 0.02 and ring(0.97) > ring(1.05) - 0.02  # (no dark limb)
    white = (o > 245).all(-1)
    assert white[under].mean() < 0.1  # its white-hot middle and channels, not a white disc
    assert (o[under, 2] - o[under, 0]).mean() > (p[under, 2] - p[under, 0]).mean() + 40  # blue
    lines = (o - out_flat[t]).mean(-1), (p - flat[t]).mean(-1)  # the palm's lines, as they come through

    def kept(sel: np.ndarray) -> float:
        return float(np.std(lines[0][sel]) / np.std(lines[1][sel]))

    assert kept((d < 0.3) & on) < 0.2 and kept((d > 0.8) & (d < 1.0) & on) > 0.25  # (hidden behind its dense
    # middle, showing through its outer part)
    round_it = (d > 1.15) & (d < 1.6) & on  # the palm just past its edge: lit, not veiled or milky
    assert lum[round_it].mean() > (lin(plate[t]) @ LUMA)[round_it].mean()
    assert kept(round_it) > 0.7
    d *= R / s
    near, far = (d > 0.6) & (d < 1.0) & ~on, (d > 1.8) & ~on
    near, far = (o - p)[near], (o - p)[far]
    assert far.mean() < -5  # the camera exposed down for it...
    assert near.mean() > far.mean() + 3  # ...and it lights the wall round it,
    assert (near[:, 2] - near[:, 0]).mean() > (far[:, 2] - far[:, 0]).mean() + 2  # a cool white


def hue(rgb) -> float:
    """Degrees: the hue of a colour as shown (code values)."""
    r, g, b = (float(v) / 255.0 for v in rgb)
    return 360.0 * colorsys.rgb_to_hsv(r, g, b)[0]


def hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_the_balls_body_is_white_hot_only_in_its_middle_and_a_clear_cyan_blue_round_that():
    """The body rolls off to white with its hue kept: white-hot only in its middle (saturation at most 0.1 within 0.15
    of its radius), brightest there and dimmer out to its edge (at 0.15 of its radius at least 1.2 times as bright as
    at 0.5), and a clear cyan-blue round that over the palm (hue 200-230, saturation 0.25-0.6 at 0.4-0.7: pale, at
    0.12-0.45, its inside read as blown out), its colour the arcs' own blue (within 25 degrees of their glow's hue: in
    a greener blue it read aqua beside them). Each channel clipped on its own, it was a flat pale-aqua disc at
    (205-215, 247-254, 253)."""
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

    assert lum[at(0.15)].mean() >= 1.2 * lum[at(0.5)].mean()
    body = (d > 0.4) & (d < 0.85)

    def sat(sel: np.ndarray) -> float:
        return colorsys.rgb_to_hsv(*(o[sel].mean(0) / 255.0))[1]

    assert sat(d < 0.15) <= 0.1  # (white in its middle)
    sel = (d > 0.4) & (d < 0.7) & (mask[t] > 0)
    assert 200.0 <= hue(o[sel].mean(0)) <= 230.0 and 0.25 <= sat(sel) <= 0.6
    assert ((o[body, 1] >= 250) & (o[body, 2] >= 250)).mean() < 0.1  # (not two channels clipped: pale aqua)
    line = np.stack([np.full(60, 50.0), np.linspace(10, 90, 60)], 1)
    E, _ = L.layer([(line, 4.0, 1.0)], [], None, 60.0, (0, 100, 0, 100), L.GLOW_CAP["arc"], 0.0, np.zeros(2), [])
    glow = E[30:70, 52:62].reshape(-1, 3).mean(0)  # (the arcs' glow, 2-12 px off a channel)
    assert hue_gap(hue(o[body].mean(0)), hue(gam(np.ascontiguousarray(glow[None, None] / glow.max()))[0, 0])) < 25.0


BALL_PROFILE = ((0.0, 250, 252, 255), (0.15, 232, 244, 255), (0.3, 175, 219, 255), (0.45, 161, 212, 254),
                (0.6, 178, 216, 254), (0.75, 204, 224, 253), (0.9, 230, 234, 252))  # (radii, r, g, b): the ball's body
# as shown, white-hot in its middle, a clear cyan-blue round that and the lit palm showing through its edge. Try it
# live's body (web/lib/lightning.ts' BODY_STOPS, held by web/lib/lightning.test.ts to its own copy of this profile)
# draws the pale one this replaced and is to follow it


def test_the_balls_body_is_the_profile_the_live_preview_draws():
    """At a phone's 1080x1920, at full charge over a palm, the median colour of each thin ring of the ball's body is
    within 8 levels of BALL_PROFILE, which the live preview's body is to be held to too: the two draw the same ball
    (the live one was a saturated cyan disc while the shot's was a pale one)."""
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
    most, not a sharp dark window through it (seen sharp, a glass marble). Its dense middle hides most of it; at 0.6 of
    its radius, where it begins to be light added over what is behind it, about a quarter of the step shows, soft."""
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
    assert np.abs(seen).max() < 0.35 * step and np.abs(np.diff(seen)).max() < 0.1 * step  # (seen sharp: 0.26, 0.22)


def test_the_ball_settles_the_exposure_once_for_a_phrase(monkeypatch):
    """Three sounds with a second's gap between each: the camera exposes down once, by at most EXPOSURE's stops, and
    holds there through the gaps instead of breathing with each sound (by 0.6 stops three times in 4 s, it pumped
    the whole room). After one sound and silence it holds while the ball fades, and comes back once it has gone
    (held a second and a half, the room stayed dark with no ball in it)."""
    depth, hand_gain, _, near_gain = L.LIGHT["ball"]
    monkeypatch.setitem(L.LIGHT, "ball", (depth, hand_gain, 0.0, near_gain))  # (the camera alone, not the ball's light)
    monkeypatch.setitem(L.KEY, "ball", (L.KEY["ball"][0], 0.0, L.KEY["ball"][2]))
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


def test_the_balls_light_leaves_the_skin_round_it_brighter_and_a_little_cooler_not_dyed():
    """Round the ball's edge (1.05-1.3 of its radius) a palm at the top of red takes its cool near-white light as
    light: clearly brighter (at least 1.5 times the plate exposed down), its texture at least 0.85 of the plate's and
    the edges of the fingers 0.75 (the edge and the haze of a ball hovering in front of them soften the skin next to
    it a little), and only a little cooler: a pale cool pink, its hue within 25 degrees of the plate's and
    turned towards pink, no more saturated than the plate's and still skin (saturation at least 0.08). Its light falls
    off with distance: the skin 2.6-3.4 radii out is lifted less than a third as much, by at most 1.35 times. An olive
    wall round its edge takes the faint blue haze of its light (hue 200-260, saturation at most 0.25). (In a saturated
    blue light, the skin there went fuchsia, hue 284-341, with hardly any more light on it: a dye; taken towards a cool
    white by a glare, it kept saturation 0.12, texture 0.8 and edges 0.84: frosted glass.)"""
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
    assert fine(g_o)[skin].std() >= 0.85 * fine(g_p)[skin].std()
    assert grad(g_o)[edge].mean() >= 0.75 * grad(g_p)[edge].mean()
    lift = Y[skin].mean() / (P[skin].mean() * exposed)
    assert lift >= 1.5
    h_skin, s_skin, _ = colorsys.rgb_to_hsv(*(o[skin].mean(0) / 255.0))
    h_plate, s_plate, _ = colorsys.rgb_to_hsv(*(plate[t][skin].astype(np.float32).mean(0) / 255.0))
    turn = (360.0 * (h_skin - h_plate) + 180.0) % 360.0 - 180.0
    assert -25.0 <= turn <= 0.0 and 0.08 <= s_skin <= s_plate  # (a pale cool pink, never fuchsia)
    tips = (d > 2.6) & (d < 3.4) & on
    assert Y[tips].mean() / (P[tips].mean() * exposed) - 1.0 < (lift - 1.0) / 3.0
    assert Y[tips].mean() / (P[tips].mean() * exposed) <= 1.35
    assert 200.0 <= hue(o[wall].mean(0)) <= 260.0 and colorsys.rgb_to_hsv(*(o[wall].mean(0) / 255.0))[1] <= 0.25


def test_the_balls_body_is_a_volume_of_light_with_no_outline():
    """The ball's body, as light added to the picture (the same over skin and wall): brightest in its middle and never
    brighter further out, falling off smoothly with no outline (between rings 0.05 of its radius apart, past half its
    radius, it drops by at most 15 % of its brightness at half its radius: a disc's edge drops all of it), a cool
    near-white in its middle, a cyan-blue at 0.6 of its radius and a faint cyan-blue haze at 1.6; and it hides what
    is behind it only where it is dense: nearly all behind its middle, under a third at its radius, nothing past 1.5."""
    out, veil = np.zeros((321, 321, 3), np.float32), np.zeros((321, 321), np.float32)
    L._sphere(out, veil, (0, 321, 0, 321), L.Ball(160.0, 160.0, 60.0, 1.0, 1.0, 1.0))
    d = np.hypot(*(np.indices(veil.shape) - 160.0)) / 60.0
    lum = out @ LUMA

    def ring(img: np.ndarray, r: float) -> np.ndarray:
        return np.median(img[np.abs(d - r) < 0.02], axis=0)

    rs = np.arange(0.0, 2.5, 0.05)
    profile = np.array([ring(lum, r) for r in rs])
    assert np.all(np.diff(profile) <= 1e-6)
    assert np.max(-np.diff(profile)[rs[1:] > 0.5]) <= 0.15 * ring(lum, 0.5)
    middle, body, haze = ring(out, 0.0), ring(out, 0.6), ring(out, 1.6)
    shown = L.hue_shoulder(middle)
    assert middle.min() > 1.0 and shown.min() / shown.max() >= 0.85  # (far over white: white-hot)
    assert body[2] > body[1] > body[0] and body[0] <= 0.5 * body[2]  # (cyan-blue)
    assert haze[2] > haze[1] > 1.5 * haze[0] and ring(lum, 1.6) < 0.03  # (cyan-blue, faint)
    assert ring(veil, 0.0) >= 0.9 and ring(veil, 1.0) <= 0.35 and ring(veil, 1.5) < 0.05


def test_the_balls_arcs_over_the_skin_are_white_hot_not_magenta():
    """An arc jumping from the ball to a fingertip is white-hot over the skin, as over the ball: the brightest 40 of
    its pixels on the skin more than 1.3 radii out are white (every channel at least 235, green at most 12 levels under
    red, blue at most 5 under green), and the glow beside it 2-3 px off its core is a cool white on skin, its hue within
    15 degrees of the plate's there (added over skin at the top of red in a blue or violet glow, it clipped red, left
    green behind and read as magenta-white scratches, (245, 197, 245))."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0, 4, 8])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    hand = L._hand(plate, track)
    bolts, _, balls = L.ball_frames(hand, tl, 0)
    picked, glow, under = [], [], []
    yy, xx = np.indices((H, W))
    for t in range(n):
        if balls[t] is None:
            continue
        x, y, R = balls[t][:3]
        core = np.zeros((H, W), np.uint8)
        for p, _, _ in bolts[t]:
            if np.linalg.norm(hand.lm[t] - p[-1, :2], axis=1).min() > 1e-6:
                continue  # (not an arc to a fingertip or knuckle)
            q = np.unique(np.round(p[:, :2]).astype(int), axis=0)
            keep = (np.hypot(q[:, 0] - x, q[:, 1] - y) > 1.3 * R) & (mask[t][q[:, 1], q[:, 0]] > 0)
            picked += [out[t][b, a] for a, b in q[keep]]
            cv2.polylines(core, [np.round(p[:, :2]).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
        beside = cv2.dilate(core, np.ones((7, 7), np.uint8)) & ~cv2.dilate(core, np.ones((3, 3), np.uint8))
        sel = (beside > 0) & (mask[t] > 0) & (np.hypot(xx - x, yy - y) > 1.3 * R)
        glow += list(out[t][sel])
        under += list(plate[t][sel])
    picked = np.array(picked, float)
    assert len(picked) >= 100 and len(glow) >= 100
    top = picked[np.argsort(picked @ LUMA)[-40:]].mean(0)
    assert top.min() >= 235 and top[1] >= top[0] - 12 and top[2] >= top[1] - 5
    assert hue_gap(hue(np.mean(glow, 0)), hue(np.mean(under, 0))) <= 15.0


def test_the_balls_light_wraps_the_edges_of_the_fingers_facing_it():
    """A finger left of the ball catches its light on its right edge, facing the ball, not on its left one."""
    m = np.zeros((120, 160), np.float32)
    m[:, 40:60] = 1.0  # a finger; the ball's middle at (100, 60)
    box, wl = L.wrap_light(m, L.Ball(100.0, 60.0, 20.0, 1.0, 1.0, 1.0), 270)
    lit = np.zeros_like(m)
    lit[box[0]:box[1], box[2]:box[3]] = wl
    assert lit[60, 56:60].max() > 0.2 and lit[60, 40:45].max() < 0.02
    assert not lit[:, :40].any() and not lit[:, 60:].any()  # (on the hand only)


def test_a_knotted_shell_arc_is_drawn_again():
    t = np.linspace(0, 4 * np.pi, 200)
    assert L.knotted(np.stack([np.cos(t) * t, np.sin(t) * t], 1))  # a spiral
    assert not L.knotted(L.displace((0.0, 0.0), (1.0, 0.0), np.random.default_rng(1), 0.1, 0.02))
    assert sum(L.knotted(L._shell(np.random.default_rng(i))[0][0]) for i in range(200)) <= 4


def test_bad_inputs_are_refused():
    plate, track, mask = take(4, lambda t: (120, 330))
    with pytest.raises(ValueError, match="form"):
        L.render(plate, track, mask, voice(np.ones(4)), "orb")
    with pytest.raises(ValueError, match="timeline"):
        L.render(plate, track, mask, voice(np.ones(5)), "arc")
    with pytest.raises(ValueError, match="mask"):
        L.render(plate, track, mask[:, ::2], voice(np.ones(4)), "arc")
