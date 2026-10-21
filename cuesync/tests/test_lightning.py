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


def test_ball_revision_core_is_distributed_without_a_stationary_white_pin():
    """A charge has several hot filaments spread through its core, rather than a fixed Gaussian dot."""
    yy, xx = np.indices((241, 241))
    d = np.hypot(xx - 120, yy - 120) / 60.0
    positions = []
    for frame in (0, 3, 6):
        out = np.zeros((241, 241, 3), np.float32)
        veil = np.zeros((241, 241), np.float32)
        L._sphere(out, veil, (0, 241, 0, 241), L.Ball(120, 120, 60, 1, 1, 1), seed=3, frame=frame)
        lum = out @ LUMA
        assert np.mean(lum[d < 0.1]) < 2.0 * np.percentile(lum[(d > 0.2) & (d < 0.45)], 75)
        positions.append(np.unravel_index(np.argmax(lum), lum.shape))
    assert len(set(positions)) > 1


def test_ball_revision_fingers_cover_rear_volume_without_hiding_the_palm():
    """Finger silhouettes remove rear filaments while the palm underneath remains the lit receiver."""
    plate, track, mask = take(1, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    hand = L._hand(plate, track)
    cover = L.finger_cover(hand, 0, mask[0], (0, H, 0, W))
    assert cover[245, 111] > 0.5
    assert cover[315, 135] == 0
    assert not cover[mask[0] == 0].any()


def test_finger_occlusion_fades_into_the_palm_without_a_capsule_end():
    """The proximal finger blends into its knuckle instead of carving a hard pill-shaped hole in the charge."""
    plate, track, mask = take(1, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    hand = L._hand(plate, track)
    cover = L.finger_cover(hand, 0, mask[0], (0, H, 0, W))
    assert 0.15 < cover[292, 111] < 0.6
    assert cover[260, 111] > 0.95
    assert cover[315, 135] == 0


def test_finger_occlusion_edges_match_soft_phone_footage():
    """A crossing finger keeps its opaque centre but has a resolved soft transition into the luminous interior."""
    plate, track, mask = take(1, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    cover = L.finger_cover(L._hand(plate, track), 0, mask[0], (0, H, 0, W))
    assert np.count_nonzero((cover[260, 98:124] > 0.05) & (cover[260, 98:124] < 0.95)) >= 8
    assert cover[260, 111] > 0.95


def test_suspended_palm_source_lights_near_skin_more_than_the_wrist():
    """A reflected local pool distinguishes the skin immediately under the charge from the distant wrist."""
    plate, track, mask = take(4, lambda t: (135, 300), 80, emitter="palm", middle_out=True,
                              wall=(20, 20, 20), palm=(80, 60, 50), texture=0)
    out = L.render(plate, track, mask, voice(np.full(4, 0.9)), "arc", seed=1)
    lift = (out.astype(float) - plate) @ LUMA
    near = lift[-1, 300:323, 125:156].mean()
    wrist = lift[-1, 360:400, 125:156].mean()
    assert near > 2.0 * wrist and near > 25


def test_source_pool_has_strong_local_falloff_toward_the_wrist():
    """A hovering source has a compact reflection under it, rather than washing the whole forearm equally."""
    pool = L.source_pool((100, 100), (100, 250), 200, 1, 480, 270)
    assert pool is not None
    (y0, y1, x0, x1), field = pool
    full = np.zeros((480, 270), np.float32)
    full[y0:y1, x0:x1] = field
    assert full[160, 100] > 0.5 and full[280, 100] < 0.02
    assert full[400, 100] == 0


def test_ball_haze_has_no_visible_edge_at_the_cropped_layer_boundary(monkeypatch):
    """The optical halo reaches zero before its work buffer ends, avoiding a rectangular composite on a dark wall."""
    plate, track, _ = take(1, lambda t: (135, 300), wall=(30, 30, 30), palm=(30, 30, 30), texture=0)
    ball = L.Ball(135, 240, 40, 1, 1, 1)
    monkeypatch.setattr(L, "ball_frames", lambda *args: ([[]], [[]], [ball]))
    out = L.render(plate, track, None, voice([0.9]), "ball", seed=1)[0]
    assert np.abs(np.diff(out[240, 10:65].astype(float), axis=0)).max() <= 3


def test_ball_revision_contact_light_follows_the_finger_surface():
    """A finger's landing light is a narrow pad response along its axis rather than a circular white bead."""
    emission, _ = L.layer([], [(50, 50, 1.6, 2.5, 0.3)], None, 220, (0, 101, 0, 101),
                          L.GLOW_CAP["ball"], 0, np.zeros(2), [], land_axes=[(0, 1)])
    lum = emission @ LUMA
    assert lum[46:55, 50].sum() > 1.8 * lum[50, 46:55].sum()


@pytest.mark.parametrize("form", ["arc", "ball"])
def test_lightning_render_is_deterministic_across_worker_counts(monkeypatch, form):
    plate, track, mask = take(8, lambda t: (135, 300), 80, emitter="palm")
    timeline = voice(np.r_[np.zeros(2), np.full(6, .8)], hits=[2])
    monkeypatch.setattr(L.fx, "THREADS", 1)
    one = L.render(plate, track, mask, timeline, form, seed=29)
    monkeypatch.setattr(L.fx, "THREADS", 4)
    assert np.array_equal(one, L.render(plate, track, mask, timeline, form, seed=29))


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
    assert E[15:75, 50, 0].max() > 1.0 > E[15:75, 120, 0].max()  # only the main channel has white-hot sections


def test_a_loud_main_channel_forks_two_to_four_times_in_clear_dimmer_lines():
    """A main channel struck at a loud voice forks at least FORK_MIN times on nearly every strike (at 0.15-0.35 of its
    brightness and 0.3 of its width, the forks were dropped as running beside it or could not be seen: at phone size
    the bolts read as unbranched wires): each fork FORK_INTEN of its brightness, up to FORK_LEN long, leaving at most
    FORK_MOST off its way out."""
    n, s = 30, 60.0
    plate, track, mask = take(n, lambda t: (120, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    k = W / 1080.0
    counts, seen = [], set()
    for seed in range(3):
        for b in L.arc_frames(hand, voice(np.full(n, 1.0)), seed=seed, matte=mask)[0]:
            main = [(p, it) for p, _, it in b if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
            if not main or (seed, *np.round(main[0][0][-1], 4)) in seen:
                continue
            p, it = main[0]
            seen.add((seed, *np.round(p[-1], 4)))
            forks = [(q, iq) for q, _, iq in b if q is not p and np.linalg.norm(p[1:-1] - q[0], axis=1).min() < 1e-6]
            counts.append(len(forks))
            way = _unit_of(p[-1] - p[0])
            for q, iq in forks:
                assert L.FORK_INTEN[0] - 1e-6 <= iq / it <= L.FORK_INTEN[1] + 1e-6
                assert np.linalg.norm(np.diff(q, axis=0), axis=1).sum() < 1.6 * L.FORK_LEN[1] * k
                d = _unit_of(q[min(len(q) - 1, 6)] - q[0])  # (as it leaves)
                assert np.degrees(np.arccos(np.clip(d @ way, -1, 1))) < L.FORK_MOST + 35.0
    assert len(counts) > 30 and np.mean(np.array(counts) >= L.FORK_MIN) >= 0.85


def _unit_of(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


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
    assert E[15:130, 60, 0].max() > 1.0 and E[60:100, 60, 0].min() < 0.6
    assert glow[75] < 0.5 and glow[100] < 0.25


def test_a_bolt_strikes_an_edge_in_the_plate_with_a_contact_flash_and_a_splash():
    """With a floor's line across the picture above the fingertip, the main channels strike it, TARGET_BORDER inside
    the frame, never the frame's margin; each new strike lands somewhere new on it (one of the best few, TARGET_APART
    from the last). Where a bright stroke strikes there is a contact flash (where it touches, and which way the
    surface runs: along the line) and forks splash out of it on its first frames, SPLASH's length."""
    n, s = 40, 60.0
    plate, track, mask = take(n, lambda t: (120, 330), s, wall=(150, 142, 128), texture=4.0)
    line = 120
    plate[:, :line] = (60, 50, 45)  # (a dark floor meeting a light wall, nearly three hands over the fingertip)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    bolts, spots, struck = L.arc_frames(hand, voice(np.full(n, 1.0)), seed=2, matte=mask, plate=plate)
    border, k = L.TARGET_BORDER * W, W / 1080.0
    ends, on_line, flashed = [], 0, 0
    for t in range(n):
        main = [(p, it) for p, _, it in bolts[t] if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
        for p, it in main:
            e = p[-1]
            assert border - 1 <= e[0] <= W - border and border - 1 <= e[1] <= H - border
            on_line += abs(e[1] - line) < 6.0
            if not ends or tuple(np.round(e, 6)) != ends[-1]:  # (each new strike's end, in turn)
                ends.append(tuple(np.round(e, 6)))
            hit = [(x, y, ax, ay) for x, y, ax, ay, *_ in struck[t] if np.hypot(x - e[0], y - e[1]) < 1e-6]
            if it >= L.IMPACT_MIN:
                assert hit and abs(hit[0][3]) < 0.3  # (the flash, the surface running across the picture)
                flashed += 1
    assert on_line > 0.8 * sum(1 for t in range(n) for p, _, _ in bolts[t]
                               if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s)
    assert flashed > n // 2
    assert len(ends) > n // 3 and min(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(ends, ends[1:])) \
        >= L.TARGET_APART * k - 1  # (somewhere new each time)
    splash = [p for t in range(n) for p, _, _ in bolts[t] if not np.allclose(p[0], tip) and
              any(np.allclose(p[0], e) for e in ends)]  # (out of where a channel strikes)
    assert len(splash) > n and all(np.linalg.norm(p - p[0], axis=1).max() < 1.3 * L.SPLASH[1][1] * k for p in splash)


def test_with_nothing_to_strike_a_bolt_runs_off_the_frame_or_ends_in_a_brush():
    """Without a receiver image there is nothing to strike: a main channel runs off the frame where it would leave it
    within its reach, else ends in the air in a brush of short dim forks; neither has a contact flash. And none ends
    pinned to the frame's margin (struck as far as it could get inside it, every bolt ended in a bead there)."""
    n, s = 30, 60.0
    for at, aim in (((120, 330), None), ((135, 330), 270.0)):  # (up, two hands from the top; left, far from the edge)
        plate, track, mask = take(n, lambda t: at, s, texture=0.0)
        hand = L._hand(plate, track)
        tip = np.asarray(track["landmarks"][0][8][:2])
        e = np.r_[np.full(10, 0.4), np.full(n - 10, 1.0)]
        bolts, _, struck = L.arc_frames(hand, voice(e, aim=aim), seed=3, matte=mask)
        assert not any(struck)
        off = air = brushed = 0
        for t in range(n):
            for p, _, _ in bolts[t]:
                if not (np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s):
                    continue
                x, y = p[-1]
                if not (0 <= x < W and 0 <= y < H):
                    off += 1
                    assert max(-x, x - W, -y, y - H) > 0.5 * L.EXIT_PAD * W  # (well past the edge)
                else:
                    air += 1
                    brush = [q for q, _, _ in bolts[t] if np.linalg.norm(p[-len(p) // 5:] - q[0], axis=1).min() < 1e-6
                             and q is not p]
                    brushed += bool(brush)
                    assert min(x, W - 1 - x, y, H - 1 - y) > 0.08 * W  # (not on the margin)
        assert off and (air or aim is not None)  # (quiet, up the frame: in the air; loud, or left: off it)
        assert brushed >= 0.7 * air  # (it ends in a brush)


def striped(n: int, s: float, step: int = 24):
    """A take whose wall above the hand is striped across (dark and light bands ``step`` px high down to row 260): an
    edge to strike at every distance up the picture."""
    plate, track, mask = take(n, lambda t: (120, 330), s, wall=(150, 142, 128), texture=4.0)
    for y in range(0, 260, 2 * step):
        plate[:, y: y + step][mask[:, y: y + step] == 0] = (60, 50, 45)
    return plate, track, mask


def test_a_louder_voice_strikes_further_and_each_strike_lands_somewhere_new():
    """With something to strike at every distance up the picture, a loud voice strikes further than a quiet one (about
    TARGET_AT's diagonals: striking as far as it could, a burst's every stroke was the same length whatever the voice
    did), and a burst's strikes go back and forth between places TARGET_APART apart, a re-strike as far as
    TARGET_CONE's third off the aim (every strike of a 1.5 s burst ended in one 140 x 60 px patch)."""
    n, s = 48, 60.0
    plate, track, mask = striped(n, s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    k, diag = W / 1080.0, float(np.hypot(W, H))
    reach = {}
    for level in (0.3, 1.0):
        bolts, _, struck = L.arc_frames(hand, voice(np.full(n, level)), seed=4, matte=mask, plate=plate)
        ends = [(t, (x, y)) for t in range(n) for x, y, *_ in struck[t][:1]]
        dist = [float(np.hypot(*(np.array(e) - tip))) for _, e in ends]
        reach[level] = np.median(dist)
        want = diag * (L.TARGET_AT[0] + (L.TARGET_AT[1] - L.TARGET_AT[0]) * L.levels(voice([level]))[0])
        assert 0.6 * want < reach[level] < 1.4 * want
        new = [e for (t, e), (t0, e0) in zip(ends[1:], ends) if np.hypot(e[0] - e0[0], e[1] - e0[1]) > 1e-6]
        assert len(new) > n // 4 and len({(round(x), round(y)) for x, y in new}) > len(new) // 3
        steps = [np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(new[1:], new)]
        assert min(steps) >= L.TARGET_APART * k - 1  # (somewhere new each time)
        off = [abs(np.degrees(np.arctan2(e[0] - tip[0], tip[1] - e[1]))) for e in new]  # (the aim is up)
        assert max(off) <= L.TARGET_CONE[2] + 1
    assert reach[1.0] > 1.2 * reach[0.3]


def test_one_strike_in_four_lands_in_two_prongs():
    """About PRONG's share of the main channels that strike a surface end in two prongs, landing on two points PRONG's
    px apart, each with its contact flash."""
    n, s = 60, 60.0
    plate, track, mask = striped(n, s)
    hand = L._hand(plate, track)
    k = W / 1080.0
    two = one = 0
    for seed in range(3):
        _, _, struck = L.arc_frames(hand, voice(np.full(n, 1.0)), seed=seed, matte=mask, plate=plate)
        for at in struck:
            fresh = [(x, y) for x, y, ax, ay, g, first in at if first]
            if len(fresh) == 2:
                two += 1
                gap = np.hypot(fresh[0][0] - fresh[1][0], fresh[0][1] - fresh[1][1])
                assert L.PRONG[1][0] * k - 1 <= gap <= L.PRONG[1][1] * k + 1
            elif len(fresh) == 1:
                one += 1
    assert 0.08 < two / (one + two) < 0.45


def test_two_fingers_striking_nothing_run_well_apart():
    """Without a receiver image, two fingers held out together with nothing to strike: where either runs off into the air, their main channels
    end at least PAIR_APART apart as seen from their tips (pointing along a floor, both ran off the frame side by side:
    two stretched cables), or the second does not strike."""
    n, s = 30, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True, texture=0.0)
    hand = L._hand(plate, track)
    tips = [np.asarray(track["landmarks"][0][f][:2]) for f in (8, 12)]
    both = 0
    for seed in range(3):
        bolts, _, struck = L.arc_frames(hand, voice(np.full(n, 0.9), aim=300.0), seed=seed, matte=mask)
        assert not any(struck)
        for b in bolts:
            ends = [mains(b, tp, s) for tp in tips]
            if ends[0] and ends[1]:
                both += 1
                u = [(e[0] - tp) / np.linalg.norm(e[0] - tp) for e, tp in zip(ends, tips)]
                assert np.degrees(np.arccos(np.clip(u[0] @ u[1], -1, 1))) >= L.PAIR_APART - 0.5
    assert both > n // 2


def test_two_fingers_with_nothing_ahead_send_the_second_to_a_surface_round_them(monkeypatch):
    """Two fingers held out together pointing left over a blank wall, a dark patch high above them, well round from
    where they point (outside both fingers' first cones, inside TARGET_CONE's third): the first finger's channel runs
    off into the air, the second's strikes the patch's edge rather than run off beside it (both left the frame to the
    left side by side); with no wider cone to look in, it runs off too."""
    n, s = 30, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, middle_out=True, texture=0.0)
    plate[:, :150, 110:200] = (60, 50, 45)
    detector = L.surfaces
    monkeypatch.setattr(L, "surfaces", lambda p, m: [edges[edges[:, 2] >= 0] for edges in detector(p, m)])
    hand = L._hand(plate, track)
    tips = [np.asarray(track["landmarks"][0][f][:2]) for f in (8, 12)]

    def second_strikes() -> tuple[int, int]:
        lone = hit = 0
        for seed in range(3):
            bolts, _, struck = L.arc_frames(hand, voice(np.full(n, 0.9), aim=290.0), seed=seed, matte=mask,
                                            plate=plate)
            for t in range(n):
                ends = [mains(bolts[t], tp, s) for tp in tips]
                on = [any(np.hypot(x - e[0], y - e[1]) < 1e-6 for x, y, *_ in struck[t] for e in es) for es in ends]
                if ends[0] and not on[0]:
                    lone += 1
                    hit += on[1]
        return lone, hit

    lone, hit = second_strikes()
    assert lone > n and hit >= 0.8 * lone
    monkeypatch.setattr(L, "TARGET_CONE", L.TARGET_CONE[:2] + (L.TARGET_CONE[1],))
    assert second_strikes()[1] == 0


def test_forks_splashes_and_brushes_keep_inside_the_frame():
    """No fork, splash or brush runs off the frame or along its very edge: each keeps FORK_EDGE inside it (forks off a
    strike on a door's edge near the top ran on up it and out of the picture)."""
    n, s = 30, 60.0
    plate, track, mask = striped(n, s, step=16)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    edge = L.FORK_EDGE * W / 1080.0
    for seed in range(3):
        bolts, _, _ = L.arc_frames(hand, voice(np.full(n, 1.0), aim=330.0), seed=seed, matte=mask, plate=plate)
        for b in bolts:
            for p, _, _ in b:
                if np.linalg.norm(p[0] - tip) < 1e-6:
                    continue  # (a main channel or the crackle, from the fingertip: a channel may run off the frame)
                q = p[1:]
                assert (q >= edge - 1e-6).all() and (q[:, 0] <= W - 1 - edge + 1e-6).all() and \
                    (q[:, 1] <= H - 1 - edge + 1e-6).all()


def test_a_strike_on_a_surface_flashes_there_on_every_stroke():
    """Every stroke down a channel that strikes a surface, a re-stroke as well as a new one, has its contact flash, as
    bright as the stroke, but for one gone dark (only bright strokes had one: a dim re-stroke ended on a shower hose in
    a hook in the dark). Its narrow surface-following streak and faint scatter avoid an isolated clipped bead."""
    n, s = 40, 60.0
    plate, track, mask = striped(n, s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    bolts, _, struck = L.arc_frames(hand, voice(np.full(n, 0.9)), seed=5, matte=mask, plate=plate)
    restruck = 0
    for t in range(n):
        for x, y, ax, ay, g, fresh in struck[t]:
            restruck += not fresh
        for p, _, it in bolts[t]:
            if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s and it >= L.IMPACT_MIN and \
                    any(np.hypot(p[-1, 0] - x, p[-1, 1] - y) < 1e-6 for at in struck[max(0, t - 3): t + 1]
                        for x, y, *_ in at):
                assert any(np.hypot(p[-1, 0] - x, p[-1, 1] - y) < 1e-6 for x, y, *_ in struck[t])
    assert restruck > 0
    box = (0, 200, 0, 200)
    E, _ = L.layer([], [], None, 60.0, box, L.GLOW_CAP["arc"], 0.0, np.zeros(2), [], side=1080,
                   impacts=[(100.0, 100.0, 1.0, 0.0, 1.0, True)])
    y_ = fx_lum(E)
    half = 0.5 * y_.max()
    assert 30 <= (y_[100] >= half).sum() <= 60
    assert (y_[:, 100] >= half).sum() <= 4 and y_.max() < 1.0
    assert y_[100, 120] > 0.05 and y_[100, 120] > 4 * y_[100, 160]  # (in a glow about 40 px across)


def fx_lum(E: np.ndarray) -> np.ndarray:
    return E @ LUMA


def test_two_strikes_side_by_side_pool_their_brighter_not_their_sum():
    """The light a strike pools on what it strikes is a soft ellipse, along the surface at most 3 times as far as
    across it and feathered over 30 px and more across it at 1080; two strikes side by side light it as the brighter
    of the two, not their sum (summed, two strikes blew a baseboard to a hard white strip)."""
    h, w = 400, 600

    def full(got) -> np.ndarray:
        (y0, y1, x0, x1), pl = got
        out = np.zeros((h, w), np.float32)
        out[y0:y1, x0:x1] = pl
        return out

    one = full(L.pooled([(300.0, 200.0, 1.0, 0.0, 1.0, True)], 1080, h, w))
    two = full(L.pooled([(300.0, 200.0, 1.0, 0.0, 1.0, True), (330.0, 200.0, 1.0, 0.0, 1.0, True)], 1080, h, w))
    assert np.isclose(two.max(), one.max(), rtol=1e-3)
    assert np.allclose(two, np.maximum(one, full(L.pooled([(330.0, 200.0, 1.0, 0.0, 1.0, True)], 1080, h, w))))
    half = one >= 0.5 * one.max()
    ys, xs = np.nonzero(half)
    assert np.ptp(xs) <= 3.0 * max(np.ptp(ys), 1)
    across = one[:, 300]
    assert across[200 + 30] > 0.2 * across[200]  # (still lit 30 px off it, across the surface)


def test_in_a_dark_room_the_arcs_light_pulses_with_their_strokes():
    """In a dark room the light the arcs throw on the room pulses with them, STROBE's swing frame to frame and a dip for
    a frame every 5-8 frames (a steady wash under a bolt jumping every frame read as a lamp), and still never flashes
    more than 3 times a second over a large area; in a lit room it holds (strobe: 1 on every frame)."""
    n = 40
    plate, track, mask = take(n, lambda t: (120, 330), wall=(30, 28, 30), palm=(70, 50, 45), texture=0.0)
    out = L.render(plate, track, mask, voice(np.full(n, 0.9), hits=[0]), "arc", seed=2)
    yy, xx = np.indices((H, W))
    room = (mask[0] == 0) & (yy > 400) & (xx > 180)
    lift = (lin(out) @ LUMA - lin(plate) @ LUMA)[:, room].mean(1)[4:]
    assert np.std(lift) > 0.1 * np.mean(lift) and (lift < 0.6 * np.median(lift)).sum() >= 3
    assert flashes(out, plate) <= 3.0
    pulse = L.strobe(np.full(n, 0.9), 2)
    on = ~np.isclose(pulse, L.STROBE[4])
    assert (pulse[on] >= L.STROBE[0] - 1e-6).all() and (pulse[on] <= L.STROBE[1] + 1e-6).all()
    dips = np.flatnonzero(~on)
    assert len(dips) >= 4 and np.diff(dips).min() >= L.STROBE[3][0] and np.diff(dips).max() <= L.STROBE[3][1]
    assert np.array_equal(L.strobe(np.r_[np.full(10, 0.9), np.zeros(5)], 2)[10:], np.ones(5))
    assert np.allclose(L.strobe(np.full(n, 0.9), 2, hold=1.0), 1.0)


def test_over_a_bright_wall_a_bolt_stands_off_it():
    """Over a wall brighter than BRIGHT_WALL (a beige wall at 155, where a channel was a faint scribble that all but
    vanished at phone size) the main channel leaves the finger BRIGHT_WIDTH wider, each new stroke burns at full
    strength for BRIGHT_HOLD frames, and round it the wall is darker and bluer (a band, and a saturated blue glow);
    over a mid-grey wall none of that."""
    n, s, kk = 16, 240.0, 4
    tl = voice(np.full(n, 0.8), hits=[0])
    found = {}
    for wall in ((205, 195, 175), (110, 105, 100)):
        plate, track, mask = take(n, lambda t: (120 * kk, 330 * kk), s, wall=wall, texture=0.0, size=(W * kk, H * kk))
        hand = L._hand(plate, track)
        bright = L._bright(plate, mask)
        tip = np.asarray(track["landmarks"][0][8][:2])
        bolts, _, _ = L.arc_frames(hand, tl, seed=1, matte=mask, plate=plate)
        widths = [wd for b in bolts for p, wd, _ in b if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s]
        found[wall] = bright, max(widths)
        if bright < 1.0:
            continue
        strokes = []
        for b in bolts:
            main = [(tuple(np.round(p[-1], 3)), it) for p, wd, it in b if np.allclose(p[0], tip) and
                    np.isclose(wd, max(widths))]
            strokes.append(main[0] if main else None)
        runs, last = [], None
        for st in strokes:
            if st is not None and last is not None and st[0] == last[0]:
                runs[-1].append(st[1])
            elif st is not None:
                runs.append([st[1]])
            last = st
        assert all(np.allclose(r[:L.BRIGHT_HOLD], r[0]) for r in runs if len(r) >= L.BRIGHT_HOLD)
        out = L.render(plate, track, mask, tl, "arc", seed=1)
        core = (out[4].min(-1) >= 235) & (mask[4] == 0)
        dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
        band = (dist > 6) & (dist < 14) & (mask[4] == 0)
        o, pl = out[4][band].astype(np.float32), plate[4][band].astype(np.float32)
        assert (o @ LUMA).mean() < 0.97 * (pl @ LUMA).mean() or \
            (o[:, 2] - o[:, 0]).mean() > (pl[:, 2] - pl[:, 0]).mean() + 25
    (b1, w1), (b0, w0) = found[(205, 195, 175)], found[(110, 105, 100)]
    assert b1 == 1.0 and b0 == 0.0 and np.isclose(w1, (1.0 + L.BRIGHT_WIDTH) * w0)


def test_a_palm_tap_on_a_two_finger_gesture_still_uses_the_extended_tips():
    """A palm tap on a two-finger gesture still chooses its tips; only a spread open hand holds a central source."""
    n, s = 12, 60.0
    plate, track, mask = take(n, lambda t: (135, 330), s, emitter="palm", middle_out=True)
    for row in track["landmarks"]:
        for base in (13, 17):
            x = row[base][0]
            row[base:base + 4] = [[x, 333, 0], [x, 315, 0], [x, 324, -30], [x, 336, -70]]
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


def test_no_fork_runs_alongside_another_channel():
    """A fork lies within ALONG's distance of another channel for at most ALONG's share of its length once it has
    fanned out of its channel (a fork ran 30 px beside its channel for 150 px: two wires side by side)."""
    n, s = 24, 60.0
    plate, track, _ = take(n, lambda t: (135, 330), s)
    hand = L._hand(plate, track)
    tip = np.asarray(track["landmarks"][0][8][:2])
    near = 0.8 * L.ALONG[0] * W / 1080.0  # (a little inside the rule's own distance: it is drawn on whole pixels)
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


def test_a_branch_fades_along_its_length_into_a_fine_dim_tip():
    """A fork loses both current and width toward its tip, while its parent has independent hot sections."""
    line = np.stack([np.full(80, 30.0), np.linspace(10, 150, 80)], 1)
    other = line + (60.0, 0.0)
    box = (0, 160, 0, 120)
    branch = L.draw([(line, 2.0, 0.3), (other, 2.0, 1.0)], box)[..., 0]  # (the same stretches as the main: drawn first)
    start, end = slice(12, 25), slice(135, 148)  # (the first and last tenth of it)
    assert branch[end, 26:35].sum(1).mean() < 0.08 * branch[start, 26:35].sum(1).mean()
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


def test_a_fingertip_lit_by_the_arcs_records_cool_light_not_hot_pink():
    """A soft contact records cool pale light while the wider reflection keeps the photographed skin colour.
    The contact is neutral (saturation at most 0.15, blue over red); the rest keeps SKIN_KEEP of its saturation (under the
    arcs' violet blue a pointing finger went a grey mauve), and nowhere on the hand does red rise more than green and
    SKIN_RED (a cool light on skin already red lifted it most in red and blue: the fingertips went hot pink, the hand
    magenta)."""
    s = 60.0
    for palm, wall, most in (((75, 45, 49), (160, 150, 165), 100.0), ((225, 150, 130), (150, 142, 128), 70.0)):
        plate, track, mask = take(12, lambda t: (120, 330), s, palm=palm, wall=wall, texture=0.0)
        out = L.render(plate, track, mask, voice(np.full(12, 0.9), hits=[0]), "arc", seed=2)
        o, p = out[4:].astype(np.float32), plate[4:].astype(np.float32)
        skin = cv2.erode(mask[0], np.ones((7, 7), np.uint8))[None] > 0
        lift = o - p
        lit = skin & (lift @ LUMA > most)
        yy, xx = np.indices((H, W))
        tip = np.asarray(track["landmarks"][0][8][:2])
        contact = (np.hypot(xx - tip[0], yy - tip[1]) <= 0.06 * s) & (mask[0] > 0)
        hot = o[:, contact]
        c = np.mean([frame[np.argmax(frame @ LUMA)] for frame in hot], axis=0)
        assert c.min() >= 200  # optical spreading can take the tiny source below sensor clipping
        _, spots, _ = L.arc_frames(L._hand(plate, track), voice(np.full(12, 0.9), hits=[0]),
                                   seed=2, matte=mask, plate=plate)
        emission, _ = L.layer([], spots[4], None, s, (0, H, 0, W), L.GLOW_CAP["arc"], 0,
                               np.zeros(2), [], seed=2)
        source_luma = emission @ LUMA
        # Bound the source independently of already bright skin clipping under reflected illumination.
        assert np.count_nonzero(source_luma >= 0.5 * source_luma.max()) <= max(2, contact.sum() / 4)
        assert colorsys.rgb_to_hsv(*(c / 255.0))[1] <= 0.15 and c[2] >= c[0]
        rest = lit & (o.max(-1) < L.SKIN_KEEP_UNDER - 30)
        if rest.any():
            sat = (o.max(-1) - o.min(-1)) / o.max(-1)
            assert sat[rest].mean() >= 0.9 * L.SKIN_KEEP * float((p.max(-1) - p.min(-1)).mean() / p.max(-1).mean())
        moved = skin & (lift @ LUMA > 20)
        assert moved.sum() > 100 and lit.any()
        assert (lift[moved][:, 0] - lift[moved][:, 1]).max() <= L.SKIN_RED + 1.5


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
    values), a cool white keeping the skin's colour, its blue rising at least 50, as much as its red within a few
    levels; the room round them comes up, more near them than far off; and the light flickers with the strikes, under
    the flash limit."""
    n = 24
    plate, track, mask = take(n, lambda t: (120, 330), wall=(30, 28, 30), palm=(70, 50, 45), texture=0.0)
    out = L.render(plate, track, mask, voice(np.full(n, 0.9), hits=[0, 12]), "arc", seed=2)
    hand = (mask[0] > 0) & (np.indices((H, W))[0] > 300)
    g_out = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in out[4:]]).astype(np.float32)
    g_plate = cv2.cvtColor(plate[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    assert g_plate[hand].mean() < 60 and g_out[:, hand].mean() > 100
    rise = (out[4:].astype(np.float32) - plate[4:].astype(np.float32))[:, hand].mean((0, 1))
    assert rise[2] >= 50 and rise[2] > rise[0] - 5 > 0
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


def test_the_bolts_go_behind_the_hand_not_across_it(monkeypatch):
    """Struck back down across the hand (the index up, the studio's arrow pointing down over the palm and the forearm):
    no fork lies on the hand's matte for more than ON_HAND's share of its length, and drawn, nothing of the bolts shows
    over the hand further than HIDE_NEAR from the fingertip (left on top out to half a hand, a channel ran down the
    palm like a drawing on the skin), nor where they strike behind it: the hand hides them (their light on the hand
    aside, which lights it to a cool white by the fingertip)."""
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
    depth, _, room, near = L.LIGHT
    monkeypatch.setattr(L, "LIGHT", (depth, 0.0, room, near))  # (the channels alone over the hand)
    monkeypatch.setattr(L, "KEY", (0.0, *L.KEY[1:]))
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
    """When the voice stops a crackle stays at the burning tip, and RETURN's frames on the last channel is struck once
    more, half as bright, and fades over two frames: a return stroke down the same path."""
    n, s = 30, 60.0
    plate, track, _ = take(n, lambda t: (120, 330), s)
    e = np.zeros(n)
    e[:8] = 0.9
    bolts, spots, _ = L.arc_frames(L._hand(plate, track), voice(e), seed=2)
    tip = np.asarray(track["landmarks"][0][8][:2])
    assert all(mains(bolts[t], tip, s) for t in range(8))
    back = 7 + L.RETURN[0]
    for t in range(8, back):  # the voice has stopped: no stream, a crackle a fraction of a hand long, the tip burning
        assert bolts[t] and not mains(bolts[t], tip, s)
        assert max(np.linalg.norm(p - tip, axis=1).max() for p, _, _ in bolts[t]) < 0.8 * s
        assert spots[t] and spots[t][0][3] > 0
    for t, share in zip(range(back, back + 3), (L.RETURN[2], *L.RETURN[3])):
        ends = mains(bolts[t], tip, s)
        assert len(ends) == 1 and np.allclose(ends[0], mains(bolts[7], tip, s)[0])  # (down the same path)
        it = max(it for p, _, it in bolts[t] if np.allclose(p[0], tip) and np.linalg.norm(p[-1] - tip) > s)
        assert np.isclose(it, share * (0.55 + 0.45 * 0.9))  # (its share of a fresh stroke)
    assert not any(mains(bolts[t], tip, s) for t in range(back + 3, n))
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
                        ("SPILL", (3.0, 5.0)), ("SHELL", (30, 40)), ("GROUND", (0.5, 0.5)), ("LIFT_PX", (0.0, 1.0)),
                        ("BALL_LIGHT", (20.0, 20.0, 5.0, 0.1, 9.0)), ("BALL_COOL", 0.9), ("LAND", (5.0, 9.0, 9.0, 9.0)),
                        ("SHELL_GONE", (0.9, 0.8)), ("TIPS", (8,))):
        monkeypatch.setattr(L, name, value)
    monkeypatch.setitem(L.LIGHT_RGB, "ball", (rgb, rgb))
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


def test_the_light_goes_with_the_last_channel_and_holds_through_a_short_gap():
    """In a dark room the bolts light the hand and the room while a channel is on screen, a fifth of that (as shown)
    for one frame after the last, and then only the crackle at the tip lights them, faintly: between two strikes 4
    frames apart and after the last, no lit room with no bolt in it (let go over several frames, it stayed at 45 % of
    its lift, like a phone's exposure lagging), but for the return stroke down the last path (RETURN), which lights
    them as its own, dimmer bolt. A gap of LIGHT_BRIDGE frames between strokes holds it (a "ka-ka-ka" would strobe the
    room)."""
    n = 32
    plate, track, mask = take(n, lambda t: (120, 330), wall=(30, 28, 30), palm=(70, 50, 45), texture=0.0)
    e = np.zeros(n)
    e[2:10], e[14:20], e[22:24] = 0.9, 0.9, 0.9  # (a gap of 4 frames, then one of 2)
    out = L.render(plate, track, mask, voice(e, hits=[2, 14, 22]), "arc", seed=2)
    yy, xx = np.indices((H, W))
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in out]).astype(np.float32) - \
        cv2.cvtColor(plate[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    back = 23 + L.RETURN[0]  # (the return stroke after the last; the gap of 4 is too short for one)
    # The wrist receives weaker bounce than the room; its lift must still exceed output quantisation clearly.
    for sel, minimum in (((mask[0] == 0) & (yy > 400) & (xx > 180), 8.0),
                         ((mask[0] > 0) & (yy > 380), 3.0)):
        lift = gray[:, sel].mean(1)
        lit = lift[4:10].mean()
        assert lit > minimum
        assert lift[10] <= 0.25 * lit and lift[24] <= 0.25 * lift[22:24].mean()  # (a fifth, for one frame)
        assert lift[11:14].max() <= 0.08 * lit and lift[25:back].max() <= 0.08 * lit  # (then the crackle's alone)
        assert 0.08 * lit < lift[back] <= 0.75 * lit and lift[back + 3:].max() <= 0.08 * lit  # (its return stroke)
        assert lift[20:22].min() >= 0.7 * lift[16:20].mean()  # (held through the short gap)


def test_the_light_only_adds_to_the_plate(monkeypatch):
    """With the camera's exposure held, no pixel is darker than the plate's but in the narrow band round a channel over
    a bright wall (CONTRAST: at most its share darker, as a phone's sharpening rings a bright line): the roll-off to
    white that softens the channels never dims the plate's own highlights, and a window blown out to white stays
    white."""
    monkeypatch.setitem(L.EXPOSURE, "arc", (0.0, 1, 1.0, 1.0, 1))
    n = 8
    plate, track, mask = take(n, lambda t: (120, 330), wall=(250, 248, 245))
    plate[:, :60] = 255  # a window blown out to white
    tl = voice(np.full(n, 0.9), hits=[0])
    out = L.render(plate, track, mask, tl, "arc", seed=2)
    bolts, _, _ = L.arc_frames(L._hand(plate, track), tl, seed=2, matte=mask, plate=plate)
    band = np.zeros(out.shape[:3], np.uint8)
    for t in range(n):
        for p, wd, _ in bolts[t]:
            cv2.polylines(band[t], [np.round(p[:, :2]).astype(np.int32).reshape(-1, 1, 2)], False, 1,
                          int(wd + 2 * 3 * L.CONTRAST[1] * W / 1080.0) + 3)
    darker = (out.astype(int) < plate).any(-1)
    assert not (darker & (band == 0)).any() and (out[:, :60] == 255).all()
    shown = lin(out) @ LUMA / np.maximum(lin(plate) @ LUMA, 1e-6)
    assert shown[darker].min(initial=1.0) >= 1.0 - L.CONTRAST[0] - 0.02


def test_the_ball_shows_from_a_soft_start_without_sputtering_and_hovers_off_the_palm():
    """A voice that begins softly and dips for a frame before it swells shows the ball on every frame from its start,
    fading in; a palm seen from above (its fingers further from the camera than its wrist) holds the ball well up the
    picture from the palm's point, its lower envelope clear of the skin (at its point it was a flare stamped on the palm), wandering a few px
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
    radii = np.array([b.R for b in balls[3:]])
    lower_gap = off[:, 1] - radii
    # The visible lower envelope hovers above the palm; the centre also includes the sphere's radius.
    assert np.all(lower_gap > 0)
    assert np.all(lower_gap < L.LIFT_PX[1] * k + 0.5 * radii + wander)
    # The source can sit beside the fingers in the thumb-side opening, within roughly half its own radius.
    assert np.all(np.abs(off[:, 0]) <= .55 * radii + wander)
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


def jumps(bolts: list, sparks: list) -> list:
    """The arcs a ball throws to the fingers on a frame: the channels ending where a spark lands (LAND's, with a
    glow)."""
    lands = [np.array(sp[:2]) for sp in sparks if len(sp) == 5]
    return [p for p, _, _ in bolts if any(np.allclose(p[-1, :2], q) for q in lands)]


def thrown(hand, balls, t: int, bolts: list, sparks: list) -> list[tuple[np.ndarray, int, int, bool]]:
    """The arcs a ball throws to the fingers on frame t: (points, the landmark it goes to, which side of the ball that
    is, -1 or 1, as ball_frames splits them by its palm's way up turned a right angle, whether it is struck anew there:
    its landing spark at full strength)."""
    marks = list(L.TIPS) + list(L.GROUND_PAD[2])
    up = L._unit(L._palm_lift(hand)[t])
    across = np.array([-up[1], up[0]])
    out, left = [], [q for q, _, _ in bolts[t]]
    for sp in sparks[t]:
        if len(sp) != 5:
            continue
        p = next(q for q in left if np.allclose(q[-1, :2], sp[:2]))  # (several may land on one point: each once)
        left = [q for q in left if q is not p]
        f = marks[int(np.argmin(np.linalg.norm(hand.lm[t, marks] - p[-1], axis=1)))]
        side = 1 if float((hand.lm[t, f] - (balls[t].x, balls[t].y)) @ across) >= 0 else -1
        out.append((p, f, side, bool(np.isclose(sp[3], L.LAND[1]))))  # (fully shown: on a re-stroke 0.55-0.9 of it)
    return out


def test_the_ball_is_ringed_by_a_shell_of_arcs_and_throws_uneven_arcs_to_the_fingers(monkeypatch):
    """Its arcs crawl over its skin brightest where they run along its outline, which draws it (filaments from its
    middle out were a radial starburst: a flare). Its arcs to the fingers each land within LAND_NEAR of a landmark of
    theirs, spread over them, and come a side at a time: the two sides strike anew on the same frame only on a hit, at
    most three a side, now one side, now the other, now none, so the ball is not a pair of wings (both sides every
    frame, each an arch from its top) nor a spider (6-10 legs); now and then a short one to a finger's pad."""
    # Keep fingertips on both sides outside the formed charge; the full-size sphere can cover the near tips.
    monkeypatch.setattr(L, "BALL_R", (L.BALL_R[0], 0.35))
    n, s = 60, 80.0
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm")
    hand = L._hand(plate, track)
    tl = voice(np.full(n, 0.95), hits=[0, 40])
    bolts, sparks, balls = L.ball_frames(hand, tl, seed=0, matte=mask)
    k = W / 1080.0
    rim, seen, both_new, pads, spread = [], [], [], 0, set()
    rear_energy, rear_peak = [], []
    for t in range(8, n):
        b = balls[t]
        arcs = thrown(hand, balls, t, bolts, sparks)
        for p, f, _, _ in arcs:
            assert np.linalg.norm(hand.lm[t, f] - p[-1]) <= L.LAND_NEAR * k + 1e-6  # (on a finger)
        tips = [(f, sd, new) for _, f, sd, new in arcs if f in L.TIPS]
        pads += len(arcs) - len(tips)
        spread |= {f for f, _, _ in tips}
        seen.append(([sd for _, sd, _ in tips].count(-1), [sd for _, sd, _ in tips].count(1)))
        both_new.append(len({sd for _, sd, new in tips if new}) == 2)
        for p, _, it in bolts[t]:
            if any(p is q for q, *_ in arcs):
                continue
            r = np.linalg.norm(p - (b.x, b.y), axis=1).mean() / b.R
            if r > 0.9:
                rim.append(it)
        front = [c for c in bolts[t] if isinstance(c, L._Filament) and c.depth == 0]
        rear = [c for c in bolts[t] if isinstance(c, L._Filament) and c.depth > 0]
        front_image = L.draw(front, (0, H, 0, W), seed=0, ball=b)[..., 0] ** L.CORE_GAMMA
        rear_image = L.draw(rear, (0, H, 0, W), seed=0, ball=b)[..., 0] ** L.CORE_GAMMA
        assert front_image.max() > 0
        rear_energy.append(rear_image.sum() / front_image.sum())
        rear_peak.append(rear_image.max() / front_image.max())
    left, right = np.array(seen).T
    assert left.max() <= 3 and right.max() <= 3
    assert np.mean((left > 0) & (right > 0)) < 0.6 and np.mean(left + right <= 1) >= 0.25  # at least a quarter quiet
    assert np.mean(left > 0) > 0.2 and np.mean(right > 0) > 0.2 and len(spread) >= 3  # (both sides, in turn)
    assert not any(b_ for t, b_ in zip(range(8, n), both_new) if t not in tl.hits)  # (a side strikes a frame)
    assert pads >= 3
    assert len(rim) > n - 8  # enough disconnected surface paths to establish the shared envelope
    # Dim defocused fragments can outnumber the front paths without describing a bright wire cage.
    assert max(rear_energy) < 0.25 and max(rear_peak) < 0.2


def test_the_balls_shell_stays_on_its_skin_and_its_arcs_are_kinked_channels_through_the_air(monkeypatch):
    """Grown, its shell arcs lie within SHELL_SKIN of its skin as seen (thrown 30 % past it, they lay on the palm like
    glowing veins). Each arc to a finger leaves its skin within GROUND_LEAVE of the point facing where it goes (from its
    top, they rose as arches), bows GROUND_BOW of its chord through the air, either way, in straight steps (LEADER)
    kinked between them, its course never turning more than GROUND_TURN in all (one looped round a fingertip), and
    passes over the hand only as GROUND_ON lets it, coming down onto the finger (evenly wiggling, they were ribbons)."""
    monkeypatch.setattr(L, "BALL_R", (L.BALL_R[0], 0.35))
    n, s = 40, 80.0
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm")
    hand = L._hand(plate, track)
    bolts, sparks, balls = L.ball_frames(hand, voice(np.full(n, 0.95), hits=[0]), seed=0, matte=mask)
    k = W / 1080.0
    leave, kinks, bows, count = [], [], [], 0
    for t in range(12, n):
        b = balls[t]
        arcs = thrown(hand, balls, t, bolts, sparks)
        for channel in bolts[t]:
            p, _, _ = channel
            if not getattr(channel, "streamer", False) and not any(p is q for q, *_ in arcs):
                r = np.hypot(p[:, 0] - b.x, p[:, 1] - b.y) / b.R
                assert r.max() <= 1.0 + L.SHELL_SKIN + 0.10  # the contained shell stays separate from short escapes
        for p, f, _, _ in arcs:
            count += f in L.TIPS
            c, a, e = np.array([b.x, b.y]), p[0], p[-1]
            assert abs(np.hypot(*(a - c)) - b.R) < 1e-6  # (from its skin)
            leave.append(np.degrees(np.arccos(np.clip(L._unit(a - c) @ L._unit(hand.lm[t, f] - c), -1.0, 1.0))))
            assert L._swing(p) <= L.GROUND_TURN + 1e-6
            chord = e - a
            off = (p - a) @ np.array([-chord[1], chord[0]]) / np.hypot(*chord) ** 2
            bows.append(off[np.argmax(np.abs(off))])
            steps = p[::2]  # (its straight steps' ends)
            d = np.diff(steps, axis=0)
            ang = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))))
            kinks += list(ang[:-1])  # (the last, straight onto its end, aside)
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            q = np.round(p).astype(int)
            on = mask[t][np.clip(q[:, 1], 0, H - 1), np.clip(q[:, 0], 0, W - 1)] >= 128
            away = np.hypot(p[:, 0] - b.x, p[:, 1] - b.y) > L.GROUND_FREE * b.R
            free = away & (cum < cum[-1] - L.LAND_NEAR * k)
            over = free[1:] & free[:-1] & on[1:] & on[:-1]
            # Measure unwanted skin overlap against the whole route outside the sphere. The observed landing can
            # reserve less than LAND_NEAR, so removing that maximum from both numerator and denominator biases short routes.
            assert f not in L.TIPS or seg[over].sum() <= L.GROUND_ON[0] * seg[away[1:] & away[:-1]].sum() + 1e-6
    assert count > n - 12
    assert np.percentile(leave, 90) <= L.GROUND_LEAVE + 2.0
    assert min(bows) < -0.05 and max(bows) > 0.05  # (bowed either way)
    assert np.median(kinks) > 15.0  # (kinked, not wiggling)


def test_a_jumping_arc_is_straight_steps_kinked_twenty_to_fifty_degrees():
    """_leader: from (0, 0) to (1, 0), its steps (before each is broken a hair off its line) LEADER's px long, kinking
    20-50 degrees between them on most, and its course bowed as asked."""
    rng = np.random.default_rng(3)
    kinks, lengths = [], []
    for _ in range(50):
        q = L._leader(300.0, 0.25, rng, 1.0)
        assert np.allclose(q[0], 0.0) and np.allclose(q[-1], (1.0, 0.0))
        steps = q[::2] * 300.0
        d = np.diff(steps, axis=0)
        lengths += list(np.hypot(d[:-1, 0], d[:-1, 1]))
        kinks += list(np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))))[:-1])
        assert 0.1 < q[:, 1].max() < 0.45 and L._swing(q) < 180.0
    assert 9.9 <= min(lengths) and max(lengths) <= 40.1
    assert np.mean((np.array(kinks) >= 15.0) & (np.array(kinks) <= 60.0)) > 0.6


def test_the_balls_arcs_and_shell_go_with_it():
    """As the voice stops the ball shrinks; its arcs to the fingers thin out as it does, none once it is under
    GROUND_GONE of the largest it grew to, and its shell's crackle goes with its body (arcs crawled over the fingers
    for 4 frames after it had gone, with no source)."""
    n, s = 40, 80.0
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.r_[np.zeros(4), np.full(20, 0.9), np.zeros(16)]
    bolts, sparks, balls = L.ball_frames(L._hand(plate, track), voice(e, hits=[4]), seed=0, matte=mask)
    top = max(b.R for b in balls if b is not None)
    for t in range(24, n):
        b = balls[t]
        if b is None:
            assert not bolts[t]
            continue
        if b.R < L.GROUND_GONE * top:
            assert not jumps(bolts[t], sparks[t])
        if L._dense(b.level, b.charge) == 0.0:
            assert not any(it > 0 for _, _, it in bolts[t])  # (its body gone: its crackle too)
    sizes = [(balls[t].R, len(jumps(bolts[t], sparks[t]))) for t in range(18, n) if balls[t] is not None]
    assert sizes[-1][1] < sizes[0][1]  # (thinning out as it shrinks)


def test_the_ball_is_a_blue_light_over_the_palm_that_keeps_its_lines(monkeypatch):
    """Over already bright skin, hot knots remain brighter than the illuminated palm while the bulk stays faint.
    Foreground fingers keep their photographed colour and texture; the unoccluded volume scatters the view behind it.
    Its outer part has no dark outline, nearby skin keeps its lines, and the exposure and local wall light still respond.
    """
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    flat, _, _ = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), texture=0.0)
    out_flat = L.render(flat, track, mask, tl, "ball", seed=0)  # the same lightning over a palm with no lines
    t = n - 1
    hand = L._hand(plate, track)
    x, y, R = L.ball_frames(hand, tl, 0)[2][t][:3]
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - x, yy - y) / R
    on = mask[t] > 0
    clear = L.finger_cover(hand, t, mask[t], (0, H, 0, W)) < 0.2
    under = (d < 0.85) & on & clear
    o, p = out[t].astype(np.float32), plate[t].astype(np.float32)
    lum = lin(out[t]) @ LUMA
    hot = (d < 0.5) & on & clear
    assert hot.sum() > 100
    exposed_plate = (lin(plate[t]) @ LUMA) * 2.0 ** -L.EXPOSURE["ball"][0]
    # The sparse hot knots outshine their photographed receiver; a broad bright core would hide skin unnecessarily.
    assert np.percentile(lum[hot] / exposed_plate[hot], 99) > 1.15

    monkeypatch.setattr(L, "SHELL", (0, 0))  # (its body alone, the arcs crossing its edge aside)
    monkeypatch.setattr(L, "GROUND", (1.0,))
    monkeypatch.setattr(L, "GROUND_PAD", (0.0, *L.GROUND_PAD[1:]))
    body = lin(L.render(plate, track, mask, tl, "ball", seed=0)[t]) @ LUMA
    monkeypatch.undo()

    def ring(r: float) -> float:
        return float(body[(np.abs(d - r) < 0.03) & on].mean())

    assert ring(0.97) > 0.9 * ring(0.8) and ring(0.97) > ring(1.05) - 0.05  # no dark outline over the asymmetric limb
    white = (o > 245).all(-1)
    assert white[under].mean() < 0.1  # its white-hot middle and channels, not a white disc
    assert (o[under, 2] - o[under, 0]).mean() > (p[under, 2] - p[under, 0]).mean() + 40  # blue
    lines = (o - out_flat[t]).mean(-1), (p - flat[t]).mean(-1)  # the palm's lines, as they come through

    def kept(sel: np.ndarray) -> float:
        return float(np.std(lines[0][sel]) / np.std(lines[1][sel]))

    assert kept((d < 0.3) & on & clear) < 0.25 and kept((d > 0.8) & (d < 1.0) & on & clear) > 0.18  # (softened behind its dense
    # middle, showing through its outer part, seen through its blue: TRANSMIT)
    round_it = (d > 1.15) & (d < 1.6) & on  # the palm just past its edge: lit, not veiled or milky
    assert lum[round_it].mean() > (lin(plate[t]) @ LUMA)[round_it].mean()
    assert kept(round_it) > 0.3  # (lit 30-50 levels, skin already at the top of red rolls off)
    d *= R / s
    near, far = (d > 0.6) & (d < 1.0) & ~on, (d > 1.8) & ~on
    near, far = (o - p)[near], (o - p)[far]
    assert far.mean() < -2  # the camera exposed down for it...
    assert near.mean() > far.mean() + 3  # ...and it lights the wall round it,
    assert (near[:, 2] - near[:, 0]).mean() > (far[:, 2] - far[:, 0]).mean() + 2  # a cool white


def hue(rgb) -> float:
    """Degrees: the hue of a colour as shown (code values)."""
    r, g, b = (float(v) / 255.0 for v in rgb)
    return 360.0 * colorsys.rgb_to_hsv(r, g, b)[0]


def hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_the_balls_body_has_hot_regions_inside_a_clear_cyan_blue_volume():
    """Hot regions occupy the inside without requiring a stationary white centre. The volume retains a blue hue
    beside its white filaments, and avoids a broad disc of clipped green and blue."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    t = n - 1
    hand = L._hand(plate, track)
    bolts, spots, balls = L.ball_frames(hand, tl, 0)
    x, y, R = balls[t][:3]
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - x, yy - y) / R
    o = out[t].astype(np.float32)
    lum = lin(out[t]) @ LUMA

    clear = L.finger_cover(hand, t, mask[t], (0, H, 0, W)) < 0.2
    body = (d > 0.4) & (d < 0.85) & clear
    assert np.percentile(lum[(d < 0.8) & clear], 95) >= 1.15 * np.median(lum[body])

    def sat(sel: np.ndarray) -> float:
        return colorsys.rgb_to_hsv(*(o[sel].mean(0) / 255.0))[1]

    hot = (d < 0.8) & clear & (lum >= np.percentile(lum[(d < 0.8) & clear], 90))
    assert sat(hot) <= 0.25
    sel = (d > 0.4) & (d < 0.7) & (mask[t] > 0) & clear
    material, _ = L.layer(bolts[t], spots[t], balls[t], hand.s[t], (0, H, 0, W), L.GLOW_CAP["ball"],
                          0.5, np.zeros(2), [], seed=0, frame=t)
    own = gam(L.hue_shoulder(material)).astype(np.float32)
    # Open pockets reveal the receiving skin; colour is a property of the emitted material, not the holes.
    own_sat = colorsys.rgb_to_hsv(*(own[sel].mean(0) / 255))[1]
    assert 195.0 <= hue(own[sel].mean(0)) <= 240.0 and 0.15 <= own_sat <= 0.8
    assert ((o[body, 1] >= 250) & (o[body, 2] >= 250)).mean() < 0.1  # (not two channels clipped: pale aqua)
    line = np.stack([np.full(60, 50.0), np.linspace(10, 90, 60)], 1)
    E, _ = L.layer([(line, 4.0, 1.0)], [], None, 60.0, (0, 100, 0, 100), L.GLOW_CAP["arc"], 0.0, np.zeros(2), [])
    glow = E[30:70, 52:62].reshape(-1, 3).mean(0)  # (the arcs' glow, 2-12 px off a channel)
    assert hue_gap(hue(own[body].mean(0)), hue(gam(np.ascontiguousarray(glow[None, None] / glow.max()))[0, 0])) < 25.0


def test_the_balls_body_has_interior_contrast_without_a_bright_circular_rim():
    """The emitted volume retains contrast and a faint boundary at phone resolution. Material radiance is measured
    before hand reflections and foreground occlusion, which can legitimately outshine or hide its interior."""
    k, n = 4, 24
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135 * k, 300 * k), 80.0 * k, emitter="palm", palm=(80, 65, 60), wall=(45, 40, 35),
                              size=(W * k, H * k))
    hand = L._hand(plate, track)
    bolts, spots, balls = L.ball_frames(hand, tl, 0, mask)
    composite = L.render(plate, track, mask, tl, "ball", seed=0)
    yy, xx = np.indices(plate.shape[1:3])
    for t in (n - 6, n - 1):
        x, y, R = balls[t][:3]
        d = np.hypot(xx - x, yy - y) / R
        emission, _ = L.layer(bolts[t], spots[t], balls[t], hand.s[t], (0, H * k, 0, W * k),
                              L.GLOW_CAP["ball"], 1.5, np.zeros(2), [], seed=0, frame=t)
        y = gam(L.hue_shoulder(emission + 0.02)).astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        inside = y[(d > 0.25) & (d < 0.8)]
        rim = y[(d > 0.9) & (d < 1.05)]
        assert inside.std() >= 14.0
        assert inside.mean() >= rim.mean() - 5.0
        phone = cv2.resize(composite[t], (360, 640), interpolation=cv2.INTER_AREA).astype(np.float32) @ \
            np.array([0.299, 0.587, 0.114], np.float32)
        distance = cv2.resize(d.astype(np.float32), (360, 640), interpolation=cv2.INTER_AREA)
        cover = cv2.resize(L.finger_cover(hand, t, mask[t], (0, H * k, 0, W * k)), (360, 640),
                           interpolation=cv2.INTER_AREA)
        visible = (distance > 0.25) & (distance < 0.8) & (cover < 0.2)
        assert visible.sum() > 30 and phone[visible].std() >= 10.0


def test_the_hue_keeping_roll_off_keeps_a_little_over_exposed_blue_blue_and_takes_a_lot_to_white():
    blue = np.array([0.2, 0.5, 1.0], np.float32)
    P = np.stack([blue * k for k in (0.6, 1.3, 5.0)])
    Q = L.hue_shoulder(P)
    assert np.allclose(Q[0], P[0]) and (Q <= 1.0).all()
    assert hue_gap(hue(gam(Q[1:2])[0]), hue(gam(blue[None])[0])) < 8.0 and Q[1, 0] / Q[1, 2] < 0.4
    assert Q[2].min() / Q[2].max() > 0.85  # (white)


def test_a_gap_between_the_fingers_does_not_show_through_the_ball(monkeypatch):
    """A narrow contrasting gap behind dense charge is attenuated and softened instead of becoming a sharp window.

    Pin this optical fixture over the palm so changes to the ball's hovering height cannot move the input gap onto
    already identical wall pixels. The open outer volume is covered separately by texture-transmission assertions.
    """
    n, s = 12, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), texture=0.0)
    t = n - 1
    x, y, R = 135.0, 328.0, 40.0
    ball = L.Ball(x, y, R, 1, 1, 1)
    monkeypatch.setattr(L, "ball_frames", lambda *args, **kwargs: ([[] for _ in range(n)],
                                                                   [[] for _ in range(n)], [ball] * n))
    # Only transmission is compared; changing the receiver from skin to a gap must not change the lighting fixture.
    monkeypatch.setattr(L, "BALL_LIGHT", (0.0, 0.0, *L.BALL_LIGHT[2:]))
    monkeypatch.setattr(L, "WRAP_GAIN", 0.0)
    monkeypatch.setattr(L, "SPILL", (L.SPILL[0], 0.0))
    monkeypatch.setattr(L, "FILL", (0.0, *L.FILL[1:]))
    gap, gmask = plate.copy(), mask.copy()
    x0, x1 = int(x - 0.12 * R), int(x + 0.12 * R)
    gap[:, : int(y + 0.25 * R), x0:x1] = (150, 142, 128)
    gmask[:, : int(y + 0.25 * R), x0:x1] = 0
    a, b = (lin(L.render(pl, track, mk, tl, "ball", seed=0)[t]) @ LUMA for pl, mk in ((plate, mask), (gap, gmask)))
    row = int(y - 0.1 * R)
    seen = (b - a)[row, int(x - 0.5 * R): int(x + 0.5 * R)]  # what the gap does to the ball, across it
    step = abs(float((lin(gap[t]) @ LUMA - lin(plate[t]) @ LUMA)[row, int(x)]))
    assert step > 0.1
    # A single code value accommodates output quantisation; the gap must still have no sharp window edge.
    assert np.abs(seen).max() < 0.35 * step + 1 / 255 and np.abs(np.diff(seen)).max() < 0.2 * step


def test_the_ball_settles_the_exposure_once_for_a_phrase(monkeypatch):
    """Three sounds with a second's gap between each: the camera exposes down once, by at most EXPOSURE's stops (an
    eighth: at a third the whole room went 8-19 levels darker for a light a twentieth of the frame across), and holds
    there through the gaps instead of breathing with each sound (by 0.6 stops three times in 4 s, it pumped the whole
    room). After one sound and silence it holds while the ball fades, and comes back once it has gone (held a second
    and a half, the room stayed dark with no ball in it)."""
    monkeypatch.setattr(L, "BALL_LIGHT", (L.BALL_LIGHT[0], 0.0, *L.BALL_LIGHT[2:]))  # (the camera alone, not the
    # ball's light on the room)
    n, s = 80, 80.0
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm")
    e = np.zeros(n)
    e[4:20], e[34:48], e[60:76] = 0.9, 0.9, 0.9
    out = L.render(plate, track, mask, voice(e, hits=[4, 34, 60]), "ball", seed=0)
    far = np.zeros((H, W), bool)
    far[:70] = True  # (the wall at the top, far from the ball)
    stops = np.log2((lin(out) @ LUMA)[:, far].mean(1) / (lin(plate) @ LUMA)[:, far].mean(1))
    assert -L.EXPOSURE["ball"][0] - 0.03 < stops.min() < -0.08
    assert np.ptp(stops[20:78]) < 0.05  # (through both gaps)
    e[34:] = 0.0  # one sound, then silence: the camera holds while the ball fades, then comes back once it has gone
    out = L.render(plate, track, mask, voice(e, hits=[4]), "ball", seed=0)
    stops = np.log2((lin(out) @ LUMA)[:, far].mean(1) / (lin(plate) @ LUMA)[:, far].mean(1))
    balls = L.ball_frames(L._hand(plate, track), voice(e, hits=[4]), 0)[2]
    gone = next(t for t in range(20, n) if balls[t] is None)
    assert stops[gone - 1] < -0.08 and stops[-1] > -0.03 and np.all(np.diff(stops[gone:]) > -0.005)


def test_the_camera_holds_through_a_gap_between_phrases_and_comes_back_after_a_long_one():
    """The render looks ahead: the ball out for longer than it takes the camera to let go, but back within the frames
    it holds, the exposure holds through the gap (no rise over 0.05 stops: let go, it rose by 0.2 and fell again); out
    for longer than that, it comes back in between, most of the way."""
    bright = np.zeros(150)
    bright[0:20], bright[44:64], bright[120:140] = 0.9, 0.9, 0.9  # (gaps of 24 and 56 frames)
    dip = L.exposure_dip(bright, *L.EXPOSURE["ball"])
    assert np.ptp(dip[16:64]) < 0.05
    assert dip[110] < dip[64] - 0.08


def test_the_balls_light_lights_the_skin_round_it_and_nothing_further_off():
    """The ball is a light, not a tint: a palm at the top of red just past its edge (1.05-1.6 of its radius) is lit
    25-55 levels, a cool white with blue over red by 20-60 (a tint, not a dye: lifting it +2-6 while its blue rose
    +68-81 over its red read as an unlit hand under a grade), and further round it (1.3-2.0) lit still, keeping at least
    half its saturation, its hue skin's (never violet). Its light falls off with distance and is
    gone by BALL_LIGHT's reach: the skin 3.2 radii and more out and the room far off are the plate exposed down by at
    most EXPOSURE's eighth of a stop, their colour as it was (the far hallway shifted blue, a colour grade), while an
    olive wall round its edge records more added blue than red without requiring an opaque blue halo."""
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148), wall=(110, 104, 90))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    t = n - 1
    x, y, R = L.ball_frames(L._hand(plate, track), tl, 0)[2][t][:3]
    yy, xx = np.indices((H, W))
    d = np.hypot(xx - x, yy - y) / R
    o, p = out[t].astype(np.float32), plate[t].astype(np.float32)
    on = mask[t] > 0
    edge, palm, gone = (d > 1.05) & (d < 1.6) & on, (d > 1.3) & (d < 2.0) & on, (d > 3.2) & on
    luma = np.array([0.299, 0.587, 0.114], np.float32)

    def lift(sel):
        return (o - p)[sel].mean(0)

    assert 25.0 <= lift(edge) @ luma <= 55.0 and 20.0 <= lift(edge)[2] - lift(edge)[0] <= 70.0
    assert lift(palm) @ luma >= 15.0
    h_skin, s_skin, _ = colorsys.rgb_to_hsv(*(o[palm].mean(0) / 255.0))
    h_plate, s_plate, _ = colorsys.rgb_to_hsv(*(p[palm].mean(0) / 255.0))
    assert s_skin >= 0.5 * s_plate
    turn = (360.0 * (h_skin - h_plate) + 180.0) % 360.0 - 180.0
    assert -40.0 <= turn <= 5.0 and not 284.0 <= 360.0 * h_skin <= 341.0
    far = (yy < 70) & ~on
    for sel in (gone, far):
        stops = np.log2((lin(out[t]) @ LUMA)[sel].mean() / (lin(plate[t]) @ LUMA)[sel].mean())
        assert -L.EXPOSURE["ball"][0] - 0.02 <= stops <= 0.0
        assert hue_gap(hue(o[sel].mean(0)), hue(p[sel].mean(0))) < 2.0  # (no colour cast)
    wall = (d > 1.05) & (d < 1.6) & ~on
    wall_lift = (o - p)[wall].mean(0)
    assert wall_lift.mean() > 0 and wall_lift[2] > 1.5 * max(wall_lift[0], 1.0)


def test_the_balls_body_is_a_volume_of_light_with_no_outline(monkeypatch):
    """The ball's body, as light added to the picture (the same over skin and wall), its plasma aside (PLASMA):
    brightest in its middle and never brighter further out, falling off smoothly with no outline (between rings 0.05
    of its radius apart, beyond its compact core at 0.7 radii, it drops by at most 15 % of its half-radius brightness: a disc's
    edge drops all of it), a cool near-white in its middle, a cyan-blue at 0.6 of its radius and a faint blue haze at
    1.6; and it hides what is behind it only where it is dense: a translucent middle, under a third at its
    radius, nothing past 1.5."""
    monkeypatch.setattr(L, "PLASMA", (0.0, *L.PLASMA[1:5], 0.0))
    out, veil = np.zeros((321, 321, 3), np.float32), np.zeros((321, 321), np.float32)
    L._sphere(out, veil, (0, 321, 0, 321), L.Ball(160.0, 160.0, 60.0, 1.0, 1.0, 1.0))
    d = np.hypot(*(np.indices(veil.shape) - 160.0)) / 60.0
    lum = out @ LUMA

    def ring(img: np.ndarray, r: float) -> np.ndarray:
        return np.median(img[np.abs(d - r) < 0.02], axis=0)

    rs = np.arange(0.0, 2.5, 0.05)
    profile = np.array([ring(lum, r) for r in rs])
    assert np.all(np.diff(profile) <= 1e-6)
    assert np.max(-np.diff(profile)[rs[1:] > 0.7]) <= 0.15 * ring(lum, 0.5)
    middle, body, haze = ring(out, 0.0), ring(out, 0.6), ring(out, 1.6)
    assert middle[2] > middle[1] > middle[0] and middle @ LUMA > body @ LUMA  # a distributed blue-white core
    assert body[2] > body[1] > body[0] and body[0] <= 0.5 * body[2]  # (cyan-blue)
    assert haze[2] > haze[1] > 1.5 * haze[0] and ring(lum, 1.6) < 0.03  # (blue, faint)
    assert 0.4 <= ring(veil, 0.0) <= 0.95 and ring(veil, 1.0) <= 0.35 and ring(veil, 1.5) < 0.05


def test_the_balls_body_is_moving_plasma_not_an_airbrushed_marble():
    """Through its body (0.3-0.7 of its radius) its glow is a fifth or more brighter and darker than each ring's own
    (between its 10th and 90th percentiles), filaments of plasma, and they move: from one frame to the next the
    pattern changes at PLASMA's rate while the volume remains luminous. A smooth gradient there read as an
    airbrushed marble; sparse hot channels can rise well above the translucent body."""
    d = np.hypot(*(np.indices((321, 321)) - 160.0)) / 60.0
    rings = [(d >= r) & (d < r + 0.05) for r in np.arange(0.3, 0.7, 0.05)]
    seen, medians = [], []
    for frame in (0, 2):
        out, veil = np.zeros((321, 321, 3), np.float32), np.zeros((321, 321), np.float32)
        L._sphere(out, veil, (0, 321, 0, 321), L.Ball(160.0, 160.0, 60.0, 1.0, 1.0, 1.0), seed=3, frame=frame)
        lum = out @ LUMA
        rel = np.concatenate([lum[r] / np.median(lum[r]) for r in rings])
        lo, hi = np.percentile(rel, [10, 90])
        assert 0.15 <= lo <= 0.82 and 1.18 <= hi <= 8.0  # sparse hot channels above a translucent body
        seen.append(rel)
        medians.append([np.median(lum[r]) for r in rings])
    assert np.corrcoef(*seen)[0, 1] < 0.8
    assert 0.5 < np.mean(medians[0]) / np.mean(medians[1]) < 2.0


def test_the_balls_finger_contacts_are_white_hot_and_its_dim_leads_do_not_turn_skin_magenta(monkeypatch):
    """White landing contacts stay neutral while subdued long leads keep the receiving skin's hue.

    A quiet outer path should not leave a broad white cable on the skin just to satisfy the contact colour check.
    """
    # Keep this contact-colour fixture close to its fingers; longer leads intentionally carry less current.
    monkeypatch.setattr(L, "BALL_R", (L.BALL_R[0], .35))
    n, s = 24, 80.0
    tl = voice(np.full(n, 0.95), hits=[0, 4, 8])
    plate, track, mask = take(n, lambda t: (135, 300), s, emitter="palm", palm=(240, 165, 148))
    out = L.render(plate, track, mask, tl, "ball", seed=0)
    hand = L._hand(plate, track)
    bolts, sparks, balls = L.ball_frames(hand, tl, 0, mask)  # (as render draws them, round the hand's matte)
    picked, glow, under = [], [], []
    yy, xx = np.indices((H, W))
    for t in range(n):
        if balls[t] is None:
            continue
        x, y, R = balls[t][:3]
        core = np.zeros((H, W), np.uint8)
        for spot in sparks[t]:
            a, b = np.rint(spot[:2]).astype(int)
            patch = out[t, max(0, b - 1):b + 2, max(0, a - 1):a + 2].reshape(-1, 3)
            if len(patch):
                picked.append(patch[np.argmax(patch @ LUMA)])
        for p, *_ in thrown(hand, balls, t, bolts, sparks):
            cv2.polylines(core, [np.round(p[:, :2]).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
        beside = cv2.dilate(core, np.ones((7, 7), np.uint8)) & ~cv2.dilate(core, np.ones((3, 3), np.uint8))
        off = cv2.dilate(core, np.ones((15, 15), np.uint8)) & ~cv2.dilate(core, np.ones((11, 11), np.uint8))
        clear = (mask[t] > 0) & (np.hypot(xx - x, yy - y) > 1.3 * R)
        glow += list(out[t][(beside > 0) & clear])
        under += list(out[t][(off > 0) & clear])
    picked = np.array(picked, float)
    assert len(picked) >= 12 and len(glow) >= 100
    top = picked[np.argsort(picked @ LUMA)[-8:]].mean(0)
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


def test_arc_revision_a_bent_side_finger_does_not_emit_a_second_channel():
    """Depth can call a middle finger extended while its projected joints fold toward the index's side."""
    plate, track, _ = take(8, lambda t: (135, 330), middle_out=True)
    for row in track["landmarks"]:
        row[9:13] = [[135, 333, 0], [135, 303, 0], [148, 292, -50], [160, 285, -120]]
    hand = L._hand(plate, track)
    bolts, spots, _ = L.arc_frames(hand, voice(np.full(8, 0.95)), seed=1)
    side = hand.lm[0, 12]
    assert not any(np.allclose(p[0], side) for frame in bolts for p, _, _ in frame)
    assert all(len(frame) == 1 for frame in spots)


def test_arc_revision_a_curled_primary_finger_stops_emitting_on_that_frame():
    """An emitter selected while pointing cannot keep a cable rooted to its last pose when it curls."""
    plate, track, _ = take(12, lambda t: (135, 330))
    for row in track["landmarks"][6:]:
        row[6:9] = [[117, 315, 0], [117, 324, -30], [117, 336, -70]]
    bolts, spots, impacts = L.arc_frames(L._hand(plate, track), voice(np.full(12, 0.95)), seed=1)
    assert all(bolts[:6])
    assert not any(bolts[6:]) and not any(spots[6:]) and not any(impacts[6:])


def test_arc_revision_second_finger_switches_off_without_a_hysteresis_tail():
    """Smoothing extension can delay starting a channel, but a folded landmark must stop it immediately."""
    plate, track, _ = take(12, lambda t: (135, 330), middle_out=True)
    folded = landmarks(135, 330, 60, middle_out=False)
    for row in track["landmarks"][6:]:
        row[9:13] = folded[9:13].tolist()
    assert L._pair_out(L._hand(plate, track), [8, 12]).tolist() == [True] * 6 + [False] * 6


@pytest.mark.parametrize("lost,confidence,motion", [(True, 0.0, 0.0), (False, 0.3, 0.0), (False, 1.0, 15.0)])
def test_arc_revision_return_strokes_need_an_observed_tip_near_the_burst_end(lost, confidence, motion):
    """Lost confidence or a moving hand cannot restrike an old channel floating above the real fingertip."""
    plate, track, _ = take(22, lambda t: (120 + (motion if t >= 8 else 0), 330))
    track["conf"][8:] = [confidence] * 14
    if lost:
        track["src"][8:] = ["lost"] * 14
    energy = np.r_[np.full(8, 0.9), np.zeros(14)]
    hand = L._hand(plate, track)
    bolts, _, _ = L.arc_frames(hand, voice(energy), seed=2)
    assert all(bolts[:8])
    assert not any(mains(bolts[t], hand.xy[t], 60) for t in range(8, 22))


def test_arc_revision_the_main_core_has_local_width_and_brightness_changes():
    """A long channel has narrow dim necks between its hot thicker sections, beyond its overall taper."""
    p = np.c_[np.linspace(20, 480, 200), np.full(200, 30.0)]
    core = L.draw([(p, 5.0, 1.0)], (0, 61, 0, 500), seed=4)[..., 0]
    columns = core[:, 45:240]
    widths = (columns > 0.25).sum(0)
    peaks = columns.max(0)
    assert widths.max() >= 1.6 * widths.min()
    assert peaks.max() >= 1.7 * peaks.min()


def test_channel_current_variation_survives_phone_downsampling():
    """Bright and faint stretches remain distinct after bloom, the camera shoulder and phone reduction."""
    from cuesync.bending import shoulder

    p = np.c_[np.linspace(20, 1060, 300), np.full(300, 60.0)]
    e, _ = L.layer([(p, 5.0, 1.0)], [], None, 330.0, (0, 121, 0, 1080), L.GLOW_CAP["arc"],
                   1.2, np.zeros(2), [], seed=4)
    picture = e + 0.1
    shoulder(picture)
    phone = cv2.resize(gam(picture), (360, 40), interpolation=cv2.INTER_AREA).astype(np.float32)
    contrast = phone.mean(2) - 89.0
    peaks = contrast[:, 20:250].max(0)
    lo, hi = np.percentile(peaks, [10, 90])
    assert hi > 2.2 * lo


def test_rear_channel_disappears_through_core_while_front_crosses_it():
    """Core thickness hides a rear arc at its centre while the same front arc stays resolved."""
    p = np.c_[np.linspace(35, 205, 180), np.full(180, 120.0)]
    ball = L.Ball(120, 120, 70, 1, 1, 1)
    front = L.draw([L._Filament(p, 3, 1, 0)], (0, 241, 0, 241), seed=5, ball=ball)[..., 0]
    rear = L.draw([L._Filament(p, 3, 1, 2)], (0, 241, 0, 241), seed=5, ball=ball)[..., 0]
    ratio = rear.sum(0) / np.maximum(front.sum(0), 1e-6)
    assert np.median(ratio[100:140]) < 0.15
    assert np.median(ratio[45:60]) > 3 * np.median(ratio[100:140])


def test_a_soft_foreground_channel_is_not_extinguished_as_a_rear_channel():
    """Phone defocus and depth are independent: equally soft paths lose different light through the charge."""
    p = np.c_[np.linspace(35, 205, 180), np.full(180, 120.0)]
    box = (0, 241, 0, 241)
    ball = L.Ball(120, 120, 70, 1, 1, 1)
    near = L._Filament(p, 3, .4, 2, depth=0)
    rear = L._Filament(p, 3, .4, 2, depth=1)
    foreground = L.draw([near], box, seed=5, ball=ball)[..., 0]
    background = L.draw([rear], box, seed=5, ball=ball)[..., 0]
    np.testing.assert_array_equal(foreground, L.draw([near], box, seed=5)[..., 0])
    assert foreground[110:131, 110:131].sum() > 4 * background[110:131, 110:131].sum()


@pytest.mark.parametrize('scale', [1, 2])
def test_near_current_ends_recede_at_both_phone_and_larger_resolutions(scale):
    """A near channel's endpoint depth cue survives scaling past one pixel of optical softness."""
    p = np.c_[np.linspace(60, 180, 90), np.full(90, 120)] * scale
    box = (0, 241 * scale, 0, 241 * scale)
    ball = L.Ball(120 * scale, 120 * scale, 80 * scale, 1, 1, 1)
    plain = L._Filament(p, 3 * scale, 1, .65 * scale, 0)
    receding = L._Filament(p, 3 * scale, 1, .65 * scale, 0, .75)
    baseline = L.draw([plain], box, seed=3, ball=ball)[..., 0]
    faded = L.draw([receding], box, seed=3, ball=ball)[..., 0]
    assert faded[:, 60 * scale:75 * scale].sum() < .65 * baseline[:, 60 * scale:75 * scale].sum()
    assert faded[:, 110 * scale:130 * scale].sum() > .9 * baseline[:, 110 * scale:130 * scale].sum()


def test_local_palm_light_faces_the_source_and_preserves_finger_shadow():
    """A finger blocks new light beyond itself and its source-facing edge catches the local reflection."""
    m = np.zeros((300, 300), np.float32)
    m[70:250, 130:170] = 1
    lm = np.tile((150.0, 130.0), (21, 1))
    for tip in L.TIPS:
        lm[tip - 2:tip + 1] = [(150, 90), (150, 130), (150, 170)]
    got = L.source_pool((90, 130), (130, 200), 140, 1, 300, 300, matte=m, landmarks=lm)
    (y0, y1, x0, x1), field = got
    assert field[130 - y0, 132 - x0] > 2 * field[130 - y0, 167 - x0]
    shadow = L.source_visibility(lm, (90, 130), 140, (0, 300, 0, 300))
    assert shadow[130, 205] < 0.3
    assert shadow[270, 205] > 0.9


def test_source_discharge_and_reflection_share_the_observed_skin_contact():
    """The hovering root connects to the first real skin surface, rather than a separate invented palm point."""
    m = np.zeros((160, 160), np.uint8)
    m[85:, 30:130] = 255
    fields = []
    for matte in (m, m.astype(np.float32) / 255):
        p = L._source_contact((80, 30), (80, 130), matte)
        assert 84.5 <= p[1] <= 86.0 and p[0] == 80
        _, light = L.source_pool((80, 30), (80, 130), 70, 1, 160, 160, matte=matte)
        assert light.max() > 0.5
        fields.append(light)
    np.testing.assert_allclose(*fields, atol=1e-6)
    border = np.zeros_like(m)
    border[:, 0] = 255
    assert L._source_contact((-40, 40), (80, 120), border)[0] >= 0


def test_charge_halo_is_subordinate_to_the_luminous_volume():
    """The broad optical haze cannot dominate the volume with a large uniform blue disc."""
    out = np.zeros((241, 241, 3), np.float32)
    veil = np.zeros((241, 241), np.float32)
    L._sphere(out, veil, (0, 241, 0, 241), L.Ball(120, 120, 50, 1, 1, 1), seed=3)
    r = np.hypot(*(np.indices(veil.shape) - 120)) / 50
    lum = out @ LUMA
    assert lum[r > 1.4].sum() < 0.08 * lum[r < 1].sum()


def test_bulk_light_gathers_around_the_discharge_with_open_pockets_between():
    """A moving arc changes the surrounding luminous volume instead of sitting on a separate fixed blue dome."""
    box = (0, 241, 0, 241)
    ball = L.Ball(120, 120, 60, 1, 1, 1)
    bare = np.zeros((241, 241, 3), np.float32)
    shaped = np.zeros_like(bare)
    channels = np.zeros((241, 241), np.float32)
    channels[65:176, 95:98] = 1
    L._sphere(bare, np.zeros((241, 241), np.float32), box, ball, seed=3)
    L._sphere(shaped, np.zeros((241, 241), np.float32), box, ball, seed=3, channels=channels)
    ratio = (shaped @ LUMA) / np.maximum(bare @ LUMA, 1e-6)
    assert np.median(ratio[100:140, 92:103]) > 0.65
    assert np.median(ratio[100:140, 140:151]) < 0.5


def test_open_palm_arc_has_one_source_over_the_middle_of_the_palm():
    """An open palm carries a compact hovering source rather than selecting its outermost extended finger."""
    plate, track, mask = take(8, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    hand = L._hand(plate, track)
    bolts, spots, _ = L.arc_frames(hand, voice(np.full(8, 0.9)), seed=1, matte=mask)
    for t in range(8):
        assert len(spots[t]) == 1
        x, y, sigma, _ = spots[t][0]
        assert 130 < x < 155 and 250 < y < 280
        assert sigma < 4.0
        assert any(np.linalg.norm(p[0] - (x, y)) < 1e-6 for p, _, _ in bolts[t])


def test_a_palm_source_stops_when_the_observed_fingers_close():
    """Choosing the source on an open hand must not let a later closed fist continue to emit or restrike."""
    plate, track, mask = take(12, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    for row in track["landmarks"][6:]:
        for base in (5, 9, 13, 17):
            x = row[base][0]
            row[base:base + 4] = [[x, 304, 0], [x, 280, 0], [x, 292, -30], [x, 308, -70]]
    bolts, spots, impacts = L.arc_frames(L._hand(plate, track), voice(np.full(12, 0.9)), seed=1, matte=mask)
    assert all(bolts[:6])
    assert not any(bolts[6:]) and not any(spots[6:]) and not any(impacts[6:])


def test_foreground_finger_hides_the_source_core_as_well_as_the_channel():
    """A source behind a crossing finger cannot leave a white spot painted on top of the skin."""
    box = (0, 101, 0, 101)
    args = ([], [(50, 50, 3.0, 1.0)], None, 220, box, L.GLOW_CAP["arc"], 0, np.zeros(2), [])
    exposed, _ = L.layer(*args)
    cover = np.zeros((101, 101), np.float32)
    cover[:, 40:61] = 1.0
    hidden, _ = L.layer(*args, hide=cover)
    assert (hidden[50, 50] @ LUMA) < 0.15 * (exposed[50, 50] @ LUMA)


def test_contact_spill_is_concentrated_at_the_receiving_point():
    """The wall or floor carries a bright local contact, with little light a hundred pixels along it."""
    (y0, _, x0, _), light = L.pool_light(300, 200, 1, 0, 1, 1080, 400, 600)
    center = light[200 - y0, 300 - x0]
    assert light[200 - y0, 400 - x0] < 0.06 * center
    assert light[230 - y0, 300 - x0] > 0.2 * center


def test_ball_rear_filaments_have_depth_softness_before_compositing():
    """Rear channels retain their depth when projected, so the rasterizer can soften them independently."""
    plate, track, mask = take(12, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    bolts, _, _ = L.ball_frames(L._hand(plate, track), voice(np.full(12, 0.95)), seed=3, matte=mask)
    rear = [c for frame in bolts for c in frame if getattr(c, "depth", 0) > 0.5]
    assert len(rear) > 5
    assert all(c.softness > 0 for c in rear)
    assert max(c.softness for c in rear) > 2.0 * W / 1080.0


def test_front_filament_stops_at_its_depth_boundary_before_the_rear_vertex(monkeypatch):
    """The first rear vertex must not inherit the nearer run's brightness across an occluding finger."""
    segment = np.array([[-0.7, 0.0, 0.5], [0.6, 0.0, -0.6]])
    monkeypatch.setattr(L, "_shell", lambda rng: [(segment, 1.0, 1.0)])
    plate, track, _ = take(10, lambda t: (135, 300), 80, emitter="palm")
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(np.full(10, 0.95)), seed=3)
    b = balls[8]
    # The injected two-vertex path is separate from the independently tested equator and upper-limb channels.
    near = [p for c in bolts[8] for p, _, _ in [c]
            if isinstance(c, L._Filament) and c.depth < .5 and len(p) <= 3]
    assert near
    assert max(p[:, 0].max() for p in near) < b.x + 0.2 * b.R


def test_most_charge_filaments_wander_through_the_volume_instead_of_a_spherical_cage():
    """Most channels move between depths and radii; nested constant-radius arcs would recreate hollow wire loops."""
    radial_spread = [np.std(np.linalg.norm(L._shell(np.random.default_rng(seed))[0][0], axis=1))
                     for seed in range(100)]
    assert np.mean(np.asarray(radial_spread) > 0.08) > 0.45


def test_crossing_fingers_hide_the_bulk_of_the_rear_luminous_interior():
    """A finger crossing the charge reveals its silhouette, with only the thin front layer and haze before it."""
    ball = L.Ball(120, 120, 60, 1, 1, 1)
    out = np.zeros((241, 241, 3), np.float32)
    hidden = np.zeros_like(out)
    veil = np.zeros((241, 241), np.float32)
    cover = np.zeros_like(veil)
    cover[:, 110:131] = 1.0
    L._sphere(out, veil, (0, 241, 0, 241), ball, seed=3, frame=5)
    L._sphere(hidden, veil.copy(), (0, 241, 0, 241), ball, seed=3, frame=5, cover=cover)
    assert hidden[105:135, 115:126].sum() < 0.6 * out[105:135, 115:126].sum()


def test_arc_revision_a_tap_beside_a_curled_middle_still_uses_the_extended_index():
    """A point placed on the finger's side cannot select the tucked middle or carry the root beside the real tip."""
    plate, track, _ = take(8, lambda t: (135, 330))
    track["xy"] = [[135, 310]] * 8
    for row in track["landmarks"]:
        for base in (13, 17):
            x = row[base][0]
            row[base:base + 4] = [[x, 333, 0], [x, 315, 0], [x, 324, -30], [x, 336, -70]]
    hand = L._hand(plate, track)
    fingers = L._fingers(hand, 0)
    assert fingers[0] == 8
    bolts, _, _ = L.arc_frames(hand, voice(np.full(8, 0.9)), seed=1)
    assert all(mains(frame, hand.lm[0, 8], 60) for frame in bolts)
    assert not any(np.allclose(p[0], (135, 310)) for frame in bolts for p, _, _ in frame)


def test_arc_revision_a_blank_wall_receives_the_bolt_inside_the_picture():
    """A broad low-texture wall is a receiver even without a seam; the strike gets diffuse contact metadata."""
    plate, track, mask = take(12, lambda t: (170, 330), texture=4.0)
    hand = L._hand(plate, track)
    bolts, _, impacts = L.arc_frames(hand, voice(np.full(12, 0.9)), seed=2, matte=mask, plate=plate)
    assert all(impacts)
    for t, contacts in enumerate(impacts):
        for x, y, ax, ay, gain, fresh, plane in contacts:
            assert plane and 35 < x < 235 and 35 < y < 445 and mask[t, round(y), round(x)] < 64
            assert any(np.linalg.norm(p[-1] - (x, y)) < 1e-6 for p, _, _ in bolts[t])
            assert not any(np.allclose(p[0], (x, y)) for p, _, _ in bolts[t])


def test_arc_revision_a_visible_seam_wide_of_the_aim_precedes_a_flat_receiver():
    """A real seam can receive a curving discharge before a featureless wall patch is used."""
    surfaces = np.array([[50, 330, 40, 1, 0], [140, 90, -1, 1, 0]], np.float32)
    end, along, _ = L._target(np.array([180.0, 260.0]), np.array([0.0, -1.0]), 0.0, 250.0, 200.0,
                             (270, 480), surfaces, np.random.default_rng(1))
    assert np.allclose(end, (50, 330)) and len(along) == 2


def test_arc_revision_a_wide_surface_turn_keeps_only_a_short_finger_aligned_departure(monkeypatch):
    """A backward surface turn leaves along the fingertip briefly, without a large drawn hook beside it."""
    plate, track, _ = take(4, lambda t: (135, 330), texture=0.0)
    known = np.array([[35, 315, 40, 1, 0], [120, 100, -1, 1, 0]], np.float32)
    monkeypatch.setattr(L, "surfaces", lambda p, m: [known] * len(p))
    bolts, _, _ = L.arc_frames(L._hand(plate, track), voice(np.full(4, 0.9)), seed=2, plate=plate)
    channel = bolts[0][0][0]
    reach = np.linalg.norm(channel[-1] - channel[0])
    assert channel[1, 1] < channel[0, 1]
    assert channel[0, 1] - channel[:, 1].min() < 0.1 * reach


def test_tip_reflection_runs_down_the_finger_and_rolls_off_around_it():
    """A contact illuminates the curved finger sides softly and loses strength down its length and past its tip."""
    box, light = L.tip_reflection((100, 80), (0, 1), 160, 1, 300, 240)
    field = np.zeros((300, 240), np.float32)
    field[box[0]:box[1], box[2]:box[3]] = light
    assert field[100, 100] > 0.5
    assert field[100, 100] > field[100, 111]
    assert field[100, 100] > 5 * field[160, 100]
    assert field[65, 100] == 0
    assert .4 * field[100, 100] < field[100, 125] < .7 * field[100, 100]


def test_tip_contact_is_smaller_than_the_reflected_skin_light():
    """Only a tiny contact emits directly; the wider finger records reflected light instead of a white cap."""
    plate, track, mask = take(4, lambda t: (120, 330), 80)
    _, spots, _ = L.arc_frames(L._hand(plate, track), voice(np.full(4, .9)), seed=2, matte=mask, plate=plate)
    assert all(spot[2] <= 1.5 for frame in spots for spot in frame)
    box, light = L.tip_reflection((100, 80), (0, 1), 160, 1, 300, 240)
    field = np.zeros((300, 240), np.float32)
    field[box[0]:box[1], box[2]:box[3]] = light
    assert field[96, 122] > .5 * field[96, 100]
    assert field.max() < 2.5


def test_tip_reflection_wraps_the_observed_width_of_the_finger():
    """The same tracked tip on narrower skin must not carry an equally broad cylindrical light patch."""
    def reflected(radius):
        matte = np.zeros((240, 240), np.float32)
        cv2.ellipse(matte, (100, 140), (radius, 60), 0, 0, 360, 1, -1)
        box, patch = L.tip_reflection((100, 80), (0, 1), 160, 1, 240, 240, matte=matte)
        field = np.zeros_like(matte)
        field[box[0]:box[1], box[2]:box[3]] = patch
        return field

    narrow, broad = reflected(14), reflected(30)
    assert narrow[96, 116] < .65 * broad[96, 116]
    assert narrow[96, 100] > .8 * broad[96, 100]


def test_adjacent_finger_does_not_widen_the_emitting_finger_reflection():
    """Only the emitter's contiguous skin section determines its curved cap, even beside another finger."""
    matte = np.zeros((240, 240), np.float32)
    matte[80:181, 93:108] = 1
    _, alone = L.tip_reflection((100, 80), (0, 1), 160, 1, 240, 240, matte=matte)
    matte[80:181, 120:133] = 1
    _, adjacent = L.tip_reflection((100, 80), (0, 1), 160, 1, 240, 240, matte=matte)
    np.testing.assert_allclose(adjacent, alone)


def test_textured_fingertip_keeps_its_contrast_outside_a_small_clipped_contact():
    """A brighter finger retains the plate's surface variation instead of becoming a smooth pale strip."""
    plate, track, mask = take(6, lambda t: (135, 300), 100, palm=(100, 65, 60), texture=0)
    yy, xx = np.indices((H, W))
    texture = 8 * np.sin(yy * .7) + 4 * np.sin(xx * .6)
    plate = np.clip(plate.astype(float) + texture[None, ..., None] * (mask > 0)[..., None], 0, 255).astype(np.uint8)
    out = L.render(plate, track, mask, voice(np.full(6, .9)), "arc", seed=2)
    selected = (xx >= 101) & (xx <= 109) & (yy >= 228) & (yy <= 250)
    photo, lit = plate[-1].astype(float) @ LUMA, out[-1].astype(float) @ LUMA
    before = photo - cv2.GaussianBlur(photo, (0, 0), 2)
    after = lit - cv2.GaussianBlur(lit, (0, 0), 2)
    assert np.corrcoef(before[selected], after[selected])[0, 1] > .9
    assert np.std(after[selected]) > np.std(before[selected])
    assert (lit - photo)[selected].mean() > 30
    assert np.count_nonzero((out[-1].min(-1) > 235) & (mask[-1] > 0)) < 30


def test_palm_reflection_has_a_small_hotspot_without_a_wide_pale_shoulder():
    """The source-facing palm carries the highlight; adjacent skin receives little broad fill."""
    box, light = L.source_pool((100, 100), (100, 250), 200, 1, 480, 270)
    field = np.zeros((480, 270), np.float32)
    field[box[0]:box[1], box[2]:box[3]] = light
    assert field[160, 100] > .6
    assert field[160, 150] < .045 * field[160, 100]


def test_hand_reflectance_removes_background_colour_from_the_soft_skin_boundary():
    """A bright wall mixed into a blurred silhouette must not turn into an added cyan skin outline."""
    matte = np.zeros((160, 160), np.float32)
    matte[30:130, 40:120] = 1
    alb = np.tile(np.array([.8, .85, .9], np.float32), (160, 160, 1))
    alb[35:125, 45:115] = [.25, .12, .08]
    corrected = L.hand_reflectance(alb, matte, 100)
    assert corrected[80, 41, 0] > 1.4 * corrected[80, 41, 2]
    assert corrected[80, 41, 2] < .35
    np.testing.assert_allclose(corrected[80, 80], alb[80, 80], atol=1e-5)


def test_near_finger_channel_shares_focus_while_the_distant_stroke_stays_resolved():
    """A soft moving finger cannot carry a razor-sharp arc; distant electricity retains its crisp core."""
    p = np.c_[np.linspace(40, 360, 140), np.full(140, 100)]
    box = (0, 201, 0, 401)
    emission, _ = L.layer([(p, 4, 1)], [], None, 100, box, L.GLOW_CAP['arc'], .7,
                          np.zeros(2), [(40, 100)], seed=2, root_soft=4.5)
    lum = emission @ LUMA
    assert lum[97:104, 60:90].max() < .75 * lum[97:104, 280:320].max()


def test_source_core_and_bloom_share_the_finger_focus_and_shutter_motion():
    """A moving defocused fingertip cannot carry a crisp stationary source inside a softer channel."""
    box = (0, 101, 0, 161)
    spots = [(80, 50, 1.2, 1)]
    def source(soft, move):
        emission, _ = L.layer([], spots, None, 160, box, L.GLOW_CAP['arc'], .7,
                              np.asarray(move), [(80, 50)], root_soft=soft)
        return emission @ LUMA

    sharp = source(.7, (0, 0))
    focused = source(4.5, (0, 0))
    moving = source(4.5, (16, 0))
    assert focused.max() < .6 * sharp.max()
    yy, xx = np.indices(sharp.shape)
    variance = lambda light, axis: float((light * axis ** 2).sum() / light.sum())
    # The capped 12 px shutter adds about 12 square pixels of axial variance, including bloom.
    assert variance(moving, xx - 80) > variance(focused, xx - 80) + 8
    assert variance(moving, yy - 50) < 1.1 * variance(focused, yy - 50)


def test_palm_source_near_an_index_tip_still_lights_its_facing_skin(monkeypatch):
    """As the palm turns, a source projected beside a fingertip retains a soft surface return there."""
    plate, track, mask = take(4, lambda t: (135, 300), 80, emitter='palm', middle_out=True,
                              wall=(20, 20, 20), palm=(80, 60, 50), texture=0)
    monkeypatch.setattr(L, '_palm_source', lambda hand, tap: hand.lm[:, 8] - (0, 4))
    timeline = voice(np.full(4, .9))
    wrapped = L.render(plate, track, mask, timeline, 'arc', seed=1)
    monkeypatch.setattr(L, 'tip_reflection', lambda *args, **kwargs: None)
    unwrapped = L.render(plate, track, mask, timeline, 'arc', seed=1)
    assert (wrapped[-1, 248:268, 107:115, 2].astype(float) - unwrapped[-1, 248:268, 107:115, 2]).max() > 5


def test_proximal_finger_casts_a_distinct_shadow_beyond_its_root():
    """A source beside an MCP joint must be blocked even if all distal joints lie outside its light rays."""
    lm = np.tile((160.0, 70.0), (21, 1))
    lm[5:9] = [(150, 150), (150, 115), (150, 90), (150, 70)]
    shadow = L.source_visibility(lm, (80, 150), 160, (0, 300, 0, 300))
    assert shadow[150, 195] < 0.3
    assert shadow[210, 195] > 0.8


def test_finger_shadow_leaves_the_near_surface_without_a_stamped_boundary():
    """The near-surface shadow transition is continuous rather than a ring around each sampled joint."""
    lm = np.tile((150.0, 130.0), (21, 1))
    shadow = L.source_visibility(lm, (80, 130), 160, (0, 300, 0, 300))
    assert np.abs(np.diff(shadow[130, 145:195])).max() < .16
    assert shadow[130, 195] < .3


def test_palm_shadow_buffer_feathers_out_on_the_observed_wrist(monkeypatch):
    """A bounded shadow field cannot leave a horizontal exposure step across the photographed wrist."""
    plate, track, mask = take(4, lambda t: (135, 240), 80, emitter="palm", middle_out=True,
                              wall=(20, 20, 20), palm=(80, 60, 50), texture=0)
    # Strong flat shadow coverage exposes a buffer edge that normal changing finger shadows could conceal.
    monkeypatch.setattr(L, "source_visibility", lambda lm, source, scale, box:
                        np.zeros((box[1] - box[0], box[3] - box[2]), np.float32))
    monkeypatch.setattr(L.fx, "shadow_noise", lambda *args: 0.0)
    monkeypatch.setattr(L, "LIGHT", (L.LIGHT[0], 20.0, 20.0, 15.0))
    out = L.render(plate, track, mask, voice(np.full(4, .9)), "arc", seed=1)
    hand = L._hand(plate, track)
    boundary = L._around(*hand.xy[-1], 1.7 * hand.s[-1], H, W)[1]
    lift = (out[-1].astype(float) - plate[-1]) @ LUMA
    rows = lift[boundary - 3:boundary + 4, 130:141].mean(1)
    assert np.abs(np.diff(rows)).max() < 3.0


def test_a_short_jumping_arc_cannot_walk_forever_after_passing_its_endpoint():
    """This short finger connection overshot its final tangent and formerly appended points without terminating."""
    rng = np.random.default_rng()
    rng.bit_generator.state = {"bit_generator": "PCG64", "state": {
        "state": 192067346315275282861957193538235975710,
        "inc": 222608255215073204495496610942370383039},
        "has_uint32": 1, "uinteger": 3560187575}

    class LimitedWalk:
        def __init__(self):
            self.calls = 0

        def __getattr__(self, name):
            return getattr(rng, name)

        def uniform(self, *args, **kwargs):
            self.calls += 1
            assert self.calls < 1024, "a short connection kept walking beyond its endpoint"
            return rng.uniform(*args, **kwargs)

    path = L._leader(15.02311492946489, -.25516836740603377, LimitedWalk(), .25)
    assert np.isfinite(path).all() and len(path) < 256
    np.testing.assert_array_equal(path[0], (0, 0))
    np.testing.assert_array_equal(path[-1], (1, 0))


def test_ball_thumb_opening_stays_stable_when_the_thumb_crosses_the_lift_axis():
    """Subpixel thumb tracking jitter cannot switch the sphere abruptly between opposite palm openings."""
    plate, track, _ = take(16, lambda t: (135, 300), 80, emitter="palm")
    hand = L._hand(plate, track)
    hand.lm[:, 4, 0] = hand.xy[:, 0]
    hand.lm[:, 4, 1] = hand.xy[:, 1] - 40
    reference = L.ball_frames(hand, voice(np.full(16, .9)), seed=1)[2]
    hand.lm[:, 4, 0] += np.linspace(-.1, .1, 16)
    balls = L.ball_frames(hand, voice(np.full(16, .9)), seed=1)[2]
    shift = np.array([(b.x - a.x, b.y - a.y) for a, b in zip(reference[6:], balls[6:])])
    assert np.linalg.norm(np.diff(shift, axis=0), axis=1).max() < 1.0


def test_ball_has_a_dominant_continuous_front_crossing_at_its_equator():
    """Every formed charge has a connected near arc crossing its middle, rather than only random short sparks."""
    plate, track, mask = take(16, lambda t: (135, 300), 80, emitter="palm")
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(np.full(16, .9)), seed=1, matte=mask)
    for t in range(6, 16):
        ball = balls[t]
        crossing = []
        for channel in bolts[t]:
            if not isinstance(channel, L._Filament) or channel.softness > 0:
                continue
            xy = (channel[0] - (ball.x, ball.y)) / ball.R
            if xy[:, 0].min() < -.7 and xy[:, 0].max() > .7 and np.abs(xy[:, 1]).max() < .55:
                crossing.append(channel)
        assert crossing
        assert any((c[0][:, 1].max() - ball.y) / ball.R > .1 for c in crossing)


def test_ball_has_soft_rear_arcs_around_both_sides_of_the_lower_hemisphere():
    """Every formed sphere has a faint lower return behind its crisp crossing, rather than only an upper dome."""
    plate, track, _ = take(16, lambda t: (135, 300), 80, emitter="palm")
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(np.full(16, .9)), seed=1)
    for t in range(6, 16):
        ball = balls[t]
        lower = []
        for channel in bolts[t]:
            if not isinstance(channel, L._Filament) or channel.depth <= .5:
                continue
            xy = (channel[0] - (ball.x, ball.y)) / ball.R
            if xy[:, 1].max() > .65 and np.linalg.norm(xy, axis=1).mean() > .85:
                assert channel[2] < .55
                lower.extend(xy.tolist())
        assert lower
        assert np.min(np.asarray(lower)[:, 0]) < -.4
        assert np.max(np.asarray(lower)[:, 0]) > .4


def test_ball_has_a_second_near_current_through_the_upper_interior():
    """Two nonparallel near currents cross the volume; a bright lower crescent alone reads as a ring."""
    plate, track, _ = take(16, lambda t: (135, 300), 80, emitter='palm')
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(np.full(16, .9)), seed=1)
    for t in range(6, 16):
        ball = balls[t]
        near = []
        for channel in bolts[t]:
            if not isinstance(channel, L._Filament) or channel.depth != 0 or channel[2] < .5:
                continue
            xy = (channel[0] - (ball.x, ball.y)) / ball.R
            if xy[:, 1].min() < -.35 and xy[:, 1].max() > .3 and np.linalg.norm(xy, axis=1).min() < .3:
                near.append(xy)
        assert near
        assert any(p[:, 0].max() - p[:, 0].min() > .8 for p in near)


def test_ball_grounding_has_one_quieter_path_per_finger_instead_of_a_separate_loop():
    """Repeated paths to one thumb cannot enclose a second bright volume beside the charge."""
    plate, track, mask = take(35, lambda t: (135, 300), 80, emitter="palm")
    bolts, sparks, _ = L.ball_frames(L._hand(plate, track), voice(np.full(35, .95), hits=[0, 20]),
                                    seed=0, matte=mask)
    for channels, contacts in zip(bolts, sparks):
        landings = [tuple(np.round(spot[:2], 4)) for spot in contacts]
        assert len(landings) == len(set(landings))
        for channel in channels:
            if any(np.allclose(channel[0][-1], spot[:2]) for spot in contacts):
                assert channel[2] <= .55 * L.GROUND_ARC[1] + 1e-6


def test_ball_surface_currents_share_one_radius_and_bow_toward_the_camera():
    """Inclined surface currents retain small electrical kinks while their broad bends describe one sphere."""
    for angle, tilt in ((.08, .36), (-.88, -.22)):
        for seed in (1, 7, 29):
            points = L._sphere_current(angle, tilt, -1.22, 1.22, 61, seed)
            np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, atol=1e-6)
            assert (points[:, 2] > 0).all()
            xy = points[:, :2]
            axis = xy[-1] - xy[0]
            off = (xy - xy[0]) @ np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
            assert np.abs(np.median(off[22:39])) > .12
            assert np.linalg.norm(xy[[0, -1]], axis=1).min() > .9
            np.testing.assert_array_equal(points, L._sphere_current(angle, tilt, -1.22, 1.22, 61, seed))


def test_ball_dominant_front_current_bends_around_its_common_envelope():
    """At phone size the leading current has a broad spherical bow rather than an unrelated zigzag chord."""
    plate, track, mask = take(12, lambda t: (135, 300), 80, emitter="palm")
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(np.full(12, .9)), seed=1, matte=mask)
    for t in range(6, 12):
        ball = balls[t]
        current = next(c for c in bolts[t] if isinstance(c, L._Filament) and c.softness == 0)
        xy = (current[0] - (ball.x, ball.y)) / ball.R
        axis = xy[-1] - xy[0]
        off = (xy - xy[0]) @ np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
        assert np.abs(np.median(off[len(off) // 3:2 * len(off) // 3])) > .17


def test_ball_short_escaping_streamers_cast_local_light_on_adjacent_skin():
    """A few tapered escapes interrupt the limb and light nearby skin, with no added white contact beads."""
    ball = L.Ball(120, 120, 50, 1, 1, 1)
    lm = np.tile((185., 120.), (21, 1))
    paths = L._ball_streamers(np.array(ball[:2]), ball.R, lm, seed=3, frame=8)
    assert 2 <= len(paths) <= 3
    channels = [L._Filament(p, 2, .6, 0, streamer=True) for p in paths]
    for p in paths:
        radius = np.linalg.norm(p - ball[:2], axis=1) / ball.R
        assert .80 <= radius[0] <= 1.0
        assert 1.12 <= radius.max() <= 1.4
        assert np.linalg.norm(np.diff(p, axis=0), axis=1).sum() < .7 * ball.R
    matte = np.zeros((241, 241), np.float32)
    matte[95:146, 180:196] = 1
    got = L.streamer_light(matte, ball, channels)
    assert got is not None
    (y0, y1, x0, x1), field = got
    full = np.zeros_like(matte)
    full[y0:y1, x0:x1] = field
    assert full[120, 183] > .1
    assert not full[matte == 0].any()
    assert full[:80].sum() == 0
    assert L.streamer_light(matte, ball, []) is None


def test_ball_streamers_stop_before_a_finger_intercepts_a_free_escape():
    """A fallback escape must stop at a crossing finger too; aiming only its endpoint cannot protect the skin."""
    c, radius = np.array([120., 120.]), 50.
    matte = np.zeros((241, 241), np.uint8)
    angle = -.94 + .12 * np.sin(8 / 24 * 1.3 + 3)
    direction = np.array([np.cos(angle), np.sin(angle)])
    cv2.line(matte, tuple(np.rint(c + 1.06 * radius * direction).astype(int)),
             tuple(np.rint(c + 1.6 * radius * direction).astype(int)), 255, 10)
    paths = L._ball_streamers(c, radius, None, seed=3, frame=8, matte=matte)
    assert paths  # the unobstructed escapes remain visible
    for p in paths:
        dense = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) * 2) + 1))
                                for a, b in zip(p[:-1], p[1:])])
        xi, yi = np.rint(dense).astype(int).T
        assert not matte[yi, xi].any()


def test_ball_generated_streamer_cores_stay_in_air_beside_observed_fingers():
    """The production ball path must use the matte check, including when several fingers overlap the limb."""
    plate, track, mask = take(12, lambda t: (135, 300), 80, emitter="palm", middle_out=True)
    bolts, _, _ = L.ball_frames(L._hand(plate, track), voice(np.full(12, .9)), seed=1, matte=mask)
    streamers = [c[0] for frame in bolts[6:] for c in frame if getattr(c, "streamer", False)]
    assert streamers
    for p in streamers:
        xi, yi = np.rint(p).astype(int).T
        assert not mask[6, yi, xi].any()  # this fixture's observed hand is stationary


def test_ball_escaping_streamers_remain_connected_to_a_foreground_current():
    """A short escape starts on a visible current; an arbitrary bright limb root reads as a floating firefly."""
    plate, track, _ = take(12, lambda t: (135, 300), 80, emitter="palm")
    bolts, _, balls = L.ball_frames(L._hand(plate, track), voice(np.full(12, .9)), seed=1)
    for channels, ball in zip(bolts[6:], balls[6:]):
        currents = np.concatenate([c[0] for c in channels if isinstance(c, L._Filament)
                                   and c.depth == 0 and not c.streamer])
        escapes = [c[0] for c in channels if getattr(c, "streamer", False)]
        assert escapes
        for p in escapes:
            assert np.linalg.norm(currents - p[0], axis=1).min() < .025 * ball.R
