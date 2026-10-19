"""WP2: the hand + forearm matte (cuesync.hand_mask).

The prompt schedule, the clean-up and the npz contract run anywhere (numpy + OpenCV). The SAM 2.1 tests need the
matte's own environment (torch with CUDA, sam2: runs/bending/venv-mask) and the dev plates; elsewhere they skip:

    PYTHONPATH=cuesync/src runs/bending/venv-mask/Scripts/python.exe -m pytest cuesync/tests/bending/test_hand_mask.py

The GT mattes are side agent A's polygons (gt/<clip>.json "masks") when that file has them, else the provisional
ones in runs/bending/mask_lab/gt (see its README).
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from cuesync import hand_mask as hm

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import test_metrics as tm  # the measures every bending package reports against (side agent A's)
except ImportError:
    tm = None

REPO = Path(__file__).resolve().parents[3]
PLATES = REPO / "runs" / "bending" / "plates"
LAB = REPO / "runs" / "bending" / "mask_lab"
CLIPS = ["C_palm_push", "A_point_left", "two_palms", "cupped_palm"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if tm is not None:
        return tm.mask_iou(a, b)
    a, b = a >= 128, b >= 128  # the same measure, should test_metrics not be there
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


# An open right hand, fingers up, in hand-scales from the wrist (MediaPipe ids: 0 wrist, 1-4 thumb, 5-8 index,
# 9-12 middle, 13-16 ring, 17-20 pinky); the middle knuckle is one scale above the wrist, so scale = |lm0 - lm9|.
OPEN_HAND = np.array([
    (0, 0), (-0.35, -0.25), (-0.6, -0.5), (-0.8, -0.75), (-0.95, -1.0),
    (-0.3, -0.95), (-0.35, -1.35), (-0.38, -1.65), (-0.4, -1.9),
    (0, -1.0), (0, -1.45), (0, -1.75), (0, -2.0),
    (0.25, -0.95), (0.28, -1.35), (0.3, -1.62), (0.32, -1.85),
    (0.48, -0.85), (0.55, -1.15), (0.6, -1.38), (0.65, -1.58)])


def hand(cx: float, cy: float, s: float = 40.0) -> list:
    """The open hand's 21 landmarks with the wrist at (cx, cy), ``s`` px to the scale."""
    return [[float(cx + x * s), float(cy + y * s), 0.0] for x, y in OPEN_HAND]


def track(T: int, W: int, H: int, centre, src=None, emitter="two_finger", s=40.0) -> dict:
    lms = [hand(*centre(t), s=s) for t in range(T)]
    src = src or ["mp"] * T
    lms = [None if src[t] in ("flow", "interp", "lost") else lms[t] for t in range(T)]
    return {"version": 1, "fps": 24, "W": W, "H": H, "emitter": emitter, "hand": "right",
            "xy": [list(centre(t)) for t in range(T)], "dir": [[0.0, -1.0]] * T, "scale": [s] * T,
            "conf": [0.0 if x == "lost" else 0.9 for x in src], "src": src, "speed": [0.0] * T, "landmarks": lms,
            "gesture": ["None"] * T, "events": []}


# ------------------------------------------------------------------------------------- prompts (anywhere)


def test_prompts_every_16_frames_and_on_each_source_change():
    src = ["mp"] * 60
    src[20:24] = ["flow"] * 4  # the tracker bridged a blur: re-prompt on the last good frame and the re-acquire
    tr = track(60, 320, 240, lambda t: (160, 150), src=src)
    frames = hm.prompt_frames(tr)
    assert {0, 16, 32, 48, 19, 24} <= set(frames)
    assert not set(frames) & set(range(20, 24))  # never prompted from frames without landmarks
    assert len(frames) == len(set(frames))


def test_prompts_are_on_the_hand_and_negatives_off_it():
    tr = track(1, 640, 480, lambda t: (320, 300))
    (p,) = hm.prompts(tr)
    lm = np.asarray(tr["landmarks"][0])[:, :2]
    pos, neg = p.points[p.labels == 1], p.points[p.labels == 0]
    assert len(pos) == 6  # wrist, index, middle and pinky knuckles, index and middle tips (two_finger)
    assert np.allclose(pos, lm[[0, 5, 9, 17, 8, 12]])
    assert 1 <= len(neg) <= 2
    c = lm[[0, 5, 9, 13, 17]].mean(0)
    for q in neg:  # clear of the hand, and never down the forearm (straight down from the wrist here)
        assert np.min(np.linalg.norm(lm - q, axis=1)) >= 1.2 * 40
        u = (q - c) / np.linalg.norm(q - c)
        assert math.degrees(math.acos(float(u @ [0.0, 1.0]))) > 60
    x0, y0, x1, y1 = p.box
    assert x0 <= lm[:, 0].min() and x1 >= lm[:, 0].max() and y0 <= lm[:, 1].min() and y1 >= lm[:, 1].max()
    assert y1 >= lm[0, 1] + 1.5 * 40  # a bare forearm: room for it past the wrist
    (cuff,) = hm.prompts(tr, reach=0.0)  # a sleeved arm: the box ends at the wrist, keeping SAM at the cuff
    assert cuff.box[3] <= lm[0, 1] + 0.5 * 40


def test_points_off_the_picture_are_dropped():
    tr = track(1, 320, 240, lambda t: (160, 250))  # a hand coming in at the bottom edge: the wrist is off frame
    (p,) = hm.prompts(tr)
    assert np.all((p.points[:, 0] >= 0) & (p.points[:, 0] < 320) & (p.points[:, 1] >= 0) & (p.points[:, 1] < 240))
    gone = track(1, 320, 240, lambda t: (160, 420))  # all of it off frame: no prompt at all
    assert hm.prompts(gone) == []


def test_track_is_fitted_to_the_frames():
    tr = track(10, 960, 540, lambda t: (480, 300))
    fit = hm._fit(tr, 12, 1920, 1080)  # measured at half size, two frames short
    assert len(fit["src"]) == 12 and fit["src"][-1] == "lost" and fit["landmarks"][-1] is None
    assert fit["landmarks"][0][0][:2] == [960.0, 600.0]
    assert fit["scale"][0] == pytest.approx(80.0)


# ------------------------------------------------------------------------------------- clean-up (anywhere)


def disc(T, h, w, cx, cy, r):
    yy, xx = np.mgrid[:h, :w]
    return np.stack([(xx - cx(t)) ** 2 + (yy - cy) ** 2 <= r * r for t in range(T)])


def test_temporal_median_drops_flicker_and_keeps_a_fast_hand():
    T, h, w = 9, 120, 200
    truth = disc(T, h, w, lambda t: 40 + 14 * t, 60, 18)  # 14 px a frame: a fast hand at this size
    m = truth.copy()
    m[4] = False  # SAM dropped one frame
    m[6, 5:15, 5:15] = True  # and flickered a speck on another
    out = hm.temporal_median(m)
    assert iou(out[4].astype(np.uint8) * 255, truth[4].astype(np.uint8) * 255) > 0.95
    assert not out[6, 5:15, 5:15].any()
    for t in (2, 3, 5):  # the moving edges aren't eroded by the vote
        assert iou(out[t].astype(np.uint8) * 255, truth[t].astype(np.uint8) * 255) > 0.97


def test_holes_inside_a_hand_are_filled_but_a_real_gap_stays():
    m = np.zeros((100, 100), bool)
    m[10:90, 10:90] = True
    m[30:33, 30:33] = False  # a speck SAM let go
    m[50:80, 50:80] = False  # a big enclosed gap (thumb touching a finger)
    out = hm.fill_holes(m)
    assert out[31, 31] and not out[60, 60]
    a = np.zeros((100, 100), np.uint8)
    a[out] = 255
    assert hm.holes(a) == pytest.approx(900 / out.sum(), rel=1e-6)


def test_finish_feathers_2px_and_blurs_only_fast_frames_along_the_motion():
    T, h, w = 7, 108, 192  # work size; output is 2x
    W, H = 384, 216
    yy, xx = np.mgrid[:H, :W]
    plate = np.zeros((T, H, W, 3), np.uint8)
    masks = np.zeros((T, 1, h, w), bool)
    for t in range(T):
        x0 = 30 + 6 * t  # 6 work px a frame = 0.055 of the short side: fast
        masks[t, 0, 40:70, x0:x0 + 30] = True
        plate[t, 80:140, 2 * x0:2 * x0 + 60] = 200
        plate[t] = cv2.GaussianBlur(plate[t], (0, 0), 1.0) + ((xx * 7 + yy * 3) % 23).astype(np.uint8)[..., None]
    still = hm.finish(np.repeat(masks[:1], T, axis=0), None, (W, H))
    row = still[3, 110, :].astype(int)
    ramp = np.flatnonzero((row > 20) & (row < 235))
    assert 2 <= len(ramp) <= 10  # a soft edge of a few px on each side, not a hard one or a wide blur
    moving = hm.finish(masks, plate, (W, H))
    r = moving[3, 110, :].astype(int)
    c = moving[3, :, int(np.argmax(r))].astype(int)
    soft_x = np.count_nonzero((r > 20) & (r < 235))
    soft_y = np.count_nonzero((c > 20) & (c < 235))
    assert soft_x > 2 * soft_y  # smeared along x (the motion), not along y


def test_npz_follows_the_contract(tmp_path):
    alpha = np.zeros((3, 8, 10), np.uint8)
    alpha[1, 2:5, 3:7] = 255
    path = hm.save(tmp_path / "run" / "hand_mask.npz", alpha, ["sam2", "selfie", "hull"])
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {"alpha", "src", "fps", "version"}
        assert z["alpha"].dtype == np.uint8 and z["alpha"].shape == (3, 8, 10)
        assert [str(s) for s in z["src"]] == ["sam2", "selfie", "hull"]
        assert int(z["fps"]) == 24 and int(z["version"]) == 1
    a, src = hm.load(path)
    assert np.array_equal(a, alpha) and src == ["sam2", "selfie", "hull"]
    streamed = hm.save(tmp_path / "streamed.npz", (f for f in alpha), ["sam2", "selfie", "hull"])  # frame by frame
    b, src = hm.load(streamed)
    assert np.array_equal(b, alpha) and src == ["sam2", "selfie", "hull"]
    with pytest.raises(ValueError):
        hm.save(tmp_path / "short.npz", (f for f in alpha[:2]), ["sam2", "selfie", "hull"])


def test_packed_frames_hold_the_plate(tmp_path, make_video):
    from cuesync.media import read_video

    video = make_video(tmp_path, 6, 24, 160, 96, lambda i: (20 + 30 * i, 120, 200 - 20 * i))
    packed = hm.PackedFrames(video)
    plain = read_video(video).frames
    assert packed.shape == plain.shape and len(packed) == len(plain)
    for t in range(len(plain)):  # JPEG at q95: within a level or two of the decoded plate
        assert np.abs(packed[t].astype(int) - plain[t]).mean() < 1.5


def test_cli_writes_the_npz_with_the_hull_fallback(tmp_path, make_video):
    T, W, H = 12, 320, 240
    video = make_video(tmp_path, T, 24, W, H, lambda i: (30, 30, 30))
    tp = tmp_path / "hand_track.json"
    tp.write_text(json.dumps(track(T, W, H, lambda t: (160 + 2 * t, 160))), encoding="utf-8")
    out = tmp_path / "hand_mask.npz"
    assert hm.main([str(video), "--track", str(tp), "--out", str(out), "--method", "hull"]) == 0
    alpha, src = hm.load(out)
    assert alpha.shape == (T, H, W) and src == ["hull"] * T
    lm = np.asarray(track(T, W, H, lambda t: (160 + 2 * t, 160))["landmarks"][5])[:, :2]
    assert all(alpha[5, int(y), int(x)] > 128 for x, y in lm)  # the hull covers every landmark
    assert alpha[5, 10, 10] == 0


# ------------------------------------------------------------------------------- SAM 2.1 on the dev plates


def _tracks_for(clip):
    """Side agent A's real track when it exists, else the MediaPipe stub tracks in mask_lab (both hands)."""
    real = sorted((REPO / "runs" / "bending" / "track_lab" / "out").glob(f"{clip}.hand_track.json"))
    stub = sorted((LAB / "stub").glob(f"{clip}_*.json"))
    paths = real or stub
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _gt_for(clip):
    """{frame: matte} and the blurriest frames: side agent A's mask polygons when gt/<clip>.json has them, else
    mask_lab's provisional mattes (runs/bending/mask_lab/gt/README.md)."""
    try:
        labels = tm.load_gt(clip) if tm is not None else {}
    except FileNotFoundError:
        labels = {}
    if labels.get("masks"):
        H, W = int(labels["H"]), int(labels["W"])
        return {t: tm.polygon_mask(p, H, W) for t, p in labels["masks"].items()}, labels.get("blurry", [])
    d = LAB / "gt" / clip
    meta = json.loads((d / "frames.json").read_text())
    gt = {t: cv2.imread(str(d / f"{t:04d}.png"), cv2.IMREAD_GRAYSCALE) for t in meta["frames"]}
    return {t: g for t, g in gt.items() if g is not None}, meta["blurry"]


def _read(clip):
    import av

    with av.open(str(PLATES / f"{clip}.mp4")) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


needs_sam = pytest.mark.skipif(not hm.sam2_available(), reason="SAM 2.1 needs torch + CUDA + sam2 (venv-mask)")


def _has_mediapipe() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _has_mediapipe(), reason="needs mediapipe (venv-mask)")
def test_the_other_palm_is_found_when_one_is_tracked():
    clip = "two_palms"
    real = REPO / "runs" / "bending" / "track_lab" / "out" / f"{clip}.hand_track.json"
    if not (PLATES / f"{clip}.mp4").exists() or not real.exists():
        pytest.skip("dev plate or A's track not here")
    frames = _read(clip)
    tr = hm._fit(json.loads(real.read_text(encoding="utf-8")), len(frames), frames.shape[2], frames.shape[1])
    found = hm.other_hands(frames, [tr])
    assert len(found) == 1  # both palms are up from about 1.5 s to 5.5 s; the track follows one of them
    seen = [t for t, lm in enumerate(found[0]["landmarks"]) if lm is not None]
    assert seen[0] <= 48 and seen[-1] >= 120
    for t in seen:  # never the tracked palm
        if tr["landmarks"][t] is not None:
            d = np.linalg.norm(hm._palm(np.asarray(found[0]["landmarks"][t])[:, :2]) - hm._palm(hm._lm(tr, t)))
            assert d > 1.5 * tr["scale"][t]  # the distance other_hands tells two hands apart by


@needs_sam
@pytest.mark.parametrize("clip", CLIPS)
def test_sam2_matte_matches_the_gt(clip):
    if not (PLATES / f"{clip}.mp4").exists():
        pytest.skip("dev plate not here")
    try:
        gt, blurry = _gt_for(clip)
    except FileNotFoundError:
        pytest.skip("no GT mattes for this clip")
    tracks = _tracks_for(clip)
    if not tracks:
        pytest.skip("no hand track for this clip")
    frames = _read(clip)
    alpha, src = hm.hand_matte(frames, tracks)
    del frames
    scores = {t: iou(alpha[t], g) for t, g in gt.items()}
    assert np.mean(list(scores.values())) >= 0.85, scores
    assert np.mean([scores[t] for t in blurry]) >= 0.75, scores
    assert max(hm.holes(a) for a in alpha) <= 0.05
    assert set(src) <= {"sam2", "selfie", "hull"} and len(src) == len(alpha)


def test_the_drawn_hand_leaves_the_air_between_spread_fingers_out():
    """Without SAM, a hand the skin segmenter can't find is drawn from its landmarks: palm, fingers a finger wide,
    the wrist's stub of forearm; not their hull, which filled the air between and above the fingers."""
    lm = np.zeros((21, 2))
    lm[0] = (100, 180)  # wrist
    for k, (x0, dx) in enumerate(((60, -30), (80, -12), (100, 0), (120, 12), (140, 30))):  # five fingers fanned out
        for j in range(4):
            lm[1 + 4 * k + j] = (x0 + dx * j / 3 + (dx * 1.5 if j else 0) * 0, 130 - 25 * j)
    s = float(np.linalg.norm(lm[0] - lm[9]))
    m = hm.bones_mask(lm, s, (220, 200))
    assert all(m[int(y), int(x)] for x, y in lm)  # every joint is in it
    gap = (int((lm[8, 0] + lm[12, 0]) / 2), int(lm[8, 1]))  # between the index and middle fingertips
    assert not m[gap[1], gap[0]] and hm.hull_mask(lm, s, (220, 200))[gap[1], gap[0]]
