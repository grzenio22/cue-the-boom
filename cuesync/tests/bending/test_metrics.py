"""The measures every element-bending work package reports against (brief WP0), and their own checks.

- Tracking: how far the track's emitter is from the labelled one, as a share of the frame's short side (median and
  95th percentile); how much it shakes while the hand holds still; how many frames it lost while the hand was there.
- Masks: intersection over union of a matte and a labelled polygon (``mask_iou``, ``polygon_mask``; WP2 imports them).
- Timing: how far a detected onset is from the labelled one, in frames and milliseconds.

Ground truth lives in ``gt/<clip>.json`` (``load_gt``), clicked with ``tools/label_points.py``; a file marked
``"provisional": true`` was labelled by eye from coordinate-gridded crops, not clicked, until Grzegorz clicks it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cuesync.hand_track import PUSH  # pure numpy at import: fine without MediaPipe

GT_DIR = Path(__file__).resolve().parent / "gt"


# ------------------------------------------------------------------ ground truth

def load_gt(clip: str) -> dict:
    """``gt/<clip>.json``: {"clip", "video", "W", "H", "fps", "emitter", "hand", "seed", "provisional", "points":
    {frame: [x, y] | null}, "held": [[a, b], ...], "in_frame": [[a, b], ...], "events": [{"frame", "kind"}],
    "masks": {frame: [[x, y], ...]}, "blurry": [frame, ...]}. Ranges are inclusive frame numbers."""
    gt = json.loads((GT_DIR / f"{clip}.json").read_text(encoding="utf-8"))
    gt["points"] = {int(k): v for k, v in gt.get("points", {}).items()}
    gt["masks"] = {int(k): v for k, v in gt.get("masks", {}).items()}
    return gt


def frames_in(ranges) -> list[int]:
    return sorted({t for a, b in ranges for t in range(int(a), int(b) + 1)})


# ------------------------------------------------------------------ tracking

def track_point(track: dict, t: int) -> np.ndarray:
    """The point on the hand a labeller clicks for frame ``t`` of a hand_track.json: the emitter itself, except for
    the two-finger and fist emitters, which sit PUSH hand sizes past the fingertips / knuckles a labeller can see."""
    xy = np.asarray(track["xy"][t], np.float64)
    if track["emitter"] in ("two_finger", "fist"):
        xy = xy - PUSH * float(track["scale"][t]) * np.asarray(track["dir"][t], np.float64)
    return xy


def landmark_point(track: dict, t: int, k: int) -> np.ndarray:
    """Landmark ``k`` of frame ``t``; on a frame with no hand, the nearest one before it (as the track holds its
    emitter there), else after."""
    lms = track["landmarks"]
    for f in [*range(t, -1, -1), *range(t + 1, len(lms))]:
        if lms[f] is not None:
            return np.asarray(lms[f][k][:2], np.float64)
    return np.full(2, np.nan)


def emitter_errors(track: dict, gt: dict) -> dict[int, float]:
    """px from the labelled point, per labelled frame (frames labelled ``null``, the point not visible, are skipped).
    The labelled point is the emitter (``track_point``), or MediaPipe landmark ``gt["landmark"]`` when the file names
    one: provisional labels take the one point on the hand that can be seen unambiguously, which for a palm emitter
    (an average of five joints inside the hand) is a fingertip."""
    k = gt.get("landmark")
    at = (lambda t: track_point(track, t)) if k is None else (lambda t: landmark_point(track, t, int(k)))
    return {t: float(np.linalg.norm(at(t) - np.asarray(p, np.float64)))
            for t, p in sorted(gt["points"].items()) if p is not None and t < len(track["xy"])}


def error_stats(errors, short_side: float) -> dict:
    """Median and 95th percentile of px errors, in px and as a share of the short side."""
    e = np.asarray(list(errors.values()) if isinstance(errors, dict) else errors, np.float64)
    if e.size == 0:
        return {"n": 0, "median_px": np.nan, "p95_px": np.nan, "median": np.nan, "p95": np.nan}
    med, p95 = float(np.median(e)), float(np.percentile(e, 95))
    return {"n": int(e.size), "median_px": med, "p95_px": p95, "median": med / short_side, "p95": p95 / short_side}


def jitter_rms(track: dict, ranges, detrend: bool = False) -> float:
    """RMS frame-to-frame movement of the emitter (px) over the held stretches: what the brief calls held-pose jitter.
    Only steps between two tracked frames count. A held hand still drifts (and a camera pans), and that counts here
    too; ``detrend`` first takes out each stretch's own slow motion (a quadratic through each second of it), leaving
    the shake."""
    xy = np.asarray(track["xy"], np.float64)
    src = track["src"]
    steps = []
    for a, b in ranges:
        a, b = int(a), min(int(b), len(xy) - 1)
        seg = xy[a:b + 1].copy()
        if detrend and len(seg) > 3:
            from cuesync.hand_track import savgol

            seg -= savgol(seg, 25, 2)
        steps += [np.linalg.norm(seg[t + 1 - a] - seg[t - a]) for t in range(a, b)
                  if src[t] not in ("lost", "hold") and src[t + 1] not in ("lost", "hold")]
    return float(np.sqrt(np.mean(np.square(steps)))) if steps else np.nan


def lost_frames(track: dict, ranges) -> list[int]:
    """Frames in ``ranges`` the track has no hand for (``lost`` or ``hold``)."""
    return [t for t in frames_in(ranges) if t < len(track["src"]) and track["src"][t] in ("lost", "hold")]


# ------------------------------------------------------------------ masks

def polygon_mask(poly, H: int, W: int) -> np.ndarray:
    """A labelled polygon ([[x, y], ...] px) as a bool mask [H, W]."""
    import cv2

    m = np.zeros((H, W), np.uint8)
    pts = np.round(np.asarray(poly, np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def mask_iou(pred, gt) -> float:
    """Intersection over union of two masks the same size. Bool masks as they are; uint8 alphas (0..255) cut at 128;
    float alphas (0..1) at 0.5. Two empty masks agree perfectly (1.0)."""
    def binar(m):
        m = np.asarray(m)
        if m.dtype == bool:
            return m
        return m >= (128 if m.dtype == np.uint8 else 0.5)

    a, b = binar(pred), binar(gt)
    if a.shape != b.shape:
        raise ValueError(f"mask sizes differ: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


# ------------------------------------------------------------------ timing

def onset_offsets(detected, labelled, fps: float) -> list[dict]:
    """For each labelled onset frame, the nearest detected one: {"labelled", "detected", "frames", "ms"} with the
    offset detected - labelled (positive = late); ``detected`` None (and offsets None) when nothing was detected."""
    det = sorted(int(f) for f in detected)
    out = []
    for f in labelled:
        d = min(det, key=lambda x: abs(x - f)) if det else None
        off = None if d is None else d - int(f)
        out.append({"labelled": int(f), "detected": d, "frames": off,
                    "ms": None if off is None else 1000.0 * off / fps})
    return out


# ------------------------------------------------------------------ the measures' own checks

def _track(xy, emitter="index_tip", scale=100.0, d=(1.0, 0.0), src=None):
    T = len(xy)
    return {"xy": [list(map(float, p)) for p in xy], "dir": [list(d)] * T, "scale": [scale] * T, "emitter": emitter,
            "src": src or ["mp"] * T}


def test_emitter_errors_compare_the_clicked_point():
    gt = {"points": {0: [10.0, 0.0], 2: None, 3: [0.0, 0.0]}}
    tr = _track([[13, 4], [0, 0], [0, 0], [0, 0]])
    assert emitter_errors(tr, gt) == {0: 5.0, 3: 0.0}
    # a two-finger emitter sits 0.15 x the hand's size past the fingertips that were clicked
    tr2 = _track([[25, 0]] * 4, emitter="two_finger", scale=100.0)
    assert emitter_errors(tr2, {"points": {1: [10.0, 0.0]}}) == {1: 0.0}
    # a named landmark; a frame with no hand uses the one before
    tr3 = _track([[0, 0]] * 3)
    tr3["landmarks"] = [[[float(i), 0.0, 0.0] for i in range(21)], None, None]
    assert emitter_errors(tr3, {"landmark": 4, "points": {2: [4.0, 3.0]}}) == {2: 3.0}


def test_error_stats_are_shares_of_the_short_side():
    s = error_stats({i: float(i) for i in range(101)}, short_side=1000.0)
    assert s["median_px"] == 50.0 and s["median"] == 0.05
    assert abs(s["p95_px"] - 95.0) < 1e-9
    assert error_stats([], 1000.0)["n"] == 0


def test_jitter_counts_only_steps_inside_held_stretches_between_tracked_frames():
    xy = [[0, 0], [3, 4], [3, 4], [100, 100], [100, 100]]
    tr = _track(xy, src=["mp", "mp", "mp", "lost", "mp"])
    assert jitter_rms(tr, [[0, 2]]) == float(np.sqrt((25 + 0) / 2))
    assert jitter_rms(tr, [[2, 4]]) != jitter_rms(tr, [[2, 4]])  # no tracked pair: nan
    assert lost_frames(tr, [[0, 4]]) == [3]
    # a steady drift is movement, not shake: detrending takes it out
    drift = _track([[2.0 * t, 0.0] for t in range(40)])
    assert abs(jitter_rms(drift, [[0, 39]]) - 2.0) < 1e-9 and jitter_rms(drift, [[0, 39]], detrend=True) < 1e-6


def test_mask_iou_and_polygons():
    a = polygon_mask([[0, 0], [9, 0], [9, 9], [0, 9]], 20, 20)
    assert a.sum() == 100
    b = np.zeros((20, 20), np.uint8)
    b[0:10, 5:15] = 255
    assert abs(mask_iou(a, b) - 50 / 150) < 1e-9
    assert mask_iou(a.astype(np.float32), a) == 1.0
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) == 1.0


def test_onset_offsets_in_frames_and_ms():
    out = onset_offsets([10, 31], [11, 30, 60], fps=24.0)
    assert [o["frames"] for o in out] == [-1, 1, -29]
    assert abs(out[1]["ms"] - 1000 / 24) < 1e-9
    assert onset_offsets([], [5], 24.0)[0]["detected"] is None
