"""Plates of any length, orientation and resolution: never padded, cropped around the tapped point, bounded."""

import numpy as np
import pytest

from cuesync.conform import _crop_box, crop_point
from cuesync.effects import load_effect
from cuesync.guide import paint_guide
from cuesync.audio import CueSheet
from cuesync.media import probe_video, read_video, write_video
from cuesync.pipeline import load_plate


def marker_video(path, n, w, h, fps=24, at=(0.5, 0.5)):
    """Grey frames with a white square at ``at`` (normalised), to find where a point lands after cropping."""
    frames = np.full((n, h, w, 3), 60, np.uint8)
    cx, cy, r = int(at[0] * w), int(at[1] * h), max(4, min(w, h) // 40)
    frames[:, cy - r: cy + r, cx - r: cx + r] = 255
    write_video(path, frames, fps=fps)
    return path


def test_probe_reports_the_displayed_size_and_duration(tmp_path):
    info = probe_video(marker_video(tmp_path / "p.mp4", 48, 320, 240))
    assert (info.width, info.height) == (320, 240)
    assert info.duration_s == pytest.approx(2.0, abs=0.1)


def test_read_video_stops_at_max_s(tmp_path):
    clip = read_video(marker_video(tmp_path / "long.mp4", 240, 64, 64), max_s=2.0)
    assert 48 <= len(clip.frames) <= 64  # 2 s plus the out-of-order margin, not all 10 s


def test_a_short_plate_is_not_padded_to_the_maximum(tmp_path):
    frames, _, _ = load_plate(marker_video(tmp_path / "short.mp4", 72, 216, 384), (0.5, 0.5), 10.0)
    assert len(frames) == 72  # 3 s, not 10 s of a frozen last frame


def test_a_long_plate_is_trimmed_to_the_maximum(tmp_path):
    frames, _, _ = load_plate(marker_video(tmp_path / "long.mp4", 360, 108, 192), (0.5, 0.5), 10.0)
    assert len(frames) == 240


def test_landscape_plates_stay_landscape(tmp_path):
    frames, point, element = load_plate(marker_video(tmp_path / "land.mp4", 24, 384, 216, at=(0.3, 0.6)), (0.3, 0.6), 10.0)
    assert frames.shape[1:3] == (1080, 1920) and element == (1024, 576)
    assert point == pytest.approx((0.3, 0.6), abs=0.01)


def test_the_tapped_point_survives_a_crop(tmp_path):
    """A 3:4 clip tapped near its right edge: the 9:16 crop slides to keep a fifth of it around the point (a
    centre crop would have put it at 0.9), and the mapped point lands on the marker."""
    src = marker_video(tmp_path / "34.mp4", 24, 240, 320, at=(0.8, 0.5))
    frames, point, _ = load_plate(src, (0.8, 0.5), 10.0)
    assert point[0] == pytest.approx(0.8, abs=0.01)
    f = frames[0].astype(int).sum(-1)
    ys, xs = np.nonzero(f > 600)  # the white marker
    assert xs.mean() / frames.shape[2] == pytest.approx(point[0], abs=0.03)
    assert ys.mean() / frames.shape[1] == pytest.approx(point[1], abs=0.03)


def test_a_huge_plate_is_decoded_small(tmp_path, monkeypatch):
    import cuesync.pipeline as pipeline
    seen = {}
    real = pipeline.read_video

    def spy(path, size=None, max_s=None):
        seen["size"] = size
        return real(path, size=size, max_s=max_s)

    monkeypatch.setattr(pipeline, "read_video", spy)
    load_plate(marker_video(tmp_path / "big.mp4", 12, 2160, 3840), (0.5, 0.5), 10.0)
    assert seen["size"] is not None and seen["size"][1] <= 1920


def test_crop_box_without_focus_is_centred_and_focus_never_leaves_the_frame():
    assert _crop_box(400, 300, 9, 16) == (115, 0, 284, 300)
    left, top, right, bottom = _crop_box(400, 300, 9, 16, focus=(0.99, 0.5))
    assert right == 400 and right - left == 169  # slid to the edge, no further
    x, y = crop_point((0.99, 0.5), 400, 300, (left, top, right, bottom))
    assert 0.0 < x < 1.0 and y == pytest.approx(0.5)


def test_guide_blobs_keep_their_size_in_landscape():
    e = np.array([0.0] * 12 + [1.0] * 6 + [0.3] * 30)
    cue = CueSheet(24, 0.0, 2.0, [0.5], 0.5, e, e, 13, active_intervals=[(12, 18)])
    eff = load_effect("fireball")
    por = paint_guide(cue, eff, (0.5, 0.7), size=(576, 1024))
    land = paint_guide(cue, eff, (0.5, 0.7), size=(1024, 576))
    r = lambda plan: max(sh["rx"] for b in plan.blobs for sh in b["shapes"])
    assert r(land) == pytest.approx(r(por), rel=0.05)
