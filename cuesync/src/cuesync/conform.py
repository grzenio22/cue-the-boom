"""Conform phone footage: constant 24 fps by timestamp (VFR-safe), centre crop to the target aspect, Lanczos resize."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .media import VideoClip


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize (N,H,W,3) uint8 frames to ``size`` = (W, H) with Lanczos."""
    w, h = size
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.shape[1] == h and frames.shape[2] == w:
        return frames.copy()
    out = np.empty((frames.shape[0], h, w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.LANCZOS))
    return out


def _crop_box(src_w: int, src_h: int, dst_w: int, dst_h: int,
              focus: tuple[float, float] | None = None) -> tuple[int, int, int, int]:
    """Crop box (left, top, right, bottom) matching the destination aspect: centred, but slid (never past the
    frame) so that ``focus`` (normalised x, y in the source) keeps at least a fifth of the crop around it."""
    target = dst_w / dst_h

    def place(length: int, crop: int, f: float | None) -> int:
        start = (length - crop) // 2
        if f is not None:
            p, m = f * length, 0.2 * crop
            start = min(max(start, int(p + m - crop)), int(p - m))
        return min(max(start, 0), length - crop)

    if src_w / src_h > target:  # too wide: trim the sides
        cw = max(1, round(src_h * target))
        left = place(src_w, cw, None if focus is None else focus[0])
        return left, 0, left + cw, src_h
    ch = max(1, round(src_w / target))  # too tall: trim top and bottom
    top = place(src_h, ch, None if focus is None else focus[1])
    return 0, top, src_w, top + ch


def crop_point(point: tuple[float, float], src_w: int, src_h: int, box: tuple[int, int, int, int]) -> tuple[float, float]:
    """A normalised point in the source, in the cropped frame's normalised coordinates."""
    left, top, right, bottom = box
    return ((point[0] * src_w - left) / (right - left), (point[1] * src_h - top) / (bottom - top))


def clip_duration(clip: VideoClip, fps: int = 24) -> float:
    """Duration covered by the clip's frames: the last timestamp plus one frame."""
    times = np.asarray(clip.times_s, dtype=np.float64)
    frame_dur = float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / (clip.fps_nominal or fps)
    return float(times[-1] + frame_dur)


def conform(
    clip: VideoClip,
    size: tuple[int, int] = (1080, 1920),
    fps: int = 24,
    start_s: float = 0.0,
    duration_s: float | None = None,
    focus: tuple[float, float] | None = None,
) -> np.ndarray:
    """Resample ``clip`` to constant ``fps`` (sample-and-hold by timestamp), crop to ``size``'s aspect and resize.

    Output frame k shows the latest source frame whose timestamp is <= start_s + k/fps. ``focus``: a
    normalised source point the crop must keep in view (see _crop_box).
    """
    times = np.asarray(clip.times_s, dtype=np.float64)
    if duration_s is None:
        if len(times) > 1:
            frame_dur = float(np.median(np.diff(times)))
        else:
            frame_dur = 1.0 / (clip.fps_nominal or fps)
        duration_s = times[-1] + frame_dur - start_s
    n_out = int(round(duration_s * fps))
    if n_out <= 0:
        raise ValueError(f"nothing to conform: duration {duration_s:.3f}s at {fps} fps")
    src_h, src_w = clip.frames.shape[1:3]
    box = _crop_box(src_w, src_h, size[0], size[1], focus)
    out = np.empty((n_out, size[1], size[0], 3), dtype=np.uint8)
    last_idx = -1  # sample-and-hold only ever repeats the previous source frame: no cache needed
    for k in range(n_out):
        t = start_s + k / fps
        idx = int(np.searchsorted(times, t + 1e-9, side="right")) - 1
        idx = min(max(idx, 0), len(times) - 1)
        if idx == last_idx:
            out[k] = out[k - 1]
            continue
        img = Image.fromarray(clip.frames[idx]).crop(box)
        if img.size != size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        out[k] = np.asarray(img)
        last_idx = idx
    return out
