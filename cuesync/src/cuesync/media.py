"""Video and audio I/O on top of PyAV (FFmpeg)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import ColorRange, Colorspace


@dataclass
class VideoClip:
    """Decoded video. ``times_s`` are presentation timestamps relative to the first frame."""

    frames: np.ndarray  # (N, H, W, 3) uint8
    times_s: np.ndarray  # (N,) float64, ascending, relative to the source's first frame
    fps_nominal: float
    start_s: float = 0.0  # absolute container time of the first frame
    duration_s: float | None = None  # source duration, retained when frames are sampled during decode


_PQ, _HLG = 16, 18  # transfer characteristics: SMPTE ST 2084, ARIB STD-B67
_BT2020_SPACES = {9, 10}  # BT.2020 non-constant / constant luminance


class HDRVideoError(ValueError):
    """Kept for compatibility: HDR plates are now tone-mapped to SDR instead of rejected."""


def _hdr_kind(frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> str | None:
    """'hlg' or 'pq' for HDR video, 'bt2020' for wide-gamut primaries with an SDR transfer, else None.
    (A BT.2020 *matrix* alone is undone by the YUV->RGB conversion and needs nothing more.)"""
    ctx = stream.codec_context
    trc = getattr(frame, "color_trc", None) or getattr(ctx, "color_trc", None)
    prim = getattr(frame, "color_primaries", None) or getattr(ctx, "color_primaries", None)
    if trc is not None and int(trc) == _HLG:
        return "hlg"
    if trc is not None and int(trc) == _PQ:
        return "pq"
    if prim is not None and int(prim) == 9:
        return "bt2020"
    return None


# linear BT.2020 -> linear BT.709 (ITU-R BT.2087)
_M2020_709 = np.array([[1.6605, -0.5876, -0.0728],
                       [-0.1246, 1.1329, -0.0083],
                       [-0.0182, -0.1006, 1.1187]], dtype=np.float32)


def hdr_to_sdr(rgb16: np.ndarray, kind: str) -> np.ndarray:
    """HDR or wide-gamut RGB (uint16, as decoded: non-linear, BT.2020 primaries) -> SDR BT.709 uint8.

    Undo the transfer to linear light (HLG: inverse OETF plus the reference 1.2 system gamma; PQ: absolute
    nits), convert the primaries, put diffuse white (HLG 75 %, PQ 203 nits) at 0.8, roll the highlights off
    softly instead of clipping, and re-encode with the same gamma the rest of CueSync uses.
    """
    e = rgb16.astype(np.float32) / 65535.0
    if kind == "hlg":
        a, b, c = 0.17883277, 0.28466892, 0.55991073
        lin = np.where(e <= 0.5, e * e / 3.0, (np.exp((e - c) / a) + b) / 12.0)
        y = lin @ np.array([0.2627, 0.6780, 0.0593], np.float32)
        lin = lin * np.power(np.maximum(y, 1e-6), 0.2)[..., None]  # OOTF, system gamma 1.2
        white = 0.2647 ** 1.2  # HLG reference white (75 % signal) after the OOTF
    elif kind == "pq":
        m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
        ep = np.power(e, 1.0 / m2)
        lin = np.power(np.maximum(ep - c1, 0.0) / (c2 - c3 * ep), 1.0 / m1) * 10000.0  # nits
        white = 203.0
    else:  # wide gamut with an SDR transfer: only the primaries change (same gamma as the rest of CueSync)
        lin = np.power(e, 2.2)
        white = 0.8
    lin = (lin @ _M2020_709.T) * (0.8 / white)
    lin = np.maximum(lin, 0.0)
    m = lin.max(-1, keepdims=True)  # soft shoulder above 0.8, hue kept (compress the max channel)
    over = np.maximum(m - 0.8, 0.0)
    target = np.where(m > 0.8, 0.8 + 0.2 * (1.0 - np.exp(-over / 0.2)), m)
    lin = lin * (target / np.maximum(m, 1e-6))
    return np.clip(np.power(lin, 1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _rotation_k(frame: av.VideoFrame) -> int:
    """Quarter turns (np.rot90 k) that undo the phone's display-matrix rotation, matching ffmpeg autorotate."""
    return int(round((getattr(frame, "rotation", 0) or 0) / 90.0)) % 4


@dataclass(frozen=True)
class VideoInfo:
    width: int  # as displayed (rotation applied)
    height: int
    duration_s: float
    fps: float


def probe_video(path: str | Path) -> VideoInfo:
    """Displayed size, duration and nominal rate, from the first frame and the container (no full decode)."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        frame = next(container.decode(stream), None)
        if frame is None:
            raise ValueError(f"no decodable video frames in {path}")
        w, h = (frame.height, frame.width) if _rotation_k(frame) % 2 else (frame.width, frame.height)
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base
        elif stream.frames and fps:
            duration = stream.frames / fps
        else:
            duration = 0.0
    return VideoInfo(width=w, height=h, duration_s=duration, fps=fps)


def read_video(path: str | Path, size: tuple[int, int] | None = None, max_s: float | None = None,
               target_fps: float | None = None) -> VideoClip:
    """Decode every frame as RGB, *as displayed* (phone rotation metadata applied), with its timestamp.

    Sorted by timestamp (VFR- and out-of-order-safe). ``size`` = (W, H) of the displayed frame scales
    during decode (keeps memory bounded for previews/compares). ``max_s`` stops decoding once frames are more
    than that many seconds after the first. HDR plates (an iPhone's default HLG, or PQ) are tone-mapped
    to SDR BT.709. ``target_fps`` retains only the frames a sample-and-hold conform can use, before
    resizing and tone-mapping; ``duration_s`` still describes the original source timestamps.
    """
    if target_fps is not None and (not np.isfinite(target_fps) or target_fps <= 0):
        raise ValueError("target_fps must be positive finite")
    frames: list[np.ndarray] = []
    times: list[float] = []
    source_times: list[float] = []
    pending, pending_time, pending_slot = None, 0.0, 0
    out_of_order = False
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        hdr = None

        def append(frame, timestamp):
            k = _rotation_k(frame)
            fmt = "rgb48le" if hdr else "rgb24"
            if size is not None:
                w, h = (size[1], size[0]) if k % 2 else size  # decode in stored orientation, then rotate
                arr = frame.to_ndarray(format=fmt, width=w, height=h)
            else:
                arr = frame.to_ndarray(format=fmt)
            if hdr:
                arr = hdr_to_sdr(arr, hdr)
            frames.append(np.ascontiguousarray(np.rot90(arr, k)) if k else arr)
            times.append(timestamp)

        for frame in container.decode(stream):
            if not source_times:
                hdr = _hdr_kind(frame, stream)
            elif max_s is not None and frame.time is not None and frame.time - source_times[0] > max_s + 0.5:
                break  # a margin past max_s: out-of-order frames near the cut still land in order
            timestamp = float(frame.time) if frame.time is not None else len(source_times) / (fps or 1.0)
            if target_fps is not None and source_times and timestamp < source_times[-1]:
                out_of_order = True
                break
            source_times.append(timestamp)
            if target_fps is None:
                append(frame, timestamp)
                continue
            # The last source frame before each output tick wins, including equal timestamps. Keep the
            # decoded frame pending so discarded 60 fps frames never allocate RGB or run the HDR curve.
            slot = int(np.ceil((timestamp - source_times[0] - 1e-9) * target_fps))
            if pending is not None and slot != pending_slot:
                append(pending, pending_time)
            pending, pending_time, pending_slot = frame, timestamp, slot
        if pending is not None and not out_of_order:
            append(pending, pending_time)
    if out_of_order:
        frames.clear()
        pending = None
        # Broken presentation order is uncommon; preserve the full decoder's stable sort and duration.
        clip = read_video(path, size=size, max_s=max_s)
        slots = np.ceil((clip.times_s - 1e-9) * target_fps)
        keep = np.r_[np.flatnonzero(np.diff(slots) != 0), len(slots) - 1]
        clip.frames, clip.times_s = clip.frames[keep], clip.times_s[keep]
        return clip
    if not frames:
        raise ValueError(f"no decodable video frames in {path}")
    t = np.asarray(times, dtype=np.float64)
    if np.all(np.diff(t) >= 0):
        stacked = np.stack(frames)
    else:  # some encoders write frames out of order
        order = np.argsort(t, kind="stable")
        stacked = np.stack([frames[i] for i in order])
        t = t[order]
    del frames
    source_t = np.sort(source_times)
    frame_duration = float(np.median(np.diff(source_t))) if len(source_t) > 1 else 1.0 / (fps or target_fps or 24)
    duration = float(source_t[-1] - source_t[0] + frame_duration)
    return VideoClip(frames=stacked, times_s=t - source_t[0], fps_nominal=fps,
                     start_s=float(source_t[0]), duration_s=duration)


def write_video(
    path: str | Path,
    frames: np.ndarray,
    fps: int = 24,
    audio: np.ndarray | None = None,
    sample_rate: int = 48000,
    crf: int = 12,
    preset: str = "medium",
) -> None:
    """Encode RGB frames to H.264 (yuv420p, BT.709, tagged) MP4, optionally muxing mono float audio as AAC.

    The RGB->YUV matrix is BT.709 and the stream says so: an untagged file converted with swscale's default
    (BT.601) plays back with shifted reds and greens in browsers, which assume BT.709 for HD."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (N,H,W,3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    n, h, w, _ = frames.shape
    if h % 2 or w % 2:
        raise ValueError(f"yuv420p needs even dimensions, got {w}x{h}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        vstream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        vstream.width, vstream.height, vstream.pix_fmt = w, h, "yuv420p"
        vstream.options = {"crf": str(crf), "preset": preset}
        cc = vstream.codec_context
        cc.colorspace, cc.color_primaries, cc.color_trc, cc.color_range = 1, 1, 1, 1  # BT.709, limited range
        astream = None
        if audio is not None:
            astream = container.add_stream("aac", rate=sample_rate, layout="mono")
        for i in range(n):
            vframe = av.VideoFrame.from_ndarray(frames[i], format="rgb24").reformat(
                format="yuv420p", dst_colorspace=Colorspace.ITU709, dst_color_range=ColorRange.MPEG)
            for packet in vstream.encode(vframe):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)
        if astream is not None:
            samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
            chunk = 1024
            for start in range(0, samples.size, chunk):
                piece = samples[start:start + chunk].reshape(1, -1)
                aframe = av.AudioFrame.from_ndarray(np.ascontiguousarray(piece), format="fltp", layout="mono")
                aframe.sample_rate = sample_rate
                aframe.pts = start
                aframe.time_base = Fraction(1, sample_rate)
                for packet in astream.encode(aframe):
                    container.mux(packet)
            for packet in astream.encode():
                container.mux(packet)


def read_audio(path: str | Path, sample_rate: int = 48000) -> np.ndarray:
    """Decode the first audio stream (wav/m4a/webm/mp4) to mono float32 in [-1, 1] at ``sample_rate``."""
    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"no audio stream in {path}")
        stream = container.streams.audio[0]
        # Keep the source channel layout and average the channels ourselves: FFmpeg's mono downmix
        # sums stereo (+3 dB for L=R), which would make the "absolute" loudness depend on the file layout.
        resampler = av.AudioResampler(format="fltp", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().mean(axis=0))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().mean(axis=0))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)
