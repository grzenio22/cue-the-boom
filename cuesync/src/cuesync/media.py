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
    times_s: np.ndarray  # (N,) float64, ascending, times_s[0] == 0
    fps_nominal: float
    start_s: float = 0.0  # absolute container time of the first frame


_HDR_TRC = {16, 18}  # SMPTE ST 2084 (PQ), ARIB STD-B67 (HLG)
_BT2020_SPACES = {9, 10}  # BT.2020 non-constant / constant luminance


class HDRVideoError(ValueError):
    """The plate is HDR (HLG/PQ/BT.2020); converting it without tone mapping would look dim and washed out."""


def _check_sdr(frame: av.VideoFrame, stream: av.video.stream.VideoStream, path) -> None:
    ctx = stream.codec_context
    trc = getattr(frame, "color_trc", None) or getattr(ctx, "color_trc", None)
    space = getattr(frame, "colorspace", None) or getattr(ctx, "colorspace", None)
    if (trc is not None and int(trc) in _HDR_TRC) or (space is not None and int(space) in _BT2020_SPACES):
        raise HDRVideoError(
            f"{path} is HDR video (transfer {trc}, colorspace {space}). Turn off HDR Video on the phone "
            "(iPhone: Settings > Camera > Record Video > HDR Video), or convert it to SDR BT.709 first."
        )


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


def read_video(path: str | Path, size: tuple[int, int] | None = None, max_s: float | None = None) -> VideoClip:
    """Decode every frame as RGB, *as displayed* (phone rotation metadata applied), with its timestamp.

    Sorted by timestamp (VFR- and out-of-order-safe). ``size`` = (W, H) of the displayed frame scales
    during decode (keeps memory bounded for previews/compares). ``max_s`` stops decoding once frames are more
    than that many seconds after the first. HDR plates raise HDRVideoError.
    """
    frames: list[np.ndarray] = []
    times: list[float] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        for frame in container.decode(stream):
            if not frames:
                _check_sdr(frame, stream, path)
            elif max_s is not None and frame.time is not None and frame.time - times[0] > max_s + 0.5:
                break  # a margin past max_s: out-of-order frames near the cut still land in order
            k = _rotation_k(frame)
            if size is not None:
                w, h = (size[1], size[0]) if k % 2 else size  # decode in stored orientation, then rotate
                arr = frame.to_ndarray(format="rgb24", width=w, height=h)
            else:
                arr = frame.to_ndarray(format="rgb24")
            frames.append(np.ascontiguousarray(np.rot90(arr, k)) if k else arr)
            if frame.time is not None:
                times.append(float(frame.time))
            else:  # no timestamp: fall back to the nominal rate
                times.append(len(times) / fps if fps else float(len(times)))
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
    return VideoClip(frames=stacked, times_s=t - t[0], fps_nominal=fps, start_s=float(t[0]))


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
