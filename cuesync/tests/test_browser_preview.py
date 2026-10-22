"""A phone fallback preview keeps the source timeline used by hand taps and audio."""

from fractions import Fraction

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cuesync.media import browser_copy, probe_video, read_audio, read_video
from cuesync.server import create_app


def _vfr_clip(path, frame_numbers=None):
    """Red for one second, held across a one-second gap, then blue; sound at 2.4 s."""
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=30)
        video.width, video.height, video.pix_fmt = 64, 32, "yuv420p"
        video.codec_context.time_base = Fraction(1, 30)
        audio = container.add_stream("aac", rate=48000, layout="mono")
        for i in frame_numbers if frame_numbers is not None else list(range(30)) + list(range(60, 90)):
            rgb = np.empty((32, 64, 3), np.uint8)
            rgb[:] = (240, 0, 0) if i < 60 else (0, 0, 240)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        samples = np.zeros(144000, np.float32)
        samples[115200:120000] = 0.6 * np.sin(np.arange(4800) * 0.1)
        for start in range(0, len(samples), 1024):
            frame = av.AudioFrame.from_ndarray(samples[None, start:start + 1024], format="fltp", layout="mono")
            frame.sample_rate = 48000
            frame.pts, frame.time_base = start, Fraction(1, 48000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    return path


def test_preview_preserves_vfr_gaps_and_sound_on_the_source_timeline(tmp_path):
    src = _vfr_clip(tmp_path / "phone.mov")
    out = browser_copy(src, tmp_path / "preview.mp4")
    info, clip, audio = probe_video(out), read_video(out), read_audio(out)

    assert info.duration_s == pytest.approx(3.0, abs=1 / info.fps)
    blue = clip.frames[..., 2].mean(axis=(1, 2)) > 180
    assert clip.times_s[np.flatnonzero(blue)[0]] == pytest.approx(2.0, abs=1 / info.fps)
    assert len(audio) / 48000 == pytest.approx(3.0, abs=0.05)
    audible = np.flatnonzero(np.abs(audio) > 0.1)
    assert audible[0] / 48000 == pytest.approx(2.4, abs=0.01)


def test_preview_uses_known_duration_instead_of_extrapolating_a_long_vfr_gap(tmp_path):
    src = _vfr_clip(tmp_path / "sparse.mov", frame_numbers=[0, 400])
    # Two frames thirteen seconds apart do not mean a twenty-six-second movie: the final
    # packet lasts 1/30 s, which the stream duration knows but the median frame gap cannot.
    out = browser_copy(src, tmp_path / "preview.mp4")
    assert probe_video(out).duration_s == pytest.approx(13 + 11 / 30, abs=1 / 30)


@pytest.mark.parametrize("known_duration", [True, False])
def test_preview_rejects_an_overlong_source_instead_of_truncating_it(tmp_path, make_video, monkeypatch, known_duration):
    src = make_video(tmp_path, n=32, fps=2, w=16, h=16, color_fn=lambda i: (i, 0, 0))
    if not known_duration:
        from dataclasses import replace
        import cuesync.media as media

        info = probe_video(src)
        monkeypatch.setattr(media, "probe_video", lambda _: replace(info, duration_s=0.0))
    out = tmp_path / "preview.mp4"
    with pytest.raises(ValueError, match="15"):
        browser_copy(src, out)
    assert not out.exists()


def test_preview_endpoint_rejects_overlong_phone_files(tmp_path, make_video):
    src = make_video(tmp_path, n=32, fps=2, w=16, h=16, color_fn=lambda i: (i, 0, 0))
    with TestClient(create_app(None, tmp_path / "jobs")) as client, src.open("rb") as fh:
        result = client.post("/api/preview", files={"plate": ("long.mov", fh, "video/quicktime")})
    assert result.status_code == 400
    assert "15" in result.json()["detail"]
