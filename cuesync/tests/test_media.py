import numpy as np

from cuesync.media import read_audio, read_video, write_video


def test_roundtrip_video(tmp_path, make_video):
    p = make_video(tmp_path, n=48, fps=24, w=64, h=128, color_fn=lambda i: (i*5 % 256, 0, 0))
    clip = read_video(p)
    assert clip.frames.shape == (48, 128, 64, 3)
    assert abs(clip.times_s[1] - 1/24) < 1e-3
    assert abs(int(clip.frames[10, 64, 32, 0]) - 50) <= 6   # H.264 tolerance


def test_write_video_with_audio_then_read_audio(tmp_path):
    frames = np.zeros((24, 64, 64, 3), np.uint8)
    tone = (0.5*np.sin(2*np.pi*440*np.arange(48000)/48000)).astype(np.float32)
    write_video(tmp_path/"a.mp4", frames, fps=24, audio=tone, sample_rate=48000)
    a = read_audio(tmp_path/"a.mp4")
    assert 47000 < len(a) < 50000 and 0.3 < np.abs(a).max() <= 1.0


def test_read_wav(tmp_path, make_wav):
    x = (0.25*np.sin(np.arange(24000)*0.05)).astype(np.float32)
    a = read_audio(make_wav(tmp_path, x, 48000))
    assert len(a) == 24000 and abs(np.abs(a).max() - 0.25) < 0.01


def test_black_stays_black(tmp_path):
    write_video(tmp_path/"k.mp4", np.zeros((9, 64, 64, 3), np.uint8), fps=24)
    assert read_video(tmp_path/"k.mp4").frames.max() <= 2


def test_written_video_is_bt709_and_tagged(tmp_path):
    """Browsers decode HD as BT.709: the file must be converted with that matrix and say so."""
    import av
    path = tmp_path / "c.mp4"
    frames = np.zeros((4, 64, 64, 3), np.uint8)
    frames[:] = (200, 60, 30)
    write_video(path, frames)
    with av.open(str(path)) as c:
        cc = c.streams.video[0].codec_context
        assert (cc.colorspace, cc.color_primaries, cc.color_trc) == (1, 1, 1)
        y = next(c.decode(video=0)).to_ndarray(format="yuv420p")[32, 32]
    assert abs(int(y) - 91) <= 1  # BT.709 luma of (200, 60, 30) in limited range; BT.601 would give 101
    assert np.abs(read_video(path).frames[1, 32, 32].astype(int) - (200, 60, 30)).max() <= 3
