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
