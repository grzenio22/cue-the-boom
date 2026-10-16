import numpy as np

from cuesync.conform import conform, resize_frames
from cuesync.media import VideoClip, read_video


def test_vfr_and_30fps_resample_by_timestamp():
    times = np.array([0, 1/30, 2/30, 0.2, 0.25, 0.4, 0.5])          # variable gaps
    frames = np.stack([np.full((4, 4, 3), i*30, np.uint8) for i in range(7)])
    out = conform(VideoClip(frames, times, 30.0), size=(4, 4), fps=24, duration_s=0.5)
    assert out.shape[0] == 12                                         # round(0.5*24)
    assert out[5, 0, 0, 0] == 90                                      # t=0.2083 -> latest ts <= t is 0.2 (index 3)


def test_center_crop_to_9x16():
    frames = np.zeros((1, 100, 200, 3), np.uint8); frames[:, :, 100:] = 255   # landscape, right half white
    out = conform(VideoClip(frames, np.array([0.0]), 24.0), size=(90, 160), fps=24, duration_s=1/24)
    assert out.shape == (1, 160, 90, 3)
    assert out[0, 80, 10, 0] < 30 and out[0, 80, 80, 0] > 225         # crop centred on the boundary


def test_30fps_file_conforms_to_24fps_length(tmp_path, make_video):
    p = make_video(tmp_path, n=90, fps=30, w=180, h=320, color_fn=lambda i: (i % 256, 0, 0))
    out = conform(read_video(p), size=(90, 160), fps=24)
    assert out.shape == (72, 160, 90, 3)                              # 3.0 s at 24 fps


def test_resize_frames_shape():
    assert resize_frames(np.zeros((2, 10, 20, 3), np.uint8), (8, 6)).shape == (2, 6, 8, 3)
