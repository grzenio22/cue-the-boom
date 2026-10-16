import os
import wave
from pathlib import Path

import numpy as np
import pytest

from cuesync.media import write_video

# Tests reference repo-level files (effects/*.json) by relative path, as the plan does.
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def make_video():
    def _make(tmp_path, n, fps, w, h, color_fn):
        frames = np.zeros((n, h, w, 3), np.uint8)
        for i in range(n):
            frames[i] = np.array(color_fn(i), np.uint8)
        path = Path(tmp_path) / f"video_{n}_{fps}_{w}x{h}_{os.urandom(3).hex()}.mp4"
        write_video(path, frames, fps=fps)
        return path

    return _make


@pytest.fixture
def make_wav():
    def _make(tmp_path, samples, sr):
        path = Path(tmp_path) / f"noise_{os.urandom(3).hex()}.wav"
        pcm = (np.clip(np.asarray(samples, np.float32), -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return path

    return _make
