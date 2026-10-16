"""The web app's API end to end, with a stand-in renderer (no GPU)."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cuesync.media import write_video  # noqa: E402
from cuesync.server import create_app  # noqa: E402

SR = 48000


class FakeRenderer:
    """Writes a fire element as long as the guide, so the whole pipeline runs without a GPU."""

    name = "fake"

    def __call__(self, guide_path, prompt, seed, out_path):
        from cuesync.media import read_video

        n = read_video(guide_path).frames.shape[0]
        f = np.zeros((n, 256, 128, 3), np.uint8)
        f[:, 120:200, 40:88] = (255, 150, 30)
        write_video(out_path, f, fps=24)
        return 0.1


@pytest.fixture
def client(tmp_path, make_video):
    app = create_app(FakeRenderer(), tmp_path / "jobs",
                     shot_defaults={"plate_size": (144, 256), "element_size": (128, 256), "max_plate_s": 3.0})
    return TestClient(app)


def make_voice(tmp_path, make_wav):
    x = np.random.default_rng(1).normal(0, 0.002, 3 * SR).astype(np.float32)
    x[SR: SR + 9600] += (0.5 * np.sin(np.arange(9600) * 0.06)).astype(np.float32)
    return make_wav(tmp_path, x, SR)


def test_health_and_effects(client):
    assert client.get("/api/health").json()["renderer"] == "fake"
    ids = {e["id"] for e in client.get("/api/effects").json()}
    assert {"fireball", "flame_plume", "smoke_puff", "steam_jet"} <= ids


def test_a_shot_end_to_end(client, tmp_path, make_video, make_wav):
    plate = make_video(tmp_path, n=60, fps=24, w=180, h=320, color_fn=lambda i: (40, 40, 40))
    voice = make_voice(tmp_path, make_wav)
    timing = {"version": 1, "noise_offset_s": 0.25, "input_latency_s": 0.05, "emission_point": [0.5, 0.6], "face_offset_s": None}
    with open(plate, "rb") as p, open(voice, "rb") as v:
        r = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("mic.wav", v, "audio/wav"),
                                            "timing": ("timing.json", json.dumps(timing), "application/json")},
                        data={"effect": "flame_plume", "seed": "3"})
    assert r.status_code == 200, r.text
    job = r.json()
    for _ in range(600):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    assert set(job["files"]) >= {"final.mp4", "composite.mp4", "breakdown.mp4", "receipt.json"}
    rec = job["receipt"]
    assert rec["inputs"]["offset_s"] == pytest.approx(0.2) and rec["inputs"]["point"] == [0.5, 0.6] and rec["inputs"]["seed"] == 3
    assert rec["measured"]["onset_error_frames"] is not None
    video = client.get(job["files"]["final.mp4"])
    assert video.status_code == 200 and len(video.content) > 1000
    events = client.get(f"/api/jobs/{job['id']}/events").text
    assert "rendering" in events and "event: end" in events


def test_bad_effect_is_rejected(client, tmp_path, make_video, make_wav):
    plate = make_video(tmp_path, n=10, fps=24, w=180, h=320, color_fn=lambda i: (40, 40, 40))
    voice = make_voice(tmp_path, make_wav)
    with open(plate, "rb") as p, open(voice, "rb") as v:
        r = client.post("/api/jobs", files={"plate": ("p.mp4", p), "voice": ("v.wav", v)}, data={"effect": "nope"})
    assert r.status_code == 400
