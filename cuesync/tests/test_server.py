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


def wait(client, job_id):
    for _ in range(600):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    return job


@pytest.fixture
def instant_client(tmp_path, make_video, make_wav):
    """An instant-only server (no GPU renderer) with one sample plate and one sample noise."""
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    plate = make_video(tmp_path, n=48, fps=24, w=180, h=320, color_fn=lambda i: (60, 60, 60))
    plate.rename(samples / "plates" / "mug.mp4")
    make_voice(tmp_path, make_wav).rename(samples / "voices" / "pf.wav")
    (samples / "samples.json").write_text(json.dumps({"plates": {"mug.mp4": {"title": "Mug", "point": [0.5, 0.6]}},
                                                      "voices": {"pf.wav": {"title": "pf.", "effect": "fireball"}}}))
    app = create_app(None, tmp_path / "jobs", instant=FakeRenderer(), samples_dir=samples,
                     shot_defaults={"plate_size": (144, 256), "element_size": (128, 256), "max_plate_s": 2.0})
    return TestClient(app)


def test_samples_are_listed_and_served(instant_client):
    s = instant_client.get("/api/samples").json()
    assert s["plates"][0]["title"] == "Mug" and s["plates"][0]["point"] == [0.5, 0.6]
    assert s["voices"][0]["effect"] == "fireball"
    assert instant_client.get(s["plates"][0]["url"]).status_code == 200
    assert instant_client.get("/api/samples/plates/../samples.json").status_code == 404


def test_an_instant_only_server_refuses_full_renders(instant_client):
    h = instant_client.get("/api/health").json()
    assert h["modes"] == ["instant"] and h["renderer"] is None
    r = instant_client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "mug.mp4", "voice_sample": "pf.wav",
                                               "mode": "render"})
    assert r.status_code == 400 and "not available" in r.text


def test_two_sample_takes_and_their_compare(instant_client):
    ids = []
    for seed in (1, 2):
        r = instant_client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "mug.mp4", "voice_sample": "pf.wav",
                                                   "mode": "instant", "x": "0.5", "y": "0.6", "seed": str(seed)})
        assert r.status_code == 200, r.text
        job = wait(instant_client, r.json()["id"])
        assert job["status"] == "done" and job["mode"] == "instant", job
        ids.append(job["id"])
    c = instant_client.post("/api/compare", data={"a": ids[0], "b": ids[1]})
    assert c.status_code == 200, c.text
    assert instant_client.get(c.json()["files"]["compare.mp4"]).status_code == 200


def test_a_job_needs_exactly_one_plate_and_one_voice(instant_client):
    r = instant_client.post("/api/jobs", data={"effect": "fireball", "voice_sample": "pf.wav", "mode": "instant"})
    assert r.status_code == 400
    r = instant_client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "nope.mp4", "voice_sample": "pf.wav",
                                               "mode": "instant"})
    assert r.status_code == 400


def test_an_offline_gpu_is_not_offered(tmp_path):
    class Offline(FakeRenderer):
        def ready(self):
            return False

    client = TestClient(create_app(Offline(), tmp_path / "jobs", instant=FakeRenderer(), samples_dir=tmp_path))
    assert client.get("/api/health").json()["modes"] == ["instant"]
    r = client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "x.mp4", "voice_sample": "y.wav", "mode": "render"})
    assert r.status_code in (400, 503)


def test_a_sample_made_ahead_of_time_is_answered_at_once_and_shown_in_the_gallery(tmp_path, make_video, make_wav):
    from cuesync.precompute import cache_key
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    make_video(tmp_path, n=24, fps=24, w=90, h=160, color_fn=lambda i: (60, 60, 60)).rename(samples / "plates" / "mug.mp4")
    make_voice(tmp_path, make_wav).rename(samples / "voices" / "pf.wav")
    (samples / "samples.json").write_text(json.dumps({"plates": {"mug.mp4": {"point": [0.5, 0.6]}}, "voices": {"pf.wav": {}}}))
    shot = tmp_path / "pre" / "instant" / "mug__pf__fireball"
    shot.mkdir(parents=True)
    for name in ("final.mp4", "composite.mp4", "receipt.json", "cue.json", "guide.json"):
        (shot / name).write_bytes(b"{}" if name.endswith(".json") else b"\x00" * 64)
    key = cache_key("instant", "mug.mp4", "pf.wav", "fireball", (0.5, 0.6))
    (tmp_path / "pre" / "instant" / "index.json").write_text(json.dumps(
        {key: {"dir": "mug__pf__fireball", "title": "pf. · Mug", "plate": "mug.mp4", "voice": "pf.wav",
               "effect": "fireball", "mode": "instant"}}))
    client = TestClient(create_app(None, tmp_path / "jobs", instant=FakeRenderer(), samples_dir=samples,
                                   precomputed=tmp_path / "pre"))
    g = client.get("/api/gallery").json()
    assert len(g) == 1 and g[0]["title"] == "pf. · Mug" and "final.mp4" in g[0]["files"]
    r = client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "mug.mp4", "voice_sample": "pf.wav",
                                       "mode": "instant", "x": "0.5", "y": "0.6", "seed": "1"}).json()
    assert r["status"] == "done" and r["cached"] is True and "final.mp4" in r["files"]
    other = client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "mug.mp4", "voice_sample": "pf.wav",
                                           "mode": "instant", "x": "0.3", "y": "0.6"}).json()
    assert other["cached"] is False  # a different spot is made fresh


def test_a_finished_shot_is_retimed_through_the_api(instant_client):
    r = instant_client.post("/api/jobs", data={"effect": "fireball", "plate_sample": "mug.mp4", "voice_sample": "pf.wav",
                                               "mode": "instant", "x": "0.5", "y": "0.6"})
    job = wait(instant_client, r.json()["id"])
    assert job["status"] == "done", job
    rt = instant_client.post(f"/api/jobs/{job['id']}/retime", data={"shift": "4"})
    assert rt.status_code == 200, rt.text
    done = wait(instant_client, rt.json()["id"])
    assert done["status"] == "done", done
    assert done["receipt"]["retimed"]["hit_shift_frames"] == 4
    assert instant_client.post(f"/api/jobs/{job['id']}/retime", data={"shift": "99"}).status_code == 400
