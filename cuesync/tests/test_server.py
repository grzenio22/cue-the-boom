"""The web app's API end to end, with a stand-in renderer (no GPU)."""

import json
import shutil
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


@pytest.fixture(autouse=True)
def no_hand_tracker(monkeypatch):
    """Tests don't depend on this machine's helper environments: no hand tracker unless a test installs one."""
    import cuesync.bending as bending

    monkeypatch.setattr(bending, "helpers_ready", lambda: False)


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


@pytest.mark.parametrize("again", [None, 0, 1, 2])
def test_gallery_jobs_preserve_again_from_the_index(tmp_path, again):
    shot = tmp_path / "pre" / "scene" / "sample"
    shot.mkdir(parents=True)
    (shot / "final.mp4").write_bytes(b"finished shot")
    entry = {"dir": "sample", "title": "Sample", "plate": "sample.mp4", "mode": "scene"}
    if again is not None:
        entry["again"] = again
    (shot.parent / "index.json").write_text(json.dumps({"sample": entry}))
    client = TestClient(create_app(None, tmp_path / "jobs", precomputed=tmp_path / "pre"))

    gallery = client.get("/api/gallery").json()
    assert len(gallery) == 1
    assert gallery[0].get("again") == again
    if again is not None:
        assert type(gallery[0]["again"]) is int
    assert client.get(f"/api/jobs/{gallery[0]['id']}").json().get("again") == again


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


# ---------------------------------------------------------------- cues, drawings, the noise deciding


class FakeSceneRenderer:
    """Renders 'into the scene': the start frame held, a warm disc from the painted hit on."""

    name, scenes = "fake-scene", True

    def __init__(self):
        self.prompts = []

    def __call__(self, guide_path, prompt, seed, out_path, image=None):
        from PIL import Image

        from cuesync.media import read_video

        self.prompts.append(prompt)
        n, h, w = read_video(guide_path).frames.shape[:3]
        start = np.asarray(Image.open(image).convert("RGB"))
        at = json.loads((Path(guide_path).parent / "guide.json").read_text())["events"][0]["at"]
        f = np.repeat(start[None], n, 0)
        f[at:, h // 3: h // 2, w // 3: 2 * w // 3] = (250, 170, 50)
        write_video(out_path, f, fps=24, audio=np.zeros(int(n / 24 * SR), np.float32))
        return 0.1

    def ready(self):
        return True


@pytest.fixture
def scene_client(tmp_path):
    r = FakeSceneRenderer()
    app = create_app(r, tmp_path / "jobs", shot_defaults={"plate_size": (128, 256), "max_plate_s": 3.0,
                                                          "matte": lambda f, p: None})
    return TestClient(app), r


def wait(client, job_id, limit=120):
    for _ in range(limit * 4):
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.25)
    raise AssertionError("job did not finish")


def test_a_held_drawn_scene_shot_needs_no_voice(scene_client, tmp_path, make_video):
    client, r = scene_client
    assert "scene" in client.get("/api/health").json()["modes"]
    plate = make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))
    strokes = [{"points": [[0.3, 0.6], [0.6, 0.4]], "width": 0.08}]
    with open(plate, "rb") as p:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                        data={"effect": "smoke_puff", "mode": "scene", "cue": "hold", "holds": json.dumps([[0.8, 1.6]]),
                              "strokes": json.dumps(strokes), "describe": "smoke pours out of the box"}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    rc = done["receipt"]
    assert rc["inputs"]["cue"] == "hold" and rc["inputs"]["drawn"] and rc["voice"]["source"] == "hold"
    assert rc["inputs"]["holds"] == [[0.8, 1.6]]  # so a re-time is cued the same way
    assert r.prompts[-1].startswith("ainvfxfluid, smoke pours out of the box,")


def test_health_reports_model_preflight_and_observes_new_configuration_immediately(scene_client, monkeypatch):
    from dataclasses import dataclass, replace

    @dataclass(frozen=True)
    class Readiness:
        ready: bool = False
        scene_ready: bool = False
        connected: bool = True
        status: str = "missing_models"
        message: str = "Install the configured checkpoint"
        missing_nodes: tuple = ()
        missing_models: tuple = ("model.safetensors",)

    client, renderer = scene_client
    state = Readiness()
    monkeypatch.setattr(renderer, "preflight", lambda: state, raising=False)
    health = client.get("/api/health").json()
    assert health["render_backend"]["missing_models"] == ["model.safetensors"]
    assert "render" not in health["modes"] and "scene" not in health["modes"]
    state = replace(state, ready=True, status="ready", missing_models=(), missing_nodes=("LoadImage",))
    health = client.get("/api/health").json()
    assert "render" in health["modes"] and "scene" not in health["modes"]
    state = replace(state, scene_ready=True, missing_nodes=())
    assert "scene" in client.get("/api/health").json()["modes"]


def test_scene_generation_rejects_backend_missing_scene_nodes(scene_client, tmp_path, make_video, monkeypatch):
    from types import SimpleNamespace

    client, renderer = scene_client
    monkeypatch.setattr(renderer, "preflight", lambda: SimpleNamespace(ready=True, scene_ready=False), raising=False)
    plate = make_video(tmp_path, n=12, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))
    with plate.open("rb") as source:
        result = client.post("/api/jobs", files={"plate": ("plate.mp4", source, "video/mp4")},
                             data={"effect": "fire_burst", "mode": "scene", "cue": "hold", "holds": "[[0.1, 0.2]]"})
    assert result.status_code == 503, result.text
    assert not renderer.prompts


@pytest.mark.parametrize("prepared_sample", [False, True])
def test_bending_generation_only_needs_element_readiness(tmp_path, monkeypatch, prepared_sample):
    from dataclasses import dataclass

    import cuesync.bending as bending
    from cuesync.server import JobQueue

    @dataclass(frozen=True)
    class Readiness:
        ready: bool = True
        scene_ready: bool = False

    renderer = FakeElementRenderer()
    monkeypatch.setattr(renderer, "preflight", lambda: Readiness(), raising=False)
    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    monkeypatch.setattr(bending, "library_dir", lambda: None)
    queued = []

    def capture(queue, job):
        queued.append(job)
        queue.jobs[job.id] = job
        return job

    monkeypatch.setattr(JobQueue, "submit", capture)
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "plates" / "hand.mp4").write_bytes(b"prepared plate")
    (samples / "samples.json").write_text(json.dumps({"plates": {"hand.mp4": {
        "point": [0.4, 0.6], "hand": {"frame": 10}}}}))
    shot = samples / "made_ahead" / "bending" / "sample"
    shot.mkdir(parents=True)
    (shot / "final.mp4").write_bytes(b"prepared result")
    (shot.parent / "index.json").write_text(json.dumps({"hand": {
        "dir": "sample", "title": "Hand", "plate": "hand.mp4", "mode": "bending", "effect": "fire_burst"}}))
    client = TestClient(create_app(renderer, tmp_path / "jobs", samples_dir=samples,
                                   precomputed=samples / "made_ahead"))
    modes = client.get("/api/health").json()["modes"]
    assert "render" in modes and "bending" in modes and "scene" not in modes
    data = {"mode": "bending", "effect": "fire_burst", "x": "0.4", "y": "0.6"}
    files = {"voice": ("mic.wav", b"recorded voice", "audio/wav")}
    if prepared_sample:
        data["plate_sample"] = "hand.mp4"
    else:
        files["plate"] = ("hand.mp4", b"uploaded plate", "video/mp4")
    response = client.post("/api/jobs", data=data, files=files)
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "bending"
    assert len(queued) == 1 and JobQueue.queue_of(queued[0].mode) == "render"
    assert queued[0].replay is None


def test_call_and_response_sources_go_through_the_api(scene_client, tmp_path, make_video):
    client, r = scene_client
    plate = make_video(tmp_path, n=72, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))
    sources = [[0.25, 0.7], [0.5, 0.5], [0.75, 0.3]]
    holds = [[0.5, 0.65], [1.1, 1.25], [1.7, 2.1]]  # pf, pf, KA-BOOM pressed on the button
    with open(plate, "rb") as p:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                        data={"effect": "fire_burst", "mode": "scene", "cue": "hold", "holds": json.dumps(holds),
                              "sources": json.dumps(sources)}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    rc = done["receipt"]
    assert rc["inputs"]["sources"] == sources and rc["plate"]["point"] == sources[0]
    assert rc["guide"]["sources"] == sources

    def refused(**data):
        with open(plate, "rb") as p:
            return client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                               data={"effect": "fire_burst", "cue": "hold", "holds": json.dumps(holds), **data}).status_code

    assert refused(mode="scene", sources=json.dumps([[0.5, 0.5]])) == 400  # one source is just a point
    assert refused(mode="scene", sources=json.dumps([[0.5, 0.5], [1.5, 0.5]])) == 400  # off the frame
    assert refused(mode="scene", sources="[[0.5]]") == 400
    assert refused(mode="instant", sources=json.dumps(sources)) == 400  # it is an in-scene render
    assert refused(mode="scene", sources=json.dumps(sources),
                   strokes=json.dumps([{"points": [[0.3, 0.6], [0.6, 0.4]], "width": 0.08}])) == 400


def test_a_scene_shot_is_performed_again_with_no_new_render(scene_client, tmp_path, make_video, make_wav):
    client, r = scene_client
    plate = make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))  # a still
    voice = make_voice(tmp_path, make_wav)
    with open(plate, "rb") as p, open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                        data={"effect": "fire_burst", "mode": "scene", "x": "0.5", "y": "0.45"}).json()
    first = wait(client, s["id"])
    assert first["status"] == "done", first.get("error")
    renders = len(r.prompts)
    holds = [[0.5, 0.65], [1.1, 1.25], [1.7, 2.1]]  # a new rhythm, pressed on the button
    again = client.post(f"/api/jobs/{s['id']}/replay", data={"cue": "hold", "holds": json.dumps(holds)}).json()
    done = wait(client, again["id"])
    assert done["status"] == "done", done.get("error")
    rc = done["receipt"]
    assert rc["replayed"]["no_new_render"] and rc["replayed"]["new_hits"] == 3 and len(rc["hits"]) == 3
    assert len(r.prompts) == renders and "final.mp4" in done["files"]  # nothing rendered again
    twice = client.post(f"/api/jobs/{again['id']}/replay", data={"cue": "hold", "holds": json.dumps(holds[:1])}).json()
    assert wait(client, twice["id"])["status"] == "done"  # performed again from a replay: its render's shot again
    assert client.post(f"/api/jobs/{s['id']}/replay", data={"cue": "voice"}).status_code == 400  # a voice cue needs a voice


def test_shots_on_one_plate_are_performed_together(scene_client, tmp_path, make_video, make_wav):
    client, r = scene_client
    plate = make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))  # a still
    voice = make_voice(tmp_path, make_wav)

    def shot(plate_path, x, y):
        with open(plate_path, "rb") as p, open(voice, "rb") as v:
            s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                            data={"effect": "fire_burst", "mode": "scene", "x": str(x), "y": str(y)}).json()
        assert wait(client, s["id"])["status"] == "done"
        return s["id"]

    ids = [shot(plate, 0.3, 0.7), shot(plate, 0.7, 0.3)]  # one render per source
    renders = len(r.prompts)
    holds = [[0.5, 0.65], [1.1, 1.25]]  # a rhythm for both
    again = client.post(f"/api/jobs/{ids[0]}/replay",
                        data={"cue": "hold", "holds": json.dumps(holds), "together": json.dumps(ids[1:])}).json()
    done = wait(client, again["id"])
    assert done["status"] == "done", done.get("error")
    rc = done["receipt"]
    assert rc["replayed"]["renders"] == 2 and rc["replayed"]["render_hits"] == 2 and len(r.prompts) == renders
    assert sorted(h["source"] for h in rc["hits"]) == [0, 1]  # each hit at its own source, from its own render
    twice = client.post(f"/api/jobs/{again['id']}/replay", data={"cue": "hold", "holds": json.dumps(holds)}).json()
    assert wait(client, twice["id"])["receipt"]["replayed"]["renders"] == 2  # performed again: both renders again
    elsewhere = shot(make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (20, 60, 90)), 0.5, 0.5)
    refused = client.post(f"/api/jobs/{ids[0]}/replay",
                          data={"cue": "hold", "holds": json.dumps(holds), "together": json.dumps([elsewhere])})
    assert refused.status_code == 409  # another plate
    assert client.post(f"/api/jobs/{ids[0]}/replay", data={"cue": "hold", "holds": json.dumps(holds),
                                                           "together": "nope"}).status_code == 400


def test_cue_inputs_are_checked(scene_client, tmp_path, make_video, make_wav):
    client, _ = scene_client
    plate = make_video(tmp_path, n=24, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))
    voice = make_voice(tmp_path, make_wav)

    def post(data, with_voice=False):
        files = {"plate": ("plate.mp4", open(plate, "rb"), "video/mp4")}
        if with_voice:
            files["voice"] = ("mic.wav", open(voice, "rb"), "audio/wav")
        return client.post("/api/jobs", files=files, data={"effect": "fireball", "mode": "scene"} | data)

    assert post({"cue": "voice"}).status_code == 400  # a voice cue needs a voice
    assert post({"cue": "hold"}, with_voice=True).status_code == 400  # a hold takes none
    assert post({"cue": "hold", "holds": "[[2, 1]]"}).status_code == 400  # release before press
    assert post({"cue": "hold", "holds": "not json"}).status_code == 400
    assert post({"cue": "tap"}).status_code == 400
    assert post({"cue": "hold", "strokes": "[]"}).status_code == 400
    assert post({"cue": "hold", "holds": "[[NaN, 1]]"}).status_code == 400
    assert post({"cue": "hold", "holds": "[[5, 6]]"}).status_code == 400  # pressed after the end of the shot
    for bad in ('[{"points": "no"}]', "[1, 2]", '[{"points": [[0.1, NaN]]}]', '{"points": 3}'):
        assert post({"cue": "hold", "strokes": bad}).status_code == 400, bad  # refused, never a server error
    r = client.post("/api/jobs", files={"plate": ("plate.mp4", open(plate, "rb"), "video/mp4")},
                    data={"effect": "fireball", "mode": "render", "cue": "hold", "describe": "fire"})
    assert r.status_code == 400  # describing needs scene mode


def test_the_noise_decides_the_effect_through_the_plate_s_map(tmp_path, make_wav):
    from cuesync.gestures import effect_for, gesture
    from cuesync.audio import analyze

    short = np.zeros(2 * SR, np.float32)
    short[SR: SR + 2400] = 0.6 * np.sin(np.arange(2400) * 0.05)  # "pf."
    cue = analyze(short, SR, fps=24)
    assert gesture(cue, short) == "hit"
    candle = {"scene": {"hit": "blow_out", "sustained": "wick_flare"}}
    assert effect_for("hit", candle) == "blow_out" and effect_for("sustained", candle) == "wick_flare"
    assert effect_for("sustained", {"direction": -90}) == "fire_breath"
    assert effect_for("hiss", {}) == "steam_jet"
    long_voiced = np.zeros(3 * SR, np.float32)
    long_voiced[SR: 2 * SR] = 0.5 * np.sin(np.arange(SR) * 2 * np.pi * 140 / SR)
    assert gesture(analyze(long_voiced, SR, fps=24), long_voiced) == "sustained"
    hiss = np.zeros(3 * SR, np.float32)
    hiss[SR: 2 * SR] = np.random.default_rng(0).normal(0, 0.3, SR)
    assert gesture(analyze(hiss, SR, fps=24), hiss) == "hiss"


def test_an_edited_effect_retires_its_old_sample_shots():
    from cuesync.precompute import cache_key

    k = cache_key("scene", "a.mp4", "b.wav", "fireball", (0.5, 0.5), 1, None)
    assert "fireball@" in k and len(k.split("@")[1].split("|")[0]) == 8


def test_auto_with_a_sample_noise_is_answered_from_the_shots_made_ahead(tmp_path, make_video, make_wav):
    from cuesync.precompute import cache_key
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    make_video(tmp_path, n=24, fps=24, w=90, h=160, color_fn=lambda i: (60, 60, 60)).rename(samples / "plates" / "mug.mp4")
    make_voice(tmp_path, make_wav).rename(samples / "voices" / "pf.wav")
    (samples / "samples.json").write_text(json.dumps({"plates": {"mug.mp4": {"point": [0.5, 0.6]}},
                                                      "voices": {"pf.wav": {"gesture": "hit"}}}))
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
    r = client.post("/api/jobs", data={"effect": "auto", "plate_sample": "mug.mp4", "voice_sample": "pf.wav",
                                       "mode": "instant", "x": "0.5", "y": "0.6", "seed": "1"}).json()
    assert r["cached"] is True and r["effect"] == "fireball" and r["gesture"] == "hit"  # the "pf." decided, at once


def test_with_the_gpu_offline_the_sample_scene_shots_made_ahead_still_play(tmp_path, make_video, make_wav):
    from cuesync.precompute import cache_key
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    make_video(tmp_path, n=24, fps=24, w=90, h=160, color_fn=lambda i: (60, 60, 60)).rename(samples / "plates" / "candle.mp4")
    make_voice(tmp_path, make_wav).rename(samples / "voices" / "pf.wav")
    (samples / "samples.json").write_text(json.dumps({
        "plates": {"candle.mp4": {"point": [0.5, 0.6], "scene": {"pf.wav": "blow_out", "hit": "blow_out"},
                                  "clip_cue": {"effect": "blow_out"}}},
        "voices": {"pf.wav": {"gesture": "hit"}}}))
    index = {}
    for voice in ("pf.wav", "clip"):
        name = f"candle__{voice.split('.')[0]}__blow_out"
        shot = tmp_path / "pre" / "scene" / name
        shot.mkdir(parents=True)
        for f in ("final.mp4", "composite.mp4", "receipt.json", "cue.json", "guide.json"):
            (shot / f).write_bytes(b"{}" if f.endswith(".json") else b"\x00" * 64)
        index[cache_key("scene", "candle.mp4", voice, "blow_out", (0.5, 0.6))] = {
            "dir": name, "title": name, "plate": "candle.mp4", "voice": voice, "effect": "blow_out", "mode": "scene"}
    (tmp_path / "pre" / "scene" / "index.json").write_text(json.dumps(index))
    client = TestClient(create_app(None, tmp_path / "jobs", instant=FakeRenderer(), samples_dir=samples,
                                   precomputed=tmp_path / "pre"))
    h = client.get("/api/health").json()
    assert "scene" not in h["modes"] and "scene" in h["made_ahead"]
    base = {"effect": "auto", "plate_sample": "candle.mp4", "mode": "scene", "x": "0.5", "y": "0.6"}
    said = client.post("/api/jobs", data=base | {"voice_sample": "pf.wav"}).json()
    assert said["cached"] is True and said["effect"] == "blow_out"
    snap = client.post("/api/jobs", data=base | {"cue": "clip"}).json()
    assert snap["cached"] is True and snap["effect"] == "blow_out"  # the snap in the clip, no GPU
    moved = client.post("/api/jobs", data=base | {"voice_sample": "pf.wav", "x": "0.3"})
    assert moved.status_code == 400 and "made ahead" in moved.text  # anything else needs the GPU, and says so


def test_with_the_gpu_offline_your_own_take_plays_the_samples_scene_shot_made_ahead(tmp_path, make_video, make_wav):
    from cuesync.pipeline import ShotSpec, make_shot
    from cuesync.precompute import cache_key

    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40)).rename(samples / "plates" / "jack.mp4")
    make_voice(tmp_path, make_wav).rename(samples / "voices" / "ka-boom.wav")
    (samples / "samples.json").write_text(json.dumps({
        "plates": {"jack.mp4": {"point": [0.5, 0.45], "subject": "the jack-o'-lantern", "scene": {"hit": "fire_burst"}}},
        "voices": {"ka-boom.wav": {"gesture": "hit"}}}))
    shot = tmp_path / "pre" / "scene" / "jack__ka-boom__fire_burst"
    make_shot(ShotSpec(plate=samples / "plates" / "jack.mp4", voice=samples / "voices" / "ka-boom.wav", effect="fire_burst",
                       point=(0.5, 0.45), mode="scene", plate_size=(128, 256), max_plate_s=3.0, matte=lambda f, p: None),
              FakeSceneRenderer(), shot)
    (tmp_path / "pre" / "scene" / "index.json").write_text(json.dumps({
        cache_key("scene", "jack.mp4", "ka-boom.wav", "fire_burst", (0.5, 0.45)): {
            "dir": shot.name, "title": "KA-BOOM · jack", "plate": "jack.mp4", "voice": "ka-boom.wav",
            "effect": "fire_burst", "mode": "scene"}}))
    client = TestClient(create_app(None, tmp_path / "jobs", instant=FakeRenderer(), samples_dir=samples,
                                   precomputed=tmp_path / "pre", shot_defaults={"plate_size": (128, 256), "max_plate_s": 3.0}))
    holds = [[0.5, 0.65], [1.3, 1.8]]  # the performer's own timing, not a sample noise
    s = client.post("/api/jobs", data={"effect": "fire_burst", "plate_sample": "jack.mp4", "mode": "scene", "x": "0.5",
                                       "y": "0.45", "cue": "hold", "holds": json.dumps(holds)}).json()
    assert s["mode"] == "replay"
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["replayed"]["no_new_render"] and done["receipt"]["replayed"]["new_hits"] == 2


def test_with_the_gpu_offline_a_samples_own_drawing_replays(tmp_path, make_video, make_wav):
    from cuesync.precompute import cache_key
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    make_video(tmp_path, n=24, fps=24, w=90, h=160, color_fn=lambda i: (90, 60, 30)).rename(samples / "plates" / "pumpkin.mp4")
    make_voice(tmp_path, make_wav).rename(samples / "voices" / "fwoooom.wav")
    strokes = [{"points": [[0.36, 0.53], [0.5, 0.4], [0.7, 0.54]], "width": 0.09}]
    (samples / "samples.json").write_text(json.dumps({
        "plates": {"pumpkin.mp4": {"point": [0.53, 0.6], "scene": {"sustained": "flame_plume"},
                                   "drawn": [{"title": "drawn", "voice": "fwoooom.wav", "effect": "flame_plume",
                                              "strokes": strokes, "describe": "flames spread along its top"}]}},
        "voices": {"fwoooom.wav": {"gesture": "sustained"}}}))
    shot = tmp_path / "pre" / "scene" / "drawn"
    shot.mkdir(parents=True)
    for f in ("final.mp4", "composite.mp4", "receipt.json", "cue.json", "guide.json"):
        (shot / f).write_bytes(b"{}" if f.endswith(".json") else b"\x00" * 64)
    (tmp_path / "pre" / "scene" / "index.json").write_text(json.dumps({
        cache_key("scene", "pumpkin.mp4", "fwoooom.wav", "flame_plume", (0.36, 0.53)): {
            "dir": "drawn", "title": "drawn", "plate": "pumpkin.mp4", "voice": "fwoooom.wav", "effect": "flame_plume",
            "mode": "scene", "drawn": True}}))
    client = TestClient(create_app(None, tmp_path / "jobs", instant=FakeRenderer(), samples_dir=samples,
                                   precomputed=tmp_path / "pre"))
    base = {"effect": "auto", "plate_sample": "pumpkin.mp4", "mode": "scene", "voice_sample": "fwoooom.wav",
            "x": "0.36", "y": "0.53"}
    same = client.post("/api/jobs", data=base | {"strokes": json.dumps(strokes)}).json()
    assert same["cached"] is True and same["effect"] == "flame_plume"
    nudged = [{"points": [[0.362, 0.531], [0.5, 0.4], [0.7, 0.54]], "width": 0.09}]  # the same drawing, a pixel off
    assert client.post("/api/jobs", data=base | {"strokes": json.dumps(nudged)}).json()["cached"] is True
    other = [{"points": [[0.36, 0.53], [0.6, 0.3]], "width": 0.09}]
    r = client.post("/api/jobs", data=base | {"strokes": json.dumps(other)})
    assert r.status_code == 400 and "made ahead" in r.text  # a drawing of your own needs the GPU, and says so


def test_your_own_clip_says_what_it_is_and_the_noise_does_to_it_what_it_would_to_a_sample(scene_client, tmp_path, make_video):
    client, r = scene_client
    assert any(o["id"] == "candle" for o in client.get("/api/objects").json())
    plate = make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40))
    with open(plate, "rb") as p:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                        data={"effect": "auto", "mode": "scene", "cue": "hold", "holds": json.dumps([[1.0, 1.15]]),
                              "object": "candle", "x": "0.5", "y": "0.5"}).json()
    assert s["effect"] == "blow_out" and s["gesture"] == "hit"  # a short press on a candle blows it out
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert "the candle" in r.prompts[-1]  # and the model is told what it is
    with open(plate, "rb") as p:
        bad = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                          data={"effect": "auto", "mode": "scene", "cue": "hold", "object": "volcano"})
    assert bad.status_code == 400


def test_on_a_hand_the_sound_picks_the_effect(scene_client, tmp_path, make_video, make_wav):
    client, _ = scene_client
    plate = make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40 + (i % 2)))
    voice = make_voice(tmp_path, make_wav)  # a voiced burst: a boom
    with open(plate, "rb") as p, open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                        data={"effect": "auto", "mode": "scene", "x": "0.5", "y": "0.5", "object": "hand"}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    auto = done["receipt"]["auto"]
    assert done["heard"] == auto["heard"] == "boom" and auto["picked"] == done["effect"] == "fire_burst"
    assert done["receipt"]["inputs"]["subject"] == "the outstretched hand" and "carved" not in done["receipt"]["render"]["prompt"]


def test_a_sample_noise_on_a_hand_is_read_by_its_sound(scene_client, tmp_path, make_video):
    client, _ = scene_client
    plate = make_video(tmp_path, n=48, fps=24, w=128, h=256, color_fn=lambda i: (60, 50, 40 + (i % 2)))
    voices = client.get("/api/samples").json()["voices"]
    pf = next((v["name"] for v in voices if v["name"].startswith("pf.")), None)
    if pf is None:
        return
    with open(plate, "rb") as p:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                        data={"effect": "auto", "mode": "scene", "x": "0.5", "y": "0.5", "object": "hand",
                              "voice_sample": pf}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["auto"]["heard"] == "puff" and done["effect"] == "smoke_puff"  # not the default fireball


def test_together_needs_shots_cropped_alike_and_merges_one_spot(scene_client, tmp_path, make_video, make_wav):
    client, r = scene_client
    plate = make_video(tmp_path, n=48, fps=24, w=320, h=240, color_fn=lambda i: (60, 50, 40))  # 4:3: cropped to 9:16 here
    voice = make_voice(tmp_path, make_wav)

    def shot(x, y):
        with open(plate, "rb") as p, open(voice, "rb") as v:
            s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                            data={"effect": "fire_burst", "mode": "scene", "x": str(x), "y": str(y)}).json()
        assert wait(client, s["id"])["status"] == "done"
        return s["id"]

    top, bottom, again = shot(0.15, 0.5), shot(0.85, 0.5), shot(0.15, 0.5)  # left and right of a clip cropped narrower
    holds = json.dumps([[0.5, 0.65], [1.1, 1.25]])
    far = client.post(f"/api/jobs/{top}/replay", data={"cue": "hold", "holds": holds, "together": json.dumps([bottom])})
    assert far.status_code == 409  # cropped differently from the clip
    assert client.post(f"/api/jobs/{top}/replay", data={"cue": "hold", "holds": holds, "together": '"abc"'}).status_code == 400
    same = client.post(f"/api/jobs/{top}/replay", data={"cue": "hold", "holds": holds, "together": json.dumps([again])}).json()
    done = wait(client, same["id"])
    assert done["status"] == "done", done.get("error")
    assert len({h["source"] for h in done["receipt"]["hits"]}) == 1  # the same spot is one source: they hand over


# ---------------------------------------------------------------- element bending: your own hand in the shot


class FakeElementRenderer:
    """Renders 'the element over black' as its own guide (cuesync.bending's renders), with the GPU answering."""

    name, scenes = "fake-element", True

    def __init__(self):
        self.prompts = []

    def ready(self):
        return True

    def __call__(self, guide_path, prompt, seed, out_path, image=None):
        from cuesync.media import read_video

        self.prompts.append(prompt)
        write_video(out_path, read_video(guide_path).frames, fps=24)
        return 0.1


def test_a_shot_from_your_hand_is_tracked_carried_and_performed_again(tmp_path, make_video, make_wav, monkeypatch):
    import cuesync.bending as bending

    seen = {}

    def fake_prepare(plate, out_dir, tap, frame, emitter, say=None):  # the tracker's subprocess: a still hand at the tap
        seen.update(tap=tap, frame=frame, emitter=emitter, calls=seen.get("calls", 0) + 1)
        t = {"version": 1, "fps": 24, "W": 256, "H": 128, "emitter": emitter, "hand": "right", "xy": [list(tap)] * 72,
             "dir": [[0.0, -1.0]] * 72, "scale": [20.0] * 72, "conf": [1.0] * 72, "src": ["mp"] * 72,
             "speed": [0.0] * 72, "landmarks": [None] * 72, "gesture": ["Open_Palm"] * 72, "events": []}
        (out_dir / "hand_track.json").write_text(json.dumps(t))
        return out_dir / "hand_track.json", None

    monkeypatch.setattr(bending, "prepare_hand", fake_prepare)
    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    r = FakeElementRenderer()
    client = TestClient(create_app(r, tmp_path / "jobs", shot_defaults={"plate_size": (256, 128), "max_plate_s": 3.0}))
    assert "bending" in client.get("/api/health").json()["modes"]
    plate = make_video(tmp_path, n=72, fps=24, w=256, h=128, color_fn=lambda i: (40, 35, 30))
    voice = make_voice(tmp_path, make_wav)

    def post(**data):
        with open(plate, "rb") as p, open(voice, "rb") as v:
            return client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                               data={"mode": "bending", "x": "0.4", "y": "0.6", **data})

    assert post(effect="fireball").status_code == 400  # not one a hand throws
    assert post(effect="fire_burst", frame="-1").status_code == 400
    s = post(effect="fire_burst", frame="12", direction="90").json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    rc = done["receipt"]
    assert rc["mode"] == "bending" and seen["frame"] == 12 and seen["emitter"] == "palm"
    assert abs(seen["tap"][0] - 0.4 * 256) < 1 and len(r.prompts) == 1 and "to the right" in r.prompts[0]  # the arrow
    again = client.post(f"/api/jobs/{s['id']}/replay", data={"cue": "hold", "holds": json.dumps([[0.5, 0.7], [1.5, 1.8]])})
    done = wait(client, again.json()["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["performed_again"]["no_new_render"] and len(r.prompts) == 1
    assert len(done["receipt"]["bending"]["hits"]) == 2
    other = post(effect="fire_breath", frame="12", reuse=s["id"]).json()  # the badge: another effect, same take
    done = wait(client, other["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["effect"]["id"] == "fire_breath" and len(r.prompts) == 2 and seen["calls"] == 1  # not tracked again


def test_your_tap_is_followed_before_the_shot_and_the_shot_reuses_it(tmp_path, make_video, make_wav, monkeypatch):
    """The studio draws your hand's path under your aim before anything is made: /api/hands/prepare follows the tapped
    hand (its track served as soon as it is followed), and a shot sent with reuse=<that job> doesn't follow it again."""
    import cuesync.bending as bending

    seen = {"calls": 0}

    def fake_track(plate, out_dir, tap, frame, emitter):
        seen.update(tap=tap, frame=frame, calls=seen["calls"] + 1)
        t = {"version": 1, "fps": 24, "W": 256, "H": 128, "emitter": emitter, "hand": "right", "xy": [list(tap)] * 72,
             "dir": [[0.0, -1.0]] * 72, "scale": [20.0] * 72, "conf": [1.0] * 72, "src": ["mp"] * 60 + ["lost"] * 12,
             "speed": [0.0] * 72, "landmarks": [None] * 72, "gesture": ["Open_Palm"] * 72, "events": []}
        (out_dir / "hand_track.json").write_text(json.dumps(t))
        return out_dir / "hand_track.json"

    monkeypatch.setattr(bending, "track_hand", fake_track)
    monkeypatch.setattr(bending, "hand_python", lambda kind: "no-such-python")  # (no matte: the shot holds the root out)
    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    r = FakeElementRenderer()
    client = TestClient(create_app(r, tmp_path / "jobs", shot_defaults={"plate_size": (256, 128), "max_plate_s": 3.0}))
    plate = make_video(tmp_path, n=72, fps=24, w=256, h=128, color_fn=lambda i: (40, 35, 30))
    with open(plate, "rb") as p:
        assert client.post("/api/hands/prepare", files={"plate": ("plate.mp4", p, "video/mp4")},
                           data={"x": "1.4", "y": "0.6"}).status_code == 400
    with open(plate, "rb") as p:
        prep = client.post("/api/hands/prepare", files={"plate": ("plate.mp4", p, "video/mp4")},
                           data={"x": "0.4", "y": "0.6", "frame": "12"}).json()
    done = wait(client, prep["id"])
    assert done["status"] == "done", done.get("error")
    track = client.get(done["files"]["hand_track.json"]).json()
    assert track["src"].count("lost") == 12 and seen["frame"] == 12 and abs(seen["tap"][0] - 0.4 * 256) < 1
    voice = make_voice(tmp_path, make_wav)
    with open(plate, "rb") as p, open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                        data={"mode": "bending", "x": "0.4", "y": "0.6", "frame": "12", "effect": "fire_burst",
                              "reuse": prep["id"]}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert seen["calls"] == 1 and done["receipt"]["bending"]["tap_frame"] == 12  # followed once, for the preview


def test_a_sample_hand_plays_your_take_on_its_element_with_the_gpu_offline(tmp_path, make_video, make_wav):
    from cuesync.bending import make_bending_shot
    from cuesync.pipeline import ShotSpec
    from cuesync.precompute import cache_key

    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    plate = make_video(samples / "plates", n=72, fps=24, w=256, h=128, color_fn=lambda i: (40, 35, 30))
    plate = plate.rename(samples / "plates" / "hand.mp4")
    (samples / "samples.json").write_text(json.dumps({"voices": {"boom.wav": {}}, "plates": {"hand.mp4": {
        "title": "Your hand", "point": [0.4, 0.6], "hand": {"frame": 10}}}}))
    shot = samples / "made_ahead" / "bending" / "hand__boom"
    shot.mkdir(parents=True)
    t = {"version": 1, "fps": 24, "W": 256, "H": 128, "emitter": "palm", "hand": "right", "xy": [[102.0, 77.0]] * 72,
         "dir": [[0.0, -1.0]] * 72, "scale": [20.0] * 72, "conf": [1.0] * 72, "src": ["mp"] * 72, "speed": [0.0] * 72,
         "landmarks": [None] * 72, "gesture": ["Open_Palm"] * 72, "events": []}
    (shot / "hand_track.json").write_text(json.dumps(t))
    voice = make_voice(tmp_path, make_wav)
    r = FakeElementRenderer()
    make_bending_shot(ShotSpec(plate=plate, voice=voice, effect="fire_burst", point=(0.4, 0.6), plate_size=(256, 128),
                               max_plate_s=3.0), r, shot, shot / "hand_track.json")
    key = cache_key("bending", "hand.mp4", "", "fire_burst", (0.4, 0.6))
    (shot.parent / "index.json").write_text(json.dumps({key: {"dir": shot.name, "title": "a fireball in your palm",
        "plate": "hand.mp4", "voice": "", "effect": "fire_burst", "mode": "bending", "seed": 1, "tap_frame": 10}}))

    class Offline(FakeElementRenderer):
        def ready(self):
            return False

    client = TestClient(create_app(Offline(), tmp_path / "jobs", samples_dir=samples, precomputed=samples / "made_ahead",
                                   shot_defaults={"plate_size": (256, 128), "max_plate_s": 3.0}))
    assert "bending" in client.get("/api/health").json()["made_ahead"]
    with open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"voice": ("v.wav", v, "audio/wav")},
                        data={"mode": "bending", "plate_sample": "hand.mp4", "effect": "auto", "x": "0.4", "y": "0.6"}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["performed_again"]["no_new_render"] and done["mode"] == "replay"
    shutil.copy(voice, samples / "voices" / "boom.wav")  # an element made here, on the sample hand with a sample noise
    s = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "boom.wav",
                                       "effect": "air_blast", "x": "0.4", "y": "0.6"})
    assert s.status_code == 200, s.text  # (its made-ahead lookup once asked for a definition it has none of)
    done = wait(client, s.json()["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["effect"]["id"] == "air_blast" and done["receipt"]["effect"]["made"] == "air"
    buzz = Path(__file__).resolve().parents[2] / "samples" / "voices" / "zzzzz-tsst.wav"  # your noise decides: a buzz
    with open(buzz, "rb") as v:  # over the sample's open palm is ball lightning, made on its hand (not its fire again)
        s = client.post("/api/jobs", files={"voice": ("z.wav", v, "audio/wav")},
                        data={"mode": "bending", "plate_sample": "hand.mp4", "effect": "auto", "x": "0.4", "y": "0.6"}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["effect"]["id"] == "lightning_ball"
    with open(voice, "rb") as v:  # a hand clip of your own still needs the GPU
        with open(plate, "rb") as p:
            assert client.post("/api/jobs", files={"voice": ("v.wav", v, "audio/wav"), "plate": ("p.mp4", p, "video/mp4")},
                               data={"mode": "bending", "effect": "auto", "x": "0.4", "y": "0.6"}).status_code in (400, 503)


@pytest.fixture
def sample_hand(tmp_path, make_video, make_wav):
    from cuesync.bending import make_bending_shot
    from cuesync.pipeline import ShotSpec
    from cuesync.precompute import cache_key

    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    plate = make_video(samples / "plates", n=72, fps=24, w=256, h=128, color_fn=lambda i: (40, 35, 30))
    plate = plate.rename(samples / "plates" / "hand.mp4")
    voice = samples / "voices" / "pf.wav"
    shutil.copyfile(make_voice(tmp_path, make_wav), voice)
    (samples / "samples.json").write_text(json.dumps({"voices": {"pf.wav": {}}, "plates": {"hand.mp4": {
        "point": [0.4, 0.6], "hand": {"frame": 10, "strike": 36}}}}))
    shot = samples / "made_ahead" / "bending" / "hand__pf"
    shot.mkdir(parents=True)
    track = {"version": 1, "fps": 24, "W": 256, "H": 128, "emitter": "palm", "hand": "right",
             "xy": [[102.4, 76.8]] * 72, "dir": [[0.0, -1.0]] * 72, "scale": [20.0] * 72,
             "conf": [1.0] * 72, "src": ["mp"] * 72, "speed": [0.0] * 72, "landmarks": [None] * 72,
             "gesture": ["Open_Palm"] * 72, "events": []}
    (shot / "hand_track.json").write_text(json.dumps(track))
    make_bending_shot(ShotSpec(plate=plate, voice=voice, effect="fire_burst", point=(0.4, 0.6), offset_s=0.25,
                               plate_size=(256, 128), max_plate_s=3.0), FakeElementRenderer(), shot,
                      shot / "hand_track.json", tap_frame=10)
    key = cache_key("bending", "hand.mp4", "pf.wav", "fire_burst", (0.4, 0.6))
    (shot.parent / "index.json").write_text(json.dumps({key: {"dir": shot.name, "title": "Your hand",
        "plate": "hand.mp4", "voice": "pf.wav", "effect": "fire_burst", "mode": "bending", "seed": 1,
        "tap_frame": 10}}))
    return samples


def hand_client(samples, jobs, renderer=None):
    return TestClient(create_app(renderer, jobs, samples_dir=samples, precomputed=samples / "made_ahead",
                                 shot_defaults={"plate_size": (256, 128), "max_plate_s": 3.0}))


def test_explicit_sample_reuse_ignores_ready_gpu_and_tracker_and_can_replay(sample_hand, tmp_path, monkeypatch):
    import cuesync.bending as bending

    renderer = FakeElementRenderer()
    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    monkeypatch.setattr(renderer, "ready", lambda: pytest.fail("reuse must not probe or use the GPU"))
    monkeypatch.setattr(bending, "prepare_hand", lambda *a, **kw: pytest.fail("the prepared hand must be reused"))
    client = hand_client(sample_hand, tmp_path / "jobs", renderer)
    with (sample_hand / "voices" / "pf.wav").open("rb") as voice:
        first = client.post("/api/jobs", data={"mode": "bending", "performance_mode": "reuse",
                            "plate_sample": "hand.mp4", "effect": "fire_burst"},
                            files={"voice": ("microphone.wav", voice, "audio/wav")})
    assert first.status_code == 200, first.text
    assert first.json()["mode"] == "replay" and not first.json()["cached"]
    done = wait(client, first.json()["id"])
    assert done["status"] == "done", done
    assert done["receipt"]["inputs"]["point"] == [0.4, 0.6]
    assert done["receipt"]["bending"]["tap_frame"] == 10
    assert done["receipt"]["bending"]["performed_again"]["no_new_render"]
    again = client.post(f"/api/jobs/{done['id']}/replay", data={"voice_sample": "pf.wav"})
    assert again.status_code == 200, again.text
    twice = wait(client, again.json()["id"])
    assert twice["status"] == "done", twice
    assert twice["receipt"]["bending"]["performed_again"]["no_new_render"]
    assert not renderer.prompts


def test_a_library_element_asked_for_on_a_sample_hand_is_carried_on_its_track_with_no_gpu_or_tracker(sample_hand, tmp_path,
                                                                                                     monkeypatch):
    import cuesync.bending as bending

    lib = tmp_path / "library"
    (lib / "air_vortex").mkdir(parents=True)
    from cuesync.effects import load_effect

    shot = sample_hand / "made_ahead" / "bending" / "hand__pf"  # a stored element, rendered ahead (here: the sample's own
    shutil.copyfile(shot / "element.mp4", lib / "air_vortex" / "element.mp4")  # render, named as the library's air)
    rc = json.loads((shot / "receipt.json").read_text()) | {"effect": load_effect("air_vortex").to_json()}
    (lib / "air_vortex" / "receipt.json").write_text(json.dumps(rc))
    monkeypatch.setattr(bending, "library_dir", lambda: lib)
    monkeypatch.setattr(bending, "helpers_ready", lambda: False)
    monkeypatch.setattr(bending, "prepare_hand", lambda *a, **kw: pytest.fail("the sample's own track must be reused"))
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "air_vortex", "x": "0.4", "y": "0.6"})
    assert first.status_code == 200, first.text
    assert first.json()["mode"] == "replay"
    done = wait(client, first.json()["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["effect"]["id"] == "air_vortex"
    assert Path(done["receipt"]["bending"]["instant"]["library"]).name == "air_vortex"  # the library's air, not the fire


def test_explicit_reuse_composes_even_an_exact_cached_sample_voice(sample_hand, tmp_path):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "performance_mode": "reuse",
                         "plate_sample": "hand.mp4", "effect": "fire_burst", "voice_sample": "pf.wav",
                         "x": "0.4", "y": "0.6"})
    assert first.status_code == 200, first.text
    assert not first.json()["cached"] and first.json()["mode"] == "replay"
    assert wait(client, first.json()["id"])["status"] == "done"


@pytest.mark.parametrize("changes", [
    {"performance_mode": "silent-gpu-fallback"}, {"mode": "scene"}, {"effect": "auto"},
    {"effect": "flame_plume"}, {"plate_sample": "missing.mp4"}, {"reuse": "another-job"},
])
def test_explicit_reuse_rejects_invalid_requests_without_fallback(sample_hand, tmp_path, changes):
    renderer = FakeElementRenderer()
    client = hand_client(sample_hand, tmp_path / "jobs", renderer)
    first = client.post("/api/jobs", data={"mode": "bending", "performance_mode": "reuse",
                         "plate_sample": "hand.mp4", "effect": "fire_burst", "voice_sample": "pf.wav", **changes})
    assert first.status_code in (400, 409), first.text
    assert not renderer.prompts
    assert not list((tmp_path / "jobs").iterdir())


@pytest.mark.parametrize("missing", ["element.mp4", "hand_track.json", "receipt.json", "hand_mask.npz"])
def test_explicit_reuse_rejects_incomplete_prepared_resources(sample_hand, tmp_path, missing):
    shot = next((sample_hand / "made_ahead").rglob("receipt.json")).parent
    if missing == "hand_mask.npz":
        receipt = json.loads((shot / "receipt.json").read_text())
        receipt["bending"]["mask"] = "hand_mask.npz"
        (shot / "receipt.json").write_text(json.dumps(receipt))
    else:
        (shot / missing).unlink()
    renderer = FakeElementRenderer()
    client = hand_client(sample_hand, tmp_path / "jobs", renderer)
    first = client.post("/api/jobs", data={"mode": "bending", "performance_mode": "reuse",
                         "plate_sample": "hand.mp4", "effect": "fire_burst", "voice_sample": "pf.wav"})
    assert first.status_code == 409, first.text
    assert "prepared" in first.json()["detail"].lower()
    assert not renderer.prompts


def test_result_file_mime_types_and_raw_scene_download(tmp_path):
    shot = tmp_path / "pre" / "scene" / "sample"
    shot.mkdir(parents=True)
    for name in ("final.mp4", "render.mp4", "poster.jpg", "receipt.json"):
        (shot / name).write_bytes(b"{}" if name.endswith(".json") else b"result bytes")
    (shot.parent / "index.json").write_text(json.dumps({"sample": {
        "dir": "sample", "title": "Sample", "plate": "sample.mp4", "mode": "scene"}}))
    client = TestClient(create_app(None, tmp_path / "jobs", precomputed=tmp_path / "pre"))
    job = client.get("/api/gallery").json()[0]
    for name, media_type in (("render.mp4", "video/mp4"), ("final.mp4", "video/mp4"),
                             ("poster.jpg", "image/jpeg"), ("receipt.json", "application/json")):
        assert name in job["files"]
        result = client.get(job["files"][name])
        assert result.status_code == 200
        assert result.headers["content-type"] == media_type
    assert client.get(f"/api/jobs/{job['id']}/files/secret.txt").status_code == 404


def test_raw_scene_render_stays_private_until_job_finishes(instant_client, monkeypatch):
    from cuesync.server import JobQueue

    def hold(queue, job):
        queue.jobs[job.id] = job
        (job.dir / "render.mp4").write_bytes(b"partial video")
        return job

    monkeypatch.setattr(JobQueue, "submit", hold)
    job = instant_client.post("/api/jobs", data={"mode": "instant", "plate_sample": "mug.mp4",
                              "voice_sample": "pf.wav", "effect": "fireball"}).json()
    assert "render.mp4" not in job["files"]
    assert instant_client.get(f"/api/jobs/{job['id']}/files/render.mp4").status_code == 404


def test_job_public_keeps_terminal_status_files_and_receipt_consistent(tmp_path, monkeypatch):
    from cuesync.server import Job

    (tmp_path / "final.mp4").write_bytes(b"finished result")
    (tmp_path / "cue.json").write_text("{}")
    (tmp_path / "receipt.json").write_text(json.dumps({"seconds_total": 1.0}))
    job = Job(id="transition", dir=tmp_path, spec=None, status="running")
    exists = Path.exists

    def finish_during_snapshot(path):
        if path == tmp_path / "cue.json":
            job.status = "done"
        return exists(path)

    monkeypatch.setattr(Path, "exists", finish_during_snapshot)
    first = job.public()
    assert first["status"] == "running"
    assert "final.mp4" not in first["files"] and first["receipt"] is None
    second = job.public()
    assert second["status"] == "done"
    assert "final.mp4" in second["files"] and second["receipt"] == {"seconds_total": 1.0}


def test_a_cached_sample_hand_can_be_performed_again_on_its_strike(sample_hand, tmp_path):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "fire_burst", "x": "0.4", "y": "0.6"}).json()
    assert first["cached"] and first["status"] == "done"
    again = client.post(f"/api/jobs/{first['id']}/replay", data={"voice_sample": "pf.wav"}).json()
    done = wait(client, again["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["tap_frame"] == 10
    assert done["receipt"]["bending"]["k_launch"] == 36


def test_a_samples_authored_aim_does_not_prevent_its_default_cache_hit(sample_hand, tmp_path):
    receipt = next((sample_hand / "made_ahead").rglob("receipt.json"))
    rc = json.loads(receipt.read_text())
    rc["bending"]["aim"] = 0.0
    receipt.write_text(json.dumps(rc))
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "fire_burst", "x": "0.4", "y": "0.6"}).json()
    assert first["cached"] and first["status"] == "done"


def test_a_sample_voice_replay_keeps_the_first_shots_strike(sample_hand, tmp_path):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "auto", "x": "0.4", "y": "0.6", "offset_s": "0.5"}).json()
    done = wait(client, first["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["k_launch"] == 36
    again = client.post(f"/api/jobs/{first['id']}/replay", data={"voice_sample": "pf.wav"}).json()
    twice = wait(client, again["id"])
    assert twice["status"] == "done", twice.get("error")
    assert twice["receipt"]["bending"]["k_launch"] == 36
    assert abs(twice["receipt"]["inputs"]["offset_s"] - done["receipt"]["inputs"]["offset_s"]) < 1 / 24


def test_a_sample_noise_uses_the_strike_instead_of_the_tap_frame(sample_hand, tmp_path):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "auto", "x": "0.4", "y": "0.6"}).json()
    done = wait(client, first["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["k_launch"] == 36
    assert done["receipt"]["bending"]["tap_frame"] == 10


def test_a_hold_replay_uses_its_holds_when_a_sample_voice_is_also_sent(sample_hand, tmp_path):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "fire_burst", "x": "0.4", "y": "0.6"}).json()
    again = client.post(f"/api/jobs/{first['id']}/replay", data={"cue": "hold", "holds": "[[1.5, 1.7]]",
                                                               "voice_sample": "pf.wav"}).json()
    done = wait(client, again["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["k_launch"] == 36


def test_a_sample_hand_uses_its_shipped_track_when_only_the_gpu_is_ready(sample_hand, tmp_path):
    renderer = FakeElementRenderer()
    client = hand_client(sample_hand, tmp_path / "jobs", renderer)
    assert "bending" not in client.get("/api/health").json()["modes"]
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "auto", "x": "0.4", "y": "0.6"}).json()
    assert first["mode"] == "replay"
    done = wait(client, first["id"])
    assert done["status"] == "done", done.get("error")
    assert not renderer.prompts


@pytest.mark.parametrize("changes,aim", [
    ({"frame": "30"}, None),  # no arrow: the throw follows the tracked hand
    ({"direction": "90"}, 90.0),
])
def test_a_custom_hand_frame_or_aim_does_not_use_the_sample_cache(sample_hand, tmp_path, changes, aim):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "fire_burst", "x": "0.4", "y": "0.6", **changes}).json()
    assert not first["cached"]
    done = wait(client, first["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["bending"]["tap_frame"] == int(changes.get("frame", 10))
    assert done["receipt"]["bending"]["aim"] == aim


def test_a_sample_hand_keeps_a_new_tap_point_and_frame(sample_hand, tmp_path):
    client = hand_client(sample_hand, tmp_path / "jobs")
    first = client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4", "voice_sample": "pf.wav",
                                            "effect": "fire_burst", "x": "0.45", "y": "0.35", "frame": "30",
                                            "direction": "90"}).json()
    done = wait(client, first["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["inputs"]["point"] == [0.45, 0.35]
    assert done["receipt"]["bending"]["tap_frame"] == 30


def test_your_own_hand_with_no_gpu_gets_an_element_rendered_ahead(tmp_path, make_video, make_wav, monkeypatch):
    import cuesync.bending as bending
    from cuesync.bending import make_bending_shot
    from cuesync.pipeline import ShotSpec

    w, h, n = 256, 128, 72
    plate = make_video(tmp_path, n=n, fps=24, w=w, h=h, color_fn=lambda i: (40, 35, 30))
    voice = make_voice(tmp_path, make_wav)
    t = {"version": 1, "fps": 24, "W": w, "H": h, "emitter": "palm", "hand": "right", "xy": [[102.0, 77.0]] * n,
         "dir": [[0.0, -1.0]] * n, "scale": [20.0] * n, "conf": [1.0] * n, "src": ["mp"] * n, "speed": [0.0] * n,
         "landmarks": [None] * n, "gesture": ["Open_Palm"] * n, "events": []}
    lib = tmp_path / "library"  # the library: one element rendered ahead of time (as samples/library/bending holds)
    (tmp_path / "t.json").write_text(json.dumps(t))
    make_bending_shot(ShotSpec(plate=plate, voice=voice, effect="fire_burst", point=(0.4, 0.6), plate_size=(w, h),
                               max_plate_s=3.0), FakeElementRenderer(), lib / "fire_burst", tmp_path / "t.json")

    def fake_prepare(plate_, out_dir, tap, frame, emitter, say=None):
        (out_dir / "hand_track.json").write_text(json.dumps(t))
        return out_dir / "hand_track.json", None

    monkeypatch.setattr(bending, "prepare_hand", fake_prepare)
    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    monkeypatch.setattr(bending, "library_dir", lambda: lib)
    client = TestClient(create_app(None, tmp_path / "jobs", instant=FakeRenderer(),
                                   shot_defaults={"plate_size": (w, h), "max_plate_s": 3.0}))
    assert "bending" in client.get("/api/health").json()["modes"]  # no GPU, and still a hand of your own
    with open(plate, "rb") as p, open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"plate": ("p.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                        data={"mode": "bending", "effect": "fire_burst", "x": "0.4", "y": "0.6", "frame": "10"}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    b = done["receipt"]["bending"]
    assert b["instant"]["no_gpu"] and b["performed_again"] is None and done["receipt"]["effect"]["id"] == "fire_burst"


def test_an_uploaded_hand_keeps_its_clip_cue_without_a_gpu(tmp_path, monkeypatch):
    """Your hand cued by your clip's own sound: the clip loaded for the hand keeps that sound (it was written silent,
    and the shot failed after the hand was tracked: "your clip has no sound to lock to")."""
    import cuesync.bending as bending
    from cuesync.pipeline import ShotResult, shot_cue

    plate = tmp_path / "phone.mp4"
    sound = np.zeros(2 * SR, np.float32)
    sound[SR // 2:SR] = 0.5 * np.sin(np.arange(SR // 2) * 0.06)
    write_video(plate, np.full((48, 128, 72, 3), 50, np.uint8), audio=sound)
    seen = {}

    def prepare(loaded, out_dir, tap, frame, emitter, say=None):
        seen.update(tap=tap, frame=frame)
        return out_dir / "hand_track.json", None

    def carry(spec, renderer, out_dir, track, mask, progress, tap_frame):
        cue = shot_cue(spec, 2.0)
        seen.update(source=cue.source, samples=cue.samples, renderer=renderer, direction=spec.direction)
        progress("done", 1.0, "Done")
        return ShotResult(out_dir, out_dir / "final.mp4", out_dir / "composite.mp4", None, {})

    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    monkeypatch.setattr(bending, "library_dir", lambda: tmp_path)
    monkeypatch.setattr(bending, "prepare_hand", prepare)
    monkeypatch.setattr(bending, "carry", carry)
    client = TestClient(create_app(None, tmp_path / "jobs", shot_defaults={"plate_size": (72, 128)}))
    with open(plate, "rb") as p:
        response = client.post("/api/jobs", files={"plate": ("phone.mp4", p, "video/mp4")}, data={
            "mode": "bending", "cue": "clip", "effect": "auto", "seed": "1", "direction": "0",
            "frame": "12", "x": "0.5", "y": "0.7"})
    assert response.status_code == 200, response.text
    done = wait(client, response.json()["id"])
    assert done["status"] == "done", done.get("error")
    assert seen["frame"] == 12 and seen["direction"] == 0 and seen["source"] == "clip"
    assert np.corrcoef(sound, seen["samples"][:len(sound)])[0, 1] > 0.99


def test_the_studio_is_served_beside_the_api_and_sets_up_the_hand_tracker_only_locally(tmp_path, monkeypatch):
    studio = tmp_path / "studio"
    studio.mkdir()
    (studio / "index.html").write_text("<h1>Cue the Boom</h1>")
    remote = TestClient(create_app(None, tmp_path / "jobs", instant=FakeRenderer(), studio=studio))
    assert "Cue the Boom" in remote.get("/").text and remote.get("/api/health").json()["ok"]  # one address, both
    assert remote.post("/api/hands/install").status_code == 403  # not served locally: no installing
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    local = TestClient(create_app(None, tmp_path / "jobs2", instant=FakeRenderer(), local=True))
    assert local.get("/api/health").json()["hand"]["can_install"]
    assert local.post("/api/hands/install").json()["running"]
    for _ in range(40):
        if calls:
            break
        time.sleep(0.05)
    assert calls and calls[0][-2:] == ["hands", "--install"]


def test_a_finished_shot_becomes_a_vertical_reel_to_share(tmp_path, make_video, make_wav, monkeypatch):
    import cuesync.bending as bending
    from cuesync.media import read_video

    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    t = {"version": 1, "fps": 24, "W": 256, "H": 128, "emitter": "palm", "hand": "right", "xy": [[102.0, 77.0]] * 72,
         "dir": [[0.0, -1.0]] * 72, "scale": [20.0] * 72, "conf": [1.0] * 72, "src": ["mp"] * 72, "speed": [0.0] * 72,
         "landmarks": [None] * 72, "gesture": ["Open_Palm"] * 72, "events": []}

    def fake_prepare(plate_, out_dir, tap, frame, emitter, say=None):
        (out_dir / "hand_track.json").write_text(json.dumps(t))
        return out_dir / "hand_track.json", None

    monkeypatch.setattr(bending, "prepare_hand", fake_prepare)
    client = TestClient(create_app(FakeElementRenderer(), tmp_path / "jobs",
                                   shot_defaults={"plate_size": (256, 128), "max_plate_s": 3.0}))
    plate = make_video(tmp_path, n=72, fps=24, w=256, h=128, color_fn=lambda i: (40, 35, 30))
    voice = make_voice(tmp_path, make_wav)
    with open(plate, "rb") as p, open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"plate": ("p.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                        data={"mode": "bending", "effect": "fire_burst", "x": "0.4", "y": "0.6"}).json()
    assert wait(client, s["id"])["status"] == "done"
    shared = client.post(f"/api/jobs/{s['id']}/share").json()
    reel = read_video(Path(str(tmp_path / "jobs" / s["id"] / "share.mp4"))).frames
    assert "share.mp4" in shared["files"] and reel.shape[1:3] == (1280, 720)  # 9:16, to post


def test_the_live_preview_gets_the_library_elements_with_their_poses(tmp_path):
    """Try it live plays the hand elements rendered ahead of time on your hand: each with its pose in its render and its
    frames (ignition, the end of its steady burn, its last lit frame), and the video itself."""
    client = TestClient(create_app(None, tmp_path / "jobs"))
    lib = client.get("/api/library").json()
    assert {e["effect"] for e in lib} >= {"fire_burst", "fire_breath", "flame_plume"}
    for e in lib:
        assert 0 <= e["ignition"] < e["release"] <= e["end"] and len(e["ref"]["xy"]) == 2 and e["ref"]["scale"] > 0
        assert client.get(e["url"]).status_code == 200
    assert client.get("/api/library/nope/element.mp4").status_code == 404


def test_a_clip_the_browser_cannot_play_gets_a_copy_it_can(instant_client, tmp_path, make_video):
    """An iPhone's HEVC stayed black in Firefox and couldn't be tapped: the studio asks for a copy any browser plays (H.264,
    the same timeline)."""
    from cuesync.media import probe_video

    clip = make_video(tmp_path, n=48, fps=24, w=320, h=240, color_fn=lambda i: (i * 5, 60, 60))
    with open(clip, "rb") as fh:
        r = instant_client.post("/api/preview", files={"plate": ("clip.mov", fh, "video/quicktime")})
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    out = tmp_path / "preview.mp4"
    out.write_bytes(r.content)
    info = probe_video(out)
    assert (info.width, info.height) == (320, 240) and abs(info.duration_s - 2.0) < 0.15
    bad = instant_client.post("/api/preview", files={"plate": ("x.mov", b"not a video", "video/quicktime")})
    assert bad.status_code == 400


def test_an_element_made_here_is_thrown_from_a_hand_with_no_render(tmp_path, make_video, make_wav, monkeypatch):
    """Lightning, water, air and the explosion (cuesync.bending.MADE) need no GPU and no library: a hand shot asking
    for one is made on the CPU worker, the renderer never called, and they are listed among the effects."""
    import cuesync.bending as bending
    from cuesync import explosion

    def fake_prepare(plate, out_dir, tap, frame, emitter, say=None):
        t = {"version": 1, "fps": 24, "W": 256, "H": 128, "emitter": emitter, "hand": "right", "xy": [list(tap)] * 72,
             "dir": [[0.0, -1.0]] * 72, "scale": [20.0] * 72, "conf": [1.0] * 72, "src": ["mp"] * 72,
             "speed": [0.0] * 72, "landmarks": [None] * 72, "gesture": ["Closed_Fist"] * 72, "events": []}
        (out_dir / "hand_track.json").write_text(json.dumps(t))
        return out_dir / "hand_track.json", None

    el = np.full((30, 128, 128, 3), 4, np.uint8)
    yy, xx = np.mgrid[:128, :128]
    for k in range(4, 30):
        el[k][(xx - 64) ** 2 + (yy - 64) ** 2 <= min(36, 4 + 5 * (k - 4)) ** 2] = (255, 160, 40)
    monkeypatch.setattr(explosion, "library_element", lambda: el)
    monkeypatch.setattr(bending, "prepare_hand", fake_prepare)
    monkeypatch.setattr(bending, "helpers_ready", lambda: True)
    r = FakeElementRenderer()
    client = TestClient(create_app(r, tmp_path / "jobs", shot_defaults={"plate_size": (256, 128), "max_plate_s": 3.0}))
    made = {e["id"] for e in client.get("/api/effects").json() if e.get("made")}
    assert {"explosion", "lightning_arc", "water_orb", "air_blast"} <= made
    plate = make_video(tmp_path, n=72, fps=24, w=256, h=128, color_fn=lambda i: (60, 55, 50))
    voice = make_voice(tmp_path, make_wav)
    with open(plate, "rb") as p, open(voice, "rb") as v:
        s = client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4"), "voice": ("v.wav", v, "audio/wav")},
                        data={"mode": "bending", "x": "0.5", "y": "0.6", "frame": "10", "effect": "explosion"}).json()
    done = wait(client, s["id"])
    assert done["status"] == "done", done.get("error")
    assert done["receipt"]["effect"]["made"] == "explosion" and not r.prompts  # never rendered
    with open(plate, "rb") as p:
        assert client.post("/api/jobs", files={"plate": ("plate.mp4", p, "video/mp4")},
                           data={"mode": "render", "effect": "explosion", "cue": "clip"}).status_code == 400
