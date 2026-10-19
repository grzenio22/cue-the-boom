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
        seen.update(tap=tap, frame=frame, emitter=emitter)
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


def test_a_sample_hand_plays_your_take_on_its_element_with_the_gpu_offline(tmp_path, make_video, make_wav):
    from cuesync.bending import make_bending_shot
    from cuesync.pipeline import ShotSpec
    from cuesync.precompute import cache_key

    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "voices").mkdir()
    plate = make_video(samples / "plates", n=72, fps=24, w=256, h=128, color_fn=lambda i: (40, 35, 30))
    plate = plate.rename(samples / "plates" / "hand.mp4")
    (samples / "samples.json").write_text(json.dumps({"voices": {}, "plates": {"hand.mp4": {
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
    with open(voice, "rb") as v:  # a hand clip of your own still needs the GPU
        with open(plate, "rb") as p:
            assert client.post("/api/jobs", files={"voice": ("v.wav", v, "audio/wav"), "plate": ("p.mp4", p, "video/mp4")},
                               data={"mode": "bending", "effect": "auto", "x": "0.4", "y": "0.6"}).status_code in (400, 503)
