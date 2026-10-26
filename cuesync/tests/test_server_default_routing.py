"""Default voice choices stay curated; explicit CPU experiments remain available."""

import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cuesync.server import JobQueue, create_app  # noqa: E402


EXPERIMENTAL_EFFECTS = (
    "water_orb", "water_whip", "water_surge", "water_ribbon", "water_shield", "water_jet",
    "air_ball", "air_blast", "earth_rock", "earth_shatter", "earth_volley", "earth_lance",
)


@pytest.fixture
def prepared_hand_client(tmp_path, monkeypatch):
    import cuesync.bending as bending

    # Stop before rendering: the API's selected effect is the contract under test.
    def hold(queue, job):
        queue.jobs[job.id] = job
        return job

    monkeypatch.setattr(JobQueue, "submit", hold)
    monkeypatch.setattr(bending, "helpers_ready", lambda: False)
    samples = tmp_path / "samples"
    (samples / "plates").mkdir(parents=True)
    (samples / "plates" / "hand.mp4").write_bytes(b"prepared plate")
    (samples / "samples.json").write_text(json.dumps({"plates": {"hand.mp4": {
        "point": [0.4, 0.6], "hand": {"frame": 0}}}}))
    shot = samples / "made_ahead" / "bending" / "hand"
    shot.mkdir(parents=True)
    (shot / "final.mp4").write_bytes(b"prepared result")
    (shot / "hand_track.json").write_text(json.dumps({"version": 1, "W": 100, "H": 100,
                                                     "xy": [[40.0, 60.0]], "dir": [[0.0, -1.0]]}))
    (shot.parent / "index.json").write_text(json.dumps({"hand": {
        "dir": "hand", "title": "Fire from your palm", "plate": "hand.mp4",
        "mode": "bending", "effect": "fire_burst"}}))
    return TestClient(create_app(None, tmp_path / "jobs", samples_dir=samples,
                                 precomputed=samples / "made_ahead"))


def submit_voice(client, effect):
    return client.post("/api/jobs", data={"mode": "bending", "plate_sample": "hand.mp4",
                                         "effect": effect, "x": "0.4", "y": "0.6"},
                       files={"voice": ("mic.wav", b"recorded voice", "audio/wav")})


@pytest.mark.parametrize("picked", EXPERIMENTAL_EFFECTS)
def test_auto_for_a_prepared_hand_does_not_select_experimental_cpu_effects(prepared_hand_client, monkeypatch, picked):
    import cuesync.bending as bending

    monkeypatch.setattr(bending, "instant_effect", lambda *args: (picked, {"picked": picked}))
    response = submit_voice(prepared_hand_client, "auto")

    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "replay"
    assert response.json()["effect"] == "auto"  # keep the prepared fire performance


@pytest.mark.parametrize("picked", ("explosion", "lightning_arc", "lightning_ball"))
def test_auto_for_a_prepared_hand_keeps_curated_cpu_effects(prepared_hand_client, monkeypatch, picked):
    import cuesync.bending as bending

    monkeypatch.setattr(bending, "instant_effect", lambda *args: (picked, {"picked": picked}))
    response = submit_voice(prepared_hand_client, "auto")

    assert response.status_code == 200, response.text
    assert response.json()["effect"] == picked


@pytest.mark.parametrize("effect", EXPERIMENTAL_EFFECTS)
def test_a_prepared_hand_accepts_explicit_experimental_cpu_effects(prepared_hand_client, effect):
    response = submit_voice(prepared_hand_client, effect)

    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "replay"
    assert response.json()["effect"] == effect
