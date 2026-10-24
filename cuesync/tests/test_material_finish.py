"""A GPU material pass must not repaint the performer's hand or invent cue frames."""
import numpy as np
import pytest
import json
import time

from cuesync.material_finish import prompt_for, replace_material, safe_material_coverage, stone_surface_detail


def test_surface_detail_cannot_turn_solid_geometry_into_generated_background():
    base = np.full((120, 80, 3), (100, 95, 90), np.uint8)
    generated = np.full_like(base, 170)
    generated[20:100, 20:60] = 12  # a model-invented hole through the boulder
    generated[50:52] = 250  # fine surface information may survive, with bounded contrast
    detail = stone_surface_detail(base, generated)
    ratio = detail.astype(float) / base
    assert ratio.min() >= .77 and ratio.max() <= 1.23
    assert np.abs(detail[70:90, 30:50].astype(int) - base[70:90, 30:50]).max() <= 2
    assert np.any(detail[49:54] != base[49:54])


def test_uniform_model_relighting_cannot_replace_the_original_stone_shading():
    base = np.arange(20, 140, dtype=np.uint8)[:, None, None] * np.ones((1, 80, 3), np.uint8)
    for tone in (15, 240):
        assert np.array_equal(stone_surface_detail(base, np.full_like(base, tone)), base)


def test_material_replacement_preserves_every_uncovered_pixel_and_empty_frame():
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, (3, 24, 16, 3), dtype=np.uint8)
    generated = 255 - base
    mask = np.zeros(base.shape[:-1], np.uint8)
    mask[1:, 8:16, 5:12] = 255
    mask[1:, 7, 5:12] = 128
    out = replace_material(base, generated, mask)
    assert np.array_equal(out[mask == 0], base[mask == 0])
    assert np.array_equal(out[mask == 255], generated[mask == 255])
    assert np.max(np.abs(out[mask == 128].astype(float) - 127.5)) < 1.6


def test_only_the_validated_stone_materials_are_offered():
    assert "granite" in prompt_for("earth_rock")
    assert "fragments" in prompt_for("earth_shatter")
    with pytest.raises(ValueError):
        prompt_for("water_orb")


def test_material_replacement_rejects_a_misaligned_mask():
    base = np.zeros((2, 12, 8, 3), np.uint8)
    with pytest.raises(ValueError):
        replace_material(base, base, np.zeros((1, 12, 8), np.uint8))


def test_material_guard_rejects_changed_colours_and_preserves_translucent_release():
    base = np.full((12, 8, 3), (100, 95, 90), np.uint8)
    generated = np.full_like(base, (130, 124, 118))
    generated[:3] = (220, 80, 100)  # a hallucinated finger through opaque stone
    coverage = np.full((12, 8), 255, np.uint8)
    coverage[8:] = 180  # geometry fading after release
    safe = safe_material_coverage(base, generated, coverage)
    assert not safe[:3].any() and not safe[8:].any()
    assert safe[3:8].min() == 255


@pytest.fixture
def stone_source(tmp_path):
    from cuesync.media import write_video

    src = tmp_path / "stone"
    src.mkdir()
    frames = np.full((10, 64, 32, 3), 80, np.uint8)
    frames[3:, 20:32, 10:22] = (120, 110, 100)
    audio = .15 * np.sin(np.arange(20000) * .04).astype(np.float32)
    for name in ("composite.mp4", "final.mp4"):
        write_video(src / name, frames, fps=24, audio=audio)
    mask = np.zeros(frames.shape[:-1], np.uint8)
    mask[3:, 20:32, 10:22] = 255
    np.savez_compressed(src / "effect_mask.npz", alpha=mask)
    (src / "receipt.json").write_text(json.dumps({"mode": "bending", "effect": {"id": "earth_rock"},
        "inputs": {"seed": 17}, "render": {"seconds": 0}, "seconds_total": 1,
        "bending": {"holds": [[3, 9]], "hits": [{"frame": 3}], "performed_again": None}}))
    return src


class MaterialRenderer:
    name = "fake"
    scenes = True
    calls = 0

    def ready(self):
        return True

    def refine(self, guide, prompt, seed, out, *, image):
        from cuesync.media import read_video, write_video

        self.calls += 1
        clip = read_video(guide)
        assert len(clip.frames) == 81  # the reviewed 8k+1 temporal context, even for short takes
        assert image.exists()
        # Deliberately change the entire generated frame: only the stone may survive.
        write_video(out, np.full_like(clip.frames, (140, 130, 120)), fps=24)
        return .1


def test_material_finish_restores_duration_sound_and_writes_truthful_provenance(stone_source, tmp_path):
    from cuesync.material_finish import finish_shot
    from cuesync.media import read_video, read_audio

    renderer = MaterialRenderer()
    out = tmp_path / "finished"
    receipt = finish_shot(stone_source, renderer, out, parent_id="original")
    result, source = read_video(out / "composite.mp4"), read_video(stone_source / "composite.mp4")
    assert len(result.frames) == 10 and result.fps_nominal == 24
    assert np.abs(result.frames[:, :10].astype(float) - source.frames[:, :10]).max() <= 3
    assert np.corrcoef(read_audio(out / "composite.mp4")[:19000], read_audio(stone_source / "composite.mp4")[:19000])[0, 1] > .99
    assert receipt["material_finish"]["new_gpu_render"] is True
    assert receipt["material_finish"]["parent_id"] == "original"
    assert receipt["bending"]["hits"] == [{"frame": 3}]
    assert receipt["render"]["seconds"] == .1 and renderer.calls == 1
    assert json.loads((stone_source / "receipt.json").read_text())["render"]["seconds"] == 0


def material_client(monkeypatch, tmp_path, src, renderer):
    from fastapi.testclient import TestClient
    from cuesync.server import create_app
    import cuesync.precompute
    import cuesync.bending

    monkeypatch.setattr(cuesync.bending, "helpers_ready", lambda: False)
    monkeypatch.setattr(cuesync.precompute, "load_cache", lambda _: {
        "stone": {"mode": "bending", "path": str(src), "title": "stone", "plate": "hand.mp4", "effect": "earth_rock"}})
    client = TestClient(create_app(renderer, tmp_path / "jobs"))
    job = client.get("/api/gallery").json()[0]
    return client, job


def test_gpu_pass_is_explicit_and_duplicate_requests_reuse_the_same_job(monkeypatch, tmp_path, stone_source):
    renderer = MaterialRenderer()
    client, source = material_client(monkeypatch, tmp_path, stone_source, renderer)
    assert source["can_finish"] and renderer.calls == 0
    assert "finish" in client.get("/api/health").json()["modes"]
    url = f'/api/jobs/{source["id"]}/finish'
    first = client.post(url)
    assert first.status_code == 200, first.text
    second = client.post(url)
    assert second.json()["id"] == first.json()["id"]
    for _ in range(100):
        job = client.get('/api/jobs/' + first.json()["id"]).json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(.1)
    assert job["status"] == "done", job
    assert renderer.calls == 1 and not job["can_finish"]
    assert client.post('/api/jobs/' + job["id"] + '/finish').status_code == 409
    assert client.post(url).json()["id"] == job["id"]


def test_offline_material_request_leaves_the_original_and_creates_no_job(monkeypatch, tmp_path, stone_source):
    client, source = material_client(monkeypatch, tmp_path, stone_source, None)
    assert "finish" not in client.get("/api/health").json()["modes"]
    response = client.post(f'/api/jobs/{source["id"]}/finish')
    assert response.status_code == 409
    assert not list((tmp_path / "jobs").iterdir())
    assert client.get('/api/jobs/' + source["id"]).json()["status"] == "done"
