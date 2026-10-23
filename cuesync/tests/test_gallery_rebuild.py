"""The gallery rebuild must preserve a finished shot's real cue source/timing."""
from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import shutil

import pytest

import cuesync.cues


REPOSITORY = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gallery_rebuild_recipe", REPOSITORY / "tools/rebuild_hand_fire_gallery.py")
RECIPE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RECIPE)


@pytest.fixture
def palm_gallery(tmp_path, monkeypatch):
    """Keep real shipped media/analysis, with a conflicting future-performance alias."""
    samples = tmp_path / "samples"
    gallery = samples / "made_ahead/bending"
    name = RECIPE.BASE
    source = REPOSITORY / "samples/made_ahead/bending" / name
    target = gallery / name
    target.mkdir(parents=True)
    for file in ("receipt.json", "element.mp4", "hand_track.json", "hand_mask.npz"):
        shutil.copyfile(source / file, target / file)
    (samples / "plates").mkdir()
    shutil.copyfile(REPOSITORY / "samples/plates/hand_palm.mp4", samples / "plates/hand_palm.mp4")
    (samples / "voices").mkdir()
    shutil.copyfile(REPOSITORY / "samples/voices/ka-boom.wav", samples / "voices/ka-boom.wav")
    entry = {"dir": name, "plate": "hand_palm.mp4", "voice": "", "effect": "fire_burst", "seed": 2}
    (gallery / "index.json").write_text(json.dumps({"original": entry,
        "future_voice_preset": {**entry, "voice": "ka-boom.wav"}}), encoding="utf-8")
    monkeypatch.setattr(RECIPE, "ROOT", tmp_path)
    monkeypatch.setattr(RECIPE, "SAMPLES", samples)
    monkeypatch.setattr(RECIPE, "GALLERY", gallery)
    return name, target


def test_clip_cue_ignores_conflicting_suggested_voice_alias(palm_gallery, monkeypatch):
    name, _ = palm_gallery
    real_load_cue = cuesync.cues.load_cue
    calls = []

    def observe(source, **kwargs):
        calls.append((source, kwargs.get("voice"), kwargs["plate"].name))
        return real_load_cue(source, **kwargs)

    monkeypatch.setattr(cuesync.cues, "load_cue", observe)
    item = RECIPE.inventory([name])[0]
    assert calls == [("clip", None, "hand_palm.mp4")]
    assert "voice" not in item["assets"]
    assert item["raw_audio_hit_frames"] == [64]
    assert item["cue_voice_end_frame"] == 69


def test_inventory_rejects_actual_audio_that_no_longer_matches_cached_timing(palm_gallery):
    name, directory = palm_gallery
    path = directory / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["bending"]["hits"][0]["snap"]["cue"] = 65
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Cue reproduction changed.*\[64\].*\[65\]"):
        RECIPE.inventory([name])


@pytest.fixture
def walk_gallery(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    gallery = samples / "made_ahead/bending"
    name = "hand_walk__fffff__flame_plume"
    source = REPOSITORY / "samples/made_ahead/bending" / name
    target = gallery / name
    target.mkdir(parents=True)
    for file in ("receipt.json", "element.mp4", "hand_track.json", "hand_mask.npz"):
        shutil.copyfile(source / file, target / file)
    # This is explicitly the historical receipt, even after the shipped gallery
    # is regenerated. The source audio still goes through the real detector.
    receipt = json.loads((target / "receipt.json").read_text(encoding="utf-8"))
    prototype = receipt["bending"]["hits"][-1]
    receipt["bending"]["hits"] = []
    for frame in (57, 60):
        hit = copy.deepcopy(prototype)
        hit["frame"] = frame
        hit.setdefault("snap", {})["cue"] = frame
        receipt["bending"]["hits"].append(hit)
    receipt["bending"]["held"]["holds"] = [92, 89]
    receipt.pop("recomposite", None)
    (target / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (samples / "plates").mkdir()
    shutil.copyfile(REPOSITORY / "samples/plates/hand_walk.mp4", samples / "plates/hand_walk.mp4")
    entry = {"dir": name, "plate": "hand_walk.mp4", "voice": "", "effect": "flame_plume", "seed": 2}
    (gallery / "index.json").write_text(json.dumps({"original": entry}), encoding="utf-8")
    monkeypatch.setattr(RECIPE, "ROOT", tmp_path)
    monkeypatch.setattr(RECIPE, "SAMPLES", samples)
    monkeypatch.setattr(RECIPE, "GALLERY", gallery)
    return name, target


def test_walk_charge_window_preserves_historical_rendered_timing(walk_gallery):
    name, _ = walk_gallery
    item = RECIPE.inventory([name])[0]
    assert item["raw_audio_hit_frames"] == [60]
    assert item["render_cue_hit_frames"] == [57, 60]
    assert item["charge_window"] == [35, 57]
    assert item["cue_voice_end_frame"] == 149
    assert item["expected_rendered_hit_frames"] == [57, 60]
    assert item["expected_hold_frames"] == [92, 89]


def test_walk_inventory_rejects_hold_drift_with_unchanged_onsets(walk_gallery, monkeypatch):
    name, _ = walk_gallery
    real_load_cue = cuesync.cues.load_cue
    def shorten(*args, **kwargs):
        cue = real_load_cue(*args, **kwargs)
        cue.cue.active_intervals = [(35, 140)]
        return cue
    monkeypatch.setattr(cuesync.cues, "load_cue", shorten)
    with pytest.raises(ValueError, match="hold duration changed"):
        RECIPE.inventory([name])


def test_walk_inventory_rejects_loss_of_charge_release(walk_gallery):
    name, directory = walk_gallery
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["bending"]["hits"] = [receipt["bending"]["hits"][-1]]
    receipt["bending"]["held"]["holds"] = [89]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="Cue reproduction changed"):
        RECIPE.inventory([name])


def test_cupped_anchor_choice_is_private_current_intent_not_historical_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(RECIPE, "ROOT", tmp_path)
    source = tmp_path / "original"; source.mkdir()
    original = {"bending": {"hits": [{"frame": 48}, {"frame": 60}]}}
    (source / "receipt.json").write_text(json.dumps(original))
    (source / "element.mp4").write_bytes(b"existing generated element bytes")
    receipt_hash = RECIPE.sha256(source / "receipt.json")
    element_hash = RECIPE.sha256(source / "element.mp4")
    item = {"name": "hand_cupped__kaboom__fire_burst", "source": "original", "receipt": original,
            "assets": {"element": {"path": "original/element.mp4", "sha256": element_hash},
                       "element_receipt": {"path": "original/receipt.json", "sha256": receipt_hash}}}
    staged, intent = RECIPE.prepare_reuse_source(item, tmp_path / "run")
    copied = json.loads((staged / "receipt.json").read_text())
    assert copied["bending"]["thrown"] is None
    assert copied["current_reconstruction_choice"] == intent
    assert intent["historical_field_present"] is False
    assert intent["original_source_receipt_sha256"] == receipt_hash
    assert intent["field"] == "bending.thrown" and intent["authored_value"] is None
    assert RECIPE.sha256(source / "receipt.json") == receipt_hash
    assert "thrown" not in original["bending"]
    assert RECIPE.sha256(staged / "element.mp4") == element_hash


@pytest.fixture
def source_override(palm_gallery, tmp_path):
    name, current = palm_gallery
    selected = tmp_path / "candidate"
    selected.mkdir()
    shutil.copyfile(current / "element.mp4", selected / "element.mp4")
    adapter = json.loads((current / "receipt.json").read_text())
    adapter["bending"]["loud_ref"] = 99.0
    adapter["bending"]["thrown"] = {"after": 1, "hands": 1.0}
    (selected / "receipt.json").write_text(json.dumps(adapter))
    generation = tmp_path / "original-generation.json"
    generation.write_text(json.dumps({"model": "existing LTX", "seed": 11}))
    config = {"schema": 1, "source_assets": {"selected-fire": {"generation.json": {
        "path": "original-generation.json", "sha256": RECIPE.sha256(generation)}}},
        "shots": {name: {"directory": "candidate", "adapter_sha256": RECIPE.sha256(selected / "receipt.json"),
            "element_sha256": RECIPE.sha256(selected / "element.mp4"), "source_asset_id": "selected-fire",
            "source_profile": {"scale": 0.7, "exposure": 1.5}, "thrown": None,
            "intent_reason": "Current authored anchored palm choice"}}}
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(config))
    return name, current, selected, path


def test_source_override_keeps_voice_calibration_and_authors_motion_without_rewriting_generation(source_override, tmp_path):
    name, current, selected, path = source_override
    original = (selected / "receipt.json").read_bytes()
    overrides = RECIPE.load_source_overrides(path)
    item = RECIPE.inventory([name], overrides)[0]
    staged, intent = RECIPE.prepare_reuse_source(item, tmp_path / "run")
    adapted = json.loads((staged / "receipt.json").read_text())
    historical = json.loads((current / "receipt.json").read_text())
    assert adapted["source_profile"] == {"scale": 0.7, "exposure": 1.5}
    assert adapted["bending"]["thrown"] is None
    assert adapted["bending"]["loud_ref"] == historical["bending"]["loud_ref"]
    assert intent["authored_as_current_choice"] is True
    assert intent["field"] == "bending.thrown"
    assert item["expected_rendered_hit_frames"] == [64]
    assert (selected / "receipt.json").read_bytes() == original
    assert (staged / "element.mp4").read_bytes() == (selected / "element.mp4").read_bytes()


def test_source_override_rejects_element_hash_mismatch_before_rebuilding(source_override):
    name, _, selected, path = source_override
    overrides = RECIPE.load_source_overrides(path)
    (selected / "element.mp4").write_bytes(b"substituted element")
    with pytest.raises((ValueError, RuntimeError), match="hash|changed"):
        RECIPE.inventory([name], overrides)


def test_public_provenance_rejects_credentials_without_echoing_secret(source_override, tmp_path):
    _, _, _, path = source_override
    secret = "not-a-real-provider-credential"
    generation = tmp_path / "original-generation.json"
    generation.write_text(json.dumps({"api_key": secret}))
    config = json.loads(path.read_text())
    config["source_assets"]["selected-fire"]["generation.json"]["sha256"] = RECIPE.sha256(generation)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError) as error:
        RECIPE.load_source_overrides(path)
    assert secret not in str(error.value)


def test_install_replaces_element_and_provenance_with_validated_output_and_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(RECIPE, "ROOT", tmp_path)
    gallery = tmp_path / "gallery"
    monkeypatch.setattr(RECIPE, "GALLERY", gallery)
    name = "hand_punch__boom__fire_burst"
    target = gallery / name; target.mkdir(parents=True)
    (target / "element.mp4").write_bytes(b"old element")
    staged = tmp_path / "run/staged" / name; staged.mkdir(parents=True)
    payloads = {"composite.mp4": b"composite", "final.mp4": b"final", "poster.jpg": b"poster",
        "receipt.json": json.dumps({"element_sha256": __import__("hashlib").sha256(b"new element").hexdigest()}).encode(),
        "element.mp4": b"new element", "source-adapter.json": b"{}", "source-provenance.json": json.dumps({
            "element_sha256": __import__("hashlib").sha256(b"new element").hexdigest(),
            "playback_adapter_sha256": __import__("hashlib").sha256(b"{}").hexdigest(), "source_files": {}}).encode()}
    for file, value in payloads.items():
        (staged / file).write_bytes(value)
    validation = {"name": name, "output_sha256": {f: RECIPE.sha256(staged / f) for f in payloads}}
    item = {"name": name, "assets": {"element": {"path": "unused", "sha256": validation["output_sha256"]["element.mp4"]}}}
    (staged / "element.mp4").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed|hash"):
        RECIPE.install_validated([item], tmp_path / "run", [validation])
    assert (target / "element.mp4").read_bytes() == b"old element"
    (staged / "element.mp4").write_bytes(b"new element")
    RECIPE.install_validated([item], tmp_path / "run", [validation])
    assert (target / "element.mp4").read_bytes() == b"new element"
    assert (target / "source-adapter.json").read_bytes() == b"{}"
    assert (target / "source-provenance.json").read_bytes() == payloads["source-provenance.json"]


def test_install_rejects_empty_source_provenance_before_modifying_gallery(tmp_path, monkeypatch):
    monkeypatch.setattr(RECIPE, "ROOT", tmp_path)
    name = "hand_punch__boom__fire_burst"
    staged = tmp_path / "run/staged" / name
    staged.mkdir(parents=True)
    for file in RECIPE.OUTPUT_FILES:
        (staged / file).write_bytes(b"fixture")
    element_hash = RECIPE.sha256(staged / "element.mp4")
    (staged / "receipt.json").write_text(json.dumps({"element_sha256": element_hash}))
    for file in RECIPE.SOURCE_FILES:
        (staged / file).write_text("{}")
    hashes = {p.name: RECIPE.sha256(p) for p in staged.iterdir()}
    item = {"name": name, "assets": {"element": {"sha256": element_hash}}}
    with pytest.raises(RuntimeError, match="provenance"):
        RECIPE.validate_staged(item, tmp_path / "run", {"output_sha256": hashes})


@pytest.mark.parametrize("motion", [{"after": 8, "hands": 1.0}, {"after": 1, "hands": 20.0}, {}, {"after": 1, "hands": 1.0, "speed": 2}])
def test_source_override_rejects_release_parameters_the_renderer_does_not_honor(source_override, motion):
    _, _, _, path = source_override
    config = json.loads(path.read_text())
    config["shots"][RECIPE.BASE]["thrown"] = motion
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="release|thrown"):
        RECIPE.load_source_overrides(path)


def test_open_puffs_use_source_span_when_audio_has_no_sustained_interval(tmp_path, monkeypatch):
    samples = tmp_path / "samples"; gallery = samples / "made_ahead/bending"
    name = "hand_open__fff__flame_plume"
    current = REPOSITORY / "samples/made_ahead/bending" / name
    target = gallery / name; target.mkdir(parents=True)
    for file in ("receipt.json", "element.mp4", "hand_track.json", "hand_mask.npz"):
        shutil.copyfile(current / file, target / file)
    historical = json.loads((target / "receipt.json").read_text())
    historical["bending"]["ignition"] = 109
    historical["bending"]["held"] = {"release": 121, "end": 121, "holds": [12, 12, 12]}
    (target / "receipt.json").write_text(json.dumps(historical))
    (samples / "plates").mkdir()
    shutil.copyfile(REPOSITORY / "samples/plates/hand_open.mp4", samples / "plates/hand_open.mp4")
    (gallery / "index.json").write_text(json.dumps({"open": {"dir": name, "plate": "hand_open.mp4", "voice": "", "effect": "flame_plume", "seed": 1}}))
    monkeypatch.setattr(RECIPE, "ROOT", tmp_path); monkeypatch.setattr(RECIPE, "SAMPLES", samples); monkeypatch.setattr(RECIPE, "GALLERY", gallery)
    item = RECIPE.inventory([name])[0]
    assert item["raw_audio_hold_frames"] == [None, None, 12]
    assert item["expected_hold_frames"] == [12, 12, 12]
    assert item["expected_rendered_hit_frames"] == [24, 57, 90]
