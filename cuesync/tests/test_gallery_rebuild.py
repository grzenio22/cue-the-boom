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
