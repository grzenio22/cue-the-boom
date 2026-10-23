"""Contracts for recipe-based CPU fire derivatives, without model assets."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def module():
    return importlib.import_module("derive_fire_element")


def recipe(mapping, **extra):
    return {"fps": 24, "frames": len(mapping), "time_mapping": [
        {"frame": i, "source_frame": source, "linear_radiance_gain": gain}
        for i, (source, gain) in enumerate(mapping)], **extra}


def test_mapping_reorders_and_repeats_source_frames_with_linear_light_gain():
    source = np.stack([np.full((2, 4, 3), n, np.uint8) for n in (0, 128, 255)])
    before = source.copy()
    result = module().derive_frames(source, recipe([(2, .25), (1, 1), (2, 0), (1, 1)]))
    assert result[:, 0, 0, 0].tolist() == [136, 128, 0, 128]
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("source_index", [-1, 3, 1.5, True])
def test_invalid_source_index_is_rejected(source_index):
    with pytest.raises(ValueError, match="source_frame"):
        module().derive_frames(np.zeros((3, 2, 4, 3), np.uint8), recipe([(source_index, 1)]))


@pytest.mark.parametrize("gain", [-.1, float("nan"), float("inf"), "1", True])
def test_invalid_gain_is_rejected(gain):
    with pytest.raises(ValueError, match="gain"):
        module().derive_frames(np.zeros((1, 2, 4, 3), np.uint8), recipe([(0, gain)]))


def test_mapping_cannot_silently_skip_output_frames_or_disagree_with_declared_length():
    source = np.zeros((1, 2, 4, 3), np.uint8)
    bad = recipe([(0, 1)])
    bad["time_mapping"][0]["frame"] = 1
    with pytest.raises(ValueError, match="contiguous"):
        module().derive_frames(source, bad)
    with pytest.raises(ValueError, match="frame count"):
        module().derive_frames(source, recipe([(0, 1)], frames=2))
    with pytest.raises(ValueError, match="nonempty"):
        module().derive_frames(source, recipe([]))


def test_crown_feather_is_applied_in_linear_light_without_moving_pixels():
    source = np.full((1, 500, 648, 3), 255, np.uint8)
    r = recipe([(0, 1)], mask={"formula": "(1 - smoothstep(edge, edge+140, y))^2; edge=340-.0025*(x-646)^2"})
    result = module().derive_frames(source, r)
    assert result.shape == source.shape
    assert result[0, [270, 340, 410, 480], 646, 0].tolist() == [255, 255, 136, 0]


def test_unknown_mask_formula_is_rejected_instead_of_evaluated():
    with pytest.raises(ValueError, match="mask formula"):
        module().derive_frames(np.zeros((1, 2, 4, 3), np.uint8), recipe([(0, 1)], mask={"formula": "unknown"}))


def test_source_hash_and_overwrite_guards_preserve_original(tmp_path):
    source = tmp_path / "original.mp4"
    source.write_bytes(b"untouched source")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({"original_element_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="source hash"):
        module().reproduce(source, recipe_path, tmp_path / "derived.mp4")
    assert source.read_bytes() == b"untouched source"
    assert not (tmp_path / "derived.mp4").exists()
    with pytest.raises(ValueError, match="overwrite"):
        module().reproduce(source, recipe_path, source)
