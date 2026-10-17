import json
from pathlib import Path

import pytest

from cuesync.effects import load_effect


def test_all_four_load():
    for i in ["fireball", "flame_plume", "smoke_puff", "steam_jet"]:
        assert load_effect(i).prompt.startswith("ainvfxfluid, ")


def test_rejects_bad_stride(tmp_path):
    bad = json.loads(Path("effects/fireball.json").read_text()) | {"keyframe_stride": 12}
    (tmp_path/"b.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_effect(tmp_path/"b.json")


def test_rejects_bad_prompt_prefix(tmp_path):
    bad = json.loads(Path("effects/fireball.json").read_text()) | {"prompt": "a fireball"}
    (tmp_path/"p.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_effect(tmp_path/"p.json")


def test_rejects_rmin_not_below_rmax(tmp_path):
    bad = json.loads(Path("effects/smoke_puff.json").read_text()) | {"r_min": 0.5, "r_max": 0.3}
    (tmp_path/"r.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_effect(tmp_path/"r.json")


def test_an_effect_file_must_be_named_after_its_id(tmp_path, monkeypatch):
    # a shot carries the effect's id and finds it again by file name, so a renamed id would fail later, in the shot
    d = json.loads(Path("effects/fireball.json").read_text()) | {"id": "inner"}
    (tmp_path/"outer.json").write_text(json.dumps(d))
    monkeypatch.setenv("CUESYNC_EFFECTS", str(tmp_path))
    with pytest.raises(ValueError, match="named after its id"):
        load_effect("outer")
