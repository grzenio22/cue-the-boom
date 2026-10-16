import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cuesync.media import write_video
from cuesync.runner import GraphBindError, RenderContractError, bind_fluid_graph, check_render_contract, render_element

G = json.loads((Path(__file__).parent / "fixtures" / "fluid_element.api.json").read_text())


def test_binds_by_class_type():
    with pytest.warns(UserWarning):
        b = bind_fluid_graph(G, prompt="ainvfxfluid, fire", seed=7)
    assert b.load_video_id == "12" and b.output_id == "90"
    assert b.graph["105:104"]["inputs"]["value"] == "ainvfxfluid, fire"
    assert b.graph["105:7"]["inputs"]["value"] == "blurry, low quality" and b.graph["33:2"]["inputs"]["noise_seed"] == 7
    assert G["33:2"]["inputs"]["noise_seed"] == 42                    # input not mutated


def test_two_loadvideo_raises():
    g = copy.deepcopy(G); g["13"] = {"class_type": "LoadVideo", "inputs": {"file": "y.mp4"}}
    with pytest.raises(GraphBindError):
        bind_fluid_graph(g, prompt="ainvfxfluid, x", seed=1)


def test_contract_mismatch_raises(tmp_path):
    write_video(tmp_path/"g.mp4", np.zeros((25, 64, 64, 3), np.uint8), fps=24)
    write_video(tmp_path/"e.mp4", np.zeros((17, 64, 64, 3), np.uint8), fps=24)
    with pytest.raises(RenderContractError):
        check_render_contract(tmp_path/"g.mp4", tmp_path/"e.mp4")
    check_render_contract(tmp_path/"g.mp4", tmp_path/"g.mp4")         # identical: no error


def test_missing_sdk_gives_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "comfy_sdk", None)
    with pytest.raises(ImportError, match=r"cuesync\[comfy\]"):
        render_element(tmp_path/"g.mp4", prompt="ainvfxfluid, x", seed=1, workflow_path=tmp_path/"w.json", out_path=tmp_path/"o.mp4")
