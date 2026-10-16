"""The one-workflow shot: the graph built from the element graph, and its fit with the CueSync node pack."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from cuesync.graph import GraphConvertError, shot_graph

REPO = Path(__file__).resolve().parents[2]
ELEMENT = REPO / "workflows" / "fluid_element.api.json"
SHOT = REPO / "workflows" / "cue_the_boom.api.json"


def element() -> dict:
    return json.loads(ELEMENT.read_text(encoding="utf-8"))


def by_class(g: dict, cls: str) -> list[str]:
    return [k for k, n in g.items() if n["class_type"] == cls]


def links(g: dict):
    for key, node in g.items():
        for name, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int):
                yield key, name, value


@pytest.fixture(scope="module")
def nodes():
    """Import the node pack without ComfyUI: torch and comfy_api are only touched when a node runs."""
    saved = {m: sys.modules.get(m) for m in ("torch",)}
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = importlib.util.spec_from_file_location("cuesync_nodes", REPO / "comfyui-cuesync" / "nodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod.NODE_CLASS_MAPPINGS
    for m, v in saved.items():
        if v is None:
            sys.modules.pop(m, None)


def test_the_shot_graph_wires_voice_guide_element_and_composite():
    g = shot_graph(element(), effect="steam_jet", point=(0.4, 0.6))
    (voice,), (cue,), (guide,), (shot,) = (by_class(g, c) for c in
                                           ("LoadAudio", "CueSyncAnalyzeVoice", "CueSyncPaintGuide", "CueSyncComposite"))
    (plate,) = by_class(g, "LoadVideo")  # the guide's Load Video is gone, only the plate is loaded
    assert g[plate]["_meta"]["title"] == "Your plate"
    assert g[guide]["inputs"]["effect"] == "steam_jet"
    assert (g[guide]["inputs"]["x"], g[guide]["inputs"]["y"]) == (0.4, 0.6)
    comps = {g[k]["inputs"]["video"][0]: k for k in by_class(g, "GetVideoComponents")}
    assert set(comps) == {guide, plate}  # the guide feeds the model, the plate's frames feed the matte
    (matte,) = by_class(g, "CueSyncObjectMatte")
    assert g[matte]["inputs"]["images"] == [comps[plate], 0] and g[matte]["inputs"]["plan"] == [guide, 2]
    assert g[shot]["inputs"]["object_mask"] == [matte, 0]
    texts = [g[k]["inputs"]["text"] for k in by_class(g, "CLIPTextEncode")]
    assert [guide, 1] in texts  # the effect's prompt, not the sample's
    assert not any("ainvfxfluid" in str(n["inputs"].get("value", "")) for n in g.values())
    (save,) = by_class(g, "SaveVideo")
    assert g[save]["inputs"]["video"] == [shot, 0]
    assert g[shot]["inputs"]["plate"] == [plate, 0] and g[shot]["inputs"]["voice"] == [voice, 0]
    assert g[shot]["inputs"]["plan"] == [guide, 2] and g[shot]["inputs"]["cue"] == [cue, 0]
    assert g[g[shot]["inputs"]["element"][0]]["class_type"] == "CreateVideo"  # the decoded element


def test_every_link_points_at_a_node():
    g = shot_graph(element())
    dangling = [(k, name, v) for k, name, v in links(g) if v[0] not in g]
    assert dangling == []


def test_the_committed_workflow_is_the_generated_one():
    assert json.loads(SHOT.read_text(encoding="utf-8")) == shot_graph(element())


def test_the_shot_graph_needs_the_element_graph():
    with pytest.raises(GraphConvertError):
        shot_graph({"1": {"class_type": "SaveVideo", "inputs": {}}})


def test_the_graph_matches_the_node_pack(nodes):
    g = shot_graph(element())
    for key, node in g.items():
        cls = nodes.get(node["class_type"])
        if cls is None:
            continue
        spec = cls.INPUT_TYPES()
        required, optional = spec["required"], spec.get("optional", {})
        assert set(required) <= set(node["inputs"]) <= set(required) | set(optional), node["class_type"]
        for name, value in node["inputs"].items():
            declared = (required | optional)[name][0]
            if isinstance(value, list) and value[0] in g and g[value[0]]["class_type"] in nodes:
                src = nodes[g[value[0]]["class_type"]]
                assert src.RETURN_TYPES[value[1]] == declared, (node["class_type"], name)
            elif isinstance(declared, list):  # a combo: the value must be one of its options
                assert value in declared, (node["class_type"], name, value)
    # outputs consumed by stock nodes: guide video -> Get Video Components, prompt -> CLIP Text Encode
    guide = nodes["CueSyncPaintGuide"]
    assert guide.RETURN_TYPES[:2] == ("VIDEO", "STRING")
    assert nodes["CueSyncComposite"].RETURN_TYPES == ("VIDEO",)
