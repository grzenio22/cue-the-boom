import json

import pytest

from cuesync.graph import GraphConvertError, element_graph
from cuesync.runner import bind_fluid_graph


def sample_graph() -> dict:
    """The ainvfx-fluid sample's shape after ComfyUI flattens its subgraphs (abridged)."""
    return {
        "9794": {"class_type": "Painter", "inputs": {"image": "first.png [input]", "width": 512, "height": 512}},
        "9797": {"class_type": "Painter", "inputs": {"image": ["9794", 0], "width": 512, "height": 512}},
        "9821": {"class_type": "EmptyImage", "inputs": {"width": 512, "height": 512, "batch_size": 119, "color": 0}},
        "9822": {"class_type": "BatchImagesNode",
                 "inputs": {"images.image0": ["9794", 0], "images.image1": ["9821", 0], "images.image2": ["9797", 0]}},
        "9795": {"class_type": "EmptyAudio", "inputs": {"duration": 5, "sample_rate": 44100, "channels": 2}},
        "9798": {"class_type": "CreateVideo", "inputs": {"images": ["9822", 0], "audio": ["9795", 0], "fps": 25}},
        "9812:9741": {"class_type": "GetVideoComponents", "inputs": {"video": ["9798", 0]}},
        "9812:9740": {"class_type": "GetImageSize", "inputs": {"image": ["9812:9741", 0]}},
        "9811:9725": {"class_type": "ResizeImageMaskNode", "inputs": {"input": ["9822", 0], "resize_type": "scale longer dimension"}},
        "9807": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "ainvfxfluid, smoke plume"}},
        "9811:9732": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9807", 0], "clip": ["9802:9723", 0]}},
        "9802:9723": {"class_type": "CLIPLoader", "inputs": {"clip_name": "gemma4-12b.safetensors", "type": "ltxv"}},
        "9814:9756": {"class_type": "LTXVImgToVideoInplace", "inputs": {"image": ["9811:9725", 0], "bypass": True}},
        "9814:9755": {"class_type": "LTXAddVideoICLoRAGuide",
                      "inputs": {"image": ["9812:9741", 0], "latent": ["9814:9756", 0], "positive": ["9811:9732", 0]}},
        "9813:9745": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "9813:9750": {"class_type": "SamplerCustomAdvanced",
                      "inputs": {"noise": ["9813:9745", 0], "latent_image": ["9814:9755", 2]}},
        "9815:9761": {"class_type": "CreateVideo",
                      "inputs": {"images": ["9813:9750", 0], "fps": ["9812:9741", 2], "audio": ["9812:9741", 1]}},
        "9816": {"class_type": "SaveVideo", "inputs": {"video": ["9815:9761", 0], "filename_prefix": "ainvfx-fluid/x"}},
        "9824": {"class_type": "MarkdownNote", "inputs": {}},
    }


def test_swaps_the_painted_control_video_for_load_video():
    g = element_graph(sample_graph(), video_name="guide.mp4")
    loads = [k for k, n in g.items() if n["class_type"] == "LoadVideo"]
    assert len(loads) == 1 and g[loads[0]]["inputs"] == {"file": "guide.mp4"}
    assert g["9812:9741"]["inputs"]["video"] == [loads[0], 0]
    gone = {"Painter", "EmptyImage", "BatchImagesNode", "EmptyAudio", "MarkdownNote"}
    assert not gone & {n["class_type"] for n in g.values()}
    assert "9798" not in g and "9815:9761" in g  # the control CreateVideo goes, the decode one stays


def test_i2v_branch_gets_the_loaded_frames():
    g = element_graph(sample_graph())
    assert g["9811:9725"]["inputs"]["input"] == ["9812:9741", 0]


def test_new_node_id_does_not_collide_and_downstream_is_untouched():
    src = sample_graph()
    g = element_graph(src)
    load = next(k for k, n in g.items() if n["class_type"] == "LoadVideo")
    assert load not in src and int(load) > 9824
    for key in ("9814:9755", "9813:9750", "9815:9761", "9816", "9813:9745"):
        assert g[key] == src[key]
    json.dumps(g)


def test_result_binds_with_the_runner():
    bound = bind_fluid_graph(element_graph(sample_graph()), prompt="ainvfxfluid, bright orange fire", seed=7)
    assert bound.graph[bound.load_video_id]["class_type"] == "LoadVideo"
    assert bound.graph["9807"]["inputs"]["value"] == "ainvfxfluid, bright orange fire"
    assert bound.graph["9813:9745"]["inputs"]["noise_seed"] == 7


def with_optional_branches() -> dict:
    """Add the sample's If/Else switches: hosted Gemma API encoder and the LLM prompt enhancer."""
    g = sample_graph()
    g.update({
        "9811:9733": {"class_type": "StringContains", "inputs": {"string": "", "substring": "ltxv_"}},
        "9811:9738": {"class_type": "GemmaAPITextEncode", "inputs": {"api_key": "", "prompt": ["9807", 0]}},
        "9811:9729": {"class_type": "TextGenerateLTX2Prompt", "inputs": {"clip": ["9802:9722", 0], "prompt": ["9807", 0]}},
        "9802:9722": {"class_type": "CLIPLoader", "inputs": {"clip_name": "gemma4_e2b_it_int8_convrot.safetensors"}},
        # enhancer on/off: off -> the raw prompt
        "9811:9737": {"class_type": "ComfySwitchNode",
                      "inputs": {"on_false": ["9807", 0], "on_true": ["9811:9729", 0], "switch": False}},
        # conditioning source: API key present -> the hosted encoder
        "9811:9736": {"class_type": "ComfySwitchNode",
                      "inputs": {"on_false": ["9811:9732", 0], "on_true": ["9811:9738", 0], "switch": ["9811:9733", 0]}},
    })
    g["9811:9732"]["inputs"]["text"] = ["9811:9737", 0]
    g["9814:9755"]["inputs"]["positive"] = ["9811:9736", 0]
    return g


def test_partner_api_and_enhancer_branches_are_removed():
    g = element_graph(with_optional_branches())
    kinds = {n["class_type"] for n in g.values()}
    assert not kinds & {"GemmaAPITextEncode", "TextGenerateLTX2Prompt", "ComfySwitchNode", "StringContains"}
    assert "9802:9722" not in g  # the enhancer's text model is no longer loaded
    assert g["9811:9732"]["inputs"]["text"] == ["9807", 0]
    assert g["9814:9755"]["inputs"]["positive"] == ["9811:9732", 0]
    bind_fluid_graph(g, prompt="ainvfxfluid, white steam", seed=1)


def test_a_switch_with_no_local_branch_is_an_error():
    g = with_optional_branches()
    g["9811:9736"]["inputs"]["on_false"] = ["9811:9738", 0]  # both branches hosted
    with pytest.raises(GraphConvertError, match="GemmaAPITextEncode"):
        element_graph(g)


def test_rejects_other_graphs():
    g = sample_graph()
    g["9812:9741"]["inputs"]["video"] = ["9816", 0]
    with pytest.raises(GraphConvertError, match="Create Video"):
        element_graph(g)
    g = sample_graph()
    del g["9812:9741"]
    with pytest.raises(GraphConvertError, match="GetVideoComponents"):
        element_graph(g)
