"""Derive the CueSync element graph from the ainvfx-fluid sample graph (both in ComfyUI API format).

The sample paints its control video inside the graph (Painter -> Batch Images -> Create Video).
CueSync paints the guide itself, so this swaps that front end for one Load Video node. It also
points the image input of the sample's I2V branch (bypassed by default) at the loaded frames.

The sample also carries optional branches behind If/Else switches: a hosted Gemma API text
encoder (a partner API) and a prompt enhancer. Both are off by default. Each switch is
replaced by its local branch, so the result runs on open weights only.

Finally every node that no longer feeds the output is dropped. Everything else stays
exactly as the model's author shipped it.
"""

from __future__ import annotations

import copy

# Branches the element graph must not depend on: a partner API, and the optional LLM prompt enhancer.
UNWANTED = ("GemmaAPITextEncode", "TextGenerateLTX2Prompt")


class GraphConvertError(ValueError):
    """The input isn't the ainvfx-fluid sample graph this converter expects."""


def _is_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _nodes(graph: dict, class_type: str) -> list[str]:
    return [k for k, n in graph.items() if n.get("class_type") == class_type]


def _ancestors(graph: dict, roots: list[str]) -> set[str]:
    seen, stack = set(), list(roots)
    while stack:
        key = stack.pop()
        if key in seen or key not in graph:
            continue
        seen.add(key)
        stack.extend(v[0] for v in graph[key].get("inputs", {}).values() if _is_link(v))
    return seen


def _new_id(graph: dict) -> str:
    numbers = [int(part) for key in graph for part in key.split(":") if part.isdigit()]
    return str(max(numbers, default=0) + 1)


def _rewire(graph: dict, old: list, new: list) -> None:
    for node in graph.values():
        inputs = node.get("inputs", {})
        for name, value in inputs.items():
            if value == old:
                inputs[name] = new


def _take_local_branches(graph: dict) -> None:
    """Replace each If/Else switch with whichever branch avoids UNWANTED nodes (when exactly one does)."""

    def unwanted(link) -> bool:
        return any(graph[k].get("class_type") in UNWANTED for k in _ancestors(graph, [link[0]]))

    changed = True
    while changed:
        changed = False
        for key, node in list(graph.items()):
            if node.get("class_type") != "ComfySwitchNode":
                continue
            on_true, on_false = node.get("inputs", {}).get("on_true"), node.get("inputs", {}).get("on_false")
            if not (_is_link(on_true) and _is_link(on_false)):
                continue
            bad_true, bad_false = unwanted(on_true), unwanted(on_false)
            if bad_true == bad_false:
                continue
            _rewire(graph, [key, 0], on_false if bad_true else on_true)
            del graph[key]
            changed = True


def element_graph(sample: dict, *, video_name: str = "guide.mp4") -> dict:
    """Return the sample graph with its in-graph control video replaced by a LoadVideo node."""
    g = copy.deepcopy(sample)
    components = _nodes(g, "GetVideoComponents")
    if len(components) != 1:
        raise GraphConvertError(f"expected one GetVideoComponents (the Source Video input), found {len(components)}")
    comp = components[0]
    source = g[comp].get("inputs", {}).get("video")
    if not _is_link(source) or g.get(source[0], {}).get("class_type") != "CreateVideo":
        raise GraphConvertError("the source video isn't built by a Create Video node; is this the ainvfx-fluid sample?")
    painted = g[source[0]].get("inputs", {}).get("images")

    load = _new_id(g)
    g[load] = {"class_type": "LoadVideo", "inputs": {"file": video_name},
               "_meta": {"title": "Guide video (painted by CueSync)"}}
    g[comp]["inputs"]["video"] = [load, 0]
    if _is_link(painted):  # e.g. the I2V branch's resize node: give it the loaded frames instead
        _rewire(g, painted, [comp, 0])
    _take_local_branches(g)

    savers = _nodes(g, "SaveVideo")
    if len(savers) != 1:
        raise GraphConvertError(f"expected one SaveVideo, found {len(savers)}")
    keep = _ancestors(g, savers)
    g = {k: v for k, v in g.items() if k in keep}

    leftovers = [g[k]["class_type"] for k in g if g[k]["class_type"] in ("Painter", "EmptyImage", "BatchImagesNode", "EmptyAudio")]
    if leftovers or source[0] in g:
        raise GraphConvertError(f"the old control-video front end still feeds the output: {leftovers or ['CreateVideo']}")
    unwanted = sorted({g[k]["class_type"] for k in g if g[k]["class_type"] in UNWANTED})
    if unwanted:
        raise GraphConvertError(f"the output still depends on {unwanted}")
    if load not in g:
        raise GraphConvertError("the Load Video node doesn't reach the output")
    return g
