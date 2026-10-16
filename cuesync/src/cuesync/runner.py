"""Run the exported ainvfx-fluid ComfyUI API graph through the Comfy SDK (Comfy API v2).

Nodes are bound by class_type, never by hard-coded IDs (subgraph exports use IDs like "105:104").
The target is chosen by COMFY_BASE_URL (e.g. http://127.0.0.1:8189 for a self-hosted comfy-api-proxy).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .graph import UNWANTED
from .media import read_video


class GraphBindError(ValueError):
    """The API graph doesn't have the nodes the fluid render needs."""


class RenderContractError(RuntimeError):
    """The returned element doesn't match its guide (frame count / fps)."""


@dataclass
class BoundGraph:
    graph: dict
    load_video_id: str
    output_id: str


@dataclass
class RenderResult:
    path: Path
    seconds: float


def _nodes_of(graph: dict, class_type: str) -> list[str]:
    return [k for k, v in graph.items() if isinstance(v, dict) and v.get("class_type") == class_type]


def bind_fluid_graph(graph: dict, *, prompt: str, seed: int) -> BoundGraph:
    """Return a deep-copied graph with prompt and seed set, plus the LoadVideo and SaveVideo node ids."""
    if not prompt.startswith("ainvfxfluid, "):
        raise GraphBindError("prompt must start with 'ainvfxfluid, ' (the LoRA's trigger)")
    g = copy.deepcopy(graph)
    loaders = _nodes_of(g, "LoadVideo")
    if len(loaders) != 1:
        raise GraphBindError(f"expected exactly one LoadVideo node, found {len(loaders)}")
    prompts = [k for k in _nodes_of(g, "PrimitiveStringMultiline") if "ainvfxfluid" in str(g[k].get("inputs", {}).get("value", ""))]
    if len(prompts) != 1:
        raise GraphBindError(f"expected exactly one prompt node containing 'ainvfxfluid', found {len(prompts)}")
    g[prompts[0]]["inputs"]["value"] = prompt
    noises = _nodes_of(g, "RandomNoise")
    if not noises:
        raise GraphBindError("no RandomNoise node to seed")
    for k in noises:
        g[k]["inputs"]["noise_seed"] = int(seed)
    savers = _nodes_of(g, "SaveVideo")
    if len(savers) != 1:
        raise GraphBindError(f"expected exactly one SaveVideo node, found {len(savers)}")
    unwanted = sorted({v["class_type"] for v in g.values() if isinstance(v, dict) and v.get("class_type") in UNWANTED})
    if unwanted:
        raise GraphBindError(f"graph depends on {unwanted} (partner API / prompt enhancer); "
                             "derive it with `cuesync graph` so it runs on open weights only")
    return BoundGraph(graph=g, load_video_id=loaders[0], output_id=savers[0])


def check_render_contract(guide_path: str | Path, element_path: str | Path) -> None:
    """Raise RenderContractError unless the element has the guide's frame count and fps (+/-0.01)."""
    guide, element = read_video(guide_path), read_video(element_path)
    gn, en = guide.frames.shape[0], element.frames.shape[0]
    if gn != en or abs(guide.fps_nominal - element.fps_nominal) > 0.01:
        raise RenderContractError(
            f"element {element_path} has {en} frames @ {element.fps_nominal:.3f} fps; "
            f"guide {guide_path} has {gn} frames @ {guide.fps_nominal:.3f} fps"
        )


def render_element(
    guide_path: str | Path,
    *,
    prompt: str,
    seed: int,
    workflow_path: str | Path,
    out_path: str | Path,
    timeout: float = 600.0,
) -> RenderResult:
    """Render one fluid element from ``guide_path`` and download it to ``out_path``.

    ``timeout`` bounds the wait for the job (seconds): a server that lost its upstream must not hang a batch.
    """
    try:
        from comfy_sdk import Comfy, InvalidWorkflow, JobFailed
    except ImportError as exc:  # optional dependency
        raise ImportError("comfy_sdk is not installed: pip install cuesync[comfy]") from exc

    graph = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    bound = bind_fluid_graph(graph, prompt=prompt, seed=seed)
    client = Comfy()  # COMFY_BASE_URL / COMFY_API_KEY from the environment
    started = time.perf_counter()
    wf = client.workflows.from_json(bound.graph)
    wf.set_input(bound.load_video_id, "file", client.assets.from_file(str(guide_path)))
    try:
        job = client.run(wf, timeout=timeout)
    except TimeoutError as exc:
        raise RuntimeError(f"render of {guide_path} did not finish within {timeout:.0f} s; "
                           "is ComfyUI behind the proxy still running?") from exc
    except JobFailed as exc:  # surface the node that broke, not just "job failed"
        err = exc.error
        detail = f" [{err.code}] node {err.node_id}: {err.message}" if err else ""
        raise RuntimeError(f"render failed for {guide_path}{detail}") from exc
    except InvalidWorkflow as exc:
        raise RuntimeError(f"ComfyUI rejected the graph: {exc} {json.dumps(getattr(exc, 'details', None), default=str)[:2000]}") from exc
    outputs = job.get_outputs(bound.output_id)
    if not outputs:
        raise RuntimeError(f"job produced no output on node {bound.output_id}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].to_file(str(out_path))
    seconds = time.perf_counter() - started
    check_render_contract(guide_path, out_path)
    return RenderResult(path=out_path, seconds=seconds)
