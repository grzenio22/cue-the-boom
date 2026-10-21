"""Render the project's ComfyUI node-pack close-up for the v15 demo.

The display names and socket definitions are read with AST from the node pack,
without importing its runtime. The two displayed node-to-node cables and the
guide's entry into the LTX workflow are checked against the scene API template.
The layout is a fresh, labelled graph view, matching the existing LTX cards.
All text and graph content end before y=840, leaving narration-caption space.

    python tools/demo/v15_nodes.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import ACCENT, BG, FG, MUTED, font  # noqa: E402
from v13_cards import LINE, NODE, PORT, TITLE, text, wrapped  # noqa: E402

W, H = 1920, 1080
PACK = ("CueSyncAnalyzeVoice", "CueSyncPaintScene", "CueSyncSceneShot")
CAPTION = "Turns a sound into the guide the LTX render follows."


def definitions(path: Path) -> dict:
    """Extract only literal public names and socket types, never run the pack."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "NODE_DISPLAY_NAME_MAPPINGS" for t in n.targets))
    found = {}
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in PACK:
            continue
        values = {t.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and t.id in ("RETURN_TYPES", "RETURN_NAMES")}
        inputs = {}
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
        returned = next(n.value for n in ast.walk(method) if isinstance(n, ast.Return))
        for section, fields in zip(returned.keys, returned.values):
            for key, spec in zip(fields.keys, fields.values):
                first = spec.elts[0]
                inputs[ast.literal_eval(key)] = {
                    "type": ast.literal_eval(first) if isinstance(first, ast.Constant) else "CHOICE",
                    "section": ast.literal_eval(section),
                }
        found[cls.name] = {"display_name": names[cls.name], "inputs": inputs,
                           "outputs": dict(zip(values["RETURN_NAMES"], values["RETURN_TYPES"]))}
    if set(found) != set(PACK):
        raise ValueError(f"Missing node definitions: {set(PACK) - set(found)}")
    return found


def source_graph(path: Path, specs: dict) -> tuple[dict, dict, list[dict]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    ids = {kind: next(k for k, v in graph.items() if v["class_type"] == kind) for kind in PACK}
    analyze, paint, shot = (ids[kind] for kind in PACK)
    checked = []

    def edge(source: str, output: str, target: str, input_name: str) -> None:
        kind = graph[source]["class_type"]
        index = list(specs[kind]["outputs"]).index(output)
        actual = graph[target]["inputs"][input_name]
        if actual != [source, index]:
            raise ValueError(f"Not a real scene-workflow edge: {source}.{output} -> {target}.{input_name}")
        checked.append({"source": source, "output": output, "output_index": index,
                        "target": target, "input": input_name})

    edge(analyze, "cue", paint, "cue")
    edge(paint, "plan", shot, "plan")
    components = next(k for k, v in graph.items()
                      if v["class_type"] == "GetVideoComponents" and v["inputs"].get("video") == [paint, 0])
    edge(paint, "guide", components, "video")
    # The compact LTX rectangle represents a workflow group, not an extra node.
    # Check the actual endpoint feeding Scene Shot's render port as well.
    render_source, render_index = graph[shot]["inputs"]["render"]
    if graph[render_source]["class_type"] != "CreateVideo" or render_index != 0:
        raise ValueError("Scene Shot render is not the template's CreateVideo output")
    checked.append({"source": render_source, "output_index": render_index,
                    "target": shot, "input": "render"})
    return graph, ids, checked


def cable(d: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, color=ACCENT) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = max(40, (x1 - x0) * .5)
    points = []
    for i in range(61):
        t, s = i / 60, 1 - i / 60
        points.append((round(s**3*x0 + 3*s*s*t*(x0+dx) + 3*s*t*t*(x1-dx) + t**3*x1),
                       round(s**3*y0 + 3*s*s*t*y0 + 3*s*t*t*y1 + t**3*y1)))
    d.line(points, fill=color, width=5)


def node(d: ImageDraw.ImageDraw, spec: dict, box: tuple[int, int, int, int],
         *, title_size: int, header: int = 76) -> None:
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=15, fill=NODE, outline=ACCENT, width=5)
    d.rounded_rectangle((x0 + 3, y0 + 3, x1 - 3, y0 + header), radius=12, fill=TITLE)
    d.rectangle((x0 + 3, y0 + header - 16, x1 - 3, y0 + header), fill=TITLE)
    lines = wrapped(d, spec["display_name"], title_size, x1 - x0 - 42, True)
    if len(lines) * (title_size + 8) > header - 15:
        raise ValueError(f"Node title does not fit: {spec['display_name']}")
    for i, value in enumerate(lines):
        text(d, (x0 + 20, y0 + 15 + i * (title_size + 8)), value, title_size, FG, True)


def socket(d: ImageDraw.ImageDraw, spec: dict, key: str, xy: tuple[int, int],
           *, output: bool = False, size: int = 33) -> tuple[int, int]:
    x, y = xy
    fields = spec["outputs"] if output else spec["inputs"]
    if key not in fields:
        raise ValueError(f"No defined socket {spec['display_name']}.{key}")
    d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=PORT)
    text(d, (x - 19 if output else x + 19, y), key, size, FG,
         anchor="rm" if output else "lm")
    return xy


def render(specs: dict, graph: dict, ids: dict) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    text(d, (64, 31), "Our ComfyUI node pack", 66, bold=True)
    text(d, (66, 131), "Sound → guide → LTX render → finished shot", 47, ACCENT, True)
    text(d, (67, 207), "comfyui-cuesync/nodes.py · actual scene-workflow nodes", 28, MUTED)
    d.rounded_rectangle((62, 258, W - 62, 754), radius=18,
                        fill=(20, 20, 25), outline=LINE, width=2)
    for x in range(76, W - 64, 32):
        for y in range(272, 750, 32):
            d.point((x, y), fill=(39, 39, 44))

    a, p, s = (specs[kind] for kind in PACK)
    node(d, a, (101, 344, 680, 694), title_size=43, header=122)
    node(d, p, (790, 299, 1290, 710), title_size=40)
    node(d, s, (1410, 440, 1820, 710), title_size=36)
    socket(d, a, "audio", (101, 514))
    offset = graph[ids[PACK[0]]]["inputs"]["offset_s"]
    text(d, (122, 564), f"offset_s    {offset}", 29, MUTED)
    cue = socket(d, a, "cue", (680, 620), output=True)
    socket(d, a, "envelope", (680, 666), output=True, size=29)
    target_cue = socket(d, p, "cue", (790, 420))
    socket(d, p, "plate", (790, 475))
    effect = graph[ids[PACK[1]]]["inputs"]["effect"]
    text(d, (811, 534), "effect", 28, MUTED)
    text(d, (811, 575), effect, 32, FG)
    guide = socket(d, p, "guide", (1290, 420), output=True)
    plan = socket(d, p, "plan", (1290, 654), output=True)
    socket(d, s, "plate", (1410, 548), size=31)
    socket(d, s, "render", (1410, 594), size=31)
    text(d, (1540, 594), "from LTX", 25, MUTED, anchor="lm")
    target_plan = socket(d, s, "plan", (1410, 650), size=31)
    socket(d, s, "shot", (1820, 682), output=True, size=31)

    # This is explicitly the existing render-workflow group, not a fabricated node.
    d.rounded_rectangle((1420, 290, 1810, 388), radius=15, fill=(28, 28, 33), outline=LINE, width=3)
    text(d, (1615, 322), "LTX workflow", 38, FG, True, "mm")
    text(d, (1615, 365), "guide video in", 29, MUTED, anchor="mm")
    d.ellipse((1413, 332, 1427, 346), fill=PORT)
    cable(d, cue, target_cue)
    cable(d, plan, target_plan)
    cable(d, guide, (1420, 339))

    d.line((66, 778, W - 66, 778), fill=LINE, width=2)
    if d.textlength(CAPTION, font=font(36, True)) > W - 132:
        raise ValueError("Node-pack caption exceeds its safe width")
    text(d, (W // 2, 802), CAPTION, 36, FG, True, "mm")
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("D:/openHiggsfield/runs/demo_v15/cards"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    pack = root / "comfyui-cuesync/nodes.py"
    workflow = root / "workflows/cue_the_boom_scene.api.json"
    specs = definitions(pack)
    graph, ids, edges = source_graph(workflow, specs)
    card = render(specs, graph, ids)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "comfy_cuesync.png"
    card.save(target)
    evidence = {"image": str(target), "size": [W, H], "caption_safe_y": 840,
                "caption": CAPTION, "definitions": specs, "checked_edges": edges,
                "sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (pack, workflow)},
                "note": "Fresh graph layout from actual node definitions and checked scene-template links; LTX is a workflow group."}
    (args.out / "comfy_cuesync_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
