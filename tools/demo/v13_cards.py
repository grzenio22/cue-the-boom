"""Render v13's four readable, sequential ComfyUI workflow close-ups.

The checkout contains fluid_element.api.json, but no fluid_element.json canvas.
These are therefore fresh, explicitly labelled graph views of the actual API
nodes, rather than purported screenshots. Node titles, settings, socket names,
and connections are read from that file. The small local layout is presentational;
it never changes the workflow or invents a node/value. The two loader views,
the five sampler inputs, and Create Video -> Save Video all use real links.

All cards are 1920x1080 and reserve y >= 840 for narration captions. The exact
API template's prompt and output prefix are outside these close-ups, since the
app supplies the shot-specific overrides when generating an element.

    python tools/demo/v13_cards.py --out D:/openHiggsfield/runs/demo_v13/cards
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import ACCENT, BG, FG, MUTED, font  # noqa: E402

W, H = 1920, 1080
LINE = (67, 65, 61)
NODE = (44, 44, 48)
TITLE = (62, 62, 66)
PORT = (183, 212, 176)
MODEL = "9802:9720"
LORA = "9802:9724"
SAMPLER = "9813:9750"
OUTPUT = "9816"


def title_of(node: dict) -> str:
    # The API metadata uses enclosed L/T/X glyphs in this one title. Plain LTX
    # is the identical readable label, without depending on an emoji font.
    return node.get("_meta", {}).get("title", node["class_type"]).replace("🅛🅣🅧", "LTX")


def wrapped(d: ImageDraw.ImageDraw, value: str, size: int, width: int,
            bold: bool = False) -> list[str]:
    """Wrap even an exact long filename; never clip, abbreviate or ellipsize it."""
    f = font(size, bold)
    lines: list[str] = []
    for paragraph in value.split("\n"):
        rest = paragraph
        while rest:
            k = len(rest)
            while k > 1 and d.textlength(rest[:k], font=f) > width:
                k -= 1
            if k < len(rest):
                space = rest.rfind(" ", 0, k + 1)
                if space > k // 2:
                    k = space
            lines.append(rest[:k])
            rest = rest[k:].lstrip(" ")
    return lines


def text(d: ImageDraw.ImageDraw, xy, value: str, size: int, fill=FG,
         bold: bool = False, anchor="la") -> None:
    d.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def canvas(step: int, heading: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    text(d, (64, 31), f"{step} / 4   {heading}", 66, bold=True)
    text(d, (66, 131), "App → Comfy Python SDK → ComfyUI → LTX-2.5", 47, ACCENT, True)
    text(d, (67, 207), "fluid_element.api.json · actual workflow nodes", 28, MUTED)
    # The graph grid is confined to the readable diagram area.
    d.rounded_rectangle((62, 258, W - 62, 754), radius=18,
                        fill=(20, 20, 25), outline=LINE, width=2)
    for x in range(76, W - 64, 32):
        for y in range(272, 750, 32):
            d.point((x, y), fill=(39, 39, 44))
    d.line((66, 778, W - 66, 778), fill=LINE, width=2)
    text(d, (W // 2, 802), "The app sends this workflow through the Comfy Python SDK.",
         36, FG, True, "mm")
    return im, d


def node(d: ImageDraw.ImageDraw, graph: dict, key: str,
         box: tuple[int, int, int, int], *, focus=False, fields=(),
         inputs=(), output: str | None = None, title_size=43, body_size=32) -> dict:
    """Draw one exact API node and return socket positions for its real links."""
    data = graph[key]
    x0, y0, x1, y1 = box
    stroke = ACCENT if focus else LINE
    d.rounded_rectangle(box, radius=15, fill=NODE, outline=stroke, width=6 if focus else 3)
    d.rounded_rectangle((x0 + 3, y0 + 3, x1 - 3, y0 + 70), radius=12,
                        fill=TITLE if focus else (46, 46, 50))
    d.rectangle((x0 + 3, y0 + 55, x1 - 3, y0 + 70), fill=TITLE if focus else (46, 46, 50))
    node_title = title_of(data)
    title_lines = wrapped(d, node_title, title_size, x1 - x0 - 44, True)
    if len(title_lines) != 1:
        raise ValueError(f"Title must fit at its readable size: {node_title}")
    text(d, (x0 + 22, y0 + 16), node_title, title_size,
         FG if focus else (205, 202, 195), True)
    y = y0 + 88
    for field in fields:
        value = data["inputs"][field]
        if isinstance(value, list):
            raise ValueError(f"{key}.{field} is a link, not a setting")
        text(d, (x0 + 24, y), field, body_size - 4, MUTED)
        y += body_size + 7
        value = str(value).lower() if isinstance(value, bool) else str(value)
        for row in wrapped(d, value, body_size, x1 - x0 - 48):
            text(d, (x0 + 24, y), row, body_size, ACCENT if focus else FG)
            y += body_size + 7
        y += 14
    ports = {}
    for name in inputs:
        if name not in data["inputs"] or not isinstance(data["inputs"][name], list):
            raise ValueError(f"{key}.{name} is not an API connection")
        py = y + body_size // 2
        d.ellipse((x0 - 7, py - 7, x0 + 7, py + 7), fill=PORT)
        text(d, (x0 + 24, y), name, body_size, FG)
        ports[name] = (x0, py)
        y += body_size + 27
    if output:
        py = y1 - 37
        d.ellipse((x1 - 7, py - 7, x1 + 7, py + 7), fill=PORT)
        text(d, (x1 - 24, py), output, body_size - 3, MUTED, anchor="rm")
        ports["output"] = (x1, py)
    if (fields or inputs) and y > y1 + 5:
        raise ValueError(f"Contents run outside {key}: {y} > {y1}")
    return ports


def link(d: ImageDraw.ImageDraw, graph: dict, source: str, target: str, field: str,
         start: tuple[int, int], end: tuple[int, int], bright=True) -> None:
    """Assert each displayed edge against the source file before drawing it."""
    actual = graph[target]["inputs"][field]
    if not isinstance(actual, list) or actual[0] != source:
        raise ValueError(f"No actual link {source} -> {target}.{field}: {actual}")
    # A smooth cubic is a familiar graph cable; sampled here to avoid dependencies.
    x0, y0 = start
    x1, y1 = end
    dx = max(70, (x1 - x0) * .5)
    points = []
    for i in range(61):
        t, s = i / 60, 1 - i / 60
        points.append((round(s**3*x0 + 3*s*s*t*(x0+dx) + 3*s*t*t*(x1-dx) + t**3*x1),
                       round(s**3*y0 + 3*s*s*t*y0 + 3*s*t*t*y1 + t**3*y1)))
    d.line(points, fill=ACCENT if bright else PORT, width=5)


def loaders(graph: dict, out: Path) -> None:
    im, d = canvas(1, "LTX-2.5 model loader")
    # Compact downstream context is still the real LoRA node, with the real model edge.
    source = node(d, graph, MODEL, (105, 302, 1235, 703), focus=True,
                  fields=("unet_name", "weight_dtype"), output="MODEL", title_size=49, body_size=34)
    dest = node(d, graph, LORA, (1370, 393, 1815, 648),
                inputs=("model",), title_size=20, body_size=30)
    link(d, graph, MODEL, LORA, "model", source["output"], dest["model"])
    im.save(out / "comfy_model.png")

    im, d = canvas(2, "ainvfx-fluid IC-LoRA")
    source = node(d, graph, MODEL, (104, 366, 685, 615),
                  output="MODEL", title_size=33, body_size=29)
    dest = node(d, graph, LORA, (832, 301, 1813, 716), focus=True,
                fields=("lora_name", "strength_model"), inputs=("model",),
                title_size=43, body_size=34)
    link(d, graph, MODEL, LORA, "model", source["output"], dest["model"])
    im.save(out / "comfy_lora.png")


def sampler(graph: dict, out: Path) -> None:
    im, d = canvas(3, "Sampler / generation")
    # Five actual API inputs, in the JSON's own order. Their full settings are
    # available in the workflow; these local input views show the exact titles.
    slots = list(graph[SAMPLER]["inputs"])
    target = node(d, graph, SAMPLER, (1040, 287, 1810, 730), focus=True,
                  inputs=slots, title_size=43, body_size=32)
    for i, name in enumerate(slots):
        source = graph[SAMPLER]["inputs"][name][0]
        y = 285 + i * 91
        ports = node(d, graph, source, (113, y, 826, y + 77),
                     output=None, title_size=31, body_size=26)
        del ports
        point = (826, y + 39)
        d.ellipse((point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=PORT)
        link(d, graph, source, SAMPLER, name, point, target[name])
    im.save(out / "comfy_sampler.png")


def output(graph: dict, out: Path) -> None:
    im, d = canvas(4, "The generated video output")
    create = graph[OUTPUT]["inputs"]["video"][0]
    source = node(d, graph, create, (110, 305, 827, 716),
                  fields=("bit_depth", "color_space", "codec"), output="VIDEO",
                  title_size=45, body_size=31)
    target = node(d, graph, OUTPUT, (1054, 328, 1810, 702), focus=True,
                  fields=("format", "codec"), inputs=("video",),
                  title_size=49, body_size=34)
    link(d, graph, create, OUTPUT, "video", source["output"], target["video"])
    im.save(out / "comfy_output.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflow", type=Path,
                        default=Path(__file__).resolve().parents[2] / "workflows/fluid_element.api.json")
    parser.add_argument("--out", type=Path, default=Path("D:/openHiggsfield/runs/demo_v13/cards"))
    args = parser.parse_args()
    graph = json.loads(args.workflow.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    loaders(graph, args.out)
    sampler(graph, args.out)
    output(graph, args.out)
    for name in ("model", "lora", "sampler", "output"):
        print(args.out / f"comfy_{name}.png")


if __name__ == "__main__":
    main()
