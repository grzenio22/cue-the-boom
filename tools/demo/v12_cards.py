"""Make the v12 workflow zooms, contribution, invitation and credit cards.

The workflow panels are enlarged crops of the project's actual graph, with
annotations identifying the model and LoRA. No graph nodes are reconstructed.
All narrated cards keep the bottom 200 pixels free for the spoken captions.

    python tools/demo/v12_cards.py --root D:/openHiggsfield
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import v11_cards as base  # noqa: E402
from build_demo import ACCENT, BG, FG, MUTED  # noqa: E402

W, H = base.W, base.H
LINE = base.LINE
text, fit_text = base.text, base.fit_text


def graph_crop(source: Image.Image, box: tuple[int, int, int, int], scale: float) -> Image.Image:
    crop = source.crop(box)
    return crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)


def highlight(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
              source_box: tuple[int, int, int, int], xy: tuple[int, int], scale: float) -> None:
    x, y = xy
    x0, y0, x1, y1 = source_box
    del x1, y1
    bx0, by0, bx1, by1 = box
    d.rounded_rectangle((x + round((bx0 - x0) * scale), y + round((by0 - y0) * scale),
                         x + round((bx1 - x0) * scale), y + round((by1 - y0) * scale)),
                        radius=14, outline=ACCENT, width=5)


def comfy(root: Path, out: Path) -> None:
    source = Image.open(root / "media/comfyui-scene-workflow.png").convert("RGB")
    im, d = base.canvas()
    text(d, (64, 39), "One ComfyUI workflow", 76, bold=True)
    fit_text(d, (66, 141), "LTX-2.5 · Lightricks     ainvfx-fluid IC-LoRA · AInVFX",
             47, W - 132, ACCENT, True)
    # The group title and both actual loader nodes stay together at twice size.
    box, xy, scale = (15, 580, 882, 778), (92, 257), 2.0
    im.paste(graph_crop(source, box, scale), xy)
    highlight(d, (35, 655, 222, 698), box, xy, scale)
    highlight(d, (591, 655, 820, 698), box, xy, scale)
    d.line((66, 707, W - 66, 707), fill=LINE, width=3)
    text(d, (W / 2, 722), "The app sends this workflow to ComfyUI", 43, bold=True, anchor="ma")
    text(d, (W / 2, 777), "through the Comfy Python SDK.", 43, ACCENT, True, "ma")
    im.save(out / "comfy_model.png")

    im, d = base.canvas()
    text(d, (64, 39), "The workflow's output", 76, bold=True)
    fit_text(d, (65, 141), "Create Video  →  CueSync Scene Shot  →  Save Video",
             46, W - 130, ACCENT, True)
    # The output branch crosses the two groups in the real graph. Enlarge its
    # Scene Shot / Save Video pair together; the Create Video node is inset.
    top_box, top_xy, top_scale = (898, 580, 1590, 773), (130, 242), 2.4
    im.paste(graph_crop(source, top_box, top_scale), top_xy)
    highlight(d, (914, 623, 1214, 770), top_box, top_xy, top_scale)
    highlight(d, (1264, 622, 1573, 762), top_box, top_xy, top_scale)
    inset_box, inset_xy, inset_scale = (583, 944, 800, 998), (130, 699), 2.5
    im.paste(graph_crop(source, inset_box, inset_scale), inset_xy)
    highlight(d, (592, 951, 785, 993), inset_box, inset_xy, inset_scale)
    text(d, (760, 728), "One generated fire element", 45, ACCENT, True)
    text(d, (762, 785), "Ready to perform again with your voice", 36)
    im.save(out / "comfy_output.png")


def pipeline(root: Path, out: Path) -> None:
    base.pipeline(root, out)
    im = Image.open(out / "pipeline.png").convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 276), fill=BG)
    fit_text(d, (64, 37), "Separating generation from performance", 73, W - 128, bold=True)
    text(d, (65, 139), "One LTX effect from ComfyUI becomes a reusable instrument.", 48, ACCENT, True)
    text(d, (65, 208), "Your voice controls when it hits and how big it plays.", 44)
    # Two-line captions begin at y=840. The primary stage labels stay visible;
    # their explanatory sublabels duplicate the narration and yield that space.
    d.rectangle((0, 832, W, H), fill=BG)
    im.save(out / "pipeline.png")


def try_it(out: Path) -> None:
    im, d = base.canvas()
    text(d, (W / 2, 46), "Cue the Boom", 98, bold=True, anchor="ma")
    text(d, (W / 2, 181), "Try the included LTX effect on your laptop,", 58, ACCENT, True, "ma")
    text(d, (W / 2, 256), "no GPU needed", 58, ACCENT, True, "ma")
    text(d, (274, 356), "From the cloned repo:", 32, MUTED)
    d.rounded_rectangle((270, 407, 1650, 624), radius=22,
                        fill=(22, 22, 27), outline=LINE, width=3)
    text(d, (330, 431), '$ pip install "./cuesync[server]"', 57, bold=True)
    text(d, (330, 526), "$ cuesync studio", 57, ACCENT, True)
    text(d, (W / 2, 699), "github.com/grzenio22/cue-the-boom", 57, bold=True, anchor="ma")
    text(d, (W / 2, 775), "#OpenSourceHiggsfield", 49, ACCENT, True, "ma")
    im.save(out / "try_it.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--out", type=Path, help="Defaults to ROOT/runs/demo_v12/cards")
    args = parser.parse_args()
    out = args.out or args.root / "runs/demo_v12/cards"
    out.mkdir(parents=True, exist_ok=True)
    comfy(args.root, out)
    pipeline(args.root, out)
    try_it(out)
    base.credits(out)
    for path in sorted(out.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
