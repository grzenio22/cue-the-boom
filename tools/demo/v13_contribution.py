"""Add narrated stage highlights to the existing v12 contribution card.

The v12 card is the pixel source: its heading, four source images, labels and
caption-safe area stay in place. Only stage emphasis changes between frames.

    python tools/demo/v13_contribution.py --root D:/openHiggsfield
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from build_demo import ACCENT, BG, FG
from v11_cards import LINE, fit_text


SIZE = (1920, 1080)
PANEL_WIDTH = 400
PANEL_HEIGHT = 440
PANEL_TOP = 308
PANEL_LEFTS = (70, 530, 990, 1450)
LABEL_Y = PANEL_TOP + PANEL_HEIGHT + 30
CAPTION_TOP = 832
LABELS = ("Your clip + sound", "Track + matte", "One LTX render", "Perform + light")
STAGES = {"asset": 2, "voice": 0, "result": 3}


def border(draw: ImageDraw.ImageDraw, index: int, color, width: int) -> None:
    x = PANEL_LEFTS[index]
    draw.rounded_rectangle(
        (x - 4, PANEL_TOP - 4, x + PANEL_WIDTH + 4, PANEL_TOP + PANEL_HEIGHT + 4),
        radius=15, outline=color, width=width,
    )


def title(draw: ImageDraw.ImageDraw, index: int, color) -> None:
    x = PANEL_LEFTS[index]
    draw.rectangle((x - 18, 765, x + PANEL_WIDTH + 18, CAPTION_TOP - 1), fill=BG)
    fit_text(draw, (x + PANEL_WIDTH / 2, LABEL_Y), LABELS[index], 39,
             PANEL_WIDTH + 35, color, True, "ma")


def render(source: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    original = Image.open(source).convert("RGB")
    if original.size != SIZE:
        raise ValueError(f"Expected {SIZE}, got {original.size}: {source}")

    overview = out / "pipeline_overview.png"
    shutil.copy2(source, overview)
    paths = [overview]

    # v12 starts with the generated asset selected. Establish a neutral version
    # of that one border/label, using the exact original geometry and typography.
    neutral = original.copy()
    draw = ImageDraw.Draw(neutral)
    border(draw, 2, BG, 6)
    border(draw, 2, LINE, 3)
    title(draw, 2, FG)

    for name, selected in STAGES.items():
        image = neutral.copy()
        for index, x in enumerate(PANEL_LEFTS):
            if index == selected:
                continue
            bounds = (x - 8, PANEL_TOP - 8, x + PANEL_WIDTH + 8, CAPTION_TOP)
            crop = image.crop(bounds)
            image.paste(Image.blend(crop, Image.new("RGB", crop.size, BG), 0.64), bounds[:2])
        draw = ImageDraw.Draw(image)
        border(draw, selected, ACCENT, 6)
        title(draw, selected, ACCENT)
        path = out / f"pipeline_{name}.png"
        image.save(path)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.source or args.root / "runs/demo_v12/cards/pipeline.png"
    out = args.out or args.root / "runs/demo_v13/cards"
    for path in render(source, out):
        print(path)


if __name__ == "__main__":
    main()
