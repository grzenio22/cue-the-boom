"""Build media/hand_hero.jpg from the shipped hand gallery.

    python tools/demo/hand_hero_card.py [--output path/to/hand_hero.jpg]

To reproduce the README card with the complete flame and hand in each panel:
    python tools/demo/hand_hero_card.py --frames 70 120 41 --contain-crop 1 0 700 1080 1600 --contain-crop 2 0 400 1080 1500 --contain-crop 3 0 100 1080 1550

Paths default to this checkout, regardless of the working directory. Frame
numbers are zero-based. Only crop, resize and optional dark bars affect pixels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SHOTS = [
    ("hand_palm__boom__fire_burst", "BOOM  (fire in your palm)"),
    ("hand_walk__fffff__flame_plume", "fffff  (a flame in your palm, one take)"),
    ("hand_fist__boom__fire_burst", "BOOM  (a fireball as your hand closes)"),
]
W, LABEL, GAP = 645, 40, 2
DEFAULT_CONTAIN = [(1, 0, 700, 1080, 1600), (2, 0, 400, 1080, 1500), (3, 0, 100, 1080, 1550)]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def frame(path: Path, index: int) -> Image.Image:
    if index < 0:
        raise ValueError("Frame numbers must be nonnegative")
    with av.open(str(path)) as container:
        for i, decoded in enumerate(container.decode(video=0)):
            if i == index:
                return decoded.to_image()
    raise ValueError(f"{path}: frame {index} is outside the video")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, default=ROOT / "samples/made_ahead/bending")
    parser.add_argument("--output", type=Path, default=ROOT / "media/hand_hero.jpg")
    parser.add_argument("--frames", type=int, nargs=3, default=(70, 120, 41),
                        metavar=("PALM", "WALK", "FIST"))
    parser.add_argument("--tops", type=float, nargs=3, default=(0.27, 0.38, 0.36),
                        metavar=("PALM", "WALK", "FIST"),
                        help="Top of each 16:9 crop, as a fraction of source height")
    parser.add_argument("--contain-crop", type=int, nargs=5, action="append",
                        metavar=("PANEL", "LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Fit this source-pixel crop inside panel 1, 2 or 3 with dark bars; repeat per panel")
    args = parser.parse_args()
    contained = {}
    for panel, left, top, right, bottom in (args.contain_crop if args.contain_crop is not None else DEFAULT_CONTAIN):
        if panel not in range(1, len(SHOTS) + 1) or panel in contained:
            parser.error("--contain-crop requires a unique panel number from 1 to 3")
        contained[panel] = (left, top, right, bottom)
    height = round(W * 9 / 16)
    out = Image.new("RGB", (W * len(SHOTS) + GAP * (len(SHOTS) - 1), height + LABEL), (11, 11, 15))
    draw = ImageDraw.Draw(out)
    for k, ((name, label), index, top) in enumerate(zip(SHOTS, args.frames, args.tops)):
        source = args.gallery / name / "composite.mp4"
        im = frame(source, index)
        band, y = round(im.width * 9 / 16), round(top * im.height)
        box = contained.get(k + 1, (0, y, im.width, y + band))
        left, upper, right, bottom = box
        if not (0 <= left < right <= im.width and 0 <= upper < bottom <= im.height):
            raise ValueError(f"{name}: crop {box} is outside the {im.width}x{im.height}px frame")
        crop = im.crop(box)
        if k + 1 in contained:
            scale = min(W / crop.width, height / crop.height)
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                               Image.Resampling.LANCZOS)
        else:
            crop = crop.resize((W, height), Image.Resampling.LANCZOS)
        x = k * (W + GAP)
        out.paste(crop, (x + (W - crop.width) // 2, (height - crop.height) // 2))
        face = font(22)
        for size in range(22, 13, -1):
            face = font(size)
            if draw.textlength(label, font=face) <= W - 20:
                break
        draw.text((x + 10, height + LABEL / 2), label, font=face, fill=(255, 176, 32), anchor="lm")
        print(f"{source}: frame {index}; crop {box}; fitted size {crop.size}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output, quality=92, subsampling=0)
    print(f"{args.output}: {out.width}x{out.height}")


if __name__ == "__main__":
    main()
