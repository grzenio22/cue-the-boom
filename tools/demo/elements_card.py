"""Build media/elements.jpg from five shipped gallery shots.

    python tools/demo/elements_card.py [--output path/to/elements.jpg]

Frame numbers are zero-based. Each crop keeps the hand and effect, omitting the
shot's bottom timeline. Only crop and resize are applied to the video pixels.
"""

import argparse
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SHOTS = [  # sample directory, frame, crop top in source pixels, cue, description
    # Hit 35: the fist has closed and the fireball has developed by frame 46.
    ("hand_fist__boooom__explosion", 46, 400, "BOOOOM", "Closing-fist explosion"),
    # Hit 44, held 45-75: a full channel from finger to wall at frame 56.
    ("hand_point__zzzzz__lightning", 56, 150, "zzzzz", "Pointing-finger lightning"),
    # Hits 47 and 70, held 48-77: channel and palm light at frame 49.
    ("hand_palm__zzzzz__lightning", 49, 240, "zzzzz", "Bathroom lightning"),
    # Same cue: the ball's arcs and light on the palm are clear at frame 60.
    ("hand_palm__zzzzz__lightning_ball", 60, 240, "zzzzz", "Bathroom ball lightning"),
    # Held 2-54, final hit 43: the orb is formed over the palm at frame 30.
    ("hand_open__shhhhh__water_orb", 30, 400, "shhhhh", "Water orb"),
]
W, H, LABEL, GAP = 432, 576, 96, 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "media/elements.jpg")
    args = parser.parse_args()
    for face in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            cue_font = ImageFont.truetype(face, 32)
            label_font = ImageFont.truetype(face, 23)
            break
        except OSError:
            continue
    else:
        cue_font = label_font = ImageFont.load_default()
    out = Image.new("RGB", (W * len(SHOTS) + GAP * (len(SHOTS) - 1), H + LABEL), (11, 11, 15))
    draw = ImageDraw.Draw(out)
    for k, (name, n, top, cue, label) in enumerate(SHOTS):
        shot = ROOT / "samples/made_ahead/bending" / name
        receipt = json.loads((shot / "receipt.json").read_text(encoding="utf-8"))
        with av.open(str(shot / "final.mp4")) as c:
            im = next((fr.to_image() for i, fr in enumerate(c.decode(video=0)) if i == n), None)
        if im is None:
            raise ValueError(f"{name}: frame {n} is outside final.mp4")
        band = round(im.width * H / W)
        if top < 0 or top + band > im.height:
            raise ValueError(f"{name}: crop is outside the frame")
        im = im.crop((0, top, im.width, top + band)).resize((W, H), Image.Resampling.LANCZOS)
        x = k * (W + GAP)
        out.paste(im, (x, 0))
        draw.text((x + 14, H + 8), cue, font=cue_font, fill=(255, 176, 32))
        draw.text((x + 14, H + 52), label, font=label_font, fill=(238, 238, 240))
        hits = [hit["frame"] for hit in receipt["bending"]["hits"]]
        print(f"{name}/final.mp4: frame {n}; receipt hits {hits}; holds {receipt['bending']['holds']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output, quality=92, subsampling=0)
    print(f"{args.output}: {out.width}x{out.height}")


if __name__ == "__main__":
    main()
