"""Build media/pipeline.jpg from one shipped shot's matching pipeline stages.

    python tools/demo/pipeline_card.py [--frame 70] [--output path/to/pipeline.jpg]

The plate, track, matte and composite use the same zero-based frame. For the
single-hit base shot, the raw element frame is derived from the receipt offset.
No generation, relighting or enhancement is done here: only crop/resize, a
labelled matte/landmark overlay, and explanatory text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H = 1920, 1080
BG, ACCENT, FG, MUTED = (11, 11, 15), (255, 176, 32), (244, 241, 234), (163, 159, 151)
BW, BH, GAP = 400, 520, 80
HAND_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
              (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def frame(path: Path, index: int) -> np.ndarray:
    if index < 0:
        raise ValueError("Frame numbers must be nonnegative")
    with av.open(str(path)) as container:
        for i, decoded in enumerate(container.decode(video=0)):
            if i == index:
                return decoded.to_ndarray(format="rgb24")
    raise ValueError(f"{path}: frame {index} is outside the video")


def portrait(a: np.ndarray, cx: float = 0.5, cy: float = 0.60) -> Image.Image:
    """Use the exact same crop for the plate, its matte overlay and the result."""
    im = Image.fromarray(a)
    scale = max(BW / im.width, BH / im.height) * 1.15
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    x = int(np.clip(cx * im.width - BW / 2, 0, im.width - BW))
    y = int(np.clip(cy * im.height - BH / 2, 0, im.height - BH))
    return im.crop((x, y, x + BW, y + BH))


def element_detail(a: np.ndarray) -> Image.Image:
    """Fit the visible fire plus a margin on black without cutting off its shape."""
    ys, xs = np.nonzero(a.max(axis=2) > 20)
    if not len(xs):
        raise ValueError("Selected element frame has no visible fire; choose a later --frame")
    pad = max(24, round(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.12))
    box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
           min(a.shape[1], int(xs.max()) + pad + 1), min(a.shape[0], int(ys.max()) + pad + 1))
    im = Image.fromarray(a).crop(box)
    scale = min(BW / im.width, BH / im.height)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (BW, BH), "black")
    out.paste(im, ((BW - im.width) // 2, (BH - im.height) // 2))
    return out


def element_frame(receipt: dict, index: int) -> int:
    """Base burst only: refuse to imply a single raw frame for blended/held hits."""
    bending = receipt["bending"]
    hits = bending.get("hits", [])
    if (receipt["effect"]["behaviour"] != "burst" or len(hits) != 1
            or bending.get("held") or bending.get("holds") or bending.get("charge")):
        raise ValueError("Pipeline card requires the single-hit base shot without a held or charged cue")
    hit = hits[0]
    ignition = int(bending["ignition"])
    offset = int(bending["offset"])
    if offset != int(hit["frame"]) - ignition:
        raise ValueError("Receipt offset disagrees with the selected hit")
    if index < int(hit["frame"]):
        return index - offset
    size = float(hit.get("size", 1.0))
    if size <= 0:
        raise ValueError("Receipt hit size must be positive")
    return ignition + round((index - int(hit["frame"])) / size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=Path, default=ROOT / "samples/made_ahead/bending/hand_palm__boom__fire_burst")
    parser.add_argument("--plate", type=Path, default=ROOT / "samples/plates/hand_palm.mp4")
    parser.add_argument("--frame", type=int, default=70)
    parser.add_argument("--output", type=Path, default=ROOT / "media/pipeline.jpg")
    args = parser.parse_args()
    receipt = json.loads((args.shot / "receipt.json").read_text(encoding="utf-8"))
    track = json.loads((args.shot / receipt["bending"]["track"]).read_text(encoding="utf-8"))
    element_index = element_frame(receipt, args.frame)
    plate = frame(args.plate, args.frame)
    composite = frame(args.shot / "composite.mp4", args.frame)
    if plate.shape != composite.shape:
        raise ValueError("Plate and composite dimensions must match for comparable crops")
    with np.load(args.shot / receipt["bending"]["mask"], allow_pickle=False) as masks:
        alpha = masks["alpha"][args.frame].astype(np.float32) / 255.0
        mask_source = str(masks["src"][args.frame])
    if alpha.shape != plate.shape[:2]:
        raise ValueError("Hand mask and plate dimensions must match")
    matte = Image.fromarray((plate * 0.45 + alpha[..., None] * np.array([255, 150, 40]) * 0.55)
                            .clip(0, 255).astype(np.uint8))
    landmarks = np.asarray(track["landmarks"][args.frame])
    if landmarks.shape == (21, 3) and np.isfinite(landmarks).all():
        overlay = ImageDraw.Draw(matte)
        for start, end in HAND_EDGES:
            overlay.line([tuple(landmarks[start, :2]), tuple(landmarks[end, :2])], fill=FG, width=4)
        for x, y in landmarks[:, :2]:
            overlay.ellipse((x - 5, y - 5, x + 5, y + 5), fill=ACCENT)
    sources = set(track.get("src", []))
    tracker = "MediaPipe + RTMPose tracking" if {"mp", "rtm"} <= sources else "Stored hand tracking"
    matte_label = "SAM 2.1 matte" if mask_source == "sam2" else f"Hand matte ({mask_source})"
    stages = [
        (portrait(plate), "Your clip, your noise", "phone footage; your sound sets\nwhen, how big and how long"),
        (portrait(np.asarray(matte)), "Your hand, found", f"{tracker}\n{matte_label} + landmarks"),
        (element_detail(frame(args.shot / "element.mp4", element_index)), "The fire, rendered once",
         "LTX-2.5 + ainvfx-fluid IC-LoRA\nin ComfyUI, over black"),
        (portrait(composite), "Carried, lit, timed", "tracked fire and local lighting\ncomposited on the CPU"),
    ]
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((60, 44), "How it's made", font=font(72, True), fill=FG)
    draw.text((W - 60, 84), "Open weights. Generate once. Perform again.",
              font=font(32, True), fill=ACCENT, anchor="rm")
    x0, top = (W - (4 * BW + 3 * GAP)) // 2, 190
    for k, (thumb, title, sub) in enumerate(stages):
        x = x0 + k * (BW + GAP)
        img.paste(thumb, (x, top))
        draw.rounded_rectangle((x - 4, top - 4, x + BW + 4, top + BH + 4), radius=16,
                               outline=ACCENT if k == 2 else (70, 68, 64), width=6 if k == 2 else 3)
        title_font = font(38, True)
        for size in range(38, 23, -1):
            title_font = font(size, True)
            if draw.textlength(title, font=title_font) <= BW:
                break
        draw.text((x + BW / 2, top + BH + 50), title, font=title_font,
                  fill=ACCENT if k == 2 else FG, anchor="mm")
        draw.multiline_text((x + BW / 2, top + BH + 96), sub, font=font(27), fill=MUTED,
                            anchor="ma", align="center", spacing=10)
        if k < 3:
            ax, ay = x + BW + 14, top + BH // 2
            draw.line((ax, ay, ax + GAP - 28, ay), fill=FG, width=6)
            draw.polygon(((ax + GAP - 28, ay - 16), (ax + GAP - 6, ay), (ax + GAP - 28, ay + 16)), fill=FG)
    draw.text((W / 2, 985), "2.5D approximation: tracked placement, a hand matte and local lighting.",
              font=font(27), fill=MUTED, anchor="mm")
    draw.text((W / 2, 1030), f"Matching source frame {args.frame}  |  raw fire frame {element_index} (detail)",
              font=font(23), fill=MUTED, anchor="mm")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output, quality=92, subsampling=0)
    print(f"{args.shot}: composite/plate/matte frame {args.frame}; element frame {element_index}; "
          f"receipt offset {receipt['bending']['offset']}; mask {mask_source}")
    print(f"{args.output}: {img.width}x{img.height}")


if __name__ == "__main__":
    main()
