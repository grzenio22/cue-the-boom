"""Render the v11 demo's 1080p proof, workflow, innovation, invitation and credit cards.

The workflow is the project's actual ComfyUI graph. The four pipeline images
are frames from the author's plate, hand matte, LTX render and finished shot.
All cards reserve their bottom 150 pixels for spoken-line captions.

    python tools/demo/v11_cards.py --root D:/openHiggsfield
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import ACCENT, BG, FG, MUTED, font  # noqa: E402

W, H = 1920, 1080
SAFE_BOTTOM = 930
LINE = (67, 65, 61)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def text(d: ImageDraw.ImageDraw, xy, value: str, size: int, fill=FG,
         bold: bool = False, anchor: str = "la") -> None:
    d.multiline_text(xy, value, font=font(size, bold), fill=fill, anchor=anchor,
                     spacing=12, align="center" if anchor[0] == "m" else "left")


def fit_text(d: ImageDraw.ImageDraw, xy, value: str, size: int, max_width: int,
             fill=FG, bold: bool = False, anchor: str = "la") -> None:
    while max(d.textlength(s, font=font(size, bold)) for s in value.split("\n")) > max_width:
        size -= 1
    text(d, xy, value, size, fill, bold, anchor)


def frame(path: Path, k: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for i, fr in enumerate(container.decode(video=0)):
            if i == k:
                return fr.to_ndarray(format="rgb24")
    raise IndexError(f"Frame {k} not found in {path}")


def portrait(a: np.ndarray, cx: float, cy: float, w: int, h: int,
             zoom: float = 1.0) -> Image.Image:
    im = Image.fromarray(a)
    scale = max(w / im.width, h / im.height) * zoom
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    x = int(np.clip(cx * im.width - w / 2, 0, im.width - w))
    y = int(np.clip(cy * im.height - h / 2, 0, im.height - h))
    return im.crop((x, y, x + w, y + h))


def proof(root: Path, out: Path) -> None:
    """A clearly labelled still of the one element that the next clips perform."""
    im, d = canvas()
    text(d, (64, 38), "One generated element", 79, bold=True)
    text(d, (67, 144), "LTX-2.5 + ainvfx-fluid IC-LoRA · raw render over black", 38, ACCENT, True)
    raw = frame(root / "runs/demo_v11/clips/how_raw.mp4", 70)
    thumb = portrait(raw, .53, .55, 840, 650, 2.0)
    im.paste(thumb, (62, 239))
    d.rounded_rectangle((58, 235, 906, 893), radius=18, outline=LINE, width=3)
    text(d, (976, 267), "Generate the effect once.", 58, bold=True)
    text(d, (976, 360), "Perform it again\nwith your voice.", 61, ACCENT, True)
    d.line((978, 567, W - 68, 567), fill=LINE, width=3)
    text(d, (978, 601), "Three performances", 37, MUTED)
    for y, value in ((668, "BOOM"), (742, "pf · pf · BOOM"), (816, "pf pf pf PF")):
        text(d, (980, y), value, 49, bold=True)
    im.save(out / "proof.png")


def comfy(root: Path, out: Path) -> None:
    im, d = canvas()
    text(d, (64, 35), "LTX-2.5 is the core video model", 70, bold=True)
    text(d, (66, 126), "ComfyUI workflow  ·  open weights only", 39, ACCENT, True)

    source = Image.open(root / "media/comfyui-scene-workflow.png").convert("RGB")
    graph = source.crop((0, source.height // 2, int(0.55 * source.width), source.height))
    graph = graph.crop((15, 40, 880, 500))
    scale = 1.5
    graph = graph.resize((round(graph.width * scale), round(graph.height * scale)), Image.Resampling.LANCZOS)
    dim = Image.blend(graph, Image.new("RGB", graph.size), 0.35)
    boxes = [(27, 80, 198, 107), (585, 80, 790, 107), (585, 380, 720, 407)]
    ox, oy = 60, 207
    for box in boxes:
        box = tuple(round(v * scale) for v in (box[0] - 6, box[1] - 6, box[2] + 6, box[3] + 6))
        dim.paste(graph.crop(box), box[:2])
    im.paste(dim, (ox, oy))
    for index, box in enumerate(boxes, 1):
        x0, y0, x1, y1 = (ox + round((box[0] - 6) * scale), oy + round((box[1] - 6) * scale),
                          ox + round((box[2] + 6) * scale), oy + round((box[3] + 6) * scale))
        d.rounded_rectangle((x0, y0, x1, y1), radius=12, outline=ACCENT, width=5)
        d.ellipse((x0 - 17, y0 - 25, x0 + 25, y0 + 17), fill=ACCENT)
        text(d, (x0 + 4, y0 - 4), str(index), 27, BG, True, "mm")

    x = 1410
    text(d, (x, 232), "1   LTX-2.5", 43, ACCENT, True)
    text(d, (x, 295), "Lightricks\nCore video model", 33)
    text(d, (x, 421), "2   ainvfx-fluid", 40, ACCENT, True)
    text(d, (x, 479), "IC-LoRA · AInVFX", 34)
    text(d, (x, 584), "3   The fire render", 39, ACCENT, True)
    text(d, (x, 643), "Generated once", 33)
    d.line((x, 731, W - 60, 731), fill=LINE, width=3)
    text(d, (x, 755), "Comfy SDK", 40, bold=True)
    text(d, (x, 817), "sends this workflow\nto ComfyUI", 33)
    im.save(out / "comfy.png")


def pipeline(root: Path, out: Path) -> None:
    im, d = canvas()
    fit_text(d, (64, 37), "A generated VFX element becomes an instrument.", 68, W - 128, bold=True)
    text(d, (65, 132), "Your voice sets when and how big.", 46, ACCENT, True)
    text(d, (65, 194), "Perform it again without regenerating.", 44)

    bw, bh, gap, top = 400, 440, 60, 308
    plate = frame(root / "runs/takes/plates/IMG_6658.mp4", 54)
    alpha = np.load(root / "runs/takes/hand/IMG_6658/hand_mask.npz")["alpha"][54].astype(np.float32)
    alpha /= max(1.0, float(alpha.max()))
    matte = (plate * 0.35 + alpha[..., None] * np.array([255, 150, 40]) * 0.65).clip(0, 255).astype(np.uint8)
    stages = [
        (portrait(plate, .5, .62, bw, bh, 1.15), "Your clip + sound", "The performance"),
        (portrait(matte, .5, .62, bw, bh, 1.15), "Track + matte", "Your hand carries it"),
        (portrait(frame(root / "runs/demo_v11/clips/how_raw.mp4", 70), .52, .42, bw, bh, 1.3),
         "One LTX render", "The reusable element"),
        (portrait(frame(root / "runs/demo_v11/clips/proof_boom/composite.mp4", 70),
                  .5, .48, bw, bh), "Perform + light", "Timed and sized by voice"),
    ]
    x0 = (W - (4 * bw + 3 * gap)) // 2
    for k, (thumb, title, sub) in enumerate(stages):
        x = x0 + k * (bw + gap)
        im.paste(thumb, (x, top))
        d.rounded_rectangle((x - 4, top - 4, x + bw + 4, top + bh + 4), radius=15,
                            outline=ACCENT if k == 2 else LINE, width=6 if k == 2 else 3)
        fit_text(d, (x + bw / 2, top + bh + 30), title, 39, bw + 35,
                 ACCENT if k == 2 else FG, True, "ma")
        fit_text(d, (x + bw / 2, top + bh + 88), sub, 31, bw + 25, MUTED, anchor="ma")
        if k < 3:
            ax, ay = x + bw + 10, top + bh / 2
            d.line((ax, ay, ax + gap - 29, ay), fill=FG, width=5)
            d.polygon(((ax + gap - 31, ay - 12), (ax + gap - 12, ay),
                       (ax + gap - 31, ay + 12)), fill=FG)
    im.save(out / "pipeline.png")


def try_it(out: Path) -> None:
    im, d = canvas()
    text(d, (W / 2, 65), "Cue the Boom", 100, bold=True, anchor="ma")
    text(d, (W / 2, 204), "Try it in two minutes · no GPU", 60, ACCENT, True, "ma")
    text(d, (274, 290), "From the cloned repo:", 32, MUTED)
    box = (270, 342, 1650, 584)
    d.rounded_rectangle(box, radius=22, fill=(22, 22, 27), outline=LINE, width=3)
    text(d, (330, 371), '$ pip install "./cuesync[server]"', 57, bold=True)
    text(d, (330, 471), "$ cuesync studio", 57, ACCENT, True)
    text(d, (W / 2, 674), "github.com/grzenio22/cue-the-boom", 57, bold=True, anchor="ma")
    text(d, (W / 2, 791), "#OpenSourceHiggsfield", 49, ACCENT, True, "ma")
    im.save(out / "try_it.png")


def credits(out: Path) -> None:
    im, d = canvas()
    text(d, (W / 2, 34), "Cue the Boom · credits", 66, bold=True, anchor="ma")
    rows = [
        "LTX-2.5 · Lightricks    ainvfx-fluid IC-LoRA · AInVFX    Gemma text encoder · Google",
        "ComfyUI + Comfy SDK · Comfy Org",
        "Hands: MediaPipe · Google    RTMPose · OpenMMLab    SAM 2.1 · Meta",
        "The author's own hand, voice and phone    Stock footage: Pexels, Pixabay",
        "Voices · rykenn, unfa and others (CC0, Freesound)",
        "Narration · Kokoro-82M, a synthetic voice    Foley · synthesised per shot",
        "Open weights only · generation self-hosted on one rented GPU",
    ]
    for i, row in enumerate(rows):
        fit_text(d, (W / 2, 160 + i * 73), row, 37, W - 120, anchor="ma")
    d.line((120, 727, W - 120, 727), fill=LINE, width=3)
    text(d, (W / 2, 763), "Fire + explosion fireball: LTX-2.5 renders", 40, ACCENT, True, "ma")
    text(d, (W / 2, 829), "Lightning + water: procedural CPU effects", 40, ACCENT, True, "ma")
    im.save(out / "credits.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--out", type=Path, help="Defaults to ROOT/runs/demo_v11/cards")
    args = parser.parse_args()
    out = args.out or args.root / "runs/demo_v11/cards"
    out.mkdir(parents=True, exist_ok=True)
    proof(args.root, out)
    comfy(args.root, out)
    pipeline(args.root, out)
    try_it(out)
    credits(out)
    for path in sorted(out.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
