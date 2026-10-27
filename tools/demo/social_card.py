"""Build the YouTube thumbnail and GitHub's social preview: the same palm and the same stored LTX-2.5 render, twice,
a small fire under a small "pf" and a big one under a huge "BOOM!", from the shipped pf, pf, BOOM performance.

    python tools/demo/social_card.py                      # media/social_preview.jpg, 1280x640 (GitHub)
    python tools/demo/social_card.py --thumbnail OUT.jpg  # also the 1280x720 YouTube thumbnail

Upload the preview under the repository's Settings > General > Social preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SHOT = ROOT / "samples/made_ahead/bending/hand_palm__pfpfboom__again/final.mp4"  # hits on frames 29, 38 (pf) and 64
PF, BOOM = 36, 67      # each fire near its largest
PICTURE_ROWS = 1920    # the shot without its export footer
INK, FIRE, BG = (246, 243, 236), (255, 140, 26), (10, 10, 14)


def font(size: int, black: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (("ariblk.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if black else ("arialbd.ttf", "DejaVuSans-Bold.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def frames(path: Path, wanted: tuple[int, ...]) -> dict[int, Image.Image]:
    out = {}
    with av.open(str(path)) as container:
        for i, decoded in enumerate(container.decode(video=0)):
            if i in wanted:
                out[i] = decoded.to_image().crop((0, 0, decoded.width, PICTURE_ROWS))
            if len(out) == len(wanted):
                return out
    raise ValueError(f"{path}: frames {wanted} are not all in the video")


def card(size: tuple[int, int], shots: dict[int, Image.Image], strip: str, top: float, title: bool) -> Image.Image:
    w, h = size
    band = round(h * 0.17)
    pw, ph = (w - 4) // 2, h - band
    out = Image.new("RGB", (w, h), BG)
    for k, (frame, word, size_) in enumerate(((PF, "pf", 0.13), (BOOM, "BOOM!", 0.22))):
        src = shots[frame]
        ch = round(src.width * ph / pw)
        y0 = round(top * src.height)
        out.paste(src.crop((0, y0, src.width, y0 + ch)).resize((pw, ph), Image.Resampling.LANCZOS), (k * (pw + 4), 0))
        d = ImageDraw.Draw(out)
        d.text((k * (pw + 4) + pw // 2, round(ph * (0.17 if k else 0.15))), word, font=font(round(h * size_), True),
               fill=FIRE, anchor="mm", stroke_width=max(4, round(h * 0.012)), stroke_fill=(0, 0, 0))
    d = ImageDraw.Draw(out)
    face = font(round(band * 0.42), True)
    while d.textlength(strip, font=face) > w - 80:
        face = font(face.size - 1, True)
    d.text((w // 2, h - band // 2), strip, font=face, fill=INK, anchor="mm")
    if title:
        d.text((24, 22), "Cue the Boom", font=font(30), fill=INK, stroke_width=3, stroke_fill=(0, 0, 0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "media/social_preview.jpg")
    parser.add_argument("--thumbnail", type=Path, help="also write the 1280x720 YouTube thumbnail here")
    parser.add_argument("--top", type=float, default=0.22, help="top of each crop, as a fraction of the picture's height")
    args = parser.parse_args()
    shots = frames(SHOT, (PF, BOOM))
    outputs = [(args.output, (1280, 640), "ONE LTX-2.5 RENDER. YOUR VOICE PLAYS IT.", True)]
    if args.thumbnail:
        outputs.append((args.thumbnail, (1280, 720), "SAME AI FIRE. YOUR VOICE PLAYS IT.", False))
    for path, size, strip, title in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        card(size, shots, strip, args.top, title).save(path, quality=92, subsampling=0)
        print(f"{path}: {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
