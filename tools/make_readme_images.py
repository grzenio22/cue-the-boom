"""The README's images from the sample scene shots: media/hero.jpg (six shots on real footage, a frame each,
labelled with their cue), media/stand-ins.jpg (six shots on the stand-in plates) and media/how-it-was-made.jpg (a
frame of one shot's breakdown: plate, guide, render, shot).

    python tools/make_readme_images.py samples/made_ahead/scene
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuesync" / "src"))
from cuesync.media import read_video  # noqa: E402

REAL = [  # shot on real footage, frames after its hit, label
    ("real_candles_glasses__fwoooom", 30, "“fwoooOOOM”"),
    ("real_lanterns__kaboom", 10, "“KA-BOOM”"),
    ("real_candle_stool__pf", 12, "“pf.”"),
    ("real_trex_photo__fwoooom", 24, "“fwoooOOOM”"),
    ("real_firepit__fwoooom", 24, "“fwoooOOOM”"),
    ("real_jack_photo__kaboom", 12, "“KA-BOOM”"),
]
STAND_INS = [  # shot, frames after its hit, label
    ("jackolantern_kitchen__ka-boom__fire_burst", 12, "“KA-BOOM”"),
    ("toy_desk__fwoooom__fire_breath", 12, "“fwoooOOOM”"),
    ("pumpkin_kitchen__fwoooom__flame_plume", 22, "drawn + “fwoooom”"),
    ("candle_shelf__fwoooom__wick_flare", 22, "“fwoooom” on a candle"),
    ("candle_snap__clip__blow_out", 22, "a snap in the clip"),
    ("mug_counter__hissss__steam_jet", 22, "“hissss”"),
]
BREAKDOWN = ("jackolantern_kitchen__ka-boom__fire_burst", 8)


def font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def frame_after_hit(d: Path, name: str, after: int) -> Image.Image:
    hit = json.loads((d / "receipt.json").read_text())["alignment"]["hit_frame"]
    frames = read_video(d / name).frames
    return Image.fromarray(frames[min(max(hit + after, 0), len(frames) - 1)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("shots", nargs="?", default="samples/made_ahead/scene")
    ap.add_argument("-o", "--out", default="media")
    a = ap.parse_args()
    root, out = Path(a.shots), Path(a.out)
    f = font(30)
    tw, th = 480, 270
    real = Image.new("RGB", (tw * 3, th * 2))
    d = ImageDraw.Draw(real)
    for i, (shot, after, label) in enumerate(REAL):
        im = frame_after_hit(root / shot, "final.mp4", after)
        im = im.crop((0, 0, im.width, int(im.width * 9 / 16)))  # the shot, without the cue ribbon under it
        x, y = (i % 3) * tw, (i // 3) * th
        real.paste(im.resize((tw, th), Image.Resampling.LANCZOS), (x, y))
        d.text((x + 14, y + th - 16), label, font=f, fill=(255, 255, 255), anchor="ls", stroke_width=3, stroke_fill=(0, 0, 0))
    real.save(out / "hero.jpg", quality=88)
    w, h = 360, 640
    hero = Image.new("RGB", (w * len(STAND_INS), h))
    d = ImageDraw.Draw(hero)
    for i, (shot, after, label) in enumerate(STAND_INS):
        im = frame_after_hit(root / shot, "final.mp4", after)
        im = im.crop((0, 0, im.width, int(im.width * 16 / 9)))  # the shot, without the cue ribbon under it
        hero.paste(im.resize((w, h), Image.Resampling.LANCZOS), (i * w, 0))
        d.text((i * w + 14, h - 30), label, font=f, fill=(255, 255, 255), anchor="ls", stroke_width=3, stroke_fill=(0, 0, 0))
    hero.save(out / "stand-ins.jpg", quality=88)
    shot, after = BREAKDOWN
    bd = frame_after_hit(root / shot, "breakdown.mp4", after)
    bd.resize((540, int(540 * bd.height / bd.width)), Image.Resampling.LANCZOS).save(out / "how-it-was-made.jpg", quality=88)
    print(f"{out / 'hero.jpg'}, {out / 'stand-ins.jpg'}, {out / 'how-it-was-made.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
