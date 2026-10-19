"""media/elements.jpg: the elements a hand throws beyond fire, from the author's own takes, each a portrait band labelled
with its cue. python tools/demo/elements_card.py [shot.mp4 frame top label]...

With no arguments, the shots in SHOTS (made by runs/takes/try_made.py)."""

import sys

import av
from PIL import Image, ImageDraw, ImageFont

SHOTS = [  # composite, frame, top of the band (share of the height), label
    ("runs/takes/made/arc_6667/composite.mp4", 45, 0.10, "zzzzz  (lightning from a fingertip)"),
    ("runs/takes/made/orb_6661/composite.mp4", 60, 0.40, "shhhhh  (water held over the palm)"),
    ("runs/takes/made/airball_6661/composite.mp4", 74, 0.40, "hooooo  (a ball of air)"),
    ("runs/takes/made/expl_6666/composite.mp4", 45, 0.05, "BOOOOM  (an explosion off the palm)"),
]
W, H, LABEL = 480, 640, 40  # each band: a 3:4 window of the portrait shot


def main() -> None:
    args = sys.argv[1:]
    shots = [(args[i], int(args[i + 1]), float(args[i + 2]), args[i + 3]) for i in range(0, len(args), 4)] or SHOTS
    f = ImageFont.truetype("arialbd.ttf", 18)
    out = Image.new("RGB", (W * len(shots) + 2 * (len(shots) - 1), H + LABEL), (11, 11, 15))
    for k, (src, n, top, label) in enumerate(shots):
        with av.open(src) as c:
            im = next(fr.to_image() for i, fr in enumerate(c.decode(video=0)) if i == n)
        band = round(im.width * H / W)
        y = min(round(top * im.height), im.height - band)
        im = im.crop((0, y, im.width, y + band)).resize((W, H), Image.Resampling.LANCZOS)
        x = k * (W + 2)
        out.paste(im, (x, 0))
        ImageDraw.Draw(out).text((x + 10, H + 10), label, font=f, fill=(255, 176, 32))
    out.save("media/elements.jpg", quality=88)
    print("media/elements.jpg", out.size)


if __name__ == "__main__":
    main()
