"""media/hand_hero.jpg from the author's own takes: three 16:9 bands of the portrait shots, labelled with their cue.

    python tools/demo/hand_hero_card.py
"""

import av
from PIL import Image, ImageDraw, ImageFont

G = "runs/takes/gpu"
SHOTS = [  # composite, frame, top of the 16:9 band (share of the height), label
    (f"{G}/IMG_6658_fire_burst_s2_c5/composite.mp4", 70, 0.42, "BOOM  (a fireball off your palm)"),
    (f"{G}/IMG_6653_fire_breath_c5/composite.mp4", 44, 0.29, "fwoooOOOM  (a flamethrower from a fingertip)"),
    ("runs/takes/rhythms/pf_x4/composite.mp4", 62, 0.42, "pf pf pf PF  (the same render, performed again)"),
]
W, LABEL = 645, 40
f = ImageFont.truetype("arialbd.ttf", 22)
out = Image.new("RGB", (W * len(SHOTS) + 2 * (len(SHOTS) - 1), round(W * 9 / 16) + LABEL), (11, 11, 15))
for k, (src, n, top, label) in enumerate(SHOTS):
    with av.open(src) as c:
        im = next(fr.to_image() for i, fr in enumerate(c.decode(video=0)) if i == n)
    band = round(im.width * 9 / 16)
    y = round(top * im.height)
    im = im.crop((0, y, im.width, y + band)).resize((W, round(W * 9 / 16)), Image.Resampling.LANCZOS)
    x = k * (W + 2)
    out.paste(im, (x, 0))
    ImageDraw.Draw(out).text((x + 10, im.height + 8), label, font=f, fill=(255, 176, 32))
out.save("media/hand_hero.jpg", quality=88)
print("media/hand_hero.jpg", out.size)
