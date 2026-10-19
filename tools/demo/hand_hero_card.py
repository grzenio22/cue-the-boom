"""media/hand_hero.jpg from the author's own takes: three 16:9 bands of the portrait shots, labelled with their cue
(a punch's BOOM, a flame carried through the house in one take, and a render performed again).

    python tools/demo/hand_hero_card.py
"""

import av
from PIL import Image, ImageDraw, ImageFont

G, G2 = "runs/takes/gpu", "runs/takes/gpu2"
SHOTS = [  # composite, frame, top of the 16:9 band (share of the height), label
    (f"{G2}/IMG_6663_fire_burst_comp/composite.mp4", 66, 0.27, "BOOM  (a fireball off your fist)"),
    (f"{G2}/IMG_6668_flame_plume_s2_again/composite.mp4", 120, 0.38, "fffff  (a flame in your palm, one take)"),
    ("runs/takes/rhythms/pf_x4/composite.mp4", 64, 0.36, "pf pf pf PF  (one render, performed again)"),
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
