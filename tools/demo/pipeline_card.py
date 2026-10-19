"""The demo's "how it's made" card: the hero shot's own frames at each stage, and the open model doing it.

    python tools/demo/pipeline_card.py   (writes runs/demo_v9/pipeline.png)
"""

import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import ACCENT, BG, FG, MUTED, font  # noqa: E402

W, H = 1920, 1080
HERO = Path("runs/takes/gpu/IMG_6658_fire_burst_s2_c5/composite.mp4")
PLATE = Path("runs/takes/plates/IMG_6658.mp4")
RENDER = Path("runs/demo_v8/how_render.mp4")
MASK = Path("runs/takes/hand/IMG_6658/hand_mask.npz")


def frame(path: Path, k: int) -> np.ndarray:
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == k:
                return fr.to_ndarray(format="rgb24")
    raise IndexError(k)


def portrait(a: np.ndarray, cx: float, cy: float, w: int, h: int, zoom: float = 1.0) -> Image.Image:
    """A w x h window of the picture around (cx, cy) as fractions, zoomed in."""
    im = Image.fromarray(a)
    s = max(w / im.width, h / im.height) * zoom
    im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    x = int(np.clip(cx * im.width - w / 2, 0, im.width - w))
    y = int(np.clip(cy * im.height - h / 2, 0, im.height - h))
    return im.crop((x, y, x + w, y + h))


BW, BH, GAP = 400, 520, 80
plate = frame(PLATE, 54)
alpha = np.load(MASK)["alpha"][54].astype(np.float32)
alpha = alpha / max(1.0, float(alpha.max()))
matte = (plate * 0.35 + alpha[..., None] * np.array([255, 150, 40]) * 0.65).clip(0, 255).astype(np.uint8)
stages = [
    (portrait(plate, 0.5, 0.62, BW, BH, 1.15), "Your clip, your noise", "filmed on a phone; the sound\nsets when, how big, how long"),
    (portrait(matte, 0.5, 0.62, BW, BH, 1.15), "Your hand, found", "MediaPipe + RTMPose track it\nSAM 2.1 mattes it"),
    (portrait(frame(RENDER, 70), 0.52, 0.42, BW, BH, 1.3), "The fire, rendered once", "LTX-2.5 + ainvfx-fluid IC-LoRA\nin ComfyUI, over black"),
    (portrait(frame(HERO, 70), 0.5, 0.6, BW, BH, 1.15), "Carried, lit, timed", "on your hand, lit by its own\nflame, on every hit you make"),
]
img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
d.text((60, 44), "How it's made", font=font(72, True), fill=FG)
d.text((W - 60, 84), "open weights · one rented GPU · a laptop for the rest", font=font(40, True), fill=ACCENT, anchor="rm")
x0 = (W - (4 * BW + 3 * GAP)) // 2
top = 190
for k, (thumb, title, sub) in enumerate(stages):
    x = x0 + k * (BW + GAP)
    img.paste(thumb, (x, top))
    d.rounded_rectangle([x - 4, top - 4, x + BW + 4, top + BH + 4], radius=16, outline=ACCENT if k == 2 else (70, 68, 64), width=6 if k == 2 else 3)
    d.text((x + BW / 2, top + BH + 50), title, font=font(42, True), fill=ACCENT if k == 2 else FG, anchor="mm")
    d.multiline_text((x + BW / 2, top + BH + 96), sub, font=font(32), fill=MUTED, anchor="ma", align="center", spacing=10)
    if k < 3:
        ax, ay = x + BW + 14, top + BH // 2
        d.line([ax, ay, ax + GAP - 28, ay], fill=FG, width=6)
        d.polygon([(ax + GAP - 28, ay - 16), (ax + GAP - 6, ay), (ax + GAP - 28, ay + 16)], fill=FG)
img.save("runs/demo_v9/pipeline.png")
print("runs/demo_v9/pipeline.png")
