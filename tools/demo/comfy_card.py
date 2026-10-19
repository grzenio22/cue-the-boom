"""The demo's ComfyUI beat: the LTX-2.5 group of the scene workflow, the model, the IC-LoRA and the output lit, the
rest dimmed, and what runs where. Writes runs/demo_v9/comfy.png (1920x1080).

    python tools/demo/comfy_card.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from build_demo import ACCENT, BG, FG, font  # noqa: E402

W, H = 1920, 1080
src = Image.open("media/comfyui-scene-workflow.png").convert("RGB")
crop = src.crop((0, src.height // 2, int(0.55 * src.width), src.height)).crop((15, 40, 880, 500))  # group 3: LTX-2.5 + ainvfx-fluid
S = 2.0
crop = crop.resize((round(crop.width * S), round(crop.height * S)), Image.Resampling.LANCZOS)
LIT = [  # node boxes in the crop (before scaling), tag
    ((27, 80, 198, 107), "LTX-2.5 (Lightricks)"),
    ((585, 80, 790, 107), "ainvfx-fluid IC-LoRA (AInVFX)"),
    ((585, 380, 720, 407), "the render"),
]
dim = Image.blend(crop, Image.new("RGB", crop.size, (0, 0, 0)), 0.55)
for (x0, y0, x1, y1), _ in LIT:
    box = tuple(round(v * S) for v in (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    dim.paste(crop.crop(box), box[:2])
img = Image.new("RGB", (W, H), BG)
ox, oy = (W - dim.width) // 2, 84
img.paste(dim, (ox, oy))
d = ImageDraw.Draw(img)
d.text((60, 18), "One ComfyUI workflow, open weights", font=font(52, True), fill=FG)
for (x0, y0, x1, y1), tag in LIT:
    bx0, by0, bx1, by1 = (ox + round((x0 - 6) * S), oy + round((y0 - 6) * S), ox + round((x1 + 6) * S), oy + round((y1 + 6) * S))
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=14, outline=ACCENT, width=6)
    f = font(36, True)
    tw = d.textlength(tag, font=f)
    tx = min(bx0, W - 40 - tw - 32)
    d.rounded_rectangle([tx, by1 + 10, tx + tw + 32, by1 + 62], radius=10, fill=ACCENT)
    d.text((tx + 16, by1 + 36), tag, font=f, fill=BG, anchor="lm")
d.text((W // 2, H - 40), "GPU: render the fire once   →   CPU: perform it again, as often as you like", font=font(44, True),
       fill=ACCENT, anchor="mm")
img.save("runs/demo_v9/comfy.png")
print("runs/demo_v9/comfy.png")
