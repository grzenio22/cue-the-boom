"""'Same footage, different performance': two composites side by side, synced on the hit, each with its
label and a live envelope ribbon so the voice shaping the effect is readable with the sound off."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RIBBON_H = 64
RIBBON_BG = (16, 16, 20)
RIBBON_FILL = (255, 170, 40)
RIBBON_FUTURE = (70, 70, 80)
PLAYHEAD = (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _pad_to(frames: np.ndarray, lead: int, total: int) -> np.ndarray:
    parts = []
    if lead > 0:
        parts.append(np.repeat(frames[:1], lead, axis=0))
    parts.append(frames)
    tail = total - lead - frames.shape[0]
    if tail > 0:
        parts.append(np.repeat(frames[-1:], tail, axis=0))
    return np.concatenate(parts, axis=0)[:total]


def _decorate(frame: np.ndarray, label: str, env: np.ndarray, t: int, total: int) -> np.ndarray:
    h, w = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    fsize = max(18, w // 14)
    draw.text((w // 2, fsize // 2 + 8), label, font=_font(fsize), fill=(255, 255, 255), anchor="mt",
              stroke_width=max(2, fsize // 12), stroke_fill=(0, 0, 0))
    top = h - RIBBON_H
    draw.rectangle([0, top, w, h], fill=RIBBON_BG)
    n = max(total - 1, 1)
    for i in range(total):
        v = float(env[i]) if i < env.size else 0.0
        x0, x1 = int(i / n * (w - 1)), int((i + 1) / n * (w - 1))
        bar = int(v * (RIBBON_H - 8))
        if bar > 0:
            draw.rectangle([x0, h - 4 - bar, max(x0, x1 - 1), h - 4], fill=RIBBON_FILL if i <= t else RIBBON_FUTURE)
    px = int(t / n * (w - 1))
    draw.line([px, top, px, h], fill=PLAYHEAD, width=2)
    return np.asarray(img)


def side_by_side(
    a: np.ndarray,
    b: np.ndarray,
    a_anchor: int,
    b_anchor: int,
    env_a: np.ndarray,
    env_b: np.ndarray,
    labels: tuple[str, str] | list[str],
) -> np.ndarray:
    """Return (L, H, 2W, 3) frames with both anchors on the same output frame."""
    if a.shape[1:3] != b.shape[1:3]:
        h, w = a.shape[1:3]
        b = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.Resampling.LANCZOS)) for f in b])
    lead = max(a_anchor, b_anchor)
    lead_a, lead_b = lead - a_anchor, lead - b_anchor
    total = max(a.shape[0] + lead_a, b.shape[0] + lead_b)
    pa, pb = _pad_to(a, lead_a, total), _pad_to(b, lead_b, total)
    ea = np.concatenate([np.full(lead_a, env_a[0] if len(env_a) else 0.0), np.asarray(env_a, float)])
    eb = np.concatenate([np.full(lead_b, env_b[0] if len(env_b) else 0.0), np.asarray(env_b, float)])
    out = np.empty((total, a.shape[1], a.shape[2] * 2, 3), dtype=np.uint8)
    for t in range(total):
        out[t, :, : a.shape[2]] = _decorate(pa[t], labels[0], ea, t, total)
        out[t, :, a.shape[2]:] = _decorate(pb[t], labels[1], eb, t, total)
    return out
