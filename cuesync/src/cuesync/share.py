"""A finished shot as a vertical reel to share: 9:16 around where it happened, your cue's word popping on each hit, a
small tag, and the shot's own sound (your voice, its Foley)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SIZE = (720, 1280)  # (w, h): a phone's reel
POP_FRAMES = 14  # the word on screen after each hit, popping in and settling
TAG = "Cue the Boom"
WORDS = {"fire_burst": "BOOM", "fireball": "BOOM", "fire_breath": "fwoooOOOM", "flame_plume": "fwoooOOOM",
         "wick_flare": "fwoooOOOM", "steam_jet": "hissss", "smoke_puff": "pf", "blow_out": "pf",
         "explosion": "BOOM", "lightning_arc": "zzzzz", "lightning_ball": "zzzzz",
         "water_orb": "sploosh", "water_whip": "shhhhh", "water_surge": "sploosh",
         "water_ribbon": "shhhhh", "water_shield": "shhhhh", "water_jet": "sploosh",
         "air_ball": "hoooooo", "air_blast": "HOO!",
         "earth_rock": "rrrrr", "earth_shatter": "CRACK"}


def _font(size: int):
    from PIL import ImageFont

    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def hits_of(receipt: dict) -> list[int]:
    """The shot's hit frames, whatever made it (element bending, a scene shot, an element shot)."""
    b = receipt.get("bending") or {}
    if b.get("hits"):
        return [int(h["frame"]) for h in b["hits"]]
    if receipt.get("hits"):
        return [int(h["frame"]) for h in receipt["hits"]]
    g = receipt.get("guide") or {}
    return [int(g["anchor_frame"])] if "anchor_frame" in g else []


def share_reel(shot_dir: Path, out: Path) -> Path:
    """Write ``out`` (mp4, 720x1280): the shot's clean composite cropped to 9:16 round where it was tapped (a portrait
    shot as it is), the cue's word big on each hit, the tag at the bottom, the shot's sound."""
    from PIL import Image, ImageDraw

    from .media import read_audio, read_video, write_video

    shot_dir = Path(shot_dir)
    rc = json.loads((shot_dir / "receipt.json").read_text(encoding="utf-8"))
    frames = read_video(shot_dir / "composite.mp4").frames
    audio = read_audio(shot_dir / "composite.mp4", 48000)
    n, h, w = frames.shape[:3]
    cw = min(w, int(round(h * SIZE[0] / SIZE[1])))
    px = float(((rc.get("inputs") or {}).get("point") or [0.5, 0.5])[0])
    x0 = int(np.clip(px * w - cw / 2, 0, w - cw))
    # Custom effects have no known sound: keep their picture and audio without inventing a fire caption.
    word = WORDS.get((rc.get("effect") or {}).get("id", ""))
    hits = hits_of(rc)
    small = _font(int(SIZE[0] * 0.05))
    reel = np.empty((n, SIZE[1], SIZE[0], 3), np.uint8)
    for t in range(n):
        im = Image.fromarray(cv2.resize(frames[t, :, x0:x0 + cw], SIZE, interpolation=cv2.INTER_AREA))
        d = ImageDraw.Draw(im)
        since = [t - k for k in hits if 0 <= t - k < POP_FRAMES]
        if since and word:  # pops in a little big, settles, fades
            a = min(since) / POP_FRAMES
            font = _font(int(SIZE[0] * (0.2 - 0.04 * min(1.0, a * 3))))
            alpha = 1.0 if a < 0.6 else 1.0 - (a - 0.6) / 0.4
            bw = d.textlength(word, font=font)
            x, y = (SIZE[0] - bw) / 2, SIZE[1] * 0.68
            for dx, dy in ((-3, 3), (3, 3), (0, 4)):  # a dark edge, so it reads over fire and over black
                d.text((x + dx, y + dy), word, font=font, fill=(0, 0, 0))
            d.text((x, y), word, font=font, fill=(255, int(176 + 60 * (1 - alpha)), 32))
            if alpha < 1.0:
                im = Image.blend(Image.fromarray(cv2.resize(frames[t, :, x0:x0 + cw], SIZE, interpolation=cv2.INTER_AREA)),
                                 im, alpha)
                d = ImageDraw.Draw(im)
        tw = d.textlength(TAG, font=small)
        d.text(((SIZE[0] - tw) / 2 + 2, SIZE[1] * 0.93 + 2), TAG, font=small, fill=(0, 0, 0))
        d.text(((SIZE[0] - tw) / 2, SIZE[1] * 0.93), TAG, font=small, fill=(255, 176, 32))
        reel[t] = np.asarray(im)
    del frames
    write_video(out, reel, fps=24, audio=audio, crf=20)
    return Path(out)
