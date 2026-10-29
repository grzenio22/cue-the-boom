"""Cut the short native clip for the X post: 1080x1350 (4:5), about 26 s, 24 fps, H.264 + AAC, captions burned in,
made to work muted in a phone feed. Everything in it is shipped material; nothing is generated.

    python tools/demo/social_cut.py                                   # with the stand-in live clip
    python tools/demo/social_cut.py --scan --live D:/takes/live.mp4   # where its fire and loud sounds are
    python tools/demo/social_cut.py --live D:/takes/live.mp4 --live-start 4.2 [--live-crop 0.1,0.2,0.9,0.95]

Writes D:/openHiggsfield_tmp/social/social_cut.mp4, social_cut_sheet.jpg (one frame a second) and social_cut.json
(segments, loudness, where each caption and noise word sits), and prints each segment's loudness (FFmpeg ebur128).

Segments: the live take (cover-cropped, "LIVE WEBCAM · NO RENDER RUNNING", its loudest sound popped as BOOM) - the
pf · pf · BOOM proof - four strangers' voices on the same render, in a 2x2 grid - water - lightning on the author's own zzzzz -
the dinosaur and the lanterns (letterboxed over a blurred copy) - an end card. X shows frame 0 as the poster, so it
must already show fire: when the live take's first frame has none (--poster auto), the cut opens on half a second of
the proof's BOOM first. The live window starts at --live-start (default: picked from the clip's loudest sound,
preferring a start that already shows fire). --live-crop is the part of the upright live picture to use, as fractions
x0,y0,x1,y1 (default: all of it; for the stand-in, the Studio's camera panel): the largest 4:5 window in it slides
along its long side to keep the fire in the picture and out from under the caption, which goes at the bottom or top.
Every segment carries a small stamp saying what it is, worded as the 16:9 demo's beats (runs/demo_v19/storyboard.json).
Sound: each segment levelled towards -16 LUFS by gain alone, never past -1.7 dBTP before encoding (no limiter), so the
quiet pf stays quiet beside the BOOM; the whole cut is checked at about -16 LUFS and at most -1.5 dBTP after AAC.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_demo as bd  # noqa: E402  (fonts, pop, mark, wave, Clip, read_window, cover; puts cuesync on the path)
from cuesync.media import read_audio  # noqa: E402

ROOT = HERE.parents[1]
W, H, FPS, SR = 1080, 1350, bd.FPS, bd.SR
OUT_DIR = Path("D:/openHiggsfield_tmp/social")
STANDIN = Path("D:/openHiggsfield_tmp/rehearsal/standin-phone.mp4")
STANDIN_CROP = (0.3734, 0.1444, 0.6273, 0.9458)  # the Studio's camera panel in that 1280x720 screen capture
CLIPS = ROOT / "runs/demo_v11/clips"
MADE = ROOT / "samples/made_ahead/bending"
V19 = ROOT / "runs/demo_v19/storyboard.json"
PROOF = MADE / "hand_palm__pfpfboom__again/composite.mp4"  # the fire as the gallery plays it
PROOF_VOICE = CLIPS / "proof_pf_pf_boom.wav"  # the author's recorded voice, as v19's proof beats play it
PROOF_TOP = 420                                # the 4:5 window's top row in the 1080x1920 proof: fire and hand
CRACKLE = ROOT / "runs/demo_v10/credits_crackle.wav"
STRANGERS = [("sadiquecat", "Boom"), ("unfa", "Ka-Boom!"), ("huw2k8", "Powf"), ("jedimichael", "Boom")]

INK, ACCENT, MUTED, BG = (255, 255, 255), bd.ACCENT, (190, 186, 178), (11, 11, 15)
CAPTION_SIZE = 88       # Arial Bold: a 63 px cap height on the 1080-wide frame
STAMP_SIZE = 30
STAMP_ZONE = 106        # rows at the top kept for the stamp
BOTTOM_SAFE = 72        # rows at the bottom kept clear of text (the feed puts its mute button and timer there)
SIDE = 24
TP_BEFORE_AAC = -1.7    # dBTP ceiling for each segment's gain (checked again after AAC)
TARGET_LUFS = -15.8     # a little over -16: the proof's BOOM caps its own gain lower
FIRE_MIN = 0.002        # share of the frame that must be fire for frame 0 to count as showing it

# The 16:9 demo's own wording (runs/demo_v19/storyboard.json, by beat id), used when that file is not present.
STAMPS = {
    "water": "The author's hand · sample noise (synthetic) · water made on the CPU",
    "lightning": "The author's hand, filmed 28 Oct, and the author's own zzzzz · lightning made on the CPU",
    "dino": "Real stock footage (Pexels) · sample noise (synthetic) · LTX-2.5 render",
    "lanterns": "Pixabay footage, first frame held · real CC0 voices (Freesound) · LTX-2.5 fire",
    "strangers": "One LTX-2.5 render · voices: Sadiquecat, unfa, huw2k8, Jedimichael (CC0, Freesound)",
}
LIGHTNING_28 = ROOT / "runs/demo_v21/shots/18766d702104/composite.mp4"  # the author's own zzzzz, as the studio made it
PROOF_STAMP ="The author's hand + recorded voice · one stored LTX-2.5 fire · no new render"
PHONE_STAMP = "Filmed on a phone · the author, live · the stored LTX-2.5 fire, no render running"  # as v20_build
STANDIN_STAMP = "STAND-IN, not the live take · a Studio screen capture · the stored LTX-2.5 fire"
LANTERN_MARKS = [[1.0, 0.42, 0.72, "1"], [1.67, 0.64, 0.58, "2"], [2.38, 0.43, 0.3, "3"]]
LANTERN_POPS = [[1.0, "pf"], [1.67, "pf"], [2.38, "BOOM"]]


def v19_beats() -> dict[str, dict]:
    try:
        board = json.loads(V19.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return {b.get("id"): b for b in (board["beats"] if isinstance(board, dict) else board)}


# ---------------------------------------------------------------- text


@lru_cache(maxsize=64)
def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return bd.font(size, bold)


@lru_cache(maxsize=16)
def stamp_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_MEASURE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = (line + " " + word).strip()
            if line and _MEASURE.textlength(trial, font=f) > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


@dataclass
class Caption:
    lines: list[str]
    font: ImageFont.FreeTypeFont
    rect: tuple[int, int, int, int]
    line_h: int


def caption_layout(text: str, where: str | int, size: int = CAPTION_SIZE) -> Caption:
    """A caption's lines and its dark box: at the bottom (above the feed's controls), the top (under the stamp), or
    with its top at a given row."""
    f = font(size)
    pad_x, pad_y = 24, 22
    lines = wrap(text, f, W - 2 * SIDE - 2 * pad_x)
    line_h = round(size * 1.12)
    w = max(_MEASURE.textlength(s, font=f) for s in lines) + 2 * pad_x
    h = line_h * len(lines) + 2 * pad_y
    if where == "bottom":
        y0 = H - BOTTOM_SAFE - h
    elif where == "top":
        y0 = STAMP_ZONE + 8
    else:
        y0 = int(where)
    x0 = round((W - w) / 2)
    return Caption(lines, f, (x0, int(y0), round(x0 + w), int(y0 + h)), line_h)


def draw_caption(img: Image.Image, cap: Caption, alpha: float = 1.0) -> Image.Image:
    if not cap.lines or alpha <= 0:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = cap.rect
    d.rounded_rectangle(cap.rect, radius=22, fill=(0, 0, 0, round(165 * alpha)))
    y = y0 + (y1 - y0 - cap.line_h * len(cap.lines)) / 2 + cap.line_h / 2
    for line in cap.lines:
        d.text((W / 2, y), line, font=cap.font, fill=INK + (round(255 * alpha),), anchor="mm",
               stroke_width=4, stroke_fill=(0, 0, 0, round(255 * alpha)))
        y += cap.line_h
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


def stamp_layout(text: str) -> tuple[list[str], ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    size = STAMP_SIZE
    f = stamp_font(size)
    lines = wrap(text, f, W - 2 * SIDE - 28)
    while len(lines) > 2 and size > 22:
        size -= 1
        f = stamp_font(size)
        lines = wrap(text, f, W - 2 * SIDE - 28)
    if len(lines) == 2:  # two lines of like length, split after a " · " where it can be
        words = text.split(" ")
        cuts = [(max(_MEASURE.textlength(" ".join(words[:k]), font=f), _MEASURE.textlength(" ".join(words[k:]), font=f))
                 - (60 if words[k - 1] == "·" or words[k] == "·" else 0), k) for k in range(1, len(words))]
        k = min(cuts)[1]
        if words[k] == "·":
            k += 1
        trial = [" ".join(words[:k]).removesuffix(" ·"), " ".join(words[k:])]
        if max(_MEASURE.textlength(s, font=f) for s in trial) <= W - 2 * SIDE - 28:
            lines = trial
    lh = round(size * 1.22)
    w = max(_MEASURE.textlength(s, font=f) for s in lines) + 28
    h = lh * len(lines) + 12
    x0 = round((W - w) / 2)
    return lines, f, (x0, 12, round(x0 + w), 12 + h)


def draw_stamp(img: Image.Image, text: str) -> Image.Image:
    """What the shot is made of: small, at the top centre, on a dim pill so it reads on a bright kitchen too."""
    if not text:
        return img
    lines, f, rect = stamp_layout(text)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(rect, radius=14, fill=(0, 0, 0, 140))
    lh = (rect[3] - rect[1] - 12) / len(lines)
    for k, line in enumerate(lines):
        d.text((W / 2, rect[1] + 6 + lh * (k + 0.5)), line, font=f, fill=(236, 232, 224, 255), anchor="mm",
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")


@lru_cache(maxsize=32)
def _pop_extent(word: str, scale: float) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in _MEASURE.textbbox((0, 0), word, font=font(int(120 * scale)), anchor="mm", stroke_width=7))


def pop_box(word: str, cx: int, cy: int, scale: float = 1.15) -> tuple[int, int, int, int]:
    """Where bd.pop's word sits (it starts at 1.3x and settles at 1x)."""
    x0, y0, x1, y1 = _pop_extent(word, scale)
    return cx + x0, cy + y0, cx + x1, cy + y1


# ---------------------------------------------------------------- pictures


def fire_mask(a: np.ndarray) -> np.ndarray:
    """Flame orange with its red channel clipped (the white-hot core is ringed by it): a sunlit wooden floor, skin
    and a lit kitchen come close in hue, but stay under 250."""
    a = a.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r >= 250) & (g > 110) & (b < 150) & (r - b > 110)


def cover_region(frame: np.ndarray, region: tuple[float, float, float, float] | None, focus=None) -> np.ndarray:
    """The largest 4:5 window inside region (fractions of the frame; default all of it), centred on focus
    (fractions of the frame) or on the region's middle."""
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = region or (0.0, 0.0, 1.0, 1.0)
    rx0, ry0, rx1, ry1 = x0 * fw, y0 * fh, x1 * fw, y1 * fh
    rw, rh = rx1 - rx0, ry1 - ry0
    if rw / rh > W / H:
        ww, wh = rh * W / H, rh
    else:
        ww, wh = rw, rw * H / W
    cx, cy = ((focus[0] * fw, focus[1] * fh) if focus else ((rx0 + rx1) / 2, (ry0 + ry1) / 2))
    left = min(max(cx - ww / 2, rx0), rx1 - ww)
    top = min(max(cy - wh / 2, ry0), ry1 - wh)
    return frame[int(round(top)):int(round(top + wh)), int(round(left)):int(round(left + ww))]


def to_frame(a: np.ndarray) -> np.ndarray:
    if a.shape[:2] == (H, W):
        return np.ascontiguousarray(a)
    return np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((W, H), Image.Resampling.LANCZOS))


def index_at(times: np.ndarray, start: float, i: int) -> int:
    k = int(np.searchsorted(times, start + i / FPS + 1e-4, side="right")) - 1
    return min(max(k, 0), len(times) - 1)


def portrait_clip(src: Path, start: float, dur: float, top: int) -> tuple[Callable[[int], Image.Image], np.ndarray]:
    """A portrait clip's 4:5 window from row `top` (frame width kept), and its own sound."""
    c = bd.Clip(str(src), start, dur)
    fh, fw = c.frames.shape[1:3]
    rows = round(fw * H / W)
    top = min(max(0, top), fh - rows)
    frames = [to_frame(f[top:top + rows]) for f in c.frames]
    times = c.times
    sound = c.sound()
    del c
    return (lambda i: Image.fromarray(frames[index_at(times, start, i)])), sound


def letterbox_clip(src: Path, start: float, dur: float, y_pic: int):
    """A landscape clip at full width, over a dimmed, blurred copy of itself filling the 4:5 frame."""
    c = bd.Clip(str(src), start, dur)
    pics, backs = [], []
    for f in c.frames:
        im = Image.fromarray(f)
        pics.append(im.resize((W, round(W * im.height / im.width)), Image.Resampling.LANCZOS))
        back = bd.cover(f, W // 4, H // 4).filter(ImageFilter.GaussianBlur(9))
        backs.append(ImageEnhance.Brightness(back).enhance(0.42))
    times, sound = c.times, c.sound()
    del c

    def picture(i: int) -> Image.Image:
        k = index_at(times, start, i)
        img = backs[k].resize((W, H), Image.Resampling.BILINEAR)
        img.paste(pics[k], (0, y_pic))
        return img
    return picture, sound, pics[0].size


# ---------------------------------------------------------------- the live take


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    rot = int(float(v.get("tags", {}).get("rotate", 0)))
    for side in v.get("side_data_list", []):
        if "rotation" in side:
            rot = int(float(side["rotation"]))
    w, h = int(v["width"]), int(v["height"])
    return dict(size=(h, w) if rot % 180 else (w, h), duration=float(data["format"]["duration"]),
                audio=any(s["codec_type"] == "audio" for s in data["streams"]))


def decode(path: Path, start: float, dur: float, short_side: int) -> tuple[np.ndarray, np.ndarray]:
    """Frames (upright, as FFmpeg applies the rotation) at 24 fps, the short side at most short_side, and the mono
    sound at 48 kHz, of [start, start + dur)."""
    info = probe(path)
    w, h = info["size"]
    s = min(1.0, short_side / min(w, h))
    ow, oh = max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)
    n = round(dur * FPS)
    seek = ["-ss", f"{start:.3f}"] if start > 0 else []
    v = subprocess.run(["ffmpeg", "-v", "error", *seek, "-i", str(path), "-t", f"{dur + 0.2:.3f}", "-an",
                        "-vf", f"fps={FPS},scale={ow}:{oh}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True).stdout
    frames = np.frombuffer(v, np.uint8).reshape(-1, oh, ow, 3)
    if len(frames) < n:
        frames = np.concatenate([frames, np.repeat(frames[-1:], n - len(frames), axis=0)])
    return frames[:n], decode_audio(path, start, n / FPS) if info["audio"] else np.zeros(round(n / FPS * SR), np.float32)


def decode_audio(path: Path, start: float, dur: float) -> np.ndarray:
    seek = ["-ss", f"{start:.3f}"] if start > 0 else []
    a = subprocess.run(["ffmpeg", "-v", "error", *seek, "-i", str(path), "-t", f"{dur:.3f}", "-vn", "-ac", "1",
                        "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    out = np.zeros(round(dur * SR), np.float32)
    a = np.frombuffer(a, np.float32)
    out[:min(out.size, a.size)] = a[:out.size]
    return out


def onsets(audio: np.ndarray, floor_db: float = 14.0, quiet_s: float = 0.3) -> list[dict]:
    """Where a loud sound starts (s) and its 20 ms peak level: rising floor_db over the median after quiet_s below
    it (as v20_build finds a phone take's BOOMs), and within 30 dB of the loudest (digital silence has no median)."""
    hop = SR // 50
    if audio.size < hop:
        return []
    level = 20 * np.log10(np.sqrt(np.mean(audio[:audio.size // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-9)
    threshold = max(float(np.median(level)) + floor_db, float(level.max()) - 30)
    found, quiet, k = [], int(quiet_s * 50), 0
    while k < level.size:
        if level[k] >= threshold and (k < quiet or (level[max(0, k - quiet):k] < threshold).all()):
            end = k
            while end < level.size and level[end] >= threshold - 6:
                end += 1
            found.append(dict(at=round(k / 50, 2), peak_db=round(float(level[k:end].max()), 1)))
            k = end + 1
        else:
            k += 1
    return found


def loudest_at(audio: np.ndarray) -> float:
    """The start of the loudest sound: back from its loudest 10 ms to where it rose within 12 dB of it."""
    hop = SR // 100
    level = 20 * np.log10(np.sqrt(np.mean(audio[:audio.size // hop * hop].reshape(-1, hop) ** 2, axis=1)) + 1e-9)
    k = int(np.argmax(level))
    while k > 0 and level[k - 1] > level.max() - 12:
        k -= 1
    return k / 100


def scan_live(path: Path, region, dur: float) -> dict:
    """The whole clip at a small size: its fire per frame (inside region) and its loud sounds; and the start of a
    dur-second window holding the loudest sound, preferring one whose first frame already shows fire."""
    info = probe(path)
    w, h = info["size"]
    s = 270 / min(w, h)
    ow, oh = round(w * s / 2) * 2, round(h * s / 2) * 2
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
                             f"fps={FPS},scale={ow}:{oh}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    fire, thumbs = [], []
    try:
        size = ow * oh * 3
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            f = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
            x0, y0, x1, y1 = region or (0.0, 0.0, 1.0, 1.0)
            win = f[round(y0 * oh):round(y1 * oh), round(x0 * ow):round(x1 * ow)]
            fire.append(float(fire_mask(win).mean()))
            if len(fire) % (FPS // 2) == 1:
                thumbs.append((len(fire) - 1, Image.fromarray(np.ascontiguousarray(win))))
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
    audio = decode_audio(path, 0.0, info["duration"]) if info["audio"] else np.zeros(1, np.float32)
    found = onsets(audio)
    fire = np.asarray(fire)
    lit = fire >= FIRE_MIN
    nd = int(round(dur * FPS))
    loud = max(found, key=lambda o: o["peak_db"]) if found else None
    best = None
    for f0 in range(max(1, len(fire) - nd + 1)):
        off = loud["at"] - f0 / FPS if loud else 0.5
        if not 0.2 <= off <= dur - 0.8:  # the loudest sound, early enough for its fire to show
            continue
        score = float(lit[f0:f0 + nd].mean()) + (0.15 if lit[f0] else 0.0) - 0.02 * abs(off - 0.5)
        if best is None or score > best[0]:
            best = (score, f0)
    f0 = best[1] if best else 0
    why = (f"{'the loudest sound at ' + format(loud['at'], '.2f') + ' s inside, ' if loud else 'no loud sound; '}"
           f"fire on {lit[f0:f0 + nd].mean():.0%} of the window, "
           f"{'already on its first frame' if lit[f0] else 'none on its first frame (the cut opens on the proof BOOM)'}")
    return dict(fire=fire, onsets=found, start=round(f0 / FPS, 3), why=why, thumbs=thumbs, duration=info["duration"])


# ---------------------------------------------------------------- segments


@dataclass
class Segment:
    key: str
    n: int
    picture: Callable[[int], Image.Image]
    audio: np.ndarray
    caption: str
    stamp: str = ""
    caption_at: str | int = "auto"    # "auto" (bottom, or the top when the fire is under the bottom), or a row
    pops: list = field(default_factory=list)
    pop_at: tuple[int, int] | None = None   # None: placed beside the fire
    decorate: Callable[[Image.Image, ImageDraw.ImageDraw, int], None] | None = None
    target: float = TARGET_LUFS
    source: str = ""
    caption_alpha: Callable[[int], float] | None = None
    over: Callable[[Image.Image, int], Image.Image] | None = None
    report: dict = field(default_factory=dict)


def place(seg: Segment) -> None:
    """Settle the caption (bottom unless the fire sits under it more than under the top) and the noise words (beside
    the fire, off the caption and the stamp, and as clear as can be of any fire while they show)."""
    q = 4
    masks = [fire_mask(np.asarray(seg.picture(i).resize((W // q, H // q), Image.Resampling.BILINEAR)))
             for i in range(seg.n)]
    union = np.any(masks, axis=0)

    def fire_in(rect, ms) -> int:
        x0, y0, x1, y1 = (max(0, v // q) for v in rect)
        return int(sum(m[y0:y1, x0:x1].sum() for m in ms))

    if seg.caption_at == "auto":
        bottom, top = caption_layout(seg.caption, "bottom"), caption_layout(seg.caption, "top")
        seg.caption_at = "top" if fire_in(top.rect, masks) < fire_in(bottom.rect, masks) * 0.5 else "bottom"
    cap = caption_layout(seg.caption, seg.caption_at)
    keep_off = [cap.rect, stamp_layout(seg.stamp)[2] if seg.stamp else (0, 0, 0, 0)]
    if seg.pops and seg.pop_at is None:
        showing = sorted({i for at, _ in seg.pops for i in range(int(float(at) * FPS), int((float(at) + 0.8) * FPS))
                          if 0 <= i < seg.n})
        word = max((w for _, w in seg.pops), key=len)
        ys, xs = np.nonzero(union)
        fc = (xs.mean() * q, ys.mean() * q) if xs.size else (W / 2, H / 2)
        best = None
        for cy in range(140, H - 120, 20):
            for cx in range(150, W - 149, 20):
                box = pop_box(word, cx, cy)
                if box[0] < 10 or box[2] > W - 10 or box[1] < STAMP_ZONE or box[3] > H - 20:
                    continue
                if any(box[0] < r[2] and r[0] < box[2] and box[1] < r[3] and r[1] < box[3] for r in keep_off):
                    continue
                cost = 50 * fire_in(box, [masks[i] for i in showing]) + fire_in(box, [union]) * 4 \
                    + 0.6 * float(np.hypot(cx - fc[0], cy - fc[1]))
                if best is None or cost < best[0]:
                    best = (cost, (cx, cy))
        seg.pop_at = best[1] if best else (W // 2, H // 2)
    under = max(fire_in(cap.rect, [m]) for m in masks) if seg.caption else 0
    seg.report.update(caption_at=seg.caption_at, caption_rect=list(cap.rect),
                      fire_px_under_caption_max=under * q * q,
                      fire_share_under_caption_max=round(under * q * q / max(1, (cap.rect[2] - cap.rect[0]) * (cap.rect[3] - cap.rect[1])), 4))
    if seg.pops:
        word = max((w for _, w in seg.pops), key=len)
        box = pop_box(word, *seg.pop_at)
        seg.report.update(pop_at=list(seg.pop_at), fire_px_under_pop_max=max(
            (fire_in(box, [masks[int(float(at) * FPS) + k]]) for at, _ in seg.pops for k in range(int(0.8 * FPS))
             if 0 <= int(float(at) * FPS) + k < seg.n), default=0) * q * q)
    seg.report["frame0_fire_share"] = round(float(masks[0].mean()), 4)


def compose(seg: Segment, i: int) -> Image.Image:
    img = seg.picture(i)
    d = ImageDraw.Draw(img)
    if seg.decorate:
        seg.decorate(img, d, i)
    if seg.pops:
        bd.pop(d, seg.pop_at[0], seg.pop_at[1], i / FPS, seg.pops)
    if seg.over:
        img = seg.over(img, i)
    if seg.caption:
        img = draw_caption(img, caption_layout(seg.caption, seg.caption_at), seg.caption_alpha(i) if seg.caption_alpha else 1.0)
    return draw_stamp(img, seg.stamp)


def frames_of(dur: float) -> int:
    return int(round(dur * FPS))


def fit_len(a: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(round(n / FPS * SR), np.float32)
    out[:min(out.size, a.size)] = a[:out.size]
    return out


def proof_voice(start: float, n: int) -> np.ndarray:
    return fit_len(read_audio(PROOF_VOICE, SR)[int(round(start * SR)):], n)


def boom_frame_time() -> float:
    """The proof's biggest fire around its BOOM (2.6-3.0 s)."""
    frames, times = bd.read_window(PROOF, 2.6, 3.0)
    keep = [(float(fire_mask(f[PROOF_TOP:PROOF_TOP + H]).sum()), float(t)) for f, t in zip(frames, times) if 2.6 <= t <= 3.0]
    return max(keep)[1]


def seg_poster(dur: float) -> Segment:
    t = boom_frame_time()
    n = frames_of(dur)
    picture, _ = portrait_clip(PROOF, t, dur, PROOF_TOP)
    return Segment("poster", n, picture, proof_voice(t, n), "Generate once.\nPerform again.", PROOF_STAMP,
                   caption_at="bottom", pops=[[0.0, "BOOM"]],
                   source=f"{PROOF.relative_to(ROOT)} {t:.3f}-{t + dur:.3f} s (its BOOM)")


LIVE_CAPTION = "LIVE WEBCAM\nNO RENDER RUNNING"
LIVE_POP = "BOOM"  # the word popped on the live take's loudest sound


def frame_live(frames: np.ndarray, region, stamp: str) -> tuple[tuple[float, float, float, float], str, dict]:
    """Where the 4:5 window sits in region (fractions; default the whole picture) and whether the caption goes at
    the bottom or the top: the window slides along region's long side to keep the fire in the picture, out from
    under the caption first and then from under the stamp."""
    q = 4
    fh, fw = frames.shape[1:3]
    heat = np.zeros((fh // q, fw // q), np.float64)
    for f in frames:
        heat += fire_mask(f[::q, ::q][:fh // q, :fw // q])
    table = np.pad(heat.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def fire(x0, y0, x1, y1) -> float:  # fire (summed over the frames) inside a rectangle of the full picture
        c = [int(round(min(max(v / q, 0), lim))) for v, lim in ((x0, heat.shape[1]), (y0, heat.shape[0]),
                                                                (x1, heat.shape[1]), (y1, heat.shape[0]))]
        return float(table[c[3], c[2]] - table[c[1], c[2]] - table[c[3], c[0]] + table[c[1], c[0]])
    x0, y0, x1, y1 = region or (0.0, 0.0, 1.0, 1.0)
    rx0, ry0, rx1, ry1 = x0 * fw, y0 * fh, x1 * fw, y1 * fh
    wide = (rx1 - rx0) / (ry1 - ry0) > W / H
    ww, wh = ((ry1 - ry0) * W / H, ry1 - ry0) if wide else (rx1 - rx0, (rx1 - rx0) * H / W)
    span = (rx1 - rx0 - ww) if wide else (ry1 - ry0 - wh)
    total, best = fire(rx0, ry0, rx1, ry1), None
    stamp_rect = stamp_layout(stamp)[2]
    for k in range(41):
        left, top = (rx0 + span * k / 40, ry0) if wide else (rx0, ry0 + span * k / 40)
        s = ww / W
        inside = fire(left, top, left + ww, top + wh)
        under = lambda r: fire(left + r[0] * s, top + r[1] * s, left + r[2] * s, top + r[3] * s)  # noqa: E731
        for where in ("bottom", "top"):
            caption, on_stamp = under(caption_layout(LIVE_CAPTION, where).rect), under(stamp_rect)
            cost = 10 * caption + on_stamp + 3 * (total - inside) + 0.01 * total * abs(k - 20) / 20 + (where == "top")
            if best is None or cost < best[0]:
                best = (cost, (left, top), where, dict(fire_under_caption=caption, fire_under_stamp=on_stamp,
                                                       fire_cut_off=total - inside, fire_total=total))
    (left, top), where, why = best[1], best[2], best[3]
    window = (left / fw, top / fh, (left + ww) / fw, (top + wh) / fh)
    return window, where, {k: round(v / max(total, 1), 4) for k, v in why.items() if k != "fire_total"}


def seg_live(path: Path, start: float, dur: float, region, stamp: str) -> Segment:
    n = frames_of(dur)
    frames, audio = decode(path, start, dur, 1350)
    window, where, shares = frame_live(frames, region, stamp)
    pictures = [to_frame(cover_region(f, window)) for f in frames]
    del frames
    at = loudest_at(audio)
    seg = Segment("live", n, lambda i: Image.fromarray(pictures[i]), audio, LIVE_CAPTION, stamp, caption_at=where,
                  pops=[[at, LIVE_POP]], source=f"{path} {start:.3f}-{start + dur:.3f} s, window "
                                              f"{tuple(round(v, 4) for v in window)} of region {region or 'whole'}")
    seg.report.update(loudest_sound_at=round(at, 3), live_window=[round(v, 4) for v in window],
                      live_fire_shares=shares)
    return seg


def seg_proof() -> Segment:
    n = frames_of(3.5)
    picture, _ = portrait_clip(PROOF, 0.0, 3.5, PROOF_TOP)
    audio = proof_voice(0.0, n)
    env = np.array([np.sqrt(np.mean(audio[k * (SR // FPS):(k + 1) * (SR // FPS)] ** 2)) for k in range(n)])
    env = (env / env.max()) ** 0.8
    cap = caption_layout("Quiet → small.\nLoud → big.", "bottom")

    def decorate(img, d, i):  # the voice as bars: two small pf, one big BOOM
        bd.wave(d, 90, cap.rect[1] - 96, W - 180, 76, env, i)
    return Segment("proof", n, picture, audio, "Quiet → small.\nLoud → big.", PROOF_STAMP, caption_at="bottom",
                   pops=[[1.21, "pf"], [1.58, "pf"], [2.67, "BOOM"]], decorate=decorate,
                   source=f"{PROOF.relative_to(ROOT)} 0-3.5 s, sound {PROOF_VOICE.relative_to(ROOT)}")


def seg_strangers(start: float, dur: float, stamp: str) -> Segment:
    """The same render, four strangers' voices, all playing at once; each name lights as its voice lands."""
    n = frames_of(dur)
    caption = "Same fire.\n4 strangers’ voices.\n0 new renders."
    cap_probe = caption_layout(caption, 0)
    gap = 6
    cap_h = cap_probe.rect[3] - cap_probe.rect[1] + 16
    cw = (W - gap) // 2
    ch = (H - STAMP_ZONE - cap_h - gap) // 2
    rows = [STAMP_ZONE, STAMP_ZONE + ch + cap_h + gap]
    cells, sounds, hits = [], [], []
    for who, _ in STRANGERS:
        c = bd.Clip(str(MADE / f"hand_palm__{who}__again/final.mp4"), start, dur)
        fw = c.frames.shape[2]
        sh = round(fw * ch / cw)
        top = min(max(0, 800 - sh // 2), c.frames.shape[1] - sh)  # the fire (rows 380-1190) whole, the palm under it
        cells.append(([np.asarray(Image.fromarray(f[top:top + sh]).resize((cw, ch), Image.Resampling.LANCZOS))
                       for f in c.frames], c.times))
        a = c.sound()
        sounds.append(a)
        hits.append([o["at"] for o in onsets(a, floor_db=20)])
        del c
    audio = sum(sounds) / 2.0  # four voices, mostly a beat apart: levelled together below

    def picture(i: int) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        for k, (fs, times) in enumerate(cells):
            img.paste(Image.fromarray(fs[index_at(times, start, i)]), ((cw + gap) * (k % 2), rows[k // 2]))
        return img

    def decorate(img, d, i):
        t = i / FPS
        for k, (who, label) in enumerate(STRANGERS):
            x, y = (cw + gap) * (k % 2) + 22, rows[k // 2] + ch - 14  # bottom left, over the arm: the fire rises
            ages = [t - h for h in hits[k] if t >= h]
            lit = bool(ages)
            grow = 1 + 0.35 * max(0.0, 1 - min(ages) / 0.3) if lit else 1.0
            d.text((x, y), f"“{label}”", font=font(int(56 * grow)), fill=ACCENT if lit else MUTED, anchor="ld",
                   stroke_width=5, stroke_fill=(0, 0, 0))
    seg = Segment("strangers", n, picture, audio, caption, stamp, caption_at=rows[0] + ch + 8, decorate=decorate,
                  source=f"{MADE.relative_to(ROOT)}/hand_palm__{{{','.join(w for w, _ in STRANGERS)}}}__again/final.mp4 "
                         f"{start}-{start + dur} s, all four sounds mixed")
    seg.report["voice_onsets_s"] = {w: hs for (w, _), hs in zip(STRANGERS, hits)}
    return seg


def seg_portrait(key: str, src: Path, start: float, dur: float, top: int, caption: str, stamp: str,
                 caption_at: str = "auto") -> Segment:
    picture, sound = portrait_clip(src, start, dur, top)
    return Segment(key, frames_of(dur), picture, sound, caption, stamp, caption_at=caption_at,
                   source=f"{src.relative_to(ROOT)} {start}-{start + dur} s, rows {top}+")


def seg_scene(key: str, src: Path, start: float, dur: float, caption: str, stamp: str, marks=None, pops=None) -> Segment:
    cap = caption_layout(caption, "bottom")
    pic_h = round(W * 1080 / 1920)
    y_pic = STAMP_ZONE + (cap.rect[1] - STAMP_ZONE - pic_h) // 2
    picture, sound, size = letterbox_clip(src, start, dur, y_pic)
    decorate = None
    if marks:
        def decorate(img, d, i):  # where each hit goes off, numbered in turn
            bd.mark(d, 0, y_pic, size[0], size[1], i / FPS, marks)
    # the noise words over the blurred band above the picture: on it they would cover a lit lantern
    return Segment(key, frames_of(dur), picture, sound, caption, stamp, caption_at="bottom", pops=pops or [],
                   pop_at=(W // 2, y_pic - 76) if pops else None, decorate=decorate,
                   source=f"{src.relative_to(ROOT)} {start}-{start + dur} s, letterboxed")


def seg_end(dur: float) -> Segment:
    n = frames_of(dur)
    t = boom_frame_time()
    frames, times = bd.read_window(PROOF, t, t + 0.05)
    back = Image.fromarray(np.ascontiguousarray(frames[index_at(times, t, 0)][PROOF_TOP:PROOF_TOP + H]))
    back = ImageEnhance.Brightness(back.filter(ImageFilter.GaussianBlur(14))).enhance(0.3)
    text = "1 LTX-2.5 render in ComfyUI. Performed by any voice."
    lines = [("1 LTX-2.5 render", INK), ("in ComfyUI.", INK), ("Performed by", ACCENT), ("any voice.", ACCENT)]
    url, tag = "github.com/grzenio22/cue-the-boom", "#OpenSourceHiggsfield"
    size = 64
    while _MEASURE.textlength(url, font=font(size)) > W - 2 * SIDE and size > 40:
        size -= 1

    def fade(i: int, at: float) -> int:
        return round(255 * min(1.0, max(0.0, (i / FPS - at) / 0.2)))

    def over(img: Image.Image, i: int) -> Image.Image:  # the claim, then where to find it
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.text((W / 2, 215), "CUE THE BOOM", font=font(50), fill=ACCENT + (255,), anchor="mm")
        for k, (line, colour) in enumerate(lines):
            d.text((W / 2, 340 + k * 102 + (28 if k > 1 else 0)), line, font=font(CAPTION_SIZE), fill=colour + (255,),
                   anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0, 255))
        a = fade(i, 0.8)
        d.text((W / 2, 845), url, font=font(size), fill=INK + (a,), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, a))
        a = fade(i, 1.2)
        d.text((W / 2, 965), tag, font=font(78), fill=ACCENT + (a,), anchor="mm", stroke_width=4, stroke_fill=(0, 0, 0, a))
        return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    crackle = read_audio(CRACKLE, SR)[int(0.8 * SR):]
    audio = fit_len(crackle, n)
    audio *= np.linspace(1.0, 0.0, audio.size, dtype=np.float32) ** 1.5
    seg = Segment("end", n, lambda i: back.copy(), audio, "", caption_at=330, over=over, target=-24.0,
                  source=f"end card over the proof's BOOM frame, blurred; sound {CRACKLE.relative_to(ROOT)} fading")
    seg.report.update(text=text, url=url, url_font_px=size, hashtag=tag)
    return seg


# ---------------------------------------------------------------- sound


def measure(samples: np.ndarray, tmp: Path) -> tuple[float, float]:
    """Integrated loudness (LUFS) and true peak (dBTP) of a mono mix played as dual mono, as the cut carries it."""
    path = tmp / "measure.f32"
    path.write_bytes(np.repeat(samples.astype("<f4")[:, None], 2, axis=1).tobytes())
    return ebur128(["-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(path)])


def ebur128(inputs: list[str], trim: tuple[float, float] | None = None, momentary: bool = False):
    """FFmpeg's ebur128: integrated loudness (LUFS) and true peak (dBTP); with momentary, also the loudest 400 ms."""
    af = (f"atrim=start={trim[0]:.4f}:end={trim[1]:.4f}," if trim else "") + "ebur128=peak=true"
    r = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", *inputs, "-map", "0:a", "-af", af, "-f", "null", "-"],
                       capture_output=True, text=True)
    summary = r.stderr[r.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary)
    p = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    if not i or not p:
        raise RuntimeError(f"ebur128 gave no summary: {r.stderr[-500:]}")
    result = float(i.group(1).replace("-inf", "-120")), float(p.group(1).replace("-inf", "-120"))
    if momentary:
        ms = [float(m) for m in re.findall(r"\bM:\s*(-?[\d.]+)", r.stderr[:r.stderr.rfind("Summary:")])]
        result += (max(ms) if ms else -120.0,)
    return result


def level(segments: list[Segment], tmp: Path) -> np.ndarray:
    """Each segment towards its target loudness by gain alone, never past TP_BEFORE_AAC (a quiet pf stays quiet
    beside its BOOM: no limiter, no compressor), with 6 ms fades at the cuts."""
    out = []
    for seg in segments:
        a = seg.audio.astype(np.float32).copy()
        k = int(0.006 * SR)
        a[:k] *= np.linspace(0, 1, k, dtype=np.float32)
        a[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
        loud, peak = measure(a, tmp)
        gain = 0.0 if loud < -60 else min(seg.target - loud, TP_BEFORE_AAC - peak)
        a *= np.float32(10 ** (gain / 20))
        seg.report.update(source_lufs=loud, source_true_peak=peak, gain_db=round(gain, 2),
                          peak_limited=bool(loud >= -60 and TP_BEFORE_AAC - peak < seg.target - loud))
        out.append(a)
    return np.concatenate(out)


# ---------------------------------------------------------------- the cut


def sheet(thumbs: list[tuple[float, Image.Image]], path: Path, cols: int = 6) -> None:
    tw, th = 270, 338
    rows = -(-len(thumbs) // cols)
    out = Image.new("RGB", (cols * tw + (cols + 1) * 6, rows * (th + 30) + 6), (24, 24, 28))
    d = ImageDraw.Draw(out)
    for k, (t, im) in enumerate(thumbs):
        x, y = 6 + (k % cols) * (tw + 6), 6 + (k // cols) * (th + 30)
        im = im.copy()
        im.thumbnail((tw, th), Image.Resampling.LANCZOS)
        out.paste(im, (x + (tw - im.width) // 2, y + 24 + (th - im.height) // 2))
        d.text((x + 4, y + 2), f"{t:g} s", font=font(20), fill=(230, 230, 230))
    out.save(path, quality=90)


def build(args) -> int:
    out = Path(args.output)
    tmp = Path(args.tmp)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    live = Path(args.live)
    standin = live.resolve() == STANDIN.resolve()
    region = parse_crop(args.live_crop) if args.live_crop else (STANDIN_CROP if standin else None)
    start = args.live_start
    if start is None:
        scan = scan_live(live, region, args.live_dur)
        start = scan["start"]
        print(f"live start {start:.3f} s: {scan['why']}")
    beats = v19_beats()
    stamp = lambda key: beats.get(key, {}).get("stamp") or STAMPS[key]  # noqa: E731
    live_stamp = args.live_stamp or (STANDIN_STAMP if standin else PHONE_STAMP)

    seg1 = seg_live(live, start, args.live_dur, region, live_stamp)
    first = np.asarray(seg1.picture(0))
    live_fire = float(fire_mask(first).mean())
    poster = args.poster == "always" or (args.poster == "auto" and live_fire < FIRE_MIN)
    print(f"live frame 0 fire share {live_fire:.4f} -> {'opens on the proof BOOM' if poster else 'opens on the live take'}")
    lanterns = beats.get("lanterns", {})
    builders: list[Callable[[], Segment]] = ([lambda: seg_poster(args.poster_dur)] if poster else []) + [
        lambda: seg1,
        seg_proof,
        lambda: seg_strangers(1.4, 2.1, stamp("strangers")),  # the fires land 2.2 s in; the 16:9 demo uses 1.4-3.5 s too
        # water on the dim palm, then lightning on the author's own zzzzz (filmed 28 Oct): the element changes with the sound
        lambda: seg_portrait("water", MADE / "hand_palm__shhhhh__water_whip/composite.mp4", 0.6, 2.6, 420,
                             "shhhh, sploosh:\nwater.", STAMPS["water"], "bottom"),
        lambda: seg_portrait("lightning", LIGHTNING_28, 2.25, 2.4, 160, "zzzzz:\nlightning.", STAMPS["lightning"], "bottom"),
        lambda: seg_scene("dino", CLIPS / "scene_dino/composite.mp4", 0.25, 3.5, "Or render it\ninto the scene.",
                          stamp("dino")),
        lambda: seg_scene("lanterns", CLIPS / "scene_lanterns/composite.mp4", 0.0, 4.5,
                          "Each hit lights the\nnext lantern.", stamp("lanterns"),
                          marks=lanterns.get("marks", LANTERN_MARKS), pops=lanterns.get("pops", LANTERN_POPS)),
        lambda: seg_end(3.0),
    ]
    video = tmp / "social_cut_video.mp4"
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-an", "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                            "setparams=range=tv:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-level:v", "4.1",
                            "-g", "48", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                            "-color_range", "tv", str(video)], stdin=subprocess.PIPE)
    done, thumbs, total = [], [], 0
    try:
        for make in builders:
            seg = make()
            place(seg)
            seg.report.update(begin=round(total / FPS, 3), end=round((total + seg.n) / FPS, 3))
            for i in range(seg.n):
                img = compose(seg, i)
                if (total + i) % FPS == 0:
                    thumbs.append(((total + i) / FPS, img))
                enc.stdin.write(img.tobytes())
            total += seg.n
            print(f"{seg.key:13s} {seg.report['begin']:5.2f}-{seg.report['end']:5.2f} s  caption {seg.report['caption_at']}"
                  f"  fire under caption (max px) {seg.report['fire_px_under_caption_max']}", flush=True)
            seg.picture = seg.decorate = seg.over = None  # free its frames
            done.append(seg)
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("the H.264 encode failed")
    except BaseException:
        enc.kill()
        enc.wait()
        raise
    audio = level(done, tmp)
    audio = fit_len(audio, total)
    raw = tmp / "social_cut_audio.f32"
    for attempt in range(3):
        raw.write_bytes(np.repeat(audio.astype("<f4")[:, None], 2, axis=1).tobytes())
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-f", "f32le", "-ar", str(SR), "-ac", "2",
                        "-i", str(raw), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", "-shortest", str(out)], check=True)
        loud, peak = ebur128(["-i", str(out)])
        if peak <= -1.5 or attempt == 2:
            break
        audio *= np.float32(10 ** ((-1.6 - peak) / 20))  # the AAC encode overshot: the whole cut down a little
        print(f"true peak {peak:+.1f} dBTP after AAC: lowering the cut by {peak + 1.6:.1f} dB")
    for seg in done:
        seg.report["lufs"], seg.report["true_peak"], seg.report["max_momentary"] = ebur128(
            ["-i", str(out)], (seg.report["begin"], seg.report["end"]), momentary=True)
    duration = total / FPS
    final = decode_audio(out, 0.0, duration)
    proof = next(s for s in done if s.key == "proof")

    def burst_db(t: float) -> float:  # the loudest 50 ms of a sound starting at t s into the proof
        a = final[int((proof.report["begin"] + t - 0.05) * SR):int((proof.report["begin"] + t + 0.35) * SR)]
        hop = SR // 20
        return round(float(20 * np.log10(max(np.sqrt(np.mean(a[k:k + hop] ** 2)) for k in range(0, a.size - hop, hop // 4)))), 1)
    levels = [burst_db(float(at)) for at, _ in proof.pops]
    proof.report["pf_pf_boom_db"] = levels
    proof.report["boom_over_pf_db"] = round(levels[-1] - max(levels[:-1]), 1)
    sheet(thumbs, out.with_name(out.stem + "_sheet.jpg"))
    report = dict(output=str(out), duration=duration, size=[W, H], fps=FPS, bytes=out.stat().st_size,
                  lufs=loud, true_peak=peak, live=str(live), live_start=start, live_crop=region,
                  live_frame0_fire_share=round(live_fire, 4), opens_on="proof BOOM" if poster else "live take",
                  caption_cap_height_px=font(CAPTION_SIZE).getbbox("H")[3] - font(CAPTION_SIZE).getbbox("H")[1],
                  segments=[dict(key=s.key, source=s.source, caption=s.caption, stamp=s.stamp, **s.report) for s in done])
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    video.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    (tmp / "measure.f32").unlink(missing_ok=True)
    print(f"\n{out}: {duration:.2f} s, {W}x{H}, {out.stat().st_size / 1e6:.1f} MB, {loud:.1f} LUFS, {peak:+.1f} dBTP")
    for s in done:
        r = s.report
        print(f"  {s.key:13s} {r['begin']:5.2f}-{r['end']:5.2f} s  {r['lufs']:6.1f} LUFS  {r['true_peak']:+5.1f} dBTP"
              f"  max M {r['max_momentary']:6.1f}  gain {r['gain_db']:+5.1f} dB"
              f"{'  (peak-limited gain)' if r['peak_limited'] else ''}")
    print(f"  proof: pf, pf, BOOM loudest 50 ms {proof.report['pf_pf_boom_db']} dBFS: the BOOM "
          f"{proof.report['boom_over_pf_db']:+.1f} dB over the louder pf")
    print(f"sheet: {out.with_name(out.stem + '_sheet.jpg')}")
    return 0


def parse_crop(text: str) -> tuple[float, float, float, float]:
    vals = tuple(float(v) for v in text.split(","))
    if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
        raise argparse.ArgumentTypeError("--live-crop takes x0,y0,x1,y1 as fractions, x0 < x1, y0 < y1")
    return vals


def scan(args) -> int:
    live = Path(args.live)
    standin = live.resolve() == STANDIN.resolve()
    region = parse_crop(args.live_crop) if args.live_crop else (STANDIN_CROP if standin else None)
    s = scan_live(live, region, args.live_dur)
    print(f"{live}: {s['duration']:.2f} s; fire share inside crop {region or 'whole frame'}, every 0.25 s:")
    for f0 in range(0, len(s["fire"]), FPS // 4):
        bar = "#" * min(40, int(s["fire"][f0] * 400))
        print(f"  {f0 / FPS:6.2f} s  {s['fire'][f0]:.4f} {bar}")
    print("loud sounds (s, 20 ms peak dBFS): " + ", ".join(f"{o['at']:.2f} ({o['peak_db']:.1f})" for o in s["onsets"]))
    print(f"suggested --live-start {s['start']:.2f}: {s['why']}")
    path = Path(args.output).with_name("live_scan_sheet.jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet([(f0 / FPS, im) for f0, im in s["thumbs"][:60]], path, cols=8)
    print(f"frames every 0.5 s (the crop): {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", default=str(STANDIN), help="the live take (default: the stand-in)")
    ap.add_argument("--live-start", type=float, default=None, help="s into the live take (default: picked, see --scan)")
    ap.add_argument("--live-dur", type=float, default=3.0)
    ap.add_argument("--live-crop", default=None,
                    help="x0,y0,x1,y1: the part of the upright live picture the 4:5 window may use (fractions)")
    ap.add_argument("--live-stamp", default=None, help="the live segment's stamp (default: the phone or stand-in one)")
    ap.add_argument("--live-caption", default=None, help="the live segment's caption, its lines split by | "
                                                         "(default: LIVE WEBCAM | NO RENDER RUNNING)")
    ap.add_argument("--live-pop", default=None, help="the word popped on its loudest sound (default: BOOM)")
    ap.add_argument("--poster", choices=("auto", "always", "never"), default="auto",
                    help="open on the proof's BOOM first: auto = when the live take's first frame shows no fire")
    ap.add_argument("--poster-dur", type=float, default=0.5)
    ap.add_argument("--scan", action="store_true", help="only report the live take's fire and loud sounds")
    ap.add_argument("-o", "--output", default=str(OUT_DIR / "social_cut.mp4"))
    ap.add_argument("--tmp", default=str(OUT_DIR / "tmp"))
    a = ap.parse_args()
    global LIVE_CAPTION, LIVE_POP
    LIVE_CAPTION = a.live_caption.replace("|", "\n") if a.live_caption else LIVE_CAPTION
    LIVE_POP = a.live_pop or LIVE_POP
    sys.stdout.reconfigure(encoding="utf-8")
    if a.live_crop:
        parse_crop(a.live_crop)
    return scan(a) if a.scan else build(a)


if __name__ == "__main__":
    raise SystemExit(main())
