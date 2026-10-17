"""Build the demo video from a storyboard: a JSON list of beats, rendered to one 1920x1080 MP4 with sound.

    python tools/demo/build_demo.py tools/demo/storyboard.json -o runs/demo/demo.mp4

Beats (all times in seconds; text may use \\n):
  {"type": "title", "dur": 3, "text": "...", "sub": "..."}
  {"type": "clip", "src": "path.mp4", "start": 0, "dur": 4, "caption": "...", "side": "...", "label": "..."}
      a vertical (or any) clip, fitted in the centre; caption under it, side text on the left
  {"type": "triptych", "srcs": ["a.mp4", "b.mp4", "c.mp4"], "labels": [...], "start": 0, "dur": 5,
   "audio": 0 | 1 | 2 | "mix", "caption": "..."}       three clips side by side, same timeline
  {"type": "image", "src": "path.png", "dur": 3, "caption": "...", "zoom": 1.08}   a still with a slow push-in
  a clip may add "speed": 8 (silent while sped up) and "note": "sped up x8" (a tag at the top right)
      ("split": [0.4] cuts a wide image at those fractions of its width and stacks the parts as rows)
  {"type": "credits", "dur": 6, "title": "...", "lines": ["...", "..."]}
Every beat may set "stamp": a small corner label (e.g. "stand-in plate, generated with LTX-2.5").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuesync" / "src"))
from cuesync.media import read_audio, read_video, write_video  # noqa: E402

W, H, FPS, SR = 1920, 1080, 24, 48000
BG = (11, 11, 15)
ACCENT = (255, 176, 32)
FG = (244, 241, 234)
MUTED = (163, 159, 151)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_block(d: ImageDraw.ImageDraw, xy, text: str, size: int, fill, bold=False, anchor="la", spacing=10,
               max_w: int | None = None):
    f = font(size, bold)
    lines = []
    for para in text.split("\n"):
        words, line = para.split(" "), ""
        for w_ in words:
            trial = (line + " " + w_).strip()
            if max_w and d.textlength(trial, font=f) > max_w and line:
                lines.append(line)
                line = w_
            else:
                line = trial
        lines.append(line)
    d.multiline_text(xy, "\n".join(lines), font=f, fill=fill, anchor=anchor, spacing=spacing,
                     align="center" if anchor[0] == "m" else "left")


def fit(frame: np.ndarray, box_w: int, box_h: int) -> Image.Image:
    im = Image.fromarray(frame)
    s = min(box_w / im.width, box_h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.Resampling.LANCZOS)


def stacked(src: Image.Image, split: list[float] | None) -> Image.Image:
    """A wide image cut at fractions of its width, the parts stacked as left-aligned rows."""
    if not split:
        return src
    xs = [0] + [int(f * src.width) for f in split] + [src.width]
    parts = [src.crop((a, 0, b_, src.height)) for a, b_ in zip(xs, xs[1:])]
    gap = 24
    out = Image.new("RGB", (max(p.width for p in parts), sum(p.height for p in parts) + gap * (len(parts) - 1)), BG)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + gap
    return out


def stamp(img: Image.Image, text: str | None) -> None:
    if text:
        ImageDraw.Draw(img).text((W - 24, H - 20), text, font=font(22), fill=MUTED, anchor="rd")


class Clip:
    """A stretch of a video from ``start`` s, played at ``speed`` (frames by their own timestamps, so a screen
    recording's uneven frame rate is fine). A sped-up stretch is silent."""

    def __init__(self, src: str, start: float, dur: float, speed: float = 1.0):
        clip = read_video(src)
        self.frames, self.times = clip.frames, clip.times_s
        self.start_s, self.speed = start, speed
        try:
            self.audio = read_audio(src, SR)
        except Exception:
            self.audio = np.zeros(0, np.float32)
        self.n = int(round(dur * FPS))

    def frame(self, i: int) -> np.ndarray:
        k = int(np.searchsorted(self.times, self.start_s + i / FPS * self.speed, side="right")) - 1
        return self.frames[min(max(k, 0), len(self.frames) - 1)]

    def sound(self) -> np.ndarray:
        out = np.zeros(int(self.n / FPS * SR), np.float32)
        if self.speed != 1.0:
            return out
        a = self.audio[int(self.start_s * SR):]
        out[: min(out.size, a.size)] = a[: out.size]
        return out


def render_beat(b: dict) -> tuple[list[np.ndarray], np.ndarray]:
    kind, dur = b["type"], float(b.get("dur", 3))
    n = int(round(dur * FPS))
    frames, audio = [], np.zeros(int(n / FPS * SR), np.float32)
    if kind == "title":
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        text_block(d, (W // 2, H // 2 - 40), b["text"], 96, FG, bold=True, anchor="mm", max_w=W - 240)
        if b.get("sub"):
            text_block(d, (W // 2, H // 2 + 90), b["sub"], 40, ACCENT, anchor="mm", max_w=W - 300)
        stamp(img, b.get("stamp"))
        frames = [np.asarray(img)] * n
    elif kind == "credits":
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((W // 2, 120), b.get("title", "Built with"), font=font(64, True), fill=FG, anchor="mm")
        lines = b["lines"]
        col = (len(lines) + 1) // 2
        for k, line in enumerate(lines):
            x = 180 if k < col else W // 2 + 60
            y = 230 + (k % col) * 64
            d.text((x, y), line, font=font(32), fill=FG if not line.startswith("·") else MUTED)
        stamp(img, b.get("stamp"))
        frames = [np.asarray(img)] * n
    elif kind == "image":
        src = stacked(Image.open(b["src"]).convert("RGB"), b.get("split"))
        zoom = float(b.get("zoom", 1.06))
        for i in range(n):
            z = 1.0 + (zoom - 1.0) * i / max(1, n - 1)
            img = Image.new("RGB", (W, H), BG)
            box_h = H - (170 if b.get("caption") else 80)
            s = min((W - 120) / src.width, box_h / src.height) * z
            im = src.resize((int(src.width * s), int(src.height * s)), Image.Resampling.LANCZOS)
            img.paste(im, ((W - im.width) // 2, (box_h - im.height) // 2 + 40))
            if b.get("caption"):
                text_block(ImageDraw.Draw(img), (W // 2, H - 70), b["caption"], 44, FG, anchor="mm", max_w=W - 200)
            stamp(img, b.get("stamp"))
            frames.append(np.asarray(img))
    elif kind == "clip":
        c = Clip(b["src"], float(b.get("start", 0)), dur, float(b.get("speed", 1.0)))
        audio = c.sound()
        box_h = H - (150 if b.get("caption") else 60)
        for i in range(n):
            img = Image.new("RGB", (W, H), BG)
            im = fit(c.frame(i), W - 900 if b.get("side") else W - 120, box_h)
            x0 = (W - im.width) // 2 + (220 if b.get("side") else 0)
            img.paste(im, (x0, (box_h - im.height) // 2 + 20))
            d = ImageDraw.Draw(img)
            if b.get("side"):  # the label heads the side text (the shot carries its own effect label)
                y = H // 2 - 120
                if b.get("label"):
                    text_block(d, (90, y - 110), b["label"], 60, ACCENT, bold=True, max_w=x0 - 160)
                text_block(d, (90, y), b["side"], 46, FG, max_w=x0 - 160, spacing=16)
            elif b.get("label"):
                d.text((x0 + 20, 40), b["label"], font=font(40, True), fill=ACCENT)
            if b.get("caption"):
                text_block(d, (W // 2, H - 64), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            if b.get("note"):  # e.g. that a stretch is sped up
                f = font(30, True)
                tw = d.textlength(b["note"], font=f)
                d.rounded_rectangle([W - tw - 70, 24, W - 30, 74], radius=12, fill=ACCENT)
                d.text((W - 50, 49), b["note"], font=f, fill=BG, anchor="rm")
            stamp(img, b.get("stamp"))
            frames.append(np.asarray(img))
    elif kind == "triptych":
        clips = [Clip(s, float(b.get("start", 0)), dur) for s in b["srcs"]]
        mix = b.get("audio", "mix")
        sounds = [c.sound() for c in clips]
        audio = (sum(sounds) / len(sounds)) if mix == "mix" else sounds[int(mix)]
        box_h = H - (150 if b.get("caption") else 60)
        cw = (W - 80) // len(clips)
        labels = b.get("labels", [""] * len(clips))
        for i in range(n):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            for k, c in enumerate(clips):
                im = fit(c.frame(i), cw - 30, box_h - 70)
                x0 = 40 + k * cw + (cw - im.width) // 2
                img.paste(im, (x0, 80 + (box_h - 70 - im.height) // 2))
                d.text((40 + k * cw + cw // 2, 46), labels[k], font=font(40, True), fill=ACCENT, anchor="mm")
            if b.get("caption"):
                text_block(d, (W // 2, H - 64), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            stamp(img, b.get("stamp"))
            frames.append(np.asarray(img))
    else:
        raise ValueError(f"unknown beat type {kind!r}")
    return frames, audio


def build(storyboard: Path, out: Path) -> float:
    beats = json.loads(Path(storyboard).read_text(encoding="utf-8"))
    beats = beats["beats"] if isinstance(beats, dict) else beats
    all_frames, all_audio = [], []
    for k, b in enumerate(beats):
        f, a = render_beat(b)
        all_frames.extend(f)
        all_audio.append(a)
        print(f"beat {k + 1}/{len(beats)} {b['type']}: {len(f) / FPS:.1f} s", flush=True)
    write_video(out, np.stack(all_frames), fps=FPS, audio=np.concatenate(all_audio), crf=20, preset="medium")
    return len(all_frames) / FPS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("storyboard")
    ap.add_argument("-o", "--output", default="runs/demo/demo.mp4")
    a = ap.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    total = build(Path(a.storyboard), Path(a.output))
    print(f"demo: {total:.1f} s -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
