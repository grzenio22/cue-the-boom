"""Build the demo video from a storyboard: a JSON list of beats, rendered to one 1920x1080 MP4 with sound.

    python tools/demo/build_demo.py tools/demo/storyboard.json -o runs/demo/demo.mp4

Beats (all times in seconds; text may use \\n):
  {"type": "title", "dur": 3, "text": "...", "sub": "..."}
  {"type": "clip", "src": "path.mp4", "start": 0, "dur": 4, "caption": "...", "side": "...", "label": "..."}
      a vertical (or any) clip, fitted in the centre; caption under it, side text on the left
  {"type": "triptych", "srcs": ["a.mp4", "b.mp4", "c.mp4"], "labels": [...], "start": 0, "dur": 5,
   "audio": 0 | 1 | 2 | "mix", "caption": "..."}       three clips side by side, same timeline
  {"type": "turns", "panels": [{"src": "a.mp4", "start": 0, "dur": 2, "label": "BOOM", "crop": [...], "pops": [...]},
   ...], "headline": "...", "subline": "...", "caption": "..."}   clips side by side that play in turn, each with its
      own sound under it (the others wait, dimmed): one shot performed several ways; the beat lasts as long as they do
  {"type": "image", "src": "path.png", "dur": 3, "caption": "...", "zoom": 1.08}   a still with a slow push-in
  a clip may add "speed": 8 (silent while sped up) and "note": "sped up x8" (a tag at the top right),
  "crop": [x0, y0, x1, y1] (fractions, to zoom in), and "audio_src" + "audio_from" + "audio_at" (another file's
  sound, from audio_from s in it, starting audio_at s into the beat), "wave": true (the beat's sound drawn under
  the clip, a playhead running along it) and "pops": [[t, "KA-BOOM"], ...] (the noise word, big, at t s), "marks":
  [[t, x, y, "1"], ...] (a numbered ring on the clip at x, y as fractions, from t s: where a hit goes off) and
  "headline" / "subline" (a title over the clip itself, top left), "fill": true (the clip, or its crop, fills the
  whole frame, the text over it; a crop of the frame's own shape is not cut further) and "chip": "..." (a step's
  name, big, top left); "pop_x" / "pop_y": where the noise words sit, as shares of the clip's width and height
  (beside the fire, not on it); "audio_gain": how much louder audio_src plays (a take said quietly); "wave_box":
  [x, y, w, h] as shares of the frame, where a filled clip's sound is drawn; "stamp_top": the stamp at the top
      (an image may add "crop" too; "split": [0.4] cuts a wide image at those fractions of its width and stacks
      the parts as rows)
  {"type": "versus", "shot": "runs/.../shot", "start": 0.4, "dur": 2.4, "speed": 1, "labels": ["...", "..."],
   "cue_at": 1.0}   the shot's raw render (as LTX-2.5 made it, on the plate's timeline) beside the finished shot,
      the sound drawn under both with a line where the noise is (cue_at: s on the shot's timeline)
  {"type": "credits", "dur": 6, "title": "...", "lines": ["...", "  · a sub-line", "..."], "url": "..."}
Every beat may set "stamp": a small corner label (e.g. "stand-in plate, generated with LTX-2.5"), and "vo": a
narration WAV with "vo_at" (s into the beat, default 0.15) and "vo_gain": it is laid over the beat's own sound,
which is ducked while it speaks; narration longer than its beat runs on into the next beats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuesync" / "src"))
from cuesync.media import read_audio  # noqa: E402

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


def cover(frame: np.ndarray, box_w: int, box_h: int) -> Image.Image:
    """The frame scaled to fill the box, cut at its middle where its shape differs."""
    im = Image.fromarray(frame)
    s = max(box_w / im.width, box_h / im.height)
    im = im.resize((max(box_w, round(im.width * s)), max(box_h, round(im.height * s))), Image.Resampling.LANCZOS)
    x, y = (im.width - box_w) // 2, (im.height - box_h) // 2
    return im.crop((x, y, x + box_w, y + box_h))


def chip(d: ImageDraw.ImageDraw, x: int, y: int, text: str, size: int = 50) -> None:
    """A step's name on an orange tab."""
    f = font(size, True)
    tw = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x + tw + 1.2 * size, y + 1.7 * size], radius=0.4 * size, fill=ACCENT)
    d.text((x + 0.6 * size, y + 0.85 * size), text, font=f, fill=BG, anchor="lm")


def cropped(frames: np.ndarray, crop: list[float] | None) -> np.ndarray:
    """[x0, y0, x1, y1] as fractions of the picture."""
    if not crop:
        return frames
    h, w = frames.shape[1:3]
    x0, y0, x1, y1 = int(crop[0] * w) // 2 * 2, int(crop[1] * h) // 2 * 2, int(crop[2] * w) // 2 * 2, int(crop[3] * h) // 2 * 2
    return frames[:, y0:y1, x0:x1]


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


def stamp(img: Image.Image, text: str | None, top: bool = False) -> None:
    if text:  # what the shot is made of: as readable as the claims it qualifies (at the top where the UI is under it)
        ImageDraw.Draw(img).text((W - 24, 12) if top else (W - 24, H - 12), text, font=font(32), fill=(214, 210, 202),
                                 anchor="ra" if top else "rd")


def envelope(audio: np.ndarray, n: int) -> np.ndarray:
    """The loudness of a beat's sound per video frame, 0-1 (compressed, so a quiet "pf." still shows)."""
    hop = SR // FPS
    env = np.array([float(np.sqrt(np.mean(audio[k * hop:(k + 1) * hop] ** 2))) if k * hop < audio.size else 0.0
                    for k in range(n)])
    return (env / env.max()) ** 0.6 if env.max() > 0 else env


def wave(d: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, env: np.ndarray, i: int,
         cue: float | None = None, cue_label: str = "the noise") -> None:
    """The sound as bars, what has played in orange, a playhead at frame i, and a line where the noise is."""
    n = max(1, len(env))
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=12, fill=(29, 28, 34))
    bw = w / n
    for k, v in enumerate(env):
        bh = max(2, int(v * (h - 20)))
        x = x0 + k * bw
        d.rectangle([x, y0 + h - 10 - bh, x + max(1.0, bw - 1), y0 + h - 10], fill=ACCENT if k <= i else (88, 86, 82))
    if cue is not None:
        cx = x0 + cue * bw
        for y in range(y0 - 8, y0 + h + 8, 14):
            d.line([cx, y, cx, y + 7], fill=FG, width=3)
        d.text((cx + 10, y0 - 8), cue_label, font=font(32, True), fill=FG, anchor="lb")
    px = x0 + (min(i, n - 1) + 0.5) * bw
    d.line([px, y0 - 4, px, y0 + h + 4], fill=FG, width=4)


def pop(d: ImageDraw.ImageDraw, cx: int, cy: int, t: float, pops: list) -> None:
    """The noise word, big, for 0.8 s from the moment it is heard (a quick scale-in)."""
    for at, word in pops or []:
        age = t - float(at)
        if 0 <= age < 0.8:
            size = int(120 * (1.3 - 0.3 * min(1.0, age / 0.15)))
            d.text((cx, cy), word, font=font(size, True), fill=ACCENT, anchor="mm", stroke_width=7, stroke_fill=(0, 0, 0))


def mark(d: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int, t: float, marks: list) -> None:
    """A numbered ring where each hit goes off: it springs up as the hit is heard and stays, quieter, after."""
    for at, x, y, label in marks or []:
        age = t - float(at)
        if age < 0:
            continue
        r = int(46 * (1.4 - 0.4 * min(1.0, age / 0.2)))
        cx, cy = x0 + int(float(x) * w), y0 + int(float(y) * h)
        col = ACCENT if age < 0.9 else (200, 150, 60)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=6 if age < 0.9 else 3)
        d.text((cx + r + 8, cy - r), str(label), font=font(52 if age < 0.9 else 40, True), fill=col, anchor="lt",
               stroke_width=5, stroke_fill=(0, 0, 0))


def read_window(src: str | Path, t0: float, t1: float, every: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The frames of a video shown from t0 to t1 s, with their timestamps, decoded from a seek near t0 (a long
    screen recording is never decoded, or held, whole). ``every``: keep one frame per that many seconds (a
    stretch played sped up shows no more; holding all of 84 s of a screen recording took 6 GB)."""
    import av

    frames, times = [], []
    with av.open(str(src)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if every:  # keep frames closer than the step shown, minus one source frame, so none it shows is dropped
            rate = float(st.average_rate or st.guessed_rate or 24)
            every = max(0.0, every - 1.0 / rate)
        if t0 > 2.0:
            c.seek(int((t0 - 2.0) / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            if fr.time is None:
                continue
            if fr.time > t1 + 0.2:
                break
            if fr.time < t0 - 0.25:  # before the window: keep only the latest, shown until the window's first frame
                frames, times = [fr.to_ndarray(format="rgb24")], [fr.time]
                continue
            if every and times and times[-1] >= t0 and fr.time - times[-1] < every:
                continue
            frames.append(fr.to_ndarray(format="rgb24"))
            times.append(fr.time)
    order = np.argsort(times)
    return np.stack([frames[k] for k in order]), np.asarray(times)[order]


class Clip:
    """A stretch of a video from ``start`` s, played at ``speed`` (frames by their own timestamps, so a screen
    recording's uneven frame rate is fine). A sped-up stretch is silent."""

    def __init__(self, src: str, start: float, dur: float, speed: float = 1.0, crop: list[float] | None = None):
        self.frames, self.times = read_window(src, start, start + dur * speed, every=speed / FPS if speed > 1 else 0.0)
        self.frames = cropped(self.frames, crop)  # zoom in on part of a screen recording
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
    elif kind == "credits":  # one column: a sub-line ("  · ...") sits under the line it belongs to
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((W // 2, 110), b.get("title", "Built with"), font=font(64, True), fill=FG, anchor="mm")
        for k, line in enumerate(b["lines"]):
            sub = line.startswith("  ")
            d.text((W // 2, 210 + k * 76), line.strip(), font=font(44), fill=MUTED if sub or line.startswith("·") else FG, anchor="ma")
        if b.get("url"):
            d.text((W // 2, H - 130), b["url"], font=font(60, True), fill=ACCENT, anchor="mm")
        stamp(img, b.get("stamp"))
        frames = [np.asarray(img)] * n
    elif kind == "image":
        src = Image.open(b["src"]).convert("RGB")
        if b.get("crop"):  # [x0, y0, x1, y1] as fractions
            c = b["crop"]
            src = src.crop((int(c[0] * src.width), int(c[1] * src.height), int(c[2] * src.width), int(c[3] * src.height)))
        src = stacked(src, b.get("split"))
        zoom = float(b.get("zoom", 1.06))
        if src.size == (W, H) and zoom == 1.0 and not b.get("caption"):  # a card made for the frame: as it is
            frames = [np.asarray(src)] * n
        for i in range(0 if frames else n):
            z = 1.0 + (zoom - 1.0) * i / max(1, n - 1)
            img = Image.new("RGB", (W, H), BG)
            box_h = H - (170 if b.get("caption") else 80)
            s = min((W - 120) / src.width, box_h / src.height) * z
            im = src.resize((int(src.width * s), int(src.height * s)), Image.Resampling.LANCZOS)
            img.paste(im, ((W - im.width) // 2, (box_h - im.height) // 2 + 40))
            if b.get("caption"):
                text_block(ImageDraw.Draw(img), (W // 2, H - 84), b["caption"], 44, FG, anchor="mm", max_w=W - 200)
            stamp(img, b.get("stamp"))
            frames.append(np.asarray(img))
    elif kind == "clip":
        c = Clip(b["src"], float(b.get("start", 0)), dur, float(b.get("speed", 1.0)), b.get("crop"))
        audio = c.sound()
        if b.get("audio_src"):  # another file's sound under this stretch (a screen recording has none)
            a = read_audio(b["audio_src"], SR)[int(float(b.get("audio_from", 0)) * SR):]
            at = int(float(b.get("audio_at", 0)) * SR)
            if at < audio.size:
                n_ = min(audio.size - at, a.size)
                audio[at: at + n_] += a[:n_] * float(b.get("audio_gain", 1.0))
        filled = bool(b.get("fill"))
        tall = c.frames.shape[1] > c.frames.shape[2] and not filled  # a portrait clip: its sound goes beside it
        box_h = H - (150 if b.get("caption") else 60) - (130 if b.get("wave") and not tall else 0)
        env = envelope(audio, n) if b.get("wave") else None
        for i in range(n):
            img = Image.new("RGB", (W, H), BG)
            if filled:  # the whole frame, the text over it
                im, x0, y0 = cover(c.frame(i), W, H), 0, 0
            else:
                im = fit(c.frame(i), W - 900 if b.get("side") else W - 120, box_h)
                x0 = (W - im.width) // 2 + (220 if b.get("side") else 0)
                y0 = (box_h - im.height) // 2 + 20
            img.paste(im, (x0, y0))
            d = ImageDraw.Draw(img)
            if env is not None:  # the sound: the playhead reaches the noise as the effect hits
                if filled and b.get("wave_box"):
                    bx, by, bw, bh = b["wave_box"]
                    wave(d, int(bx * W), int(by * H), int(bw * W), int(bh * H), env, i)
                elif filled:
                    wave(d, 60, H - (190 if b.get("caption") else 120), W - 120, 70, env, i)
                elif not tall:
                    wave(d, x0, y0 + im.height + 30, im.width, 90, env, i)
                elif b.get("side"):  # under the side text
                    wave(d, 90, H // 2 + 130, max(200, x0 - 180), 110, env, i)
                else:  # to the right of the clip
                    wave(d, x0 + im.width + 70, y0 + im.height - 220, W - 120 - (x0 + im.width + 70), 110, env, i)
            mark(d, x0, y0, im.width, im.height, i / FPS, b.get("marks"))
            pop(d, x0 + int(im.width * float(b.get("pop_x", 0.5))), y0 + int(im.height * float(b.get("pop_y", 0.62 if filled else 0.8))),
                i / FPS, b.get("pops"))
            if b.get("chip"):  # the step it shows
                chip(d, 60, 50, b["chip"])
            if b.get("headline"):  # the title over the shot itself, not on a card of its own
                d.text((x0 + 40, y0 + 30), b["headline"], font=font(88, True), fill=FG, stroke_width=6, stroke_fill=(0, 0, 0))
                if b.get("subline"):
                    d.text((x0 + 44, y0 + 140), b["subline"], font=font(44, True), fill=ACCENT, stroke_width=5, stroke_fill=(0, 0, 0))
            if b.get("side"):  # the label heads the side text (the shot carries its own effect label)
                y = H // 2 - 120
                if b.get("label"):
                    text_block(d, (90, y - 110), b["label"], 60, ACCENT, bold=True, max_w=x0 - 160)
                text_block(d, (90, y), b["side"], 46, FG, max_w=x0 - 160, spacing=16)
            elif b.get("label"):
                d.text((x0 + 20, 40), b["label"], font=font(40, True), fill=ACCENT)
            if b.get("caption") and filled:  # over the picture: outlined to read on anything
                d.text((W // 2, H - 70), b["caption"], font=font(46, True), fill=FG, anchor="mm", stroke_width=6, stroke_fill=(0, 0, 0))
            elif b.get("caption"):
                text_block(d, (W // 2, H - 80), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            if b.get("note"):  # e.g. that a stretch is sped up
                f = font(30, True)
                tw = d.textlength(b["note"], font=f)
                d.rounded_rectangle([W - tw - 70, 24, W - 30, 74], radius=12, fill=ACCENT)
                d.text((W - 50, 49), b["note"], font=f, fill=BG, anchor="rm")
            stamp(img, b.get("stamp"), top=bool(b.get("stamp_top")))
            frames.append(np.asarray(img))
    elif kind == "versus":  # LTX-2.5's render as it came, beside the finished shot, on the same timeline
        import json as _json
        shot = Path(b["shot"])
        start_f = int(_json.loads((shot / "guide.json").read_text())["start"])  # render frame 0 is this plate frame
        t0, speed = float(b.get("start", 0)), float(b.get("speed", 1.0))
        t1 = t0 + dur * speed
        comp_f, comp_t = read_window(shot / "composite.mp4", t0, t1)
        rend_f, rend_t = read_window(shot / "render.mp4", t0 - start_f / FPS, t1 - start_f / FPS)
        pick = lambda fs, ts, sec: fs[min(max(int(np.searchsorted(ts, sec + 1e-3, side="right")) - 1, 0), len(fs) - 1)]  # noqa: E731
        full = read_audio(shot / "composite.mp4", SR)
        if speed == 1.0:
            a = full[int(t0 * SR):]
            audio[: min(audio.size, a.size)] = a[: audio.size]
        hop = SR // FPS
        shown = [int(round((t0 + i / FPS * speed) * FPS)) for i in range(n)]  # the shot frame on screen
        env = np.array([float(np.sqrt(np.mean(full[f * hop:(f + 1) * hop] ** 2))) if 0 <= f * hop < full.size else 0.0
                        for f in shown])
        env = (env / env.max()) ** 0.6 if env.max() > 0 else env
        cue = (float(b["cue_at"]) - t0) / speed * FPS if b.get("cue_at") is not None else None
        pw, box_h = (W - 120) // 2, H - (150 if b.get("caption") else 60) - 150
        labels = b.get("labels", ["", ""])
        for i, f in enumerate(shown):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            for k, frame in enumerate((pick(rend_f, rend_t, (f - start_f) / FPS), pick(comp_f, comp_t, f / FPS))):
                im = fit(frame, pw - 20, box_h - 60)
                x0 = 60 + k * pw + (pw - im.width) // 2
                img.paste(im, (x0, 90 + (box_h - 60 - im.height) // 2))
                d.text((60 + k * pw + pw // 2, 56), labels[k], font=font(40, True), fill=ACCENT if k else FG, anchor="mm")
            wave(d, 60, box_h + 70, W - 120, 90, env, i, cue)
            if b.get("caption"):
                text_block(d, (W // 2, H - 80), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            if b.get("note"):
                f_ = font(30, True)
                tw = d.textlength(b["note"], font=f_)
                d.rounded_rectangle([W - tw - 70, 24, W - 30, 74], radius=12, fill=ACCENT)
                d.text((W - 50, 49), b["note"], font=f_, fill=BG, anchor="rm")
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
                text_block(d, (W // 2, H - 80), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            stamp(img, b.get("stamp"))
            frames.append(np.asarray(img))
    elif kind == "turns":  # one shot performed several ways: each plays in turn, its own sound under it
        panels = b["panels"]
        clips = [Clip(q["src"], float(q.get("start", 0)), float(q["dur"]), crop=q.get("crop")) for q in panels]
        sounds = [c.sound() for c in clips]
        audio = np.concatenate(sounds)
        envs = [envelope(a, c.n) for a, c in zip(sounds, clips)]
        starts = np.cumsum([0] + [c.n for c in clips])
        top, wave_h = (170 if b.get("headline") else 60), 80
        box_h = H - top - wave_h - (160 if b.get("caption") else 80)
        cw = (W - 120) // len(clips)
        for i in range(int(starts[-1])):
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            now = int(np.searchsorted(starts, i, side="right")) - 1
            for k, (c, q) in enumerate(zip(clips, panels)):
                j = min(max(i - int(starts[k]), 0), c.n - 1)  # waiting on its first frame, done on its last
                im = cover(c.frame(j), cw - 36, box_h)
                if k != now:
                    im = Image.blend(Image.new("RGB", im.size, BG), im, 0.4)
                x0 = 60 + k * cw + 18
                img.paste(im, (x0, top))
                if k == now:
                    d.rounded_rectangle([x0 - 6, top - 6, x0 + im.width + 6, top + im.height + 6], radius=14, outline=ACCENT, width=6)
                d.text((x0 + im.width // 2, top + 56), q.get("label", ""), font=font(56, True), fill=ACCENT if k == now else MUTED,
                       anchor="mm", stroke_width=6, stroke_fill=(0, 0, 0))
                wave(d, x0, top + box_h + 24, im.width, wave_h, envs[k], j if k <= now else -1)
                if k == now:
                    pop(d, x0 + im.width // 2, top + int(box_h * 0.3), j / FPS, q.get("pops"))
            if b.get("headline"):
                d.text((60, 40), b["headline"], font=font(72, True), fill=FG)
                if b.get("subline"):
                    d.text((W - 60, 80), b["subline"], font=font(44, True), fill=ACCENT, anchor="rm")
            if b.get("caption"):
                text_block(d, (W // 2, H - 80), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            stamp(img, b.get("stamp"))
            frames.append(np.asarray(img))
    else:
        raise ValueError(f"unknown beat type {kind!r}")
    return frames, audio


DUCK = 0.3  # the beat's own sound under narration


def narrate(audio: np.ndarray, parts: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """The beat's sound with narration laid over it (each part from a sample offset), ducked while it speaks; what
    runs past the beat's end comes back, to go on under the next beat."""
    if not parts:
        return audio, np.zeros(0, np.float32)
    speech = np.zeros(max([audio.size] + [at + a.size for at, a in parts]), np.float32)
    for at, a in parts:
        speech[at: at + a.size] += a
    speaking = (np.abs(speech[: audio.size]) > 0).astype(np.float32)
    k = int(0.15 * SR)
    speaking = np.convolve(np.maximum.accumulate(speaking[::-1])[::-1] * np.maximum.accumulate(speaking), np.ones(k, np.float32) / k,
                           mode="same")  # ducked from the first word to the last, not between words
    out = audio * (1.0 - (1.0 - DUCK) * speaking) + speech[: audio.size]
    return out.astype(np.float32), speech[audio.size:]


def build(storyboard: Path, out: Path) -> float:
    """Render the beats one at a time and encode each as it is made (a whole demo held in memory is ~10 GB):
    H.264 yuv420p tagged BT.709 like cuesync's write_video, then the sound as AAC."""
    import av
    from fractions import Fraction

    from av.video.reformatter import ColorRange, Colorspace

    beats = json.loads(Path(storyboard).read_text(encoding="utf-8"))
    beats = beats["beats"] if isinstance(beats, dict) else beats
    n_frames, all_audio, carry = 0, [], np.zeros(0, np.float32)
    with av.open(str(out), "w") as container:
        vs = container.add_stream("libx264", rate=Fraction(FPS))
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        vs.options = {"crf": "20", "preset": "medium"}
        cc = vs.codec_context
        cc.colorspace, cc.color_primaries, cc.color_trc, cc.color_range = 1, 1, 1, 1  # BT.709, limited range
        astream = container.add_stream("aac", rate=SR, layout="mono")
        for k, b in enumerate(beats):
            frames, audio = render_beat(b)
            parts = [(0, carry)] if carry.size else []
            if b.get("vo"):
                parts.append((int(float(b.get("vo_at", 0.15)) * SR), read_audio(b["vo"], SR) * float(b.get("vo_gain", 1.0))))
            audio, carry = narrate(audio, parts)
            for fr in frames:
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24").reformat(
                    format="yuv420p", dst_colorspace=Colorspace.ITU709, dst_color_range=ColorRange.MPEG)
                for packet in vs.encode(vf):
                    container.mux(packet)
            n_frames += len(frames)
            all_audio.append(audio)
            print(f"beat {k + 1}/{len(beats)} {b['type']}: {len(frames) / FPS:.1f} s", flush=True)
            del frames
        for packet in vs.encode():
            container.mux(packet)
        if carry.size:
            print(f"the last narration runs {carry.size / SR:.2f} s past the end: cut", flush=True)
        samples = np.clip(np.concatenate(all_audio), -1.0, 1.0).astype(np.float32)
        for start in range(0, samples.size, 1024):
            af = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples[start:start + 1024].reshape(1, -1)), format="fltp", layout="mono")
            af.sample_rate, af.pts = SR, start
            for packet in astream.encode(af):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    return n_frames / FPS


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
