"""Build the demo video from a storyboard: a JSON list of beats, rendered to one 1920x1080 MP4 with sound.

    python tools/demo/build_demo.py tools/demo/storyboard.json -o runs/demo/demo.mp4

Beats (all times in seconds; text may use \\n):
  {"type": "title", "dur": 3, "text": "...", "sub": "..."}
  {"type": "clip", "src": "path.mp4", "start": 0, "dur": 4, "caption": "...", "side": "...", "label": "..."}
      a vertical (or any) clip, fitted in the centre; caption under it, side text on the left
  {"type": "triptych", "srcs": ["a.mp4", "b.mp4", "c.mp4"], "labels": [...], "start": 0, "dur": 5,
   "audio": 0 | 1 | 2 | "mix", "caption": "..."}       three clips side by side, same timeline
  {"type": "image", "src": "path.png", "dur": 3, "caption": "...", "zoom": 1.08}   a still with a slow push-in
  a clip may add "speed": 8 (silent while sped up) and "note": "sped up x8" (a tag at the top right),
  "crop": [x0, y0, x1, y1] (fractions, to zoom in), and "audio_src" + "audio_from" + "audio_at" (another file's
  sound, from audio_from s in it, starting audio_at s into the beat), "wave": true (the beat's sound drawn under
  the clip, a playhead running along it) and "pops": [[t, "KA-BOOM"], ...] (the noise word, big, at t s)
      (an image may add "crop" too; "split": [0.4] cuts a wide image at those fractions of its width and stacks
      the parts as rows)
  {"type": "versus", "shot": "runs/.../shot", "start": 0.4, "dur": 2.4, "speed": 1, "labels": ["...", "..."],
   "cue_at": 1.0}   the shot's raw render (as LTX-2.5 made it, on the plate's timeline) beside the finished shot,
      the sound drawn under both with a line where the noise is (cue_at: s on the shot's timeline)
  {"type": "credits", "dur": 6, "title": "...", "lines": ["...", "  · a sub-line", "..."], "url": "..."}
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
    if text:  # what the shot is made of: as readable as the claims it qualifies
        ImageDraw.Draw(img).text((W - 24, H - 12), text, font=font(32), fill=(214, 210, 202), anchor="rd")


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


def read_window(src: str | Path, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """The frames of a video shown from t0 to t1 s, with their timestamps, decoded from a seek near t0 (a long
    screen recording is never decoded, or held, whole)."""
    import av

    frames, times = [], []
    with av.open(str(src)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
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
            frames.append(fr.to_ndarray(format="rgb24"))
            times.append(fr.time)
    order = np.argsort(times)
    return np.stack([frames[k] for k in order]), np.asarray(times)[order]


class Clip:
    """A stretch of a video from ``start`` s, played at ``speed`` (frames by their own timestamps, so a screen
    recording's uneven frame rate is fine). A sped-up stretch is silent."""

    def __init__(self, src: str, start: float, dur: float, speed: float = 1.0, crop: list[float] | None = None):
        self.frames, self.times = read_window(src, start, start + dur * speed)
        if crop:  # [x0, y0, x1, y1] as fractions: zoom in on part of a screen recording
            h, w = self.frames.shape[1:3]
            x0, y0, x1, y1 = int(crop[0] * w) // 2 * 2, int(crop[1] * h) // 2 * 2, int(crop[2] * w) // 2 * 2, int(crop[3] * h) // 2 * 2
            self.frames = self.frames[:, y0:y1, x0:x1]
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
            d.text((520 + (50 if sub else 0), 200 + k * 60), line.strip(), font=font(38), fill=MUTED if sub or line.startswith("·") else FG)
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
        for i in range(n):
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
                audio[at: at + n_] += a[:n_]
        tall = c.frames.shape[1] > c.frames.shape[2]  # a portrait clip: its sound goes beside it, not under it
        box_h = H - (150 if b.get("caption") else 60) - (130 if b.get("wave") and not tall else 0)
        env = envelope(audio, n) if b.get("wave") else None
        for i in range(n):
            img = Image.new("RGB", (W, H), BG)
            im = fit(c.frame(i), W - 900 if b.get("side") else W - 120, box_h)
            x0 = (W - im.width) // 2 + (220 if b.get("side") else 0)
            y0 = (box_h - im.height) // 2 + 20
            img.paste(im, (x0, y0))
            d = ImageDraw.Draw(img)
            if env is not None:  # the sound: the playhead reaches the noise as the effect hits
                if not tall:
                    wave(d, x0, y0 + im.height + 30, im.width, 90, env, i)
                elif b.get("side"):  # under the side text
                    wave(d, 90, H // 2 + 130, max(200, x0 - 180), 110, env, i)
                else:  # to the right of the clip
                    wave(d, x0 + im.width + 70, y0 + im.height - 220, W - 120 - (x0 + im.width + 70), 110, env, i)
            pop(d, x0 + im.width // 2, y0 + int(im.height * 0.8), i / FPS, b.get("pops"))
            if b.get("side"):  # the label heads the side text (the shot carries its own effect label)
                y = H // 2 - 120
                if b.get("label"):
                    text_block(d, (90, y - 110), b["label"], 60, ACCENT, bold=True, max_w=x0 - 160)
                text_block(d, (90, y), b["side"], 46, FG, max_w=x0 - 160, spacing=16)
            elif b.get("label"):
                d.text((x0 + 20, 40), b["label"], font=font(40, True), fill=ACCENT)
            if b.get("caption"):
                text_block(d, (W // 2, H - 80), b["caption"], 42, FG, anchor="mm", max_w=W - 200)
            if b.get("note"):  # e.g. that a stretch is sped up
                f = font(30, True)
                tw = d.textlength(b["note"], font=f)
                d.rounded_rectangle([W - tw - 70, 24, W - 30, 74], radius=12, fill=ACCENT)
                d.text((W - 50, 49), b["note"], font=f, fill=BG, anchor="rm")
            stamp(img, b.get("stamp"))
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
    else:
        raise ValueError(f"unknown beat type {kind!r}")
    return frames, audio


def build(storyboard: Path, out: Path) -> float:
    """Render the beats one at a time and encode each as it is made (a whole demo held in memory is ~10 GB):
    H.264 yuv420p tagged BT.709 like cuesync's write_video, then the sound as AAC."""
    import av
    from fractions import Fraction

    from av.video.reformatter import ColorRange, Colorspace

    beats = json.loads(Path(storyboard).read_text(encoding="utf-8"))
    beats = beats["beats"] if isinstance(beats, dict) else beats
    n_frames, all_audio = 0, []
    with av.open(str(out), "w") as container:
        vs = container.add_stream("libx264", rate=Fraction(FPS))
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        vs.options = {"crf": "20", "preset": "medium"}
        cc = vs.codec_context
        cc.colorspace, cc.color_primaries, cc.color_trc, cc.color_range = 1, 1, 1, 1  # BT.709, limited range
        astream = container.add_stream("aac", rate=SR, layout="mono")
        for k, b in enumerate(beats):
            frames, audio = render_beat(b)
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
