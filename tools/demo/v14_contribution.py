"""Prepare the eight-second v14 recap and the complete, pitch-preserved voice.

Reuse v13's exact card layout and v11's stored fire. The source panel moves
through the same short fire window already shown in the demo; stage emphasis
advances through asset, tracking/matte, voice and the finished lighting result.
Only new12.wav is played faster (1.28x); no words or phrases are removed.
All generated assets and voice manifests are written into demo_v14.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import av
import numpy as np
from PIL import Image, ImageDraw

from build_demo import ACCENT, BG, FG, FPS, H, SR, W, Clip
from v11_cards import LINE
from v13_contribution import (
    CAPTION_TOP, PANEL_HEIGHT, PANEL_LEFTS, PANEL_TOP, PANEL_WIDTH, border, title,
)


TEMPO = 1.28
DURATION = 8.0
# Frame-exact holds: overview, source, matte, voice, finished result.
STAGES = (("innovation", 24, None), ("innovation_asset", 48, 2),
          ("innovation_track", 36, 1), ("innovation_voice", 24, 0),
          ("innovation_result", 60, 3))


def portrait(frame: np.ndarray) -> Image.Image:
    """Match the original source-panel crop in v11_cards.pipeline exactly."""
    im = Image.fromarray(frame)
    scale = max(PANEL_WIDTH / im.width, PANEL_HEIGHT / im.height) * 1.3
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    x = int(np.clip(.52 * im.width - PANEL_WIDTH / 2, 0, im.width - PANEL_WIDTH))
    y = int(np.clip(.42 * im.height - PANEL_HEIGHT / 2, 0, im.height - PANEL_HEIGHT))
    return im.crop((x, y, x + PANEL_WIDTH, y + PANEL_HEIGHT))


def recap(root: Path, run: Path, *, stages=STAGES, duration=DURATION,
          source_card: Path | None = None, source_clip: Path | None = None) -> Path:
    original = Image.open(source_card or root / "runs/demo_v13/cards/pipeline_overview.png").convert("RGB")
    if original.size != (W, H):
        raise ValueError(f"Unexpected recap card size: {original.size}")
    neutral = original.copy()
    draw = ImageDraw.Draw(neutral)
    border(draw, 2, BG, 6)
    border(draw, 2, LINE, 3)
    title(draw, 2, FG)
    # Keep the original 2.4--3.3 s source window, slowed over the shorter recap.
    source = Clip(str(source_clip or root / "runs/demo_v11/clips/how_raw.mp4"), 2.4, duration, speed=.9 / duration)
    path = run / "cards/pipeline_motion.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(FPS))
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        cc = stream.codec_context
        cc.colorspace, cc.color_primaries, cc.color_trc, cc.color_range = 1, 1, 1, 1
        index = 0
        for _, count, selected in stages:
            for _ in range(count):
                image = original.copy() if selected is None else neutral.copy()
                image.paste(portrait(source.frame(index)), (PANEL_LEFTS[2], PANEL_TOP))
                if selected is not None:
                    for panel, x in enumerate(PANEL_LEFTS):
                        if panel == selected:
                            continue
                        bounds = (x - 8, PANEL_TOP - 8, x + PANEL_WIDTH + 8, CAPTION_TOP)
                        crop = image.crop(bounds)
                        image.paste(Image.blend(crop, Image.new("RGB", crop.size, BG), .64), bounds[:2])
                    draw = ImageDraw.Draw(image)
                    border(draw, selected, ACCENT, 6)
                    title(draw, selected, ACCENT)
                frame = av.VideoFrame.from_image(image).reformat(format="yuv420p", dst_colorspace=1,
                                                               dst_color_range=1)
                for packet in stream.encode(frame):
                    container.mux(packet)
                index += 1
        for packet in stream.encode():
            container.mux(packet)
    if index != round(duration * FPS):
        raise ValueError(f"Recap has {index} frames instead of {round(duration * FPS)}")
    return path


def narration(root: Path, run: Path) -> Path:
    source_dir = root / "runs/demo_v12/vo12"
    source = source_dir / "new12.wav"
    duration = json.loads((source_dir / "durations.json").read_text(encoding="utf-8"))["new12"] / TEMPO
    samples = round(duration * SR)
    dest = run / "vo14/new12.wav"
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for pitch-preserving narration timing")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-af", f"aresample={SR},atempo={TEMPO},apad,atrim=end_sample={samples}",
                    "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(dest)], check=True)
    with wave.open(str(dest), "rb") as audio:
        if audio.getnframes() != samples or audio.getframerate() != SR:
            raise ValueError("Retimed narration has an unexpected sample count")
    spoken = json.loads((source_dir.parent / "vo_lines.json").read_text(encoding="utf-8"))["new12"]
    phrases = json.loads((source_dir / "caption_phrases.json").read_text(encoding="utf-8"))["new12"]
    payloads = {
        run / "vo_lines.json": {"new12": spoken},
        dest.parent / "durations.json": {"new12": duration},
        dest.parent / "caption_phrases.json": {"new12": [[a / TEMPO, z / TEMPO, text] for a, z, text in phrases]},
        dest.parent / "provenance.json": dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                             tempo=TEMPO, pitch_preserved=True, duration=duration,
                                             complete_transcript=spoken, words_removed=0),
    }
    for path, payload in payloads.items():
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--run", type=Path, default=Path("D:/openHiggsfield/runs/demo_v14"))
    args = parser.parse_args()
    print(narration(args.root, args.run))
    print(recap(args.root, args.run))


if __name__ == "__main__":
    main()
