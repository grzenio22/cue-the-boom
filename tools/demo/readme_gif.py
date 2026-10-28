"""Build media/one_render_five_performances.gif: one stored LTX-2.5 palm render, five recorded performances side by
side, opening on the fires at their biggest.

    python tools/demo/readme_gif.py [--output path/to/one_render_five_performances.gif]

Needs ffmpeg on the PATH. Only crop, resize, the caption bars and GIF quantisation touch the shots' pixels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHOTS = [  # gallery shot, cue, who performed it
    ("hand_palm__boom__fire_burst", "BOOM", "the author"),
    ("hand_palm__pfpfboom__again", "pf, pf, BOOM", "the author"),
    ("hand_palm__pfx4__again", "pf pf pf PF", "the author"),
    ("hand_palm__huw2k8__again", "Powf", "huw2k8 (CC0)"),
    ("hand_palm__unfa__again", "Ka-Boom!", "unfa (CC0)"),
]
FRAMES = range(16, 84, 2)  # every other frame of the shots' 84, from just before the first hit
OPEN_AT = 64               # the loop starts here: the five fires at their biggest
H, BAR, HEAD, GAP, WIDTH, FPS = 330, 54, 34, 6, 1040, 12
INK = (18, 14, 12)


def read(name: str) -> list[np.ndarray]:
    cap, frames = cv2.VideoCapture(str(ROOT / "samples/made_ahead/bending" / name / "final.mp4")), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def panel(shots: list[list[np.ndarray]], k: int) -> np.ndarray:
    tiles = []
    for frames, (_, cue, who) in zip(shots, SHOTS):
        f = frames[min(k, len(frames) - 1)]
        f = f[int(f.shape[0] * 0.1):int(f.shape[0] * 0.8)]  # the hand and its fire, not the shot's footer
        f = cv2.resize(f, (int(f.shape[1] * H / f.shape[0]), H), interpolation=cv2.INTER_AREA)
        bar = np.full((BAR, f.shape[1], 3), INK, np.uint8)
        cv2.putText(bar, cue, (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.62, (80, 190, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, who, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 205, 205), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([f, bar], 0))
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, np.full((t.shape[0], GAP, 3), INK, np.uint8), t], 1)
    head = np.full((HEAD, row.shape[1], 3), INK, np.uint8)
    cv2.putText(head, "One LTX render, five performances, no new generation", (8, 24), cv2.FONT_HERSHEY_DUPLEX, 0.62,
                (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([head, row], 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "media/one_render_five_performances.gif")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("needs ffmpeg on the PATH")
    shots = [read(name) for name, _, _ in SHOTS]
    order = list(FRAMES)
    order = order[order.index(OPEN_AT):] + order[:order.index(OPEN_AT)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, k in enumerate(order):
            cv2.imwrite(str(Path(tmp) / f"f{i:03d}.png"), panel(shots, k))
        frames, palette = str(Path(tmp) / "f%03d.png"), str(Path(tmp) / "palette.png")
        run = lambda *a: subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)  # noqa: E731
        run("-framerate", str(FPS), "-i", frames,
            "-vf", f"scale={WIDTH}:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff", palette)
        run("-framerate", str(FPS), "-i", frames, "-i", palette, "-lavfi",
            f"scale={WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            "-loop", "0", str(args.output))
    print(f"{args.output}: {len(order)} frames, {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
