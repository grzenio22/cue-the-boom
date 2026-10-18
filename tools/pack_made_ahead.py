"""Copy the scene shots made ahead of time into the repository, small enough to ship, so a fresh clone serves
them with no GPU (the gallery, the sample shots, and re-timing them by hand).

    python tools/pack_made_ahead.py runs/precomputed -o samples/made_ahead

Each shot keeps what serving and re-timing need (the shot, its breakdown, the render and its guide, the
receipt, the cue, the plan), re-encoded at a lower bitrate; the clean composite is left out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuesync" / "src"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from cuesync.media import read_audio, read_video, write_video  # noqa: E402

VIDEOS = {"final.mp4": 0, "render.mp4": 2, "breakdown.mp4": 4, "guide.mp4": 4, "guide_preview.mp4": 4}  # CRF over the shot's
FILES = ("receipt.json", "cue.json", "guide.json", "matte.png")


def pack(src: Path, out: Path, crf: int, modes: tuple[str, ...] = ("scene",), max_side: int | None = None) -> int:
    """Copy and re-encode every shot under ``src``/<mode>; ``max_side`` also scales the videos down to it, except
    the render, which stays at its size: the timing check measures it as LTX-2.5 made it."""
    n = 0
    for mode in modes:
        index = src / mode / "index.json"
        if not index.exists():
            continue
        entries = json.loads(index.read_text(encoding="utf-8"))
        dst_mode = out / mode
        if dst_mode.exists():
            shutil.rmtree(dst_mode)
        for entry in entries.values():
            s, d = src / mode / entry["dir"], dst_mode / entry["dir"]
            d.mkdir(parents=True)
            for name in FILES:
                if (s / name).exists():
                    shutil.copyfile(s / name, d / name)
            for name, extra in VIDEOS.items():
                if not (s / name).exists():
                    continue
                clip = read_video(s / name)
                frames = clip.frames
                h, w = frames.shape[1:3]
                if max_side and max(h, w) > max_side and name != "render.mp4":
                    k = max_side / max(h, w)
                    size = (int(w * k) // 2 * 2, int(h * k) // 2 * 2)
                    frames = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])
                try:
                    audio = read_audio(s / name, 48000)
                except Exception:
                    audio = None
                write_video(d / name, frames, fps=round(clip.fps_nominal), audio=audio, crf=crf + extra, preset="slow")
            n += 1
        (dst_mode / "index.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", nargs="?", default="runs/precomputed")
    ap.add_argument("-o", "--out", default="samples/made_ahead")
    ap.add_argument("--crf", type=int, default=22)
    a = ap.parse_args()
    n = pack(Path(a.src), Path(a.out), a.crf)
    size = sum(f.stat().st_size for f in Path(a.out).rglob("*") if f.is_file())
    print(f"{n} shots -> {a.out} ({size / 1e6:.1f} MB)")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
