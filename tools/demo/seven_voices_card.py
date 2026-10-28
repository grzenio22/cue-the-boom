"""Build media/one_render_seven_voices.jpg: one stored LTX-2.5 palm render, its seven recorded performances, each at
its biggest hit (four frames after it).

    python tools/demo/seven_voices_card.py [--output path/to/one_render_seven_voices.jpg]

Only crop, resize and the caption bars touch the shots' pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHOTS = [  # gallery shot, cue, who performed it
    ("hand_palm__boom__fire_burst", "BOOM", "the author"),
    ("hand_palm__pfpfboom__again", "pf, pf, BOOM", "the author"),
    ("hand_palm__pfx4__again", "pf pf pf PF", "the author"),
    ("hand_palm__sadiquecat__again", "Boom", "Sadiquecat"),
    ("hand_palm__unfa__again", "Ka-Boom!", "unfa"),
    ("hand_palm__huw2k8__again", "Powf", "huw2k8"),
    ("hand_palm__jedimichael__again", "Boom", "Jedimichael"),
]
H, BAR, GAP = 420, 70, 8
INK = (18, 14, 12)


def tile(name: str, cue: str, who: str) -> np.ndarray:
    shot = ROOT / "samples/made_ahead/bending" / name
    hits = json.loads((shot / "receipt.json").read_text(encoding="utf-8"))["bending"]["hits"]
    cap = cv2.VideoCapture(str(shot / "final.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(hits, key=lambda h: h.get("size", 0))["frame"] + 4)
    ok, f = cap.read()
    if not ok:
        raise SystemExit(f"can't read {shot / 'final.mp4'}")
    f = f[int(f.shape[0] * 0.08):int(f.shape[0] * 0.80)]  # the hand and its fire, not the shot's footer
    f = cv2.resize(f, (int(f.shape[1] * H / f.shape[0]), H))
    bar = np.full((BAR, f.shape[1], 3), INK, np.uint8)
    cv2.putText(bar, cue, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (80, 190, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, who, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([f, bar], 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "media/one_render_seven_voices.jpg")
    args = parser.parse_args()
    tiles = [tile(*shot) for shot in SHOTS]
    row = tiles[0]
    for t in tiles[1:]:
        row = np.concatenate([row, np.full((t.shape[0], GAP, 3), INK, np.uint8), t], 1)
    cv2.imwrite(str(args.output), row, [cv2.IMWRITE_JPEG_QUALITY, 84])
    print(f"{args.output}: {row.shape[1]}x{row.shape[0]}")


if __name__ == "__main__":
    main()
