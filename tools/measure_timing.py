"""How well do scene shots land on their cue? Raw LTX-2.5 render vs the finished, re-timed shot.

    python tools/measure_timing.py runs/precomputed/scene [more shot folders or parents ...] -o runs/timing

For every scene shot (a folder with receipt.json, guide.json, render.mp4, composite.mp4):
- the cue frame: where the hit is (the sound's onset; a burst a frame before it, light beats sound);
- raw error: where the render, used as it came out of LTX-2.5, shows the effect, minus the cue frame
  (measured on the render, as the pipeline does);
- final error: where the finished shot shows it, minus the cue frame, measured on the finished shot with a
  different detector from the pipeline's: the first frame whose frame-to-frame change inside the painted region
  jumps above a third of its largest jump (a rhythm: the largest jump near its biggest hit; a blow-out: the
  frame its flame is gone, since flicker would trip a jump detector).
Writes timing.csv, timing.md (per shot + hit rates within +-1 frame) and timing.png (raw vs final per shot).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cuesync" / "src"))
from cuesync.media import read_video  # noqa: E402
from cuesync.scene_shot import flame_out  # noqa: E402
from cuesync.transfer import effect_region  # noqa: E402

JUMP_SHARE = 0.33


def first_jump(frames: np.ndarray, region: np.ndarray, after: int = 1) -> int | None:
    """First frame whose change from the previous frame (inside the region) exceeds JUMP_SHARE of the largest."""
    small = frames[:, ::4, ::4].astype(np.float32)
    reg = region[::4, ::4]
    d = np.array([0.0] + [float(np.abs(small[i] - small[i - 1]).max(-1)[reg].mean()) for i in range(1, len(small))])
    peak = float(d[after:].max()) if len(d) > after else 0.0
    if peak < 1.5:
        return None
    hit = np.nonzero(d[after:] > JUMP_SHARE * peak)[0]
    return int(after + hit[0]) if hit.size else None


def biggest_jump(frames: np.ndarray, region: np.ndarray, around: int, half: int = 8) -> int | None:
    """For a rhythm: the frame of the largest frame-to-frame change inside the region near ``around``."""
    small = frames[:, ::4, ::4].astype(np.float32)
    reg = region[::4, ::4]
    lo, hi = max(1, around - half), min(len(small), around + half + 1)
    if hi <= lo:
        return None
    d = [float(np.abs(small[i] - small[i - 1]).max(-1)[reg].mean()) for i in range(lo, hi)]
    return lo + int(np.argmax(d)) if max(d) >= 1.5 else None


def shot_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "receipt.json").exists():
            out.append(p)
        else:
            out += sorted(d for d in p.iterdir() if (d / "receipt.json").exists())
    return out


def measure(d: Path) -> dict | None:
    rc = json.loads((d / "receipt.json").read_text())
    if rc.get("mode") != "scene":
        return None
    g = json.loads((d / "guide.json").read_text())
    al = rc["alignment"]
    start, lead, painted = int(g["start"]), int(al["lead"]), int(al["painted_hit"])
    cue = start + painted - lead + int(al.get("retime_frames", 0))  # where the hit belongs on the shot
    comp = read_video(d / "composite.mp4").frames
    h, w = comp.shape[1:3]
    region = effect_region(g["blobs"], tuple(g["size"]), (w, h))
    raw = None if al["model_onset"] is None else (start + al["model_onset"]) - cue
    rhythm = bool(rc.get("hits"))  # a rhythm lines up its biggest hit: measure that one
    if rc["effect"]["behaviour"] == "extinguish":  # a blow-out: when the flame is gone in the finished shot
        final_on = flame_out(comp, region, max(1, cue - 8))
    else:
        final_on = biggest_jump(comp, region, cue) if rhythm else first_jump(comp, region, after=max(1, cue - 12))
    return {"shot": d.name, "effect": rc["effect"]["id"], "cue_frame": cue,
            "raw_error": raw, "final_error": None if final_on is None else final_on - cue,
            "auto_shift": al["auto_shift"], "clamped": al.get("clamped", False)}


def rate(vals: list[int | None], tol: int = 1) -> str:
    v = [x for x in vals if x is not None]
    return f"{sum(abs(x) <= tol for x in v)}/{len(v)}" if v else "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="runs/timing")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (measure(d) for d in shot_dirs(a.paths)) if r]
    if not rows:
        print("no scene shots found")
        return 1
    with open(out / "timing.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    raw = [r["raw_error"] for r in rows]
    fin = [r["final_error"] for r in rows]
    mae = lambda v: np.mean([abs(x) for x in v if x is not None]) if any(x is not None for x in v) else float("nan")  # noqa: E731
    lines = ["| shot | effect | cue frame | raw LTX error | final error |", "|---|---|---|---|---|"]
    lines += [f"| {r['shot']} | {r['effect']} | {r['cue_frame']} | {r['raw_error']} | {r['final_error']} |" for r in rows]
    lines += ["", f"Within one frame (42 ms) of the cue: raw render {rate(raw)}, finished shot {rate(fin)}.",
              f"Mean absolute error: raw {mae(raw):.1f} frames, finished {mae(fin):.1f} frames.",
              "Final error is measured with a different detector than the one the pipeline re-times with."]
    (out / "timing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-3:]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.7), 3.6), dpi=150)
        ax.bar(x - 0.2, [v if v is not None else 0 for v in raw], 0.4, label="LTX-2.5 render as it comes", color="#888")
        ax.bar(x + 0.2, [v if v is not None else 0 for v in fin], 0.4, label="finished shot", color="#ff8c00")
        ax.axhspan(-1, 1, color="#ff8c00", alpha=0.08)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xticks(x, [r["shot"].replace("__", "\n")[:28] for r in rows], fontsize=6, rotation=0)
        ax.set_ylabel("frames from the cue (24 fps)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "timing.png")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
