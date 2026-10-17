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



BG, FG, MUTED, GREY, ACCENT = (11, 11, 15), (244, 241, 234), (163, 159, 151), (120, 118, 114), (255, 176, 32)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def chart(raw: list, fin: list, path: Path, lo: int = -10, hi: int = 16) -> None:
    """Every shot as a dot, frames from its cue: the render as LTX-2.5 made it, then the finished shot."""
    from PIL import Image, ImageDraw

    w, h = 1600, 720
    left, right = 330, w - 60
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img, "RGBA")
    xf = lambda v: left + (min(max(v, lo), hi) - lo) / (hi - lo) * (right - left)  # noqa: E731
    rows = [("LTX-2.5 render,", "as it comes", raw, GREY, 200), ("the finished", "shot", fin, ACCENT, 450)]
    top, bottom = 90, 600
    d.rectangle([xf(-1), top, xf(1), bottom], fill=ACCENT + (34,))
    d.text(((xf(-1) + xf(1)) / 2, top - 14), "within one frame (42 ms)", font=_font(26), fill=ACCENT, anchor="md")
    for f in range(lo, hi + 1, 2):
        x = xf(f)
        d.line([x, bottom, x, bottom + 10], fill=MUTED, width=2)
        d.text((x, bottom + 16), f"{f:+d}" if f else "0", font=_font(24, bold=f == 0), fill=FG if f == 0 else MUTED,
               anchor="ma")
    d.line([left, bottom, right, bottom], fill=MUTED, width=2)
    d.line([xf(0), top, xf(0), bottom], fill=(244, 241, 234, 90), width=2)
    d.text((xf(0), bottom + 56), "frames from the cue (24 fps)", font=_font(24), fill=MUTED, anchor="ma")
    d.text((left, bottom + 56), "early", font=_font(24), fill=MUTED, anchor="la")
    d.text((right, bottom + 56), "late", font=_font(24), fill=MUTED, anchor="ra")
    r = 13
    for a, b, vals, colour, y in rows:
        d.text((40, y - 34), a, font=_font(30, bold=True), fill=FG)
        d.text((40, y + 4), b, font=_font(30, bold=True), fill=FG)
        got = [v for v in vals if v is not None]
        within = sum(abs(v) <= 1 for v in got)
        mean = float(np.mean([abs(v) for v in got])) if got else float("nan")
        d.text((40, y + 50), f"{within}/{len(got)} on the cue · {mean:.1f} frames off", font=_font(24),
               fill=colour if colour != GREY else MUTED)
        seen: dict[int, int] = {}
        for v in sorted(got, key=abs):
            k = int(round(min(max(v, lo), hi)))
            n = seen.get(k, 0)
            seen[k] = n + 1
            cy = y + (n // 2 + n % 2) * (2 * r + 3) * (-1 if n % 2 else 1)  # stack out from the row's line
            d.ellipse([xf(k) - r, cy - r, xf(k) + r, cy + r], fill=colour)
    img.save(path)


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
    chart(raw, fin, out / "timing.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
