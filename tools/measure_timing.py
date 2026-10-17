"""How well do scene shots land on their cue? Every hit, as LTX-2.5 rendered it and in the finished shot.

    python tools/measure_timing.py runs/precomputed/scene runs/precomputed/scene.r5 runs/extra_takes -o runs/timing

Where each hit shows on the raw render (render.mp4, before re-timing) is read from tools/timing_annotations.json:
marked by eye, frame by frame (the first frame the effect is clearly there; for a blow-out, the first with no
flame left; nothing where the model merged a hit into its neighbour). The "calibration" shots set the
pipeline's onset threshold; the "held_out" shots were marked before it ran on them, and are reported apart.
A shot with no marks falls back to a detector (the steepest rise of the visibly changed share of its region).
The raw error is where the render shows the hit, laid on the plate from the render's start, minus its cue frame
(a burst one frame before its sound: light beats sound). The finished shot is the render moved by the shot's
re-time, so its error is the same mark moved by the same frames (the flash added to it can't sway this).
Flames that should stop with the voice: where their flame-coloured area falls below a quarter of its peak.
Writes timing.csv (one row per hit), timing.md and timing.png.
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
from cuesync.transfer import effect_region  # noqa: E402

CHANGED = 30  # 8-bit, largest channel: a pixel the effect visibly changed
WARM = (170, 70)  # a flame's pixel: red above this, red minus blue above this
SMALL = 4  # measured at a quarter of the render's size


def _rise(c: np.ndarray) -> np.ndarray:
    """Frame-to-frame rise of a curve, lightly smoothed (rise[i]: from frame i-1 to i)."""
    cs = np.convolve(np.pad(c, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return np.concatenate([[0.0], np.diff(cs)])


def onset_in(c: np.ndarray, lo: int, hi: int, share: float = 0.35, least: float = 0.004) -> int | None:
    """First frame in [lo, hi] whose rise is at least ``share`` of the steepest rise there."""
    r = _rise(c)
    lo, hi = max(1, lo), min(len(c) - 1, hi)
    if hi < lo or r[lo:hi + 1].max() < least:
        return None
    return lo + int(np.nonzero(r[lo:hi + 1] >= share * r[lo:hi + 1].max())[0][0])


def flame_gone(frames: np.ndarray, point: tuple[float, float], ref_frames: int, hold: int = 3) -> int | None:
    """The frame a candle's flame goes out: the brightest pixel near the wick falls halfway from lit to out."""
    h, w = frames.shape[1:3]
    x, y, r = int(point[0] * w), int(point[1] * h), max(2, int(0.04 * w))
    luma = frames[:, max(0, y - 2 * r): y + r, max(0, x - r): x + r].astype(np.float32).mean(-1)
    peak = luma.reshape(len(frames), -1).max(1)
    lit, out = float(np.median(peak[:ref_frames])), float(np.median(peak[-10:]))
    if lit - out < 20:
        return None
    below = peak < (lit + out) / 2
    for i in range(ref_frames, len(peak) - hold):
        if below[i: i + hold].all():
            return i
    return None


def measure(d: Path, eye: list | None = None, group: str = "") -> dict | None:
    rc = json.loads((d / "receipt.json").read_text())
    if rc.get("mode") != "scene" or not (d / "render.mp4").exists():
        return None
    g = json.loads((d / "guide.json").read_text())
    al, eff = rc["alignment"], rc["effect"]
    start, lead, retime = int(g["start"]), int(al["lead"]), int(al.get("retime_frames", 0))
    shift = int(al["auto_shift"]) + retime  # the finished shot is the render moved by this many frames
    events = g["events"]
    gw, gh = g["size"]
    size = (gw // SMALL, gh // SMALL)
    render = read_video(d / "render.mp4", size=size).frames.astype(np.int16)
    region = effect_region(g["blobs"], (gw, gh), size)
    ref_frames = max(1, events[0]["at"] - 3)
    ref = np.median(render[:ref_frames], axis=0)
    diff = np.abs(render - ref).max(-1)
    changed = np.array([float((f > CHANGED)[region].mean()) for f in diff])
    warm = (render[..., 0] > WARM[0]) & (render[..., 0] - render[..., 2] > WARM[1])
    fire = np.array([float(((diff[i] > CHANGED) & warm[i])[region].mean()) for i in range(len(render))])
    want = [ev["at"] - lead for ev in events]  # render frame each hit should show on
    if eye is not None:  # marked by eye
        on = list(eye)
        want = want[: len(on)]
    elif eff["behaviour"] == "extinguish":
        on = [flame_gone(render, tuple(g["point"]), ref_frames)]
        want = want[:1]
    else:
        main = max(range(len(events)), key=lambda k: events[k]["size"])
        on = [None] * len(events)
        on[main] = onset_in(changed, want[main] - 16, want[main] + 16)
        late = 0 if on[main] is None else on[main] - want[main]
        for k in range(len(events)):
            if k != main:
                on[k] = onset_in(changed, want[k] + late - 6, want[k] + late + 6)
    hits = [{"shot": f"{d.parent.name}/{d.name}", "effect": eff["id"], "set": group or "detector", "hit": k + 1, "of": len(want),
             "cue_frame": start + want[k] + retime,
             "raw_error": None if o is None else o - want[k] - retime,
             "final_error": None if o is None else o + shift - want[k] - retime}
            for k, o in enumerate(on)]
    stop = None
    if eff["fluid"] == "fire" and eff["behaviour"] in ("jet", "plume"):
        v = rc["voice"].get("voice_end_frame")
        if v is not None and fire.max() > 0.01:
            pk = int(np.argmax(fire))
            low = np.nonzero(fire[pk:] < 0.25 * fire[pk])[0]
            s_ = None if not low.size else pk + int(low[0])
            stop = {"shot": f"{d.parent.name}/{d.name}", "effect": eff["id"], "voice_end": v,
                    "raw_latency": None if s_ is None else start + s_ - v,
                    "final_latency": None if s_ is None else start + shift + s_ - v}
    return {"hits": hits, "stop": stop}


def shot_dirs(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "receipt.json").exists():
            out.append(p)
        else:
            out += sorted(d for d in p.iterdir() if (d / "receipt.json").exists())
    return out


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
    # dots stack out from each row's line; size them so the tallest stack stays inside its own row
    tallest = max((sum(1 for v in vals if v is not None and int(round(min(max(v, lo), hi))) == k)
                   for _, _, vals, _, _ in rows for k in range(lo, hi + 1)), default=1)
    step = min(29.0, 120.0 / max(1, tallest // 2))
    r = max(4, int((step - 3) / 2))
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
            cy = y + (n // 2 + n % 2) * step * (-1 if n % 2 else 1)  # stack out from the row's line
            d.ellipse([xf(k) - r, cy - r, xf(k) + r, cy + r], fill=colour)
    img.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="runs/timing")
    ap.add_argument("--marks", default=str(Path(__file__).with_name("timing_annotations.json")))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    marks = json.loads(Path(a.marks).read_text(encoding="utf-8")) if Path(a.marks).exists() else {}
    eye = {name: (group, m) for group in ("calibration", "held_out") for name, m in marks.get(group, {}).items()}
    key = lambda d: f"{d.parent.name}/{d.name}"  # noqa: E731
    found = [r for r in (measure(d, *(eye[key(d)][::-1] if key(d) in eye else (None, ""))) for d in shot_dirs(a.paths)) if r]
    rows = [h for r in found for h in r["hits"]]
    stops = [r["stop"] for r in found if r["stop"]]
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
    later = [r for r in rows if r["of"] > 1]
    lines = ["| shot | effect | set | hit | cue frame | raw LTX error | finished error |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['shot']} | {r['effect']} | {r['set']} | {r['hit']}/{r['of']} | {r['cue_frame']} | "
              f"{('merged' if r['of'] > 1 else 'unclear') if r['raw_error'] is None else r['raw_error']} | {'' if r['final_error'] is None else r['final_error']} |"
              for r in rows]
    if stops:
        lines += ["", "| shot | effect | voice ends (frame) | flame stops, raw | flame stops, finished |", "|---|---|---|---|---|"]
        lines += [f"| {s_['shot']} | {s_['effect']} | {s_['voice_end']} | {s_['raw_latency']} | {s_['final_latency']} |"
                  for s_ in stops]
    fl = [s_["final_latency"] for s_ in stops if s_["final_latency"] is not None]
    by = lambda g, k: [r[k] for r in rows if r["set"] == g]  # noqa: E731
    merged = sum(r["raw_error"] is None and r["of"] > 1 for r in rows)
    unclear = sum(r["raw_error"] is None and r["of"] == 1 for r in rows)
    lines += ["", f"Every hit within one frame (42 ms) of its cue: raw render {rate(raw)}, finished shot {rate(fin)}.",
              f"Mean absolute error: raw {mae(raw):.1f} frames, finished {mae(fin):.1f} frames.",
              f"Held-out takes only: raw {rate(by('held_out', 'raw_error'))}, finished {rate(by('held_out', 'final_error'))}."
              if by("held_out", "raw_error") else "",
              f"{merged} rhythm hit(s) the model merged into a neighbour have no onset of their own and are not counted." if merged else "",
              f"{unclear} take(s) whose start can't be called by eye are not counted." if unclear else "",
              f"Rhythm hits (each one measured): raw {rate([r['raw_error'] for r in later])}, "
              f"finished {rate([r['final_error'] for r in later])}." if later else "",
              f"Flames that should stop with the voice stop {int(np.median(fl))} frames after it (median, finished; "
              f"range {min(fl)} to {max(fl)})." if fl else "",
              "Onsets marked by eye on the raw render; the finished error is that mark moved by the shot's re-time."]
    (out / "timing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(x for x in lines[-7:] if x))
    chart(raw, fin, out / "timing.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
