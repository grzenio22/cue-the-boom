"""Click ground truth for the element-bending tests: the emitter point on every 4th frame, the held stretches, the
strikes, and about 10 hand+forearm polygons (the 3 blurriest frames among them). Saves
``cuesync/tests/bending/gt/<clip>.json`` (see ``cuesync/tests/bending/test_metrics.py`` for the format), keeping what
was there, so it can be run again to fix or extend a file.

The core venv's OpenCV is headless, so run it from the tracking venv:

    runs/bending/venv-track/Scripts/python.exe tools/label_points.py runs/bending/plates/A.mp4 --emitter two_finger

Points (the default mode)
    left click      the emitter on this frame          right click  clear it
    x               not visible on this frame (the hand is there, the point hidden)
    space / n, b    next / previous frame to label (every --every frames); . and , step one frame
    h               start / end a held stretch here    f            start / end a stretch with the hand in frame
    e               an arm locks here (extension)      o            a palm opens here (pressing again removes it)
    t               show / hide the tracker's emitter (runs/.../hand_track.json, --track), off by default
Masks (m switches)
    left click      add a corner of the hand+forearm outline       right click  remove the last one
    enter           close it for this frame            c            clear this frame's outline
    k               go to the next suggested frame (the 3 blurriest where the hand is, then spread through the clip)
Always: s saves, q or Esc saves and quits. A magnifier follows the mouse in the top-left corner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TARGETS = {
    "two_finger": "midway between the index and middle fingertips (each a third of a finger's width in from its end)",
    "index_tip": "the index fingertip: on the finger's axis, a third of a finger's width in from its end",
    "palm": "the palm centre: from the wrist to the middle of the knuckle row, 4/5 of the way to the knuckles",
    "fist": "the middle of the knuckle row (index to little finger)",
}
VIEW_W, VIEW_H, ZOOM, MAG = 1600, 900, 4, 220


def decode(path: str) -> tuple[list[bytes], float, int, int]:
    """Every frame JPEG-packed in memory (a 10 s 1080p clip is ~100 MB, not 1.5 GB), its rate and size."""
    import av

    packed = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate or 24)
        for f in c.decode(s):
            img = f.to_ndarray(format="bgr24")
            packed.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
    h, w = img.shape[:2]
    return packed, fps, w, h


def ranges_toggle(ranges: list, open_at: list, t: int) -> None:
    """Start a range at ``t``, or close the open one there."""
    if open_at:
        a = open_at.pop()
        ranges.append([min(a, t), max(a, t)])
        ranges.sort()
    else:
        open_at.append(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--emitter", choices=list(TARGETS))
    ap.add_argument("--hand", choices=("left", "right"), help="the performer's hand that is labelled")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", help="default: cuesync/tests/bending/gt/<clip>.json")
    ap.add_argument("--track", help="a hand_track.json to show with t")
    a = ap.parse_args(argv)

    clip = Path(a.video).stem
    out = Path(a.out or Path(__file__).resolve().parents[1] / "cuesync" / "tests" / "bending" / "gt" / f"{clip}.json")
    gt = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    print(f"decoding {a.video} ...")
    packed, fps, W, H = decode(a.video)
    T = len(packed)
    emitter = a.emitter or gt.get("emitter") or "two_finger"
    gt.update({"clip": clip, "video": Path(a.video).as_posix(), "W": W, "H": H, "fps": fps, "emitter": emitter})
    if a.hand:
        gt["hand"] = a.hand
    points = {int(k): v for k, v in gt.get("points", {}).items()}
    masks = {int(k): v for k, v in gt.get("masks", {}).items()}
    held, in_frame = gt.get("held", []), gt.get("in_frame", [])
    events = gt.get("events", [])
    track = json.loads(Path(a.track).read_text(encoding="utf-8")) if a.track else None

    sharp = []  # variance of the Laplacian: low = blurry
    for b in packed:
        g = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        sharp.append(cv2.Laplacian(cv2.resize(g, (W // 4, H // 4)), cv2.CV_64F).var())
    where = sorted({t for r in in_frame for t in range(r[0], r[1] + 1)}) or list(range(T))
    blurry = sorted(where, key=lambda t: sharp[t])[:3]
    spread = [where[int(i)] for i in np.linspace(0, len(where) - 1, 9)[1:-1]]
    suggest = blurry + [t for t in spread if t not in blurry][:7]

    view = min(VIEW_W / W, VIEW_H / H, 1.0)
    state = {"t": 0, "mode": "points", "mouse": (0, 0), "poly": [], "show_track": False}
    open_held, open_frame = [], []

    def save():
        gt.update({"provisional": False, "labeller": "human (tools/label_points.py)",
                   "points": {str(k): points[k] for k in sorted(points)},
                   "masks": {str(k): masks[k] for k in sorted(masks)}, "held": held, "in_frame": in_frame,
                   "events": sorted(events, key=lambda e: e["frame"]), "blurry": blurry})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(gt, indent=1), encoding="utf-8")
        print(f"saved {out}: {sum(v is not None for v in points.values())} points, {len(masks)} masks")

    def on_mouse(ev, x, y, flags, _):
        fx, fy = x / view, y / view
        state["mouse"] = (fx, fy)
        t = state["t"]
        if state["mode"] == "points":
            if ev == cv2.EVENT_LBUTTONDOWN:
                points[t] = [round(fx, 1), round(fy, 1)]
            elif ev == cv2.EVENT_RBUTTONDOWN:
                points.pop(t, None)
        else:
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["poly"].append([round(fx, 1), round(fy, 1)])
            elif ev == cv2.EVENT_RBUTTONDOWN and state["poly"]:
                state["poly"].pop()

    cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("label", on_mouse)
    while True:
        t = state["t"]
        full = cv2.imdecode(np.frombuffer(packed[t], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(full, (round(W * view), round(H * view)), interpolation=cv2.INTER_AREA)

        def px(p):
            return int(round(p[0] * view)), int(round(p[1] * view))

        if t in points and points[t] is not None:
            cv2.drawMarker(img, px(points[t]), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        if state["show_track"] and track and t < len(track["xy"]):
            cv2.circle(img, px(track["xy"][t]), 6, (255, 0, 255), 1)
        poly = masks.get(t) if state["mode"] == "points" else (state["poly"] or masks.get(t))
        if poly:
            cv2.polylines(img, [np.array([px(p) for p in poly], np.int32)], t in masks, (0, 200, 0), 1)
        mx, my = state["mouse"]  # magnifier
        x0, y0 = int(mx) - MAG // (2 * ZOOM), int(my) - MAG // (2 * ZOOM)
        patch = cv2.copyMakeBorder(full, MAG, MAG, MAG, MAG, cv2.BORDER_CONSTANT)[
            y0 + MAG:y0 + MAG + MAG // ZOOM, x0 + MAG:x0 + MAG + MAG // ZOOM]
        mag = cv2.resize(patch, (MAG, MAG), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(mag, (MAG // 2, MAG // 2), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
        img[:MAG, :MAG] = mag
        label_frame = t % a.every == 0
        mark = "x" if points.get(t, 0) is None else ("*" if t in points else " ")
        flags = ["held" for r in held if r[0] <= t <= r[1]] + [e["kind"] for e in events if e["frame"] == t]
        lines = [f"{clip}  frame {t}/{T - 1}  t={t / fps:.2f}s  [{mark}]{' label' if label_frame else ''}  "
                 f"mode {state['mode']}  {' '.join(flags)}  sharpness {sharp[t]:.0f}",
                 f"click: {TARGETS[emitter] if state['mode'] == 'points' else 'hand + forearm outline'}"]
        for i, s in enumerate(lines):
            cv2.putText(img, s, (MAG + 10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, s, (MAG + 10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("label", img)
        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        c = chr(k & 0xFF) if k & 0xFF < 128 else ""
        if c in ("q", "\x1b"):
            save()
            break
        if c == "s":
            save()
        elif c in (" ", "n"):
            state["t"] = min(T - 1, (t // a.every + 1) * a.every)
        elif c == "b":
            state["t"] = max(0, (t - 1) // a.every * a.every)
        elif c == ".":
            state["t"] = min(T - 1, t + 1)
        elif c == ",":
            state["t"] = max(0, t - 1)
        elif c == "x":
            points[t] = None
        elif c == "h":
            ranges_toggle(held, open_held, t)
        elif c == "f":
            ranges_toggle(in_frame, open_frame, t)
        elif c in ("e", "o"):
            kind = "extension" if c == "e" else "open"
            hit = [e for e in events if e["frame"] == t and e["kind"] == kind]
            for e in hit:
                events.remove(e)
            if not hit:
                events.append({"frame": t, "kind": kind})
        elif c == "t":
            state["show_track"] = not state["show_track"]
        elif c == "m":
            state["mode"] = "masks" if state["mode"] == "points" else "points"
            state["poly"] = []
        elif c == "\r" and state["mode"] == "masks" and len(state["poly"]) >= 3:
            masks[t] = state["poly"]
            state["poly"] = []
        elif c == "c":
            masks.pop(t, None)
            state["poly"] = []
        elif c == "k":
            later = [f for f in suggest if f > t] or suggest
            state["t"] = later[0]
            state["poly"] = []
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
