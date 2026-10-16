"""cuesync command line: analyze, guide, graph, render, compare, experiment, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _point(s: str) -> tuple[float, float]:
    x, y = (float(v) for v in s.split(","))
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError("point must be X,Y in [0,1]")
    return x, y


def _size(s: str) -> tuple[int, int]:
    w, h = (int(v) for v in s.lower().split("x"))
    if w % 64 or h % 64:
        raise argparse.ArgumentTypeError("width and height must be multiples of 64 (e.g. 576x1024, 512x960)")
    return w, h


def _ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v.strip()]


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def _offset(args) -> float:
    if getattr(args, "timing", None):
        from .experiment import offset_from_timing

        return offset_from_timing(json.loads(Path(args.timing).read_text()))
    return float(args.offset or 0.0)


def cmd_analyze(args) -> int:
    from .audio import analyze
    from .media import read_audio

    cue = analyze(read_audio(args.noise), 48000, offset_s=_offset(args))
    Path(args.output).write_text(json.dumps(cue.to_json(), indent=1))
    print(f"hero onset {cue.hero_onset_s:.3f}s (frame {cue.frame_of(cue.hero_onset_s)}), "
          f"{len(cue.onsets_s)} onsets, peak frame {cue.peak_frame}, max level {cue.envelope.max():.2f} -> {args.output}")
    return 0


def cmd_guide(args) -> int:
    from .audio import CueSheet
    from .effects import load_effect
    from .guide import paint_guide, write_guide

    cue = CueSheet.from_json(json.loads(Path(args.cue).read_text()))
    plan = paint_guide(cue, load_effect(args.effect), args.point, size=args.size, length=args.length,
                       tail=args.tail, canvas=args.canvas)
    write_guide(plan, args.output)
    Path(args.output).with_suffix(".json").write_text(json.dumps(plan.to_json(), indent=1))
    print(f"guide {args.output}: {plan.frames.shape[0]} frames, anchor plate frame {plan.anchor_frame}, "
          f"voice ends at element frame {plan.voice_end}, keyframes {plan.keyframes}")
    return 0


def cmd_graph(args) -> int:
    from .graph import element_graph, shot_graph

    g = element_graph(json.loads(Path(args.sample).read_text(encoding="utf-8")), video_name=args.video_name)
    if args.shot:
        g = shot_graph(g)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
    print(f"element graph {args.output}: {len(g)} nodes")
    return 0


def cmd_render(args) -> int:
    from .runner import render_element

    res = render_element(args.guide, prompt=args.prompt, seed=args.seed, workflow_path=args.workflow,
                         out_path=args.output, timeout=args.timeout)
    print(f"rendered {res.path} in {res.seconds:.1f}s")
    return 0


def cmd_library(args) -> int:
    from .library import render_library
    from .pipeline import ComfyRenderer

    seeds = tuple(int(x) for x in args.seeds.split(","))
    m = render_library(ComfyRenderer(Path(args.workflow), timeout=args.timeout), Path(args.out), seeds=seeds)
    print(f"library: {len(m['entries'])} elements -> {args.out}")
    return 0


def cmd_shot(args) -> int:
    from .pipeline import ComfyRenderer, ShotSpec, make_shot

    if args.instant:
        from .library import DEFAULT_DIR, LibraryRenderer
        renderer = LibraryRenderer(Path(args.library) if args.library else DEFAULT_DIR)
    else:
        renderer = ComfyRenderer(Path(args.workflow), timeout=args.timeout)
    spec = ShotSpec(plate=Path(args.plate), voice=Path(args.voice), effect=args.effect, point=(args.x, args.y),
                    seed=args.seed, offset_s=args.offset_s, face=Path(args.face) if args.face else None,
                    label=args.label or "")
    last = [""]

    def progress(stage: str, frac: float, msg: str) -> None:
        if msg != last[0]:
            print(f"{int(frac * 100):3d}%  {msg}")
            last[0] = msg

    r = make_shot(spec, renderer, Path(args.output), progress)
    m = r.receipt["measured"]
    print(f"done in {r.receipt['seconds_total']:.0f} s -> {r.final}  (hit error {m['hit_error_frames']} frames)")
    return 0


def cmd_compare(args) -> int:
    from .audio import CueSheet
    from .compare import side_by_side
    from .media import read_video, write_video

    cue_a = CueSheet.from_json(json.loads(Path(args.cue_a).read_text()))
    cue_b = CueSheet.from_json(json.loads(Path(args.cue_b).read_text()))

    def anchor(explicit, guide_json, cue):  # explicit > the guide's own anchor rule > hero onset
        if explicit is not None:
            return explicit
        if guide_json:
            return int(json.loads(Path(guide_json).read_text())["anchor_frame"])
        return cue.frame_of(cue.hero_onset_s)

    anchor_a = anchor(args.anchor_a, args.guide_a, cue_a)
    anchor_b = anchor(args.anchor_b, args.guide_b, cue_b)
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else [Path(args.a).stem, Path(args.b).stem]
    size = None
    if args.half:
        probe = read_video(args.a).frames
        size = (probe.shape[2] // 2, probe.shape[1] // 2)
    frames = side_by_side(read_video(args.a, size=size).frames, read_video(args.b, size=size).frames,
                          anchor_a, anchor_b, np.asarray(cue_a.envelope), np.asarray(cue_b.envelope), labels[:2])
    write_video(args.output, frames, fps=24)
    print(f"compare -> {args.output} ({frames.shape[0]} frames)")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    origins = [o.strip() for o in args.allow_origin.split(",")] if args.allow_origin else None
    serve(workflow=None if args.instant_only else Path(args.workflow), jobs_dir=Path(args.jobs), host=args.host,
          port=args.port, allow_origins=origins, timeout=args.timeout,
          library=Path(args.library) if args.library else None)
    return 0


def cmd_experiment(args) -> int:
    from .experiment import run

    if args.timings and args.offsets:
        raise SystemExit("use --offsets or --timings, not both")
    m = run(plates=args.plates, noises=args.noises, offsets=_floats(args.offsets) if args.offsets else None,
            timings=args.timings, effect=args.effect, point=args.point, seeds=_ints(args.seeds), length=args.length,
            tail=args.tail, canvas=args.canvas, allow_stale=args.allow_stale, elements_dir=args.elements_dir,
            workflow=args.workflow, out=args.out,
            labels=[s.strip() for s in args.labels.split(",")] if args.labels else None)
    for r in m["runs"]:
        lat = r["stop_latency_frames"]
        stops = "n/a" if lat is None else (f"{lat} f after the voice" if r["stopped_within_plate"] else f">= {lat} f (still on at plate end)")
        print(f"{r['plate']:>16} n{r['noise_index']} seed {r['seed']}: onset err {r['measured_onset_error_frames']} f, "
              f"hit err {r['hit_error_frames']} f, stops {stops}, length {r['element_length']}, render {r['render_s']}s"
              + ("  [STALE element]" if r["stale_element"] else ""))
    print(f"metrics -> {Path(args.out) / 'metrics.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cuesync", description="Voice-shaped VFX: cue sheets, fluid guides, renders, composites.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="noise recording -> cue sheet JSON")
    a.add_argument("noise")
    g_off = a.add_mutually_exclusive_group()
    g_off.add_argument("--offset", type=float, help="plate time (s) of the recording's first sample")
    g_off.add_argument("--timing", help="Dub Mode timing.json (offset = noise_offset_s - input_latency_s)")
    a.add_argument("-o", "--output", required=True)
    a.set_defaults(func=cmd_analyze)

    g = sub.add_parser("guide", help="cue sheet -> ainvfx-fluid guide video")
    g.add_argument("--cue", required=True)
    g.add_argument("--effect", required=True)
    g.add_argument("--point", type=_point, required=True, help="emission point X,Y in [0,1]")
    g.add_argument("--length", type=int, default=None, help="element frames, 8k+1 (default: fit the performance)")
    g.add_argument("--tail", choices=["wisp", "fade"], default="wisp", help="how the effect is painted to end")
    g.add_argument("--canvas", action="store_true", help="paint keyframes on a dark canvas")
    g.add_argument("--size", type=_size, default=(576, 1024), help="guide WxH, multiples of 64 (default 576x1024)")
    g.add_argument("-o", "--output", required=True)
    g.set_defaults(func=cmd_guide)

    gr = sub.add_parser("graph", help="ainvfx-fluid sample graph (API format) -> element graph with a Load Video input")
    gr.add_argument("sample", help="the sample workflow exported in API format")
    gr.add_argument("--video-name", default="guide.mp4", help="placeholder file for the Load Video node")
    gr.add_argument("--shot", action="store_true", help="the whole shot as one workflow (needs the CueSync node pack)")
    gr.add_argument("-o", "--output", required=True)
    gr.set_defaults(func=cmd_graph)

    r = sub.add_parser("render", help="guide video -> element via Comfy SDK (COMFY_BASE_URL)")
    r.add_argument("--guide", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--workflow", required=True, help="exported API graph (workflows/fluid_element.api.json)")
    r.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for the job")
    r.add_argument("-o", "--output", required=True)
    r.set_defaults(func=cmd_render)

    c = sub.add_parser("compare", help="two composites side by side, synced on the hit, with envelope ribbons")
    c.add_argument("--a", required=True)
    c.add_argument("--cue-a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--cue-b", required=True)
    c.add_argument("--anchor-a", type=int)
    c.add_argument("--anchor-b", type=int)
    c.add_argument("--guide-a", help="guide.json written by 'cuesync guide' (uses its anchor_frame)")
    c.add_argument("--guide-b", help="guide.json written by 'cuesync guide' (uses its anchor_frame)")
    c.add_argument("--labels", help='e.g. "pf.,fwoooOOOM"')
    c.add_argument("--half", action="store_true", help="half-resolution sides")
    c.add_argument("-o", "--output", required=True)
    c.set_defaults(func=cmd_compare)

    lb = sub.add_parser("library", help="render the instant-mode element library (needs a GPU backend)")
    lb.add_argument("--workflow", default="workflows/fluid_element.api.json")
    lb.add_argument("--out", default="samples/library")
    lb.add_argument("--seeds", default="1,2")
    lb.add_argument("--timeout", type=float, default=600.0)
    lb.set_defaults(func=cmd_library)

    sh = sub.add_parser("shot", help="plate + performance -> the finished shot (final, breakdown, clean, receipt)")
    sh.add_argument("--plate", required=True)
    sh.add_argument("--voice", required=True, help="your noise (wav/m4a/webm...)")
    sh.add_argument("--effect", required=True, choices=["fireball", "smoke_puff", "flame_plume", "steam_jet"])
    sh.add_argument("--x", type=float, default=0.5, help="emission point, 0..1 across the plate")
    sh.add_argument("--y", type=float, default=0.7, help="emission point, 0..1 down the plate")
    sh.add_argument("--offset-s", type=float, default=0.0, help="plate time of the recording's first sample")
    sh.add_argument("--seed", type=int, default=1)
    sh.add_argument("--face", help="optional face clip for the inset")
    sh.add_argument("--label", help="text on the final video (default: the effect)")
    sh.add_argument("--instant", action="store_true",
                    help="no GPU: use the pre-rendered element library (timed and placed by your performance)")
    sh.add_argument("--library", help="library folder (default: samples/library)")
    sh.add_argument("--workflow", default="workflows/fluid_element.api.json")
    sh.add_argument("--timeout", type=float, default=600.0)
    sh.add_argument("-o", "--output", default="runs/shot")
    sh.set_defaults(func=cmd_shot)

    sv = sub.add_parser("serve", help="HTTP API for the web app (renders through COMFY_BASE_URL)")
    sv.add_argument("--workflow", default="workflows/fluid_element.api.json")
    sv.add_argument("--jobs", default="runs/jobs", help="where job inputs and results are kept")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--allow-origin", help="comma list of browser origins allowed to call the API directly")
    sv.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for one render")
    sv.add_argument("--library", help="instant-mode element library (default: samples/library)")
    sv.add_argument("--instant-only", action="store_true", help="no GPU backend: serve instant mode only")
    sv.set_defaults(func=cmd_serve)

    e = sub.add_parser("experiment", help="performances x seeds -> composites on every plate + metrics + compares")
    e.add_argument("--plates", nargs="+", required=True)
    e.add_argument("--noises", nargs="+", required=True)
    e.add_argument("--offsets", help="comma list, one per noise")
    e.add_argument("--timings", nargs="+", help="Dub Mode timing.json per noise")
    e.add_argument("--effect", required=True)
    e.add_argument("--point", type=_point, required=True)
    e.add_argument("--seeds", default="1,2,3")
    e.add_argument("--length", type=int, default=None, help="element frames, 8k+1 (default: fit each performance)")
    e.add_argument("--tail", choices=["wisp", "fade"], default="wisp")
    e.add_argument("--canvas", action="store_true")
    e.add_argument("--allow-stale", action="store_true", help="composite elements rendered from older guides (A/B only)")
    e.add_argument("--elements-dir", help="reuse element_n{i}_seed{s}.mp4 instead of rendering")
    e.add_argument("--workflow", help="exported API graph (needed unless --elements-dir)")
    e.add_argument("--labels", help="comma list of noise labels for compares")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_experiment)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
