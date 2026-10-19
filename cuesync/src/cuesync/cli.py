"""cuesync command line: shot, bend, hands, studio, retime, serve, and the lower-level analyze, guide, graph, render, compare, experiment."""

from __future__ import annotations

import argparse
import json
import os
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
    from .graph import element_graph, scene_graph, shot_graph

    g = element_graph(json.loads(Path(args.sample).read_text(encoding="utf-8")), video_name=args.video_name)
    if args.shot:
        g = shot_graph(g)
    elif args.scene:
        g = scene_graph(g)
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
    holds = None
    if args.hold:
        holds = [tuple(float(v) for v in h.split("-", 1)) for h in args.hold]
    if args.cue == "voice" and not args.voice:
        raise SystemExit("--voice is required for a voice cue (or use --cue clip / --cue hold)")
    if not args.scene and (args.strokes or args.describe or args.effect not in ("auto", "fireball", "flame_plume", "smoke_puff", "steam_jet")):
        raise SystemExit("--strokes, --describe and the scene effects (fire_breath, fire_burst, wick_flare, blow_out) "
                         "need --scene")
    strokes = json.loads(Path(args.strokes).read_text(encoding="utf-8")) if args.strokes else None
    sources = [[float(v) for v in p.split(",")] for p in args.sources.split(";")] if args.sources else None
    if sources and not args.scene:
        raise SystemExit("--sources needs --scene")
    spec = ShotSpec(plate=Path(args.plate), voice=Path(args.voice) if args.voice else None, effect=args.effect,
                    point=(args.x, args.y), seed=args.seed, offset_s=args.offset_s,
                    face=Path(args.face) if args.face else None, label=args.label or "", scale=args.scale,
                    mode="scene" if args.scene else "element", direction=args.direction, subject=args.subject,
                    cue=args.cue, holds=holds, strokes=strokes, describe=args.describe, sources=sources)
    last = [""]

    def progress(stage: str, frac: float, msg: str) -> None:
        if msg != last[0]:
            print(f"{int(frac * 100):3d}%  {msg}")
            last[0] = msg

    r = make_shot(spec, renderer, Path(args.output), progress)
    m, al = r.receipt["measured"], r.receipt["alignment"]
    if r.receipt.get("mode") == "scene":  # how far off the model was, and what moving it couldn't take out
        early = m.get("model_early_frames")
        what = "no clear onset: not re-timed" if early is None else (
            f"the model was {abs(early)} frames {'early' if early > 0 else 'late'}"
            + (f", moved hit by hit onto {len(al['warp'])} cues" if al.get("warp") else f", moved by {al['auto_shift']}")
            + (f", {m['residual_frames']} left" if m.get("residual_frames") else ""))
    else:
        what = f"hit error {m['hit_error_frames']} frames"
    print(f"done in {r.receipt['seconds_total']:.0f} s -> {r.final}  ({what})")
    return 0


def cmd_precompute(args) -> int:
    from .precompute import precompute

    if args.mode == "instant":
        from .library import DEFAULT_DIR, LibraryRenderer
        renderer = LibraryRenderer(DEFAULT_DIR)
    else:
        from .pipeline import ComfyRenderer
        renderer = ComfyRenderer(Path(args.workflow), timeout=args.timeout)
    index = precompute(renderer, args.mode, Path(args.samples), Path(args.out))
    print(f"{len(index)} {args.mode} sample shots in {args.out}/{args.mode}")
    return 0


def cmd_retime(args) -> int:
    from .pipeline import ShotSpec, retime_shot

    rc = json.loads((Path(args.shot) / "receipt.json").read_text(encoding="utf-8"))
    cue = rc.get("inputs", {}).get("cue", "voice")  # the shot knows what timed it
    if cue == "voice" and not args.voice:
        raise SystemExit("this shot was cued by a voice: pass --voice")
    holds = rc.get("inputs", {}).get("holds")
    spec = ShotSpec(plate=Path(args.plate), voice=Path(args.voice) if args.voice else None, effect=args.effect,
                    point=(args.x, args.y), offset_s=args.offset_s, cue=cue,
                    holds=[tuple(h) for h in holds] if holds else None,
                    mode="scene" if rc.get("mode") == "scene" else "element")
    r = retime_shot(Path(args.shot), spec, Path(args.output), hit_shift=args.shift, stop_frame=args.stop)
    print(f"re-timed in {r.receipt['seconds_total']:.0f} s -> {r.final}")
    return 0


def _progress():
    """A progress callback that says each new stage on the terminal (a long CPU step looked hung with nothing said)."""
    last = {"msg": None, "pct": -10}

    def say(stage, frac, msg) -> None:
        pct = int(round(100 * float(frac)))
        if msg != last["msg"] or pct >= last["pct"] + 10:
            print(f"  [{pct:3d}%] {msg}", flush=True)
            last.update(msg=msg, pct=pct)

    return say


def cmd_bend(args) -> int:
    """A shot from your hand (element bending). With --again: a finished bending shot (the sample hand's, made ahead
    of time) performed by this take, no GPU. Without: your clip's hand tracked and matted, the element rendered on the
    GPU (COMFY_BASE_URL) and carried on it; the helpers run from cuesync/.venv-track and cuesync/.venv-mask."""
    from .bending import bending_again, bending_job
    from .pipeline import ShotSpec

    if args.again:
        rc = json.loads((Path(args.again) / "receipt.json").read_text(encoding="utf-8"))
        spec = ShotSpec(plate=Path(args.plate), voice=Path(args.voice), effect=rc["effect"]["id"],
                        point=tuple(rc["inputs"]["point"]), offset_s=args.offset_s, mode="element",
                        extra={"tap_frame": rc["bending"].get("tap_frame")})
        r = bending_again(Path(args.again), spec, Path(args.output), progress=_progress())
    else:
        from .pipeline import ComfyRenderer

        spec = ShotSpec(plate=Path(args.plate), voice=Path(args.voice), effect=args.effect, point=(args.x, args.y),
                        offset_s=args.offset_s, mode="element", direction=args.direction, extra={"tap_frame": args.frame})
        renderer = None if args.instant else ComfyRenderer(Path(args.workflow), timeout=args.timeout)
        r = bending_job(spec, renderer, Path(args.output), progress=_progress())
    b = r.receipt["bending"]
    print(f"{r.receipt['effect']['id']}: {len(b['hits'])} hit(s)" + (f", charged {b['charge']['seconds']} s" if b.get("charge") else "")
          + (", no new render" if b.get("performed_again") else "") + f", {r.receipt['seconds_total']:.0f} s -> {r.final}")
    return 0


def cmd_hands(args) -> int:
    """Set up the hand tracker's environment (cuesync/.venv-track: MediaPipe + RTMPose, CPU, about 250 MB), or with
    --mask the hand matte's too (cuesync/.venv-mask: SAM 2.1, an NVIDIA GPU, about 3 GB). With uv when it is there
    (it fetches a Python 3.12), else this Python's venv. --check says what is installed."""
    import shutil
    import subprocess

    from .bending import hand_python, helpers_ready

    root = next((r for r in (Path.cwd(), *Path.cwd().parents) if (r / "cuesync" / "pyproject.toml").exists()), None)
    if args.check or not (args.install or args.mask):
        print(f"tracker: {'ready, ' + hand_python('track') if helpers_ready() else 'not installed (cuesync hands --install)'}")
        mask = hand_python("mask")
        print(f"matte: {mask if mask != sys.executable else 'not installed (cuesync hands --mask; the root fades without it)'}")
        return 0
    if root is None:
        raise SystemExit("run this from the cue-the-boom checkout (the folder with cuesync/ in it)")
    uv = shutil.which("uv")

    def venv(name: str, packages: list[str], index: str | None = None, more: list[list[str]] = ()) -> None:
        d = root / "cuesync" / name
        py = d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.exists():
            print(f"creating {d}")
            subprocess.run([uv, "venv", str(d), "--python", "3.12"] if uv else [sys.executable, "-m", "venv", str(d)], check=True)
        pip = [uv, "pip", "install", "--python", str(py)] if uv else [str(py), "-m", "pip", "install"]
        subprocess.run(pip + packages + (["--index-url", index] if index else []), check=True)
        for extra in more:
            subprocess.run(pip + extra, check=True)

    if args.install:
        from .bending import READY_MARK

        d = root / "cuesync" / ".venv-track"
        (d / READY_MARK).unlink(missing_ok=True)
        venv(".venv-track", ["mediapipe==1.1.1", "onnxruntime==1.31.0", "av==19.0.1"])
        py = d / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(py), "-c", "import mediapipe, onnxruntime, av"], check=True)  # it works: only then ready
        (d / READY_MARK).write_text("ok\n", encoding="utf-8")
        print("the hand tracker is ready: pick Your hand")
    if args.mask:
        venv(".venv-mask", ["torch==2.11.0", "torchvision==0.26.0"], index="https://download.pytorch.org/whl/cu128",
             more=[["hydra-core==1.3.7", "iopath==0.1.10", "huggingface-hub==2.2.0", "mediapipe==1.1.1", "av==19.0.1",
                    "tqdm", "setuptools", "wheel"]])
        os.environ["SAM2_BUILD_CUDA"] = "0"
        py = root / "cuesync" / ".venv-mask" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        pip = [uv, "pip", "install", "--python", str(py)] if uv else [str(py), "-m", "pip", "install"]
        subprocess.run(pip + ["--no-build-isolation", "--no-deps",
                              "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"],
                       check=True)
        print("the hand matte is ready")
    return 0


def cmd_studio(args) -> int:
    """The studio in one command: the render API (instant-only unless COMFY_BASE_URL points at a GPU) and the web app,
    both stopped with Ctrl+C; opens http://localhost:3000."""
    import shutil
    import subprocess
    import time
    import webbrowser

    from .server import STUDIO_STATIC

    import socket
    import urllib.request

    with socket.socket() as probe:  # a port another server holds: said now, not after opening a browser on it
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            raise SystemExit(f"port {args.port} is in use (another cuesync studio or serve?): "
                             f"cuesync studio --port {args.port + 1}") from None
    url = f"http://127.0.0.1:{args.port}"
    serve_cmd = [sys.executable, "-m", "cuesync.cli", "serve", "--port", str(args.port)] + (
        [] if os.environ.get("COMFY_BASE_URL") else ["--instant-only"])
    if not args.dev and (STUDIO_STATIC / "index.html").exists():  # the studio as shipped: the API serves it, no Node
        api = subprocess.Popen(serve_cmd)
        try:
            for _ in range(120):  # up and answering before it says where it is
                if api.poll() is not None:
                    raise SystemExit(f"the studio's server stopped (exit code {api.returncode}): see the lines above")
                try:
                    urllib.request.urlopen(f"{url}/api/health", timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.5)
            print(f"the studio: {url}  (Ctrl+C stops it)", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return api.wait()
        except KeyboardInterrupt:
            return 0
        finally:
            api.terminate()
    root = next((r for r in (Path.cwd(), *Path.cwd().parents) if (r / "web" / "package.json").exists()), None)
    npm = shutil.which("npm")
    if root is None or npm is None:
        raise SystemExit("run this from the cue-the-boom checkout, with Node 24 installed (npm on the PATH)")
    if not (root / "web" / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=root / "web", check=True)
    procs = [subprocess.Popen(serve_cmd, cwd=root), subprocess.Popen([npm, "run", "dev"], cwd=root / "web")]
    print(f"the studio: http://localhost:3000  (the render API: {url}; Ctrl+C stops both)")
    try:
        time.sleep(6)
        if not args.no_browser:
            webbrowser.open("http://localhost:3000")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
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
    from .server import SAMPLES, serve

    made_ahead = SAMPLES / "made_ahead"  # the scene shots that ship with the repository

    origins = [o.strip() for o in args.allow_origin.split(",")] if args.allow_origin else None
    serve(workflow=None if args.instant_only else Path(args.workflow), jobs_dir=Path(args.jobs), host=args.host,
          port=args.port, allow_origins=origins, timeout=args.timeout,
          library=Path(args.library) if args.library else None,
          precomputed=Path(args.precomputed) if args.precomputed else (made_ahead if made_ahead.is_dir() else None))
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
    gr.add_argument("--scene", action="store_true", help="the whole shot in scene mode (the effect rendered into the plate) as one workflow")
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
    sh.add_argument("--voice", help="your noise (wav/m4a/webm...); needed for --cue voice")
    sh.add_argument("--cue", choices=["voice", "clip", "hold"], default="voice",
                    help="what times it: your voice, a sound already in the plate (snap, click, slam), or --hold")
    sh.add_argument("--hold", action="append", metavar="PRESS-RELEASE",
                    help="--cue hold: press and release times in seconds, e.g. 1.0-2.5 (repeat for more)")
    sh.add_argument("--effect", required=True, help="auto (the sound picks it), or an effect id from effects/ (fireball, fire_breath, wick_flare, "
                                                    "blow_out, flame_plume, smoke_puff, steam_jet)")
    sh.add_argument("--x", type=float, default=0.5, help="emission point, 0..1 across the plate")
    sh.add_argument("--y", type=float, default=0.7, help="emission point, 0..1 down the plate")
    sh.add_argument("--offset-s", type=float, default=0.0, help="plate time of the recording's first sample")
    sh.add_argument("--seed", type=int, default=1)
    sh.add_argument("--scale", type=float, help="effect size, e.g. 0.75 or 1.3 (default: from the object's size)")
    sh.add_argument("--face", help="optional face clip for the inset")
    sh.add_argument("--label", help="text on the final video (default: the effect)")
    sh.add_argument("--scene", action="store_true",
                    help="render the effect into the plate itself (the scene reacts; needs a GPU backend)")
    sh.add_argument("--direction", type=float,
                    help="scene mode: where the effect goes, degrees clockwise from up (-90 left, 90 right)")
    sh.add_argument("--subject", help='scene mode: what it happens to, e.g. "the toy dinosaur on the table"')
    sh.add_argument("--strokes", help='scene mode: a JSON file of drawn strokes [{"points": [[x, y], ...], "width": w}]')
    sh.add_argument("--sources", help='scene mode, call and response: one source per hit, "x1,y1;x2,y2;x3,y3" '
                                      "(each hit of the rhythm fires at its own; pair it with --describe)")
    sh.add_argument("--describe", help='scene mode: say what happens, e.g. "thick smoke pours out of the lamp"')
    sh.add_argument("--instant", action="store_true",
                    help="no GPU: use the pre-rendered element library (timed and placed by your performance)")
    sh.add_argument("--library", help="library folder (default: samples/library)")
    sh.add_argument("--workflow", default="workflows/fluid_element.api.json")
    sh.add_argument("--timeout", type=float, default=600.0)
    sh.add_argument("-o", "--output", default="runs/shot")
    sh.set_defaults(func=cmd_shot)

    rt = sub.add_parser("retime", help="move the hit / the stop of a finished shot, no new render")
    rt.add_argument("--shot", required=True, help="the finished shot's folder")
    rt.add_argument("--plate", required=True)
    rt.add_argument("--voice", help="the voice the shot was cued with (not needed for a clip or hold cue)")
    rt.add_argument("--effect", required=True)
    rt.add_argument("--x", type=float, default=0.5)
    rt.add_argument("--y", type=float, default=0.7)
    rt.add_argument("--offset-s", type=float, default=0.0)
    rt.add_argument("--shift", type=int, default=0, help="frames to move every hit (+ later, - earlier)")
    rt.add_argument("--stop", type=int, help="plate frame where a flame or steam jet stops")
    rt.add_argument("-o", "--output", default="runs/retimed")
    rt.set_defaults(func=cmd_retime)

    bd = sub.add_parser("bend", help="fire, steam or air off your hand (element bending): your clip, your take")
    bd.add_argument("--plate", required=True, help="your clip, your hand in it")
    bd.add_argument("--voice", required=True, help="your take: the noise that cues it")
    bd.add_argument("--again", help="a finished bending shot's folder: perform its element again, no GPU "
                                    "(the sample hand's: samples/made_ahead/bending/...)")
    bd.add_argument("--effect", default="auto", help="auto (your sound and your hand pick it), fire_breath, fire_burst, "
                                                     "flame_plume, steam_jet or smoke_puff")
    bd.add_argument("--x", type=float, default=0.5, help="where on the hand it comes from, as a fraction of the width")
    bd.add_argument("--y", type=float, default=0.5)
    bd.add_argument("--frame", type=int, default=0, help="the frame (24 fps) --x --y are on")
    bd.add_argument("--direction", type=float, help="aim, degrees clockwise from up (default: the hand's own)")
    bd.add_argument("--offset-s", type=float, default=0.0, help="where your take starts in the clip, seconds")
    bd.add_argument("--instant", action="store_true", help="no GPU: an element rendered ahead of time "
                                                          "(samples/library/bending), carried on your hand")
    bd.add_argument("--workflow", default="workflows/fluid_element.api.json")
    bd.add_argument("--timeout", type=float, default=900.0)
    bd.add_argument("-o", "--output", default="runs/bend")
    bd.set_defaults(func=cmd_bend)

    hd = sub.add_parser("hands", help="set up (or check) the hand tracker and matte environments for element bending")
    hd.add_argument("--install", action="store_true", help="the tracker: MediaPipe + RTMPose, CPU, about 250 MB")
    hd.add_argument("--mask", action="store_true", help="the matte too: SAM 2.1, an NVIDIA GPU, about 3 GB")
    hd.add_argument("--check", action="store_true", help="say what is installed")
    hd.set_defaults(func=cmd_hands)

    st = sub.add_parser("studio", help="the studio in one command (the API serves it; --dev: the Next.js dev server)")
    st.add_argument("--no-browser", action="store_true")
    st.add_argument("--port", type=int, default=8765, help="where the studio is served (default 8765)")
    st.add_argument("--dev", action="store_true", help="run web/ with Next.js (Node 24) instead of the built studio")
    st.set_defaults(func=cmd_studio)

    sv = sub.add_parser("serve", help="HTTP API for the web app (renders through COMFY_BASE_URL)")
    sv.add_argument("--workflow", default="workflows/fluid_element.api.json")
    sv.add_argument("--jobs", default="runs/jobs", help="where job inputs and results are kept")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--allow-origin", help="comma list of browser origins allowed to call the API directly")
    sv.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for one render")
    sv.add_argument("--library", help="instant-mode element library (default: samples/library)")
    sv.add_argument("--instant-only", action="store_true", help="no GPU backend: serve instant mode only")
    sv.add_argument("--precomputed", help="sample shots made ahead of time (cuesync precompute) for the gallery; "
                                          "default: the ones in samples/made_ahead")

    pc = sub.add_parser("precompute", help="make every sample shot ahead of time (gallery + instant first clicks)")
    pc.add_argument("--mode", choices=["instant", "render", "scene"], default="instant")
    pc.add_argument("--out", default="runs/precomputed")
    pc.add_argument("--samples", default="samples")
    pc.add_argument("--workflow", default="workflows/fluid_element.api.json")
    pc.add_argument("--timeout", type=float, default=600.0)
    pc.set_defaults(func=cmd_precompute)
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
