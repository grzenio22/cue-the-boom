"""Reproducible CPU element renders on the author's prepared bright/dim iPhone takes.

Uses the production shot pipeline, sample voice cues and recorded hand tracking.
Outputs are review evidence, never published. Before/after runs use identical inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import numpy as np

from cuesync.bending import make_bending_shot, sample_noise_offset
from cuesync.pipeline import ShotSpec
from cuesync.media import probe_video

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "bright": ("hand_open.mp4", "hand_open__fff__flame_plume"),
    "dark": ("hand_palm.mp4", "hand_palm__boom__fire_burst"),
    "point": ("hand_point.mp4", "hand_point__fwoooom__fire_breath"),
}
VOICES = {
    "air_ball": "hoooo.wav", "air_blast": "hoo-hoo-HOO.wav",
    "water_orb": "shhhhh-sploosh.wav", "water_whip": "shhhhh-sploosh.wav",
    "water_surge": "shhhhh-sploosh.wav",
    "water_ribbon": "shhhhh-sploosh.wav", "water_shield": "shhhhh-sploosh.wav", "water_jet": "shhhhh-sploosh.wav",
    "lightning_ball": "zzzzz-tsst.wav", "lightning_arc": "zzzzz-tsst.wav",
    "earth_rock": "hoooo.wav", "earth_shatter": "pf-pf-boom-author.wav",
    "earth_volley": "pf-pf-boom-author.wav", "earth_lance": "pf-pf-boom-author.wav",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inputs(out, case, width):
    plate, prepared = CASES[case]
    meta = json.loads((ROOT / "samples/samples.json").read_text(encoding="utf-8"))["plates"][plate]
    src = ROOT / "samples/made_ahead/bending" / prepared
    track = json.loads((src / "hand_track.json").read_text())
    ratio = width / track["W"]
    height = round(track["H"] * ratio / 2) * 2
    target = out / "inputs" / case
    target.mkdir(parents=True, exist_ok=True)
    for key in ("xy", "scale", "speed"):
        if key in track:
            track[key] = (np.asarray(track[key]) * ratio).tolist()
    track["landmarks"] = [None if lm is None else (np.asarray(lm) * ratio).tolist()
                          for lm in track.get("landmarks", [])]
    track["W"], track["H"] = width, height
    (target / "hand_track.json").write_text(json.dumps(track), encoding="utf-8")
    with np.load(src / "hand_mask.npz") as data:
        mask = np.stack([cv2.resize(m, (width, height), interpolation=cv2.INTER_AREA) for m in data["alpha"]])
    np.savez_compressed(target / "hand_mask.npz", alpha=mask, fps=24, version=1)
    return ROOT / "samples/plates" / plate, meta, target, (width, height), src


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--width", type=int, default=540)
    p.add_argument("--plates", nargs="+", choices=CASES, default=["bright", "dark"])
    p.add_argument("--effects", nargs="+", choices=VOICES, default=list(VOICES))
    p.add_argument("--baseline-ref", help="Compare older air/water/lightning code on identical current shot inputs")
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit("Use a fresh output directory; review evidence is immutable")
    args.out.mkdir(parents=True)
    renderer_sources = {}
    if args.baseline_ref:
        for name in {e.split("_")[0] for e in args.effects}:
            filename = f"cuesync/src/cuesync/{name}.py"
            source = subprocess.check_output(["git", "show", f"{args.baseline_ref}:{filename}"], cwd=ROOT)
            module = types.ModuleType(f"cuesync.{name}")
            module.__file__ = str(ROOT / filename)
            sys.modules[module.__name__] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            if name=="earth" and not hasattr(module,"choreography"):
                module.choreography=lambda *a:[]
            renderer_sources[filename] = hashlib.sha256(source).hexdigest()
    manifest = {"width": args.width, "seed": 17, "cloud_spend_usd": 0,
                "voice_note": "Bundled sample recordings; synthetic except the labelled author's pf-pf-BOOM.",
                "source_sha256": {f.relative_to(ROOT).as_posix(): sha(f) for f in [* (ROOT / "cuesync/src/cuesync").glob("*.py"), * (ROOT / "cuesync/src/cuesync/assets/stone").glob("*")]},
                "cases": []}
    manifest["renderer_baseline_ref"] = args.baseline_ref
    manifest["source_sha256"].update(renderer_sources)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for case in args.plates:
        plate, meta, inp, size, src = inputs(args.out, case, args.width)
        for effect in args.effects:
            voice = ROOT / "samples/voices" / VOICES[effect]
            offset = sample_noise_offset(voice, probe_video(plate).duration_s, int(meta["hand"]["strike"]),
                                         fit_held=effect.startswith("water_") or effect in ("air_ball", "lightning_ball", "earth_rock"))
            spec = ShotSpec(plate=plate, voice=voice, effect=effect, point=tuple(meta["point"]), seed=17,
                            offset_s=offset, plate_size=size, label=effect.replace("_", " "))
            job = args.out / f"{case}-{effect}"
            start = time.perf_counter()
            print(f"START {job.name}", flush=True)
            def progress(stage, share, message):
                if stage in ("done", "finishing"):
                    print(f"{job.name}: {message}", flush=True)
            result = make_bending_shot(spec, None, job, inp / "hand_track.json", inp / "hand_mask.npz",
                                       tap_frame=meta["hand"]["frame"], progress=progress)
            seconds = time.perf_counter() - start
            # A renderer edited during a batch would make its evidence ambiguous.
            for file, digest in manifest["source_sha256"].items():
                if file not in renderer_sources and sha(ROOT / file) != digest:
                    raise RuntimeError(f"Source changed during render: {file}; rerun in a fresh directory")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(result.final), "-f", "null", "-"], check=True)
            entry = {"case": case, "effect": effect, "seconds": round(seconds, 3), "offset_s": offset,
                     "plate": str(plate), "plate_sha256": sha(plate), "voice": str(voice), "voice_sha256": sha(voice),
                     "track_source_sha256": sha(src / "hand_track.json"), "mask_source_sha256": sha(src / "hand_mask.npz"),
                     "final": str(result.final.resolve()), "sha256": sha(result.final), "decode": "passed"}
            manifest["cases"].append(entry)
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"DONE {job.name} {seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
