"""Rebuild v11's footage with the current checkout, without generating another model render.

The three palm performances share exactly one shipped LTX-2.5 element. The two new
rhythms are cut from the author's recorded pf and BOOM, as runs/takes/rhythms.py does.
Explosion shots use the current compositor and the library's LTX fireball; lightning
and water are procedural CPU effects. The scenes are replayed/transferred again from
their cached model renders by current code. The studio recording is existing footage;
its resulting shot is composited again. clips/manifest.json records source hashes.

    python tools/demo/make_v11_clips.py
    python tools/demo/make_v11_clips.py --only explosion_palm proof_boom --force

All inputs are read-only. Output and temporary roots are configurable; each render
runs in a separate process so decoded 1080p arrays are released. Optional
``--gallery-inputs`` uses an immutable copy of gallery assets; ``--workers 2``
runs two independent jobs while the main process alone records their receipts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave


JOBS = (
    "explosion_palm", "proof_boom", "proof_pf_pf_boom", "proof_pf_x4",
    "how_light_off", "flame_walk", "flamethrower", "lightning_point",
    "lightning_dark", "lightning_ball", "water_orb", "explosion_punch",
    "scene_lanterns", "scene_dino", "studio",
)
SCRIPT = Path(__file__).resolve()
CHECKOUT = SCRIPT.parents[2]
SOURCE = CHECKOUT / "cuesync" / "src"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(cmd: list[str]) -> None:
    subprocess.run([str(v) for v in cmd], check=True)


def outputs(job: str, clips: Path) -> list[Path]:
    if job == "studio":
        return [clips / "studio.mp4", clips / "studio_say_own.wav", clips / "studio_result.mp4",
                clips / "studio_result_current" / "receipt.json"]
    result = [clips / job / "composite.mp4", clips / job / "receipt.json"]
    if job == "proof_boom":
        result += [clips / "how_raw.mp4", clips / "how_carried.mp4", clips / "proof_boom.wav"]
    if job in ("proof_pf_pf_boom", "proof_pf_x4"):
        result.append(clips / f"{job}.wav")
    return result


def recipe_input_paths(job: str, root: Path, gallery: Path | None = None) -> list[Path]:
    takes = root / "runs" / "takes"
    samples = root / "samples"
    ahead = gallery or samples / "made_ahead" / "bending"
    if job == "studio":
        job_dir = root / "runs/jobs/fd58657de8da"
        return [root / "runs/demo_v9/studio_own3/studio.mp4", root / "runs/demo_v9/studio_say_own.wav",
                job_dir / "in/plate24.mp4", job_dir / "in/voice.wav", job_dir / "hand_track.json",
                job_dir / "hand_mask.npz", samples / "library/bending/fire_burst/element.mp4",
                samples / "library/bending/fire_burst/receipt.json"]
    if job == "scene_lanterns":
        paths = [root / "runs/real/plates/s2_lanterns_held_frame.mp4",
                 root / "runs/round10/human/pf-pf-boom__voices.wav"]
        for name in ("held2_small", "held2_ghost", "held2_big"):
            paths.extend(root / "runs/round10/shots" / name / f for f in
                         ("receipt.json", "guide.json", "guide.mp4", "render.mp4"))
        return paths
    if job == "scene_dino":
        src = root / "runs/real/final/dino_books__fwoooom"
        return [root / "runs/real/plates/s4_pexels_8014678_a_toy_dinosaur_on_a_stack_of_book.mp4",
                samples / "voices/fwoooom.wav", *[src / n for n in
                ("receipt.json", "guide.json", "guide.mp4", "guide_preview.mp4", "render.mp4", "start_frame.png",
                 "matte.png", "cue.json")]]
    if job.startswith("proof_") or job == "how_light_off":
        return [takes / "plates/IMG_6658.mp4", takes / "voice_6644_pf.wav",
                *[ahead / "hand_palm__boom__fire_burst" / n for n in
                  ("element.mp4", "receipt.json", "hand_track.json", "hand_mask.npz")]]
    if job in ("explosion_palm", "explosion_punch"):
        take = "IMG_6666" if job == "explosion_palm" else "IMG_6663"
        return [takes / "plates" / f"{take}.mp4", takes / "hand" / take / "hand_track.json",
                takes / "hand" / take / "hand_mask.npz", samples / "library/bending/explosion/element.mp4"]
    if job in ("flame_walk", "flamethrower"):
        take, element = ("IMG_6668", "hand_walk__fffff__flame_plume") if job == "flame_walk" else (
            "IMG_6653", "hand_point__fwoooom__fire_breath")
        return [takes / "plates" / f"{take}.mp4", *[ahead / element / n for n in
                ("element.mp4", "receipt.json", "hand_track.json", "hand_mask.npz")]]
    plate, voice, element = {
        "lightning_point": ("hand_point.mp4", "zzzzz-tsst.wav", "hand_point__fwoooom__fire_breath"),
        "lightning_dark": ("hand_palm.mp4", "zzzzz-tsst.wav", "hand_palm__boom__fire_burst"),
        "lightning_ball": ("hand_palm.mp4", "zzzzz-tsst.wav", "hand_palm__boom__fire_burst"),
        "water_orb": ("hand_open.mp4", "shhhhh-sploosh.wav", "hand_open__fff__flame_plume"),
    }[job]
    return [samples / "plates" / plate, samples / "voices" / voice, samples / "samples.json",
            *[ahead / element / n for n in ("receipt.json", "hand_track.json", "hand_mask.npz")]]


def source_lineage(element: Path, root: Path, gallery: Path | None = None) -> dict | None:
    """Bind playback bytes to the installed original generation and CPU recipe."""
    provenance = element.with_name("source-provenance.json")
    if not provenance.is_file():
        receipt = element.with_name("receipt.json")
        if receipt.is_file() and json.loads(receipt.read_text(encoding="utf-8")).get("source_provenance"):
            raise FileNotFoundError(f"Playback source declares a missing provenance sidecar: {provenance}")
        return None
    data = json.loads(provenance.read_text(encoding="utf-8"))
    adapter = element.with_name("source-adapter.json")
    if sha(element) != data["element_sha256"] or sha(adapter) != data["playback_adapter_sha256"]:
        raise ValueError(f"Playback element or adapter differs from its source provenance: {element}")
    assets = gallery.parent / "source-assets" if gallery and element.is_relative_to(gallery) else root
    source_files = {}
    for relative, expected in data["source_files"].items():
        path = (assets / relative).resolve()
        if not path.is_relative_to(assets.resolve()) or not path.is_file() or sha(path) != expected:
            raise ValueError(f"Original-generation source asset is missing or changed: {path}")
        source_files[relative] = dict(path=str(path), sha256=expected)
    original_sha, derivative = data["element_sha256"], None
    for item in source_files.values():
        path = Path(item["path"])
        if path.name == "derivative-recipe.json":
            recipe = json.loads(path.read_text(encoding="utf-8"))
            if recipe.get("derivative_sha256") == data["element_sha256"]:
                if derivative is not None:
                    raise ValueError(f"Ambiguous derivative recipes for {element}")
                derivative = str(path)
                original_sha = recipe["original_element_sha256"]
    originals = [item["path"] for item in source_files.values()
                 if item["sha256"] == original_sha and Path(item["path"]).name == "element.mp4"]
    if len(originals) != 1:
        raise ValueError(f"Exactly one original generation must establish playback ancestry: {element}")
    return dict(provenance_path=str(provenance.resolve()), source_adapter_path=str(adapter.resolve()),
                source_files=source_files, original_generation_element=originals[0],
                original_generation_sha256=original_sha, derivative_recipe=derivative)


def input_paths(job: str, root: Path, gallery: Path | None = None) -> list[Path]:
    inputs = recipe_input_paths(job, root, gallery)
    for element in [p for p in inputs if p.name == "element.mp4"]:
        lineage = source_lineage(element, root, gallery)
        if lineage:
            inputs.extend(Path(lineage[key]) for key in ("provenance_path", "source_adapter_path"))
            inputs.extend(Path(item["path"]) for item in lineage["source_files"].values())
    return list(dict.fromkeys(inputs))


def read_mono(path: Path, sr: int = 48000):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr),
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def wav(path: Path, samples, sr: int = 48000) -> None:
    import numpy as np
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def proof_voice(job: str, root: Path, clips: Path) -> Path:
    """The same cuts, room bed, positions and gains as the original rhythms.py."""
    import numpy as np
    take = read_mono(root / "runs/takes/plates/IMG_6658.mp4")
    path = clips / f"{job}.wav"
    if job == "proof_boom":
        wav(path, take)
        return path
    sr = 48000
    pf = read_mono(root / "runs/takes/voice_6644_pf.wav")[int(.98 * sr):int(1.16 * sr)]
    boom = take[int(2.60 * sr):int(3.20 * sr)]
    room = take[:int(.5 * sr)]
    track = np.tile(room, int(np.ceil(take.size / room.size)))[:take.size].copy()
    parts = [(pf, 1.20, -7), (pf, 1.56, -7), (boom, 2.60, 0)] if job == "proof_pf_pf_boom" else [
        (pf, 1.04, -12), (pf, 1.56, -9), (pf, 2.04, -6), (pf, 2.50, -3)]
    for clip, at, db in parts:
        i = int(at * sr)
        track[i:i + clip.size] += clip[:max(0, track.size - i)] * 10 ** (db / 20)
    wav(path, track)
    return path


def render_worker(job: str, root: Path, clips: Path, temp: Path, gallery: Path | None = None) -> None:
    # Absolute paths keep all input directories read-only. The cwd only helps the
    # current library resolver locate the already rendered explosion element.
    os.chdir(root)
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    import cuesync.bending as b
    from cuesync.media import probe_video, write_video
    from cuesync.pipeline import ShotSpec

    if job == "scene_lanterns":
        from cuesync.replay import replay_scene_shot
        (clips / job / "source.json").unlink(missing_ok=True)
        src = root / "runs/round10/shots"
        spec = ShotSpec(plate=root / "runs/real/plates/s2_lanterns_held_frame.mp4",
                        voice=root / "runs/round10/human/pf-pf-boom__voices.wav", effect="fire_burst",
                        point=(.42, .72), mode="scene")
        replay_scene_shot([src / n for n in ("held2_small", "held2_ghost", "held2_big")], spec, clips / job)
        return
    if job == "scene_dino":
        from cuesync.scene_shot import retime_scene_shot
        (clips / job / "source.json").unlink(missing_ok=True)
        spec = ShotSpec(plate=root / "runs/real/plates/s4_pexels_8014678_a_toy_dinosaur_on_a_stack_of_book.mp4",
                        voice=root / "samples/voices/fwoooom.wav", effect="fire_breath", point=(.36, .42), seed=1,
                        offset_s=0.0, mode="scene", direction=-80, subject="the toy dinosaur on the stack of books")
        retime_scene_shot(root / "runs/real/final/dino_books__fwoooom", spec, clips / job, hit_shift=0)
        return
    if job == "studio":
        shutil.copyfile(root / "runs/demo_v9/studio_own3/studio.mp4", clips / "studio.mp4")
        shutil.copyfile(root / "runs/demo_v9/studio_say_own.wav", clips / "studio_say_own.wav")
        src = root / "runs/jobs/fd58657de8da"
        spec = ShotSpec(plate=src / "in/plate24.mp4", voice=src / "in/voice.wav", effect="fire_burst",
                        point=(.509, .654), seed=1, offset_s=-.0636, mode="element", cue="voice")
        result = b.make_bending_shot(spec, None, clips / "studio_result_current", src / "hand_track.json",
                                     src / "hand_mask.npz", element_from=root / "samples/library/bending/fire_burst",
                                     tap_frame=54, aim=0.0, instant=True)
        shutil.copyfile(result.composite, clips / "studio_result.mp4")
        return

    ahead = gallery or root / "samples/made_ahead/bending"
    takes = root / "runs/takes"
    dest = clips / job
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Current renderer: {Path(b.__file__).resolve()}", flush=True)

    def progress(stage, frac, message):
        print(f"{job} {stage} {frac:.0%}: {message}", flush=True)

    if job in ("explosion_palm", "explosion_punch"):
        take, x, y, tap = ("IMG_6666", 540, 1000, 35) if job == "explosion_palm" else (
            "IMG_6663", 621, 807, 68)
        spec = ShotSpec(plate=takes / "plates" / f"{take}.mp4", voice=None, cue="clip", effect="explosion",
                        point=(x / 1080, y / 1920), mode="element", seed=1)
        b.make_bending_shot(spec, None, dest, takes / "hand" / take / "hand_track.json",
                            takes / "hand" / take / "hand_mask.npz", progress, tap_frame=tap)
        return

    if job in ("lightning_point", "lightning_dark", "lightning_ball", "water_orb"):
        plate, voice, effect, first = {
            "lightning_point": ("hand_point.mp4", "zzzzz-tsst.wav", "lightning_arc", "hand_point__fwoooom__fire_breath"),
            "lightning_dark": ("hand_palm.mp4", "zzzzz-tsst.wav", "lightning_arc", "hand_palm__boom__fire_burst"),
            "lightning_ball": ("hand_palm.mp4", "zzzzz-tsst.wav", "lightning_ball", "hand_palm__boom__fire_burst"),
            "water_orb": ("hand_open.mp4", "shhhhh-sploosh.wav", "water_orb", "hand_open__fff__flame_plume"),
        }[job]
        sample = root / "samples"
        meta = json.loads((sample / "samples.json").read_text(encoding="utf-8"))["plates"][plate]
        rc = json.loads((ahead / first / "receipt.json").read_text(encoding="utf-8"))
        offset = b.sample_noise_offset(sample / "voices" / voice, probe_video(sample / "plates" / plate).duration_s,
                                       int(meta["hand"].get("frame", 0)))
        spec = ShotSpec(plate=sample / "plates" / plate, voice=sample / "voices" / voice, effect=effect,
                        point=tuple(meta["point"]), seed=1, cue="voice", offset_s=offset)
        spec.extra["tap_frame"] = rc["bending"]["tap_frame"]
        b.bending_again(ahead / first, spec, dest, progress)
        return

    first = {"flame_walk": "hand_walk__fffff__flame_plume", "flamethrower": "hand_point__fwoooom__fire_breath"}.get(
        job, "hand_palm__boom__fire_burst")
    element = ahead / first
    rc = json.loads((element / "receipt.json").read_text(encoding="utf-8"))
    i, bending = rc["inputs"], rc["bending"]
    take = {"flame_walk": "IMG_6668", "flamethrower": "IMG_6653"}.get(job, "IMG_6658")
    voice = proof_voice(job, root, clips) if job.startswith("proof_") else None
    spec = ShotSpec(plate=takes / "plates" / f"{take}.mp4", voice=None if job == "proof_boom" else voice,
                    cue="voice" if voice is not None and job != "proof_boom" else "clip", effect=rc["effect"]["id"],
                    point=tuple(i["point"]), offset_s=0.0, mode="element", seed=i["seed"])
    capture = None
    if job == "proof_boom":
        from cuesync.transfer import gam
        track = json.loads((element / "hand_track.json").read_text(encoding="utf-8"))
        capture = np.memmap(temp / "how_carried.rgb", dtype=np.uint8, mode="w+",
                            shape=(len(track["xy"]), track["H"], track["W"], 3))
        capture[:] = 0
        original_warp = b.warp_element

        def capture_warp(*args, **kwargs):
            frame = original_warp(*args, **kwargs)
            k = int(args[3])
            if 0 <= k < len(capture):
                # Fire may pack its transported extinction alongside RGB. The
                # carried-on-black proof shows only emission, never that channel.
                capture[k] = np.maximum(capture[k], gam(frame[..., :3]))
            return frame

        b.warp_element = capture_warp
    if job == "how_light_off":
        import cuesync.fx_light as fx
        fx.light_plate = lambda *a, **k: None
        fx.flash = lambda *a, **k: None
        b.cap_light = lambda *a, **k: None
        b.contact_light = lambda *a, **k: None
        if hasattr(b, "fire_light"):
            b.fire_light.relight = lambda plate, *a, **k: (plate.copy(), np.zeros_like(plate), {})

    # A resolved default of zero is still a default, not a user-authored arrow.
    # Preserve that intent when a refreshed gallery receipt is replayed.
    aim = None if bending.get("aim_mode") == "upward_default" else bending.get("aim")
    result = b.make_bending_shot(spec, None, dest, element / "hand_track.json", element / "hand_mask.npz", progress,
                                 element_from=element, tap_frame=bending["tap_frame"], aim=aim)
    if capture is not None:
        n = round(probe_video(result.composite).duration_s * 24)
        write_video(clips / "how_carried.mp4", capture[:n], fps=24, crf=16)
        del capture
        (temp / "how_carried.rgb").unlink(missing_ok=True)
        off = result.receipt["bending"]["offset"] / 24
        duration = n / 24
        vf = f"tpad=start_duration={max(0, off):.6f}:start_mode=add:color=black,trim=duration={duration:.6f}"
        if off < 0:
            vf = f"trim=start={-off:.6f},setpts=PTS-STARTPTS,{vf}"
        vf += f",tpad=stop_mode=add:stop_duration={duration:.6f}:color=black,trim=duration={duration:.6f}"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(element / "element.mp4"), "-vf", vf,
             "-r", "24", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(clips / "how_raw.mp4")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--out", type=Path, default=Path("D:/openHiggsfield/runs/demo_v11"))
    parser.add_argument("--temp", type=Path, default=Path("D:/openHiggsfield_tmp/sol12/demo"))
    parser.add_argument("--only", nargs="+", choices=JOBS)
    parser.add_argument("--gallery-inputs", type=Path, help="Optional immutable gallery element/receipt/track/matte snapshot")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1,
                        help="Independent render subprocesses; concurrent timings are not a performance benchmark")
    parser.add_argument("--force", action="store_true", help="rebuild even if input and source hashes match")
    parser.add_argument("--refresh-unchanged", action="store_true",
                        help="refresh selected unchanged recipes after tooling-only edits; verify every source/output hash")
    parser.add_argument("--job", choices=JOBS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, out, temp = args.input_root.resolve(), args.out.resolve(), args.temp.resolve()
    gallery = args.gallery_inputs.resolve() if args.gallery_inputs else None
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(temp)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = str(SOURCE)
    if args.job:
        render_worker(args.job, root, clips, temp, gallery)
        return

    manifest_path = clips / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"jobs": {}}
    old_code = dict(manifest.get("source_sha256", {}))
    code = {str(p.relative_to(CHECKOUT)): sha(p) for p in sorted((SOURCE / "cuesync").glob("*.py"))}
    code[str(SCRIPT.relative_to(CHECKOUT))] = sha(SCRIPT)
    manifest.update({"input_root": str(root), "output_root": str(out), "source_root": str(SOURCE),
                     "source_sha256": code,
                     "claims": {"proof": "one shared playback element from one LTX-2.5 generation, three performances, zero new model renders; original and CPU-derivative ancestry recorded where applicable",
                                "explosion": "LTX-2.5 fireball plus CPU lighting, smoke treatment and compositing",
                                "lightning_and_water": "procedural CPU effects, synthetic sample noise",
                                "narration": "Kokoro-82M, a synthetic voice"}})
    hashes, tasks = {}, []
    for job in args.only or JOBS:
        ins = input_paths(job, root, gallery)
        missing = [str(p) for p in ins if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
        source_hashes = {}
        for p in ins:
            key = str(p)
            if key not in hashes:
                hashes[key] = sha(p)
            source_hashes[key] = hashes[key]
        fingerprint = hashlib.sha256(json.dumps({"code": code, "inputs": source_hashes}, sort_keys=True).encode()).hexdigest()
        previous = manifest["jobs"].get(job, {})
        expected = outputs(job, clips)
        if args.refresh_unchanged:
            core = lambda values: {k: v for k, v in values.items() if k != str(SCRIPT.relative_to(CHECKOUT))}
            if (not previous or core(old_code) != core(code) or previous.get("inputs_sha256") != source_hashes
                    or not all(p.exists() and previous.get("outputs_sha256", {}).get(str(p)) == sha(p) for p in expected)):
                raise ValueError(f"{job}: source or output changed; rebuild instead of refreshing provenance")
            previous.setdefault("cache_tool_sha256", old_code.get(str(SCRIPT.relative_to(CHECKOUT))))
            previous["fingerprint"] = fingerprint
            previous["provenance_note"] = "Recipe unchanged; source/output hashes verified after unrelated tooling edits"
            dump(manifest_path, manifest)
            print(f"{job}: unchanged recipe, provenance refreshed", flush=True)
            continue
        if not args.force and previous.get("fingerprint") == fingerprint and all(p.exists() for p in expected):
            print(f"{job}: up to date", flush=True)
            continue
        tasks.append((job, ins, source_hashes, fingerprint, expected))

    def rebuild(task):
        job, ins, source_hashes, fingerprint, expected = task
        log = clips / f"{job}.log"
        print(f"{job}: rebuilding (log: {log})", flush=True)
        started = time.time()
        cmd = [sys.executable, str(SCRIPT), "--input-root", str(root), "--out", str(out), "--temp", str(temp), "--job", job]
        if gallery:
            cmd += ["--gallery-inputs", str(gallery)]
        with log.open("w", encoding="utf-8") as f:
            completed = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        if completed.returncode or not all(p.exists() for p in expected):
            print(log.read_text(encoding="utf-8")[-12000:], flush=True)
            raise RuntimeError(f"{job} failed: see {log}")
        entry = {"fingerprint": fingerprint, "elapsed_seconds": round(time.time() - started, 2),
                 "cache_tool_sha256": code[str(SCRIPT.relative_to(CHECKOUT))],
                 "inputs_sha256": source_hashes, "outputs_sha256": {str(p): sha(p) for p in expected},
                 "no_new_render": True, "no_gpu": True,
                 "concurrent_workers": args.workers, "timing_is_benchmark": False,
                 "method": "existing studio recording; resulting shot recomposited by current checkout" if job == "studio" else
                           "current checkout CPU render/composite from cached model render; no new model render"}
        # An original-generation chain can contain several element.mp4 files.
        # Identify playback from the actual render recipe, never its basename alone.
        elements = {str(p): source_hashes[str(p)] for p in recipe_input_paths(job, root, gallery)
                    if p.name == "element.mp4"}
        if len(elements) == 1:
            playback = next(iter(elements))
            entry["playback_element"] = playback
            entry["element_sha256"] = elements[playback]
            lineage = source_lineage(Path(playback), root, gallery)
            if lineage:
                entry["source_lineage"] = lineage
                if lineage["derivative_recipe"]:
                    entry["method"] = "Current checkout CPU composite from one existing generation's CPU temporal derivative; no new model generation"
        receipt = clips / ("studio_result_current" if job == "studio" else job) / "receipt.json"
        if receipt.exists():
            entry["receipt"] = json.loads(receipt.read_text(encoding="utf-8"))
        return job, entry

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(rebuild, task): task[0] for task in tasks}
        try:
            for future in as_completed(pending):
                job, entry = future.result()
                # Only this main thread writes the manifest; workers own disjoint outputs.
                manifest["jobs"][job] = entry
                dump(manifest_path, manifest)
                print(f"{job}: done in {entry['elapsed_seconds']:.1f}s", flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    dump(manifest_path, manifest)


if __name__ == "__main__":
    main()
