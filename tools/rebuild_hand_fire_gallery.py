"""Recomposite the shipped fire gallery on CPU from its existing cached assets.

Default: inventory only. --run stages and fully decodes the requested shots;
--install additionally copies verified composites, finals, posters and receipts
into samples/made_ahead/bending. No element, track, matte or source is generated.
The renderer is a fail-closed sentinel, and CUDA is hidden in worker processes.

    python tools/rebuild_hand_fire_gallery.py
    python tools/rebuild_hand_fire_gallery.py --run --install
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
GALLERY = SAMPLES / "made_ahead/bending"
DEFAULT_RUN = ROOT / "runs/submission-polish-20261022/gallery"
BASE = "hand_palm__boom__fire_burst"
NAMES = (BASE, "hand_palm__pfpfboom__again", "hand_palm__pfx4__again",
         "hand_walk__fffff__flame_plume", "hand_fist__boom__fire_burst",
         "hand_punch__boom__fire_burst", "hand_open__fff__flame_plume",
         "hand_point__fwoooom__fire_breath", "hand_cupped__kaboom__fire_burst")
OUTPUT_FILES = ("composite.mp4", "final.mp4", "poster.jpg", "receipt.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def renderer_hashes() -> dict:
    # Snapshot actual rendering and timing modules, including newly integrated helpers.
    source = ROOT / "cuesync/src/cuesync"
    names = {"bending", "fx_light", "pipeline", "media", "audio", "cues", "effects", "transfer",
             "scene", "composite", "conform", "sfx_synth", "gestures", "hand_track"}
    paths = [p for p in source.glob("*.py") if p.stem in names or any(s in p.stem for s in ("fire", "surface", "volume", "relight"))]
    paths.extend((ROOT / "effects").glob("*.json"))
    return {relative(path): sha256(path) for path in sorted(paths)}


def inventory(names: list[str]) -> list[dict]:
    import av
    import numpy as np
    sys.path.insert(0, str(ROOT / "cuesync/src"))
    from cuesync.cues import load_cue
    from cuesync.bending import charge_window, hold_of, snap_launch

    index = json.loads((GALLERY / "index.json").read_text(encoding="utf-8"))
    entries: dict[str, list[dict]] = {}
    for entry in index.values():
        entries.setdefault(entry["dir"], []).append(entry)
    result = []
    for name in names:
        directory = GALLERY / name
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        cue_kind = receipt["inputs"].get("cue", "clip")
        candidates = entries[name]
        if len({entry["plate"] for entry in candidates}) != 1:
            raise ValueError(f"Gallery aliases disagree about the source plate: {name}")
        # Later index aliases may preselect a synthetic voice for a future user
        # performance. They are not evidence of this finished shot's input audio.
        entry = next((entry for entry in candidates if not entry.get("voice")), candidates[0]) if cue_kind == "clip" else candidates[0]
        source = GALLERY / (BASE if name.endswith("__again") else name)
        required = {"plate": SAMPLES / "plates" / entry["plate"], "element": source / "element.mp4",
                    "element_receipt": source / "receipt.json", "track": directory / "hand_track.json",
                    "mask": directory / "hand_mask.npz", "receipt": receipt_path}
        if cue_kind == "voice" and entry.get("voice"):
            required["voice"] = SAMPLES / "voices" / entry["voice"]
        elif cue_kind == "voice":
            # The CC0 unfa recording used by the stock cupped-hand example was absent
            # from the legacy index. Its original bytes now travel with that shot.
            required["voice"] = directory / "source_voice.wav"
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("CPU rebuild requires cached assets; missing: " + ", ".join(missing))
        track = json.loads(required["track"].read_text(encoding="utf-8"))
        n, w, h = len(track["xy"]), int(track["W"]), int(track["H"])
        with np.load(required["mask"]) as archive:
            mask_shape = list(archive["alpha"].shape)
        if mask_shape != [n, h, w]:
            raise ValueError(f"Cached matte does not match track: {name}: {mask_shape} != {[n, h, w]}")
        for key in ("plate", "element"):
            with av.open(str(required[key])) as container:
                video = container.streams.video[0]
                if video.average_rate is None or video.width <= 0 or video.height <= 0:
                    raise ValueError(f"Invalid cached {key}: {required[key]}")
                if key == "plate" and (Fraction(video.average_rate) != 24 or video.frames != n
                                       or abs(video.width / video.height - w / h) > 1e-4):
                    raise ValueError(f"Cached plate must match the track's aspect and 24fps timeline: {required[key]}")
        if cue_kind not in ("clip", "voice"):
            raise ValueError(f"Original authored hold timings are required for this cue mode: {name}")
        cue = load_cue(cue_kind, plate=required["plate"], voice=required.get("voice"),
                       offset_s=float(receipt["inputs"].get("offset_s", 0)), duration_s=n / 24)
        cue_frames = [hit.frame for hit in cue.cue.hits()]
        env = np.asarray(cue.cue.envelope, np.float64)
        raw_hits = cue_frames or [cue.cue.frame_of(cue.cue.hero_onset_s)]
        loud = [float(env[f:f + 8].max()) if 0 <= f < env.size else 0.0 for f in raw_hits]
        hero = raw_hits[int(np.argmax(loud))]
        charge = charge_window(cue.cue, env, hero, 24)
        # The production renderer promotes a sustained build-up's release into a
        # launch before gesture snapping. Raw audio onsets are not final hits.
        render_cues = (sorted({f for f in raw_hits if not charge[0] - 2 <= f < charge[1] + 3}
                              | {charge[1]}) if charge else raw_hits)
        expected_cues = [hit.get("snap", {}).get("cue", hit["frame"]) for hit in receipt["bending"]["hits"]]
        if render_cues != expected_cues:
            raise ValueError(f"Cue reproduction changed for {name}: {render_cues} != original {expected_cues}")
        expected_rendered = [hit["frame"] for hit in receipt["bending"]["hits"]]
        planned_rendered = [snap_launch(frame, track.get("events", []))["frame"] for frame in render_cues]
        if planned_rendered != expected_rendered:
            raise ValueError(f"Gesture-snapped timing changed for {name}: {planned_rendered} != {expected_rendered}")
        old_held = receipt["bending"].get("held")
        expected_holds = old_held.get("holds") if old_held else None
        if expected_holds is not None:
            analyzed_holds = [hold_of(frame, cue.cue.intervals()) for frame in expected_rendered]
            if analyzed_holds != expected_holds:
                raise ValueError(f"Cue hold duration changed for {name}: {analyzed_holds} != {expected_holds}")
        result.append({"name": name, "index": entry, "receipt": receipt,
                       "source": relative(source), "frames": n, "size": [w, h],
                       "raw_audio_hit_frames": cue_frames,
                       "render_cue_hit_frames": render_cues,
                       "charge_window": list(charge) if charge else None,
                       "cue_voice_end_frame": cue.cue.voice_end_frame,
                       "expected_rendered_hit_frames": expected_rendered,
                       "expected_hold_frames": expected_holds,
                       "landmark_frames": sum(bool(x) for x in track.get("landmarks", [])),
                       "assets": {key: {"path": relative(path), "sha256": sha256(path)} for key, path in required.items()}})
    return result


def assert_immutable_assets(item: dict) -> None:
    for kind, asset in item["assets"].items():
        if kind in ("receipt", "element_receipt"):
            continue  # The only inputs whose metadata is intentionally refreshed on install.
        if sha256(ROOT / asset["path"]) != asset["sha256"]:
            raise RuntimeError(f"Cached {kind} changed during the CPU rebuild: {item['name']}")


def decode_video(path: Path, n: int, size: list[int] | None = None) -> dict:
    import av
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        dims = [video.width, video.height]
        if Fraction(video.average_rate) != 24 or (size is not None and dims != size):
            raise ValueError(f"Unexpected output geometry or frame rate: {path}")
        cc = video.codec_context
        tags = {"colorspace": cc.colorspace, "primaries": cc.color_primaries,
                "transfer": cc.color_trc, "range": cc.color_range}
        if any(tag != 1 for tag in tags.values()):
            raise ValueError(f"Missing explicit limited-range BT709 tags: {path}: {tags}")
        count = sum(1 for _ in container.decode(video=0))
    if count != n:
        raise ValueError(f"Wrong decoded frame count for {path}: {count} != {n}")
    with av.open(str(path)) as container:
        if len(container.streams.audio) != 1:
            raise ValueError(f"Expected one audio stream: {path}")
        samples = sum(frame.samples for frame in container.decode(audio=0))
    if samples == 0:
        raise ValueError(f"Empty output audio stream: {path}")
    return {"frames": count, "size": dims, "fps": 24, "audio_samples_decoded": samples,
            "color_tags": tags, "sha256": sha256(path)}


def prepare_reuse_source(item: dict, run: Path) -> tuple[Path, dict | None]:
    """Author explicit current intent for the legacy stationary cupped example."""
    original = ROOT / item["source"]
    if item["name"] != "hand_cupped__kaboom__fire_burst":
        return original, None
    prior = item["receipt"].get("recomposite", {}).get("authored_reuse_intent")
    intent = prior or {
        "field": "bending.thrown", "authored_value": None,
        "historical_field_present": "thrown" in item["receipt"]["bending"],
        "original_source_receipt_sha256": item["assets"]["element_receipt"]["sha256"],
        "basis": "Current anchored-intent reconstruction from the historical stationary cupped-hand composition. The original legacy receipt did not encode a thrown/anchored choice.",
        "authored_as_current_choice": True,
    }
    if intent.get("field") != "bending.thrown" or intent.get("authored_value") is not None:
        raise ValueError("The cupped example requires its explicit anchored reconstruction intent")
    copied = json.loads(json.dumps(item["receipt"]))
    copied["bending"]["thrown"] = None
    copied["current_reconstruction_choice"] = intent
    destination = run / "reuse-inputs" / item["name"]
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / item["assets"]["element"]["path"], destination / "element.mp4")
    if sha256(destination / "element.mp4") != item["assets"]["element"]["sha256"]:
        raise RuntimeError("The private reuse snapshot changed the original element bytes")
    write_json(destination / "receipt.json", copied)
    return destination, intent


def worker(name: str, run: Path) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, str(ROOT / "cuesync/src"))
    import cv2
    from cuesync.bending import make_bending_shot
    from cuesync.pipeline import ShotSpec

    cv2.setNumThreads(4)
    item = inventory([name])[0]
    receipt, entry = item["receipt"], item["index"]
    assets = item["assets"]
    params, bend = receipt["inputs"], receipt["bending"]
    effective_aim = None if bend.get("aim_mode") == "upward_default" else bend.get("aim")
    codes = renderer_hashes()
    recipe = sha256(Path(__file__))
    spec = ShotSpec(plate=ROOT / assets["plate"]["path"],
                    voice=ROOT / assets["voice"]["path"] if params.get("cue") == "voice" and "voice" in assets else None,
                    effect=receipt["effect"]["id"], point=tuple(params["point"]),
                    seed=int(params.get("seed", entry.get("seed", 1))),
                    offset_s=float(params.get("offset_s", 0.0)), cue=params.get("cue", "clip"),
                    direction=effective_aim, mode="element", plate_size=tuple(item["size"]))
    if spec.cue not in ("clip", "voice"):
        raise ValueError(f"No authored hold reconstruction is permitted without its original timings: {name}")
    destination = run / "staged" / name
    destination.mkdir(parents=True, exist_ok=True)
    reuse_source, intent = prepare_reuse_source(item, run)
    if intent is not None:
        item["authored_reuse_intent"] = intent
        item["reuse_input_snapshot"] = {
            "path": relative(reuse_source), "receipt_sha256": sha256(reuse_source / "receipt.json"),
            "element_sha256": sha256(reuse_source / "element.mp4"),
        }

    def deny_render(*args, **kwargs):
        raise RuntimeError("New model generation is forbidden: only existing LTX elements may be reused")

    def progress(stage, fraction, message):
        print(f"{name} | {stage} {fraction:.0%} | {message}", flush=True)

    start = time.perf_counter()
    result = make_bending_shot(spec, deny_render, destination, ROOT / assets["track"]["path"],
                              ROOT / assets["mask"]["path"], progress,
                              element_from=reuse_source, tap_frame=int(bend["tap_frame"]),
                              aim=effective_aim)
    elapsed = time.perf_counter() - start
    if renderer_hashes() != codes or sha256(Path(__file__)) != recipe:
        raise RuntimeError("Rendering source or recipe changed during this shot; discard it and rebuild after freeze")
    validate_render(item, run, result.receipt, elapsed, codes, recipe)


def validate_render(item: dict, run: Path, updated: dict, elapsed: float, codes: dict, recipe: str) -> None:
    """Check completed media against current source analysis and record provenance."""
    name, assets = item["name"], item["assets"]
    destination = run / "staged" / name
    params = item["receipt"]["inputs"]
    assert_immutable_assets(item)
    expected_hits = item["expected_rendered_hit_frames"]
    actual_hits = [hit["frame"] for hit in updated["bending"]["hits"]]
    if actual_hits != expected_hits:
        raise RuntimeError(f"Composite timing changed for {name}: {actual_hits} != original {expected_hits}")
    if item["expected_hold_frames"] is not None:
        actual_holds = (updated["bending"].get("held") or {}).get("holds")
        if actual_holds != item["expected_hold_frames"]:
            raise RuntimeError(f"Composite hold duration changed for {name}: {actual_holds} != {item['expected_hold_frames']}")
    if updated["inputs"]["cue"] != params.get("cue", "clip") or updated["inputs"]["offset_s"] != float(params.get("offset_s", 0)):
        raise RuntimeError(f"Composite cue source/offset changed for {name}")
    if float(updated["render"]["seconds"]) != 0.0:
        raise RuntimeError("CPU-only gallery rebuild unexpectedly recorded generation time")
    if "authored_reuse_intent" in item:
        if "thrown" not in updated["bending"] or updated["bending"]["thrown"] is not None:
            raise RuntimeError("Cupped anchored reconstruction was not honored by the renderer")
        snapshot = item["reuse_input_snapshot"]
        if (sha256(ROOT / snapshot["path"] / "receipt.json") != snapshot["receipt_sha256"]
                or sha256(ROOT / snapshot["path"] / "element.mp4") != snapshot["element_sha256"]):
            raise RuntimeError("The private reuse snapshot changed while rendering")
    element_hash = assets["element"]["sha256"]
    updated["element_sha256"] = element_hash
    if "voice" in assets:
        updated["inputs"]["voice"] = ("source_voice.wav" if Path(assets["voice"]["path"]).parent == Path(relative(GALLERY / name))
                                         else assets["voice"]["path"])
        updated["inputs"]["voice_sha256"] = assets["voice"]["sha256"]
    updated["bending"]["track"] = "hand_track.json"
    updated["bending"]["mask"] = "hand_mask.npz"
    updated["bending"]["performed_again"] = (
        {"from": item["source"], "no_new_render": True, "no_gpu": True, "element_sha256": element_hash}
        if name.endswith("__again") else None)
    updated["recomposite"] = {"method": "CPU compositing from the existing LTX element, track and matte",
                              "no_new_render": True, "no_gpu": True, "element_sha256": element_hash,
                              "source_receipt_sha256": assets["receipt"]["sha256"],
                              "raw_audio_hit_frames": item["raw_audio_hit_frames"],
                              "render_cue_hit_frames": item["render_cue_hit_frames"],
                              "charge_window": item["charge_window"],
                              "cue_voice_end_frame": item["cue_voice_end_frame"],
                              "utc": datetime.now(timezone.utc).isoformat(), "seconds": round(elapsed, 3),
                              "renderer_sources": codes, "rebuild_recipe_sha256": recipe}
    if "authored_reuse_intent" in item:
        updated["recomposite"]["authored_reuse_intent"] = item["authored_reuse_intent"]
        updated["recomposite"]["reuse_input_snapshot"] = item["reuse_input_snapshot"]
    write_json(destination / "receipt.json", updated)
    validation = {"composite": decode_video(destination / "composite.mp4", item["frames"], item["size"]),
                  "final": decode_video(destination / "final.mp4", item["frames"])}
    from PIL import Image
    with Image.open(destination / "poster.jpg") as poster:
        poster.load()
        if list(poster.size) != item["size"]:
            raise ValueError(f"Poster geometry changed: {name}")
    write_json(run / "validation" / f"{name}.json", {"name": name, "seconds": elapsed,
               "inputs": item, "renderer_sources": codes, "rebuild_recipe_sha256": recipe, "validation": validation,
               "output_sha256": {file: sha256(destination / file) for file in OUTPUT_FILES}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", choices=NAMES, action="append")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--run", action="store_true", help="Recomposite locally after production sources are frozen")
    parser.add_argument("--install", action="store_true", help="Install only after every selected shot decodes successfully")
    parser.add_argument("--reuse-validated", action="store_true", help="Reuse staged pilot outputs only when all inputs, rendering sources and outputs still match")
    parser.add_argument("--worker", choices=NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run = args.run_root.resolve()
    if args.worker:
        worker(args.worker, run)
        return
    if args.install and not args.run:
        parser.error("--install requires --run")
    names = list(dict.fromkeys(args.shot or NAMES))
    items = inventory(names)
    codes = renderer_hashes()
    recipe = sha256(Path(__file__))
    write_json(run / "inventory.json", {"shots": items, "renderer_sources": codes, "rebuild_recipe_sha256": recipe, "cpu_only": True})
    print(json.dumps({"shots": len(items), "missing_assets": [], "frames": sum(x["frames"] for x in items),
                      "inventory": str(run / "inventory.json"), "run_requested": args.run}, indent=2), flush=True)
    if not args.run:
        return
    for item in items:
        backup = run / "before" / item["name"]
        backup.mkdir(parents=True, exist_ok=True)
        for file in OUTPUT_FILES:
            source = GALLERY / item["name"] / file
            target = backup / file
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONUNBUFFERED"] = "1"
    for item in items:
        name = item["name"]
        validated = run / "validation" / f"{name}.json"
        if args.reuse_validated and validated.exists():
            previous = json.loads(validated.read_text(encoding="utf-8"))
            same_inputs = previous["inputs"]["assets"] == item["assets"]
            same_outputs = all((run / "staged" / name / file).exists()
                               and sha256(run / "staged" / name / file) == expected
                               for file, expected in previous["output_sha256"].items())
            if (same_inputs and same_outputs and previous["renderer_sources"] == codes
                    and previous.get("rebuild_recipe_sha256") == recipe):
                print(f"Reusing verified pilot output: {name}", flush=True)
                continue
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name, "--run-root", str(run)],
                       cwd=ROOT, env=env, check=True)
    if renderer_hashes() != codes or sha256(Path(__file__)) != recipe:
        raise RuntimeError("Rendering sources or recipe changed during the batch; no gallery files were installed")
    completed = []
    for item in items:
        assert_immutable_assets(item)
        name = item["name"]
        validation = json.loads((run / "validation" / f"{name}.json").read_text(encoding="utf-8"))
        if validation["renderer_sources"] != codes:
            raise RuntimeError(f"Shot was rendered with a different source snapshot: {name}")
        for file, expected in validation["output_sha256"].items():
            if sha256(run / "staged" / name / file) != expected:
                raise RuntimeError(f"Staged artifact changed after validation: {name}/{file}")
        completed.append(validation)
    if args.install:
        for item in items:
            name = item["name"]
            for file in OUTPUT_FILES:
                shutil.copy2(run / "staged" / name / file, GALLERY / name / file)
        base_hash = sha256(GALLERY / BASE / "element.mp4")
        for name in (n for n in names if n.endswith("__again")):
            receipt = json.loads((GALLERY / name / "receipt.json").read_text(encoding="utf-8"))
            replay = receipt["bending"]["performed_again"]
            if replay.get("element_sha256") != base_hash or not replay.get("no_new_render") or not replay.get("no_gpu"):
                raise RuntimeError(f"Replay provenance verification failed: {name}")
    write_json(run / "manifest.json", {"installed": args.install, "cpu_only": True, "new_model_renders": 0,
                                      "gpu_used": False, "renderer_sources": codes, "rebuild_recipe_sha256": recipe, "shots": completed})
    print(json.dumps({"installed": args.install, "shots": len(completed), "manifest": str(run / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
