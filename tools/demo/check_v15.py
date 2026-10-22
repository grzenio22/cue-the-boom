"""Check the final cut, regenerated sources, and the v15/v16 edit constraints.

    python tools/demo/check_v15.py
    python tools/demo/check_v15.py --metadata-only

The decoded video is checked frame by frame and sampled every two seconds for
timecoded contact sheets. Source hashes and storyboard dictionaries establish
preservation after rebuilding the effects; old encoded pictures are not used
as the reference. Waveform measurements establish presence and level, while a
listening pass remains necessary to judge intelligibility and the edit.
The default video follows the storyboard version; v16 preserves v15 timing and
speech while refreshing production fire pictures and separating Studio step labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cuesync" / "src"))
from cuesync.media import read_audio  # noqa: E402
from build_demo import FPS, H, SR, W  # noqa: E402
from check_v11 import (  # noqa: E402
    author_voice_check, levels, narration_check, pronunciation_key, ranges,
    scan_video, source_bounds_check, timeline, voice_presence,
)


ROOT = Path("D:/openHiggsfield")
CHECKOUT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/demo_v15"
RECAP = {"innovation", "innovation_asset", "innovation_track", "innovation_voice", "innovation_result"}
AUDIO_REPAIRS = {"intro_raw", "intro_finished", "workflow_fire", "how", "light"}
AUDIO_FIELDS = {"audio_src", "audio_from", "audio_at", "audio_gain", "audio_replace", "source_gain",
                "audio_speed", "sound_speed", "slow_audio", "slow_sound", "room_tone_dbfs"}
VOICE_FIELDS = {"vo", "vo_at", "vo_gain", "vo_text", "vo_captions"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def beats_of(board: dict | list) -> list[dict]:
    return board["beats"] if isinstance(board, dict) else board


def structure_check(board: dict | list, baseline: dict | list) -> tuple[dict, list[str]]:
    """Compare unchanged beat fields, with explicit exceptions for this edit."""
    if isinstance(board, dict) and board.get("version") == 16:
        return integration_structure_check(board, baseline)
    beats, previous = beats_of(board), beats_of(baseline)
    errors, changed, unchanged = [], [], []
    old = {b["id"]: b for b in previous}
    ids = [b["id"] for b in beats]
    if len(set(ids)) != len(ids):
        errors.append("Duplicate beat identifiers")
    alias = {"lightning_palm": "water", "palm_lightning": "water"}
    original_ids = [alias.get(name, name) for name in ids if alias.get(name, name) in old]
    if original_ids != [b["id"] for b in previous]:
        errors.append("An approved v14 beat was removed, reordered, or duplicated")
    inserted = [b for b in beats if alias.get(b["id"], b["id"]) not in old]
    inserted_seconds = sum(round(float(b["dur"]) * FPS) / FPS for b in inserted)
    if not 2.75 <= inserted_seconds <= 3.25:
        errors.append(f"Own node close-ups must total about 3 seconds, found {inserted_seconds:.6f}")
    if inserted:
        left = ids.index("comfy_output") if "comfy_output" in ids else -1
        right = ids.index("workflow_fire") if "workflow_fire" in ids else -1
        if any(not left < ids.index(b["id"]) < right for b in inserted):
            errors.append("New node close-ups are outside the LTX/ComfyUI workflow beat")
        if any(b["type"] not in {"image", "clip"} for b in inserted):
            errors.append("Own node close-ups must use readable cards or video close-ups")
        caption = " ".join(str(b.get(field, "")) for b in inserted
                           for field in ("caption", "headline", "subline", "vo_text", "stamp", "node_caption"))
        key = pronunciation_key(caption)
        if not all(word in key for word in ("sound", "guide", "ltx")):
            errors.append("Own node caption must explain sound to guide to LTX")
    paid_seconds = 0.0
    for beat in beats:
        identity = alias.get(beat["id"], beat["id"])
        if identity not in old:
            continue
        prior = old[identity]
        fields = {k for k in set(beat) | set(prior) if beat.get(k) != prior.get(k)}
        allowed = set()
        if identity in AUDIO_REPAIRS:
            allowed |= AUDIO_FIELDS
        if identity == "cold_open":
            allowed.add("dur")
            paid_seconds += float(prior["dur"]) - float(beat["dur"])
            if not 6 <= float(beat["dur"]) <= float(prior["dur"]):
                errors.append("Cold open shortening extends before its clean final hold or lengthens the beat")
        if identity == "water":
            allowed |= {"src", "start", "id"}
            if "lightning" not in str(beat.get("src", "")).lower():
                errors.append("Procedural water beat was not replaced with palm lightning")
            if beat.get("chip") != "Procedural CPU effects":
                errors.append("Palm lightning lost its Procedural CPU effects label")
        if identity in RECAP:
            allowed |= {"dur", "start", "src"}
            if float(beat["dur"]) > float(prior["dur"]) + 1 / FPS:
                errors.append(f"Time-payment recap stage lengthened: {identity}")
            paid_seconds += float(prior["dur"]) - float(beat["dur"])
            if "src" in fields and Path(beat["src"]).name != Path(prior["src"]).name:
                errors.append(f"Recap source changed to a different animation: {identity}")
        if identity == "innovation":
            allowed |= VOICE_FIELDS
        extra = fields - allowed
        if extra:
            errors.append(f"Unapproved beat fields changed: {identity} ({', '.join(sorted(extra))})")
        if fields:
            changed.append(dict(id=identity, changed_fields=sorted(fields), allowed_fields=sorted(allowed)))
        else:
            unchanged.append(identity)
    if paid_seconds + 1 / FPS < inserted_seconds:
        errors.append(f"Own node time was not paid from the recap/clean hold: added {inserted_seconds:.6f}s, saved {paid_seconds:.6f}s")
    current, before = timeline(beats), timeline(previous)
    duration = current[-1]["end"] if current else 0
    if duration > 118 + .000001:
        errors.append(f"Picture exceeds 118 seconds: {duration:.6f}s")
    if isinstance(board, dict):
        if board.get("version") != 15:
            errors.append("Storyboard version is not 15")
        if abs(float(board.get("duration", duration)) - duration) > .002:
            errors.append("Storyboard duration does not match its frame-rounded beats")
    return dict(unchanged_beats=unchanged, permitted_changes=changed,
                inserted_beats=[b["id"] for b in inserted], inserted_seconds=inserted_seconds,
                recap_and_clean_hold_seconds_saved=round(paid_seconds, 6), baseline_duration=before[-1]["end"],
                duration=duration, expected_frames=sum(b["frames"] for b in current),
                timeline=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in current]), errors


def integration_structure_check(board: dict, baseline: dict | list) -> tuple[dict, list[str]]:
    """The integration cut may replace media and separate confirmed Studio overlaps."""
    beats, previous = beats_of(board), beats_of(baseline)
    errors, changed = [], []
    if [b["id"] for b in beats] != [b["id"] for b in previous]:
        errors.append("The v16 integration cut changed the approved beat sequence")

    def editorial(value):
        if isinstance(value, dict):
            return {k: editorial(v) for k, v in value.items()
                    if k not in {"src", "srcs", "element_src", "chip_header"}}
        return [editorial(v) for v in value] if isinstance(value, list) else value

    for beat, prior in zip(beats, previous):
        comparable = dict(beat)
        if beat["id"] == "perform":
            if beat.get("wave_box") != [.54, .025, .4, .115]:
                errors.append("The Studio voice waveform must stay within the reserved header")
            comparable["wave_box"] = prior.get("wave_box")
        if editorial(comparable) != editorial(prior):
            errors.append(f"Integration changed timing, narration, audio, or editorial content: {beat['id']}")
        if beat.get("chip_header") and beat["id"] not in {"upload", "perform", "make"}:
            errors.append(f"Unexpected header layout change: {beat['id']}")
        if beat != prior:
            changed.append(dict(id=beat["id"], changed_fields=sorted(k for k in set(beat) | set(prior)
                                                                   if beat.get(k) != prior.get(k))))
    for identity in ("upload", "perform", "make"):
        step = next((b for b in beats if b["id"] == identity), {})
        if not step.get("chip_header"):
            errors.append(f"{identity} chip has no reserved header band")
    current, before = timeline(beats), timeline(previous)
    duration = current[-1]["end"]
    if abs(duration - before[-1]["end"]) > 1e-6 or duration > 118:
        errors.append("Integration changed the approved demo duration")
    if abs(float(board.get("duration", duration)) - duration) > .002:
        errors.append("Storyboard duration does not match its frame-rounded beats")
    return dict(permitted_changes=changed, baseline_duration=before[-1]["end"], duration=duration,
                expected_frames=sum(b["frames"] for b in current),
                timeline=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in current]), errors


def source_provenance_check(board: dict | list) -> tuple[dict, list[str]]:
    """Verify current renderer fingerprints and every recorded output digest."""
    from make_v11_clips import JOBS

    errors, manifests, paths, hashes = [], [], set(), {}

    def digest(path: Path) -> str:
        if path not in hashes:
            hashes[path] = sha256(path)
        return hashes[path]

    def collect(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name in {"src", "audio_src", "vo", "element_src"} and isinstance(child, str):
                    paths.add(Path(child))
                elif name == "srcs":
                    paths.update(Path(p) for p in child)
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(board)
    current_manifest = Path(board["provenance"]) if isinstance(board, dict) and board.get("provenance") else (
        ROOT / "runs/demo_v11/clips/manifest.json")
    candidates = {current_manifest}
    for p in paths:
        for ancestor in p.parents:
            if ancestor.name == "clips" and (ancestor / "manifest.json").exists():
                candidates.add(ancestor / "manifest.json")
                break
    production_outputs = set()
    for path in sorted(candidates):
        if not path.is_file():
            errors.append(f"Missing rebuilt clip manifest: {path}")
            continue
        data = read_json(path)
        historical = path != current_manifest
        checked_code, checked_inputs, checked_outputs = [], [], []
        code = data.get("source_sha256", {})
        if not code:
            errors.append(f"No current source fingerprints in {path}")
        for relative, expected in code.items():
            source = CHECKOUT / relative
            good = source.is_file() and digest(source) == expected
            checked_code.append(dict(source=str(source), matches=good))
            if not good and not historical:
                errors.append(f"Rebuilt clips do not match current code: {source}")
        missing = sorted(set(JOBS) - set(data.get("jobs", {})))
        if not historical and path.parent == ROOT / "runs/demo_v11/clips" and missing:
            errors.append(f"Rebuilt manifest lacks jobs: {', '.join(missing)}")
        for name, job in data.get("jobs", {}).items():
            for filename, expected in job.get("inputs_sha256", {}).items():
                if historical:
                    continue
                source = Path(filename)
                good = source.is_file() and digest(source) == expected
                checked_inputs.append(dict(job=name, source=filename, matches=good))
                if not good:
                    errors.append(f"Rebuilt input differs from manifest: {source}")
            if not job.get("outputs_sha256"):
                errors.append(f"No rebuilt output hashes for {name}")
            for filename, expected in job.get("outputs_sha256", {}).items():
                output = Path(filename)
                if not historical:
                    production_outputs.add(output)
                if historical and output not in paths:
                    continue
                good = output.is_file() and digest(output) == expected
                checked_outputs.append(dict(job=name, source=filename, matches=good))
                if not good:
                    errors.append(f"Rebuilt output differs from manifest: {output}")
            input_code = {"code": code, "inputs": job.get("inputs_sha256", {})}
            expected_fingerprint = hashlib.sha256(json.dumps(input_code, sort_keys=True).encode()).hexdigest()
            if job.get("fingerprint") != expected_fingerprint:
                errors.append(f"Job fingerprint does not use current code and recorded inputs: {name}")
        if not historical and isinstance(board, dict) and board.get("version", 15) >= 16:
            proof = [data.get("jobs", {}).get(name, {}) for name in
                     ("proof_boom", "proof_pf_pf_boom", "proof_pf_x4")]
            shared = {job.get("element_sha256") for job in proof}
            if len(shared) != 1 or None in shared:
                errors.append("Three demo performances do not prove reuse of the identical element SHA-256")
            if any({sha for filename, sha in job.get("inputs_sha256", {}).items()
                    if Path(filename).name == "element.mp4"} != {job.get("element_sha256")} for job in proof):
                errors.append("Demo proof element receipt differs from its verified source element")
            if not all(job.get("no_new_render") is True and job.get("no_gpu") is True for job in proof):
                errors.append("Three demo performances lack explicit CPU/no-new-model-render provenance")
        manifests.append(dict(path=str(path),
                              policy="Historical retained media: referenced output hashes preserved; no claim of a rebuild by current code"
                                     if historical else "Current production composites: current code, input and output hashes required",
                              code=checked_code, inputs=checked_inputs, outputs=checked_outputs))
    if isinstance(board, dict) and board.get("version", 15) >= 16:
        for path in paths:
            if path.is_relative_to(current_manifest.parent) and path not in production_outputs:
                errors.append(f"Storyboard production clip has no rendered-output provenance: {path}")
    assets = []
    for path in sorted(paths):
        if not path.is_file():
            errors.append(f"Missing storyboard source: {path}")
        else:
            assets.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    derived = []
    for receipt in sorted({p.parent / "provenance.json" for p in paths
                           if p.parent.name == "cards" and (p.parent / "provenance.json").is_file()}):
        data = read_json(receipt)
        checks = []
        for field in ("sources_sha256", "outputs_sha256"):
            for filename, expected in data.get(field, {}).items():
                path = Path(filename)
                good = path.is_file() and digest(path) == expected
                checks.append(dict(path=filename, matches=good, kind=field))
                if not good:
                    errors.append(f"Derived recap media differs from provenance: {path}")
        derived.append(dict(path=str(receipt), checks=checks))
    if isinstance(board, dict) and board.get("version", 15) >= 16:
        for path in paths:
            if path.is_relative_to(current_manifest.parent.parent / "cards"):
                receipt = path.parent / "provenance.json"
                if not receipt.is_file() or str(path) not in read_json(receipt).get("outputs_sha256", {}):
                    errors.append(f"Storyboard recap has no derived-output provenance: {path}")
    return dict(method="Current checkout code and rebuilt source/output SHA-256 digests; approved beat dictionaries and valid source windows",
                manifests=manifests, storyboard_assets=assets, derived_cards=derived), errors


def voice_provenance_check(schedule: list[dict]) -> tuple[dict, list[str]]:
    errors, rows, contribution = [], [], None
    for entry in schedule:
        beat = entry["beat"]
        if not beat.get("vo"):
            continue
        path = Path(beat["vo"])
        if not path.is_file():
            continue
        provenance_path = path.parent / "provenance.json"
        if not provenance_path.is_file():
            errors.append(f"Missing normal-tempo narration provenance: {provenance_path}")
            continue
        provenance = read_json(provenance_path)
        speed = provenance.get("speeds", {}).get(path.stem, provenance.get("speed"))
        row = dict(id=entry["id"], path=str(path), speed=speed, provenance=str(provenance_path))
        if speed is None or abs(float(speed) - 1.0) > 1e-8 or float(provenance.get("tempo", 1)) != 1:
            errors.append(f"Narration is not documented at normal tempo: {entry['id']}")
        spoken = provenance.get("lines", {}).get(path.stem, "")
        if pronunciation_key(spoken) != pronunciation_key(beat.get("vo_text", "")):
            errors.append(f"Narration provenance transcript differs from captions: {entry['id']}")
        tokens_path = path.parent / "token_timings.json"
        tokens = read_json(tokens_path).get(path.stem, []) if tokens_path.is_file() else []
        words = [t for t in tokens if pronunciation_key(t.get("text", ""))]
        if words:
            row["token_words_match_transcript"] = pronunciation_key(" ".join(t["text"] for t in words)) == pronunciation_key(spoken)
            if not row["token_words_match_transcript"]:
                errors.append(f"Narration token words differ from transcript: {entry['id']}")
            row["final_word"] = words[-1]
            duration = read_audio(path, SR).size / SR
            last_end = words[-1].get("end")
            if last_end is None or float(last_end) > duration + .002:
                errors.append(f"Final narration word extends past source: {entry['id']}")
            boundaries = [float(c[1]) for c in beat.get("vo_captions", [])[:-1]]
            cuts = [dict(boundary=t, word=w) for t in boundaries for w in words
                    if w.get("start") is not None and w.get("end") is not None
                    and float(w["start"]) + .001 < t < float(w["end"]) - .001]
            row["caption_changes_inside_words"] = cuts
            if cuts:
                errors.append(f"Narration caption changes inside a word: {entry['id']}")
        elif entry["id"] == "innovation":
            errors.append("Contribution lacks word timestamps to verify complete final word")
        if entry["id"] == "innovation":
            contribution = row
            text = pronunciation_key(beat.get("vo_text", ""))
            if not all(part in text for part in ("contribution", "separatinggenerationfromperformance", "ltx", "instrument", "voice", "generation")):
                errors.append("Short contribution does not retain generation/performance/instrument meaning")
            if path == ROOT / "runs/demo_v14/vo14/new12.wav":
                errors.append("Contribution still uses the sped-up v14 narration")
            row["transcript"] = beat.get("vo_text", "")
        rows.append(row)
    if contribution is None:
        errors.append("Contribution narration is missing")
    return dict(lines=rows, contribution=contribution), errors


def node_evidence_check(beats: list[dict]) -> tuple[dict, list[str]]:
    """Read exact display names, sockets and real links from the node sources."""
    from PIL import Image
    from v15_nodes import definitions, render, source_graph

    errors, rows = [], []
    for beat in beats:
        if beat["id"] != "comfy_cuesync":
            continue
        image_path = Path(beat["src"])
        evidence_path = image_path.with_name(image_path.stem + "_evidence.json")
        if not evidence_path.is_file():
            errors.append(f"Missing own-node source evidence: {evidence_path}")
            continue
        evidence = read_json(evidence_path)
        pack = CHECKOUT / "comfyui-cuesync/nodes.py"
        workflow = CHECKOUT / "workflows/cue_the_boom_scene.api.json"
        specs = definitions(pack)
        graph, ids, edges = source_graph(workflow, specs)
        if specs != evidence.get("definitions") or edges != evidence.get("checked_edges"):
            errors.append("Own node display names, ports or cables differ from the repository definitions/workflow")
        for relative, expected in evidence.get("sources", {}).items():
            path = CHECKOUT / relative
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"Own node card evidence refers to different source: {path}")
        if len(evidence.get("sources", {})) < 2:
            errors.append("Own node card lacks node-pack and workflow source fingerprints")
        if evidence.get("size") != [W, H] or evidence.get("caption_safe_y", H) > 840:
            errors.append("Own node card does not reserve the narration caption safe area")
        if evidence.get("caption") != beat.get("node_caption"):
            errors.append("Own node explanatory caption differs from storyboard metadata")
        expected = np.asarray(render(specs, graph, ids))
        actual = np.asarray(Image.open(image_path).convert("RGB"))
        matches = actual.shape == expected.shape and np.array_equal(actual, expected)
        if not matches:
            errors.append("Own node image differs from the verified display names, sockets and explanatory caption")
        rows.append(dict(id=beat["id"], evidence=str(evidence_path), image_matches_definitions=matches,
                         display_names=[s["display_name"] for s in specs.values()],
                         caption=evidence.get("caption"), checked_edges=edges))
    if not rows:
        errors.append("No verified own node close-up found")
    return dict(cards=rows), errors


def silence_check(audio: np.ndarray, threshold_dbfs: float = -60.0) -> dict:
    """Find sustained practical and digital silence at ten-millisecond resolution."""
    hop = round(.01 * SR)
    audio = np.asarray(audio, dtype=np.float64)
    rms = [math.sqrt(float(np.mean(audio[i:i + hop] ** 2))) for i in range(0, audio.size, hop)]

    def events(threshold):
        mask = [value < 10 ** (threshold / 20) for value in rms]
        return [dict(start=round(a * hop / SR, 6), end=round(min(b * hop, audio.size) / SR, 6),
                     duration=round((min(b * hop, audio.size) - a * hop) / SR, 6))
                for a, b in ranges(mask)]

    quiet, digital = events(threshold_dbfs), events(-100)
    return dict(block_seconds=hop / SR, practical_silence_threshold_dbfs=threshold_dbfs,
                maximum_allowed_seconds=1.0, longest_practical_silence_seconds=max((r["duration"] for r in quiet), default=0),
                longest_digital_silence_seconds=max((r["duration"] for r in digital), default=0),
                practical_silence_over_one_second=[r for r in quiet if r["duration"] > 1],
                digital_silence_over_one_second=[r for r in digital if r["duration"] > 1])


def source_picture_check(video: Path, schedule: list[dict]) -> tuple[dict, list[str]]:
    """Reconstruct source pictures without replaying obsolete v14 effects."""
    from build_demo import render_beat

    rows, errors = [], []
    # This region excludes spoken captions, bottom waveforms and top labels.
    # The Studio microphone view has its waveform above this region as well.
    bounds = (round(.28 * W), round(.32 * H), round(.72 * W), round(.68 * H))
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for entry in schedule:
            beat = entry["beat"]
            if beat["type"] not in {"clip", "image"}:
                continue
            try:
                comparison_bounds = (0, 0, W, 840) if beat["id"] == "comfy_cuesync" else bounds
                left, top, right, bottom = comparison_bounds
                sample = dict(beat, dur=1 / FPS)
                references, _ = render_beat(sample)
                reference = references[0][top:bottom, left:right].astype(np.float32)
                container.seek(int(entry["start"] / float(stream.time_base)), stream=stream, backward=True)
                actual = None
                for frame in container.decode(stream):
                    if frame.time is not None and float(frame.time) >= entry["start"] - .001:
                        actual = frame.to_ndarray(format="rgb24")[top:bottom, left:right].astype(np.float32)
                        break
                if actual is None or actual.shape != reference.shape:
                    errors.append(f"Cannot decode source-picture sample: {entry['id']}")
                    continue
                absolute = np.abs(actual - reference)
                mean, mse = float(np.mean(absolute)), float(np.mean(absolute ** 2))
                psnr = 10 * math.log10(255 ** 2 / mse) if mse else None
                row = dict(id=entry["id"], time=entry["start"], source=beat["src"],
                           source_from=float(beat.get("start", 0)), mean_absolute_error=round(mean, 6),
                           psnr_db=round(psnr, 5) if psnr is not None else None, bounds=comparison_bounds)
                rows.append(row)
                if mean > 4 or (psnr is not None and psnr < 30):
                    errors.append(f"Decoded picture does not match its current source: {entry['id']}")
            except (OSError, ValueError, IndexError, KeyError) as exc:
                errors.append(f"Source-picture sample failed: {entry['id']} ({exc})")
            print(f"Checked source picture: {entry['id']}", flush=True)
    return dict(method="First frame of each clip/card reconstructed with the current renderer, compared with decoded video above captions and outside waveform regions",
                comparison_bounds=bounds, maximum_mean_absolute_error=4, minimum_psnr_db=30,
                checked_beats=rows), errors


def chip_header_layout_check(video: Path, schedule: list[dict], out: Path,
                             identity: str) -> tuple[dict, list[str]]:
    """Decode every protected UI frame and verify its title band has no source pixels."""
    from build_demo import render_beat
    from PIL import Image

    entry = next(e for e in schedule if e["id"] == identity)
    beat = entry["beat"]
    references, _ = render_beat(dict(beat, dur=1 / FPS))
    reference = references[0][:170].astype(np.float32)
    compare_mask = np.ones(reference.shape[:2], dtype=bool)
    excluded_wave_box = None
    if beat.get("wave") and beat.get("wave_box"):
        bx, by, bw, bh = beat["wave_box"]
        # The live voice envelope belongs in the header. Exclude just its pixels
        # (plus codec edges), while checking the chip and all remaining space.
        x1, y1 = max(0, int(bx * W) - 6), max(0, int(by * H) - 6)
        x2, y2 = min(W, int((bx + bw) * W) + 7), min(170, int((by + bh) * H) + 7)
        compare_mask[y1:y2, x1:x2] = False
        excluded_wave_box = [x1, y1, x2, y2]
    errors, differences, samples = [], [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        container.seek(int(entry["start"] / float(stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            at = float(frame.time or 0)
            if at < entry["start"] - .001:
                continue
            if at >= entry["end"] - .001:
                break
            image = frame.to_ndarray(format="rgb24")
            differences.append(float(np.abs(image[:170].astype(np.float32) - reference)[compare_mask].mean()))
            for target in {"upload": (76., 77., 78.), "perform": (84., 86., 87.), "make": (88., 89.)}[identity]:
                if abs(at - target) < .001:
                    path = out / f"{identity}-{int(target)}s.jpg"
                    Image.fromarray(image).save(path, quality=95)
                    samples.append(str(path))
    if len(differences) != entry["frames"]:
        errors.append(f"{identity} layout scan did not decode every frame")
    maximum = max(differences, default=999.)
    if maximum > 3:
        errors.append(f"Recorded Studio pixels overlap or alter the reserved {identity} title band")
    return dict(decoded_frames=len(differences), title_band_y=[0, 170],
                excluded_animated_waveform_bounds=excluded_wave_box,
                maximum_mean_absolute_error=maximum, threshold=3, samples=samples), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--storyboard", type=Path, default=Path(__file__).with_name("storyboard.json"))
    parser.add_argument("--baseline", "--base-storyboard", type=Path)
    parser.add_argument("--output", type=Path, help="Report and contact-sheet directory; defaults to the video's directory")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    board = read_json(args.storyboard)
    version = board.get("version", 15) if isinstance(board, dict) else 15
    args.video = args.video or ROOT / f"runs/demo_v{version}/demo_v{version}.mp4"
    out = args.output or args.video.parent
    out.mkdir(parents=True, exist_ok=True)
    args.baseline = args.baseline or ROOT / f"runs/demo_v{15 if version == 16 else 14}/storyboard.json"
    baseline = read_json(args.baseline)
    schedule = timeline(beats_of(board))
    total = schedule[-1]["end"]
    structure, errors = structure_check(board, baseline)
    provenance, source_errors = source_provenance_check(board)
    errors.extend(source_errors)
    narrations, narration_errors = narration_check(schedule, total, out)
    errors.extend(narration_errors)
    voices, voice_errors = voice_provenance_check(schedule)
    errors.extend(voice_errors)
    nodes, node_errors = node_evidence_check(beats_of(board))
    errors.extend(node_errors)
    windows, window_errors = source_bounds_check(schedule)
    errors.extend(window_errors)
    report = dict(video=str(args.video), storyboard=str(args.storyboard), baseline=str(args.baseline),
                  metadata_only=args.metadata_only, structure=structure, source_provenance=provenance,
                  narration=narrations, narration_provenance=voices, source_windows=windows,
                  own_nodes=nodes,
                  errors=errors, review=[],
                  limitations="Waveform presence, word timing, caption pixels and levels require a companion listening and visual review for intelligibility and edit quality.")
    if not args.metadata_only:
        if version >= 16:
            build_path = args.video.with_suffix(".build.json")
            if not build_path.is_file():
                errors.append(f"Missing final encode source manifest: {build_path}")
            else:
                from build_demo import build_source_hashes
                build = read_json(build_path)
                report["encode_manifest"] = dict(path=str(build_path), **build)
                if build.get("sources_sha256") != build_source_hashes(board, args.storyboard):
                    errors.append("Media, storyboard or renderer changed after final encoding")
                if build.get("video_sha256") != sha256(args.video):
                    errors.append("Final video differs from its encoding manifest")
                if build.get("frames") != structure["expected_frames"]:
                    errors.append("Encoding manifest frame count differs from storyboard")
        picture = scan_video(args.video, schedule, narrations, out)
        report["picture"] = picture
        if picture["decode_error"]:
            errors.append(f"Video decode failed: {picture['decode_error']}")
        if (picture["width"], picture["height"]) != (W, H):
            errors.append(f"Expected 1920x1080, got {picture['width']}x{picture['height']}")
        if abs(picture["nominal_fps"] - FPS) > 1e-6:
            errors.append(f"Unexpected frame rate: {picture['nominal_fps']}")
        if picture["decoded_frames"] != structure["expected_frames"]:
            errors.append(f"Decoded frame count differs: {picture['decoded_frames']} != {structure['expected_frames']}")
        if picture["decoded_duration"] > 118:
            errors.append(f"Decoded picture exceeds 118 seconds: {picture['decoded_duration']}")
        if picture["pts_delta_min"] is not None and (abs(picture["pts_delta_min"] - 1 / FPS) > 1e-5
                                                     or abs(picture["pts_delta_max"] - 1 / FPS) > 1e-5):
            errors.append("Nonuniform frame presentation timestamps")
        for row in picture["black_events"] + picture["freezes"]:
            if row["kind"] == "full_black":
                errors.append(f"Whole black frames at {row['start']:.3f}s to {row['end']:.3f}s ({row['beat']})")
            elif row["classification"] == "review":
                report["review"].append(row)
        source_pictures, source_picture_errors = source_picture_check(args.video, schedule)
        report["source_picture_comparison"] = source_pictures
        errors.extend(source_picture_errors)
        if version >= 16:
            for identity in ("upload", "perform", "make"):
                layout, layout_errors = chip_header_layout_check(args.video, schedule, out, identity)
                report[f"{identity}_layout"] = layout
                errors.extend(layout_errors)
        audio = read_audio(args.video, SR)
        sound = dict(duration=audio.size / SR, levels=levels(audio), silence=silence_check(audio), beats=[])
        report["audio"] = sound
        if abs(audio.size / SR - total) > .08:
            errors.append(f"Audio/picture duration mismatch: {audio.size / SR:.6f}s vs {total:.6f}s")
        for quiet in sound["silence"]["practical_silence_over_one_second"]:
            errors.append(f"Silence over one second: {quiet['start']:.3f}s to {quiet['end']:.3f}s")
        for entry in schedule:
            segment = audio[round(entry["start"] * SR):round(entry["end"] * SR)]
            sound["beats"].append(dict(id=entry["id"], start=entry["start"], end=entry["end"], **levels(segment)))
        for row in narrations:
            beat = next(e["beat"] for e in schedule if e["id"] == row["id"])
            voice = read_audio(row["source"], SR) * float(beat.get("vo_gain", 1))
            presence = voice_presence(audio, voice, row["start"])
            row["waveform_presence"] = presence
            for caption in row["captions"]:
                if caption.get("decoded_caption_white_glyph_coverage", 0) < .9:
                    errors.append(f"Decoded caption coverage below 90%: {caption['id']}")
            if presence.get("estimated_narration_scale", 0) < .8 or abs(presence.get("alignment_offset_seconds", 1)) > .04:
                errors.append(f"Full narration waveform is weak, absent, or misaligned: {row['id']}")
            fitted = voice * float(presence.get("estimated_narration_scale", 0))
            active = np.abs(voice) > max(.001, float(np.max(np.abs(voice))) * .05)
            row["estimated_active_voice_levels"] = levels(fitted[active])
            if row["estimated_active_voice_levels"]["rms_dbfs"] < -36:
                errors.append(f"Narration active level is below -36 dBFS: {row['id']}")
            if presence.get("narration_to_residual_db", -240) < -6:
                report["review"].append(dict(kind="narration_audibility", id=row["id"], **presence))
            if row["id"] == "innovation":
                tail_seconds = min(1.5, voice.size / SR)
                tail = voice[-round(tail_seconds * SR):]
                row["final_word_region_waveform_presence"] = voice_presence(audio, tail, row["end"] - tail.size / SR)
                if row["final_word_region_waveform_presence"].get("estimated_narration_scale", 0) < .8:
                    errors.append("Contribution final word region is weak or truncated in output")
        report["author_voice"] = author_voice_check(schedule, narrations, audio)
        for row in report["author_voice"]:
            if row["waveform_presence"].get("estimated_author_voice_scale", 0) < .5:
                errors.append(f"Author voice waveform weak or absent in mix: {row['id']}")
        if sound["levels"]["near_full_scale_samples"]:
            report["review"].append(dict(kind="audio_near_full_scale", **sound["levels"]))
    report["status"] = "errors" if errors else "metadata_passed" if args.metadata_only else "measured_checks_passed"
    destination = out / (f"qa_v{version}_metadata.json" if args.metadata_only else f"qa_v{version}.json")
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {total:.3f}s, {structure['expected_frames']} expected frames")
    print(f"{len(errors)} errors; {len(report['review'])} review items -> {destination}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
