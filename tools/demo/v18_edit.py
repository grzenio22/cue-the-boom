"""Editorial constraints for the proof-first v18 cut; never generate new effects."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from check_v11 import timeline


STUDIO_IDS = ("try_sample", "perform", "make", "download", "result")
REPLACED_IDS = {"cold_open", "intro_raw", "intro_finished", "upload", "tap", "perform", "make", "result"}
PROOF_STAMP = "Author's recorded voice, edited rhythms · same footage + one LTX element"
HERO_STAMP = "Author's hand · stored LTX fire + CPU compositing and lighting · slow motion"


def capture_provenance_check(board: dict) -> tuple[dict, list[str]]:
    """Bind the new UI actions, uploaded PCM and downloaded result to one actual job."""
    errors = []
    binding = board.get("browser_capture", {})
    path = Path(binding.get("manifest", ""))
    if not path.is_file():
        return {}, ["Fresh Studio capture manifest is missing"]
    def digest(source):
        return hashlib.sha256(source.read_bytes()).hexdigest()
    if digest(path) != binding.get("manifest_sha256"):
        errors.append("Fresh Studio capture manifest changed after editorial selection")
    capture = json.loads(path.read_text(encoding="utf-8"))
    assets = capture.get("assets", {})
    checked = []
    for name, asset in assets.items():
        source = Path(asset["path"])
        good = source.is_file() and digest(source) == asset.get("sha256")
        checked.append({"name": name, "path": str(source), "matches": good})
        if not good:
            errors.append(f"Fresh Studio capture asset changed: {name}")
    required = {"raw_screen_video", "recorded_microphone", "actual_final", "downloaded_final", "receipt", "browser_evidence"}
    if not required.issubset(assets):
        errors.append("Fresh Studio capture lacks required screen, microphone, output or request evidence")
        return {"assets": checked}, errors
    if (capture.get("passed") is not True or capture.get("new_model_generation") is not False
            or capture.get("request", {}).get("fields", {}).get("performance_mode") != "reuse"):
        errors.append("Fresh Studio job must document a successful forced-reuse request without model generation")
    if assets["downloaded_final"]["sha256"] != assets["actual_final"]["sha256"]:
        errors.append("Downloaded Studio file differs from the actual backend result")
    by_id = {beat["id"]: beat for beat in board["beats"]}
    for identity in STUDIO_IDS:
        beat = by_id.get(identity, {})
        source = assets["actual_final" if identity == "result" else "raw_screen_video"]["path"]
        if beat.get("src") != source:
            errors.append(f"Fresh Studio beat uses a source outside the verified capture: {identity}")
    recording = by_id.get("perform", {})
    if recording.get("audio_src") != assets["recorded_microphone"]["path"]:
        errors.append("Fresh Studio recording must play the actual uploaded microphone PCM")
    expected_offset = float(capture.get("pcm_first_sample_page_relative_s", 0)) - float(recording.get("start", 0))
    if (recording.get("audio_from", 0) != 0
            or abs(float(recording.get("audio_at", 0)) - expected_offset) > 1 / 24
            or float(recording.get("dur", 0)) - float(recording.get("audio_at", 0))
            < float(capture.get("recorded_voice", {}).get("duration", 0)) - 1 / 24):
        errors.append("Fresh Studio microphone sound must retain the whole recording and its documented clock alignment")
    actual_wait = float(capture.get("actual_cpu_composition_seconds", 0))
    if actual_wait <= 0 or f"{actual_wait:.1f}s" not in by_id.get("make", {}).get("chip", ""):
        errors.append("Shortened render wait must show its measured actual CPU duration")
    if "wait shortened" not in by_id.get("make", {}).get("chip", "").lower():
        errors.append("The demo must disclose the shortened render wait")
    return dict(manifest=str(path), job_id=capture.get("job_id"), assets=checked,
                actual_cpu_composition_seconds=actual_wait,
                capture_disclosure=capture.get("capture_disclosure"),
                screen_audio_alignment_limitation=capture.get("stage_clock")), errors


def hero_beat(baseline: dict) -> dict:
    """Use an already-active interval of the final dark-palm composite."""
    old = {beat["id"]: beat for beat in baseline["beats"]}
    hero = deepcopy(old["intro_finished"])
    hero.update(id="hero_open", start=2.8, dur=4.75, speed=.125,
                subline="Generate once. Perform again.", stamp=HERO_STAMP)
    for key, value in old["intro_raw"].items():
        if key == "vo" or key.startswith("vo_"):
            hero[key] = deepcopy(value)
    hero["vo_at"] = .15
    return hero


def editorial_structure_check(board: dict, baseline: dict) -> tuple[dict, list[str]]:
    """Protect complete performances and narration while permitting the new UI footage."""
    errors = []
    beats = board["beats"]
    old = {beat["id"]: beat for beat in baseline["beats"]}
    ids = [beat["id"] for beat in beats]
    tail = [beat["id"] for beat in baseline["beats"]
            if beat["id"] not in REPLACED_IDS | {"performance_turns", "small_and_big"}]
    expected = ["hero_open", "performance_turns", "small_and_big", *STUDIO_IDS, *tail]
    if ids != expected:
        errors.append("The v18 cut must open with the hero and complete proof, followed by the fresh short flow")
    for beat in beats:
        identity = beat["id"]
        if identity == "hero_open":
            if beat != hero_beat(baseline):
                errors.append("The v18 hero differs from the approved active dark-palm source, timing or narration")
        elif identity in STUDIO_IDS:
            if identity != "result":
                stamp = str(beat.get("stamp", "")).lower()
                if "virtual microphone" not in stamp or "recorded voice" not in stamp:
                    errors.append(f"The fresh Studio capture must disclose virtual microphone and recorded voice: {identity}")
                if not beat.get("chip_header"):
                    errors.append(f"The fresh Studio capture must reserve a clear label header: {identity}")
            if beat.get("speed", 1) != 1:
                errors.append(f"Studio action speed must remain real time: {identity}")
        elif identity in old:
            expected_beat = deepcopy(old[identity])
            if identity in {"performance_turns", "small_and_big"}:
                expected_beat["stamp"] = PROOF_STAMP
            if beat != expected_beat:
                errors.append(f"Preserved performance, source, narration or editorial content changed: {identity}")
    schedule = timeline(beats)
    duration = schedule[-1]["end"] if schedule else 0
    proof = next((entry for entry in schedule if entry["id"] == "performance_turns"), None)
    if proof is None or proof["end"] > 15:
        errors.append("All three complete performances must finish within the opening 15 seconds")
    studio = next((entry for entry in schedule if entry["id"] == "try_sample"), None)
    if studio is None or studio["start"] > 20:
        errors.append("The actual short-flow demonstration must begin within 20 seconds")
    if not 0 < duration < 120:
        errors.append("The demo must remain shorter than 120 seconds")
    if abs(float(board.get("duration", 0)) - duration) > .002:
        errors.append("Storyboard duration does not match frame-rounded beats")
    return dict(duration=duration, baseline_duration=baseline["duration"],
                expected_frames=sum(entry["frames"] for entry in schedule),
                proof_complete_seconds=proof["end"] if proof else None,
                studio_start_seconds=studio["start"] if studio else None,
                timeline=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in schedule]), errors
