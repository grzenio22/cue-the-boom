"""Demo v19: the proof-first v18 cut on the approved v11 fire renders, plus four strangers' voices and air.

    python tools/demo/v19_build.py prepare   # runs/demo_v19/storyboard.json from the v18 storyboard
    python tools/demo/v19_build.py build     # runs/demo_v19/demo_v19.mp4 and its .build.json
    python tools/demo/v19_build.py post      # FFmpeg loudness/silence record and the demo regression tests
    python tools/demo/check_v15.py --video runs/demo_v19/demo_v19.mp4 --storyboard runs/demo_v19/storyboard.json

The v18 beats keep their windows, speeds, captions, narration and audio. Every runs/demo_v17/clips
source becomes the runs/demo_v11/clips render of the same name (the renders demo v15 used). The
composites share their strike frames with v17, so their windows stay; the workflow close-up returns
to v15's window for the v11 element, the window its Foley was resampled from. The recap animation is
v15's, built from the v11 pictures with the same stages and timing. Two existing renders are
inserted: the palm render performed again by four strangers' recorded voices (CC0) and the stored
air element held in the palm. The five Studio beats use a new browser capture made with the current
code (runs/contest-polish-20261026/browser), at the same visible UI moments and durations as v18.
No model generation, GPU service or narration generation is used.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from check_v11 import timeline

FPS = 24
ROOT = Path("D:/openHiggsfield")
RUN = ROOT / "runs/demo_v19"
SCRATCH = Path("D:/openHiggsfield_tmp/demo19")
BASELINE = ROOT / "runs/demo_v18/storyboard.json"
OLD_CLIPS = ROOT / "runs/demo_v17/clips"
NEW_CLIPS = ROOT / "runs/demo_v11/clips"
MANIFEST = NEW_CLIPS / "manifest.json"
SWAPPED = ("proof_boom/composite.mp4", "proof_pf_pf_boom/composite.mp4", "proof_pf_x4/composite.mp4",
           "how_raw.mp4", "how_carried.mp4", "how_light_off/composite.mp4",
           "flame_walk/composite.mp4", "flamethrower/composite.mp4")
OLD_RECAP = ROOT / "runs/demo_v17/cards/pipeline_motion.mp4"
NEW_RECAP = ROOT / "runs/demo_v15/cards/pipeline_motion.mp4"
RECAP_QA = ROOT / "runs/demo_v15/qa_v15.json"
RECAP_IDS = ("innovation", "innovation_asset", "innovation_track", "innovation_voice", "innovation_result")
# v18 sampled the v17 derivative from 2.65 s, where its fireball appears (black before 2.625 s).
# The v11 element's fireball appears at 2.417 s; v15 showed it from 2.4 s at 0.45x, and
# workflow_fire.wav was resampled from that window.
WINDOWS = {"workflow_fire": {"start": 2.4, "speed": .45}}
# The v11 element is the LTX output re-timed over black, not v17's CPU temporal derivative.
LABELS = {"how": {0: "LTX-2.5 over black"}}

BENDING = ROOT / "samples/made_ahead/bending"
PROOF_ELEMENT_DIR = "samples/made_ahead/bending/hand_palm__boom__fire_burst"
STRANGERS = (("sadiquecat", "Boom"), ("unfa", "Ka-Boom!"), ("huw2k8", "Powf"), ("jedimichael", "Boom"))
STRANGERS_HEADLINE = "The same render · 4 strangers' voices · 0 new generations"
STRANGERS_STAMP = "One LTX-2.5 render · voices: Sadiquecat, unfa, huw2k8, Jedimichael (CC0, Freesound)"
TURN_START_FRAME, TURN_FRAMES = 34, 50          # 1.417-3.5 s of each 3.5 s take: every voice and its whole flight
# The same picture rows as the author's panels ([0, .18, 1, .96] of a 1920-row composite);
# the export's 64-row footer below row 1920 is left out.
FINAL_CROP = [0, .1734, 1, .9294]
STRANGERS_GAIN = .4                              # -14.7 LUFS takes -> about -22.6 LUFS, the author's turns
AIR_SRC = BENDING / "hand_palm__hoooo__air_vortex/final.mp4"
AIR_START, AIR_FRAMES = .75, 66                  # palm, then the hoooo at 1.3 s and the vortex from 1.4 s to the end
AIR_CROP = [0, 0, 1, .96775]                     # rows 0-1920: the composite without the export footer
AIR_GAIN = .27                                   # -12.4 LUFS take -> about -23.8 LUFS, the montage level
AIR_CHIP = "hoooo · sample noise (synthetic)"
AIR_STAMP = "Air: a whirlwind of mist rendered once by LTX-2.5, held with your voice"
# The Studio flow, captured again on 26 October with the current code (same Playwright, Chromium,
# quick-flow script and virtual microphone file as the 23 October capture it replaces).
OLD_CAPTURE = ROOT / "runs/contest-polish-20261023/browser/capture-manifest.json"
CAPTURE = ROOT / "runs/contest-polish-20261026/browser/capture-manifest.json"
STUDIO_IDS = ("try_sample", "perform", "make", "download", "result")
# Window starts on the new screen recording, placed at v18's distances from the same visible UI changes
# (measured frame differences in the UI column, old -> new): CTA to record view 3.00 -> 2.16 s,
# microphone armed 4.40 -> 3.56 s, composition view 8.96 -> 8.04 s, result view settled 27.72 -> 41.52 s.
# make starts where perform ends, as in v18; durations are v18's.
STUDIO_STARTS = {"try_sample": 1.96, "perform": 3.46, "download": 41.55}
METADATA_INPUTS = {"receipt.json", "samples.json"}
MEDIA_FIELDS = ("src", "audio_src", "vo", "element_src")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def swap_path(value: str) -> str:
    path = Path(value)
    for name in SWAPPED:
        if path == OLD_CLIPS / name:
            return str(NEW_CLIPS / name)
    return str(NEW_RECAP) if path == OLD_RECAP else value


def migrate(beat: dict) -> dict:
    """The v18 beat with only the approved source swap, window and label corrections."""
    def walk(value):
        if isinstance(value, dict):
            return {k: (swap_path(v) if k in ("src", "element_src") and isinstance(v, str)
                        else [swap_path(p) for p in v] if k == "srcs" else walk(v))
                    for k, v in value.items()}
        return [walk(v) for v in value] if isinstance(value, list) else value
    result = walk(deepcopy(beat))
    result.update(WINDOWS.get(beat["id"], {}))
    for index, label in LABELS.get(beat["id"], {}).items():
        result["labels"][index] = label
    return result


def capture() -> dict:
    """The new capture manifest, plus the page time its exported shot started playing in the UI preview."""
    data = read_json(CAPTURE)
    evidence = read_json(Path(data["assets"]["browser_evidence"]["path"]))["results"][0]
    first = evidence["stages"][0]
    opened = first["epochMs"] - first["relativeToPageOpenS"] * 1000
    final = f"/api/jobs/{data['job_id']}/files/final.mp4"
    play = next(s for s in evidence["capture"]["stages"]
                if s["name"] == "video-play" and str(s.get("src", "")).endswith(final))
    data["preview_play_page_relative_s"] = (evidence["capture"]["timeOriginMs"] + play["at"] - opened) / 1000
    return data


def studio_beat(beat: dict, cap: dict) -> dict:
    """v18's Studio beat on the new capture: its crop, chip, stamp, duration and audio rules, new sources and windows.

    As in v18, the uploaded microphone PCM and the exported sound are placed by the browser page clock
    (first PCM sample; UI preview play event); screen-clock correspondence stays approximate."""
    b, assets = deepcopy(beat), cap["assets"]
    if b["id"] == "result":
        b["src"] = assets["actual_final"]["path"]
        return b
    b["src"] = assets["raw_screen_video"]["path"]
    b["start"] = STUDIO_STARTS["perform"] + 122 / FPS if b["id"] == "make" else STUDIO_STARTS[b["id"]]
    if b["id"] == "make":
        b["chip"] = f"{cap['actual_cpu_composition_seconds']:.1f}s CPU composition · wait shortened"
    elif b["id"] == "perform":
        b.update(audio_src=assets["recorded_microphone"]["path"],
                 audio_at=cap["pcm_first_sample_page_relative_s"] - b["start"])
    elif b["id"] == "download":
        b.update(audio_src=assets["actual_final"]["path"],
                 audio_at=round(cap["preview_play_page_relative_s"] - b["start"], 3))
    return b


def expected_beat(beat: dict, cap: dict) -> dict:
    """The v19 form of a v18 beat."""
    return studio_beat(migrate(beat), cap) if beat["id"] in STUDIO_IDS else migrate(beat)


def strangers_beat() -> dict:
    panels = [dict(src=str(BENDING / f"hand_palm__{name}__again/final.mp4"), start=TURN_START_FRAME / FPS,
                   dur=TURN_FRAMES / FPS, crop=list(FINAL_CROP), label=label) for name, label in STRANGERS]
    return dict(id="strangers", type="turns", dur=len(panels) * TURN_FRAMES / FPS, audio_gain=STRANGERS_GAIN,
                panels=panels, headline=STRANGERS_HEADLINE, element_src=str(NEW_CLIPS / "how_raw.mp4"),
                element_at=2.92, stamp=STRANGERS_STAMP)


def air_beat() -> dict:
    return dict(id="air", type="clip", src=str(AIR_SRC), start=AIR_START, dur=AIR_FRAMES / FPS, crop=list(AIR_CROP),
                fill=True, portrait_backdrop=True, source_gain=AIR_GAIN, chip=AIR_CHIP, stamp=AIR_STAMP)


def expected_ids(baseline: dict) -> list[str]:
    ids = []
    for beat in baseline["beats"]:
        ids.append(beat["id"])
        if beat["id"] == "small_and_big":
            ids.append("strangers")
        elif beat["id"] == "flamethrower":
            ids.append("air")
    return ids


def media_paths(value) -> set[Path]:
    paths = set()
    if isinstance(value, dict):
        for name, child in value.items():
            if name in MEDIA_FIELDS and isinstance(child, str):
                paths.add(Path(child))
            elif name == "srcs":
                paths.update(Path(p) for p in child)
            else:
                paths |= media_paths(child)
    elif isinstance(value, list):
        for child in value:
            paths |= media_paths(child)
    return paths


def video_duration(path: Path) -> float:
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base)


def prepare() -> dict:
    baseline = read_json(BASELINE)
    if baseline.get("version") != 18:
        raise ValueError("The v19 baseline must be the v18 storyboard")
    cap = capture()
    beats = []
    for beat in baseline["beats"]:
        beats.append(expected_beat(beat, cap))
        if beat["id"] == "small_and_big":
            beats.append(strangers_beat())
        elif beat["id"] == "flamethrower":
            beats.append(air_beat())
    schedule = timeline(beats)
    swaps = []
    for name in SWAPPED + ("pipeline_motion.mp4",):
        old, new = (OLD_RECAP, NEW_RECAP) if name == "pipeline_motion.mp4" else (OLD_CLIPS / name, NEW_CLIPS / name)
        used = sorted({b["id"] for b in baseline["beats"] if old in media_paths(b)})
        swaps.append(dict(old=str(old), new=str(new), new_sha256=sha256(new), beats=used))
    takes = []
    for name, label in STRANGERS:
        folder = BENDING / f"hand_palm__{name}__again"
        receipt = read_json(folder / "receipt.json")
        takes.append(dict(voice=name, label=label, src=str(folder / "final.mp4"), sha256=sha256(folder / "final.mp4"),
                          performed_again=receipt["bending"]["performed_again"]))
    air_receipt = read_json(AIR_SRC.parent / "receipt.json")
    board = deepcopy(baseline)
    board.update(
        version=19, beats=beats, duration=schedule[-1]["end"],
        schedule=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in schedule],
        baseline=str(BASELINE), baseline_sha256=sha256(BASELINE), provenance=str(MANIFEST),
        source_policy=("Editorial reuse of v18: each v17 fire render replaced by the approved v11 render of the same beat "
                       "(as used by demo v15), v15's recap animation, and two inserted existing renders. No new model "
                       "generation, GPU service or narration generation. Preserved bytes are not claimed to be rebuilt."),
        source_reuse=dict(manifest=str(MANIFEST), manifest_sha256=sha256(MANIFEST), new_model_generations=0,
                          composites_rebuilt_for_this_edit=False),
        source_swaps=swaps,
        window_adjustments=[dict(id="workflow_fire", v18=dict(start=2.65, speed=.4), v19=WINDOWS["workflow_fire"],
                                 reason="The fireball first appears at 2.625 s in the v17 derivative (black before) and at "
                                        "2.417 s in the v11 element, which grows to its peak near 2.8 s. v15's window for "
                                        "this clip shows that appearance; workflow_fire.wav was resampled from it.")],
        label_adjustments=[dict(id="how", panel=1, v18="LTX fire · CPU derivative", v19=LABELS["how"][0],
                                reason="The v11 element is the LTX output re-timed over black (v15's label), "
                                       "not v17's CPU temporal derivative.")],
        recap=dict(src=str(NEW_RECAP), sha256=sha256(NEW_RECAP), replaces=str(OLD_RECAP), verified_in=str(RECAP_QA),
                   note="v15's recap, made with v12_cards.pipeline and v14_contribution.recap from the v11 pictures; "
                        "the same five stage windows and narration as v18."),
        inserted_beats=dict(
            strangers=dict(takes=takes, credits="samples/CREDITS.md (CC0 1.0, Freesound)",
                           window_seconds=[TURN_START_FRAME / FPS, 3.5], audio_gain=STRANGERS_GAIN),
            air=dict(src=str(AIR_SRC), sha256=sha256(AIR_SRC), element_source=air_receipt["bending"]["element_source"],
                     render_prompt=air_receipt["render"]["prompt"], window_seconds=[AIR_START, AIR_START + AIR_FRAMES / FPS],
                     voice="synthetic sample noise voices/hoooo.wav (samples/CREDITS.md)", source_gain=AIR_GAIN)),
        browser_capture=dict(manifest=str(CAPTURE), manifest_sha256=sha256(CAPTURE), job_id=cap["job_id"],
                             replaces=dict(manifest=str(OLD_CAPTURE), job_id=read_json(OLD_CAPTURE)["job_id"],
                                           reason="Its exported shot was composed by the 23 October fire code (a "
                                                  "standing flame column), since reverted."),
                             audio_alignment=baseline["browser_capture"]["audio_alignment"],
                             shortened_wait=True, actual_cpu_composition_seconds=cap["actual_cpu_composition_seconds"],
                             preview_play_page_relative_s=cap["preview_play_page_relative_s"],
                             windows=dict(starts={**STUDIO_STARTS, "make": STUDIO_STARTS["perform"] + 122 / FPS},
                                          method="Placed at v18's distances from the same visible UI changes on the "
                                                 "new screen recording; v18 durations, crops, chips and stamps.",
                                          visible_events_old_to_new_seconds={
                                              "record_view": [3.0, 2.16], "microphone_armed": [4.4, 3.56],
                                              "recording": [5.12, 4.32], "composition_view": [8.96, 8.04],
                                              "result_view_settled": [27.72, 41.52], "perform_again": [32.92, 47.12]})),
        standalone_source_window_note=("The opening starts on the active v11 dark-palm composite at 2.8 s in disclosed "
                                       "slow motion. Proof performances, narration and the remaining source windows are "
                                       "v18's; v11 composites share v17's strike frames. The Studio beats are a new "
                                       "capture with the current code, in real time within each segment; the "
                                       "composition wait is shortened and its measured duration displayed."))
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "storyboard.json").write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return board


def structure_check(board: dict, baseline: dict) -> tuple[dict, list[str]]:
    """Only the approved source swaps and the two inserted beats may differ from v18."""
    errors = []
    beats = board["beats"]
    ids = [b["id"] for b in beats]
    if baseline.get("version") != 18:
        errors.append("The v19 baseline must be the v18 storyboard")
    if ids != expected_ids(baseline):
        errors.append("v19 must be v18's beat order with strangers after small_and_big and air after flamethrower")
    old = {b["id"]: b for b in baseline["beats"]}
    cap = capture()
    if board.get("browser_capture", {}).get("manifest") != str(CAPTURE):
        errors.append("v19 must use the 26 October Studio capture")
    changed = []
    for beat in beats:
        identity = beat["id"]
        if identity in old:
            if beat != expected_beat(old[identity], cap):
                errors.append(f"Preserved v18 beat changed beyond the approved source swaps: {identity}")
            if beat != old[identity]:
                changed.append(dict(id=identity, fields=sorted(k for k in set(beat) | set(old[identity])
                                                              if beat.get(k) != old[identity].get(k))))
    by_id = {b["id"]: b for b in beats}
    strangers, air = by_id.get("strangers"), by_id.get("air")
    if strangers != strangers_beat():
        errors.append("The strangers beat differs from its approved sources, windows, gain or text")
    if strangers:
        panels = strangers.get("panels", [])
        if [q.get("label") for q in panels] != [label for _, label in STRANGERS]:
            errors.append("Strangers labels must read Boom, Ka-Boom!, Powf, Boom")
        if strangers.get("headline") != STRANGERS_HEADLINE or strangers.get("stamp") != STRANGERS_STAMP:
            errors.append("Strangers headline or stamp differs from the approved text")
        for q in panels:
            if q.get("audio_src") or not Path(q["src"]).is_file():
                errors.append(f"Each stranger must play its own final.mp4 with its own sound: {q.get('src')}")
            elif q["start"] + q["dur"] > video_duration(Path(q["src"])) + 1e-6:
                errors.append(f"Stranger window runs past its take: {q['src']}")
        seconds = round(float(strangers["dur"]) * FPS) / FPS
        if not 8 <= seconds <= 9.5:
            errors.append(f"Strangers beat must last about 8-9 seconds, found {seconds:.3f}")
    if air != air_beat():
        errors.append("The air beat differs from its approved source, window, gain or text")
    if air:
        end = float(air["start"]) + float(air["dur"])
        if air.get("stamp") != AIR_STAMP or not 2.5 <= float(air["dur"]) <= 3.5 or float(air.get("source_gain", 0)) <= 0:
            errors.append("Air must carry its stamp and own sound for about three seconds")
        if not (Path(air["src"]).is_file() and float(air["start"]) <= 1.4 and end <= video_duration(Path(air["src"])) + 1e-6):
            errors.append("Air must show the vortex from 1.4 s to the end of its take")
    stale = sorted(str(p) for p in media_paths(beats) if "demo_v17" in str(p) or "contest-polish-20261023" in str(p))
    if stale:
        errors.append(f"Replaced v17 or 23 October capture media remain in the cut: {', '.join(stale)}")
    schedule = timeline(beats)
    duration = schedule[-1]["end"] if schedule else 0
    at = {e["id"]: e for e in schedule}
    proof = at.get("performance_turns")
    if proof is None or proof["end"] > 15:
        errors.append("All three complete performances must finish within the opening 15 seconds")
    if duration > 118 + 1e-6:
        errors.append(f"Picture exceeds 118 seconds: {duration:.6f}s")
    if abs(float(board.get("duration", 0)) - duration) > .002:
        errors.append("Storyboard duration does not match frame-rounded beats")
    before = {e["id"]: e for e in timeline(baseline["beats"])}
    shifts = [dict(id=e["id"], v18_start=before[e["id"]]["start"], v19_start=e["start"],
                   shift=round(e["start"] - before[e["id"]]["start"], 6)) for e in schedule if e["id"] in before]
    return dict(duration=duration, baseline_duration=baseline["duration"],
                expected_frames=sum(e["frames"] for e in schedule),
                proof_complete_seconds=proof["end"] if proof else None,
                strangers_seconds=[at["strangers"]["start"], at["strangers"]["end"]] if "strangers" in at else None,
                air_seconds=[at["air"]["start"], at["air"]["end"]] if "air" in at else None,
                studio_start_seconds=at["try_sample"]["start"] if "try_sample" in at else None,
                changed_preserved_beats=changed, start_shifts=shifts,
                timeline=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in schedule]), errors


def source_provenance_check(board: dict) -> tuple[dict, list[str]]:
    """Bind the preserved v11 renders, the inserted takes and the v15 recap to recorded hashes."""
    from check_v15 import CHECKOUT

    errors, hashes = [], {}

    def digest(path: Path) -> str:
        if path not in hashes:
            hashes[path] = sha256(path)
        return hashes[path]

    paths = media_paths(board)
    reuse = board.get("source_reuse", {})
    manifest = Path(board.get("provenance", ""))
    if (manifest != MANIFEST or reuse.get("manifest") != str(MANIFEST) or not MANIFEST.is_file()
            or reuse.get("manifest_sha256") != digest(MANIFEST) or reuse.get("new_model_generations") != 0
            or reuse.get("composites_rebuilt_for_this_edit") is not False):
        errors.append("v19 editorial reuse must bind the immutable v11 source manifest and disclose no rebuild")
        return {}, errors
    data = read_json(MANIFEST)
    code = data.get("source_sha256", {})
    checked_code = [dict(source=str(CHECKOUT / rel), matches=(CHECKOUT / rel).is_file() and digest(CHECKOUT / rel) == h)
                    for rel, h in code.items()]
    inputs, outputs, edited_metadata, produced = [], [], [], set()
    for name, job in data.get("jobs", {}).items():
        for filename, expected in job.get("inputs_sha256", {}).items():
            source = Path(filename)
            good = source.is_file() and digest(source) == expected
            inputs.append(dict(job=name, source=filename, matches=good))
            if not good and source.name in METADATA_INPUTS and source.is_file():
                edited_metadata.append(dict(job=name, source=filename, recorded_sha256=expected,
                                            current_sha256=digest(source)))
            elif not good:
                errors.append(f"Preserved render input differs from its manifest: {source}")
        if not job.get("outputs_sha256"):
            errors.append(f"No preserved output hashes for {name}")
        for filename, expected in job.get("outputs_sha256", {}).items():
            output = Path(filename)
            produced.add(output)
            good = output.is_file() and digest(output) == expected
            outputs.append(dict(job=name, source=filename, matches=good))
            if not good:
                errors.append(f"Preserved render output differs from its manifest: {output}")
        fingerprint = hashlib.sha256(json.dumps({"code": code, "inputs": job.get("inputs_sha256", {})},
                                                sort_keys=True).encode()).hexdigest()
        if job.get("fingerprint") != fingerprint:
            errors.append(f"Job fingerprint does not match its recorded code and inputs: {name}")
    for path in paths:
        if path.is_relative_to(NEW_CLIPS) and path not in produced:
            errors.append(f"Storyboard v11 clip has no rendered-output provenance: {path}")
        if "demo_v17" in str(path) or "contest-polish-20261023" in str(path):
            errors.append(f"Replaced v17 or 23 October capture media remain in the cut: {path}")
    jobs = data.get("jobs", {})
    elements = {value for name in ("proof_boom", "proof_pf_pf_boom", "proof_pf_x4")
                for filename, value in jobs.get(name, {}).get("inputs_sha256", {}).items()
                if Path(filename).name == "element.mp4"}
    element = next(iter(elements)) if len(elements) == 1 else None
    if element is None:
        errors.append("The three proof performances do not share one LTX element SHA-256")
    elif digest(ROOT / PROOF_ELEMENT_DIR / "element.mp4") != element:
        errors.append("The shared proof element no longer matches its stored bytes")
    if not all("no new model render" in jobs.get(n, {}).get("method", "") for n in ("proof_boom", "proof_pf_pf_boom", "proof_pf_x4")):
        errors.append("Proof renders do not document reuse without a new model render")
    takes = []
    for name, _ in STRANGERS:
        folder = BENDING / f"hand_palm__{name}__again"
        again = read_json(folder / "receipt.json").get("bending", {}).get("performed_again") or {}
        good = (again.get("element_sha256") == element and again.get("no_new_render") is True
                and again.get("no_gpu") is True and again.get("from") == PROOF_ELEMENT_DIR)
        takes.append(dict(take=str(folder / "final.mp4"), sha256=digest(folder / "final.mp4"), performed_again=again,
                          same_element_as_proof=good))
        if not good:
            errors.append(f"Stranger take is not the same stored element without a new render: {name}")
    air_receipt = read_json(AIR_SRC.parent / "receipt.json")
    air = dict(take=str(AIR_SRC), sha256=digest(AIR_SRC), effect=air_receipt["effect"]["id"],
               element_source=air_receipt["bending"].get("element_source"),
               element_sha256=digest(AIR_SRC.parent / "element.mp4"))
    if "LTX-2.5" not in str(air["element_source"]) or "no GPU" not in str(air["element_source"]):
        errors.append("Air take does not document its stored LTX-2.5 element and CPU shot")
    recap_rows = read_json(RECAP_QA).get("source_provenance", {}).get("storyboard_assets", [])
    recorded = next((r["sha256"] for r in recap_rows if Path(r["path"]) == NEW_RECAP), None)
    recap_qa = read_json(RECAP_QA)
    recap = dict(src=str(NEW_RECAP), sha256=digest(NEW_RECAP), recorded_in_v15_qa=recorded,
                 v15_qa_status=recap_qa.get("status"), v15_qa_errors=len(recap_qa.get("errors", [])))
    if (recorded != recap["sha256"] or recap_qa.get("status") != "measured_checks_passed" or recap_qa.get("errors")
            or board.get("recap", {}).get("sha256") != recap["sha256"]):
        errors.append("The v15 recap animation differs from the bytes verified by the passing v15 QA")
    derived = []
    for receipt in sorted({p.parent / "provenance.json" for p in paths
                           if p.parent.name == "cards" and (p.parent / "provenance.json").is_file()}):
        rows = []
        for field in ("sources_sha256", "outputs_sha256"):
            for filename, expected in read_json(receipt).get(field, {}).items():
                good = Path(filename).is_file() and digest(Path(filename)) == expected
                rows.append(dict(path=filename, matches=good, kind=field))
                if not good:
                    errors.append(f"Derived card media differs from provenance: {filename}")
        derived.append(dict(path=str(receipt), checks=rows))
    assets = []
    for path in sorted(paths):
        if not path.is_file():
            errors.append(f"Missing storyboard source: {path}")
        else:
            assets.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    return dict(method=("Editorial reuse: immutable v11 manifest with its original code fingerprints; every recorded output "
                        "and every media input verified; receipt/catalog metadata edited after the render is listed, "
                        "not hidden; inserted takes bound to their receipts; recap bound to the passing v15 QA"),
                manifests=[dict(path=str(MANIFEST), sha256=digest(MANIFEST),
                                policy="Editorial reuse of the approved v11 renders; no claim of a rebuild by current code",
                                code=checked_code, inputs=inputs, outputs=outputs,
                                metadata_inputs_edited_after_render=edited_metadata)],
                proof_element_sha256=element, strangers=takes, air=air, recap=recap,
                derived_cards=derived, storyboard_assets=assets), errors


def build() -> float:
    """Encode the storyboard; frame buffers go to D: (the system drive is nearly full)."""
    temp = SCRATCH / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temp), TMP=str(temp))
    tempfile.tempdir = str(temp)
    from build_demo import build as encode
    return encode(RUN / "storyboard.json", RUN / "demo_v19.mp4")


def post() -> None:
    """FFmpeg loudness and silence record, then the demo regression tests, bound to the finished video."""
    video = RUN / "demo_v19.mp4"
    video_sha = sha256(video)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-vn", "-af",
                        "silencedetect=noise=-50dB:d=1,ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    record = {"video_sha256": video_sha, "ffmpeg_exit_code": r.returncode,
              "silence_threshold_dbfs": -50, "silence_minimum_seconds": 1,
              "silence_events": [line for line in r.stderr.splitlines() if "silence_start:" in line or "silence_end:" in line],
              "loudness_summary": "Summary:" + r.stderr.rsplit("Summary:", 1)[-1]}
    (RUN / "ffmpeg_audio_checks.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    results, log = [], []
    for name in ("test_v18_edit.py", "test_demo_provenance.py", "test_demo_layout.py", "test_v11_render.py"):
        path = ROOT / "tools/demo" / name
        command = [sys.executable, str(path)]
        p = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        results.append({"test_file": str(path), "sha256": sha256(path), "returncode": p.returncode,
                        "output": p.stdout + p.stderr})
        log.extend(["COMMAND: " + " ".join(command), p.stdout, p.stderr])
    (RUN / "demo-tests.log").write_text("\n".join(log), encoding="utf-8")
    (RUN / "demo-tests.json").write_text(json.dumps({"video_sha256": video_sha,
                                                     "all_passed": all(x["returncode"] == 0 for x in results),
                                                     "results": results}, indent=2) + "\n", encoding="utf-8")
    print(record["loudness_summary"])
    print("silence events:", record["silence_events"] or "none")
    print("tests:", "all passed" if all(x["returncode"] == 0 for x in results) else "FAILED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "build", "post"))
    step = parser.parse_args().step
    if step == "prepare":
        board = prepare()
        _, errors = structure_check(board, read_json(BASELINE))
        for entry in board["schedule"]:
            print(f"{entry['start']:8.3f} {entry['end']:8.3f}  {entry['id']}")
        print(f"duration {board['duration']:.3f} s; {len(errors)} structure errors")
        for error in errors:
            print("ERROR:", error)
        return 1 if errors else 0
    if step == "build":
        print(f"demo: {build():.3f} s -> {RUN / 'demo_v19.mp4'}")
        return 0
    post()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
