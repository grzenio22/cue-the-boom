"""Demo v21: v20's edits with a cold open of five elements on the author's own hands, and the fire as the gallery now
plays it (streaming in moving air).

    python tools/demo/v21_build.py prepare   # runs/demo_v21/storyboard.json from v19's, through v20's edits
    python tools/demo/v21_build.py build     # runs/demo_v21/demo_v21.mp4 and its .build.json
    python tools/demo/v21_build.py post      # FFmpeg loudness/silence record, then the full demo check

The author filmed new hand clips on a phone on 28 October (no webcam takes); the studio made shots from them on the
CPU, cued by the clip's own sound where the author made one and by a sample noise (named on screen) where the clip
was silent. v21 is v20's storyboard (v19 with v20's EDITS, REMOVED and inserted()) without its live block, then:
OPENING (five 1.5 s beats, one per element, before the title), SWAPS (each clip of the stored LTX-2.5 fire on a
gallery hand replaced by the gallery's current composite of the same shot, frame for frame the same timeline),
DROPPED (the tour beats the opening now shows) and RESOURCED (the fingertip lightning from the new clips). Each shot
the opening plays is copied with its receipt into runs/demo_v21/shots, and its SHA-256 is recorded.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_v11 import timeline
import v20_build
from v19_build import ROOT, read_json, sha256

RUN = ROOT / "runs/demo_v21"
BASELINE = ROOT / "runs/demo_v19/storyboard.json"
BENDING = ROOT / "samples/made_ahead/bending"
JOBS = ROOT / "runs/jobs"
SCRATCH = Path("D:/openHiggsfield_tmp/demo21")
MAX_SECONDS = 118
OPEN_S = 1.5
CLIPS_28 = "the author's hand, filmed on a phone on 28 Oct"

# id, shot (a gallery folder or a studio job), first second shown, chip, stamp
OPENING = [
    ("open_fire", BENDING / "hand_walk__fffff__flame_plume", 2.6, "FIRE · fffff",
     "The author's hand and voice, one take, the camera walking · the stored LTX-2.5 fire · no new render"),
    ("open_water", JOBS / "46f4d63b6f63", 1.7, "WATER · shhhh",
     f"{CLIPS_28} · sample noise (synthetic) · water made on the CPU"),
    ("open_air", BENDING / "hand_palm__hoooo__air_vortex", 1.0, "AIR · hoooo",
     "The author's hand · sample noise (synthetic) · an LTX-2.5 whirlwind, rendered once"),
    ("open_explosion", JOBS / "8bf6ba31033b", 0.95, "EXPLOSION · BOOOOM",
     f"{CLIPS_28} · sample noise (synthetic) · the stored LTX-2.5 fireball, its blast made on the CPU"),
    ("open_lightning", JOBS / "18766d702104", 2.2, "LIGHTNING · zzzzz",
     f"{CLIPS_28}, and his own zzzzz · lightning made on the CPU"),
]
OPEN_HEADLINE, OPEN_SUBLINE = "One voice. Five elements.", "Real hands, filmed on a phone"
SWAPS = {  # v11's clips of the stored fire on a gallery hand -> that shot as the gallery now plays it
    "runs/demo_v11/clips/proof_boom/composite.mp4": "hand_palm__boom__fire_burst",
    "runs/demo_v11/clips/proof_pf_pf_boom/composite.mp4": "hand_palm__pfpfboom__again",
    "runs/demo_v11/clips/proof_pf_x4/composite.mp4": "hand_palm__pfx4__again",
    "runs/demo_v11/clips/flame_walk/composite.mp4": "hand_walk__fffff__flame_plume",
    "runs/demo_v11/clips/flamethrower/composite.mp4": "hand_point__fwoooom__fire_breath",
}
DROPPED = {
    "air": "The air vortex now plays in the opening",
    "water": "A second lightning clip under a water id; water now plays in the opening",
    "dino": "Time for the opening; the lanterns keep the render into a scene",
}
RESOURCED = {  # beat id -> (studio job, start, stamp)
    "lightning_point": ("2c35b55cfe7d", 0.85, f"{CLIPS_28} · sample noise (synthetic) · procedural CPU effect"),
}


def rel(path: str | Path) -> str:
    return str(path).replace("\\", "/").split("openHiggsfield/", 1)[-1]


def shot_copy(src: Path) -> Path:
    """A studio job's shot copied (its picture, sound and receipt) where the demo keeps its sources."""
    if src.parent != JOBS:
        return src
    dst = RUN / "shots" / src.name
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("composite.mp4", "final.mp4", "receipt.json", "poster.jpg"):
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)
    return dst


def opening_beats() -> list[dict]:
    beats = []
    for k, (bid, shot, start, chip, stamp) in enumerate(OPENING):
        clip = shot_copy(shot) / "composite.mp4"
        b = dict(id=bid, type="clip", src=str(clip), start=start, dur=OPEN_S, fill=True, portrait_backdrop=True,
                 chip=chip, stamp=stamp, source_sha256=sha256(clip),
                 source_gain=v20_build.level_gain(clip, start, OPEN_S))
        if k == 0:
            b.update(headline=OPEN_HEADLINE, subline=OPEN_SUBLINE)
        beats.append(b)
    return beats


def swapped(beat: dict) -> dict:
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str) and rel(x) in SWAPS:
            return str(BENDING / SWAPS[rel(x)] / "composite.mp4")
        return x
    return walk(beat)


def resourced(beat: dict) -> dict:
    job, start, stamp = RESOURCED[beat["id"]]
    clip = shot_copy(JOBS / job) / "composite.mp4"
    beat = dict(beat, src=str(clip), start=start, stamp=stamp, source_sha256=sha256(clip))
    beat["source_gain"] = v20_build.level_gain(clip, start, float(beat["dur"]))
    return beat


def v21_beats() -> list[dict]:
    beats = opening_beats()
    for beat in v20_build.edited(read_json(BASELINE)["beats"]):
        if beat["id"] in DROPPED:
            continue
        beat = swapped(deepcopy(beat))
        beats.append(resourced(beat) if beat["id"] in RESOURCED else beat)
    return beats


def prepare() -> dict:
    v20_build.write_node_evidence()
    baseline = read_json(BASELINE)
    beats = v21_beats()
    schedule = timeline(beats)
    board = deepcopy(baseline)
    board.pop("browser_capture", None)
    shots = {p.name: dict(receipt_sha256=sha256(p / "receipt.json"), composite_sha256=sha256(p / "composite.mp4"))
             for p in sorted((RUN / "shots").iterdir())}
    board.update(
        version=21, beats=beats, duration=schedule[-1]["end"],
        schedule=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in schedule],
        baseline=str(BASELINE), baseline_sha256=sha256(BASELINE),
        v20_edits=v20_build.edits_record(),
        v21_edits=dict(opening=[b[0] for b in OPENING], swaps=SWAPS, dropped=DROPPED,
                       resourced={k: v[0] for k, v in RESOURCED.items()}, shots=shots),
        source_policy=baseline["source_policy"] + " v21 opens on five elements made by the studio on the CPU from "
                                                  "the author's own phone clips (28 Oct) and the gallery, and plays "
                                                  "the gallery's current fire; see v21_edits.")
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "storyboard.json").write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return board


def structure_check(board: dict, baseline: dict) -> tuple[dict, list[str]]:
    """v20's edited beats, in order, with only v21's listed changes; the opening and new sources as recorded."""
    errors = []
    expected = v21_beats()
    ids = [b["id"] for b in board["beats"]]
    if ids != [b["id"] for b in expected]:
        errors.append("The beat sequence differs from v20's edits with v21's opening and drops")
    for got, want in zip(board["beats"], expected):
        if got != want:
            errors.append(f"Beat differs from its v21 recipe: {got['id']}")
    for b in board["beats"]:
        if "source_sha256" in b and sha256(Path(b["src"])) != b["source_sha256"]:
            errors.append(f"{b['id']}: its source changed after it was recorded")
    if len({b["id"] for b in board["beats"]}) != len(ids):
        errors.append("Duplicate beat identifiers")
    schedule = timeline(board["beats"])
    duration = schedule[-1]["end"]
    if duration > MAX_SECONDS + 1e-6:
        errors.append(f"Picture exceeds {MAX_SECONDS} seconds: {duration:.3f}")
    opening = [b for b in board["beats"] if b["id"].startswith("open_")]
    if len(opening) != 5 or any(abs(b["dur"] - OPEN_S) > 1e-6 for b in opening):
        errors.append("The opening must be five 1.5 s element beats")
    return dict(duration=duration, expected_frames=sum(e["frames"] for e in schedule), beats=len(ids), opening=[b["id"] for b in opening], dropped=list(DROPPED)), errors


def source_provenance_check(board: dict) -> tuple[dict, list[str]]:
    """v19's source checks for every beat it keeps, plus each new shot made from stored elements with no model."""
    import v19_build
    report, errors = v19_build.source_provenance_check(board)
    shots = []
    for name, rec in board.get("v21_edits", {}).get("shots", {}).items():
        receipt = RUN / "shots" / name / "receipt.json"
        r = read_json(receipt)
        shots.append(dict(job=name, effect=r["effect"]["id"], cue=r["inputs"].get("cue"),
                          render_seconds=r["render"]["seconds"]))
        if sha256(receipt) != rec["receipt_sha256"]:
            errors.append(f"A new shot's receipt changed after it was recorded: {name}")
        if r["render"]["seconds"] not in (0, 0.0):
            errors.append(f"A new shot was rendered with a model, not made from stored elements: {name}")
    report["v21_shots"] = shots
    return report, errors


def build() -> float:
    temp = SCRATCH / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temp), TMP=str(temp))
    tempfile.tempdir = str(temp)
    from build_demo import build as encode
    video = RUN / "demo_v21.mp4"
    seconds = encode(RUN / "storyboard.json", video)
    record_path = video.with_suffix(".build.json")
    record = read_json(record_path)
    record["mastering"] = v20_build.master(video, read_json(RUN / "storyboard.json")["beats"])
    record["video_sha256"] = sha256(video)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return seconds


def post() -> int:
    v20_build.run = RUN
    video = RUN / "demo_v21.mp4"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-vn", "-af",
                        "silencedetect=noise=-50dB:d=1,ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    record = {"video_sha256": sha256(video), "ffmpeg_exit_code": r.returncode,
              "silence_events": [line for line in r.stderr.splitlines() if "silence_start:" in line or "silence_end:" in line],
              "loudness_summary": "Summary:" + r.stderr.rsplit("Summary:", 1)[-1]}
    (RUN / "ffmpeg_audio_checks.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record["loudness_summary"])
    print("silence events:", record["silence_events"] or "none")
    check = subprocess.run([sys.executable, str(Path(__file__).with_name("check_v15.py")), "--video", str(video),
                            "--storyboard", str(RUN / "storyboard.json"), "--baseline", str(BASELINE)], cwd=ROOT)
    return check.returncode or r.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("prepare", "build", "post"))
    args = parser.parse_args()
    if args.step == "prepare":
        board = prepare()
        summary, errors = structure_check(board, read_json(BASELINE))
        print(json.dumps(summary, indent=2))
        for e in errors:
            print("ERROR", e)
        return 1 if errors else 0
    if args.step == "build":
        print(f"{build():.2f} s")
        return 0
    return post()


if __name__ == "__main__":
    sys.exit(main())
