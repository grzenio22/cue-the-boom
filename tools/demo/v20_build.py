"""Demo v20: v19 with the author's live takes, filmed on a phone, in place of the automated Studio capture.

    python tools/demo/v20_build.py ingest PHONE.mp4 [PHONE.mp4 ...] [--job JOB_ID ...]
    python tools/demo/v20_build.py prepare   # runs/demo_v20/storyboard.json from v19's and runs/demo_v20/live.json
    python tools/demo/v20_build.py build     # runs/demo_v20/demo_v20.mp4 and its .build.json
    python tools/demo/v20_build.py post      # FFmpeg loudness/silence record, then the full demo check
    python tools/demo/check_v15.py --storyboard runs/demo_v20/storyboard.json

ingest: each phone clip upright (its rotation applied), at a constant 24 fps, no more than 1080 rows (1080
columns if portrait), AAC 48 kHz, in live/phone_N.mp4, with a contact sheet of one frame a second and the loud
onsets of its sound (where a BOOM is) to pick windows from. Each Studio job named with --job (default: the jobs
finished while the clips were filmed, from the phone's creation time) is copied with its inputs and receipt to
live/jobs/<id>. live.json records the sources (original path, SHA-256, creation time, rotation) and, the first
time, draft beats to edit: a window of the first phone clip from its first loud sound, and the newest job's shot
from its first hit.
prepare: v19's beats with its five automated Studio beats (try_sample ... result) replaced, in place, by
live.json's beats. Every other beat is v19's, unchanged. No model generation, GPU service or narration
generation is used. --run-dir builds elsewhere (a stand-in test of the plumbing, never shipped).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from check_v11 import timeline
import v19_build
from v19_build import FPS, ROOT, media_paths, read_json, sha256, video_duration

BASELINE = ROOT / "runs/demo_v19/storyboard.json"
JOBS = ROOT / "runs/jobs"
SCRATCH = Path("D:/openHiggsfield_tmp/demo20")
STUDIO_IDS = v19_build.STUDIO_IDS
CAPTURE_DIR = "contest-polish-20261026b"  # the automated capture the live takes replace
FOOTER = 64          # rows of the studio's export footer under the picture (the receipt line and the waveform)
MAX_SECONDS = 118
SR = 48000
PHONE_STAMP = "Filmed on a phone · the author, live · the studio on a laptop CPU, no GPU"
RESULT_STAMP = "Hand followed, matted and lit on the CPU · the stored LTX-2.5 fire · no new render"
run = Path(os.environ.get("CUE_DEMO20_RUN", ROOT / "runs/demo_v20"))


def live_dir() -> Path:
    return run / "live"


def live_json() -> Path:
    return run / "live.json"


def probe(path: Path) -> dict:
    """Size, length, rotation, creation time and frame rate, as the file states them (ffprobe)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
                       capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"{path} has no picture")
    rotation = int(float(video.get("tags", {}).get("rotate", 0)))
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = int(float(side["rotation"]))
    tags = {**data["format"].get("tags", {}), **video.get("tags", {})}
    num, den = (int(x) for x in video.get("avg_frame_rate", "0/1").split("/"))
    return dict(width=int(video["width"]), height=int(video["height"]), rotation=rotation,
                upright=[int(video["height"]), int(video["width"])] if rotation % 180 else
                        [int(video["width"]), int(video["height"])],
                duration=float(data["format"]["duration"]), fps=round(num / den, 3) if den else None,
                creation_time=tags.get("creation_time") or tags.get("com.apple.quicktime.creationdate"),
                audio=any(s["codec_type"] == "audio" for s in data["streams"]))


def normalize(src: Path, dst: Path) -> None:
    """Upright (FFmpeg applies the rotation), 24 fps constant, at most 1080 on the short side, 48 kHz AAC."""
    shape = "scale='if(gt(iw,ih),-2,min(1080,iw))':'if(gt(iw,ih),min(1080,ih),-2)'"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"fps={FPS},{shape},format=yuv420p",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-c:a", "aac", "-b:a", "192k",
                    "-ar", str(SR), "-movflags", "+faststart", str(dst)], check=True)


def onsets(path: Path, floor_db: float = 14.0, quiet_s: float = 0.3) -> list[dict]:
    """Where a loud sound starts (in s), with its peak: a 20 ms level rising floor_db over the clip's median
    level after at least quiet_s below it."""
    from cuesync.media import read_audio

    audio = read_audio(path, SR)
    hop = SR // 50
    if audio.size < hop:
        return []
    frames = audio[: audio.size // hop * hop].reshape(-1, hop)
    level = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    threshold = float(np.median(level)) + floor_db
    found, quiet = [], int(quiet_s * 50)
    k = 0
    while k < level.size:
        if level[k] >= threshold and (k < quiet or (level[max(0, k - quiet):k] < threshold).all()):
            end = k
            while end < level.size and level[end] >= threshold - 6:
                end += 1
            found.append(dict(at=round(k / 50, 2), length=round((end - k) / 50, 2),
                              peak_dbfs=round(float(level[k:end].max()), 1)))
            k = end + 1
        else:
            k += 1
    return found


def sheet(path: Path, out: Path, every: float = 1.0) -> None:
    """One frame every ``every`` s, 6 to a row, each labelled with its time: to pick a window by eye."""
    from PIL import Image, ImageDraw

    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps=1/{every},scale=320:-2",
                        str(Path(tmp) / "f%04d.jpg")], check=True)
        frames = sorted(Path(tmp).glob("f*.jpg"))
        if not frames:
            return
        tiles = [Image.open(f).convert("RGB") for f in frames]
        w, h = tiles[0].size
        cols = 6
        board = Image.new("RGB", (w * cols, h * ((len(tiles) + cols - 1) // cols)), (11, 11, 15))
        for k, tile in enumerate(tiles):
            d = ImageDraw.Draw(tile)
            d.rectangle([0, 0, 70, 26], fill=(0, 0, 0))
            d.text((6, 6), f"{k * every:.0f} s", fill=(255, 160, 40))
            board.paste(tile, ((k % cols) * w, (k // cols) * h))
        board.save(out, quality=85)


def stamp_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def jobs_during(clips: list[dict]) -> list[Path]:
    """The Studio jobs finished from a minute before the first clip began to half an hour after the last ended (the
    shot takes a few minutes; the clip of its result may come later); without creation times, the newest three."""
    done = [d for d in JOBS.iterdir() if (d / "final.mp4").is_file() and (d / "receipt.json").is_file()] \
        if JOBS.is_dir() else []
    begins = [t for t in (stamp_time(c["creation_time"]) for c in clips) if t is not None]
    if not begins:
        return sorted(done, key=lambda d: (d / "final.mp4").stat().st_mtime)[-3:]
    first = min(begins) - 60
    last = max(t + c["duration"] for t, c in zip(begins, clips)) + 1800
    return sorted((d for d in done if first <= (d / "final.mp4").stat().st_mtime <= last),
                  key=lambda d: (d / "final.mp4").stat().st_mtime)


def copy_job(job: Path) -> dict:
    """The job's shot, receipt, poster and inputs (your voice or your take) under live/jobs/<id>."""
    dst = live_dir() / "jobs" / job.name
    (dst / "in").mkdir(parents=True, exist_ok=True)
    for name in ("final.mp4", "receipt.json", "poster.jpg"):
        if (job / name).is_file():
            shutil.copy2(job / name, dst / name)
    for f in (job / "in").glob("*") if (job / "in").is_dir() else []:
        if f.is_file():
            shutil.copy2(f, dst / "in" / f.name)
    receipt = read_json(dst / "receipt.json")
    hits = (receipt.get("bending") or {}).get("hits") or []
    return dict(folder=str(job), final=str(dst / "final.mp4"), final_sha256=sha256(dst / "final.mp4"),
                mode=receipt.get("mode"), effect=(receipt.get("effect") or {}).get("id"),
                cue=(receipt.get("inputs") or {}).get("cue"), seconds_total=receipt.get("seconds_total"),
                render_seconds=(receipt.get("render") or {}).get("seconds"),
                render_prompt=(receipt.get("render") or {}).get("prompt"),
                performed_again=(receipt.get("bending") or {}).get("performed_again"),
                first_hit_s=round(hits[0]["frame"] / FPS, 3) if hits else None,
                finished=datetime.fromtimestamp((job / "final.mp4").stat().st_mtime, timezone.utc).isoformat())


def draft_beats(phones: dict, jobs: dict) -> list[dict]:
    """A first cut to edit: the first phone clip from just before its first loud sound, then the newest job's shot
    from just before its first hit."""
    beats = []
    if phones:
        name, info = next(iter(phones.items()))
        first = info["onsets"][0]["at"] if info["onsets"] else 0.0
        start = max(0.0, first - 1.0)
        beats.append(dict(id="live_preview", src=f"live/{name}", start=round(start, 2),
                          dur=round(min(7.0, info["duration"] - start), 2),
                          chip="Try it live · your webcam and your voice", stamp=PHONE_STAMP))
    if jobs:
        job_id, info = list(jobs.items())[-1]
        start = max(0.0, (info["first_hit_s"] or 0.5) - 0.45)
        beats.append(dict(id="live_result", src=f"live/jobs/{job_id}/final.mp4", start=round(start, 3), dur=3.5,
                          hold_tail=True, hold_reason="The shot's final frame is held to the end of the beat",
                          chip="The shot made from that take", stamp=RESULT_STAMP))
    return beats


def ingest(phones: list[Path], job_ids: list[str]) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    live_dir().mkdir(parents=True, exist_ok=True)
    data = read_json(live_json()) if live_json().is_file() else dict(version=20, phones={}, jobs={}, beats=[])
    for src in phones:
        known = next((n for n, p in data["phones"].items() if p["original"] == str(src)), None)
        name = known or f"phone_{len(data['phones']) + 1}.mp4"
        info = probe(src)
        normalize(src, live_dir() / name)
        sheet(live_dir() / name, live_dir() / (Path(name).stem + "_sheet.jpg"))
        data["phones"][name] = dict(original=str(src), original_sha256=sha256(src), sha256=sha256(live_dir() / name),
                                    creation_time=info["creation_time"], rotation=info["rotation"],
                                    original_size=[info["width"], info["height"]], upright=info["upright"],
                                    original_fps=info["fps"], duration=video_duration(live_dir() / name),
                                    audio=info["audio"], onsets=onsets(live_dir() / name) if info["audio"] else [])
        print(f"{name}: {src.name}, {info['upright'][0]}x{info['upright'][1]} upright, {info['duration']:.1f} s, "
              f"rotation {info['rotation']}, created {info['creation_time']}")
        print("  loud sounds at:", ", ".join(f"{o['at']:.2f} s" for o in data["phones"][name]["onsets"][:12]) or "none")
    clips = [dict(creation_time=p["creation_time"], duration=p["duration"]) for p in data["phones"].values()]
    jobs = [Path(j) if Path(j).is_dir() else JOBS / j for j in job_ids] if job_ids else jobs_during(clips)
    for job in jobs:
        if not (job / "final.mp4").is_file():
            raise FileNotFoundError(f"no finished shot in {job}")
        data["jobs"][job.name] = copy_job(job)
        j = data["jobs"][job.name]
        print(f"job {job.name}: {j['effect']} ({j['mode']}, cue {j['cue']}), first hit {j['first_hit_s']} s, "
              f"{j['seconds_total']} s to make, finished {j['finished']}")
    if not data["beats"]:
        data["beats"] = draft_beats(data["phones"], data["jobs"])
        data["draft"] = True
    live_json().write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {live_json()} ({'draft beats: edit the windows, then set draft to false' if data.get('draft') else 'beats kept'})")
    return data


def resolve(src: str) -> Path:
    path = Path(src)
    return path if path.is_absolute() else run / path


def live_beat(raw: dict, live: dict) -> dict:
    """A live.json beat as a storyboard clip beat: a phone clip fills the frame (a portrait one over its blurred
    copy); a job's shot loses its export footer and fills the frame the same way."""
    b = dict(type="clip", fill=True, source_gain=1.0)
    b.update(deepcopy(raw))
    path = resolve(raw["src"])
    b["src"] = str(path)
    upright = None
    if path.parent == live_dir() and path.name in live["phones"]:
        upright = live["phones"][path.name]["upright"]
    elif path.name == "final.mp4" and path.parent.name in live["jobs"]:
        import av
        with av.open(str(path)) as c:
            w, h = c.streams.video[0].width, c.streams.video[0].height
        if "crop" not in raw and (h - FOOTER) in (1080, 1920, 1440):
            b["crop"] = [0, 0, 1, round((h - FOOTER) / h, 6)]
        upright = [w, h - FOOTER]
    if upright and upright[1] > upright[0]:
        b.setdefault("portrait_backdrop", True)
    return b


def expected_ids(baseline: dict, live: dict) -> list[str]:
    ids, placed = [], False
    for beat in baseline["beats"]:
        if beat["id"] in STUDIO_IDS:
            if not placed:
                ids.extend(b["id"] for b in live["beats"])
                placed = True
        else:
            ids.append(beat["id"])
    return ids


def prepare() -> dict:
    baseline, live = read_json(BASELINE), read_json(live_json())
    if baseline.get("version") != 19:
        raise ValueError("The v20 baseline must be the v19 storyboard")
    beats, placed = [], False
    for beat in baseline["beats"]:
        if beat["id"] in STUDIO_IDS:
            if not placed:
                beats.extend(live_beat(b, live) for b in live["beats"])
                placed = True
        else:
            beats.append(deepcopy(beat))
    schedule = timeline(beats)
    board = deepcopy(baseline)
    for key in ("browser_capture",):
        board.pop(key, None)
    board.update(
        version=20, beats=beats, duration=schedule[-1]["end"],
        schedule=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in schedule],
        baseline=str(BASELINE), baseline_sha256=sha256(BASELINE),
        live=dict(record=str(live_json()), record_sha256=sha256(live_json()), draft=bool(live.get("draft")),
                  replaces=list(STUDIO_IDS), phones=live["phones"], jobs=live["jobs"],
                  method="The author's own takes, filmed on a phone at the laptop while the studio ran on its CPU; "
                         "each phone clip upright at 24 fps, unedited within its window; each shot as the studio "
                         "exported it, without its footer."),
        source_policy=baseline["source_policy"] + " v20 replaces the automated Studio capture with the author's live "
                                                  "takes filmed on a phone and the shots the studio made from them.")
    run.mkdir(parents=True, exist_ok=True)
    (run / "storyboard.json").write_text(json.dumps(board, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return board


def structure_check(board: dict, baseline: dict) -> tuple[dict, list[str]]:
    """v19's beats unchanged around the live block; the live beats from their recorded sources, within them."""
    errors = []
    record = Path(board.get("live", {}).get("record", live_json()))
    live = read_json(record) if record.is_file() else dict(beats=[], phones={}, jobs={})
    beats, ids = board["beats"], [b["id"] for b in board["beats"]]
    if baseline.get("version") != 19:
        errors.append("The v20 baseline must be the v19 storyboard")
    if ids != expected_ids(baseline, live):
        errors.append("v20 must be v19's beat order with the Studio beats replaced in place by live.json's beats")
    old = {b["id"]: b for b in baseline["beats"]}
    for beat in beats:
        if beat["id"] in old and beat != old[beat["id"]]:
            errors.append(f"A v19 beat changed: {beat['id']}")
    live_beats = [b for b in beats if b["id"] not in old]
    if [b for b in live_beats if not b["id"].startswith("live_")]:
        errors.append("Live beats are named live_*")
    if any(b["id"] in STUDIO_IDS for b in beats):
        errors.append("An automated Studio beat remains")
    phones = jobs = 0
    for b in live_beats:
        path = Path(b["src"])
        if not path.is_file():
            errors.append(f"Missing live source: {path}")
            continue
        if path.parent == record.parent / "live" and path.name in live["phones"]:
            phones += 1
            if sha256(path) != live["phones"][path.name]["sha256"]:
                errors.append(f"Phone clip differs from its ingest record: {path}")
            if "phone" not in b.get("stamp", ""):
                errors.append(f"A phone beat says it was filmed on a phone: {b['id']}")
        elif path.name == "final.mp4" and path.parent.name in live["jobs"]:
            jobs += 1
            if sha256(path) != live["jobs"][path.parent.name]["final_sha256"]:
                errors.append(f"Shot differs from its ingest record: {path}")
        else:
            errors.append(f"A live beat plays something ingest did not record: {path}")
        end = float(b.get("start", 0)) + float(b["dur"]) * float(b.get("speed", 1))
        if end > video_duration(path) + 1 / FPS and not b.get("hold_tail"):
            errors.append(f"Live window runs past its source: {b['id']}")
        if float(b.get("speed", 1)) != 1 and not b.get("note"):
            errors.append(f"A sped-up live beat says so (note): {b['id']}")
        if not b.get("stamp"):
            errors.append(f"Live beat without its stamp: {b['id']}")
    if not phones:
        errors.append("No beat from a phone clip")
    if not jobs:
        errors.append("No beat shows a shot the studio made from a live take")
    stale = sorted(str(p) for p in media_paths(beats) if CAPTURE_DIR in p.parts or "demo_v17" in str(p))
    if stale:
        errors.append(f"Automated capture or v17 media remain: {', '.join(stale)}")
    schedule = timeline(beats)
    duration = schedule[-1]["end"] if schedule else 0
    at = {e["id"]: e for e in schedule}
    if "performance_turns" not in at or at["performance_turns"]["end"] > 15:
        errors.append("All three complete performances must finish within the opening 15 seconds")
    if duration > MAX_SECONDS + 1e-6:
        errors.append(f"Picture exceeds {MAX_SECONDS} seconds: {duration:.3f}")
    if abs(float(board.get("duration", 0)) - duration) > .002:
        errors.append("Storyboard duration does not match frame-rounded beats")
    live_ids = [b["id"] for b in live_beats]
    return dict(duration=duration, baseline_duration=baseline["duration"],
                expected_frames=sum(e["frames"] for e in schedule),
                proof_complete_seconds=at["performance_turns"]["end"] if "performance_turns" in at else None,
                live_seconds=[at[live_ids[0]]["start"], at[live_ids[-1]]["end"]] if live_ids else None,
                live_beats=live_ids, draft=bool(board.get("live", {}).get("draft")),
                timeline=[dict(id=e["id"], start=e["start"], end=e["end"]) for e in schedule]), errors


def source_provenance_check(board: dict) -> tuple[dict, list[str]]:
    """v19's source checks for every beat it keeps, plus the live takes bound to their ingest record."""
    report, errors = v19_build.source_provenance_check(board)
    live = board.get("live", {})
    takes = []
    for name, p in live.get("phones", {}).items():
        original = Path(p["original"])
        takes.append(dict(phone=name, original=str(original), original_present=original.is_file(),
                          original_matches=original.is_file() and sha256(original) == p["original_sha256"],
                          creation_time=p.get("creation_time")))
    shots = [dict(job=j, effect=info.get("effect"), render_seconds=info.get("render_seconds"),
                  render_prompt=info.get("render_prompt"), seconds_total=info.get("seconds_total"))
             for j, info in live.get("jobs", {}).items()]
    for shot in shots:
        if shot["render_seconds"] not in (0, 0.0, None):
            errors.append(f"A live shot was rendered with a model, not made from stored elements: {shot['job']}")
    report["live"] = dict(phones=takes, shots=shots)
    return report, errors


def build() -> float:
    """Encode the storyboard; frame buffers go to D: (the system drive is nearly full)."""
    temp = SCRATCH / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(temp), TMP=str(temp))
    tempfile.tempdir = str(temp)
    from build_demo import build as encode
    return encode(run / "storyboard.json", run / "demo_v20.mp4")


def post() -> int:
    """FFmpeg loudness and silence record bound to the finished video, then the full demo check."""
    video = run / "demo_v20.mp4"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-vn", "-af",
                        "silencedetect=noise=-50dB:d=1,ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    record = {"video_sha256": sha256(video), "ffmpeg_exit_code": r.returncode,
              "silence_events": [line for line in r.stderr.splitlines() if "silence_start:" in line or "silence_end:" in line],
              "loudness_summary": "Summary:" + r.stderr.rsplit("Summary:", 1)[-1]}
    (run / "ffmpeg_audio_checks.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record["loudness_summary"])
    print("silence events:", record["silence_events"] or "none")
    check = subprocess.run([sys.executable, str(Path(__file__).with_name("check_v15.py")), "--video", str(video),
                            "--storyboard", str(run / "storyboard.json")], cwd=ROOT)
    return check.returncode or r.returncode


def main() -> int:
    global run
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("ingest", "prepare", "build", "post"))
    parser.add_argument("phones", nargs="*", type=Path, help="ingest: the phone clips, as copied off the phone")
    parser.add_argument("--job", action="append", default=[], help="ingest: a Studio job id (runs/jobs/<id>) or folder")
    parser.add_argument("--run-dir", type=Path, help="build somewhere other than runs/demo_v20 (a stand-in test)")
    args = parser.parse_args()
    if args.run_dir:
        run = args.run_dir
        os.environ["CUE_DEMO20_RUN"] = str(run)
    if args.step == "ingest":
        ingest(args.phones, args.job)
        return 0
    if args.step == "prepare":
        board = prepare()
        _, errors = structure_check(board, read_json(BASELINE))
        for entry in board["schedule"]:
            print(f"{entry['start']:8.3f} {entry['end']:8.3f}  {entry['id']}")
        print(f"duration {board['duration']:.3f} s; {len(errors)} structure errors"
              + ("; DRAFT live windows (live.json draft: true)" if board["live"]["draft"] else ""))
        for error in errors:
            print("ERROR:", error)
        return 1 if errors else 0
    if args.step == "build":
        if read_json(run / "storyboard.json")["live"]["draft"] and run == ROOT / "runs/demo_v20":
            raise SystemExit("live.json is still a draft: review the windows and set draft to false, then prepare")
        print(f"demo: {build():.3f} s -> {run / 'demo_v20.mp4'}")
        return 0
    return post()


if __name__ == "__main__":
    raise SystemExit(main())
