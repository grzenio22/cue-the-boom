"""V15 contest finish, preserving the approved v14 cut except targeted edits.

FIRST rebuild every clip from this checkout:
  make_v11_clips.py --force --out D:/openHiggsfield/runs/demo_v11 --temp D:/openHiggsfield_tmp/sol12/demo
Then prepare the normal-tempo contribution, current-code recap, slowed Foley
and exact CueSync node-pack close-up:
  v15_assets.py
  v15_nodes.py
  make_storyboard.py
  build_demo.py tools/demo/storyboard.json -o D:/openHiggsfield/runs/demo_v15/demo_v15.mp4
  check_v11.py --video D:/openHiggsfield/runs/demo_v15/demo_v15.mp4
  check_v15.py

The baseline is runs/demo_v14/storyboard.json (113.375 s). Keep every original
beat and dictionary, apart from: pay for the three-second node insert with
recap/opening-tail holds; restore the slow-motion shots' own slowed Foley, ringing out on the BOOM's own echo;
replace water with the freshly remade dark-bathroom palm lightning; shorten
and re-voice the contribution at normal Kokoro-82M tempo. The moving recap
retains its five highlights and the complete 3.5-second quiet/loud reprise.
No other source windows, captions, voices or beat lengths change. Recorded
room tone fades into quiet gaps at -45 dBFS, leaving audible sound intact.

The render is frame-exact 1920x1080 at 24 fps and remains 113.375 s, safely
below the stricter 118 s limit. Checks write timecoded sheet_*.jpg every 2 s.
All earlier runs are read-only except the regenerated v11 clip folder; all
new cards, audio, manifests and rendered output are written into demo_v15.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

FPS = 24


def make(root: Path, run: Path) -> dict:
    baseline = root / "runs/demo_v14/storyboard.json"
    board = json.loads(baseline.read_text(encoding="utf-8"))
    beats = copy.deepcopy(board["beats"])
    timing = json.loads((run / "edit_timing.json").read_text(encoding="utf-8"))
    lookup = {beat["id"]: beat for beat in beats}
    lookup["cold_open"]["dur"] = timing["cold_open_duration"]
    for name in ("intro_finished", "workflow_fire", "how", "light"):
        lookup[name].update(audio_src=str(run / "audio" / f"{name}.wav"), audio_from=0, audio_gain=.8)
        if lookup[name]["type"] == "clip":
            lookup[name]["audio_replace"] = True
    # Both lightning discharges (about 1.96 and 2.92 s) fit this two-second window.
    lookup["water"].update(src=str(root / "runs/demo_v11/clips/lightning_dark/composite.mp4"), start=1.5)
    node = dict(id="comfy_cuesync", type="image", src=str(run / "cards/comfy_cuesync.png"),
                dur=3.0, zoom=1,
                node_caption="Turns a sound into the guide the LTX render follows.",
                node_evidence=str(run / "cards/comfy_cuesync_evidence.json"))
    beats.insert(next(i for i, beat in enumerate(beats) if beat["id"] == "workflow_fire"), node)

    voice = run / "vo15"
    spoken = json.loads((run / "vo_lines.json").read_text(encoding="utf-8"))["contribution15"]
    phrases = json.loads((voice / "caption_phrases.json").read_text(encoding="utf-8"))["contribution15"]
    start = 0
    for name, count, _ in timing["stages"]:
        beat = lookup[name]
        beat.update(src=str(run / "cards/pipeline_motion.mp4"), start=start/FPS, dur=count/FPS)
        start += count
    lookup["innovation"].update(vo=str(voice / "contribution15.wav"), vo_text=spoken, vo_captions=phrases)

    elapsed, narration_end, schedule = 0, 0.0, []
    for beat in beats:
        frames = round(float(beat["dur"])*FPS)
        beat["dur"] = frames/FPS
        begin, end = elapsed/FPS, (elapsed+frames)/FPS
        schedule.append(dict(id=beat["id"], start=begin, end=end))
        if beat.get("vo"):
            path = Path(beat["vo"])
            duration = json.loads((path.parent / "durations.json").read_text(encoding="utf-8"))[path.stem]
            voice_begin = begin+float(beat.get("vo_at", .15))
            if voice_begin < narration_end-.002:
                raise ValueError(f"Overlapping narration at {beat['id']}")
            narration_end = voice_begin+duration
        elapsed += frames
    duration = elapsed/FPS
    if duration > 118 or narration_end > duration or abs(duration-board["duration"]) > 1/FPS:
        raise ValueError(f"Invalid duration: picture {duration:.6f}s; narration ends {narration_end:.6f}s")
    invitation = next(entry["start"] for entry in schedule if entry["id"] == "try")
    contribution_start = next(entry["start"] for entry in schedule if entry["id"] == "innovation")+.1
    voice_duration = json.loads((voice / "durations.json").read_text(encoding="utf-8"))["contribution15"]
    if contribution_start+voice_duration > invitation-.1:
        raise ValueError("Contribution must finish before the invitation")
    board.update(version=15, duration=duration, beats=beats, schedule=schedule,
        baseline=str(baseline), narration=str(voice / "provenance.json"),
        room_tone=dict(src=str(run / "audio/room_tone.wav"), target_dbfs=-45, threshold_dbfs=-50),
        edit_timing=str(run / "edit_timing.json"),
        source_policy="Every v11 clip rebuilt from the current checkout; no new model renders; narration at normal tempo")
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--run", type=Path, default=Path("D:/openHiggsfield/runs/demo_v15"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("storyboard.json"))
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    board = make(args.root, args.run)
    payload = json.dumps(board, indent=2, ensure_ascii=False)+"\n"
    args.output.write_text(payload, encoding="utf-8")
    (args.run / "storyboard.json").write_text(payload, encoding="utf-8")
    print(f"{len(board['beats'])} beats, {board['duration']:.3f}s -> {args.output}")
    for beat in board["schedule"]:
        print(f"{beat['start']:7.3f} - {beat['end']:7.3f} {beat['id']}")


if __name__ == "__main__":
    main()
