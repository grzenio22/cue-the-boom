"""Prepare v15's normal-tempo contribution, slowed Foley and refreshed recap.

Run make_v11_clips.py --force first. All new assets live in demo_v15; source
footage, narration and earlier cards remain read-only. The short contribution
uses the established Kokoro-82M am_michael voice at speed 1.0. Its measured
duration determines the recap hold; a clean opening hold pays the remaining
time, so the three-second node insert adds no net duration.

Use --foley-only to refresh the shot audio and its provenance without
regenerating narration, cards or edit timing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np

from build_demo import FPS, SR, narrate
from cuesync.media import read_audio

TEXT = ("My contribution: separating generation from performance. "
        "Your voice plays one LTX instrument, without regenerating.")
TEMP = Path("D:/openHiggsfield_tmp/sol12/demo")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, SR, 0, "NONE", "not compressed"))
        output.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())


def stages_for(duration: float) -> list[tuple]:
    # Same five highlights as v14, scaled to the measured normal-tempo line.
    weights = (24, 48, 36, 24, 60)
    boundaries = np.rint(np.cumsum((0, *weights)) / sum(weights) * round(duration*FPS)).astype(int)
    return [(name, int(boundaries[i+1]-boundaries[i]), panel)
            for i, (name, panel) in enumerate((("innovation", None), ("innovation_asset", 2),
                 ("innovation_track", 1), ("innovation_voice", 0), ("innovation_result", 3)))]


def boom_tail(audio: np.ndarray) -> np.ndarray:
    t = np.arange(audio.size) / SR
    source = audio * np.clip((t-1.48)/.02, 0, 1) * np.clip((3.26-t)/.02, 0, 1)
    frequencies = np.fft.rfftfreq(audio.size, 1/SR)
    colour = 1/np.sqrt(1+(frequencies/250)**8) * np.minimum(frequencies/45, 1)**2
    noise = np.random.default_rng(117).standard_normal(audio.size)
    reflections = np.fft.irfft(np.fft.rfft(noise)*colour, audio.size)
    delay = t-.3
    reflections *= np.clip(delay/.06, 0, 1) * np.exp(-np.log(1000)*np.maximum(delay-1, 0)/8)
    size = 1 << (audio.size*2-2).bit_length()
    wet = np.fft.irfft(np.fft.rfft(source, size)*np.fft.rfft(reflections, size), size)[:audio.size]
    wet *= np.minimum(np.linspace(audio.size/SR/.35, 0, audio.size), 1)
    wet *= .48*float(np.max(np.abs(audio))) / max(float(np.max(np.abs(wet))), 1e-8)
    return wet.astype(np.float32)


def prepare_foley(root: Path, run: Path) -> None:
    clips = root / "runs/demo_v11/clips"
    sound = read_audio(clips / "proof_boom/composite.mp4", SR)
    prepared = {}
    for name, length, speed in (("intro_finished", 6, .15), ("workflow_fire", 2, .45),
                                 ("how", 6, .15), ("light", 3, .3)):
        positions = 2.4*SR + np.arange(round(length*SR)) * speed
        slowed = np.interp(positions, np.arange(sound.size), sound, left=0, right=0).astype(np.float32)
        method = "Shot Foley resampled at the picture speed; narration never retimed"
        if name in {"intro_finished", "how"}:
            wet = boom_tail(slowed)
            if name == "how":
                voice = read_audio(root / "runs/demo_v12/vo12/how12.wav", SR)*1.3
                at = round(.65*SR)
                speech = np.zeros_like(slowed)
                speech[at:at+voice.size] = voice
                duck, _ = narrate(np.ones_like(slowed), [(at, voice)])
                duck -= speech
                end = .65 + np.flatnonzero(np.abs(voice) > .001)[-1]/SR
                release = np.clip((np.arange(slowed.size)/SR-end)/(5.5-end), 0, 1)
                gain = .25*np.exp(np.log(4)*release)
                # Cancel the card fade and existing duck only on the wet return.
                fade = np.linspace(1, 0, slowed.size, dtype=np.float32)
                wet = np.divide(wet*gain, fade*duck, out=np.zeros_like(wet), where=fade > 0)
            slowed += wet
            method = ("Shot Foley resampled at picture speed; own BOOM convolved with low-passed "
                      "decaying reflections (45-250 Hz, 300 ms predelay, 1 s diffusion, 8 s low-frequency RT60); "
                      "dry transient retained, wet peak 48% of the dry peak, 350 ms tail fade; narration never retimed")
            if name == "how":
                method += "; wet return compensates the card fade with smooth ducking under the unchanged narration"
        path = run / "audio" / f"{name}.wav"
        wav(path, slowed)
        prepared[name] = dict(path=str(path), source=str(clips / "proof_boom/composite.mp4"),
                              source_from=2.4, picture_speed=speed, duration=length,
                              method=method)
    room_source = clips / "proof_boom.wav"
    room = read_audio(room_source, SR)[:SR//2].copy()
    room -= room.mean()
    # Crossfade the seam of a real half-second room recording for quiet gaps.
    cross = SR//25
    blend = np.linspace(0, 1, cross, dtype=np.float32)
    loop = np.concatenate((room[cross:-cross], room[-cross:]*(1-blend)+room[:cross]*blend))
    room_path = run / "audio/room_tone.wav"
    wav(room_path, loop / max(float(np.sqrt(np.mean(loop**2))), 1e-8) * 10**(-45/20))
    dump(run / "audio/provenance.json", dict(foley=prepared,
         room_tone=dict(path=str(room_path), source=str(room_source), source_from=0, source_duration=.5,
                        target_dbfs=-45, method="Recorded room, DC removed and loop seam crossfaded")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--run", type=Path, default=Path("D:/openHiggsfield/runs/demo_v15"))
    parser.add_argument("--foley-only", action="store_true", help="Refresh only Foley, room tone and audio provenance")
    args = parser.parse_args()
    root, run = args.root, args.run
    TEMP.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), PYTHONDONTWRITEBYTECODE="1")
    if args.foley_only:
        prepare_foley(root, run)
        return

    from v12_cards import pipeline
    from v14_contribution import recap

    dump(run / "vo_lines.json", {"contribution15": TEXT})
    subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("make_v11_voice.py")),
                    "--lines", str(run / "vo_lines.json"), "--output", str(run / "vo15"),
                    "--voice", "am_michael", "--speed", "1.0"], check=True)
    durations = json.loads((run / "vo15/durations.json").read_text(encoding="utf-8"))
    # .10 s lead + .15 s clearance before the unchanged invitation line.
    duration = max(5.0, np.ceil((durations["contribution15"] + .25 - 3.5)*FPS)/FPS)
    cold_open = 13.0-duration
    if cold_open < 6.0:
        raise ValueError("Contribution needs a shorter rewrite at normal tempo")
    stages = stages_for(duration)
    cards = run / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    pipeline(root, cards)
    recap(root, run, stages=stages, duration=duration, source_card=cards / "pipeline.png")
    prepare_foley(root, run)
    dump(run / "edit_timing.json", dict(recap_duration=duration, cold_open_duration=cold_open,
         stages=stages, narration_duration=durations["contribution15"], insert_duration=3.0,
         removed_hold_seconds=8-cold_open, removed_recap_seconds=8-duration,
         narration_speed=1.0))
    print(f"Normal narration {durations['contribution15']:.4f}s; recap {duration:.3f}s; opening {cold_open:.3f}s")


if __name__ == "__main__":
    main()
