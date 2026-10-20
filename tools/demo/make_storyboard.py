"""V11 contest demo: generate the effect once; perform it again with your voice.

118 seconds, 1920x1080. IMG_6666's palm explosion opens (current code, x=540,
y=1000, tap=35); its clean final frame is held to 7s. The proof gets 30 seconds:
ONE stored LTX-2.5 element on IMG_6658, performed with the author's BOOM,
pf/pf/BOOM, and four rising pf sounds. Complete 3.5s takes play individually,
then in synchronized comparison passes. No frozen fire in the comparisons.

Then: raw element, tracking, light off/on; walk, flamethrower, lightning, water,
punch; real scenes; recorded Studio; actual ComfyUI workflow and innovation;
no-GPU invitation and full credits. Fire and the explosion fireball are LTX
renders. Lightning/water are CPU procedural; synthetic samples are labelled.
Kokoro-82M narration is credited as synthetic and captioned for its full audio.

Run make_v11_clips.py, make_v11_voice.py, v11_cards.py, then this script, then:
    python tools/demo/build_demo.py tools/demo/storyboard.json -o D:/openHiggsfield/runs/demo_v11/demo_v11.mp4
All inputs remain read-only. make_v11_clips.py rebuilds with the current code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def make(root: Path, run: Path) -> dict:
    clips, cards, voice = run / "clips", run / "cards", run / "vo11"
    durations = json.loads((voice / "durations.json").read_text(encoding="utf-8"))
    spoken = json.loads((run / "vo_lines.json").read_text(encoding="utf-8"))
    mine = "Author's phone footage + own voice · fire rendered with LTX-2.5 + ainvfx-fluid"
    sample = "Author's phone footage · sample noise (synthetic) · CPU procedural effect"
    studio = "Recorded Studio · author's clip + voice · existing LTX element · GPU offline"

    def shot(name):
        return str(clips / name / "composite.mp4")

    def band(y):
        return [0, y, 1, round(y + 9 / 16 * 1080 / 1920, 5)]

    def say(name, at=.2, text=None):
        return dict(vo=str(voice / f"{name}.wav"), vo_at=at, vo_gain=1.3, vo_text=text or spoken[name])

    tall = [0, .18, 1, .96]
    panels = [
        dict(src=shot("proof_boom"), start=0, crop=tall, label="BOOM",
             audio_src=str(clips / "proof_boom.wav"), pops=[[2.67, "BOOM"]]),
        dict(src=shot("proof_pf_pf_boom"), start=0, crop=tall, label="pf · pf · BOOM",
             audio_src=str(clips / "proof_pf_pf_boom.wav"), pops=[[1.21, "pf"], [1.58, "pf"], [2.67, "BOOM"]]),
        dict(src=shot("proof_pf_x4"), start=0, crop=tall, label="pf pf pf PF",
             audio_src=str(clips / "proof_pf_x4.wav"), pops=[[1.04, "pf"], [1.58, "pf"], [2.04, "pf"], [2.50, "PF"]]),
    ]
    beats = [
        # 00:00–00:07. Room, hand, author's BOOM; no narration on the hook.
        dict(id="cold_open", type="clip", src=shot("explosion_palm"), start=0, dur=7, fill=True,
             portrait_backdrop=True, pops=[[1.46, "BOOM"]], pop_x=1.35, pop_y=.55,
             source_gain=.2, audio_src=str(root / "runs/takes/plates/IMG_6666.mp4"), audio_gain=10,
             stamp="Author's hand + voice · LTX-2.5 fireball · CPU compositing, lighting and shock effects"),
        # 00:07–00:12. Name the instrument after hearing it.
        dict(id="title", type="title", text="Cue the Boom", sub="Your voice performs the effect.", dur=5,
             **say("title11")),
        # 00:12–00:42. One generation and the author's three performances.
        dict(id="one_generation", type="image", src=str(cards / "proof.png"), dur=5, zoom=1, **say("once11")),
    ]
    for k, panel in enumerate(panels):
        beats.append(dict(id=f"performance_{k+1}", type="clip", src=panel["src"], start=0, dur=3.5,
                          fill=True, portrait_backdrop=True, crop=tall, wave=True,
                          audio_src=panel["audio_src"], audio_replace=True,
                          audio_gain=4.5, audio_from=0, pops=panel["pops"], pop_x=.82, pop_y=.6,
                          headline=f"Performance {k+1}: {panel['label']}",
                          subline="Same LTX-2.5 render. New voice timing and size.",
                          stamp="Author's own voice · recorded pf and BOOM cut into new rhythms · 0 new renders"))
    # All three pictures play in step. Focus one actual voice at a time.
    for k, panel in enumerate(panels):
        beat = dict(id=f"comparison_{k+1}", type="compare", panels=panels, dur=3.5, audio=k, audio_gain=4.5,
                    headline="1 generation · 3 performances · 0 new renders",
                    subline=f"All three in step · listening to: {panel['label']}",
                    stamp="Same hand footage and generated element · three voice-only recordings")
        if k == 2:
            beat.update(say("size11", at=1.5))  # continues over the next cut
            beat["subline"] = "Three rhythms, the same LTX-2.5 element."
        beats.append(beat)
    beats.append(dict(id="small_and_big", type="compare", panels=panels, dur=4, loop=3.5, audio=1, audio_gain=3,
                      headline="Quiet pf → small. Loud BOOM → big.",
                      subline="The voice controls the timing and the size.",
                      stamp="One LTX-2.5 element · no new generation"))
    # 00:42–00:55. The same shot at every stage, then the fire's light.
    beats += [
        dict(id="how", type="triptych", srcs=[str(clips / "how_raw.mp4"), str(clips / "how_carried.mp4"),
             shot("how_light_off"), shot("proof_boom")], start=2.4, dur=7, speed=.15,
             labels=["LTX-2.5 over black", "Carried on the hand", "Light off", "Finished + light"], audio=3,
             stamp="Same generated element · slow motion · track, matte, composite and light", **say("how11")),
        dict(id="light", type="triptych", srcs=[shot("how_light_off"), shot("proof_boom")],
             labels=["Light off", "Light on · from the fire's own image"], start=2.4, dur=6, speed=.15,
             audio=1, stamp="The same tracked hand and LTX element · slow motion to show the light"),
        # 00:55–01:12. Every effect keeps its own cue and an honest origin label.
        dict(id="walk", type="clip", src=shot("flame_walk"), start=3, dur=3.25, fill=True, portrait_backdrop=True,
             chip="fffff · carried through a walk, uncut", stamp=mine),
        dict(id="flamethrower", type="clip", src=shot("flamethrower"), start=1.3, dur=2, fill=True,
             portrait_backdrop=True, chip="fwoooom · fingertip flamethrower", stamp=mine),
        dict(id="lightning_point", type="clip", src=shot("lightning_point"), start=1.5, dur=2, fill=True,
             portrait_backdrop=True, source_gain=.35, chip="zzzzz · fingertip lightning", stamp=sample, **say("cpu11", at=.15)),
        dict(id="lightning_dark", type="clip", src=shot("lightning_dark"), start=1.5, dur=2, fill=True,
             portrait_backdrop=True, source_gain=.35, chip="zzzzz · in the dark", stamp=sample),
        dict(id="lightning_ball", type="clip", src=shot("lightning_ball"), start=1.5, dur=2, fill=True,
             portrait_backdrop=True, source_gain=.35, chip="zzzzz · ball lightning", stamp=sample),
        dict(id="water", type="clip", src=shot("water_orb"), start=0, dur=2.75, fill=True, portrait_backdrop=True,
             chip="shhhhh · water orb", stamp=sample),
        dict(id="punch", type="clip", src=shot("explosion_punch"), start=.5, dur=3, fill=True,
             portrait_backdrop=True, pops=[[2.17, "BOOM"]], pop_x=1.35, pop_y=.5,
             stamp="Author's fist + own BOOM · LTX-2.5 fireball · CPU finish"),
        # 01:12–01:25. Each sound fires its own object in real-footage scenes.
        dict(id="lanterns", type="clip", src=shot("scene_lanterns"), start=0, dur=6.5, wave=True,
             marks=[[1.0, .42, .72, "1"], [1.67, .64, .58, "2"], [2.38, .43, .30, "3"]],
             pops=[[1.0, "pf"], [1.67, "pf"], [2.38, "BOOM"]], headline="Each sound, its own lantern",
             subline="Two people's voices: pf · pf · BOOM",
             stamp="Pixabay real footage, first frame held · real CC0 voices (Freesound) · LTX-2.5 fire",
             **say("scene11", at=3.4)),
        dict(id="dino", type="clip", src=shot("scene_dino"), start=.25, dur=6.5, fill=True,
             chip="fwoooom · into the scene", stamp="Real stock footage (Pexels) · sample noise (synthetic) · LTX-2.5 render"),
        # 01:25–01:37. Recorded controls: upload, palm, perform, make and result.
        dict(id="upload", type="clip", src=str(clips / "studio.mp4"), start=1, dur=3, fill=True,
             crop=[0, .03, 1, .78], chip="1. Upload a clip", stamp=studio, **say("upload11")),
        dict(id="tap", type="clip", src=str(clips / "studio.mp4"), start=7, dur=3, fill=True,
             crop=[.078, .208, .578, .583], chip="2. Tap the palm, drag to aim", stamp=studio, **say("tap11")),
        dict(id="perform", type="clip", src=str(clips / "studio.mp4"), start=33, dur=3, fill=True,
             crop=[.078, .5, .634, .917], chip="3. Perform", audio_src=str(clips / "studio_say_own.wav"),
             audio_from=3.23, audio_gain=4, wave=True, wave_box=[.54, .16, .40, .12],
             pops=[[.87, "pf"], [1.27, "pf"], [2.27, "BOOM"]], pop_x=.76, pop_y=.57, stamp=studio),
        dict(id="make", type="clip", src=str(clips / "studio.mp4"), start=37.2, dur=1, fill=True,
             crop=[.06, .27, .79, .82], chip="4. Make the shot", stamp=studio, **say("make11", at=.05)),
        dict(id="result", type="clip", src=str(clips / "studio_result.mp4"), start=1.34, dur=2, fill=True,
             crop=band(.42), chip="The finished shot · existing render", stamp=studio),
        # 01:37–01:48. Model, actual workflow and SDK, plus the innovation.
        dict(id="comfy", type="image", src=str(cards / "comfy.png"), dur=6, zoom=1,
             **say("ltx11", at=.1, text="LTX-2.5, with the fluid LoRA, in ComfyUI.")),
        dict(id="innovation", type="image", src=str(cards / "pipeline.png"), dur=5, zoom=1, **say("new11")),
        # 01:48–01:58. Installation, repo, contest tag, and full credits.
        dict(id="try", type="image", src=str(cards / "try_it.png"), dur=5, zoom=1, **say("try11")),
        dict(id="credits", type="image", src=str(cards / "credits.png"), dur=5, zoom=1,
             audio_src=str(root / "runs/demo_v10/credits_crackle.wav"), audio_from=.8, audio_gain=.3),
    ]
    elapsed, narration_end, schedule = 0.0, 0.0, []
    for b in beats:
        start, end = round(elapsed, 4), round(elapsed + b["dur"], 4)
        schedule.append(dict(id=b["id"], start=start, end=end))
        if b.get("vo"):
            name = Path(b["vo"]).stem
            begin = start + b["vo_at"]
            if begin < narration_end - .01:
                raise ValueError(f"Narration overlaps before {name}: {begin:.2f} < {narration_end:.2f}")
            narration_end = begin + durations[name]
        elapsed = end
    if elapsed > 120 or narration_end > elapsed:
        raise ValueError(f"Overlength: picture={elapsed:.2f}s, narration end={narration_end:.2f}s")
    return dict(version=11, duration=elapsed, beats=beats, schedule=schedule,
                provenance="clips/manifest.json", narration="vo11/provenance.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--run", type=Path, default=Path("D:/openHiggsfield/runs/demo_v11"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("storyboard.json"))
    args = parser.parse_args()
    board = make(args.root, args.run)
    payload = json.dumps(board, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    (args.run / "storyboard.json").write_text(payload, encoding="utf-8")
    print(f"{len(board['beats'])} beats, {board['duration']:.2f}s -> {args.output}")
    for beat in board["schedule"]:
        print(f"{beat['start']:6.2f}–{beat['end']:6.2f} {beat['id']}")


if __name__ == "__main__":
    main()
