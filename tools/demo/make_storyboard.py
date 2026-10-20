"""V12 contest cut: generation becomes a reusable voice instrument (<=120 s).

Edit plan: cold open with recorded-voice cue; moving raw fire -> hand with the
name over the picture; 18 s unified proof triptych (three complete voice takes,
all together, quiet/loud); readable actual ComfyUI model/LoRA/output nodes and
Comfy Python SDK role; matching fire -> hand breakdown; walk; 4 s procedural
lightning/water; sequential lanterns; dinosaur; expanded recorded Studio with
its full finished fire; explicit contribution over pipeline; qualified laptop
invitation, installation, repository and credits.

All footage is reused read-only from demo_v11/clips. Only cards and Kokoro-82M
narration are new. No footage rebuild or model generation is needed. Measured
phrase captions preserve the complete contribution recording. Full voice takes
and the Studio result are never shortened to meet the time limit.

Run with the project Python and this checkout's cuesync/src on PYTHONPATH:
  make_v11_voice.py --lines tools/demo/vo12_lines.json
  make_v11_voice.py --lines tools/demo/vo12_lines.json --only new12 --speed 1.15
  v12_cards.py
  make_storyboard.py
  build_demo.py tools/demo/storyboard.json -o D:/openHiggsfield/runs/demo_v12/demo_v12.mp4
  check_v11.py --video D:/openHiggsfield/runs/demo_v12/demo_v12.mp4
The second narration pass gives the complete contribution a natural 14.4 s.
Final beat groups: 00:00 hook; 00:08 moving introduction; 00:18 proof;
00:36 workflow; 00:48 compositing and light; 00:57 walk/flamethrower;
01:02.25 procedural effects; 01:06.25 lanterns/dinosaur; 01:16.125 Studio;
01:35.125 contribution; 01:49.75 invitation; 01:55.50 credits; 02:00 end.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FPS = 24
CLAIM = "1 LTX generation · 3 performances · 0 additional AI generations"


def make(root: Path, run: Path) -> dict:
    clips = root / "runs/demo_v11/clips"
    cards, voice = run / "cards", run / "vo12"
    durations = json.loads((voice / "durations.json").read_text(encoding="utf-8"))
    spoken = json.loads((run / "vo_lines.json").read_text(encoding="utf-8"))
    mine = "Author's phone footage + own voice · fire rendered with LTX-2.5 + ainvfx-fluid"
    sample = "Author's phone footage · sample noise (synthetic) · procedural CPU effect"
    studio = "Recorded Studio · author's clip + voice · included LTX effect · GPU offline"

    def shot(name):
        return str(clips / name / "composite.mp4")

    def say(name, at=.2, text=None):
        return dict(vo=str(voice / f"{name}.wav"), vo_at=at, vo_gain=1.3,
                    vo_text=text or spoken[name].replace("Comfy UI", "ComfyUI"))

    tall = [0, .18, 1, .96]
    panels = [
        dict(src=shot("proof_boom"), start=0, dur=3.5, crop=tall, label="BOOM",
             audio_src=str(clips / "proof_boom.wav"), pops=[[2.67, "BOOM"]]),
        dict(src=shot("proof_pf_pf_boom"), start=0, dur=3.5, crop=tall, label="pf · pf · BOOM",
             audio_src=str(clips / "proof_pf_pf_boom.wav"), pops=[[1.21, "pf"], [1.58, "pf"], [2.67, "BOOM"]]),
        dict(src=shot("proof_pf_x4"), start=0, dur=3.5, crop=tall, label="pf pf pf PF",
             audio_src=str(clips / "proof_pf_x4.wav"), pops=[[1.04, "pf"], [1.58, "pf"], [2.04, "pf"], [2.50, "PF"]]),
    ]
    proof = dict(panels=panels, headline=CLAIM, element_src=str(clips / "how_raw.mp4"), element_at=2.92,
                 stamp="Same hand footage + one LTX-2.5 element · author's recorded pf and BOOM")
    beats = [
        dict(id="cold_open", type="clip", src=shot("explosion_palm"), start=0, dur=8, fill=True,
             portrait_backdrop=True, pops=[[1.46, "BOOM"]], pop_x=1.35, pop_y=.55,
             cue_text="the author's recorded voice: BOOM", cue_at=1.32, cue_dur=2.3,
             source_gain=.2, audio_src=str(root / "runs/takes/plates/IMG_6666.mp4"), audio_gain=10,
             stamp="Author's hand + voice · LTX-2.5 fireball · CPU compositing, lighting and shock effects"),
        dict(id="intro_raw", type="clip", src=str(clips / "how_raw.mp4"), start=2.4, dur=4,
             speed=.225, fill=True, portrait_backdrop=True, crop=[.22, .06, .82, .94], headline="Cue the Boom",
             subline="Generate once. Perform again.",
             stamp="Raw LTX-2.5 fire · ainvfx-fluid IC-LoRA · shown in slow motion", **say("once12")),
        dict(id="intro_finished", type="clip", src=shot("proof_boom"), start=2.4, dur=6,
             speed=.15, fill=True, portrait_backdrop=True, headline="Cue the Boom",
             subline="Your voice performs the effect.", stamp="The same fire, carried on the hand · slow motion"),
        dict(id="performance_turns", type="turns", dur=10.5, audio_gain=4.5,
             subline="Listen to each complete performance", **proof),
        dict(id="comparison_together", type="compare", dur=3.5, audio="mix", audio_gain=4.5,
             subline="All three together · the same generated element", **proof),
        dict(id="small_and_big", type="compare", dur=4, loop=3.5, audio=1, audio_gain=4.5, highlight=1,
             subline="Quiet pf → small. Loud BOOM → big.", **proof, **say("size12", at=0)),
        dict(id="comfy_model", type="image", src=str(cards / "comfy_model.png"), dur=6, zoom=1,
             **say("ltx12", at=.5, text="LTX-2.5, with the fluid LoRA, makes this fire in ComfyUI.")),
        dict(id="comfy_output", type="image", src=str(cards / "comfy_output.png"), dur=4, zoom=1,
             **say("sdk12", at=.7)),
        dict(id="workflow_fire", type="clip", src=str(clips / "how_raw.mp4"), start=2.4, dur=2,
             speed=.45, crop=[.22, .06, .82, .94], fill=True, portrait_backdrop=True,
             chip="The generated LTX fire", stamp="The same stored output · slow motion"),
        dict(id="how", type="triptych", srcs=[str(clips / "how_raw.mp4"), str(clips / "how_carried.mp4"),
             shot("how_light_off"), shot("proof_boom")], start=2.4, dur=6, speed=.15,
             labels=["LTX-2.5 over black", "Carried on the hand", "Light off", "Finished + light"], audio=3,
             stamp="Same generated element · slow motion · track, matte, composite and light", **say("how12", at=.65)),
        dict(id="light", type="triptych", srcs=[shot("how_light_off"), shot("proof_boom")],
             labels=["Light off", "Light on · from the fire's own image"], start=2.4, dur=3, speed=.3,
             audio=1, stamp="The same tracked hand and LTX element · slow motion to show the light"),
        dict(id="walk", type="clip", src=shot("flame_walk"), start=3, dur=3.25, fill=True, portrait_backdrop=True,
             chip="fffff · carried through a walk, uncut", stamp=mine),
        dict(id="flamethrower", type="clip", src=shot("flamethrower"), start=1.3, dur=2, fill=True,
             portrait_backdrop=True, chip="fwoooom · fingertip flamethrower", stamp=mine),
        dict(id="lightning_point", type="clip", src=shot("lightning_point"), start=1.5, dur=2, fill=True,
             portrait_backdrop=True, source_gain=.35, chip="Procedural CPU effects", stamp=sample,
             **say("cpu12", at=.1)),
        dict(id="water", type="clip", src=shot("water_orb"), start=.25, dur=2, fill=True, portrait_backdrop=True,
             source_gain=.35, chip="Procedural CPU effects", stamp=sample),
        dict(id="lanterns", type="clip", src=shot("scene_lanterns"), start=0, dur=6.5, wave=True,
             marks=[[1.0, .42, .72, "1"], [1.67, .64, .58, "2"], [2.38, .43, .30, "3"]],
             pops=[[1.0, "pf"], [1.67, "pf"], [2.38, "BOOM"]], headline="Each hit lights the next lantern",
             subline="In turn: 1 → 2 → 3 · pf · pf · BOOM",
             stamp="Pixabay footage, first frame held · real CC0 voices (Freesound) · LTX-2.5 fire",
             **say("scene12", at=3.1)),
        dict(id="dino", type="clip", src=shot("scene_dino"), start=.25, dur=4.25, fill=True,
             chip="fwoooom · into the scene", stamp="Real stock footage (Pexels) · sample noise (synthetic) · LTX-2.5 render"),
        dict(id="upload", type="clip", src=str(clips / "studio.mp4"), start=1, dur=4, fill=True,
             crop=[0, .03, 1, .78], chip="1. Upload a clip", stamp=studio, **say("upload12")),
        dict(id="tap", type="clip", src=str(clips / "studio.mp4"), start=6.5, dur=4, fill=True,
             crop=[.078, .208, .578, .583], chip="2. Tap the palm, drag to aim", stamp=studio, **say("tap12")),
        dict(id="perform", type="clip", src=str(clips / "studio.mp4"), start=31.5, dur=5, fill=True,
             crop=[.078, .5, .634, .917], chip="3. Record your voice", audio_src=str(clips / "studio_say_own.wav"),
             audio_from=1.73, audio_gain=4, audio_replace=True, wave=True, wave_box=[.54, .16, .40, .12],
             pops=[[2.37, "pf"], [2.77, "pf"], [3.77, "BOOM"]], pop_x=.76, pop_y=.57, stamp=studio),
        dict(id="make", type="clip", src=str(clips / "studio.mp4"), start=36.5, dur=2.5, fill=True,
             crop=[.06, .27, .79, .82], chip="4. Make the shot", stamp=studio, **say("make12", at=.15)),
        dict(id="result", type="clip", src=str(clips / "studio_result.mp4"), start=0, dur=3.5, fill=True,
             portrait_backdrop=True, crop=[0, .18, 1, .84],
             chip="The finished shot · included LTX effect", stamp=studio),
        dict(id="innovation", type="image", src=str(cards / "pipeline.png"), dur=14, zoom=1, **say("new12", at=.1)),
        dict(id="try", type="image", src=str(cards / "try_it.png"), dur=5.75, zoom=1, **say("try12", at=.1)),
        dict(id="credits", type="image", src=str(cards / "credits.png"), dur=4.5, zoom=1,
             audio_src=str(root / "runs/demo_v10/credits_crackle.wav"), audio_from=.8, audio_gain=.3),
    ]
    # Keep full narration tails; trim only dinosaur time if needed for 120 s.
    by_id = {b["id"]: b for b in beats}
    for bid, key in (("innovation", "new12"), ("try", "try12")):
        by_id[bid]["dur"] = max(by_id[bid]["dur"], math.ceil((durations[key] + .2) * FPS) / FPS)
    excess = sum(b["dur"] for b in beats) - 120
    if excess > 0:
        by_id["dino"]["dur"] -= math.ceil(excess * FPS) / FPS
        if by_id["dino"]["dur"] < 2.5:
            raise ValueError("Narration leaves too little time for the retained dinosaur demonstration")
    phrase_path = voice / "caption_phrases.json"
    if phrase_path.exists():
        phrases = json.loads(phrase_path.read_text(encoding="utf-8"))
        for b in beats:
            if b.get("vo") and Path(b["vo"]).stem in phrases:
                b["vo_captions"] = [[a, z, s.replace("Comfy UI", "ComfyUI").replace("LTX two point five", "LTX-2.5")]
                                    for a, z, s in phrases[Path(b["vo"]).stem]]

    elapsed, narration_end, schedule = 0.0, 0.0, []
    for b in beats:
        b["dur"] = round(b["dur"] * FPS) / FPS
        start, end = round(elapsed, 6), round(elapsed + b["dur"], 6)
        schedule.append(dict(id=b["id"], start=start, end=end))
        if b.get("vo"):
            name = Path(b["vo"]).stem
            begin = start + b["vo_at"]
            if begin < narration_end - .01:
                raise ValueError(f"Narration overlaps before {name}: {begin:.2f} < {narration_end:.2f}")
            narration_end = begin + durations[name]
        elapsed = end
    if elapsed > 120.00001 or narration_end > elapsed:
        raise ValueError(f"Overlength: picture={elapsed:.2f}s, narration end={narration_end:.2f}s")
    return dict(version=12, duration=elapsed, beats=beats, schedule=schedule,
                provenance=str(clips / "manifest.json"), narration="vo12/provenance.json",
                source_policy="Read-only v11 clips reused; no new AI generations or footage rebuilds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("D:/openHiggsfield"))
    parser.add_argument("--run", type=Path, default=Path("D:/openHiggsfield/runs/demo_v12"))
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
