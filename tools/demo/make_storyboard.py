"""The demo's storyboard (v10, the cut submitted). Its footage is the author's own takes and their renders, made with
this repository's tools and kept outside it (runs/: the phone clips, runs/takes/rhythms.py's and nolight.py's
performances, the studio recorded with tools/demo/record_studio.py --hd); the cards come from tools/demo/*_card.py.

Demo v9, after the v8 reviews: the author's portrait takes fill the frame (no pillarbox), the hero's one render
performed three ways right after it, the studio legible (recorded at twice the pixels, on the author's own clip:
upload, tap the palm, perform, make the shot), light off / light on, and a readable pipeline.
Demo v10: the author's well-lit takes of 2026-10-19 rendered for their own hands: a punch's BOOM opens it, a held
"fffff" lights a flame in an opening palm, and one uncut take carries a flame in the palm from a hallway into a kitchen.
Writes tools/demo/storyboard.json for tools/demo/build_demo.py.

    python tools/demo/make_storyboard.py && python tools/demo/build_demo.py tools/demo/storyboard.json -o runs/demo_v10/demo_v10.mp4
"""

import json
from pathlib import Path

V10, V9, V8, V7, V6 = (Path("runs/demo_v10"), Path("runs/demo_v9"), Path("runs/demo_v8"), Path("runs/demo_v7"),
                       Path("runs/demo_v6"))
VO, VO_DIR = {}, {}
for d in (V6 / "vo", V7 / "vo7", V8 / "vo8", V9 / "vo9", V10 / "vo10"):
    for k, s in json.loads((d / "durations.json").read_text(encoding="utf-8")).items():
        VO[k], VO_DIR[k] = s, d
G, G2, R = "runs/takes/gpu", "runs/takes/gpu2", "runs/takes/rhythms"
HERO = f"{G}/IMG_6658_fire_burst_s2_c5/composite.mp4"
MINE = "the author's own hand and voice, filmed on an iPhone · the fire rendered once over black (LTX-2.5 + ainvfx-fluid)"
STOCK = "real stock clip (Pexels) · CC0 voices · the fire rendered once over black (LTX-2.5 + ainvfx-fluid)"
LANTERNS = "real stock clip (Pixabay), held on its first frame · real people's voices (CC0, Freesound) · LTX-2.5 + ainvfx-fluid"
STUDIO = "the studio in a browser, recorded: the author's own clip, the GPU offline (a fireball rendered ahead with LTX-2.5)"
OWN = f"{V9}/studio_own3/studio.mp4"  # (1920x1440: the edit crops it)
SRC = {"1": (0.42, 0.72), "2": (0.64, 0.58), "3": (0.43, 0.30)}
WIDE = 9 / 16 * 1080 / 1920  # a 16:9 window of a 1080x1920 phone clip, as a share of its height


def band(y0: float) -> list[float]:
    return [0, y0, 1, round(y0 + WIDE, 4)]


TALL = [0, 0.36, 1, 0.92]  # the hand and the fire above it, for a panel
beats = [
    # 1. the hook: the author's punch, the author's BOOM, filling the frame, no narration over it
    dict(type="clip", src=f"{G2}/IMG_6663_fire_burst/composite.mp4", start=2.15, dur=1.35, fill=True, wave=True,
         crop=band(0.31), pop_x=0.78, pop_y=0.5, pops=[[0.52, "BOOM"]], headline="Cue the Boom",
         subline="Make the noise. Throw fire.", stamp=MINE),
    # 2. right after it, the same render performed three ways
    dict(type="turns", headline="One render. Perform it again.", subline="0 new renders", vo="again9", vo_at=0.1,
         panels=[dict(src=f"{R}/pf_pf_boom/composite.mp4", start=0.9, dur=2.3, crop=TALL, label="pf · pf · BOOM",
                      pops=[[0.31, "pf"], [0.68, "pf"], [1.77, "BOOM"]]),
                 dict(src=f"{R}/pf_x4/composite.mp4", start=0.8, dur=2.6, crop=TALL, label="pf pf pf PF",
                      pops=[[0.24, "pf"], [0.78, "pf"], [1.24, "pf"], [1.7, "PF"]])],
         stamp="the author's own hand · the author's pf and BOOM, re-cut into new rhythms · one LTX-2.5 render"),
    # 3. first person: a roar, a flamethrower from the fingertip; a finger gun
    dict(type="clip", src=f"{G}/IMG_6653_fire_breath_c5/composite.mp4", start=1.55, dur=0.9, fill=True, wave=True,
         crop=band(0.29), pops=[[0.0, "fwooooOOM"]], stamp=MINE),  # only the stretch where the flame leaves the finger
    dict(type="clip", src=f"{G2}/IMG_6660_flame_plume/composite.mp4", start=1.0, dur=2.4, fill=True, wave=True,
         crop=band(0.5), pops=[[0.4, "fffff"]], stamp=MINE),  # a fist opens, a held fffff: a flame in the palm
    # 4. how: rendered once over black, carried on the tracked hand, light off / light on
    dict(type="triptych", srcs=[str(V9 / f"{n}_slow.mp4") for n in ("how_render", "how_carried", "nolight", "hero")],
         labels=["LTX-2.5, over black", "on your hand", "light off", "light on"], start=0.4, dur=3.6, audio=3,
         caption="Rendered once, carried on the tracked hand, matted, and lit by its own fire.",
         vo="how9", vo_at=0.0, stamp=MINE + " · half speed"),
    # 4b. one take, uncut: the camera walks from a hallway into a kitchen, the flame stays in the palm
    dict(type="clip", src=f"{G2}/IMG_6668_flame_plume_comp/composite.mp4", start=1.9, dur=4.8, fill=True, wave=True,
         crop=band(0.42), headline="One take, uncut", subline="It stays in your palm.", stamp=MINE + " · the camera walking"),
    # 6. the studio on the author's own clip, each step full frame
    dict(type="clip", src=OWN, start=7.0, dur=2.6, fill=True, crop=[0.078, 0.208, 0.578, 0.583], chip="Tap your palm, drag to aim",
         vo="studio9a", vo_at=0.1, stamp=STUDIO),
    dict(type="clip", src=OWN, start=23.6, dur=2.6, fill=True, crop=[0.052, 0.181, 0.615, 0.603],
         chip="It follows your hand before anything is made", vo="studio9c", vo_at=0.2, stamp=STUDIO),
    dict(type="clip", src=OWN, start=33.0, dur=2.9, fill=True, crop=[0.078, 0.5, 0.634, 0.917], chip="Make the noise while it plays",
         pop_x=0.74, pop_y=0.52, audio_src=f"{V9}/studio_say_own.wav", audio_from=3.23, audio_gain=4.0, wave=True,
         wave_box=[0.53, 0.16, 0.42, 0.13], stamp_top=True,
         pops=[[0.87, "pf"], [1.27, "pf"], [2.27, "BOOM"]], stamp=STUDIO),
    dict(type="clip", src="runs/jobs/fd58657de8da/composite.mp4", start=0.8, dur=2.6, fill=True, wave=True, crop=band(0.42),
         chip="Your shot · no GPU, about a minute on a laptop", pops=[[0.53, "pf"], [0.95, "pf"], [1.99, "BOOM"]], stamp=STUDIO),
    # 7. the scene: one sound, one source
    dict(type="clip", src="runs/round10/shots/held_together__pf-pf-boom/composite.mp4", start=0.6, dur=5.0, wave=True,
         marks=[[t, *SRC[k], k] for t, k in ((0.4, "1"), (1.07, "2"), (1.78, "3"))],
         pops=[[0.4, "pf"], [1.07, "pf"], [1.78, "BOOM"]], headline="Or tap anything in your shot",
         subline="One sound, one source.", vo="scene7", vo_at=1.9, stamp=LANTERNS),
    # 8. how it's made, readable; then the ComfyUI graph itself, close up
    dict(type="image", src=str(V9 / "comfy.png"), dur=2.6, zoom=1.0, vo="pipe10b", vo_at=0.2),
    dict(type="image", src=str(V9 / "pipeline.png"), dur=3.0, zoom=1.0),
    # 9. end on the shot, and the claim
    dict(type="clip", src=f"{R}/pf_pf_boom/composite.mp4", start=0.5, dur=3.2, fill=True, wave=True, crop=band(0.42),
         pop_x=0.8, pop_y=0.5, pops=[[0.71, "pf"], [1.08, "pf"], [2.17, "BOOM"]], headline="Render the fire once.",
         subline="Your voice decides when, how big, how long.", stamp=MINE),
    dict(type="credits", dur=4.2, title="Cue the Boom", lines=[
        "LTX-2.5 · Lightricks    ainvfx-fluid IC-LoRA · AInVFX    Gemma text encoder · Google", "ComfyUI + Comfy SDK · Comfy Org",
        "Hands: MediaPipe · Google    RTMPose · OpenMMLab    SAM 2.1 · Meta",
        "The author's own hand, voice and phone · stock footage: Pexels, Pixabay", "Voices · rykenn, unfa and others (CC0, Freesound)",
        "Narration · Kokoro-82M, a synthetic voice    Foley · synthesised per shot",
        "Open weights only, self-hosted on one rented GPU", "Film your hand. Make the noise. Throw fire."]),
]

t, said_until = 0.0, 0.0
for b in beats:
    if b["type"] == "turns":
        b["dur"] = round(sum(float(q["dur"]) for q in b["panels"]), 3)
    if "vo" in b:
        name = b.pop("vo")
        start = max(t + float(b.pop("vo_at", 0.15)), said_until + 0.3)
        b["vo"], b["vo_at"] = str(VO_DIR[name] / f"{name}.wav"), round(start - t, 3)
        said_until = start + VO[name]
        if start - t > b["dur"]:
            print(f"  {name} starts after its beat ends ({start - t:.1f} s into a {b['dur']} s beat)")
    t += float(b["dur"])
over = said_until - t
(Path(__file__).parent / "storyboard.json").write_text(json.dumps({"beats": beats}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(beats)} beats, {t:.1f} s; narration ends at {said_until:.1f} s" + (f" ({over:.1f} s past the end!)" if over > 0 else ""))
