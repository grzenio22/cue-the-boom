"""Write tools/demo/storyboard.json from the shots it uses: every time a noise word pops or a cue line is drawn
comes from that shot's receipt (when its noise is heard on the shot's timeline), not from a hand-typed number.

    python tools/demo/make_storyboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIO, CROP = "runs/demo/studio_live/studio.webm", [0.205, 0.0, 0.785, 1.0]
NOISE = {"blow_out": "pf.", "fire_burst": "KA-BOOM"}  # the word for each shot's noise (default: fwoooOOOM)


def receipt(shot: str) -> dict:
    return json.loads((ROOT / shot / "receipt.json").read_text())


def heard(shot: str) -> list[float]:
    """When each hit's noise is heard on the shot's timeline (s): its hit frame plus the effect's lead (a burst is
    shown a frame before its sound)."""
    rc = receipt(shot)
    lead = int(rc["alignment"].get("lead", 0))
    frames = [h["frame"] for h in rc["hits"]] if rc.get("hits") else [rc["alignment"]["hit_frame"]]
    return [(f + lead) / 24 for f in frames]


def real(shot: str, start: float, dur: float, caption: str | None, stamp: str, word: str | None = None) -> dict:
    """A real-footage shot, clean (no burned-in labels), its sound drawn under it and its noise word popping."""
    rc = receipt(shot)
    word = word or NOISE.get(rc["effect"]["id"], "fwoooOOOM")
    b = {"type": "clip", "src": f"{shot}/composite.mp4", "start": start, "dur": dur, "wave": True,
         "pops": [[round(t - start, 3), word] for t in heard(shot)], "stamp": stamp}
    if caption:
        b["caption"] = caption
    return b


def versus(shot: str, before: float, after: float, speed: float, caption: str, note: str | None = None) -> dict:
    t = heard(shot)[0]
    b = {"type": "versus", "shot": shot, "start": round(t - before, 3), "dur": round((before + after) / speed, 2),
         "speed": speed, "cue_at": round(t, 3), "labels": ["LTX-2.5, as it comes", "the finished shot"],
         "caption": caption, "stamp": "real stock clip (Pexels) · synthetic noise · the clip's own pixels, plus what changed"}
    if note:
        b["note"] = note
    return b


def main() -> None:
    pump = "runs/real/heldout/table_pumpkin__fwoooom"
    early = int(receipt(pump)["alignment"]["auto_shift"])
    rhythm = "runs/warp/toy_desk__pf-pf-ka-boom__fire_breath"
    beats = [
        real("runs/real/heldout/lanterns__kaboom", 0.35, 2.6, None,
             "real stock clip (Pixabay) · synthetic noise · LTX-2.5 + ainvfx-fluid, open weights"),
        {"type": "title", "dur": 1.6, "text": "Cue the Boom", "sub": "Draw where. Cue it with a sound. The scene reacts."},
        real("runs/real/heldout/trex_photo__fwoooom", 0.45, 2.6, "On a photo: LTX-2.5 brings it to life, timed by the noise.",
             "real stock photo (Pixabay), animated by LTX-2.5 · synthetic noise"),
        real("runs/real/final/pumpkins_tap__fwoooom", 0.45, 2.6, "On a clip: its own pixels stay, and the fire lights them.",
             "real stock clip (Pexels) · synthetic noise · the clip's own pixels, plus what changed"),
        real("runs/real/final/cake_candle__pf", 0.5, 2.6, "Blown out, it stays out.",
             "real stock clip (Pexels) · synthetic noise · the clip's own pixels, plus what changed"),
        versus(pump, 0.9, 1.5, 1.0, f"The same render, shifted {early} frames: LTX-2.5 started it early, CueSync moved it onto the noise."),
        versus(pump, 0.75, 0.35, 1 / 3, f"Slowed down: the render {early / 24:.2f} s early, the shot on the noise.", "slowed ×3"),
        {"type": "clip", "src": STUDIO, "start": 1.9, "dur": 3.2, "speed": 2.1, "crop": CROP,
         "side": "1. Pick a shot.\nIts spot is set;\ntap to move it.", "note": "the studio · sped up ×2",
         "stamp": "the studio's sample plates: stand-ins made with LTX-2.5"},
        {"type": "clip", "src": STUDIO, "start": 10.3, "dur": 5.0, "crop": CROP, "audio_src": "runs/demo/live_dino_final.mp4",
         "audio_from": 0.0, "audio_at": 0.47, "side": "2. Cue it: make the noise\nwhile the clip plays.\n(Here: a synthetic\nsample noise.)",
         "note": "real time · scripted take", "stamp": "a synthetic sample noise fed to the microphone input · no person yet"},
        {"type": "clip", "src": STUDIO, "start": 16.0, "dur": 2.0, "speed": 42, "crop": CROP,
         "side": "3. Into the scene:\nLTX-2.5 renders it from\nyour plate's frame.", "note": "82 s in total on one GPU · sped up ×42",
         "stamp": "LTX-2.5 image-to-video + ainvfx-fluid, with the painted guide"},
        {"type": "clip", "src": "runs/demo/live_dino_final.mp4", "start": 0.3, "dur": 3.8, "label": "that take",
         "side": "As long and as loud\nas the noise,\nthen it dies down.",
         "stamp": "stand-in plate (LTX-2.5) · synthetic noise · the shot from that take"},
        {"type": "clip", "src": STUDIO, "start": 106.6, "dur": 1.6, "crop": CROP, "side": "4. Drag the hit\nto re-time it.",
         "note": "real time"},
        {"type": "clip", "src": STUDIO, "start": 120.3, "dur": 3.6, "crop": CROP,
         "side": "Composited again\nfrom the same render:\nno new render.", "note": "12 s later"},
        {"type": "clip", "src": "runs/demo/studio_draw/studio.webm", "start": 4.8, "dur": 5.0, "speed": 2.3, "crop": CROP,
         "side": "Or draw where it goes,\nthen cue it.", "note": "sped up ×2.3",
         "stamp": "drawn in the studio · the cue: a synthetic sample noise fed to the microphone input"},
        {"type": "clip", "src": "runs/demo/live_pumpkin_final.mp4", "start": 0.8, "dur": 4.2, "label": "draw it + “fwoooOOOM”",
         "side": "The noise fills the drawing;\nwhen it stops, it dies down.",
         "stamp": "stand-in plate (LTX-2.5) · synthetic noise · the shot from that take"},
        {"type": "clip", "src": f"{rhythm}/composite.mp4", "start": 0.4, "dur": 3.4, "wave": True,
         "pops": [[round(t - 0.4, 3), w] for t, w in zip(heard(rhythm), ("pf", "pf", "KA-BOOM"))],
         "side": "A rhythm: every hit\nthe model keeps apart\nis re-timed to its beat.",
         "stamp": "stand-in plate (LTX-2.5) · synthetic noise · re-timed hit by hit"},
        {"type": "clip", "src": "runs/real/heldout/lanterns__kaboom/breakdown.mp4", "start": 0.4, "dur": 4.0,
         "caption": "Your sound sets the hit. LTX-2.5 renders it; CueSync moves the real onset onto your cue."},
        {"type": "image", "src": "media/timing.png", "dur": 5.0, "zoom": 1.02,
         "caption": "47 of 54 measurable hits land within a frame of the cue (47 of all 64 asked for). Straight from the model: 4.",
         "stamp": "marked by eye on 50 renders, 19 of them real footage · all cued by synthetic sample noises"},
        {"type": "image", "src": "media/comfyui-scene-workflow.png", "dur": 1.5, "zoom": 1.0,
         "caption": "The same shot as one ComfyUI workflow."},
        {"type": "image", "src": "media/comfyui-scene-workflow.png", "crop": [0.0, 0.0, 0.662, 0.489], "dur": 2.2, "zoom": 1.04,
         "caption": "Your cue and your plate in: CueSync Analyze Sound, CueSync Paint Scene."},
        {"type": "image", "src": "media/comfyui-scene-workflow.png", "crop": [0.534, 0.535, 0.955, 0.72], "dur": 1.8, "zoom": 1.04,
         "caption": "Code MIT / Apache-2.0; open weights only, through the Comfy SDK."},
        {"type": "credits", "dur": 5.0, "title": "Cue the Boom", "lines": [
            "LTX-2.5 · Lightricks",
            "ainvfx-fluid · AInVFX / Adrien Toupet",
            "SAM 2.1 · Meta",
            "ComfyUI + Comfy SDK · Comfy Org",
            "Real footage · Pexels, Pixabay, Wikimedia Commons",
            "  · authors in samples/CREDITS.md",
            "Open weights only, self-hosted on a rented GPU",
            "Some sample plates: stand-ins made with LTX-2.5",
            "Noises: synthetic, until real ones are recorded",
        ], "url": "github.com/grzenio22/cue-the-boom"},
    ]
    out = ROOT / "tools/demo/storyboard.json"
    out.write_text(json.dumps({"beats": beats}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(beats)} beats, {sum(float(b.get('dur', 3)) for b in beats):.1f} s -> {out}")


if __name__ == "__main__":
    main()
