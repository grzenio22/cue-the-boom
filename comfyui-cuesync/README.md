# CueSync for ComfyUI

Put fire, smoke or steam into your own shot where it should come from (a point, or a drawing) and when it should happen: on a sound (your voice, or a sound already in the clip) or on a press-and-hold. Nodes around the LTX-2.5 + [ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid) IC-LoRA graph.

**Scene mode** (the effect rendered into the plate, so the scene reacts):

```
Load Audio (your voice) or the plate's own audio ──> CueSync Analyze Sound ─┐      (or CueSync Hold Cue, no sound)
Load Video (your plate) ─────────────────────────────────────────────────> CueSync Paint Scene ──> guide video ──> LTX-2.5 + ainvfx-fluid
                                                                                   │                start frame ─> LTXVImgToVideoInplace (strength 1)
                                                                                   │                prompt ──────> text encoder
                                                                                   └─ plan ──> CueSync Scene Shot <── render ──> Save Video
```

**Element mode** (rendered on black, composited onto the untouched plate): Analyze Sound ──> Paint Guide ──> LTX-2.5 + ainvfx-fluid ──> Composite (with Object Matte).

| Node | In | Out | What it does |
|---|---|---|---|
| **CueSync Analyze Sound** | AUDIO, offset_s | CUESYNC_CUE, envelope IMAGE | Onsets, the hit, an absolute loudness envelope per video frame, and when the sound is actually there. Feed it your performance, or the plate's own soundtrack (Get Video Components) to lock the effect to a snap, a click or a slam in the clip. |
| **CueSync Hold Cue** | holds ("1.0-2.5, 3.2-3.6"), duration | CUESYNC_CUE | Press-and-hold timing when there is no sound: each press is a hit, each hold how long it lasts. |
| **CueSync Paint Scene** | cue, plate VIDEO, effect, x, y, direction, subject; optional strokes (JSON), describe | VIDEO (guide), IMAGE (start frame), STRING (prompt), CUESYNC_SCENE | Paints the control the way the LoRA was trained: the plate in six flat colour zones on every keyframe, held over its latent until the hit, with the effect painted at its source and direction (or filling your drawing), timed and sized by the cue. The start frame goes to LTXVImgToVideoInplace at full strength. |
| **CueSync Scene Shot** | plate VIDEO, render VIDEO, plan, (voice AUDIO) | VIDEO | Measures where the render's effect visibly starts and moves it onto the cue. A still plate becomes the render; a moving plate keeps its own pixels and gets only what changed (checked; it falls back to the render if the render drifted). Adds a flash and a camera shake on blasts. |
| **CueSync Paint Guide** | cue, effect, x, y, size, tail | VIDEO, STRING, CUESYNC_PLAN | Element mode: blobs on black that appear on the hit, grow with the loudness and die away when it stops. |
| **CueSync Object Matte** | IMAGE, x, y or plan | MASK | Element mode: SAM 2.1 (Apache-2.0) segments the object under the point; it stays in front of the effect. |
| **CueSync Composite** | plate, element, cue, plan, (voice, mask) | VIDEO | Element mode: aligns, gates and composites in linear light onto the untouched plate. |

Whole-shot workflows: [`workflows/cue_the_boom_scene.api.json`](../workflows/cue_the_boom_scene.api.json) (scene mode) and [`workflows/cue_the_boom.api.json`](../workflows/cue_the_boom.api.json) (element mode). They need the LTX-2.5 models and ainvfx-fluid listed in [`build/model-files.md`](../build/model-files.md).

## Install

Copy (or clone) this folder into `ComfyUI/custom_nodes/`, then install its dependencies into ComfyUI's Python:

```bash
pip install -r custom_nodes/comfyui-cuesync/requirements.txt
pip install "git+https://github.com/grzenio22/cue-the-boom#subdirectory=cuesync"
```

The effects (fireball, fire breath, fire burst, flame plume, flare-up, blow it out, smoke puff, steam) are JSON files in the repository's `effects/` folder; set `CUESYNC_EFFECTS` to use your own.

Licence: MIT (this node pack), Apache-2.0 (cuesync). Models: LTX-2.5 by Lightricks and ainvfx-fluid by Adrien Toupet / AInVFX (LTX-2.x Community License), SAM 2.1 by Meta (Apache-2.0).
