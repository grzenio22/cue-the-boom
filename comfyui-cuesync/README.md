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
| **CueSync Analyze Sound** | audio, offset_s | CUESYNC_CUE, envelope IMAGE | Onsets, the hit, an absolute loudness envelope per video frame, and when the sound is actually there. Feed it your performance, or the plate's own soundtrack (Get Video Components) to lock the effect to a snap, a click or a slam in the clip. |
| **CueSync Hold Cue** | holds ("1.0-2.5, 3.2-3.6"), duration_s | CUESYNC_CUE | Press-and-hold timing when there is no sound: each press is a hit, each hold how long it lasts. |
| **CueSync Paint Scene** | cue, plate, effect, x, y, aim, direction (used only with aim on), subject, max_seconds; optional strokes (JSON), describe | VIDEO (guide), IMAGE (start frame), STRING (prompt), CUESYNC_SCENE | Paints the control the way the LoRA was trained: the plate in six flat colour zones on every keyframe, held over its latent until the hit, with the effect painted at its source and direction (or filling your drawing), timed and sized by the cue. The start frame goes to LTXVImgToVideoInplace at full strength. |
| **CueSync Scene Shot** | plate, render, plan, max_seconds; optional voice | VIDEO | Measures where the render's effect visibly starts and moves it onto the cue. A still plate becomes the render; a moving plate keeps its own pixels and gets only what changed (checked; it falls back to the render if the render drifted). Adds a flash and a camera shake on blasts. |
| **CueSync Paint Guide** | cue, effect, x, y, width, height, tail, canvas | VIDEO, STRING, CUESYNC_PLAN | Element mode: blobs on black that appear on the hit, grow with the loudness and die away when it stops. |
| **CueSync Object Matte** | images, x, y, model; optional plan | MASK | Element mode: SAM 2.1 (Apache-2.0) segments the object under the point; it stays in front of the effect. |
| **CueSync Composite** | plate, element, cue, plan, max_seconds; optional voice, object_mask | VIDEO | Element mode: aligns, gates and composites in linear light onto the untouched plate. |

Whole-shot workflows: [`workflows/cue_the_boom_scene.json`](../workflows/cue_the_boom_scene.json) (scene mode; drag it onto the ComfyUI canvas) and [`workflows/cue_the_boom.json`](../workflows/cue_the_boom.json) (element mode), each with an `.api.json` twin for the Comfy SDK. Put your noise and your clip in `ComfyUI/input` as `voice.wav` and `plate.mp4`, or pick them in Load Audio and Load Video; to start from a sample, copy `samples/voices/fwoooom.wav` and `samples/plates/toy_desk.mp4` there and queue it as it is (Paint Scene is set up for that toy dinosaur). Plates are rendered at their own size, long side at most 1920 px (a 1080x1920 phone clip renders at 1088x1920).

**Requires:** ComfyUI v0.41 or later, [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo), and the LTX-2.5 models and ainvfx-fluid listed in [`build/model-files.md`](../build/model-files.md). A scene render fits a 32 GB card (tested with the GPU held to 31.5 GiB free: about 3 minutes a render, against about 1 minute on a 96 GB card).

## Install

Clone the whole repository into `ComfyUI/custom_nodes/` (or paste its URL into ComfyUI-Manager's "Install via Git URL"), next to [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo); the nodes then use the `cuesync` library and the effects right next to them:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/grzenio22/cue-the-boom
pip install -r cue-the-boom/requirements.txt
```

Use the Python that runs ComfyUI: with the Windows portable build, from `ComfyUI\custom_nodes` that is `..\..\python_embeded\python.exe -m pip install -r cue-the-boom\requirements.txt`.

To install this folder on its own instead, copy it into `custom_nodes/` and install `cuesync` with it (its `install.py` does this for ComfyUI-Manager):

```bash
pip install -r custom_nodes/comfyui-cuesync/requirements.txt
python custom_nodes/comfyui-cuesync/install.py
```

The standalone installer and package metadata select the same tested core revision. The installer resolves its dependencies, then replaces only `cuesync` so an older checkout with the same version number cannot be retained. It does not force-reinstall ComfyUI's dependency stack. A full repository checkout uses its adjacent core instead.

The effects (fireball, fire breath, fire burst, flame plume, flare-up, blow it out, smoke puff, steam) are the JSON files in the repository's `effects/` folder, and the install above ships them inside `cuesync`. To add your own, put JSON files in a folder and point `CUESYNC_EFFECTS` at it; name each file after the `id` inside it (`my_fire.json` holds `"id": "my_fire"`).

Licence: MIT (this node pack), Apache-2.0 (cuesync). Models: LTX-2.5 by Lightricks and ainvfx-fluid by Adrien Toupet / AInVFX (LTX-2.x Community License), SAM 2.1 by Meta (Apache-2.0).
