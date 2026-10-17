# Cue the Boom

**Draw where. Cue it with a sound. The scene reacts.**

[Try it without a GPU](#try-it) · [How it works](#how-it-works) · [How well it lands](#how-well-it-lands)

Tap or draw where fire, smoke or steam should come from in your clip, then make the noise ("KA-BOOM", "fwoooOOOM", "hissss"), or point at a sound that is already in the clip (a finger snap). The sound sets the frame it hits and how big and how long it gets. **LTX-2.5** with the **[ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid)** IC-LoRA by AInVFX (Adrien Toupet) renders it into the scene, so it lights the room and leaves its mark: a blown-out candle stays out. Then the render's real onset is measured and moved onto your cue: on 31 test shots, 22 of 24 single hits land within a frame (42 ms) of the cue; straight from the model, none do.

**New here:** a performed sound becomes the hit frame and growth curve of a generated effect, painted in the control format the LoRA was trained on; the render's real onset is then measured and corrected, with the error written into every shot's receipt.

![The scene reacts](media/hero.jpg)
<sub>Scene-mode shots. The plates are stand-ins generated with LTX-2.5 and the noises are synthetic, until real phone footage is added; the effects are rendered into each plate by LTX-2.5 + ainvfx-fluid, from the cue, and timed by CueSync.</sub>

An entry for Comfy's **Open Source Higgsfield** contest, built on the Comfy Developer Platform (the Comfy Python SDK against a self-hosted Comfy API), on open weights only.

## Why

Preset effects decide for you where and when the effect happens; prompt-based video edits make you type the timing ("at about 2.2 seconds, on his finger snap"). Cue the Boom keeps you in charge of the two things a shot is made of:

- **Where.** Tap the thing it comes from (a toy's mouth, a candle's wick), drag to aim it, or draw it (flames along the top of a pumpkin).
- **When, and how much.** A sound pins the frame, and its loudness and length shape the effect. Your voice is the fun way to do it; the sound already in your clip is the editor's way (hits are cut to sound anyway); press-and-hold works in a quiet room.

The effect is rendered into the scene, so it lights and changes what is around it. On a moving clip CueSync adds only what the effect changed and checks the result (if the check fails, the shot says so and uses the render); the sample plates so far are still.

## Try it

Python 3.10+ (CI runs 3.12); Node 24 for the studio.

**In a minute, no GPU, no npm:** re-time a shipped scene shot, and re-measure the timing of the shipped shots.

```bash
git clone https://github.com/grzenio22/cue-the-boom && cd cue-the-boom
pip install "./cuesync[server]"
cuesync retime --shot samples/made_ahead/scene/jackolantern_kitchen__ka-boom__fire_burst --plate samples/plates/jackolantern_kitchen.mp4 \
  --voice samples/voices/ka-boom.wav --effect fire_burst --x 0.48 --y 0.39 --shift 6 -o runs/retimed
python tools/measure_timing.py samples/made_ahead/scene -o runs/timing
```

`runs/retimed/final.mp4` is the shot moved a quarter of a second later, composited again from the same render; `runs/timing/timing.md` is the timing table for those shots.

**The studio, with no GPU.** The sample scene shots in `samples/made_ahead` play straight away (and you can drag their hit to re-time them, in seconds); on your own clip, instant mode places pre-rendered LTX-2.5 elements, timed by your cue. In two terminals:

```bash
cuesync serve --instant-only           # terminal 1: the render API, http://127.0.0.1:8765
cd web && npm install && npm run dev   # terminal 2: the studio, http://localhost:3000
```

1. Pick a sample shot or your own clip (phone held still, 1-15 s; the shot uses the first 10 s).
2. Tap where it comes from, or draw it.
3. Cue it: say it into the microphone (or pick a sample noise), use the clip's own sound, or press and hold while it plays.
4. Make the shot. **Into the scene** renders it into your shot (GPU); **Instant** places a pre-rendered element (no GPU).

**From the command line:**

```bash
cuesync shot --instant --plate samples/plates/pumpkin_kitchen.mp4 --voice samples/voices/ka-boom.wav --effect fireball --x 0.5 --y 0.56
```

`runs/shot/` then holds `final.mp4` (the shot with a live ribbon of the cue), `breakdown.mp4` (plate | guide | render | shot), `composite.mp4` (the clean shot) and `receipt.json` (what was measured and what made it).

## Into the scene (GPU)

The full pipeline needs ComfyUI with LTX-2.5 22B (int8) and ainvfx-fluid. A render fits a 32 GB card (tested with the GPU held to 31.5 GiB free: about 3 minutes a render); we run it on a rented RTX PRO 6000 (96 GB), about a minute a render:

```bash
python infra/pod/podctl.py up && python infra/pod/podctl.py setup && python infra/pod/podctl.py sync
python infra/pod/podctl.py tunnel                                   # in another terminal
pip install "./cuesync[server,comfy]"
COMFY_BASE_URL=http://127.0.0.1:8189 cuesync serve
```

`podctl` creates, sets up and terminates a disposable pod through the Runpod API; a watchdog terminates it when idle ([infra/pod/README.md](infra/pod/README.md)). A scene shot takes about two minutes, most of it the LTX-2.5 render.

From the command line, add `--scene`:

```bash
# the toy dinosaur breathes fire to the left, as long and as hard as your "fwoooom"
cuesync shot --scene --plate samples/plates/toy_desk.mp4 --voice samples/voices/fwoooom.wav --effect fire_breath \
  --x 0.375 --y 0.485 --direction -90 --subject "the small plastic toy dinosaur on the bedside table"
# lock it to the snap that is already in the clip
cuesync shot --scene --plate samples/plates/candle_snap.mp4 --cue clip --effect blow_out --x 0.53 --y 0.52
# draw it, time it with a hold, say what happens
cuesync shot --scene --plate samples/plates/pumpkin_kitchen.mp4 --cue hold --hold 1.0-2.5 --effect flame_plume \
  --strokes samples/strokes/pumpkin_top.json --describe "the small pumpkin catches fire and flames spread along its top"
```

**In ComfyUI.** [`workflows/cue_the_boom_scene.json`](workflows/cue_the_boom_scene.json) (drag it onto the canvas) is the whole scene shot as one workflow: your cue and your plate go in, the finished shot comes out. [`cue_the_boom_scene.api.json`](workflows/cue_the_boom_scene.api.json) is the same graph for the API.

![The scene shot in ComfyUI: your cue, CueSync Paint Scene, LTX-2.5 with ainvfx-fluid from your plate, CueSync Scene Shot](media/comfyui-scene-workflow.png)

Install: clone this repository into `ComfyUI/custom_nodes/` (or paste its URL into ComfyUI-Manager's "Install via Git URL"), install its `requirements.txt` into ComfyUI's Python, and get the models in [build/model-files.md](build/model-files.md):

```bash
cd ComfyUI/custom_nodes && git clone https://github.com/grzenio22/cue-the-boom && pip install -r cue-the-boom/requirements.txt
```

The nodes are documented in [comfyui-cuesync/README.md](comfyui-cuesync/README.md).

## How it works

![How it was made](media/how-it-was-made.jpg)

1. **Read the cue** ([`cues.py`](cuesync/src/cuesync/cues.py), [`audio.py`](cuesync/src/cuesync/audio.py)). Onsets, the hits of a rhythm ("pf, pf, KA-BOOM"), an absolute loudness envelope per video frame, and when the sound is actually there. From your voice, from the clip's own soundtrack, or from press-and-hold. Its shape (a hit, a rhythm, a sustained sound, a breath, a hiss) picks the effect when you let it ([`gestures.py`](cuesync/src/cuesync/gestures.py)): a sample plate knows what it shows; on your own clip you say what it is (a candle, a jack-o'-lantern, a toy's mouth, a hot drink), since nothing here recognises objects.
2. **Paint the scene** ([`scene.py`](cuesync/src/cuesync/scene.py)). ainvfx-fluid was trained on controls that paint the whole frame in six flat colour zones, with the real first frame as conditioning. So CueSync paints your plate in its own zones on every keyframe and paints the effect over them: at its source, going its way (a jet out of a mouth that curls up as hot gas rises; a flare that grows out of the candle's own flame and settles back), or filling your drawing as far as your cue says. The scene is held over each 8-frame latent until the hit (a blank latent lets the model start the effect early), and every hit lands on a keyframe.
3. **Render into the scene.** LTX-2.5 image-to-video starts from your plate's frame at full strength, with the painted guide, through the Comfy SDK ([`runner.py`](cuesync/src/cuesync/runner.py)). The model lights the room, the table and the prop itself, and leaves what the effect leaves behind.
4. **Time it** ([`scene_shot.py`](cuesync/src/cuesync/scene_shot.py)). The first frame where the effect is visible is measured on the render and moved onto your cue (a burst shows a frame before its "BOOM": light beats sound). A rhythm moves by the middle error of the hits the model kept apart. A blast adds a warm flash at its peak and a camera shake just after it.
5. **Keep your footage** ([`transfer.py`](cuesync/src/cuesync/transfer.py)). A still plate becomes the render. A moving clip keeps its own pixels: the render is registered onto its own scene, and only what changed is added, in linear light; where the structure changed (the fire, smoke, debris) the render replaces the plate. The transfer is checked, and if the render drifted it is not used blindly; the shot says so.

Every shot writes a receipt: the cue, where the hits were painted, how early or late the model put them, how far they were moved, whether the transfer passed its check. LTX-2.5 can only take the guide on an 8-frame grid (a third of a second at 24 fps), and it nearly always starts the effect a few frames early, so this measure-and-move step is what puts the hit on your cue.

### How well it lands

Marked by eye frame by frame on the raw renders of 31 shots ([`tools/timing_annotations.json`](tools/timing_annotations.json), [`tools/measure_timing.py`](tools/measure_timing.py)):

| hits | within one frame (42 ms) of the cue: LTX-2.5 render as it comes → finished shot | mean error |
|---|---|---|
| all 36 | 3 → 26 | 4.6 → 1.1 frames |
| single hits (24) | 0 → 22 | 5.5 → 0.5 frames |
| hits in a rhythm (12) | 3 → 4 | 2.9 → 2.1 frames |
| renders the onset threshold was not fitted on (23) | 2 → 15 | 4.5 → 1.3 frames |

Of the 43 hits painted, 36 have an onset that can be marked; the other 7 are 6 rhythm hits the model merged into a neighbour and 1 start too faint to call. Every render the marks refer to ships with the repository (`samples/made_ahead/scene` and `samples/timing_set/`), so the whole table re-measures in under a minute:

```bash
python tools/measure_timing.py samples/made_ahead/scene samples/timing_set/before_fixes samples/timing_set/other_seeds samples/timing_set/dropped
```

All of these are LTX-2.5 stand-in plates: real phone footage is still untested.

![Timing: raw render vs finished shot](media/timing.png)

Before any of that, the take itself has to land where it was performed: [`tools/check_take_timing.py`](tools/check_take_timing.py) replaces the studio's microphone with a synthetic one that makes a noise at the moment the clip shows a chosen frame; in 8 takes the noise was found within 45 ms of it (6 within 20 ms). A synthetic microphone has no input latency; a real one reports its own, and the studio subtracts it.

The limits, measured the same way: in a rhythm LTX-2.5 often merges close hits into one (6 of 18 here); those are reported, not faked, and the rest move together. A flame dies down about 0.6 s after the voice stops (0.3 to 1.0 s).

**Element mode** (Advanced in the studio) is the other route: the effect is rendered alone on black and composited onto the untouched plate in linear light ([`composite.py`](cuesync/src/cuesync/composite.py)); the object stays in front through a GrabCut matte in the studio and the command line ([`matte.py`](cuesync/src/cuesync/matte.py)), or SAM 2.1 in the ComfyUI node.

## What is new, and what it builds on

What we found nobody else doing is the loop: a performed or in-clip sound becomes a hit frame and a growth curve, painted into an IC-LoRA control in the format the LoRA was trained on (the plate's own flat zones plus the effect), rendered by LTX-2.5 from the real plate frame, then the render's actual onset is measured and moved onto the cue, with the error written down; and for a moving clip, a checked transfer of only what changed. Each piece rests on prior work:

- **[ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid)** (AInVFX, Adrien Toupet) is the painted-control fluid model everything here is rendered with. Its author describes painting over a photo as "inspired by" the photo, not a plate matcher; CueSync's zones, timing and transfer are what keep the shot yours.
- **Sound-guided video effects** have been studied before (for example [Soundini](https://arxiv.org/abs/2304.06818)); voice as a timing and loudness control exists for sound effects (vocal-imitation tools such as Sketch2Sound); audio-reactive and onset nodes exist for ComfyUI; LTX-2.5 has its own audio-to-video.
- **Drawing where an effect goes** and **scene reaction** exist in other tools (region edits, image-to-video effect presets). Here they are the inputs and the output of the cue, not the novelty.

## Repository

| Path | What |
|---|---|
| `cuesync/` | The Python core and CLI: cues, scene guides, the Comfy SDK runner, timing and transfer, the element compositor, the render API (Apache-2.0) |
| `comfyui-cuesync/` | ComfyUI node pack: Analyze Sound, Hold Cue, Paint Scene, Scene Shot, plus the element-mode nodes |
| `workflows/` | ComfyUI workflows, canvas (`*.json`) and API (`*.api.json`) format (LTX-2.x Community License, derived from the ainvfx-fluid sample) |
| `web/` | The studio (Next.js) |
| `effects/` | The effects, as JSON |
| `samples/` | Sample plates (stand-ins), noises (synthetic), the scene shots made ahead of time, the timing evaluation renders and the instant-mode element library |
| `tools/` | Timing measurement, the take-timing check, the demo tools ([tools/README.md](tools/README.md)) |
| `infra/pod/` | Disposable Runpod GPU pods with a self-terminating watchdog |
| `build/model-files.md` | Exact models, licences and versions |

Tests: `pip install "./cuesync[server,dev]" && pytest cuesync/tests` (CPU only), plus `cd web && npm test`.

## Credits and licences

- **Code:** MIT, with `cuesync/` under Apache-2.0.
- **Workflows:** `workflows/` is under the LTX-2.x Community License (`workflows/LICENSE-LTX-2.x`).
- **Models:**
  - **LTX-2.5** by Lightricks (LTX-2.x Community License);
  - **ainvfx-fluid** by Adrien Toupet / AInVFX (LTX-2.x Community License): the painted-control fluid IC-LoRA this builds on;
  - **SAM 2.1** by Meta (Apache-2.0).
  - Rendered shots carry the LTX-2.x terms.
- **Software:** **ComfyUI** (GPL-3.0), the **Comfy Python SDK** and **comfy-api-proxy** by Comfy Org, **ComfyUI-LTXVideo** by Lightricks, **Next.js**, **PyAV/FFmpeg**, **OpenCV**.
- **Samples:** the sample noises (and the snap in `candle_snap.mp4`) are synthetic. The sample plates are stand-ins generated with LTX-2.5.
