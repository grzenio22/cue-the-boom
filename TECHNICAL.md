# Cue the Boom: how it works, in detail

The [README](README.md) is the short version. This is the long one: the pipeline step by step, the hand, the timing
evaluation, setting the helpers up by hand, and the other routes (into the scene, element mode, ComfyUI).

## Generation and performance

Generation makes a stored element; performance gives it a new timeline. Near the top of the studio, **One render. Seven performances.** uses `hand_palm__boom__fire_burst` and its two replays, `hand_palm__pfpfboom__again` and `hand_palm__pfx4__again`, in `samples/made_ahead/bending/`, then four more by strangers' recorded voices (`hand_palm__sadiquecat__again`, `hand_palm__unfa__again`, `hand_palm__huw2k8__again`, `hand_palm__jedimichael__again`). The author's palm footage and one LTX-2.5 fire element stay the same. **BOOM** uses the original recorded cue; **pf, pf, BOOM** and **pf pf pf PF** are edited rhythms assembled from the author's recordings and room tone.

The replay receipts name the base shot and record the same element SHA-256, `no_new_render: true` and `no_gpu: true`. The new voice determines the hit frames, size and held duration; tracking, masking, compositing and light run on the CPU. The generated effect becomes a reusable instrument.

For hand effects, LTX-2.5 with ainvfx-fluid generates the fire, the air vortex and the explosion's fireball once over black. [`runner.py`](cuesync/src/cuesync/runner.py) loads the API workflow JSON, constructs the Comfy Python SDK workflow and sends it to the self-hosted Comfy API with `client.run`. Lightning is a procedural CPU effect. The explosion reuses an LTX fireball; its blast, light, smoke compositing and dissipation are handled on the CPU.

## The pipeline

![How it was made](media/how-it-was-made.jpg)

1. **Read the cue** ([`cues.py`](cuesync/src/cuesync/cues.py), [`audio.py`](cuesync/src/cuesync/audio.py)). The onsets, the hits of a rhythm ("pf, pf, KA-BOOM"), an absolute loudness envelope per video frame, and when the sound is actually there. It comes from your voice, from the clip's own soundtrack, or from press-and-hold. Its shape (a hit, a rhythm, a sustained sound, a breath, a hiss) picks the effect when you let it ([`gestures.py`](cuesync/src/cuesync/gestures.py)):
   - A sample clip knows what it shows. On your own clip you say what it is (a candle, a jack-o'-lantern, a toy's mouth, a hot drink), since nothing here recognises objects.
   - With your hand in the shot, the sound and the hand's gesture decide: a voiced BOOM off an open palm is a fireball, a long "fffff" from pointed fingers a flamethrower, a hiss steam, a "pf" a pop of fire.
   - Voiced or not is read from pitch. An acoustic sound classifier ([`cue_detect.py`](cuesync/src/cuesync/cue_detect.py)) misread too many of the 19 real and sample takes to pick with, so its reading is only kept in the receipt.
2. **Paint the guide** ([`scene.py`](cuesync/src/cuesync/scene.py)). ainvfx-fluid was trained on controls that paint the whole frame in six flat colour zones, with the real first frame as conditioning. So the plate is painted in its own zones on every keyframe, and the effect over them: at its source, going its way, or filling your drawing as far as your cue says. The words fit the subject ([`effects/`](effects)). A hand's element is painted alone, over black.
   - The guide is held over each 8-frame latent until the hit (a blank latent lets the model start the effect early), and every hit lands on a keyframe.
3. **Render.** LTX-2.5 with the ainvfx-fluid IC-LoRA, in ComfyUI, through the Comfy Python SDK ([`runner.py`](cuesync/src/cuesync/runner.py)). Into the scene, it starts from your plate's frame at full strength and lights the room, the table and the prop itself. For your hand, the element is rendered once over black.
4. **Time it** ([`scene_shot.py`](cuesync/src/cuesync/scene_shot.py)). The first frame where the effect shows is measured on the render and moved onto your cue (a burst shows a frame before its "BOOM": light beats sound).
   - In a rhythm, each hit the model kept apart goes onto its own cue. Between them the render plays up to a third faster or a quarter slower; when not every hit can be reached, the big blast is the one that lands.
   - In call and response, each hit is measured at its own source. A blast adds a warm flash at its peak and a camera shake just after it.
5. **Keep your footage** ([`transfer.py`](cuesync/src/cuesync/transfer.py)). A still plate becomes the render. A moving clip keeps its own pixels when it can:
   - The render and the clip are tracked onto the start frame, and the render's exposure dip (LTX darkens a room by up to three stops for a fireball) is taken out before anything is compared.
   - Its light carries over as a gain on your pixels, in linear light, so the room goes warm around the fire while keeping its own texture. Where the structure changed (the fire, smoke, debris), the render replaces the plate.
   - The transfer is checked. If the render moved or re-shaped the scene, the shot uses the render and says so: that happened on 4 of 18 real moving clips.
6. **Perform it again** ([`replay.py`](cuesync/src/cuesync/replay.py), [`bending.py`](cuesync/src/cuesync/bending.py)). A finished shot's render already holds each hit's effect. A new take fires it on its own hits, with no new render:
   - Each hit is as big as it is loud, against the take the render was made for: a lone quiet "pf" on a BOOM's render is a pop.
   - A held flame (a flamethrower, a plume) burns as long as the new voice holds it, looping its steady burn, and dies down when the voice stops.
   - **Perform all together** plays several shots on one still plate with one rhythm, each effect at its own spot.
7. **Give it its sound** ([`sfx_synth.py`](cuesync/src/cuesync/sfx_synth.py)). Procedural Foley of what the effect does (a fire's whoomph and crackle, a puff of smoke, a candle blown out), synthesised for the shot, each hit on the frame where it shows, under your cue.

Every shot writes a receipt (`receipt.json`) that records:
- the cue, and where the hits were painted;
- how early or late the model put them, and how far they were moved;
- whether the transfer passed its check, and what made the shot.

LTX-2.5 can only take the guide on an 8-frame grid (a third of a second at 24 fps), and it nearly always starts the effect a few frames early, so the measure-and-move step is what puts the hit on your cue.

## The hand (element bending)

The element is rendered once over black with LTX-2.5 and ainvfx-fluid (the LoRA learned static sources, so the hand itself is never regenerated). Then it is carried on your hand frame by frame:
- **Tracking** ([`hand_track.py`](cuesync/src/cuesync/hand_track.py)): MediaPipe follows the hand you tapped; RTMPose re-finds it where MediaPipe loses it; optical flow fills the gaps. In the studio you see its path before anything is made, with the frames it lost in red.
- **History:** each part of the flame keeps the direction it was thrown in, so a swing whips through the air.
- **Size** ([`bending.fit_play`](cuesync/src/cuesync/bending.py)): measured on what the render draws, not on what its guide asked for (the model paints well inside it). A palm's flame plays 2.8 hands tall, a fireball 2.2 hands wide (1.1 from a fingertip), within the frame's room, never smaller than rendered.
- **Thrown:** a fireball leaves the fist (or the palm) a frame after the strike and flies on along it, slowing and rising, with the room; the fist pulls back without it. A flame stays on the hand.
- **A flame in the palm** comes up out of the skin: its foot fades over a quarter of a hand, broken by noise that rises with the flame, and throws a pool of light on the palm it stands in.
- **Matte** ([`hand_mask.py`](cuesync/src/cuesync/hand_mask.py)): SAM 2.1 cuts the hand and forearm out, so the fingers in front of the flame's root hide it. Without SAM, the matte is the skin MediaPipe's segmenter finds, filled out by the hand drawn from its landmarks (and the landmarks alone where the skin isn't found).
- **Timing:** each hit fires on your arm's strike when the strike falls on the beat.
- **Light** ([`fx_light.py`](cuesync/src/cuesync/fx_light.py)):
  - The flame's own image is spread as light, and your hand is lit as if round, on the side facing the flame.
  - The room gets less. Soot only where flame is, and its dim edge hides no more of the room than it gives back.
  - Skin under a flame is no brighter than the flame times its albedo: the light on it rolls off towards that, so a fist next to a fireball keeps its texture instead of blowing out.
- **A camera's colour** (`bending.fire_hdr`): in a lit room a camera exposes for the room and sees a flame's core blown to white. The render's core is brightened by its heat (not the whole flame, which only went orange, yellow, lemon), its overexposure spills through warm white, and a lens glow surrounds it; the flame lights the room as brightly as it now reads.
- **Filmed like the hand:** the fire is smeared by the hand's motion over half a frame (a phone's shutter) and softened to the plate's sharpness.
- **Charge-up:** hum before the strike ("mmmmmmBOOM") and an ember grows in your palm with your voice. It lights your fingers until the strike releases the fireball; the hum launches nothing.
- **First person:** with the phone at your eyes and your hand coming in from below, a handheld camera is measured on the background, so what was already thrown moves with the room, not with the screen. A camera on a tripod is left alone.
  - On a handheld stand-in with a known camera path, the error is 0.6 px median over the flame's 10-frame history, against 27 px of camera motion.
  - The tracker's ground truth in [`cuesync/tests/bending/gt/`](cuesync/tests/bending/gt) is provisional: points marked by eye on gridded crops (AI-assisted), not hand-clicked.

### Air

The air vortex is the second generated instrument. Its guide is a funnel of steam rings, narrow on the hand and wide at
the top, each swaying round the axis a little further every keyframe ([`scene._vortex`](cuesync/src/cuesync/scene.py)).
LTX-2.5 with ainvfx-fluid turns it into a twisting column of white mist over black, once (110 s on a rented A100);
[`samples/library/bending/air_vortex`](samples/library/bending/air_vortex) keeps the element and its guide. Like the fire,
it is then carried on the hand on the CPU: held while the voice holds (a hoooo), sized like a flame in the palm (2.8 hands),
composited over the plate as white vapour, and let go when the voice stops. On a sample hand with no GPU and no tracker,
the stored element is carried on the sample's own track and matte.

### Lightning and explosion

These effects are composited on the CPU from the plate, hand track and matte, and the voice's timeline. Each module follows the contract in [`elements.py`](cuesync/src/cuesync/elements.py): `render(plate, track, mask, timeline, form, seed)`. A shot is deterministic per seed.

- **Lightning** ([`lightning.py`](cuesync/src/cuesync/lightning.py)) uses branching channels from a tracked fingertip or a plasma ball over the palm, with hand occlusion and cool light on the skin and room. The voice sets timing and strength; no model generation is needed.
- **Explosion** ([`explosion.py`](cuesync/src/cuesync/explosion.py)) uses a stored LTX-2.5 + ainvfx-fluid fireball. Its blast is measured on the source render, placed ahead of the fist on the strike and carried with the room after release. The CPU adds the blast, light, sparks, smoke compositing and dissipation without a new generation.

Water, earth and the procedural air ball and blast are experimental CPU effects under the collapsed **Experimental** disclosure; their materials and motion can still look synthetic.

The curated gallery leads with the fist explosion, palm fire and its re-performances in the author's and four strangers' voices, followed by the walking flame, pointing fire, the air vortex and lightning from a fingertip and a dim-room palm. Each finished shot keeps its source and performance receipts. The live preview draws fire and lightning in the browser.

**CPU-route evaluation, on nineteen phone takes.** Nineteen of the author's own takes went through the studio's no-GPU route: no SAM, the hand followed and matted on the CPU, each take's own sound as the cue, the effect picked by the sound and the gesture, and an element from the library. The timings below are from a desktop CPU, not a laptop performance guarantee.
- Every one of the nineteen made a shot, in 32 to 100 s each on a desktop CPU.
- The hand was lost on at most 23 % of a clip's frames.
- The picks: a palm's "pf" a pop of fire, a fist's strike a burst, a roar from pointing fingers a flamethrower, a held "fffff" over an open palm a flame in it, a BOOM a fireball. One snap with no voice became a pop of fire where a lighter's flame was meant; a tap on the badge changes it.

| take | the room | the cue | picked | seconds | hand lost |
|---|---|---|---|---|---|
| side palm | white wall, daylight | pf | pop of fire | 54 | 0 % |
| fist punch | white wall | its own sound | burst | 32 | 4 % |
| pointing, first person | white wall | hiss | flamethrower | 75 | 9 % |
| open hand into the room, first person | living room | pf | pop of fire | 34 | 0 % |
| palm to the camera, first person | white wall | pf | pop of fire | 48 | 0 % |
| snap, then a thumb | white wall | its own sound | burst | 60 | 6 % |
| open palm | dark bathroom | BOOM | fireball | 44 | 0 % |
| finger gun sweeping, first person | dim room | roar | flamethrower | 61 | 0 % |
| finger gun | dim room | BOOM | fireball | 37 | 0 % |
| snap, then a thumb | dark bathroom | snap | pop of fire | 39 | 14 % |
| fist punch | olive wall, lamplight | BOOM | burst | 35 | 0 % |
| fist punch | olive wall | BOOM | burst | 51 | 23 % |
| fist thrown at the wall | olive wall | BOOM | burst | 60 | 15 % |
| finger gun | olive wall | bang | fireball | 36 | 12 % |
| still palm, three beats | olive wall | fff, fff, fff | a flame each | 70 | 0 % |
| fist opening into a palm | olive wall | pf | pop of fire | 47 | 0 % |
| fist opening, a held breath | olive wall | fffff | flame | 54 | 0 % |
| palm, fist, palm | olive wall | its own sound | fireball | 79 | 0 % |
| palm carried through the house, the camera walking | hallway to kitchen | fffff | flame | 100 | 0 % |

**Try it live** ([`Live.tsx`](web/components/Live.tsx), [`live.ts`](web/lib/live.ts)) runs MediaPipe's hand landmarker in the browser. Its runtime and model load from jsDelivr and Google's model storage. The microphone's level is read every frame:
- a fast rise well over the room is a hit;
- a sound still going after a third of a second is held.

On each hit the library's fireball, rendered ahead with LTX-2.5, plays from your palm, added as light and sized against your loudest noise so far (and as big as the shot will play it), then leaves the hand and rises. A held sound plays its flame (pointing: its jet) looped until you stop. A buzzing "zzzzz" (a pitch with a hiss on it) throws lightning instead ([`lightning.ts`](web/lib/lightning.ts)): arcs from a pointed fingertip, or a crackling ball over an open palm. It's a preview: no matte, no tracking history, a glow for the light. **Record this** keeps the raw take, which goes through the pipeline above.

A take that tracks well:
- **Light:** your hand lit, in front of a plain, darker background.
- **Framing:** the whole hand in frame on the strike, with room for the flame in the direction it goes.
- **Camera:** 1080p at 24-30 fps, held still or steady. First person (the phone at your eyes) works too.
- **Cue:** make the noise as you strike.

### The helpers, set up by hand

`cuesync hands --install` (or **Set up the hand tracker** in the studio) does the first one. They live apart from the core's environment because MediaPipe's opencv-contrib clashes with its opencv-python. `cuesync serve` finds them in `cuesync/.venv-track` and `cuesync/.venv-mask`; `CUESYNC_TRACK_PY` and `CUESYNC_MASK_PY` point it at others. On Linux and macOS the interpreter is `bin/python`.

The hand tracker (MediaPipe + RTMPose, Apache-2.0; their models download on first use into `~/.cache/cuesync`), about 250 MB, CPU:

```bash
uv venv cuesync/.venv-track --python 3.12
uv pip install --python cuesync/.venv-track/Scripts/python.exe mediapipe==1.1.1 onnxruntime==1.31.0 av==19.0.1
```

The SAM 2.1 matte (SAM 2.1-small, Apache-2.0; an NVIDIA GPU, CUDA 12.8 wheels, about 3 GB), in PowerShell:

```powershell
uv venv cuesync/.venv-mask --python 3.12
$env:VIRTUAL_ENV = "$PWD\cuesync\.venv-mask"
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install hydra-core==1.3.7 iopath==0.1.10 huggingface-hub==2.2.0 mediapipe==1.1.1 av==19.0.1 tqdm setuptools wheel
$env:SAM2_BUILD_CUDA = "0"
uv pip install --no-build-isolation --no-deps "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
Remove-Item Env:VIRTUAL_ENV, Env:SAM2_BUILD_CUDA
```

(In bash: `export VIRTUAL_ENV="$PWD/cuesync/.venv-mask"` and `SAM2_BUILD_CUDA=0` before the last install.)

From the command line, your own hand with no GPU (an element rendered ahead of time, from `samples/library/bending`), tapped at x, y on frame 40:

```bash
cuesync bend --instant --plate my_hand.mp4 --voice my_kaboom.wav --x 0.55 --y 0.6 --frame 40
```

With the GPU, leave out `--instant` and the element is rendered for your hand's own pose.

## How well it lands

Marked by eye, frame by frame, on the raw renders of 57 scene shots ([`tools/timing_annotations.json`](tools/timing_annotations.json), [`tools/measure_timing.py`](tools/measure_timing.py)):

| Hits within 1 frame (42 ms) of the frame they aim for: raw LTX-2.5 render → finished shot | | mean error |
|---|---|---|
| all 66 scorable hits | 7 → 53 | 5.2 → 1.1 frames |
| single hits (42) | 0 → 37 | 6.0 → 0.7 frames |
| hits in a rhythm (24), each on its own cue | 7 → 16 | 3.7 → 1.7 frames |
| hits on renders the onset code was not fitted on (46, on 39 renders) | 6 → 35 | 5.2 → 1.3 frames |
| real footage rendered after the last onset fix, synthetic noises (11) | 1 → 10 | 6.4 → 0.9 frames |
| real footage cued by real people's recorded voices, marked blind (12) | 3 → 6 | 5.3 → 2.5 frames |

- **The shots:** 31 are on LTX-2.5 stand-in plates (11 setups, most rendered more than once) and 26 on real stock footage.
- **The hits:** of the 76 painted, 66 have an onset that can be marked. The other 10 are 6 rhythm hits the model merged into a neighbour and 4 that can't be called. Counting those as misses, 53 of 76 land within a frame.
- **How it's marked:** the onsets are marked by eye on the raw renders; the finished error is that mark moved by the shot's re-time.
- **On real footage:**
  - The first 7 real shots showed two misses the stand-ins never did (a real candle's flame flaring white as it was blown out; a scene LTX-2.5 re-shaped at a latent block boundary). Both were fixed.
  - A second batch of 12 was then rendered with no change to the onset code: 8 of its 11 measurable hits landed within a frame.
  - Of the 18 real moving clips, 14 keep their own pixels and 4 fall back to the render, each saying so.
- **Real people's voices:** 7 shots are cued by CC0 recordings from Freesound, marked blind by two annotators (a third deciding): 6 of their 12 hits land within a frame. The misses are the model's spacing: in two rhythms LTX-2.5 put the hits further apart than the voice did, beyond the third the re-time allows.
- **Limits:** in a rhythm LTX-2.5 often merges close hits into one (6 of 30 here); those are reported, not faked. A flame dies down about 0.6 s after the voice stops (0.2 to 1.4 s).

Every render the marks refer to ships with the repository (`samples/made_ahead/scene`, `samples/timing_set/`). Rechecking all 57 scene shots found zero receipt retiming drift. That confirms the recorded timing transformations, not perfect cue accuracy: 53 of 66 scorable hits, including 37 of 42 single hits, land within one frame; the 10 unscorable hits remain disclosed above. A hit aims for its cue, except a burst, which aims one frame before its sound so its flash is seen as its bang is heard. Measured against the sound itself, 43 of 66 hits (30 of 42 single hits) are within a frame. The tool re-runs the pipeline's own onset and re-time code on every render rather than trusting the receipts, so the table re-measures from a clone (about five minutes, one line per shot):

```bash
python tools/measure_timing.py samples/made_ahead/scene samples/timing_set/before_fixes samples/timing_set/other_seeds samples/timing_set/dropped samples/timing_set/real
```

![Timing: raw render vs finished shot](media/timing.png)

The take itself has to land where it was performed. [`tools/check_take_timing.py`](tools/check_take_timing.py) replaces the studio's microphone with a synthetic one that makes a noise at the moment the clip shows a chosen frame: in 8 takes the noise was found within 45 ms of it (6 within 20 ms). Chrome reports a real microphone's input latency and the studio subtracts it (20 ms when the browser doesn't say).

## Into the scene, from the command line

```bash
cuesync shot --scene --plate samples/plates/toy_desk.mp4 --voice samples/voices/fwoooom.wav --effect fire_breath --x 0.375 --y 0.485 --direction -90 --subject "the small plastic toy dinosaur on the bedside table"
cuesync shot --scene --plate samples/plates/candle_snap.mp4 --cue clip --effect blow_out --x 0.53 --y 0.52
cuesync shot --scene --plate samples/plates/pumpkin_kitchen.mp4 --cue hold --hold 1.0-2.5 --effect flame_plume --strokes samples/strokes/pumpkin_top.json --describe "the small pumpkin catches fire and flames spread along its top"
```

The first breathes fire out of the toy dinosaur to the left, as long and as hard as the "fwoooom". The second locks to the snap already in the clip. The third is drawn, timed with a press-and-hold, and described in words.

Re-timing a finished shot from the command line (the same render, composited again a quarter of a second later):

```bash
cuesync retime --shot samples/made_ahead/scene/jackolantern_kitchen__ka-boom__fire_burst --plate samples/plates/jackolantern_kitchen.mp4 --voice samples/voices/ka-boom.wav --effect fire_burst --x 0.48 --y 0.39 --shift 6 -o runs/retimed
```

**Element mode** (Advanced in the studio, `cuesync shot --instant`) is the no-GPU route for a scene: an effect rendered ahead of time, alone on black, composited onto the untouched plate in linear light ([`composite.py`](cuesync/src/cuesync/composite.py)). The object stays in front through a GrabCut matte ([`matte.py`](cuesync/src/cuesync/matte.py)), or SAM 2.1 in the ComfyUI node.

## Models and licences

The app uses open weights, self-hosted, with no partner nodes or closed APIs:

- **LTX-2.5** 22B (int8), its video and audio VAEs, and its **Gemma 4 12B** text encoder (Lightricks; LTX-2.x Community License, Gemma terms).
- **ainvfx-fluid** IC-LoRA (AInVFX, Adrien Toupet; LTX-2.x Community License).
- **MediaPipe** gesture recognizer, hand landmarker and selfie segmenter (Google; Apache-2.0). The browser live preview also uses the landmarker.
- **RTMPose**-m hand (OpenMMLab; Apache-2.0).
- **SAM 2.1**-small (Meta; Apache-2.0).
- Optional: LTX-2.5 IC-LoRA **Alpha-Gen** (Lightricks), a smoke-matte fallback.

The demo's narration uses **Kokoro-82M** (Apache-2.0), a synthetic voice, separate from the app and from the author's performed cues. The studio's **pf, pf, BOOM** and **pf pf pf PF** samples are edited rhythms assembled from the author's recorded pf/BOOM sounds and room tone; the other preset noises are synthetic. [Model files, versions and revisions](build/model-files.md).

## With a GPU

Your own renders, for your hand's pose or into the scene, need ComfyUI with LTX-2.5 22B (int8) and ainvfx-fluid ([models](build/model-files.md); gated LTX-2.5 access on Hugging Face). A render fits a 32 GB card (about 3 minutes); on a rented RTX PRO 6000 (96 GB) it takes about a minute.

- **Your own GPU:** run ComfyUI with this repository and ComfyUI-LTXVideo in `custom_nodes/` ([installation below](#in-comfyui)), run comfy-api-proxy in front of it, and point `COMFY_BASE_URL` at the proxy.
- **A rented GPU:** [`infra/pod/`](infra/pod/README.md) creates, sets up and terminates a disposable Runpod pod (it needs `RUNPOD_API_KEY` and `HF_TOKEN`; the models live on a network volume, about 120 GB). A watchdog terminates the pod when it is idle.

Connecting the app to ComfyUI:

- **Comfy Cloud:** set `COMFY_API_KEY`. With `COMFY_BASE_URL` unset, the SDK defaults to `https://cloud.comfy.org`. For `cuesync studio`, also set `COMFY_BASE_URL=https://cloud.comfy.org`. GPU options appear only after the API and Cloud's documented node/model inventory confirm the configured workflow's required resources.
- **Self-hosted ComfyUI:** run comfy-api-proxy in front of ComfyUI and set `COMFY_BASE_URL` to the proxy URL, for example `http://127.0.0.1:8189`. The standard local setup checks native ComfyUI at `http://127.0.0.1:8188`; for a different upstream, also set `COMFYUI_BASE_URL` to its native URL. `COMFY_API_KEY` authenticates the proxy; a separately protected native endpoint uses `COMFYUI_API_KEY`. Leave keys unset when their endpoints need none.
- **Managed Comfy API deployments:** v2 alone has no documented node/model inventory route. The app reports an accessible endpoint as **unverified**, without submitting a paid probe. Set `COMFYUI_BASE_URL` only if the deployment exposes its native ComfyUI inventory; otherwise use the SDK/CLI generation route and the app's CPU reuse flow. The app does not advertise unverified GPU modes as ready.

Preflight results are cached for 15 seconds and refresh immediately when the workflow or connection settings change. `/api/health` includes the missing node/model names and setup guidance. This verifies availability, not model loading or sufficient GPU memory.

The studio's GPU preflight ([`readiness.py`](cuesync/src/cuesync/readiness.py)) checks API access and the configured graph's node classes/model filename choices, rather than the proxy-only health route. Native inventory comes from `COMFYUI_BASE_URL/object_info` (automatic for the standard local 8189→8188 setup), or Cloud's documented `/api/object_info`. Checks expire after 15 seconds; workflow/connection changes invalidate them immediately. Missing scene-only support disables scene generation without disabling an otherwise valid element workflow. Optional SAM matting nodes are not required. A managed v2-only deployment with no exposed native inventory is reported as **unverified** and retains the CPU reuse flow; no paid job is submitted by preflight. A passing check does not establish GPU memory capacity or model-load success.

```bash
python infra/pod/podctl.py volume create
python infra/pod/podctl.py up
python infra/pod/podctl.py setup
python infra/pod/podctl.py sync
python infra/pod/podctl.py tunnel
```

The first command runs once; run `tunnel` in another terminal. Then, with `COMFY_BASE_URL=http://127.0.0.1:8189` set:

```bash
pip install "./cuesync[server,comfy]"
cuesync studio
```

## In ComfyUI

[`workflows/cue_the_boom_scene.json`](workflows/cue_the_boom_scene.json) (drag it onto the canvas) is the whole scene shot as one workflow: your cue and your plate go in, the finished shot comes out. [`cue_the_boom_scene.api.json`](workflows/cue_the_boom_scene.api.json) is the same graph for the API, and [`fluid_element.api.json`](workflows/fluid_element.api.json) renders a hand's element over black.

![The scene shot in ComfyUI: your cue, CueSync Paint Scene, LTX-2.5 with ainvfx-fluid from your plate, CueSync Scene Shot](media/comfyui-scene-workflow.png)
<sub>The picture carries the workflow: drag it onto the ComfyUI canvas to open the same graph.</sub>

To install it, clone this repository and [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) into `ComfyUI/custom_nodes/` (or paste their URLs into ComfyUI-Manager's "Install via Git URL"). Install each one's `requirements.txt` into ComfyUI's Python, and get the models in [build/model-files.md](build/model-files.md).

To try it on a sample, copy `samples/voices/fwoooom.wav` to `ComfyUI/input/voice.wav` and `samples/plates/toy_desk.mp4` to `ComfyUI/input/plate.mp4`, then queue the workflow as it is. The nodes are documented in [comfyui-cuesync/README.md](comfyui-cuesync/README.md).

## A try-it link of your own

`deploy/space/` builds the no-GPU studio into a Hugging Face Docker Space (CPU basic, free). Run `python deploy/space/make_space.py`, push the folder it writes, and set the Space secret `CUE_PASSWORD` to keep it behind a password.

## Prior work

The contribution is separating generation from performance in this loop:
1. A performed or in-clip sound becomes a hit frame and a growth curve.
2. It is painted into an IC-LoRA control in the format the LoRA was trained on, and rendered by LTX-2.5.
3. The render's actual onset is measured and moved onto the cue, with the error written down.
4. On your hand, the render is carried on the tracked hand, lit and timed to every hit, and performed again with no new render.

Each piece rests on prior work:
- **[ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid)** (AInVFX, Adrien Toupet) is the painted-control fluid model everything here is rendered with.
- **Sound-guided video effects** have been studied before (for example [Soundini](https://arxiv.org/abs/2304.06818)). Voice as a timing and loudness control exists for sound effects (Sketch2Sound). Audio-reactive and onset nodes exist for ComfyUI, and LTX-2.5 has its own audio-to-video.
- **Drawing where an effect goes** and **scene reaction** exist in other tools (region edits, image-to-video effect presets). Here they are the inputs and the output of the cue, not the novelty.

## Tests

```bash
pip install "./cuesync[server,dev]"
pytest cuesync/tests
```

The final Python integration run passed **1,066 tests**, with **20 skipped**. The skips require optional MediaPipe, SAM 2.1 or development plates. The suite runs on the CPU; duration depends on the machine and concurrent rendering. The studio's **295 tests**, TypeScript, ESLint, production build and static export passed. Its tests need Node 24: `npm install`, then `npm test`, in `web/`.
