# Cue the Boom

**Generate the effect once. Perform it again with your voice.**

**Film your hand. Make the noise. Perform fire, air, explosions and lightning.**

Choose **Fire**, **Air**, **Explosion** or **Lightning**, then **Perform this … with your voice**: record over a prepared hand clip and download your newly composed shot. No camera, tracker setup or GPU is needed for these prepared samples.

Below the creation flow, **One render. Three performances.** shows the author's palm and one LTX-2.5 fire render with three voice-driven performances: **BOOM**, **pf, pf, BOOM**, and **pf pf pf PF**. The first uses the original recorded BOOM; the two rhythms are edited from the author's recorded pf and BOOM sounds with room tone from the hand take. Play them together or hear them one by one. A quiet "pf" makes a small hit; a loud "BOOM" makes a big one. Each performance changes the timing and size of the same generated element, without another generation or a GPU. [Audio provenance](samples/CREDITS.md#the-authors-voice-and-the-three-performances).

For hand effects, the fire, the air vortex and the explosion's fireball are rendered once over black by **LTX-2.5** with the **[ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid)** IC-LoRA, in a ComfyUI workflow sent through the Comfy Python SDK. **Lightning is a procedural CPU effect.** The CPU also handles the explosion's blast, compositing, light and smoke handling, using the stored LTX fireball.

![The author's own hand: BOOM throws a fireball off the fist, a held fffff keeps a flame in the palm through a walk from a hallway into a kitchen in one take, and a render performed again as pf pf pf PF](media/hand_hero.jpg)
<sub>The author's own hand and voice, filmed on a phone. The third is a render performed again in a new rhythm, with no new render.</sub>

An entry for Comfy's **[Open Source Higgsfield](https://comfy.org/open-source-higgsfield)** contest, on open weights only.

[Try it](#try-it-in-two-minutes-no-gpu) · [How it works](#how-it-works) · [The entry at a glance](#the-entry-at-a-glance) · [With a GPU](#with-a-gpu) · [In detail](TECHNICAL.md)

## Why it's new

The contribution is separating generation from performance: one generated effect becomes a reusable instrument.
Your voice controls when it hits, how big it plays and how long it burns.
A new take reuses the same element while the CPU tracks, masks, lights and composites it on your hand.
The three performances below the creation flow let you hear and see that separation.

## Try it in two minutes, no GPU

You need Python 3.10+ and git. No GPU and no model download: the studio ships built, and the sample hands come with their fire rendered ahead of time.

```bash
git clone --depth 1 https://github.com/grzenio22/cue-the-boom
cd cue-the-boom
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`; macOS and Linux: `source .venv/bin/activate`), then:

```bash
pip install "./cuesync[server]"
cuesync studio
```

The studio opens at http://127.0.0.1:8765 (`cuesync studio --port 8766` if that port is taken). In it:

1. Choose **Fire**, **Air**, **Explosion** or **Lightning**, then a form where offered. Click **Perform this … with your voice** to select its prepared hand clip.
2. Click **Use my microphone**, then **Record & make …**. Make the suggested sound as the clip plays. Microphone access starts only when you request it.
3. When recording ends, the app automatically makes your performance. Fire, air and explosion reuse stored generated elements; lightning is composed procedurally from your recording and the tracked hand. **Download** saves the finished shot.
4. **Perform again** gives the same element a new performance. **Try a sample voice instead** works without microphone access and is explicitly labelled as a recorded example.

The prepared fire path always reuses the stored element, even when a GPU is connected. Three local full-quality palm runs with a single-hit recording took 14.8–16.8 seconds from HTTP submission through MP4 download on a Ryzen 9 7900X; the bundled three-hit author rhythm took 26.6 seconds (84 frames at 24 fps; no new model generation). Those measurements exclude microphone permission and recording time and do not predict hosted or arbitrary-clip performance. Run `python tools/benchmark_prepared_palm.py --out runs/palm-benchmark` to measure the bundled author rhythm on your computer; it uses a separate local CPU-only server and writes its inputs, timings and decode checks.

Under **Made with it**, play **fffff · a flame in the palm, one take**: the camera walks from a hallway into a kitchen, and the flame stays in the palm. To perform it yourself, pick **Your hand: a flame through the house**, then **Use my microphone** and hold an "fffff".

**Your own hand:** open **Use your own clip or explore effects**, then use **Film it here** or **Use your own clip** and **Set up the hand tracker** once (needs Python 3.12 or `uv`). Tap your palm or fingertips, drag to aim, and confirm **That's my hand**. Record your noise and **Make the shot**. **Try it live** previews stored fire and procedural lightning through your webcam. It needs internet access to load its hand model and runtime from a CDN. **Record this** turns the take into a finished shot. [Tracking and live-preview details](TECHNICAL.md#the-hand-element-bending).

The same from the command line (the shot lands in `runs/bend/final.mp4`):

```bash
cuesync bend --plate samples/plates/hand_palm.mp4 --voice samples/voices/pf-pf-ka-boom.wav --again samples/made_ahead/bending/hand_palm__boom__fire_burst
```

![The studio](media/studio.jpg)

## How it works

![How it's made: your clip and your noise, your hand found, the fire rendered once, carried, lit and timed](media/pipeline.jpg)

1. **Your sound** sets when each hit fires, how big it is and how long it lasts. With your hand's gesture, it also picks the effect.
2. **Your hand** is followed by MediaPipe and RTMPose. SAM 2.1 supplies the matte with a GPU; the CPU route uses MediaPipe's skin segmenter and hand landmarks.
3. **The fire** is rendered once, over black, by LTX-2.5 with the ainvfx-fluid IC-LoRA in a ComfyUI workflow, from a guide your sound paints.
4. **Carried, lit, timed:** the fire follows your hand, lights your skin, passes behind your fingers and fires on every hit. A thrown fireball keeps moving after your hand pulls back.
5. **Perform it again:** a new take re-composites the same render on the CPU. A held flame burns as long as your new voice holds it; longer holds and overlapping hits take more time to process.

**Or tap anything in your shot.** "KA-BOOM" at a jack-o'-lantern, "fwoooOOOM" at a toy dinosaur: LTX-2.5 renders the effect into the scene, so it lights the room and leaves its mark.

For scene renders, Cue the Boom measures where the effect starts and moves it onto your sound. On 42 single hits, 37 land within one frame (1/24 s) of the cue; straight from the model, none do. [Evaluation and limits](TECHNICAL.md#how-well-it-lands).

![The scene reacts: six real clips and photos, each with the sound that cued it](media/hero.jpg)
<sub>Real stock clips and photos ([credits](samples/CREDITS.md)), each with the sound that cued it. These play in the studio's gallery with no GPU.</sub>

The details (the pipeline step by step, the timing evaluation, the matte, the light) are in [TECHNICAL.md](TECHNICAL.md).

## The curated gallery

**Made with it** leads with the fist explosion, the palm fire and its two new voice rhythms, the same render in four strangers' recorded voices, then the walking flame, pointing fire, the air vortex and lightning. The dim-room lightning arcs and ball show procedural light on the author's hand. These are finished shots you can play without a GPU.

Water and earth, and the procedural air ball and blast, are experimental CPU effects under the collapsed **Experimental** disclosure; their materials and motion can still look synthetic.

Choose an effect, then perform its cue. Voice controls timing and strength; it does not generate arbitrary new effects from spoken instructions. The live webcam preview supports fire and lightning. [Footage, audio and retained comparison-media credits](samples/CREDITS.md).

## The entry at a glance

- **Core video model:** LTX-2.5 (Lightricks) with the ainvfx-fluid IC-LoRA (AInVFX, Adrien Toupet). Every fire effect, and the explosion's fireball, is an LTX-2.5 render.
- **Open weights only, self-hosted.** No partner nodes and no closed APIs. [Models and licences](TECHNICAL.md#models-and-licences); [versions and revisions](build/model-files.md).
- **The app:** a browser studio and its render API (`cuesync studio`), a CLI (`cuesync`), and ComfyUI nodes.

## Comfy Developer Platform

The app is written to the Comfy Python SDK and Comfy API v2 exposed by the Comfy Developer Platform. The [runner](cuesync/src/cuesync/runner.py) submits workflow JSON through the SDK.

- **Comfy Cloud:** set `COMFY_API_KEY`. With `COMFY_BASE_URL` unset, the SDK defaults to `https://cloud.comfy.org`. For `cuesync studio`, also set `COMFY_BASE_URL=https://cloud.comfy.org`. GPU options appear only after the API and Cloud's documented node/model inventory confirm the configured workflow's required resources.
- **Self-hosted ComfyUI:** run comfy-api-proxy in front of ComfyUI and set `COMFY_BASE_URL` to the proxy URL, for example `http://127.0.0.1:8189`. The standard local setup checks native ComfyUI at `http://127.0.0.1:8188`; for a different upstream, also set `COMFYUI_BASE_URL` to its native URL. `COMFY_API_KEY` authenticates the proxy; a separately protected native endpoint uses `COMFYUI_API_KEY`. Leave keys unset when their endpoints need none.
- **Managed Comfy API deployments:** v2 alone has no documented node/model inventory route. The app reports an accessible endpoint as **unverified**, without submitting a paid probe. Set `COMFYUI_BASE_URL` only if the deployment exposes its native ComfyUI inventory; otherwise use the SDK/CLI generation route and the app's CPU reuse flow. The app does not advertise unverified GPU modes as ready.

Preflight results are cached for 15 seconds and refresh immediately when the workflow or connection settings change. `/api/health` includes the missing node/model names and setup guidance. This verifies availability, not model loading or sufficient GPU memory.

The renders here ran on self-hosted ComfyUI to load the ainvfx-fluid IC-LoRA and install the [CueSync node pack](comfyui-cuesync). A target server needs the models and nodes used by the workflow.

## With a GPU

To generate a new hand element or render into a scene, use ComfyUI with LTX-2.5 22B (int8) and ainvfx-fluid. A render fits a 32 GB card (about 3 minutes); on a rented RTX PRO 6000 (96 GB) it takes about a minute. [Local and rented GPU setup](TECHNICAL.md#with-a-gpu), including the Comfy SDK connection and model downloads.

## Repository

| Path | What |
|---|---|
| `cuesync/` | The Python core, the CLI and the render API: cues, guides, the Comfy SDK runner, timing, the compositor, the hand (Apache-2.0) |
| `web/` | The studio (Next.js); built into `cuesync/src/cuesync/studio_static/` |
| `comfyui-cuesync/` | The ComfyUI node pack |
| `workflows/` | The ComfyUI workflows, canvas and API format (LTX-2.x Community License, derived from the ainvfx-fluid sample) |
| `effects/` | The effects, as JSON |
| `samples/` | The sample clips and noises, the shots made ahead of time, the timing evaluation renders, the element library ([credits](samples/CREDITS.md)) |
| `tools/`, `infra/pod/` | Timing measurement, the demo tools; disposable GPU pods |

## Credits and licences

- **Code:** [MIT](LICENSE), with [`cuesync/` under Apache-2.0](cuesync/LICENSE). The workflows are under the [LTX-2.x Community License](workflows/LICENSE-LTX-2.x), and LTX-rendered shots carry its terms. See [NOTICE.md](NOTICE.md) for exceptions and third-party notices.
- **Models:** LTX-2.5 and ainvfx-fluid are open weights under the LTX-2.x Community License; the LTX-2.5 download is gated by a click-through on Hugging Face. [Files, versions and revisions](build/model-files.md).
- **Software:** ComfyUI (GPL-3.0); the Comfy Python SDK and comfy-api-proxy (Comfy Org); ComfyUI-LTXVideo (Lightricks); Next.js; PyAV/FFmpeg; OpenCV.
- **Demo narration:** Kokoro-82M (Apache-2.0), a synthetic narration voice used in the demo video; it is not part of the app. The performed fire cues use the author's recorded voice, including the two edited rhythms described above.
- **Footage and sounds:**
  - `hand_palm`, `hand_point`, `hand_walk`, `hand_punch`, `hand_fist` and `hand_open` are the author's own hand and voice.
  - `hand_cupped` is a Pexels clip (Kindel Media).
  - The other sample clips are stand-ins generated with LTX-2.5. The preset noises are synthetic except **pf, pf, BOOM** and **pf pf pf PF**, which are edited rhythms made from the author's recordings. The procedural lightning examples use synthetic noises over the author's footage.
  - The real-footage scene shots are made from stock clips and photos, and real people's CC0 voices, by the authors in [samples/CREDITS.md](samples/CREDITS.md). Only the processed scene shots are in the repository; the Pexels hand clip ships as-is.
