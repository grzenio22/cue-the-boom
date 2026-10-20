# Cue the Boom

**Film your hand. Make the noise. Throw fire, lightning, water and air.**

Your voice drives visual effects. Hold your hand up to a phone, tap your palm or fingertips, and make the noise:
"BOOM" throws a fireball, a held "fffff" burns a flame in your open palm, a long "fffff" from pointed fingers is a flamethrower, a "pf" is a pop of fire, a hum charges an ember.
A buzzing "zzzzz" strikes lightning from your fingertips, and a long, rolling "BOOOOM" with a punch sets off an explosion. Water and air bend in your open palm too.
The sound sets when each hit fires, how big it is and how long it burns.

The fire is rendered once, over black, by **LTX-2.5** with the **[ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid)** IC-LoRA (open weights, in ComfyUI). Cue the Boom tracks your hand and carries the fire on it, frame by frame:
- it lights your skin from the side that faces it;
- your fingers hide its root;
- what you threw keeps flying the way you threw it.

Then **perform it again**: a new rhythm on the same fire, with no new render and no GPU.

Lightning, water and air are made on your computer, frame by frame, from your hand's track, its matte and your voice, with no render and no GPU ([below](#beyond-fire-lightning-water-air-and-an-explosion)).

![The author's own hand: BOOM throws a fireball off the fist, a held fffff keeps a flame in the palm through a walk from a hallway into a kitchen in one take, and a render performed again as pf pf pf PF](media/hand_hero.jpg)
<sub>The author's own hand and voice, filmed on a phone. The third is a render performed again in a new rhythm, with no new render.</sub>

An entry for Comfy's **Open Source Higgsfield** contest, built on the Comfy Developer Platform (the Comfy Python SDK against a self-hosted Comfy API), on open weights only.

[Try it](#try-it-in-two-minutes-no-gpu) · [How it works](#how-it-works) · [The entry at a glance](#the-entry-at-a-glance) · [With a GPU](#with-a-gpu) · [In detail](TECHNICAL.md)

## Try it in two minutes, no GPU

You need Python 3.10+ and git. No GPU and no model download: the studio ships built, and the sample hands come with their fire rendered ahead of time.

```bash
git clone https://github.com/grzenio22/cue-the-boom
cd cue-the-boom
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`; macOS and Linux: `source .venv/bin/activate`), then:

```bash
pip install "./cuesync[server]"
cuesync studio
```

The studio opens at http://127.0.0.1:8765 (`cuesync studio --port 8766` if that port is taken). In it:

1. Pick **Your hand: open palm** (the author's hand).
2. Cue it: pick the sample noise **pf, pf, KA-BOOM**, or **Use my microphone** and make your own noise while the clip plays.
3. **Make the shot**: three fireballs in the palm, one on each beat, each as big as it was loud, in about a minute on a laptop CPU.
4. **Perform it again** with another rhythm: the same fire, no new render.

Then, under **Made with it**, play **fffff · a flame in the palm, one take**: the camera walks from a hallway into a kitchen in one take, and the flame stays in the palm the whole way. To perform it yourself, pick **Your hand: a flame through the house**, then **Use my microphone** and hold an "fffff": the flame burns as long as you hold it.

**Live, in a second.** **Try it live** opens your webcam: raise your hand and say BOOM, and the fire is in your palm as you say it. A held "fffff" holds a flame, and pointing turns it into a jet. It's a rough preview of fire rendered ahead of time with LTX-2.5; **Record this** makes the real shot from that take.

**Your own hand.** Film it on your phone (held still, your hand in front of a plain, darker background), or with **Film it here**, then:
1. Pick **Use your own clip** (a take filmed here is already picked), then **Set up the hand tracker** (once, about a minute, CPU).
2. Tap your palm or fingertips and drag to aim, then **That's my hand**. The studio follows your hand through the clip and shows its path (red where it lost it) before anything is made.
3. Make the noise, then **Make the shot**. With no GPU, the fire is an element LTX-2.5 rendered ahead of time, carried on your hand.

The same from the command line (the shot lands in `runs/bend/final.mp4`):

```bash
cuesync bend --plate samples/plates/hand_palm.mp4 --voice samples/voices/pf-pf-ka-boom.wav --again samples/made_ahead/bending/hand_palm__boom__fire_burst
```

![The studio](media/studio.jpg)

## How it works

![How it's made: your clip and your noise, your hand found, the fire rendered once, carried, lit and timed](media/pipeline.jpg)

1. **Your sound** sets when each hit fires, how big it is and how long it lasts. With your hand's gesture, it also picks the effect.
2. **Your hand** is followed by MediaPipe and RTMPose, and cut out by SAM 2.1 (on a CPU, by MediaPipe's skin segmenter and the hand's own landmarks).
3. **The fire** is rendered once, over black, by LTX-2.5 with the ainvfx-fluid IC-LoRA in a ComfyUI workflow, from a guide your sound paints.
4. **Carried, lit, timed:** the fire is carried on your hand, lit by its own flame, smeared by the hand's motion like the hand itself, and fired on every hit.
5. **Perform it again:** a new take re-composites the same render in about a minute on a CPU. A held flame burns as long as your new voice holds it.

**Or tap anything in your shot.** "KA-BOOM" at a jack-o'-lantern, "fwoooOOOM" at a toy dinosaur: LTX-2.5 renders the effect into the scene, so it lights the room and leaves its mark.

Video models start an effect a few frames early. Cue the Boom measures where the fire really starts and moves it onto your sound. On 42 single hits, 37 land within one frame (1/24 s) of the cue; straight from the model, none do.

![The scene reacts: six real clips and photos, each with the sound that cued it](media/hero.jpg)
<sub>Real stock clips and photos ([credits](samples/CREDITS.md)), each with the sound that cued it. These play in the studio's gallery with no GPU.</sub>

The details (the pipeline step by step, the timing evaluation, the matte, the light) are in [TECHNICAL.md](TECHNICAL.md).

## Beyond fire: lightning, water, air and an explosion

In the studio, pick one of the author's hands, then a sample noise made for one of these four (**zzzzz · tsst!**, **BOOOOM**, **shhhhh · sploosh**, **hoooooo**, **hoo · hoo · HOO!**). You can also make the noise yourself and **Choose it yourself** under Make the shot. On a clip of your own, a buzz picks lightning and a long BOOM with a punch picks the explosion by themselves. All four play on your own clip with no GPU, in a minute or two.

- **Lightning** ("zzzzz"): channels struck from your fingertips into the room behind your hand, re-struck for a frame or two and then forking anew, or a crackling ball of plasma over your open palm. Your skin and the room flash cool-white with it.
- **An explosion** (a long "BOOOOM" with a punch): a fireball rendered with LTX-2.5 goes off ahead of your fist and stays where it went off as your hand moves. It rolls into black smoke that clears thinnest-first. With it come the flash, a shock front bending the room, sparks and embers, and the camera's knock.
- **Water** ("shhhhh"): a ball of water gathers over your palm and wobbles, the room bent and mirrored in it. Throw it and it bursts on the wall and runs down it. Or a stream rises from your palm and lashes like a whip.
- **Air** ("hoooo"): a spinning ball of air over your palm, seen only by what it bends and the dust it carries. Or a push of air that bends the room and blows dust at the lens.

Each is ray-traced or shaded in linear light over the plate itself, with its own Foley. How each is made: [TECHNICAL.md](TECHNICAL.md#lightning-water-air-and-an-explosion).

## The entry at a glance

- **Core video model:** LTX-2.5 (Lightricks) with the ainvfx-fluid IC-LoRA (AInVFX, Adrien Toupet). Every fire effect, and the explosion's fireball, is an LTX-2.5 render. Lightning, water and air are made procedurally on the CPU from your clip.
- **Comfy Developer Platform:** the renders are ComfyUI workflows ([`workflows/`](workflows)), run through the Comfy Python SDK against a self-hosted comfy-api-proxy. The ComfyUI node pack is in [`comfyui-cuesync/`](comfyui-cuesync).
- **Open weights only, self-hosted.** No partner nodes and no closed APIs. All models used:
  - **LTX-2.5** 22B (int8), its video and audio VAEs, and its **Gemma 4 12B** text encoder (Lightricks; LTX-2.x Community License, Gemma terms);
  - **ainvfx-fluid** IC-LoRA (AInVFX; LTX-2.x Community License);
  - **MediaPipe** gesture recognizer, hand landmarker and selfie segmenter (Google; Apache-2.0), the landmarker also in the browser for Try it live;
  - **RTMPose**-m hand (OpenMMLab; Apache-2.0);
  - **SAM 2.1**-small (Meta; Apache-2.0).
  - Optional: LTX-2.5 IC-LoRA **Alpha-Gen** (Lightricks), a smoke-matte fallback.
  - The demo video's narration is **Kokoro-82M** (Apache-2.0), a synthetic voice; it is not part of the app.
  - Versions and revisions: [build/model-files.md](build/model-files.md).
- **The app:** a browser studio and its render API (`cuesync studio`), a CLI (`cuesync`), and ComfyUI nodes.

## With a GPU

Your own renders, for your hand's own pose or into the scene, need ComfyUI with LTX-2.5 22B (int8) and ainvfx-fluid ([models](build/model-files.md); gated LTX-2.5 access on Hugging Face). A render fits a 32 GB card (about 3 minutes); on a rented RTX PRO 6000 (96 GB) it takes about a minute.

- **Your own GPU:** run ComfyUI with this repository and ComfyUI-LTXVideo in `custom_nodes/` ([TECHNICAL.md](TECHNICAL.md#in-comfyui)), run comfy-api-proxy in front of it, and point `COMFY_BASE_URL` at the proxy.
- **A rented GPU:** [`infra/pod/`](infra/pod/README.md) creates, sets up and terminates a disposable Runpod pod (it needs `RUNPOD_API_KEY` and `HF_TOKEN`; the models live on a network volume, about 120 GB). A watchdog terminates the pod when it is idle.

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

- **Code:** MIT, with `cuesync/` under Apache-2.0. The workflows are under the LTX-2.x Community License, and rendered shots carry its terms.
- **Models:** see [The entry at a glance](#the-entry-at-a-glance).
- **Software:** ComfyUI (GPL-3.0); the Comfy Python SDK and comfy-api-proxy (Comfy Org); ComfyUI-LTXVideo (Lightricks); Next.js; PyAV/FFmpeg; OpenCV.
- **Footage and sounds:**
  - `hand_palm`, `hand_point`, `hand_walk`, `hand_punch` and `hand_fist` are the author's own hand and voice.
  - `hand_cupped` is a Pexels clip (Kindel Media).
  - The other sample clips are stand-ins generated with LTX-2.5, and the sample noises are synthetic.
  - The real-footage scene shots are made from stock clips and photos, and real people's CC0 voices, by the authors in [samples/CREDITS.md](samples/CREDITS.md). Only the processed shots are in the repository.
