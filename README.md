# Cue the Boom

**Make the noise. Get the shot.**

Film an object, tap where the effect should come from, then *perform* it with your mouth: "pf.", "fwoooOOOM", "pfffff", "hissss". Your voice picks fire, smoke or steam and shapes it: when it erupts, how fast it builds, how big it gets and when it stops. The effect is rendered on open weights with **LTX-2.5** and the **ainvfx-fluid** IC-LoRA, then composited onto your footage. Your footage is never regenerated.

![Five noises, five shots](media/hero.jpg)
<sub>Instant-mode shots made on a laptop from the sample noises. The background plates are stand-ins generated with LTX-2.5; the effects are LTX-2.5 + ainvfx-fluid elements, timed, matted and composited by CueSync.</sub>

An entry for Comfy's **Open Source Higgsfield** contest, built on the Comfy Developer Platform (the Comfy Python SDK against a self-hosted Comfy API).

## Try it in two minutes (no GPU)

You need Python 3.10+.

```bash
git clone https://github.com/grzenio22/cue-the-boom && cd cue-the-boom
pip install "./cuesync[server]"
cuesync shot --instant --plate samples/plates/pumpkin_kitchen.mp4 --voice samples/voices/ka-boom.wav --effect fireball --x 0.5 --y 0.56
```

About 20 seconds later, `runs/shot/` holds:
- `final.mp4`: the shot, with the effect label and a live ribbon of your voice;
- `breakdown.mp4`: how it was made (plate, guide, rendered element, composite);
- `composite.mp4`: the clean shot;
- `receipt.json`: what was measured (where the hit landed, when it stopped) and what rendered it.

**Use your own.** Swap in your clip (phone held still, 1-15 s) and your noise (any audio file). `--x/--y` is where the effect comes from, from 0 to 1 across and down the frame. `--effect` is `fireball`, `smoke_puff`, `flame_plume` or `steam_jet`.

**Instant mode** picks one of 24 pre-rendered LTX-2.5 elements (`samples/library/`) for your performance. Then it does everything else from *your* recording:
- the hit lands on your hit, and a sustained effect follows your onset;
- the effect stops when you stop;
- it sits where you tapped, and is bigger or smaller with how loud you were.

The full mode below renders a new element from your voice, so the fluid itself takes the shape of your performance.

## The studio (web app)

```bash
cuesync serve --instant-only          # the render API, http://127.0.0.1:8765
cd web && npm install && npm run dev  # the app, http://localhost:3000
```

1. Pick a sample shot or your own clip.
2. Tap the spot.
3. Perform the noise: the clip plays muted while you do it. It only needs the microphone; adding your face as an inset is optional. Or use a sample noise.
4. Choose **Instant** or **Render it from my voice**.

Then perform it differently, on the same shot with the same seed, and compare the takes side by side.

## Full render (GPU)

The full pipeline needs ComfyUI with LTX-2.5 22B (int8) and ainvfx-fluid; peak VRAM is about 45 GB. We run it on a rented RTX PRO 6000 (96 GB).

```bash
python infra/pod/podctl.py up && python infra/pod/podctl.py setup   # Runpod pod: models, ComfyUI, Comfy API
python infra/pod/podctl.py tunnel                                   # in another terminal
COMFY_BASE_URL=http://127.0.0.1:8189 cuesync serve                   # full renders + instant mode
```

`podctl` creates, sets up and terminates a disposable pod through the Runpod API. A watchdog on the pod terminates it when idle. See [infra/pod/README.md](infra/pod/README.md).

A full shot takes about 45 s:

| Step | Time |
|---|---|
| Render (GPU) | ~20 s |
| SAM 2.1 object matte | ~2 s |
| Composite and finishing (CPU) | ~20 s |

`cuesync shot` without `--instant` does the same from the command line.

**In ComfyUI.** `workflows/cue_the_boom.api.json` is the whole shot as one workflow: your voice and your plate go in, the finished shot comes out.
1. Install the node pack: put `comfyui-cuesync/` in `ComfyUI/custom_nodes/`, then `pip install ./cuesync`.
2. Get the models: they're listed in [build/model-files.md](build/model-files.md).

`workflows/fluid_element.api.json` renders one element from a guide video, on open weights only.

## How it works

![How it was made](media/how-it-was-made.jpg)

1. **Listen.** Onsets, the hero hit, an absolute loudness envelope per video frame, and the stretches where you are actually making the noise ([`audio.py`](cuesync/src/cuesync/audio.py)).
2. **Paint the guide.** ainvfx-fluid is steered by a control video of coloured blobs. CueSync paints it from your voice: the blobs appear on your hit, grow with your loudness and die away when you stop ([`guide.py`](cuesync/src/cuesync/guide.py)).
3. **Render the element** with LTX-2.5 + ainvfx-fluid through the Comfy SDK. It is rendered on black, on its own ([`runner.py`](cuesync/src/cuesync/runner.py)).
4. **Align and gate.** The element's measured hit lands on your hit; light leads sound by a frame. A voice gate makes sustained fire collapse, lifting off its base, within a fraction of a second of you stopping, while smoke thins out ([`align.py`](cuesync/src/cuesync/align.py), [`composite.py`](cuesync/src/cuesync/composite.py)).
5. **Matte the object.** SAM 2.1, prompted by your tap, finds the object the effect comes from. That object stays in front of the effect, and nothing reaches below the surface it stands on ([`matte.py`](cuesync/src/cuesync/matte.py), the CueSync Object Matte node).
6. **Composite in linear light.**
   - Fire is light with an opaque core: it never draws a dark outline on a bright wall.
   - Smoke gets a density matte and is relit by the room: white smoke stays white on a bright wall, and a fireball's smoke is soot.
   - The fire lights the plate and the object.
   - Grain matches the plate.
   - Pixels nothing touches stay bit-exact.

**What is new here:** the performance *is* the control. Timing, size and the stop come from a noise you make in a second, not from a prompt or a keyframe editor. It runs as a reusable ComfyUI node pack. Sound-driven video effects have been explored before (for example [Soundini](https://arxiv.org/abs/2304.06818)). This is a usable, real-footage, open-weights tool built around that idea.

## Repository

| Path | What |
|---|---|
| `cuesync/` | The Python core and CLI: audio, guides, Comfy SDK runner, alignment, matte, compositor, pipeline, instant library, render API (Apache-2.0) |
| `comfyui-cuesync/` | ComfyUI node pack: CueSync Analyze Voice, Paint Guide, Object Matte, Composite |
| `workflows/` | ComfyUI API graphs (LTX-2.x Community License, derived from the ainvfx-fluid sample) |
| `web/` | The studio (Next.js) |
| `effects/` | The four effects, as JSON |
| `samples/` | Sample plates and noises, and the instant-mode element library |
| `infra/pod/` | Disposable Runpod GPU pods with a self-terminating watchdog |
| `build/model-files.md` | Exact models, licences and pinned revisions |

Tests: `pip install "./cuesync[server,dev]" && pytest cuesync/tests` (CPU only), plus `cd web && npm test`.

## Credits and licences

- **Code:** MIT, with `cuesync/` under Apache-2.0.
- **Workflows:** `workflows/` is under the LTX-2.x Community License (`workflows/LICENSE-LTX-2.x`).
- **Models:**
  - **LTX-2.5** by Lightricks (LTX-2.x Community License);
  - **ainvfx-fluid** by Adrien Toupet / AInVFX (LTX-2.x Community License);
  - **SAM 2.1** by Meta (Apache-2.0).
  - Rendered elements carry the LTX-2.x terms.
- **Software:** **ComfyUI** (GPL-3.0), the **Comfy Python SDK** and **comfy-api-proxy** by Comfy Org, **ComfyUI-LTXVideo** by Lightricks, **Next.js**, **PyAV/FFmpeg**, **OpenCV**.
- **Samples:** the sample noises are synthetic. The sample plates are stand-ins generated with LTX-2.5.
