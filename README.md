# Cue the Boom

**Make the noise. Get the shot.** Film a pumpkin, a mug or a toy, then perform the noise ("pf.", "fwoooOOOM", "KA-BOOM"). Your voice picks a fire, smoke or steam effect and shapes it: when it erupts, how fast it builds and how big it gets. The effect is rendered by LTX-2.5 with the `ainvfx-fluid` IC-LoRA, then composited onto your untouched footage, all through the Comfy Developer Platform on open weights.

An entry for Comfy's Open Source Higgsfield contest. It is work in progress.

| Path | What |
|---|---|
| `cuesync/` | Python core: cue sheets, voice-painted fluid guides, Comfy SDK runner, alignment, compositing (Apache-2.0) |
| `effects/` | Effect definitions (JSON) |
| `workflows/` | The ComfyUI API graph that renders one element (derived from the ainvfx-fluid sample; LTX-2.x Community License) |
| `infra/pod/` | On-demand Runpod GPU pod: `podctl.py` (create, set up, tunnel, terminate; self-terminating when idle) and the pod setup script |
| `build/model-files.md` | Exact model files, licences, pinned commits |
| `web/` | Web app (Next.js): Dub Mode recorder |

Licence: MIT (repo), Apache-2.0 (`cuesync/`), LTX-2.x Community License (`workflows/`, a derivative of the ainvfx-fluid sample workflow). Models: see `build/model-files.md`.
