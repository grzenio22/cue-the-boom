# Cue the Boom

**Make the noise. Get the shot.** Film a pumpkin, a mug or a toy, then perform the noise ("pf.", "fwoooOOOM", "KA-BOOM"). Your voice picks a fire, smoke or steam effect and shapes it: when it erupts, how fast it builds and how big it gets. The effect is rendered by LTX-2.5 with the `ainvfx-fluid` IC-LoRA, then composited onto your untouched footage, all through the Comfy Developer Platform on open weights.

An entry for Comfy's Open Source Higgsfield contest. It is work in progress.

| Path | What |
|---|---|
| `cuesync/` | Python core: cue sheets, voice-painted fluid guides, Comfy SDK runner, alignment, compositing (Apache-2.0) |
| `effects/` | Effect definitions (JSON) |
| `infra/pod/` | GPU pod setup (ComfyUI v0.41.0 + LTX-2.5 + comfy-api-proxy) |
| `build/model-files.md` | Exact model files, licences, pinned commits |
| `web/` | Web app (Next.js): Dub Mode recorder |

Licence: MIT (repo), Apache-2.0 (`cuesync/`). Models: see `build/model-files.md`.
