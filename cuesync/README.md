# cuesync

The core library behind **Cue the Boom**: a performed or in-clip sound sets when and how big fire, smoke or steam
gets; LTX-2.5 with the `ainvfx-fluid` IC-LoRA renders it into the scene; the render is re-timed onto the cue.

Scene mode (the default):

| Module | What it does |
|---|---|
| `cues` | The cue sheet from a voice take, the clip's own sound, or press-and-hold. |
| `audio` | Onsets, the hits of a rhythm, an absolute loudness envelope per video frame. |
| `gestures` | A cue's shape (hit, rhythm, sustained, breath, hiss) and the effect it asks for on a plate. |
| `scene` | The control video, painted the way ainvfx-fluid was trained: the plate in six flat zones, the effect at its source (or filling a drawing), timed and sized by the cue. |
| `runner` | Renders it through the Comfy SDK: LTX-2.5 image-to-video from the plate's frame. |
| `scene_shot` | Measures where the render's effect first shows, moves it onto the cue, adds flash and shake on blasts. |
| `transfer` | A moving plate keeps its own pixels: only what changed is added, in linear light, and checked. |
| `server`, `cli`, `precompute` | The studio's render API, the command line, the sample shots made ahead of time. |

Element mode (the effect rendered alone on black and composited onto the untouched plate) is `guide`, `score`,
`align`, `matte` and `composite`.

```powershell
if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }; .\.venv\Scripts\python -m pip install -e ".[server,comfy,dev]"; .\.venv\Scripts\python -m pytest -q
```

Copyright 2026 Grzegorz Szmulik. Licence: Apache-2.0 (LICENSE).
