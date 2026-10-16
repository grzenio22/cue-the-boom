# cuesync

The core library behind **Cue the Boom**. It turns a performed noise ("pf.", "fwoooOOOM", "KA-BOOM") into a voice-shaped fire, smoke or steam element for the LTX-2.5 `ainvfx-fluid` IC-LoRA, then composites that element onto untouched real footage.

Pipeline:
1. `conform`: phone clip → 24 fps CFR, 9:16.
2. `audio.analyze`: onsets plus an absolute-loudness envelope (below 11 kHz).
3. `guide.paint_guide`: blob guide video for `ainvfx-fluid`.
4. `runner.render_element`: Comfy SDK → ComfyUI.
5. `score.measure_element`: visible onset and peak.
6. `align.plan_alignment`: onset on the cue, peak on the loudness peak.
7. `composite.composite`: colour-split matte plus interactive glow.
8. `compare.side_by_side`: same footage, different performance.

```powershell
if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }; .\.venv\Scripts\python -m pip install -e ".[comfy,dev]"; .\.venv\Scripts\python -m pytest -q
```

Licence: Apache-2.0.
