# Workflows

| File | What |
|---|---|
| `cue_the_boom_scene.json` / `.api.json` | The whole scene shot, canvas and API format: your cue and plate in, the finished shot out. |
| `cue_the_boom.json` / `.api.json` | The whole element-mode shot: the effect rendered on black and composited onto the plate. |
| `fluid_element.api.json` | The render on its own: one fire, smoke or steam element (or a scene render) from a CueSync guide video. |

## Origin and modifications

It is a **modified version** of `workflow/ainvfx-fluid_painted_smoke_plume.json` from [AInVFX/ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid) (revision `c90ad4eb423b5d2f410193e9669e589b5b8c5af5`), by Adrien Toupet / AInVFX. That workflow is Complementary Material of LTX-2.5, so this derivative is distributed under the **LTX-2.x Community License Agreement** (`LICENSE-LTX-2.x` in this folder), not under this repository's MIT licence.

How it was produced:

1. ComfyUI v0.41.0 (`98cc0bf8`) with ComfyUI-LTXVideo (`44c8a36b`) loaded the original workflow, and its own exporter converted it to API format: `app.graphToPrompt()`.
2. `cuesync graph` (see `cuesync/src/cuesync/graph.py`) then made these changes:
   - The in-graph control video (Painter → Batch Images → Create Video, plus an empty audio track) is replaced by a single **Load Video** node. CueSync paints the guide video itself.
   - The image input of the (bypassed) image-to-video branch now takes the loaded frames.
   - The optional **Gemma API text encoder** (a hosted partner API) and the optional **LLM prompt enhancer** branches are removed. Their If/Else switches are replaced by the local branch, so the graph runs on open weights only.
   - Nodes that no longer feed the output are dropped.

Every remaining node and setting (models, sampler, sigmas, IC-LoRA guide, decode) is exactly as the original author shipped it.

Regenerate it:

```bash
cuesync graph ainvfx-fluid_sample.api.json -o workflows/fluid_element.api.json
```

## Models it loads

Download all of these with `infra/pod/pod-setup.sh`:
- `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`, `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`, and the LTX-2.5 video and audio VAEs, from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5).
- `ainvfx-fluid.safetensors`, from AInVFX/ainvfx-fluid.

LTX-2.5 by Lightricks; ainvfx-fluid by Adrien Toupet / AInVFX.

## The whole shot

**Scene mode.** `cue_the_boom_scene.json` (canvas format: open it in ComfyUI) and `cue_the_boom_scene.api.json` (API
format: the Comfy SDK) are the render graph with the CueSync node pack around it: Load Audio -> CueSync Analyze Sound
-> CueSync Paint Scene (with the plate from Load Video) feed the guide, the plate's first frame and the prompt into
LTX-2.5 image-to-video with ainvfx-fluid, and CueSync Scene Shot measures the render's onset, moves it onto the cue and
writes the shot for Save Video. `cuesync graph --scene` generates the API version.

**Element mode.** `cue_the_boom.json` and `cue_the_boom.api.json` are the element graph with Load Audio -> CueSync Analyze
Sound -> CueSync Paint Guide in place of its Load Video, and CueSync Object Matte + CueSync Composite putting the rendered
element onto the plate. `cuesync graph --shot` generates the API version.

All of these are modified from the ainvfx-fluid sample workflow as described above, and carry the same licence.

