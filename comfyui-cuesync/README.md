# CueSync for ComfyUI

Perform a noise ("pf.", "KA-BOOM", "fwoooOOOM", "hissss") and get fire, smoke or steam that erupts on your hit, grows with your voice and stops when you stop, composited onto your footage. Four nodes around the LTX-2.5 + [ainvfx-fluid](https://huggingface.co/AInVFX/ainvfx-fluid) IC-LoRA graph:

```
Load Audio ──> CueSync Analyze Voice ──> CueSync Paint Guide ──> guide video ─> LTX-2.5 + ainvfx-fluid ─> element
                                               │ prompt, plan                                              │
Load Video (your plate) ──┬──> CueSync Object Matte (SAM 2.1) ──────────────────────> CueSync Composite <──┘
                          └────────────────────────────────────────────────────────────────┘        └──> Save Video
```

| Node | In | Out | What it does |
|---|---|---|---|
| **CueSync Analyze Voice** | AUDIO, offset_s | CUESYNC_CUE, envelope IMAGE | Onsets, the hero hit, an absolute loudness envelope per video frame, and when you are actually making the noise. |
| **CueSync Paint Guide** | cue, effect, x, y, size, tail | VIDEO (guide), STRING (prompt), CUESYNC_PLAN | Paints the ainvfx-fluid control video from your voice: blobs appear on your hit, grow with your loudness and die away when you stop. Feed it to Get Video Components in place of the sample's painter. |
| **CueSync Object Matte** | IMAGE (plate frames), x, y or plan | MASK | Segments the object under the emission point with SAM 2.1 (Apache-2.0); it stays in front of the effect below that point. |
| **CueSync Composite** | plate VIDEO, element VIDEO, cue, plan, (voice AUDIO, object MASK) | VIDEO | Aligns the element's measured hit to your hit, gates it with your voice, and composites it in linear light: fire as light with an opaque core, smoke relit by the room, the object in front, the fire lighting the scene. |

The whole shot as one workflow: [`workflows/cue_the_boom.api.json`](../workflows/cue_the_boom.api.json). It needs the LTX-2.5 models and ainvfx-fluid listed in [`build/model-files.md`](../build/model-files.md).

## Install

Copy (or clone) this folder into `ComfyUI/custom_nodes/`, then install its dependencies into ComfyUI's Python:

```bash
pip install -r custom_nodes/comfyui-cuesync/requirements.txt
pip install "git+https://github.com/grzenio22/cue-the-boom#subdirectory=cuesync"
```

The effects (fireball, smoke puff, flame plume, steam jet) are JSON files in the repository's `effects/` folder; set `CUESYNC_EFFECTS` to use your own.

Licence: MIT (this node pack), Apache-2.0 (cuesync). Models: LTX-2.5 by Lightricks and ainvfx-fluid by Adrien Toupet / AInVFX (LTX-2.x Community License), SAM 2.1 by Meta (Apache-2.0).
