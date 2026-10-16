# Model files, licences and pinned versions

All weights are downloaded by `infra/pod/pod-setup.sh`. None are committed to this repo.

## Pinned software

| Component | Version | Source |
|---|---|---|
| ComfyUI | v0.41.0, commit `98cc0bf84b7cb1383654c5b402e7612d875d6d22` | https://github.com/Comfy-Org/ComfyUI |
| ComfyUI-LTXVideo | commit `44c8a36b7c31ee00b4271daed11dc2a4aa481140` | https://github.com/Lightricks/ComfyUI-LTXVideo |
| comfy-api-proxy | latest 0.1.x from PyPI | https://github.com/Comfy-Org/comfy-api-proxy |
| comfy-sdk (Python) | 0.5.x | https://github.com/Comfy-Org/comfy-python-sdk |

## Weights (Tier 0)

| File (as placed under `ComfyUI/models/`) | Hugging Face repo | Licence | Role |
|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 (gated, click-through) | LTX-2.x Community License | Base video model |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | Lightricks/LTX-2.5 | LTX-2.x Community License | Video VAE |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | Lightricks/LTX-2.5 | LTX-2.x Community License | Audio VAE (the graph expects an audio track) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 | Gemma terms | Local text encoder |
| `text_encoders/gemma4_e2b_it_int8_convrot.safetensors` | Comfy-Org/gemma-4 | Gemma terms | Prompt enhancer node (kept off) |
| `loras/ainvfx-fluid.safetensors` | AInVFX/ainvfx-fluid (`ltx-2.5/ainvfx-fluid.safetensors`) | LTX-2.x Community License | Fire, smoke and steam IC-LoRA (by Adrien Toupet) |

## Weights (fallback, `--with-alphagen`)

| File | Repo | Licence | Role |
|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 | LTX-2.x Community License | Full model needed by Alpha-Gen |
| `loras/alpha-gen/*` | Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen (gated) | LTX-2.x Community License | Smoke matte fallback |

## Licence notes

- **LTX-2.x Community License:** free below $10M annual revenue. Its Attachment A clause 20 restricts use in apps that "directly compete with Licensor's commercial products" without a commercial licence. This applies to every LTX-based app; check the licence before commercial use.
- **Outputs** carry the base model's terms. Credit "LTX-2.5 by Lightricks" and "ainvfx-fluid by Adrien Toupet / AInVFX" in the app and the README.
