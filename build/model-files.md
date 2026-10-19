# Model files, licences and pinned versions

All weights are downloaded by `infra/pod/pod-setup.sh`. None are committed to this repo.

## Pinned software

| Component | Version | Source |
|---|---|---|
| ComfyUI | v0.41.0, commit `98cc0bf84b7cb1383654c5b402e7612d875d6d22` | https://github.com/Comfy-Org/ComfyUI |
| ComfyUI-LTXVideo | commit `44c8a36b7c31ee00b4271daed11dc2a4aa481140` | https://github.com/Lightricks/ComfyUI-LTXVideo |
| comfy-api-proxy | latest 0.1.x from PyPI | https://github.com/Comfy-Org/comfy-api-proxy |
| comfy-sdk (Python) | 0.5.x | https://github.com/Comfy-Org/comfy-python-sdk |
| Pod image | `runpod/pytorch:1.4.0-cu1300-torch2121-ubuntu2404` (torch 2.12.1+cu130, Python 3.12) | https://hub.docker.com/r/runpod/pytorch |

ComfyUI v0.41.0 only enables comfy-kitchen's optimized int8 kernels with a cu130+ torch, so pods are placed on CUDA 13 hosts only.

## Model repo revisions

| Repo | Commit |
|---|---|
| Lightricks/LTX-2.5 | `49b093a39c19169a8a81721dcc541e07e945232f` |
| AInVFX/ainvfx-fluid | `c90ad4eb423b5d2f410193e9669e589b5b8c5af5` |
| Comfy-Org/gemma-4 | `63d0f7c476756b88910170c1df75e2384ea1af31` |
| Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen | `6184df14b1b560bf9b6447d3ee126bd7e4a88513` |

## Weights (Tier 0)

| File (as placed under `ComfyUI/models/`) | Hugging Face repo | Licence | Role |
|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 (gated, click-through) | LTX-2.x Community License | Base video model |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | Lightricks/LTX-2.5 | LTX-2.x Community License | Video VAE |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | Lightricks/LTX-2.5 | LTX-2.x Community License | Audio VAE (the graph expects an audio track) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 | Gemma terms | Local text encoder |
| `text_encoders/gemma4_e2b_it_int8_convrot.safetensors` | Comfy-Org/gemma-4 | Gemma terms | Prompt enhancer (not used: `cuesync graph` removes that branch) |
| `loras/ainvfx-fluid.safetensors` | AInVFX/ainvfx-fluid (`ltx-2.5/ainvfx-fluid.safetensors`) | LTX-2.x Community License | Fire, smoke and steam IC-LoRA (by Adrien Toupet) |

## Weights (fallback, `--with-alphagen`)

| File | Repo | Licence | Role |
|---|---|---|---|
| `diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors` | Lightricks/LTX-2.5 | LTX-2.x Community License | Full model needed by Alpha-Gen |
| `loras/alpha-gen/*` | Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen (gated) | LTX-2.x Community License | Smoke matte fallback |

## Hands (element bending), local, CPU or a consumer GPU

The tracker and the matte run in their own environments (`cuesync/.venv-track`, `cuesync/.venv-mask`, see the README); their models download on first use.

| Model | Source | Licence | Role |
|---|---|---|---|
| `gesture_recognizer.task` (float16; HandLandmarker + gesture classifier) | storage.googleapis.com/mediapipe-models (MediaPipe 1.1.1) | Apache-2.0 (MediaPipe model card) | Hand landmarks and gesture per frame |
| `rtmpose-m_hand5_256x256.onnx` (from `rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip`) | download.openmmlab.com (mmpose RTMPose), run with onnxruntime 1.31.0 | Apache-2.0 | Re-finds the hand where MediaPipe loses it |
| `sam2.1_hiera_small.pt` (facebook/sam2.1-hiera-small, rev `ee5bba1d`) | Hugging Face; code facebookresearch/sam2 at `2b90b9f5` | Apache-2.0 | The hand and forearm matte (video predictor) |
| `hand_landmarker.task`, `selfie_multiclass_256x256.tflite` | storage.googleapis.com/mediapipe-models | Apache-2.0 (MediaPipe model cards) | Other hands in the frame; the matte's fallback |

The demo's narration is Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0), a synthetic voice; it is not part of the app.

## Licence notes

- **LTX-2.x Community License:** free below $10M annual revenue. Its Attachment A clause 20 restricts use in apps that "directly compete with Licensor's commercial products" without a commercial licence. This applies to every LTX-based app; check the licence before commercial use.
- **Outputs** carry the base model's terms. Credit "LTX-2.5 by Lightricks" and "ainvfx-fluid by Adrien Toupet / AInVFX" in the app and the README.
