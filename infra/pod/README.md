# GPU pod: setup and Day-1 smoke test

Heavy rendering (LTX-2.5, 22B) runs on a rented GPU. Your RTX 4070 (12 GB) can't hold it. This page takes you from nothing to "a fire element rendered through the Comfy SDK" in about an hour, most of it downloads.

## 1. Before you rent (5 min, on your laptop)

1. **Hugging Face:** sign in and click **Agree and access** on
   - https://huggingface.co/Lightricks/LTX-2.5 (required)
   - https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen (optional; the smoke-matte fallback)

   Then create a **read** token at https://huggingface.co/settings/tokens (enable "access to public gated repos"). Keep it in your password manager: you'll type it on the pod, and it never goes in this repo.
2. **SSH key:** if you don't have one, run `ssh-keygen -t ed25519`. Paste `~/.ssh/id_ed25519.pub` into Runpod → Settings → SSH Public Keys.

## 2. Rent the pod (Runpod)

- **GPU:** **RTX PRO 6000 (96 GB)** is preferred: everything stays resident, and it's the judge-backend target. H100/A100 80 GB also work. An RTX 5090 (32 GB) works but swaps models (slower).
- **Template:** an official **PyTorch** template with **CUDA 12.8+** (Blackwell GPUs need torch built for cu128).
- **Storage:** a **volume of 150 GB or more** mounted at `/workspace` (models are ~45 GB, plus ~42 GB with Alpha-Gen) and a container disk of 30 GB or more.
- **Expose TCP port 22** so you get the "SSH over exposed TCP" connection string. The basic ssh.runpod.io proxy can't do port forwarding.

## 3. Run the setup

From the repo on your laptop, copy the script, then SSH in:

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 infra/pod/pod-setup.sh root@<POD_IP>:/workspace/
```

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

On the pod:

```bash
export HF_TOKEN=hf_xxx            # paste your token here; never commit it
bash /workspace/pod-setup.sh --with-alphagen
```

It ends with **READY** once ComfyUI (`:8188`) and the Comfy API v2 proxy (`:8189`) are both healthy. Logs go to `/workspace/logs/`. Re-running is safe: it skips what's already done and restarts whatever isn't running.

## 4. Tunnel from your laptop

Keep this running in its own terminal:

```bash
ssh -N -L 8188:127.0.0.1:8188 -L 8189:127.0.0.1:8189 root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

Now `http://127.0.0.1:8188` is the pod's ComfyUI, and `http://127.0.0.1:8189` is its Comfy API v2. Both stay private to your laptop.

## 5. Graph export (one time, tonight)

1. Open http://127.0.0.1:8188. Workflow → Open → `ainvfx-fluid_painted_smoke_plume.json`. Click **Queue** unmodified; it must render the sample smoke plume to `output/ainvfx-fluid/`.
2. **Check frame 0** of that render. If the flat painted disc shows as the first frame, open the **Preprocess** subgraph, set `LTXVImgToVideoInplace` to **bypass = true**, and re-run.
3. At the **top level** of the graph, delete **Create Video** and **Empty Audio**, add a **Load Video** node, and connect it to the **Source Video** node's video input. Leave Painter → Batch Images → Input Parameters in place: Batch Images also feeds Input Parameters, and it's unused while I2V is bypassed. Frame rate and length are read from the loaded video, so there are no fps widgets to change.
4. Open **Input Parameters**. Make sure the **local Gemma** encoder branch is the one in use, not `GemmaAPITextEncode`, and leave the e2b prompt enhancer **off**.
5. In Load Video, pick **`03_control.mp4`** (the portrait fire sample, already in `input/`). Set the prompt to `ainvfxfluid, bright orange fire`. Click **Queue** and confirm a fire element renders.
6. Workflow → **Export (API)**. Save it as `workflows/fluid_element.api.json` in this repo.

## 6. SDK smoke test (laptop PowerShell, once `cuesync` is installed)

First create the folders, then copy the sample guide (the setup script already gave it a silent audio track) into `samples/` in this repo:

```powershell
New-Item -ItemType Directory -Force samples, workflows, runs | Out-Null
```

```powershell
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<POD_IP>:/workspace/ComfyUI/input/03_control.mp4 samples/
```

Install `cuesync` into a virtual environment:

```powershell
cd cuesync; if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }; .\.venv\Scripts\python -m pip install -e ".[comfy,dev]"; cd ..
```

(Uses Python 3.12, matching the pod. Calling the venv's own `python` avoids picking up the system Python 3.14. Run the CLI as `cuesync\.venv\Scripts\cuesync`, or activate the venv first.)

Point the SDK at the tunnelled proxy:

```powershell
$env:COMFY_BASE_URL = "http://127.0.0.1:8189"
```

Render the sample guide:

```powershell
cuesync\.venv\Scripts\cuesync render --guide samples/03_control.mp4 --prompt "ainvfxfluid, bright orange fire" --seed 42 --workflow workflows/fluid_element.api.json -o runs/smoke.mp4
```

Expected: `runs/smoke.mp4` comes back with the same frame count and frame rate as the guide (the sample may be 25 fps; our own guides are 24).

## 7. Costs and hygiene

- **Stop the pod** whenever you're not rendering. The `/workspace` volume keeps models and outputs; storage is billed while paused.
- **Never expose ports 8188 or 8189 publicly.** Use the SSH tunnel only.
- See `build/model-files.md` for exact files, licences and pinned commits.
