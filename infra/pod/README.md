# GPU pod: setup and Day-1 smoke test

Heavy rendering (LTX-2.5, 22B) runs on a rented Runpod GPU; an RTX 4070 (12 GB) can't hold it. `podctl.py` drives the whole thing from your laptop through the Runpod API:

- **The models live on a network volume** (about 120 GB, roughly $8/month), so a pod is disposable. It's rented when there's rendering to do and terminated afterwards.
- **A watchdog on the pod terminates it** after 30 idle minutes, or at a hard deadline (4 h by default), so a forgotten pod can't run up a bill. The pod's start command arms it before anything else, it survives container restarts, and `up` won't leave a pod running until it has confirmed the watchdog is armed.
- **The image is `runpod/pytorch` with torch 2.12.1 on CUDA 13.0.** ComfyUI v0.41.0 only enables its fast int8 kernels with cu130 or newer, so pods are placed only on CUDA 13 hosts.
- **ComfyUI and the Comfy API stay on the pod's loopback.** Only SSH is exposed, and you reach them through an SSH tunnel.

## 1. One-time prerequisites

1. **Hugging Face:** sign in and click **Agree and access** on
   - https://huggingface.co/Lightricks/LTX-2.5 (required)
   - https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen (optional; the smoke-matte fallback)

   Then create a **read** token at https://huggingface.co/settings/tokens (enable "access to public gated repos").
2. **Runpod:** create an API key in Settings → API Keys, and add some credit.
3. **Environment variables** on your laptop: `RUNPOD_API_KEY` (or `RUNPOD_API`), plus `HF_TOKEN` for the first setup, while weights are still downloading. Neither goes in this repo.

`podctl.py` uses only the Python standard library and Runpod's REST API v2; any Python 3.10+ runs it. It creates its own SSH key (`~/.ssh/cue_pod_ed25519`) and passes the public half to each pod. Your Runpod account settings aren't touched.

The Hugging Face token reaches the setup script as an environment variable over SSH stdin. It is never written to a file, the volume or a command line. (The image keeps `HF_HOME` on the volume, so avoid `hf auth login` on the pod.)

## 2. Create the model volume (once)

Check what's in stock in data centers that support volumes:

```bash
python infra/pod/podctl.py gpus
```

```bash
python infra/pod/podctl.py volume create --dc EUR-IS-1 --size 120
```

Pick a data center with an **RTX PRO 6000 (96 GB)** on **CUDA 13**:
- **Fastest card:** Runpod's LTX benchmarks have it fastest, at about the same cost per clip as the cheap cards.
- **Tuned kernels:** ComfyUI's int8 kernels are tuned for it.
- **No swapping:** all ~37 GB of weights stay in VRAM.

Avoid the A100: its int8 path is slow. Pods can only use GPUs in the volume's data center.

## 3. Each session

Rent a pod (prints its GPU and price, then waits for SSH):

```bash
python infra/pod/podctl.py up
```

The first run downloads the models (~45 GB, plus ~42 GB with Alpha-Gen) and installs everything onto the volume. Later runs only start the services:

```bash
python infra/pod/podctl.py setup --with-alphagen
```

Then open the tunnel and keep it running:

```bash
python infra/pod/podctl.py tunnel
```

`http://127.0.0.1:8188` is now the pod's ComfyUI and `http://127.0.0.1:8189` its Comfy API v2.

When you're done, terminate the pod. The volume keeps everything:

```bash
python infra/pod/podctl.py down
```

Other commands:

| Command | What it does |
|---|---|
| `status` | Lists pods, the cost so far, and the watchdog state. It also flags any pod this laptop doesn't track. |
| `ssh -- <cmd>` | Runs a command on the pod. Without a command, it opens a shell. |
| `push` / `pull` | Copy files to or from the pod. |
| `extend <hours>` | Moves the hard deadline. `extend 0` terminates the pod within a minute, which also tests the watchdog. |
| `down --all` | Also terminates stray `cue-the-boom*` pods. |

## 4. Graph export (one time)

1. Open http://127.0.0.1:8188. Workflow → Open → `ainvfx-fluid_painted_smoke_plume.json`. Click **Queue** unmodified; it must render the sample smoke plume to `output/ainvfx-fluid/`.
2. **Check frame 0** of that render. If the flat painted disc shows as the first frame, open the **Preprocess** subgraph, set `LTXVImgToVideoInplace` to **bypass = true**, and re-run.
3. At the **top level** of the graph:
   - Delete **Create Video** and **Empty Audio**.
   - Add a **Load Video** node and connect it to the **Source Video** node's video input.
   - Leave Painter → Batch Images → Input Parameters in place: Batch Images also feeds Input Parameters, and it's unused while I2V is bypassed.

   Frame rate and length are read from the loaded video, so there are no fps widgets to change.
4. Open **Input Parameters**. Make sure the **local Gemma** encoder branch is the one in use, not `GemmaAPITextEncode`, and leave the e2b prompt enhancer **off**.
5. In Load Video, pick **`03_control.mp4`** (the portrait fire sample, already in `input/`). Set the prompt to `ainvfxfluid, bright orange fire`. Click **Queue** and confirm a fire element renders.
6. Workflow → **Export (API)**. Save it as `workflows/fluid_element.api.json` in this repo.

## 5. SDK smoke test (laptop PowerShell, once `cuesync` is installed)

Create the folders and fetch the sample guide (the setup script already gave it a silent audio track):

```powershell
New-Item -ItemType Directory -Force samples, workflows, runs | Out-Null
```

```powershell
python infra/pod/podctl.py pull /workspace/ComfyUI/input/03_control.mp4 samples/
```

Install `cuesync` into a virtual environment:

```powershell
cd cuesync; if (-not (Test-Path .venv)) { uv venv --python 3.12 .venv }; .\.venv\Scripts\python -m pip install -e ".[comfy,dev]"; cd ..
```

Point the SDK at the tunnelled proxy, then render the sample guide:

```powershell
$env:COMFY_BASE_URL = "http://127.0.0.1:8189"
```

```powershell
cuesync\.venv\Scripts\cuesync render --guide samples/03_control.mp4 --prompt "ainvfxfluid, bright orange fire" --seed 42 --workflow workflows/fluid_element.api.json -o runs/smoke.mp4
```

Expected: `runs/smoke.mp4` comes back with the same frame count and frame rate as the guide (the sample may be 25 fps; our own guides are 24).

## 6. Costs and hygiene

- **Pods bill per second while they exist.** `podctl.py down` ends billing. The watchdog is a safety net, not the plan.
- **The network volume bills monthly** (about $0.07/GB). To delete it, use Runpod → Storage when the project is over.
- **Never expose ports 8188 or 8189 publicly.** Use the SSH tunnel only.
- See `build/model-files.md` for exact files, licences and pinned commits.

### Without podctl

Any CUDA 13 GPU pod with the volume mounted at `/workspace` works. Copy `pod-setup.sh` over and `export HF_TOKEN=...` in that shell, then run:

```bash
bash /workspace/pod-setup.sh --with-alphagen
```

Then tunnel:

```bash
ssh -N -L 8188:127.0.0.1:8188 -L 8189:127.0.0.1:8189 root@<POD_IP> -p <PORT>
```
