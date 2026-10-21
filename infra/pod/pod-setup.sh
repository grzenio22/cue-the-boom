#!/usr/bin/env bash
# Cue the Boom: Runpod GPU pod setup. Idempotent: safe to re-run.
#
# Installs ComfyUI v0.41.0 (pinned), the ComfyUI-LTXVideo node pack (pinned) and
# comfy-api-proxy, downloads the LTX-2.5 + ainvfx-fluid models, then starts
# ComfyUI on 127.0.0.1:8188 and the Comfy API v2 proxy on 127.0.0.1:8189.
# Both bind to loopback only; reach them from your laptop over an SSH tunnel.
#
# Everything lands on /workspace (a Runpod network volume), so a fresh pod with the
# same volume skips straight to starting the services. `podctl.py setup` runs this for you.
#
# Usage (on the pod):
#   export HF_TOKEN=hf_...; bash pod-setup.sh [--with-alphagen]
# A token is only needed while weights are still missing. Prefer the environment variable:
# runpod/pytorch images keep HF_HOME on the volume, so `hf auth login` would store the token there.
#
# Prerequisite: accept the LTX-2.5 terms at https://huggingface.co/Lightricks/LTX-2.5
# (and the Alpha-Gen repo, if you use --with-alphagen) with the same HF account.
set -euo pipefail
shopt -s nullglob

COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
COMFY_COMMIT="98cc0bf84b7cb1383654c5b402e7612d875d6d22"   # ComfyUI v0.41.0 (2026-10-15)
LTXV_COMMIT="44c8a36b7c31ee00b4271daed11dc2a4aa481140"    # Lightricks/ComfyUI-LTXVideo (2026-10-14)
VENV="${VENV:-/workspace/venv}"
# Model repos pinned to exact commits, so every pod gets byte-identical weights.
declare -A REV=(
  [Lightricks/LTX-2.5]=49b093a39c19169a8a81721dcc541e07e945232f
  [AInVFX/ainvfx-fluid]=c90ad4eb423b5d2f410193e9669e589b5b8c5af5
  [Comfy-Org/gemma-4]=63d0f7c476756b88910170c1df75e2384ea1af31
  [Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen]=6184df14b1b560bf9b6447d3ee126bd7e4a88513
)
SAM_REV=ee5bba1d82bb8749febdf90f45e84b687142ba03   # facebook/sam2.1-hiera-small (keep in sync with comfyui-cuesync/nodes.py)
# The image's interpreter that has torch (Runpod images differ: `python` vs `python3`).
BASE_PY="${PYTHON:-}"
if [ -z "$BASE_PY" ]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import torch' >/dev/null 2>&1; then BASE_PY="$(command -v "$c")"; break; fi
  done
fi
: "${BASE_PY:?no python with torch found; pick a PyTorch image or set PYTHON=/path/to/python}"
LOG_DIR="/workspace/logs"
export HF_XET_HIGH_PERFORMANCE=1
WITH_ALPHAGEN=0

for arg in "$@"; do
  case "$arg" in
    --with-alphagen) WITH_ALPHAGEN=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- system tools
log "System tools"
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq git curl >/dev/null
fi
"$BASE_PY" - <<'PYCHECK'
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU visible: pick a GPU pod with a PyTorch (CUDA 13) image.")
if int((torch.version.cuda or "0").split(".")[0]) < 13:
    print("WARNING: torch is built for CUDA", torch.version.cuda, "- ComfyUI only enables its fast int8 "
          "kernels with cu130+, so renders will be slower. Use a runpod/pytorch cu1300 image.")
PYCHECK

# A venv on the volume that reuses the image's torch, so pip installs survive pod turnover.
# It is rebuilt whenever the image's interpreter or torch build changes.
log "Python environment ($VENV)"
STAMP="$("$BASE_PY" -c 'import sys, torch; print(sys.executable, sys.version.split()[0], torch.__version__)')"
if [ ! -x "$VENV/bin/python" ] || [ "$(cat "$VENV/.base" 2>/dev/null)" != "$STAMP" ]; then
  rm -rf "$VENV"
  if ! "$BASE_PY" -m venv --system-site-packages "$VENV"; then
    echo "venv failed; installing python3-venv and retrying" >&2
    rm -rf "$VENV"
    apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null || true
    "$BASE_PY" -m venv --system-site-packages "$VENV" \
      || { "$BASE_PY" -m pip install -q virtualenv && "$BASE_PY" -m virtualenv -q --system-site-packages "$VENV"; }
  fi
  echo "$STAMP" > "$VENV/.base"
fi
PY="$VENV/bin/python"
echo "$STAMP"
# Never let a requirements file replace the image's CUDA build of torch.
"$BASE_PY" - > "$VENV/constraints.txt" <<'PIN'
import importlib.metadata as md
for name in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PIN
PIP=("$PY" -m pip install -q -c "$VENV/constraints.txt")

# ------------------------------------------------------------ pinned checkouts
clone_at() {  # dir url commit
  local dir="$1" url="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then git clone -q "$url" "$dir"; fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q --all
    git -C "$dir" checkout -q "$commit"
  fi
  echo "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

log "ComfyUI (pinned)"
clone_at "$COMFY_DIR" https://github.com/Comfy-Org/ComfyUI "$COMFY_COMMIT"
"${PIP[@]}" -r "$COMFY_DIR/requirements.txt"

log "ComfyUI-LTXVideo node pack (pinned)"
clone_at "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo" https://github.com/Lightricks/ComfyUI-LTXVideo "$LTXV_COMMIT"
if [ -f "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt" ]; then
  "${PIP[@]}" -r "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt"
fi

log "comfy-api-proxy + Hugging Face CLI"
"${PIP[@]}" comfy-api-proxy "huggingface_hub[cli]>=0.34"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
if [ -x "$VENV/bin/hf" ]; then HF=("$VENV/bin/hf" download); else HF=("$VENV/bin/huggingface-cli" download); fi

# Hugging Face auth, checked only once something actually needs downloading:
# HF_TOKEN from the environment, or a login saved earlier with `hf auth login`.
# Keep hf-xet's chunk cache on the container disk, not the volume.
export HF_XET_CACHE=/tmp/hf-xet
HF_AUTH_OK=0
ensure_hf_auth() {
  [ "$HF_AUTH_OK" = 1 ] && return
  if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN
  elif ! "$PY" -c 'import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)'; then
    echo "No Hugging Face token found: export HF_TOKEN=hf_... (a read token), then re-run this script." >&2
    echo "First accept the terms at https://huggingface.co/Lightricks/LTX-2.5 with the same account." >&2
    exit 1
  fi
  HF_AUTH_OK=1
}

# ----------------------------------------------------------------- downloads
M="$COMFY_DIR/models"
STAGE="$COMFY_DIR/_downloads"
mkdir -p "$M/loras" "$STAGE" "$COMFY_DIR/input" "$COMFY_DIR/user/default/workflows"

get() {  # repo repo-relative-file dest-root  -> dest-root/repo-relative-file
  local repo="$1" file="$2" dest="$3"
  if [ -s "$dest/$file" ]; then echo "ok    $file"; return; fi
  ensure_hf_auth
  echo "fetch $repo :: $file"
  "${HF[@]}" "$repo" "$file" --revision "${REV[$repo]}" --local-dir "$dest" >/dev/null
}

log "LTX-2.5 base models (gated: terms must be accepted)"
LTX="Lightricks/LTX-2.5"
get "$LTX" diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors "$M"
get "$LTX" vae/ltx-2.5-video-vae-bf16.safetensors "$M"
get "$LTX" vae/ltx-2.5-audio-vae-bf16.safetensors "$M"
get "$LTX" text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors "$M"
get Comfy-Org/gemma-4 text_encoders/gemma4_e2b_it_int8_convrot.safetensors "$M"

log "ainvfx-fluid IC-LoRA, workflow, sample control videos"
get AInVFX/ainvfx-fluid ltx-2.5/ainvfx-fluid.safetensors "$STAGE/ainvfx-fluid"
if [ ! -f "$STAGE/ainvfx-fluid/.extras-complete" ]; then
  ensure_hf_auth
  for pattern in "workflow/*" "samples/*_control.mp4" "LICENSE"; do   # one pattern per call: CLI flag semantics differ across versions
    "${HF[@]}" AInVFX/ainvfx-fluid --include "$pattern" --revision "${REV[AInVFX/ainvfx-fluid]}" --local-dir "$STAGE/ainvfx-fluid" >/dev/null
  done
  touch "$STAGE/ainvfx-fluid/.extras-complete"
fi

place() {  # src dst: copy unless an identical-size copy exists; never leave a partial file behind
  local src="$1" dst="$2"
  if [ -s "$dst" ] && [ "$(stat -c %s "$src")" = "$(stat -c %s "$dst")" ]; then return; fi
  cp "$src" "$dst.partial" && mv -f "$dst.partial" "$dst"
}
place "$STAGE/ainvfx-fluid/ltx-2.5/ainvfx-fluid.safetensors" "$M/loras/ainvfx-fluid.safetensors"
for f in "$STAGE"/ainvfx-fluid/workflow/*.json; do place "$f" "$COMFY_DIR/user/default/workflows/$(basename "$f")"; done
for f in "$STAGE"/ainvfx-fluid/workflow/inputs/*.png; do [ -e "$f" ] && place "$f" "$COMFY_DIR/input/$(basename "$f")"; done
# The sample control videos have no audio track, but the fluid graph decodes audio from its source video
# (VAEEncodeAudio), so add a silent track while copying them into input/.
for f in "$STAGE"/ainvfx-fluid/samples/*_control.mp4; do
  d="$COMFY_DIR/input/$(basename "$f")"
  [ -s "$d" ] && continue
  command -v ffmpeg >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }
  ffmpeg -loglevel error -y -i "$f" -f lavfi -i anullsrc=r=48000:cl=mono \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest -f mp4 "$d.partial" && mv -f "$d.partial" "$d"
done
ls -1 "$COMFY_DIR/input" | sed 's/^/  input: /'

if [ "$WITH_ALPHAGEN" = 1 ]; then
  log "Alpha-Gen fallback (dev transformer + IC-LoRA) downloading in the BACKGROUND -> $LOG_DIR/alphagen-download.log"
  nohup bash -c "
    set -e
    if [ ! -s '$M/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors' ]; then
      ${HF[*]} '$LTX' diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors --revision ${REV[$LTX]} --local-dir '$M'
    fi
    ${HF[*]} Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen --revision ${REV[Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen]} --local-dir '$M/loras/alpha-gen'
    echo ALPHAGEN_DONE
  " > "$LOG_DIR/alphagen-download.log" 2>&1 &
fi

log "SAM 2.1 small for CueSync Object Matte (public, Apache-2.0) + CueSync's own dependencies"
"${PIP[@]}" "transformers>=4.56" "opencv-python-headless>=4.10" "comfy-sdk>=0.5,<0.6" "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"
"$PY" - "$SAM_REV" <<'SAM'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download("facebook/sam2.1-hiera-small", revision=sys.argv[1],
                         allow_patterns=["*.json", "model.safetensors"])
print("sam2.1-hiera-small @", sys.argv[1][:7], "->", path)
SAM

# ------------------------------------------------------------------ services
unset HF_TOKEN   # the services never need it

# Read the big weights once now so the first render loads them from RAM instead of the
# network volume; the log records the volume's read speed.
log "Pre-reading model weights into the page cache (background) -> $LOG_DIR/prewarm.log"
setsid -f nohup "$PY" - "$M" > "$LOG_DIR/prewarm.log" 2>&1 <<'WARM'
import glob, os, sys, time
m = sys.argv[1]
for f in sorted(glob.glob(f"{m}/diffusion_models/ltx-2.5-22b-distilled-*.safetensors")
                + glob.glob(f"{m}/text_encoders/gemma4-12b-*.safetensors")):
    t, n = time.time(), 0
    with open(f, "rb", buffering=0) as fh:
        while chunk := fh.read(64 << 20):
            n += len(chunk)
    dt = max(time.time() - t, 1e-3)
    print(f"{os.path.basename(f)}: {n / 1e9:.1f} GB in {dt:.0f} s ({n / 1e6 / dt:.0f} MB/s)", flush=True)
WARM

log "Starting services (loopback only)"
# Torch sizes its CPU pool from the host's cores (255 on a shared EPYC), but the container may get only a
# quota of them (27): every CPU-side step then stalls on throttling, and a warm render took 14 min instead
# of 2. Size the pools to the quota (cgroup v2, else v1).
CPUS=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r q p < /sys/fs/cgroup/cpu.max
  [ "$q" != max ] && CPUS=$(( q / p ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)" -gt 0 ]; then
  CPUS=$(( $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us) ))
fi
[ "$CPUS" -ge 1 ] || CPUS=1
export OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS
echo "CPU threads: $CPUS (the container's share)"
if pgrep -f "main.py --listen 127.0.0.1 --port 8188" >/dev/null; then
  echo "ComfyUI already running"
else
  # --highvram keeps every model resident on the GPU; without it, DynamicVRAM re-streams the weights
  # from host RAM and renders slow from ~22 s to 45-90 s within a session.
  # setsid -f: fully detached, so this script (and the SSH session running it) can exit
  (cd "$COMFY_DIR" && setsid -f nohup "$PY" main.py --listen 127.0.0.1 --port 8188 --highvram < /dev/null > "$LOG_DIR/comfyui.log" 2>&1)
  echo "ComfyUI starting -> $LOG_DIR/comfyui.log"
fi
if pgrep -f "comfy-api-proxy" >/dev/null; then
  echo "comfy-api-proxy already running"
else
  setsid -f nohup "$VENV/bin/comfy-api-proxy" --comfyui http://127.0.0.1:8188 --port 8189 --comfyui-base-dir "$COMFY_DIR" \
    --max-upload-mb 500 < /dev/null > "$LOG_DIR/proxy.log" 2>&1
  echo "comfy-api-proxy starting -> $LOG_DIR/proxy.log"
fi

log "Waiting for ComfyUI (:8188) and the Comfy API v2 proxy (:8189), up to 5 minutes"
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1 \
     && curl -fsS http://127.0.0.1:8189/api/v2/health >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != 1 ]; then
  echo "Services did not come up. Last log lines:" >&2
  tail -n 30 "$LOG_DIR/comfyui.log" "$LOG_DIR/proxy.log" >&2 || true
  exit 1
fi
curl -fsS http://127.0.0.1:8189/api/v2/health; echo
if grep -q "pytorch with cu130" "$LOG_DIR/comfyui.log"; then
  echo "WARNING: ComfyUI disabled its optimized CUDA int8 kernels (torch is not cu130+)." >&2
fi
grep -E "Total VRAM|pytorch version|Model storage policy" "$LOG_DIR/comfyui.log" | sed 's/^/  /' || true

cat <<'NEXT'

READY.
1. On your LAPTOP:  python infra/pod/podctl.py tunnel
   (or: ssh -N -L 8188:127.0.0.1:8188 -L 8189:127.0.0.1:8189 root@<POD_IP> -p <PORT>)
2. ComfyUI is at http://127.0.0.1:8188; the element graph is workflows/fluid_element.api.json (see infra/pod/README.md).
3. Smoke test from the laptop with COMFY_BASE_URL=http://127.0.0.1:8189 (see the README).
When done:  python infra/pod/podctl.py down   (the /workspace volume keeps models and outputs; the watchdog terminates an idle pod anyway).
NEXT
