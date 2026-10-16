#!/usr/bin/env bash
# Cue the Boom: Runpod GPU pod setup. Idempotent: safe to re-run.
#
# Installs ComfyUI v0.41.0 (pinned), the ComfyUI-LTXVideo node pack (pinned) and
# comfy-api-proxy, downloads the LTX-2.5 + ainvfx-fluid models, then starts
# ComfyUI on 127.0.0.1:8188 and the Comfy API v2 proxy on 127.0.0.1:8189.
# Both bind to loopback only; reach them from your laptop over an SSH tunnel.
#
# Usage (on the pod), either:
#   hf auth login                    # once: paste your token; it's saved on the pod, never in this repo
#   bash pod-setup.sh [--with-alphagen]
# or, for a single session:
#   export HF_TOKEN=hf_...; bash pod-setup.sh [--with-alphagen]
#
# Prerequisite: accept the LTX-2.5 terms at https://huggingface.co/Lightricks/LTX-2.5
# (and the Alpha-Gen repo, if you use --with-alphagen) with the same HF account.
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
COMFY_COMMIT="98cc0bf84b7cb1383654c5b402e7612d875d6d22"   # ComfyUI v0.41.0 (2026-10-15)
LTXV_COMMIT="44c8a36b7c31ee00b4271daed11dc2a4aa481140"    # Lightricks/ComfyUI-LTXVideo (2026-10-14)
# Use the interpreter that actually has torch (Runpod images differ: `python` vs `python3`).
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import torch' >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
: "${PY:?no python with torch found; pick a PyTorch template or set PYTHON=/path/to/python}"
LOG_DIR="/workspace/logs"
WITH_ALPHAGEN=0

for arg in "$@"; do
  case "$arg" in
    --with-alphagen) WITH_ALPHAGEN=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- system tools
log "System tools"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq ffmpeg git curl >/dev/null
fi
"$PY" - <<'PYCHECK'
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU visible: pick a GPU pod with a PyTorch (CUDA 12.8+) template.")
PYCHECK

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
"$PY" -m pip install -q -r "$COMFY_DIR/requirements.txt"

log "ComfyUI-LTXVideo node pack (pinned)"
clone_at "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo" https://github.com/Lightricks/ComfyUI-LTXVideo "$LTXV_COMMIT"
if [ -f "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt" ]; then
  "$PY" -m pip install -q -r "$COMFY_DIR/custom_nodes/ComfyUI-LTXVideo/requirements.txt"
fi

log "comfy-api-proxy + Hugging Face CLI"
"$PY" -m pip install -q comfy-api-proxy "huggingface_hub[cli]>=0.34"
if command -v hf >/dev/null 2>&1; then HF=(hf download); else HF=(huggingface-cli download); fi

# Hugging Face auth: HF_TOKEN from the environment, or a login saved earlier with `hf auth login`.
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
elif ! "$PY" -c 'import sys; from huggingface_hub import get_token; sys.exit(0 if get_token() else 1)'; then
  echo "No Hugging Face login found. On the pod run:  hf auth login   (paste a read token), then re-run this script." >&2
  echo "First accept the terms at https://huggingface.co/Lightricks/LTX-2.5 with the same account." >&2
  exit 1
fi

# ----------------------------------------------------------------- downloads
M="$COMFY_DIR/models"
STAGE="$COMFY_DIR/_downloads"
mkdir -p "$M/loras" "$STAGE" "$COMFY_DIR/input" "$COMFY_DIR/user/default/workflows"

get() {  # repo repo-relative-file dest-root  -> dest-root/repo-relative-file
  local repo="$1" file="$2" dest="$3"
  if [ -s "$dest/$file" ]; then echo "ok    $file"; return; fi
  echo "fetch $repo :: $file"
  "${HF[@]}" "$repo" "$file" --local-dir "$dest" >/dev/null
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
for pattern in "workflow/*" "samples/*_control.mp4" "LICENSE"; do   # one pattern per call: CLI flag semantics differ across versions
  "${HF[@]}" AInVFX/ainvfx-fluid --include "$pattern" --local-dir "$STAGE/ainvfx-fluid" >/dev/null
done
cp -n "$STAGE/ainvfx-fluid/ltx-2.5/ainvfx-fluid.safetensors" "$M/loras/ainvfx-fluid.safetensors" 2>/dev/null || true
cp -n "$STAGE"/ainvfx-fluid/workflow/*.json "$COMFY_DIR/user/default/workflows/" 2>/dev/null || true
cp -n "$STAGE"/ainvfx-fluid/workflow/inputs/*.png "$COMFY_DIR/input/" 2>/dev/null || true
# The sample control videos have no audio track, but the fluid graph decodes audio from its source video
# (VAEEncodeAudio), so add a silent track while copying them into input/.
for f in "$STAGE"/ainvfx-fluid/samples/*_control.mp4; do
  d="$COMFY_DIR/input/$(basename "$f")"
  [ -s "$d" ] || ffmpeg -loglevel error -y -i "$f" -f lavfi -i anullsrc=r=48000:cl=mono \
    -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "$d"
done
ls -1 "$COMFY_DIR/input" | sed 's/^/  input: /'

if [ "$WITH_ALPHAGEN" = 1 ]; then
  log "Alpha-Gen fallback (dev transformer + IC-LoRA) downloading in the BACKGROUND -> $LOG_DIR/alphagen-download.log"
  nohup bash -c "
    set -e
    if [ ! -s '$M/diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors' ]; then
      ${HF[*]} '$LTX' diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors --local-dir '$M'
    fi
    ${HF[*]} Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen --local-dir '$M/loras/alpha-gen'
    echo ALPHAGEN_DONE
  " > "$LOG_DIR/alphagen-download.log" 2>&1 &
fi

# ------------------------------------------------------------------ services
log "Starting services (loopback only)"
if pgrep -f "main.py --listen 127.0.0.1 --port 8188" >/dev/null; then
  echo "ComfyUI already running"
else
  (cd "$COMFY_DIR" && nohup "$PY" main.py --listen 127.0.0.1 --port 8188 > "$LOG_DIR/comfyui.log" 2>&1 &)
  echo "ComfyUI starting -> $LOG_DIR/comfyui.log"
fi
if pgrep -f "comfy-api-proxy" >/dev/null; then
  echo "comfy-api-proxy already running"
else
  nohup comfy-api-proxy --comfyui http://127.0.0.1:8188 --port 8189 --comfyui-base-dir "$COMFY_DIR" \
    > "$LOG_DIR/proxy.log" 2>&1 &
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

cat <<'NEXT'

READY.
1. On your LAPTOP, open the tunnel (use Runpod's "SSH over exposed TCP" details):
     ssh -N -L 8188:127.0.0.1:8188 -L 8189:127.0.0.1:8189 root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
2. Browse http://127.0.0.1:8188 and follow infra/pod/README.md, section "Graph export".
3. Smoke test from the laptop with COMFY_BASE_URL=http://127.0.0.1:8189 (see the README).
Stop the pod in the Runpod console when idle; the /workspace volume keeps models and outputs.
NEXT
