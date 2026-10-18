#!/usr/bin/env bash
# Cue the Boom: terminate this pod when it is idle, so a forgotten pod can't run up a bill.
#
# The pod's start command launches this before anything else (see podctl.py), so it is
# armed from the first second and again after any container restart. Every minute:
#   1. the hard deadline is checked first (it lives on the volume, so restarts keep it;
#      `podctl.py extend` moves it);
#   2. then activity: a ComfyUI job running or queued, the setup script or a model
#      download running, GPU utilisation above 10%, or a keepalive touch from the
#      laptop (podctl commands touch it).
# After IDLE_MINUTES without activity, or at the deadline, the pod terminates itself
# with its pod-scoped API key, retrying every minute until it is gone. Every external
# call is time-bounded so a stuck driver, disk or API can't stall the loop.
set -u

IDLE_MINUTES="${IDLE_MINUTES:-30}"
MAX_HOURS="${MAX_HOURS:-4}"
POD="${RUNPOD_POD_ID:-unknown}"
# The CUE_* overrides exist for tests.
POLL="${CUE_POLL_SECONDS:-60}"
IDLE_LIMIT_S="${CUE_IDLE_SECONDS:-$(( IDLE_MINUTES * 60 ))}"
API="${CUE_API_BASE:-https://api.runpod.io/v2}/pods/$POD"
GRAPHQL="${CUE_GRAPHQL_URL:-https://api.runpod.io/graphql}"
COMFY="${CUE_COMFY_URL:-http://127.0.0.1:8188}"
UA="cue-the-boom-watchdog"
# The pod-scoped key Runpod injects; fall back to the container's init environment.
KEY="${RUNPOD_API_KEY:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_API_KEY=//p')}"
STATE_DIR="${CUE_STATE_DIR:-/workspace/.cue}"
TMP="${CUE_TMP:-/tmp}"
KEEPALIVE="$TMP/cue-keepalive"
LOG="$TMP/cue-watchdog.log"
DEADLINE_FILE="$STATE_DIR/deadline-$POD"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }
now() { date +%s; }

touch "$KEEPALIVE"
timeout 10 mkdir -p "$STATE_DIR" 2>/dev/null
if ! deadline="$(timeout 5 cat "$DEADLINE_FILE" 2>/dev/null)" || [ -z "$deadline" ]; then
  deadline=$(( $(now) + $(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }') ))
  echo "$deadline" | timeout 5 tee "$DEADLINE_FILE" >/dev/null 2>&1 || log "could not persist the deadline"
fi
log "armed: pod $POD key=$([ -n "$KEY" ] && echo yes || echo NO)" \
    "runpodctl=$(command -v runpodctl >/dev/null 2>&1 && echo yes || echo no)," \
    "idle limit $(( IDLE_LIMIT_S / 60 )) min, deadline $(date -u -d "@$deadline" +%FT%TZ)"

comfy_busy() {
  curl -fsS --max-time 5 "$COMFY/queue" 2>/dev/null | timeout 5 python3 -c '
import json, sys
q = json.load(sys.stdin)
sys.exit(0 if q.get("queue_running") or q.get("queue_pending") else 1)' 2>/dev/null
}

gpu_busy() {
  local util
  util="$(timeout 15 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -n1)"
  [ -n "$util" ] && [ "$util" -gt 10 ] 2>/dev/null
}

setup_busy() { pgrep -f 'pod-setup.sh|hf download|huggingface-cli download' >/dev/null 2>&1; }

graphql() {  # mutation -> success only if the response carries data and no errors
  local out
  out="$(curl -sS --max-time 30 -A "$UA" -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d "{\"query\":\"$1\"}" "$GRAPHQL" 2>>"$LOG")" || return 1
  log "graphql: ${out:0:200}"
  case "$out" in *'"errors"'*|'') return 1 ;; *'"data"'*) return 0 ;; *) return 1 ;; esac
}

rest() {  # method path [body] -> success on 2xx (or 404: already gone)
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 -A "$UA" -X "$1" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' ${3:+-d "$3"} "$API$2" 2>>"$LOG")"
  log "REST $1 $2 -> ${code:-no response}"
  case "$code" in 2??|404) return 0 ;; *) return 1 ;; esac
}

terminate() {
  # In order: the only in-pod path confirmed with pod-scoped keys (GraphQL), then runpodctl, then
  # REST v2. Stopping is the last resort: a stopped pod whose only storage is a network volume
  # bills nothing.
  log "terminating: $1"
  graphql "mutation { podTerminate(input: {podId: \\\"$POD\\\"}) }" && return
  if command -v runpodctl >/dev/null 2>&1; then
    timeout 60 runpodctl remove pod "$POD" >>"$LOG" 2>&1 && { log "runpodctl remove pod: ok"; return; }
  fi
  rest DELETE "" && return
  graphql "mutation { podStop(input: {podId: \\\"$POD\\\"}) { id desiredStatus } }" && return
  rest POST /action '{"action":"stop"}' && return
  log "could not terminate or stop; will retry next minute"
}

while sleep "$POLL"; do
  # `podctl.py extend` rewrites the deadline file; keep the last good value if the volume stalls.
  if d="$(timeout 5 cat "$DEADLINE_FILE" 2>/dev/null)" && [ -n "$d" ] && [ "$d" -eq "$d" ] 2>/dev/null; then
    deadline="$d"
  fi
  if [ "$(now)" -ge "$deadline" ]; then terminate "hard deadline reached"; continue; fi
  if comfy_busy || setup_busy || gpu_busy; then touch "$KEEPALIVE"; fi
  idle_s=$(( $(now) - $(stat -c %Y "$KEEPALIVE" 2>/dev/null || now) ))
  if [ "$idle_s" -ge "$IDLE_LIMIT_S" ]; then terminate "idle for $(( idle_s / 60 )) min"; fi
done
