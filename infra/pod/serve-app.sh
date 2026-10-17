#!/usr/bin/env bash
# Serve the judge app from the GPU pod: the render API (full renders through the local Comfy API, plus
# instant mode) and the studio (Next.js production build) on 0.0.0.0:3000 behind HTTP Basic auth.
# Run by `podctl.py serve-app` after `up --judge`, `setup` and `sync`. CUE_PASSWORD comes on stdin.
set -euo pipefail
: "${CUE_PASSWORD:?set CUE_PASSWORD}"

SRC=/workspace/cue-src
LOG=/workspace/logs
NODE_DIR=/workspace/node
NODE_VERSION=v24.11.0
WEB=/workspace/web-build        # node_modules persists here across syncs
JOBS=/root/cue-jobs             # container disk: job files must not crawl through the network volume
API=http://127.0.0.1:8765
mkdir -p "$LOG" "$JOBS" "$WEB"
log() { printf '\n==> %s\n' "$*"; }

log "Node.js $NODE_VERSION"
if [ ! -x "$NODE_DIR/bin/node" ] || [ "$("$NODE_DIR/bin/node" --version)" != "$NODE_VERSION" ]; then
  curl -fsSL "https://nodejs.org/dist/$NODE_VERSION/node-$NODE_VERSION-linux-x64.tar.xz" -o /tmp/node.tar.xz
  rm -rf "$NODE_DIR" && mkdir -p "$NODE_DIR" && tar -xJf /tmp/node.tar.xz -C "$NODE_DIR" --strip-components=1
fi
export PATH="$NODE_DIR/bin:$PATH" NEXT_TELEMETRY_DISABLED=1
node --version

log "Studio build"
cp -r "$SRC/web/." "$WEB/"
cd "$WEB"
LOCK_SHA="$(sha256sum package-lock.json | cut -c1-16)"
if [ "$(cat node_modules/.lock-sha 2>/dev/null)" != "$LOCK_SHA" ]; then
  npm ci --no-audit --no-fund --loglevel=error && echo "$LOCK_SHA" > node_modules/.lock-sha
fi
CUESYNC_API_URL="$API" npm run build > "$LOG/web-build.log" 2>&1 || { tail -n 40 "$LOG/web-build.log"; exit 1; }

log "Services"
# Free both ports: an old instance would keep serving stale code (or an old password). Next renames its
# process to "next-server", so match that too; fuser catches anything else on the port.
pkill -f 'cuesync serv[e]' || true
pkill -f 'next star[t]' || true
pkill -f 'next-serve[r]' || true
command -v fuser >/dev/null 2>&1 && { fuser -k 3000/tcp 8765/tcp >/dev/null 2>&1 || true; }
for _ in $(seq 1 20); do
  if ! (exec 3<>/dev/tcp/127.0.0.1/3000) 2>/dev/null && ! (exec 3<>/dev/tcp/127.0.0.1/8765) 2>/dev/null; then break; fi
  sleep 0.5
done
cd "$SRC"
COMFY_BASE_URL=http://127.0.0.1:8189 setsid -f nohup /workspace/venv/bin/cuesync serve --jobs "$JOBS" \
  --host 127.0.0.1 --port 8765 --workflow workflows/fluid_element.api.json --library samples/library \
  < /dev/null > "$LOG/serve.log" 2>&1
cd "$WEB"
CUE_PASSWORD="$CUE_PASSWORD" CUESYNC_API_URL="$API" setsid -f nohup npx next start -H 0.0.0.0 -p 3000 \
  < /dev/null > "$LOG/web.log" 2>&1

log "Waiting for the render API and the studio"
for _ in $(seq 1 60); do
  if curl -fsS "$API/api/health" >/dev/null 2>&1 \
     && [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/)" = "401" ]; then break; fi
  sleep 2
done
curl -s "$API/api/health"; echo
echo "studio: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/) without a password (401 = gated)"
