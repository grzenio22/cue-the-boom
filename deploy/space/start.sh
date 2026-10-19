#!/usr/bin/env bash
# The render API (instant mode, on 127.0.0.1 only) and the studio (port 7860, behind CUE_PASSWORD when it is set).
set -euo pipefail
cd /app
mkdir -p /tmp/cue-jobs
cuesync serve --instant-only --host 127.0.0.1 --port 8765 --jobs /tmp/cue-jobs > /tmp/cue-serve.log 2>&1 &
for _ in $(seq 1 60); do curl -fs http://127.0.0.1:8765/api/health > /dev/null && break; sleep 1; done
cd web
exec npx next start -H 0.0.0.0 -p "${PORT:-7860}"
