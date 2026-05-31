#!/usr/bin/env bash
# run-prod.sh — single-process production mode.
#
# FastAPI serves the built React app + the /api/* endpoints from one
# process on one port. This is the right pattern for Raspberry Pi: no
# Node.js / Vite running on the device, no separate frontend process,
# fits cleanly behind a systemd unit.
#
# Prerequisites on this host:
#   - .venv with requirements installed (python3 -m venv .venv &&
#     .venv/bin/pip install -r requirements.txt)
#   - backend/app/static/index.html present (built bundle)
#       * either: install npm on this host and run
#         `cd frontend && npm install && npm run build`
#       * or: build on a workstation and `scp -r backend/app/static/`
#         here.
#   - models/ecoli_yolov9c.pt present (trained weights)

set -euo pipefail

export ECOLI_CONFIG="${ECOLI_CONFIG:-config.yaml}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PYTHON="$REPO/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

# Pre-flight: if there's no built bundle, attempt to build it; if we
# can't, fail with a clear message rather than silently serving an
# API-only host.
if [[ ! -f "$REPO/backend/app/static/index.html" ]]; then
    if command -v npm > /dev/null 2>&1; then
        echo "No built frontend — running 'npm run build'..."
        (cd "$REPO/frontend" && \
         [[ -d node_modules ]] || npm ci --no-fund --no-audit; \
         npm run build)
    else
        cat <<EOF >&2
ERROR: backend/app/static/index.html doesn't exist and npm isn't on this host.

You need a built frontend bundle for production mode. Either:

  (a) Install Node + npm here:
        sudo apt-get install -y nodejs npm   # Raspberry Pi OS / Debian
      then re-run this script.

  (b) Build on a workstation and copy the bundle over:
        # on workstation:
        cd frontend && npm run build
        # then:
        scp -r backend/app/static/ <user>@<this-host>:$REPO/backend/app/

EOF
        exit 1
    fi
fi

echo "Serving UI + API on http://0.0.0.0:8003 (config=$ECOLI_CONFIG)"
exec "$PYTHON" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8003
