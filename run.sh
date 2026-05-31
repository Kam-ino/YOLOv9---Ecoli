#!/usr/bin/env bash
# run.sh — start backend + frontend in one shot (Linux / macOS, incl. Raspberry Pi).
#
# Two processes:
#   - backend (uvicorn)    http://<host>:8003
#   - frontend (vite dev)  http://<host>:3003
#
# Ctrl+C stops both — the trap fires and SIGTERMs the backend.
# Switch ECOLI_CONFIG to config.smoke.yaml to fall back to the COCO model.

set -euo pipefail

export ECOLI_CONFIG="${ECOLI_CONFIG:-config.yaml}"

# Resolve repo root from this script's location so it works no matter
# where you invoke it from.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

# Pick the venv python: .venv on Linux/macOS, fall back to system if absent.
PYTHON="$REPO/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

cleanup() {
    if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
        wait "$BACKEND_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting backend on http://0.0.0.0:8003 (config=$ECOLI_CONFIG)..."
"$PYTHON" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8003 &
BACKEND_PID=$!

echo "Starting frontend on http://0.0.0.0:3003..."
cd "$REPO/frontend"
npm run dev
