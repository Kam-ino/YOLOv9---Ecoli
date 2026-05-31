#!/usr/bin/env bash
# start.sh — one command from any state to a running app.
#
# Idempotent bootstrap + run:
#   1. create .venv if missing
#   2. install Python deps if anything's not importable
#   3. install + build the frontend if backend/app/static is empty
#   4. pre-download the pretrained yolov9c.pt if no model is present
#   5. exec uvicorn on 0.0.0.0:8003 (single process — serves UI and API)
#
# Usage:
#   ./start.sh
#   ECOLI_CONFIG=config.smoke.yaml ./start.sh   # force the COCO smoke model
#
# First run on a fresh Pi takes ~15 minutes (torch wheel on ARM is huge).
# Subsequent runs start in seconds.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

log() { printf '\033[1;36m→\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m✓\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m✗\033[0m %s\n' "$*" >&2; }

# --- 1. Python venv -------------------------------------------------------

if [[ ! -x ".venv/bin/python" ]]; then
    if ! command -v python3 >/dev/null; then
        err "python3 not installed."
        err "  sudo apt install -y python3 python3-venv python3-pip"
        exit 1
    fi
    log "Creating .venv ..."
    python3 -m venv .venv
    ok "venv created"
fi
PYTHON="$REPO/.venv/bin/python"

# --- 2. Python deps -------------------------------------------------------

# A quick import probe: faster than re-running pip every launch.
if ! "$PYTHON" -c "import fastapi, ultralytics, cv2, uvicorn" 2>/dev/null; then
    log "Installing Python deps (this is ~15 min on a Pi the first time)..."
    "$PYTHON" -m pip install --upgrade pip >/dev/null
    "$PYTHON" -m pip install -r requirements.txt
    ok "Python deps installed"
fi

# --- 3. Frontend bundle ---------------------------------------------------

if [[ ! -f "backend/app/static/index.html" ]]; then
    if ! command -v npm >/dev/null; then
        err "npm not installed (needed once to build the React bundle)."
        err "  sudo apt install -y nodejs npm"
        err ""
        err "Or build on a workstation and copy backend/app/static/ here:"
        err "  scp -r backend/app/static/ <user>@<this-host>:$REPO/backend/app/"
        exit 1
    fi
    if [[ ! -d "frontend/node_modules" ]]; then
        log "Installing frontend deps (~5 min on a Pi)..."
        (cd frontend && npm install --no-fund --no-audit)
    fi
    log "Building frontend..."
    (cd frontend && npm run build)
    ok "Frontend built into backend/app/static/"
fi

# --- 4. Make sure SOME model is available --------------------------------

# If you transferred a trained models/ecoli_yolov9c.pt this is a no-op.
# If you didn't, fall back to the smoke config + auto-download yolov9c.pt
# so the app can at least start.
if [[ ! -f "models/ecoli_yolov9c.pt" ]] && [[ ! -f "models/yolov9c.pt" ]]; then
    log "No model in models/ — downloading pretrained yolov9c.pt..."
    "$PYTHON" -c "from ultralytics import YOLO; YOLO('yolov9c.pt')"
    # Ultralytics drops the file in cwd; move it where the config expects.
    [[ -f yolov9c.pt ]] && mv yolov9c.pt models/yolov9c.pt
    ok "yolov9c.pt → models/"
fi

CONFIG="${ECOLI_CONFIG:-}"
if [[ -z "$CONFIG" ]]; then
    if [[ -f "models/ecoli_yolov9c.pt" ]]; then
        CONFIG="config.yaml"
    else
        log "No fine-tuned models/ecoli_yolov9c.pt — using config.smoke.yaml"
        log "(detect tab will run the COCO model and won't find bacteria;"
        log " scp your trained best.pt over and re-run for the real model)"
        CONFIG="config.smoke.yaml"
    fi
fi
export ECOLI_CONFIG="$CONFIG"

# --- 5. Run --------------------------------------------------------------

ok "All set. Open http://localhost:8003 (or http://<this-host>:8003 from another device)."
exec "$PYTHON" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8003
