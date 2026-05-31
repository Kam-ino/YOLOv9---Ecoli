#!/usr/bin/env bash
# deploy/install-service.sh — register the backend (and optionally a
# Cloudflare tunnel) as systemd services so the Pi serves the UI on
# boot, restarts on crash, and survives power cuts.
#
# Run ONCE on the Pi from anywhere:
#     cd ~/YOLOv9-Ecoli
#     sudo ./deploy/install-service.sh
#
# After this, you never need to touch the backend again — power on the
# Pi and the Vercel UI is reachable as soon as it's on the network.
#
# Management:
#     sudo systemctl status ecoli-backend
#     sudo systemctl restart ecoli-backend
#     sudo journalctl -u ecoli-backend -f      # tail live logs
#
# Re-run this script after editing CORS_ORIGINS or moving the project.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This script needs root (systemctl writes under /etc/systemd/)."
    echo "Re-run with:  sudo $0"
    exit 1
fi

# ---------------------------------------------------------------------
# Detect repo path + the user who'll own the process.
# ---------------------------------------------------------------------

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# When invoked via sudo, $SUDO_USER is the real user; fall back to ubuntu.
APP_USER="${SUDO_USER:-ubuntu}"
APP_GROUP="$(id -gn "$APP_USER")"
VENV_PY="$REPO/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: $VENV_PY not found."
    echo "Run ./start.sh once as $APP_USER first — it bootstraps the venv."
    exit 1
fi

echo "Project:   $REPO"
echo "Runs as:   $APP_USER:$APP_GROUP"
echo "Python:    $VENV_PY"
echo

# ---------------------------------------------------------------------
# Collect runtime settings.
# ---------------------------------------------------------------------

read -r -p "Vercel UI origin (for CORS, e.g. https://my-app.vercel.app) [skip]: " CORS_ORIGIN
read -r -p "Port the backend should listen on [8003]: " PORT
PORT="${PORT:-8003}"
read -r -p "Use config.yaml (trained model) or config.smoke.yaml (COCO fallback)? [config.yaml]: " CFG
CFG="${CFG:-config.yaml}"

# ---------------------------------------------------------------------
# Write the backend service.
# ---------------------------------------------------------------------

BACKEND_SVC=/etc/systemd/system/ecoli-backend.service
echo
echo "Writing $BACKEND_SVC ..."

cat > "$BACKEND_SVC" <<EOF
[Unit]
Description=E. coli detection backend (uvicorn)
# Wait for the network to be fully up before we try to bind to 0.0.0.0.
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_GROUP
WorkingDirectory=$REPO
Environment=ECOLI_CONFIG=$CFG
Environment=CORS_ORIGINS=${CORS_ORIGIN}
# Unbuffered + UTF-8 — same hygiene as the in-process train subprocess.
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8
ExecStart=$VENV_PY -m uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
# Restart on crash but back off so a broken config doesn't spin the CPU.
Restart=on-failure
RestartSec=5
# Stream stdout / stderr into journald — view with 'journalctl -u ecoli-backend'.
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# ---------------------------------------------------------------------
# Optional: Cloudflare tunnel as a sibling service.
# ---------------------------------------------------------------------

echo
read -r -p "Also install a Cloudflare Tunnel service? (you must have already run 'cloudflared tunnel login' and 'cloudflared tunnel create <name>') [y/N]: " WANT_TUNNEL

if [[ "${WANT_TUNNEL,,}" == "y" ]]; then
    read -r -p "Tunnel name (the one passed to 'cloudflared tunnel create'): " TUNNEL_NAME
    if ! command -v cloudflared >/dev/null; then
        echo "cloudflared isn't installed. Install:"
        echo "    curl -fsSL https://pkg.cloudflare.com/install.sh | sudo bash"
        echo "    sudo apt install -y cloudflared"
        exit 1
    fi
    TUNNEL_SVC=/etc/systemd/system/ecoli-tunnel.service
    echo "Writing $TUNNEL_SVC ..."
    cat > "$TUNNEL_SVC" <<EOF
[Unit]
Description=Cloudflare Tunnel for E. coli backend
After=network-online.target ecoli-backend.service
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
# cloudflared reads credentials from ~/.cloudflared by default.
ExecStart=/usr/local/bin/cloudflared tunnel --no-autoupdate run $TUNNEL_NAME
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
fi

# ---------------------------------------------------------------------
# Activate.
# ---------------------------------------------------------------------

echo
echo "Reloading systemd and enabling services..."
systemctl daemon-reload
systemctl enable --now ecoli-backend.service
[[ -f /etc/systemd/system/ecoli-tunnel.service ]] && systemctl enable --now ecoli-tunnel.service

echo
echo "Done. Status:"
systemctl --no-pager --lines=3 status ecoli-backend.service || true
[[ -f /etc/systemd/system/ecoli-tunnel.service ]] && \
    systemctl --no-pager --lines=3 status ecoli-tunnel.service || true

echo
cat <<'EOF'
Management commands:
    sudo systemctl status ecoli-backend
    sudo systemctl restart ecoli-backend
    sudo journalctl -u ecoli-backend -f         # tail live logs
    sudo systemctl disable --now ecoli-backend  # turn it off
EOF
