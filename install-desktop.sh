#!/usr/bin/env bash
# install-desktop.sh — one-time: install a clickable launcher for start.sh.
#
# Creates:
#   ~/.local/share/applications/ecoli-detector.desktop   (Apps menu entry)
#   ~/Desktop/ecoli-detector.desktop                      (if ~/Desktop exists)
#
# After this, double-click "E. coli Detector" — a terminal window opens
# and start.sh runs. Closing the window stops the app.
#
# Ubuntu only — on a headless Pi just run ./start.sh from SSH.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS="$HOME/.local/share/applications"
DESKTOP="$HOME/Desktop"

mkdir -p "$APPS"
DEST="$APPS/ecoli-detector.desktop"

cat > "$DEST" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=E. coli Detector
Comment=Bootstrap and launch the E. coli detection app
Exec=bash -c "cd '$REPO' && ./start.sh; echo; read -p '[press enter to close] '"
Path=$REPO
Terminal=true
Icon=applications-science
Categories=Science;Education;
EOF
chmod +x "$DEST"
echo "Installed: $DEST"

if [[ -d "$DESKTOP" ]]; then
    cp "$DEST" "$DESKTOP/ecoli-detector.desktop"
    chmod +x "$DESKTOP/ecoli-detector.desktop"
    # Nautilus by default refuses to launch untrusted .desktop files;
    # this marks it trusted so the icon name + double-click both work.
    # gio is part of glib (standard on Ubuntu).
    gio set "$DESKTOP/ecoli-detector.desktop" metadata::trusted true 2>/dev/null || true
    echo "Installed: $DESKTOP/ecoli-detector.desktop"
fi

# Refresh the desktop database so Apps menu picks it up immediately.
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true

echo
echo "Done. Either:"
echo "  - Double-click 'E. coli Detector' on your desktop, or"
echo "  - Find it in the Apps menu (Show Apps → 'E. coli Detector')."
