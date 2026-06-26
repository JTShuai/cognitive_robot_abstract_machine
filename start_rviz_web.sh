#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Launch RViz2 headless (Xvfb :99) AND expose it in a WEB BROWSER via noVNC.
#
# This is a PURELY ADDITIVE wrapper around start_rviz_vnc.sh -- it changes
# nothing in the original VNC flow. It brings up a websockify + noVNC web proxy
# in front of the existing x11vnc (VNC_PORT, 5900 by default), then hands off to
# start_rviz_vnc.sh unchanged. The plain VNC port keeps working exactly as
# before; the browser view is an extra entry point on WEB_PORT (6080).
#
# Usage (inside the container):
#   start_rviz_web.sh [path/to/config.rviz]
#
# Then, from a browser on the host (e.g. inside the Remmina :0 desktop):
#   http://localhost:6080/vnc.html?autoconnect=1&resize=remote
# No SSH tunnel is needed when the browser runs on the same host that publishes
# the container's WEB_PORT.
#
# Override defaults via env vars:
#   WEB_PORT(6080)  VNC_PORT(5900)  NOVNC_WEB(/usr/share/novnc)
#   (plus everything start_rviz_vnc.sh understands: DISPLAY_NUM, SCREEN, ...)
# ----------------------------------------------------------------------------
set -euo pipefail

WEB_PORT="${WEB_PORT:-6080}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_WEB="${NOVNC_WEB:-/usr/share/novnc}"

# noVNC + websockify are not in the base cram:jazzy image. Install on first use
# so this works without rebuilding the image (idempotent: skipped once present,
# e.g. when baked in via Dockerfile.cram-web).
if ! command -v websockify >/dev/null 2>&1 || [[ ! -e "${NOVNC_WEB}/vnc.html" ]]; then
    echo ">>> installing noVNC + websockify (one-time)"
    apt-get update
    apt-get install -y --no-install-recommends novnc websockify
fi

# Web proxy: wrap the RFB stream (x11vnc on VNC_PORT) as WebSocket for the
# browser. websockify dials its target lazily per client connection, so starting
# it before x11vnc is up is fine. Reuse if already running.
if ! pgrep -f "websockify.*${WEB_PORT}" >/dev/null 2>&1; then
    echo ">>> noVNC web on ${WEB_PORT} -> localhost:${VNC_PORT}"
    websockify --web="${NOVNC_WEB}" "${WEB_PORT}" "localhost:${VNC_PORT}" >/tmp/novnc.log 2>&1 &
    sleep 1
fi

echo ">>> Browser view: http://localhost:${WEB_PORT}/vnc.html?autoconnect=1&resize=remote"

# Hand off to the unchanged VNC script: it sets up Xvfb/openbox/x11vnc and runs
# RViz in the foreground (exec). The backgrounded websockify survives the exec,
# exactly as Xvfb/openbox already do in that script.
exec start_rviz_vnc.sh "$@"
