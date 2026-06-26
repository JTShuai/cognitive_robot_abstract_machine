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

# Locate start_rviz_vnc.sh robustly: it may be on PATH (baked image), next to
# this script, or only in the mounted source tree (older base images that
# predate the COPY of start_rviz_vnc.sh). Override with RVIZ_VNC_SCRIPT if needed.
RVIZ_VNC_SCRIPT="${RVIZ_VNC_SCRIPT:-}"
if [[ -z "${RVIZ_VNC_SCRIPT}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    for candidate in \
        "$(command -v start_rviz_vnc.sh 2>/dev/null || true)" \
        "${script_dir}/start_rviz_vnc.sh" \
        "/root/cognitive_robot_abstract_machine/start_rviz_vnc.sh"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            RVIZ_VNC_SCRIPT="${candidate}"
            break
        fi
    done
fi
if [[ -z "${RVIZ_VNC_SCRIPT}" ]]; then
    echo "ERROR: cannot find start_rviz_vnc.sh -- set RVIZ_VNC_SCRIPT=/path/to/start_rviz_vnc.sh" >&2
    exit 1
fi

# RViz restores its saved main-window geometry from the .rviz config. demo.rviz
# stores X: 3135, which is off-screen on a 1920-wide virtual display, so the
# window lands outside the framebuffer and VNC/noVNC shows only the empty root.
# Rewrite just the Window Geometry block into an on-screen copy under /tmp (the
# original config is left untouched) and launch RViz with that copy.
SCREEN="${SCREEN:-1920x1080x24}"
screen_width="${SCREEN%%x*}"
screen_rest="${SCREEN#*x}"
screen_height="${screen_rest%%x*}"
usable_height="$(( screen_height - 40 ))"  # leave room for the WM title bar
source_config="${1:-/root/cognitive_robot_abstract_machine/coraplex/demos/coraplex_bullet_world_demo/demo.rviz}"
if [[ -f "${source_config}" ]]; then
    onscreen_config="/tmp/rviz_onscreen.rviz"
    sed -E "/^Window Geometry:/,/^[^[:space:]]/ {
        s/^(  X:) .*/\1 0/
        s/^(  Y:) .*/\1 0/
        s/^(  Width:) .*/\1 ${screen_width}/
        s/^(  Height:) .*/\1 ${usable_height}/
    }" "${source_config}" > "${onscreen_config}"
    set -- "${onscreen_config}"
    echo ">>> on-screen RViz config: ${onscreen_config} (from ${source_config})"
fi

# Hand off to the unchanged VNC script: it sets up Xvfb/openbox/x11vnc and runs
# RViz in the foreground (exec). The backgrounded websockify survives the exec,
# exactly as Xvfb/openbox already do in that script.
echo ">>> handing off to ${RVIZ_VNC_SCRIPT}"
exec bash "${RVIZ_VNC_SCRIPT}" "$@"
