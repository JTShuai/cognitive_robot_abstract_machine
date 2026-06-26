#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Launch RViz2 on a HEADLESS virtual display inside the container and expose it
# over VNC. Built for remote / display-less hosts: RViz runs on Xvfb :99 and
# never touches the host's :0.
#
# Usage (inside the container):
#   start_rviz_vnc.sh [path/to/config.rviz]
#
# Override defaults via env vars:
#   DISPLAY_NUM(:99)  SCREEN(1920x1080x24)  VNC_PORT(5900)
#
# View it: connect a VNC viewer to <host>:VNC_PORT. If the port is firewalled,
# use an SSH tunnel:
#   ssh -L 5900:localhost:5900 <user>@<host>   # then VNC to localhost:5900
# ----------------------------------------------------------------------------
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-:99}"
SCREEN="${SCREEN:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5900}"
RVIZ_CONFIG="${1:-/root/cognitive_robot_abstract_machine/coraplex/demos/coraplex_bullet_world_demo/demo.rviz}"
DISPLAY_INDEX="${DISPLAY_NUM#:}"

# rviz2 not on PATH means ROS/cram-env wasn't sourced (e.g. docker exec skips
# the entrypoint) -- source it here.
if ! command -v rviz2 >/dev/null 2>&1; then
    source /opt/ros/jazzy/setup.bash
    [[ -f /opt/ros/overlay_ws/install/setup.bash ]] && source /opt/ros/overlay_ws/install/setup.bash
    source /opt/ros/cram-env/bin/activate
fi

# 1) Virtual X display (with GLX) -- reuse if already running.
if [[ ! -e "/tmp/.X11-unix/X${DISPLAY_INDEX}" ]]; then
    echo ">>> Xvfb ${DISPLAY_NUM} (${SCREEN})"
    Xvfb "${DISPLAY_NUM}" -screen 0 "${SCREEN}" +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
    sleep 1
fi
export DISPLAY="${DISPLAY_NUM}"

# 2) Window manager -- without it RViz's window lands off-screen (VNC stays black).
if ! pgrep -x openbox >/dev/null 2>&1; then
    echo ">>> openbox"
    openbox >/tmp/openbox.log 2>&1 &
    sleep 1
fi

# 3) VNC server -- -noxdamage polls the framebuffer, since RViz's GL window
#    emits no X DAMAGE events.
if ! pgrep -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1; then
    echo ">>> x11vnc ${DISPLAY_NUM} -> port ${VNC_PORT}"
    x11vnc -display "${DISPLAY_NUM}" -rfbport "${VNC_PORT}" -forever -shared -nopw -noxdamage -bg \
        >/tmp/x11vnc.log 2>&1
    sleep 1
fi

echo ">>> VNC ready: connect to <host>:${VNC_PORT} (SSH-tunnel to localhost:${VNC_PORT} if firewalled)"
echo ">>> RViz config: ${RVIZ_CONFIG}"

# 4) Run RViz with software GL (foreground; exec so it becomes the main process).
LIBGL_ALWAYS_SOFTWARE=1 exec rviz2 -d "${RVIZ_CONFIG}"
