#!/usr/bin/env bash
# Run RViz on an isolated X display and expose it through VNC and noVNC.
set -euo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-:99}"
SCREEN="${SCREEN:-1920x1080x24}"
VNC_PORT="${VNC_PORT:-5900}"
WEB_PORT="${WEB_PORT:-6080}"
NOVNC_WEB="${NOVNC_WEB:-/usr/share/novnc}"
RVIZ_CONFIG="${1:?usage: start_rviz_web.sh CONFIG.rviz}"
DISPLAY_INDEX="${DISPLAY_NUM#:}"

source /opt/ros/jazzy/setup.bash
[[ -f /opt/ros/overlay_ws/install/setup.bash ]] +    && source /opt/ros/overlay_ws/install/setup.bash
source /opt/ros/cram-env/bin/activate

if [[ ! -e "/tmp/.X11-unix/X${DISPLAY_INDEX}" ]]; then
    Xvfb "${DISPLAY_NUM}" -screen 0 "${SCREEN}" +        +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
    sleep 1
fi
export DISPLAY="${DISPLAY_NUM}"

if ! pgrep -x openbox >/dev/null 2>&1; then
    openbox >/tmp/openbox.log 2>&1 &
    sleep 1
fi

if ! pgrep -f "x11vnc.*-rfbport ${VNC_PORT}" >/dev/null 2>&1; then
    x11vnc -display "${DISPLAY_NUM}" -rfbport "${VNC_PORT}" +        -forever -shared -nopw -noxdamage -bg >/tmp/x11vnc.log 2>&1
fi

if ! pgrep -f "websockify.*${WEB_PORT}" >/dev/null 2>&1; then
    websockify --web="${NOVNC_WEB}" "${WEB_PORT}" +        "localhost:${VNC_PORT}" >/tmp/novnc.log 2>&1 &
fi

screen_width="${SCREEN%%x*}"
screen_rest="${SCREEN#*x}"
screen_height="${screen_rest%%x*}"
usable_height="$((screen_height - 40))"
onscreen_config=/tmp/resym_rviz_onscreen.rviz
sed -E "/^Window Geometry:/,/^[^[:space:]]/ {
    s/^(  X:) .*/\1 0/
    s/^(  Y:) .*/\1 0/
    s/^(  Width:) .*/\1 ${screen_width}/
    s/^(  Height:) .*/\1 ${usable_height}/
}" "${RVIZ_CONFIG}" >"${onscreen_config}"

LIBGL_ALWAYS_SOFTWARE=1 exec rviz2 -d "${onscreen_config}"
