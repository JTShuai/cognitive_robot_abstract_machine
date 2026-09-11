#!/usr/bin/env bash
# Run the open-drawer demo with live RViz2 visualization.
#
#   ./scripts/run_viz_demo.sh [apartment|kitchen] [local|web]
#   ./scripts/run_viz_demo.sh stop                # tear the container down
#
# Modes (second argument, default: local when $DISPLAY is set, else web):
#   local  RViz window opens directly on this machine's X display
#          (X11 socket mounted into the container; no VNC involved).
#   web    RViz runs headless on Xvfb :99 inside the container, viewable in a
#          browser via noVNC on port 6080 (for a headless rendering PC):
#          http://localhost:6080/vnc.html?autoconnect=1&resize=remote
#
# Both modes use the cram:jazzy-resym image (ROS lives only there). The demo
# publishes /tf and /semworld/viz_marker while the actual Coraplex plan runs.
# A live symbolic-execution page is served on port 5000 from the same trace.
# GPU for local mode: CRAM_GPU=nvidia|intel (default: software rendering).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRAM_ROOT="$(dirname "$PROJECT_ROOT")"
CONTAINER_ROOT=/opt/cram
IMAGE="${CRAM_IMAGE:-cram:jazzy-resym}"
CONTAINER="${CONTAINER:-resym_viz}"
WEB_PORT="${WEB_PORT:-6080}"
VNC_PORT="${VNC_PORT:-5900}"
VIEWER_PORT="${VIEWER_PORT:-5000}"
PLAN_PREVIEW_SECONDS="${RESYM_VIZ_PLAN_PREVIEW_SECONDS:-6}"
ACTION_PAUSE_SECONDS="${RESYM_VIZ_ACTION_PAUSE_SECONDS:-1}"
SECONDS_PER_UPDATE="${RESYM_VIZ_SECONDS_PER_UPDATE:-0.01}"
RECORD_VIDEO="${RESYM_VIZ_RECORD:-0}"
SCENE="${1:-apartment}"
if [[ -n "${2:-}" ]]; then
  MODE="$2"
elif [[ -n "${DISPLAY:-}" ]]; then
  MODE="local"
else
  MODE="web"
fi
RVIZ_CONFIG=${CONTAINER_ROOT}/experiments/src/experiments/resym/rviz/open_drawer.rviz

if [[ "$SCENE" == "stop" ]]; then
  docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "removed $CONTAINER" || echo "no container to remove"
  exit 0
fi

case "${RECORD_VIDEO,,}" in
  1|true|yes) RECORD_VIDEO=1 ;;
  0|false|no) RECORD_VIDEO=0 ;;
  *) echo "RESYM_VIZ_RECORD must be 0/1, false/true, or no/yes" >&2; exit 2 ;;
esac

if [[ "$RECORD_VIDEO" == 1 && "$MODE" != "web" ]]; then
  echo "RViz recording is supported only in web mode (isolated Xvfb :99)" >&2
  exit 2
fi

# The docker flags differ per mode and exposed port; recreate when either changed.
container_config="${MODE}-${VIEWER_PORT}-${WEB_PORT}-${VNC_PORT}"
current_mode="$(docker inspect -f '{{index .Config.Labels "resym.viz_mode"}}' "$CONTAINER" 2>/dev/null || true)"
if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true \
   || [[ "$current_mode" != "$container_config" ]]; then
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  run_args=(
    -d --name "$CONTAINER"
    --label "resym.viz_mode=${container_config}"
    -v "$CRAM_ROOT:${CONTAINER_ROOT}"
    -e PYTHONPATH=${CONTAINER_ROOT}/resym/src:${CONTAINER_ROOT}/experiments/src
    -p "${VIEWER_PORT}:${VIEWER_PORT}"
  )
  if [[ "$MODE" == "local" ]]; then
    command -v xhost >/dev/null 2>&1 && xhost +local:root >/dev/null 2>&1 || true
    run_args+=( -e "DISPLAY=${DISPLAY:-}" -v /tmp/.X11-unix:/tmp/.X11-unix )
    case "${CRAM_GPU:-}" in
      nvidia) run_args+=( --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all ) ;;
      intel)  run_args+=( --device /dev/dri ) ;;
    esac
  else
    run_args+=(
      -p "${WEB_PORT}:${WEB_PORT}"
      -p "${VNC_PORT}:${VNC_PORT}"
      -e "WEB_PORT=${WEB_PORT}"
      -e "VNC_PORT=${VNC_PORT}"
    )
  fi
  docker run "${run_args[@]}" "$IMAGE" sleep infinity >/dev/null
  echo "container $CONTAINER started (mode=${MODE})"
fi

if [[ "$RECORD_VIDEO" == 1 ]] && ! docker exec "$CONTAINER" sh -c 'command -v ffmpeg' >/dev/null; then
  echo "FFmpeg is missing from $IMAGE; rebuild it with: docker build -t $IMAGE ." >&2
  exit 2
fi

docker exec -d "$CONTAINER" bash -c \
  "source /opt/ros/cram-env/bin/activate;
   cd ${CONTAINER_ROOT}/resym;
   exec uv run --active --no-sync python -m resym.observability.viewer runs \
     --host 0.0.0.0 --port ${VIEWER_PORT} > /tmp/resym_viewer.log 2>&1"
echo ">>> symbolic execution: http://localhost:${VIEWER_PORT}/live"

if [[ "$MODE" == "local" ]]; then
  echo ">>> starting RViz on your display (DISPLAY=${DISPLAY:-unset})..."
  software_gl=""
  [[ -z "${CRAM_GPU:-}" ]] && software_gl="LIBGL_ALWAYS_SOFTWARE=1 "
  docker exec -d "$CONTAINER" bash -c \
    "source /opt/ros/jazzy/setup.bash; source /opt/ros/overlay_ws/install/setup.bash 2>/dev/null;
     ${software_gl}exec rviz2 -d $RVIZ_CONFIG > /tmp/rviz_local.log 2>&1"
  echo ">>> no window? run 'xhost +local:root' on the host; for HW GL set CRAM_GPU=nvidia|intel"
else
  echo ">>> starting RViz (headless :99 -> noVNC :${WEB_PORT})..."
  # Source the ROS environment before starting the headless RViz helper.
  docker exec -d "$CONTAINER" bash -c \
    "source /opt/ros/jazzy/setup.bash; source /opt/ros/overlay_ws/install/setup.bash 2>/dev/null;
     export AMENT_TRACE_SETUP_FILES='' COLCON_TRACE='';
     start_resym_rviz_web.sh $RVIZ_CONFIG > /tmp/rviz_web.log 2>&1"
  echo ">>> open in your browser:  http://localhost:${WEB_PORT}/vnc.html?autoconnect=1&resize=remote"
fi

echo ">>> starting the demo (Ctrl-C stops the demo, container keeps running)..."
echo ">>> playback: plan preview ${PLAN_PREVIEW_SECONDS}s, action pause ${ACTION_PAUSE_SECONDS}s, state-update delay ${SECONDS_PER_UPDATE}s"
record_option=""
if [[ "$RECORD_VIDEO" == 1 ]]; then
  record_option="--record-video"
  echo ">>> recording RViz to the run's D_visualization/rviz_execution.mp4"
fi
# -it only when we actually have a terminal (detached/ssh runs have none)
tty_args=()
[[ -t 0 && -t 1 ]] && tty_args=(-it)
docker exec "${tty_args[@]}" "$CONTAINER" bash -c \
  "source /opt/ros/jazzy/setup.bash && source /opt/ros/overlay_ws/install/setup.bash 2>/dev/null; \
   source /opt/ros/cram-env/bin/activate \
   && cd ${CONTAINER_ROOT}/resym \
   && uv run --active --no-sync python -m experiments.resym.open_drawer_viz $SCENE \
      --plan-preview-seconds $PLAN_PREVIEW_SECONDS \
      --action-pause-seconds $ACTION_PAUSE_SECONDS \
      --seconds-per-update $SECONDS_PER_UPDATE \
      $record_option"
