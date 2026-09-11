#!/usr/bin/env bash
# Run the demos inside the self-contained cram:jazzy-resym image (built from
# this project's Dockerfile; override with CRAM_IMAGE). Mounts the current CRAM
# monorepo at the same path used when the image was built.
#
#   ./scripts/run_in_docker.sh [apartment|kitchen|both] [kinematic|coraplex]
#   ./scripts/run_in_docker.sh test [pytest arguments...]
#   ./scripts/run_in_docker.sh calibrate [calibration arguments...]
#   ./scripts/run_in_docker.sh smoke [smoke arguments...]
#   ./scripts/run_in_docker.sh experiment [experiment arguments...]
#
# QT_QPA_PLATFORM=offscreen keeps WorldReasoner's Qt-based RDR machinery
# headless-safe.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRAM_ROOT="$(dirname "$PROJECT_ROOT")"
CONTAINER_ROOT=/opt/cram
IMAGE="${CRAM_IMAGE:-cram:jazzy-resym}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
MODE="${1:-both}"

case "$MODE" in
  test)
    shift
    if (( $# )); then
      COMMAND=(uv run --active --no-sync python -m pytest "$@")
    else
      COMMAND=(uv run --active --no-sync python -m pytest ${CONTAINER_ROOT}/test/resym_test)
    fi
    ;;
  calibrate)
    shift
    COMMAND=(uv run --active --no-sync python -m experiments.resym.icra.calibrate_splits "$@")
    ;;
  smoke)
    shift
    COMMAND=(uv run --active --no-sync python -m experiments.resym.icra.smoke_llm_episode "$@")
    ;;
  experiment)
    shift
    COMMAND=(uv run --active --no-sync python -m experiments.resym.icra.run "$@")
    ;;
  *)
    SCENE="$MODE"
    BACKEND="${2:-kinematic}"
    COMMAND=(uv run --active --no-sync python -m experiments.resym.open_drawer "$SCENE" "$BACKEND")
    ;;
esac

# Forward LLM selection + credentials for real-model runs. llm-agent-kit reads
# CLIENT_TYPE/API_KEY/BASE_URL from the environment (it does NOT auto-load .env),
# so the project .env is injected as real env vars via --env-file. Needs an image
# with llm-agent-kit installed, e.g. the local Dockerfile (cram:jazzy-resym).
LLM_ENV=()
if [[ "$MODE" == "smoke" || "$MODE" == "experiment" ]]; then
  for var in RESYM_LLM_MODEL RESYM_LLM_CONFIG; do
    [[ -n "${!var:-}" ]] && LLM_ENV+=(-e "$var=${!var}")
  done
  if [[ -f "$PROJECT_ROOT/.env" ]]; then
    LLM_ENV+=(--env-file "$PROJECT_ROOT/.env")
  fi
fi

# Persist the Hugging Face cache across containers: the RAG embedding model
# would otherwise be re-downloaded on every --rm run.
HF_CACHE="${RESYM_HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

docker run --rm \
  -v "$CRAM_ROOT:${CONTAINER_ROOT}" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -w "${CONTAINER_ROOT}/resym" \
  -e PYTHONPATH=${CONTAINER_ROOT}/resym/src:${CONTAINER_ROOT}/experiments/src \
  -e QT_QPA_PLATFORM=offscreen \
  -e RESYM_HOST_UID="$(id -u)" \
  -e RESYM_HOST_GID="$(id -g)" \
  -e RESYM_CONTAINER_IMAGE="$IMAGE" \
  -e RESYM_CONTAINER_IMAGE_ID="$IMAGE_ID" \
  "${LLM_ENV[@]}" \
  "$IMAGE" \
  bash -c '
    source /opt/ros/jazzy/setup.bash
    source /opt/ros/overlay_ws/install/setup.bash 2>/dev/null
    source /opt/ros/cram-env/bin/activate
    "$@"
    status=$?
    for path in config library runs .pytest_cache .ruff_cache src; do
      [[ -e "$path" ]] && chown -R "$RESYM_HOST_UID:$RESYM_HOST_GID" "$path"
    done
    for path in /opt/cram/experiments/src/experiments/resym /opt/cram/test/resym_test; do
      [[ -e "$path" ]] && chown -R "$RESYM_HOST_UID:$RESYM_HOST_GID" "$path"
    done
    chown -R "$RESYM_HOST_UID:$RESYM_HOST_GID" /root/.cache/huggingface 2>/dev/null
    exit "$status"
  ' bash "${COMMAND[@]}"
