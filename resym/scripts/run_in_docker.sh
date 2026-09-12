#!/usr/bin/env bash
# Run one command inside the self-contained cram:jazzy-resym image (built from
# this project's Dockerfile; override with CRAM_IMAGE). Mounts the current CRAM
# monorepo at the same path used when the image was built, activates ROS and the
# project environment, and hands the rest of the argument list to `uv run`.
#
# This script only deploys the container. What runs inside it is the caller's
# input, so no experiment, module path or scene is wired in here:
#
#   ./scripts/run_in_docker.sh pytest /opt/cram/test/resym_test
#   ./scripts/run_in_docker.sh python -m experiments.resym.open_drawer apartment
#   ./scripts/run_in_docker.sh --with-llm-credentials python -m <module> [arguments]
#
# Options, which must precede the command:
#
#   --with-llm-credentials  forward .env and RESYM_LLM_* into the container, for
#                           commands that call a real model
#   --                      end of options
#
# Do not prefix the command with `uv run`; this script adds it.
#
# QT_QPA_PLATFORM=offscreen keeps WorldReasoner's Qt-based RDR machinery
# headless-safe. /opt/cram/tmp is on PYTHONPATH, so scratch packages kept
# outside the tracked source trees stay importable by their own name.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRAM_ROOT="$(dirname "$PROJECT_ROOT")"
CONTAINER_ROOT=/opt/cram
IMAGE="${CRAM_IMAGE:-cram:jazzy-resym}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"

usage() {
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

WITH_LLM_CREDENTIALS=0
while (( $# )); do
  case "$1" in
    --with-llm-credentials) WITH_LLM_CREDENTIALS=1; shift ;;
    --) shift; break ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) break ;;
  esac
done

(( $# )) || usage
COMMAND=(uv run --active --no-sync "$@")

# llm-agent-kit reads CLIENT_TYPE/API_KEY/BASE_URL from the environment (it does
# NOT auto-load .env), so the project .env is injected as real env vars via
# --env-file. Needs an image with llm-agent-kit installed, e.g. the local
# Dockerfile (cram:jazzy-resym). Credentials stay out unless asked for.
LLM_ENV=()
if (( WITH_LLM_CREDENTIALS )); then
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
  -e PYTHONPATH=${CONTAINER_ROOT}/resym/src:${CONTAINER_ROOT}/experiments/src:${CONTAINER_ROOT}/tmp \
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
    for path in /opt/cram/experiments/src/experiments/resym /opt/cram/test/resym_test /opt/cram/tmp; do
      [[ -e "$path" ]] && chown -R "$RESYM_HOST_UID:$RESYM_HOST_GID" "$path"
    done
    chown -R "$RESYM_HOST_UID:$RESYM_HOST_GID" /root/.cache/huggingface 2>/dev/null
    exit "$status"
  ' bash "${COMMAND[@]}"
