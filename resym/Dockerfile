# Build from the CRAM root:
#
#   docker build -f resym/Dockerfile -t cram:jazzy-resym .
#
ARG FROM_IMAGE=osrf/ros:jazzy-desktop-full
FROM ${FROM_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG CRAM_SRC=/opt/cram
ARG OVERLAY_WS=/opt/ros/overlay_ws
ENV OVERLAY_WS=${OVERLAY_WS}
ENV CRAM_VENV=/opt/ros/cram-env
ENV UV_PROJECT_ENVIRONMENT=${CRAM_VENV}
ENV DEBIAN_FRONTEND=noninteractive

ARG INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${INDEX_URL}
ENV UV_DEFAULT_INDEX=${INDEX_URL}

# Build the ROS overlay workspace (iai_pr2 / iai_maps / ...).
COPY .github/docker/setup_ros_workspace.sh /tmp/setup_ros_workspace.sh
COPY .github/docker/setup_workspace.py /tmp/setup_workspace.py
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates ros-jazzy-ur-client-library \
    && OVERLAY_WS=${OVERLAY_WS} bash /tmp/setup_ros_workspace.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# cram-env includes the ROS Python packages exposed by the base image.
RUN source /opt/ros/jazzy/setup.bash \
    && python3 -m venv "${CRAM_VENV}" --system-site-packages \
    && source "${CRAM_VENV}/bin/activate" \
    && pip install --no-cache-dir uv

# Install this checkout, including resym, as a single CRAM workspace.
COPY . ${CRAM_SRC}
WORKDIR ${CRAM_SRC}
RUN source /opt/ros/jazzy/setup.bash \
    && source "${OVERLAY_WS}/install/setup.bash" \
    && source "${CRAM_VENV}/bin/activate" \
    && uv sync --extra dev --active \
    && uv run --active --no-sync resym-install-ontologies \
    && uv run --active --no-sync resym-install-corpus

# Keep the planner outside the source mount used by the runtime scripts.
ENV RESYM_FAST_DOWNWARD_DIRECTORY=/opt/fast-downward
RUN source "${CRAM_VENV}/bin/activate" \
    && ./resym/scripts/install_fast_downward.sh

# Qt and the RViz-over-VNC stack used by the visualization demo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        xvfb x11vnc openbox novnc websockify ffmpeg \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1 \
        libxkbcommon-x11-0 libegl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN source "${CRAM_VENV}/bin/activate" \
    && uv pip install PyQt6 qtconsole flask

# Dense retrieval uses CPU-only torch to avoid shipping the CUDA runtime.
RUN source "${CRAM_VENV}/bin/activate" \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install sentence-transformers

# OpenAI 3 uses httpx2, whose bounded Brotli decoder requires Brotli 1.2.0.
RUN source "${CRAM_VENV}/bin/activate" \
    && uv pip install "Brotli==1.2.0"

RUN install -m 0755 \
    "${CRAM_SRC}/resym/scripts/start_rviz_web.sh" \
    /usr/local/bin/start_resym_rviz_web.sh

COPY .github/docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["bash", "/entrypoint.sh"]
CMD ["bash"]
