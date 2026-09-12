"""
Open a drawer through Coraplex while publishing the live world to RViz.

The pipeline and the visualization observe the same execution. Structured events are
written under ``runs/`` for the live web viewer; there is no post-execution animation or
operator-name-based replay.

Run through ``scripts/run_viz_demo.sh [apartment|kitchen] [local|web]``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from typing_extensions import Any

from experiments.resym.scenes import Scene, load_scene
from experiments.resym.seed_library import build_seed_library
from resym.core.model import Literal
from resym.observability.runlog import RunRecorder
from experiments.resym.grounding_initialization import default_drawer_grounding_catalog
from resym.platform.kinematic import KinematicFeasibility
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import pddl_name
from resym.platform.cram_objects import task_object_universe
from resym.planning.events import PipelineEvent, PipelineEventSink
from resym.planning.execution.coraplex import CoraplexSkillRealization
from resym.planning.pipeline import solve_task
from semantic_digital_twin.callbacks.callback import StateChangeCallback


@dataclass(eq=False)
class PlaybackPacer(StateChangeCallback):
    """
    Pace live world updates without changing the simulated controller.
    """

    seconds_per_update: float = 0.01

    def on_state_change(self, **kwargs) -> None:
        time.sleep(self.seconds_per_update)


@dataclass
class RvizVideoRecorder:
    """
    Record the isolated Xvfb display and finalize one MP4 artifact.
    """

    output_path: Path
    display: str = ":99.0"
    video_size: str = "1920x1080"
    fps: int = 30
    process: Any = None
    log_stream: Any = None
    started_at: str | None = None

    @property
    def metadata_path(self) -> Path:
        return self.output_path.with_name("rviz_recording.json")

    @property
    def log_path(self) -> Path:
        return self.output_path.with_name("ffmpeg.log")

    @property
    def command(self) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "x11grab",
            "-draw_mouse",
            "0",
            "-framerate",
            str(self.fps),
            "-video_size",
            self.video_size,
            "-i",
            self.display,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]

    def start(self) -> None:
        if self.process is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_stream = self.log_path.open("wb")
        self.started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self.log_stream,
            )
        except Exception:
            self.log_stream.close()
            self.log_stream = None
            raise
        time.sleep(0.25)
        if self.process.poll() is not None:
            return_code = self.process.returncode
            self.log_stream.close()
            self.log_stream = None
            self.process = None
            raise RuntimeError(
                f"FFmpeg could not start (exit {return_code}); see {self.log_path}"
            )

    def stop(self) -> dict[str, Any] | None:
        if self.process is None:
            return None
        process = self.process
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        if self.log_stream is not None:
            self.log_stream.close()
        self.process = None
        self.log_stream = None
        size_bytes = self.output_path.stat().st_size if self.output_path.exists() else 0
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        result = {
            "path": str(self.output_path),
            "metadata_path": str(self.metadata_path),
            "log_path": str(self.log_path),
            "display": self.display,
            "video_size": self.video_size,
            "fps": self.fps,
            "codec": "H.264",
            "started_at": self.started_at,
            "finished_at": finished_at,
            "return_code": process.returncode,
            "size_bytes": size_bytes,
            "succeeded": size_bytes > 0,
            "command": self.command,
        }
        self.metadata_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result


@dataclass
class DemoVisualizationController:
    """
    Keep grounding invisible and make execution phases readable.

    Truth procedures also run *during* execution (precondition, effect and goal checks
    recompute predicates on the shared world, teleporting the robot through candidate
    poses and IK probes). Those probe states are working state, not execution, so the
    RViz publishers are paused for every check window and resumed — with one repaint of
    the true state — when platform motion starts or an action settles.
    """

    world: Any
    record_event: PipelineEventSink
    plan_preview_seconds: float
    action_pause_seconds: float
    seconds_per_update: float
    video_recorder: RvizVideoRecorder | None = None
    video_postroll_seconds: float = 2.0
    publisher_node: Any = None
    pacer: PlaybackPacer | None = None
    state_callbacks: list = field(default_factory=list)

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        self.record_event(event, data)
        if event == PipelineEvent.PLAN_GENERATED:
            self.record_event(
                PipelineEvent.PLAN_PREVIEW_STARTED,
                {"seconds": self.plan_preview_seconds},
            )
            self.publisher_node, publishers = start_publishers(self.world)
            self.state_callbacks = list(publishers)
            publish_initial_world(self.world)
            if self.video_recorder is not None:
                self.video_recorder.start()
                self.record_event(
                    "video_recording_started",
                    {
                        "path": str(self.video_recorder.output_path),
                        "fps": self.video_recorder.fps,
                        "video_size": self.video_recorder.video_size,
                    },
                )
            if self.seconds_per_update > 0:
                self.pacer = PlaybackPacer(
                    _world=self.world,
                    seconds_per_update=self.seconds_per_update,
                )
                self.state_callbacks.append(self.pacer)
            time.sleep(self.plan_preview_seconds)
            self.record_event(PipelineEvent.PLAN_PREVIEW_COMPLETED, {})
        elif event == PipelineEvent.ACTION_STARTED:
            time.sleep(self.action_pause_seconds)
            self._mute_probes()  # precondition checks follow
        elif event == PipelineEvent.PLATFORM_EXECUTION_STARTED:
            self._show_execution()  # real platform motion must be visible
        elif event == PipelineEvent.PLATFORM_RESULT:
            self._mute_probes()  # effect checks follow
        elif event == PipelineEvent.ACTION_COMPLETED:
            self._show_execution()
            time.sleep(self.action_pause_seconds)
        elif event == PipelineEvent.EXECUTION_COMPLETED:
            self._mute_probes()  # the independent goal re-check follows
        elif event in {PipelineEvent.TASK_SUCCEEDED, "task_failed"}:
            self._show_execution()
            if self.video_recorder is not None:
                time.sleep(self.video_postroll_seconds)
                self._stop_recording()

    def _mute_probes(self) -> None:
        for callback in self.state_callbacks:
            callback.pause()

    def _show_execution(self) -> None:
        for callback in self.state_callbacks:
            callback.resume()
        if self.state_callbacks:
            # Repaint the true world state: the probes above left RViz on
            # whatever candidate pose happened to be published last.
            self.world.notify_state_change()

    def _stop_recording(self) -> None:
        if self.video_recorder is None:
            return
        result = self.video_recorder.stop()
        if result is not None:
            event = (
                "video_recording_finished"
                if result["succeeded"]
                else "video_recording_failed"
            )
            self.record_event(event, result)

    def close(self) -> None:
        self._stop_recording()
        if self.pacer is not None:
            self.pacer.stop()
        if self.publisher_node is not None:
            self.publisher_node.destroy_node()


def start_publishers(world):
    """
    Bring up ROS and attach TF and marker publishers to the live world.

    Returns the node and the publisher callbacks so the demo controller can pause them
    while truth procedures probe the world.
    """
    import rclpy
    from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    rclpy.init()
    node = rclpy.create_node("resym_viz")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    publishers = [
        TFPublisher(_world=world, node=node),
        VizMarkerPublisher(_world=world, node=node),
    ]
    return node, publishers


def publish_initial_world(world) -> None:
    """
    Give a newly attached RViz enough state changes to resolve all frames.
    """
    for _ in range(6):
        world.notify_state_change()
        time.sleep(0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        nargs="?",
        choices=[scene.value for scene in Scene],
        default=Scene.APARTMENT.value,
    )
    parser.add_argument(
        "--plan-preview-seconds",
        type=float,
        default=float(os.environ.get("RESYM_VIZ_PLAN_PREVIEW_SECONDS", "6")),
    )
    parser.add_argument(
        "--action-pause-seconds",
        type=float,
        default=float(os.environ.get("RESYM_VIZ_ACTION_PAUSE_SECONDS", "1")),
    )
    parser.add_argument(
        "--seconds-per-update",
        type=float,
        default=float(os.environ.get("RESYM_VIZ_SECONDS_PER_UPDATE", "0.01")),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=os.environ.get("RESYM_VIZ_RECORD", "0").lower() in {"1", "true", "yes"},
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = Scene(args.scene)
    recorder = RunRecorder.create(
        "viz-open-drawer",
        metadata={
            "demo": "open_drawer_viz",
            "scene": scene.value,
            "backend": "coraplex",
        },
    )
    task_name = f"{scene.value}_open"

    print(f"=== live Coraplex demo: {scene.value} ===", flush=True)
    print(f"run: {recorder.directory}", flush=True)
    setup = load_scene(scene)
    video_recorder = (
        RvizVideoRecorder(recorder.directory / "D_visualization" / "rviz_execution.mp4")
        if args.record_video
        else None
    )
    visualization = DemoVisualizationController(
        world=setup.world,
        record_event=recorder.task_event_sink(task_name),
        plan_preview_seconds=max(0.0, args.plan_preview_seconds),
        action_pause_seconds=max(0.0, args.action_pause_seconds),
        seconds_per_update=max(0.0, args.seconds_per_update),
        video_recorder=video_recorder,
    )
    print(
        "Grounding is isolated from RViz; the scene appears when the plan is ready",
        flush=True,
    )
    if video_recorder is not None:
        print(f"RViz video: {video_recorder.output_path}", flush=True)

    universe = task_object_universe(setup.world, setup.robot)
    context = EvaluationContext(
        world=setup.world,
        robot=setup.robot,
        profile=setup.profile,
        grounding_catalog=default_drawer_grounding_catalog(),
        capability_feasibility=KinematicFeasibility(),
    )
    goal = (Literal("opened", (pddl_name(scene.goal_drawer_body),)),)
    recorder.record_world(universe, scene=scene.value)
    library = build_seed_library(default_drawer_grounding_catalog())
    recorder.record_library(library)

    realization = CoraplexSkillRealization.for_evaluation_context(
        context,
        event_sink=visualization,
    )
    try:
        result = solve_task(
            library=library,
            universe=universe,
            context=context,
            goal=goal,
            working_directory=Path(tempfile.mkdtemp(prefix="resym_viz_")),
            realization=realization,
            event_sink=visualization,
        )
    except KeyboardInterrupt:
        visualization.close()
        raise
    except Exception as error:
        visualization("task_failed", {"error": f"{type(error).__name__}: {error}"})
        recorder.record_task(task_name, goal, error=str(error))
        recorder.finish({"backend": "coraplex", "succeeded": False})
        visualization.close()
        raise

    recorder.record_task(task_name, goal, result=result)
    summary = {"backend": "coraplex", "succeeded": True}
    if video_recorder is not None:
        summary["rviz_video"] = str(video_recorder.output_path)
    recorder.finish(summary)
    print("plan:", flush=True)
    for action in result.plan:
        print(f"  ({action.operator} {' '.join(action.arguments)})", flush=True)
    print("execution and independent effect/goal checks succeeded", flush=True)
    if video_recorder is not None:
        print(f"video saved: {video_recorder.output_path}", flush=True)
    print("RViz remains live; press Ctrl-C to exit", flush=True)

    try:
        while True:
            setup.world.notify_state_change()
            time.sleep(1.0)
    except KeyboardInterrupt:
        visualization.close()


if __name__ == "__main__":
    main()
