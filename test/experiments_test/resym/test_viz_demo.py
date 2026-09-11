"""
Presentation control for the RViz/live-viewer demo.
"""

import json
from pathlib import Path

from experiments.resym import open_drawer_viz


class _Node:
    def __init__(self):
        self.destroyed = False

    def destroy_node(self):
        self.destroyed = True


class _VideoRecorder:
    output_path = Path("runs/demo/D_visualization/rviz_execution.mp4")
    fps = 30
    video_size = "1920x1080"

    def __init__(self, calls):
        self.calls = calls

    def start(self):
        self.calls.append("video-start")

    def stop(self):
        self.calls.append("video-stop")
        return {
            "path": str(self.output_path),
            "succeeded": True,
            "size_bytes": 123,
        }


def test_rviz_publishers_start_only_after_plan_is_generated(monkeypatch):
    events = []
    calls = []
    node = _Node()
    world = object()
    monkeypatch.setattr(
        open_drawer_viz,
        "start_publishers",
        lambda selected_world: calls.append(("start", selected_world)) or (node, []),
    )
    monkeypatch.setattr(
        open_drawer_viz,
        "publish_initial_world",
        lambda selected_world: calls.append(("publish", selected_world)),
    )
    controller = open_drawer_viz.DemoVisualizationController(
        world=world,
        record_event=lambda event, data: events.append((event, data)),
        plan_preview_seconds=0,
        action_pause_seconds=0,
        seconds_per_update=0,
    )

    controller("task_started", {"goal": []})
    controller("grounding_completed", {"evaluations": 1})
    assert calls == []

    controller("plan_generated", {"actions": []})
    assert calls == [("start", world), ("publish", world)]
    assert [event for event, _ in events] == [
        "task_started",
        "grounding_completed",
        "plan_generated",
        "plan_preview_started",
        "plan_preview_completed",
    ]

    controller.close()
    assert node.destroyed


def test_recording_starts_after_scene_publish_and_stops_after_task(monkeypatch):
    events = []
    calls = []
    node = _Node()
    world = object()
    monkeypatch.setattr(
        open_drawer_viz,
        "start_publishers",
        lambda selected_world: calls.append("publisher-start") or (node, []),
    )
    monkeypatch.setattr(
        open_drawer_viz,
        "publish_initial_world",
        lambda selected_world: calls.append("scene-published"),
    )
    controller = open_drawer_viz.DemoVisualizationController(
        world=world,
        record_event=lambda event, data: events.append((event, data)),
        plan_preview_seconds=0,
        action_pause_seconds=0,
        seconds_per_update=0,
        video_recorder=_VideoRecorder(calls),
        video_postroll_seconds=0,
    )

    controller("plan_generated", {"actions": []})
    controller("task_succeeded", {})

    assert calls == [
        "publisher-start",
        "scene-published",
        "video-start",
        "video-stop",
    ]
    assert [event for event, _ in events] == [
        "plan_generated",
        "plan_preview_started",
        "video_recording_started",
        "plan_preview_completed",
        "task_succeeded",
        "video_recording_finished",
    ]


class _Publisher:
    def __init__(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class _World:
    def __init__(self):
        self.repaints = 0

    def notify_state_change(self):
        self.repaints += 1


def test_publishers_pause_during_check_windows(monkeypatch):
    publisher = _Publisher()
    world = _World()
    monkeypatch.setattr(
        open_drawer_viz, "start_publishers", lambda w: (_Node(), [publisher])
    )
    monkeypatch.setattr(open_drawer_viz, "publish_initial_world", lambda w: None)
    controller = open_drawer_viz.DemoVisualizationController(
        world=world,
        record_event=lambda event, data: None,
        plan_preview_seconds=0,
        action_pause_seconds=0,
        seconds_per_update=0,
    )

    controller("plan_generated", {"actions": []})
    assert not publisher.paused

    controller("action_started", {"action_index": 0})
    assert publisher.paused  # precondition probes must not reach RViz

    controller("platform_execution_started", {"action_index": 0})
    assert not publisher.paused  # real motion is visible
    assert world.repaints == 1  # true state repainted after the probes

    controller("platform_result", {"action_index": 0})
    assert publisher.paused  # effect re-checks are probes again

    controller("action_completed", {"action_index": 0})
    assert not publisher.paused
    assert world.repaints == 2

    controller("execution_completed", {})
    assert publisher.paused  # independent goal re-check

    controller("task_succeeded", {})
    assert not publisher.paused
    assert world.repaints == 3


def test_ffmpeg_command_records_xvfb_as_h264(tmp_path):
    output = tmp_path / "rviz_execution.mp4"
    recorder = open_drawer_viz.RvizVideoRecorder(output)

    assert recorder.command[-1] == str(output)
    assert recorder.command[recorder.command.index("-i") + 1] == ":99.0"
    assert recorder.command[recorder.command.index("-video_size") + 1] == "1920x1080"
    assert recorder.command[recorder.command.index("-framerate") + 1] == "30"
    assert recorder.command[recorder.command.index("-c:v") + 1] == "libx264"


def test_video_recorder_finalizes_mp4_and_metadata(tmp_path, monkeypatch):
    output = tmp_path / "rviz_execution.mp4"

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, selected_signal):
            assert selected_signal == open_drawer_viz.signal.SIGINT

        def wait(self, timeout):
            output.write_bytes(b"mp4-data")
            self.returncode = 0
            return 0

        def terminate(self):
            raise AssertionError("graceful FFmpeg shutdown should not time out")

    process = Process()
    monkeypatch.setattr(
        open_drawer_viz.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(open_drawer_viz.time, "sleep", lambda seconds: None)
    recorder = open_drawer_viz.RvizVideoRecorder(output)

    recorder.start()
    result = recorder.stop()

    assert result is not None and result["succeeded"]
    assert result["size_bytes"] == len(b"mp4-data")
    metadata = json.loads(recorder.metadata_path.read_text(encoding="utf-8"))
    assert metadata["path"] == str(output)
    assert metadata["codec"] == "H.264"
