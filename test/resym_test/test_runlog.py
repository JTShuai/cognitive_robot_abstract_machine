"""
Stage-split run logging (:mod:`resym.observability.runlog`).

These exercise only the recorder's on-disk layout and serialisation, so they run without
the CRAM stack (unlike the pipeline tests). Lightweight stand-ins mimic the shape of the
real domain objects the recorder is handed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from resym.observability.runlog import (
    STAGE_REPAIR,
    STAGE_SOLVE,
    STAGE_WORLD,
    RunRecorder,
    action_str,
    literal_str,
)


@dataclass(frozen=True)
class FakeLiteral:
    predicate: str
    arguments: tuple
    negated: bool = False


@dataclass(frozen=True)
class FakeSymbolType:
    python_type_ref: str


@dataclass
class FakeAction:
    operator: str
    arguments: tuple


class FakeViolation(Enum):
    PLATFORM_FAILED = "platform execution failed"


@dataclass
class FakeExecution:
    executed: list = field(default_factory=list)
    violation: object = None
    violated_action: object = None

    @property
    def succeeded(self) -> bool:
        return self.violation is None


@dataclass
class FakeResult:
    plan: list = field(default_factory=list)
    execution: object = None
    replanning_rounds: int = 0
    evaluation_count: int = 0
    grounding_seconds: float = 0.0
    planning_seconds: float = 0.0
    domain_text: str = ""
    problem_text: str = ""


def test_literal_and_action_rendering():
    assert literal_str(FakeLiteral("opened", ("d1",))) == "opened(d1)"
    assert literal_str(FakeLiteral("at", ("r", "d1"), negated=True)) == "not at(r, d1)"
    assert action_str(FakeAction("open-drawer", ("r", "h", "d1"))) == (
        "(open-drawer r h d1)"
    )


def test_create_seeds_run_json(tmp_path):
    recorder = RunRecorder.create("smoke", root=tmp_path, metadata={"demo": "x"})
    assert recorder.directory.parent == tmp_path
    assert recorder.directory.name.endswith("-smoke")
    run_json = json.loads((recorder.directory / "run.json").read_text())
    assert run_json["label"] == "smoke"
    assert run_json["metadata"] == {"demo": "x"}
    assert "started_at" in run_json


def test_runs_started_in_the_same_second_get_distinct_directories(
    tmp_path, monkeypatch
):
    from datetime import datetime

    from resym.observability import runlog

    fixed = datetime(2026, 8, 11, 12, 0, 0)
    monkeypatch.setattr(runlog, "_now", lambda: fixed)

    first = RunRecorder.create("e1", root=tmp_path)
    second = RunRecorder.create("e1", root=tmp_path)

    assert first.directory != second.directory
    assert second.directory.name.endswith("-e1-01")


def test_records_world_repair_and_solve_stages(tmp_path):
    recorder = RunRecorder.create("smoke", root=tmp_path)

    # Stage A
    class FakeUniverse:
        objects = {
            "d1": type(
                "O",
                (),
                {"name": "d1", "symbol_type": FakeSymbolType("drawer"), "body": "b"},
            )(),
        }

    recorder.record_world(FakeUniverse(), scene="kitchen")

    # Stage B
    recorder.transcript_path.write_text(
        '{"agent_name":"repair-agent","response":"submit"}\n',
        encoding="utf-8",
    )

    # Stage C: one failure, one success
    goal = (FakeLiteral("closed", ("d1",)),)
    recorder.record_task("1_initial_failure", goal, error="no plan")
    recorder.record_task(
        "2_retry",
        goal,
        result=FakeResult(
            plan=[FakeAction("close-drawer", ("r", "h", "d1"))],
            execution=FakeExecution(
                executed=[FakeAction("close-drawer", ("r", "h", "d1"))]
            ),
            evaluation_count=42,
            domain_text="(define (domain x))",
            problem_text="(define (problem p))",
        ),
    )
    recorder.finish({"drawer_closed": True})

    run_dir = recorder.directory
    # layout
    assert (run_dir / STAGE_WORLD / "universe.json").exists()
    assert (run_dir / STAGE_REPAIR / "llm_transcript.jsonl").exists()
    assert (run_dir / STAGE_SOLVE / "1_initial_failure" / "result.json").exists()
    assert (run_dir / STAGE_SOLVE / "2_retry" / "domain.pddl").exists()
    assert (run_dir / STAGE_SOLVE / "2_retry" / "problem.pddl").exists()

    # content
    universe = json.loads((run_dir / STAGE_WORLD / "universe.json").read_text())
    assert universe["object_count"] == 1
    assert universe["objects"][0]["type"] == "drawer"

    retry = json.loads((run_dir / STAGE_SOLVE / "2_retry" / "result.json").read_text())
    assert retry["plan"] == ["(close-drawer r h d1)"]
    assert retry["execution"]["succeeded"] is True
    assert retry["evaluation_count"] == 42

    failure = json.loads(
        (run_dir / STAGE_SOLVE / "1_initial_failure" / "result.json").read_text()
    )
    assert failure["error"] == "no plan"

    run_json = json.loads((run_dir / "run.json").read_text())
    assert run_json["scene"] == "kitchen"
    assert run_json["summary"] == {"drawer_closed": True}
    assert "ended_at" in run_json

    # events accumulate across the run
    events = [
        json.loads(line)
        for line in (run_dir / STAGE_SOLVE / "events.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events] == ["task", "task"]


def test_record_world_namespaces_multiple_worlds(tmp_path):
    recorder = RunRecorder.create("open", root=tmp_path)

    def universe(object_name):
        return type(
            "U",
            (),
            {
                "objects": {
                    object_name: type(
                        "O",
                        (),
                        {
                            "name": object_name,
                            "symbol_type": FakeSymbolType("drawer"),
                            "body": "b",
                        },
                    )()
                }
            },
        )()

    recorder.record_world(universe("d_apartment"), scene="apartment", name="apartment")
    recorder.record_world(universe("d_kitchen"), scene="kitchen", name="kitchen")

    apartment = json.loads(
        (recorder.directory / STAGE_WORLD / "apartment" / "universe.json").read_text()
    )
    kitchen = json.loads(
        (recorder.directory / STAGE_WORLD / "kitchen" / "universe.json").read_text()
    )
    assert apartment["objects"][0]["name"] == "d_apartment"
    assert kitchen["objects"][0]["name"] == "d_kitchen"
    # named worlds do not claim the single-scene run.json field
    assert "scene" not in json.loads((recorder.directory / "run.json").read_text())


def test_json_default_never_crashes_on_unknown(tmp_path):
    recorder = RunRecorder.create("x", root=tmp_path)
    # An arbitrary non-serialisable object must not raise, just stringify.
    recorder.json(STAGE_WORLD, "weird.json", {"obj": object()})
    payload = json.loads((recorder.directory / STAGE_WORLD / "weird.json").read_text())
    assert isinstance(payload["obj"], str)


def test_task_event_sink_writes_ordered_task_trace(tmp_path):
    recorder = RunRecorder.create("live", root=tmp_path)
    sink = recorder.task_event_sink("open")

    sink("task_started", {"goal": [{"display": "opened(d1)"}]})
    sink("plan_generated", {"actions": [{"display": "(open d1)"}]})

    path = recorder.directory / STAGE_SOLVE / "open" / "trace.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["seq"] for record in records] == [0, 1]
    assert [record["event"] for record in records] == [
        "task_started",
        "plan_generated",
    ]
    assert records[0]["goal"][0]["display"] == "opened(d1)"
    assert "at" in records[0]
