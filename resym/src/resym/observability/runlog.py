"""
Per-run structured logs, split by pipeline stage.

Each run gets its own timestamped directory under a runs root, with one
sub-directory per stage:

* ``A_world``       -- Stage A: the kinematic world / object universe a run starts from.
* ``B_repair``      -- Stage B: planning-model agent and baseline transcripts.
* ``C_solve``       -- Stage C: the per-task solve loop (goal, projected PDDL, plan, execution).
* ``D_experiment``  -- E1/E2 episode and decision records.

Recording is opt-in and injected exactly like
:class:`~resym.llm.transcript.TranscriptRecorder`: a demo (or any
caller) creates a :class:`RunRecorder` and hands stage data to it. Nothing in the
core pipeline depends on it, so leaving it out changes no behaviour. The on-disk
layout is stable so :mod:`resym.observability.viewer` can browse it.

Serialisation never raises on an unfamiliar object: domain types get explicit,
readable summaries here, and the JSON encoder falls back to ``str`` for anything
else rather than crashing a run over a log line.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel
from typing_extensions import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from resym.core.model import Literal, SymbolLibrary
    from resym.planning.pddl import GroundAction
    from resym.planning.pipeline import TaskResult
    from resym.planning.execution.engine import ExecutionReport
    from resym.platform.universe import ObjectUniverse

STAGE_WORLD = "A_world"
STAGE_REPAIR = "B_repair"
STAGE_SOLVE = "C_solve"
STAGE_EXPERIMENT = "D_experiment"

DEFAULT_RUNS_DIR = "runs"
"""
Runs root when neither an explicit root nor ``RESYM_RUNS_DIR`` is given.
"""


def _now() -> datetime:
    return datetime.now()


def _json_default(value: Any) -> Any:
    """
    Last-resort encoder: keep a log line rather than crash the run.
    """
    if isinstance(value, set):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Enum):
        return value.value
    return str(value)


def literal_str(literal: Literal) -> str:
    """
    A goal/atom literal as ``pred(a, b)`` or ``not pred(a, b)``.
    """
    prefix = "not " if literal.negated else ""
    arguments = ", ".join(literal.arguments)
    return f"{prefix}{literal.predicate}({arguments})"


def action_str(action: GroundAction) -> str:
    """
    A ground action as ``(operator arg1 arg2)``.
    """
    return f"({action.operator} {' '.join(action.arguments)})".replace("  ", " ")


def summarize_universe(universe: ObjectUniverse) -> dict:
    """
    The typed object domain a run quantifies over.
    """
    objects = []
    for grounded in universe.objects.values():
        objects.append(
            {
                "name": grounded.name,
                "type": grounded.symbol_type.python_type_ref,
                "body": str(grounded.body),
            }
        )
    objects.sort(key=lambda entry: (entry["type"], entry["name"]))
    return {"object_count": len(objects), "objects": objects}


def summarize_execution(report: Optional[ExecutionReport]) -> Optional[dict]:
    """
    An :class:`ExecutionReport` reduced to a JSON-friendly verdict.
    """
    if report is None:
        return None
    violation = report.violation
    violated = report.violated_action
    return {
        "succeeded": report.succeeded,
        "executed": [action_str(action) for action in report.executed],
        "violation": violation.value if violation is not None else None,
        "violated_action": action_str(violated) if violated is not None else None,
    }


def summarize_task_result(result: TaskResult) -> dict:
    """A :class:`~resym.planning.pipeline.TaskResult` as one record."""
    return {
        "plan": [action_str(action) for action in result.plan],
        "execution": summarize_execution(result.execution),
        "replanning_rounds": result.replanning_rounds,
        "evaluation_count": result.evaluation_count,
        "grounding_seconds": round(result.grounding_seconds, 4),
        "planning_seconds": round(result.planning_seconds, 4),
    }


@dataclass
class RunRecorder:
    """
    Writes one run's artifacts under ``<root>/<timestamp>-<label>/``.
    """

    directory: Path
    """
    The run's own directory; everything this recorder writes lives here.
    """

    label: str
    """
    Short run label (usually the demo name), part of the directory name.
    """

    meta: dict = field(default_factory=dict)
    """
    Accumulated ``run.json`` contents (started/ended, label, summary).
    """

    @classmethod
    def create(
        cls,
        label: str,
        root: Optional[Path] = None,
        metadata: Optional[dict] = None,
    ) -> "RunRecorder":
        """
        Make a fresh timestamped run directory and seed ``run.json``.
        """
        started = _now()
        if root is None:
            root = Path(os.environ.get("RESYM_RUNS_DIR", DEFAULT_RUNS_DIR))
        root = Path(root)
        stem = f"{started.strftime('%Y%m%d-%H%M%S')}-{label}"
        directory = root / stem
        suffix = 1
        # Directory timestamps have second precision. Two short runs may
        # therefore collide; never merge their JSONL records silently.
        while True:
            try:
                directory.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                directory = root / f"{stem}-{suffix:02d}"
                suffix += 1
        recorder = cls(
            directory=directory,
            label=label,
            meta={
                "label": label,
                "started_at": started.isoformat(timespec="seconds"),
                "metadata": metadata or {},
            },
        )
        recorder._flush_meta()
        return recorder

    # -- generic sinks --------------------------------------------------------

    def stage_dir(self, stage: str) -> Path:
        """
        The (lazily created) directory for one stage.
        """
        path = self.directory / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def json(self, stage: str, relative_path: str, data: Any) -> Path:
        """
        Write ``data`` as pretty JSON under a stage directory.
        """
        path = self.stage_dir(stage) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return path

    def artifact(self, stage: str, relative_path: str, text: str) -> Path:
        """
        Write a raw text artifact (PDDL, plan, ...) under a stage directory.
        """
        path = self.stage_dir(stage) / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def event(self, stage: str, name: str, data: Optional[dict] = None) -> None:
        """
        Append one timestamped workflow event to ``<stage>/events.jsonl``.
        """
        entry = {
            "at": _now().isoformat(timespec="seconds"),
            "event": name,
            **(data or {}),
        }
        path = self.stage_dir(stage) / "events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n"
            )

    # -- stage A --------------------------------------------------------------

    def record_world(
        self, universe: Any, scene: Optional[str] = None, name: Optional[str] = None
    ) -> None:
        """
        Stage A: the object universe (and scene name) a run starts from.

        ``name`` namespaces the artifact into ``A_world/<name>/`` so a run that loads
        several worlds (e.g. open_drawer's ``both``) keeps them apart.
        """
        self.event(STAGE_WORLD, "world_loaded", {"scene": scene, "name": name})
        relative = "universe.json" if name is None else f"{name}/universe.json"
        self.json(STAGE_WORLD, relative, summarize_universe(universe))
        if scene is not None and name is None:
            self.meta.setdefault("scene", scene)
            self._flush_meta()

    # -- stage B --------------------------------------------------------------

    @property
    def transcript_path(self) -> Path:
        """
        Where the LLM transcript for this run should be written.
        """
        return self.stage_dir(STAGE_REPAIR) / "llm_transcript.jsonl"

    def record_library(
        self, library: SymbolLibrary | dict, name: str = "start"
    ) -> Path:
        """
        A symbol-library snapshot (e.g. the library a run starts from), written at the
        run root so the viewer's Results section renders it.

        ``library`` may be a :class:`SymbolLibrary` (serialized via its ``to_json``) or
        an already-serialized dict.
        """
        data = library if isinstance(library, dict) else library.to_json()
        path = self.directory / f"library_{name}.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        return path

    # -- stage C --------------------------------------------------------------

    def task_event_sink(self, name: str) -> Callable[[str, dict], None]:
        """
        Return a callback that persists one task's live pipeline trace.

        The callback shape matches ``solve_task(event_sink=...)``. Sequence numbers make
        polling deterministic even when several events share a timestamp.
        """
        path = self.stage_dir(STAGE_SOLVE) / name / "trace.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        sequence = 0

        def record(event: str, data: dict) -> None:
            nonlocal sequence
            entry = {
                "at": _now().isoformat(timespec="milliseconds"),
                "seq": sequence,
                "event": event,
                **data,
            }
            sequence += 1
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n"
                )

        return record

    def record_task(
        self,
        name: str,
        goal: tuple[Literal, ...],
        result: Optional[TaskResult] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Stage C: one task solve -- goal, projected PDDL, plan, execution.
        """
        goal_literals = [literal_str(literal) for literal in goal]
        self.event(
            STAGE_SOLVE,
            "task",
            {"name": name, "goal": goal_literals, "solved": result is not None},
        )
        self.json(
            STAGE_SOLVE,
            f"{name}/goal.json",
            {"goal": goal_literals},
        )
        if result is None:
            self.json(STAGE_SOLVE, f"{name}/result.json", {"error": error})
            return
        self.json(STAGE_SOLVE, f"{name}/result.json", summarize_task_result(result))
        domain = result.domain_text
        problem = result.problem_text
        if domain:
            self.artifact(STAGE_SOLVE, f"{name}/domain.pddl", domain)
        if problem:
            self.artifact(STAGE_SOLVE, f"{name}/problem.pddl", problem)

    # -- stage D --------------------------------------------------------------

    def record_episode(self, record: dict) -> None:
        """
        Stage D: one experiment episode/decision (the §11 record fields), appended to
        ``D_experiment/episodes.jsonl`` — every failure, budget exhaustion, and
        unsupported episode is kept, none is filtered.
        """
        path = self.stage_dir(STAGE_EXPERIMENT) / "episodes.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
            )

    # -- lifecycle ------------------------------------------------------------

    def finish(self, summary: Optional[dict] = None) -> None:
        """
        Stamp ``run.json`` with an end time and an optional summary.
        """
        self.meta["ended_at"] = _now().isoformat(timespec="seconds")
        if summary:
            self.meta["summary"] = summary
        self._flush_meta()

    def _flush_meta(self) -> None:
        (self.directory / "run.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
