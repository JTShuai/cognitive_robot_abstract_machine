"""
End-to-end demo: one persistent library, two households, recomputed truth.

Usage:
    uv run python -m experiments.resym.open_drawer apartment
    uv run python -m experiments.resym.open_drawer kitchen
    uv run python -m experiments.resym.open_drawer both

Execution uses Coraplex action designators performed by the simulated robot.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from typing_extensions import Optional

from experiments.resym.articulation import (
    articulation_connection,
    is_articulated_object,
    joint_fraction,
)
from experiments.resym.scenes import Scene, load_scene
from experiments.resym.seed_library import (
    OPENED_FRACTION_THRESHOLD,
    build_seed_library,
)
from resym.core.symbols import Literal, SymbolLibrary
from resym.planning.pipeline import solve_task
from resym.observability.runlog import RunRecorder
from resym.platform.cram_objects import task_object_universe
from experiments.resym.capability_realizations import default_capability_initialization
from experiments.resym.grounding_initialization import default_drawer_grounding_catalog
from experiments.resym.drawer_kinematic_oracle import DrawerExperimentFeasibility
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import pddl_name


def run_scene(
    scene: Scene,
    library: SymbolLibrary,
    recorder: Optional[RunRecorder] = None,
) -> dict:
    print(f"\n=== scene: {scene.value} (backend: coraplex) ===")
    setup = load_scene(scene)
    universe = task_object_universe(setup.world, setup.robot)
    drawers = [
        item.name for item in universe.objects.values() if is_articulated_object(item)
    ]
    print(f"objects: {len(universe.objects)} ({len(drawers)} drawers)")

    context = EvaluationContext(
        world=setup.world,
        robot=setup.robot,
        grounding_catalog=default_drawer_grounding_catalog(),
        capability_feasibility=DrawerExperimentFeasibility(),
    )
    goal_drawer = pddl_name(scene.goal_drawer_body)
    goal = (Literal("opened", (goal_drawer,)),)
    print(f"goal: (opened {goal_drawer})")
    if recorder is not None:
        recorder.record_world(universe, scene=scene.value, name=scene.value)
    event_sink = (
        recorder.task_event_sink(f"{scene.value}_open")
        if recorder is not None
        else None
    )

    result = solve_task(
        library=library,
        universe=universe,
        context=context,
        goal=goal,
        working_directory=Path(tempfile.mkdtemp(prefix=f"resym_{scene.value}_")),
        realization=_build_realization(context, event_sink),
        event_sink=event_sink,
    )

    print(
        f"grounding: {result.evaluation_count} evaluations, {result.grounding_seconds:.1f}s"
    )
    print(
        f"planning:  {result.planning_seconds:.1f}s, replanning rounds: {result.replanning_rounds}"
    )
    print("plan:")
    for action in result.plan:
        print(f"  ({action.operator} {' '.join(action.arguments)})")

    connection = articulation_connection(universe[goal_drawer])
    joint_position = connection.position
    opened = joint_fraction(connection) >= OPENED_FRACTION_THRESHOLD
    print(
        f"executed: {len(result.execution.executed)} actions; "
        f"drawer joint now at {joint_position:.3f} m -> "
        f"{'OPEN' if opened else 'closed'}"
    )
    if recorder is not None:
        recorder.record_task(f"{scene.value}_open", goal, result=result)
    return {
        "scene": scene.value,
        "goal_drawer": goal_drawer,
        "drawer_opened": bool(opened),
        "drawer_joint_position": round(joint_position, 4),
        "evaluation_count": result.evaluation_count,
    }


def _build_realization(context: EvaluationContext, event_sink=None):
    """
    Build the Coraplex execution backend for the loaded CRAM world.
    """
    from resym.planning.execution.coraplex import (
        CoraplexSkillRealization,
    )

    return CoraplexSkillRealization.for_evaluation_context(
        context, default_capability_initialization(), event_sink=event_sink
    )


def main() -> None:
    choice = sys.argv[1] if len(sys.argv) > 1 else "both"
    recorder = RunRecorder.create(
        "open",
        metadata={"demo": "open_drawer", "choice": choice, "backend": "coraplex"},
    )
    print(f"run logs: {recorder.directory}")
    library = build_seed_library(default_drawer_grounding_catalog())
    recorder.record_library(library)
    print(
        "library initialized from reviewed grounding and capability catalogs "
        f"({len(library.predicates)} predicates, {len(library.operators)} operators)"
        " -- reused across scenes"
    )
    scenes = [Scene.APARTMENT, Scene.KITCHEN] if choice == "both" else [Scene(choice)]
    scene_summaries = [run_scene(scene, library, recorder=recorder) for scene in scenes]
    recorder.finish({"backend": "coraplex", "scenes": scene_summaries})
    print(f"\nrun logs written to {recorder.directory}")


if __name__ == "__main__":
    main()
