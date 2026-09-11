"""
End-to-end behaviour on the apartment world.

Includes the thesis test: a declared-truth plan (skipping navigation) is refused by
recomputed preconditions, while the computed-truth pipeline succeeds.
"""

from __future__ import annotations

import pytest

from resym.planning.execution.coraplex import CoraplexSkillRealization
from experiments.resym.scenes import Scene, load_scene
from experiments.resym.seed_library import build_seed_library
from resym.platform.evaluators import EvaluationContext
from resym.platform.articulation import articulation_connection
from resym.planning.execution.engine import ExecutionViolation, execute
from resym.core.model import Literal
from resym.planning.pddl import GroundAction
from resym.planning.pipeline import solve_task
from resym.platform.universe import (
    ObjectUniverse,
    pddl_name,
    set_joint_fraction,
)
from .conftest import drawer_universe_extractors
from resym.platform.kinematic import KinematicSkillRealization

GOAL_DRAWER = "cabinet10_drawer_top"
GOAL_HANDLE = "handle_cab10_t"


@pytest.fixture()
def restored_world(apartment_setup, apartment_context, apartment_universe):
    """
    Undo the state changes an execution test makes to the shared world.
    """
    yield
    world = apartment_setup.world
    from experiments.resym.scenes import Scene

    spawn_x, spawn_y = Scene.APARTMENT.robot_spawn_xy
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

    apartment_setup.drive_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            spawn_x, spawn_y, 0.0, reference_frame=world.root
        )
    )
    set_joint_fraction(articulation_connection(apartment_universe[GOAL_DRAWER]), 0.0)
    world.notify_state_change()


def test_declared_truth_plan_is_refused_by_recomputed_preconditions(
    library, apartment_universe, apartment_context, restored_world
):
    """
    A plan whose initial state was declared rather than computed claims the robot can
    open the drawer from its spawn pose.

    The per-action query must either return false or stop execution with a structured
    grounding failure when its computation cannot finish.
    """
    stale_plan = [
        GroundAction(
            "open-drawer",
            (
                pddl_name("base_footprint"),
                pddl_name(GOAL_HANDLE),
                pddl_name(GOAL_DRAWER),
            ),
        )
    ]
    report = execute(
        stale_plan,
        library,
        apartment_universe,
        apartment_context,
        realization=KinematicSkillRealization(),
    )
    assert not report.succeeded
    assert report.violation in (
        ExecutionViolation.PRECONDITION_FALSE,
        ExecutionViolation.PRECONDITION_GROUNDING_FAILED,
    )
    assert not report.executed


def test_computed_truth_pipeline_opens_the_drawer(
    library, apartment_universe, apartment_context, restored_world, tmp_path
):
    goal = (Literal("opened", (pddl_name(GOAL_DRAWER),)),)
    result = solve_task(
        library=library,
        universe=apartment_universe,
        context=apartment_context,
        goal=goal,
        working_directory=tmp_path,
        realization=KinematicSkillRealization(),
    )
    assert result.execution.succeeded
    assert [action.operator for action in result.plan] == ["navigate", "open-drawer"]

    drawer = apartment_universe[pddl_name(GOAL_DRAWER)]
    connection = articulation_connection(drawer)
    joint_position = apartment_context.world.state[connection.dof.id].position
    assert joint_position > 0.1


def test_coraplex_backend_performs_the_plan_with_cram_designators(tmp_path):
    """
    The same task, executed through CRAM: NavigateAction and OpenAction are performed by
    giskard's QP controller under simulated execution.

    A fresh world is loaded because the designators move the whole robot, not just the
    drive and the drawer joint.
    """
    setup = load_scene(Scene.APARTMENT)
    universe = ObjectUniverse.from_world(
        setup.world, setup.robot, drawer_universe_extractors()
    )
    context = EvaluationContext(
        world=setup.world,
        robot=setup.robot,
        profile=setup.profile,
    )
    events = []

    def event_sink(event, data):
        events.append((event, data))

    result = solve_task(
        library=build_seed_library(),
        universe=universe,
        context=context,
        goal=(Literal("opened", (pddl_name(GOAL_DRAWER),)),),
        working_directory=tmp_path,
        realization=CoraplexSkillRealization.for_evaluation_context(
            context, event_sink=event_sink
        ),
        event_sink=event_sink,
    )
    assert result.execution.succeeded
    assert [outcome.code for outcome in result.execution.platform_results] == [
        "CORAPLEX_EXECUTION_SUCCEEDED",
        "CORAPLEX_EXECUTION_SUCCEEDED",
    ]
    assert [
        data["action"] for event, data in events if event == "coraplex_action_selected"
    ] == ["NavigateAction", "OpenAction"]
    drawer = universe[pddl_name(GOAL_DRAWER)]
    connection = articulation_connection(drawer)
    assert setup.world.state[connection.dof.id].position > 0.1
