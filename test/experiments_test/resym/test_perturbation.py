"""
Execution perturbations through the real closed loop: a partially completed skill and an
external disturbance are caught by the immediate postcondition check (conservative
monitoring), pass-through leaves nominal runs untouched, and the monitoring-ablation
baseline blindly reports success over a perturbed world.

Container suite (shares the session-scoped Tracy fixtures).
"""

from __future__ import annotations

import pytest

from experiments.resym.scenes import Scene
from experiments.resym.seed_library import build_fixed_arm_library
from resym.repair.diagnosis import diagnose
from resym.core.model import Literal
from resym.planning.execution.perturbation import (
    ExternalPerturbationRealization,
    PartialSkillRealization,
    jitter_drawer_fraction,
    set_drawer_fraction,
)
from resym.platform.articulation import articulation_connection
from resym.platform.universe import pddl_name

GOAL_DRAWER = pddl_name(Scene.APARTMENT.goal_drawer_body)


def drawer_fraction(setup, universe, name: str) -> float:
    drawer = universe[name]
    connection = articulation_connection(drawer)
    limits = connection.dof.limits
    position = setup.world.state[connection.dof.id].position
    return (position - limits.lower.position) / (
        limits.upper.position - limits.lower.position
    )


def close_drawer(setup, universe, name: str) -> None:
    drawer = universe[name]
    connection = articulation_connection(drawer)
    limits = connection.dof.limits
    setup.world.state[connection.dof.id].position = limits.lower.position
    setup.world.notify_state_change()


@pytest.fixture()
def open_goal():
    return (Literal("opened", (GOAL_DRAWER,)),)


def test_partial_skill_is_caught_by_the_postcondition_check(
    tracy_setup, tracy_universe, tracy_context, tmp_path, open_goal
):
    """
    The first pull completes only 30% of its motion — below the opened threshold — so
    the declared effect did not materialize and the conservative loop refuses to paper
    over it.
    """
    close_drawer(tracy_setup, tracy_universe, GOAL_DRAWER)
    diagnosis = diagnose(
        build_fixed_arm_library(),
        tracy_universe,
        tracy_context,
        open_goal,
        tmp_path,
        realization=PartialSkillRealization(schedule={0: 0.3}),
    )
    assert not diagnosis.succeeded
    assert diagnosis.failure_class_value == "postcondition_failure"
    fraction = drawer_fraction(tracy_setup, tracy_universe, GOAL_DRAWER)
    assert fraction == pytest.approx(0.27, abs=0.02)  # 30% of the 0.9 pull


def test_unscheduled_actions_pass_through_unchanged(
    tracy_setup, tracy_universe, tracy_context, tmp_path, open_goal
):
    close_drawer(tracy_setup, tracy_universe, GOAL_DRAWER)
    diagnosis = diagnose(
        build_fixed_arm_library(),
        tracy_universe,
        tracy_context,
        open_goal,
        tmp_path,
        realization=PartialSkillRealization(schedule={}),
    )
    assert diagnosis.succeeded
    assert diagnosis.result.goal_check.satisfied


def test_external_disturbance_is_caught_immediately(
    tracy_setup, tracy_universe, tracy_context, tmp_path, open_goal
):
    """
    Someone pushes the drawer shut right after the robot opened it; the immediate
    postcondition check catches the mismatch at the action, not at some later consumer.
    """
    close_drawer(tracy_setup, tracy_universe, GOAL_DRAWER)
    diagnosis = diagnose(
        build_fixed_arm_library(),
        tracy_universe,
        tracy_context,
        open_goal,
        tmp_path,
        realization=ExternalPerturbationRealization(
            schedule={0: set_drawer_fraction(GOAL_DRAWER, 0.0)}
        ),
    )
    assert not diagnosis.succeeded
    assert diagnosis.failure_class_value == "postcondition_failure"


def test_monitoring_ablation_blindly_reports_success(
    tracy_setup, tracy_universe, tracy_context, tmp_path, open_goal
):
    """
    The deliberate baseline: with postcondition and goal checks off, the same
    disturbance goes unnoticed and the task 'succeeds' over a world where the drawer is
    in fact closed.
    """
    close_drawer(tracy_setup, tracy_universe, GOAL_DRAWER)
    diagnosis = diagnose(
        build_fixed_arm_library(),
        tracy_universe,
        tracy_context,
        open_goal,
        tmp_path,
        realization=ExternalPerturbationRealization(
            schedule={0: set_drawer_fraction(GOAL_DRAWER, 0.0)}
        ),
        check_postconditions=False,
        verify_goal=False,
    )
    assert diagnosis.succeeded  # claimed
    fraction = drawer_fraction(tracy_setup, tracy_universe, GOAL_DRAWER)
    assert fraction == pytest.approx(0.0, abs=0.01)  # actually closed


def test_jitter_disturbance_is_deterministic_and_clipped(
    tracy_setup, tracy_universe, tracy_context
):
    close_drawer(tracy_setup, tracy_universe, GOAL_DRAWER)
    jitter_drawer_fraction(GOAL_DRAWER, 0.05)(tracy_context, tracy_universe)
    assert drawer_fraction(tracy_setup, tracy_universe, GOAL_DRAWER) == pytest.approx(
        0.05, abs=1e-6
    )
    jitter_drawer_fraction(GOAL_DRAWER, -0.5)(tracy_context, tracy_universe)
    assert drawer_fraction(tracy_setup, tracy_universe, GOAL_DRAWER) == pytest.approx(
        0.0, abs=1e-6
    )  # clipped at the lower limit
