"""
The fixed-arm embodiment end to end: Tracy (dual UR10e arms with Robotiq grippers on
their table) placed beside the apartment's goal drawer opens and closes it without any
navigation, and a goal whose closure needs base repositioning is refused as an
unsupported capability before anything is grounded.

Real scene load, real inverse kinematics, real Fast Downward — container suite.
"""

from __future__ import annotations

import pytest

from experiments.resym.scenes import Scene
from experiments.resym.seed_library import build_seed_library
from resym.platform.capabilities import ARTICULATION_CAPABILITY_UID
from resym.platform.kinematic import KinematicFeasibility
from resym.platform.embodiment import UnsupportedCapabilityError
from resym.core.model import Literal, is_symbol_subtype
from resym.platform.capabilities import NAVIGATION_CAPABILITY_UID
from resym.platform.kinematic import KinematicSkillRealization
from resym.planning.pipeline import solve_task
from resym.platform.articulation import is_articulated_object
from resym.platform.universe import ObjectUniverse, pddl_name

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


GOAL_DRAWER = pddl_name(Scene.APARTMENT.goal_drawer_body)


def robot_name(universe: ObjectUniverse) -> str:
    (name,) = [
        item.name
        for item in universe.objects.values()
        if is_symbol_subtype(item.symbol_type, ROBOT_TYPE)
    ]
    return name


def test_the_fixed_arm_world_contains_the_annotated_drawers(tracy_universe):
    drawers = [
        item for item in tracy_universe.objects.values() if is_articulated_object(item)
    ]
    assert GOAL_DRAWER in {o.name for o in drawers}


def test_the_arm_beside_the_furniture_reaches_the_goal_drawer(
    tracy_universe, tracy_context
):
    """
    The reachability smoke of the table placement: from beside the furniture, `ready-to-
    open` must be provably TRUE — not merely unknown — for the goal drawer.
    """
    verdict = KinematicFeasibility().feasible(
        ARTICULATION_CAPABILITY_UID,
        (
            tracy_universe[robot_name(tracy_universe)],
            tracy_universe[GOAL_DRAWER],
        ),
        tracy_context,
    )
    assert verdict is True


def test_open_then_close_without_navigation(
    tracy_setup, tracy_universe, tracy_context, tmp_path, fixed_arm_library
):
    """
    The full fixed-arm loop: one-action plans (no navigate exists), monitored dispatch,
    postconditions, and the independent goal check — in both manipulation directions.
    """
    library = fixed_arm_library
    opened = solve_task(
        library=library,
        universe=tracy_universe,
        context=tracy_context,
        goal=(Literal("opened", (GOAL_DRAWER,)),),
        working_directory=tmp_path / "open",
        realization=KinematicSkillRealization(),
    )
    assert opened.goal_check is not None and opened.goal_check.satisfied
    assert [action.operator for action in opened.plan] == ["open-drawer"]

    closed = solve_task(
        library=library,
        universe=tracy_universe,
        context=tracy_context,
        goal=(Literal("closed", (GOAL_DRAWER,)),),
        working_directory=tmp_path / "close",
        realization=KinematicSkillRealization(),
    )
    assert closed.goal_check is not None and closed.goal_check.satisfied
    assert [action.operator for action in closed.plan] == ["close-drawer"]


def test_mobile_library_on_the_fixed_arm_is_unsupported_not_a_library_gap(
    tracy_universe, tracy_context, tmp_path, grounding_catalog
):
    """
    The mobile seed library's closure needs `openable` (base-pose sampling) and the
    navigation skill; on the fixed arm that is an UNSUPPORTED_CAPABILITY refusal raised
    before grounding — the repair loop must never see it as a missing model.
    """
    with pytest.raises(UnsupportedCapabilityError) as error:
        solve_task(
            library=build_seed_library(grounding_catalog),
            universe=tracy_universe,
            context=tracy_context,
            goal=(Literal("opened", (GOAL_DRAWER,)),),
            working_directory=tmp_path,
        )
    missing = " ".join(error.value.missing)
    assert "openable" in missing
    assert NAVIGATION_CAPABILITY_UID in missing
    assert error.value.goal == (Literal("opened", (GOAL_DRAWER,)),)


def test_unsupported_capability_certificate_from_the_refusal(
    tracy_universe, tracy_context, tmp_path, grounding_catalog
):
    from resym.repair.certificate import (
        FailureClass,
        certify_unsupported_capability,
    )

    goal = (Literal("opened", (GOAL_DRAWER,)),)
    with pytest.raises(UnsupportedCapabilityError) as error:
        solve_task(
            library=build_seed_library(grounding_catalog),
            universe=tracy_universe,
            context=tracy_context,
            goal=goal,
            working_directory=tmp_path,
        )
    certificate = certify_unsupported_capability(
        goal,
        error.value.selection,
        missing="; ".join(error.value.missing),
    )
    assert certificate.failure_class is FailureClass.UNSUPPORTED_CAPABILITY
    assert "openable" in certificate.render()
