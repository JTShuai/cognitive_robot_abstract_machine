"""
A drawer the fixed arm cannot reach within the inverse-kinematics budget is not
reachable, the verdict CRAM's own ``reachable`` predicate gives, rather than a grounding
failure that stops the task.

Real scene load; container suite.
"""

from __future__ import annotations

from experiments.resym.capability_contracts import ARTICULATION_CAPABILITY_UID
from experiments.resym.drawer_kinematic_oracle import DrawerExperimentFeasibility
from resym.core.symbol_types import SymbolType, is_symbol_subtype
from semantic_digital_twin.robots.robot_parts import AbstractRobot

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)

UNREACHED_DRAWER = "cabinet11_drawer_middle"
"""
A drawer beside the goal cabinet whose handle the solver does not reach in budget.
"""


def robot_of(universe):
    (item,) = [
        item
        for item in universe.objects.values()
        if is_symbol_subtype(item.symbol_type, ROBOT_TYPE)
    ]
    return item


def test_budget_exhaustion_is_not_reachable(tracy_universe, tracy_context):
    verdict = DrawerExperimentFeasibility().feasible(
        ARTICULATION_CAPABILITY_UID,
        (robot_of(tracy_universe), tracy_universe[UNREACHED_DRAWER]),
        tracy_context,
    )
    assert verdict is False
