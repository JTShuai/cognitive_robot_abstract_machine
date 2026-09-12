"""
A task whose model needs what the robot cannot do is refused before grounding.
"""

import pytest

from resym.core.symbols import Literal
from resym.core.symbol_types import SymbolType
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import GroundedObject, ObjectUniverse
from resym.planning.pipeline import UnsupportedCapabilityError, solve_task
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.world_description.world_entity import Body

from .dataset.capability_model import (
    ARTICULATION_CAPABILITY_UID,
    INTERACTION_NAVIGATION_CAPABILITY_UID,
)

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)
DRAWER_TYPE = SymbolType.from_python_type(Drawer)


def test_feasibility_of_an_unavailable_capability_is_refused_as_a_grounding_gap(
    library, grounding_catalog, tmp_path
):
    """
    ``openable`` is decided by the feasibility factory of the navigation capability; on
    a robot without that capability it is a missing grounding factory, reported together
    with the missing capability, before any grounding runs.
    """
    universe = ObjectUniverse()
    universe.add(GroundedObject("rob", ROBOT_TYPE, Body(name=PrefixedName("rob"))))
    universe.add(GroundedObject("drawer1", DRAWER_TYPE, Body(name=PrefixedName("d"))))
    context = EvaluationContext(
        world=None, robot=None, grounding_catalog=grounding_catalog
    )
    goal = (Literal("opened", ("drawer1",)),)

    with pytest.raises(UnsupportedCapabilityError) as refusal:
        solve_task(
            library=library,
            universe=universe,
            context=context,
            goal=goal,
            working_directory=tmp_path,
            available_capabilities=frozenset({ARTICULATION_CAPABILITY_UID}),
        )

    missing = " ".join(refusal.value.missing)
    assert "openable" in missing
    assert INTERACTION_NAVIGATION_CAPABILITY_UID in missing
