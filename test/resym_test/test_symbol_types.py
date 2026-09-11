"""
CRAM type references and their task-scoped PDDL projection.
"""

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

from resym.core.model import (
    Literal,
    PredicateSymbol,
    SymbolLibrary,
    SymbolType,
    is_symbol_subtype,
    resolve_symbol_type,
)
from resym.planning.pddl import write_domain, write_problem
from resym.planning.selection import Selection
from resym.platform.universe import GroundedObject, ObjectUniverse

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Agent,
    Drawer,
    Handle,
)

AGENT_TYPE = SymbolType.from_python_type(Agent)
ARTICULATED_PART_TYPE = SymbolType.from_python_type(HasMechanicalJoint)
DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


def test_cram_type_reference_roundtrip_uses_python_class():
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="available",
            parameter_types=(DRAWER_TYPE,),
            evaluator="test_available",
            fluent=False,
        )
    )

    payload = library.to_json()
    reloaded = SymbolLibrary.from_json(payload)

    assert "types" not in payload
    assert resolve_symbol_type(DRAWER_TYPE).__name__ == "Drawer"
    assert reloaded.predicates["available"].parameter_types == (DRAWER_TYPE,)


def test_symbol_type_accepts_any_cram_python_type_reference():
    body_type = SymbolType.from_python_type(Body)

    assert body_type.python_type_ref.endswith(":Body")
    assert resolve_symbol_type(body_type) is Body


def test_library_rejects_legacy_short_type_names():
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="available",
            parameter_types=(DRAWER_TYPE,),
            evaluator="test_available",
            fluent=False,
        )
    )
    payload = library.to_json()
    payload["predicates"][0]["parameter_types"][0]["python_type_ref"] = "drawer"

    with pytest.raises(ValueError, match="not an allowed CRAM Python type reference"):
        SymbolLibrary.from_json(payload)


def test_cram_python_inheritance_drives_subtype_queries():
    universe = ObjectUniverse()
    universe.add(
        GroundedObject(
            name="robot1",
            symbol_type=ROBOT_TYPE,
            body=Body(name=PrefixedName("robot1")),
        )
    )

    assert is_symbol_subtype(ROBOT_TYPE, AGENT_TYPE)
    assert [item.name for item in universe.of_type(AGENT_TYPE)] == ["robot1"]


def test_pddl_projection_encodes_cram_types_as_static_predicates():
    predicate = PredicateSymbol(
        name="opened",
        parameter_types=(ARTICULATED_PART_TYPE,),
        evaluator="opened",
        fluent=True,
    )
    selection = Selection(predicates={predicate.name: predicate})
    universe = ObjectUniverse()
    universe.add(
        GroundedObject(
            name="drawer1",
            symbol_type=DRAWER_TYPE,
            body=Body(name=PrefixedName("drawer1")),
        )
    )

    domain = write_domain(selection, "types-test")
    problem = write_problem(
        universe,
        init_atoms=set(),
        goal=(),
        domain_name="types-test",
        name="types-test-problem",
        selection=selection,
    )

    assert "(cram-type-has-mechanical-joint " in domain
    assert "(cram-type-has-mechanical-joint " in problem
    assert " drawer1)" in problem


def test_task_universe_keeps_only_selected_symbol_types():
    drawer = GroundedObject(
        name="drawer1",
        symbol_type=DRAWER_TYPE,
        body=Body(name=PrefixedName("drawer1")),
    )
    unrelated_handle = GroundedObject(
        name="handle1",
        symbol_type=HANDLE_TYPE,
        body=Body(name=PrefixedName("handle1")),
    )
    universe = ObjectUniverse(
        objects={
            drawer.name: drawer,
            unrelated_handle.name: unrelated_handle,
        }
    )
    predicate = PredicateSymbol(
        name="opened",
        parameter_types=(DRAWER_TYPE,),
        evaluator="opened",
        fluent=True,
    )
    selection = Selection(predicates={predicate.name: predicate})

    task_universe = universe.for_task(selection, (Literal("opened", ("drawer1",)),))

    assert list(task_universe.objects) == ["drawer1"]
