"""
Binary predicate grounding and its separate failure channel.
"""

from __future__ import annotations

import pytest

from resym.core.grounding import GroundingFailure, GroundingFailureCode
from resym.core.model import Literal, PredicateSymbol, SymbolType
from resym.planning.grounding import ground
from resym.planning.selection import Selection
from resym.platform.embodiment import EmbodimentProfile
from resym.platform.evaluators import EVALUATORS, EvaluationContext
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.world_entity import Body

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


@pytest.fixture()
def miniature_universe() -> ObjectUniverse:
    universe = ObjectUniverse()
    universe.add(
        GroundedObject(
            name="robot",
            symbol_type=ROBOT_TYPE,
            body=Body(name=PrefixedName("robot")),
        )
    )
    return universe


@pytest.fixture()
def context() -> EvaluationContext:
    profile = EmbodimentProfile(
        name="stub",
        evaluators=frozenset({"is_ready", "missing_state"}),
        capabilities=frozenset(),
    )
    return EvaluationContext(world=None, robot=None, profile=profile)


def predicate(name: str, evaluator: str) -> PredicateSymbol:
    return PredicateSymbol(
        name=name,
        parameter_types=(ROBOT_TYPE,),
        evaluator=evaluator,
        fluent=True,
    )


def test_grounding_assigns_every_atom_a_boolean_value(
    miniature_universe: ObjectUniverse,
    context: EvaluationContext,
) -> None:
    EVALUATORS["is_ready"] = lambda query_context, universe, arguments: True
    try:
        selection = Selection(
            predicates={"ready": predicate("ready", "is_ready")}, operators={}
        )
        result = ground(selection, miniature_universe, context)
    finally:
        del EVALUATORS["is_ready"]

    atom = Literal("ready", ("robot",))
    assert result.value_of(atom) is True
    assert result.true_atoms == {atom}
    assert result.false_atoms == set()
    assert not hasattr(result, "unknown_atoms")


def test_missing_world_knowledge_stops_grounding_with_a_diagnostic(
    miniature_universe: ObjectUniverse,
    context: EvaluationContext,
) -> None:
    def fail(query_context, universe, arguments):
        raise GroundingFailure(
            GroundingFailureCode.MISSING_WORLD_KNOWLEDGE,
            "joint annotation is absent",
        )

    EVALUATORS["missing_state"] = fail
    try:
        selection = Selection(
            predicates={"ready": predicate("ready", "missing_state")}, operators={}
        )
        with pytest.raises(GroundingFailure) as error:
            ground(selection, miniature_universe, context)
    finally:
        del EVALUATORS["missing_state"]

    assert error.value.code is GroundingFailureCode.MISSING_WORLD_KNOWLEDGE
    assert error.value.atom == Literal("ready", ("robot",))
    assert error.value.detail == "joint annotation is absent"


def test_predicate_symbol_has_one_reviewed_query_reference() -> None:
    symbol = predicate("ready", "is_ready")

    assert symbol.implementation.evaluator_key == "is_ready"
    assert not hasattr(symbol, "program")


def test_non_boolean_query_result_is_a_grounding_failure(
    miniature_universe: ObjectUniverse,
    context: EvaluationContext,
) -> None:
    EVALUATORS["invalid_result"] = lambda *arguments: None
    try:
        selection = Selection(
            predicates={"ready": predicate("ready", "invalid_result")}, operators={}
        )
        with pytest.raises(GroundingFailure) as error:
            ground(selection, miniature_universe, context)
    finally:
        del EVALUATORS["invalid_result"]

    assert error.value.code is GroundingFailureCode.QUERY_ERROR
    assert "returned NoneType" in error.value.detail
