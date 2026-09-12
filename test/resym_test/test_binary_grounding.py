"""
Binary predicate grounding and its separate failure channel.
"""

from __future__ import annotations

import pytest

from resym.core.grounding import (
    GroundingFactoryOrigin,
    GroundingFactoryRole,
    GroundingFactorySpec,
    GroundingFailure,
    GroundingFailureCode,
    PredicateGroundingPlan,
)
from resym.core.model import Literal, PredicateSymbol, SymbolType, TruthProcedureRef
from resym.planning.grounding import ground
from resym.planning.selection import Selection
from resym.platform.embodiment import EmbodimentProfile
from resym.platform.grounding_catalog import GroundingFactoryCatalog
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.world_entity import Body

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)

STUB_CHECKSUM = "stub-checksum"


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


def factory_uid(name: str) -> str:
    return f"test:grounding/{name}"


def catalog_with(**procedures) -> GroundingFactoryCatalog:
    """
    An in-memory approved catalog holding one stub factory per procedure.
    """
    specifications = {}
    callables = {}
    for name, procedure in procedures.items():
        uid = factory_uid(name)
        specifications[uid] = GroundingFactorySpec(
            uid=uid,
            semantic_name=name,
            implementation_ref=f"test_stub:{name}",
            implementation_checksum=STUB_CHECKSUM,
            roles=(GroundingFactoryRole("object", ROBOT_TYPE),),
            origin=GroundingFactoryOrigin.PLATFORM,
            reviewed_by="test-reviewer",
            approved_at="test-time",
            active_revision_id="r-test",
        )
        callables[uid] = procedure
    return GroundingFactoryCatalog(specifications, callables)


def context_over(catalog: GroundingFactoryCatalog) -> EvaluationContext:
    profile = EmbodimentProfile(name="stub", capabilities=frozenset())
    return EvaluationContext(
        world=None, robot=None, profile=profile, grounding_catalog=catalog
    )


def predicate(name: str, factory: str) -> PredicateSymbol:
    return PredicateSymbol(
        name=name,
        parameter_types=(ROBOT_TYPE,),
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid=factory_uid(factory),
            approved_factory_checksum=STUB_CHECKSUM,
        ),
    )


def test_grounding_assigns_every_atom_a_boolean_value(
    miniature_universe: ObjectUniverse,
) -> None:
    context = context_over(
        catalog_with(is_ready=lambda context, universe, arguments, parameters: True)
    )
    selection = Selection(
        predicates={"ready": predicate("ready", "is_ready")}, operators={}
    )
    result = ground(selection, miniature_universe, context)

    atom = Literal("ready", ("robot",))
    assert result.value_of(atom) is True
    assert result.true_atoms == {atom}
    assert result.false_atoms == set()
    assert not hasattr(result, "unknown_atoms")


def test_missing_world_knowledge_stops_grounding_with_a_diagnostic(
    miniature_universe: ObjectUniverse,
) -> None:
    def fail(context, universe, arguments, parameters):
        raise GroundingFailure(
            GroundingFailureCode.MISSING_WORLD_KNOWLEDGE,
            "joint annotation is absent",
        )

    context = context_over(catalog_with(missing_state=fail))
    selection = Selection(
        predicates={"ready": predicate("ready", "missing_state")}, operators={}
    )
    with pytest.raises(GroundingFailure) as error:
        ground(selection, miniature_universe, context)

    assert error.value.code is GroundingFailureCode.MISSING_WORLD_KNOWLEDGE
    assert error.value.atom == Literal("ready", ("robot",))
    assert error.value.detail == "joint annotation is absent"


def test_predicate_symbol_has_one_reviewed_query_reference() -> None:
    symbol = predicate("ready", "is_ready")

    assert symbol.truth_procedure_ref == TruthProcedureRef.query(
        "ready", symbol.grounding_plan.version
    )
    assert not hasattr(symbol, "evaluator")
    assert not hasattr(symbol, "program")


def test_non_boolean_query_result_is_a_grounding_failure(
    miniature_universe: ObjectUniverse,
) -> None:
    context = context_over(catalog_with(invalid_result=lambda *arguments: None))
    selection = Selection(
        predicates={"ready": predicate("ready", "invalid_result")}, operators={}
    )
    with pytest.raises(GroundingFailure) as error:
        ground(selection, miniature_universe, context)

    assert error.value.code is GroundingFailureCode.QUERY_ERROR
    assert "returned NoneType" in error.value.detail


def test_query_exception_is_a_structured_grounding_failure(
    miniature_universe: ObjectUniverse,
) -> None:
    def fail(*arguments):
        raise KeyError("missing SDT annotation")

    context = context_over(catalog_with(failing_query=fail))
    selection = Selection(
        predicates={"ready": predicate("ready", "failing_query")}, operators={}
    )

    with pytest.raises(GroundingFailure) as error:
        ground(selection, miniature_universe, context)

    assert error.value.code is GroundingFailureCode.QUERY_ERROR
    assert "missing SDT annotation" in error.value.detail
    assert error.value.atom == Literal("ready", ("robot",))
