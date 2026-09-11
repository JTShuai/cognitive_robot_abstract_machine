"""
Execution consumes Boolean predicates and keeps query failures separate.
"""

from __future__ import annotations

import pytest

from resym.core.grounding import GroundingFailure, GroundingFailureCode
from resym.core.model import (
    Literal,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
    SymbolType,
)
from resym.planning.execution.engine import (
    ExecutionViolation,
    PlatformExecutionResult,
    PlatformSkillRealization,
    check_goal,
    execute,
)
from resym.planning.pddl import GroundAction
from resym.platform.embodiment import EmbodimentProfile
from resym.platform.evaluators import EVALUATORS, EvaluationContext
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.world_entity import Body

from .capability_helpers import capability_contract, execution_binding

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)
CAPABILITY_UID = "test:Noop"


class RecordingRealization(PlatformSkillRealization):
    """
    Record requests that pass symbolic precondition checks.
    """

    def __init__(self) -> None:
        self.applied = []

    def execute(self, request, context, universe):
        self.applied.append(request)
        return PlatformExecutionResult.succeeded("STUB_EXECUTION_SUCCEEDED")


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
    return EvaluationContext(
        world=None,
        robot=None,
        profile=EmbodimentProfile(
            name="stub",
            evaluators=frozenset({"true", "false", "failure"}),
            capabilities=frozenset({CAPABILITY_UID}),
        ),
    )


@pytest.fixture(autouse=True)
def predicate_queries():
    def fail(query_context, universe, arguments):
        raise GroundingFailure(
            GroundingFailureCode.RESOURCE_LIMIT,
            "query budget exhausted",
        )

    EVALUATORS.update(
        {"true": lambda *arguments: True, "false": lambda *arguments: False}
    )
    EVALUATORS["failure"] = fail
    yield
    for name in ("true", "false", "failure"):
        del EVALUATORS[name]


def library_with(
    evaluator: str,
    *,
    precondition: bool = True,
    negated: bool = False,
) -> SymbolLibrary:
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="probe",
            parameter_types=(ROBOT_TYPE,),
            evaluator=evaluator,
            fluent=True,
        )
    )
    library.add(
        Operator(
            name="act",
            parameters=(("r", ROBOT_TYPE),),
            preconditions=(
                (Literal("probe", ("r",), negated=negated),) if precondition else ()
            ),
            add_effects=() if precondition else (Literal("probe", ("r",)),),
            delete_effects=(),
            execution_binding=execution_binding(CAPABILITY_UID, (("actor", "r"),)),
        )
    )
    library.add_capability_contract(
        capability_contract(CAPABILITY_UID, (("actor", ROBOT_TYPE),), ("probe",))
    )
    return library


def run(library, miniature_universe, context):
    realization = RecordingRealization()
    report = execute(
        [GroundAction(operator="act", arguments=("robot",))],
        library,
        miniature_universe,
        context,
        realization,
    )
    return report, realization


def test_false_precondition_refuses_dispatch(miniature_universe, context) -> None:
    report, realization = run(library_with("false"), miniature_universe, context)

    assert report.violation is ExecutionViolation.PRECONDITION_FALSE
    assert realization.applied == []


def test_negated_false_precondition_dispatches(miniature_universe, context) -> None:
    report, realization = run(
        library_with("false", negated=True), miniature_universe, context
    )

    assert report.succeeded
    assert len(realization.applied) == 1


def test_precondition_query_failure_is_not_a_false_value(
    miniature_universe, context
) -> None:
    report, realization = run(library_with("failure"), miniature_universe, context)

    assert report.violation is ExecutionViolation.PRECONDITION_GROUNDING_FAILED
    assert report.violation_reason == "query budget exhausted"
    assert report.grounding_failure_code == "resource_limit"
    assert realization.applied == []


def test_unmaterialized_effect_stops_execution(miniature_universe, context) -> None:
    report, realization = run(
        library_with("false", precondition=False), miniature_universe, context
    )

    assert report.violation is ExecutionViolation.POSTCONDITION_FAILED
    assert len(realization.applied) == 1


def test_effect_query_failure_has_its_own_violation(
    miniature_universe, context
) -> None:
    report, realization = run(
        library_with("failure", precondition=False), miniature_universe, context
    )

    assert report.violation is ExecutionViolation.POSTCONDITION_GROUNDING_FAILED
    assert report.violation_reason == "query budget exhausted"
    assert report.grounding_failure_code == "resource_limit"
    assert len(realization.applied) == 1


def test_goal_check_returns_a_boolean_value(miniature_universe, context) -> None:
    goal = (Literal("probe", ("robot",)),)

    failed = check_goal(goal, library_with("false"), miniature_universe, context)
    satisfied = check_goal(goal, library_with("true"), miniature_universe, context)

    assert failed.satisfied is False
    assert failed.truth is False
    assert satisfied.satisfied is True


def test_goal_query_failure_is_not_returned_as_a_third_value(
    miniature_universe, context
) -> None:
    goal = (Literal("probe", ("robot",)),)

    with pytest.raises(GroundingFailure) as caught:
        check_goal(goal, library_with("failure"), miniature_universe, context)

    assert caught.value.code is GroundingFailureCode.RESOURCE_LIMIT
    assert caught.value.atom == goal[0]
