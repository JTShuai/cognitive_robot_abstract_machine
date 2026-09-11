"""
Closed-loop monitoring in solve_task: final goal verification, nogood recording, and
refusal to repeat an already-failed plan unchanged.

Stub evaluators and a stub backend over a miniature universe; Fast Downward runs for
real, so this lives in the container suite.
"""

from __future__ import annotations

import pytest

from resym.platform.embodiment import EmbodimentProfile
from resym.platform.evaluators import EVALUATORS, EvaluationContext
from resym.planning.execution.engine import (
    ExecutionViolation,
    PlatformExecutionResult,
    PlatformSkillRealization,
)
from resym.core.model import (
    Literal,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
)
from resym.planning.pipeline import (
    RepeatedFailedPlanError,
    solve_task,
)
from resym.repair.certificate import FailureClass
from resym.repair.diagnosis import diagnose
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from .capability_helpers import capability_contract, execution_binding

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


CAPABILITY_UID = "test:SetDone"


class StubWorld:
    """
    One boolean of world state, mutated by the stub skill and read by the stub evaluator
    — the smallest possible closed loop.
    """

    def __init__(self):
        self.done = False


class StubRealization(PlatformSkillRealization):
    def __init__(self, world: StubWorld, actually_works: bool = True):
        self.world = world
        self.actually_works = actually_works
        self.applied = []

    def execute(self, request, context, universe):
        self.applied.append(request)
        if self.actually_works:
            self.world.done = True
        return PlatformExecutionResult.succeeded("STUB_EXECUTION_SUCCEEDED")


class RejectingRealization(PlatformSkillRealization):
    def execute(self, request, context, universe):
        return PlatformExecutionResult.rejected(
            "NATIVE_PRECONDITION_FAILED", "target is no longer reachable"
        )


class UnsupportedRealization(PlatformSkillRealization):
    def execute(self, request, context, universe):
        return PlatformExecutionResult.unsupported(
            "NATIVE_CAPABILITY_NOT_IMPLEMENTED", "no platform mapping"
        )


@pytest.fixture()
def stub_world():
    return StubWorld()


@pytest.fixture()
def stub_evaluator(stub_world):
    EVALUATORS["stub_done"] = lambda context, universe, arguments: stub_world.done
    yield
    del EVALUATORS["stub_done"]


@pytest.fixture()
def miniature_universe():
    universe = ObjectUniverse()
    universe.add(
        GroundedObject(
            name="rob",
            symbol_type=ROBOT_TYPE,
            body=Body(name=PrefixedName("rob")),
        )
    )
    return universe


@pytest.fixture()
def library():
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="done",
            parameter_types=(ROBOT_TYPE,),
            evaluator="stub_done",
            fluent=True,
        )
    )
    library.add(
        Operator(
            name="act",
            parameters=(("r", ROBOT_TYPE),),
            preconditions=(),
            add_effects=(Literal("done", ("r",)),),
            delete_effects=(),
            execution_binding=execution_binding(CAPABILITY_UID, (("actor", "r"),)),
        )
    )
    library.add_capability_contract(
        capability_contract(CAPABILITY_UID, (("actor", ROBOT_TYPE),), ("done",))
    )
    return library


GOAL = (Literal("done", ("rob",)),)


def stub_context():
    profile = EmbodimentProfile(
        name="stub",
        evaluators=frozenset({"stub_done"}),
        capabilities=frozenset({CAPABILITY_UID}),
    )
    return EvaluationContext(world=None, robot=None, profile=profile)


def test_working_skill_passes_postcondition_and_goal_check(
    stub_evaluator, stub_world, miniature_universe, library, tmp_path
):
    backend = StubRealization(stub_world, actually_works=True)
    events = []
    result = solve_task(
        library,
        miniature_universe,
        stub_context(),
        GOAL,
        tmp_path,
        realization=backend,
        event_sink=lambda event, data: events.append((event, data)),
    )
    assert result.goal_check is not None and result.goal_check.satisfied
    assert result.nogoods == []
    assert [a.operator for a in result.plan] == ["act"]
    names = [event for event, _ in events]
    assert names[0] == "task_started"
    assert "grounding_completed" in names
    assert "plan_generated" in names
    assert "goal_checked" in names
    assert names[-1] == "task_succeeded"


def test_broken_skill_fails_postcondition_then_refuses_repeat(
    stub_evaluator, stub_world, miniature_universe, library, tmp_path
):
    """
    A skill that silently does nothing: round 1 catches the unmaterialized effect,
    records a nogood, and round 2 — identical world, identical plan — is refused instead
    of re-executed.
    """
    backend = StubRealization(stub_world, actually_works=False)
    with pytest.raises(RepeatedFailedPlanError) as error:
        solve_task(
            library,
            miniature_universe,
            stub_context(),
            GOAL,
            tmp_path,
            realization=backend,
        )
    assert len(backend.applied) == 1  # executed once, never repeated
    (nogood,) = error.value.nogoods
    assert nogood.violation == ExecutionViolation.POSTCONDITION_FAILED
    assert nogood.violated_action is not None


def test_repeated_platform_rejection_preserves_its_native_code(
    stub_evaluator, miniature_universe, library, tmp_path
):
    diagnosis = diagnose(
        library,
        miniature_universe,
        stub_context(),
        GOAL,
        tmp_path,
        realization=RejectingRealization(),
    )

    assert diagnosis.certificate is not None
    assert diagnosis.certificate.failure_class is FailureClass.GEOMETRIC_INFEASIBILITY
    assert diagnosis.certificate.platform_failure_code == "NATIVE_PRECONDITION_FAILED"


def test_unsupported_platform_result_is_classified_as_unsupported_capability(
    stub_evaluator, miniature_universe, library, tmp_path
):
    diagnosis = diagnose(
        library,
        miniature_universe,
        stub_context(),
        GOAL,
        tmp_path,
        realization=UnsupportedRealization(),
    )

    assert diagnosis.certificate is not None
    assert diagnosis.certificate.failure_class is FailureClass.UNSUPPORTED_CAPABILITY
    assert (
        diagnosis.certificate.platform_failure_code
        == "NATIVE_CAPABILITY_NOT_IMPLEMENTED"
    )


def test_goal_check_catches_what_disabled_postconditions_miss(
    stub_evaluator, stub_world, miniature_universe, library, tmp_path
):
    """
    With postcondition checks ablated, the broken skill slips through execution — the
    independent final goal check still refuses success.
    """
    backend = StubRealization(stub_world, actually_works=False)
    with pytest.raises(RepeatedFailedPlanError) as error:
        solve_task(
            library,
            miniature_universe,
            stub_context(),
            GOAL,
            tmp_path,
            realization=backend,
            check_postconditions=False,
        )
    (nogood,) = error.value.nogoods
    assert nogood.violation is None
    assert "goal literal" in nogood.detail


def test_fully_ablated_monitoring_reports_false_success(
    stub_evaluator, stub_world, miniature_universe, library, tmp_path
):
    """
    The one-shot baseline of E3: no postcondition check, no goal check — the broken
    skill is reported as success.

    This test documents the baseline's blindness on purpose.
    """
    backend = StubRealization(stub_world, actually_works=False)
    result = solve_task(
        library,
        miniature_universe,
        stub_context(),
        GOAL,
        tmp_path,
        realization=backend,
        check_postconditions=False,
        verify_goal=False,
    )
    assert result.goal_check.satisfied  # blind
    assert stub_world.done is False  # reality
