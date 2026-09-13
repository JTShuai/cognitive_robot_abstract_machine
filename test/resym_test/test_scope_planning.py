"""
Scope retries are independent of execution retries and planner faults.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field

from resym.planning import pipeline
from resym.planning.events import ObjectScopeExpansionReason, PipelineEvent
from resym.planning.object_scope import ObjectRecommendation, ObjectScopeAdvice
from resym.planning.pddl import (
    GroundAction,
    UnsolvableProblemError,
    PlannerTimeoutError,
)
from resym.core.grounding_model import GroundingFailure, GroundingFailureCode
from resym.repair import diagnosis
from resym.repair.certificate import FailureClass
from resym.platform.universe import GroundedObject
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body

from .test_pipeline_monitoring import (
    library,
    miniature_universe,
    stub_context,
    stub_world,
    StubRealization,
    UnsupportedRealization,
    GOAL,
    CAPABILITY_UID,
    ROBOT_TYPE,
)

# %% planning attempts


@pytest.fixture()
def world_objects(miniature_universe):
    """
    An additional available actor outside the initial goal objects.
    """
    miniature_universe.add(
        GroundedObject("reserve", ROBOT_TYPE, Body(name=PrefixedName("reserve")))
    )
    return miniature_universe


def test_unsolvable_subset_expands_without_using_execution_rounds(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    problems = []
    events = []

    def planner(domain, problem, directory):
        problems.append(problem)
        if len(problems) == 1:
            raise UnsolvableProblemError("selected objects are insufficient")
        return [GroundAction("act", GOAL[0].arguments)]

    monkeypatch.setattr(pipeline, "run_planner", planner)
    result = pipeline.solve_task(
        library,
        world_objects,
        stub_context,
        GOAL,
        tmp_path,
        realization=StubRealization(stub_world),
        maximum_replanning_rounds=1,
        event_sink=lambda event, data: events.append((event, data)),
    )

    assert "reserve" not in problems[0]
    assert "reserve" in problems[1]
    assert result.replanning_rounds == 0
    assert result.scope_expansions == 1
    assert result.evaluation_count == 3
    assert [event for event, _ in events].count(
        PipelineEvent.OBJECT_SCOPE_EXPANDED
    ) == 1


def test_planner_timeout_does_not_expand_scope(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    calls = []

    def planner(*arguments):
        calls.append(arguments)
        raise PlannerTimeoutError("timeout")

    monkeypatch.setattr(pipeline, "run_planner", planner)
    with pytest.raises(PlannerTimeoutError):
        pipeline.solve_task(
            library,
            world_objects,
            stub_context,
            GOAL,
            tmp_path,
            realization=StubRealization(stub_world),
        )
    assert len(calls) == 1


def test_exhausted_scope_preserves_the_last_grounding(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    def planner(*arguments):
        raise UnsolvableProblemError("no solution")

    monkeypatch.setattr(pipeline, "run_planner", planner)
    with pytest.raises(UnsolvableProblemError) as error:
        pipeline.solve_task(
            library,
            world_objects,
            stub_context,
            GOAL,
            tmp_path,
            realization=StubRealization(stub_world),
        )

    assert error.value.grounding.evaluation_count == len(world_objects.objects)


def test_execution_and_goal_check_receive_the_complete_world(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    observed = []
    backend = StubRealization(stub_world)
    native_execute = backend.execute

    def execute(request, context, universe):
        observed.append(universe)
        return native_execute(request, context, universe)

    monkeypatch.setattr(backend, "execute", execute)
    monkeypatch.setattr(
        pipeline,
        "run_planner",
        lambda *arguments: [GroundAction("act", GOAL[0].arguments)],
    )
    result = pipeline.solve_task(
        library, world_objects, stub_context, GOAL, tmp_path, realization=backend
    )

    assert observed == [world_objects]
    assert result.goal_check.satisfied
    assert result.evaluation_count == 1


def test_grounding_failure_does_not_try_larger_scopes(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    calls = []

    def fail(*arguments):
        calls.append(arguments)
        raise GroundingFailure(GroundingFailureCode.QUERY_ERROR, "native query failed")

    monkeypatch.setattr(
        stub_context.grounding_catalog, "resolve", lambda *args, **kwargs: fail
    )
    with pytest.raises(GroundingFailure) as error:
        pipeline.solve_task(
            library,
            world_objects,
            stub_context,
            GOAL,
            tmp_path,
            realization=StubRealization(stub_world),
        )

    assert error.value.code is GroundingFailureCode.QUERY_ERROR
    assert len(calls) == 1


def test_repeated_failed_plan_expands_before_final_refusal(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    backend = StubRealization(stub_world, actually_works=False)
    events = []
    monkeypatch.setattr(
        pipeline,
        "run_planner",
        lambda *arguments: [GroundAction("act", GOAL[0].arguments)],
    )
    with pytest.raises(pipeline.RepeatedFailedPlanError):
        pipeline.solve_task(
            library,
            world_objects,
            stub_context,
            GOAL,
            tmp_path,
            realization=backend,
            event_sink=lambda event, data: events.append((event, data)),
        )

    assert len(backend.applied) == 1
    assert [event for event, _ in events].count(
        PipelineEvent.OBJECT_SCOPE_EXPANDED
    ) == 1


def test_unsupported_execution_does_not_trigger_scope_expansion(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    tmp_path,
):
    events = []
    monkeypatch.setattr(
        pipeline,
        "run_planner",
        lambda *arguments: [GroundAction("act", GOAL[0].arguments)],
    )
    with pytest.raises(pipeline.RepeatedFailedPlanError):
        pipeline.solve_task(
            library,
            world_objects,
            stub_context,
            GOAL,
            tmp_path,
            realization=UnsupportedRealization(),
            event_sink=lambda event, data: events.append((event, data)),
        )

    assert PipelineEvent.OBJECT_SCOPE_EXPANDED not in [event for event, _ in events]


def test_diagnosis_reuses_exhausted_scope_evidence(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    def planner(*arguments):
        raise UnsolvableProblemError("no solution")

    def forbidden_regrounding(*arguments):
        pytest.fail("diagnosis must reuse the state already evaluated by planning")

    monkeypatch.setattr(pipeline, "run_planner", planner)
    monkeypatch.setattr(diagnosis, "ground", forbidden_regrounding)
    result = diagnosis.diagnose(
        library,
        world_objects,
        stub_context,
        GOAL,
        tmp_path,
        realization=StubRealization(stub_world),
    )

    assert result.certificate.failure_class is FailureClass.GEOMETRIC_INFEASIBILITY


# %% advised scope


@dataclass
class RecordingAdvisor:
    """
    Recommends the reserve actor only after the planner has rejected the subset.
    """

    requests: list = field(default_factory=list)

    def advise(self, request):
        self.requests.append(request)
        if request.expansion_reason is None:
            return ObjectScopeAdvice()
        return ObjectScopeAdvice(
            recommendations=(ObjectRecommendation("reserve", "another actor"),),
            trace=({"tool": "query_candidate_objects"},),
        )


def test_unsolvable_subset_consults_the_advisor_before_typed_growth(
    monkeypatch,
    library,
    world_objects,
    stub_context,
    stub_world,
    tmp_path,
):
    problems = []
    events = []
    advisor = RecordingAdvisor()

    def planner(domain, problem, directory):
        problems.append(problem)
        if len(problems) == 1:
            raise UnsolvableProblemError("selected objects are insufficient")
        return [GroundAction("act", GOAL[0].arguments)]

    monkeypatch.setattr(pipeline, "run_planner", planner)
    pipeline.solve_task(
        library,
        world_objects,
        stub_context,
        GOAL,
        tmp_path,
        realization=StubRealization(stub_world),
        maximum_replanning_rounds=1,
        event_sink=lambda event, data: events.append((event, data)),
        object_advisor=advisor,
    )

    assert [request.expansion_reason for request in advisor.requests] == [
        None,
        ObjectScopeExpansionReason.UNSOLVABLE_SUBSET,
    ]
    assert "selected objects are insufficient" in advisor.requests[1].planner_message
    advised = [
        data for event, data in events if event == PipelineEvent.OBJECT_SCOPE_ADVISED
    ]
    assert [record["attempt"] for record in advised] == [1, 2]
    assert advised[1]["recommendations"] == [
        {"name": "reserve", "rationale": "another actor"}
    ]
    assert advised[1]["trace"] == [{"tool": "query_candidate_objects"}]
    selected = [
        data for event, data in events if event == PipelineEvent.TASK_OBJECTS_SELECTED
    ]
    assert selected[1]["inclusion_reasons"]["reserve"] == {
        "reason": "advised",
        "related_to": None,
        "rationale": "another actor",
    }
