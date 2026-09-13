"""The per-task loop: select, ground, plan, execute, verify.

No learned component runs in here. Selection is symbol-table lookup,
grounding computes the selected object scope's Boolean state, planning is Fast Downward,
execution is monitored and effect-checked, and
the goal is independently re-verified at the end. A failed round records
a nogood and re-enters the loop on the changed world; a plan identical
to one that already failed is refused instead of re-executed, so the
loop cannot burn its rounds repeating a failure unchanged.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from typing_extensions import Optional

from resym.core.grounding_model import GroundingFailure
from resym.core.symbols import Literal, SymbolLibrary
from resym.core.validation import InvalidSymbolLibraryError, selection_model_issues
from resym.platform.feasibility import (
    FEASIBILITY_FACTORY_NAMESPACE,
    feasibility_factory_uid,
)
from resym.platform.grounding_context import EvaluationContext
from resym.planning.execution.engine import (
    ExecutionReport,
    GoalCheck,
    ExecutionViolation,
    PlatformExecutionResult,
    PlatformSkillRealization,
    check_goal,
    execute,
)
from resym.planning.state_evaluation import ground
from resym.planning.events import (
    PipelineEvent,
    PipelineEventSink,
    ObjectScopeExpansionReason,
    action_payload,
    emit_event,
    literal_payload,
)
from resym.planning.pddl import (
    GroundAction,
    UnsolvableProblemError,
    plan as run_planner,
    write_domain,
    write_problem,
)
from resym.planning.selection import select_for_goal, Selection
from resym.planning.object_scope import (
    ObjectScopeAdvisor,
    PlanningObjectScope,
    PlanningObjectSelector,
)
from resym.platform.universe import ObjectUniverse


@dataclass(frozen=True)
class Nogood:
    """One recorded failure: which action of which plan failed, and why.

    Kept across replanning rounds so a failure certificate can show what
    was already tried, and so an unchanged failing plan is refused
    instead of repeated.
    """

    plan: tuple[GroundAction, ...]
    """The full plan that failed."""

    violated_action: GroundAction | None
    """The action that was refused or whose effect failed; ``None`` when the
    plan executed fully but the goal check failed."""

    violation: ExecutionViolation | None
    """The execution-level violation, if any."""

    detail: str
    """Human-readable provenance (failed literal, grounding reason, or goal
    check result)."""

    platform_result: PlatformExecutionResult | None = None
    """The platform outcome that caused this failure, when applicable."""


@dataclass
class TaskResult:
    """Everything one task solve produced, for inspection and tests."""

    plan: list[GroundAction] = field(default_factory=list)
    """The final successfully executed plan."""

    execution: ExecutionReport | None = None
    """Report of the last execution attempt."""

    goal_check: GoalCheck | None = None
    """The independent final goal verification of the last attempt."""

    replanning_rounds: int = 0
    """How many times a failed round forced re-projection."""

    nogoods: list[Nogood] = field(default_factory=list)
    """Every failure recorded across the rounds, in order."""

    evaluation_count: int = 0
    """Total predicate evaluations across all grounding rounds."""

    scope_expansions: int = 0
    """Object-set expansions, counted independently of execution retries."""

    scope_seconds: float = 0.0
    """Time spent selecting objects and querying their dependencies."""

    grounding_seconds: float = 0.0
    """Wall-clock time spent computing truth values."""

    planning_seconds: float = 0.0
    """Wall-clock time spent inside Fast Downward."""

    domain_text: str = ""
    """The last projected PDDL domain."""

    problem_text: str = ""
    """The last projected PDDL problem."""


class ReplanningLimitExceededError(Exception):
    """Raised when failed rounds exhaust the replanning budget.

    Carries the last execution report and the recorded nogoods so the
    failure can flow into a structured failure certificate.
    """

    def __init__(
        self,
        rounds: int,
        last_execution: ExecutionReport | None = None,
        nogoods: tuple[Nogood, ...] = (),
    ):
        super().__init__(f"Gave up after {rounds} replanning rounds.")
        self.last_execution = last_execution
        self.nogoods = nogoods


class RepeatedFailedPlanError(Exception):
    """Raised when replanning produced a plan identical to one that already
    failed: the symbolic state did not change, so executing it again cannot
    succeed and would only repeat the failure.

    Carries the last execution report and the recorded nogoods so the
    failure can flow into a structured failure certificate.
    """

    def __init__(
        self,
        plan: tuple[GroundAction, ...],
        nogoods: tuple[Nogood, ...],
        last_execution: ExecutionReport | None = None,
    ):
        super().__init__(
            "Replanning repeated an already-failed plan unchanged; refusing to "
            "re-execute it."
        )
        self.plan = plan
        self.nogoods = nogoods
        self.last_execution = last_execution


class InvalidGroundingFactoryBindingError(Exception):
    """A selected predicate references no factory available to the robot."""

    def __init__(self, missing: tuple[str, ...], goal: tuple = (), selection=None):
        super().__init__("Invalid grounding factory binding: " + "; ".join(missing))
        self.missing = missing
        self.goal = goal
        self.selection = selection


class UnsupportedCapabilityError(Exception):
    """The selected operators require capabilities unavailable to the robot."""

    def __init__(
        self,
        missing: tuple[str, ...],
        goal: tuple = (),
        selection=None,
    ):
        super().__init__("Unsupported capability: " + "; ".join(missing))
        self.missing = missing
        self.goal = goal
        self.selection = selection


def solve_task(
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    goal: tuple[Literal, ...],
    working_directory: Path,
    maximum_replanning_rounds: int = 3,
    realization: Optional[PlatformSkillRealization] = None,
    verify_goal: bool = True,
    check_postconditions: bool = True,
    event_sink: Optional[PipelineEventSink] = None,
    available_capabilities: frozenset[str] | None = None,
    object_advisor: Optional[ObjectScopeAdvisor] = None,
) -> TaskResult:
    """Solve one symbolic goal on the current world using the persistent
    library.

    ``verify_goal`` and ``check_postconditions`` exist only for the monitoring-
    ablation baselines; full monitoring is the default.

    Before anything is grounded, the goal-relevant selection is checked
    against the capabilities derived from the CRAM robot. A missing execution
    capability raises :class:`UnsupportedCapabilityError`. A missing grounding-factory
    binding raises :class:`InvalidGroundingFactoryBindingError`, allowing repair to
    distinguish a bad symbol binding from a genuine platform limitation.
    """
    result = TaskResult()
    robot_name = type(context.robot).__name__ if context.robot is not None else "none"
    robot_type = (
        f"{type(context.robot).__module__}:{type(context.robot).__qualname__}"
        if context.robot is not None
        else "none"
    )
    emit_event(
        event_sink,
        PipelineEvent.TASK_STARTED,
        goal=[literal_payload(literal) for literal in goal],
        robot_type=robot_type,
    )
    selection = select_for_goal(library, goal)
    model_issues = selection_model_issues(library, selection)
    if model_issues:
        emit_event(
            event_sink,
            PipelineEvent.MODEL_VALIDATION_FAILED,
            issues=[issue.render() for issue in model_issues],
        )
        raise InvalidSymbolLibraryError(model_issues, selection)
    if available_capabilities is not None:
        capabilities = available_capabilities
    elif realization is not None:
        capabilities = realization.available_capabilities(context.robot)
    else:
        raise ValueError(
            "capability availability needs a platform realization or an explicit set"
        )
    supported_feasibility_factories = {
        feasibility_factory_uid(capability_uid) for capability_uid in capabilities
    }
    missing_grounding_factories = tuple(
        f"predicate '{predicate.name}' needs grounding factory "
        f"'{predicate.grounding_plan.factory_uid}', which robot '{robot_name}' "
        "does not implement"
        for predicate in selection.predicates.values()
        if not _grounding_factory_supported(
            predicate.grounding_plan.factory_uid,
            robot_type,
            context,
            supported_feasibility_factories,
        )
    )
    missing_capabilities = tuple(
        f"operator '{operator.name}' needs capability '{capability_uid}', "
        f"which robot '{robot_name}' does not provide"
        for operator in selection.operators.values()
        if (capability_uid := operator.execution_binding.capability_ref.uid)
        not in capabilities
    )
    if missing_capabilities:
        missing = missing_grounding_factories + missing_capabilities
        emit_event(
            event_sink,
            PipelineEvent.TASK_UNSUPPORTED,
            missing=[str(requirement) for requirement in missing],
        )
        raise UnsupportedCapabilityError(missing, goal=goal, selection=selection)
    if missing_grounding_factories:
        emit_event(
            event_sink,
            PipelineEvent.GROUNDING_FACTORY_BINDING_INVALID,
            missing=[str(requirement) for requirement in missing_grounding_factories],
        )
        raise InvalidGroundingFactoryBindingError(
            missing_grounding_factories, goal=goal, selection=selection
        )
    failed_plans: set[tuple[GroundAction, ...]] = set()
    for round_index in range(maximum_replanning_rounds):
        emit_event(event_sink, PipelineEvent.ROUND_STARTED, round=round_index + 1)
        actions = _plan_object_scopes(
            selection,
            universe,
            context,
            goal,
            working_directory,
            round_index + 1,
            result,
            failed_plans,
            event_sink,
            object_advisor,
        )

        plan_key = tuple(actions)
        if plan_key in failed_plans:
            emit_event(
                event_sink,
                PipelineEvent.REPEATED_PLAN_REFUSED,
                round=round_index + 1,
                actions=[action_payload(action) for action in actions],
            )
            raise RepeatedFailedPlanError(
                plan_key, tuple(result.nogoods), last_execution=result.execution
            )

        result.execution = execute(
            actions,
            library,
            universe,
            context,
            realization,
            check_postconditions=check_postconditions,
            event_sink=event_sink,
        )
        if result.execution.succeeded:
            result.goal_check = (
                check_goal(
                    goal,
                    library,
                    universe,
                    context,
                    event_sink=event_sink,
                )
                if verify_goal
                else GoalCheck(satisfied=True)
            )
            emit_event(
                event_sink,
                (
                    PipelineEvent.GOAL_CHECKED
                    if verify_goal
                    else PipelineEvent.GOAL_CHECK_SKIPPED
                ),
                satisfied=result.goal_check.satisfied,
                failed_literal=(
                    literal_payload(result.goal_check.failed_literal)
                    if result.goal_check.failed_literal is not None
                    else None
                ),
                truth=(
                    str(result.goal_check.truth).lower()
                    if result.goal_check.truth is not None
                    else None
                ),
            )
            if result.goal_check.satisfied:
                result.plan = actions
                emit_event(
                    event_sink,
                    PipelineEvent.TASK_SUCCEEDED,
                    round=round_index + 1,
                    actions=len(actions),
                )
                return result
            failed_plans.add(plan_key)
            result.nogoods.append(
                Nogood(
                    plan=plan_key,
                    violated_action=None,
                    violation=None,
                    detail=(
                        f"goal literal {result.goal_check.failed_literal} "
                        f"evaluated {result.goal_check.truth}"
                    ),
                )
            )
        else:
            failed_plans.add(plan_key)
            result.nogoods.append(
                Nogood(
                    plan=plan_key,
                    violated_action=result.execution.violated_action,
                    violation=result.execution.violation,
                    detail=(
                        f"{result.execution.violation.value}: "
                        f"{result.execution.violated_literal} "
                        f"({result.execution.violation_reason or 'no reason'})"
                    ),
                    platform_result=(
                        result.execution.platform_results[-1]
                        if result.execution.violation
                        in {
                            ExecutionViolation.PLATFORM_UNSUPPORTED,
                            ExecutionViolation.PLATFORM_REJECTED,
                            ExecutionViolation.PLATFORM_FAILED,
                        }
                        and result.execution.platform_results
                        else None
                    ),
                )
            )
        result.replanning_rounds += 1
        emit_event(
            event_sink,
            PipelineEvent.REPLANNING_STARTED,
            completed_round=round_index + 1,
            reason=result.nogoods[-1].detail,
        )
    emit_event(
        event_sink,
        PipelineEvent.REPLANNING_LIMIT_EXCEEDED,
        rounds=result.replanning_rounds,
    )
    raise ReplanningLimitExceededError(
        result.replanning_rounds, result.execution, tuple(result.nogoods)
    )


# %% object scope attempts


def _plan_object_scopes(
    selection: Selection,
    universe: ObjectUniverse,
    context: EvaluationContext,
    goal: tuple[Literal, ...],
    working_directory: Path,
    round_number: int,
    result: TaskResult,
    failed_plans: set[tuple[GroundAction, ...]],
    event_sink: Optional[PipelineEventSink],
    object_advisor: Optional[ObjectScopeAdvisor] = None,
) -> list[GroundAction]:
    """Plan on growing object sets against the current, unchanged world state."""
    selector = PlanningObjectSelector(
        universe, selection, goal, context.robot, advisor=object_advisor
    )
    scope_start = time.perf_counter()
    scope = selector.initial()
    scope_seconds = time.perf_counter() - scope_start
    result.scope_seconds += scope_seconds
    _emit_advice(event_sink, round_number, 1, scope, 0)
    attempt = 0
    result.domain_text = write_domain(selection, name="resym")
    while True:
        attempt += 1
        task_universe = scope.universe(universe)
        emit_event(
            event_sink,
            PipelineEvent.TASK_OBJECTS_SELECTED,
            round=round_number,
            attempt=attempt,
            total_objects=len(universe.objects),
            seconds=round(scope_seconds, 4),
            selected_objects=sorted(scope.object_names),
            inclusion_reasons={
                name: asdict(reason) for name, reason in scope.inclusion_reasons.items()
            },
        )
        grounding_start = time.perf_counter()
        try:
            grounding = ground(
                selection, universe, context, active_universe=task_universe
            )
        except GroundingFailure as error:
            emit_event(
                event_sink,
                PipelineEvent.GROUNDING_FAILED,
                round=round_number,
                code=error.code.value,
                detail=error.detail,
                atom=literal_payload(error.atom) if error.atom is not None else None,
            )
            raise
        grounding_seconds = time.perf_counter() - grounding_start
        result.grounding_seconds += grounding_seconds
        result.evaluation_count += grounding.evaluation_count
        emit_event(
            event_sink,
            PipelineEvent.GROUNDING_COMPLETED,
            round=round_number,
            attempt=attempt,
            evaluations=grounding.evaluation_count,
            true=len(grounding.true_atoms),
            false=len(grounding.false_atoms),
            seconds=round(grounding_seconds, 4),
        )
        result.problem_text = write_problem(
            task_universe,
            grounding.true_atoms,
            goal,
            domain_name="resym",
            name=f"task-round-{round_number}-scope-{attempt}",
            selection=selection,
        )
        emit_event(
            event_sink,
            PipelineEvent.PLANNING_STARTED,
            round=round_number,
            attempt=attempt,
        )
        planning_start = time.perf_counter()
        try:
            try:
                actions = run_planner(
                    result.domain_text,
                    result.problem_text,
                    working_directory / f"round-{round_number}" / f"scope-{attempt}",
                )
            finally:
                planning_seconds = time.perf_counter() - planning_start
                result.planning_seconds += planning_seconds
        except UnsolvableProblemError as error:
            expansion_reason = ObjectScopeExpansionReason.UNSOLVABLE_SUBSET
            scope_start = time.perf_counter()
            expanded = selector.expand(scope, expansion_reason, str(error))
            scope_seconds = time.perf_counter() - scope_start
            result.scope_seconds += scope_seconds
            if expanded is None:
                error.grounding = grounding
                emit_event(
                    event_sink,
                    PipelineEvent.OBJECT_SCOPE_EXHAUSTED,
                    round=round_number,
                    attempt=attempt,
                    selected_objects=sorted(scope.object_names),
                )
                raise
        else:
            expansion_reason = ObjectScopeExpansionReason.REPEATED_FAILED_PLAN
            scope_start = time.perf_counter()
            can_expand = tuple(actions) in failed_plans and (
                result.execution is None
                or result.execution.violation
                not in {
                    ExecutionViolation.PLATFORM_UNSUPPORTED,
                    ExecutionViolation.PRECONDITION_GROUNDING_FAILED,
                    ExecutionViolation.POSTCONDITION_GROUNDING_FAILED,
                }
            )
            expanded = (
                selector.expand(
                    scope,
                    expansion_reason,
                    "the plan "
                    + " ".join(str(action) for action in actions)
                    + " already failed in execution",
                )
                if can_expand
                else None
            )
            scope_seconds = time.perf_counter() - scope_start
            result.scope_seconds += scope_seconds
            if expanded is None:
                emit_event(
                    event_sink,
                    PipelineEvent.PLAN_GENERATED,
                    round=round_number,
                    actions=[action_payload(action) for action in actions],
                    seconds=round(planning_seconds, 4),
                )
                return actions
        emit_event(
            event_sink,
            PipelineEvent.OBJECT_SCOPE_EXPANDED,
            round=round_number,
            attempt=attempt,
            reason=expansion_reason,
            added_objects=sorted(expanded.object_names - scope.object_names),
        )
        result.scope_expansions += 1
        _emit_advice(event_sink, round_number, attempt + 1, expanded, len(scope.advice))
        scope = expanded


def _emit_advice(
    event_sink: Optional[PipelineEventSink],
    round_number: int,
    attempt: int,
    scope: PlanningObjectScope,
    already_reported: int,
) -> None:
    """Report every advisor consultation the scope gained since the last attempt."""
    for advice in scope.advice[already_reported:]:
        emit_event(
            event_sink,
            PipelineEvent.OBJECT_SCOPE_ADVISED,
            round=round_number,
            attempt=attempt,
            recommendations=[asdict(item) for item in advice.recommendations],
            trace=list(advice.trace),
        )


def _grounding_factory_supported(
    factory_uid: str,
    robot_type: str,
    context: EvaluationContext,
    supported_feasibility_factories: set[str],
) -> bool:
    """
    A feasibility factory is supported exactly when its capability is available on
    the robot; any other factory answers for itself.
    """
    if factory_uid.startswith(FEASIBILITY_FACTORY_NAMESPACE):
        return factory_uid in supported_feasibility_factories
    return context.grounding_catalog.supports(factory_uid, robot_type)
