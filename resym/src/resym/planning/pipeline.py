"""The per-task loop: select, ground, plan, execute, verify.

No learned component runs in here. Selection is symbol-table lookup,
grounding computes a complete Boolean state, planning is Fast Downward,
execution is monitored and effect-checked, and
the goal is independently re-verified at the end. A failed round records
a nogood and re-enters the loop on the changed world; a plan identical
to one that already failed is refused instead of re-executed, so the
loop cannot burn its rounds repeating a failure unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Optional

from resym.core.grounding import GroundingFailure
from resym.platform.embodiment import (
    InvalidEvaluatorBindingError,
    UnsupportedCapabilityError,
)
from resym.platform.evaluators import EvaluationContext
from resym.planning.execution.engine import (
    ExecutionReport,
    GoalCheck,
    ExecutionViolation,
    PlatformExecutionResult,
    PlatformSkillRealization,
    check_goal,
    execute,
)
from resym.planning.grounding import ground
from resym.planning.events import (
    PipelineEvent,
    PipelineEventSink,
    action_payload,
    emit_event,
    literal_payload,
)
from resym.core.model import Literal, SymbolLibrary
from resym.core.validation import (
    InvalidSymbolLibraryError,
    selection_model_issues,
)
from resym.planning.pddl import (
    GroundAction,
    plan as run_planner,
    write_domain,
    write_problem,
)
from resym.planning.selection import select_for_goal
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
    """Human-readable provenance (failed literal, evaluator reason, or goal
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
) -> TaskResult:
    """Solve one symbolic goal on the current world using the persistent
    library.

    ``verify_goal`` and ``check_postconditions`` exist only for the monitoring-
    ablation baselines; full monitoring is the default.

    Before anything is grounded, the goal-relevant selection is checked
    against the embodiment profile. A missing execution capability raises
    :class:`UnsupportedCapabilityError`. A missing evaluator binding raises
    :class:`InvalidEvaluatorBindingError`, allowing repair to distinguish a
    bad symbol binding from a genuine platform limitation.
    """
    result = TaskResult()
    emit_event(
        event_sink,
        PipelineEvent.TASK_STARTED,
        goal=[literal_payload(literal) for literal in goal],
        embodiment=context.profile.name,
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
    missing_evaluators = context.profile.missing_evaluators(selection)
    missing_capabilities = context.profile.missing_capabilities(selection)
    if missing_capabilities:
        missing = missing_evaluators + missing_capabilities
        emit_event(
            event_sink,
            PipelineEvent.TASK_UNSUPPORTED,
            missing=[str(requirement) for requirement in missing],
        )
        raise UnsupportedCapabilityError(missing, goal=goal, selection=selection)
    if missing_evaluators:
        emit_event(
            event_sink,
            PipelineEvent.EVALUATOR_BINDING_INVALID,
            missing=[str(requirement) for requirement in missing_evaluators],
        )
        raise InvalidEvaluatorBindingError(
            missing_evaluators, goal=goal, selection=selection
        )
    task_universe = universe.for_task(selection, goal)
    emit_event(
        event_sink,
        PipelineEvent.TASK_OBJECTS_SELECTED,
        total_objects=len(universe.objects),
        selected_objects=list(task_universe.objects),
    )
    failed_plans: set[tuple[GroundAction, ...]] = set()
    for round_index in range(maximum_replanning_rounds):
        emit_event(event_sink, PipelineEvent.ROUND_STARTED, round=round_index + 1)
        grounding_start = time.perf_counter()
        try:
            grounding = ground(selection, task_universe, context)
        except GroundingFailure as error:
            emit_event(
                event_sink,
                PipelineEvent.GROUNDING_FAILED,
                round=round_index + 1,
                code=error.code.value,
                detail=error.detail,
                atom=(literal_payload(error.atom) if error.atom is not None else None),
            )
            raise
        grounding_seconds = time.perf_counter() - grounding_start
        result.grounding_seconds += grounding_seconds
        result.evaluation_count += grounding.evaluation_count
        emit_event(
            event_sink,
            PipelineEvent.GROUNDING_COMPLETED,
            round=round_index + 1,
            evaluations=grounding.evaluation_count,
            true=len(grounding.true_atoms),
            false=len(grounding.false_atoms),
            seconds=round(grounding_seconds, 4),
        )

        result.domain_text = write_domain(selection, name="resym")
        result.problem_text = write_problem(
            task_universe,
            grounding.true_atoms,
            goal,
            domain_name="resym",
            name=f"task-round-{round_index}",
            selection=selection,
        )
        planning_start = time.perf_counter()
        emit_event(event_sink, PipelineEvent.PLANNING_STARTED, round=round_index + 1)
        actions = run_planner(
            result.domain_text, result.problem_text, working_directory
        )
        planning_seconds = time.perf_counter() - planning_start
        result.planning_seconds += planning_seconds
        emit_event(
            event_sink,
            PipelineEvent.PLAN_GENERATED,
            round=round_index + 1,
            actions=[action_payload(action) for action in actions],
            seconds=round(planning_seconds, 4),
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
            task_universe,
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
                    task_universe,
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
