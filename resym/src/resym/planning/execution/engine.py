"""Monitored execution behind a platform realization.

Before each action runs, its symbolic preconditions are recomputed against the
current world rather than trusted from the planner snapshot. Platform-native
preconditions and execution failures are reported by the realization itself.

How an admitted action changes the world is the platform realization's
business. The operator binding first creates an :class:`ExecutionRequest`;
the realization consumes only that request, never the operator name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Optional

from krrood.adapters.json_serializer import to_json
from resym.core.grounding import GroundingFailure
from resym.platform.grounding_context import EvaluationContext
from resym.planning.grounding import evaluate_predicate
from resym.core.model import (
    CapabilityContract,
    ExecutionRequest,
    Literal,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
    contract_violations,
)
from resym.planning.pddl import GroundAction
from resym.planning.events import (
    PipelineEvent,
    PipelineEventSink,
    action_payload,
    emit_event,
    literal_payload,
)
from resym.platform.universe import ObjectUniverse


class ExecutionViolation(Enum):
    """Why execution stopped before the complete plan finished."""

    PRECONDITION_FALSE = "precondition recomputed false"
    PRECONDITION_GROUNDING_FAILED = "precondition grounding failed"
    PLATFORM_UNSUPPORTED = "platform capability unsupported"
    PLATFORM_REJECTED = "platform-native precondition rejected execution"
    PLATFORM_FAILED = "platform execution failed"
    POSTCONDITION_FAILED = "declared effect did not materialize"
    POSTCONDITION_GROUNDING_FAILED = "postcondition grounding failed"


class PlatformExecutionStatus(Enum):
    """Platform-level outcome for one :class:`ExecutionRequest`."""

    SUCCEEDED = "succeeded"
    UNSUPPORTED = "unsupported"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class PlatformExecutionResult:
    """Structured result returned by a platform realization."""

    status: PlatformExecutionStatus
    code: str
    detail: str | None = None

    @classmethod
    def succeeded(cls, code: str = "EXECUTION_SUCCEEDED") -> PlatformExecutionResult:
        return cls(PlatformExecutionStatus.SUCCEEDED, code)

    @classmethod
    def unsupported(
        cls, code: str, detail: str | None = None
    ) -> PlatformExecutionResult:
        return cls(PlatformExecutionStatus.UNSUPPORTED, code, detail)

    @classmethod
    def rejected(cls, code: str, detail: str | None = None) -> PlatformExecutionResult:
        return cls(PlatformExecutionStatus.REJECTED, code, detail)

    @classmethod
    def failed(cls, code: str, detail: str | None = None) -> PlatformExecutionResult:
        return cls(PlatformExecutionStatus.FAILED, code, detail)


@dataclass
class ExecutionReport:
    """What happened while executing one plan."""

    executed: list[GroundAction] = field(default_factory=list)
    """Actions that ran, in order."""

    requests: list[ExecutionRequest] = field(default_factory=list)
    """Ground capability requests dispatched in the same order."""

    platform_results: list[PlatformExecutionResult] = field(default_factory=list)
    """Platform outcomes, including the result that stopped execution."""

    violation: ExecutionViolation | None = None
    """Set when execution stopped before the plan ended."""

    violated_action: GroundAction | None = None
    """The action that was refused or failed, if any."""

    violated_literal: Literal | None = None
    """The ground literal associated with the violation."""

    violation_reason: str | None = None
    """Query or platform evidence associated with the violation."""

    grounding_failure_code: str | None = None
    """Stable query-failure code when predicate grounding could not finish."""

    @property
    def succeeded(self) -> bool:
        return self.violation is None


class MissingWitnessPoseError(Exception):
    """Raised when execution requires a witness pose grounding did not find."""

    def __init__(self, request: ExecutionRequest):
        super().__init__(
            f"No witness base pose recorded for {request}; "
            "grounding must run 'openable' first."
        )


class UnknownCapabilityError(Exception):
    """Raised when no realization is registered for a capability."""

    def __init__(self, capability_uid: str):
        super().__init__(f"No realization registered for '{capability_uid}'.")


class UnsupportedCapabilityRealizationError(Exception):
    """A realization exists, but the current embodiment does not provide it."""

    def __init__(self, capability_uid: str, profile_name: str):
        super().__init__(
            f"Capability '{capability_uid}' is not provided by embodiment "
            f"'{profile_name}'."
        )


class PlatformSkillRealization(ABC):
    """Execute one capability request on a concrete platform."""

    @abstractmethod
    def execute(
        self,
        request: ExecutionRequest,
        context: EvaluationContext,
        universe: ObjectUniverse,
    ) -> PlatformExecutionResult:
        """Execute the request and return a structured platform outcome."""


def execution_request_for(
    operator: Operator,
    action: GroundAction,
    contract: CapabilityContract,
    predicates: dict[str, PredicateSymbol] | None = None,
) -> ExecutionRequest:
    """Ground an admitted operator binding and recheck its contract."""
    violations = contract_violations(operator, contract, predicates)
    if violations:
        raise InvalidExecutionBindingError(violations)
    return operator.execution_binding.resolve(action.binding(operator))


class InvalidExecutionBindingError(Exception):
    """
    Raised when an admitted operator's binding violates its capability contract.
    """

    def __init__(self, violations: list[str]):
        super().__init__("; ".join(violations))
        self.violations = tuple(violations)


def execute(
    plan: list[GroundAction],
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    realization: Optional[PlatformSkillRealization] = None,
    check_postconditions: bool = True,
    event_sink: Optional[PipelineEventSink] = None,
) -> ExecutionReport:
    """Run the plan action by action, recomputing truth before each step and
    verifying the declared effects immediately after it.

    ``check_postconditions=False`` exists only for the monitoring-ablation
    baselines; the full monitoring policy is the default.
    """
    if plan and realization is None:
        raise ValueError("a platform realization is required to execute a plan")
    report = ExecutionReport()
    for action_index, action in enumerate(plan):
        emit_event(
            event_sink,
            PipelineEvent.ACTION_STARTED,
            action_index=action_index,
            action=action_payload(action),
        )
        operator = library.operators[action.operator]
        refusal = _check_preconditions(
            operator,
            action,
            library,
            universe,
            context,
            event_sink=event_sink,
            action_index=action_index,
        )
        if _refused(
            report,
            action,
            refusal,
        ):
            _emit_execution_stopped(event_sink, action_index, action, report)
            return report
        capability_uid = operator.execution_binding.capability_ref.uid
        contract = library.capability_contracts.get(capability_uid)
        if contract is None:
            raise UnknownCapabilityError(capability_uid)
        request = execution_request_for(operator, action, contract, library.predicates)
        report.requests.append(request)
        emit_event(
            event_sink,
            PipelineEvent.EXECUTION_REQUEST_CREATED,
            action_index=action_index,
            action=action_payload(action),
            capability_ref=to_json(request.capability_ref),
            arguments=request.argument_map,
            role_bindings={
                role: to_json(binding)
                for role, binding in operator.execution_binding.role_bindings
            },
        )
        emit_event(
            event_sink,
            PipelineEvent.PLATFORM_EXECUTION_STARTED,
            action_index=action_index,
            realization=type(realization).__name__,
        )
        platform_result = realization.execute(request, context, universe)
        report.platform_results.append(platform_result)
        emit_event(
            event_sink,
            PipelineEvent.PLATFORM_RESULT,
            action_index=action_index,
            status=platform_result.status.value,
            code=platform_result.code,
            detail=platform_result.detail,
        )
        if platform_result.status is not PlatformExecutionStatus.SUCCEEDED:
            report.violation = {
                PlatformExecutionStatus.UNSUPPORTED: ExecutionViolation.PLATFORM_UNSUPPORTED,
                PlatformExecutionStatus.REJECTED: ExecutionViolation.PLATFORM_REJECTED,
                PlatformExecutionStatus.FAILED: ExecutionViolation.PLATFORM_FAILED,
            }[platform_result.status]
            report.violated_action = action
            report.violation_reason = platform_result.code + (
                f": {platform_result.detail}" if platform_result.detail else ""
            )
            _emit_execution_stopped(event_sink, action_index, action, report)
            return report
        report.executed.append(action)
        if check_postconditions:
            refusal = _check_postconditions(
                operator,
                action,
                library,
                universe,
                context,
                event_sink=event_sink,
                action_index=action_index,
            )
            if _refused(report, action, refusal):
                _emit_execution_stopped(event_sink, action_index, action, report)
                return report
        else:
            emit_event(
                event_sink,
                PipelineEvent.POSTCONDITION_CHECK_SKIPPED,
                action_index=action_index,
            )
        emit_event(
            event_sink,
            PipelineEvent.ACTION_COMPLETED,
            action_index=action_index,
            action=action_payload(action),
        )
    emit_event(
        event_sink, PipelineEvent.EXECUTION_COMPLETED, executed=len(report.executed)
    )
    return report


@dataclass(frozen=True)
class _Refusal:
    """One reason an action may not be dispatched."""

    violation: ExecutionViolation
    literal: Literal | None = None
    reason: str | None = None
    grounding_failure_code: str | None = None


def _refused(
    report: ExecutionReport, action: GroundAction, refusal: Optional[_Refusal]
) -> bool:
    if refusal is None:
        return False
    report.violation = refusal.violation
    report.violated_action = action
    report.violated_literal = refusal.literal
    report.violation_reason = refusal.reason
    report.grounding_failure_code = refusal.grounding_failure_code
    return True


def _check_preconditions(
    operator: Operator,
    action: GroundAction,
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    event_sink: Optional[PipelineEventSink] = None,
    action_index: int = 0,
) -> Optional[_Refusal]:
    """Require every binary precondition to hold immediately before dispatch."""
    binding = action.binding(operator)
    for precondition in operator.preconditions:
        ground_literal = precondition.substitute(binding)
        arguments = tuple(universe[name] for name in ground_literal.arguments)
        predicate = library.predicates[ground_literal.predicate]
        required = not ground_literal.negated
        try:
            value = evaluate_predicate(predicate, arguments, universe, context)
        except GroundingFailure as error:
            emit_event(
                event_sink,
                PipelineEvent.PRECONDITION_GROUNDING_FAILED,
                action_index=action_index,
                literal=literal_payload(ground_literal),
                failure_code=error.code.value,
                reason=error.detail,
            )
            return _Refusal(
                ExecutionViolation.PRECONDITION_GROUNDING_FAILED,
                ground_literal,
                error.detail,
                error.code.value,
            )
        emit_event(
            event_sink,
            PipelineEvent.PRECONDITION_CHECKED,
            action_index=action_index,
            literal=literal_payload(ground_literal),
            truth=str(value).lower(),
            required=str(required).lower(),
            satisfied=value is required,
            reason=None,
        )
        if value is not required:
            return _Refusal(ExecutionViolation.PRECONDITION_FALSE, ground_literal)
    return None


def _check_postconditions(
    operator: Operator,
    action: GroundAction,
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    event_sink: Optional[PipelineEventSink] = None,
    action_index: int = 0,
) -> Optional[_Refusal]:
    """Immediately after the platform realization ran, every add effect must
    recompute ``TRUE`` and every declared delete effect ``FALSE`` on fluent
    predicates. A false result or grounding failure stops the plan where
    the mismatch occurred."""
    binding = action.binding(operator)
    for effect, required in [
        *((literal, True) for literal in operator.add_effects),
        *((literal, False) for literal in operator.delete_effects),
    ]:
        predicate = library.predicates[effect.predicate]
        if not predicate.fluent:
            continue
        ground_literal = effect.substitute(binding)
        arguments = tuple(universe[name] for name in ground_literal.arguments)
        try:
            value = evaluate_predicate(predicate, arguments, universe, context)
        except GroundingFailure as error:
            emit_event(
                event_sink,
                PipelineEvent.EFFECT_GROUNDING_FAILED,
                action_index=action_index,
                literal=literal_payload(ground_literal),
                failure_code=error.code.value,
                reason=error.detail,
            )
            return _Refusal(
                ExecutionViolation.POSTCONDITION_GROUNDING_FAILED,
                ground_literal,
                error.detail,
                error.code.value,
            )
        emit_event(
            event_sink,
            PipelineEvent.EFFECT_CHECKED,
            action_index=action_index,
            literal=literal_payload(ground_literal),
            truth=str(value).lower(),
            required=str(required).lower(),
            satisfied=value is required,
            reason=None,
        )
        if value is not required:
            return _Refusal(ExecutionViolation.POSTCONDITION_FAILED, ground_literal)
    return None


@dataclass(frozen=True)
class GoalCheck:
    """The independent final verification of a task goal."""

    satisfied: bool
    """Every goal literal holds in the final world state."""

    failed_literal: Literal | None = None
    """The first literal that is not known to hold, if any."""

    truth: bool | None = None
    """Boolean value of the failed literal, absent when all literals hold."""


def check_goal(
    goal: tuple[Literal, ...],
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    event_sink: Optional[PipelineEventSink] = None,
) -> GoalCheck:
    """Re-evaluate the goal literals on the final world state, independently
    of what the executed actions promised. A positive literal must be proven
    true and a negated one must evaluate false."""
    for literal in goal:
        predicate = library.predicates[literal.predicate]
        arguments = tuple(universe[name] for name in literal.arguments)
        required = not literal.negated
        try:
            value = evaluate_predicate(predicate, arguments, universe, context)
        except GroundingFailure as error:
            emit_event(
                event_sink,
                PipelineEvent.GOAL_GROUNDING_FAILED,
                literal=literal_payload(literal),
                failure_code=error.code.value,
                reason=error.detail,
            )
            raise error.for_atom(literal) from error
        emit_event(
            event_sink,
            PipelineEvent.GOAL_LITERAL_CHECKED,
            literal=literal_payload(literal),
            truth=str(value).lower(),
            required=str(required).lower(),
            satisfied=value is required,
            reason=None,
        )
        if value is not required:
            return GoalCheck(
                satisfied=False,
                failed_literal=literal,
                truth=value,
            )
    return GoalCheck(satisfied=True)


def _emit_execution_stopped(
    event_sink: Optional[PipelineEventSink],
    action_index: int,
    action: GroundAction,
    report: ExecutionReport,
) -> None:
    violation = report.violation
    emit_event(
        event_sink,
        PipelineEvent.EXECUTION_STOPPED,
        action_index=action_index,
        action=action_payload(action),
        violation=violation.name if violation is not None else None,
        detail=report.violation_reason,
        literal=(
            literal_payload(report.violated_literal)
            if report.violated_literal is not None
            else None
        ),
    )
