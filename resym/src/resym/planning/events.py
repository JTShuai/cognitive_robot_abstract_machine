"""
Optional structured events emitted by task planning and execution.

The planning core only calls a supplied callback.  It does not know whether events are
printed, persisted, or displayed, and emits nothing when no callback is supplied.
"""

from __future__ import annotations

from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Any, Callable, Optional, TypeAlias

if TYPE_CHECKING:
    from resym.core.model import Literal
    from resym.planning.pddl import GroundAction

PipelineEventSink: TypeAlias = Callable[[str, dict[str, Any]], None]


class PipelineEvent(StrEnum):
    """
    One kind of structured event in a task's trace stream.

    The members are the complete vocabulary of the persisted trace: producers
    emit them through :func:`emit_event` and trace consumers match on them.
    """

    TASK_STARTED = "task_started"
    MODEL_VALIDATION_FAILED = "model_validation_failed"
    TASK_UNSUPPORTED = "task_unsupported"
    GROUNDING_FACTORY_BINDING_INVALID = "grounding_factory_binding_invalid"
    TASK_OBJECTS_SELECTED = "task_objects_selected"
    ROUND_STARTED = "round_started"
    GROUNDING_FAILED = "grounding_failed"
    GROUNDING_COMPLETED = "grounding_completed"
    PLANNING_STARTED = "planning_started"
    PLAN_GENERATED = "plan_generated"
    PLAN_PREVIEW_STARTED = "plan_preview_started"
    PLAN_PREVIEW_COMPLETED = "plan_preview_completed"
    REPEATED_PLAN_REFUSED = "repeated_plan_refused"
    ACTION_STARTED = "action_started"
    PRECONDITION_CHECKED = "precondition_checked"
    PRECONDITION_GROUNDING_FAILED = "precondition_grounding_failed"
    EXECUTION_REQUEST_CREATED = "execution_request_created"
    PLATFORM_EXECUTION_STARTED = "platform_execution_started"
    CORAPLEX_ACTION_SELECTED = "coraplex_action_selected"
    PLATFORM_RESULT = "platform_result"
    EFFECT_CHECKED = "effect_checked"
    EFFECT_GROUNDING_FAILED = "effect_grounding_failed"
    POSTCONDITION_CHECK_SKIPPED = "postcondition_check_skipped"
    ACTION_COMPLETED = "action_completed"
    EXECUTION_STOPPED = "execution_stopped"
    EXECUTION_COMPLETED = "execution_completed"
    GOAL_LITERAL_CHECKED = "goal_literal_checked"
    GOAL_GROUNDING_FAILED = "goal_grounding_failed"
    GOAL_CHECKED = "goal_checked"
    GOAL_CHECK_SKIPPED = "goal_check_skipped"
    REPLANNING_STARTED = "replanning_started"
    REPLANNING_LIMIT_EXCEEDED = "replanning_limit_exceeded"
    TASK_SUCCEEDED = "task_succeeded"


def emit_event(
    sink: Optional[PipelineEventSink], event: PipelineEvent, **data: Any
) -> None:
    """
    Send one event when observation is enabled.
    """
    if sink is not None:
        sink(event, data)


def literal_payload(literal: Literal) -> dict[str, Any]:
    """
    Return the stable display fields of a ground literal.
    """
    arguments = list(literal.arguments)
    prefix = "not " if literal.negated else ""
    return {
        "predicate": literal.predicate,
        "arguments": arguments,
        "negated": literal.negated,
        "display": f"{prefix}{literal.predicate}({', '.join(arguments)})",
    }


def action_payload(action: GroundAction) -> dict[str, Any]:
    """
    Return the stable display fields of a grounded action.
    """
    arguments = list(action.arguments)
    joined = " ".join(arguments)
    return {
        "operator": action.operator,
        "arguments": arguments,
        "display": f"({action.operator}{' ' if joined else ''}{joined})",
    }
