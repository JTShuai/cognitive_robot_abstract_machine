"""
Run one task to a verdict: success, or a structured failure certificate.

Every way :func:`~resym.planning.pipeline.solve_task` can fail — the static capability
gate, an unknown goal predicate, a planner-proved unsolvable task, an execution
violation, a refused repeated plan — maps here onto the programmatic certificate
builders, so callers (the fault diagnosis evaluation, and later the experiment harness)
get one uniform outcome type instead of five exception paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Optional

from resym.core.grounding import GroundingFailure
from resym.repair.certificate import (
    FailureCertificate,
    certify_evaluator_binding_error,
    certify_execution_failure,
    certify_grounding_failure,
    certify_invalid_model,
    certify_unsolvable,
    certify_unsupported_capability,
)
from resym.core.validation import InvalidSymbolLibraryError
from resym.platform.embodiment import (
    InvalidEvaluatorBindingError,
    UnsupportedCapabilityError,
)
from resym.platform.evaluators import EvaluationContext
from resym.planning.execution.engine import ExecutionReport
from resym.planning.grounding import ground
from resym.core.model import Literal, SymbolLibrary
from resym.planning.pddl import PlanNotFoundError
from resym.planning.pipeline import (
    Nogood,
    RepeatedFailedPlanError,
    ReplanningLimitExceededError,
    TaskResult,
    solve_task,
)
from resym.planning.selection import (
    Selection,
    UnknownPredicateError,
    select_for_goal,
)
from resym.platform.universe import ObjectUniverse


@dataclass
class TaskDiagnosis:
    """
    The uniform outcome of one attempted task.
    """

    succeeded: bool
    certificate: Optional[FailureCertificate] = None
    result: Optional[TaskResult] = None

    @property
    def failure_class_value(self) -> Optional[str]:
        """
        The certificate's class as a string, ``None`` on success — the shape the
        diagnosis evaluation consumes.
        """
        if self.certificate is None:
            return None
        return self.certificate.failure_class.value


def diagnose(
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    goal: tuple[Literal, ...],
    working_directory: Path,
    **solve_arguments,
) -> TaskDiagnosis:
    """
    Attempt the task and classify whatever happens.

    A successful solve (goal independently verified) diagnoses as success; every failure
    path becomes a certificate via the deterministic evidence rules of
    :mod:`certificate`.
    """
    try:
        result = solve_task(
            library, universe, context, goal, working_directory, **solve_arguments
        )
    except GroundingFailure as error:
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_grounding_failure(
                goal,
                select_for_goal(library, goal),
                error,
            ),
        )
    except InvalidEvaluatorBindingError as error:
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_evaluator_binding_error(
                goal,
                error.selection if error.selection is not None else Selection(),
                missing="; ".join(error.missing),
            ),
        )
    except UnsupportedCapabilityError as error:
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_unsupported_capability(
                goal,
                error.selection if error.selection is not None else Selection(),
                missing="; ".join(error.missing),
            ),
        )
    except UnknownPredicateError as error:
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_unsolvable(
                goal, library, Selection(), planner_message=str(error)
            ),
        )
    except InvalidSymbolLibraryError as error:
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_invalid_model(
                goal,
                library,
                error.selection,
                error.issues,
            ),
        )
    except PlanNotFoundError as error:
        selection = select_for_goal(library, goal)
        grounding = ground(selection, universe, context)
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_unsolvable(
                goal,
                library,
                selection,
                planner_message=str(error),
                grounding=grounding,
            ),
        )
    except (RepeatedFailedPlanError, ReplanningLimitExceededError) as error:
        return TaskDiagnosis(
            succeeded=False,
            certificate=certify_execution_failure(
                goal,
                library,
                select_for_goal(library, goal),
                _report_from_nogoods(error.nogoods, error.last_execution),
                nogoods=tuple(error.nogoods),
            ),
        )
    return TaskDiagnosis(succeeded=True, result=result)


def _report_from_nogoods(
    nogoods: tuple[Nogood, ...],
    last_execution: Optional[ExecutionReport],
) -> ExecutionReport:
    """
    The execution evidence of a replanning dead end: the real last report where the
    pipeline kept one, otherwise reconstructed from the last recorded nogood (a
    repeated-plan refusal happens before any new execution, so only the nogood log knows
    what failed).
    """
    if last_execution is not None and last_execution.violation is not None:
        return last_execution
    if not nogoods:
        return ExecutionReport()
    last = nogoods[-1]
    return ExecutionReport(
        violation=last.violation,
        violated_action=last.violated_action,
        violation_reason=last.detail,
        grounding_failure_code=(
            last_execution.grounding_failure_code
            if last_execution is not None
            else None
        ),
        platform_results=[last.platform_result] if last.platform_result else [],
    )
