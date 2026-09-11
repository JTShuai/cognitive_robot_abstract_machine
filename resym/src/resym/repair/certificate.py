"""Structured failure certificates with a programmatic taxonomy.

Every task failure is classified *by program components* — planner
result, binary grounding, query diagnostics, execution violations, and platform codes —
never by a language model. The certificate carries the evidence: the
goal, the relevant binary state, query diagnostics, the failed
action or literal, budget status, and the causal neighborhood. A repair
agent may later hypothesize among the ``alternative_classes`` the
evidence allows, but it cannot overwrite the primary class or the raw
error codes.

Missing-model and deterministic operator-model defects trigger curation.
``UNSUPPORTED_CAPABILITY`` means the platform lacks a low-level
implementation and automatic repair must stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import Optional

from resym.core.capabilities import BindingSource
from resym.core.grounding import GroundingFailure
from resym.planning.execution.engine import (
    ExecutionReport,
    ExecutionViolation,
)
from resym.planning.grounding import GroundingResult
from resym.core.model import Literal, SymbolLibrary
from resym.planning.pddl import GroundAction
from resym.planning.pipeline import Nogood
from resym.planning.selection import Selection
from resym.core.validation import ModelIssue, ModelIssueKind


class FailureClass(Enum):
    """The first-level failure taxonomy (revised plan §5.1)."""

    GOAL_SPECIFICATION_ERROR = "goal_specification_error"
    OBJECT_OR_PERCEPTION_MISSING = "object_or_perception_missing"
    GROUNDING_FAILURE = "grounding_failure"
    GEOMETRIC_INFEASIBILITY = "geometric_infeasibility"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    MISSING_PREDICATE_MODEL = "missing_predicate_model"
    PREDICATE_IMPLEMENTATION_ERROR = "predicate_implementation_error"
    MISSING_OPERATOR_MODEL = "missing_operator_model"
    OPERATOR_SIGNATURE_ERROR = "operator_signature_error"
    OPERATOR_PRECONDITION_ERROR = "operator_precondition_error"
    OPERATOR_EFFECT_ERROR = "operator_effect_error"
    PRECONDITION_INVALIDATED = "precondition_invalidated"
    PLATFORM_EXECUTION_FAILURE = "platform_execution_failure"
    POSTCONDITION_FAILURE = "postcondition_failure"


CURATION_TRIGGERS = frozenset(
    {
        FailureClass.MISSING_PREDICATE_MODEL,
        FailureClass.PREDICATE_IMPLEMENTATION_ERROR,
        FailureClass.MISSING_OPERATOR_MODEL,
        FailureClass.OPERATOR_SIGNATURE_ERROR,
        FailureClass.OPERATOR_PRECONDITION_ERROR,
        FailureClass.OPERATOR_EFFECT_ERROR,
    }
)
"""Failure classes that identify a repairable symbolic-model defect."""


@dataclass(frozen=True)
class TruthSummary:
    """The binary knowledge state relevant to the failure."""

    true_atoms: tuple[Literal, ...] = ()
    false_atoms: tuple[Literal, ...] = ()
    evaluation_count: int = 0

    @classmethod
    def from_grounding(cls, grounding: Optional[GroundingResult]) -> TruthSummary:
        if grounding is None:
            return cls()
        return cls(
            true_atoms=tuple(sorted(grounding.true_atoms, key=_atom_key)),
            false_atoms=tuple(sorted(grounding.false_atoms, key=_atom_key)),
            evaluation_count=grounding.evaluation_count,
        )


@dataclass(frozen=True)
class CausalNeighborhood:
    """The library slice around the goal: which predicates and operators
    the failure touches, and which goal predicates have no achiever."""

    predicates: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    unachievable_goal_predicates: tuple[str, ...] = ()
    """Goal predicates no selected operator adds."""

    @classmethod
    def from_selection(
        cls, selection: Selection, goal: tuple[Literal, ...]
    ) -> CausalNeighborhood:
        achievable = {
            effect.predicate
            for operator in selection.operators.values()
            for effect in operator.add_effects
        }
        return cls(
            predicates=tuple(sorted(selection.predicates)),
            operators=tuple(sorted(selection.operators)),
            unachievable_goal_predicates=tuple(
                sorted(
                    {
                        literal.predicate
                        for literal in goal
                        if not literal.negated and literal.predicate not in achievable
                    }
                )
            ),
        )


@dataclass(frozen=True)
class FailureCertificate:
    """One structured, program-classified task failure (revised plan §5.2)."""

    task_goal: tuple[Literal, ...]
    failure_class: FailureClass
    """The primary class, decided by program evidence."""

    alternative_classes: tuple[FailureClass, ...] = ()
    """Classes the same evidence also permits; a repair agent may test
    hypotheses among these, and only these."""

    context_id: str = "unspecified"
    library_version: Optional[str] = None
    embodiment_version: Optional[str] = None

    task_instruction: Optional[str] = None
    """Original user instruction, when the task entered through the language
    interface. It is retrieval evidence, not an executable condition."""

    goal_semantics: tuple[str, ...] = ()
    """Descriptions of clear desired relations absent from the local library."""

    task_object_types: tuple[tuple[str, str], ...] = ()
    """Ground object names and local symbol types referenced by the task."""

    planner_message: Optional[str] = None
    failed_action: Optional[GroundAction] = None
    violation: Optional[ExecutionViolation] = None
    violated_literal: Optional[Literal] = None
    violation_reason: Optional[str] = None
    grounding_failure_code: Optional[str] = None
    """Machine-readable predicate-grounding failure code."""

    platform_failure_code: Optional[str] = None

    truth: TruthSummary = TruthSummary()
    causal_neighborhood: CausalNeighborhood = CausalNeighborhood()
    executed_actions: tuple[GroundAction, ...] = ()
    nogoods: tuple[str, ...] = ()
    """Rendered nogood details across the replanning rounds."""

    model_issues: tuple[str, ...] = ()
    """Deterministic structural or causal model inconsistencies."""

    @property
    def triggers_curation(self) -> bool:
        return self.failure_class in CURATION_TRIGGERS

    def render(self) -> str:
        """A compact human/LLM-readable account of the evidence."""
        lines = [
            f"failure class: {self.failure_class.value}",
            "alternatives: "
            + (", ".join(c.value for c in self.alternative_classes) or "none"),
            "goal: " + _render_literals(self.task_goal),
        ]
        if self.task_instruction:
            lines.append(f"user instruction: {self.task_instruction.strip()[:500]}")
        if self.goal_semantics:
            lines.append("desired relations: " + "; ".join(self.goal_semantics))
        if self.task_object_types:
            lines.append(
                "task object types: "
                + ", ".join(
                    f"{name}={symbol_type}"
                    for name, symbol_type in self.task_object_types
                )
            )
        if self.planner_message:
            lines.append(f"planner: {self.planner_message.strip()[:500]}")
        if self.failed_action is not None:
            lines.append(
                f"failed action: ({self.failed_action.operator} "
                f"{' '.join(self.failed_action.arguments)})"
            )
        if self.violation is not None:
            lines.append(f"violation: {self.violation.value}")
        if self.violated_literal is not None:
            lines.append(
                f"violated literal: {_render_literals((self.violated_literal,))}"
            )
        if self.violation_reason:
            lines.append(f"evaluator reason: {self.violation_reason}")
        if self.grounding_failure_code:
            lines.append(f"grounding failure: {self.grounding_failure_code}")
        if self.platform_failure_code:
            lines.append(f"platform failure code: {self.platform_failure_code}")
        if self.causal_neighborhood.unachievable_goal_predicates:
            lines.append(
                "goal predicates without achiever: "
                + ", ".join(self.causal_neighborhood.unachievable_goal_predicates)
            )
        if self.causal_neighborhood.operators:
            lines.append(
                "causal operators: " + ", ".join(self.causal_neighborhood.operators)
            )
        if self.nogoods:
            lines.append("nogoods: " + " | ".join(self.nogoods))
        if self.model_issues:
            lines.append("model issues: " + " | ".join(self.model_issues))
        return "\n".join(lines)


def certify_unsolvable(
    goal: tuple[Literal, ...],
    library: SymbolLibrary,
    selection: Selection,
    planner_message: str,
    grounding: Optional[GroundingResult] = None,
    **metadata,
) -> FailureCertificate:
    """Classify a planner-proved-unsolvable failure.

    Deterministic evidence rules:

    - a goal predicate missing from the library         -> MISSING_PREDICATE_MODEL
      (the same evidence also permits GOAL_SPECIFICATION_ERROR);
    - a bound capability implies a missing goal effect  -> OPERATOR_EFFECT_ERROR;
    - a goal predicate no selected operator adds        -> MISSING_OPERATOR_MODEL;
    - every achiever requires the same goal literal     -> OPERATOR_PRECONDITION_ERROR;
    - achievers exist but no plan -> GEOMETRIC_INFEASIBILITY
      (also permits MISSING_OPERATOR_MODEL: an unmodeled intermediate step).
    """
    neighborhood = CausalNeighborhood.from_selection(selection, goal)
    missing_predicates = [
        literal.predicate
        for literal in goal
        if literal.predicate not in library.predicates
    ]
    effect_issues = _capability_effect_issues(
        library, set(neighborhood.unachievable_goal_predicates)
    )
    if effect_issues:
        neighborhood = CausalNeighborhood(
            predicates=neighborhood.predicates,
            operators=tuple(
                sorted(
                    set(neighborhood.operators)
                    | {operator for operator, _ in effect_issues}
                )
            ),
            unachievable_goal_predicates=(neighborhood.unachievable_goal_predicates),
        )
    if missing_predicates:
        failure_class = FailureClass.MISSING_PREDICATE_MODEL
        alternatives = (FailureClass.GOAL_SPECIFICATION_ERROR,)
    elif effect_issues:
        failure_class = FailureClass.OPERATOR_EFFECT_ERROR
        alternatives = ()
    elif neighborhood.unachievable_goal_predicates:
        failure_class = FailureClass.MISSING_OPERATOR_MODEL
        alternatives = ()
    elif _has_self_dependent_goal_achiever(goal, selection):
        failure_class = FailureClass.OPERATOR_PRECONDITION_ERROR
        alternatives = ()
    else:
        failure_class = FailureClass.GEOMETRIC_INFEASIBILITY
        alternatives = (FailureClass.MISSING_OPERATOR_MODEL,)
    return FailureCertificate(
        task_goal=goal,
        failure_class=failure_class,
        alternative_classes=alternatives,
        planner_message=planner_message,
        truth=_causal_truth_summary(goal, selection, grounding),
        causal_neighborhood=neighborhood,
        model_issues=(
            tuple(detail for _, detail in effect_issues)
            if failure_class is FailureClass.OPERATOR_EFFECT_ERROR
            else (
                (
                    "every achiever for a required goal literal requires that "
                    "same literal",
                )
                if failure_class is FailureClass.OPERATOR_PRECONDITION_ERROR
                else ()
            )
        ),
        **metadata,
    )


def _capability_effect_issues(
    library: SymbolLibrary, unachievable_goal_predicates: set[str]
) -> tuple[tuple[str, str], ...]:
    """Find operators whose constant capability target implies a missing effect.

    This uses only the platform-independent semantic mapping stored in the
    capability contract.  It does not infer physical target values or consult
    experiment ground truth.
    """
    issues: list[tuple[str, str]] = []
    for operator in library.operators.values():
        binding = operator.execution_binding
        contract = library.capability_contracts.get(binding.capability_ref.uid)
        if contract is None or contract.version != binding.capability_ref.version:
            continue
        role_bindings = binding.role_map
        for effect, role_name, value in contract.effect_role_values:
            if effect not in unachievable_goal_predicates:
                continue
            role_binding = role_bindings.get(role_name)
            if (
                role_binding is None
                or role_binding.source is not BindingSource.CONSTANT
                or role_binding.value != value
                or any(item.predicate == effect for item in operator.add_effects)
            ):
                continue
            observed = (
                ", ".join(sorted({item.predicate for item in operator.add_effects}))
                or "none"
            )
            issues.append(
                (
                    operator.name,
                    f"operator '{operator.name}' binds capability "
                    f"'{contract.uid}' role '{role_name}' to '{value}', which "
                    f"maps to add effect '{effect}', but its add effects are "
                    f"{observed}",
                )
            )
    return tuple(issues)


def _has_self_dependent_goal_achiever(
    goal: tuple[Literal, ...], selection: Selection
) -> bool:
    """Whether a required goal literal can only be achieved by requiring itself."""
    for required in goal:
        achievers: list[tuple] = []
        for operator in selection.operators.values():
            effects = (
                operator.delete_effects if required.negated else operator.add_effects
            )
            for effect in effects:
                binding = _unify(effect, required)
                if binding is None:
                    continue
                achievers.append((operator, binding))
        if achievers and all(
            any(
                precondition.negated == required.negated
                and _matches_bound(precondition, required, binding)
                for precondition in operator.preconditions
            )
            for operator, binding in achievers
        ):
            return True
    return False


def _causal_truth_summary(
    goal: tuple[Literal, ...],
    selection: Selection,
    grounding: Optional[GroundingResult],
) -> TruthSummary:
    if grounding is None:
        return TruthSummary()
    relevant = _causal_atoms(goal, selection, grounding)
    return TruthSummary(
        true_atoms=tuple(sorted(grounding.true_atoms & relevant, key=_atom_key)),
        false_atoms=tuple(sorted(grounding.false_atoms & relevant, key=_atom_key)),
        evaluation_count=grounding.evaluation_count,
    )


def _causal_atoms(
    goal: tuple[Literal, ...], selection: Selection, grounding: GroundingResult
) -> set[Literal]:
    """Ground atoms reachable by regressing the concrete task goal."""
    available = set(grounding.true_atoms) | set(grounding.false_atoms)
    relevant = {_positive_atom(literal) for literal in goal}
    frontier = list(goal)
    expanded: set[Literal] = set()
    while frontier:
        required = frontier.pop()
        if required in expanded:
            continue
        expanded.add(required)
        for operator in selection.operators.values():
            effects = (
                operator.delete_effects if required.negated else operator.add_effects
            )
            for effect in effects:
                binding = _unify(effect, required)
                if binding is None:
                    continue
                for precondition in operator.preconditions:
                    for atom in available:
                        grounded = Literal(
                            atom.predicate, atom.arguments, precondition.negated
                        )
                        if _matches(precondition, grounded, binding):
                            relevant.add(atom)
                            frontier.append(grounded)
    return relevant


def _unify(schema: Literal, grounded: Literal) -> Optional[dict[str, str]]:
    if schema.predicate != grounded.predicate or len(schema.arguments) != len(
        grounded.arguments
    ):
        return None
    binding: dict[str, str] = {}
    for variable, value in zip(schema.arguments, grounded.arguments):
        previous = binding.setdefault(variable, value)
        if previous != value:
            return None
    return binding


def _matches(schema: Literal, grounded: Literal, binding: dict[str, str]) -> bool:
    if schema.predicate != grounded.predicate or len(schema.arguments) != len(
        grounded.arguments
    ):
        return False
    return all(
        variable not in binding or binding[variable] == value
        for variable, value in zip(schema.arguments, grounded.arguments)
    )


def _matches_bound(schema: Literal, grounded: Literal, binding: dict[str, str]) -> bool:
    return (
        schema.predicate == grounded.predicate
        and len(schema.arguments) == len(grounded.arguments)
        and all(
            variable in binding and binding[variable] == value
            for variable, value in zip(schema.arguments, grounded.arguments)
        )
    )


def _positive_atom(literal: Literal) -> Literal:
    return Literal(literal.predicate, literal.arguments)


def certify_invalid_model(
    goal: tuple[Literal, ...],
    library: SymbolLibrary,
    selection: Selection,
    issues: tuple[ModelIssue, ...],
    **metadata,
) -> FailureCertificate:
    """Classify deterministic structural defects before grounding."""
    kinds = {issue.kind for issue in issues}
    if kinds == {ModelIssueKind.OPERATOR_SIGNATURE}:
        failure_class = FailureClass.OPERATOR_SIGNATURE_ERROR
    else:  # Keep the certificate total as new deterministic checks are added.
        failure_class = FailureClass.MISSING_OPERATOR_MODEL
    return FailureCertificate(
        task_goal=goal,
        failure_class=failure_class,
        causal_neighborhood=CausalNeighborhood.from_selection(selection, goal),
        model_issues=tuple(issue.render() for issue in issues),
        **metadata,
    )


def certify_grounding_failure(
    goal: tuple[Literal, ...],
    selection: Selection,
    failure: GroundingFailure,
    **metadata,
) -> FailureCertificate:
    """Record a query failure without turning it into a planning truth value."""
    return FailureCertificate(
        task_goal=goal,
        failure_class=FailureClass.GROUNDING_FAILURE,
        violated_literal=failure.atom,
        violation_reason=failure.detail,
        grounding_failure_code=failure.code.value,
        causal_neighborhood=CausalNeighborhood.from_selection(selection, goal),
        **metadata,
    )


_VIOLATION_CLASSES: dict[
    ExecutionViolation, tuple[FailureClass, tuple[FailureClass, ...]]
] = {
    ExecutionViolation.PRECONDITION_FALSE: (
        FailureClass.PRECONDITION_INVALIDATED,
        (FailureClass.MISSING_OPERATOR_MODEL,),
    ),
    ExecutionViolation.PRECONDITION_GROUNDING_FAILED: (
        FailureClass.GROUNDING_FAILURE,
        (FailureClass.PRECONDITION_INVALIDATED,),
    ),
    ExecutionViolation.PLATFORM_UNSUPPORTED: (
        FailureClass.UNSUPPORTED_CAPABILITY,
        (),
    ),
    ExecutionViolation.PLATFORM_REJECTED: (
        FailureClass.GEOMETRIC_INFEASIBILITY,
        (FailureClass.PRECONDITION_INVALIDATED,),
    ),
    ExecutionViolation.PLATFORM_FAILED: (
        FailureClass.PLATFORM_EXECUTION_FAILURE,
        (FailureClass.POSTCONDITION_FAILURE,),
    ),
    ExecutionViolation.POSTCONDITION_FAILED: (
        FailureClass.POSTCONDITION_FAILURE,
        (FailureClass.PLATFORM_EXECUTION_FAILURE, FailureClass.MISSING_OPERATOR_MODEL),
    ),
    ExecutionViolation.POSTCONDITION_GROUNDING_FAILED: (
        FailureClass.GROUNDING_FAILURE,
        (FailureClass.POSTCONDITION_FAILURE,),
    ),
}


def certify_execution_failure(
    goal: tuple[Literal, ...],
    library: SymbolLibrary,
    selection: Selection,
    report: ExecutionReport,
    grounding: Optional[GroundingResult] = None,
    nogoods: tuple[Nogood, ...] = (),
    platform_failure_code: Optional[str] = None,
    **metadata,
) -> FailureCertificate:
    """Classify a failure the monitored executor reported.

    The primary class follows the violation kind mechanically (the table
    above); a platform-level error code overrides toward
    ``PLATFORM_EXECUTION_FAILURE``.
    """
    explicit_platform_failure = platform_failure_code is not None
    if (
        platform_failure_code is None
        and report.violation
        in {
            ExecutionViolation.PLATFORM_UNSUPPORTED,
            ExecutionViolation.PLATFORM_REJECTED,
            ExecutionViolation.PLATFORM_FAILED,
        }
        and report.platform_results
    ):
        platform_failure_code = report.platform_results[-1].code
    if explicit_platform_failure:
        failure_class = FailureClass.PLATFORM_EXECUTION_FAILURE
        alternatives: tuple[FailureClass, ...] = (FailureClass.UNSUPPORTED_CAPABILITY,)
    elif report.violation is not None:
        failure_class, alternatives = _VIOLATION_CLASSES[report.violation]
    else:
        failure_class = FailureClass.POSTCONDITION_FAILURE
        alternatives = ()
    return FailureCertificate(
        task_goal=goal,
        failure_class=failure_class,
        alternative_classes=alternatives,
        failed_action=report.violated_action,
        violation=report.violation,
        violated_literal=report.violated_literal,
        violation_reason=report.violation_reason,
        grounding_failure_code=report.grounding_failure_code,
        platform_failure_code=platform_failure_code,
        truth=TruthSummary.from_grounding(grounding),
        causal_neighborhood=CausalNeighborhood.from_selection(selection, goal),
        executed_actions=tuple(report.executed),
        nogoods=tuple(n.detail for n in nogoods),
        **metadata,
    )


def certify_unsupported_capability(
    goal: tuple[Literal, ...],
    selection: Selection,
    missing: str,
    **metadata,
) -> FailureCertificate:
    """The platform lacks a required execution capability; repair must stop."""
    return FailureCertificate(
        task_goal=goal,
        failure_class=FailureClass.UNSUPPORTED_CAPABILITY,
        violation_reason=missing,
        causal_neighborhood=CausalNeighborhood.from_selection(selection, goal),
        **metadata,
    )


def certify_evaluator_binding_error(
    goal: tuple[Literal, ...],
    selection: Selection,
    missing: str,
    **metadata,
) -> FailureCertificate:
    """A predicate points at an evaluator absent from this embodiment.

    The binding is repairable when another registered evaluator implements
    the predicate.  If none does, the agent may select the explicitly allowed
    unsupported-capability alternative.
    """
    return FailureCertificate(
        task_goal=goal,
        failure_class=FailureClass.PREDICATE_IMPLEMENTATION_ERROR,
        alternative_classes=(FailureClass.UNSUPPORTED_CAPABILITY,),
        violation_reason=missing,
        causal_neighborhood=CausalNeighborhood.from_selection(selection, goal),
        **metadata,
    )


def _render_literals(literals: tuple[Literal, ...]) -> str:
    parts = []
    for literal in literals:
        atom = f"({literal.predicate} {' '.join(literal.arguments)})"
        parts.append(f"(not {atom})" if literal.negated else atom)
    return " and ".join(parts)


def _atom_key(atom: Literal) -> tuple:
    return (atom.predicate, atom.arguments)
