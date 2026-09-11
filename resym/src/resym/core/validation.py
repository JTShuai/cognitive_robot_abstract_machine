"""
Deterministic structural validation for planning-model operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from resym.core.capabilities import BindingSource
from resym.core.symbols import Operator, SymbolLibrary
from resym.core.types import SymbolType, is_symbol_subtype


class ModelIssueKind(StrEnum):
    """
    Structural model defects that do not require environment evidence.
    """

    OPERATOR_SIGNATURE = "operator_signature_error"


@dataclass(frozen=True)
class ModelIssue:
    kind: ModelIssueKind
    operator: str
    detail: str

    def render(self) -> str:
        return f"operator '{self.operator}': {self.detail}"


class InvalidSymbolLibraryError(Exception):
    """
    A goal-relevant operator is structurally inconsistent.
    """

    def __init__(self, issues: tuple[ModelIssue, ...], selection) -> None:
        self.issues = issues
        self.selection = selection
        super().__init__("; ".join(issue.render() for issue in issues))


def selection_model_issues(library: SymbolLibrary, selection) -> tuple[ModelIssue, ...]:
    """
    Return structural issues in the goal-relevant operator slice.
    """
    issues: list[ModelIssue] = []
    for operator in selection.operators.values():
        issues.extend(_operator_signature_issues(operator, library))
    return tuple(issues)


def _operator_signature_issues(
    operator: Operator, library: SymbolLibrary
) -> list[ModelIssue]:
    issues: list[ModelIssue] = []
    parameter_types = dict(operator.parameters)
    if len(parameter_types) != len(operator.parameters):
        issues.append(
            ModelIssue(
                ModelIssueKind.OPERATOR_SIGNATURE,
                operator.name,
                "declares a parameter more than once",
            )
        )

    for literal in (
        *operator.preconditions,
        *operator.add_effects,
        *operator.delete_effects,
    ):
        predicate = library.predicates.get(literal.predicate)
        if predicate is None:
            issues.append(
                ModelIssue(
                    ModelIssueKind.OPERATOR_SIGNATURE,
                    operator.name,
                    f"references unknown predicate '{literal.predicate}'",
                )
            )
            continue
        if len(literal.arguments) != len(predicate.parameter_types):
            issues.append(
                ModelIssue(
                    ModelIssueKind.OPERATOR_SIGNATURE,
                    operator.name,
                    f"'{literal.predicate}' takes {len(predicate.parameter_types)} "
                    f"arguments, got {len(literal.arguments)}",
                )
            )
            continue
        for argument, expected in zip(literal.arguments, predicate.parameter_types):
            actual = parameter_types.get(argument)
            if actual is None:
                issues.append(
                    ModelIssue(
                        ModelIssueKind.OPERATOR_SIGNATURE,
                        operator.name,
                        f"literal argument '{argument}' is not an operator parameter",
                    )
                )
                continue
            compatible = _compatible(actual, expected)
            if compatible is None:
                issues.append(
                    ModelIssue(
                        ModelIssueKind.OPERATOR_SIGNATURE,
                        operator.name,
                        f"cannot resolve type '{actual.python_type_ref}' or "
                        f"'{expected.python_type_ref}' of variable '{argument}'",
                    )
                )
            elif not compatible:
                issues.append(
                    ModelIssue(
                        ModelIssueKind.OPERATOR_SIGNATURE,
                        operator.name,
                        f"variable '{argument}' is '{actual.python_type_ref}', "
                        f"'{literal.predicate}' expects "
                        f"'{expected.python_type_ref}'",
                    )
                )

    contract = library.capability_contracts.get(
        operator.execution_binding.capability_ref.uid
    )
    if contract is not None:
        for role_name, binding in operator.execution_binding.role_bindings:
            role = contract.role_map.get(role_name)
            if (
                role is None
                or binding.source is not BindingSource.PARAMETER
                or not role.accepted_symbol_types
            ):
                continue
            actual = parameter_types.get(binding.value)
            if actual is None:
                continue
            checks = [
                _compatible(actual, expected) for expected in role.accepted_symbol_types
            ]
            if any(check is True for check in checks):
                continue
            expected = ", ".join(
                item.python_type_ref for item in role.accepted_symbol_types
            )
            if any(check is None for check in checks):
                detail = (
                    f"cannot resolve type '{actual.python_type_ref}' or an "
                    f"accepted type of capability role '{role_name}' "
                    f"({expected})"
                )
            else:
                detail = (
                    f"variable '{binding.value}' is "
                    f"'{actual.python_type_ref}', capability role "
                    f"'{role_name}' expects {expected}"
                )
            issues.append(
                ModelIssue(ModelIssueKind.OPERATOR_SIGNATURE, operator.name, detail)
            )
    return issues


def _compatible(actual: SymbolType, expected: SymbolType) -> bool | None:
    """
    Subtype check that reports an unresolvable type reference as ``None``.
    """
    try:
        return is_symbol_subtype(actual, expected)
    except (ImportError, AttributeError, TypeError):
        return None
