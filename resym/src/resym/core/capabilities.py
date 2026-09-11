"""
Capability contracts, operator bindings, and grounded execution requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import TYPE_CHECKING, Mapping

from resym.core.predicate_refs import PredicateRef
from resym.core.provenance import OntologyAlignment
from resym.core.types import SymbolType, is_symbol_subtype

if TYPE_CHECKING:
    from resym.core.symbols import Operator, PredicateSymbol


@dataclass(frozen=True)
class CapabilityRole:
    """
    One object-valued or constant-valued role in a capability contract.
    """

    name: str
    required: bool = True
    allowed_values: tuple[str, ...] = ()
    accepted_symbol_types: tuple[SymbolType, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_values", tuple(self.allowed_values))
        object.__setattr__(
            self, "accepted_symbol_types", tuple(self.accepted_symbol_types)
        )
        if bool(self.allowed_values) == bool(self.accepted_symbol_types):
            raise ValueError(
                f"capability role '{self.name}' must declare exactly one of "
                "allowed_values or accepted_symbol_types"
            )


@dataclass(frozen=True)
class CapabilityRef:
    uid: str
    version: str = "1"


@dataclass(frozen=True)
class CapabilityContract:
    """
    Versioned, hardware-independent contract for an execution goal.
    """

    uid: str
    label: str
    roles: tuple[CapabilityRole, ...]
    success_relation: str
    verifiable_effects: tuple[PredicateRef | str, ...]
    version: str = "1"
    ontology_alignment: OntologyAlignment = OntologyAlignment()
    effect_role_values: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(
            self,
            "effect_role_values",
            tuple(tuple(item) for item in self.effect_role_values),
        )
        object.__setattr__(
            self,
            "verifiable_effects",
            tuple(
                (
                    effect
                    if isinstance(effect, PredicateRef)
                    else PredicateRef.from_name(effect)
                )
                for effect in self.verifiable_effects
            ),
        )
        role_names = [role.name for role in self.roles]
        if len(role_names) != len(set(role_names)):
            raise ValueError(f"capability '{self.uid}' declares duplicate roles")
        for effect, role_name, value in self.effect_role_values:
            if effect not in self.verifiable_effect_names:
                raise ValueError(
                    f"capability '{self.uid}' maps unknown effect '{effect}'"
                )
            role = self.role_map.get(role_name)
            if role is None:
                raise ValueError(
                    f"capability '{self.uid}' maps unknown role '{role_name}'"
                )
            if value not in role.allowed_values:
                raise ValueError(
                    f"capability '{self.uid}' maps unsupported value '{value}' "
                    f"for role '{role_name}'"
                )

    @property
    def ref(self) -> CapabilityRef:
        return CapabilityRef(self.uid, self.version)

    @property
    def role_map(self) -> dict[str, CapabilityRole]:
        return {role.name: role for role in self.roles}

    @property
    def verifiable_effect_names(self) -> tuple[str, ...]:
        """
        Task-local labels used when constructing PDDL literals.
        """
        return tuple(
            effect.local_name or effect.uid.rsplit("/", 1)[-1]
            for effect in self.verifiable_effects
        )

    def constant_role_value(
        self,
        role_name: str,
        effects: set[PredicateRef | str],
    ) -> str | None:
        requested_refs = {
            (
                effect
                if isinstance(effect, PredicateRef)
                else PredicateRef.from_name(effect)
            )
            for effect in effects
        }
        contract_refs = {
            effect.local_name or effect.uid.rsplit("/", 1)[-1]: effect
            for effect in self.verifiable_effects
        }
        values = {
            value
            for effect, role, value in self.effect_role_values
            if role == role_name
            and contract_refs.get(effect, PredicateRef.from_name(effect))
            in requested_refs
        }
        return next(iter(values)) if len(values) == 1 else None


class BindingSource(Enum):
    PARAMETER = "parameter"
    CONSTANT = "constant"


@dataclass(frozen=True)
class RoleBinding:
    source: BindingSource
    value: str

    @classmethod
    def parameter(cls, name: str) -> RoleBinding:
        return cls(BindingSource.PARAMETER, name)

    @classmethod
    def constant(cls, value: str) -> RoleBinding:
        return cls(BindingSource.CONSTANT, value)


@dataclass(frozen=True)
class OperatorExecutionBinding:
    capability_ref: CapabilityRef
    role_bindings: tuple[tuple[str, RoleBinding], ...]
    proposal_source: str = "seed"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role_bindings",
            tuple(tuple(pair) for pair in self.role_bindings),
        )
        role_names = [role for role, _ in self.role_bindings]
        if len(role_names) != len(set(role_names)):
            raise ValueError(
                f"binding for '{self.capability_ref.uid}' maps a role more than once"
            )

    @property
    def role_map(self) -> dict[str, RoleBinding]:
        return dict(self.role_bindings)

    def resolve(self, parameters: dict[str, str]) -> ExecutionRequest:
        return ExecutionRequest(
            self.capability_ref,
            tuple(
                (
                    role,
                    (
                        parameters[binding.value]
                        if binding.source is BindingSource.PARAMETER
                        else binding.value
                    ),
                )
                for role, binding in self.role_bindings
            ),
        )


@dataclass(frozen=True)
class ExecutionRequest:
    capability_ref: CapabilityRef
    arguments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        roles = [role for role, _ in self.arguments]
        if len(roles) != len(set(roles)):
            raise ValueError(
                f"request for '{self.capability_ref.uid}' repeats an argument role"
            )

    @property
    def argument_map(self) -> dict[str, str]:
        return dict(self.arguments)

    def argument(self, role: str) -> str:
        return self.argument_map[role]


def contract_violations(
    operator: Operator,
    contract: CapabilityContract,
    predicates: Mapping[str, PredicateSymbol] | None = None,
) -> list[str]:
    """
    Return deterministic binding and effect-contract inconsistencies.
    """
    violations: list[str] = []
    binding = operator.execution_binding
    if binding.capability_ref != contract.ref:
        violations.append(
            f"operator '{operator.name}' references capability "
            f"'{binding.capability_ref.uid}' v{binding.capability_ref.version}, "
            f"not contract '{contract.uid}' v{contract.version}"
        )
        return violations
    declared_roles = contract.role_map
    bound_roles = binding.role_map
    for role in contract.roles:
        if role.required and role.name not in bound_roles:
            violations.append(
                f"operator '{operator.name}' does not bind required capability role '{role.name}'"
            )
    for role_name, role_binding in bound_roles.items():
        role = declared_roles.get(role_name)
        if role is None:
            violations.append(
                f"operator '{operator.name}' binds unknown capability role '{role_name}'"
            )
            continue
        if role_binding.source is BindingSource.PARAMETER:
            actual = dict(operator.parameters).get(role_binding.value)
            if actual is None:
                violations.append(
                    f"operator '{operator.name}' binds role '{role_name}' to unknown parameter '{role_binding.value}'"
                )
            elif role.accepted_symbol_types and not any(
                is_symbol_subtype(actual, accepted)
                for accepted in role.accepted_symbol_types
            ):
                expected = ", ".join(
                    item.python_type_ref for item in role.accepted_symbol_types
                )
                violations.append(
                    f"operator '{operator.name}' binds {actual.python_type_ref} parameter "
                    f"'{role_binding.value}' to {expected} role '{role_name}'"
                )
        elif not role.allowed_values:
            violations.append(
                f"operator '{operator.name}' binds untyped constant "
                f"'{role_binding.value}' to object role '{role_name}'"
            )
        elif role_binding.value not in role.allowed_values:
            violations.append(
                f"operator '{operator.name}' binds unsupported value "
                f"'{role_binding.value}' to role '{role_name}'"
            )
    allowed = set(contract.verifiable_effects)
    allowed_names = set(contract.verifiable_effect_names)
    violations.extend(
        f"effect '{literal.predicate}' of operator '{operator.name}' is not "
        f"verifiable through capability '{contract.uid}' "
        f"(contract allows: {', '.join(sorted(allowed_names)) or 'nothing'})"
        for literal in operator.add_effects + operator.delete_effects
        if (
            predicates[literal.predicate].ref
            if predicates is not None and literal.predicate in predicates
            else PredicateRef.from_name(literal.predicate)
        )
        not in allowed
    )
    return violations
