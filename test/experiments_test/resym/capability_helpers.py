"""
Small constructors for capability-focused experiment tests.
"""

from __future__ import annotations

from resym.core.model import (
    CapabilityContract,
    CapabilityRef,
    CapabilityRole,
    OperatorExecutionBinding,
    RoleBinding,
    SymbolType,
)


def capability_contract(
    uid: str,
    roles: tuple[tuple[str, SymbolType], ...],
    effects: tuple[str, ...],
    *,
    constants: tuple[tuple[str, tuple[str, ...]], ...] = (),
    version: str = "1",
) -> CapabilityContract:
    return CapabilityContract(
        uid=uid,
        label=uid,
        version=version,
        roles=tuple(
            CapabilityRole(name, accepted_symbol_types=(symbol_type,))
            for name, symbol_type in roles
        )
        + tuple(
            CapabilityRole(name, allowed_values=values) for name, values in constants
        ),
        success_relation="test relation",
        verifiable_effects=effects,
    )


def execution_binding(
    uid: str,
    parameters: tuple[tuple[str, str], ...],
    *,
    constants: tuple[tuple[str, str], ...] = (),
    version: str = "1",
) -> OperatorExecutionBinding:
    return OperatorExecutionBinding(
        CapabilityRef(uid, version),
        tuple(
            (role, RoleBinding.parameter(parameter)) for role, parameter in parameters
        )
        + tuple((role, RoleBinding.constant(value)) for role, value in constants),
    )
