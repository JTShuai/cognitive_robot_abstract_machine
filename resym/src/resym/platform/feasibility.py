"""
Capability feasibility as predicate truth.

A feasibility grounding factory is derived from one reviewed capability contract: its
truth procedure asks the embodiment's feasibility oracle whether that capability can
currently be realized for the bound objects. The platform therefore carries no
predicate-specific truth code — which feasibility questions exist follows entirely from
the capability catalog produced at initialization.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from resym.core.capabilities import CapabilityContract
from resym.core.grounding import (
    GroundingFactoryOrigin,
    GroundingFactoryProcedure,
    GroundingFactoryRole,
    GroundingFactorySpec,
    GroundingFailure,
    GroundingFailureCode,
    text_checksum,
)
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import GroundedObject

FEASIBILITY_FACTORY_NAMESPACE = "resym:grounding/feasible/"
"""
Reserved identity prefix of capability-derived grounding factories.
"""

CAPABILITY_CATALOG_REVIEWER = "capability-catalog"
"""
Review authority recorded on factories derived from reviewed contracts.
"""


def feasibility_factory_uid(capability_uid: str) -> str:
    """
    Grounding-factory identity derived from one capability identity.
    """
    return FEASIBILITY_FACTORY_NAMESPACE + capability_uid.split(":", 1)[-1]


class CapabilityFeasibility(ABC):
    """
    Answers whether an embodiment can currently realize a capability for concrete
    objects.
    """

    @abstractmethod
    def feasible(
        self,
        capability_uid: str,
        arguments: tuple[GroundedObject, ...],
        context: EvaluationContext,
    ) -> bool:
        """
        Decide feasibility of one capability for one bound object tuple.

        Raises :class:`~resym.core.grounding.GroundingFailure` when the question cannot
        be answered with a Boolean.
        """


def capability_feasibility_factories(
    contracts: Iterable[CapabilityContract],
    implementations: Mapping[str, Callable] | None = None,
) -> tuple[tuple[GroundingFactorySpec, GroundingFactoryProcedure], ...]:
    """
    Derive factories only for contracts with a concrete reviewed implementation.

    Factory roles are the contract's required object roles in declaration order;
    constant and optional roles take no part in feasibility grounding.
    """
    entries = []
    implementations = implementations or {}
    for contract in contracts:
        implementation = implementations.get(contract.uid)
        if implementation is None:
            continue
        roles = tuple(
            GroundingFactoryRole(role.name, role.accepted_symbol_types[0])
            for role in contract.roles
            if role.required and role.accepted_symbol_types
        )
        procedure = _feasibility_procedure(contract.uid)
        source_file = inspect.getsourcefile(implementation)
        if source_file is None:
            raise ValueError(
                f"Cannot locate feasibility source for capability '{contract.uid}'."
            )
        implementation_checksum = text_checksum(
            Path(source_file).read_text(encoding="utf-8")
        )
        checksum = text_checksum(
            inspect.getsource(_feasibility_procedure)
            + contract.uid
            + contract.version
            + implementation_checksum
        )
        specification = GroundingFactorySpec(
            uid=feasibility_factory_uid(contract.uid),
            semantic_name=f"feasible:{contract.label}",
            implementation_ref=f"{__name__}:feasible/{contract.uid}",
            implementation_checksum=checksum,
            roles=roles,
            origin=GroundingFactoryOrigin.PLATFORM,
            reviewed_by=CAPABILITY_CATALOG_REVIEWER,
            approved_at=f"capability-contract@{contract.version}",
            active_revision_id=f"r-{checksum[:12]}",
        )
        entries.append((specification, procedure))
    return tuple(entries)


def _feasibility_procedure(capability_uid: str) -> GroundingFactoryProcedure:
    """
    Truth procedure delegating one capability's feasibility to the oracle.
    """

    def evaluate(
        context: EvaluationContext,
        universe,
        arguments: tuple[GroundedObject, ...],
        parameters,
    ) -> bool:
        oracle = context.capability_feasibility
        if oracle is None:
            raise GroundingFailure(
                GroundingFailureCode.UNSUPPORTED_QUERY,
                f"no feasibility oracle is attached for capability "
                f"'{capability_uid}'",
            )
        return bool(oracle.feasible(capability_uid, arguments, context))

    return evaluate
