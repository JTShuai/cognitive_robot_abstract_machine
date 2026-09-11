"""
Predicate and operator schemas plus persistent symbol-library storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    from_json,
    to_json,
)
from typing_extensions import Any, Self

from resym.core.capabilities import (
    CapabilityContract,
    OperatorExecutionBinding,
)
from resym.core.grounding import PredicateGroundingPlan
from resym.core.predicate_refs import (
    PredicateImplementation,
    PredicateRef,
    TruthProcedureRef,
)
from resym.core.provenance import Provenance
from resym.core.types import SymbolType


@dataclass(frozen=True)
class Literal:
    """
    An optionally negated predicate applied to parameters or objects.
    """

    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", tuple(self.arguments))

    def substitute(self, binding: dict[str, str]) -> Literal:
        return Literal(
            self.predicate,
            tuple(binding[argument] for argument in self.arguments),
            self.negated,
        )


@dataclass(frozen=True)
class PredicateSymbol:
    """
    A typed predicate whose truth is computed, not asserted.
    """

    name: str
    parameter_types: tuple[SymbolType, ...]
    evaluator: str
    fluent: bool
    provenance: Provenance = Provenance()
    uid: str | None = None
    version: str = "1"
    truth_procedure_ref: TruthProcedureRef | None = None
    grounding_plan: PredicateGroundingPlan | None = None
    """
    Reviewed EQL factory binding; absent for legacy registered evaluators.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_types", tuple(self.parameter_types))
        if self.uid is None:
            object.__setattr__(self, "uid", PredicateRef.from_name(self.name).uid)
        if self.truth_procedure_ref is None:
            object.__setattr__(
                self,
                "truth_procedure_ref",
                (
                    TruthProcedureRef.query(self.name, self.grounding_plan.version)
                    if self.grounding_plan is not None
                    else TruthProcedureRef.registered(self.evaluator)
                ),
            )

    @property
    def ref(self) -> PredicateRef:
        return PredicateRef(self.uid, self.version, self.name)

    @property
    def implementation(self) -> PredicateImplementation:
        return PredicateImplementation(
            ref=self.truth_procedure_ref,
            evaluator_key=self.evaluator,
        )


@dataclass(frozen=True)
class Operator:
    """
    An action schema with typed parameters and STRIPS effects.
    """

    name: str
    parameters: tuple[tuple[str, SymbolType], ...]
    preconditions: tuple[Literal, ...]
    add_effects: tuple[Literal, ...]
    delete_effects: tuple[Literal, ...]
    execution_binding: OperatorExecutionBinding
    provenance: Provenance = Provenance()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameters", tuple(tuple(pair) for pair in self.parameters)
        )
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "add_effects", tuple(self.add_effects))
        object.__setattr__(self, "delete_effects", tuple(self.delete_effects))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(variable for variable, _ in self.parameters)


@dataclass
class SymbolLibrary(SubclassJSONSerializer):
    """
    Persistent predicates, operators, and capability contracts.
    """

    predicates: dict[str, PredicateSymbol] = field(default_factory=dict)
    operators: dict[str, Operator] = field(default_factory=dict)
    capability_contracts: dict[str, CapabilityContract] = field(default_factory=dict)

    @property
    def symbol_types(self) -> tuple[SymbolType, ...]:
        values = {
            symbol_type
            for predicate in self.predicates.values()
            for symbol_type in predicate.parameter_types
        }
        values.update(
            symbol_type
            for operator in self.operators.values()
            for _, symbol_type in operator.parameters
        )
        values.update(
            symbol_type
            for contract in self.capability_contracts.values()
            for role in contract.roles
            for symbol_type in role.accepted_symbol_types
        )
        return tuple(sorted(values))

    def add(self, symbol: PredicateSymbol | Operator) -> None:
        target = (
            self.predicates if isinstance(symbol, PredicateSymbol) else self.operators
        )
        if symbol.name in target:
            raise DuplicateSymbolError(symbol.name)
        target[symbol.name] = symbol

    def add_capability_contract(self, contract: CapabilityContract) -> None:
        if contract.uid in self.capability_contracts:
            raise DuplicateSymbolError(contract.uid)
        self.capability_contracts[contract.uid] = contract

    def to_json(self) -> dict:
        return {
            **super().to_json(),
            "predicates": [
                to_json(predicate) for predicate in self.predicates.values()
            ],
            "operators": [to_json(operator) for operator in self.operators.values()],
            "capability_contracts": [
                to_json(contract) for contract in self.capability_contracts.values()
            ],
        }

    @classmethod
    def _from_json(cls, data: dict, **kwargs: Any) -> Self:
        library = cls()
        for entry in data["predicates"]:
            library.add(from_json(entry))
        for entry in data["operators"]:
            library.add(from_json(entry))
        for entry in data.get("capability_contracts", ()):
            library.add_capability_contract(from_json(entry))
        return library

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.from_json(json.loads(path.read_text()))


class DuplicateSymbolError(Exception):
    """
    Raised when a symbol or contract is added under a name the library already holds.
    """

    def __init__(self, name: str):
        super().__init__(f"Symbol '{name}' is already in the library.")
