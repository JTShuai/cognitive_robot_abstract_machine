"""
Compute a complete binary state for one task-scoped object universe.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from resym.core.grounding import GroundingFailure, GroundingFailureCode
from resym.platform.evaluators import (
    EvaluationContext,
    resolve_evaluator,
)
from resym.core.model import Literal, PredicateSymbol
from resym.planning.selection import Selection
from resym.platform.universe import GroundedObject, ObjectUniverse


@dataclass
class GroundingResult:
    """
    The computed binary state of every evaluated ground atom.
    """

    true_atoms: set[Literal] = field(default_factory=set)
    """
    Ground literals proven true against the current world.
    """

    false_atoms: set[Literal] = field(default_factory=set)
    """
    Ground literals proven false against the current world.
    """

    evaluation_count: int = 0
    """
    How many predicate evaluations the grounding performed (the |D|^arity cost).
    """

    def value_of(self, atom: Literal) -> bool:
        """
        Return the Boolean value of one evaluated ground atom.
        """
        if atom in self.true_atoms:
            return True
        if atom in self.false_atoms:
            return False
        raise KeyError(f"Atom was not grounded: {atom}")


def ground(
    selection: Selection,
    universe: ObjectUniverse,
    context: EvaluationContext,
) -> GroundingResult:
    """
    Evaluate every selected predicate over the typed active domain.
    """
    result = GroundingResult()
    for predicate in selection.predicates.values():
        for arguments in _typed_tuples(predicate, universe):
            result.evaluation_count += 1
            atom = Literal(
                predicate=predicate.name,
                arguments=tuple(argument.name for argument in arguments),
            )
            try:
                value = evaluate_predicate(predicate, arguments, universe, context)
            except GroundingFailure as error:
                raise error.for_atom(atom) from error
            if value:
                result.true_atoms.add(atom)
            else:
                result.false_atoms.add(atom)
    return result


def evaluate_predicate(
    predicate: PredicateSymbol,
    arguments: tuple[GroundedObject, ...],
    universe: ObjectUniverse,
    context: EvaluationContext,
) -> bool:
    """
    Run one truth procedure for one ground atom against the current world through its
    single, versioned implementation reference.
    """
    procedure = resolve_evaluator(predicate.implementation.evaluator_key)
    value = procedure(context, universe, arguments)
    if not isinstance(value, bool):
        raise GroundingFailure(
            GroundingFailureCode.QUERY_ERROR,
            f"predicate query '{predicate.implementation.evaluator_key}' returned "
            f"{type(value).__name__}, expected bool",
        )
    return value


def _typed_tuples(
    predicate: PredicateSymbol, universe: ObjectUniverse
) -> itertools.product:
    domains = [
        universe.of_type(symbol_type) for symbol_type in predicate.parameter_types
    ]
    return itertools.product(*domains)
