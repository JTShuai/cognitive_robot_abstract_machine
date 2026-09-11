"""
Goal-regression selection: which library symbols matter for this goal.

Classic backward closure over the operator effect-to-precondition links; pure symbol-
table reasoning, no geometry, no learned components. This keeps the expensive grounding
step from evaluating the whole library against the whole scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from resym.core.model import (
    Literal,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
)


@dataclass
class Selection:
    """
    The goal-relevant slice of the library.
    """

    predicates: dict[str, PredicateSymbol] = field(default_factory=dict)
    """
    Relevant predicate symbols by name.
    """

    operators: dict[str, Operator] = field(default_factory=dict)
    """
    Relevant operator schemas by name.
    """


class UnknownPredicateError(Exception):
    """
    Raised when a goal or operator references a predicate the library does not hold.
    """

    def __init__(self, name: str):
        super().__init__(f"Predicate '{name}' is not in the library.")


def select_for_goal(library: SymbolLibrary, goal: tuple[Literal, ...]) -> Selection:
    """
    Regress from the goal literals to the operators that can establish them.

    Positive conditions are established by add effects; negative conditions
    are established by delete effects. Keeping the requested polarity in the
    work list matters: selecting an operator that *adds* ``opened`` for a goal
    of ``not opened`` is both unnecessary and, for a small task slice, wrong.
    """
    selection = Selection()
    open_conditions = [(literal.predicate, not literal.negated) for literal in goal]
    expanded_conditions: set[tuple[str, bool]] = set()
    while open_conditions:
        predicate_name, required_true = open_conditions.pop()
        condition = (predicate_name, required_true)
        if condition in expanded_conditions:
            continue
        expanded_conditions.add(condition)
        if predicate_name not in library.predicates:
            raise UnknownPredicateError(predicate_name)
        selection.predicates[predicate_name] = library.predicates[predicate_name]
        for operator in library.operators.values():
            if not _achieves(operator, predicate_name, required_true):
                continue
            already_selected = operator.name in selection.operators
            selection.operators[operator.name] = operator
            if not already_selected:
                for precondition in operator.preconditions:
                    open_conditions.append(
                        (precondition.predicate, not precondition.negated)
                    )
            for effect in operator.add_effects + operator.delete_effects:
                if effect.predicate not in library.predicates:
                    raise UnknownPredicateError(effect.predicate)
                selection.predicates[effect.predicate] = library.predicates[
                    effect.predicate
                ]
    return selection


def _achieves(operator: Operator, predicate_name: str, required_true: bool) -> bool:
    effects = operator.add_effects if required_true else operator.delete_effects
    return any(effect.predicate == predicate_name for effect in effects)
