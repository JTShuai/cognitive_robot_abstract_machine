"""
Goal regression over the library.
"""

from __future__ import annotations

import pytest

from resym.core.model import (
    Literal,
    Operator,
)
from resym.platform.capabilities import ARTICULATION_CAPABILITY_UID
from resym.planning.selection import (
    UnknownPredicateError,
    select_for_goal,
)
from .capability_helpers import execution_binding

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)



def test_regression_reaches_full_closure(library):
    selection = select_for_goal(library, (Literal("opened", ("d",)),))
    assert set(selection.operators) == {"navigate", "open-drawer"}
    assert set(selection.predicates) == {
        "opened",
        "closed",
        "handle-of",
        "ready-to-open",
        "openable",
    }


def test_regression_without_achievers_selects_goal_only(library):
    selection = select_for_goal(library, (Literal("handle-of", ("h", "d")),))
    assert set(selection.predicates) == {"handle-of"}
    assert not selection.operators


def test_unknown_goal_predicate_raises(library):
    with pytest.raises(UnknownPredicateError):
        select_for_goal(library, (Literal("levitating", ("d",)),))


def test_negative_goal_selects_deleting_operator(library):
    library.add(
        Operator(
            name="close-by-negation",
            parameters=(("d", DRAWER_TYPE),),
            preconditions=(),
            add_effects=(),
            delete_effects=(Literal("opened", ("d",)),),
            execution_binding=execution_binding(
                ARTICULATION_CAPABILITY_UID,
                (("patient", "d"),),
                constants=(("target_state", "CLOSED"),),
            ),
        )
    )

    selection = select_for_goal(library, (Literal("opened", ("d",), negated=True),))

    assert set(selection.operators) == {"close-by-negation"}
    assert set(selection.predicates) == {"opened"}
