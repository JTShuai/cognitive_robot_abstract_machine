"""
Executable adaptation: alignment validation, the six issue kinds, patch complexity, and
the deterministic name-matching heuristic.

Host-runnable, stdlib-only.
"""

from __future__ import annotations

import json

import pytest
from krrood.adapters.json_serializer import from_json, to_json
from krrood.adapters.exceptions import MissingTypeError

from resym.repair.patch import (
    AdaptationIssueKind,
    ModelPatch,
    OperatorAlignment,
    PredicateBinding,
    adapt_operator,
    contract_scorer_for,
    suggest_alignment,
)
from resym.knowledge.corpus import load_fragment
from resym.core.model import (
    Literal,
    Operator,
    PredicateSymbol,
    Provenance,
    SymbolLibrary,
)
from .capability_helpers import capability_contract, execution_binding

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


KNOWN_EVALUATORS = frozenset({"drawer_closed", "drawer_opened"})
CAPABILITY_UID = "test:DrawerStateChange"
AVAILABLE_CAPABILITIES = frozenset({CAPABILITY_UID})


def close_binding():
    return execution_binding(CAPABILITY_UID, (("patient", "d"),))


def local_library() -> SymbolLibrary:
    library = SymbolLibrary()
    for name in ("drawer-open", "drawer-closed"):
        library.add(
            PredicateSymbol(
                name=name,
                parameter_types=(DRAWER_TYPE,),
                evaluator="drawer_opened" if name == "drawer-open" else "drawer_closed",
                fluent=True,
            )
        )
    library.add_capability_contract(
        capability_contract(
            CAPABILITY_UID,
            (("patient", DRAWER_TYPE),),
            ("drawer-open", "drawer-closed"),
        )
    )
    return library


@pytest.fixture()
def close_fragment(tmp_path):
    record = {
        "predicates": {
            "(drawer_open ?d)": "the drawer ?d is open",
            "(drawer_closed ?d)": "the drawer ?d is closed",
        },
        "operators": {
            "close_drawer": (
                "(:action close_drawer :parameters (?d) "
                ":precondition (drawer_open ?d) "
                ":effect (and (drawer_closed ?d) (not (drawer_open ?d))))"
            )
        },
    }
    path = tmp_path / "9" / "episode_9" / "atomic_domain.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return load_fragment(path)


def full_alignment() -> OperatorAlignment:
    return OperatorAlignment(
        fragment_operator="close_drawer",
        local_name="close-drawer",
        variable_types={"?d": DRAWER_TYPE},
        bindings=(
            PredicateBinding("drawer_open", "drawer-open"),
            PredicateBinding("drawer_closed", "drawer-closed"),
        ),
        execution_binding=close_binding(),
    )


def test_valid_alignment_yields_typed_patch(close_fragment):
    result = adapt_operator(
        close_fragment,
        full_alignment(),
        local_library(),
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    assert result.succeeded, [i.render() for i in result.issues]
    (operator,) = result.patch.operators
    assert operator.name == "close-drawer"
    assert operator.parameters == (("d", DRAWER_TYPE),)
    assert operator.preconditions == (Literal("drawer-open", ("d",)),)
    assert operator.add_effects == (Literal("drawer-closed", ("d",)),)
    assert operator.delete_effects == (Literal("drawer-open", ("d",)),)
    assert operator.execution_binding.capability_ref.uid == CAPABILITY_UID
    assert result.patch.predicates == ()  # everything mapped onto existing
    edge_targets = {e.element for e in result.patch.edges}
    assert "close-drawer" in edge_targets


def test_missing_binding_is_reported_and_left_unresolved(close_fragment):
    alignment = OperatorAlignment(
        fragment_operator="close_drawer",
        local_name="close-drawer",
        variable_types={"?d": DRAWER_TYPE},
        bindings=(PredicateBinding("drawer_open", "drawer-open"),),
        execution_binding=close_binding(),
    )
    result = adapt_operator(
        close_fragment,
        alignment,
        local_library(),
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    assert not result.succeeded
    kinds = {i.kind for i in result.issues}
    assert AdaptationIssueKind.MISSING_PREDICATE_IMPLEMENTATION in kinds


def test_arity_mismatch_is_a_type_issue(close_fragment):
    library = local_library()
    library.add(
        PredicateSymbol(
            name="binary-open",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            evaluator="drawer_opened",
            fluent=True,
        )
    )
    alignment = OperatorAlignment(
        fragment_operator="close_drawer",
        local_name="close-drawer",
        variable_types={"?d": DRAWER_TYPE},
        bindings=(
            PredicateBinding("drawer_open", "binary-open"),  # arity 2 vs usage 1
            PredicateBinding("drawer_closed", "drawer-closed"),
        ),
        execution_binding=close_binding(),
    )
    result = adapt_operator(
        close_fragment, alignment, library, KNOWN_EVALUATORS, AVAILABLE_CAPABILITIES
    )
    assert any(i.kind is AdaptationIssueKind.TYPE_MISMATCH for i in result.issues)


def test_unavailable_capability_is_unsupported_embodiment(close_fragment):
    alignment = full_alignment()
    result = adapt_operator(
        close_fragment,
        alignment,
        local_library(),
        KNOWN_EVALUATORS,
        available_capabilities=frozenset(),
    )
    assert any(
        i.kind is AdaptationIssueKind.UNSUPPORTED_EMBODIMENT for i in result.issues
    )


def test_undeclared_contract_is_reported(close_fragment):
    library = local_library()
    del library.capability_contracts[CAPABILITY_UID]
    result = adapt_operator(
        close_fragment,
        full_alignment(),
        library,
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    assert any(
        i.kind is AdaptationIssueKind.MISSING_CAPABILITY_CONTRACT for i in result.issues
    )


def test_effect_beyond_contract_is_a_conflict(close_fragment):
    library = local_library()
    library.capability_contracts[CAPABILITY_UID] = capability_contract(
        CAPABILITY_UID,
        (("patient", DRAWER_TYPE),),
        ("drawer-closed",),
    )
    result = adapt_operator(
        close_fragment,
        full_alignment(),
        library,
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    assert any(
        i.kind is AdaptationIssueKind.EFFECT_CONTRACT_CONFLICT for i in result.issues
    )


def test_new_predicate_uses_a_reviewed_query(close_fragment):
    library = local_library()
    del library.predicates["drawer-open"]
    alignment = OperatorAlignment(
        fragment_operator="close_drawer",
        local_name="close-drawer",
        variable_types={"?d": DRAWER_TYPE},
        bindings=(
            PredicateBinding(
                "drawer_open",
                "drawer-open",
                parameter_types=(DRAWER_TYPE,),
                evaluator="drawer_opened",
            ),
            PredicateBinding("drawer_closed", "drawer-closed"),
        ),
        execution_binding=close_binding(),
    )
    result = adapt_operator(
        close_fragment, alignment, library, KNOWN_EVALUATORS, AVAILABLE_CAPABILITIES
    )
    assert result.succeeded, [i.render() for i in result.issues]
    (predicate,) = result.patch.predicates
    assert predicate.name == "drawer-open"
    assert predicate.evaluator == "drawer_opened"


def test_creating_over_an_existing_predicate_is_a_conflict(close_fragment):
    alignment = OperatorAlignment(
        fragment_operator="close_drawer",
        local_name="close-drawer",
        variable_types={"?d": DRAWER_TYPE},
        bindings=(
            PredicateBinding(
                "drawer_open",
                "drawer-open",
                parameter_types=(DRAWER_TYPE,),
                evaluator="drawer_opened",  # creates new, but name exists
            ),
            PredicateBinding("drawer_closed", "drawer-closed"),
        ),
        execution_binding=close_binding(),
    )
    result = adapt_operator(
        close_fragment,
        alignment,
        local_library(),
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    assert any(
        i.kind is AdaptationIssueKind.DUPLICATE_OR_CONFLICTING_SYMBOL
        for i in result.issues
    )


def test_suggest_alignment_reconstructs_the_mapping(close_fragment):
    alignment = suggest_alignment(close_fragment, "close_drawer", local_library())
    assert alignment is not None
    assert alignment.execution_binding.capability_ref.uid == CAPABILITY_UID
    assert alignment.variable_types == {"?d": DRAWER_TYPE}
    mapped = {b.fragment_predicate: b.local_name for b in alignment.bindings}
    assert mapped == {
        "drawer_open": "drawer-open",
        "drawer_closed": "drawer-closed",
    }
    result = adapt_operator(
        close_fragment,
        alignment,
        local_library(),
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    assert result.succeeded


def test_patch_complexity_is_lexicographic(close_fragment):
    library = local_library()
    result = adapt_operator(
        close_fragment,
        full_alignment(),
        library,
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    complexity = result.patch.complexity(library)
    assert complexity.as_tuple() == (0, 1, 0, 0)  # one new operator, no mods


def test_apply_to_never_mutates_the_base(close_fragment):
    library = local_library()
    result = adapt_operator(
        close_fragment,
        full_alignment(),
        library,
        KNOWN_EVALUATORS,
        AVAILABLE_CAPABILITIES,
    )
    candidate = result.patch.apply_to(library)
    assert "close-drawer" in candidate.operators
    assert "close-drawer" not in library.operators


def test_patch_json_is_complete_replayable_and_checksummed(close_fragment):
    provenance = Provenance(
        source="curation", proposal_backend="test", retrieved_ids=("fragment-1",)
    )
    predicate = PredicateSymbol(
        name="thresholded",
        parameter_types=(DRAWER_TYPE,),
        evaluator="drawer_opened",
        fluent=True,
        provenance=provenance,
    )
    operator = Operator(
        name="mark-thresholded",
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(),
        add_effects=(Literal("thresholded", ("d",)),),
        delete_effects=(),
        execution_binding=close_binding(),
        provenance=provenance,
    )
    patch = ModelPatch(
        predicates=(predicate,),
        operators=(operator,),
        capability_contracts=(
            capability_contract(
                CAPABILITY_UID,
                (("patient", DRAWER_TYPE),),
                ("thresholded",),
            ),
        ),
        rationale="audit me",
    )
    restored = from_json(to_json(patch))
    assert restored == patch
    assert to_json(patch)["operators"][0]["add_effects"][0]["predicate"] == (
        "thresholded"
    )
    with pytest.raises(MissingTypeError):
        from_json({"predicates": ["thresholded"]})


def test_contract_scorer_prefers_mappable_fragments(close_fragment, tmp_path):
    stove_record = {
        "predicates": {"(stove_on ?s)": "the stove is on"},
        "operators": {
            "turn_on_stove": (
                "(:action turn_on_stove :parameters (?s) "
                ":precondition (not (stove_on ?s)) :effect (stove_on ?s))"
            )
        },
    }
    path = tmp_path / "8" / "episode_8" / "atomic_domain.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stove_record))
    stove_fragment = load_fragment(path)

    scorer = contract_scorer_for(local_library())
    assert scorer(None, close_fragment) > scorer(None, stove_fragment)
