"""
Retrieval over a synthetic corpus: BM25 relevance, structural rerank, determinism, and a
smoke test against the frozen real release.

Host-runnable, stdlib-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resym.knowledge.corpus import DomainFragment
from resym.knowledge.retrieval import (
    FragmentIndex,
    RetrievalConfig,
    RetrievalQuery,
)

RELEASE = Path(__file__).resolve().parent.parent / "corpus_release" / "r1"


def fragment(fragment_id: str, predicates: dict[str, str], operators: dict[str, str]):
    from resym.knowledge.corpus import (
        _parse_operator,
        _parse_predicate,
    )

    return DomainFragment(
        fragment_id=fragment_id,
        predicates=tuple(
            _parse_predicate(fragment_id, s, g) for s, g in predicates.items()
        ),
        operators=tuple(
            _parse_operator(fragment_id, n, t) for n, t in operators.items()
        ),
        source_path=f"/synthetic/{fragment_id}",
    )


DRAWER = fragment(
    "1/1",
    {
        "(drawer_open ?d)": "the drawer ?d is open",
        "(drawer_closed ?d)": "the drawer ?d is closed",
    },
    {
        "close_drawer": (
            "(:action close_drawer :parameters (?d) "
            ":precondition (drawer_open ?d) "
            ":effect (and (drawer_closed ?d) (not (drawer_open ?d))))"
        )
    },
)

STOVE = fragment(
    "2/1",
    {"(stove_on ?s)": "the stove ?s burner is on"},
    {
        "turn_on_stove": (
            "(:action turn_on_stove :parameters (?s) "
            ":precondition (not (stove_on ?s)) :effect (stove_on ?s))"
        )
    },
)

PICK = fragment(
    "3/1",
    {
        "(holding ?o)": "the gripper holds object ?o",
        "(on_table ?o ?t)": "object ?o rests on table ?t",
    },
    {
        "pick_object": (
            "(:action pick_object :parameters (?o ?t) "
            ":precondition (on_table ?o ?t) "
            ":effect (and (holding ?o) (not (on_table ?o ?t))))"
        )
    },
)


@pytest.fixture()
def index():
    return FragmentIndex([DRAWER, STOVE, PICK])


def test_lexically_relevant_fragment_ranks_first(index):
    query = RetrievalQuery(text="close the drawer", goal_predicates=("closed",))
    hits = index.retrieve(query, top_k=3)
    assert hits[0].fragment_id == "1/1"


def test_causal_overlap_lifts_structurally_related_fragments(index):
    """
    A query whose lexical surface is neutral but whose causal neighborhood names
    holding/on predicates must prefer the pick fragment.
    """
    query = RetrievalQuery(
        text="capability gap",
        causal_predicates=("holding", "on-table"),
    )
    hits = index.retrieve(query, top_k=3)
    assert hits[0].fragment_id == "3/1"
    assert hits[0].causal_score > hits[1].causal_score


def test_score_decomposition_is_reported(index):
    query = RetrievalQuery(
        text="drawer",
        goal_predicates=("drawer-closed",),
        argument_arities=(1,),
    )
    top, *_ = index.retrieve(query, top_k=1)
    assert top.text_score > 0
    assert top.causal_score > 0
    assert top.type_score > 0
    assert top.contract_score == 0.0  # no scorer plugged in yet


def test_retrieval_is_deterministic(index):
    query = RetrievalQuery(text="object manipulation")
    first = [h.fragment_id for h in index.retrieve(query, top_k=3)]
    second = [h.fragment_id for h in index.retrieve(query, top_k=3)]
    assert first == second


def test_arity_alone_does_not_make_an_unrelated_fragment_relevant(index):
    query = RetrievalQuery(
        text="ferment sourdough starter",
        goal_predicates=("fermented",),
        argument_arities=(1,),
    )

    assert index.retrieve(query, top_k=3) == []


def test_contract_scorer_hook_changes_ranking():
    config = RetrievalConfig(
        text_weight=0.0,
        causal_weight=0.0,
        type_weight=0.0,
        contract_weight=1.0,
        contract_scorer=lambda query, fragment: (
            1.0 if fragment.fragment_id == "2/1" else 0.0
        ),
    )
    index = FragmentIndex([DRAWER, STOVE, PICK], config)
    hits = index.retrieve(RetrievalQuery(text="anything"), top_k=1)
    assert hits[0].fragment_id == "2/1"


def test_query_from_certificate_shape():
    """
    A certificate's failure class, goal, and causal neighborhood become the query's
    lexical and structural surface.
    """

    class Goal:
        predicate = "closed"
        arguments = ("d1",)
        negated = False

    class Neighborhood:
        predicates = ("opened", "closed", "openable")
        operators = ("open-drawer",)

    class Certificate:
        task_goal = (Goal(),)
        causal_neighborhood = Neighborhood()
        task_instruction = None
        goal_semantics = ()
        task_object_types = ()

        class failure_class:
            value = "missing_operator_model"

    query = RetrievalQuery.from_certificate(Certificate())
    assert "missing operator model" in query.text
    assert query.goal_predicates == ("closed",)
    assert "openable" in query.causal_predicates
    assert query.argument_arities == (1,)


@pytest.mark.skipif(not RELEASE.exists(), reason="corpus release r1 not built")
def test_real_release_smoke():
    from resym.knowledge.freeze import load_release

    fragments = load_release(RELEASE)
    index = FragmentIndex(fragments)
    query = RetrievalQuery(
        text="close the drawer missing operator",
        goal_predicates=("closed",),
        causal_predicates=("opened", "closed", "drawer"),
        argument_arities=(1,),
    )
    hits = index.retrieve(query, top_k=10)
    assert len(hits) == 10
    top_texts = " ".join(h.fragment.index_text().lower() for h in hits[:5])
    assert "drawer" in top_texts
