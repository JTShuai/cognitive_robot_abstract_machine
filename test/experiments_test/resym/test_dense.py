"""
The dense retrieval channel: index build/save/load, id-keyed scoring through the rerank,
development-split weight selection, and the embedding-level leakage audit — all against
a deterministic stub encoder so no model download is needed.

Host-runnable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from resym.knowledge.corpus import DomainFragment, _parse_predicate
from resym.knowledge.dense import (
    DenseIndex,
    audit_embedding_leakage,
    recall_at_k,
    select_dense_weight,
)
from experiments.resym.icra.articulation.retrieval_benchmark import (
    drawer_development_benchmark,
)
from resym.knowledge.retrieval import (
    FragmentIndex,
    RetrievalConfig,
    RetrievalQuery,
)

SYNONYM_GROUPS = (
    ("drawer", "cabinet"),
    ("open", "unlatch"),
    ("close", "shut"),
    ("stove",),
    ("burner",),
    ("cook",),
    ("handle", "knob"),
    ("pull", "tug"),
)


def stub_encode(texts):
    """
    Deterministic bag-of-concepts embedding: synonym groups share a dimension, so
    semantically similar wording gives high cosine even with zero lexical overlap — the
    property that separates the dense channel from BM25, without a model download.
    """
    rows = []
    for text in texts:
        lowered = text.lower()
        rows.append(
            [
                float(sum(lowered.count(word) for word in group))
                for group in SYNONYM_GROUPS
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def fragment(fragment_id: str, predicates: dict[str, str]) -> DomainFragment:
    return DomainFragment(
        fragment_id=fragment_id,
        predicates=tuple(
            _parse_predicate(fragment_id, signature, gloss)
            for signature, gloss in predicates.items()
        ),
        operators=(),
        source_path=f"/synthetic/{fragment_id}",
    )


DRAWER = fragment(
    "d/1", {"(drawer_open ?d)": "pull the drawer handle to open the drawer"}
)
STOVE = fragment("s/1", {"(stove_on ?s)": "the stove burner is on to cook"})
FRAGMENTS = (DRAWER, STOVE)


@pytest.fixture()
def index():
    return DenseIndex.build(FRAGMENTS, stub_encode, batch_size=1)


class TestDenseIndex:
    def test_rows_are_unit_normalized(self, index):
        norms = np.linalg.norm(index.matrix, axis=1)
        assert np.allclose(norms, 1.0)

    def test_save_load_roundtrip_preserves_checksum(self, index, tmp_path):
        path = tmp_path / "dense.npz"
        index.save(path)
        loaded = DenseIndex.load(path)
        assert loaded.fragment_ids == index.fragment_ids
        assert loaded.checksum() == index.checksum()

    def test_similarities_rank_matching_text_first(self, index):
        query = stub_encode(["open the drawer"])[0]
        drawer_score, stove_score = index.similarities(query, ("d/1", "s/1"))
        assert drawer_score > stove_score

    def test_unknown_fragment_id_scores_zero(self, index):
        query = stub_encode(["open the drawer"])[0]
        assert index.similarities(query, ("missing/1",)) == [0.0]

    def test_scorer_encodes_each_query_once(self, index):
        calls = []

        def counting_encode(texts):
            calls.append(tuple(texts))
            return stub_encode(texts)

        score = index.scorer(counting_encode)
        first = score("open the drawer", ("d/1", "s/1"))
        second = score("open the drawer", ("s/1",))
        assert len(calls) == 1
        assert second == [first[1]]


class TestRerankIntegration:
    def test_by_id_scorer_changes_the_ranking(self, index):
        """
        A query lexically ambiguous between the fragments is decided by the dense
        channel once its weight is on.
        """
        query = RetrievalQuery(text="open drawer stove burner")
        without = FragmentIndex(FRAGMENTS)
        baseline = [h.fragment_id for h in without.retrieve(query, top_k=2)]

        config = RetrievalConfig(
            text_weight=0.0,
            causal_weight=0.0,
            type_weight=0.0,
            dense_weight=1.0,
            dense_scorer_by_id=index.scorer(stub_encode),
        )
        dense_only = FragmentIndex(FRAGMENTS, config)
        ranked = dense_only.retrieve(
            RetrievalQuery(text="pull the handle to open the drawer"), top_k=2
        )
        assert [h.fragment_id for h in ranked] == ["d/1", "s/1"]
        assert set(baseline) == {"d/1", "s/1"}

    def test_zero_weight_never_calls_the_scorer(self, index):
        def exploding(query_text, ids):
            raise AssertionError("dense scorer must not run at weight 0")

        config = RetrievalConfig(dense_weight=0.0, dense_scorer_by_id=exploding)
        FragmentIndex(FRAGMENTS, config).retrieve(
            RetrievalQuery(text="drawer"), top_k=1
        )


class TestWeightSelection:
    def test_grid_prefers_the_weight_that_finds_the_relevant_fragment(self, index):
        """
        Lexically the query is a near-tie ('cook' anchors the stove, 'pull' the drawer,
        and BM25's length norm tips it to the stove); the synonyms 'unlatch cabinet
        knob' are invisible to BM25 but the dense channel maps them onto the drawer
        fragment, so nonzero λ wins the top-1.
        """
        config = RetrievalConfig(dense_scorer_by_id=index.scorer(stub_encode))
        lexical = FragmentIndex(FRAGMENTS, config)
        benchmark = [
            (
                RetrievalQuery(text="cook pull unlatch cabinet knob"),
                frozenset({"d/1"}),
            )
        ]
        best, recalls = select_dense_weight(
            lexical, benchmark, weights=(0.0, 1.0), top_k=1
        )
        assert recalls[1.0] > recalls[0.0]
        assert best == 1.0
        assert lexical.config.dense_weight == 0.0  # restored

    def test_ties_break_toward_the_smaller_weight(self, index):
        config = RetrievalConfig(dense_scorer_by_id=index.scorer(stub_encode))
        lexical = FragmentIndex(FRAGMENTS, config)
        benchmark = [(RetrievalQuery(text="open the drawer"), frozenset({"d/1"}))]
        best, recalls = select_dense_weight(
            lexical, benchmark, weights=(0.0, 0.4), top_k=2
        )
        assert recalls[0.0] == recalls[0.4] == 1.0
        assert best == 0.0

    def test_empty_benchmark_is_rejected(self, index):
        with pytest.raises(ValueError, match="empty benchmark"):
            recall_at_k(FragmentIndex(FRAGMENTS), [], top_k=1)

    def test_development_benchmark_labels_drawer_fragments(self):
        benchmark = drawer_development_benchmark(FRAGMENTS)
        assert len(benchmark) == 3
        for _, relevant in benchmark:
            assert relevant == frozenset({"d/1"})


class TestLeakageAudit:
    def test_near_duplicate_of_a_target_is_reported(self, index):
        hits = audit_embedding_leakage(
            index,
            {"opened": "pull the drawer handle to open the drawer"},
            stub_encode,
            threshold=0.95,
        )
        assert [hit["fragment_id"] for hit in hits] == ["d/1"]
        assert hits[0]["target"] == "opened"
        assert hits[0]["cosine"] > 0.95

    def test_unrelated_targets_report_nothing(self, index):
        hits = audit_embedding_leakage(
            index,
            {"navigation": "drive the mobile base to a pose"},
            stub_encode,
            threshold=0.9,
        )
        assert hits == []

    def test_no_targets_no_audit(self, index):
        assert audit_embedding_leakage(index, {}, stub_encode) == []


RELEASE = Path(__file__).resolve().parent.parent / "corpus_release" / "r1"


@pytest.mark.skipif(
    not (RELEASE / "dense_mpnet.npz").exists(),
    reason="dense index for release r1 not built",
)
def test_frozen_release_index_smoke():
    """
    The shipped artifacts agree with each other: the frozen config's checksum matches
    the index bytes, the model is the pinned one, and the frozen weight came from the
    recorded development grid.
    """
    import json

    index = DenseIndex.load(RELEASE / "dense_mpnet.npz")
    frozen = json.loads((RELEASE / "retrieval_frozen.json").read_text())
    assert frozen["model"] == "sentence-transformers/all-mpnet-base-v2"
    assert frozen["index_checksum"] == index.checksum()
    assert index.matrix.shape[1] == 768
    assert len(index.fragment_ids) == len(set(index.fragment_ids)) > 10_000
    recalls = {
        float(weight): recall
        for weight, recall in frozen["development_recalls_at_10"].items()
    }
    best = max(recalls.values())
    assert recalls[frozen["dense_weight"]] == best
    assert all(
        recalls[weight] < best for weight in recalls if weight < frozen["dense_weight"]
    )  # ties broke toward the smaller weight
