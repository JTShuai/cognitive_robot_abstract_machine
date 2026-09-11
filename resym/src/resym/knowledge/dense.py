"""
The dense retrieval channel: precomputed sentence embeddings over a frozen release.

The 12,889 fragment embeddings are encoded **once** per release with a
pinned model (:data:`MODEL_NAME`) and stored next to the release; at
query time only the query is encoded. The scorer plugs into
:class:`~resym.knowledge.retrieval.RetrievalConfig` by
fragment id (``dense_scorer_by_id``), so retrieval stays deterministic
and the heavy encode never runs inside an experiment.

Freezing protocol (revised plan §9.3/§6.2): the dense weight λ is chosen
by grid search on the *development* benchmark only
(:func:`select_dense_weight`), then written to
``retrieval_frozen.json`` beside the index with the model name and the
index checksum — held-out tasks never tune it. The embedding-level
leakage audit (:func:`audit_embedding_leakage`) reports fragments whose
embeddings sit suspiciously close to the local target library's own
descriptions, completing the audit the release manifest marked pending.

Everything here except :func:`mpnet_encoder` needs only numpy; the
sentence-transformers import is lazy so the module stays importable in
environments without torch.

CLI (one-time per release, run on a machine with sentence-transformers)::

    uv run python -m resym.knowledge.dense augment_dataset/corpus_release/r1 \
        --benchmark development_benchmark.json --targets targets.json

builds the index, runs the audit against the target texts, selects and
freezes λ on the supplied development benchmark, and writes all three
artifacts into the release directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from typing_extensions import Callable, Mapping, Optional, Sequence

from resym.knowledge.corpus import DomainFragment
from resym.knowledge.retrieval import (
    FragmentIndex,
    RetrievalQuery,
)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
INDEX_FILENAME = "dense_mpnet.npz"
AUDIT_FILENAME = "dense_leakage_audit.json"
FROZEN_CONFIG_FILENAME = "retrieval_frozen.json"

WEIGHT_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""
Λ candidates for the development grid search.
"""

LEAKAGE_COSINE_THRESHOLD = 0.9
"""
Embedding cosine above which a fragment counts as a near-duplicate of a target text.
"""

Encoder = Callable[[Sequence[str]], np.ndarray]
"""
Batch text encoder returning one embedding row per input text.
"""


def mpnet_encoder(model_name: str = MODEL_NAME) -> Encoder:
    """
    The pinned sentence-transformers encoder (lazy import; normalized embeddings so
    cosine is a dot product).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return encode


def _normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


@dataclass(frozen=True)
class DenseIndex:
    """
    Unit-normalized embedding rows, one per fragment, keyed by id.
    """

    fragment_ids: tuple[str, ...]
    matrix: np.ndarray

    _row_by_fragment_id: dict[str, int] = field(init=False, repr=False, compare=False)
    """
    Row index per fragment id, derived from ``fragment_ids``.
    """

    def __post_init__(self):
        if len(self.fragment_ids) != self.matrix.shape[0]:
            raise ValueError(
                f"{len(self.fragment_ids)} ids for "
                f"{self.matrix.shape[0]} embedding rows."
            )
        object.__setattr__(
            self,
            "_row_by_fragment_id",
            {fragment_id: row for row, fragment_id in enumerate(self.fragment_ids)},
        )

    @classmethod
    def build(
        cls,
        fragments: Sequence[DomainFragment],
        encode: Encoder,
        batch_size: int = 64,
    ) -> "DenseIndex":
        texts = [fragment.index_text() for fragment in fragments]
        rows = [
            np.asarray(encode(texts[start : start + batch_size]))
            for start in range(0, len(texts), batch_size)
        ]
        matrix = np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float32)
        return cls(
            fragment_ids=tuple(f.fragment_id for f in fragments),
            matrix=_normalized(matrix),
        )

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            matrix=self.matrix,
            fragment_ids=np.array(self.fragment_ids, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        with np.load(path, allow_pickle=True) as data:
            return cls(
                fragment_ids=tuple(str(i) for i in data["fragment_ids"]),
                matrix=np.asarray(data["matrix"], dtype=np.float32),
            )

    def checksum(self) -> str:
        digest = hashlib.sha256()
        digest.update("\n".join(self.fragment_ids).encode("utf-8"))
        digest.update(np.ascontiguousarray(self.matrix).tobytes())
        return digest.hexdigest()

    def similarities(
        self, query_vector: np.ndarray, fragment_ids: Sequence[str]
    ) -> list[float]:
        """
        Cosine similarity mapped to [0, 1]; an id absent from the index scores 0 (a
        fragment that was never embedded cannot claim dense relevance).
        """
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm > 0.0:
            query = query / norm
        scores = []
        for fragment_id in fragment_ids:
            row = self._row_by_fragment_id.get(fragment_id)
            if row is None:
                scores.append(0.0)
                continue
            cosine = float(self.matrix[row] @ query)
            scores.append((1.0 + cosine) / 2.0)
        return scores

    def scorer(self, encode: Encoder) -> Callable[[str, Sequence[str]], list[float]]:
        """
        The ``dense_scorer_by_id`` hook: encodes each distinct query text once (cached),
        scores candidates from the precomputed rows.
        """
        query_cache: dict[str, np.ndarray] = {}

        def score(query_text: str, fragment_ids: Sequence[str]) -> list[float]:
            if query_text not in query_cache:
                query_cache[query_text] = np.asarray(encode([query_text]))[0]
            return self.similarities(query_cache[query_text], fragment_ids)

        return score


# -- development-split weight selection ---------------------------------

Benchmark = Sequence[tuple[RetrievalQuery, frozenset]]
"""
Development queries with their relevant fragment-id sets.
"""


def recall_at_k(index: FragmentIndex, benchmark: Benchmark, top_k: int = 10) -> float:
    """
    Mean fraction of each query's relevant fragments found in the top-k.
    """
    if not benchmark:
        raise ValueError("Cannot score an empty benchmark.")
    total = 0.0
    for query, relevant in benchmark:
        if not relevant:
            raise ValueError(f"Query '{query.text[:40]}' has no relevant ids.")
        hits = index.retrieve(query, top_k=top_k)
        found = {hit.fragment_id for hit in hits} & set(relevant)
        total += len(found) / len(relevant)
    return total / len(benchmark)


def select_dense_weight(
    index: FragmentIndex,
    benchmark: Benchmark,
    weights: Sequence[float] = WEIGHT_GRID,
    top_k: int = 10,
) -> tuple[float, dict[float, float]]:
    """
    Grid-search λ on the development benchmark; ties break toward the smaller weight
    (prefer the simpler model).

    The index config is restored afterwards.
    """
    saved = index.config.dense_weight
    recalls: dict[float, float] = {}
    try:
        for weight in weights:
            index.config.dense_weight = weight
            recalls[weight] = recall_at_k(index, benchmark, top_k=top_k)
    finally:
        index.config.dense_weight = saved
    best = max(sorted(recalls), key=lambda weight: recalls[weight])
    return best, recalls


def load_benchmark(path: Path) -> Benchmark:
    """
    Load a domain-owned development benchmark from JSON.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    return [
        (
            RetrievalQuery(
                text=record["query"]["text"],
                goal_predicates=tuple(record["query"].get("goal_predicates", ())),
                causal_predicates=tuple(record["query"].get("causal_predicates", ())),
                argument_arities=tuple(record["query"].get("argument_arities", ())),
                available_capabilities=tuple(
                    record["query"].get("available_capabilities", ())
                ),
            ),
            frozenset(record["relevant_fragment_ids"]),
        )
        for record in records
    ]


# -- embedding-level leakage audit --------------------------------------


def audit_embedding_leakage(
    index: DenseIndex,
    targets: Mapping[str, str],
    encode: Encoder,
    threshold: float = LEAKAGE_COSINE_THRESHOLD,
) -> list[dict]:
    """
    Fragments whose embeddings are near-duplicates of a target text (the local library's
    own operator/predicate descriptions).

    Reported,
    not auto-excluded: exclusion is a release decision recorded in the
    release manifest, like the name-level filter.
    """
    if not targets:
        return []
    names = list(targets)
    target_matrix = _normalized(np.asarray(encode([targets[n] for n in names])))
    cosines = index.matrix @ target_matrix.T
    hits = []
    for row, column in zip(*np.nonzero(cosines >= threshold)):
        hits.append(
            {
                "fragment_id": index.fragment_ids[int(row)],
                "target": names[int(column)],
                "cosine": round(float(cosines[row, column]), 4),
            }
        )
    hits.sort(key=lambda hit: (-hit["cosine"], hit["fragment_id"], hit["target"]))
    return hits


# -- one-time release CLI -----------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Encode a frozen release, audit embedding leakage, and "
        "freeze the dense retrieval weight."
    )
    parser.add_argument(
        "release",
        type=Path,
        help="release directory (e.g. augment_dataset/corpus_release/r1)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="development benchmark JSON used to select the dense weight",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="JSON file {name: text} of local library descriptions to audit against",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args(argv)

    from resym.knowledge.freeze import load_release

    fragments = load_release(arguments.release)
    encode = mpnet_encoder()

    index_path = arguments.release / INDEX_FILENAME
    if index_path.exists():
        index = DenseIndex.load(index_path)
        print(f"loaded existing index: {len(index.fragment_ids)} embeddings")
    else:
        index = DenseIndex.build(fragments, encode, batch_size=arguments.batch_size)
        index.save(index_path)
        print(f"encoded {len(index.fragment_ids)} fragments -> {index_path}")

    audit: list[dict] = []
    if arguments.targets is not None:
        targets = json.loads(arguments.targets.read_text(encoding="utf-8"))
        audit = audit_embedding_leakage(index, targets, encode)
        (arguments.release / AUDIT_FILENAME).write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "threshold": LEAKAGE_COSINE_THRESHOLD,
                    "targets": sorted(targets),
                    "near_duplicates": audit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"leakage audit: {len(audit)} near-duplicates >= {LEAKAGE_COSINE_THRESHOLD}"
        )

    lexical = FragmentIndex(fragments)
    lexical.config.dense_scorer_by_id = index.scorer(encode)
    benchmark = load_benchmark(arguments.benchmark)
    best, recalls = select_dense_weight(lexical, benchmark)
    frozen = {
        "model": MODEL_NAME,
        "dense_weight": best,
        "development_recalls_at_10": {str(w): round(r, 4) for w, r in recalls.items()},
        "index_checksum": index.checksum(),
        "leakage_near_duplicates": len(audit),
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (arguments.release / FROZEN_CONFIG_FILENAME).write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )
    print(f"frozen dense weight {best} -> {arguments.release / FROZEN_CONFIG_FILENAME}")
    for weight in sorted(recalls):
        print(f"  λ={weight}: recall@10 {recalls[weight]:.4f}")


if __name__ == "__main__":
    main()
