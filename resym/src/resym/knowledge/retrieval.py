"""Two-stage retrieval over a frozen corpus release.

Stage one is lexical BM25 over the fragments' synthesized index text
(the corpus ships no instructions); stage two reranks the candidates by
structural compatibility with the failure: causal-neighborhood overlap,
type/arity fit, and — once the executable adapter can decide it —
contract mappability. The combined score is

    S(d) = λt·S_text + λc·S_causal + λy·S_type + λk·S_contract

with weights fixed in the index configuration and frozen before formal
experiments. Retrieval is deterministic and LLM-free; every query
records the ranked fragment ids so provenance survives into admission.

The dense-encoder stage and the contract scorer are pluggable hooks
(:class:`RetrievalConfig`): absent, their terms contribute zero — the
lexical+structural core has no third-party dependencies and runs
anywhere.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Callable, Optional, Sequence

from resym.knowledge.corpus import DomainFragment

if TYPE_CHECKING:
    from resym.repair.certificate import FailureCertificate

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower().replace("_", " "))


@dataclass(frozen=True)
class RetrievalQuery:
    """
    A structured query built from a failure certificate, not a free-text sentence.
    """

    text: str
    """
    Lexical surface: desired effect, failed goal, missing model class.
    """

    goal_predicates: tuple[str, ...] = ()
    """
    Predicate names of the failed goal.
    """

    causal_predicates: tuple[str, ...] = ()
    """
    Predicate names of the failure's causal neighborhood.
    """

    argument_arities: tuple[int, ...] = ()
    """
    Arities occurring in the goal/neighborhood, for type-fit scoring.
    """

    available_capabilities: tuple[str, ...] = ()
    """
    Capability UIDs of the current embodiment.
    """

    @classmethod
    def from_certificate(cls, certificate: FailureCertificate) -> RetrievalQuery:
        """
        Build the query from a failure certificate's lexical and causal surface.
        """
        goal_predicates = tuple(literal.predicate for literal in certificate.task_goal)
        neighborhood = certificate.causal_neighborhood
        text_parts = [
            certificate.failure_class.value.replace("_", " "),
            certificate.task_instruction or "",
            *certificate.goal_semantics,
            *(
                symbol_type.replace("_", " ").replace(".", " ")
                for _, symbol_type in certificate.task_object_types
            ),
            *[p.replace("-", " ") for p in goal_predicates],
            *[p.replace("-", " ") for p in neighborhood.predicates],
            *[o.replace("-", " ") for o in neighborhood.operators],
        ]
        return cls(
            text=" ".join(text_parts),
            goal_predicates=goal_predicates,
            causal_predicates=tuple(neighborhood.predicates),
            argument_arities=tuple(
                sorted({len(literal.arguments) for literal in certificate.task_goal})
            ),
        )


@dataclass(frozen=True)
class RankedFragment:
    """
    One retrieval hit with its score decomposition, for logging and the paper's per-
    component analysis.
    """

    fragment: DomainFragment
    score: float
    text_score: float
    causal_score: float
    type_score: float
    contract_score: float
    dense_score: float = 0.0
    evidence_score: float = 0.0
    """
    Score excluding arity alone, which is not semantic evidence.
    """

    @property
    def fragment_id(self) -> str:
        return self.fragment.fragment_id


ContractScorer = Callable[[RetrievalQuery, DomainFragment], float]
"""
Hook: how mappable a fragment is onto the available contracts, in [0, 1].

Provided by the executable adapter once it exists.
"""

DenseScorer = Callable[[str, Sequence[str]], Sequence[float]]
"""
Hook: dense similarity of a query against candidate index texts, in [0, 1] per
candidate.
"""

DenseScorerById = Callable[[str, Sequence[str]], Sequence[float]]
"""
Hook: dense similarity of a query against candidates named by fragment id, in [0, 1] per
candidate — the shape a precomputed embedding index
(:class:`~resym.knowledge.dense.DenseIndex`) provides, so only the query is ever encoded
at retrieval time.
"""


@dataclass
class RetrievalConfig:
    """
    Weights and hooks; freeze this with the corpus release before formal experiments.
    """

    text_weight: float = 1.0
    causal_weight: float = 0.6
    type_weight: float = 0.2
    contract_weight: float = 0.0
    """
    Zero until the adapter provides a real contract scorer.
    """

    first_stage_size: int = 200
    """
    BM25 candidates entering the rerank stage.
    """

    contract_scorer: Optional[ContractScorer] = None
    dense_scorer: Optional[DenseScorer] = None
    dense_scorer_by_id: Optional[DenseScorerById] = None
    """
    Preferred over ``dense_scorer`` when both are set: candidates are looked up by
    fragment id in a precomputed index.
    """

    dense_weight: float = 0.0

    minimum_evidence_score: float = 0.05
    """
    Minimum lexical/causal/contract/dense evidence required for a hit.

    Type compatibility alone cannot make an unrelated fragment relevant.
    """

    def describe(self) -> dict:
        return {
            "text_weight": self.text_weight,
            "causal_weight": self.causal_weight,
            "type_weight": self.type_weight,
            "contract_weight": self.contract_weight,
            "dense_weight": self.dense_weight,
            "minimum_evidence_score": self.minimum_evidence_score,
            "first_stage_size": self.first_stage_size,
            "contract_scorer": bool(self.contract_scorer),
            "dense_scorer": bool(self.dense_scorer or self.dense_scorer_by_id),
        }


class FragmentIndex:
    """
    BM25 index plus structural rerank over one frozen fragment set.
    """

    K1 = 1.5
    B = 0.75

    def __init__(
        self,
        fragments: Sequence[DomainFragment],
        config: Optional[RetrievalConfig] = None,
    ):
        self.fragments = list(fragments)
        self.config = config or RetrievalConfig()
        self._documents = [tokenize(f.index_text()) for f in self.fragments]
        self._frequencies = [Counter(document) for document in self._documents]
        self._lengths = [len(document) for document in self._documents]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._document_frequency: Counter = Counter()
        for frequency in self._frequencies:
            self._document_frequency.update(frequency.keys())

    def retrieve(self, query: RetrievalQuery, top_k: int = 10) -> list[RankedFragment]:
        """
        BM25 first stage, structural rerank second; deterministic ties by fragment id.
        """
        text_scores = self._bm25_scores(tokenize(query.text))
        candidate_indices = sorted(
            range(len(self.fragments)),
            key=lambda i: (-text_scores[i], self.fragments[i].fragment_id),
        )[: self.config.first_stage_size]

        maximum_text = max((text_scores[i] for i in candidate_indices), default=0.0)
        dense_scores = self._dense_scores(query, candidate_indices)
        ranked = []
        for position, index in enumerate(candidate_indices):
            fragment = self.fragments[index]
            text_score = text_scores[index] / maximum_text if maximum_text > 0 else 0.0
            causal_score = _overlap(
                query.causal_predicates + query.goal_predicates,
                fragment.predicate_names(),
            )
            type_score = _arity_fit(query.argument_arities, fragment)
            contract_score = (
                self.config.contract_scorer(query, fragment)
                if self.config.contract_scorer
                else 0.0
            )
            score = (
                self.config.text_weight * text_score
                + self.config.causal_weight * causal_score
                + self.config.type_weight * type_score
                + self.config.contract_weight * contract_score
                + self.config.dense_weight * dense_scores[position]
            )
            evidence_score = (
                self.config.text_weight * text_score
                + self.config.causal_weight * causal_score
                + self.config.contract_weight * contract_score
                + self.config.dense_weight * dense_scores[position]
            )
            if evidence_score < self.config.minimum_evidence_score:
                continue
            ranked.append(
                RankedFragment(
                    fragment=fragment,
                    score=score,
                    text_score=text_score,
                    causal_score=causal_score,
                    type_score=type_score,
                    contract_score=contract_score,
                    dense_score=dense_scores[position],
                    evidence_score=evidence_score,
                )
            )
        ranked.sort(key=lambda hit: (-hit.score, hit.fragment_id))
        return ranked[:top_k]

    def _bm25_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.fragments)
        document_count = len(self.fragments)
        for token in query_tokens:
            containing = self._document_frequency.get(token, 0)
            if containing == 0:
                continue
            idf = math.log(
                (document_count - containing + 0.5) / (containing + 0.5) + 1.0
            )
            for index, frequency in enumerate(self._frequencies):
                occurrences = frequency.get(token, 0)
                if occurrences == 0:
                    continue
                length_norm = (
                    1.0
                    - self.B
                    + self.B * (self._lengths[index] / self._average_length)
                )
                scores[index] += idf * (
                    occurrences
                    * (self.K1 + 1.0)
                    / (occurrences + self.K1 * length_norm)
                )
        return scores

    def _dense_scores(
        self, query: RetrievalQuery, candidate_indices: list[int]
    ) -> list[float]:
        if self.config.dense_weight == 0.0:
            return [0.0] * len(candidate_indices)
        if self.config.dense_scorer_by_id is not None:
            identifiers = [self.fragments[i].fragment_id for i in candidate_indices]
            return list(self.config.dense_scorer_by_id(query.text, identifiers))
        if self.config.dense_scorer is not None:
            texts = [self.fragments[i].index_text() for i in candidate_indices]
            return list(self.config.dense_scorer(query.text, texts))
        return [0.0] * len(candidate_indices)


def _overlap(query_names: Sequence[str], fragment_names: frozenset[str]) -> float:
    """
    Token-level Jaccard-style overlap between the query's predicate names and the
    fragment's, tolerant to naming conventions (underscores vs hyphens).
    """
    query_tokens = {t for name in query_names for t in tokenize(name)}
    fragment_tokens = {t for name in fragment_names for t in tokenize(name)}
    if not query_tokens or not fragment_tokens:
        return 0.0
    return len(query_tokens & fragment_tokens) / len(query_tokens | fragment_tokens)


def _arity_fit(arities: Sequence[int], fragment: DomainFragment) -> float:
    """
    Share of the fragment's predicates whose arity occurs in the query.
    """
    if not arities or not fragment.predicates:
        return 0.0
    wanted = set(arities)
    matching = sum(1 for p in fragment.predicates if p.arity in wanted)
    return matching / len(fragment.predicates)
