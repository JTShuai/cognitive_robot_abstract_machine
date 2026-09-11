"""
Audit metadata persisted beside symbols and capability alignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class KnowledgeSource(StrEnum):
    """
    Where a generated symbol's content came from.
    """

    RETRIEVAL = "retrieval"
    LLM_PRIOR = "llm-prior"
    MIXED = "mixed"


@dataclass(frozen=True)
class Provenance:
    """
    Where an admitted symbol came from and what evidence admitted it.
    """

    source: str = "seed"
    capability_request: str | None = None
    proposal_backend: str | None = None
    model: str | None = None
    retrieved_ids: tuple[str, ...] = ()
    knowledge_source: KnowledgeSource | None = None
    """
    Absent for seeded or hand-curated symbols.
    """

    test_suite_version: str | None = None
    admitted_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieved_ids", tuple(self.retrieved_ids))


@dataclass(frozen=True)
class OntologyAlignment:
    """
    Auditable alignment of a local capability term to an ontology.
    """

    relation: str = "NO_MATCH"
    target_iri: str | None = None
    source_version: str | None = None
    decided_by: str | None = None
    decision_model: str | None = None
    rationale: str | None = None
    evidence_iris: tuple[str, ...] = ()
    human_reviewed: bool = False
    human_reviewer: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_iris", tuple(self.evidence_iris))
