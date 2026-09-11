"""
Stable identities for predicates and their executable truth procedures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Self

PREDICATE_NAMESPACE = "resym:predicate/"
TRUTH_PROCEDURE_NAMESPACE = "resym:truth-procedure/"


@dataclass(frozen=True)
class PredicateRef:
    """
    Versioned semantic identity, independent of a task-local PDDL name.

    Equality is the ``uid`` alone: a version bump revises the same semantic identity, so
    effect gating and contract matching must keep treating the predicate as the one the
    contract lists.
    """

    uid: str
    version: str = field(default="1", compare=False)
    local_name: str | None = field(default=None, compare=False, hash=False)

    @classmethod
    def from_name(cls, name: str, version: str = "1") -> PredicateRef:
        return cls(f"{PREDICATE_NAMESPACE}{name}", version, name)


@dataclass(frozen=True)
class TruthProcedureRef:
    """
    Versioned identity of reviewed executable predicate semantics.
    """

    uid: str
    version: str = "1"

    @classmethod
    def registered(cls, evaluator: str, version: str = "1") -> TruthProcedureRef:
        return cls(f"{TRUTH_PROCEDURE_NAMESPACE}registry/{evaluator}", version)

    @classmethod
    def query(cls, predicate_name: str, version: str = "1") -> TruthProcedureRef:
        """
        Identity of the reviewed query plan owned by the named predicate.
        """
        return cls(f"{TRUTH_PROCEDURE_NAMESPACE}query/{predicate_name}", version)


@dataclass(frozen=True)
class PredicateImplementation:
    """
    One reviewed platform query behind a predicate symbol.
    """

    ref: TruthProcedureRef
    evaluator_key: str

    @classmethod
    def registered(cls, evaluator_key: str) -> Self:
        return cls(
            ref=TruthProcedureRef.registered(evaluator_key),
            evaluator_key=evaluator_key,
        )
