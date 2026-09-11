"""
Reviewed predicate-grounding records and their structured runtime failures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resym.core.symbols import Literal
    from resym.core.types import SymbolType


type GroundingParameterValue = str | int | float | bool
GROUNDING_PLAN_EVALUATOR_KEY = "grounding-plan"
"""
Compatibility key stored where legacy predicates carry an evaluator name.
"""

# %% Grounding factories and plans


class GroundingFactoryOrigin(StrEnum):
    """
    Durable source of an approved grounding factory.
    """

    PLATFORM = "platform"
    LOCAL = "local"
    SHARED = "shared"


class GroundingFactorySourceKind(StrEnum):
    """
    How a pending factory candidate was obtained.
    """

    DISCOVERED = "discovered"
    AGENT_DRAFT = "agent-draft"


class GroundingFactoryReviewStatus(StrEnum):
    """
    Human-review state of a grounding factory candidate.
    """

    PENDING_REVIEW = "pending-review"
    APPROVED = "approved"
    REJECTED = "rejected"


class GroundingFactoryParameterType(StrEnum):
    """
    JSON-compatible value kinds accepted by a factory parameter.
    """

    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"


@dataclass(frozen=True)
class GroundingFactoryRole:
    """
    Typed semantic input accepted by a grounding factory.
    """

    name: str
    """
    Semantic role name.
    """

    symbol_type: SymbolType
    """
    Most general CRAM type accepted for the role.
    """


@dataclass(frozen=True)
class GroundingFactoryParameter:
    """
    One reviewed configurable input of a grounding factory.
    """

    name: str
    """
    Stable parameter name used in a grounding plan.
    """

    value_type: GroundingFactoryParameterType
    """
    Primitive value kind accepted by the implementation.
    """

    required: bool = True
    """
    Whether each grounding plan must provide the parameter.
    """

    minimum: float | None = None
    """
    Inclusive numeric lower bound, when applicable.
    """

    maximum: float | None = None
    """
    Inclusive numeric upper bound, when applicable.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "value_type", GroundingFactoryParameterType(self.value_type)
        )

    def accepts(self, value: GroundingParameterValue) -> bool:
        """
        Return whether a plan value has the declared kind and range.
        """
        kind_matches = {
            GroundingFactoryParameterType.BOOLEAN: type(value) is bool,
            GroundingFactoryParameterType.INTEGER: type(value) is int,
            GroundingFactoryParameterType.NUMBER: (
                type(value) is int or type(value) is float
            ),
            GroundingFactoryParameterType.STRING: type(value) is str,
        }[self.value_type]
        if not kind_matches:
            return False
        if type(value) not in (int, float):
            return True
        numeric_value = float(value)
        return (self.minimum is None or numeric_value >= self.minimum) and (
            self.maximum is None or numeric_value <= self.maximum
        )


@dataclass(frozen=True)
class GroundingFactoryCandidate:
    """
    Non-executable source proposal awaiting a human decision.
    """

    candidate_id: str
    """
    Stable identity inside the local review workspace.
    """

    proposed_uid: str
    """
    Proposed semantic identity of the resulting factory.
    """

    semantic_name: str
    """
    Human-readable meaning of the proposed query.
    """

    source_code: str
    """
    Native EQL factory source kept outside importable paths.
    """

    roles: tuple[GroundingFactoryRole, ...]
    """
    Ordered semantic inputs of the proposed implementation.
    """

    generated_by: str
    """
    Agent or scanner that produced the candidate.
    """

    rationale: str
    """
    Why the candidate is expected to implement the named relation.
    """

    evidence: tuple[str, ...] = ()
    """
    Retrieval and preview evidence supplied for review.
    """

    parameters: tuple[GroundingFactoryParameter, ...] = ()
    """
    Proposed typed parameters for the generated factory.
    """

    source_kind: GroundingFactorySourceKind = GroundingFactorySourceKind.AGENT_DRAFT
    """
    Whether discovery or an agent produced the candidate.
    """

    review_status: GroundingFactoryReviewStatus = (
        GroundingFactoryReviewStatus.PENDING_REVIEW
    )
    """
    Current human-review decision.
    """

    reviewed_by: str | None = None
    """
    Reviewer identity after a decision.
    """

    review_note: str | None = None
    """
    Reviewer explanation after a decision.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(
            self, "source_kind", GroundingFactorySourceKind(self.source_kind)
        )
        object.__setattr__(
            self,
            "review_status",
            GroundingFactoryReviewStatus(self.review_status),
        )


@dataclass(frozen=True)
class GroundingFactorySpec:
    """
    Current approved implementation of one grounding-factory identity.
    """

    uid: str
    """
    Stable semantic identity.
    """

    semantic_name: str
    """
    Human-readable meaning of the query.
    """

    implementation_ref: str
    """
    Importable ``module:function`` entry point.
    """

    implementation_checksum: str
    """
    Hash of the reviewed source implementation.
    """

    roles: tuple[GroundingFactoryRole, ...]
    """
    Ordered semantic inputs accepted by the entry point.
    """

    origin: GroundingFactoryOrigin
    """
    Platform, local, or shared source of the implementation.
    """

    reviewed_by: str
    """
    Authority that admitted the current implementation.
    """

    approved_at: str
    """
    UTC timestamp of its admission.
    """

    active_revision_id: str
    """
    Audit revision of the single active implementation.
    """

    parameters: tuple[GroundingFactoryParameter, ...] = ()
    """
    Reviewed configurable inputs accepted by grounding plans.
    """

    supported_embodiments: tuple[str, ...] = ()
    """
    Compatible profiles; empty means platform-independent.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "origin", GroundingFactoryOrigin(self.origin))
        object.__setattr__(
            self, "supported_embodiments", tuple(self.supported_embodiments)
        )


@dataclass(frozen=True)
class PredicateGroundingPlan:
    """
    Reviewed binding from one predicate schema to an approved factory.
    """

    factory_uid: str
    """
    Stable identity resolved through the current factory catalog.
    """

    approved_factory_checksum: str
    """
    Implementation hash approved for this plan.
    """

    role_bindings: tuple[tuple[str, int], ...] = ()
    """
    Factory roles mapped to predicate-argument positions.
    """

    parameters: tuple[tuple[str, GroundingParameterValue], ...] = ()
    """
    Explicit semantic parameters such as thresholds.
    """

    negated: bool = False
    """
    Whether the Boolean factory result is complemented.
    """

    version: str = "1"
    """
    Version of this predicate-to-factory binding.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role_bindings",
            tuple(tuple(binding) for binding in self.role_bindings),
        )
        object.__setattr__(
            self,
            "parameters",
            tuple(tuple(parameter) for parameter in self.parameters),
        )


# %% Structured runtime failures


class GroundingFailureCode(StrEnum):
    """
    Stable reason why a predicate query produced no Boolean value.
    """

    MISSING_WORLD_KNOWLEDGE = "missing_world_knowledge"
    RESOURCE_LIMIT = "resource_limit"
    QUERY_ERROR = "query_error"
    UNSUPPORTED_QUERY = "unsupported_query"


@dataclass(eq=False)
class GroundingFailure(Exception):
    """
    A predicate query failure kept outside the planning truth domain.
    """

    code: GroundingFailureCode
    """
    Machine-readable failure category.
    """

    detail: str
    """
    Evidence explaining why the query could not return a Boolean value.
    """

    atom: Literal | None = None
    """
    Ground atom being evaluated, when the caller has attached it.
    """

    def __post_init__(self) -> None:
        Exception.__init__(self, self.detail)

    def for_atom(self, atom: Literal) -> GroundingFailure:
        """
        Return the same diagnostic attached to a concrete ground atom.
        """
        return replace(self, atom=atom)
