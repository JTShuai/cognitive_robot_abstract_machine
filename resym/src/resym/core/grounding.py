"""
Structured failures produced while grounding a binary predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from resym.core.symbols import Literal


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
