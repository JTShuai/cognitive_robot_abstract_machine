"""Initialization-stage drafting of grounding-factory candidates.

A deployment declares the predicate-grounding relations it needs as
:class:`GroundingFactoryRequest` records; a language model drafts one bounded
native-EQL candidate per request against the reviewed vocabulary. Every
accepted draft is only *submitted* to the local review workspace — a human
decision in the review Viewer is what makes a candidate executable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from resym.core.grounding import (
    GroundingFactoryCandidate,
    GroundingFactoryParameter,
    GroundingFactoryRole,
    GroundingFactorySourceKind,
    text_checksum,
)
from resym.llm.prompting import render_prompt
from resym.llm.structured import StructuredCompleter
from resym.platform.grounding_catalog import (
    GroundingFactorySourceValidator,
    GroundingFactoryWorkspace,
    GroundingVocabulary,
)

DRAFTING_AGENT_NAME = "grounding-factory-drafter"

MAXIMUM_DRAFT_ATTEMPTS = 3
"""
How many model drafts one request may consume before it is reported failed.
"""


@dataclass(frozen=True)
class GroundingFactoryRequest:
    """
    One predicate-grounding relation a deployment asks to have drafted.
    """

    proposed_uid: str
    """Semantic identity the resulting factory should carry."""

    semantic_name: str
    """Human-readable name of the relation."""

    meaning: str
    """Natural-language specification the draft must implement."""

    roles: tuple[GroundingFactoryRole, ...]
    """Ordered semantic inputs of the relation."""

    parameters: tuple[GroundingFactoryParameter, ...] = ()
    """Typed parameters grounding plans may configure."""


@dataclass(frozen=True)
class GroundingFactoryDraftFailure:
    """
    One request no accepted draft could be produced for.
    """

    proposed_uid: str
    """Identity of the failed request."""

    objections: tuple[str, ...]
    """Validator objections of the last rejected draft."""


@dataclass(frozen=True)
class GroundingFactoryDraftingReport:
    """
    Outcome of one initialization drafting pass.
    """

    submitted: tuple[GroundingFactoryCandidate, ...]
    """Candidates now pending human review."""

    failures: tuple[GroundingFactoryDraftFailure, ...]
    """Requests whose drafts never passed static validation."""


class GroundingFactoryDraftModel(BaseModel):
    """
    Structured draft the model must answer with.
    """

    source_code: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


def draft_grounding_factory_candidates(
    requests: tuple[GroundingFactoryRequest, ...],
    workspace: GroundingFactoryWorkspace,
    vocabulary: GroundingVocabulary,
    completer: StructuredCompleter,
    maximum_attempts: int = MAXIMUM_DRAFT_ATTEMPTS,
) -> GroundingFactoryDraftingReport:
    """
    Draft one pending-review candidate per request against the vocabulary.

    A draft the source validator rejects is retried with its objections shown
    to the model; a request whose attempts are exhausted is reported as a
    failure instead of entering the review queue.
    """
    validator = GroundingFactorySourceValidator(vocabulary)
    submitted: list[GroundingFactoryCandidate] = []
    failures: list[GroundingFactoryDraftFailure] = []
    for request in requests:
        objections: tuple[str, ...] = ()
        for _ in range(maximum_attempts):
            draft = completer.complete(
                DRAFTING_AGENT_NAME,
                _draft_prompt(request, vocabulary, objections),
                GroundingFactoryDraftModel,
            )
            candidate = _candidate(request, draft)
            objections = validator.candidate_objections(candidate)
            if not objections:
                workspace.submit(candidate)
                submitted.append(candidate)
                break
        else:
            failures.append(
                GroundingFactoryDraftFailure(
                    proposed_uid=request.proposed_uid,
                    objections=objections,
                )
            )
    return GroundingFactoryDraftingReport(
        submitted=tuple(submitted), failures=tuple(failures)
    )


def _candidate(
    request: GroundingFactoryRequest, draft: GroundingFactoryDraftModel
) -> GroundingFactoryCandidate:
    return GroundingFactoryCandidate(
        candidate_id=(
            f"draft-{request.semantic_name}-{text_checksum(draft.source_code)[:12]}"
        ),
        proposed_uid=request.proposed_uid,
        semantic_name=request.semantic_name,
        source_code=draft.source_code,
        roles=request.roles,
        parameters=request.parameters,
        generated_by=DRAFTING_AGENT_NAME,
        rationale=draft.rationale,
        source_kind=GroundingFactorySourceKind.AGENT_DRAFT,
    )


def _draft_prompt(
    request: GroundingFactoryRequest,
    vocabulary: GroundingVocabulary,
    objections: tuple[str, ...],
) -> str:
    roles = ", ".join(
        f"{role.name}: {role.symbol_type.python_type_ref}" for role in request.roles
    )
    parameters = (
        ", ".join(
            f"{parameter.name}: {parameter.value_type.value}"
            for parameter in request.parameters
        )
        or "none"
    )
    return render_prompt(
        "draft_grounding_factory",
        proposed_uid=request.proposed_uid,
        semantic_name=request.semantic_name,
        meaning=request.meaning,
        roles=roles,
        parameters=parameters,
        vocabulary=vocabulary.render() or "- none reviewed",
        objections="\n".join(f"- {item}" for item in objections) or "- none",
    )
