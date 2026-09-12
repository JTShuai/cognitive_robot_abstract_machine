"""
Initialization-stage LLM drafting of grounding-factory candidates.

The model only produces review material: every accepted draft lands in the workspace as
a pending candidate; nothing becomes executable without a human approval.
"""

from __future__ import annotations

import json
from pathlib import Path

from resym.core.grounding_model import (
    GroundingFactoryReviewStatus,
    GroundingFactoryRole,
    GroundingFactorySourceKind,
)
from resym.core.symbol_types import SymbolType
from resym.interfaces.grounding_drafting import (
    GroundingFactoryRequest,
    draft_grounding_factory_candidates,
)
from resym.llm.client import ScriptedCompletionClient
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.platform.grounding_catalog import (
    GroundingFactorySourceValidator,
    GroundingFactoryWorkspace,
    helper_vocabulary,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot

from .dataset.task_model import GROUNDING_QUERY_HELPERS, factory_candidates
from .test_grounding_factory_catalog import DATASET, vocabulary

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


def request() -> GroundingFactoryRequest:
    return GroundingFactoryRequest(
        proposed_uid="resym:grounding/inside-region",
        semantic_name="inside-region",
        meaning="True iff the object currently lies inside the region.",
        roles=(GroundingFactoryRole("object", ROBOT_TYPE),),
    )


def draft_response(source_file: str) -> str:
    return json.dumps(
        {
            "source_code": (DATASET / source_file).read_text(),
            "rationale": "the reviewed vocabulary can decide this relation",
        }
    )


def completer_with(
    *responses: str,
) -> tuple[StructuredCompleter, ScriptedCompletionClient]:
    client = ScriptedCompletionClient(responses=list(responses))
    return (
        StructuredCompleter(client=client, transcript=TranscriptRecorder()),
        client,
    )


def test_valid_draft_becomes_a_pending_candidate(tmp_path: Path) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    completer, client = completer_with(draft_response("valid_factory.py"))

    report = draft_grounding_factory_candidates(
        (request(),), workspace, vocabulary(), completer
    )

    (candidate,) = workspace.candidates()
    assert report.submitted == (candidate,)
    assert report.failures == ()
    assert candidate.proposed_uid == request().proposed_uid
    assert candidate.roles == request().roles
    assert candidate.review_status is GroundingFactoryReviewStatus.PENDING_REVIEW
    assert candidate.source_kind is GroundingFactorySourceKind.AGENT_DRAFT
    assert workspace.specifications() == ()  # drafting never approves


def test_rejected_draft_is_retried_with_its_objections(tmp_path: Path) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    completer, client = completer_with(
        draft_response("invalid_factory.py"),
        draft_response("valid_factory.py"),
    )

    report = draft_grounding_factory_candidates(
        (request(),), workspace, vocabulary(), completer
    )

    assert len(report.submitted) == 1
    (validator_objection,) = GroundingFactorySourceValidator(vocabulary()).objections(
        (DATASET / "invalid_factory.py").read_text()
    )
    assert validator_objection in client.received_prompts[1]


def test_exhausted_attempts_report_a_failure_without_a_candidate(
    tmp_path: Path,
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    completer, _ = completer_with(
        draft_response("invalid_factory.py"),
        draft_response("invalid_factory.py"),
    )

    report = draft_grounding_factory_candidates(
        (request(),),
        workspace,
        vocabulary(),
        completer,
        maximum_attempts=2,
    )

    assert report.submitted == ()
    (failure,) = report.failures
    assert failure.proposed_uid == request().proposed_uid
    assert failure.objections
    assert workspace.candidates() == ()


def test_task_model_factory_sources_satisfy_the_reviewed_sandbox() -> None:
    """
    Every factory the task model submits passes the same sandbox a drafted candidate
    must pass, against the helpers the task model asks to review.
    """
    validator = GroundingFactorySourceValidator(
        helper_vocabulary(GROUNDING_QUERY_HELPERS)
    )
    candidates = factory_candidates()

    objections = {
        candidate.proposed_uid: validator.candidate_objections(candidate)
        for candidate in candidates
    }

    assert candidates
    assert objections == {candidate.proposed_uid: () for candidate in candidates}
