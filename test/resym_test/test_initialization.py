"""
Initialization exports, drafts and imports without approving generated assets.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from resym.interfaces.initialization import Initialization
from resym.interfaces.initialization_models import GroundingRequest, InitializationKind
from resym.platform.capability_contract_review import CapabilityContractWorkspace
from resym.platform.coraplex_realizations import (
    CapabilityReviewStatus,
    CoraplexRealizationWorkspace,
)
from resym.platform.coraplex_catalog import approve_realization_candidate

from .test_capability_drafting import PLACE_ACTION, completer_with, draft_response
from .test_grounding_drafting import draft_response as grounding_response
from .test_grounding_drafting import request
from .test_grounding_factory_catalog import vocabulary

# %% preparation and common admission path


@pytest.fixture
def initialization(tmp_path, monkeypatch):
    """
    An isolated workspace using the installed action and a small EQL vocabulary.
    """
    monkeypatch.setattr(
        "resym.platform.grounding_catalog.discover_default_grounding_vocabulary",
        vocabulary,
    )
    return Initialization(tmp_path)


def test_external_contract_import_is_pending_and_repeatable(initialization):
    jobs = initialization.prepare(action_ids=(PLACE_ACTION,))
    (job,) = jobs
    assert job.kind is InitializationKind.CONTRACT
    assert job.response_schema
    initialization.response_path(job).write_text(draft_response())

    first = initialization.import_responses(generated_by="external-author")
    second = initialization.import_responses(generated_by="external-author")
    workspace = CapabilityContractWorkspace(initialization.contract_root)
    (candidate,) = workspace.candidates()
    assert first.failed == second.failed == {}
    assert candidate.generated_by == "external-author"
    assert candidate.review_status is CapabilityReviewStatus.PENDING
    assert workspace.approved_contracts() == ()
    assert len(workspace.candidates()) == 1


def test_api_draft_reuses_import_and_skips_saved_responses(initialization):
    initialization.prepare(action_ids=(PLACE_ACTION,))
    completer, client = completer_with(draft_response())
    first = initialization.draft(completer)
    second = initialization.draft(completer)
    assert first.failed == second.failed == {}
    assert len(client.received_prompts) == 1
    assert CapabilityContractWorkspace(initialization.contract_root).approved() == ()


def test_rejected_contract_can_be_redrafted_with_review_feedback(initialization):
    (first_job,) = initialization.prepare(action_ids=(PLACE_ACTION,))
    completer, _ = completer_with(draft_response())
    initialization.draft(completer)
    workspace = CapabilityContractWorkspace(initialization.contract_root)
    note = "Clarify the destination role."
    workspace.reject(workspace.candidates()[0].candidate_id, "human", note)

    (next_job,) = initialization.prepare(action_ids=(PLACE_ACTION,))
    assert next_job.job_id != first_job.job_id
    assert note in next_job.prompt
    assert not initialization.response_path(next_job).exists()


def test_realization_jobs_require_an_approved_contract(initialization):
    initialization.prepare(action_ids=(PLACE_ACTION,))
    completer, _ = completer_with(draft_response())
    initialization.draft(completer)
    assert initialization.prepare(action_ids=(PLACE_ACTION,)) == ()

    workspace = CapabilityContractWorkspace(initialization.contract_root)
    workspace.approve(
        workspace.candidates()[0].candidate_id, "human", initialization.actions()
    )
    (job,) = initialization.prepare(action_ids=(PLACE_ACTION,))
    assert job.kind is InitializationKind.REALIZATION
    assert job.contract_uid == workspace.approved_contracts()[0].uid

    initialization.response_path(job).write_text(
        (
            Path(__file__).parent / "dataset" / "initialization_placement.json"
        ).read_text()
    )
    report = initialization.import_responses()
    assert report.failed == {}
    realizations = CoraplexRealizationWorkspace(initialization.realization_root)
    (candidate,) = realizations.candidates()
    assert realizations.approved() == ()
    approval = approve_realization_candidate(
        realizations, candidate.candidate_id, "human", workspace.approved_contracts()
    )
    assert approval.realization.capability_uid == job.contract_uid
    assert initialization.prepare(action_ids=(PLACE_ACTION,)) == ()


def test_external_grounding_uses_the_same_source_validator(initialization):
    needed = request()
    specification = GroundingRequest(
        proposed_uid=needed.proposed_uid,
        semantic_name=needed.semantic_name,
        meaning=needed.meaning,
        roles=[
            {"name": role.name, "symbol_type": role.symbol_type.python_type_ref}
            for role in needed.roles
        ],
    )
    jobs = initialization.prepare(
        requests=(specification,), kind=InitializationKind.GROUNDING
    )
    (job,) = jobs
    initialization.response_path(job).write_text(
        grounding_response("invalid_factory.py")
    )
    refused = initialization.import_responses()
    assert job.job_id in refused.failed
    assert initialization.grounding().workspace.candidates() == ()

    initialization.response_path(job).write_text(grounding_response("valid_factory.py"))
    admitted = initialization.import_responses()
    assert admitted.failed == {}
    assert len(initialization.grounding().workspace.candidates()) == 1
    assert initialization.grounding().workspace.specifications() == ()
