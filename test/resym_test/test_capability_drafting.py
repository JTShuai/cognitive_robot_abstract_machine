"""
Initialization drafts capability contracts for human review; it never approves them.
"""

from __future__ import annotations

import json

import pytest

from resym.interfaces.capability_drafting import (
    DRAFTING_AGENT_NAME,
    draft_capability_contract_candidates,
)
from resym.llm.client import ScriptedCompletionClient
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.platform.capability_contract_review import CapabilityContractWorkspace
from resym.platform.coraplex_catalog import discover_coraplex_capability_contract_drafts
from resym.platform.coraplex_realizations import CapabilityReviewStatus

from resym.retrieval.ontology import OntologyIndex

from resym.retrieval.ontology_installation import ONTOLOGY_DIRECTORY

from .dataset.capability_model import OBJECT_TYPE

PLACING = "http://www.ease-crc.org/ont/SOMA.owl#Placing"

PLACE_ACTION = "coraplex:robot_plans.actions.core.placing.place-action"


def place_draft():
    return next(
        draft
        for draft in discover_coraplex_capability_contract_drafts()
        if draft.source_id == PLACE_ACTION
    )


def draft_response(**overrides) -> str:
    proposal = {
        "uid": "resym:ObjectPlacement",
        "label": "manipulation.place",
        "roles": [
            {"name": "actor", "accepted_symbol_types": [OBJECT_TYPE.python_type_ref]},
            {"name": "patient", "accepted_symbol_types": [OBJECT_TYPE.python_type_ref]},
            {
                "name": "destination",
                "accepted_symbol_types": [OBJECT_TYPE.python_type_ref],
            },
        ],
        "success_relation": "placed_at(patient, destination)",
        "verifiable_effects": ["placed-at"],
        "effect_role_values": [],
        "rationale": "PlaceAction puts the patient at the destination",
    }
    proposal.update(overrides)
    return json.dumps(proposal)


def completer_with(*responses: str):
    client = ScriptedCompletionClient(responses=list(responses))
    return StructuredCompleter(client=client, transcript=TranscriptRecorder()), client


def test_valid_draft_becomes_a_pending_contract_candidate(tmp_path):
    workspace = CapabilityContractWorkspace(tmp_path)
    completer, _ = completer_with(draft_response())

    report = draft_capability_contract_candidates(
        (place_draft(),), workspace, (), completer
    )

    (candidate,) = workspace.candidates()
    assert report.submitted == (candidate,)
    assert report.failures == ()
    assert candidate.contract.uid == "resym:ObjectPlacement"
    assert candidate.action_source_ids == (PLACE_ACTION,)
    assert candidate.generated_by == DRAFTING_AGENT_NAME
    assert candidate.review_status is CapabilityReviewStatus.PENDING
    assert workspace.approved_contracts() == ()  # drafting never approves


def test_rejected_draft_is_retried_with_its_objections(tmp_path):
    workspace = CapabilityContractWorkspace(tmp_path)
    completer, client = completer_with(
        draft_response(effect_role_values=[["stacked", "destination", "TOP"]]),
        draft_response(),
    )

    report = draft_capability_contract_candidates(
        (place_draft(),), workspace, (), completer
    )

    assert len(report.submitted) == 1
    assert "stacked" in client.received_prompts[1]


def test_exhausted_attempts_report_a_failure_without_a_candidate(tmp_path):
    workspace = CapabilityContractWorkspace(tmp_path)
    bad = draft_response(effect_role_values=[["stacked", "destination", "TOP"]])
    completer, _ = completer_with(bad, bad)

    report = draft_capability_contract_candidates(
        (place_draft(),), workspace, (), completer, maximum_attempts=2
    )

    assert report.submitted == ()
    (failure,) = report.failures
    assert failure.action_source_id == PLACE_ACTION
    assert failure.objections
    assert workspace.candidates() == ()


def test_actions_already_covered_by_a_contract_are_not_drafted(tmp_path):
    from resym.core.capability_model import CapabilityContract, CapabilityRole

    workspace = CapabilityContractWorkspace(tmp_path)
    completer, client = completer_with()
    covering = CapabilityContract(
        uid="resym:ObjectPlacement",
        label="manipulation.place",
        roles=(CapabilityRole("patient", accepted_symbol_types=(OBJECT_TYPE,)),),
        success_relation="placed_at(patient)",
        verifiable_effects=("placed-at",),
    )

    report = draft_capability_contract_candidates(
        (place_draft(),),
        workspace,
        (covering,),
        completer,
        covered_action_source_ids=frozenset({PLACE_ACTION}),
    )

    assert report.submitted == () and report.failures == ()
    assert client.received_prompts == []


@pytest.fixture(scope="module")
def ontology_index() -> OntologyIndex:
    if not (ONTOLOGY_DIRECTORY / "manifest.json").is_file():
        pytest.skip("ontology data is not installed")
    return OntologyIndex.from_directory(ONTOLOGY_DIRECTORY)


def alignment_submission(iri: str) -> str:
    return json.dumps(
        {
            "tool": "submit_alignment",
            "rationale": "Placing an object at a destination is SOMA placing.",
            "proposal": {
                "candidate_iri": iri,
                "relation": "SPECIALIZATION",
                "role_mapping": {},
                "cited_classes": [iri],
                "cited_properties": [],
            },
        }
    )


def test_accepted_draft_is_aligned_to_the_frozen_ontology_before_submission(
    tmp_path, ontology_index
):
    workspace = CapabilityContractWorkspace(tmp_path)
    completer, client = completer_with(draft_response(), alignment_submission(PLACING))

    report = draft_capability_contract_candidates(
        (place_draft(),), workspace, (), completer, ontology_index=ontology_index
    )

    (candidate,) = report.submitted
    assert report.unaligned == ()
    alignment = candidate.contract.ontology_alignment
    assert alignment.target_iri == PLACING
    assert alignment.decided_by == "llm-agent"
    assert alignment.source_version == "soma@2.1.0"
    assert "manipulation.place" in client.received_prompts[1]  # aligner saw the draft
    assert workspace.candidates()[0].contract.ontology_alignment == alignment


def test_draft_the_aligner_cannot_place_is_submitted_unaligned(
    tmp_path, ontology_index
):
    workspace = CapabilityContractWorkspace(tmp_path)
    completer, _ = completer_with(
        draft_response(),
        alignment_submission("http://www.ease-crc.org/ont/SOMA.owl#NoSuchClass"),
    )

    report = draft_capability_contract_candidates(
        (place_draft(),),
        workspace,
        (),
        completer,
        ontology_index=ontology_index,
        maximum_alignment_steps=1,
    )

    (candidate,) = report.submitted
    (unaligned,) = report.unaligned
    assert unaligned.candidate_id == candidate.candidate_id
    assert "NoSuchClass" in unaligned.reason
    assert candidate.contract.ontology_alignment.target_iri is None
