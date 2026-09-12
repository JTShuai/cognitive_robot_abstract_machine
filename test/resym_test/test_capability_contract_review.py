"""
Capability contracts enter the library only through the local review workspace.
"""

import pytest

from resym.core.capability_model import CapabilityContract, CapabilityRole
from resym.platform.capability_contract_review import (
    CapabilityContractCandidate,
    CapabilityContractWorkspace,
    ContractReviewError,
)
from resym.platform.coraplex_catalog import discover_coraplex_capability_contract_drafts
from resym.platform.coraplex_realizations import CapabilityReviewStatus

from .dataset.capability_model import OBJECT_TYPE, PATIENT_ROLE

REVIEWER = "tester"
PLACE_ACTION = "coraplex:robot_plans.actions.core.placing.place-action"


def placement_contract(uid: str = "resym:ObjectPlacement") -> CapabilityContract:
    return CapabilityContract(
        uid=uid,
        label="manipulation.place",
        roles=(
            CapabilityRole("actor", accepted_symbol_types=(OBJECT_TYPE,)),
            CapabilityRole(PATIENT_ROLE, accepted_symbol_types=(OBJECT_TYPE,)),
            CapabilityRole("destination", accepted_symbol_types=(OBJECT_TYPE,)),
        ),
        success_relation="placed_at(patient, destination)",
        verifiable_effects=("placed-at",),
    )


def candidate(
    contract: CapabilityContract | None = None,
    candidate_id: str = "place-contract",
    action_source_ids: tuple[str, ...] = (PLACE_ACTION,),
) -> CapabilityContractCandidate:
    return CapabilityContractCandidate(
        candidate_id=candidate_id,
        contract=placement_contract() if contract is None else contract,
        action_source_ids=action_source_ids,
        generated_by="test",
        rationale="PlaceAction puts the patient at the destination",
    )


@pytest.fixture
def workspace(tmp_path):
    return CapabilityContractWorkspace(tmp_path / "contracts")


def drafts():
    return discover_coraplex_capability_contract_drafts()


def test_submitted_contract_enters_the_library_only_after_approval(workspace):
    workspace.submit(candidate())

    assert workspace.approved_contracts() == ()
    (pending,) = workspace.candidates()
    assert pending.review_status is CapabilityReviewStatus.PENDING

    approved = workspace.approve("place-contract", REVIEWER, drafts())

    assert workspace.approved_contracts() == (placement_contract(),)
    assert approved.reviewed_by == REVIEWER
    (reviewed,) = workspace.candidates()
    assert reviewed.review_status is CapabilityReviewStatus.APPROVED
    assert reviewed.reviewed_by == REVIEWER


def test_approval_refuses_a_uid_without_a_namespace(workspace):
    workspace.submit(candidate(placement_contract(uid="ObjectPlacement")))

    with pytest.raises(ContractReviewError) as refusal:
        workspace.approve("place-contract", REVIEWER, drafts())

    assert any("namespace" in objection for objection in refusal.value.objections)
    assert workspace.approved_contracts() == ()


def test_approval_refuses_an_unknown_native_action_as_evidence(workspace):
    workspace.submit(candidate(action_source_ids=("coraplex:no.such-action",)))

    with pytest.raises(ContractReviewError) as refusal:
        workspace.approve("place-contract", REVIEWER, drafts())

    assert any("no.such-action" in objection for objection in refusal.value.objections)


def test_same_contract_identity_cannot_be_approved_twice(workspace):
    workspace.submit(candidate())
    workspace.approve("place-contract", REVIEWER, drafts())
    workspace.submit(candidate(candidate_id="place-contract-again"))

    with pytest.raises(ContractReviewError):
        workspace.approve("place-contract-again", REVIEWER, drafts())

    assert len(workspace.approved_contracts()) == 1


def test_rejection_keeps_the_contract_out_of_the_library(workspace):
    workspace.submit(candidate())

    rejected = workspace.reject("place-contract", REVIEWER, "roles are too coarse")

    assert rejected.review_status is CapabilityReviewStatus.REJECTED
    assert rejected.review_note == "roles are too coarse"
    assert workspace.approved_contracts() == ()
