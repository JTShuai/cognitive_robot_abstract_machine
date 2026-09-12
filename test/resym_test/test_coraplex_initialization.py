"""
Initialization joins scanned Coraplex actions with the realizations a workspace admits.
"""

from resym.platform.coraplex_catalog import (
    coraplex_capability_catalog,
    discover_coraplex_capability_contract_drafts,
    initialize_coraplex_capabilities,
)
from resym.platform.coraplex_realizations import CapabilityReviewStatus

from .dataset.capability_model import capability_contracts


def test_initialization_covers_every_scanned_action(capability_initialization) -> None:
    drafts = discover_coraplex_capability_contract_drafts()

    assert {record.draft.source_id for record in capability_initialization.records} == {
        draft.source_id for draft in drafts
    }


def test_admitted_realization_is_registered_once_with_its_reviewer(
    capability_initialization,
) -> None:
    open_records = tuple(
        record
        for record in capability_initialization.records
        if record.draft.action_class.endswith(".OpenAction")
    )

    assert len(open_records) == 1
    assert open_records[0].status is CapabilityReviewStatus.APPROVED
    assert open_records[0].contract is not None
    assert open_records[0].reviewed_by == "test-bootstrap"


def test_unreviewed_action_stays_out_of_the_runtime_catalog(tmp_path) -> None:
    actions = tmp_path / "robot_plans" / "actions"
    actions.mkdir(parents=True)
    (actions / "custom.py").write_text(
        "from dataclasses import dataclass\n"
        "class ActionDescription: pass\n"
        "@dataclass\n"
        "class CustomAction(ActionDescription):\n"
        "    target: object\n"
    )

    initialization = initialize_coraplex_capabilities(
        capability_contracts(), None, package_root=tmp_path
    )

    assert len(initialization.records) == 1
    record = initialization.records[0]
    assert record.status is CapabilityReviewStatus.PENDING
    assert record.contract is None
    assert initialization.approved_contracts == ()

    catalog = coraplex_capability_catalog(initialization)
    assert catalog["summary"]["pending_review"] == 1
    assert catalog["summary"]["ready"] == 0
    assert catalog["pending_actions"][0]["source_id"] == record.draft.source_id
