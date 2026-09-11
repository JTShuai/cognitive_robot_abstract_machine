"""
Initialization joins scanned Coraplex actions with reviewed semantics.
"""

from resym.platform.coraplex_catalog import (
    CapabilityReviewStatus,
    coraplex_capability_catalog,
    discover_coraplex_capability_contract_drafts,
    initialize_coraplex_capabilities,
)


def test_initialization_covers_every_scanned_action() -> None:
    drafts = discover_coraplex_capability_contract_drafts()
    initialization = initialize_coraplex_capabilities()

    assert {record.draft.source_id for record in initialization.records} == {
        draft.source_id for draft in drafts
    }


def test_reviewed_action_semantics_are_admitted_once_at_initialization() -> None:
    initialization = initialize_coraplex_capabilities()
    open_records = tuple(
        record
        for record in initialization.records
        if record.draft.action_class.endswith(".OpenAction")
    )

    assert len(open_records) == 1
    assert open_records[0].status is CapabilityReviewStatus.APPROVED
    assert open_records[0].contract is not None
    assert open_records[0].reviewed_by == "platform-maintainer"


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

    initialization = initialize_coraplex_capabilities(package_root=tmp_path)

    assert len(initialization.records) == 1
    record = initialization.records[0]
    assert record.status is CapabilityReviewStatus.PENDING
    assert record.contract is None
    assert initialization.approved_contracts == ()

    catalog = coraplex_capability_catalog(package_root=tmp_path)
    assert catalog["summary"]["pending_review"] == 1
    assert catalog["summary"]["ready"] == 0
    assert catalog["pending_actions"][0]["source_id"] == record.draft.source_id
