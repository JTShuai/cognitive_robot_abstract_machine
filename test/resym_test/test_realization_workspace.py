"""
Capability realizations become executable only through the local review workspace.
"""

import hashlib

import pytest

from resym.platform.coraplex_catalog import (
    discover_coraplex_capability_contract_drafts,
    initialize_coraplex_capabilities,
)
from resym.platform.coraplex_realizations import (
    ActionRealization,
    CapabilityReviewStatus,
    CoraplexRealizationWorkspace,
    ParameterSource,
    ParameterSourceKind,
    RealizationCandidate,
    RealizationReviewError,
)

from .dataset.capability_model import (
    BASE_NAVIGATION_CAPABILITY_UID,
    capability_contracts,
    reference_realizations,
)

REVIEWER = "tester"


@pytest.fixture
def custom_action_package(tmp_path):
    """
    A scannable package holding one native action with a pose-typed parameter.
    """
    actions = tmp_path / "package" / "robot_plans" / "actions"
    actions.mkdir(parents=True)
    (actions / "custom.py").write_text(
        "from dataclasses import dataclass\n"
        "class ActionDescription: pass\n"
        "@dataclass\n"
        "class CustomAction(ActionDescription):\n"
        "    target_location: object\n"
    )
    return tmp_path / "package"


@pytest.fixture
def workspace(tmp_path):
    return CoraplexRealizationWorkspace(tmp_path / "workspace")


def custom_candidate(role: str = "destination") -> RealizationCandidate:
    return RealizationCandidate(
        candidate_id="custom-navigation",
        action_source_id="coraplex:robot_plans.actions.custom.custom-action",
        realization=ActionRealization(
            BASE_NAVIGATION_CAPABILITY_UID,
            (ParameterSource("target_location", ParameterSourceKind.ROLE, role),),
        ),
        generated_by="test",
        rationale="navigation realized by a custom action",
    )


def approved_records(initialization):
    return [
        record
        for record in initialization.records
        if record.status is CapabilityReviewStatus.APPROVED
    ]


def test_submitted_realization_is_admitted_only_after_approval(
    workspace, custom_action_package
):
    workspace.submit(custom_candidate())

    pending = initialize_coraplex_capabilities(
        capability_contracts(), workspace, package_root=custom_action_package
    )
    assert approved_records(pending) == []
    assert pending.records[0].status is CapabilityReviewStatus.PENDING

    approved = workspace.approve(
        "custom-navigation",
        reviewer=REVIEWER,
        drafts=discover_coraplex_capability_contract_drafts(custom_action_package),
        contracts=capability_contracts(),
        package_root=custom_action_package,
    )

    admitted = initialize_coraplex_capabilities(
        capability_contracts(), workspace, package_root=custom_action_package
    )
    (record,) = approved_records(admitted)
    assert record.contract.uid == BASE_NAVIGATION_CAPABILITY_UID
    assert record.parameter_sources == custom_candidate().realization.parameter_sources
    assert record.reviewed_by == REVIEWER
    source_file = custom_action_package / "robot_plans" / "actions" / "custom.py"
    assert (
        approved.action_checksum == hashlib.sha256(source_file.read_bytes()).hexdigest()
    )
    assert workspace.candidates()[0].review_status is CapabilityReviewStatus.APPROVED


def test_approval_refuses_a_role_the_contract_does_not_declare(
    workspace, custom_action_package
):
    workspace.submit(custom_candidate(role="nowhere"))

    with pytest.raises(RealizationReviewError):
        workspace.approve(
            "custom-navigation",
            reviewer=REVIEWER,
            drafts=discover_coraplex_capability_contract_drafts(custom_action_package),
            contracts=capability_contracts(),
            package_root=custom_action_package,
        )

    assert workspace.approved() == ()
    assert workspace.candidates()[0].review_status is CapabilityReviewStatus.PENDING


def test_changed_action_source_withdraws_the_approval(workspace, custom_action_package):
    workspace.submit(custom_candidate())
    workspace.approve(
        "custom-navigation",
        reviewer=REVIEWER,
        drafts=discover_coraplex_capability_contract_drafts(custom_action_package),
        contracts=capability_contracts(),
        package_root=custom_action_package,
    )
    source_file = custom_action_package / "robot_plans" / "actions" / "custom.py"
    source_file.write_text(source_file.read_text() + "    extra: object = None\n")
    discover_coraplex_capability_contract_drafts.cache_clear()

    initialization = initialize_coraplex_capabilities(
        capability_contracts(), workspace, package_root=custom_action_package
    )

    (record,) = initialization.records
    assert record.status is CapabilityReviewStatus.SOURCE_CHANGED
    assert initialization.approved_contracts == ()


def test_bootstrapped_workspace_admits_every_reference_realization(
    capability_initialization,
):
    admitted = {
        (record.draft.action_class.rsplit(".", 1)[-1], record.contract.uid)
        for record in approved_records(capability_initialization)
    }
    expected = {
        (action_name, realization.capability_uid)
        for action_name, realizations in reference_realizations().items()
        for realization in realizations
    }

    assert admitted == expected
