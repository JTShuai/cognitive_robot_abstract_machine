"""
Coraplex capability discovery without loading the ROS runtime.
"""

from resym.core.symbol_types import SymbolType
from resym.platform.coraplex_realizations import ContextValue, ParameterSourceKind
from semantic_digital_twin.robots.robot_parts import (
    Arm,
    Camera,
    EndEffector,
    MobileBase,
    Torso,
)
from resym.platform.coraplex_catalog import (
    CapabilityRealizationStatus,
    available_coraplex_capabilities,
    coraplex_capability_catalog,
    discover_coraplex_capability_contract_drafts,
    infer_coraplex_capability_support,
    initialize_coraplex_capabilities,
    realization_evidence,
    robot_resources,
)
from resym.platform.coraplex_realizations import CoraplexRealizationWorkspace

from .dataset.capability_model import (
    ARTICULATION_CAPABILITY_UID,
    BASE_NAVIGATION_CAPABILITY_UID,
    INTERACTION_NAVIGATION_CAPABILITY_UID,
    capability_contracts,
)
from .dataset.resource_bearing_robot import Robot


def test_discovers_public_coraplex_actions_as_contract_drafts():
    drafts = discover_coraplex_capability_contract_drafts()
    by_class = {draft.action_class.rsplit(".", 1)[-1]: draft for draft in drafts}

    for expected in (
        "NavigateAction",
        "OpenAction",
        "CloseAction",
        "PickUpAction",
        "PlaceAction",
        "PouringAction",
        "WipingAction",
        "ElevatorNavigation",
    ):
        assert expected in by_class

    open_draft = by_class["OpenAction"]
    assert open_draft.source_id.endswith("core.container.open-action")
    assert [parameter.name for parameter in open_draft.parameters[:2]] == [
        "object_designator",
        "arm",
    ]
    assert open_draft.declares_precondition
    assert open_draft.declares_postcondition
    assert "semantic capability UID" in open_draft.missing_semantics


def test_field_without_default_is_still_a_required_parameter():
    drafts = discover_coraplex_capability_contract_drafts()
    transport = next(
        draft for draft in drafts if draft.action_class.endswith(".TransportAction")
    )
    object_parameter = next(
        parameter
        for parameter in transport.parameters
        if parameter.name == "object_designator"
    )
    assert object_parameter.required


def test_every_admitted_realization_carries_capability_and_resource_evidence(
    capability_initialization,
):
    draft_ids = {
        draft.source_id for draft in discover_coraplex_capability_contract_drafts()
    }
    evidence = realization_evidence(capability_initialization)

    assert evidence
    assert {item.action_source_id for item in evidence} <= draft_ids
    assert {item.capability_uid for item in evidence} <= {
        contract.uid for contract in capability_contracts()
    }
    assert all(item.required_resources for item in evidence)


def test_robot_resources_come_from_the_semantic_robot_structure():
    resources = robot_resources(Robot(mobile=True))

    assert {
        SymbolType.from_python_type(part)
        for part in (MobileBase, Arm, EndEffector, Camera, Torso)
    } <= resources
    assert all(
        item.python_type_ref.startswith("semantic_digital_twin.") for item in resources
    )


def test_fixed_robot_does_not_claim_mobile_capabilities(capability_initialization):
    support = {
        item.capability_uid: item
        for item in infer_coraplex_capability_support(
            Robot(mobile=False), capability_initialization
        )
    }

    assert (
        support[BASE_NAVIGATION_CAPABILITY_UID].status
        is CapabilityRealizationStatus.ROBOT_INCOMPATIBLE
    )
    assert (
        support[INTERACTION_NAVIGATION_CAPABILITY_UID].status
        is CapabilityRealizationStatus.ROBOT_INCOMPATIBLE
    )
    assert (
        support[ARTICULATION_CAPABILITY_UID].status is CapabilityRealizationStatus.READY
    )


def test_available_capabilities_are_derived_directly_from_the_robot(
    capability_initialization,
):
    capabilities = available_coraplex_capabilities(
        Robot(mobile=False), capability_initialization
    )

    assert ARTICULATION_CAPABILITY_UID in capabilities
    assert BASE_NAVIGATION_CAPABILITY_UID not in capabilities
    assert INTERACTION_NAVIGATION_CAPABILITY_UID not in capabilities


def test_nothing_is_available_before_any_realization_is_admitted(tmp_path):
    unreviewed = initialize_coraplex_capabilities(
        capability_contracts(), CoraplexRealizationWorkspace(tmp_path / "workspace")
    )

    assert (
        available_coraplex_capabilities(Robot(mobile=True), unreviewed) == frozenset()
    )


def test_catalog_exposes_all_contracts_actions_and_readiness(
    capability_initialization,
):
    catalog = coraplex_capability_catalog(capability_initialization)
    contracts = len(capability_contracts())

    assert catalog["summary"] == {
        "contracts": contracts,
        "actions": len(discover_coraplex_capability_contract_drafts()),
        "ready": contracts,
        "task_verification_required": contracts,
        "pending_review": len(discover_coraplex_capability_contract_drafts())
        - len(
            {
                record.draft.source_id
                for record in capability_initialization.records
                if record.contract is not None
            }
        ),
    }
    entries = {entry["contract"]["uid"]: entry for entry in catalog["contracts"]}
    assert entries[ARTICULATION_CAPABILITY_UID]["realization"]["status"] == "ready"
    assert entries[BASE_NAVIGATION_CAPABILITY_UID]["realization"]["status"] == "ready"
    assert (
        entries[BASE_NAVIGATION_CAPABILITY_UID]["realization"]["effect_verification"]
        == "task-required"
    )
    assert entries[BASE_NAVIGATION_CAPABILITY_UID]["realization"]["actions"]


def test_reviewed_realizations_reference_only_declared_roles_and_context_values(
    capability_initialization,
):
    """
    A parameter source names a role the contract declares, a value the context can
    supply, or a constant; nothing else can be filled at runtime.
    """
    for record in capability_initialization.records:
        if not record.parameter_sources:
            continue
        declared_roles = {role.name for role in record.contract.roles}
        for source in record.parameter_sources:
            if source.kind is ParameterSourceKind.ROLE:
                assert source.value in declared_roles, (record.draft.source_id, source)
            elif source.kind is ParameterSourceKind.CONTEXT:
                assert ContextValue(source.value), (record.draft.source_id, source)
        if record.applies_when is not None:
            assert record.applies_when.role in declared_roles, record.draft.source_id
