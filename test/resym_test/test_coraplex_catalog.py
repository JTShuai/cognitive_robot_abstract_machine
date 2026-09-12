"""
Coraplex capability discovery without loading the ROS runtime.
"""

from resym.platform.coraplex_catalog import (
    CapabilityRealizationStatus,
    RobotResource,
    coraplex_capability_catalog,
    coraplex_capability_realization_evidence,
    coraplex_embodiment_profile,
    discover_coraplex_capability_contract_drafts,
    infer_coraplex_capability_support,
    robot_resources,
)
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    BASE_NAVIGATION_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    capability_contracts,
)
from resym.platform.embodiment import ToolOrientation
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.world_entity import Body


class ForwardCamera(Camera):
    """
    A constructible camera sensor for exercising resource discovery.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        raise NotImplementedError

    def setup_hardware_interfaces(self):
        raise NotImplementedError

    def setup_joint_states(self):
        raise NotImplementedError


def forward_camera() -> ForwardCamera:
    return ForwardCamera(
        root=Body(name=PrefixedName("camera")),
        forward_facing_axis=Vector3.from_iterable((1.0, 0.0, 0.0)),
        field_of_view=FieldOfView(),
    )


class Robot:
    """
    Mimics the resource-bearing surface of ``AbstractRobot``.
    """

    def __init__(self, *, mobile: bool):
        self.drive = object() if mobile else None
        self.torso = object()

    def get_arms(self):
        return [object()]

    def get_end_effectors(self):
        return [object()]

    def get_sensors(self):
        return [forward_camera()]

    def get_torso_if_specified(self):
        return self.torso


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


def test_every_discovered_action_has_reviewed_capability_and_resource_evidence():
    draft_ids = {
        draft.source_id for draft in discover_coraplex_capability_contract_drafts()
    }
    evidence = coraplex_capability_realization_evidence()

    assert {item.action_source_id for item in evidence} == draft_ids
    assert {item.capability_uid for item in evidence}.issubset(
        {contract.uid for contract in capability_contracts()}
    )


def test_robot_resources_come_from_the_semantic_robot_structure():
    resources = robot_resources(Robot(mobile=True))

    assert resources == frozenset(
        {
            RobotResource.MOBILE_BASE,
            RobotResource.ARM,
            RobotResource.END_EFFECTOR,
            RobotResource.CAMERA,
            RobotResource.TORSO,
        }
    )


def test_fixed_robot_does_not_claim_mobile_capabilities():
    support = {
        item.capability_uid: item
        for item in infer_coraplex_capability_support(Robot(mobile=False))
    }

    assert (
        support[BASE_NAVIGATION_CAPABILITY_UID].status
        is CapabilityRealizationStatus.ROBOT_INCOMPATIBLE
    )
    assert (
        support[NAVIGATION_CAPABILITY_UID].status
        is CapabilityRealizationStatus.ROBOT_INCOMPATIBLE
    )
    assert ARTICULATION_CAPABILITY_UID in support


def test_profile_admits_only_ready_capabilities_and_keeps_their_sources():
    profile = coraplex_embodiment_profile(
        name="test-robot",
        robot=Robot(mobile=True),
        ready_capability_uids={ARTICULATION_CAPABILITY_UID},
        tool_orientation=ToolOrientation.BASE_ALIGNED,
    )

    assert profile.capabilities == frozenset({ARTICULATION_CAPABILITY_UID})
    assert BASE_NAVIGATION_CAPABILITY_UID not in profile.capabilities
    assert dict(profile.capability_sources)[ARTICULATION_CAPABILITY_UID]


def test_catalog_exposes_all_contracts_actions_and_readiness():
    catalog = coraplex_capability_catalog()

    assert catalog["summary"] == {
        "contracts": 20,
        "actions": len(discover_coraplex_capability_contract_drafts()),
        "ready": 20,
        "built_in_verification": 2,
        "pending_review": 0,
    }
    entries = {entry["contract"]["uid"]: entry for entry in catalog["contracts"]}
    assert entries[ARTICULATION_CAPABILITY_UID]["realization"]["status"] == "ready"
    assert entries[BASE_NAVIGATION_CAPABILITY_UID]["realization"]["status"] == "ready"
    assert (
        entries[BASE_NAVIGATION_CAPABILITY_UID]["realization"]["effect_verification"]
        == "task-required"
    )
    assert entries[BASE_NAVIGATION_CAPABILITY_UID]["realization"]["actions"]
