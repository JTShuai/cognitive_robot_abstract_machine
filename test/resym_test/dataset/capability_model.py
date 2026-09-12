"""
Test-owned capability contracts and their reviewed realizations.

A small reviewed catalog covering every way a native parameter can be filled: bodies,
poses, semantic entities, enums from roles or constants, context values (manipulation
arm, default grasp, witness base pose), constant-role dispatch, and resource dispatch.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from resym.core.capability_model import CapabilityContract, CapabilityRole
from resym.core.provenance import OntologyAlignment
from resym.core.symbol_types import SymbolType
from resym.platform.coraplex_catalog import (
    CoraplexCapabilityInitialization,
    discover_coraplex_capability_contract_drafts,
    initialize_coraplex_capabilities,
    installed_coraplex_root,
)
from resym.platform.coraplex_realizations import (
    ActionRealization,
    ContextValue,
    CoraplexRealizationWorkspace,
    ParameterSource,
    ParameterSourceKind,
    RealizationCandidate,
    RoleCondition,
)
from semantic_digital_twin.robots.robot_parts import (
    Arm,
    EndEffector,
    MobileBase,
    Torso,
)
from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Agent,
    Handle,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

AGENT_TYPE = SymbolType.from_python_type(Agent)
ARTICULATED_PART_TYPE = SymbolType.from_python_type(HasMechanicalJoint)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
OBJECT_TYPE = SymbolType.from_python_type(SemanticAnnotation)

INTERACTION_NAVIGATION_CAPABILITY_UID = "resym:ReachInteraction"
ARTICULATION_CAPABILITY_UID = "resym:ArticulationStateChange"
BASE_NAVIGATION_CAPABILITY_UID = "resym:BaseNavigation"
PICK_UP_CAPABILITY_UID = "resym:ObjectPickup"
PLACE_CAPABILITY_UID = "resym:ObjectPlacement"
TRANSPORT_CAPABILITY_UID = "resym:ObjectTransport"
GRIPPER_STATE_CAPABILITY_UID = "resym:GripperStateChange"
TORSO_STATE_CAPABILITY_UID = "resym:TorsoStateChange"
CARRY_POSTURE_CAPABILITY_UID = "resym:CarryPosture"
TOOL_PATH_CAPABILITY_UID = "resym:ToolPathFollowing"
POURING_CAPABILITY_UID = "resym:MaterialPouring"

ACTOR_ROLE = "actor"
PATIENT_ROLE = "patient"
TARGET_STATE_ROLE = "target_state"
INTERACTION_POINT_ROLE = "interaction_point"


class OpenCloseState(StrEnum):
    """
    The ``target_state`` values of the articulation capability.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"


SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"
SOMA_STATE_TRANSITION = SOMA + "StateTransition"
SOMA_OPENING = SOMA + "Opening"
SOMA_CLOSING = SOMA + "Closing"
SOMA_NAVIGATING = SOMA + "Navigating"


# %% contracts


def _object_role(name: str, *, required: bool = True) -> CapabilityRole:
    return CapabilityRole(name, required=required, accepted_symbol_types=(OBJECT_TYPE,))


def _actor_role() -> CapabilityRole:
    return CapabilityRole(ACTOR_ROLE, accepted_symbol_types=(AGENT_TYPE,))


def _constant_role(name: str, values: tuple[str, ...]) -> CapabilityRole:
    return CapabilityRole(name, allowed_values=values)


def _soma_alignment(concept: str, rationale: str) -> OntologyAlignment:
    iri = SOMA + concept
    return OntologyAlignment(
        relation="SPECIALIZATION",
        target_iri=iri,
        source_version="soma@2.1.0",
        decided_by="test-dataset",
        rationale=rationale,
        evidence_iris=(iri,),
    )


def interaction_navigation_capability_contract(
    version: str = "1",
) -> CapabilityContract:
    return CapabilityContract(
        uid=INTERACTION_NAVIGATION_CAPABILITY_UID,
        label="navigation.reach-interaction",
        version=version,
        roles=(_actor_role(), _object_role(PATIENT_ROLE)),
        success_relation="ready_to_interact(actor, patient)",
        verifiable_effects=("ready-to-interact",),
        ontology_alignment=OntologyAlignment(
            relation="SPECIALIZATION",
            target_iri=SOMA_NAVIGATING,
            source_version="soma@2.1.0",
            decided_by="llm-agent",
            decision_model="codex",
            rationale=(
                "The local capability is narrower than SOMA Navigating: it moves the "
                "robot to a witness pose from which it can interact with a task object."
            ),
            evidence_iris=(SOMA_NAVIGATING,),
        ),
    )


def articulation_capability_contract(version: str = "1") -> CapabilityContract:
    return CapabilityContract(
        uid=ARTICULATION_CAPABILITY_UID,
        label="articulation.state-transition",
        version=version,
        roles=(
            _actor_role(),
            CapabilityRole(
                PATIENT_ROLE, accepted_symbol_types=(ARTICULATED_PART_TYPE,)
            ),
            CapabilityRole(
                INTERACTION_POINT_ROLE,
                required=False,
                accepted_symbol_types=(HANDLE_TYPE,),
            ),
            _constant_role(
                TARGET_STATE_ROLE, (OpenCloseState.OPEN, OpenCloseState.CLOSED)
            ),
        ),
        success_relation="articulation_state(patient) == target_state",
        verifiable_effects=("opened", "closed"),
        effect_role_values=(
            ("opened", TARGET_STATE_ROLE, OpenCloseState.OPEN),
            ("closed", TARGET_STATE_ROLE, OpenCloseState.CLOSED),
        ),
        ontology_alignment=OntologyAlignment(
            relation="SPECIALIZATION",
            target_iri=SOMA_STATE_TRANSITION,
            source_version="soma@2.1.0",
            decided_by="llm-agent",
            decision_model="codex",
            rationale=(
                "The local capability is a state transition restricted to an "
                "articulated patient and the OPEN/CLOSED target states; SOMA Opening "
                "and Closing support the target-specific readings."
            ),
            evidence_iris=(SOMA_STATE_TRANSITION, SOMA_OPENING, SOMA_CLOSING),
        ),
    )


def capability_contracts(version: str = "1") -> tuple[CapabilityContract, ...]:
    """
    The reviewed contracts of the example task model.
    """
    common = (_actor_role(),)
    return (
        interaction_navigation_capability_contract(version),
        articulation_capability_contract(version),
        CapabilityContract(
            uid=BASE_NAVIGATION_CAPABILITY_UID,
            label="navigation.move-base",
            version=version,
            roles=common + (_object_role("destination"),),
            success_relation="at_location(actor, destination)",
            verifiable_effects=("at-location",),
            ontology_alignment=_soma_alignment("Navigating", "Base motion."),
        ),
        CapabilityContract(
            uid=PICK_UP_CAPABILITY_UID,
            label="manipulation.pick-up",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE),),
            success_relation="holding(actor, patient)",
            verifiable_effects=("holding",),
            ontology_alignment=_soma_alignment("PickingUp", "Grasping and lifting."),
        ),
        CapabilityContract(
            uid=PLACE_CAPABILITY_UID,
            label="manipulation.place",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE), _object_role("destination")),
            success_relation="placed_at(patient, destination)",
            verifiable_effects=("placed-at",),
            ontology_alignment=_soma_alignment("Placing", "Placing at a destination."),
        ),
        CapabilityContract(
            uid=TRANSPORT_CAPABILITY_UID,
            label="manipulation.transport",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE), _object_role("destination")),
            success_relation="transported_to(patient, destination)",
            verifiable_effects=("transported-to",),
            ontology_alignment=_soma_alignment("Transporting", "Moving an object."),
        ),
        CapabilityContract(
            uid=GRIPPER_STATE_CAPABILITY_UID,
            label="robot.gripper-state-transition",
            version=version,
            roles=common
            + (
                _constant_role("gripper", ("LEFT", "RIGHT", "BOTH")),
                _constant_role(TARGET_STATE_ROLE, ("OPEN", "CLOSE")),
            ),
            success_relation="gripper_state(actor, gripper) == target_state",
            verifiable_effects=("gripper-open", "gripper-closed"),
            effect_role_values=(
                ("gripper-open", TARGET_STATE_ROLE, "OPEN"),
                ("gripper-closed", TARGET_STATE_ROLE, "CLOSE"),
            ),
            ontology_alignment=_soma_alignment("StateTransition", "Gripper state."),
        ),
        CapabilityContract(
            uid=TORSO_STATE_CAPABILITY_UID,
            label="robot.torso-state-transition",
            version=version,
            roles=common + (_constant_role(TARGET_STATE_ROLE, ("LOW", "MID", "HIGH")),),
            success_relation="torso_state(actor) == target_state",
            verifiable_effects=("torso-low", "torso-mid", "torso-high"),
            effect_role_values=(
                ("torso-low", TARGET_STATE_ROLE, "LOW"),
                ("torso-mid", TARGET_STATE_ROLE, "MID"),
                ("torso-high", TARGET_STATE_ROLE, "HIGH"),
            ),
            ontology_alignment=_soma_alignment("PosturalMoving", "Torso posture."),
        ),
        CapabilityContract(
            uid=CARRY_POSTURE_CAPABILITY_UID,
            label="robot.adopt-carry-posture",
            version=version,
            roles=common + (_constant_role("arm", ("LEFT", "RIGHT", "BOTH")),),
            success_relation="carrying_posture(actor, arm)",
            verifiable_effects=("carrying-posture",),
            ontology_alignment=_soma_alignment("PosturalMoving", "Carry posture."),
        ),
        CapabilityContract(
            uid=TOOL_PATH_CAPABILITY_UID,
            label="manipulation.follow-tool-path",
            version=version,
            roles=common + (_object_role("tool"), _object_role("target")),
            success_relation="tool_path_completed(tool, target)",
            verifiable_effects=("tool-path-completed",),
            ontology_alignment=_soma_alignment("Reaching", "Following a tool path."),
        ),
        CapabilityContract(
            uid=POURING_CAPABILITY_UID,
            label="material.pour",
            version=version,
            roles=common + (_object_role("source"), _object_role("destination")),
            success_relation="poured_into(source, destination)",
            verifiable_effects=("poured-into",),
            ontology_alignment=_soma_alignment("Pouring", "Pouring material."),
        ),
    )


# %% realizations


def _role(parameter: str, role: str) -> ParameterSource:
    return ParameterSource(parameter, ParameterSourceKind.ROLE, role)


def _context(parameter: str, value: ContextValue) -> ParameterSource:
    return ParameterSource(parameter, ParameterSourceKind.CONTEXT, value)


_ARM = _context("arm", ContextValue.MANIPULATION_ARM)
_GRASP = _context("grasp_description", ContextValue.DEFAULT_GRASP)
_MOBILE = (SymbolType.from_python_type(MobileBase),)
_MANIPULATOR = (
    SymbolType.from_python_type(Arm),
    SymbolType.from_python_type(EndEffector),
)


def reference_realizations() -> dict[str, tuple[ActionRealization, ...]]:
    """
    The reviewed realizations by native action class name.
    """
    return {
        "NavigateAction": (
            ActionRealization(
                BASE_NAVIGATION_CAPABILITY_UID,
                (_role("target_location", "destination"),),
                required_resources=_MOBILE,
            ),
            ActionRealization(
                INTERACTION_NAVIGATION_CAPABILITY_UID,
                (
                    ParameterSource(
                        "target_location",
                        ParameterSourceKind.CONTEXT,
                        ContextValue.WITNESS_BASE_POSE,
                        key_roles=(ACTOR_ROLE, PATIENT_ROLE),
                    ),
                ),
                required_resources=_MOBILE,
            ),
        ),
        "OpenAction": (
            ActionRealization(
                ARTICULATION_CAPABILITY_UID,
                (_role("object_designator", INTERACTION_POINT_ROLE), _ARM),
                RoleCondition(TARGET_STATE_ROLE, OpenCloseState.OPEN),
                _MANIPULATOR,
            ),
        ),
        "CloseAction": (
            ActionRealization(
                ARTICULATION_CAPABILITY_UID,
                (_role("object_designator", INTERACTION_POINT_ROLE), _ARM),
                RoleCondition(TARGET_STATE_ROLE, OpenCloseState.CLOSED),
                _MANIPULATOR,
            ),
        ),
        "PickUpAction": (
            ActionRealization(
                PICK_UP_CAPABILITY_UID,
                (_role("object_designator", PATIENT_ROLE), _ARM, _GRASP),
                required_resources=_MANIPULATOR,
            ),
        ),
        "PlaceAction": (
            ActionRealization(
                PLACE_CAPABILITY_UID,
                (
                    _role("object_designator", PATIENT_ROLE),
                    _role("target_location", "destination"),
                    _ARM,
                ),
                required_resources=_MANIPULATOR,
            ),
        ),
        "TransportAction": (
            ActionRealization(
                TRANSPORT_CAPABILITY_UID,
                (
                    _role("object_designator", PATIENT_ROLE),
                    _role("target_location", "destination"),
                    _ARM,
                ),
                required_resources=_MOBILE + _MANIPULATOR,
            ),
        ),
        "PickAndPlaceAction": (
            ActionRealization(
                TRANSPORT_CAPABILITY_UID,
                (
                    _role("object_designator", PATIENT_ROLE),
                    _role("target_location", "destination"),
                    _ARM,
                    _GRASP,
                ),
                required_resources=_MANIPULATOR,
            ),
        ),
        "SetGripperAction": (
            ActionRealization(
                GRIPPER_STATE_CAPABILITY_UID,
                (_role("gripper", "gripper"), _role("motion", TARGET_STATE_ROLE)),
                required_resources=(SymbolType.from_python_type(EndEffector),),
            ),
        ),
        "MoveTorsoAction": (
            ActionRealization(
                TORSO_STATE_CAPABILITY_UID,
                (_role("torso_state", TARGET_STATE_ROLE),),
                required_resources=(SymbolType.from_python_type(Torso),),
            ),
        ),
        "CarryAction": (
            ActionRealization(
                CARRY_POSTURE_CAPABILITY_UID,
                (_role("arm", "arm"),),
                required_resources=(SymbolType.from_python_type(Arm),),
            ),
        ),
        "FollowToolCenterPointPathAction": (
            ActionRealization(
                TOOL_PATH_CAPABILITY_UID,
                (_role("target_locations", "target"), _ARM),
                required_resources=_MANIPULATOR,
            ),
        ),
        "PouringAction": (
            ActionRealization(
                POURING_CAPABILITY_UID,
                (
                    _ARM,
                    _role("source_container", "source"),
                    _role("target_container", "destination"),
                ),
                required_resources=_MANIPULATOR,
            ),
        ),
    }


def bootstrap_capability_realizations(
    workspace_root: Path,
    package_root: Path | None = None,
    reviewer: str = "test-bootstrap",
) -> CoraplexCapabilityInitialization:
    """
    Approve the reference realizations into a workspace and load the catalog.
    """
    workspace = CoraplexRealizationWorkspace(workspace_root)
    contracts = capability_contracts()
    drafts = discover_coraplex_capability_contract_drafts(package_root)
    root = package_root if package_root is not None else installed_coraplex_root()
    existing = {candidate.candidate_id for candidate in workspace.candidates()}
    reference = reference_realizations()
    for draft in drafts:
        action_name = draft.action_class.rsplit(".", 1)[-1]
        for index, realization in enumerate(reference.get(action_name, ())):
            candidate_id = f"reference-{action_name}-{index}"
            if candidate_id in existing:
                continue
            workspace.submit(
                RealizationCandidate(
                    candidate_id=candidate_id,
                    action_source_id=draft.source_id,
                    realization=realization,
                    generated_by=reviewer,
                    rationale="reference mapping of the example task model",
                )
            )
            workspace.approve(candidate_id, reviewer, drafts, contracts, root)
    return initialize_coraplex_capabilities(contracts, workspace, package_root)
