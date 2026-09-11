"""
Platform-independent capability contracts exposed by Coraplex.

The contracts describe semantic execution goals. Native Coraplex actions are linked to
them in :mod:`resym.platform.coraplex_catalog`; action class names are deliberately not
used as capability identifiers.
"""

from enum import StrEnum
from functools import cache
from typing import Iterable, Mapping

from resym.core.model import (
    CapabilityContract,
    CapabilityRole,
    OntologyAlignment,
    SymbolType,
    is_symbol_subtype,
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

NAVIGATION_CAPABILITY_UID = "resym:ReachArticulationInteraction"
ARTICULATION_CAPABILITY_UID = "resym:ArticulationStateChange"
BASE_NAVIGATION_CAPABILITY_UID = "resym:BaseNavigation"
VISUAL_ATTENTION_CAPABILITY_UID = "resym:VisualAttention"
DETECTION_CAPABILITY_UID = "resym:ObjectDetection"
REACH_CAPABILITY_UID = "resym:EndEffectorReach"
GRASP_CAPABILITY_UID = "resym:ObjectGrasp"
PICK_UP_CAPABILITY_UID = "resym:ObjectPickup"
PLACE_CAPABILITY_UID = "resym:ObjectPlacement"
TRANSPORT_CAPABILITY_UID = "resym:ObjectTransport"
GRIPPER_STATE_CAPABILITY_UID = "resym:GripperStateChange"
ARM_POSTURE_CAPABILITY_UID = "resym:ArmPostureChange"
TORSO_STATE_CAPABILITY_UID = "resym:TorsoStateChange"
CARRY_POSTURE_CAPABILITY_UID = "resym:CarryPosture"
TOOL_PATH_CAPABILITY_UID = "resym:ToolPathFollowing"
MIXING_CAPABILITY_UID = "resym:MaterialMixing"
POURING_CAPABILITY_UID = "resym:MaterialPouring"
CUTTING_CAPABILITY_UID = "resym:MaterialCutting"
WIPING_CAPABILITY_UID = "resym:SurfaceWiping"
ELEVATOR_NAVIGATION_CAPABILITY_UID = "resym:ElevatorNavigation"

ACTOR_ROLE = "actor"
"""
Role bound to the acting robot.
"""

PATIENT_ROLE = "patient"
"""
Role bound to the object a capability manipulates.
"""

TARGET_STATE_ROLE = "target_state"
"""
Constant role selecting the state a capability drives its patient into.
"""


class OpenCloseState(StrEnum):
    """
    The ``target_state`` values of open/close capabilities.
    """

    OPEN = "OPEN"
    CLOSED = "CLOSED"


SOMA = "http://www.ease-crc.org/ont/SOMA.owl#"
SOMA_STATE_TRANSITION = SOMA + "StateTransition"
SOMA_OPENING = SOMA + "Opening"
SOMA_CLOSING = SOMA + "Closing"
SOMA_NAVIGATING = SOMA + "Navigating"


def _object_role(name: str, *, required: bool = True) -> CapabilityRole:
    return CapabilityRole(
        name,
        required=required,
        accepted_symbol_types=(OBJECT_TYPE,),
    )


def _actor_role() -> CapabilityRole:
    return CapabilityRole(ACTOR_ROLE, accepted_symbol_types=(AGENT_TYPE,))


def _constant_role(
    name: str, values: tuple[str, ...], *, required: bool = True
) -> CapabilityRole:
    return CapabilityRole(name, required=required, allowed_values=values)


def _soma_alignment(concept: str, rationale: str) -> OntologyAlignment:
    iri = SOMA + concept
    return OntologyAlignment(
        relation="SPECIALIZATION",
        target_iri=iri,
        source_version="soma@2.1.0",
        decided_by="platform-maintainer",
        rationale=rationale,
        evidence_iris=(iri,),
    )


def navigation_capability_contract(version: str = "1") -> CapabilityContract:
    return CapabilityContract(
        uid=NAVIGATION_CAPABILITY_UID,
        label="navigation.reach-articulation-interaction",
        version=version,
        roles=(
            _actor_role(),
            CapabilityRole(
                PATIENT_ROLE,
                accepted_symbol_types=(ARTICULATED_PART_TYPE,),
            ),
        ),
        success_relation="ready_to_open(actor, patient)",
        verifiable_effects=("ready-to-open",),
        ontology_alignment=OntologyAlignment(
            relation="SPECIALIZATION",
            target_iri=SOMA_NAVIGATING,
            source_version="soma@2.1.0",
            decided_by="llm-agent",
            decision_model="codex",
            rationale=(
                "The local capability is narrower than SOMA Navigating: it "
                "moves the robot to a witness pose from which it can interact "
                "with one articulated object."
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
                PATIENT_ROLE,
                accepted_symbol_types=(ARTICULATED_PART_TYPE,),
            ),
            CapabilityRole(
                "interaction_point",
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
                "articulated patient and the OPEN/CLOSED target states; SOMA "
                "Opening and Closing support the target-specific readings."
            ),
            evidence_iris=(SOMA_STATE_TRANSITION, SOMA_OPENING, SOMA_CLOSING),
        ),
    )


@cache
def capability_contracts(version: str = "1") -> tuple[CapabilityContract, ...]:
    """
    Reviewed semantic interfaces covering the current Coraplex actions.

    Having a contract does not by itself make a capability executable on every robot.
    Robot compatibility and reSym adapter readiness are recorded by an embodiment
    profile.
    """
    common = (_actor_role(),)
    contracts = (
        navigation_capability_contract(version),
        articulation_capability_contract(version),
        CapabilityContract(
            uid=BASE_NAVIGATION_CAPABILITY_UID,
            label="navigation.move-base",
            version=version,
            roles=common + (_object_role("destination"),),
            success_relation="at_location(actor, destination)",
            verifiable_effects=("at-location",),
            ontology_alignment=_soma_alignment(
                "Navigating", "Base motion to a designated destination."
            ),
        ),
        CapabilityContract(
            uid=VISUAL_ATTENTION_CAPABILITY_UID,
            label="attention.look-at",
            version=version,
            roles=common + (_object_role("target"),),
            success_relation="looking_at(actor, target)",
            verifiable_effects=("looking-at",),
            ontology_alignment=_soma_alignment(
                "LookingAt", "Directing the robot's sensing frame toward a target."
            ),
        ),
        CapabilityContract(
            uid=DETECTION_CAPABILITY_UID,
            label="perception.detect-object",
            version=version,
            roles=common
            + (
                _object_role("target", required=False),
                _object_role("region", required=False),
            ),
            success_relation="perceived(target)",
            verifiable_effects=("perceived",),
            ontology_alignment=_soma_alignment(
                "Perceiving", "Acquiring an object observation in a bounded region."
            ),
        ),
        CapabilityContract(
            uid=REACH_CAPABILITY_UID,
            label="manipulation.reach",
            version=version,
            roles=common + (_object_role("target"),),
            success_relation="reached(actor, target)",
            verifiable_effects=("reached",),
            ontology_alignment=_soma_alignment(
                "Reaching", "Moving an end effector to a target."
            ),
        ),
        CapabilityContract(
            uid=GRASP_CAPABILITY_UID,
            label="manipulation.grasp",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE),),
            success_relation="grasped(actor, patient)",
            verifiable_effects=("grasped",),
            ontology_alignment=_soma_alignment(
                "Grasping", "Establishing a grasp on an object."
            ),
        ),
        CapabilityContract(
            uid=PICK_UP_CAPABILITY_UID,
            label="manipulation.pick-up",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE),),
            success_relation="holding(actor, patient)",
            verifiable_effects=("holding",),
            ontology_alignment=_soma_alignment(
                "PickingUp", "Grasping and lifting an object."
            ),
        ),
        CapabilityContract(
            uid=PLACE_CAPABILITY_UID,
            label="manipulation.place",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE), _object_role("destination")),
            success_relation="placed_at(patient, destination)",
            verifiable_effects=("placed-at",),
            ontology_alignment=_soma_alignment(
                "Placing", "Placing an object at a designated destination."
            ),
        ),
        CapabilityContract(
            uid=TRANSPORT_CAPABILITY_UID,
            label="manipulation.transport",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE), _object_role("destination")),
            success_relation="transported_to(patient, destination)",
            verifiable_effects=("transported-to",),
            ontology_alignment=_soma_alignment(
                "Transporting", "Moving an object to a destination."
            ),
        ),
        CapabilityContract(
            uid=GRIPPER_STATE_CAPABILITY_UID,
            label="robot.gripper-state-transition",
            version=version,
            roles=common
            + (
                _constant_role("gripper", ("LEFT", "RIGHT", "BOTH")),
                _constant_role(
                    TARGET_STATE_ROLE, (OpenCloseState.OPEN, OpenCloseState.CLOSED)
                ),
            ),
            success_relation="gripper_state(actor, gripper) == target_state",
            verifiable_effects=("gripper-open", "gripper-closed"),
            effect_role_values=(
                ("gripper-open", TARGET_STATE_ROLE, OpenCloseState.OPEN),
                ("gripper-closed", TARGET_STATE_ROLE, OpenCloseState.CLOSED),
            ),
            ontology_alignment=_soma_alignment(
                "StateTransition", "Changing a selected gripper's state."
            ),
        ),
        CapabilityContract(
            uid=ARM_POSTURE_CAPABILITY_UID,
            label="robot.park-arms",
            version=version,
            roles=common + (_constant_role("arm", ("LEFT", "RIGHT", "BOTH")),),
            success_relation="arms_parked(actor, arm)",
            verifiable_effects=("arms-parked",),
            ontology_alignment=_soma_alignment(
                "ParkingArms", "Moving selected arms to their parked posture."
            ),
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
            ontology_alignment=_soma_alignment(
                "PosturalMoving", "Changing the robot torso posture."
            ),
        ),
        CapabilityContract(
            uid=CARRY_POSTURE_CAPABILITY_UID,
            label="robot.adopt-carry-posture",
            version=version,
            roles=common + (_constant_role("arm", ("LEFT", "RIGHT", "BOTH")),),
            success_relation="carrying_posture(actor, arm)",
            verifiable_effects=("carrying-posture",),
            ontology_alignment=_soma_alignment(
                "PosturalMoving", "Moving an arm to a carrying posture."
            ),
        ),
        CapabilityContract(
            uid=TOOL_PATH_CAPABILITY_UID,
            label="manipulation.follow-tool-path",
            version=version,
            roles=common + (_object_role("tool"), _object_role("target")),
            success_relation="tool_path_completed(tool, target)",
            verifiable_effects=("tool-path-completed",),
            ontology_alignment=_soma_alignment(
                "Reaching", "Following a path with a robot tool frame."
            ),
        ),
        CapabilityContract(
            uid=MIXING_CAPABILITY_UID,
            label="material.mix",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE), _object_role("tool")),
            success_relation="mixed(patient)",
            verifiable_effects=("mixed",),
            ontology_alignment=_soma_alignment(
                "Mixing", "Mixing material in a patient object with a tool."
            ),
        ),
        CapabilityContract(
            uid=POURING_CAPABILITY_UID,
            label="material.pour",
            version=version,
            roles=common + (_object_role("source"), _object_role("destination")),
            success_relation="poured_into(source, destination)",
            verifiable_effects=("poured-into",),
            ontology_alignment=_soma_alignment(
                "Pouring", "Pouring material from a source into a destination."
            ),
        ),
        CapabilityContract(
            uid=CUTTING_CAPABILITY_UID,
            label="material.cut",
            version=version,
            roles=common + (_object_role(PATIENT_ROLE), _object_role("tool")),
            success_relation="cut(patient)",
            verifiable_effects=("cut",),
            ontology_alignment=_soma_alignment(
                "Cutting", "Cutting a patient object with a tool."
            ),
        ),
        CapabilityContract(
            uid=WIPING_CAPABILITY_UID,
            label="material.wipe-surface",
            version=version,
            roles=common + (_object_role("surface"), _object_role("tool")),
            success_relation="wiped(surface)",
            verifiable_effects=("wiped",),
            ontology_alignment=_soma_alignment(
                "Cleaning", "Cleaning a surface by wiping it with a tool."
            ),
        ),
        CapabilityContract(
            uid=ELEVATOR_NAVIGATION_CAPABILITY_UID,
            label="navigation.move-between-levels",
            version=version,
            roles=common + (_object_role("elevator"), _object_role("target_floor")),
            success_relation="at_level(actor, target_floor)",
            verifiable_effects=("at-level",),
            ontology_alignment=_soma_alignment(
                "Navigating", "Moving the robot between building levels by elevator."
            ),
        ),
    )
    assert len({contract.uid for contract in contracts}) == len(contracts)
    return contracts


def matching_capability_contracts(
    desired_effects: Iterable[str],
    required_roles: Mapping[str, SymbolType | str] | None = None,
    *,
    version: str = "1",
    contracts: Iterable[CapabilityContract] | None = None,
) -> tuple[CapabilityContract, ...]:
    """
    Return reviewed contracts covering the requested effects and roles.

    This deterministic lookup runs before an agent proposes an
    ``OperatorExecutionBinding``. Ontology labels remain descriptive metadata;
    executable matching uses declared effects and typed semantic roles.
    """
    effects = frozenset(desired_effects)
    roles = {
        name: value if isinstance(value, SymbolType) else SymbolType(value)
        for name, value in (required_roles or {}).items()
    }

    def accepts(contract: CapabilityContract) -> bool:
        if not effects.issubset(contract.verifiable_effect_names):
            return False
        for role_name, actual_type in roles.items():
            role = contract.role_map.get(role_name)
            if role is None or not role.accepted_symbol_types:
                return False
            if not any(
                is_symbol_subtype(actual_type, accepted)
                for accepted in role.accepted_symbol_types
            ):
                return False
        return True

    catalog = (
        tuple(contracts) if contracts is not None else capability_contracts(version)
    )
    matches = [contract for contract in catalog if accepts(contract)]
    return tuple(
        sorted(
            matches,
            key=lambda contract: (
                len(set(contract.verifiable_effect_names) - effects),
                len(contract.roles),
                contract.uid,
            ),
        )
    )
