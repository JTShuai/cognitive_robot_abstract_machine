"""
Identifiers of the drawer and kitchen experiments' capability contracts.

The contracts themselves are the reference artifact
``capability_references/contracts.json``, admitted into a review workspace at
initialization by :mod:`experiments.resym.capability_initialization`.
"""

from enum import StrEnum

from resym.core.symbol_types import SymbolType
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

INTERACTION_POINT_ROLE = "interaction_point"
"""
Role bound to the part a capability makes contact with.
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
