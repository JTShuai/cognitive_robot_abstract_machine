"""
Reviewed realizations of the experiment's capabilities by native Coraplex actions.

Benches and tests bootstrap these into a realization workspace through the same
candidate-review flow a live deployment drives from the review Viewer, so the runtime
catalog is identical whether a human or this bootstrap approved it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from typing_extensions import Iterable

from experiments import EXPERIMENTS_ROOT
from experiments.resym.capability_contracts import (
    ACTOR_ROLE,
    ARM_POSTURE_CAPABILITY_UID,
    ARTICULATION_CAPABILITY_UID,
    BASE_NAVIGATION_CAPABILITY_UID,
    CARRY_POSTURE_CAPABILITY_UID,
    CUTTING_CAPABILITY_UID,
    DETECTION_CAPABILITY_UID,
    ELEVATOR_NAVIGATION_CAPABILITY_UID,
    GRASP_CAPABILITY_UID,
    GRIPPER_STATE_CAPABILITY_UID,
    INTERACTION_NAVIGATION_CAPABILITY_UID,
    INTERACTION_POINT_ROLE,
    MIXING_CAPABILITY_UID,
    PATIENT_ROLE,
    PICK_UP_CAPABILITY_UID,
    PLACE_CAPABILITY_UID,
    POURING_CAPABILITY_UID,
    REACH_CAPABILITY_UID,
    TARGET_STATE_ROLE,
    TOOL_PATH_CAPABILITY_UID,
    TORSO_STATE_CAPABILITY_UID,
    TRANSPORT_CAPABILITY_UID,
    VISUAL_ATTENTION_CAPABILITY_UID,
    WIPING_CAPABILITY_UID,
    OpenCloseState,
)
from experiments.resym.capability_initialization import default_capability_contracts
from resym.core.capability_model import CapabilityContract
from resym.core.symbol_types import SymbolType
from semantic_digital_twin.robots.robot_parts import (
    Arm,
    Camera,
    EndEffector,
    MobileBase,
    Torso,
)
from resym.platform.coraplex_catalog import (
    CoraplexCapabilityInitialization,
    discover_coraplex_capability_contract_drafts,
    initialize_coraplex_capabilities,
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

DEFAULT_REALIZATION_WORKSPACE_ROOT = EXPERIMENTS_ROOT / "tmp" / "realization_workspace"
"""
Standard local workspace the experiment bootstraps its realizations into.
"""


def _role(parameter: str, role: str) -> ParameterSource:
    return ParameterSource(parameter, ParameterSourceKind.ROLE, role)


def _context(parameter: str, value: ContextValue) -> ParameterSource:
    return ParameterSource(parameter, ParameterSourceKind.CONTEXT, value)


def _constant(parameter: str, value: str) -> ParameterSource:
    return ParameterSource(parameter, ParameterSourceKind.CONSTANT, value)


_ARM = _context("arm", ContextValue.MANIPULATION_ARM)
_GRASP = _context("grasp_description", ContextValue.DEFAULT_GRASP)

_ACTION_REALIZATIONS: dict[str, tuple[ActionRealization, ...]] = {
    "FaceAtAction": (ActionRealization(VISUAL_ATTENTION_CAPABILITY_UID),),
    "MixingAction": (
        ActionRealization(
            MIXING_CAPABILITY_UID,
            (_ARM, _role("tool", "tool"), _role("container", PATIENT_ROLE)),
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
        ),
    ),
    "CuttingAction": (
        ActionRealization(
            CUTTING_CAPABILITY_UID,
            (_ARM, _role("tool", "tool"), _role("object_to_cut", PATIENT_ROLE)),
        ),
    ),
    "WipingAction": (
        ActionRealization(
            WIPING_CAPABILITY_UID,
            (_ARM, _role("tool", "tool"), _role("surface", "surface")),
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
        ),
    ),
    "MoveAndPlaceAction": (ActionRealization(PLACE_CAPABILITY_UID),),
    "MoveAndPickUpAction": (ActionRealization(PICK_UP_CAPABILITY_UID),),
    "CloseAction": (
        ActionRealization(
            ARTICULATION_CAPABILITY_UID,
            (_role("object_designator", INTERACTION_POINT_ROLE), _ARM),
            RoleCondition(TARGET_STATE_ROLE, OpenCloseState.CLOSED),
        ),
    ),
    "OpenAction": (
        ActionRealization(
            ARTICULATION_CAPABILITY_UID,
            (_role("object_designator", INTERACTION_POINT_ROLE), _ARM),
            RoleCondition(TARGET_STATE_ROLE, OpenCloseState.OPEN),
        ),
    ),
    "DetectAction": (
        ActionRealization(
            DETECTION_CAPABILITY_UID,
            (_constant("technique", "REGION"), _role("region", "region")),
            RoleCondition("region"),
        ),
        ActionRealization(
            DETECTION_CAPABILITY_UID,
            (_constant("technique", "TYPES"), _role("object_sem_annotation", "target")),
            RoleCondition("target"),
        ),
    ),
    "MoveToReach": (ActionRealization(REACH_CAPABILITY_UID),),
    "LookAtAction": (
        ActionRealization(
            VISUAL_ATTENTION_CAPABILITY_UID, (_role("target", "target"),)
        ),
    ),
    "NavigateAction": (
        ActionRealization(
            BASE_NAVIGATION_CAPABILITY_UID, (_role("target_location", "destination"),)
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
        ),
    ),
    "ElevatorNavigation": (
        ActionRealization(
            ELEVATOR_NAVIGATION_CAPABILITY_UID,
            (_role("elevator", "elevator"), _role("target_floor", "target_floor")),
        ),
    ),
    "GraspingAction": (
        ActionRealization(
            GRASP_CAPABILITY_UID,
            (_role("object_designator", PATIENT_ROLE), _ARM, _GRASP),
        ),
    ),
    "PickUpAction": (
        ActionRealization(
            PICK_UP_CAPABILITY_UID,
            (_role("object_designator", PATIENT_ROLE), _ARM, _GRASP),
        ),
    ),
    "ReachAction": (
        ActionRealization(
            REACH_CAPABILITY_UID,
            (
                _role("target_pose", "target"),
                _ARM,
                _GRASP,
                _role("object_designator", "target"),
            ),
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
        ),
    ),
    "CarryAction": (
        ActionRealization(CARRY_POSTURE_CAPABILITY_UID, (_role("arm", "arm"),)),
    ),
    "FollowToolCenterPointPathAction": (
        ActionRealization(
            TOOL_PATH_CAPABILITY_UID, (_role("target_locations", "target"), _ARM)
        ),
    ),
    "MoveManipulatorAction": (ActionRealization(REACH_CAPABILITY_UID),),
    "MoveTorsoAction": (
        ActionRealization(
            TORSO_STATE_CAPABILITY_UID, (_role("torso_state", TARGET_STATE_ROLE),)
        ),
    ),
    "ParkArmsAction": (
        ActionRealization(ARM_POSTURE_CAPABILITY_UID, (_role("arm", "arm"),)),
    ),
    "SetGripperAction": (
        ActionRealization(
            GRIPPER_STATE_CAPABILITY_UID,
            (_role("gripper", "gripper"), _role("motion", TARGET_STATE_ROLE)),
        ),
    ),
}

_MOBILE = frozenset({SymbolType.from_python_type(MobileBase)})
_ARM_ONLY = frozenset({SymbolType.from_python_type(Arm)})
_MANIPULATOR = frozenset(
    {SymbolType.from_python_type(Arm), SymbolType.from_python_type(EndEffector)}
)
_CAMERA = frozenset({SymbolType.from_python_type(Camera)})

_ACTION_REQUIREMENTS: dict[str, frozenset[SymbolType]] = {
    "FaceAtAction": _MOBILE | _CAMERA,
    "MixingAction": _MANIPULATOR,
    "PouringAction": _MANIPULATOR,
    "CuttingAction": _MANIPULATOR,
    "WipingAction": _MANIPULATOR,
    "TransportAction": _MOBILE | _MANIPULATOR,
    "PickAndPlaceAction": _MANIPULATOR,
    "MoveAndPlaceAction": _MOBILE | _MANIPULATOR,
    "MoveAndPickUpAction": _MOBILE | _MANIPULATOR,
    "CloseAction": _MANIPULATOR,
    "OpenAction": _MANIPULATOR,
    "DetectAction": _CAMERA,
    "MoveToReach": _MOBILE | _MANIPULATOR,
    "LookAtAction": _CAMERA,
    "NavigateAction": _MOBILE,
    "ElevatorNavigation": _MOBILE,
    "GraspingAction": _MANIPULATOR,
    "PickUpAction": _MANIPULATOR,
    "ReachAction": _MANIPULATOR,
    "PlaceAction": _MANIPULATOR,
    "CarryAction": _ARM_ONLY,
    "FollowToolCenterPointPathAction": _MANIPULATOR,
    "MoveManipulatorAction": _MANIPULATOR,
    "MoveTorsoAction": frozenset({SymbolType.from_python_type(Torso)}),
    "ParkArmsAction": _ARM_ONLY,
    "SetGripperAction": frozenset({SymbolType.from_python_type(EndEffector)}),
}


def reference_realizations() -> dict[str, tuple[ActionRealization, ...]]:
    """
    The reviewed realizations by native action class name, with the robot resources each
    action needs.
    """
    return {
        action_name: tuple(
            replace(
                realization,
                required_resources=tuple(sorted(_ACTION_REQUIREMENTS[action_name])),
            )
            for realization in realizations
        )
        for action_name, realizations in _ACTION_REALIZATIONS.items()
    }


def bootstrap_capability_realizations(
    workspace_root: Path,
    contracts: Iterable[CapabilityContract] | None = None,
    reviewer: str = "experiment-bootstrap",
    package_root: Path | None = None,
) -> CoraplexCapabilityInitialization:
    """
    Approve the reference realizations into a workspace and load the runtime catalog.

    Repeated calls on the same workspace are no-ops for candidates it already holds.
    """
    workspace = CoraplexRealizationWorkspace(workspace_root)
    contracts = tuple(
        default_capability_contracts() if contracts is None else contracts
    )
    drafts = discover_coraplex_capability_contract_drafts(package_root)
    existing = {candidate.candidate_id for candidate in workspace.candidates()}
    reference = reference_realizations()
    for draft in drafts:
        action_name = draft.action_class.rsplit(".", 1)[-1]
        for index, realization in enumerate(reference.get(action_name, ())):
            candidate_id = f"reference-{draft.source_id.split(':', 1)[1].replace('.', '-')}-{index}"
            if candidate_id in existing:
                continue
            workspace.submit(
                RealizationCandidate(
                    candidate_id=candidate_id,
                    action_source_id=draft.source_id,
                    realization=realization,
                    generated_by=reviewer,
                    rationale="reference mapping of the experiment platform",
                )
            )
            workspace.approve(
                candidate_id, reviewer, drafts, contracts, _package_root(package_root)
            )
    return initialize_coraplex_capabilities(contracts, workspace, package_root)


def default_capability_initialization(
    contracts: Iterable[CapabilityContract] | None = None,
) -> CoraplexCapabilityInitialization:
    """
    Bootstrap the standard local workspace (idempotently) and load it.
    """
    return bootstrap_capability_realizations(
        DEFAULT_REALIZATION_WORKSPACE_ROOT, contracts
    )


def _package_root(package_root: Path | None) -> Path:
    from resym.platform.coraplex_catalog import installed_coraplex_root

    return package_root if package_root is not None else installed_coraplex_root()
