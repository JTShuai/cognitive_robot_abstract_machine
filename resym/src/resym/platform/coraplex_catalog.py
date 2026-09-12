"""
Discover Coraplex actions and expose them as contract drafts.

Coraplex action classes reveal executable parameters and native condition hooks, but
they do not declare reSym's semantic roles or independently verifiable symbolic effects.
Discovery therefore produces drafts. A reviewed mapping links those drafts to trusted
``CapabilityContract`` objects; adapter readiness is tracked separately for each robot
embodiment.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Iterable

from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Camera
from resym.core.capabilities import CapabilityContract
from resym.platform.capabilities import (
    ARM_POSTURE_CAPABILITY_UID,
    ARTICULATION_CAPABILITY_UID,
    BASE_NAVIGATION_CAPABILITY_UID,
    CARRY_POSTURE_CAPABILITY_UID,
    CUTTING_CAPABILITY_UID,
    DETECTION_CAPABILITY_UID,
    ELEVATOR_NAVIGATION_CAPABILITY_UID,
    GRASP_CAPABILITY_UID,
    GRIPPER_STATE_CAPABILITY_UID,
    MIXING_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    PICK_UP_CAPABILITY_UID,
    PLACE_CAPABILITY_UID,
    POURING_CAPABILITY_UID,
    REACH_CAPABILITY_UID,
    TOOL_PATH_CAPABILITY_UID,
    TORSO_STATE_CAPABILITY_UID,
    TRANSPORT_CAPABILITY_UID,
    VISUAL_ATTENTION_CAPABILITY_UID,
    WIPING_CAPABILITY_UID,
    capability_contracts,
)


@dataclass(frozen=True)
class CoraplexActionParameter:
    """
    One constructor parameter found in a Coraplex action declaration.
    """

    name: str
    type_expression: str
    required: bool

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "type_expression": self.type_expression,
            "required": self.required,
        }


@dataclass(frozen=True)
class CoraplexCapabilityContractDraft:
    """
    Mechanically extracted evidence for defining a capability contract.
    """

    source_id: str
    action_class: str
    source_file: str
    parameters: tuple[CoraplexActionParameter, ...]
    summary: str
    declares_precondition: bool
    declares_postcondition: bool
    missing_semantics: tuple[str, ...] = (
        "semantic capability UID",
        "role mapping and local symbol types",
        "independently verifiable effects",
        "realization argument adapter",
    )

    def to_json(self) -> dict:
        return {
            "source_id": self.source_id,
            "action_class": self.action_class,
            "source_file": self.source_file,
            "parameters": [parameter.to_json() for parameter in self.parameters],
            "summary": self.summary,
            "declares_precondition": self.declares_precondition,
            "declares_postcondition": self.declares_postcondition,
            "missing_semantics": list(self.missing_semantics),
        }

    def render(self) -> str:
        parameters = ", ".join(
            f"{item.name}: {item.type_expression}"
            f"{' (optional)' if not item.required else ''}"
            for item in self.parameters
        )
        conditions = []
        if self.declares_precondition:
            conditions.append("native precondition")
        if self.declares_postcondition:
            conditions.append("native postcondition")
        return (
            f"- {self.source_id}: {self.action_class}({parameters}); "
            f"{', '.join(conditions) or 'default native conditions'}; "
            f"draft only"
        )


class RobotResource(str, Enum):
    """
    Robot resources referenced by Coraplex action implementations.
    """

    MOBILE_BASE = "mobile-base"
    ARM = "arm"
    END_EFFECTOR = "end-effector"
    CAMERA = "camera"
    TORSO = "torso"


@dataclass(frozen=True)
class CoraplexCapabilityRealizationEvidence:
    """
    Reviewed link from one native action to one semantic capability.
    """

    capability_uid: str
    action_source_id: str
    required_resources: frozenset[RobotResource]


@dataclass(frozen=True)
class CoraplexCapabilitySupport:
    """
    Availability of one reviewed contract on a concrete robot.
    """

    capability_uid: str
    known_action_source_ids: tuple[str, ...]
    action_source_ids: tuple[str, ...]
    status: CapabilityRealizationStatus
    verification_status: CapabilityVerificationStatus
    missing_resources: tuple[RobotResource, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is CapabilityRealizationStatus.READY


class CapabilityRealizationStatus(str, Enum):
    """
    Why a Coraplex-backed capability can or cannot be dispatched.
    """

    READY = "ready"
    ADAPTER_MISSING = "adapter-missing"
    ROBOT_INCOMPATIBLE = "robot-incompatible"


class CapabilityVerificationStatus(str, Enum):
    """
    Where independent effect truth procedures must come from.
    """

    BUILTIN = "built-in"
    TASK_REQUIRED = "task-required"


class CapabilityReviewStatus(str, Enum):
    """Admission state of a scanned Coraplex action mapping."""

    PENDING = "pending"
    APPROVED = "approved"


@dataclass(frozen=True)
class CoraplexCapabilityRegistration:
    """One scanned action joined with an optional reviewed capability mapping."""

    draft: CoraplexCapabilityContractDraft
    """Mechanically extracted action declaration."""

    status: CapabilityReviewStatus
    """Whether the semantic mapping may enter the runtime catalog."""

    contract: CapabilityContract | None = None
    """Reviewed task-level meaning, absent while review is pending."""

    required_resources: frozenset[RobotResource] = frozenset()
    """Robot resources required by the action implementation."""

    reviewed_by: str | None = None
    """Authority that approved the semantic mapping."""


@dataclass(frozen=True)
class CoraplexCapabilityInitialization:
    """Result of scanning Coraplex and applying the reviewed mapping manifest."""

    records: tuple[CoraplexCapabilityRegistration, ...]
    """All scanned actions, including mappings still awaiting review."""

    @property
    def approved_contracts(self) -> tuple[CapabilityContract, ...]:
        """Distinct contracts admitted by approved action registrations."""
        by_uid = {
            record.contract.uid: record.contract
            for record in self.records
            if record.status is CapabilityReviewStatus.APPROVED
            and record.contract is not None
        }
        return tuple(by_uid[uid] for uid in sorted(by_uid))


CORAPLEX_ADAPTER_CAPABILITY_UIDS = frozenset(
    contract.uid for contract in capability_contracts()
)
"""
Contracts with a reviewed ``ExecutionRequest`` to Coraplex adapter.
"""

CORAPLEX_BUILTIN_VERIFICATION_UIDS = frozenset(
    {NAVIGATION_CAPABILITY_UID, ARTICULATION_CAPABILITY_UID}
)
"""
Contracts whose effect predicates ship with the current task packages.
"""

_ACTION_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "FaceAtAction": (VISUAL_ATTENTION_CAPABILITY_UID,),
    "MixingAction": (MIXING_CAPABILITY_UID,),
    "PouringAction": (POURING_CAPABILITY_UID,),
    "CuttingAction": (CUTTING_CAPABILITY_UID,),
    "WipingAction": (WIPING_CAPABILITY_UID,),
    "TransportAction": (TRANSPORT_CAPABILITY_UID,),
    "PickAndPlaceAction": (TRANSPORT_CAPABILITY_UID,),
    "MoveAndPlaceAction": (PLACE_CAPABILITY_UID,),
    "MoveAndPickUpAction": (PICK_UP_CAPABILITY_UID,),
    "CloseAction": (ARTICULATION_CAPABILITY_UID,),
    "OpenAction": (ARTICULATION_CAPABILITY_UID,),
    "DetectAction": (DETECTION_CAPABILITY_UID,),
    "MoveToReach": (REACH_CAPABILITY_UID,),
    "LookAtAction": (VISUAL_ATTENTION_CAPABILITY_UID,),
    "NavigateAction": (
        BASE_NAVIGATION_CAPABILITY_UID,
        NAVIGATION_CAPABILITY_UID,
    ),
    "ElevatorNavigation": (ELEVATOR_NAVIGATION_CAPABILITY_UID,),
    "GraspingAction": (GRASP_CAPABILITY_UID,),
    "PickUpAction": (PICK_UP_CAPABILITY_UID,),
    "ReachAction": (REACH_CAPABILITY_UID,),
    "PlaceAction": (PLACE_CAPABILITY_UID,),
    "CarryAction": (CARRY_POSTURE_CAPABILITY_UID,),
    "FollowToolCenterPointPathAction": (TOOL_PATH_CAPABILITY_UID,),
    "MoveManipulatorAction": (REACH_CAPABILITY_UID,),
    "MoveTorsoAction": (TORSO_STATE_CAPABILITY_UID,),
    "ParkArmsAction": (ARM_POSTURE_CAPABILITY_UID,),
    "SetGripperAction": (GRIPPER_STATE_CAPABILITY_UID,),
}

_MOBILE = frozenset({RobotResource.MOBILE_BASE})
_ARM = frozenset({RobotResource.ARM})
_MANIPULATOR = frozenset({RobotResource.ARM, RobotResource.END_EFFECTOR})
_CAMERA = frozenset({RobotResource.CAMERA})

_ACTION_REQUIREMENTS: dict[str, frozenset[RobotResource]] = {
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
    "CarryAction": _ARM,
    "FollowToolCenterPointPathAction": _MANIPULATOR,
    "MoveManipulatorAction": _MANIPULATOR,
    "MoveTorsoAction": frozenset({RobotResource.TORSO}),
    "ParkArmsAction": _ARM,
    "SetGripperAction": frozenset({RobotResource.END_EFFECTOR}),
}


@cache
def discover_coraplex_capability_contract_drafts(
    package_root: Path | None = None,
) -> tuple[CoraplexCapabilityContractDraft, ...]:
    """
    Read Coraplex action declarations without importing the ROS runtime.
    """
    root = package_root or _installed_coraplex_root()
    actions_root = root / "robot_plans" / "actions"
    if not actions_root.is_dir():
        return ()

    records: list[tuple[Path, ast.ClassDef]] = []
    action_class_names = {"ActionDescription"}
    parsed_files: list[tuple[Path, ast.Module]] = []
    for source_file in sorted(actions_root.rglob("*.py")):
        module = ast.parse(source_file.read_text(), filename=str(source_file))
        parsed_files.append((source_file, module))

    # Resolve subclasses transitively because composite actions inherit through
    # abstract action classes declared in other files.
    changed = True
    while changed:
        changed = False
        for source_file, module in parsed_files:
            for node in module.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                base_names = {_terminal_name(base) for base in node.bases}
                if (
                    base_names & action_class_names
                    and node.name not in action_class_names
                ):
                    action_class_names.add(node.name)
                    changed = True

    for source_file, module in parsed_files:
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name == "ActionDescription" or node.name not in action_class_names:
                continue
            if _is_abstract(node):
                continue
            records.append((source_file, node))

    drafts = [_draft_from_ast(root, source_file, node) for source_file, node in records]
    return tuple(sorted(drafts, key=lambda item: item.source_id))


def render_coraplex_capability_contract_drafts() -> str:
    drafts = discover_coraplex_capability_contract_drafts()
    return "\n".join(draft.render() for draft in drafts) or "- unavailable"


def initialize_coraplex_capabilities(
    package_root: Path | None = None,
) -> CoraplexCapabilityInitialization:
    """Scan native actions and admit only platform-maintainer mappings."""
    contracts = {contract.uid: contract for contract in capability_contracts()}
    records: list[CoraplexCapabilityRegistration] = []
    for draft in discover_coraplex_capability_contract_drafts(package_root):
        action_name = draft.action_class.rsplit(".", 1)[-1]
        capability_uids = _ACTION_CAPABILITIES.get(action_name, ())
        if not capability_uids:
            records.append(
                CoraplexCapabilityRegistration(
                    draft=draft,
                    status=CapabilityReviewStatus.PENDING,
                )
            )
            continue
        for capability_uid in capability_uids:
            records.append(
                CoraplexCapabilityRegistration(
                    draft=draft,
                    status=CapabilityReviewStatus.APPROVED,
                    contract=contracts[capability_uid],
                    required_resources=_ACTION_REQUIREMENTS[action_name],
                    reviewed_by="platform-maintainer",
                )
            )
    return CoraplexCapabilityInitialization(tuple(records))


def compatible_coraplex_action_ids(profile) -> frozenset[str]:
    """
    Native actions whose resource needs match the profile's robot.
    """
    return frozenset(
        source_id
        for _, source_ids in profile.capability_sources
        for source_id in source_ids
    )


def render_coraplex_capability_candidates(profile) -> str:
    """
    Render only native actions structurally compatible with one robot.
    """
    compatible_ids = compatible_coraplex_action_ids(profile)
    capability_by_source: dict[str, set[str]] = {}
    for capability_uid, source_ids in profile.capability_sources:
        for source_id in source_ids:
            capability_by_source.setdefault(source_id, set()).add(capability_uid)
    lines = []
    for draft in discover_coraplex_capability_contract_drafts():
        if draft.source_id not in compatible_ids:
            continue
        capability_uids = ", ".join(sorted(capability_by_source[draft.source_id]))
        lines.append(
            f"{draft.render()}; compatible with {profile.name}; "
            f"reviewed semantics {{{capability_uids}}}"
        )
    return "\n".join(lines) or "- none"


@cache
def coraplex_capability_realization_evidence() -> (
    tuple[CoraplexCapabilityRealizationEvidence, ...]
):
    """
    Link every discovered action to reviewed semantics and robot needs.
    """
    evidence = [
        CoraplexCapabilityRealizationEvidence(
            capability_uid=record.contract.uid,
            action_source_id=record.draft.source_id,
            required_resources=record.required_resources,
        )
        for record in initialize_coraplex_capabilities().records
        if record.status is CapabilityReviewStatus.APPROVED
        and record.contract is not None
    ]
    return tuple(
        sorted(
            evidence,
            key=lambda item: (item.capability_uid, item.action_source_id),
        )
    )


def coraplex_capability_catalog(package_root: Path | None = None) -> dict:
    """
    Serialize the complete reviewed contract-to-Coraplex design map.

    The result is platform-structure independent and can therefore be shown by the host-
    only Viewer. Robot-specific compatibility is added by an ``EmbodimentProfile`` at
    runtime.
    """
    initialization = initialize_coraplex_capabilities(package_root)
    drafts = {record.draft.source_id: record.draft for record in initialization.records}
    approved_records = tuple(
        record
        for record in initialization.records
        if record.status is CapabilityReviewStatus.APPROVED
        and record.contract is not None
    )
    pending_records = tuple(
        record
        for record in initialization.records
        if record.status is CapabilityReviewStatus.PENDING
    )
    evidence_by_capability: dict[str, list[CoraplexCapabilityRealizationEvidence]] = {}
    for record in approved_records:
        evidence = CoraplexCapabilityRealizationEvidence(
            capability_uid=record.contract.uid,
            action_source_id=record.draft.source_id,
            required_resources=record.required_resources,
        )
        evidence_by_capability.setdefault(evidence.capability_uid, []).append(evidence)

    entries = []
    for contract in capability_contracts():
        has_reviewed_action = bool(evidence_by_capability.get(contract.uid))
        if (
            contract.uid not in CORAPLEX_ADAPTER_CAPABILITY_UIDS
            or not has_reviewed_action
        ):
            status = CapabilityRealizationStatus.ADAPTER_MISSING
        else:
            status = CapabilityRealizationStatus.READY
        verification = (
            CapabilityVerificationStatus.BUILTIN
            if contract.uid in CORAPLEX_BUILTIN_VERIFICATION_UIDS
            else CapabilityVerificationStatus.TASK_REQUIRED
        )
        actions = []
        for evidence in evidence_by_capability.get(contract.uid, []):
            draft = drafts[evidence.action_source_id]
            actions.append(
                {
                    **draft.to_json(),
                    "required_resources": sorted(
                        item.value for item in evidence.required_resources
                    ),
                }
            )
        entries.append(
            {
                "contract": to_json(contract),
                "realization": {
                    "platform": "Coraplex",
                    "status": status.value,
                    "effect_verification": verification.value,
                    "actions": actions,
                },
            }
        )
    return {
        "contracts": entries,
        "pending_actions": [record.draft.to_json() for record in pending_records],
        "summary": {
            "contracts": len(entries),
            "actions": len(drafts),
            "ready": sum(
                entry["realization"]["status"]
                == CapabilityRealizationStatus.READY.value
                for entry in entries
            ),
            "built_in_verification": sum(
                entry["realization"]["effect_verification"]
                == CapabilityVerificationStatus.BUILTIN.value
                for entry in entries
            ),
            "pending_review": len(pending_records),
        },
    }


def robot_resources(robot: AbstractRobot) -> frozenset[RobotResource]:
    """
    Read a robot's structural resources from its semantic annotation.
    """
    resources = set()
    if robot.drive is not None:
        resources.add(RobotResource.MOBILE_BASE)
    if robot.get_arms():
        resources.add(RobotResource.ARM)
    if robot.get_end_effectors():
        resources.add(RobotResource.END_EFFECTOR)
    if any(isinstance(sensor, Camera) for sensor in robot.get_sensors()):
        resources.add(RobotResource.CAMERA)
    if robot.get_torso_if_specified() is not None:
        resources.add(RobotResource.TORSO)
    return frozenset(resources)


def infer_coraplex_capability_support(
    robot: AbstractRobot,
    *,
    ready_capability_uids: Iterable[str] = (),
    adapter_capability_uids: Iterable[str] | None = None,
) -> tuple[CoraplexCapabilitySupport, ...]:
    """
    Classify every reviewed Coraplex capability for one robot.

    ``ready_capability_uids`` is the compatibility shorthand used by existing
    callers. New callers should pass the reviewed adapter set explicitly.
    Effect verification is task-dependent: selected effect predicates must
    provide a reviewed predicate query before planning can proceed.
    """
    available_resources = robot_resources(robot)
    ready = frozenset(ready_capability_uids)
    adapters = (
        ready if adapter_capability_uids is None else frozenset(adapter_capability_uids)
    )
    evidence_by_capability: dict[str, list[CoraplexCapabilityRealizationEvidence]] = {}
    for evidence in coraplex_capability_realization_evidence():
        evidence_by_capability.setdefault(evidence.capability_uid, []).append(evidence)

    support = []
    for contract in capability_contracts():
        evidence = evidence_by_capability.get(contract.uid, [])
        compatible = [
            item
            for item in evidence
            if item.required_resources.issubset(available_resources)
        ]
        if not compatible:
            missing_sets = [
                item.required_resources - available_resources for item in evidence
            ]
            missing = min(
                missing_sets,
                key=lambda items: (
                    len(items),
                    tuple(sorted(item.value for item in items)),
                ),
                default=frozenset(),
            )
            status = CapabilityRealizationStatus.ROBOT_INCOMPATIBLE
        elif contract.uid not in adapters:
            missing = frozenset()
            status = CapabilityRealizationStatus.ADAPTER_MISSING
        else:
            missing = frozenset()
            status = CapabilityRealizationStatus.READY
        verification = (
            CapabilityVerificationStatus.BUILTIN
            if contract.uid in CORAPLEX_BUILTIN_VERIFICATION_UIDS
            else CapabilityVerificationStatus.TASK_REQUIRED
        )
        support.append(
            CoraplexCapabilitySupport(
                capability_uid=contract.uid,
                known_action_source_ids=tuple(
                    sorted(item.action_source_id for item in evidence)
                ),
                action_source_ids=tuple(
                    sorted(item.action_source_id for item in compatible)
                ),
                status=status,
                verification_status=verification,
                missing_resources=tuple(sorted(missing, key=lambda item: item.value)),
            )
        )
    return tuple(support)


def coraplex_embodiment_profile(
    *,
    name: str,
    robot: AbstractRobot,
    tool_orientation,
    ready_capability_uids: Iterable[str] = (),
    adapter_capability_uids: Iterable[str] | None = None,
):
    """
    Build a profile from CRAM robot parts and reviewed realizations.
    """
    from resym.platform.embodiment import EmbodimentProfile

    support = infer_coraplex_capability_support(
        robot,
        ready_capability_uids=ready_capability_uids,
        adapter_capability_uids=adapter_capability_uids,
    )
    return EmbodimentProfile(
        name=name,
        capabilities=frozenset(item.capability_uid for item in support if item.ready),
        capability_sources=tuple(
            (item.capability_uid, item.action_source_ids)
            for item in support
            if item.action_source_ids
        ),
        tool_orientation=tool_orientation,
    )


def _installed_coraplex_root() -> Path:
    spec = importlib.util.find_spec("coraplex")
    if spec is None or not spec.submodule_search_locations:
        return Path("/__coraplex_not_installed__")
    return Path(next(iter(spec.submodule_search_locations)))


def _draft_from_ast(
    package_root: Path, source_file: Path, node: ast.ClassDef
) -> CoraplexCapabilityContractDraft:
    relative = source_file.relative_to(package_root).with_suffix("")
    module_name = ".".join(relative.parts)
    source_id = f"coraplex:{module_name}.{_kebab(node.name)}"
    parameters = []
    for child in node.body:
        if not isinstance(child, ast.AnnAssign) or not isinstance(
            child.target, ast.Name
        ):
            continue
        if child.target.id.startswith("_"):
            continue
        parameters.append(
            CoraplexActionParameter(
                name=child.target.id,
                type_expression=ast.unparse(child.annotation),
                required=_parameter_is_required(child),
            )
        )
    doc = ast.get_docstring(node) or ""
    summary = doc.strip().splitlines()[0] if doc.strip() else node.name
    method_names = {
        child.name
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return CoraplexCapabilityContractDraft(
        source_id=source_id,
        action_class=f"coraplex.{module_name}.{node.name}",
        source_file=str(relative.with_suffix(".py")),
        parameters=tuple(parameters),
        summary=summary,
        declares_precondition="pre_condition" in method_names,
        declares_postcondition="post_condition" in method_names,
    )


def _terminal_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _terminal_name(node.value)
    return ""


def _is_abstract(node: ast.ClassDef) -> bool:
    if any(_terminal_name(base) == "ABC" for base in node.bases):
        return True
    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            _terminal_name(decorator) == "abstractmethod"
            for decorator in child.decorator_list
        ):
            return True
    return False


def _parameter_is_required(node: ast.AnnAssign) -> bool:
    if node.value is None:
        return True
    if (
        not isinstance(node.value, ast.Call)
        or _terminal_name(node.value.func) != "field"
    ):
        return False
    return not any(
        keyword.arg in {"default", "default_factory"} for keyword in node.value.keywords
    )


def _kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
