"""
Discover Coraplex actions and expose them as contract drafts.

Coraplex action classes reveal executable parameters and native condition hooks, but
they do not declare reSym's semantic roles or independently verifiable symbolic effects.
Discovery therefore produces drafts. A reviewed mapping links those drafts to trusted
``CapabilityContract`` objects; adapter readiness is combined with the resources exposed
by each CRAM robot annotation.
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
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from resym.core.capability_model import CapabilityContract, ExecutionRequest
from resym.core.symbol_types import SymbolType
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobotPart, MobileBase
from resym.platform.coraplex_realizations import (
    ApprovedRealization,
    CapabilityReviewStatus,
    CoraplexRealizationWorkspace,
    ParameterSource,
    RoleCondition,
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


@dataclass(frozen=True)
class CoraplexCapabilityRealizationEvidence:
    """
    Reviewed link from one native action to one semantic capability.
    """

    capability_uid: str
    action_source_id: str
    required_resources: frozenset[SymbolType]


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
    missing_resources: tuple[SymbolType, ...] = ()

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

    TASK_REQUIRED = "task-required"


@dataclass(frozen=True)
class CoraplexCapabilityRegistration:
    """One scanned action joined with an optional reviewed capability mapping."""

    draft: CoraplexCapabilityContractDraft
    """Mechanically extracted action declaration."""

    status: CapabilityReviewStatus
    """Whether the semantic mapping may enter the runtime catalog."""

    contract: CapabilityContract | None = None
    """Reviewed task-level meaning, absent while review is pending."""

    required_resources: frozenset[SymbolType] = frozenset()
    """Robot resources required by the action implementation."""

    reviewed_by: str | None = None
    """Authority that approved the semantic mapping."""

    parameter_sources: tuple[ParameterSource, ...] = ()
    """How each native parameter is filled from a request; empty while unreviewed."""

    applies_when: RoleCondition | None = None
    """Request condition selecting this registration among a capability's variants."""


@dataclass(frozen=True)
class CoraplexCapabilityInitialization:
    """Result of scanning Coraplex and applying the reviewed mapping manifest."""

    records: tuple[CoraplexCapabilityRegistration, ...]
    """All scanned actions, including mappings still awaiting review."""

    contracts: tuple[CapabilityContract, ...] = ()
    """The reviewed contracts the actions were joined with."""

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


def adapter_capability_uids(
    initialization: CoraplexCapabilityInitialization,
) -> frozenset[str]:
    """
    Capabilities the Coraplex adapter can execute: those with an admitted realization.
    """
    return frozenset(
        record.contract.uid
        for record in initialization.records
        if record.status is CapabilityReviewStatus.APPROVED
        and record.contract is not None
        and record.parameter_sources
    )


def registration_applies(
    registration: CoraplexCapabilityRegistration, request: ExecutionRequest
) -> bool:
    """
    Whether a registration's request condition holds for one request.
    """
    condition = registration.applies_when
    if condition is None:
        return True
    arguments = request.argument_map
    if condition.role not in arguments:
        return False
    return condition.value is None or arguments[condition.role] == condition.value


def applicable_registration(
    registrations: Iterable[CoraplexCapabilityRegistration],
    request: ExecutionRequest,
    available_resources: frozenset[SymbolType],
) -> CoraplexCapabilityRegistration:
    """
    The registration realizing a request on a robot with the given resources: among the
    variants whose condition holds and whose resource needs are met, the one needing the
    most resources.
    """
    candidates = [
        registration
        for registration in registrations
        if registration_applies(registration, request)
        and not missing_resources(available_resources, registration.required_resources)
    ]
    if not candidates:
        raise NoApplicableRealizationError(request)
    return max(candidates, key=lambda item: len(item.required_resources))


class NoApplicableRealizationError(Exception):
    """
    Raised when a capability is registered but no variant fits the request and robot.
    """

    def __init__(self, request: ExecutionRequest):
        super().__init__(
            f"No registered realization of '{request.capability_ref.uid}' applies to "
            f"{request} on this robot."
        )


@cache
def discover_coraplex_capability_contract_drafts(
    package_root: Path | None = None,
) -> tuple[CoraplexCapabilityContractDraft, ...]:
    """
    Read Coraplex action declarations without importing the ROS runtime.
    """
    root = package_root or installed_coraplex_root()
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
    contracts: Iterable[CapabilityContract],
    workspace: CoraplexRealizationWorkspace | None,
    package_root: Path | None = None,
) -> CoraplexCapabilityInitialization:
    """
    Scan native actions and join each with the realizations the workspace admits for
    the given contracts; an approval whose action source changed is not admitted, and
    without a workspace nothing is.
    """
    root = package_root if package_root is not None else installed_coraplex_root()
    contracts = tuple(contracts)
    by_uid = {contract.uid: contract for contract in contracts}
    approved_by_action: dict[str, list[ApprovedRealization]] = {}
    for approved in () if workspace is None else workspace.approved():
        if approved.realization.capability_uid in by_uid:
            approved_by_action.setdefault(approved.action_source_id, []).append(
                approved
            )
    records: list[CoraplexCapabilityRegistration] = []
    for draft in discover_coraplex_capability_contract_drafts(package_root):
        reviews = approved_by_action.get(draft.source_id, [])
        if not reviews:
            records.append(
                CoraplexCapabilityRegistration(
                    draft=draft, status=CapabilityReviewStatus.PENDING
                )
            )
            continue
        for approved in reviews:
            records.append(
                CoraplexCapabilityRegistration(
                    draft=draft,
                    status=(
                        CapabilityReviewStatus.APPROVED
                        if workspace.drift(approved, root) is None
                        else CapabilityReviewStatus.SOURCE_CHANGED
                    ),
                    contract=by_uid[approved.realization.capability_uid],
                    required_resources=frozenset(
                        approved.realization.required_resources
                    ),
                    reviewed_by=approved.reviewed_by,
                    parameter_sources=approved.realization.parameter_sources,
                    applies_when=approved.realization.applies_when,
                )
            )
    return CoraplexCapabilityInitialization(tuple(records), contracts)


def approve_realization_candidate(
    workspace: CoraplexRealizationWorkspace,
    candidate_id: str,
    reviewer: str,
    contracts: Iterable[CapabilityContract],
    review_note: str | None = None,
    package_root: Path | None = None,
) -> ApprovedRealization:
    """
    Approve one candidate against the currently installed Coraplex and the given
    reviewed contracts.
    """
    root = package_root if package_root is not None else installed_coraplex_root()
    return workspace.approve(
        candidate_id,
        reviewer,
        discover_coraplex_capability_contract_drafts(package_root),
        contracts,
        root,
        review_note,
    )


def compatible_coraplex_action_ids(
    robot: AbstractRobot, initialization: CoraplexCapabilityInitialization
) -> frozenset[str]:
    """
    Native actions whose resource needs match the CRAM robot description.
    """
    return frozenset(
        source_id
        for support in infer_coraplex_capability_support(robot, initialization)
        for source_id in support.action_source_ids
    )


def render_coraplex_capability_candidates(
    robot: AbstractRobot, initialization: CoraplexCapabilityInitialization
) -> str:
    """
    Render only native actions structurally compatible with one robot.
    """
    support = infer_coraplex_capability_support(robot, initialization)
    compatible_ids = compatible_coraplex_action_ids(robot, initialization)
    capability_by_source: dict[str, set[str]] = {}
    for item in support:
        for source_id in item.action_source_ids:
            capability_by_source.setdefault(source_id, set()).add(item.capability_uid)
    lines = []
    for record in initialization.records:
        draft = record.draft
        if draft.source_id not in compatible_ids:
            continue
        capability_uids = ", ".join(sorted(capability_by_source[draft.source_id]))
        lines.append(
            f"{draft.render()}; compatible with {type(robot).__name__}; "
            f"reviewed semantics {{{capability_uids}}}"
        )
    return "\n".join(dict.fromkeys(lines)) or "- none"


def realization_evidence(
    initialization: CoraplexCapabilityInitialization,
) -> tuple[CoraplexCapabilityRealizationEvidence, ...]:
    """
    Link every admitted action to its capability and robot needs.
    """
    evidence = [
        CoraplexCapabilityRealizationEvidence(
            capability_uid=record.contract.uid,
            action_source_id=record.draft.source_id,
            required_resources=record.required_resources,
        )
        for record in initialization.records
        if record.status is CapabilityReviewStatus.APPROVED
        and record.contract is not None
    ]
    return tuple(
        sorted(evidence, key=lambda item: (item.capability_uid, item.action_source_id))
    )


def coraplex_capability_catalog(
    initialization: CoraplexCapabilityInitialization,
) -> dict:
    """
    Serialize the complete reviewed contract-to-Coraplex design map.

    The result is platform-structure independent and can therefore be shown by the host-
    only Viewer. Robot-specific compatibility is derived from the CRAM robot annotation
    at runtime.
    """
    adapters = adapter_capability_uids(initialization)
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
    for contract in initialization.contracts:
        has_reviewed_action = bool(evidence_by_capability.get(contract.uid))
        if contract.uid not in adapters or not has_reviewed_action:
            status = CapabilityRealizationStatus.ADAPTER_MISSING
        else:
            status = CapabilityRealizationStatus.READY
        verification = CapabilityVerificationStatus.TASK_REQUIRED
        actions = []
        for evidence in evidence_by_capability.get(contract.uid, []):
            draft = drafts[evidence.action_source_id]
            actions.append(
                {
                    **draft.to_json(),
                    "required_resources": sorted(
                        item.python_type_ref for item in evidence.required_resources
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
            "task_verification_required": len(entries),
            "pending_review": len(pending_records),
        },
    }


def robot_resources(robot: AbstractRobot) -> frozenset[SymbolType]:
    """
    Every CRAM robot part type the robot's declared parts belong to.
    """
    parts = [
        *robot.get_arms(),
        *robot.get_end_effectors(),
        *robot.get_sensors(),
    ]
    torso = robot.get_torso_if_specified()
    if torso is not None:
        parts.append(torso)
    part_types = {
        SymbolType.from_python_type(base)
        for part in parts
        for base in type(part).__mro__
        if issubclass(base, AbstractRobotPart)
        and base.__module__.startswith(ROBOT_PARTS)
    }
    if isinstance(robot, HasMobileBase):
        part_types.update(
            SymbolType.from_python_type(base)
            for base in type(robot.mobile_base).__mro__
            if issubclass(base, MobileBase) and base.__module__.startswith(ROBOT_PARTS)
        )
    elif robot.drive is not None:
        part_types.add(SymbolType.from_python_type(MobileBase))
    return frozenset(part_types)


ROBOT_PARTS = "semantic_digital_twin.robots"
"""
Module prefix of the CRAM robot part types resources are spelled in.
"""


def missing_resources(
    available: frozenset[SymbolType], required: Iterable[SymbolType]
) -> frozenset[SymbolType]:
    """
    Required part types the robot lacks.
    """
    return frozenset(required) - available


def infer_coraplex_capability_support(
    robot: AbstractRobot, initialization: CoraplexCapabilityInitialization
) -> tuple[CoraplexCapabilitySupport, ...]:
    """
    Classify every reviewed capability for one robot.

    Effect verification is task-dependent: selected effect predicates must provide a
    reviewed predicate query before planning can proceed.
    """
    available_resources = robot_resources(robot)
    adapters = adapter_capability_uids(initialization)
    evidence_by_capability: dict[str, list[CoraplexCapabilityRealizationEvidence]] = {}
    for evidence in realization_evidence(initialization):
        evidence_by_capability.setdefault(evidence.capability_uid, []).append(evidence)

    support = []
    for contract in initialization.contracts:
        evidence = evidence_by_capability.get(contract.uid, [])
        compatible = [
            item
            for item in evidence
            if not missing_resources(available_resources, item.required_resources)
        ]
        if not compatible:
            missing_sets = [
                missing_resources(available_resources, item.required_resources)
                for item in evidence
            ]
            missing = min(
                missing_sets,
                key=lambda items: (len(items), tuple(sorted(items))),
                default=frozenset(),
            )
            status = CapabilityRealizationStatus.ROBOT_INCOMPATIBLE
        elif contract.uid not in adapters:
            missing = frozenset()
            status = CapabilityRealizationStatus.ADAPTER_MISSING
        else:
            missing = frozenset()
            status = CapabilityRealizationStatus.READY
        verification = CapabilityVerificationStatus.TASK_REQUIRED
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
                missing_resources=tuple(sorted(missing)),
            )
        )
    return tuple(support)


def available_coraplex_capabilities(
    robot: AbstractRobot, initialization: CoraplexCapabilityInitialization
) -> frozenset[str]:
    """
    Reviewed capabilities the adapter can execute on the CRAM robot.
    """
    return frozenset(
        item.capability_uid
        for item in infer_coraplex_capability_support(robot, initialization)
        if item.ready
    )


def installed_coraplex_root() -> Path:
    """
    Root of the installed Coraplex package, or a path that exists nowhere.
    """
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
