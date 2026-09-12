"""
Reviewed realizations of capabilities by native Coraplex actions.

A realization says which native action carries out a capability and where each of the
action's parameters takes its value. Realizations are proposed as candidates, approved
by a human in a local workspace, and pinned to the checksum of the action's source so a
changed action withdraws the approval instead of silently changing what executes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from resym.core.symbol_types import SymbolType
from typing_extensions import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from resym.core.capability_model import CapabilityContract
    from resym.platform.coraplex_catalog import CoraplexCapabilityContractDraft


# %% review vocabulary


class CapabilityReviewStatus(StrEnum):
    """
    Admission state of a scanned Coraplex action mapping.
    """

    PENDING = "pending"
    """
    No human decision yet.
    """

    APPROVED = "approved"
    """
    Admitted, and the action source still matches the approval.
    """

    REJECTED = "rejected"
    """
    Turned down by a reviewer.
    """

    SOURCE_CHANGED = "source-changed"
    """
    Approved once, but the action source changed since and needs review again.
    """


class ParameterSourceKind(StrEnum):
    """
    Where a native action parameter's value comes from.
    """

    ROLE = "role"
    """
    A capability role bound in the execution request.
    """

    CONTEXT = "context"
    """
    A value the evaluation context provides for the current robot.
    """

    CONSTANT = "constant"
    """
    A fixed value spelled in the reviewed mapping.
    """


class ContextValue(StrEnum):
    """
    Values the evaluation context can supply to a native action.
    """

    MANIPULATION_ARM = "manipulation_arm"
    DEFAULT_GRASP = "default_grasp"
    WITNESS_BASE_POSE = "witness_base_pose"
    """The base pose grounding recorded for the objects bound to ``key_roles``."""


# %% realization records


@dataclass(frozen=True)
class ParameterSource:
    """
    Where one native action parameter takes its value.
    """

    parameter: str
    """
    Name of the native action's constructor parameter.
    """

    kind: ParameterSourceKind
    """
    Which kind of source supplies the value.
    """

    value: str
    """
    Role name, context value name, or constant, according to ``kind``.
    """

    key_roles: tuple[str, ...] = ()
    """
    Roles whose bound objects key a context value stored per object pair.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ParameterSourceKind(self.kind))
        object.__setattr__(self, "value", str(self.value))
        object.__setattr__(self, "key_roles", tuple(self.key_roles))


@dataclass(frozen=True)
class RoleCondition:
    """
    When a realization applies: a role is bound, optionally to one value.
    """

    role: str
    """
    The capability role inspected on the execution request.
    """

    value: str | None = None
    """
    The bound value required, or ``None`` when being bound at all suffices.
    """

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", str(self.value))


@dataclass(frozen=True)
class ActionRealization:
    """
    One reviewed way a native action realizes a capability.
    """

    capability_uid: str
    """
    The capability contract this action realizes.
    """

    parameter_sources: tuple[ParameterSource, ...] = ()
    """
    How each native parameter is filled; empty while no adapter is reviewed.
    """

    applies_when: RoleCondition | None = None
    """
    Request condition selecting this realization among a capability's variants.
    """

    required_resources: tuple[SymbolType, ...] = ()
    """
    Robot part types the action implementation needs, as CRAM type references.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_sources", tuple(self.parameter_sources))
        object.__setattr__(
            self,
            "required_resources",
            tuple(
                sorted(
                    item if isinstance(item, SymbolType) else SymbolType(item)
                    for item in self.required_resources
                )
            ),
        )


@dataclass(frozen=True)
class RealizationCandidate:
    """
    A proposed realization awaiting a human decision.
    """

    candidate_id: str
    """
    Workspace-unique identifier, also the candidate file name.
    """

    action_source_id: str
    """
    Stable id of the scanned native action the realization uses.
    """

    realization: ActionRealization
    """
    The proposed mapping.
    """

    generated_by: str
    """
    Who or what drafted the candidate.
    """

    rationale: str
    """
    Why the drafter believes the action realizes the capability.
    """

    review_status: CapabilityReviewStatus = CapabilityReviewStatus.PENDING
    """
    Current human-review decision.
    """

    reviewed_by: str | None = None
    """
    Reviewer identity after a decision.
    """

    review_note: str | None = None
    """
    Reviewer explanation after a decision.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "review_status", CapabilityReviewStatus(self.review_status)
        )


@dataclass(frozen=True)
class ApprovedRealization:
    """
    A human-approved realization pinned to the action source it was approved against.
    """

    action_source_id: str
    """
    Stable id of the native action.
    """

    action_class: str
    """
    Importable class reference of the native action.
    """

    action_source_file: str
    """
    Action source path relative to the Coraplex package root.
    """

    action_checksum: str
    """
    SHA-256 of the action source file at approval time.
    """

    realization: ActionRealization
    """
    The approved mapping.
    """

    reviewed_by: str
    """
    Reviewer who approved.
    """

    approved_at: str
    """
    Approval time in ISO 8601.
    """

    @property
    def identity(self) -> tuple[str, str, RoleCondition | None]:
        """
        What one approval stands for: an action, a capability, and a variant condition.
        """
        return (
            self.action_source_id,
            self.realization.capability_uid,
            self.realization.applies_when,
        )


# %% review


class RealizationReviewError(Exception):
    """
    Raised when a candidate cannot be approved as proposed.
    """

    def __init__(self, candidate_id: str, objections: Iterable[str]):
        self.objections = tuple(objections)
        super().__init__(f"Candidate '{candidate_id}': " + "; ".join(self.objections))


class DuplicateRealizationCandidateError(Exception):
    """
    Raised when a candidate id is submitted twice.
    """

    def __init__(self, candidate_id: str):
        super().__init__(f"Candidate '{candidate_id}' was already submitted.")


class UnknownRealizationCandidateError(Exception):
    """
    Raised when a review decision names a candidate the workspace does not hold.
    """

    def __init__(self, candidate_id: str):
        super().__init__(f"No candidate '{candidate_id}' in the workspace.")


def realization_objections(
    candidate: RealizationCandidate,
    drafts: Mapping[str, CoraplexCapabilityContractDraft],
    contracts: Mapping[str, CapabilityContract],
) -> tuple[str, ...]:
    """
    Reasons a candidate cannot be approved, checked against scanned actions and reviewed
    contracts alone.
    """
    objections: list[str] = []
    realization = candidate.realization
    if candidate.action_source_id not in drafts:
        objections.append(f"unknown native action '{candidate.action_source_id}'")
    contract = contracts.get(realization.capability_uid)
    if contract is None:
        objections.append(f"unknown capability '{realization.capability_uid}'")
        return tuple(objections)
    declared_roles = {role.name for role in contract.roles}
    context_values = {item.value for item in ContextValue}
    parameters: set[str] = set()
    for source in realization.parameter_sources:
        if source.parameter in parameters:
            objections.append(f"parameter '{source.parameter}' is bound twice")
        parameters.add(source.parameter)
        if (
            source.kind is ParameterSourceKind.ROLE
            and source.value not in declared_roles
        ):
            objections.append(
                f"role '{source.value}' is not declared by '{contract.uid}'"
            )
        if (
            source.kind is ParameterSourceKind.CONTEXT
            and source.value not in context_values
        ):
            objections.append(f"context value '{source.value}' does not exist")
        for role in source.key_roles:
            if role not in declared_roles:
                objections.append(
                    f"key role '{role}' is not declared by '{contract.uid}'"
                )
    condition = realization.applies_when
    if condition is not None and condition.role not in declared_roles:
        objections.append(
            f"condition role '{condition.role}' is not declared by '{contract.uid}'"
        )
    return tuple(objections)


def action_source_checksum(package_root: Path, source_file: str) -> str:
    """
    SHA-256 of one native action's source file under the Coraplex package root.
    """
    return hashlib.sha256((package_root / source_file).read_bytes()).hexdigest()


# %% workspace


@dataclass
class CoraplexRealizationWorkspace:
    """
    Local review workspace for realization candidates and approved realizations.
    """

    root: Path
    """
    Directory holding the candidate queue, the approved catalog, and the log.
    """

    @property
    def candidate_directory(self) -> Path:
        return self.root / "candidates"

    @property
    def catalog_path(self) -> Path:
        return self.root / "realizations.json"

    @property
    def review_log_path(self) -> Path:
        return self.root / "review_log.jsonl"

    def submit(self, candidate: RealizationCandidate) -> None:
        """
        Queue a candidate for review.
        """
        path = self._candidate_path(candidate.candidate_id)
        if path.exists():
            raise DuplicateRealizationCandidateError(candidate.candidate_id)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        self._write_candidate(candidate)

    def candidates(self) -> tuple[RealizationCandidate, ...]:
        """
        All candidate records, including completed reviews.
        """
        if not self.candidate_directory.is_dir():
            return ()
        return tuple(
            from_json(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.candidate_directory.glob("*.json"))
        )

    def approved(self) -> tuple[ApprovedRealization, ...]:
        """
        Every realization a reviewer approved, whether or not its source still matches.
        """
        if not self.catalog_path.is_file():
            return ()
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return tuple(from_json(item) for item in data["realizations"])

    def approve(
        self,
        candidate_id: str,
        reviewer: str,
        drafts: Iterable[CoraplexCapabilityContractDraft],
        contracts: Iterable[CapabilityContract],
        package_root: Path,
        review_note: str | None = None,
    ) -> ApprovedRealization:
        """
        Admit one candidate after checking it against the scanned actions and the
        reviewed contracts, pinning the action source it was approved against.
        """
        candidate = self._candidate(candidate_id)
        if candidate.review_status is not CapabilityReviewStatus.PENDING:
            raise RealizationReviewError(
                candidate_id, ("candidate already has a review decision",)
            )
        drafts_by_id = {draft.source_id: draft for draft in drafts}
        objections = realization_objections(
            candidate,
            drafts_by_id,
            {contract.uid: contract for contract in contracts},
        )
        if objections:
            raise RealizationReviewError(candidate_id, objections)
        draft = drafts_by_id[candidate.action_source_id]
        approved = ApprovedRealization(
            action_source_id=draft.source_id,
            action_class=draft.action_class,
            action_source_file=draft.source_file,
            action_checksum=action_source_checksum(package_root, draft.source_file),
            realization=candidate.realization,
            reviewed_by=reviewer,
            approved_at=datetime.now(UTC).isoformat(),
        )
        current = {item.identity: item for item in self.approved()}
        current[approved.identity] = approved
        self._write_catalog(current.values())
        self._write_candidate(
            replace(
                candidate,
                review_status=CapabilityReviewStatus.APPROVED,
                reviewed_by=reviewer,
                review_note=review_note,
            )
        )
        self._log("approved", candidate, reviewer, review_note, approved)
        return approved

    def reject(
        self, candidate_id: str, reviewer: str, review_note: str
    ) -> RealizationCandidate:
        """
        Record a human rejection.
        """
        candidate = self._candidate(candidate_id)
        if candidate.review_status is not CapabilityReviewStatus.PENDING:
            raise RealizationReviewError(
                candidate_id, ("candidate already has a review decision",)
            )
        rejected = replace(
            candidate,
            review_status=CapabilityReviewStatus.REJECTED,
            reviewed_by=reviewer,
            review_note=review_note,
        )
        self._write_candidate(rejected)
        self._log("rejected", candidate, reviewer, review_note, None)
        return rejected

    def drift(self, approved: ApprovedRealization, package_root: Path) -> str | None:
        """
        How the action source diverged from the approval, if it did.
        """
        source_file = package_root / approved.action_source_file
        if not source_file.is_file():
            return f"action source '{approved.action_source_file}' is missing"
        checksum = action_source_checksum(package_root, approved.action_source_file)
        if checksum != approved.action_checksum:
            return (
                f"action source '{approved.action_source_file}' changed from "
                f"{approved.action_checksum} to {checksum}"
            )
        return None

    def _candidate(self, candidate_id: str) -> RealizationCandidate:
        path = self._candidate_path(candidate_id)
        if not path.is_file():
            raise UnknownRealizationCandidateError(candidate_id)
        return from_json(json.loads(path.read_text(encoding="utf-8")))

    def _candidate_path(self, candidate_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id):
            raise ValueError(f"Invalid candidate id '{candidate_id}'.")
        return self.candidate_directory / f"{candidate_id}.json"

    def _write_candidate(self, candidate: RealizationCandidate) -> None:
        self._candidate_path(candidate.candidate_id).write_text(
            json.dumps(to_json(candidate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_catalog(self, approved: Iterable[ApprovedRealization]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "realizations": [
                to_json(item)
                for item in sorted(approved, key=lambda item: item.identity[:2])
            ]
        }
        self.catalog_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log(
        self,
        decision: str,
        candidate: RealizationCandidate,
        reviewer: str,
        review_note: str | None,
        approved: ApprovedRealization | None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "decision": decision,
            "candidate_id": candidate.candidate_id,
            "action_source_id": candidate.action_source_id,
            "capability_uid": candidate.realization.capability_uid,
            "reviewed_by": reviewer,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "review_note": review_note,
            "action_checksum": approved.action_checksum if approved else None,
        }
        with self.review_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
