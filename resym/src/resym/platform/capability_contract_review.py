"""
Human review of capability contracts.

A contract is proposed as a candidate — by a language model at initialization, by the
repair agent's gap report, or by a reference bootstrap — and enters the symbol library
only after a reviewer approves it in the local workspace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json
from typing_extensions import TYPE_CHECKING, Iterable, Mapping

from resym.core.capability_model import CapabilityContract
from resym.platform.coraplex_realizations import CapabilityReviewStatus

if TYPE_CHECKING:
    from resym.platform.coraplex_catalog import CoraplexCapabilityContractDraft


# %% records


@dataclass(frozen=True)
class CapabilityContractCandidate:
    """
    A proposed capability contract awaiting a human decision.
    """

    candidate_id: str
    """
    Workspace-unique identifier, also the candidate file name.
    """

    contract: CapabilityContract
    """
    The proposed contract.
    """

    action_source_ids: tuple[str, ...]
    """
    Scanned native actions cited as evidence that the platform can realize it.
    """

    generated_by: str
    """
    Who or what drafted the candidate.
    """

    rationale: str
    """
    Why the drafter believes the platform offers this capability.
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
        object.__setattr__(self, "action_source_ids", tuple(self.action_source_ids))
        object.__setattr__(
            self, "review_status", CapabilityReviewStatus(self.review_status)
        )


@dataclass(frozen=True)
class ApprovedContract:
    """
    A contract a reviewer admitted into the library.
    """

    contract: CapabilityContract
    """
    The admitted contract.
    """

    candidate_id: str
    """
    The candidate the approval decided.
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
    def identity(self) -> tuple[str, str]:
        """
        What one approval stands for: a contract uid at a version.
        """
        return (self.contract.uid, self.contract.version)


# %% review


class ContractReviewError(Exception):
    """
    Raised when a contract candidate cannot be approved as proposed.
    """

    def __init__(self, candidate_id: str, objections: Iterable[str]):
        self.objections = tuple(objections)
        super().__init__(f"Candidate '{candidate_id}': " + "; ".join(self.objections))


class DuplicateContractCandidateError(Exception):
    """
    Raised when a contract candidate id is submitted twice.
    """

    def __init__(self, candidate_id: str):
        super().__init__(f"Contract candidate '{candidate_id}' was already submitted.")


class UnknownContractCandidateError(Exception):
    """
    Raised when a review decision names a candidate the workspace does not hold.
    """

    def __init__(self, candidate_id: str):
        super().__init__(f"No contract candidate '{candidate_id}' in the workspace.")


CONTRACT_UID = re.compile(r"[a-z][a-z0-9-]*:[A-Za-z][A-Za-z0-9]*")
"""
A contract uid is a namespace and a capability name.
"""


def contract_objections(
    candidate: CapabilityContractCandidate,
    drafts: Mapping[str, CoraplexCapabilityContractDraft],
    approved: Mapping[tuple[str, str], CapabilityContract],
) -> tuple[str, ...]:
    """
    Reasons a contract candidate cannot be approved, checked against the scanned actions
    and the contracts already admitted.
    """
    contract = candidate.contract
    objections: list[str] = []
    if not CONTRACT_UID.fullmatch(contract.uid):
        objections.append(f"uid '{contract.uid}' is not '<namespace>:<Capability>'")
    if not contract.label.strip():
        objections.append("label is empty")
    if not contract.roles:
        objections.append("a contract declares at least one role")
    role_names = [role.name for role in contract.roles]
    for name in {name for name in role_names if role_names.count(name) > 1}:
        objections.append(f"role '{name}' is declared twice")
    if not contract.verifiable_effects:
        objections.append("a contract declares at least one verifiable effect")
    effect_names = set(contract.verifiable_effect_names)
    for effect, role, value in contract.effect_role_values:
        if effect not in effect_names:
            objections.append(f"effect '{effect}' is bound but not declared")
        if role not in role_names:
            objections.append(f"effect '{effect}' binds undeclared role '{role}'")
        elif value not in contract.role_map[role].allowed_values:
            objections.append(
                f"effect '{effect}' binds role '{role}' to '{value}', which it does "
                "not allow"
            )
    for source_id in candidate.action_source_ids:
        if source_id not in drafts:
            objections.append(f"unknown native action '{source_id}'")
    if (contract.uid, contract.version) in approved:
        objections.append(
            f"'{contract.uid}' version {contract.version} is already approved; a "
            "change needs a new version"
        )
    return tuple(objections)


# %% workspace


@dataclass
class CapabilityContractWorkspace:
    """
    Local review workspace for contract candidates and approved contracts.
    """

    root: Path
    """
    Directory holding the candidate queue, the approved contracts, and the log.
    """

    @property
    def candidate_directory(self) -> Path:
        return self.root / "candidates"

    @property
    def catalog_path(self) -> Path:
        return self.root / "contracts.json"

    @property
    def review_log_path(self) -> Path:
        return self.root / "review_log.jsonl"

    def submit(self, candidate: CapabilityContractCandidate) -> None:
        """
        Queue a candidate for review.
        """
        path = self._candidate_path(candidate.candidate_id)
        if path.exists():
            raise DuplicateContractCandidateError(candidate.candidate_id)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        self._write_candidate(candidate)

    def candidates(self) -> tuple[CapabilityContractCandidate, ...]:
        """
        All candidate records, including completed reviews.
        """
        if not self.candidate_directory.is_dir():
            return ()
        return tuple(
            from_json(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.candidate_directory.glob("*.json"))
        )

    def approved(self) -> tuple[ApprovedContract, ...]:
        """
        Every approval on record.
        """
        if not self.catalog_path.is_file():
            return ()
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return tuple(from_json(item) for item in data["contracts"])

    def approved_contracts(self) -> tuple[CapabilityContract, ...]:
        """
        The contracts admitted into the library, in uid order.
        """
        return tuple(item.contract for item in self.approved())

    def approve(
        self,
        candidate_id: str,
        reviewer: str,
        drafts: Iterable[CoraplexCapabilityContractDraft],
        review_note: str | None = None,
    ) -> ApprovedContract:
        """
        Admit one candidate after checking it against the scanned actions and the
        contracts already admitted.
        """
        candidate = self._candidate(candidate_id)
        if candidate.review_status is not CapabilityReviewStatus.PENDING:
            raise ContractReviewError(
                candidate_id, ("candidate already has a review decision",)
            )
        current = {item.identity: item for item in self.approved()}
        objections = contract_objections(
            candidate,
            {draft.source_id: draft for draft in drafts},
            {identity: item.contract for identity, item in current.items()},
        )
        if objections:
            raise ContractReviewError(candidate_id, objections)
        approved = ApprovedContract(
            contract=candidate.contract,
            candidate_id=candidate_id,
            reviewed_by=reviewer,
            approved_at=datetime.now(UTC).isoformat(),
        )
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
        self._log("approved", candidate, reviewer, review_note)
        return approved

    def reject(
        self, candidate_id: str, reviewer: str, review_note: str
    ) -> CapabilityContractCandidate:
        """
        Record a human rejection.
        """
        candidate = self._candidate(candidate_id)
        if candidate.review_status is not CapabilityReviewStatus.PENDING:
            raise ContractReviewError(
                candidate_id, ("candidate already has a review decision",)
            )
        rejected = replace(
            candidate,
            review_status=CapabilityReviewStatus.REJECTED,
            reviewed_by=reviewer,
            review_note=review_note,
        )
        self._write_candidate(rejected)
        self._log("rejected", candidate, reviewer, review_note)
        return rejected

    def _candidate(self, candidate_id: str) -> CapabilityContractCandidate:
        path = self._candidate_path(candidate_id)
        if not path.is_file():
            raise UnknownContractCandidateError(candidate_id)
        return from_json(json.loads(path.read_text(encoding="utf-8")))

    def _candidate_path(self, candidate_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id):
            raise ValueError(f"Invalid candidate id '{candidate_id}'.")
        return self.candidate_directory / f"{candidate_id}.json"

    def _write_candidate(self, candidate: CapabilityContractCandidate) -> None:
        self._candidate_path(candidate.candidate_id).write_text(
            json.dumps(to_json(candidate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_catalog(self, approved: Iterable[ApprovedContract]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "contracts": [
                to_json(item)
                for item in sorted(approved, key=lambda item: item.identity)
            ]
        }
        self.catalog_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log(
        self,
        decision: str,
        candidate: CapabilityContractCandidate,
        reviewer: str,
        review_note: str | None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "decision": decision,
            "candidate_id": candidate.candidate_id,
            "contract_uid": candidate.contract.uid,
            "contract_version": candidate.contract.version,
            "reviewed_by": reviewer,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "review_note": review_note,
        }
        with self.review_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
