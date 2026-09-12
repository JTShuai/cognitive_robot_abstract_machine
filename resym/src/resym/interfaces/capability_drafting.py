"""Initialization-stage drafting of capability-contract candidates.

For every scanned native action no approved contract covers, a language model drafts
the semantic interface the action offers: roles with their CRAM types, verifiable
effects, and the success relation. Every accepted draft is only *submitted* to the
local review workspace — a human decision in the review Viewer is what admits a
contract into the library.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic import BaseModel, Field

from resym.core.capability_model import CapabilityContract, CapabilityRole
from resym.core.grounding_model import text_checksum
from resym.core.symbol_types import SymbolType
from resym.llm.prompting import render_prompt
from resym.llm.structured import StructuredCompleter
from resym.platform.capability_contract_review import (
    CapabilityContractCandidate,
    CapabilityContractWorkspace,
    contract_objections,
)
from resym.platform.coraplex_catalog import CoraplexCapabilityContractDraft
from resym.retrieval.ontology import OntologyIndex
from resym.retrieval.ontology_alignment import align_capability_contract
from typing_extensions import Iterable

DRAFTING_AGENT_NAME = "capability-contract-drafter"

MAXIMUM_DRAFT_ATTEMPTS = 3
"""
How many model drafts one native action may consume before it is reported failed.
"""

MAXIMUM_ALIGNMENT_STEPS = 8
"""
How many aligner steps one accepted draft may consume before it is submitted unaligned.
"""


@dataclass(frozen=True)
class CapabilityContractDraftFailure:
    """
    One native action no accepted contract draft could be produced for.
    """

    action_source_id: str
    """The scanned action the drafts were about."""

    objections: tuple[str, ...]
    """Review objections of the last rejected draft."""


@dataclass(frozen=True)
class CapabilityContractDraftingReport:
    """
    Outcome of one initialization drafting pass.
    """

    submitted: tuple[CapabilityContractCandidate, ...]
    """Candidates now pending human review."""

    failures: tuple[CapabilityContractDraftFailure, ...]
    """Actions whose drafts never passed review checks."""

    unaligned: tuple[UnalignedContractDraft, ...] = ()
    """Submitted candidates whose ontology alignment the aligner could not decide."""


@dataclass(frozen=True)
class UnalignedContractDraft:
    """
    A submitted candidate that carries no ontology alignment.
    """

    candidate_id: str
    """The candidate pending review without an alignment."""

    reason: str
    """Why the aligner submitted no admissible alignment."""


class CapabilityRoleModel(BaseModel):
    """
    One role of a drafted contract: typed by CRAM classes or by allowed constants.
    """

    name: str = Field(min_length=1)
    required: bool = True
    accepted_symbol_types: list[str] = Field(default_factory=list)
    allowed_values: list[str] = Field(default_factory=list)


class CapabilityContractDraftModel(BaseModel):
    """
    Structured draft the model must answer with.
    """

    uid: str = Field(min_length=1)
    label: str = Field(min_length=1)
    roles: list[CapabilityRoleModel] = Field(min_length=1)
    success_relation: str = Field(min_length=1)
    verifiable_effects: list[str] = Field(min_length=1)
    effect_role_values: list[list[str]] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


def draft_capability_contract_candidates(
    drafts: Iterable[CoraplexCapabilityContractDraft],
    workspace: CapabilityContractWorkspace,
    existing_contracts: Iterable[CapabilityContract],
    completer: StructuredCompleter,
    maximum_attempts: int = MAXIMUM_DRAFT_ATTEMPTS,
    covered_action_source_ids: frozenset[str] = frozenset(),
    ontology_index: OntologyIndex | None = None,
    maximum_alignment_steps: int = MAXIMUM_ALIGNMENT_STEPS,
) -> CapabilityContractDraftingReport:
    """
    Draft one pending-review contract per scanned action not yet covered.

    A draft the review checks reject is retried with its objections shown to the
    model; an action whose attempts are exhausted is reported as a failure instead
    of entering the review queue. With an ontology index, every accepted draft is
    aligned to the frozen ontologies by the alignment agent before submission; a
    draft the aligner cannot place is submitted unaligned and reported as such.
    """
    drafts = tuple(drafts)
    drafts_by_id = {draft.source_id: draft for draft in drafts}
    existing = tuple(existing_contracts)
    approved = {(contract.uid, contract.version): contract for contract in existing}
    submitted: list[CapabilityContractCandidate] = []
    failures: list[CapabilityContractDraftFailure] = []
    unaligned: list[UnalignedContractDraft] = []
    for draft in drafts:
        if draft.source_id in covered_action_source_ids:
            continue
        objections: tuple[str, ...] = ()
        for _ in range(maximum_attempts):
            proposal = completer.complete(
                DRAFTING_AGENT_NAME,
                _draft_prompt(draft, existing, objections),
                CapabilityContractDraftModel,
            )
            candidate, objections = _candidate(draft, proposal)
            if candidate is not None:
                objections = contract_objections(candidate, drafts_by_id, approved)
            if candidate is not None and not objections:
                if ontology_index is not None:
                    candidate, rejection = _aligned(
                        candidate, ontology_index, completer, maximum_alignment_steps
                    )
                    if rejection is not None:
                        unaligned.append(
                            UnalignedContractDraft(candidate.candidate_id, rejection)
                        )
                workspace.submit(candidate)
                submitted.append(candidate)
                break
        else:
            failures.append(
                CapabilityContractDraftFailure(
                    action_source_id=draft.source_id, objections=objections
                )
            )
    return CapabilityContractDraftingReport(
        submitted=tuple(submitted),
        failures=tuple(failures),
        unaligned=tuple(unaligned),
    )


def _aligned(
    candidate: CapabilityContractCandidate,
    index: OntologyIndex,
    completer: StructuredCompleter,
    maximum_steps: int,
) -> tuple[CapabilityContractCandidate, str | None]:
    """
    The candidate with the alignment the aligner admitted, or unchanged with the
    reason none was.
    """
    result = align_capability_contract(
        candidate.contract, index, completer, maximum_steps=maximum_steps
    )
    if not result.admitted:
        return candidate, result.rejection_reason
    return replace(candidate, contract=result.apply(candidate.contract)), None


def _candidate(
    draft: CoraplexCapabilityContractDraft, proposal: CapabilityContractDraftModel
) -> tuple[CapabilityContractCandidate | None, tuple[str, ...]]:
    """
    Build the candidate a proposal describes, or the reasons it cannot be built.
    """
    roles = []
    objections: list[str] = []
    for role in proposal.roles:
        if bool(role.accepted_symbol_types) == bool(role.allowed_values):
            objections.append(
                f"role '{role.name}' must declare exactly one of accepted_symbol_types "
                "or allowed_values"
            )
            continue
        symbol_types = []
        for reference in role.accepted_symbol_types:
            try:
                symbol_types.append(SymbolType(reference))
            except ValueError as error:
                objections.append(str(error))
        roles.append(
            CapabilityRole(
                role.name,
                required=role.required,
                allowed_values=tuple(role.allowed_values),
                accepted_symbol_types=tuple(symbol_types),
            )
            if not objections
            else None
        )
    for binding in proposal.effect_role_values:
        if len(binding) != 3:
            objections.append(
                "effect_role_values entries are [effect, role, value] triples"
            )
    if objections:
        return None, tuple(objections)
    try:
        contract = CapabilityContract(
            uid=proposal.uid,
            label=proposal.label,
            roles=tuple(roles),
            success_relation=proposal.success_relation,
            verifiable_effects=tuple(proposal.verifiable_effects),
            effect_role_values=tuple(
                tuple(item) for item in proposal.effect_role_values
            ),
        )
    except ValueError as error:
        # The contract model refuses inconsistent proposals; its reason is the
        # objection the next draft must answer.
        return None, (str(error),)
    candidate = CapabilityContractCandidate(
        candidate_id=(
            f"draft-{draft.source_id.rsplit('.', 1)[-1]}-"
            f"{text_checksum(proposal.model_dump_json())[:12]}"
        ),
        contract=contract,
        action_source_ids=(draft.source_id,),
        generated_by=DRAFTING_AGENT_NAME,
        rationale=proposal.rationale,
    )
    return candidate, ()


def _draft_prompt(
    draft: CoraplexCapabilityContractDraft,
    existing: tuple[CapabilityContract, ...],
    objections: tuple[str, ...],
) -> str:
    return render_prompt(
        "draft_capability_contract",
        action=draft.render(),
        summary=draft.summary or "(no docstring)",
        existing_contracts=(
            "\n".join(
                f"- {contract.uid}: {contract.label}; effects "
                + ", ".join(contract.verifiable_effect_names)
                for contract in existing
            )
            or "- none"
        ),
        objections="\n".join(f"- {item}" for item in objections) or "- none",
    )
