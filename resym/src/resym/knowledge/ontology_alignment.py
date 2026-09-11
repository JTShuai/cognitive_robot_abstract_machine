"""
Agent-decided alignment of capability terms to frozen ontologies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from typing_extensions import Callable, Optional

from resym.core.model import (
    CapabilityContract,
    OntologyAlignment,
)
from resym.knowledge.ontology import (
    InvalidOntologyAlignmentError,
    OntologyHit,
    OntologyIndex,
    OntologyKindMismatchError,
    UnknownOntologyIriError,
)
from resym.llm.client import CompletionUsage
from resym.llm.prompting import render_prompt
from resym.llm.schemas import (
    OntologyAgentAction,
    OntologyAlignmentProposal,
)
from resym.llm.structured import StructuredCompleter

ONTOLOGY_ALIGNER_AGENT = "ontology-aligner"


@dataclass(frozen=True)
class OntologyAlignmentResult:
    """
    Agent decision, evidence, and deterministic validation outcome.
    """

    contract_uid: str
    candidates: tuple[OntologyHit, ...]
    proposal: OntologyAlignmentProposal | None
    alignment: OntologyAlignment | None
    rejection_reason: str | None = None
    trajectory: tuple[dict, ...] = ()

    @property
    def admitted(self) -> bool:
        return self.alignment is not None

    def apply(self, contract: CapabilityContract) -> CapabilityContract:
        if contract.uid != self.contract_uid:
            raise ValueError(
                f"result is for '{self.contract_uid}', not '{contract.uid}'"
            )
        if self.alignment is None:
            raise ValueError(f"ontology proposal was rejected: {self.rejection_reason}")
        return replace(contract, ontology_alignment=self.alignment)


def align_capability_contract(
    contract: CapabilityContract,
    index: OntologyIndex,
    completer: StructuredCompleter,
    *,
    top_k: int = 20,
    maximum_steps: int = 8,
    charge_usage: Optional[Callable[[CompletionUsage], None]] = None,
) -> OntologyAlignmentResult:
    """
    Let a bounded LLM agent investigate and decide, then validate locally.
    """
    if maximum_steps < 1:
        raise ValueError("maximum_steps must be positive")
    candidates = tuple(index.retrieve_for_contract(contract, top_k=top_k))
    trajectory: list[dict] = []
    last_proposal: OntologyAlignmentProposal | None = None
    last_error = "agent reached its step limit without submitting an alignment"

    for _ in range(maximum_steps):
        action = completer.complete(
            ONTOLOGY_ALIGNER_AGENT,
            render_prompt(
                "ontology_alignment",
                contract=_render_contract(contract),
                candidates=_render_candidates(candidates, index),
                history=_render_trajectory(trajectory),
            ),
            OntologyAgentAction,
            charge_usage=charge_usage,
        )
        observation: str
        if action.tool == "search":
            if not action.query:
                observation = "search rejected: query is required"
            else:
                hits = tuple(index.retrieve(action.query, top_k=top_k))
                observation = _render_candidates(hits, index)
        elif action.tool == "inspect_entity":
            if not action.iri:
                observation = "inspection rejected: iri is required"
            else:
                try:
                    observation = _render_entity(index.require(action.iri), index)
                except UnknownOntologyIriError as error:
                    observation = f"inspection rejected: {error}"
        else:
            if action.proposal is None or not action.rationale:
                observation = "submission rejected: proposal and rationale are required"
            else:
                last_proposal = action.proposal
                try:
                    _validate_role_mapping(contract, action.proposal)
                    alignment = index.admit_alignment(**action.proposal.model_dump())
                except (
                    InvalidOntologyAlignmentError,
                    OntologyKindMismatchError,
                    UnknownOntologyIriError,
                ) as error:
                    last_error = str(error)
                    observation = f"submission rejected: {error}"
                else:
                    trajectory.append(_step_record(action, "alignment accepted"))
                    alignment = replace(
                        alignment,
                        decided_by="llm-agent",
                        decision_model=completer.client.description,
                        rationale=action.rationale,
                        evidence_iris=_evidence_iris(action.proposal),
                    )
                    return OntologyAlignmentResult(
                        contract_uid=contract.uid,
                        candidates=candidates,
                        proposal=action.proposal,
                        alignment=alignment,
                        trajectory=tuple(trajectory),
                    )
        trajectory.append(_step_record(action, observation))

    return OntologyAlignmentResult(
        contract_uid=contract.uid,
        candidates=candidates,
        proposal=last_proposal,
        alignment=None,
        rejection_reason=last_error,
        trajectory=tuple(trajectory),
    )


def apply_human_alignment_review(
    contract: CapabilityContract,
    index: OntologyIndex,
    proposal: OntologyAlignmentProposal,
    *,
    reviewer: str,
    rationale: str,
) -> CapabilityContract:
    """
    Optionally confirm or correct an agent decision after structural checks.
    """
    _validate_role_mapping(contract, proposal)
    alignment = index.admit_alignment(**proposal.model_dump())
    alignment = replace(
        alignment,
        decided_by="human",
        rationale=rationale,
        evidence_iris=_evidence_iris(proposal),
        human_reviewed=True,
        human_reviewer=reviewer,
    )
    return replace(contract, ontology_alignment=alignment)


def _step_record(action: OntologyAgentAction, observation: str) -> dict:
    return {"action": action.model_dump(), "observation": observation}


def _render_trajectory(trajectory: list[dict]) -> str:
    if not trajectory:
        return "(no tool calls yet)"
    return "\n\n".join(
        f"Step {position}: {step['action']}\nObservation: {step['observation']}"
        for position, step in enumerate(trajectory, start=1)
    )


def _evidence_iris(proposal: OntologyAlignmentProposal) -> tuple[str, ...]:
    values = (
        ([proposal.candidate_iri] if proposal.candidate_iri else [])
        + proposal.cited_classes
        + proposal.cited_properties
        + list(proposal.role_mapping.values())
    )
    return tuple(dict.fromkeys(values))


def _validate_role_mapping(
    contract: CapabilityContract, proposal: OntologyAlignmentProposal
) -> None:
    unknown_roles = sorted(set(proposal.role_mapping) - set(contract.role_map))
    if unknown_roles:
        raise InvalidOntologyAlignmentError(
            "role mapping references unknown local roles: " + ", ".join(unknown_roles)
        )


def _render_contract(contract: CapabilityContract) -> str:
    roles = "\n".join(
        f"- {role.name}"
        + (
            f"; accepted symbol types: "
            f"{', '.join(symbol_type.python_type_ref for symbol_type in role.accepted_symbol_types)}"
            if role.accepted_symbol_types
            else ""
        )
        + (
            f"; allowed values: {', '.join(role.allowed_values)}"
            if role.allowed_values
            else ""
        )
        for role in contract.roles
    )
    return (
        f"UID: {contract.uid}\n"
        f"label: {contract.label}\n"
        f"roles:\n{roles}\n"
        f"success relation: {contract.success_relation}\n"
        f"verifiable effects: {', '.join(contract.verifiable_effect_names)}"
    )


def _render_candidates(
    candidates: tuple[OntologyHit, ...], index: OntologyIndex
) -> str:
    if not candidates:
        return "(no lexical candidates)"
    rendered = []
    for position, hit in enumerate(candidates, start=1):
        entity = hit.entity
        sources = ", ".join(
            f"{source_id}@{index.sources[source_id].version}"
            for source_id in entity.source_ids
        )
        definition = entity.comments[0] if entity.comments else "(no definition)"
        rendered.append(
            f"{position}. IRI: {entity.iri}\n"
            f"   kind: {', '.join(sorted(kind.value for kind in entity.kinds))}\n"
            f"   labels: {', '.join(entity.labels) or entity.local_name}\n"
            f"   definition: {definition[:500]}\n"
            f"   parents: {', '.join(entity.parent_iris) or '(none)'}\n"
            f"   source: {sources}; retrieval score: {hit.score:.3f}"
        )
    return "\n".join(rendered)


def _render_entity(entity, index: OntologyIndex) -> str:
    sources = ", ".join(
        f"{source_id}@{index.sources[source_id].version}"
        for source_id in entity.source_ids
    )
    return (
        f"IRI: {entity.iri}\n"
        f"kind: {', '.join(sorted(kind.value for kind in entity.kinds))}\n"
        f"labels: {', '.join(entity.labels) or entity.local_name}\n"
        f"definitions: {' | '.join(entity.comments) or '(none)'}\n"
        f"parents: {', '.join(entity.parent_iris) or '(none)'}\n"
        f"source: {sources}"
    )
