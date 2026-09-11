"""Repair backends behind one interface, under one budget meter.

E1's fairness requirement (revised plan §9.4) is *matched budgets*:
every automatic backend runs with the same LLM and tool
whitelist, the same corpus and top-k, and the same
candidate/tool-call/probe/token budgets, metered by the same
:class:`BudgetMeter` — not by each backend's own bookkeeping. A backend
returns candidate patches and an episode log; it never touches the
library. Admission is the trusted curator's, and only the curator's,
decision.

The main method is agentic RAG: a stateful agent decides when and how to
retrieve UniDomain fragments, adapt them, inspect feedback, and retry.  The
backends in this module are comparison conditions for separating retrieval
from agentic control:

- :class:`ClosedBookBackend` — one-shot proposal, no retrieval;
- :class:`RagOneShotBackend` — non-agentic RAG with top-k rendered into
  one proposal prompt;
- :class:`FixedPipelineBackend` — non-agentic RAG with a fixed-order loop:
  retrieve once,
  then propose → static-check → feed objections back → repropose, no
  state beyond the last objection list, no probe selection.

The stateful agentic-RAG method and typed enumeration join this registry
through the same interface and meter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from itertools import combinations

from typing_extensions import TYPE_CHECKING, Callable, Optional, Sequence

from resym.repair.patch import ModelPatch
from resym.knowledge.retrieval import (
    FragmentIndex,
    RankedFragment,
    RetrievalQuery,
)
from resym.core.provenance import KnowledgeSource
from resym.llm.client import CompletionUsage, UsageSource
from resym.llm.prompting import (
    render_operators,
    render_predicates,
    render_prompt,
)
from resym.llm.schemas import LibraryProposal
from resym.llm.structured import (
    StructuredCompleter,
    StructuredOutputRetriesExceededError,
)
from resym.core.model import (
    CapabilityContract,
    CapabilityRole,
    Literal,
    Operator,
    OperatorExecutionBinding,
    PredicateRef,
    Provenance,
    RoleBinding,
    SymbolLibrary,
    SymbolType,
    is_symbol_subtype,
    resolve_symbol_type,
)
from resym.core.grounding import GroundingFactoryCandidate

if TYPE_CHECKING:
    from resym.repair.certificate import FailureCertificate

AGENTIC_RAG_BACKEND_NAME = "agentic-rag"
"""Canonical experiment name for the retrieval-augmented stateful agent."""


class RetrievalStatus(StrEnum):
    """
    Outcome of corpus retrieval within one repair episode.
    """

    NOT_ATTEMPTED = "not_attempted"
    CORPUS_UNAVAILABLE = "corpus_unavailable"
    RELEVANT_EVIDENCE_FOUND = "relevant_evidence_found"
    NO_RELEVANT_EVIDENCE = "no_relevant_evidence"


class EpisodeStep(StrEnum):
    """
    Step labels of the episode events the comparison backends record.
    """

    RETRIEVE = "retrieve"
    PROPOSE = "propose"
    CHECK = "check"
    ENUMERATE = "enumerate"
    BUDGET = "budget"
    INVALID_OUTPUT = "invalid-output"
    INVALID_PROPOSAL = "invalid-proposal"


class CapabilityGapKind(StrEnum):
    """
    What a reported execution-capability gap is missing.
    """

    MISSING_CONTRACT = "missing-capability-contract"
    MISSING_REALIZATION = "missing-capability-realization"


class BudgetExhaustedError(Exception):
    """Raised when a backend crosses the episode budget; the harness reports
    the episode as budget-exhausted, it is not an infrastructure error."""

    def __init__(self, dimension: str):
        super().__init__(f"Episode budget exhausted: {dimension}.")
        self.dimension = dimension


@dataclass(frozen=True)
class Budget:
    """One episode's allowance, identical across compared backends."""

    candidates: int = 3
    """Model patch proposals the backend may produce."""

    tool_calls: int = 24
    """Deterministic tool invocations (checks, retrievals, alignments)."""

    probes: int = 6
    """Two diagnostic probes for each of at most three candidates."""

    estimated_tokens: int = 200_000
    """Prompt+response budget, estimated at four characters per token —
    an honest approximation applied identically to every backend."""


@dataclass
class BudgetMeter:
    """Shared spending record; every backend spends through these methods."""

    budget: Budget = Budget()
    candidates_used: int = 0
    tool_calls_used: int = 0
    probes_used: int = 0
    estimated_tokens_used: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    provider_reported_calls: int = 0
    estimated_usage_calls: int = 0

    def spend_candidate(self) -> None:
        if self.candidates_used >= self.budget.candidates:
            raise BudgetExhaustedError("candidates")
        self.candidates_used += 1

    def spend_tool_call(self) -> None:
        if self.tool_calls_used >= self.budget.tool_calls:
            raise BudgetExhaustedError("tool_calls")
        self.tool_calls_used += 1

    def spend_probe(self) -> None:
        if self.probes_used >= self.budget.probes:
            raise BudgetExhaustedError("probes")
        self.probes_used += 1

    def spend_usage(self, usage: CompletionUsage) -> None:
        """Charge one complete model call using provider counts when known."""
        self.input_tokens_used += usage.input_tokens
        self.output_tokens_used += usage.output_tokens
        self.estimated_tokens_used += usage.total_tokens
        if usage.source is UsageSource.PROVIDER:
            self.provider_reported_calls += 1
        else:
            self.estimated_usage_calls += 1
        if self.estimated_tokens_used > self.budget.estimated_tokens:
            raise BudgetExhaustedError("estimated_tokens")

    def snapshot(self) -> dict:
        return {
            "candidates_used": self.candidates_used,
            "tool_calls_used": self.tool_calls_used,
            "probes_used": self.probes_used,
            "estimated_tokens_used": self.estimated_tokens_used,
            "input_tokens_used": self.input_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "provider_reported_calls": self.provider_reported_calls,
            "estimated_usage_calls": self.estimated_usage_calls,
        }


@dataclass
class RepairTask:
    """Everything a backend may look at."""

    certificate: FailureCertificate
    library: SymbolLibrary
    evaluator_listing: str
    """Menu of registered truth procedures, one per line."""

    capability_listing: str
    """Menu of available capability contracts, one per line."""

    capability_catalog: tuple[CapabilityContract, ...] = ()
    """Reviewed platform contracts, including implementations not ready here."""

    capability_draft_listing: str = ""
    """Untrusted platform action metadata that may suggest a missing contract."""

    capability_draft_ids: frozenset[str] = frozenset()
    """Platform source identifiers that a structured gap report may cite."""

    index: Optional[FragmentIndex] = None
    """The frozen corpus index; ``None`` for closed-book settings."""

    retrieval_top_k: int = 5
    check_patch: Optional[Callable[[ModelPatch], list[str]]] = None
    """The curator's static review as a read-only tool: objections, empty
    means clean. Backends may consult it; only the curator admits."""

    run_probe: Optional[Callable[[str, ModelPatch], dict]] = None
    """Diagnostic probe executor: (probe name, candidate patch) -> structured
    observation. Probe runs are metered; the mandatory suite is not
    reachable through this hook."""

    probe_listing: str = ""
    """Menu of available diagnostic probes, one per line."""

    required_probes: tuple[str, ...] = ()
    """Proposal-split probes that must pass before agent submission."""

    grounding_vocabulary_listing: str = ""
    """Reviewed EQL symbols from which a Tier 2 candidate may be composed."""

    grounding_factory_listing: str = ""
    """Current approved factories that a predicate grounding plan may reference."""

    validate_grounding_candidate: Optional[
        Callable[[GroundingFactoryCandidate], tuple[str, ...]]
    ] = None
    """Static source boundary applied before a candidate enters review."""

    grounding_candidate_sink: Optional[Callable[[GroundingFactoryCandidate], None]] = (
        None
    )
    """Pending-review persistence supplied by the application."""


class OutcomeStatus(Enum):
    PATCH_PROPOSED = "patch_proposed"
    NO_CANDIDATE = "no_candidate"
    MISSING_EXECUTION_CAPABILITY = "missing_execution_capability"
    GROUNDING_FACTORY_PROPOSED = "grounding_factory_proposed"
    MISSING_GROUNDING_CAPABILITY = "missing_grounding_capability"
    UNSUPPORTED_DECLARED = "unsupported_declared"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class MissingExecutionCapability:
    """Agent-proposed description of an execution interface the library lacks.

    This is a reviewable gap report, not authority to create a trusted
    ``CapabilityContract`` or a Coraplex program.
    """

    suggested_label: str
    desired_effects: tuple[str, ...]
    required_roles: tuple[tuple[str, str], ...]
    candidate_realizations: tuple[str, ...]
    reason: str
    gap_kind: CapabilityGapKind = CapabilityGapKind.MISSING_CONTRACT
    matching_contract_uids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MissingGroundingCapability:
    """Structured relation that approved world-query mechanisms cannot compute."""

    required_relation: str
    """Task-level relation whose truth cannot be obtained."""

    input_types: tuple[tuple[str, str], ...]
    """Semantic input roles and their required CRAM types."""

    missing_computation: str
    """World-query or computational ability absent from the platform."""

    reason: str
    """Evidence supporting the reported gap."""


@dataclass
class RepairOutcome:
    """What one backend episode produced, with its full event log."""

    backend: str
    status: OutcomeStatus
    patch: Optional[ModelPatch] = None
    missing_execution_capability: Optional[MissingExecutionCapability] = None
    missing_grounding_capability: Optional[MissingGroundingCapability] = None
    grounding_factory_candidate: Optional[GroundingFactoryCandidate] = None
    retrieved_ids: tuple[str, ...] = ()
    events: list[dict] = field(default_factory=list)
    budget: dict = field(default_factory=dict)


class RepairBackend(ABC):
    """One repair strategy; stateless across episodes."""

    name: str = "abstract"

    @abstractmethod
    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        """Run one episode under the shared meter."""


# -- shared machinery ---------------------------------------------------


def proposal_to_patch(
    proposal: LibraryProposal,
    backend_name: str,
    retrieved_ids: tuple[str, ...] = (),
    library: Optional[SymbolLibrary] = None,
    knowledge_source: Optional[KnowledgeSource] = None,
) -> ModelPatch:
    """Convert a proposal to a complete, replayable library increment.

    Partial operator updates are overlaid on the named operator in ``library``;
    new operators must still be complete.
    """
    provenance = Provenance(
        source="curation",
        proposal_backend=backend_name,
        retrieved_ids=retrieved_ids,
        knowledge_source=knowledge_source
        or (KnowledgeSource.RETRIEVAL if retrieved_ids else KnowledgeSource.LLM_PRIOR),
    )
    predicates = tuple(
        replace(p.to_predicate_symbol(), provenance=provenance)
        for p in proposal.predicates
    )
    operators = tuple(
        replace(
            o.to_operator(library.operators.get(o.name) if library else None),
            provenance=provenance,
        )
        for o in proposal.operators
    )
    return ModelPatch(
        predicates=predicates,
        operators=operators,
        rationale=proposal.rationale,
    )


def render_proposal_prompt(task: RepairTask, retrieved: str = "") -> str:
    """The shared proposal prompt: certificate evidence plus the library
    and registry menus; RAG settings append the retrieved fragments."""
    prompt = render_prompt(
        "propose_symbols",
        request=task.certificate.render(),
        predicates=render_predicates(task.library, mark_fluents=True),
        operators=render_operators(task.library),
        evaluators=task.evaluator_listing,
        predicate_queries=(
            task.grounding_factory_listing
            or task.evaluator_listing
            or "- none registered"
        ),
        skills=task.capability_listing,
        capability_candidates=task.capability_draft_listing or "- none",
        types=", ".join(t.python_type_ref for t in task.library.symbol_types),
    )
    if retrieved:
        prompt += (
            "\n\nRetrieved external fragments (untrusted, adapt rather than"
            " copy; cite nothing you cannot ground in the menus above):\n" + retrieved
        )
    return prompt


def render_fragments(hits: Sequence[RankedFragment]) -> str:
    parts = []
    for hit in hits:
        fragment = hit.fragment
        predicate_lines = "\n".join(
            f"    {p.signature}: {p.gloss}" for p in fragment.predicates
        )
        operator_lines = "\n".join(f"    {o.raw}" for o in fragment.operators)
        parts.append(
            f"- fragment {fragment.fragment_id} (score {hit.score:.2f}):\n"
            f"{predicate_lines}\n{operator_lines}"
        )
    return "\n".join(parts)


def retrieve_for(task: RepairTask, meter: BudgetMeter) -> list[RankedFragment]:
    if task.index is None:
        return []
    meter.spend_tool_call()
    query = RetrievalQuery.from_certificate(task.certificate)
    return task.index.retrieve(query, top_k=task.retrieval_top_k)


def retrieve_and_record(
    task: RepairTask, meter: BudgetMeter, outcome: RepairOutcome
) -> list[RankedFragment]:
    """Retrieve once for the task and record ids and status on the outcome."""
    hits = retrieve_for(task, meter)
    outcome.retrieved_ids = tuple(hit.fragment_id for hit in hits)
    outcome.events.append(
        {
            "step": EpisodeStep.RETRIEVE,
            "status": (
                RetrievalStatus.RELEVANT_EVIDENCE_FOUND
                if hits
                else RetrievalStatus.NO_RELEVANT_EVIDENCE
            ),
            "ids": list(outcome.retrieved_ids),
        }
    )
    return hits


PROPOSER_AGENT = "symbol-proposer"


def complete_proposal(
    completer: StructuredCompleter, prompt: str, meter: BudgetMeter
) -> LibraryProposal:
    meter.spend_candidate()
    proposal = completer.complete(
        PROPOSER_AGENT,
        prompt,
        LibraryProposal,
        charge_usage=meter.spend_usage,
    )
    return proposal


# -- the backends -------------------------------------------------------


@dataclass
class ProposalCompletionBackend(RepairBackend):
    """Episode shell shared by the proposal-completing comparison backends.

    ``repair`` owns the outcome bookkeeping every such backend repeats: it maps
    budget exhaustion, invalid model output, and invalid proposals onto outcome
    events and records the final budget snapshot. Subclasses implement only the
    proposal strategy.
    """

    completer: StructuredCompleter

    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        outcome = RepairOutcome(backend=self.name, status=OutcomeStatus.NO_CANDIDATE)
        try:
            self._propose(task, meter, outcome)
        except BudgetExhaustedError as error:
            outcome.status = OutcomeStatus.BUDGET_EXHAUSTED
            outcome.events.append(
                {"step": EpisodeStep.BUDGET, "dimension": error.dimension}
            )
        except StructuredOutputRetriesExceededError as error:
            outcome.events.append(
                {"step": EpisodeStep.INVALID_OUTPUT, "error": str(error)}
            )
        except ValueError as error:
            outcome.events.append(
                {"step": EpisodeStep.INVALID_PROPOSAL, "error": str(error)}
            )
        outcome.budget = meter.snapshot()
        return outcome

    @abstractmethod
    def _propose(
        self, task: RepairTask, meter: BudgetMeter, outcome: RepairOutcome
    ) -> None:
        """Run the proposal strategy, mutating the outcome."""


@dataclass
class ClosedBookBackend(ProposalCompletionBackend):
    """One-shot proposal from the library and menus alone."""

    name: str = "closed-book"

    def _propose(
        self, task: RepairTask, meter: BudgetMeter, outcome: RepairOutcome
    ) -> None:
        prompt = render_proposal_prompt(task)
        proposal = complete_proposal(self.completer, prompt, meter)
        outcome.patch = proposal_to_patch(proposal, self.name, library=task.library)
        outcome.status = OutcomeStatus.PATCH_PROPOSED
        outcome.events.append(
            {"step": EpisodeStep.PROPOSE, "rationale": proposal.rationale}
        )


@dataclass
class RagOneShotBackend(ProposalCompletionBackend):
    """Non-agentic RAG: retrieve once and propose once."""

    name: str = "rag-one-shot"

    def _propose(
        self, task: RepairTask, meter: BudgetMeter, outcome: RepairOutcome
    ) -> None:
        hits = retrieve_and_record(task, meter, outcome)
        prompt = render_proposal_prompt(task, render_fragments(hits))
        proposal = complete_proposal(self.completer, prompt, meter)
        outcome.patch = proposal_to_patch(
            proposal, self.name, outcome.retrieved_ids, task.library
        )
        outcome.status = OutcomeStatus.PATCH_PROPOSED


@dataclass
class FixedPipelineBackend(ProposalCompletionBackend):
    """The non-agentic RAG baseline: retrieve once, then propose → static check →
    feed the objection list back → repropose. No memory beyond the last
    objections, no probe selection, no source switching — exactly the
    baseline the agentic claim is measured against."""

    name: str = "fixed-pipeline"

    def _propose(
        self, task: RepairTask, meter: BudgetMeter, outcome: RepairOutcome
    ) -> None:
        hits = retrieve_and_record(task, meter, outcome)
        base_prompt = render_proposal_prompt(task, render_fragments(hits))
        prompt = base_prompt
        while True:
            proposal = complete_proposal(self.completer, prompt, meter)
            patch = proposal_to_patch(
                proposal, self.name, outcome.retrieved_ids, task.library
            )
            objections = task.check_patch(patch) if task.check_patch else []
            if task.check_patch:
                meter.spend_tool_call()
            outcome.events.append(
                {"step": EpisodeStep.CHECK, "objections": list(objections)}
            )
            if not objections:
                outcome.patch = patch
                outcome.status = OutcomeStatus.PATCH_PROPOSED
                return
            prompt = (
                f"{base_prompt}\n\nYour previous proposal was rejected:\n- "
                + "\n- ".join(objections)
                + "\nReply again with a corrected JSON document."
            )


@dataclass
class EnumerationBackend(RepairBackend):
    """Typed enumerative synthesis: no language model at all.

    Candidate operators are enumerated over the *existing* typed
    vocabulary in lexicographic complexity order — one add effect
    achieving a goal predicate, an optional same-signature delete
    effect, and up to ``maximum_preconditions`` typed preconditions over
    the operator's variables — each bound to a capability contract that
    covers the effects. Every candidate is spent against the same meter
    and checked through the same ``check_patch`` tool as the LLM
    backends; this is the baseline that measures whether the language
    model prior buys anything (revised plan §9.4). Predicate invention
    is out of scope for the enumerator by design: it composes, it does
    not invent semantics.
    """

    name: str = "typed-enumeration"
    maximum_preconditions: int = 2

    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        outcome = RepairOutcome(backend=self.name, status=OutcomeStatus.NO_CANDIDATE)
        try:
            for patch in self._candidates(task):
                meter.spend_candidate()
                objections = task.check_patch(patch) if task.check_patch else []
                if task.check_patch:
                    meter.spend_tool_call()
                outcome.events.append(
                    {
                        "step": EpisodeStep.ENUMERATE,
                        "operators": [o.name for o in patch.operators],
                        "objections": len(objections),
                    }
                )
                if not objections:
                    outcome.patch = patch
                    outcome.status = OutcomeStatus.PATCH_PROPOSED
                    break
        except BudgetExhaustedError as error:
            outcome.status = OutcomeStatus.BUDGET_EXHAUSTED
            outcome.events.append(
                {"step": EpisodeStep.BUDGET, "dimension": error.dimension}
            )
        outcome.budget = meter.snapshot()
        return outcome

    def _candidates(self, task: RepairTask):
        """Ordered candidate stream: fewest preconditions first, no delete
        effect before one, contracts and predicates in sorted order for
        determinism."""
        library = task.library
        goal_predicates = [
            library.predicates[literal.predicate]
            for literal in task.certificate.task_goal
            if not literal.negated and literal.predicate in library.predicates
        ]
        for goal in goal_predicates:
            same_signature = [
                p
                for p in sorted(library.predicates.values(), key=lambda p: p.name)
                if p.parameter_types == goal.parameter_types and p.fluent
            ]
            contracts = [
                contract
                for _, contract in sorted(library.capability_contracts.items())
                if goal.ref in contract.verifiable_effects
            ]
            for size in range(self.maximum_preconditions + 1):
                for contract in contracts:
                    object_roles = [
                        role for role in contract.roles if role.accepted_symbol_types
                    ]
                    variables = tuple(
                        (f"v{i}", _local_type_for_role(library, role))
                        for i, role in enumerate(object_roles)
                    )
                    role_bindings = [
                        (role.name, RoleBinding.parameter(variable))
                        for role, (variable, _) in zip(object_roles, variables)
                    ]
                    constant_bindings = _constant_bindings_for_effects(
                        contract,
                        {goal.ref},
                    )
                    if constant_bindings is None:
                        continue
                    role_bindings.extend(constant_bindings)
                    goal_arguments = _first_typed_arguments(
                        variables, goal.parameter_types
                    )
                    if goal_arguments is None:
                        continue
                    goal_literal = Literal(goal.name, goal_arguments)
                    instantiable = [
                        Literal(p.name, goal_arguments)
                        for p in same_signature
                        if p.name != goal.name
                    ]
                    delete_options: list[tuple[Literal, ...]] = [()]
                    delete_options += [
                        (Literal(p.name, goal_arguments),)
                        for p in same_signature
                        if p.name != goal.name
                    ]
                    for preconditions in combinations(instantiable, size):
                        for deletes in delete_options:
                            if any(
                                d.predicate not in contract.verifiable_effect_names
                                for d in deletes
                            ):
                                continue
                            operator = Operator(
                                name=f"synth-{goal.name}",
                                parameters=variables,
                                preconditions=preconditions,
                                add_effects=(goal_literal,),
                                delete_effects=deletes,
                                execution_binding=OperatorExecutionBinding(
                                    contract.ref,
                                    tuple(role_bindings),
                                    proposal_source="enumeration",
                                ),
                                provenance=Provenance(
                                    source="curation",
                                    proposal_backend=self.name,
                                ),
                            )
                            yield ModelPatch(
                                operators=(operator,),
                                rationale=(
                                    f"enumerated achiever for {goal.name} via "
                                    f"capability {contract.uid}"
                                ),
                            )


def _constant_bindings_for_effects(
    contract: CapabilityContract,
    effects: set[PredicateRef | str],
) -> list[tuple[str, RoleBinding]] | None:
    bindings = []
    for role in contract.roles:
        if not role.allowed_values:
            continue
        value = contract.constant_role_value(role.name, effects)
        if value is None:
            if role.required:
                return None
            continue
        bindings.append((role.name, RoleBinding.constant(value)))
    return bindings


def _first_typed_arguments(
    variables: tuple[tuple[str, SymbolType], ...],
    required: tuple[SymbolType, ...],
) -> tuple[str, ...] | None:
    """Choose distinct in-scope variables for an ordered type signature."""
    chosen = []
    used = set()
    for required_type in required:
        match = next(
            (
                name
                for name, actual_type in variables
                if name not in used and is_symbol_subtype(actual_type, required_type)
            ),
            None,
        )
        if match is None:
            return None
        chosen.append(match)
        used.add(match)
    return tuple(chosen)


def _local_type_for_role(library: SymbolLibrary, role: CapabilityRole) -> SymbolType:
    """Choose the most specific local type compatible with a contract role."""
    used_types = {
        symbol_type
        for predicate in library.predicates.values()
        for symbol_type in predicate.parameter_types
    } | {
        symbol_type
        for operator in library.operators.values()
        for _, symbol_type in operator.parameters
    }
    candidates = [
        symbol_type
        for symbol_type in used_types
        if any(
            is_symbol_subtype(symbol_type, accepted)
            for accepted in role.accepted_symbol_types
        )
    ]
    if not candidates:
        return role.accepted_symbol_types[0]

    def depth(symbol_type: SymbolType) -> int:
        return len(resolve_symbol_type(symbol_type).__mro__)

    return sorted(candidates, key=lambda item: (-depth(item), item.python_type_ref))[0]
