"""
The bounded retrieval-augmented planning-model agent.

UniDomain retrieval is one of the agent's tools, not a competing method. When a corpus
is available, the agent retrieves declarative Predicate and Operator candidates before
constructing a local executable patch.  The admitted patch enriches the symbolic model
used by the PDDL planner; the agent does not replace the planner or copy a retrieved
task plan directly.

The agent is a loop, not a chain: every turn the model sees the planning failure
certificate, the library slice, its *entire* episode history (tool trajectory,
observations, counterexamples, its own previous candidates), and the remaining budget,
and answers with exactly one whitelisted tool call. The loop is what makes the policy
falsifiable (revised plan §6.5): tool choice may depend on history, probes are actively
selected, patches are locally revised against observations, and the agent may stop early
or declare the gap unsupported.

Authority boundaries: every tool is read-only against the world and the library.
``check_patch`` is the curator's static review as a consultable tool;
``submit_for_admission`` ends the episode with the candidate — it does not admit
anything. The mandatory suite is not reachable from here.

The model loop delegates every tool call to :mod:`agent_harness`.  That boundary
validates typed arguments, applies the episode profile and budget, and stores the
complete result before producing a bounded model observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from krrood.adapters.json_serializer import to_json
from resym.core.grounding import (
    GroundingFactoryCandidate,
    GroundingFactoryParameter,
    GroundingFactoryRole,
)
from resym.repair.patch import ModelPatch
from resym.core.model import SymbolLibrary, SymbolType
from resym.platform.capabilities import matching_capability_contracts
from resym.core.provenance import KnowledgeSource
from resym.repair.backends import (
    BudgetExhaustedError,
    BudgetMeter,
    AGENTIC_RAG_BACKEND_NAME,
    CapabilityGapKind,
    MissingExecutionCapability,
    MissingGroundingCapability,
    OutcomeStatus,
    RepairBackend,
    RepairOutcome,
    RepairTask,
    RetrievalStatus,
    proposal_to_patch,
    render_fragments,
)
from resym.knowledge.retrieval import RetrievalQuery
from resym.llm.prompting import (
    render_capability_contracts,
    render_operators,
    render_prompt,
    render_symbol_types,
)
from resym.llm.schemas import AgentActionModel
from resym.llm.structured import (
    StructuredCompleter,
    StructuredOutputRetriesExceededError,
)
from resym.llm.transcript import LanguageModelExchange
from resym.repair.agent_harness import (
    AGENTIC_RAG_PROFILE,
    AgentProfile,
    CapabilitySearchArguments,
    CapabilitySearchResult,
    CandidateResult,
    EmbodimentResult,
    EpisodeEventLog,
    FragmentArguments,
    GroundingCatalogResult,
    GroundingFactoryCandidateArguments,
    GroundingFactoryCandidateResult,
    LibraryResult,
    MissingExecutionCapabilityArguments,
    MissingExecutionCapabilityResult,
    MissingGroundingCapabilityArguments,
    MissingGroundingCapabilityResult,
    NoArguments,
    PatchCheckResult,
    PatchComparisonResult,
    ProbeArguments,
    ProbeResult,
    ProposalArguments,
    ProvenanceResult,
    RepairTool,
    RetrievalArguments,
    RetrievalResult,
    SubmissionResult,
    ToolExecutionPipeline,
    ToolRegistry,
    UnsupportedArguments,
    UnsupportedResult,
)

AGENT_NAME = "planning-model-agent"


@dataclass
class EpisodeState:
    """
    Mutable working state; the episode log is the persistent source record.
    """

    candidate: Optional[ModelPatch] = None
    previous_candidate: Optional[ModelPatch] = None
    retrieval_attempted: bool = False
    retrieval_status: RetrievalStatus = RetrievalStatus.NOT_ATTEMPTED
    retrieved_ids: tuple[str, ...] = ()
    unsupported_reason: Optional[str] = None
    missing_execution_capability: Optional[MissingExecutionCapability] = None
    missing_grounding_capability: Optional[MissingGroundingCapability] = None
    grounding_factory_candidate: Optional[GroundingFactoryCandidate] = None
    submitted: bool = False
    candidate_checked: bool = False
    candidate_static_clean: bool = False
    probe_results: dict[str, bool] = field(default_factory=dict)


@dataclass
class RetrievalAugmentedPlanningAgent(RepairBackend):
    """
    Main method: stateful agentic RAG for the symbolic planning model.
    """

    completer: StructuredCompleter
    name: str = AGENTIC_RAG_BACKEND_NAME
    maximum_turns: int = 32
    """
    Backstop only; the effective bound is the shared budget.
    """

    profile: AgentProfile = AGENTIC_RAG_PROFILE
    """
    Tool and retrieval policy frozen into every episode record.
    """

    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        state = EpisodeState()
        outcome = RepairOutcome(backend=self.name, status=OutcomeStatus.NO_CANDIDATE)
        event_log = EpisodeEventLog()
        registry = self._tools(task, meter, state)
        pipeline = ToolExecutionPipeline(registry, self.profile, meter, event_log)
        event_log.append(
            "episode_started",
            event_schema_version=1,
            backend=self.name,
            profile=self.profile.to_json(),
            maximum_turns=self.maximum_turns,
            budget={
                "candidates": meter.budget.candidates,
                "tool_calls": meter.budget.tool_calls,
                "probes": meter.budget.probes,
                "estimated_tokens": meter.budget.estimated_tokens,
            },
        )
        termination_reason = "maximum_turns_reached"
        try:
            for turn_index in range(self.maximum_turns):
                turn = turn_index + 1
                prompt = self._prompt(task, meter, event_log, registry, state)

                def record_exchange(exchange: LanguageModelExchange) -> None:
                    event_log.append(
                        "model_request",
                        turn=turn,
                        attempt=exchange.attempt,
                        agent=exchange.agent_name,
                        model=exchange.model_description,
                        prompt=exchange.prompt,
                        budget=meter.snapshot(),
                    )
                    event_log.append(
                        "model_response",
                        turn=turn,
                        attempt=exchange.attempt,
                        response=exchange.response,
                        parse_error=exchange.parse_error,
                        parse_recovery=exchange.parse_recovery,
                        input_tokens=exchange.input_tokens,
                        output_tokens=exchange.output_tokens,
                        usage_source=exchange.usage_source,
                    )

                try:
                    action = self.completer.complete(
                        AGENT_NAME,
                        prompt,
                        AgentActionModel,
                        charge_usage=meter.spend_usage,
                        exchange_sink=record_exchange,
                    )
                except StructuredOutputRetriesExceededError as error:
                    event_log.append(
                        "model_action",
                        turn=turn,
                        succeeded=False,
                        error=f"{type(error).__name__}: {error}",
                        budget=meter.snapshot(),
                    )
                    termination_reason = "invalid_model_action"
                    break
                event_log.append(
                    "model_action",
                    turn=turn,
                    succeeded=True,
                    action=action.model_dump(mode="json"),
                    budget=meter.snapshot(),
                )
                previous_candidate = state.candidate
                pipeline.execute(action.tool, action.arguments, turn)
                if (
                    state.candidate is not None
                    and state.candidate is not previous_candidate
                ):
                    event_log.append(
                        "candidate_proposed",
                        turn=turn,
                        patch=to_json(state.candidate),
                        budget=meter.snapshot(),
                    )
                if state.submitted:
                    outcome.status = OutcomeStatus.PATCH_PROPOSED
                    outcome.patch = state.candidate
                    termination_reason = "submitted_for_admission"
                    break
                if state.missing_execution_capability is not None:
                    outcome.status = OutcomeStatus.MISSING_EXECUTION_CAPABILITY
                    outcome.missing_execution_capability = (
                        state.missing_execution_capability
                    )
                    termination_reason = "missing_execution_capability_reported"
                    break
                if state.grounding_factory_candidate is not None:
                    outcome.status = OutcomeStatus.GROUNDING_FACTORY_PROPOSED
                    outcome.grounding_factory_candidate = (
                        state.grounding_factory_candidate
                    )
                    termination_reason = "grounding_factory_candidate_proposed"
                    break
                if state.missing_grounding_capability is not None:
                    outcome.status = OutcomeStatus.MISSING_GROUNDING_CAPABILITY
                    outcome.missing_grounding_capability = (
                        state.missing_grounding_capability
                    )
                    termination_reason = "missing_grounding_capability_reported"
                    break
                if state.unsupported_reason is not None:
                    outcome.status = OutcomeStatus.UNSUPPORTED_DECLARED
                    termination_reason = "unsupported_capability"
                    break
        except BudgetExhaustedError as error:
            outcome.status = OutcomeStatus.BUDGET_EXHAUSTED
            termination_reason = f"budget_exhausted:{error.dimension}"
        outcome.retrieved_ids = state.retrieved_ids
        outcome.budget = meter.snapshot()
        event_log.append(
            "episode_finished",
            status=outcome.status.value,
            reason=state.unsupported_reason or termination_reason,
            retrieval_status=state.retrieval_status,
            retrieved_ids=list(state.retrieved_ids),
            budget=outcome.budget,
        )
        outcome.events = event_log.to_json()
        return outcome

    # -- the whitelist --------------------------------------------------

    def _tools(
        self, task: RepairTask, meter: BudgetMeter, state: EpisodeState
    ) -> ToolRegistry:
        def inspect_library(arguments: NoArguments) -> LibraryResult:
            predicates = _predicates_of(task)
            operators = _operators_of(task)
            return LibraryResult(
                message=f"predicates:\n{predicates}\noperators:\n{operators}",
                predicates=predicates,
                operators=operators,
            )

        def inspect_embodiment_profile(
            arguments: NoArguments,
        ) -> EmbodimentResult:
            return EmbodimentResult(
                message=(
                    f"capabilities:\n{task.capability_listing}\n"
                    f"evaluators:\n{task.evaluator_listing}\n"
                    "platform contract drafts:\n"
                    f"{task.capability_draft_listing or '- none'}"
                ),
                capabilities=task.capability_listing,
                evaluators=task.evaluator_listing,
            )

        def inspect_grounding_catalog(
            arguments: NoArguments,
        ) -> GroundingCatalogResult:
            primitives = (
                task.grounding_factory_listing
                or task.evaluator_listing
                or "- none registered"
            )
            vocabulary = task.grounding_vocabulary_listing or "- none discovered"
            return GroundingCatalogResult(
                message=(
                    f"approved grounding factories:\n{primitives}\n"
                    f"reviewed EQL vocabulary:\n{vocabulary}"
                ),
                primitives=f"{primitives}\n{vocabulary}",
            )

        def search_capability_catalog(
            arguments: CapabilitySearchArguments,
        ) -> CapabilitySearchResult:
            matches = matching_capability_contracts(
                arguments.desired_effects,
                arguments.required_roles,
                contracts=task.capability_catalog,
            )
            records = [to_json(contract) for contract in matches]
            return CapabilitySearchResult(
                message=(
                    "matching reviewed contracts:\n"
                    + render_capability_contracts(
                        SymbolLibrary(
                            capability_contracts={
                                contract.uid: contract for contract in matches
                            }
                        )
                    )
                    if matches
                    else "no reviewed capability contract matches these effects and roles"
                ),
                matches=records,
            )

        def retrieve_domain_fragments(
            arguments: RetrievalArguments,
        ) -> RetrievalResult:
            if task.index is None:
                state.retrieval_status = RetrievalStatus.CORPUS_UNAVAILABLE
                return RetrievalResult(
                    message="no corpus is available in this setting (closed book)",
                    status=state.retrieval_status,
                    retrieved_ids=[],
                    hits=[],
                )
            state.retrieval_attempted = True
            base = RetrievalQuery.from_certificate(task.certificate)
            query_text = arguments.query or base.text
            query = RetrievalQuery(
                text=query_text,
                goal_predicates=base.goal_predicates,
                causal_predicates=base.causal_predicates,
                argument_arities=base.argument_arities,
            )
            top_k = arguments.top_k or task.retrieval_top_k
            hits = task.index.retrieve(query, top_k=top_k)
            state.retrieval_status = (
                RetrievalStatus.RELEVANT_EVIDENCE_FOUND
                if hits
                else RetrievalStatus.NO_RELEVANT_EVIDENCE
            )
            state.retrieved_ids = tuple(
                dict.fromkeys(state.retrieved_ids + tuple(h.fragment_id for h in hits))
            )
            return RetrievalResult(
                message=render_fragments(hits) or "no fragments matched",
                status=state.retrieval_status,
                retrieved_ids=[hit.fragment_id for hit in hits],
                hits=[
                    {
                        "fragment": to_json(hit.fragment),
                        "score": hit.score,
                        "text_score": hit.text_score,
                        "causal_score": hit.causal_score,
                        "type_score": hit.type_score,
                        "contract_score": hit.contract_score,
                        "dense_score": hit.dense_score,
                        "evidence_score": hit.evidence_score,
                    }
                    for hit in hits
                ],
            )

        def inspect_source_provenance(
            arguments: FragmentArguments,
        ) -> ProvenanceResult:
            fragment_id = arguments.fragment_id
            if task.index is None:
                return ProvenanceResult(
                    message="no corpus is available in this setting", found=False
                )
            for fragment in task.index.fragments:
                if fragment.fragment_id == fragment_id:
                    return ProvenanceResult(
                        message=str(to_json(fragment)),
                        found=True,
                        fragment=to_json(fragment),
                    )
            return ProvenanceResult(
                message=f"unknown fragment id '{fragment_id}'", found=False
            )

        def propose_patch(arguments: ProposalArguments) -> CandidateResult:
            if (
                self.profile.retrieval_required_when_available
                and task.index is not None
                and not state.retrieval_attempted
            ):
                return CandidateResult(
                    message=(
                        "retrieve_domain_fragments must be called before proposing: "
                        "UniDomain is the external knowledge source for this setting"
                    ),
                    registered=False,
                )
            required_step = _required_candidate_step(task, state)
            if required_step is not None:
                return CandidateResult(
                    message=(
                        f"{required_step} is the required next step for the "
                        "current candidate; do not propose another candidate yet"
                    ),
                    registered=False,
                )
            proposal = arguments.proposal
            try:
                candidate = proposal_to_patch(
                    proposal,
                    self.name,
                    state.retrieved_ids,
                    task.library,
                    knowledge_source=(
                        KnowledgeSource.MIXED
                        if state.retrieved_ids
                        else KnowledgeSource.LLM_PRIOR
                    ),
                )
            except ValueError as error:
                meter.spend_candidate()
                return CandidateResult(
                    message=f"candidate rejected: {error}", registered=False
                )
            if state.candidate is not None and _candidate_signature(
                candidate
            ) == _candidate_signature(state.candidate):
                return CandidateResult(
                    message=(
                        "duplicate candidate ignored; the current candidate is "
                        "unchanged and candidate budget was not spent. Validate "
                        "it with check_patch and every required probe, then submit."
                    ),
                    registered=False,
                )
            meter.spend_candidate()
            state.previous_candidate = state.candidate
            state.candidate = candidate
            state.candidate_checked = False
            state.candidate_static_clean = False
            state.probe_results.clear()
            complexity = state.candidate.complexity(task.library)
            predicates = [p.name for p in state.candidate.predicates]
            operators = [o.name for o in state.candidate.operators]
            return CandidateResult(
                message=(
                    "candidate registered: "
                    f"{predicates} predicates, {operators} operators, "
                    f"complexity {complexity.as_tuple()}. Validate it with "
                    "check_patch and execute_diagnostic_probe, then "
                    "submit_for_admission; re-proposing the same content is "
                    "ignored."
                ),
                registered=True,
                predicates=predicates,
                operators=operators,
                complexity=complexity.as_tuple(),
            )

        def check_patch(arguments: NoArguments) -> PatchCheckResult:
            if state.candidate is None:
                return PatchCheckResult(
                    message="no candidate to check; propose_patch first",
                    available=False,
                )
            if task.check_patch is None:
                return PatchCheckResult(
                    message="static review unavailable in this setting",
                    available=False,
                )
            objections = list(task.check_patch(state.candidate))
            objections.extend(_scope_objections(task, state.candidate))
            state.candidate_checked = True
            state.candidate_static_clean = not objections
            return PatchCheckResult(
                message=(
                    "static review clean"
                    if not objections
                    else "objections:\n- " + "\n- ".join(objections)
                ),
                available=True,
                objections=list(objections),
            )

        def execute_diagnostic_probe(arguments: ProbeArguments) -> ProbeResult:
            if state.candidate is None:
                return ProbeResult(
                    message="no candidate to probe; propose_patch first",
                    executed=False,
                    probe=arguments.probe,
                )
            if task.run_probe is None:
                return ProbeResult(
                    message="no probes available in this setting",
                    executed=False,
                    probe=arguments.probe,
                )
            meter.spend_probe()
            result = task.run_probe(arguments.probe, state.candidate)
            state.probe_results[arguments.probe] = bool(result.get("succeeded"))
            return ProbeResult(
                message=str(result),
                executed=True,
                probe=arguments.probe,
                observation=result,
            )

        def compare_patch_versions(
            arguments: NoArguments,
        ) -> PatchComparisonResult:
            if state.candidate is None or state.previous_candidate is None:
                return PatchComparisonResult(
                    message="fewer than two candidates so far", comparable=False
                )
            current = state.candidate.complexity(task.library).as_tuple()
            previous = state.previous_candidate.complexity(task.library).as_tuple()
            previous_operators = [o.name for o in state.previous_candidate.operators]
            current_operators = [o.name for o in state.candidate.operators]
            return PatchComparisonResult(
                message=(
                    f"previous complexity {previous} -> current {current}; "
                    f"previous operators {previous_operators} -> "
                    f"{current_operators}"
                ),
                comparable=True,
                previous_complexity=previous,
                current_complexity=current,
                previous_operators=previous_operators,
                current_operators=current_operators,
            )

        def submit_for_admission(arguments: NoArguments) -> SubmissionResult:
            if state.candidate is None:
                return SubmissionResult(
                    message="nothing to submit; propose_patch first", submitted=False
                )
            if task.check_patch is not None and not state.candidate_checked:
                return SubmissionResult(
                    message="run check_patch on the current candidate before submission",
                    submitted=False,
                )
            if task.check_patch is not None and not state.candidate_static_clean:
                return SubmissionResult(
                    message="the current candidate has unresolved static objections",
                    submitted=False,
                )
            missing_probes = [
                probe
                for probe in task.required_probes
                if probe not in state.probe_results
            ]
            failed_probes = [
                probe
                for probe in task.required_probes
                if state.probe_results.get(probe) is False
            ]
            if missing_probes:
                return SubmissionResult(
                    message="run required probes before submission: "
                    + ", ".join(missing_probes),
                    submitted=False,
                )
            if failed_probes:
                return SubmissionResult(
                    message="the current candidate failed required probes: "
                    + ", ".join(failed_probes),
                    submitted=False,
                )
            state.submitted = True
            return SubmissionResult(
                message="submitted; the independent curator takes over",
                submitted=True,
            )

        def declare_unsupported(
            arguments: UnsupportedArguments,
        ) -> UnsupportedResult:
            state.unsupported_reason = arguments.reason
            return UnsupportedResult(
                message="recorded as UNSUPPORTED_CAPABILITY; episode ends",
                reason=arguments.reason,
            )

        def report_missing_execution_capability(
            arguments: MissingExecutionCapabilityArguments,
        ) -> MissingExecutionCapabilityResult:
            unknown = sorted(
                set(arguments.candidate_realizations) - task.capability_draft_ids
            )
            if unknown:
                return MissingExecutionCapabilityResult(
                    message=(
                        "unknown Coraplex action candidates: " + ", ".join(unknown)
                    ),
                    recorded=False,
                    unknown_candidate_realizations=unknown,
                )
            matching_contracts = matching_capability_contracts(
                arguments.desired_effects,
                arguments.required_roles,
                contracts=task.capability_catalog,
            )
            gap = MissingExecutionCapability(
                suggested_label=arguments.suggested_label,
                desired_effects=tuple(arguments.desired_effects),
                required_roles=tuple(sorted(arguments.required_roles.items())),
                candidate_realizations=tuple(arguments.candidate_realizations),
                reason=arguments.reason,
                gap_kind=(
                    CapabilityGapKind.MISSING_REALIZATION
                    if matching_contracts
                    else CapabilityGapKind.MISSING_CONTRACT
                ),
                matching_contract_uids=tuple(
                    contract.uid for contract in matching_contracts
                ),
            )
            state.missing_execution_capability = gap
            return MissingExecutionCapabilityResult(
                message=(
                    "recorded MissingExecutionCapability for review; no trusted "
                    "contract or Coraplex implementation was written"
                ),
                recorded=True,
                gap=to_json(gap),
            )

        def propose_grounding_factory_candidate(
            arguments: GroundingFactoryCandidateArguments,
        ) -> GroundingFactoryCandidateResult:
            candidate = GroundingFactoryCandidate(
                candidate_id=arguments.candidate_id,
                proposed_uid=arguments.proposed_uid,
                semantic_name=arguments.semantic_name,
                source_code=arguments.source_code,
                roles=tuple(
                    GroundingFactoryRole(role.name, SymbolType(role.symbol_type))
                    for role in arguments.roles
                ),
                generated_by=self.name,
                rationale=arguments.rationale,
                evidence=tuple(arguments.evidence),
                parameters=tuple(
                    GroundingFactoryParameter(
                        name=parameter.name,
                        value_type=parameter.value_type,
                        required=parameter.required,
                        minimum=parameter.minimum,
                        maximum=parameter.maximum,
                    )
                    for parameter in arguments.parameters
                ),
            )
            meter.spend_candidate()
            if task.validate_grounding_candidate is None:
                return GroundingFactoryCandidateResult(
                    message="grounding-factory review is unavailable in this setting",
                    registered=False,
                    candidate=to_json(candidate),
                    objections=["no grounding source validator is configured"],
                )
            objections = list(task.validate_grounding_candidate(candidate))
            if objections:
                return GroundingFactoryCandidateResult(
                    message="candidate source rejected:\n- " + "\n- ".join(objections),
                    registered=False,
                    candidate=to_json(candidate),
                    objections=objections,
                )
            if task.grounding_candidate_sink is None:
                return GroundingFactoryCandidateResult(
                    message="candidate is valid, but no pending-review store is configured",
                    registered=False,
                    candidate=to_json(candidate),
                    objections=["no grounding candidate store is configured"],
                )
            task.grounding_candidate_sink(candidate)
            state.grounding_factory_candidate = candidate
            return GroundingFactoryCandidateResult(
                message=(
                    "candidate stored as pending-review; a human must approve it "
                    "before it can be materialized or executed"
                ),
                registered=True,
                candidate=to_json(candidate),
            )

        def report_missing_grounding_capability(
            arguments: MissingGroundingCapabilityArguments,
        ) -> MissingGroundingCapabilityResult:
            gap = MissingGroundingCapability(
                required_relation=arguments.required_relation,
                input_types=tuple(sorted(arguments.input_types.items())),
                missing_computation=arguments.missing_computation,
                reason=arguments.reason,
            )
            state.missing_grounding_capability = gap
            return MissingGroundingCapabilityResult(
                message=(
                    "recorded MissingGroundingCapability; platform work is "
                    "required before this predicate can be grounded"
                ),
                recorded=True,
                gap=to_json(gap),
            )

        return ToolRegistry(
            [
                RepairTool(
                    "inspect_library",
                    "current predicates and operators",
                    NoArguments,
                    LibraryResult,
                    inspect_library,
                ),
                RepairTool(
                    "inspect_embodiment_profile",
                    "capabilities, contracts, and evaluators of this platform",
                    NoArguments,
                    EmbodimentResult,
                    inspect_embodiment_profile,
                ),
                RepairTool(
                    "inspect_grounding_catalog",
                    "typed CRAM query primitives for constructing a new predicate",
                    NoArguments,
                    GroundingCatalogResult,
                    inspect_grounding_catalog,
                ),
                RepairTool(
                    "search_capability_catalog",
                    "find reviewed capability contracts by desired effects and typed roles",
                    CapabilitySearchArguments,
                    CapabilitySearchResult,
                    search_capability_catalog,
                ),
                RepairTool(
                    "retrieve_domain_fragments",
                    "search the frozen corpus",
                    RetrievalArguments,
                    RetrievalResult,
                    retrieve_domain_fragments,
                ),
                RepairTool(
                    "inspect_source_provenance",
                    "full record of one retrieved fragment",
                    FragmentArguments,
                    ProvenanceResult,
                    inspect_source_provenance,
                ),
                RepairTool(
                    "propose_patch",
                    "register one candidate patch in the library JSON shape",
                    ProposalArguments,
                    CandidateResult,
                    propose_patch,
                ),
                RepairTool(
                    "check_patch",
                    "run the curator's read-only static review",
                    NoArguments,
                    PatchCheckResult,
                    check_patch,
                ),
                RepairTool(
                    "execute_diagnostic_probe",
                    f"run one diagnostic probe. Available: "
                    f"{task.probe_listing or '(none)'}",
                    ProbeArguments,
                    ProbeResult,
                    execute_diagnostic_probe,
                ),
                RepairTool(
                    "compare_patch_versions",
                    "compare the current and previous candidates",
                    NoArguments,
                    PatchComparisonResult,
                    compare_patch_versions,
                ),
                RepairTool(
                    "submit_for_admission",
                    "end the episode and hand the candidate to the curator",
                    NoArguments,
                    SubmissionResult,
                    submit_for_admission,
                ),
                RepairTool(
                    "report_missing_execution_capability",
                    (
                        "end with a structured gap report when no published "
                        "capability contract can realize the required effects"
                    ),
                    MissingExecutionCapabilityArguments,
                    MissingExecutionCapabilityResult,
                    report_missing_execution_capability,
                ),
                RepairTool(
                    "propose_grounding_factory_candidate",
                    (
                        "draft a bounded native-EQL evaluator for the human review "
                        "queue when no reviewed grounding factory is sufficient"
                    ),
                    GroundingFactoryCandidateArguments,
                    GroundingFactoryCandidateResult,
                    propose_grounding_factory_candidate,
                ),
                RepairTool(
                    "report_missing_grounding_capability",
                    (
                        "end with a structured grounding gap when the reviewed "
                        "EQL vocabulary cannot compute the required relation"
                    ),
                    MissingGroundingCapabilityArguments,
                    MissingGroundingCapabilityResult,
                    report_missing_grounding_capability,
                ),
                RepairTool(
                    "declare_unsupported",
                    "end because the platform lacks a required capability",
                    UnsupportedArguments,
                    UnsupportedResult,
                    declare_unsupported,
                ),
            ]
        )

    # -- loop internals -------------------------------------------------

    def _prompt(
        self,
        task: RepairTask,
        meter: BudgetMeter,
        event_log: EpisodeEventLog,
        tools: ToolRegistry,
        state: EpisodeState,
    ) -> str:
        tool_lines = tools.render(self.profile.allowed_tools)
        history = event_log.render_tool_history()
        budget = meter.budget
        remaining = (
            f"candidates {budget.candidates - meter.candidates_used}, "
            f"tool calls {budget.tool_calls - meter.tool_calls_used}, "
            f"probes {budget.probes - meter.probes_used}, "
            f"~tokens {budget.estimated_tokens - meter.estimated_tokens_used}"
        )
        return render_prompt(
            "planning_model_agent",
            certificate=task.certificate.render(),
            predicates=_predicates_of(task),
            operators=_operators_of(task),
            contracts=render_capability_contracts(
                SymbolLibrary(
                    capability_contracts={
                        contract.uid: contract for contract in task.capability_catalog
                    }
                )
            ),
            evaluators=task.evaluator_listing,
            predicate_queries=(
                task.grounding_factory_listing
                or task.evaluator_listing
                or "- none registered"
            ),
            grounding_vocabulary=(
                task.grounding_vocabulary_listing or "- none discovered"
            ),
            capability_candidates=task.capability_draft_listing or "- none",
            skills=task.capability_listing,
            types=render_symbol_types(task.library.symbol_types),
            tools=tool_lines,
            retrieval_rule=self._retrieval_rule(task),
            required_probes=(", ".join(task.required_probes) or "none"),
            budget=remaining,
            history=history or "(no steps yet)",
            next_step=_next_step(
                task,
                state,
                retrieval_required=self.profile.retrieval_required_when_available,
            ),
        )

    def _retrieval_rule(self, task: RepairTask) -> str:
        if task.index is None:
            return "No external corpus is available; retrieval is not required."
        if self.profile.retrieval_required_when_available:
            return (
                "The corpus is available. Call retrieve_domain_fragments before "
                "the first propose_patch; refine the query or inspect provenance "
                "when useful."
            )
        return "The corpus is available, but retrieval is optional in this profile."


def _candidate_signature(patch: ModelPatch) -> tuple:
    """
    Semantic candidate identity, excluding rationale and provenance.
    """
    predicates = tuple(
        sorted(
            (
                predicate.name,
                predicate.parameter_types,
                predicate.evaluator,
                predicate.grounding_plan,
                predicate.fluent,
            )
            for predicate in patch.predicates
        )
    )
    operators = tuple(
        sorted(
            (
                operator.name,
                operator.parameters,
                tuple(sorted(map(_literal_signature, operator.preconditions))),
                tuple(sorted(map(_literal_signature, operator.add_effects))),
                tuple(sorted(map(_literal_signature, operator.delete_effects))),
                (
                    operator.execution_binding.capability_ref.uid,
                    operator.execution_binding.capability_ref.version,
                    tuple(
                        sorted(
                            (
                                role,
                                binding.source.value,
                                binding.value,
                            )
                            for role, binding in (
                                operator.execution_binding.role_bindings
                            )
                        )
                    ),
                ),
            )
            for operator in patch.operators
        )
    )
    contracts = tuple(sorted(patch.capability_contracts, key=lambda item: item.uid))
    return predicates, operators, contracts, tuple(sorted(patch.unresolved))


def _required_candidate_step(task: RepairTask, state: EpisodeState) -> Optional[str]:
    """
    Return the validation action that must precede another revision.
    """
    if state.candidate is None:
        return None
    if task.check_patch is not None and not state.candidate_checked:
        return "check_patch"
    if task.check_patch is not None and not state.candidate_static_clean:
        return None
    if any(state.probe_results.get(probe) is False for probe in task.required_probes):
        return None
    missing_probes = [
        probe for probe in task.required_probes if probe not in state.probe_results
    ]
    if missing_probes:
        return "execute_diagnostic_probe"
    return "submit_for_admission"


def _next_step(
    task: RepairTask,
    state: EpisodeState,
    *,
    retrieval_required: bool,
) -> str:
    required = _required_candidate_step(task, state)
    if required is not None:
        if required == "execute_diagnostic_probe":
            missing = [
                probe
                for probe in task.required_probes
                if probe not in state.probe_results
            ]
            return f"{required} for: {', '.join(missing)}"
        return required
    if state.candidate is not None:
        return "revise the current candidate in response to failed checks or probes"
    if task.index is not None and not state.retrieval_attempted and retrieval_required:
        return "retrieve_domain_fragments"
    return "propose_patch, or declare_unsupported only if an allowed hypothesis requires it"


def _literal_signature(literal) -> tuple:
    return literal.predicate, literal.arguments, literal.negated


def _scope_objections(task: RepairTask, patch: ModelPatch) -> list[str]:
    """
    Reject edits to existing operators outside the certificate slice.
    """
    relevant = set(task.certificate.causal_neighborhood.operators)
    if not relevant:
        return []
    unrelated = sorted(
        operator.name
        for operator in patch.operators
        if operator.name in task.library.operators and operator.name not in relevant
    )
    if not unrelated:
        return []
    return [
        "existing operator edits fall outside the failure's causal neighborhood: "
        + ", ".join(unrelated)
    ]


def _predicates_of(task: RepairTask) -> str:
    return "\n".join(
        f"- {p.name}({', '.join(t.python_type_ref for t in p.parameter_types)})"
        f"{' [fluent]' if p.fluent else ''}"
        for p in task.library.predicates.values()
    )


def _operators_of(task: RepairTask) -> str:
    return render_operators(task.library)
