"""
Typed tool boundary and append-only episode log for the repair agent.

The model chooses tools; this module owns everything between that choice and the Python
handler.  It validates inputs, enforces the active profile, charges the shared
experiment budget, validates outputs, and records a lossless result. Only a bounded
rendering of that result is returned to the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Callable, Optional

from resym.llm.schemas import LibraryProposal
from resym.core.model import GroundingFactoryParameterType, SymbolType
from resym.repair.backends import BudgetExhaustedError, BudgetMeter

MODEL_OBSERVATION_LIMIT = 1200
"""
Maximum characters of one tool result placed back in model context.
"""


class ToolArguments(BaseModel):
    """
    Strict base class for model-supplied tool arguments.
    """

    model_config = ConfigDict(extra="forbid")


class NoArguments(ToolArguments):
    pass


class RetrievalArguments(ToolArguments):
    query: Optional[str] = None
    top_k: Optional[int] = Field(default=None, gt=0)


class FragmentArguments(ToolArguments):
    fragment_id: str = Field(min_length=1)


class ProposalArguments(ToolArguments):
    proposal: LibraryProposal

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_flattened_proposal(cls, value: Any) -> Any:
        """
        Keep compatibility with models that omit the ``proposal`` wrapper.
        """
        if isinstance(value, dict) and "proposal" not in value:
            return {"proposal": value}
        return value


class ProbeArguments(ToolArguments):
    probe: str = Field(min_length=1)


class UnsupportedArguments(ToolArguments):
    reason: str = Field(min_length=1)


class MissingExecutionCapabilityArguments(ToolArguments):
    suggested_label: str = Field(min_length=1)
    desired_effects: list[str] = Field(min_length=1)
    required_roles: dict[str, str] = Field(default_factory=dict)
    candidate_realizations: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def role_types_are_valid(self) -> MissingExecutionCapabilityArguments:
        for type_reference in self.required_roles.values():
            SymbolType(type_reference)
        return self


class MissingGroundingCapabilityArguments(ToolArguments):
    """
    Structured report for a world-query ability absent from the platform.
    """

    required_relation: str = Field(min_length=1)
    input_types: dict[str, str] = Field(default_factory=dict)
    missing_computation: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def input_types_are_valid(self) -> MissingGroundingCapabilityArguments:
        for type_reference in self.input_types.values():
            SymbolType(type_reference)
        return self


class GroundingFactoryParameterArguments(ToolArguments):
    """
    Reviewed shape proposed for one factory configuration value.
    """

    name: str = Field(min_length=1)
    value_type: GroundingFactoryParameterType
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def numeric_range_is_ordered(self) -> GroundingFactoryParameterArguments:
        if self.value_type not in {
            GroundingFactoryParameterType.INTEGER,
            GroundingFactoryParameterType.NUMBER,
        } and (self.minimum is not None or self.maximum is not None):
            raise ValueError("only numeric parameters may declare bounds")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum cannot exceed maximum")
        return self


class GroundingFactoryRoleArguments(ToolArguments):
    """
    One semantic input of a drafted factory, in declaration order.
    """

    name: str = Field(min_length=1)
    symbol_type: str = Field(min_length=1)

    @model_validator(mode="after")
    def symbol_type_is_valid(self) -> GroundingFactoryRoleArguments:
        SymbolType(self.symbol_type)
        return self


class GroundingFactoryCandidateArguments(ToolArguments):
    """
    Native EQL source proposal for the human grounding-review queue.

    Role order is positional semantics: it must match how the drafted
    ``evaluate`` body reads its ``arguments`` tuple.
    """

    candidate_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    proposed_uid: str = Field(min_length=1)
    semantic_name: str = Field(min_length=1)
    source_code: str = Field(min_length=1, max_length=20_000)
    roles: list[GroundingFactoryRoleArguments] = Field(default_factory=list)
    parameters: list[GroundingFactoryParameterArguments] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def names_are_unique(self) -> GroundingFactoryCandidateArguments:
        role_names = [role.name for role in self.roles]
        if len(role_names) != len(set(role_names)):
            raise ValueError("grounding role names must be unique")
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("grounding parameter names must be unique")
        return self


class CapabilitySearchArguments(ToolArguments):
    desired_effects: list[str] = Field(min_length=1)
    required_roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def role_types_are_valid(self) -> CapabilitySearchArguments:
        for type_reference in self.required_roles.values():
            SymbolType(type_reference)
        return self


class ToolResult(BaseModel):
    """
    Base for canonical tool results.

    ``message`` is rendered to the model.
    """

    model_config = ConfigDict(extra="forbid")
    message: str


class LibraryResult(ToolResult):
    predicates: str
    operators: str


class EmbodimentResult(ToolResult):
    capabilities: str
    grounding_factories: str


class GroundingCatalogResult(ToolResult):
    primitives: str


class CapabilitySearchResult(ToolResult):
    matches: list[dict[str, Any]]


class RetrievalResult(ToolResult):
    status: str
    retrieved_ids: list[str]
    hits: list[dict[str, Any]]


class ProvenanceResult(ToolResult):
    found: bool
    fragment: Optional[dict[str, Any]] = None


class CandidateResult(ToolResult):
    registered: bool
    predicates: list[str] = Field(default_factory=list)
    operators: list[str] = Field(default_factory=list)
    complexity: Optional[tuple[int, ...]] = None


class PatchCheckResult(ToolResult):
    available: bool
    objections: list[str] = Field(default_factory=list)


class ProbeResult(ToolResult):
    executed: bool
    probe: str
    observation: Optional[dict[str, Any]] = None


class PatchComparisonResult(ToolResult):
    comparable: bool
    previous_complexity: Optional[tuple[int, ...]] = None
    current_complexity: Optional[tuple[int, ...]] = None
    previous_operators: list[str] = Field(default_factory=list)
    current_operators: list[str] = Field(default_factory=list)


class SubmissionResult(ToolResult):
    submitted: bool


class UnsupportedResult(ToolResult):
    reason: str


class MissingExecutionCapabilityResult(ToolResult):
    recorded: bool
    gap: Optional[dict[str, Any]] = None
    unknown_candidate_realizations: list[str] = Field(default_factory=list)


class MissingGroundingCapabilityResult(ToolResult):
    """
    Persistence result for one structured grounding-gap report.
    """

    recorded: bool
    gap: dict[str, Any]


class GroundingFactoryCandidateResult(ToolResult):
    """
    Static verdict and persistence result for one EQL source proposal.
    """

    registered: bool
    candidate: Optional[dict[str, Any]] = None
    objections: list[str] = Field(default_factory=list)


ToolHandler = Callable[[ToolArguments], ToolResult]


@dataclass(frozen=True)
class RepairTool:
    """
    One registered tool with explicit input and output contracts.
    """

    name: str
    description: str
    arguments_type: type[ToolArguments]
    result_type: type[ToolResult]
    handler: ToolHandler

    def prompt_description(self) -> str:
        schema = self.arguments_type.model_json_schema()
        compact_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        return f"- {self.name}: {self.description}; arguments schema: {compact_schema}"


class ToolRegistry:
    """
    Immutable-by-convention registry constructed once per episode.
    """

    def __init__(self, tools: list[RepairTool]):
        self._tools: dict[str, RepairTool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate repair tool '{tool.name}'.")
            self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[RepairTool]:
        return self._tools.get(name)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def render(self, allowed_tools: frozenset[str]) -> str:
        return "\n".join(
            self._tools[name].prompt_description()
            for name in sorted(self._tools)
            if name in allowed_tools
        )


AGENT_TOOL_NAMES = frozenset(
    {
        "inspect_library",
        "inspect_embodiment_profile",
        "inspect_grounding_catalog",
        "search_capability_catalog",
        "retrieve_domain_fragments",
        "inspect_source_provenance",
        "propose_patch",
        "check_patch",
        "execute_diagnostic_probe",
        "compare_patch_versions",
        "submit_for_admission",
        "report_missing_execution_capability",
        "report_missing_grounding_capability",
        "propose_grounding_factory_candidate",
        "declare_unsupported",
    }
)


@dataclass(frozen=True)
class AgentProfile:
    """
    Declarative policy fixed at episode start, independent of prompts.
    """

    profile_id: str
    allowed_tools: frozenset[str]
    retrieval_required_when_available: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "allowed_tools": sorted(self.allowed_tools),
            "retrieval_required_when_available": (
                self.retrieval_required_when_available
            ),
        }


AGENTIC_RAG_PROFILE = AgentProfile(
    profile_id="agentic-rag-v6",
    allowed_tools=AGENT_TOOL_NAMES,
    retrieval_required_when_available=True,
)


@dataclass
class EpisodeEventLog:
    """
    Ordered source record.

    Events are appended and never edited in place.
    """

    _events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, event: str, **data: Any) -> dict[str, Any]:
        record = {"sequence": len(self._events) + 1, "event": event, **data}
        # Fail close at the logging boundary: an experiment event must be
        # persistable before it becomes part of the source record.
        json.dumps(record, ensure_ascii=False)
        self._events.append(record)
        return record

    def to_json(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def render_tool_history(self) -> str:
        completed = [e for e in self._events if e["event"] == "tool_result"]
        return "\n".join(
            f"{index}. {event['tool']}"
            f"({json.dumps(event['arguments'], ensure_ascii=False)[:200]}) -> "
            f"{event['observation']}"
            for index, event in enumerate(completed, start=1)
        )


@dataclass(frozen=True)
class ToolExecution:
    """
    Result returned to the loop after the full tool pipeline.
    """

    observation: str
    succeeded: bool
    canonical_result: dict[str, Any]


class ToolExecutionPipeline:
    """
    Policy/validation -> handler -> validated logging/model rendering.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        profile: AgentProfile,
        meter: BudgetMeter,
        event_log: EpisodeEventLog,
    ):
        unknown = profile.allowed_tools - registry.names
        if unknown:
            raise ValueError(f"Profile references unknown tools: {sorted(unknown)}")
        self.registry = registry
        self.profile = profile
        self.meter = meter
        self.event_log = event_log

    def execute(self, tool_name: str, arguments: dict, turn: int) -> ToolExecution:
        self.event_log.append(
            "tool_call", turn=turn, tool=tool_name, arguments=arguments
        )
        try:
            # Invalid decisions still consume a tool call: this keeps budgets
            # comparable across methods and preserves the old E1 semantics.
            self.meter.spend_tool_call()
        except BudgetExhaustedError as error:
            self._record_error(
                tool_name, arguments, turn, "budget_exhausted", str(error)
            )
            raise

        tool = self.registry.get(tool_name)
        if tool is None:
            allowed = ", ".join(sorted(self.profile.allowed_tools))
            return self._record_error(
                tool_name,
                arguments,
                turn,
                "unknown_tool",
                f"unknown tool '{tool_name}'; whitelisted: {allowed}",
            )
        if tool_name not in self.profile.allowed_tools:
            return self._record_error(
                tool_name,
                arguments,
                turn,
                "tool_not_allowed",
                f"tool '{tool_name}' is disabled by profile "
                f"'{self.profile.profile_id}'",
            )
        try:
            parsed_arguments = tool.arguments_type.model_validate(arguments)
        except ValidationError as error:
            return self._record_error(
                tool_name,
                arguments,
                turn,
                "invalid_arguments",
                f"arguments rejected by schema: {error}",
            )

        try:
            raw_result = tool.handler(parsed_arguments)
            result = tool.result_type.model_validate(raw_result)
        except BudgetExhaustedError as error:
            self._record_error(
                tool_name, arguments, turn, "budget_exhausted", str(error)
            )
            raise

        canonical = result.model_dump(mode="json")
        observation = result.message[:MODEL_OBSERVATION_LIMIT]
        self.event_log.append(
            "tool_result",
            turn=turn,
            tool=tool_name,
            arguments=parsed_arguments.model_dump(mode="json"),
            succeeded=True,
            result=canonical,
            observation=observation,
            budget=self.meter.snapshot(),
        )
        return ToolExecution(observation, True, canonical)

    def _record_error(
        self,
        tool_name: str,
        arguments: dict,
        turn: int,
        code: str,
        message: str,
    ) -> ToolExecution:
        canonical = {"error": {"code": code, "message": message}}
        observation = message[:MODEL_OBSERVATION_LIMIT]
        self.event_log.append(
            "tool_result",
            turn=turn,
            tool=tool_name,
            arguments=arguments,
            succeeded=False,
            result=canonical,
            observation=observation,
            budget=self.meter.snapshot(),
        )
        return ToolExecution(observation, False, canonical)
