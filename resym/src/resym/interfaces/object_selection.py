"""
Language-model selection of task-relevant planning objects.

The model searches the world directory with typed queries and relation lookups, then
proposes objects to add to the planning scope with a rationale each. Every proposal is
validated against the world directory and the typed candidate bound before it reaches
the selector; the model never supplies truth values and cannot replace goal objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Optional

from resym.core.symbols import SymbolLibrary
from resym.llm.prompting import (
    render_literal,
    render_operators,
    render_predicates,
    render_prompt,
)
from resym.llm.schemas import AgentActionModel
from resym.llm.structured import (
    StructuredCompleter,
    StructuredOutputRetriesExceededError,
)
from resym.llm.transcript import LanguageModelExchange
from resym.platform.krrood_queries import KrroodObjectResolver, ObjectQuerySpec
from resym.planning.object_scope import (
    ObjectRecommendation,
    ObjectScopeAdvice,
    ObjectScopeAdviceRequest,
)
from resym.repair.agent_harness import (
    AgentProfile,
    EpisodeEventLog,
    RepairTool,
    ToolArguments,
    ToolExecutionPipeline,
    ToolRegistry,
    ToolResult,
)
from resym.repair.backends import Budget, BudgetExhaustedError, BudgetMeter

AGENT_NAME = "planning-object-agent"
"""
Transcript identity of this agent.
"""

PROMPT_TEMPLATE = "select_planning_objects"

# %% tool contracts


class ObjectQueryArguments(ToolArguments):
    """
    Typed description of the objects to look up in the world directory.
    """

    type: Optional[str] = None
    color: Optional[str] = None
    name_contains: Optional[str] = None


class ObjectSummary(BaseModel):
    """
    One directory entry as shown to the model.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    type: str
    eligible: bool
    selected: bool


class ObjectQueryResult(ToolResult):
    objects: list[ObjectSummary]


class ObjectRelationsArguments(ToolArguments):
    object_name: str = Field(min_length=1)


class ObjectRelation(BaseModel):
    """
    One native relationship reached from the inspected object.
    """

    model_config = ConfigDict(extra="forbid")
    object_name: str
    related_to: str


class ObjectRelationsResult(ToolResult):
    relations: list[ObjectRelation]


class ProposedObjectArguments(ToolArguments):
    name: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class PlanningObjectProposalArguments(ToolArguments):
    objects: list[ProposedObjectArguments]


class PlanningObjectProposalResult(ToolResult):
    accepted: list[str]
    rejected: dict[str, str]


OBJECT_SCOPE_TOOL_NAMES = frozenset(
    {
        "query_candidate_objects",
        "inspect_object_relations",
        "propose_planning_objects",
    }
)

OBJECT_SCOPE_PROFILE = AgentProfile(
    profile_id="planning-objects-v1",
    allowed_tools=OBJECT_SCOPE_TOOL_NAMES,
    retrieval_required_when_available=False,
)

OBJECT_SCOPE_BUDGET = Budget(candidates=1, tool_calls=12, probes=0)
"""
Default allowance for one selection episode.
"""


class EpisodeEnd(StrEnum):
    """
    Termination reasons recorded on the episode log.
    """

    OBJECTS_PROPOSED = "objects_proposed"
    INVALID_MODEL_ACTION = "invalid_model_action"
    MAXIMUM_TURNS = "maximum_turns_reached"


# %% the agent


@dataclass
class SelectionEpisodeState:
    """
    Mutable working state of one selection episode.
    """

    proposal: Optional[tuple[ObjectRecommendation, ...]] = None


@dataclass
class PlanningObjectAgent:
    """
    Bounded tool loop that recommends planning objects for one task.
    """

    completer: StructuredCompleter
    """
    The structured language-model channel.
    """

    instruction: Optional[str] = None
    """
    The natural-language task the goal was derived from, when one exists.
    """

    budget: Budget = OBJECT_SCOPE_BUDGET
    """
    Allowance for every consultation of this agent.
    """

    maximum_turns: int = 8
    """
    Backstop on model replies per consultation.
    """

    profile: AgentProfile = OBJECT_SCOPE_PROFILE
    """
    Tool policy frozen into every episode record.
    """

    def advise(self, request: ObjectScopeAdviceRequest) -> ObjectScopeAdvice:
        """
        Run one selection episode and return the validated recommendations.
        """
        state = SelectionEpisodeState()
        meter = BudgetMeter(budget=self.budget)
        event_log = EpisodeEventLog()
        registry = self._tools(request, state)
        pipeline = ToolExecutionPipeline(registry, self.profile, meter, event_log)
        event_log.append(
            "episode_started",
            event_schema_version=1,
            agent=AGENT_NAME,
            profile=self.profile.to_json(),
            maximum_turns=self.maximum_turns,
        )
        termination_reason = EpisodeEnd.MAXIMUM_TURNS
        try:
            for turn_index in range(self.maximum_turns):
                turn = turn_index + 1
                prompt = self._prompt(request, meter, event_log, registry)

                def record_exchange(exchange: LanguageModelExchange) -> None:
                    event_log.append(
                        "model_request",
                        turn=turn,
                        attempt=exchange.attempt,
                        agent=exchange.agent_name,
                        model=exchange.model_description,
                        prompt=exchange.prompt,
                    )
                    event_log.append(
                        "model_response",
                        turn=turn,
                        attempt=exchange.attempt,
                        response=exchange.response,
                        parse_error=exchange.parse_error,
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
                    )
                    termination_reason = EpisodeEnd.INVALID_MODEL_ACTION
                    break
                event_log.append(
                    "model_action",
                    turn=turn,
                    succeeded=True,
                    action=action.model_dump(mode="json"),
                )
                pipeline.execute(action.tool, action.arguments, turn)
                if state.proposal is not None:
                    termination_reason = EpisodeEnd.OBJECTS_PROPOSED
                    break
        except BudgetExhaustedError as error:
            termination_reason = f"budget_exhausted:{error.dimension}"
        event_log.append(
            "episode_finished",
            reason=str(termination_reason),
            recommended=[item.name for item in state.proposal or ()],
            budget=meter.snapshot(),
        )
        return ObjectScopeAdvice(
            recommendations=state.proposal or (),
            trace=tuple(event_log.to_json()),
        )

    # %% tools

    def _tools(
        self, request: ObjectScopeAdviceRequest, state: SelectionEpisodeState
    ) -> ToolRegistry:
        resolver = KrroodObjectResolver(request.world_objects)
        required_types = {
            symbol_type
            for predicate in request.selection.predicates.values()
            for symbol_type in predicate.parameter_types
        } | {
            symbol_type
            for operator in request.selection.operators.values()
            for _, symbol_type in operator.parameters
        }

        def summary(name: str) -> ObjectSummary:
            item = request.world_objects[name]
            return ObjectSummary(
                name=name,
                type=item.symbol_type.python_type_ref,
                eligible=name in request.eligible_names,
                selected=name in request.scope.object_names,
            )

        def query_candidate_objects(
            arguments: ObjectQueryArguments,
        ) -> ObjectQueryResult:
            matches = resolver.resolve(
                ObjectQuerySpec(
                    type_ref=arguments.type,
                    color=arguments.color,
                    name_contains=arguments.name_contains,
                )
            )
            summaries = [summary(item.name) for item in matches]
            lines = [
                f"- {item.name} ({item.type})"
                + (" eligible" if item.eligible else " not eligible for planning")
                + (" already selected" if item.selected else "")
                for item in summaries
            ]
            return ObjectQueryResult(
                message="\n".join(lines) or "no objects match this description",
                objects=summaries,
            )

        def inspect_object_relations(
            arguments: ObjectRelationsArguments,
        ) -> ObjectRelationsResult:
            if arguments.object_name not in request.world_objects.objects:
                return ObjectRelationsResult(
                    message=f"unknown object '{arguments.object_name}'", relations=[]
                )
            dependencies = resolver.dependencies(
                {arguments.object_name}, required_types
            )
            relations = [
                ObjectRelation(
                    object_name=item.object_name,
                    related_to=item.related_to,
                )
                for item in dependencies
            ]
            return ObjectRelationsResult(
                message="\n".join(
                    f"- {item.object_name} reached from {item.related_to}"
                    for item in relations
                )
                or "no native relations recorded for this object",
                relations=relations,
            )

        def propose_planning_objects(
            arguments: PlanningObjectProposalArguments,
        ) -> PlanningObjectProposalResult:
            rejected: dict[str, str] = {}
            accepted: list[ObjectRecommendation] = []
            for proposed in arguments.objects:
                if proposed.name not in request.world_objects.objects:
                    rejected[proposed.name] = "not in the world directory"
                elif proposed.name not in request.eligible_names:
                    rejected[proposed.name] = (
                        "its type is unused by the selected symbols"
                    )
                else:
                    accepted.append(
                        ObjectRecommendation(proposed.name, proposed.rationale.strip())
                    )
            if rejected:
                return PlanningObjectProposalResult(
                    message="proposal rejected; correct these objects and propose again: "
                    + "; ".join(
                        f"{name}: {reason}" for name, reason in rejected.items()
                    ),
                    accepted=[item.name for item in accepted],
                    rejected=rejected,
                )
            state.proposal = tuple(accepted)
            return PlanningObjectProposalResult(
                message="proposal accepted: "
                + (", ".join(item.name for item in accepted) or "nothing to add"),
                accepted=[item.name for item in accepted],
                rejected={},
            )

        return ToolRegistry(
            [
                RepairTool(
                    "query_candidate_objects",
                    "find world objects by exact type reference, named colour, or "
                    "name fragment",
                    ObjectQueryArguments,
                    ObjectQueryResult,
                    query_candidate_objects,
                ),
                RepairTool(
                    "inspect_object_relations",
                    "list the objects one object references, contains, or rests on",
                    ObjectRelationsArguments,
                    ObjectRelationsResult,
                    inspect_object_relations,
                ),
                RepairTool(
                    "propose_planning_objects",
                    "finish by naming the objects to add, most important first, "
                    "each with a rationale",
                    PlanningObjectProposalArguments,
                    PlanningObjectProposalResult,
                    propose_planning_objects,
                ),
            ]
        )

    # %% prompt

    def _prompt(
        self,
        request: ObjectScopeAdviceRequest,
        meter: BudgetMeter,
        event_log: EpisodeEventLog,
        registry: ToolRegistry,
    ) -> str:
        library = SymbolLibrary(
            predicates=dict(request.selection.predicates),
            operators=dict(request.selection.operators),
        )
        counts: dict[str, int] = {}
        for name in sorted(request.eligible_names):
            type_ref = request.world_objects[name].symbol_type.python_type_ref
            counts[type_ref] = counts.get(type_ref, 0) + 1
        expansion = (
            "The planner proved the current selection insufficient "
            f"({request.expansion_reason.value}): {request.planner_message}"
            if request.expansion_reason is not None
            else "This is the first selection for the task."
        )
        return render_prompt(
            PROMPT_TEMPLATE,
            instruction=self.instruction or "(none; the goal was given directly)",
            goal="\n".join(f"- {render_literal(literal)}" for literal in request.goal)
            or "- none",
            predicates=render_predicates(library) or "- none",
            operators=render_operators(library) or "- none",
            scope="\n".join(
                f"- {name}: {inclusion.reason.value}"
                + (f" (via {inclusion.related_to})" if inclusion.related_to else "")
                for name, inclusion in sorted(request.scope.inclusion_reasons.items())
            )
            or "- none",
            candidate_types="\n".join(
                f"- {type_ref}: {count}" for type_ref, count in sorted(counts.items())
            )
            or "- none",
            expansion=expansion,
            tools=registry.render(self.profile.allowed_tools),
            budget=str(meter.snapshot()),
            history=event_log.render_tool_history() or "(no tool calls yet)",
        )
