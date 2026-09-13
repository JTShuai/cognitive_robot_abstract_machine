"""
The planning-object agent searches the world directory and proposes objects.
"""

from __future__ import annotations

import json

import pytest

from resym.core.symbols import Literal
from resym.interfaces.object_selection import (
    AGENT_NAME,
    OBJECT_SCOPE_TOOL_NAMES,
    PlanningObjectAgent,
)
from resym.llm.client import ScriptedCompletionClient
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.planning.events import ObjectScopeExpansionReason
from resym.planning.object_scope import (
    ObjectInclusion,
    ObjectInclusionReason,
    ObjectRecommendation,
    ObjectScopeAdviceRequest,
    PlanningObjectScope,
)
from resym.planning.selection import Selection
from resym.repair.backends import Budget
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup

from .test_object_scope import object_predicate, objects, resolvable_robot_type

# %% fixtures


def action(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


def agent_with(
    *responses: str, **settings
) -> tuple[PlanningObjectAgent, ScriptedCompletionClient]:
    client = ScriptedCompletionClient(responses=list(responses))
    completer = StructuredCompleter(client=client, transcript=TranscriptRecorder())
    return PlanningObjectAgent(completer, **settings), client


def request_over(objects, **overrides) -> ObjectScopeAdviceRequest:
    universe, robot = objects
    selection = Selection(predicates={"done": object_predicate("done", Cup)})
    scope = PlanningObjectScope(
        {
            "cup-0": ObjectInclusion(ObjectInclusionReason.GOAL),
            "task_robot": ObjectInclusion(ObjectInclusionReason.ROBOT),
        }
    )
    arguments = dict(
        goal=(Literal("done", ("cup-0",)),),
        selection=selection,
        scope=scope,
        world_objects=universe,
        eligible_names=frozenset(f"cup-{index}" for index in range(12)),
    )
    arguments.update(overrides)
    return ObjectScopeAdviceRequest(**arguments)


# %% proposals


def test_agent_queries_inspects_and_proposes_with_rationale(objects):
    agent, client = agent_with(
        action("query_candidate_objects", name_contains="cup-7"),
        action("inspect_object_relations", object_name="cup-7"),
        action(
            "propose_planning_objects",
            objects=[{"name": "cup-7", "rationale": "the instruction names cup seven"}],
        ),
        instruction="bring me cup seven",
    )

    advice = agent.advise(request_over(objects))

    assert advice.recommendations == (
        ObjectRecommendation("cup-7", "the instruction names cup seven"),
    )
    assert [event["event"] for event in advice.trace] == [
        "episode_started",
        "model_request",
        "model_response",
        "model_action",
        "tool_call",
        "tool_result",
        "model_request",
        "model_response",
        "model_action",
        "tool_call",
        "tool_result",
        "model_request",
        "model_response",
        "model_action",
        "tool_call",
        "tool_result",
        "episode_finished",
    ]
    assert "bring me cup seven" in client.received_prompts[0]
    assert "cup-0" in client.received_prompts[0]
    query_result = advice.trace[5]["result"]
    assert query_result["objects"] == [
        {
            "name": "cup-7",
            "type": "semantic_digital_twin.semantic_annotations.semantic_annotations.Cup",
            "eligible": True,
            "selected": False,
        }
    ]
    assert advice.trace[-1]["reason"] == "objects_proposed"


def test_unknown_or_ineligible_proposals_are_sent_back_for_correction(objects):
    agent, client = agent_with(
        action(
            "propose_planning_objects",
            objects=[
                {"name": "cup-99", "rationale": "guess"},
                {"name": "task_robot", "rationale": "actor"},
            ],
        ),
        action(
            "propose_planning_objects", objects=[{"name": "cup-3", "rationale": "near"}]
        ),
    )

    advice = agent.advise(request_over(objects))

    assert advice.recommendations == (ObjectRecommendation("cup-3", "near"),)
    rejection = advice.trace[5]["result"]
    assert rejection["rejected"] == {
        "cup-99": "not in the world directory",
        "task_robot": "its type is unused by the selected symbols",
    }
    assert "cup-99" in client.received_prompts[1]


def test_empty_proposal_means_nothing_to_add(objects):
    agent, _ = agent_with(action("propose_planning_objects", objects=[]))

    advice = agent.advise(request_over(objects))

    assert advice.recommendations == ()
    assert advice.trace[-1]["reason"] == "objects_proposed"


def test_budget_exhaustion_yields_no_recommendations(objects):
    agent, _ = agent_with(
        action("query_candidate_objects"),
        action("query_candidate_objects"),
        action("query_candidate_objects"),
        budget=Budget(tool_calls=2),
    )

    advice = agent.advise(request_over(objects))

    assert advice.recommendations == ()
    assert advice.trace[-1]["reason"] == "budget_exhausted:tool_calls"


def test_unparseable_model_action_ends_the_episode(objects):
    agent, _ = agent_with("not json", "still not json", "nope")

    advice = agent.advise(request_over(objects))

    assert advice.recommendations == ()
    assert advice.trace[-1]["reason"] == "invalid_model_action"


def test_expansion_prompt_carries_the_planner_evidence(objects):
    agent, client = agent_with(action("propose_planning_objects", objects=[]))

    agent.advise(
        request_over(
            objects,
            expansion_reason=ObjectScopeExpansionReason.UNSOLVABLE_SUBSET,
            planner_message="Fast Downward found no plan",
        )
    )

    prompt = client.received_prompts[0]
    assert "unsolvable_subset" in prompt
    assert "Fast Downward found no plan" in prompt


def test_tools_outside_the_profile_are_refused(objects):
    agent, _ = agent_with(
        action("propose_patch", proposal={}),
        action("propose_planning_objects", objects=[]),
    )

    advice = agent.advise(request_over(objects))

    refused = advice.trace[5]
    assert refused["succeeded"] is False
    assert refused["result"]["error"]["code"] == "unknown_tool"
    assert OBJECT_SCOPE_TOOL_NAMES == frozenset(
        {
            "query_candidate_objects",
            "inspect_object_relations",
            "propose_planning_objects",
        }
    )
    assert advice.trace[1]["agent"] == AGENT_NAME
