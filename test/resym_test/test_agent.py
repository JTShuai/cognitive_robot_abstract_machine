"""
The retrieval-augmented planning agent: tool loop, RAG, budgets, and termination
conditions.

Host-runnable: scripted action sequences drive whole episodes.
"""

from __future__ import annotations

import json

import pytest

from resym.repair.agent import RetrievalAugmentedPlanningAgent
from resym.repair.agent_harness import AgentProfile
from resym.platform.grounding_catalog import GroundingFactorySourceValidator
from resym.repair.backends import (
    Budget,
    BudgetMeter,
    CapabilityGapKind,
    OutcomeStatus,
    RepairTask,
    RetrievalStatus,
)
from resym.core.provenance import KnowledgeSource
from resym.knowledge.corpus import load_fragment
from resym.knowledge.retrieval import FragmentIndex
from resym.llm.client import ScriptedCompletionClient
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.core.model import (
    PredicateSymbol,
    SymbolLibrary,
)
from .capability_helpers import capability_contract
from experiments.resym.seed_library import build_fixed_arm_library
from .grounding_helpers import STUB_GROUNDING_PLAN
from .test_grounding_factory_catalog import DATASET, vocabulary

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
OBJECT_TYPE = SymbolType.from_python_type(SemanticAnnotation)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


CAPABILITY_UID = "test:DrawerStateChange"


class StubGoal:
    predicate = "closed"
    arguments = ("d1",)
    negated = False


class StubNeighborhood:
    predicates = ("opened", "closed")
    operators = ("open-drawer",)
    unachievable_goal_predicates = ("closed",)


class StubFailureClass:
    value = "missing_operator_model"


class StubCertificate:
    task_goal = (StubGoal(),)
    causal_neighborhood = StubNeighborhood()
    failure_class = StubFailureClass()
    task_instruction = None
    goal_semantics = ()
    task_object_types = ()

    def render(self) -> str:
        return "failure class: missing_operator_model\ngoal: (closed d1)"


def library() -> SymbolLibrary:
    lib = SymbolLibrary()
    for name in ("opened", "closed"):
        lib.add(
            PredicateSymbol(
                name=name,
                parameter_types=(DRAWER_TYPE,),
                fluent=True,
                grounding_plan=STUB_GROUNDING_PLAN,
            )
        )
    lib.add_capability_contract(
        capability_contract(
            CAPABILITY_UID,
            (("patient", DRAWER_TYPE),),
            ("opened", "closed"),
        )
    )
    return lib


def action(tool: str, **arguments) -> str:
    return json.dumps({"tool": tool, "arguments": arguments})


def proposal_arguments(name: str = "close-drawer") -> dict:
    return {
        "proposal": {
            "rationale": "add the missing close operator",
            "predicates": [],
            "operators": [
                {
                    "name": name,
                    "parameters": [
                        {
                            "variable": "d",
                            "type": DRAWER_TYPE.python_type_ref,
                        }
                    ],
                    "preconditions": [
                        {"predicate": "opened", "arguments": ["d"], "negated": False}
                    ],
                    "add_effects": [
                        {"predicate": "closed", "arguments": ["d"], "negated": False}
                    ],
                    "delete_effects": [
                        {"predicate": "opened", "arguments": ["d"], "negated": False}
                    ],
                    "capability_uid": CAPABILITY_UID,
                    "role_bindings": {"patient": "d"},
                }
            ],
        }
    }


@pytest.fixture()
def corpus_index(tmp_path):
    record = {
        "predicates": {
            "(drawer_open ?d)": "drawer is open",
            "(drawer_closed ?d)": "drawer is closed",
        },
        "operators": {
            "close_drawer": (
                "(:action close_drawer :parameters (?d) "
                ":precondition (drawer_open ?d) "
                ":effect (and (drawer_closed ?d) (not (drawer_open ?d))))"
            )
        },
    }
    path = tmp_path / "1" / "episode_1" / "atomic_domain.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    return FragmentIndex([load_fragment(path)])


def agent_with(
    *responses: str,
) -> tuple[RetrievalAugmentedPlanningAgent, ScriptedCompletionClient]:
    client = ScriptedCompletionClient(responses=list(responses))
    completer = StructuredCompleter(client=client, transcript=TranscriptRecorder())
    return RetrievalAugmentedPlanningAgent(completer), client


def task_with(index=None, check_patch=None, run_probe=None) -> RepairTask:
    task_library = library()
    return RepairTask(
        certificate=StubCertificate(),
        library=task_library,
        grounding_factory_listing="- resym:grounding/joint-fraction-opened: joint above threshold",
        capability_listing=f"- {CAPABILITY_UID}: set drawer state",
        capability_catalog=tuple(task_library.capability_contracts.values()),
        index=index,
        check_patch=check_patch,
        run_probe=run_probe,
        probe_listing="- close-probe: try closing in the witness world",
    )


def test_full_episode_retrieve_propose_check_probe_submit(corpus_index):
    probe_calls = []

    def run_probe(name, patch):
        probe_calls.append(name)
        return {"probe": name, "result": "goal reached"}

    agent, client = agent_with(
        action("retrieve_domain_fragments"),
        action("propose_patch", **proposal_arguments()),
        action("check_patch"),
        action("execute_diagnostic_probe", probe="close-probe"),
        action("submit_for_admission"),
    )
    meter = BudgetMeter()
    outcome = agent.repair(
        task_with(index=corpus_index, check_patch=lambda p: [], run_probe=run_probe),
        meter,
    )
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    (operator,) = outcome.patch.operators
    assert operator.name == "close-drawer"
    assert operator.provenance.retrieved_ids == ("1/1",)
    assert outcome.retrieved_ids == ("1/1",)
    assert probe_calls == ["close-probe"]
    assert meter.tool_calls_used == 5
    assert meter.candidates_used == 1
    assert meter.probes_used == 1


def test_agentic_rag_retrieves_before_proposing_when_corpus_exists(corpus_index):
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("retrieve_domain_fragments"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )

    outcome = agent.repair(task_with(index=corpus_index), BudgetMeter())

    assert agent.name == "agentic-rag"
    assert "must be called before proposing" in client.received_prompts[1]
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.retrieved_ids == ("1/1",)
    assert outcome.budget["candidates_used"] == 1


def test_agent_can_use_its_prior_after_retrieval_finds_no_evidence():
    agent, _ = agent_with(
        action("retrieve_domain_fragments"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )

    outcome = agent.repair(task_with(index=FragmentIndex([])), BudgetMeter())

    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.retrieved_ids == ()
    assert outcome.patch.operators[0].provenance.knowledge_source is (
        KnowledgeSource.LLM_PRIOR
    )
    finished = outcome.events[-1]
    assert finished["retrieval_status"] is RetrievalStatus.NO_RELEVANT_EVIDENCE


def test_history_reaches_the_model_each_turn(corpus_index):
    agent, client = agent_with(
        action("retrieve_domain_fragments"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )
    agent.repair(task_with(index=corpus_index), BudgetMeter())
    final_prompt = client.received_prompts[-1]
    assert "retrieve_domain_fragments" in final_prompt  # step 1 visible
    assert "candidate registered" in final_prompt  # step 2's observation visible
    assert "close_drawer" in client.received_prompts[1]  # retrieval observation


def test_objections_are_observations_and_revision_is_local(corpus_index):
    checks = iter([["operator 'close-drawer': unknown skill"], []])
    revision = proposal_arguments()
    revision["proposal"]["operators"][0]["preconditions"] = []
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("check_patch"),
        action("propose_patch", **revision),
        action("compare_patch_versions"),
        action("check_patch"),
        action("submit_for_admission"),
    )
    outcome = agent.repair(task_with(check_patch=lambda p: next(checks)), BudgetMeter())
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert "unknown skill" in client.received_prompts[2]  # counterexample visible
    compare_step = next(
        event
        for event in outcome.events
        if event["event"] == "tool_result" and event["tool"] == "compare_patch_versions"
    )
    assert compare_step["tool"] == "compare_patch_versions"
    assert "complexity" in compare_step["observation"]


def test_declare_unsupported_terminates():
    agent, _ = agent_with(
        action("inspect_embodiment_profile"),
        action("declare_unsupported", reason="no pull skill on this arm"),
    )
    outcome = agent.repair(task_with(), BudgetMeter())
    assert outcome.status is OutcomeStatus.UNSUPPORTED_DECLARED
    assert any(e.get("reason") == "no pull skill on this arm" for e in outcome.events)
    assert outcome.patch is None


def test_agent_reports_a_structured_missing_execution_capability():
    source_id = "coraplex:robot_plans.actions.core.placing.place-action"
    task = task_with()
    task.capability_draft_listing = f"- {source_id}: PlaceAction; draft only"
    task.capability_draft_ids = frozenset({source_id})
    agent, _ = agent_with(
        action(
            "report_missing_execution_capability",
            suggested_label="placement.put-into-container",
            desired_effects=["inside"],
            required_roles={
                "actor": ROBOT_TYPE.python_type_ref,
                "item": HANDLE_TYPE.python_type_ref,
                "container": DRAWER_TYPE.python_type_ref,
            },
            candidate_realizations=[source_id],
            reason="PlaceAction exists, but no reviewed placement contract is published",
        )
    )

    outcome = agent.repair(task, BudgetMeter())

    assert outcome.status is OutcomeStatus.MISSING_EXECUTION_CAPABILITY
    assert outcome.patch is None
    assert outcome.missing_execution_capability is not None
    assert outcome.missing_execution_capability.suggested_label == (
        "placement.put-into-container"
    )
    assert outcome.missing_execution_capability.candidate_realizations == (source_id,)
    assert outcome.missing_execution_capability.gap_kind is (
        CapabilityGapKind.MISSING_CONTRACT
    )


def test_agent_can_submit_a_bounded_eql_factory_candidate_for_human_review():
    submitted = []
    reviewed_vocabulary = vocabulary()
    validator = GroundingFactorySourceValidator(reviewed_vocabulary)
    task = task_with()
    task.grounding_vocabulary_listing = reviewed_vocabulary.render()
    task.validate_grounding_candidate = validator.candidate_objections
    task.grounding_candidate_sink = submitted.append
    agent, _ = agent_with(
        action(
            "propose_grounding_factory_candidate",
            candidate_id="inside-region-candidate",
            proposed_uid="resym:grounding/inside-region",
            semantic_name="inside-region",
            source_code=(DATASET / "valid_factory.py").read_text(),
            roles=[{"name": "object", "symbol_type": ROBOT_TYPE.python_type_ref}],
            rationale="No reviewed factory implements the required relation.",
            evidence=["unidomain:inside"],
        )
    )

    outcome = agent.repair(task, BudgetMeter())

    assert outcome.status is OutcomeStatus.GROUNDING_FACTORY_PROPOSED
    assert outcome.grounding_factory_candidate is submitted[0]
    assert submitted[0].proposed_uid == "resym:grounding/inside-region"
    assert submitted[0].generated_by == agent.name


def test_candidate_roles_keep_their_declared_order():
    """
    Role order is positional semantics; it must never be re-sorted.
    """
    submitted = []
    reviewed_vocabulary = vocabulary()
    task = task_with()
    task.validate_grounding_candidate = GroundingFactorySourceValidator(
        reviewed_vocabulary
    ).candidate_objections
    task.grounding_candidate_sink = submitted.append
    agent, _ = agent_with(
        action(
            "propose_grounding_factory_candidate",
            candidate_id="ordered-roles-candidate",
            proposed_uid="resym:grounding/ordered-roles",
            semantic_name="ordered-roles",
            source_code=(DATASET / "valid_factory.py").read_text(),
            roles=[
                {"name": "object", "symbol_type": OBJECT_TYPE.python_type_ref},
                {"name": "container", "symbol_type": OBJECT_TYPE.python_type_ref},
            ],
            rationale="Role order must match the drafted evaluate body.",
        )
    )

    outcome = agent.repair(task, BudgetMeter())

    assert outcome.status is OutcomeStatus.GROUNDING_FACTORY_PROPOSED
    assert tuple(role.name for role in submitted[0].roles) == ("object", "container")


def test_agent_reports_a_structured_grounding_gap_when_eql_is_insufficient():
    agent, _ = agent_with(
        action(
            "report_missing_grounding_capability",
            required_relation="contains-liquid",
            input_types={"container": OBJECT_TYPE.python_type_ref},
            missing_computation="fluid state simulation",
            reason="The reviewed EQL vocabulary has no fluid-state source.",
        )
    )

    outcome = agent.repair(task_with(), BudgetMeter())

    assert outcome.status is OutcomeStatus.MISSING_GROUNDING_CAPABILITY
    assert outcome.missing_grounding_capability is not None
    assert outcome.missing_grounding_capability.required_relation == "contains-liquid"


def test_agent_can_search_the_complete_reviewed_capability_catalog():
    from resym.platform.capabilities import (
        PLACE_CAPABILITY_UID,
        capability_contracts,
    )

    task = task_with()
    task.capability_catalog = capability_contracts()
    agent, _ = agent_with(
        action(
            "search_capability_catalog",
            desired_effects=["placed-at"],
            required_roles={
                "patient": OBJECT_TYPE.python_type_ref,
                "destination": OBJECT_TYPE.python_type_ref,
            },
        ),
        action("declare_unsupported", reason="test complete"),
    )

    outcome = agent.repair(task, BudgetMeter())

    search = next(
        event
        for event in outcome.events
        if event.get("event") == "tool_result"
        and event.get("tool") == "search_capability_catalog"
    )
    assert search["result"]["matches"][0]["uid"] == PLACE_CAPABILITY_UID


def test_unknown_coraplex_action_cannot_be_cited_in_capability_gap():
    agent, client = agent_with(
        action(
            "report_missing_execution_capability",
            suggested_label="teleport",
            desired_effects=["at"],
            required_roles={},
            candidate_realizations=["coraplex:no-such-action"],
            reason="not implemented",
        ),
        action("declare_unsupported", reason="no matching native action"),
    )

    outcome = agent.repair(task_with(), BudgetMeter())

    assert outcome.status is OutcomeStatus.UNSUPPORTED_DECLARED
    assert "unknown Coraplex action candidates" in client.received_prompts[1]


def test_unknown_tool_is_an_observation_not_a_crash():
    agent, client = agent_with(
        action("teleport_library"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )
    outcome = agent.repair(task_with(), BudgetMeter())
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert "unknown tool 'teleport_library'" in client.received_prompts[1]


def test_submit_without_candidate_is_refused():
    agent, client = agent_with(
        action("submit_for_admission"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )
    outcome = agent.repair(task_with(), BudgetMeter())
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert "nothing to submit" in client.received_prompts[1]


def test_duplicate_candidate_is_idempotent_and_does_not_spend_candidate_budget():
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )

    outcome = agent.repair(task_with(), BudgetMeter())

    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.budget["candidates_used"] == 1
    assert (
        "submit_for_admission is the required next step" in client.received_prompts[2]
    )


def test_duplicate_candidate_with_predicates_is_recognized():
    arguments = proposal_arguments()
    arguments["proposal"]["predicates"] = [
        {
            "name": "latched",
            "parameter_types": [DRAWER_TYPE.python_type_ref],
            "fluent": True,
            "grounding_plan": {
                "factory_uid": "resym:grounding/joint-fraction-opened",
                "approved_factory_checksum": "reviewed-checksum",
                "role_bindings": {"articulated_object": 0},
                "parameters": {"threshold": 0.5},
            },
        }
    ]
    agent, client = agent_with(
        action("propose_patch", **arguments),
        action("check_patch"),
        action("propose_patch", **arguments),
        action("check_patch"),
        action("submit_for_admission"),
    )
    reviews = iter((["not admissible yet"], []))

    outcome = agent.repair(
        task_with(check_patch=lambda patch: next(reviews)), BudgetMeter()
    )

    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.budget["candidates_used"] == 1
    assert any(
        "duplicate candidate ignored" in prompt for prompt in client.received_prompts
    )


def test_submission_requires_static_check_and_required_probe():
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
        action("check_patch"),
        action("submit_for_admission"),
        action("execute_diagnostic_probe", probe="close-probe"),
        action("submit_for_admission"),
    )
    task = task_with(
        check_patch=lambda patch: [],
        run_probe=lambda name, patch: {"succeeded": True},
    )
    task.required_probes = ("close-probe",)

    outcome = agent.repair(task, BudgetMeter())

    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert "run check_patch" in client.received_prompts[2]
    assert "run required probes" in client.received_prompts[4]


def test_submission_refuses_a_failed_required_probe():
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("check_patch"),
        action("execute_diagnostic_probe", probe="close-probe"),
        action("submit_for_admission"),
        action("declare_unsupported", reason="test complete"),
    )
    task = task_with(
        check_patch=lambda patch: [],
        run_probe=lambda name, patch: {"succeeded": False},
    )
    task.required_probes = ("close-probe",)

    outcome = agent.repair(task, BudgetMeter())

    assert outcome.status is OutcomeStatus.UNSUPPORTED_DECLARED
    assert "failed required probes" in client.received_prompts[4]


def test_static_check_rejects_unrelated_existing_operator_edit(grounding_catalog):
    task = task_with(check_patch=lambda patch: [])
    task.library = build_fixed_arm_library(grounding_catalog)
    unrelated = {
        "proposal": {
            "rationale": "unrelated edit",
            "operators": [{"name": "close-drawer", "preconditions": []}],
        }
    }
    agent, client = agent_with(
        action("propose_patch", **unrelated),
        action("check_patch"),
        action("declare_unsupported", reason="test complete"),
    )

    agent.repair(task, BudgetMeter())

    assert "outside the failure's causal neighborhood" in client.received_prompts[2]


def test_budget_exhaustion_ends_the_episode():
    agent, _ = agent_with(
        *[action("inspect_library")] * 10,
    )
    outcome = agent.repair(task_with(), BudgetMeter(budget=Budget(tool_calls=3)))
    assert outcome.status is OutcomeStatus.BUDGET_EXHAUSTED
    assert outcome.budget["tool_calls_used"] == 3


def test_new_candidate_waits_for_current_candidate_validation():
    revised = proposal_arguments("alternative-close-drawer")
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("propose_patch", **revised),
        action("check_patch"),
        action("submit_for_admission"),
    )

    outcome = agent.repair(task_with(check_patch=lambda patch: []), BudgetMeter())

    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.patch.operators[0].name == "close-drawer"
    assert outcome.budget["candidates_used"] == 1
    assert "check_patch is the required next step" in client.received_prompts[2]


def test_episode_steps_carry_budget_snapshots_for_replay():
    agent, _ = agent_with(
        action("inspect_library"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )
    outcome = agent.repair(task_with(), BudgetMeter())
    steps = [e for e in outcome.events if e["event"] == "tool_result"]
    assert [s["tool"] for s in steps] == [
        "inspect_library",
        "propose_patch",
        "submit_for_admission",
    ]
    assert all("budget" in s and "observation" in s for s in steps)


def test_episode_log_orders_model_tools_candidate_and_termination():
    agent, _ = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )
    outcome = agent.repair(task_with(), BudgetMeter())

    assert [event["sequence"] for event in outcome.events] == list(
        range(1, len(outcome.events) + 1)
    )
    assert outcome.events[0]["event"] == "episode_started"
    assert outcome.events[0]["profile"]["profile_id"] == "agentic-rag-v6"
    assert outcome.events[-1]["event"] == "episode_finished"
    assert len([e for e in outcome.events if e["event"] == "model_request"]) == 2
    assert len([e for e in outcome.events if e["event"] == "model_response"]) == 2
    assert len([e for e in outcome.events if e["event"] == "model_action"]) == 2
    assert len([e for e in outcome.events if e["event"] == "tool_call"]) == 2
    assert len([e for e in outcome.events if e["event"] == "tool_result"]) == 2
    assert any(event["event"] == "candidate_proposed" for event in outcome.events)


def test_episode_log_records_every_structured_output_retry():
    agent, _ = agent_with(
        "not json",
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )

    outcome = agent.repair(task_with(), BudgetMeter())

    requests = [e for e in outcome.events if e["event"] == "model_request"]
    responses = [e for e in outcome.events if e["event"] == "model_response"]
    actions = [e for e in outcome.events if e["event"] == "model_action"]
    assert len(requests) == len(responses) == 3
    assert [event["attempt"] for event in responses[:2]] == [1, 2]
    assert responses[0]["parse_error"] is not None
    assert responses[1]["parse_error"] is None
    assert len(actions) == 2


def test_episode_log_marks_safe_structured_output_recovery():
    truncated_action = action("inspect_library")[:-1]
    agent, _ = agent_with(
        truncated_action,
        action("declare_unsupported", reason="recovery test complete"),
    )

    outcome = agent.repair(task_with(), BudgetMeter())

    response = next(
        event
        for event in outcome.events
        if event["event"] == "model_response" and event["turn"] == 1
    )
    assert response["parse_error"] is None
    assert response["parse_recovery"] == "closed_one_trailing_container"


def test_profile_rejects_disabled_tool_before_handler():
    agent, client = agent_with(
        action("inspect_library"),
        action("declare_unsupported", reason="profile test complete"),
    )
    agent.profile = AgentProfile(
        profile_id="declaration-only",
        allowed_tools=frozenset({"declare_unsupported"}),
    )

    outcome = agent.repair(task_with(), BudgetMeter())

    rejected = next(
        event
        for event in outcome.events
        if event["event"] == "tool_result" and event["tool"] == "inspect_library"
    )
    assert rejected["succeeded"] is False
    assert rejected["result"]["error"]["code"] == "tool_not_allowed"
    assert "disabled by profile" in client.received_prompts[1]


def test_tool_schema_rejects_invalid_arguments_before_retrieval(corpus_index):
    agent, client = agent_with(
        action("retrieve_domain_fragments", top_k=0),
        action("retrieve_domain_fragments"),
        action("propose_patch", **proposal_arguments()),
        action("submit_for_admission"),
    )

    outcome = agent.repair(task_with(index=corpus_index), BudgetMeter())

    rejected = next(
        event
        for event in outcome.events
        if event["event"] == "tool_result"
        and event["tool"] == "retrieve_domain_fragments"
        and not event["succeeded"]
    )
    assert rejected["result"]["error"]["code"] == "invalid_arguments"
    assert "arguments rejected by schema" in client.received_prompts[1]
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED


def test_full_probe_result_is_logged_but_model_observation_is_bounded():
    payload = "A" * 1500 + "END"
    agent, client = agent_with(
        action("propose_patch", **proposal_arguments()),
        action("execute_diagnostic_probe", probe="long-probe"),
        action("submit_for_admission"),
    )
    outcome = agent.repair(
        task_with(run_probe=lambda name, patch: {"payload": payload}), BudgetMeter()
    )

    probe_result = next(
        event
        for event in outcome.events
        if event["event"] == "tool_result"
        and event["tool"] == "execute_diagnostic_probe"
    )
    assert probe_result["result"]["observation"]["payload"].endswith("END")
    assert len(probe_result["observation"]) == 1200
    assert "END" not in client.received_prompts[-1]
