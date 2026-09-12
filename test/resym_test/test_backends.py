"""
Repair backends under one budget meter: closed-book, RAG one-shot, and the fixed
pipeline.

Host-runnable: scripted LLM client, duck-typed certificate, synthetic corpus index.
"""

from __future__ import annotations

import json

import pytest

from resym.repair.backends import (
    Budget,
    BudgetExhaustedError,
    BudgetMeter,
    ClosedBookBackend,
    FixedPipelineBackend,
    OutcomeStatus,
    RagOneShotBackend,
    RepairTask,
)
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
from .grounding_helpers import STUB_GROUNDING_PLAN

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)


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


def close_proposal_json() -> str:
    return json.dumps(
        {
            "rationale": "add the missing close operator",
            "predicates": [],
            "operators": [
                {
                    "name": "close-drawer",
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
    )


def completer_with(
    *responses: str,
) -> tuple[StructuredCompleter, ScriptedCompletionClient]:
    client = ScriptedCompletionClient(responses=list(responses))
    return (
        StructuredCompleter(client=client, transcript=TranscriptRecorder()),
        client,
    )


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


def task_with(index=None, check_patch=None) -> RepairTask:
    return RepairTask(
        certificate=StubCertificate(),
        library=library(),
        grounding_factory_listing="- resym:grounding/joint-fraction-opened: joint above threshold",
        capability_listing=f"- {CAPABILITY_UID}: set drawer state",
        index=index,
        check_patch=check_patch,
    )


def test_closed_book_produces_a_patch_with_provenance():
    completer, client = completer_with(close_proposal_json())
    outcome = ClosedBookBackend(completer).repair(task_with(), BudgetMeter())
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    (operator,) = outcome.patch.operators
    assert operator.name == "close-drawer"
    assert operator.provenance.proposal_backend == "closed-book"
    assert outcome.retrieved_ids == ()
    assert outcome.budget["candidates_used"] == 1


def test_rag_one_shot_renders_fragments_and_records_ids(corpus_index):
    completer, client = completer_with(close_proposal_json())
    outcome = RagOneShotBackend(completer).repair(
        task_with(index=corpus_index), BudgetMeter()
    )
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.retrieved_ids == ("1/1",)
    (operator,) = outcome.patch.operators
    assert operator.provenance.retrieved_ids == ("1/1",)
    prompt = client.received_prompts[0]
    assert "close_drawer" in prompt  # the fragment reached the model
    assert "untrusted" in prompt


def test_fixed_pipeline_feeds_objections_back(corpus_index):
    completer, client = completer_with(close_proposal_json(), close_proposal_json())
    verdicts = iter([["operator 'close-drawer': unknown skill"], []])
    outcome = FixedPipelineBackend(completer).repair(
        task_with(index=corpus_index, check_patch=lambda patch: next(verdicts)),
        BudgetMeter(),
    )
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    assert outcome.budget["candidates_used"] == 2
    assert "unknown skill" in client.received_prompts[1]
    check_events = [e for e in outcome.events if e["step"] == "check"]
    assert len(check_events) == 2


def test_fixed_pipeline_reports_budget_exhaustion(corpus_index):
    completer, _ = completer_with(*[close_proposal_json()] * 5)
    outcome = FixedPipelineBackend(completer).repair(
        task_with(index=corpus_index, check_patch=lambda patch: ["never good"]),
        BudgetMeter(budget=Budget(candidates=2)),
    )
    assert outcome.status is OutcomeStatus.BUDGET_EXHAUSTED
    assert outcome.patch is None
    assert outcome.budget["candidates_used"] == 2


def test_meter_is_shared_state_not_per_backend():
    meter = BudgetMeter(budget=Budget(candidates=2))
    completer, _ = completer_with(close_proposal_json())
    ClosedBookBackend(completer).repair(task_with(), meter)
    completer2, _ = completer_with(close_proposal_json())
    ClosedBookBackend(completer2).repair(task_with(), meter)
    assert meter.candidates_used == 2
    with pytest.raises(BudgetExhaustedError):
        meter.spend_candidate()


def test_token_metering_counts_prompt_and_response():
    meter = BudgetMeter()
    completer, _ = completer_with(close_proposal_json())
    ClosedBookBackend(completer).repair(task_with(), meter)
    assert meter.estimated_tokens_used > 0


def test_malformed_retries_are_metered_and_recorded_as_model_output():
    completer, _ = completer_with("not json", "still not json", "bad again")
    meter = BudgetMeter()
    outcome = ClosedBookBackend(completer).repair(task_with(), meter)
    assert outcome.status is OutcomeStatus.NO_CANDIDATE
    assert outcome.events[-1]["step"] == "invalid-output"
    assert meter.estimated_usage_calls == 3
    assert meter.input_tokens_used > 0 and meter.output_tokens_used > 0


# -- typed enumeration --------------------------------------------------


def test_enumeration_synthesizes_the_close_operator_llm_free():
    """
    The no-LLM baseline: with the curator's static review as the check, enumeration
    finds an achiever for `closed` bound to the covering contract — spending candidates
    through the same meter.
    """
    from resym.repair.backends import EnumerationBackend
    from resym.repair.curator import static_objections

    lib = library()
    task = RepairTask(
        certificate=StubCertificate(),
        library=lib,
        capability_listing="",
        check_patch=lambda patch: static_objections(
            patch,
            lib,
            available_capabilities=frozenset({CAPABILITY_UID}),
        ),
    )
    meter = BudgetMeter(budget=Budget(candidates=50, tool_calls=100))
    outcome = EnumerationBackend().repair(task, meter)
    assert outcome.status is OutcomeStatus.PATCH_PROPOSED
    from resym.core.model import Literal

    (operator,) = outcome.patch.operators
    assert operator.add_effects == (Literal("closed", ("v0",)),)
    assert operator.execution_binding.capability_ref.uid == CAPABILITY_UID
    assert operator.provenance.proposal_backend == "typed-enumeration"
    assert meter.candidates_used >= 1


def test_enumeration_is_deterministic_and_bounded():
    from resym.repair.backends import EnumerationBackend

    lib = library()
    task = RepairTask(
        certificate=StubCertificate(),
        library=lib,
        capability_listing="",
        check_patch=lambda patch: ["never admissible"],
    )
    first = EnumerationBackend().repair(
        task, BudgetMeter(budget=Budget(candidates=1000, tool_calls=5000))
    )
    second = EnumerationBackend().repair(
        task, BudgetMeter(budget=Budget(candidates=1000, tool_calls=5000))
    )
    assert first.status is OutcomeStatus.NO_CANDIDATE  # space exhausted, honest
    assert [e["operators"] for e in first.events] == [
        e["operators"] for e in second.events
    ]


def test_enumeration_respects_the_shared_budget():
    from resym.repair.backends import EnumerationBackend

    lib = library()
    task = RepairTask(
        certificate=StubCertificate(),
        library=lib,
        capability_listing="",
        check_patch=lambda patch: ["no"],
    )
    outcome = EnumerationBackend().repair(
        task, BudgetMeter(budget=Budget(candidates=2))
    )
    assert outcome.status is OutcomeStatus.BUDGET_EXHAUSTED
