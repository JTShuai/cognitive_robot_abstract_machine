"""
The language-model seam: scripted client, transcripts, structured output.

Everything here runs offline; the real llm-agent-kit adapter is exercised only where the
package is importable (it needs ``openai``, absent in the cram container).
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel


from resym.llm.client import (
    ScriptedCompletionClient,
    ScriptExhaustedError,
    UsageSource,
)
from resym.llm.structured import (
    MalformedResponseError,
    StructuredCompleter,
    StructuredOutputRetriesExceededError,
    extract_json_block,
)
from resym.llm.transcript import TranscriptRecorder

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Agent,
    Drawer,
    Handle,
)

AGENT_TYPE = SymbolType.from_python_type(Agent)
ARTICULATED_PART_TYPE = SymbolType.from_python_type(HasMechanicalJoint)
DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


class Answer(BaseModel):
    """
    A minimal structured reply for exercising the completer.
    """

    value: int


class TestScriptedClient:
    def test_replays_responses_and_records_prompts(self):
        client = ScriptedCompletionClient(responses=["first", "second"])
        assert client.complete("prompt one") == "first"
        assert client.complete("prompt two") == "second"
        assert client.received_prompts == ["prompt one", "prompt two"]

    def test_exhaustion_raises(self):
        client = ScriptedCompletionClient(responses=[])
        with pytest.raises(ScriptExhaustedError):
            client.complete("anything")


class TestJsonExtraction:
    def test_plain_object(self):
        assert extract_json_block('{"value": 1}') == '{"value": 1}'

    def test_fenced_with_prose(self):
        text = 'Sure! Here is the JSON:\n```json\n{"value": 2}\n```\nDone.'
        assert extract_json_block(text) == '{"value": 2}'

    def test_nested_braces_and_strings(self):
        text = 'reply {"a": {"b": "}"}, "c": [1, 2]} trailing'
        assert extract_json_block(text) == '{"a": {"b": "}"}, "c": [1, 2]}'

    def test_skips_balanced_non_json_braces_in_prose(self):
        text = 'Allowed values are {OPEN, CLOSED}. Use {"value": 2}.'
        assert extract_json_block(text) == '{"value": 2}'

    def test_recovers_one_missing_outer_object_close(self):
        text = '{"tool": "inspect_library", "arguments": {}'
        assert extract_json_block(text) == (
            '{"tool": "inspect_library", "arguments": {}}'
        )

    def test_recovers_one_missing_outer_array_close(self):
        assert extract_json_block('[{"value": 1}') == '[{"value": 1}]'

    @pytest.mark.parametrize(
        "text",
        [
            '{"value": 1',
            '{"value": "unfinished',
            '{"value": [1, 2}',
            '{"value": [{"nested": 1}',
        ],
    )
    def test_does_not_guess_ambiguous_or_malformed_content(self, text):
        with pytest.raises(MalformedResponseError):
            extract_json_block(text)

    def test_no_json_raises(self):
        with pytest.raises(MalformedResponseError):
            extract_json_block("no json here")


class TestStructuredCompleter:
    def test_valid_first_attempt(self):
        completer = StructuredCompleter(
            client=ScriptedCompletionClient(responses=['{"value": 7}']),
            transcript=TranscriptRecorder(),
        )
        answer = completer.complete("agent", "give a value", Answer)
        assert answer.value == 7
        assert completer.transcript.exchanges[0].parse_error is None

    def test_validation_error_is_fed_back(self):
        client = ScriptedCompletionClient(
            responses=['{"value": "not a number"}', '{"value": 3}']
        )
        completer = StructuredCompleter(client=client, transcript=TranscriptRecorder())
        answer = completer.complete("agent", "give a value", Answer)
        assert answer.value == 3
        assert "could not be used" in client.received_prompts[1]
        assert completer.transcript.exchanges[0].parse_error is not None
        assert completer.transcript.exchanges[1].parse_error is None

    def test_safe_recovery_is_audited_without_a_retry(self):
        client = ScriptedCompletionClient(responses=['{"value": {"nested": 7}'])

        class NestedAnswer(BaseModel):
            value: dict[str, int]

        transcript = TranscriptRecorder()
        answer = StructuredCompleter(client=client, transcript=transcript).complete(
            "agent", "give a value", NestedAnswer
        )
        assert answer.value == {"nested": 7}
        assert len(client.received_prompts) == 1
        assert transcript.exchanges[0].parse_error is None
        assert transcript.exchanges[0].parse_recovery == "closed_one_trailing_container"

    def test_retries_exhausted(self):
        completer = StructuredCompleter(
            client=ScriptedCompletionClient(responses=["nonsense"] * 3),
            transcript=TranscriptRecorder(),
            maximum_attempts=3,
        )
        with pytest.raises(StructuredOutputRetriesExceededError):
            completer.complete("agent", "give a value", Answer)

    def test_every_format_retry_reports_its_full_usage(self):
        client = ScriptedCompletionClient(responses=["not-json", '{"value": 3}'])
        completer = StructuredCompleter(client=client, transcript=TranscriptRecorder())
        charged = []
        completer.complete("agent", "give a value", Answer, charge_usage=charged.append)
        assert len(charged) == 2
        assert sum(usage.total_tokens for usage in charged) == sum(
            exchange.input_tokens + exchange.output_tokens
            for exchange in completer.transcript.exchanges
        )
        assert {usage.source for usage in charged} == {UsageSource.ESTIMATED}


class TestTranscriptFile:
    def test_exchanges_append_as_json_lines(self, tmp_path):
        path = tmp_path / "transcript.jsonl"
        completer = StructuredCompleter(
            client=ScriptedCompletionClient(responses=['{"value": 1}']),
            transcript=TranscriptRecorder(path=path),
        )
        completer.complete("agent", "prompt", Answer)
        lines = path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        assert entry["agent_name"] == "agent"
        assert entry["model"] == "scripted"
        assert entry["parse_error"] is None
        assert entry["parse_recovery"] is None
        assert entry["input_tokens"] > 0
        assert entry["output_tokens"] > 0

    def test_responses_of_supports_replay(self):
        transcript = TranscriptRecorder()
        completer = StructuredCompleter(
            client=ScriptedCompletionClient(responses=['{"value": 5}']),
            transcript=transcript,
        )
        completer.complete("agent", "prompt", Answer)
        replay_client = ScriptedCompletionClient(
            responses=transcript.responses_of("agent")
        )
        replayed = StructuredCompleter(
            client=replay_client, transcript=TranscriptRecorder()
        ).complete("agent", "prompt", Answer)
        assert replayed.value == 5


class TestProposalSchemas:
    def test_symbol_lists_default_to_empty(self):
        from resym.llm.schemas import LibraryProposal

        proposal = LibraryProposal.model_validate({"rationale": "operator only"})
        assert proposal.predicates == [] and proposal.operators == []

    def test_cram_type_reference_crosses_the_model_boundary(self):
        from resym.llm.schemas import PredicateProposal

        proposal = PredicateProposal.model_validate(
            {
                "name": "held",
                "parameter_types": [HANDLE_TYPE.python_type_ref],
                "fluent": True,
                "grounding_plan": {
                    "factory_uid": "resym:grounding/joint-fraction-opened",
                    "approved_factory_checksum": "reviewed-checksum",
                    "role_bindings": {"articulated_object": 0},
                },
            }
        )
        assert proposal.to_predicate_symbol().parameter_types == (HANDLE_TYPE,)

    def test_predicate_rejects_an_inline_grounding_program(self):
        from pydantic import ValidationError

        from resym.llm.schemas import PredicateProposal

        with pytest.raises(ValidationError, match="program"):
            PredicateProposal.model_validate(
                {
                    "name": "near",
                    "parameter_types": [DRAWER_TYPE.python_type_ref],
                    "fluent": True,
                    "grounding_plan": {
                        "factory_uid": "resym:grounding/joint-fraction-opened",
                        "approved_factory_checksum": "reviewed-checksum",
                        "role_bindings": {"articulated_object": 0},
                    },
                    "program": {},
                }
            )

    def test_predicate_requires_a_reviewed_query(self):
        from pydantic import ValidationError

        from resym.llm.schemas import PredicateProposal

        with pytest.raises(ValidationError, match="grounding_plan"):
            PredicateProposal.model_validate(
                {
                    "name": "new-predicate",
                    "parameter_types": [DRAWER_TYPE.python_type_ref],
                    "fluent": True,
                }
            )

    def test_malformed_operator_parameter_type_is_a_validation_error(self):
        from pydantic import ValidationError

        from resym.llm.schemas import ParameterModel

        with pytest.raises(ValidationError, match="CRAM Python type reference"):
            ParameterModel.model_validate({"variable": "a", "type": "Bad Type"})

    def test_object_query_color_must_be_a_named_cram_color(self):
        from pydantic import ValidationError

        from resym.llm.schemas import ObjectQueryModel

        with pytest.raises(ValidationError, match="color must be one of"):
            ObjectQueryModel.model_validate({"reference": "$target", "color": "purple"})

    def test_object_query_color_is_normalized_to_the_named_color_key(self):
        from resym.llm.schemas import ObjectQueryModel
        from resym.platform.krrood_queries import named_colors

        model = ObjectQueryModel.model_validate(
            {"reference": "$target", "color": "Red"}
        )

        assert model.color in named_colors()

    def test_partial_operator_update_preserves_omitted_fields(self, fixed_arm_library):
        from resym.llm.schemas import OperatorProposal

        library = fixed_arm_library
        original = library.operators["open-drawer"]
        proposal = OperatorProposal.model_validate(
            {
                "name": "open-drawer",
                "add_effects": [
                    {
                        "predicate": "opened",
                        "arguments": ["d"],
                        "negated": False,
                    }
                ],
                "delete_effects": [
                    {
                        "predicate": "closed",
                        "arguments": ["d"],
                        "negated": False,
                    }
                ],
            }
        )

        updated = proposal.to_operator(original)

        assert updated.parameters == original.parameters
        assert updated.preconditions == original.preconditions
        assert updated.execution_binding == original.execution_binding
        assert updated.add_effects[0].predicate == "opened"

    def test_parameter_type_edit_preserves_the_rest_of_the_signature(
        self, fixed_arm_library
    ):
        from resym.llm.schemas import OperatorProposal

        original = fixed_arm_library.operators["open-drawer"]
        proposal = OperatorProposal.model_validate(
            {
                "name": "open-drawer",
                "parameter_type_edits": {
                    "h": HANDLE_TYPE.python_type_ref,
                },
            }
        )

        updated = proposal.to_operator(original)

        assert updated.parameters == original.parameters
        assert dict(updated.parameters)["h"] == HANDLE_TYPE

    def test_parameter_type_edit_rejects_an_unknown_parameter(self, fixed_arm_library):
        from resym.llm.schemas import OperatorProposal

        original = fixed_arm_library.operators["open-drawer"]
        proposal = OperatorProposal.model_validate(
            {
                "name": "open-drawer",
                "parameter_type_edits": {
                    "missing": HANDLE_TYPE.python_type_ref,
                },
            }
        )

        with pytest.raises(ValueError, match="unknown parameters: missing"):
            proposal.to_operator(original)

    def test_literal_edits_change_one_precondition_without_replacing_the_list(
        self, fixed_arm_library
    ):
        from resym.core.model import Literal
        from resym.llm.schemas import OperatorProposal

        original = fixed_arm_library.operators["open-drawer"]
        wrong = Literal("opened", ("d",))
        corrupted = original.__class__(
            name=original.name,
            parameters=original.parameters,
            preconditions=tuple(
                wrong if literal.predicate == "closed" else literal
                for literal in original.preconditions
            ),
            add_effects=original.add_effects,
            delete_effects=original.delete_effects,
            execution_binding=original.execution_binding,
            provenance=original.provenance,
        )
        proposal = OperatorProposal.model_validate(
            {
                "name": "open-drawer",
                "precondition_edits": {
                    "remove": [
                        {
                            "predicate": "opened",
                            "arguments": ["d"],
                            "negated": False,
                        }
                    ],
                    "add": [
                        {
                            "predicate": "closed",
                            "arguments": ["d"],
                            "negated": False,
                        }
                    ],
                },
            }
        )

        updated = proposal.to_operator(corrupted)

        assert Literal("ready-to-open", ("r", "d")) in updated.preconditions
        assert Literal("handle-of", ("h", "d")) in updated.preconditions
        assert Literal("closed", ("d",)) in updated.preconditions
        assert wrong not in updated.preconditions

    def test_literal_edits_cannot_be_combined_with_field_replacement(self):
        from pydantic import ValidationError

        from resym.llm.schemas import OperatorProposal

        with pytest.raises(ValidationError, match="cannot be combined"):
            OperatorProposal.model_validate(
                {
                    "name": "open-drawer",
                    "preconditions": [],
                    "precondition_edits": {"add": [], "remove": []},
                }
            )

    def test_literal_edits_reject_removal_of_an_absent_literal(self, fixed_arm_library):
        from resym.llm.schemas import OperatorProposal

        original = fixed_arm_library.operators["open-drawer"]
        proposal = OperatorProposal.model_validate(
            {
                "name": "open-drawer",
                "precondition_edits": {
                    "remove": [
                        {
                            "predicate": "misspelled-condition",
                            "arguments": ["d"],
                        }
                    ]
                },
            }
        )

        with pytest.raises(ValueError, match="not present"):
            proposal.to_operator(original)

    def test_partial_definition_cannot_create_a_new_operator(self):
        from resym.llm.schemas import OperatorProposal

        proposal = OperatorProposal.model_validate(
            {"name": "new-operator", "add_effects": []}
        )

        with pytest.raises(ValueError, match="requires complete fields"):
            proposal.to_operator()

    def test_agent_action_rejects_extra_envelope_fields(self):
        from pydantic import ValidationError

        from resym.llm.schemas import AgentActionModel

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AgentActionModel.model_validate(
                {"tool": "inspect_library", "arguments": {}, "comment": "run it"}
            )


class TestPromptListings:
    def test_contracts_and_existing_bindings_are_fully_visible(self, fixed_arm_library):
        from resym.llm.prompting import (
            render_capability_contracts,
            render_operators,
        )

        library = fixed_arm_library
        contracts = render_capability_contracts(library)
        operators = render_operators(library)

        assert AGENT_TYPE.python_type_ref in contracts
        assert "interaction_point: parameter types" in contracts
        assert HANDLE_TYPE.python_type_ref in contracts
        assert "optional" in contracts
        assert "target_state: constants {OPEN, CLOSED}, required" in contracts
        assert "capability_uid=resym:ArticulationStateChange" in contracts
        assert "capability_version=1" in contracts
        assert "actor <- ?r" in operators
        assert "interaction_point <- ?h" in operators
        assert "target_state <- 'OPEN'" in operators
        assert "resym:ArticulationStateChange@1" not in contracts + operators

    def test_type_listing_includes_subtype_relations(self, fixed_arm_library):
        from resym.llm.prompting import render_symbol_types

        listing = render_symbol_types(fixed_arm_library.symbol_types)

        assert (
            f"{ROBOT_TYPE.python_type_ref} <: " f"{AGENT_TYPE.python_type_ref}"
        ) in listing
        assert (
            f"{DRAWER_TYPE.python_type_ref} <: "
            f"{ARTICULATED_PART_TYPE.python_type_ref}"
        ) in listing


class TestExperimentConfiguration:
    def test_loads_from_json_file(self, tmp_path):
        from resym.llm.configuration import ExperimentConfiguration

        path = tmp_path / "llm.json"
        path.write_text(
            json.dumps(
                {
                    "agent": {"model": "test-model", "kwargs": {"temperature": 0.5}},
                    "tasks": {"call_max_retries": 4},
                    "structured_maximum_attempts": 5,
                }
            )
        )
        configuration = ExperimentConfiguration.load(path)
        assert configuration.agent["model"] == "test-model"
        assert configuration.tasks == {"call_max_retries": 4}
        assert configuration.structured_maximum_attempts == 5
        assert configuration.to_metadata() == {
            "agent": {"model": "test-model", "kwargs": {"temperature": 0.5}},
            "tasks": {"call_max_retries": 4},
            "structured_maximum_attempts": 5,
        }


class TestRealAdapter:
    def test_adapter_builds_from_llm_agent_kit_schema(self):
        """
        The adapter validates the raw sections with llm-agent-kit's own Pydantic models
        and constructs an LLMAgent; no call is made.
        """
        pytest.importorskip("llm_agent_kit")
        import os

        os.environ.setdefault("API_KEY", "test-key-never-used")
        from resym.llm.configuration import (
            ExperimentConfiguration,
            build_completion_client,
        )

        client = build_completion_client(
            ExperimentConfiguration(
                agent={"model": "test-model"}, tasks={"call_max_retries": 3}
            )
        )
        assert client.description == "llm-agent-kit:test-model"
