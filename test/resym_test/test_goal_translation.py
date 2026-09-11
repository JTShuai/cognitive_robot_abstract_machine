"""
Stage C1: instructions become validated goals; bad translations are repaired.

The universe here is a hand-built miniature (no world needed): goal validation only
consults object names and types.
"""

from __future__ import annotations

import json

import pytest

from .library_fixtures import build_seed_library
from resym.interfaces.goal_translation import (
    GoalTranslator,
    TaskUnderstandingStatus,
    UntranslatableGoalError,
    diagnose_instruction,
)
from resym.knowledge.retrieval import RetrievalQuery
from resym.llm.client import ScriptedCompletionClient
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.core.model import PredicateSymbol, SymbolType
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup
from semantic_digital_twin.world_description.geometry import Box, Color
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


@pytest.fixture()
def miniature_universe():
    universe = ObjectUniverse()
    for name, symbol_type in (
        ("pr2", ROBOT_TYPE),
        ("cabinet10-drawer-top", DRAWER_TYPE),
        ("handle-cab10-t", HANDLE_TYPE),
    ):
        universe.add(
            GroundedObject(
                name=name,
                symbol_type=symbol_type,
                body=Body(name=PrefixedName(name.replace("-", "_"))),
            )
        )
    return universe


def translator_with(*responses: str) -> GoalTranslator:
    return GoalTranslator(
        completer=StructuredCompleter(
            client=ScriptedCompletionClient(responses=list(responses)),
            transcript=TranscriptRecorder(),
        )
    )


def goal_json(predicate: str, *arguments: str) -> str:
    return json.dumps(
        {
            "literals": [
                {"predicate": predicate, "arguments": list(arguments), "negated": False}
            ]
        }
    )


def model_gap_json(predicate: str, description: str, *arguments: str) -> str:
    return json.dumps(
        {
            "status": "model_gap",
            "literals": [],
            "unresolved_literals": [
                {
                    "suggested_predicate": predicate,
                    "arguments": list(arguments),
                    "description": description,
                    "negated": False,
                }
            ],
            "message": "",
        }
    )


def cup(name: str, color: Color | None = None) -> GroundedObject:
    shapes = (
        ShapeCollection([Box(color=color)]) if color is not None else ShapeCollection()
    )
    body = Body(name=PrefixedName(name.replace("-", "_")), visual=shapes)
    annotation = Cup(root=body)
    return GroundedObject(
        name=name,
        symbol_type=SymbolType.from_python_type(Cup),
        body=body,
        semantic_entity=annotation,
    )


def query_goal_json(reference: str = "$target", **filters: str) -> str:
    cup_type = SymbolType.from_python_type(Cup).python_type_ref
    return json.dumps(
        {
            "status": "ready",
            "object_queries": [{"reference": reference, "type": cup_type, **filters}],
            "literals": [
                {
                    "predicate": "selected",
                    "arguments": [reference],
                    "negated": False,
                }
            ],
            "unresolved_literals": [],
            "message": "",
        }
    )


class TestGoalTranslation:
    def test_eql_object_query_resolves_a_property_description(self):
        universe = ObjectUniverse()
        universe.add(cup("red-cup"))
        universe.add(cup("blue-cup"))
        library = build_seed_library()
        library.add(
            PredicateSymbol(
                name="selected",
                parameter_types=(SymbolType.from_python_type(Cup),),
                evaluator="test-only",
                fluent=False,
            )
        )

        goal = translator_with(query_goal_json(name_contains="red")).translate(
            "select the red cup", library, universe
        )

        assert goal[0].arguments == ("red-cup",)

    def test_color_query_classifies_asset_colors_to_the_nearest_name(self):
        universe = ObjectUniverse()
        universe.add(cup("cup-one", Color(0.9, 0.05, 0.1)))
        universe.add(cup("cup-two", Color.BLUE()))
        library = build_seed_library()
        library.add(
            PredicateSymbol(
                name="selected",
                parameter_types=(SymbolType.from_python_type(Cup),),
                evaluator="test-only",
                fluent=False,
            )
        )

        goal = translator_with(query_goal_json(color="red")).translate(
            "select the red cup", library, universe
        )

        assert goal[0].arguments == ("cup-one",)

    def test_ambiguous_eql_object_query_requests_a_revised_interpretation(self):
        universe = ObjectUniverse()
        universe.add(cup("red-cup-one"))
        universe.add(cup("red-cup-two"))
        library = build_seed_library()
        library.add(
            PredicateSymbol(
                name="selected",
                parameter_types=(SymbolType.from_python_type(Cup),),
                evaluator="test-only",
                fluent=False,
            )
        )
        clarification = json.dumps(
            {
                "status": "clarification_needed",
                "object_queries": [],
                "literals": [],
                "unresolved_literals": [],
                "message": "Which red cup should I use?",
            }
        )
        translator = translator_with(
            query_goal_json(name_contains="red"), clarification
        )

        understanding = translator.interpret("select the red cup", library, universe)

        assert understanding.status is TaskUnderstandingStatus.CLARIFICATION_NEEDED
        assert "red-cup-one" in translator.completer.client.received_prompts[1]

    def test_valid_goal_passes(self, miniature_universe):
        translator = translator_with(goal_json("opened", "cabinet10-drawer-top"))
        goal = translator.translate(
            "open the top drawer of cabinet 10",
            build_seed_library(),
            miniature_universe,
        )
        assert goal[0].predicate == "opened"
        assert goal[0].arguments == ("cabinet10-drawer-top",)

    def test_semantic_error_is_repaired(self, miniature_universe):
        translator = translator_with(
            goal_json("opened", "no-such-drawer"),
            goal_json("opened", "cabinet10-drawer-top"),
        )
        goal = translator.translate(
            "open the drawer", build_seed_library(), miniature_universe
        )
        assert goal[0].arguments == ("cabinet10-drawer-top",)
        client = translator.completer.client
        assert "no-such-drawer" in client.received_prompts[1]

    def test_wrong_argument_type_is_reported(self, miniature_universe):
        translator = translator_with(
            goal_json("opened", "handle-cab10-t"),
            goal_json("opened", "cabinet10-drawer-top"),
        )
        goal = translator.translate("open it", build_seed_library(), miniature_universe)
        assert goal[0].arguments == ("cabinet10-drawer-top",)

    def test_unrepairable_goal_raises(self, miniature_universe):
        translator = translator_with(*[goal_json("polish", "cabinet10-drawer-top")] * 3)
        with pytest.raises(UntranslatableGoalError):
            translator.translate(
                "polish the drawer", build_seed_library(), miniature_universe
            )

    def test_clear_missing_relation_becomes_curation_certificate(
        self, miniature_universe, tmp_path
    ):
        translator = translator_with(
            model_gap_json(
                "polished",
                "the drawer surface has been polished",
                "cabinet10-drawer-top",
            )
        )

        outcome = diagnose_instruction(
            translator,
            "polish the top drawer",
            build_seed_library(),
            miniature_universe,
            context=None,
            working_directory=tmp_path,
        )

        assert outcome.understanding.status is TaskUnderstandingStatus.MODEL_GAP
        assert outcome.triggers_curation
        certificate = outcome.diagnosis.certificate
        assert certificate.task_instruction == "polish the top drawer"
        assert certificate.task_goal[0].predicate == "polished"
        assert certificate.task_object_types == (
            (
                "cabinet10-drawer-top",
                miniature_universe["cabinet10-drawer-top"].symbol_type.python_type_ref,
            ),
        )
        query = RetrievalQuery.from_certificate(certificate)
        assert "polish the top drawer" in query.text
        assert "drawer surface has been polished" in query.text

    def test_ambiguous_reference_requests_clarification(self, miniature_universe):
        translator = translator_with(
            json.dumps(
                {
                    "status": "clarification_needed",
                    "literals": [],
                    "unresolved_literals": [],
                    "message": "Which drawer should I open?",
                }
            )
        )

        understanding = translator.interpret(
            "open it", build_seed_library(), miniature_universe
        )

        assert understanding.status is TaskUnderstandingStatus.CLARIFICATION_NEEDED
        assert understanding.message == "Which drawer should I open?"
        assert understanding.goal == ()

    def test_existing_predicate_cannot_be_reported_as_model_gap(
        self, miniature_universe
    ):
        translator = translator_with(
            model_gap_json("opened", "the drawer is open", "cabinet10-drawer-top"),
            goal_json("opened", "cabinet10-drawer-top"),
        )

        understanding = translator.interpret(
            "open the top drawer", build_seed_library(), miniature_universe
        )

        assert understanding.status is TaskUnderstandingStatus.READY
        assert "already exists" in translator.completer.client.received_prompts[1]
