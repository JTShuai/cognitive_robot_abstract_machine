"""
Task object selection and world-aware evaluation without a live planner.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field, replace

from resym.core.symbols import Literal
from resym.core.symbol_types import SymbolType
from resym.planning.events import ObjectScopeExpansionReason
from resym.planning.object_scope import (
    InvalidObjectRecommendationError,
    ObjectInclusion,
    ObjectInclusionReason,
    ObjectRecommendation,
    ObjectScopeAdvice,
    ObjectScopeAdviceRequest,
    PlanningObjectSelector,
    UnknownGoalObjectError,
)
from resym.planning.selection import Selection
from resym.planning.state_evaluation import ground
from resym.platform.cram_objects import CramObjectCatalog
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Cup,
    Door,
    Handle,
    Drawer,
)
from semantic_digital_twin.world_description.world_entity import Body

from .test_binary_grounding import catalog_with, context_over, predicate
from .test_world_extraction import CupWorld, StubRobot
from semantic_digital_twin.robots import robot_parts
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from ..semantic_digital_twin_test.test_worlds.test_predicates import two_block_world

# %% object scope


@pytest.fixture(autouse=True)
def resolvable_robot_type(monkeypatch):
    """
    Register the shared robot mimic under its declared import reference.
    """
    monkeypatch.setattr(robot_parts, "StubRobot", StubRobot, raising=False)


def body(name: str) -> Body:
    """
    Build a named body without a simulator.
    """
    return Body(name=PrefixedName(name))


def object_predicate(name: str, annotation_type: type):
    """
    Give a test predicate the requested CRAM signature.
    """
    symbol = predicate(name, name)
    return replace(
        symbol, parameter_types=(SymbolType.from_python_type(annotation_type),)
    )


@pytest.fixture()
def objects():
    """
    A target cup plus unrelated cups of exactly the same type.
    """
    cups = [Cup(root=body(f"cup-{index}")) for index in range(12)]
    world = CupWorld(cups[0])
    world.semantic_annotations = cups
    robot = StubRobot()
    return CramObjectCatalog.from_world(world, robot).universe(), robot


def test_initial_scope_does_not_include_all_instances_of_a_relevant_type(objects):
    universe, robot = objects
    selection = Selection(predicates={"done": object_predicate("done", Cup)})
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("cup-0",)),), robot
    )

    scope = selector.initial()

    assert scope.object_names == frozenset({"cup-0", "task_robot"})
    assert scope.inclusion_reasons["cup-0"].reason is ObjectInclusionReason.GOAL


def test_scope_expansion_is_monotonic_and_eventually_covers_candidates(objects):
    universe, robot = objects
    selection = Selection(predicates={"done": object_predicate("done", Cup)})
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("cup-0",)),), robot
    )
    scope = selector.initial()
    sizes = [len(scope.object_names)]

    while (expanded := selector.expand(scope)) is not None:
        assert scope.object_names < expanded.object_names
        scope = expanded
        sizes.append(len(scope.object_names))

    assert scope.object_names == frozenset(universe.objects)
    assert len(sizes) < len(universe.objects)


def test_storage_and_structural_parts_are_kept_without_other_occupants():
    target = Cup(root=body("target"))
    unrelated = Cup(root=body("unrelated"))
    handle = Handle(root=body("handle"))
    door = Door(root=body("door"), handle=handle)
    cabinet = Cabinet(root=body("cabinet"), doors=[door], objects=[target, unrelated])
    world = CupWorld(target)
    world.semantic_annotations = [cabinet]
    robot = StubRobot()
    universe = CramObjectCatalog.from_world(world, robot).universe()
    selection = Selection(
        predicates={
            "done": object_predicate("done", Cup),
            "open": object_predicate("open", Door),
            "usable": object_predicate("usable", Handle),
        }
    )
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("target",)),), robot
    )

    scope = selector.initial()

    assert scope.object_names == frozenset({"target", "door", "handle", "task_robot"})
    assert scope.inclusion_reasons["door"].related_to == "cabinet"


def test_new_scope_reads_changed_storage_relationships():
    target = Cup(root=body("target"))
    door = Door(root=body("door"))
    cabinet = Cabinet(root=body("cabinet"), doors=[door], objects=[])
    world = CupWorld(target)
    world.semantic_annotations = [cabinet, target]
    robot = StubRobot()
    universe = CramObjectCatalog.from_world(world, robot).universe()
    selection = Selection(
        predicates={
            "done": object_predicate("done", Cup),
            "open": object_predicate("open", Door),
        }
    )
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("target",)),), robot
    )

    assert "door" not in selector.initial().object_names
    cabinet.objects.append(target)
    assert "door" in selector.initial().object_names


def test_parent_lookup_does_not_pull_in_sibling_drawers():
    target_handle = Handle(root=body("target-handle"))
    other_handle = Handle(root=body("other-handle"))
    target = Drawer(root=body("target"), handle=target_handle)
    other = Drawer(root=body("other"), handle=other_handle)
    cabinet = Cabinet(root=body("cabinet"), drawers=[target, other])
    world = CupWorld(target)
    world.semantic_annotations = [cabinet]
    robot = StubRobot()
    universe = CramObjectCatalog.from_world(world, robot).universe()
    selection = Selection(
        predicates={
            "done": object_predicate("done", Drawer),
            "usable": object_predicate("usable", Handle),
        }
    )
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("target",)),), robot
    )

    assert selector.initial().object_names == frozenset(
        {"target", "target-handle", "task_robot"}
    )


def test_spatial_support_is_found_without_a_stored_occupant_annotation(two_block_world):
    supporting, supported = two_block_world
    world = supporting._world
    with world.modify_world():
        supported.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=supporting,
                z=1.0,
            )
        )
    target = Cup(root=supported)
    cabinet = Cabinet(root=supporting)
    universe = ObjectUniverse(
        {
            "target": GroundedObject(
                "target", SymbolType.from_python_type(Cup), supported, target
            ),
            "cabinet": GroundedObject(
                "cabinet", SymbolType.from_python_type(Cabinet), supporting, cabinet
            ),
        }
    )
    selection = Selection(
        predicates={
            "done": object_predicate("done", Cup),
            "accessible": object_predicate("accessible", Cabinet),
        }
    )
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("target",)),), None
    )

    assert selector.initial().object_names == frozenset({"target", "cabinet"})

    with world.modify_world():
        supported.parent_connection.parent_T_connection_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                reference_frame=supporting,
                x=10.0,
                z=1.0,
            )
        )
    assert selector.initial().object_names == frozenset({"target"})


# %% complete world access


def test_restricted_enumeration_keeps_full_world_visible_to_factory(objects):
    universe, _ = objects
    target = universe["cup-0"]
    observed = []

    def has_other_objects(context, world_objects, arguments, parameters):
        observed.append(world_objects)
        return len(world_objects.of_type(target.symbol_type)) > 1

    context = context_over(catalog_with(has_other_objects=has_other_objects))
    symbol = object_predicate("has_other_objects", Cup)
    selection = Selection(predicates={symbol.name: symbol})
    result = ground(
        selection,
        universe,
        context,
        active_universe=ObjectUniverse({target.name: target}),
    )

    assert observed == [universe]
    assert result.evaluation_count == 1
    assert result.true_atoms == {Literal(symbol.name, (target.name,))}


def test_unselected_atoms_are_not_reported_as_false(objects):
    universe, _ = objects
    symbol = object_predicate("ready", Cup)
    context = context_over(catalog_with(ready=lambda *arguments: True))
    result = ground(
        Selection(predicates={symbol.name: symbol}),
        universe,
        context,
        active_universe=ObjectUniverse({"cup-0": universe["cup-0"]}),
    )

    with pytest.raises(KeyError):
        result.value_of(Literal(symbol.name, ("cup-1",)))


# %% advised selection


@dataclass
class ScriptedAdvisor:
    """
    Returns fixed recommendations and records every request it received.
    """

    recommendations: list[tuple[ObjectRecommendation, ...]]
    requests: list[ObjectScopeAdviceRequest] = field(default_factory=list)

    def advise(self, request: ObjectScopeAdviceRequest) -> ObjectScopeAdvice:
        self.requests.append(request)
        index = min(len(self.requests), len(self.recommendations)) - 1
        return ObjectScopeAdvice(
            recommendations=self.recommendations[index],
            trace=({"consultation": len(self.requests)},),
        )


def cup_selector(objects, advisor):
    universe, robot = objects
    selection = Selection(predicates={"done": object_predicate("done", Cup)})
    return PlanningObjectSelector(
        universe, selection, (Literal("done", ("cup-0",)),), robot, advisor=advisor
    )


def test_advised_objects_join_the_initial_scope_with_their_rationale(objects):
    advisor = ScriptedAdvisor(
        [(ObjectRecommendation("cup-7", "the instruction mentions the seventh cup"),)]
    )
    selector = cup_selector(objects, advisor)

    scope = selector.initial()

    assert scope.object_names == frozenset({"cup-0", "cup-7", "task_robot"})
    assert scope.inclusion_reasons["cup-7"] == ObjectInclusion(
        ObjectInclusionReason.ADVISED,
        rationale="the instruction mentions the seventh cup",
    )
    assert scope.advice == (
        ObjectScopeAdvice(
            recommendations=(
                ObjectRecommendation(
                    "cup-7", "the instruction mentions the seventh cup"
                ),
            ),
            trace=({"consultation": 1},),
        ),
    )
    (request,) = advisor.requests
    assert request.expansion_reason is None
    assert request.scope.object_names == frozenset({"cup-0", "task_robot"})
    assert request.eligible_names == frozenset(f"cup-{index}" for index in range(12))


def test_advice_cannot_name_an_object_outside_the_typed_bound(objects):
    advisor = ScriptedAdvisor([(ObjectRecommendation("task_robot", "actor"),)])
    universe, robot = objects
    selection = Selection(predicates={"done": object_predicate("done", Cup)})
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("cup-0",)),), None, advisor=advisor
    )

    with pytest.raises(InvalidObjectRecommendationError) as error:
        selector.initial()

    assert error.value.name == "task_robot"


def test_advice_cannot_name_an_unknown_object(objects):
    advisor = ScriptedAdvisor([(ObjectRecommendation("cup-99", "guess"),)])

    with pytest.raises(InvalidObjectRecommendationError) as error:
        cup_selector(objects, advisor).initial()

    assert error.value.name == "cup-99"


def test_expansion_takes_advice_before_typed_growth(objects):
    advisor = ScriptedAdvisor(
        [(), (ObjectRecommendation("cup-5", "the planner lacked a second cup"),)]
    )
    selector = cup_selector(objects, advisor)
    scope = selector.initial()

    expanded = selector.expand(
        scope, ObjectScopeExpansionReason.UNSOLVABLE_SUBSET, "no plan"
    )

    assert expanded.object_names == frozenset({"cup-0", "cup-5", "task_robot"})
    assert expanded.inclusion_reasons["cup-5"].reason is ObjectInclusionReason.ADVISED
    request = advisor.requests[1]
    assert request.expansion_reason is ObjectScopeExpansionReason.UNSOLVABLE_SUBSET
    assert request.planner_message == "no plan"
    assert len(expanded.advice) == 2


def test_expansion_falls_back_to_typed_growth_when_advice_adds_nothing(objects):
    advisor = ScriptedAdvisor([()])
    selector = cup_selector(objects, advisor)
    scope = selector.initial()

    expanded = selector.expand(scope, ObjectScopeExpansionReason.UNSOLVABLE_SUBSET)

    assert expanded.object_names == frozenset({"cup-0", "cup-1", "task_robot"})
    assert expanded.inclusion_reasons["cup-1"].reason is ObjectInclusionReason.EXPANSION


def test_goal_objects_stay_regardless_of_advice(objects):
    advisor = ScriptedAdvisor([()])
    scope = cup_selector(objects, advisor).initial()

    assert scope.inclusion_reasons["cup-0"].reason is ObjectInclusionReason.GOAL


def test_goal_naming_an_unknown_object_is_rejected(objects):
    universe, robot = objects
    selection = Selection(predicates={"done": object_predicate("done", Cup)})
    selector = PlanningObjectSelector(
        universe, selection, (Literal("done", ("cup-99",)),), robot
    )

    with pytest.raises(UnknownGoalObjectError):
        selector.initial()
