"""
Task objects are added by platform extractors, not ObjectUniverse code.
"""

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cup
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from resym.core.model import SymbolType
from resym.platform.cram_objects import (
    CramObjectCatalog,
    SemanticAnnotationObjectExtractor,
    task_object_universe,
)
from resym.platform.universe import ObjectUniverse


class CupWorld:
    def __init__(self, cup):
        self.cup = cup
        self.semantic_annotations = [cup]

    def get_semantic_annotations_by_type(self, annotation_type):
        assert annotation_type in (Cup, SemanticAnnotation)
        return [self.cup]


class StubRobot:
    def __init__(self):
        self.root = Body(name=PrefixedName("task_robot"))


StubRobot.__module__ = "semantic_digital_twin.robots.robot_parts"


def test_non_drawer_task_object_is_discovered_without_universe_changes():
    cup = Cup(root=Body(name=PrefixedName("breakfast_cup")))
    universe = ObjectUniverse.from_world(
        CupWorld(cup),
        robot=None,
        extractors=(SemanticAnnotationObjectExtractor(Cup),),
    )

    grounded = universe["breakfast_cup"]
    assert grounded.semantic_entity is cup
    assert grounded.symbol_type == SymbolType.from_python_type(Cup)


def test_cram_catalog_discovers_object_types_without_domain_extractors():
    cup = Cup(root=Body(name=PrefixedName("breakfast_cup")))
    world = CupWorld(cup)
    robot = StubRobot()

    catalog = CramObjectCatalog.from_world(world, robot)
    universe = catalog.universe()

    assert set(universe.objects) == {"task_robot", "breakfast_cup"}
    assert universe["breakfast_cup"].semantic_entity is cup


def test_cram_catalog_queries_only_the_current_world_domain():
    first = Cup(root=Body(name=PrefixedName("first_world_cup")))
    Cup(root=Body(name=PrefixedName("second_world_cup")))

    universe = CramObjectCatalog.from_world(CupWorld(first), StubRobot()).universe()

    assert "first_world_cup" in universe.objects
    assert "second_world_cup" not in universe.objects


def test_task_object_universe_can_select_annotation_types():
    cup = Cup(root=Body(name=PrefixedName("breakfast_cup")))
    universe = task_object_universe(CupWorld(cup), StubRobot(), (Cup,))

    assert set(universe.objects) == {"task_robot", "breakfast_cup"}
