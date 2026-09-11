"""
Composable CRAM object discovery for the symbolic universe.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Callable, Iterable

from resym.core.model import SymbolType
from resym.platform.universe import (
    GroundedObject,
    ObjectUniverse,
    pddl_name,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot, AbstractRobotPart
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)


@dataclass(frozen=True)
class RobotObjectExtractor:
    """
    Expose the active robot as one typed planning object.
    """

    def extract(self, world, robot: AbstractRobot) -> Iterable[GroundedObject]:
        yield GroundedObject(
            name=pddl_name(robot.root.name.name),
            symbol_type=SymbolType.from_python_type(type(robot)),
            body=robot.root,
            semantic_entity=robot,
        )


@dataclass(frozen=True)
class SemanticAnnotationObjectExtractor:
    """
    Expose one CRAM annotation type and selected related annotations.

    This is the extension point for new task objects. For example, a door package can
    request ``Door`` plus its ``handle`` while a pick/place package can request ``Cup``
    without changing ``ObjectUniverse``.
    """

    annotation_type: type
    related_entities: tuple[Callable[[object], object | None], ...] = ()

    def extract(self, world, robot: AbstractRobot) -> Iterable[GroundedObject]:
        for annotation in world.get_semantic_annotations_by_type(self.annotation_type):
            yield _ground(annotation)
            for get_related in self.related_entities:
                related = get_related(annotation)
                if related is not None:
                    yield _ground(related)


def _ground(annotation: object) -> GroundedObject:
    body = annotation.root
    return GroundedObject(
        name=pddl_name(body.name.name),
        symbol_type=SymbolType.from_python_type(type(annotation)),
        body=body,
        semantic_entity=annotation,
    )


@dataclass(frozen=True)
class CramObjectCatalog:
    """
    All object-like semantic annotations currently exposed by CRAM.

    Discovery is domain-independent.  A caller may select annotation types when
    constructing the active task universe; no drawer-specific extraction rule is needed
    for a new semantic-annotation class.
    """

    robot: GroundedObject
    annotations: tuple[GroundedObject, ...]

    @classmethod
    def from_world(cls, world, robot: AbstractRobot) -> "CramObjectCatalog":
        # The CRAM adapter owns object discovery; predicate queries consume the
        # resulting object identities instead of maintaining a second world model.
        robot_object = next(iter(RobotObjectExtractor().extract(world, robot)))
        # The world's own annotation list gives the world-restricted seed set; some
        # annotations (a drawer's handle, for example) are reachable only through
        # references from a seed annotation, so the closure is still walked.
        grounded: list[GroundedObject] = []
        seen_annotations: set[int] = set()
        pending = list(world.semantic_annotations)
        while pending:
            annotation = pending.pop(0)
            if id(annotation) in seen_annotations:
                continue
            seen_annotations.add(id(annotation))
            if isinstance(annotation, AbstractRobot):
                continue
            if isinstance(annotation, HasRootKinematicStructureEntity):
                grounded.append(_ground(annotation))
            # The reference walk has no public accessor; the semantic digital
            # twin's own World traverses annotations through this method too.
            pending.extend(annotation._referenced_semantic_annotations())

        grounded.sort(key=lambda item: (item.name, item.symbol_type.python_type_ref))
        return cls(robot=robot_object, annotations=tuple(grounded))

    def universe(
        self,
        annotation_types: Iterable[type] | None = None,
    ) -> ObjectUniverse:
        """
        Build a task universe from selected CRAM annotation types.

        With no selection, scene annotations are exposed so natural-language grounding
        can resolve references before task-relevant symbolic types are known. Robot
        internals remain behind the single robot object unless their annotation type is
        requested explicitly.
        """
        selected_types = tuple(annotation_types or ())
        universe = ObjectUniverse()
        universe.add(self.robot)
        for grounded in self.annotations:
            if not selected_types and isinstance(
                grounded.semantic_entity, AbstractRobotPart
            ):
                continue
            if selected_types and not isinstance(
                grounded.semantic_entity, selected_types
            ):
                continue
            universe.add(grounded)
        return universe


def task_object_universe(
    world,
    robot: AbstractRobot,
    annotation_types: Iterable[type] | None = None,
) -> ObjectUniverse:
    """
    Discover CRAM objects and construct the active task universe.
    """
    return CramObjectCatalog.from_world(world, robot).universe(annotation_types)
