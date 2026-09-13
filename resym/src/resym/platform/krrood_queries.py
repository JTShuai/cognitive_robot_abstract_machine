"""
Typed krrood/EQL access to one Semantic Digital Twin world.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from collections import deque

from krrood.entity_query_language.factories import entity, variable
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.semantic_annotations.mixins import (
    IsStorageSpace,
    HasSupportingSurface,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.reasoning.predicates import InsideOf, is_supported_by
from typing_extensions import Iterable, Mapping

from resym.core.symbol_types import SymbolType, resolve_symbol_type
from resym.platform.universe import GroundedObject, ObjectUniverse


@dataclass(frozen=True)
class ObjectQuerySpec:
    """
    A typed, serializable object description chosen by the task agent.
    """

    type_ref: str | None = None
    color: str | None = None
    name_contains: str | None = None


@dataclass(frozen=True)
class ObjectDependency:
    """
    A native annotation required to access a related object.
    """

    object_name: str
    """
    Name of the discovered object.
    """

    related_to: str
    """
    Object from which this dependency was reached.
    """


@cache
def named_colors() -> Mapping[str, Color]:
    """
    The named color constants ``semantic_digital_twin`` defines, keyed by their lower-
    case names.
    """
    return {
        name.lower(): getattr(Color, name)()
        for name in dir(Color)
        if name.isupper() and len(name) > 1 and callable(getattr(Color, name))
    }


def _color_distance(color: Color, reference: Color) -> float:
    return (
        (color.R - reference.R) ** 2
        + (color.G - reference.G) ** 2
        + (color.B - reference.B) ** 2
    )


def nearest_color_name(color: Color) -> str:
    """
    The named color a concrete RGBA value is closest to.

    Real assets rarely carry the exact named constants, so color matching classifies to
    the nearest name instead of testing equality.
    """
    colors = named_colors()
    return min(colors, key=lambda name: _color_distance(color, colors[name]))


def _body_colors(body: Body) -> Iterable[Color]:
    for collection in (body.visual, body.collision):
        for shape in collection.shapes:
            yield shape.color


def _color_matches(body: Body, color_name: str) -> bool:
    return any(nearest_color_name(color) == color_name for color in _body_colors(body))


# %% object resolution


def _normalized_name(text: object) -> str:
    """
    PDDL names use hyphens where internal body names use underscores; matching treats
    the two spellings as the same name.
    """
    return str(text).lower().replace("_", "-")


def _name_matches(item: GroundedObject, needle: str) -> bool:
    """
    Match against the PDDL name the model was shown, and the internal root-body name as
    a fallback.
    """
    names = [item.name, str(item.body.name.name)]
    wanted = _normalized_name(needle)
    return any(wanted in _normalized_name(name) for name in names)


@dataclass(frozen=True)
class KrroodObjectResolver:
    """
    Resolve a structured description over one task's CRAM objects.

    The typed candidate set comes from an EQL query over the task universe's semantic
    entities; the name and color filters are deterministic Python over that candidate
    set, matching the PDDL names shown to the model and the nearest named
    ``semantic_digital_twin`` color of any of the object's shapes.
    """

    universe: ObjectUniverse

    def dependencies(
        self,
        seeds: Iterable[str],
        required_types: Iterable[SymbolType],
    ) -> tuple[ObjectDependency, ...]:
        """
        Follow annotation references and storage owners in the current world.

        Owner traversal fills missing typed roles without including every sibling.
        Occupants lead to their storage owner, not to other occupants. Native references
        are discovery hints, not predicate truth.
        """
        by_identity = {
            id(item.semantic_entity): name
            for name, item in self.universe.objects.items()
            if isinstance(item.semantic_entity, SemanticAnnotation)
        }
        references: dict[str, set[str]] = {
            name: set() for name in self.universe.objects
        }
        owners: dict[str, set[str]] = {name: set() for name in self.universe.objects}
        for name, item in self.universe.objects.items():
            annotation = item.semantic_entity
            if not isinstance(annotation, SemanticAnnotation):
                continue
            occupants = (
                {id(occupant) for occupant in annotation.objects}
                if isinstance(annotation, IsStorageSpace)
                else set()
            )
            for reference in annotation._referenced_semantic_annotations():
                related = by_identity.get(id(reference))
                if related is None:
                    continue
                owners[related].add(name)
                if id(reference) not in occupants:
                    references[name].add(related)
        seeds = frozenset(seeds)
        for dependency in self._spatial_owners(seeds):
            owners[dependency.related_to].add(dependency.object_name)
        typed_candidates = [
            {item.name for item in self.universe.of_type(symbol_type)}
            for symbol_type in required_types
        ]
        pending = deque(sorted(seeds))
        visited = set(pending)
        branches = set(seeds)
        result = []
        while pending:
            source = pending.popleft()
            for target in sorted(references[source]):
                if target in visited:
                    continue
                if source not in branches and not any(
                    target in candidates and candidates.isdisjoint(visited)
                    for candidates in typed_candidates
                ):
                    continue
                branches.add(target)
                visited.add(target)
                pending.append(target)
                result.append(ObjectDependency(target, source))
            for target in sorted(owners[source]):
                if target in visited:
                    continue
                visited.add(target)
                pending.append(target)
                result.append(ObjectDependency(target, source))
        return tuple(result)

    def _spatial_owners(self, seeds: Iterable[str]) -> Iterable[ObjectDependency]:
        """
        Query support or containment overlap only for seeded bodies with geometry.

        Any containment overlap is a candidate hint, not an ``inside`` truth value.
        Unloaded geometry has no spatial evidence; typed expansion remains available.
        """
        owners = {
            id(item.semantic_entity): item
            for item in self.universe.objects.values()
            if isinstance(item.semantic_entity, HasSupportingSurface)
            and item.body._world is not None
            and item.body.combined_mesh is not None
        }
        for name in sorted(seeds):
            item = self.universe[name]
            if (
                isinstance(item.semantic_entity, AbstractRobot)
                or item.body._world is None
                or item.body.combined_mesh is None
            ):
                continue
            domain = tuple(
                owner.semantic_entity
                for owner in owners.values()
                if owner.body is not item.body and owner.body._world is item.body._world
            )
            if not domain:
                continue
            candidate = variable(HasSupportingSurface, domain=domain)
            supporting = (
                entity(candidate)
                .where(is_supported_by(item.body, candidate.root))
                .evaluate()
            )
            supported_by = {id(owner) for owner in supporting}
            for annotation in domain:
                if (
                    id(annotation) in supported_by
                    or InsideOf(item.body, annotation.root)() > 0.0
                ):
                    yield ObjectDependency(owners[id(annotation)].name, name)

    def resolve(self, spec: ObjectQuerySpec) -> tuple[GroundedObject, ...]:
        semantic_domain = tuple(
            item.semantic_entity
            for item in self.universe.objects.values()
            if item.semantic_entity is not None
        )
        object_type = (
            resolve_symbol_type(SymbolType(spec.type_ref))
            if spec.type_ref is not None
            else object
        )
        candidate = variable(object_type, domain=semantic_domain)
        matches = list(entity(candidate).evaluate())

        by_identity = {
            id(item.semantic_entity): item
            for item in self.universe.objects.values()
            if item.semantic_entity is not None
        }
        resolved = tuple(
            by_identity[id(match)] for match in matches if id(match) in by_identity
        )
        if spec.name_contains:
            resolved = tuple(
                item for item in resolved if _name_matches(item, spec.name_contains)
            )
        if spec.color:
            resolved = tuple(
                item for item in resolved if _color_matches(item.body, spec.color)
            )
        return resolved
