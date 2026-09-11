"""
Typed krrood/EQL access to one Semantic Digital Twin world.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from krrood.entity_query_language.factories import entity, variable
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Iterable, Mapping

from resym.core.model import SymbolType, resolve_symbol_type
from resym.platform.universe import GroundedObject, ObjectUniverse


@dataclass(frozen=True)
class ObjectQuerySpec:
    """
    A typed, serializable object description chosen by the task agent.
    """

    type_ref: str | None = None
    color: str | None = None
    name_contains: str | None = None


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
