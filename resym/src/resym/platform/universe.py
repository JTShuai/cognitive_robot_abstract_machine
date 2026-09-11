"""
Typed object universe extracted from a kinematic world.

The universe is the active domain of quantification: predicates are only evaluated over
the entities that actually exist in the current world, keyed by the symbol types of the
library.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Iterable, Protocol, Self

from resym.core.model import (
    SymbolType,
    is_symbol_subtype,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GroundedObject:
    """
    A PDDL object together with the world entities it denotes.
    """

    name: str
    """PDDL-safe object name."""

    symbol_type: SymbolType
    """
    Type of the object in the library's signature language.
    """

    body: Body
    """The world body this object denotes."""

    semantic_entity: object | None = None
    """
    Optional CRAM semantic annotation or robot denoted by this object.

    Task-specific access belongs to the platform adapter (for example, articulation
    helpers), not to this task-independent record.
    """

    def denotes(self, semantic_type: type) -> bool:
        return isinstance(self.semantic_entity, semantic_type)


class WorldObjectExtractor(Protocol):
    """
    One platform-owned rule for discovering typed objects in a world.
    """

    def extract(
        self,
        world: World,
        robot: AbstractRobot,
    ) -> Iterable[GroundedObject]: ...


@dataclass
class ObjectUniverse:
    """
    All typed objects of the current world, indexed by name and by type.
    """

    objects: dict[str, GroundedObject] = field(default_factory=dict)
    """
    Objects by PDDL name.
    """

    def add(self, grounded_object: GroundedObject) -> None:
        if grounded_object.name in self.objects:
            raise DuplicateObjectError(grounded_object.name)
        self.objects[grounded_object.name] = grounded_object

    def of_type(self, symbol_type: SymbolType) -> list[GroundedObject]:
        return [
            item
            for item in self.objects.values()
            if is_symbol_subtype(item.symbol_type, symbol_type)
        ]

    def of_type_name(self, type_name: str) -> list[GroundedObject]:
        """
        Objects by the type's string value used in external fragments.
        """
        return self.of_type(SymbolType(type_name))

    def __getitem__(self, name: str) -> GroundedObject:
        return self.objects[name]

    def for_task(self, selection, goal: tuple) -> ObjectUniverse:
        """
        Keep only objects needed by a selected symbolic task.

        Selection happens after natural-language grounding, so initial object reference
        resolution still sees the full CRAM catalog.  The planner and predicate
        grounding then receive only types used by the selected model.
        """
        relevant_types = {
            symbol_type
            for predicate in selection.predicates.values()
            for symbol_type in predicate.parameter_types
        }
        relevant_types.update(
            symbol_type
            for operator in selection.operators.values()
            for _, symbol_type in operator.parameters
        )
        named_objects = {argument for literal in goal for argument in literal.arguments}
        return ObjectUniverse(
            objects={
                name: item
                for name, item in self.objects.items()
                if name in named_objects
                or any(
                    is_symbol_subtype(item.symbol_type, symbol_type)
                    for symbol_type in relevant_types
                )
            }
        )

    @classmethod
    def from_world(
        cls,
        world: World,
        robot: AbstractRobot,
        extractors: Iterable[WorldObjectExtractor],
    ) -> Self:
        """
        Build the universe using explicitly selected object extractors.
        """
        universe = cls()
        for extractor in extractors:
            for grounded_object in extractor.extract(world, robot):
                universe.add(grounded_object)
        return universe


class DuplicateObjectError(Exception):
    """
    Raised when two world entities map to the same PDDL object name.
    """

    def __init__(self, name: str):
        super().__init__(f"Object name '{name}' is already taken in this universe.")


def pddl_name(raw: str) -> str:
    """
    Map a world entity name to a PDDL-safe identifier.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "-", raw).strip("-").lower()
    if not sanitized:
        raise ValueError(f"Cannot derive a PDDL name from '{raw}'.")
    return sanitized


def joint_fraction(connection: ActiveConnection1DOF) -> float:
    """
    Normalized joint position in [0, 1] of a 1-DOF connection.
    """
    limits = connection.dof.limits
    return (connection.position - limits.lower.position) / (
        limits.upper.position - limits.lower.position
    )


def set_joint_fraction(connection: ActiveConnection1DOF, fraction: float) -> None:
    """
    Move a 1-DOF connection to a fraction of its range.

    The CRAM position setter applies the connection's multiplier/offset and notifies the
    world of the state change itself.
    """
    limits = connection.dof.limits
    connection.position = limits.lower.position + fraction * (
        limits.upper.position - limits.lower.position
    )
