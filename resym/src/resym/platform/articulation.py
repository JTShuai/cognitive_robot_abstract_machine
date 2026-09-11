"""
Task-neutral access to CRAM articulated semantic objects.
"""

from __future__ import annotations

from resym.platform.universe import GroundedObject
from semantic_digital_twin.semantic_annotations.mixins import (
    HasHandle,
    HasMechanicalJoint,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


def is_articulated_object(grounded: GroundedObject) -> bool:
    return grounded.denotes(HasMechanicalJoint)


def articulation_connection(grounded: GroundedObject) -> ActiveConnection1DOF:
    annotation = _require(grounded, HasMechanicalJoint, "articulated object")
    try:
        return annotation.root.get_first_parent_connection_of_type(ActiveConnection1DOF)
    except ValueError:
        raise MissingArticulationJointError(grounded.name) from None


def interaction_point_body(grounded: GroundedObject):
    annotation = _require(grounded, HasHandle, "object with a handle")
    if annotation.handle is None:
        raise MissingInteractionPointError(grounded.name)
    return annotation.handle.root


def interaction_point_belongs_to(
    interaction_point: GroundedObject,
    articulated: GroundedObject,
) -> bool:
    return interaction_point_body(articulated) is interaction_point.body


def _require(grounded: GroundedObject, semantic_type: type, label: str):
    if not grounded.denotes(semantic_type):
        raise TypeError(f"'{grounded.name}' does not denote an {label}")
    return grounded.semantic_entity


class MissingArticulationJointError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Articulated object '{name}' has no active 1-DOF joint.")


class MissingInteractionPointError(Exception):
    def __init__(self, name: str):
        super().__init__(f"Articulated object '{name}' has no interaction point.")
