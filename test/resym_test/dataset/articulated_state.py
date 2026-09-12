"""
World accessors the test task model's grounding factories are reviewed against.

The platform ships no domain query helpers, so the suite carries its own: these stand in
for the accessors an application package reviews into the EQL vocabulary.
"""

from __future__ import annotations

from resym.platform.universe import GroundedObject
from semantic_digital_twin.semantic_annotations.mixins import (
    HasHandle,
    HasMechanicalJoint,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


def articulation_connection(grounded: GroundedObject) -> ActiveConnection1DOF:
    """
    The active 1-DOF joint an articulated object moves on.
    """
    annotation = grounded.semantic_entity
    if not isinstance(annotation, HasMechanicalJoint):
        raise TypeError(f"'{grounded.name}' does not denote an articulated object")
    return annotation.root.get_first_parent_connection_of_type(ActiveConnection1DOF)


def joint_fraction(connection: ActiveConnection1DOF) -> float:
    """
    Normalized joint position in [0, 1] of a 1-DOF connection.
    """
    limits = connection.dof.limits
    return (connection.position - limits.lower.position) / (
        limits.upper.position - limits.lower.position
    )


def interaction_point_belongs_to(
    interaction_point: GroundedObject,
    articulated: GroundedObject,
) -> bool:
    """
    Whether the interaction point is the one mounted on the articulated object.
    """
    annotation = articulated.semantic_entity
    if not isinstance(annotation, HasHandle):
        raise TypeError(f"'{articulated.name}' does not denote an object with a handle")
    return annotation.handle.root is interaction_point.body
