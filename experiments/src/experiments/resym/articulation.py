"""
Access to articulated CRAM objects for the articulation experiments.

These accessors are the domain vocabulary the experiments' grounding factories are
drafted against and the scene setup drives joints with; the platform itself carries no
predicate semantics for them.
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


def has_interaction_point(grounded: GroundedObject) -> bool:
    """
    Whether the articulated object carries a part to interact with.
    """
    return _require(grounded, HasHandle, "object with a handle").handle is not None


def interaction_point_belongs_to(
    interaction_point: GroundedObject,
    articulated: GroundedObject,
) -> bool:
    """
    Whether the interaction point is the one mounted on the articulated object; an
    object without one has no interaction point, so nothing belongs to it.
    """
    annotation = _require(articulated, HasHandle, "object with a handle")
    return (
        annotation.handle is not None
        and annotation.handle.root is interaction_point.body
    )


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
