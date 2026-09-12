"""
What a predicate truth procedure may consult besides its arguments.

The context carries only handles: the world, CRAM robot annotation, approved factory
catalog, and optional feasibility oracle. It holds no predicate semantics of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Optional

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

if TYPE_CHECKING:
    from resym.platform.feasibility import CapabilityFeasibility
    from resym.platform.grounding_catalog import GroundingFactoryCatalog
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.world import World


@dataclass
class EvaluationContext:
    """
    Everything a truth procedure may consult besides its arguments.
    """

    world: World
    """The kinematic world truth is computed against."""

    robot: AbstractRobot
    """
    The robot semantic annotation.
    """

    grounding_catalog: GroundingFactoryCatalog
    """
    Approved factory view used by predicates with a grounding plan.
    """

    capability_feasibility: Optional[CapabilityFeasibility] = None
    """
    Oracle answering capability-feasibility grounding factories.
    """

    witness_base_poses: dict[tuple[str, str], HomogeneousTransformationMatrix] = field(
        default_factory=dict
    )
    """
    Feasible base pose per (robot, target) found during grounding, reused by execution.
    """

    def manipulation_arm(self) -> Arm:
        """
        The arm used for manipulation; deterministically the robot's right arm where
        there is a choice.
        """
        right_arm = self.robot.get_right_arm_if_specified()
        if right_arm is not None:
            return right_arm
        return self.robot.get_arms()[0]
