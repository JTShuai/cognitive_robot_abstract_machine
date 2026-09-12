"""
What a predicate truth procedure may consult besides its arguments.

The context carries only handles: the kinematic world, the robot annotation, the
embodiment's declared capability surface, the approved factory catalog, and the
feasibility oracle. It holds no predicate semantics of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Optional

from resym.platform.embodiment import EmbodimentProfile
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

if TYPE_CHECKING:
    from resym.platform.feasibility import CapabilityFeasibility
    from resym.platform.grounding_catalog import GroundingFactoryCatalog
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import WheeledDrive


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

    profile: EmbodimentProfile
    """What this embodiment can do."""

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

    @property
    def drive_connection(self) -> Optional[WheeledDrive]:
        """
        The drive that moves the base, read off the robot description; ``None`` on a
        fixed arm.
        """
        return None if self.robot is None else self.robot.drive

    def arms(self) -> list[Arm]:
        """
        Every arm of the robot, from its robot-part annotation tree.
        """
        return self.robot.get_arms()

    def manipulation_arm(self) -> Arm:
        """
        The arm used for manipulation; deterministically the robot's right arm where
        there is a choice.
        """
        right_arm = self.robot.get_right_arm_if_specified()
        if right_arm is not None:
            return right_arm
        return self.arms()[0]
