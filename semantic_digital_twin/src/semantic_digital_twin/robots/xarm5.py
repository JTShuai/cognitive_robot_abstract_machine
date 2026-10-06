"""
Semantic annotation for the UFACTORY xArm 5.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Self

from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_part_mixins import HasOneArm
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm, EndEffector
from semantic_digital_twin.spatial_types import Quaternion
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


class XArm5Joint(StrEnum):
    """
    Names of the xArm 5's commandable connections, as spelled in its URDF.
    """

    JOINT_1 = "joint1"
    JOINT_2 = "joint2"
    JOINT_3 = "joint3"
    JOINT_4 = "joint4"
    JOINT_5 = "joint5"


@dataclass(eq=False)
class XArm5Flange(EndEffector):
    """
    The bare mounting flange at the end of the arm, with nothing attached to it. A real
    gripper will be added in the future version.

    ``root`` and ``tool_frame`` are both ``link_eef``, the frame the description puts at
    the tool centre point. ``joint_eef`` is fixed with a zero origin, so that frame
    coincides with ``link5``.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        flange = robot_root._world.get_body_in_branch_by_name(robot_root, "link_eef")
        return cls(
            root=flange,
            tool_frame=flange,
            # should be re-defined after mounting a real gripper
            front_facing_orientation=Quaternion(0, -0.70710678, 0, 0.70710678),
        )

    def setup_hardware_interfaces(self):
        return None

    def setup_joint_states(self) -> List[JointState]:
        return []


@dataclass(eq=False)
class XArm5Arm(Arm[XArm5Flange]):
    """
    The arm chain from ``link_base`` to ``link5``, carrying the bare flange as its end
    effector.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(
        cls, robot_root: KinematicStructureEntity
    ) -> Self:
        return cls(
            root=robot_root._world.get_body_in_branch_by_name(robot_root, "link_base"),
            tip=robot_root._world.get_body_in_branch_by_name(robot_root, "link5"),
        )

    def setup_hardware_interfaces(self):
        self._setup_hardware_interfaces_for_active_connections()

    def setup_joint_states(self) -> List[JointState]:
        arm_park = JointState.from_mapping(
            name=PrefixedName("arm_park", prefix=self.name.name),
            mapping=dict(
                zip(self.active_connections, [0.0] * len(self.active_connections))
            ),
            state_type=StaticJointState.PARK,
        )
        return [arm_park]


@dataclass(eq=False)
class XArm5(AbstractRobot, HasOneArm[XArm5Arm]):
    """
    UFACTORY xArm 5: a 5-DoF arm.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://xarm_description/urdf/xarm_device.urdf.xacro"

    DEFAULT_SERIAL_NUMBER: ClassVar[str] = "XF1305122503B6"
    """
    https://github.com/xArm-Developer/xarm_ros2/issues/176
    """

    @classmethod
    def get_xacro_mappings(cls, serial_number: Optional[str] = None) -> Dict[str, str]:
        """
        Selects the xArm 5 from the parameterized device description, whose defaults
        describe the 7-DoF model, and the hardware variant the serial number encodes.

        :param serial_number: The serial number of the arm to describe, as printed on
            it. Defaults to :attr:`DEFAULT_SERIAL_NUMBER`.
        """
        return {
            "robot_type": "xarm",
            "dof": "5",
            "robot_sn": serial_number or cls.DEFAULT_SERIAL_NUMBER,
        }

    @classmethod
    def _get_root_body_name(cls) -> str:
        return "link_base"

    def _setup_collision_rules(self):
        srdf_path = os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "collision_configs",
            "xarm5.srdf",
        )
        self._world.collision_manager.add_ignore_collision_rule(
            SelfCollisionMatrixRule.from_collision_srdf(srdf_path, self._world)
        )
        self._world.collision_manager.add_default_rule(
            AvoidExternalCollisions(
                buffer_zone_distance=0.05, violated_distance=0.0, robot=self
            )
        )
