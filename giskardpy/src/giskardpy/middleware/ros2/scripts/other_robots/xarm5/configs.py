from dataclasses import dataclass, field
from typing import List, Optional

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.model.world_config import WorldWithFixedRobot
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.xarm5 import XArm5, XArm5Joint


@dataclass
class WorldWithXArm5Config(WorldWithFixedRobot):
    """
    A world containing only the xArm 5, whose base is fixed to the world root.
    """

    root_name: PrefixedName = field(default=PrefixedName("map"))
    """
    Name of the body the xArm 5 is attached to.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=XArm5, init=False)
    """
    Semantic view that is applied to the parsed urdf.
    """

    def setup_world(self, robot_name: Optional[str] = None) -> None:
        super().setup_world()
        self.robot = self.world.get_semantic_annotations_by_type(XArm5)[0]


@dataclass
class XArm5StandAloneRobotInterfaceConfig(StandAloneRobotInterfaceConfig):
    """
    Simulates the arm of the xArm 5 without talking to hardware.
    """

    joint_names: List[str] = field(
        init=False,
        default_factory=lambda: [
            XArm5Joint.JOINT_1,
            XArm5Joint.JOINT_2,
            XArm5Joint.JOINT_3,
            XArm5Joint.JOINT_4,
            XArm5Joint.JOINT_5,
        ],
    )
    """
    The arm joints of the xArm 5.
    """
