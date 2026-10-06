from dataclasses import dataclass, field

import pytest
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.xarm5.configs import (
    WorldWithXArm5Config,
    XArm5StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.xarm5 import XArm5, XArm5Joint
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% fixtures


@pytest.fixture()
def default_joint_state():
    return {}


@pytest.fixture()
def better_pose(default_joint_state):
    return default_joint_state


@dataclass
class XArm5Tester(GiskardTester):
    tip: KinematicStructureEntity = field(init=False)
    base: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.tip = self.api.world.get_kinematic_structure_entity_by_name("link_eef")
        self.base = self.api.world.get_kinematic_structure_entity_by_name("link_base")
        self.map = self.api.world.root

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro(
            XArm5.get_ros_file_path(), mappings=XArm5.get_xacro_mappings()
        )
        return Giskard(
            world_config=WorldWithXArm5Config(urdf=robot_desc),
            robot_interface_config=XArm5StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(
                execution_mode=ExecutionMode.STANDALONE,
                debug_mode=True,
                plot_gantt_chart=True,
                plot_trajectory=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> XArm5:
        return self.giskard.executor.context.world.get_semantic_annotations_by_type(
            XArm5
        )[0]


@pytest.fixture()
def robot():
    c = XArm5Tester()
    try:
        yield c
    finally:
        print("tear down")
        c.close()


# %% tests


class TestSetup:

    def test_small_msc(self, giskard: XArm5Tester):
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {
                        XArm5Joint.JOINT_1: 1.23,
                        XArm5Joint.JOINT_2: 1.23,
                        XArm5Joint.JOINT_3: 1.23,
                    },
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)
