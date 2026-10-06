from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.xarm5.configs import (
    WorldWithXArm5Config,
    XArm5StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from rclpy import Parameter
from semantic_digital_twin.robots.xarm5 import XArm5


def main():
    rospy.init_node("giskard")
    rospy.get_node().declare_parameters(
        namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
    )
    # The device xacro is parameterized and defaults to the 7-DoF model, so it is
    # expanded here with the xArm 5's mappings rather than read from the parameter.
    robot_description = load_xacro(
        XArm5.get_ros_file_path(), mappings=XArm5.get_xacro_mappings()
    )

    giskard = Giskard(
        world_config=WorldWithXArm5Config(urdf=robot_description),
        robot_interface_config=XArm5StandAloneRobotInterfaceConfig(),
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.STANDALONE, debug_mode=True
        ),
        qp_controller_config=QPControllerConfig(target_frequency=33),
    )
    giskard.live()


if __name__ == "__main__":
    main()
