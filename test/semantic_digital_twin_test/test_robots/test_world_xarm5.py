from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.xarm5 import XArm5


def test_xarm5_description_ships_with_package():
    """
    The xArm5 URDF and meshes ship in ``resources/xarm_description``;
    ``get_ros_file_path`` must resolve them without a ROS installation.
    """
    world = URDFParser.from_file(file_path=XArm5.get_ros_file_path()).parse()
    robot = XArm5.from_world(world)
    assert robot.root.name.name == "link_base"
    assert robot.arm is not None
