"""
A robot stand-in exposing only the structural resources the capability catalog reads.
"""

from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Arm, Camera, EndEffector, Torso
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world_description.world_entity import Body


class ForwardCamera(Camera):
    """
    A constructible camera sensor for exercising resource discovery.
    """

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        raise NotImplementedError

    def setup_hardware_interfaces(self):
        raise NotImplementedError

    def setup_joint_states(self):
        raise NotImplementedError


def forward_camera() -> ForwardCamera:
    return ForwardCamera(
        root=Body(name=PrefixedName("camera")),
        forward_facing_axis=Vector3.from_iterable((1.0, 0.0, 0.0)),
        field_of_view=FieldOfView(),
    )


class _PartStandIn:
    """
    Instantiable without the kinematic structure a real part needs; only its type
    matters to resource discovery.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        raise NotImplementedError

    def setup_hardware_interfaces(self):
        raise NotImplementedError

    def setup_joint_states(self):
        raise NotImplementedError


class MimicArm(_PartStandIn, Arm):
    """
    An arm by type alone.
    """


class MimicEndEffector(_PartStandIn, EndEffector):
    """
    An end effector by type alone.
    """


class MimicTorso(_PartStandIn, Torso):
    """
    A torso by type alone.
    """


class Robot:
    """
    Mimics the resource-bearing surface of ``AbstractRobot``.
    """

    def __init__(self, *, mobile: bool):
        self.drive = object() if mobile else None
        self.torso = MimicTorso()

    def get_arms(self):
        return [MimicArm()]

    def get_end_effectors(self):
        return [MimicEndEffector()]

    def get_sensors(self):
        return [forward_camera()]

    def get_torso_if_specified(self):
        return self.torso
