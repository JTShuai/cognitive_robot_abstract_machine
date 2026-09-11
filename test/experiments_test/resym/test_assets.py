"""
ReSym resolves scene and robot descriptions from CRAM packages.
"""

from __future__ import annotations

from experiments.resym.scenes import Scene
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.tracy import Tracy


def test_scene_descriptions_come_from_semantic_digital_twin():
    for scene in Scene:
        assert scene.urdf_path.is_file()


def test_robot_descriptions_are_resolved_by_cram():
    assert PR2.get_ros_file_path().startswith("package://")
    assert Tracy.get_ros_file_path().startswith("package://")
