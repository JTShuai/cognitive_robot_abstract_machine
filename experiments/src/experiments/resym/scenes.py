"""
Scene construction: a robot merged into a CRAM household world, drawers annotated.

Stage A of the pipeline: the kinematic world model is given (URDF assets from the CRAM
monorepo), and the semantic annotation layer marks which bodies are drawers and handles.
Both are inputs the paper assumes, not contributions. The annotation itself is CRAM's:
:class:`WorldReasoner` classifies drawers, handles, doors and wardrobes from the
kinematic structure with its persisted ripple-down rules.

Two embodiments are assembled here: the mobile PR2 (P3 material) and the fixed-arm Tracy
(the IAI TraceBot dual-UR10e table setup with Robotiq grippers) the P2 experiments run
on. Each scene setup carries the :class:`EmbodimentProfile` its capability checks and
truth procedures dispatch on.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path

import numpy as np
from typing_extensions import TYPE_CHECKING, Optional

from resym.platform.embodiment import EmbodimentProfile, ToolOrientation

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import OmniDrive

CRAM_RESOURCES = (
    Path(files("semantic_digital_twin")).parent.parent / "resources" / "urdf"
)
"""
Scene descriptions owned by the Semantic Digital Twin package.
"""


class Scene(Enum):
    """
    The two demo households; same library, different geometry.
    """

    APARTMENT = "apartment"
    KITCHEN = "kitchen"

    @property
    def urdf_path(self) -> Path:
        return (
            CRAM_RESOURCES
            / {
                Scene.APARTMENT: "apartment.urdf",
                Scene.KITCHEN: "kitchen-small.urdf",
            }[self]
        )

    @property
    def robot_spawn_xy(self) -> tuple[float, float]:
        """
        A free-floor spawn pose away from the target furniture.
        """
        return {
            Scene.APARTMENT: (2.2, 3.4),
            Scene.KITCHEN: (1.5, 1.5),
        }[self]

    @property
    def goal_drawer_body(self) -> str:
        """
        The drawer each demo run is asked to open.
        """
        return {
            Scene.APARTMENT: "cabinet10_drawer_top",
            Scene.KITCHEN: "kitchen_island_left_upper_drawer_main",
        }[self]


@dataclass
class SceneSetup:
    """
    A ready-to-plan world: robot merged, drawers annotated.
    """

    world: World
    """
    The combined robot + household world.
    """

    robot: AbstractRobot
    """
    The robot semantic annotation.
    """

    profile: EmbodimentProfile
    """
    The declared capability surface of this embodiment.
    """

    scene: Scene
    """
    Which household this is.
    """

    @property
    def drive_connection(self) -> Optional[OmniDrive]:
        """
        The planar drive that navigation actuates, read off the robot description;
        ``None`` on the fixed arm.
        """
        return self.robot.drive


def mobile_profile(robot: AbstractRobot) -> EmbodimentProfile:
    """
    Derive the PR2 capability surface from its CRAM robot annotation.
    """
    from resym.platform.evaluators import EVALUATORS
    from resym.platform.coraplex_catalog import (
        CORAPLEX_ADAPTER_CAPABILITY_UIDS,
        coraplex_embodiment_profile,
    )

    return coraplex_embodiment_profile(
        name="pr2-mobile",
        robot=robot,
        evaluators=frozenset(EVALUATORS),
        adapter_capability_uids=CORAPLEX_ADAPTER_CAPABILITY_UIDS,
        tool_orientation=ToolOrientation.BASE_ALIGNED,
    )


def fixed_arm_profile(robot: AbstractRobot) -> EmbodimentProfile:
    """
    The Tracy embodiment: no drive, so no witness-pose sampling (``openable``) and no
    navigation skill — asking for either is an unsupported capability, not a library
    gap.
    """
    from resym.platform.evaluators import EVALUATORS
    from resym.platform.coraplex_catalog import (
        CORAPLEX_ADAPTER_CAPABILITY_UIDS,
        coraplex_embodiment_profile,
    )

    return coraplex_embodiment_profile(
        name="tracy-fixed",
        robot=robot,
        evaluators=frozenset(EVALUATORS) - {"openable"},
        adapter_capability_uids=CORAPLEX_ADAPTER_CAPABILITY_UIDS,
        tool_orientation=ToolOrientation.APPROACH_ALIGNED,
    )


def load_scene(scene: Scene) -> SceneSetup:
    """
    Build the PR2 world, merge the household, annotate its drawers.
    """
    from semantic_digital_twin.adapters.urdf import URDFParser
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.robots.pr2 import PR2
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
    from semantic_digital_twin.world_description.connections import (
        Connection6DoF,
        OmniDrive,
    )
    from semantic_digital_twin.world_description.world_entity import Body

    world = URDFParser.from_file(file_path=PR2.get_ros_file_path()).parse()
    robot = PR2.from_world(world)
    with world.modify_world():
        robot_root = world.root
        map_body = Body(name=PrefixedName("map"))
        localization_body = Body(name=PrefixedName("odom_combined"))
        map_c_localization = Connection6DoF.create_with_dofs(
            world, map_body, localization_body
        )
        world.add_connection(map_c_localization)
        drive_connection = OmniDrive.create_with_dofs(
            parent=localization_body, child=robot_root, world=world
        )
        world.add_connection(drive_connection)
        drive_connection.has_hardware_interface = True

    household = URDFParser.from_file(file_path=str(scene.urdf_path)).parse()
    world.merge_world(household)

    spawn_x, spawn_y = scene.robot_spawn_xy
    drive_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        spawn_x, spawn_y, 0.0, reference_frame=world.root
    )
    world.notify_state_change()

    annotate_with_world_reasoner(world)
    return SceneSetup(
        world=world,
        robot=robot,
        profile=mobile_profile(robot),
        scene=scene,
    )


FIXED_ARM_STANDOFF_FRONT = 0.6
"""
Table offset (meters) in front of the goal drawer's handle.

The robot must not stand square in the travel path: this scene's goal drawer opens
0.45 m toward the table, so the front offset keeps the fully opened
handle clear of the tabletop while both drawer states stay well inside
the UR10e's 1.3 m reach.
"""

FIXED_ARM_STANDOFF_SIDE = 0.35
"""
Table offset (meters) to the side of the drawer's travel path.
"""


def load_fixed_arm_scene(scene: Scene = Scene.APARTMENT, variation=None) -> SceneSetup:
    """
    Build the Tracy world (dual UR10e arms with Robotiq grippers on their table), merge
    the household, and place the table beside the scene's goal drawer.

    The table pose is derived from the annotated geometry: it stands on
    the floor, offset diagonally from the goal drawer's handle —
    ``FIXED_ARM_STANDOFF_FRONT`` meters along the furniture front (the
    drawer-to-handle direction) and ``FIXED_ARM_STANDOFF_SIDE`` meters
    beside the travel path — turned to face the handle. ``variation``
    (a :class:`~experiments.resym.icra.articulation.faults.SceneVariation`) jitters the
    two standoffs deterministically for the randomized experiment
    scenes.
    """
    from semantic_digital_twin.adapters.urdf import URDFParser
    from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
    from semantic_digital_twin.robots.tracy import Tracy
    from semantic_digital_twin.world_description.connections import Connection6DoF
    from semantic_digital_twin.world_description.world_entity import Body

    world = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).parse()
    robot = Tracy.from_world(world)
    with world.modify_world():
        robot_root = world.root  # the URDF's own 'map' link above the table
        mount_body = Body(name=PrefixedName("mount"))
        mount = Connection6DoF.create_with_dofs(world, mount_body, robot_root)
        world.add_connection(mount)

    household = URDFParser.from_file(file_path=str(scene.urdf_path)).parse()
    world.merge_world(household)
    annotate_with_world_reasoner(world)

    mount.origin = _fixed_arm_mount_pose(world, scene, variation)
    world.notify_state_change()
    return SceneSetup(
        world=world, robot=robot, profile=fixed_arm_profile(robot), scene=scene
    )


def _fixed_arm_mount_pose(world: World, scene: Scene, variation=None):
    """
    The table pose beside the scene's goal drawer, computed from the annotated handle
    and drawer geometry.

    The table stands on the floor; only its planar placement is derived.
    """
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Drawer,
    )
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix

    drawers = world.get_semantic_annotations_by_type(Drawer)
    matching = [d for d in drawers if d.root.name.name == scene.goal_drawer_body]
    if not matching:
        raise ValueError(
            f"No annotated drawer named '{scene.goal_drawer_body}' in scene "
            f"'{scene.value}'."
        )
    (drawer,) = matching
    handle_position = drawer.handle.root.global_pose.to_np()[:3, 3]
    drawer_position = drawer.root.global_pose.to_np()[:3, 3]
    front = handle_position - drawer_position
    front[2] = 0.0
    norm = float(np.linalg.norm(front))
    if norm < 1e-6:
        raise ValueError(
            f"Handle of '{scene.goal_drawer_body}' sits directly above its "
            "drawer; cannot derive the furniture front direction."
        )
    front /= norm
    side = np.cross([0.0, 0.0, 1.0], front)
    front_standoff = FIXED_ARM_STANDOFF_FRONT
    side_standoff = FIXED_ARM_STANDOFF_SIDE
    if variation is not None:
        front_standoff += variation.front_offset_delta
        side_standoff += variation.side_offset_delta
    offset = front_standoff * front + side_standoff * side
    base_x = handle_position[0] + offset[0]
    base_y = handle_position[1] + offset[1]
    # The bench mesh extends ~1.2 m in the table frame's +x; +x must point
    # away from the handle so the bench clears the furniture row while the
    # arms (mounted near the table origin) stay within reach.
    yaw = math.atan2(offset[1], offset[0])
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        base_x, base_y, 0.0, 0.0, 0.0, yaw, reference_frame=world.root
    )


def annotate_with_world_reasoner(world: World) -> int:
    """
    Infer the semantic annotations (drawers, handles, doors, ...) of the world with
    CRAM's persisted ripple-down rules. Returns how many annotations were inferred.

    .. note:: The ripple-down-rules expert interface is Qt-based and aborts on
       machines without a display; forcing the offscreen Qt platform keeps the
       purely rule-driven inference path headless-safe.
    """
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner

    return len(WorldReasoner(world).infer_semantic_annotations())
