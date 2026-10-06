from pathlib import Path

from semantic_digital_twin.collision_checking.collision_rules import (
    SelfCollisionMatrixRule,
)
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.xarm5 import XArm5, XArm5Flange, XArm5Joint


def test_xarm5_loads_from_workspace_description(xarm5_world):
    """
    The xArm5 description comes from the ``xarm_description`` package of the workspace
    (see ``.github/docker/setup_workspace.py``); its device xacro must be expanded with
    :meth:`XArm5.get_xacro_mappings` because the xArm 5 is not the default model of that
    parameterized description.
    """
    robot = xarm5_world.get_semantic_annotations_by_type(XArm5)[0]
    assert robot.root.name.name == "link_base"
    assert robot.arm.tip.name.name == "link5"
    assert robot.arm.end_effector.root.name.name == "link_eef"
    assert [
        connection.raw_dof.name.name for connection in robot.arm.active_connections
    ] == list(XArm5Joint)


# %% hardware variant


def test_the_default_serial_number_selects_the_1305_flange():
    """
    Characters three to six of the serial number are the model number the description
    switches the end-effector flange on.

    A model number below 1300 selects the oldest flange, which is what an absent serial
    number falls back to, so the mapping has to carry one.
    """
    mappings = XArm5.get_xacro_mappings()

    assert mappings["robot_sn"] == XArm5.DEFAULT_SERIAL_NUMBER
    assert mappings["robot_sn"][2:6] == "1305"


def test_a_caller_describes_its_own_arm_by_serial_number():
    serial_number = "XF1300123456A1"

    mappings = XArm5.get_xacro_mappings(serial_number)

    assert mappings["robot_sn"] == serial_number
    assert mappings["dof"] == "5"


def test_the_flange_mesh_follows_the_serial_numbers_model(xarm5_world):
    """
    The 1305 arm carries a different flange from the pre-1300 one, and the mesh
    ``link5`` is built from is the only thing that makes the difference visible: the two
    variants share every link and joint name, so nothing else in the world distinguishes
    them.
    """
    flange = xarm5_world.get_body_by_name("link5")

    mesh_directories = {
        Path(shape.filename).parent.parent.name for shape in flange.visual
    }

    assert mesh_directories == {f"xarm5_{XArm5.DEFAULT_SERIAL_NUMBER[2:6]}"}


# %% the bare flange as end effector


def test_the_flange_is_the_arms_end_effector(xarm5_world):
    """
    Nothing is bolted to this arm, but ``link_eef`` is a real frame the description puts
    at the tool centre point, so the flange stands as the end effector.

    That keeps the arm an :class:`Arm`, which is what the plan layer reaches for.
    """
    robot = xarm5_world.get_semantic_annotations_by_type(XArm5)[0]

    assert isinstance(robot.arm, Arm)
    assert robot.get_arms() == [robot.arm]
    assert robot.get_end_effectors() == [robot.arm.end_effector]
    assert isinstance(robot.arm.end_effector, XArm5Flange)


def test_the_flange_faces_along_its_own_z_axis(xarm5_world):
    """
    A tool bolted to the flange points along the flange's z axis, which ``joint_eef``
    makes coincide with joint5's axis of rotation.

    An identity orientation would instead claim the arm reaches along x.
    """
    flange = xarm5_world.get_semantic_annotations_by_type(XArm5)[0].arm.end_effector

    front_facing_axis = flange.front_facing_axis.to_np()[:3]

    assert [round(float(value), 4) for value in front_facing_axis] == [0.0, 0.0, 1.0]


def test_the_flange_holds_nothing_until_something_is_grasped(xarm5_world):
    """
    ``held_bodies`` is where a grasped object hangs; a bare flange has none.
    """
    flange = xarm5_world.get_semantic_annotations_by_type(XArm5)[0].arm.end_effector

    assert flange.held_bodies == []


# %% self-collision matrix


def test_the_self_collision_matrix_matches_the_arms_geometry(xarm5_world):
    """
    The shipped matrix is generated against the geometry
    :meth:`XArm5.get_xacro_mappings` expands, and the generator is seeded, so it has a
    single right answer. Recomputing it here catches the matrix drifting away from the
    meshes - which a different hardware variant causes, since the model number selects
    a different mesh set for every link.
    """
    robot = xarm5_world.get_semantic_annotations_by_type(XArm5)[0]
    shipped = {
        frozenset((check.body_a.name.name, check.body_b.name.name))
        for rule in xarm5_world.collision_manager.ignore_collision_rules
        if isinstance(rule, SelfCollisionMatrixRule)
        for check in rule.allowed_collision_pairs
    }

    recomputed_rule = SelfCollisionMatrixRule()
    recomputed_rule.compute_self_collision_matrix(robot=robot)
    recomputed = {
        frozenset((check.body_a.name.name, check.body_b.name.name))
        for check in recomputed_rule.allowed_collision_pairs
    }

    assert shipped == recomputed
