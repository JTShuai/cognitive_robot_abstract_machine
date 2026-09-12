"""
Native actions built from reviewed capability registrations match the bundled handlers.
"""

import dataclasses

import numpy
import pytest

pytest.importorskip("rclpy")

from resym.core.capability_model import CapabilityRef, ExecutionRequest
from resym.core.symbol_types import SymbolType
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import GroundedObject, ObjectUniverse
from resym.planning.execution.coraplex import CoraplexSkillRealization, _action_type
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.definitions import TorsoState
from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.robot_plans.actions.composite.tool_based import PouringAction
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    CarryAction,
    FollowToolCenterPointPathAction,
    MoveTorsoAction,
    SetGripperAction,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    PouringCup,
    Sponge,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from .dataset.capability_model import (
    ACTOR_ROLE,
    ARTICULATION_CAPABILITY_UID,
    BASE_NAVIGATION_CAPABILITY_UID,
    CARRY_POSTURE_CAPABILITY_UID,
    GRIPPER_STATE_CAPABILITY_UID,
    INTERACTION_NAVIGATION_CAPABILITY_UID,
    INTERACTION_POINT_ROLE,
    PATIENT_ROLE,
    PICK_UP_CAPABILITY_UID,
    PLACE_CAPABILITY_UID,
    POURING_CAPABILITY_UID,
    TARGET_STATE_ROLE,
    TOOL_PATH_CAPABILITY_UID,
    TORSO_STATE_CAPABILITY_UID,
    TRANSPORT_CAPABILITY_UID,
    OpenCloseState,
)

OBJECT_TYPE = SymbolType.from_python_type(SemanticAnnotation)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


def assert_same_action(actual, expected) -> None:
    """
    Constructor-parameter equality where poses compare by value, not by identity.
    """
    assert type(actual) is type(expected)
    for field in dataclasses.fields(expected):
        if field.init:
            assert_same_value(vars(actual)[field.name], vars(expected)[field.name])


def assert_same_value(actual, expected) -> None:
    if isinstance(expected, Pose):
        numpy.testing.assert_allclose(actual.to_np(), expected.to_np())
    elif isinstance(expected, PoseTrajectory):
        assert len(actual.poses) == len(expected.poses)
        for actual_pose, expected_pose in zip(actual.poses, expected.poses):
            assert_same_value(actual_pose, expected_pose)
    else:
        assert actual == expected


def posed_body(world, name: str, x: float, y: float, z: float) -> Body:
    """
    A body fixed to the world root at a position, so its global pose is defined.
    """
    body = Body(name=PrefixedName(name))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x, y, z
                ),
            )
        )
    return body


@pytest.fixture
def pr2_context(pr2_world_copy, grounding_catalog) -> EvaluationContext:
    robot = next(
        annotation
        for annotation in pr2_world_copy.semantic_annotations
        if isinstance(annotation, AbstractRobot)
    )
    return EvaluationContext(
        world=pr2_world_copy, robot=robot, grounding_catalog=grounding_catalog
    )


@pytest.fixture
def kitchen_universe(pr2_context) -> ObjectUniverse:
    """
    Posed task objects covering every role the reviewed realizations bind.
    """
    world = pr2_context.world
    universe = ObjectUniverse()
    universe.add(
        GroundedObject("pr2", ROBOT_TYPE, pr2_context.robot.root, pr2_context.robot)
    )
    for name, x in (("cup", 1.0), ("table", 1.5), ("handle", 0.9), ("drawer", 0.8)):
        universe.add(
            GroundedObject(name, OBJECT_TYPE, posed_body(world, name, x, 0.0, 0.8))
        )
    bottle = posed_body(world, "bottle", 1.2, 0.0, 0.8)
    universe.add(GroundedObject("bottle", OBJECT_TYPE, bottle, PouringCup(root=bottle)))
    sponge = posed_body(world, "sponge", 1.3, 0.0, 0.8)
    universe.add(GroundedObject("sponge", OBJECT_TYPE, sponge, Sponge(root=sponge)))
    return universe


def default_grasp(context: EvaluationContext) -> GraspDescription:
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        context.manipulation_arm().end_effector,
    )


def body(universe: ObjectUniverse, name: str) -> Body:
    return universe[name].body


def entity(universe: ObjectUniverse, name: str):
    return universe[name].semantic_entity


WITNESS_POSE = HomogeneousTransformationMatrix.from_xyz_rpy(0.4, 0.0, 0.0)

EXPECTED_ACTIONS = {
    BASE_NAVIGATION_CAPABILITY_UID: (
        (("destination", "cup"),),
        lambda c, u: NavigateAction(target_location=body(u, "cup").global_pose),
    ),
    INTERACTION_NAVIGATION_CAPABILITY_UID: (
        ((PATIENT_ROLE, "drawer"),),
        lambda c, u: NavigateAction(target_location=WITNESS_POSE.to_pose()),
    ),
    CARRY_POSTURE_CAPABILITY_UID: (
        (("arm", "LEFT"),),
        lambda c, u: CarryAction(arm=Arms.LEFT),
    ),
    GRIPPER_STATE_CAPABILITY_UID: (
        (("gripper", "LEFT"), (TARGET_STATE_ROLE, "CLOSE")),
        lambda c, u: SetGripperAction(gripper=Arms.LEFT, motion=GripperState.CLOSE),
    ),
    PLACE_CAPABILITY_UID: (
        ((PATIENT_ROLE, "cup"), ("destination", "table")),
        lambda c, u: PlaceAction(
            object_designator=body(u, "cup"),
            target_location=body(u, "table").global_pose,
            arm=Arms.RIGHT,
        ),
    ),
    POURING_CAPABILITY_UID: (
        (("source", "bottle"), ("destination", "cup")),
        lambda c, u: PouringAction(
            arm=Arms.RIGHT,
            source_container=entity(u, "bottle"),
            target_container=body(u, "cup"),
        ),
    ),
    TOOL_PATH_CAPABILITY_UID: (
        (("target", "cup"),),
        lambda c, u: FollowToolCenterPointPathAction(
            target_locations=PoseTrajectory(poses=[body(u, "cup").global_pose]),
            arm=Arms.RIGHT,
        ),
    ),
    TORSO_STATE_CAPABILITY_UID: (
        ((TARGET_STATE_ROLE, "HIGH"),),
        lambda c, u: MoveTorsoAction(torso_state=TorsoState.HIGH),
    ),
    PICK_UP_CAPABILITY_UID: (
        ((PATIENT_ROLE, "cup"),),
        lambda c, u: PickUpAction(
            object_designator=body(u, "cup"),
            arm=Arms.RIGHT,
            grasp_description=default_grasp(c),
        ),
    ),
    TRANSPORT_CAPABILITY_UID: (
        ((PATIENT_ROLE, "cup"), ("destination", "table")),
        lambda c, u: TransportAction(
            object_designator=body(u, "cup"),
            target_location=body(u, "table").global_pose,
            arm=Arms.RIGHT,
        ),
    ),
    f"{ARTICULATION_CAPABILITY_UID}/OPEN": (
        (
            (INTERACTION_POINT_ROLE, "handle"),
            (PATIENT_ROLE, "drawer"),
            (TARGET_STATE_ROLE, OpenCloseState.OPEN),
        ),
        lambda c, u: OpenAction(object_designator=body(u, "handle"), arm=Arms.RIGHT),
    ),
    f"{ARTICULATION_CAPABILITY_UID}/CLOSED": (
        (
            (INTERACTION_POINT_ROLE, "handle"),
            (PATIENT_ROLE, "drawer"),
            (TARGET_STATE_ROLE, OpenCloseState.CLOSED),
        ),
        lambda c, u: CloseAction(object_designator=body(u, "handle"), arm=Arms.RIGHT),
    ),
}


@pytest.mark.parametrize("case", list(EXPECTED_ACTIONS), ids=list(EXPECTED_ACTIONS))
def test_admitted_realization_builds_the_expected_native_action(
    pr2_context, kitchen_universe, capability_initialization, case
):
    """
    On a PR2 (right arm, mobile base) each admitted realization yields exactly the
    native action a platform maintainer would have written by hand.
    """
    arguments, expected_action = EXPECTED_ACTIONS[case]
    request = ExecutionRequest(
        CapabilityRef(case.split("/")[0]), ((ACTOR_ROLE, "pr2"),) + arguments
    )
    pr2_context.witness_base_poses[("pr2", "drawer")] = WITNESS_POSE
    realization = CoraplexSkillRealization.for_evaluation_context(
        pr2_context, capability_initialization
    )

    actual = realization.realize(request, pr2_context, kitchen_universe)

    assert_same_action(actual, expected_action(pr2_context, kitchen_universe))


def test_every_reviewed_realization_binds_declared_and_required_parameters(
    capability_initialization,
):
    """
    A realization may only fill parameters its action declares, and must fill every
    parameter the action requires.
    """
    for record in capability_initialization.records:
        if not record.parameter_sources:
            continue
        action_type = _action_type(record.draft.action_class)
        declared = {
            field.name for field in dataclasses.fields(action_type) if field.init
        }
        required = {
            field.name
            for field in dataclasses.fields(action_type)
            if field.init
            and field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        }
        bound = {source.parameter for source in record.parameter_sources}
        assert bound <= declared, (record.draft.source_id, bound - declared)
        assert required <= bound, (record.draft.source_id, required - bound)
