"""
Kinematic capability layer: realizations and feasibility.

Executes capability requests by writing joint states directly into the world, and
answers capability-feasibility questions for the same capabilities: an inverse-
kinematics solve within a fixed budget followed by a collision check of the found
configuration, with base-pose candidates proposed by CRAM's costmaps when the capability
may reposition the base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING, Callable, Optional

from resym.core.model import ExecutionRequest
from resym.planning.execution.engine import (
    MissingWitnessPoseError,
    PlatformExecutionResult,
    PlatformSkillRealization,
    UnknownCapabilityError,
    UnsupportedCapabilityRealizationError,
)
from resym.core.grounding import GroundingFailure, GroundingFailureCode
from resym.platform.articulation import articulation_connection, interaction_point_body
from resym.platform.capabilities import (
    ACTOR_ROLE,
    ARTICULATION_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    PATIENT_ROLE,
    TARGET_STATE_ROLE,
    OpenCloseState,
)
from resym.platform.embodiment import ToolOrientation
from resym.platform.feasibility import CapabilityFeasibility
from resym.platform.grounding_context import EvaluationContext
from resym.platform.universe import (
    GroundedObject,
    ObjectUniverse,
    set_joint_fraction,
)
from semantic_digital_twin.collision_checking.collision_matrix import (
    CollisionCheck,
    CollisionMatrix,
)
from semantic_digital_twin.reasoning.robot_predicates import is_pose_free_for_robot
from semantic_digital_twin.spatial_computations.ik_solver import (
    MaxIterationsException,
    UnreachableException,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
    Vector3,
)

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types.spatial_types import Point3
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedom,
    )
    from semantic_digital_twin.world_description.world_entity import Body

INVERSE_KINEMATICS_ITERATION_BUDGET = 1000
"""
Iteration budget of one reachability attempt (the same budget CRAM's ``reachable``
predicate uses); exhausting it makes the query fail with a resource-limit diagnostic.
"""

COLLISION_DETECTION_DISTANCE = 0.005
"""
Bodies closer than this (meters) in the reaching configuration count as colliding.
"""

KinematicRealizationFunction = Callable[
    [EvaluationContext, ObjectUniverse, ExecutionRequest], None
]

KINEMATIC_REALIZATIONS: dict[str, KinematicRealizationFunction] = {}


def kinematic_realization(
    capability_uid: str,
) -> Callable[[KinematicRealizationFunction], KinematicRealizationFunction]:
    def decorate(
        function: KinematicRealizationFunction,
    ) -> KinematicRealizationFunction:
        KINEMATIC_REALIZATIONS[capability_uid] = function
        return function

    return decorate


@kinematic_realization(NAVIGATION_CAPABILITY_UID)
def reach_articulation_interaction(
    context: EvaluationContext, universe: ObjectUniverse, request: ExecutionRequest
) -> None:
    if context.drive_connection is None:
        raise UnsupportedCapabilityRealizationError(
            NAVIGATION_CAPABILITY_UID, context.profile.name
        )
    key = (request.argument(ACTOR_ROLE), request.argument(PATIENT_ROLE))
    if key not in context.witness_base_poses:
        raise MissingWitnessPoseError(request)
    context.drive_connection.origin = context.witness_base_poses[key]


OPEN_TARGET_FRACTION = 0.9
CLOSED_TARGET_FRACTION = 0.0


@kinematic_realization(ARTICULATION_CAPABILITY_UID)
def set_articulation_state(
    context: EvaluationContext, universe: ObjectUniverse, request: ExecutionRequest
) -> None:
    fraction = {
        OpenCloseState.OPEN: OPEN_TARGET_FRACTION,
        OpenCloseState.CLOSED: CLOSED_TARGET_FRACTION,
    }[OpenCloseState(request.argument(TARGET_STATE_ROLE))]
    articulated = universe[request.argument(PATIENT_ROLE)]
    set_joint_fraction(articulation_connection(articulated), fraction)


@dataclass
class KinematicSkillRealization(PlatformSkillRealization):
    """
    Execute capability requests by writing joint states directly into the world.
    """

    def execute(
        self,
        request: ExecutionRequest,
        context: EvaluationContext,
        universe: ObjectUniverse,
    ) -> PlatformExecutionResult:
        uid = request.capability_ref.uid
        if uid not in KINEMATIC_REALIZATIONS:
            raise UnknownCapabilityError(uid)
        if uid not in context.profile.capabilities:
            raise UnsupportedCapabilityRealizationError(uid, context.profile.name)
        KINEMATIC_REALIZATIONS[uid](context, universe, request)
        return PlatformExecutionResult.succeeded("KINEMATIC_EXECUTION_SUCCEEDED")


# %% capability feasibility

KinematicFeasibilityProcedure = Callable[
    ["KinematicFeasibility", EvaluationContext, tuple[GroundedObject, ...]], bool
]

KINEMATIC_FEASIBILITY: dict[str, KinematicFeasibilityProcedure] = {}


def kinematic_feasibility(
    capability_uid: str,
) -> Callable[[KinematicFeasibilityProcedure], KinematicFeasibilityProcedure]:
    """
    Register the procedure deciding one capability's kinematic feasibility.
    """

    def decorate(
        function: KinematicFeasibilityProcedure,
    ) -> KinematicFeasibilityProcedure:
        KINEMATIC_FEASIBILITY[capability_uid] = function
        return function

    return decorate


@kinematic_feasibility(ARTICULATION_CAPABILITY_UID)
def interaction_reachable_from_current_base(
    feasibility: KinematicFeasibility,
    context: EvaluationContext,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """
    True iff the arm reaches the patient's interaction point from where the robot
    currently stands.
    """
    actor, patient = arguments
    return _arm_reaches(
        context, actor.body.global_transform, interaction_point_body(patient)
    )


@kinematic_feasibility(NAVIGATION_CAPABILITY_UID)
def interaction_reachable_from_sampled_base(
    feasibility: KinematicFeasibility,
    context: EvaluationContext,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """
    True iff some collision-free base pose lets the arm reach the patient's interaction
    point; the first such pose becomes the execution witness.
    """
    actor, patient = arguments
    if context.drive_connection is None:
        raise GroundingFailure(
            GroundingFailureCode.UNSUPPORTED_QUERY,
            f"embodiment '{context.profile.name}' has no drive: base "
            "repositioning cannot be sampled",
        )
    target_body = interaction_point_body(patient)
    witness = feasibility.find_feasible_base_pose(context, target_body)
    if witness is None:
        return False
    context.witness_base_poses[(actor.name, patient.name)] = witness
    return True


@dataclass
class KinematicFeasibility(CapabilityFeasibility):
    """
    Reachability-based feasibility of the registered kinematic capabilities.
    """

    base_pose_sample_count: int = 32
    """
    Maximum number of costmap-ranked base-pose candidates tried per base-sampling query,
    split evenly across the distance rings.
    """

    base_pose_distance_factors: tuple[float, ...] = (0.66, 0.8)
    """
    Ring distances to try, as fractions of the manipulation arm's approximate length.

    Score-ranked costmap cells cluster at one ring's peak, so multiple rings hedge
    across approach distances.
    """

    def feasible(
        self,
        capability_uid: str,
        arguments: tuple[GroundedObject, ...],
        context: EvaluationContext,
    ) -> bool:
        if capability_uid not in KINEMATIC_FEASIBILITY:
            raise GroundingFailure(
                GroundingFailureCode.UNSUPPORTED_QUERY,
                f"no kinematic feasibility procedure for capability "
                f"'{capability_uid}'",
            )
        return KINEMATIC_FEASIBILITY[capability_uid](self, context, arguments)

    def find_feasible_base_pose(
        self,
        context: EvaluationContext,
        target_body: Body,
    ) -> Optional[HomogeneousTransformationMatrix]:
        """
        Try base-pose candidates around the target, best first; return the first pose
        that is collision-free and from which the arm reaches it.

        Candidates come from CRAM's costmaps — occupancy (obstacles inflated by the base
        footprint) merged with rings at several arm-length-relative distances — each
        ring iterated highest-score first, facing the target. The default (non-random)
        costmap sampling keeps the query deterministic. Each candidate is still verified
        by the exact-footprint pose check and the binary reachability query; the
        costmaps only propose.
        """
        # Imported here, not at the top of the module: the costmap module's
        # import chain reaches ROS message packages, and this module must stay
        # importable on a ROS-less host for every capability that needs no
        # costmap.
        from coraplex.datastructures.dataclasses import Context as CoraplexContext
        from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap

        plan_context = CoraplexContext(world=context.world, robot=context.robot)
        occupancy = OccupancyCostmap.default_map(plan_context, target_body.global_pose)
        arm_length = float(context.manipulation_arm().approximate_length())
        factors = self.base_pose_distance_factors
        samples_per_ring = max(1, self.base_pose_sample_count // len(factors))
        for factor in factors:
            ring = RingCostmap(
                resolution=occupancy.resolution,
                width=occupancy.width,
                height=occupancy.height,
                std=15,
                distance=arm_length * factor,
                world=context.world,
                origin=occupancy.origin,
            )
            costmap = occupancy & ring
            costmap.number_of_samples = samples_per_ring
            for candidate in costmap:
                if not is_pose_free_for_robot(context.robot, candidate):
                    continue
                world_T_base = candidate.to_homogeneous_matrix()
                if _arm_reaches(context, world_T_base, target_body):
                    return world_T_base
        return None


def _arm_reaches(
    context: EvaluationContext,
    world_T_base: HomogeneousTransformationMatrix,
    target_body: Body,
) -> bool:
    """
    Reachability from a hypothetical base pose: the target is the target body's pose
    expressed in the base frame.

    Two CRAM checks in sequence: ``compute_inverse_kinematics`` must find
    a joint solution within the budget, and the collision check must
    find that solution collision-free (the target and its parent are
    expected contacts of a grasp and therefore ignored). A proven-unreachable
    target or a colliding configuration returns false; exhausting the inverse-
    kinematics budget raises a grounding failure.
    """
    base_T_target = world_T_base.inverse() @ target_body.global_transform
    base_P_target = base_T_target.to_position()
    rotation: Optional[RotationMatrix] = None  # None targets the identity
    if context.profile.tool_orientation is ToolOrientation.APPROACH_ALIGNED:
        rotation = _approach_aligned_rotation(base_P_target)
    arm = context.manipulation_arm()
    base_body = context.robot.root
    try:
        solution = context.world.compute_inverse_kinematics(
            root=base_body,
            tip=arm.end_effector.tool_frame,
            target=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=base_P_target,
                rotation_matrix=rotation,
                reference_frame=base_body,
            ),
            max_iterations=INVERSE_KINEMATICS_ITERATION_BUDGET,
        )
    except UnreachableException:
        return False
    except MaxIterationsException:
        raise GroundingFailure(
            GroundingFailureCode.RESOURCE_LIMIT,
            f"IK iteration budget ({INVERSE_KINEMATICS_ITERATION_BUDGET}) "
            "exhausted without a Boolean result",
        )
    return not _reaching_configuration_collides(
        context, world_T_base, solution, target_body
    )


def _approach_aligned_rotation(base_P_target: Point3) -> Optional[RotationMatrix]:
    """
    Tool orientation for a fixed yaw-pitch-pitch-pitch-roll arm: the tool z-axis points
    along the horizontal direction from the arm base to the target, which is exactly the
    set of approach directions such an arm can attain.

    The remaining roll is fixed arbitrarily (tool y-axis up); the wrist roll joint
    absorbs it.
    """
    horizontal = base_P_target.to_np()[:2]
    if float(np.linalg.norm(horizontal)) < 1e-6:
        return None  # target directly above the base: keep identity
    return RotationMatrix.from_vectors(
        y=Vector3.Z(),
        z=Vector3(horizontal[0], horizontal[1], 0.0),
    )


def _reaching_configuration_collides(
    context: EvaluationContext,
    world_T_base: HomogeneousTransformationMatrix,
    solution: dict[DegreeOfFreedom, float],
    target_body: Body,
) -> bool:
    """
    Whether the arms, with the robot at the candidate base pose and the manipulation arm
    in the reaching configuration, collide with the world.

    The check runs CRAM's collision detector over an explicit matrix of arm bodies
    against environment bodies; the target and its parent are expected contacts of a
    grasp and excluded. (The whole-robot :func:`robot_in_collision` cannot be used here:
    it builds checks for every robot body, and geometry-less mount bodies make it
    raise.) The candidate pose and joint solution are written into the world state
    inside a reset context and restored afterwards.
    """
    world = context.world
    with world.reset_state_context():
        if context.drive_connection is not None:
            # a fixed base already stands at world_T_base
            context.drive_connection.origin = world_T_base
        for degree_of_freedom, position in solution.items():
            world.state[degree_of_freedom.id].position = position
        world.notify_state_change()
        contacts = world.collision_manager.collision_detector.check_collisions(
            _arms_against_environment(context, target_body)
        )
        return any(
            closest_points.distance <= COLLISION_DETECTION_DISTANCE
            for closest_points in contacts.contacts
        )


def _arms_against_environment(
    context: EvaluationContext, target_body: Body
) -> CollisionMatrix:
    """
    Collision checks between every body of *both* arms and every environment body,
    excluding the robot itself and the grasped target with its parent.

    The non-manipulating arm is included deliberately: it hangs in its
    parked configuration, and a witness pose that rams it into the
    neighbouring furniture is just as infeasible.
    """
    arm_bodies = [
        body
        for arm in context.arms()
        for body in arm.bodies_with_collision
        if body is not arm.root  # the chain root is the mount, not the arm
    ]
    excluded = set(context.robot.bodies)
    excluded.add(target_body)
    excluded.add(target_body.parent_connection.parent)
    environment_bodies = [
        body for body in context.world.bodies_with_collision if body not in excluded
    ]
    return CollisionMatrix(
        collision_checks={
            CollisionCheck.create_and_validate(
                arm_body, environment_body, distance=COLLISION_DETECTION_DISTANCE
            )
            for arm_body in arm_bodies
            for environment_body in environment_bodies
        }
    )
