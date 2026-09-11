"""Reviewed Boolean queries over the Semantic Digital Twin.

Each procedure computes one predicate for one tuple of objects. A query
returns :class:`bool` when the world provides enough evidence. Missing
knowledge, unsupported queries, and resource exhaustion raise a structured
:class:`~resym.core.grounding.GroundingFailure` instead of entering the
planning truth domain.

Reachability is two CRAM checks in sequence: an inverse-kinematics solve
(:meth:`World.compute_inverse_kinematics`) and a collision-detector check
of the found configuration over an explicit arm-versus-environment
matrix; the procedures here only phrase the question (which pose, which
kinematic chain, which expected contacts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING, Callable, Optional

from resym.platform.articulation import (
    articulation_connection,
    interaction_point_belongs_to,
    interaction_point_body,
)
from resym.platform.embodiment import EmbodimentProfile, ToolOrientation
from resym.core.grounding import GroundingFailure, GroundingFailureCode
from resym.core.model import EvaluatorSpec, SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from resym.platform.universe import (
    GroundedObject,
    ObjectUniverse,
    joint_fraction,
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

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)

INVERSE_KINEMATICS_ITERATION_BUDGET = 1000
"""Iteration budget of one reachability attempt (the same budget CRAM's
``reachable`` predicate uses); exhausting it makes the query fail with a
resource-limit diagnostic."""

COLLISION_DETECTION_DISTANCE = 0.005
"""Bodies closer than this (meters) in the reaching configuration count as
colliding."""

if TYPE_CHECKING:
    from resym.platform.grounding_catalog import GroundingFactoryCatalog
    from semantic_digital_twin.robots.robot_parts import Arm
    from semantic_digital_twin.spatial_types.spatial_types import Point3
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.connections import WheeledDrive
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedom,
    )
    from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class EvaluationBudgets:
    """Deterministic evaluation parameters, kept apart from the world handles
    so experiment configurations can vary them without touching the scene."""

    open_fraction: float = 0.4
    """Fraction of the joint range above which a drawer counts as opened."""

    base_pose_sample_count: int = 32
    """Maximum number of costmap-ranked base-pose candidates tried per
    ``openable`` query, split evenly across the distance rings."""

    base_pose_distance_factors: tuple[float, ...] = (0.66, 0.8)
    """Ring distances to try, as fractions of the manipulation arm's
    approximate length. Score-ranked costmap cells cluster at one ring's
    peak, so multiple rings hedge across approach distances the way the
    former fixed radii did."""


@dataclass
class EvaluationContext:
    """Everything a truth procedure may consult besides its arguments: the
    world handles, the embodiment's declared capability surface, the
    deterministic evaluation budgets, and the witness poses that grounding
    hands over to execution."""

    world: World
    """The kinematic world truth is computed against."""

    robot: AbstractRobot
    """The robot semantic annotation."""

    profile: EmbodimentProfile
    """What this embodiment can do."""

    grounding_catalog: GroundingFactoryCatalog | None = None
    """Approved factory view used by predicates with a grounding plan."""

    budgets: EvaluationBudgets = field(default_factory=EvaluationBudgets)
    """Evaluation parameters (thresholds, sampling resolution)."""

    witness_base_poses: dict[tuple[str, str], HomogeneousTransformationMatrix] = field(
        default_factory=dict
    )
    """Feasible base pose per (robot, drawer) found by ``openable``, reused by
    execution."""

    @property
    def drive_connection(self) -> Optional[WheeledDrive]:
        """The drive that moves the base, read off the robot description;
        ``None`` on a fixed arm."""
        return None if self.robot is None else self.robot.drive

    def arms(self) -> list[Arm]:
        """Every arm of the robot, from its robot-part annotation tree."""
        return self.robot.get_arms()

    def manipulation_arm(self) -> Arm:
        """The arm used for manipulation; deterministically the robot's right
        arm where there is a choice."""
        right_arm = self.robot.get_right_arm_if_specified()
        if right_arm is not None:
            return right_arm
        return self.arms()[0]


EvaluatorFunction = Callable[
    [EvaluationContext, ObjectUniverse, tuple[GroundedObject, ...]], bool
]

EVALUATORS: dict[str, EvaluatorFunction] = {}
"""Registry mapping evaluator names (as persisted in the library) to
procedures."""

EVALUATOR_SPECS: dict[str, EvaluatorSpec] = {}
"""Trusted, versioned ABI beside every evaluator implementation."""


class UnknownEvaluatorError(Exception):
    """Raised when a library symbol references an evaluator that is not
    registered."""

    def __init__(self, name: str):
        super().__init__(f"No evaluator registered under '{name}'.")


def evaluator(
    name: str,
    parameter_types: tuple[SymbolType, ...],
) -> Callable[[EvaluatorFunction], EvaluatorFunction]:
    """Register a truth procedure under the name the library persists."""

    def decorate(function: EvaluatorFunction) -> EvaluatorFunction:
        EVALUATORS[name] = function
        EVALUATOR_SPECS[name] = EvaluatorSpec(
            name=name,
            parameter_types=parameter_types,
        )
        return function

    return decorate


def resolve_evaluator(name: str) -> EvaluatorFunction:
    if name not in EVALUATORS:
        raise UnknownEvaluatorError(name)
    return EVALUATORS[name]


@evaluator("handle_of", (HANDLE_TYPE, DRAWER_TYPE))
def handle_of(
    context: EvaluationContext,
    universe: ObjectUniverse,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """True iff the handle is the one mounted on the drawer (from the scene
    graph)."""
    handle, drawer = arguments
    return interaction_point_belongs_to(handle, drawer)


@evaluator("drawer_closed", (DRAWER_TYPE,))
def drawer_closed(
    context: EvaluationContext,
    universe: ObjectUniverse,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """True iff the drawer joint sits below the opened fraction of its
    range."""
    (drawer,) = arguments
    return not _drawer_open_fraction_reached(context, drawer)


@evaluator("drawer_opened", (DRAWER_TYPE,))
def drawer_opened(
    context: EvaluationContext,
    universe: ObjectUniverse,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """True iff the drawer joint sits at or above the opened fraction of its
    range."""
    (drawer,) = arguments
    return _drawer_open_fraction_reached(context, drawer)


@evaluator("ready_to_open", (ROBOT_TYPE, DRAWER_TYPE))
def ready_to_open(
    context: EvaluationContext,
    universe: ObjectUniverse,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """True iff the arm reaches the drawer's handle by inverse kinematics from
    the base pose the robot currently stands at.

    Execution recomputes this symbolic precondition immediately before
    dispatch, so a stale planning snapshot cannot authorize the action.
    """
    robot, drawer = arguments
    world_T_base = robot.body.global_transform
    return _arm_reaches(context, world_T_base, interaction_point_body(drawer))


@evaluator("openable", (ROBOT_TYPE, DRAWER_TYPE))
def openable(
    context: EvaluationContext,
    universe: ObjectUniverse,
    arguments: tuple[GroundedObject, ...],
) -> bool:
    """True iff some collision-free base pose near the drawer's handle lets the
    manipulation arm reach the handle by inverse kinematics.

    The first feasible pose is recorded as the witness that execution
    navigates to. Exhausting a lower-level query budget stops grounding.
    """
    robot, drawer = arguments
    if context.drive_connection is None:
        raise GroundingFailure(
            GroundingFailureCode.UNSUPPORTED_QUERY,
            f"embodiment '{context.profile.name}' has no drive: base "
            "repositioning cannot be sampled",
        )
    handle_body = interaction_point_body(drawer)
    witness = _find_feasible_base_pose(context, robot, handle_body)
    if witness is None:
        return False
    context.witness_base_poses[(robot.name, drawer.name)] = witness
    return True


def _drawer_open_fraction_reached(
    context: EvaluationContext, drawer: GroundedObject
) -> bool:
    return (
        joint_fraction(articulation_connection(drawer)) >= context.budgets.open_fraction
    )


def _find_feasible_base_pose(
    context: EvaluationContext,
    robot: GroundedObject,
    handle_body: Body,
) -> Optional[HomogeneousTransformationMatrix]:
    """Try base-pose candidates around the handle, best first; return the first
    pose that is collision-free and from which the arm reaches the handle.

    Candidates come from CRAM's costmaps — occupancy (obstacles inflated by
    the base footprint) merged with rings at several arm-length-relative
    distances — each ring iterated highest-score first, facing the handle.
    The default (non-random) costmap sampling keeps ``openable``
    deterministic. Each candidate is still verified by the exact-footprint
    pose check and the binary reachability query; the costmaps only propose.
    """
    # Imported here, not at the top of the module: the costmap module's
    # import chain reaches ROS message packages, and this module must stay
    # importable on a ROS-less host for every evaluator that needs no costmap.
    from coraplex.datastructures.dataclasses import Context as CoraplexContext
    from coraplex.locations.costmaps import OccupancyCostmap, RingCostmap

    plan_context = CoraplexContext(world=context.world, robot=context.robot)
    occupancy = OccupancyCostmap.default_map(plan_context, handle_body.global_pose)
    arm_length = float(context.manipulation_arm().approximate_length())
    factors = context.budgets.base_pose_distance_factors
    samples_per_ring = max(1, context.budgets.base_pose_sample_count // len(factors))
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
            if _arm_reaches(context, world_T_base, handle_body):
                return world_T_base
    return None


def _arm_reaches(
    context: EvaluationContext,
    world_T_base: HomogeneousTransformationMatrix,
    handle_body: Body,
) -> bool:
    """Reachability from a hypothetical base pose: the target is the handle
    pose expressed in the base frame.

    Two CRAM checks in sequence: ``compute_inverse_kinematics`` must find
    a joint solution within the budget, and the collision check must
    find that solution collision-free (the handle and its drawer are
    expected contacts of a grasp and therefore ignored). A proven-unreachable
    target or a colliding configuration returns false; exhausting the inverse-
    kinematics budget raises a grounding failure.
    """
    base_T_handle = world_T_base.inverse() @ handle_body.global_transform
    base_P_handle = base_T_handle.to_position()
    rotation: Optional[RotationMatrix] = None  # None targets the identity
    if context.profile.tool_orientation is ToolOrientation.APPROACH_ALIGNED:
        rotation = _approach_aligned_rotation(base_P_handle)
    arm = context.manipulation_arm()
    base_body = context.robot.root
    try:
        solution = context.world.compute_inverse_kinematics(
            root=base_body,
            tip=arm.end_effector.tool_frame,
            target=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=base_P_handle,
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
        context, world_T_base, solution, handle_body
    )


def _approach_aligned_rotation(base_P_handle: Point3) -> Optional[RotationMatrix]:
    """Tool orientation for a fixed yaw-pitch-pitch-pitch-roll arm: the tool
    z-axis points along the horizontal direction from the arm base to the
    target, which is exactly the set of approach directions such an arm can
    attain. The remaining roll is fixed arbitrarily (tool y-axis up); the
    wrist roll joint absorbs it."""
    horizontal = base_P_handle.to_np()[:2]
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
    handle_body: Body,
) -> bool:
    """Whether the arms, with the robot at the candidate base pose and the
    manipulation arm in the reaching configuration, collide with the world.

    The check runs CRAM's collision detector over an explicit matrix of
    arm bodies against environment bodies; the handle and its drawer are
    expected contacts of a grasp and excluded. (The whole-robot
    :func:`robot_in_collision` cannot be used here: it builds checks for
    every robot body, and geometry-less mount bodies such as Tracy's
    ``left_arm_mount`` make it raise.) The candidate pose and joint
    solution are written into the world state inside a reset context and
    restored afterwards.
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
            _arms_against_environment(context, handle_body)
        )
        return any(
            closest_points.distance <= COLLISION_DETECTION_DISTANCE
            for closest_points in contacts.contacts
        )


def _arms_against_environment(
    context: EvaluationContext, handle_body: Body
) -> CollisionMatrix:
    """Collision checks between every body of *both* arms and every environment
    body, excluding the robot itself and the grasped handle with its drawer.

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
    excluded.add(handle_body)
    excluded.add(handle_body.parent_connection.parent)
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
