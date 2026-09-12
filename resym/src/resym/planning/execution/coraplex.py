"""
Execute capability requests through Coraplex action designators.

The small capability-to-action mapping is an internal implementation detail of this
realization. Coraplex remains responsible for its native action preconditions, motion
execution and native postconditions. reSym receives a structured outcome and
independently verifies the operator effects afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING, Callable, Mapping

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    DetectionTechnique,
    VerticalAlignment,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.exceptions import ConditionNotSatisfied, MotionDidNotFinish
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.container import CloseAction, OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from resym.core.model import ExecutionRequest
from resym.platform.capabilities import (
    ACTOR_ROLE,
    ARM_POSTURE_CAPABILITY_UID,
    ARTICULATION_CAPABILITY_UID,
    BASE_NAVIGATION_CAPABILITY_UID,
    CARRY_POSTURE_CAPABILITY_UID,
    CUTTING_CAPABILITY_UID,
    DETECTION_CAPABILITY_UID,
    ELEVATOR_NAVIGATION_CAPABILITY_UID,
    GRASP_CAPABILITY_UID,
    GRIPPER_STATE_CAPABILITY_UID,
    MIXING_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    PATIENT_ROLE,
    PICK_UP_CAPABILITY_UID,
    PLACE_CAPABILITY_UID,
    POURING_CAPABILITY_UID,
    REACH_CAPABILITY_UID,
    TARGET_STATE_ROLE,
    TOOL_PATH_CAPABILITY_UID,
    TORSO_STATE_CAPABILITY_UID,
    TRANSPORT_CAPABILITY_UID,
    VISUAL_ATTENTION_CAPABILITY_UID,
    WIPING_CAPABILITY_UID,
    OpenCloseState,
)
from resym.platform.grounding_context import EvaluationContext
from resym.platform.articulation import interaction_point_body
from resym.platform.universe import ObjectUniverse
from resym.planning.events import PipelineEvent, PipelineEventSink, emit_event
from resym.planning.execution.engine import (
    MissingWitnessPoseError,
    PlatformExecutionResult,
    PlatformSkillRealization,
    UnknownCapabilityError,
)

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.base import ActionDescription
    from semantic_digital_twin.world_description.world_entity import Body


CoraplexActionFactory = Callable[
    [ExecutionRequest, EvaluationContext, ObjectUniverse], "ActionDescription"
]


@dataclass
class CoraplexSkillRealization(PlatformSkillRealization):
    """
    Resolve and execute supported requests using native Coraplex actions.
    """

    plan_context: Context
    """
    The Coraplex plan context used by native action designators.
    """

    external_collision_avoidance: bool = False
    """
    Enable Coraplex's global external-collision goal.

    Container actions intentionally contact their handle and currently manage those
    contacts in their own motions. The global goal therefore remains an explicit
    platform option until its expected-contact policy is compatible with those motions;
    this is not replaced by a reSym guard.
    """

    event_sink: PipelineEventSink | None = None
    """
    Optional observer shared with the task pipeline by visualization demos.
    """

    capability_handlers: Mapping[str, CoraplexActionFactory] = field(
        default_factory=lambda: default_coraplex_capability_handlers()
    )
    """
    Platform-owned dispatch from a capability UID to a native action.
    """

    @classmethod
    def for_evaluation_context(
        cls,
        context: EvaluationContext,
        event_sink: PipelineEventSink | None = None,
    ) -> CoraplexSkillRealization:
        """
        Build the backend on the same world and robot the grounding queries observe.
        """
        return cls(
            plan_context=Context(world=context.world, robot=context.robot),
            event_sink=event_sink,
        )

    def execute(
        self,
        request: ExecutionRequest,
        context: EvaluationContext,
        universe: ObjectUniverse,
    ) -> PlatformExecutionResult:
        try:
            designator = self.realize(request, context, universe)
        except UnknownCapabilityError as error:
            return PlatformExecutionResult.unsupported(
                "CORAPLEX_CAPABILITY_NOT_IMPLEMENTED", str(error)
            )
        except MissingWitnessPoseError as error:
            return PlatformExecutionResult.rejected(
                "CORAPLEX_CONTEXT_MISSING", str(error)
            )
        except MissingHandleArgumentError as error:
            return PlatformExecutionResult.rejected(
                "CORAPLEX_REQUEST_ARGUMENT_MISSING", str(error)
            )
        except InvalidCapabilityArgumentError as error:
            return PlatformExecutionResult.rejected(
                "CORAPLEX_REQUEST_ARGUMENT_INVALID", str(error)
            )
        except KeyError as error:
            return PlatformExecutionResult.rejected(
                "CORAPLEX_REQUEST_ARGUMENT_MISSING",
                f"request has no role '{error.args[0]}'",
            )
        emit_event(
            self.event_sink,
            PipelineEvent.CORAPLEX_ACTION_SELECTED,
            capability_ref=to_json(request.capability_ref),
            action=type(designator).__name__,
        )
        plan_node = sequential([designator], self.plan_context)
        try:
            with simulated_robot(collision_avoidance=self.external_collision_avoidance):
                plan_node.perform()
        except ConditionNotSatisfied as error:
            if error.pre_condition:
                return PlatformExecutionResult.rejected(
                    "CORAPLEX_PRECONDITION_FAILED", str(error)
                )
            return PlatformExecutionResult.failed(
                "CORAPLEX_POSTCONDITION_FAILED", str(error)
            )
        except CollisionViolatedError as error:
            return PlatformExecutionResult.failed(
                "CORAPLEX_COLLISION_ABORTED", str(error)
            )
        except MotionDidNotFinish as error:
            return PlatformExecutionResult.failed("CORAPLEX_MOTION_FAILED", str(error))
        return PlatformExecutionResult.succeeded("CORAPLEX_EXECUTION_SUCCEEDED")

    def realize(
        self,
        request: ExecutionRequest,
        context: EvaluationContext,
        universe: ObjectUniverse,
    ) -> ActionDescription:
        """
        Instantiate one native action from the abstract request.
        """
        capability_uid = request.capability_ref.uid
        handler = self.capability_handlers.get(capability_uid)
        if handler is None:
            raise UnknownCapabilityError(capability_uid)
        return handler(request, context, universe)


def _arm_designation(context: EvaluationContext) -> Arms:
    """
    The Coraplex designation of the arm reachability was proven for — the same selection
    rule as :meth:`EvaluationContext.manipulation_arm`.
    """
    if context.manipulation_arm() is context.robot.get_left_arm_if_specified():
        return Arms.LEFT
    return Arms.RIGHT


def _handle_body(universe: ObjectUniverse, request: ExecutionRequest) -> Body:
    """
    Resolve an explicit interaction point or derive it from the patient.
    """
    arguments = request.argument_map
    if "interaction_point" in arguments:
        return universe[arguments["interaction_point"]].body
    if PATIENT_ROLE not in arguments:
        raise MissingHandleArgumentError(request)
    return interaction_point_body(universe[arguments[PATIENT_ROLE]])


def default_coraplex_capability_handlers() -> dict[str, CoraplexActionFactory]:
    """
    Capabilities bundled with the current Coraplex adapter.
    """
    return {
        NAVIGATION_CAPABILITY_UID: _navigation_action,
        ARTICULATION_CAPABILITY_UID: _articulation_action,
        BASE_NAVIGATION_CAPABILITY_UID: _base_navigation_action,
        VISUAL_ATTENTION_CAPABILITY_UID: _visual_attention_action,
        DETECTION_CAPABILITY_UID: _detection_action,
        REACH_CAPABILITY_UID: _reach_action,
        GRASP_CAPABILITY_UID: _grasp_action,
        PICK_UP_CAPABILITY_UID: _pick_up_action,
        PLACE_CAPABILITY_UID: _place_action,
        TRANSPORT_CAPABILITY_UID: _transport_action,
        GRIPPER_STATE_CAPABILITY_UID: _gripper_state_action,
        ARM_POSTURE_CAPABILITY_UID: _arm_posture_action,
        TORSO_STATE_CAPABILITY_UID: _torso_state_action,
        CARRY_POSTURE_CAPABILITY_UID: _carry_posture_action,
        TOOL_PATH_CAPABILITY_UID: _tool_path_action,
        MIXING_CAPABILITY_UID: _mixing_action,
        POURING_CAPABILITY_UID: _pouring_action,
        CUTTING_CAPABILITY_UID: _cutting_action,
        WIPING_CAPABILITY_UID: _wiping_action,
        ELEVATOR_NAVIGATION_CAPABILITY_UID: _elevator_navigation_action,
    }


def _navigation_action(
    request: ExecutionRequest,
    context: EvaluationContext,
    universe: ObjectUniverse,
) -> ActionDescription:
    key = (request.argument(ACTOR_ROLE), request.argument(PATIENT_ROLE))
    if key not in context.witness_base_poses:
        raise MissingWitnessPoseError(request)
    return NavigateAction(target_location=context.witness_base_poses[key].to_pose())


def _articulation_action(
    request: ExecutionRequest,
    context: EvaluationContext,
    universe: ObjectUniverse,
) -> ActionDescription:
    action_type = {
        OpenCloseState.OPEN: OpenAction,
        OpenCloseState.CLOSED: CloseAction,
    }.get(request.argument(TARGET_STATE_ROLE))
    if action_type is None:
        raise UnknownCapabilityError(request.capability_ref.uid)
    return action_type(
        object_designator=_handle_body(universe, request),
        arm=_arm_designation(context),
    )


def _grounded_role(universe: ObjectUniverse, request: ExecutionRequest, role: str):
    try:
        return universe[request.argument(role)]
    except KeyError as error:
        raise InvalidCapabilityArgumentError(
            request, f"role '{role}' does not resolve to a world object"
        ) from error


def _body_role(universe: ObjectUniverse, request: ExecutionRequest, role: str):
    return _grounded_role(universe, request, role).body


def _entity_role(universe: ObjectUniverse, request: ExecutionRequest, role: str):
    grounded = _grounded_role(universe, request, role)
    if grounded.semantic_entity is None:
        raise InvalidCapabilityArgumentError(
            request, f"role '{role}' has no CRAM semantic annotation"
        )
    return grounded.semantic_entity


def _object_pose(universe: ObjectUniverse, request: ExecutionRequest, role: str):
    return _body_role(universe, request, role).global_pose


def _default_grasp(context: EvaluationContext) -> GraspDescription:
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        context.manipulation_arm().end_effector,
    )


def _arm_constant(request: ExecutionRequest, role: str = "arm") -> Arms:
    try:
        return Arms[request.argument(role)]
    except (KeyError, ValueError) as error:
        raise InvalidCapabilityArgumentError(
            request, f"role '{role}' must be LEFT, RIGHT, or BOTH"
        ) from error


def _base_navigation_action(request, context, universe):
    return NavigateAction(
        target_location=_object_pose(universe, request, "destination")
    )


def _visual_attention_action(request, context, universe):
    from coraplex.robot_plans.actions.core.navigation import LookAtAction

    return LookAtAction(target=_object_pose(universe, request, "target"))


def _detection_action(request, context, universe):
    from coraplex.robot_plans.actions.core.misc import DetectAction

    arguments = request.argument_map
    if "region" in arguments:
        return DetectAction(
            technique=DetectionTechnique.REGION,
            region=_entity_role(universe, request, "region"),
        )
    if "target" in arguments:
        return DetectAction(
            technique=DetectionTechnique.TYPES,
            object_sem_annotation=type(_entity_role(universe, request, "target")),
        )
    raise InvalidCapabilityArgumentError(
        request, "object detection needs either a target or region role"
    )


def _reach_action(request, context, universe):
    from coraplex.robot_plans.actions.core.pick_up import ReachAction

    target = _body_role(universe, request, "target")
    return ReachAction(
        target_pose=target.global_pose,
        arm=_arm_designation(context),
        grasp_description=_default_grasp(context),
        object_designator=target,
    )


def _grasp_action(request, context, universe):
    from coraplex.robot_plans.actions.core.pick_up import GraspingAction

    return GraspingAction(
        object_designator=_body_role(universe, request, PATIENT_ROLE),
        arm=_arm_designation(context),
        grasp_description=_default_grasp(context),
    )


def _pick_up_action(request, context, universe):
    from coraplex.robot_plans.actions.core.pick_up import PickUpAction

    return PickUpAction(
        object_designator=_body_role(universe, request, PATIENT_ROLE),
        arm=_arm_designation(context),
        grasp_description=_default_grasp(context),
    )


def _place_action(request, context, universe):
    from coraplex.robot_plans.actions.core.placing import PlaceAction

    return PlaceAction(
        object_designator=_body_role(universe, request, PATIENT_ROLE),
        target_location=_object_pose(universe, request, "destination"),
        arm=_arm_designation(context),
    )


def _transport_action(request, context, universe):
    from coraplex.robot_plans.actions.composite.transporting import (
        PickAndPlaceAction,
        TransportAction,
    )

    arguments = {
        "object_designator": _body_role(universe, request, PATIENT_ROLE),
        "target_location": _object_pose(universe, request, "destination"),
        "arm": _arm_designation(context),
    }
    if context.drive_connection is not None:
        return TransportAction(**arguments, grasp_description=_default_grasp(context))
    return PickAndPlaceAction(**arguments, grasp_description=_default_grasp(context))


def _gripper_state_action(request, context, universe):
    from coraplex.robot_plans.actions.core.robot_body import SetGripperAction

    try:
        state = {
            OpenCloseState.OPEN: GripperState.OPEN,
            OpenCloseState.CLOSED: GripperState.CLOSE,
        }[request.argument(TARGET_STATE_ROLE)]
    except KeyError as error:
        raise InvalidCapabilityArgumentError(
            request, "target_state must be OPEN or CLOSED"
        ) from error
    return SetGripperAction(gripper=_arm_constant(request, "gripper"), motion=state)


def _arm_posture_action(request, context, universe):
    from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction

    return ParkArmsAction(arm=_arm_constant(request))


def _torso_state_action(request, context, universe):
    from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction

    try:
        state = TorsoState[request.argument(TARGET_STATE_ROLE)]
    except (KeyError, ValueError) as error:
        raise InvalidCapabilityArgumentError(
            request, "target_state must be LOW, MID, or HIGH"
        ) from error
    return MoveTorsoAction(torso_state=state)


def _carry_posture_action(request, context, universe):
    from coraplex.robot_plans.actions.core.robot_body import CarryAction

    return CarryAction(arm=_arm_constant(request))


def _tool_path_action(request, context, universe):
    from coraplex.robot_plans.actions.core.robot_body import (
        FollowToolCenterPointPathAction,
    )

    return FollowToolCenterPointPathAction(
        target_locations=PoseTrajectory(
            poses=[_object_pose(universe, request, "target")]
        ),
        arm=_arm_designation(context),
    )


def _mixing_action(request, context, universe):
    from coraplex.robot_plans.actions.composite.tool_based import MixingAction

    return MixingAction(
        arm=_arm_designation(context),
        tool=_entity_role(universe, request, "tool"),
        container=_body_role(universe, request, PATIENT_ROLE),
    )


def _pouring_action(request, context, universe):
    from coraplex.robot_plans.actions.composite.tool_based import PouringAction

    return PouringAction(
        arm=_arm_designation(context),
        source_container=_entity_role(universe, request, "source"),
        target_container=_body_role(universe, request, "destination"),
    )


def _cutting_action(request, context, universe):
    from coraplex.robot_plans.actions.composite.tool_based import CuttingAction

    return CuttingAction(
        arm=_arm_designation(context),
        tool=_entity_role(universe, request, "tool"),
        object_to_cut=_body_role(universe, request, PATIENT_ROLE),
    )


def _wiping_action(request, context, universe):
    from coraplex.robot_plans.actions.composite.tool_based import WipingAction

    return WipingAction(
        arm=_arm_designation(context),
        tool=_entity_role(universe, request, "tool"),
        surface=_body_role(universe, request, "surface"),
    )


def _elevator_navigation_action(request, context, universe):
    from coraplex.robot_plans.actions.core.navigation import ElevatorNavigation

    return ElevatorNavigation(
        elevator=_entity_role(universe, request, "elevator"),
        target_floor=_entity_role(universe, request, "target_floor"),
    )


class MissingHandleArgumentError(Exception):
    """
    Raised when neither an interaction point nor articulated part exists.
    """

    def __init__(self, request: ExecutionRequest):
        super().__init__(f"No interaction point or patient in {request}.")


class InvalidCapabilityArgumentError(Exception):
    """
    A semantic request role cannot be adapted to a native action argument.
    """

    def __init__(self, request: ExecutionRequest, reason: str):
        super().__init__(f"Invalid {request.capability_ref.uid} request: {reason}.")
