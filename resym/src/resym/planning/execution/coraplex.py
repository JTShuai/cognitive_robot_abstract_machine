"""
Execute capability requests through Coraplex action designators.

The small capability-to-action mapping is an internal implementation detail of this
realization. Coraplex remains responsible for its native action preconditions, motion
execution and native postconditions. reSym receives a structured outcome and
independently verifies the operator effects afterwards.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import (
    TYPE_CHECKING,
    Callable,
    Mapping,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.exceptions import ConditionNotSatisfied, MotionDidNotFinish
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.view_manager import ViewManager
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from krrood.adapters.json_serializer import to_json
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from resym.core.capability_model import ExecutionRequest
from resym.platform.coraplex_catalog import (
    CoraplexCapabilityInitialization,
    CoraplexCapabilityRegistration,
    NoApplicableRealizationError,
    applicable_registration,
    available_coraplex_capabilities,
    robot_resources,
)
from resym.platform.coraplex_realizations import (
    CapabilityReviewStatus,
    ContextValue,
    ParameterSource,
    ParameterSourceKind,
)
from resym.platform.grounding_context import EvaluationContext
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
        default_factory=dict
    )
    """
    Dispatch from a capability UID to a native action; empty until admitted realizations
    are supplied.
    """

    initialization: CoraplexCapabilityInitialization | None = None
    """
    The reviewed catalog the handlers were built from, when they were.
    """

    @classmethod
    def for_evaluation_context(
        cls,
        context: EvaluationContext,
        initialization: CoraplexCapabilityInitialization,
        event_sink: PipelineEventSink | None = None,
    ) -> CoraplexSkillRealization:
        """
        Build the backend on the same world and robot the grounding queries observe,
        executing the realizations the initialization admits.
        """
        return cls(
            plan_context=Context(world=context.world, robot=context.robot),
            event_sink=event_sink,
            capability_handlers=default_coraplex_capability_handlers(initialization),
            initialization=initialization,
        )

    def available_capabilities(self, robot: AbstractRobot) -> frozenset[str]:
        if self.initialization is None:
            return frozenset(self.capability_handlers)
        return available_coraplex_capabilities(robot, self.initialization)

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
        except MissingRoleArgumentError as error:
            return PlatformExecutionResult.rejected(
                "CORAPLEX_REQUEST_ARGUMENT_MISSING", str(error)
            )
        except NoApplicableRealizationError as error:
            return PlatformExecutionResult.rejected(
                "CORAPLEX_NO_APPLICABLE_REALIZATION", str(error)
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
    The Coraplex designation of the manipulation arm, by Coraplex's own arm lookup.
    """
    arm = context.manipulation_arm()
    for designation in (Arms.LEFT, Arms.RIGHT):
        if ViewManager.get_arm_view(designation, context.robot) is arm:
            return designation
    raise UndesignatedArmError(context.robot)


def default_coraplex_capability_handlers(
    initialization: CoraplexCapabilityInitialization,
) -> dict[str, CoraplexActionFactory]:
    """
    One native-action factory per capability the initialization admits.
    """
    registrations: dict[str, list[CoraplexCapabilityRegistration]] = {}
    for record in initialization.records:
        if (
            record.status is CapabilityReviewStatus.APPROVED
            and record.parameter_sources
        ):
            registrations.setdefault(record.contract.uid, []).append(record)
    return {
        capability_uid: _registered_capability_factory(tuple(records))
        for capability_uid, records in registrations.items()
    }


def _registered_capability_factory(
    registrations: tuple[CoraplexCapabilityRegistration, ...],
) -> CoraplexActionFactory:
    factories = {
        registration: action_factory_for(registration) for registration in registrations
    }

    def build(
        request: ExecutionRequest, context: EvaluationContext, universe: ObjectUniverse
    ) -> ActionDescription:
        registration = applicable_registration(
            registrations, request, robot_resources(context.robot)
        )
        return factories[registration](request, context, universe)

    return build


def _grounded_role(universe: ObjectUniverse, request: ExecutionRequest, role: str):
    if role not in request.argument_map:
        raise MissingRoleArgumentError(request, role)
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
    """
    The grasp Coraplex assumes itself when an action is given none: front approach
    without vertical alignment, on the manipulation arm's end effector.
    """
    return GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        ViewManager.get_arm_view(_arm_designation(context), context.robot).end_effector,
    )


def action_factory_for(
    registration: CoraplexCapabilityRegistration,
) -> CoraplexActionFactory:
    """
    Build native actions for one reviewed registration, filling every parameter from its
    reviewed source and converting it to the type the action declares.
    """
    resolved: dict[str, object] = {}

    def build(
        request: ExecutionRequest, context: EvaluationContext, universe: ObjectUniverse
    ) -> ActionDescription:
        if not resolved:
            action_type = _action_type(registration.draft.action_class)
            resolved["type"] = action_type
            resolved["declared"] = _declared_parameter_types(action_type)
        action_type = resolved["type"]
        declared_types = resolved["declared"]
        return action_type(
            **{
                source.parameter: _parameter_value(
                    source,
                    declared_types[source.parameter],
                    request,
                    context,
                    universe,
                )
                for source in registration.parameter_sources
            }
        )

    return build


def _declared_parameter_types(action_type: type) -> dict[str, object]:
    """
    The declared type of every constructor parameter of a native action.

    Coraplex annotates some attributes with names imported only for type checking; those
    resolve to ``object`` here, which only matters for parameters never bound.
    """
    placeholders: dict[str, type] = {}
    while True:
        try:
            return get_type_hints(action_type, localns=placeholders)
        except NameError as error:
            placeholders[error.name] = object


def _action_type(action_class: str) -> type:
    module_name, _, class_name = action_class.rpartition(".")
    return vars(importlib.import_module(module_name))[class_name]


def _parameter_value(
    source: ParameterSource,
    declared_type: object,
    request: ExecutionRequest,
    context: EvaluationContext,
    universe: ObjectUniverse,
):
    if source.kind is ParameterSourceKind.CONTEXT:
        return _context_value(source, context, request)
    target = _without_optional(declared_type)
    if source.kind is ParameterSourceKind.CONSTANT:
        return target[source.value] if _is_enum(target) else source.value
    if _is_enum(target):
        return target[request.argument(source.value)]
    return _role_value(universe, request, source.value, target)


def _context_value(
    source: ParameterSource, context: EvaluationContext, request: ExecutionRequest
):
    value = ContextValue(source.value)
    if value is ContextValue.MANIPULATION_ARM:
        return _arm_designation(context)
    if value is ContextValue.DEFAULT_GRASP:
        return _default_grasp(context)
    key = tuple(request.argument(role) for role in source.key_roles)
    if key not in context.witness_base_poses:
        raise MissingWitnessPoseError(request)
    return context.witness_base_poses[key].to_pose()


def _role_value(
    universe: ObjectUniverse, request: ExecutionRequest, role: str, target: object
):
    """
    The grounded object bound to a role, in the form the declared parameter type asks
    for.
    """
    if get_origin(target) is type:
        return type(_entity_role(universe, request, role))
    if target is Pose:
        return _object_pose(universe, request, role)
    if target is PoseTrajectory:
        return PoseTrajectory(poses=[_object_pose(universe, request, role)])
    if target is HasRootBody or (isinstance(target, type) and issubclass(target, Body)):
        return _body_role(universe, request, role)
    if isinstance(target, type) and issubclass(target, SemanticAnnotation):
        return _entity_role(universe, request, role)
    raise UnsupportedParameterTypeError(role, target)


def _without_optional(declared_type: object) -> object:
    if get_origin(declared_type) is Union:
        return next(
            argument
            for argument in get_args(declared_type)
            if argument is not type(None)
        )
    return declared_type


def _is_enum(target: object) -> bool:
    return isinstance(target, type) and issubclass(target, Enum)


class UnsupportedParameterTypeError(Exception):
    """
    Raised when a role is bound to a native parameter of a type reSym cannot supply.
    """

    def __init__(self, role: str, declared_type: object):
        super().__init__(
            f"Role '{role}' cannot fill a parameter declared as {declared_type!r}."
        )


class UndesignatedArmError(Exception):
    """
    Raised when the manipulation arm is none of the arms Coraplex can designate.
    """

    def __init__(self, robot: AbstractRobot):
        super().__init__(
            f"The manipulation arm of {type(robot).__name__} is neither its left nor "
            "its right arm."
        )


class MissingRoleArgumentError(Exception):
    """
    Raised when a request does not bind a role its realization reads.
    """

    def __init__(self, request: ExecutionRequest, role: str):
        super().__init__(f"Role '{role}' is not bound in {request}.")


class InvalidCapabilityArgumentError(Exception):
    """
    A semantic request role cannot be adapted to a native action argument.
    """

    def __init__(self, request: ExecutionRequest, reason: str):
        super().__init__(f"Invalid {request.capability_ref.uid} request: {reason}.")
