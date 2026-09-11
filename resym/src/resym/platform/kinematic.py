"""
Bundled kinematic realizations for the articulation demonstration.
"""

from dataclasses import dataclass

from typing_extensions import Callable

from resym.core.model import ExecutionRequest
from resym.planning.execution.engine import (
    MissingWitnessPoseError,
    PlatformExecutionResult,
    PlatformSkillRealization,
    UnknownCapabilityError,
    UnsupportedCapabilityRealizationError,
)
from resym.platform.articulation import articulation_connection
from resym.platform.capabilities import (
    ACTOR_ROLE,
    ARTICULATION_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    PATIENT_ROLE,
    TARGET_STATE_ROLE,
    OpenCloseState,
)
from resym.platform.evaluators import EvaluationContext
from resym.platform.universe import ObjectUniverse, set_joint_fraction

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
