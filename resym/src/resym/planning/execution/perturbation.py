"""
Execution perturbations: wrappers around an execution backend that inject the §9.6-style
disturbances into simulated runs.

Two wrappers, both deterministic by construction (explicit schedules
keyed by the wrapper's own apply-call counter — no hidden randomness;
derive schedules from case seeds in the harness):

- :class:`PartialSkillRealization` — an actuation fault: on scheduled
  actions the skill only completes a fraction of its motion. Realized
  by snapshotting the articulated joints (drawer connections) before
  the inner backend runs and interpolating each changed joint between
  its pre- and post-state. The very next postcondition check sees the
  incomplete effect — that is the point.
- :class:`ExternalPerturbationRealization` — the world changes behind the
  robot's back: after scheduled actions an external disturbance
  callback mutates the state (someone pushes the drawer shut, a sensor
  reads a jittered joint). Ready-made disturbances below.

Both wrappers preserve the backend contract so postcondition checks, goal verification, and
nogood replanning all see the perturbed world through the ordinary
monitoring path — the E1 robustness grids and the E3 monitoring
ablations wrap the backend and change nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Callable, Mapping, Optional

from resym.platform.articulation import (
    articulation_connection,
    is_articulated_object,
)
from resym.platform.evaluators import EvaluationContext
from resym.planning.execution.engine import (
    PlatformExecutionResult,
    PlatformExecutionStatus,
    PlatformSkillRealization,
)
from resym.platform.kinematic import KinematicSkillRealization
from resym.core.model import ExecutionRequest
from resym.platform.universe import (
    ObjectUniverse,
    joint_fraction,
    set_joint_fraction,
)

Disturbance = Callable[[EvaluationContext, ObjectUniverse], None]
"""
An external world change, applied right after a scheduled action.
"""


def _articulated_connections(universe: ObjectUniverse):
    """
    The actuating connections of every annotated articulated object — the state a
    drawer-domain skill can move.
    """
    return [
        articulation_connection(grounded)
        for grounded in universe.objects.values()
        if is_articulated_object(grounded)
    ]


@dataclass
class PartialSkillRealization(PlatformSkillRealization):
    """
    Scheduled actions complete only a fraction of their joint motion.
    """

    schedule: Mapping[int, float]
    """
    Apply-call index (0-based, counted by this wrapper) -> completion fraction in [0,
    1]; unscheduled calls pass through unchanged.
    """

    inner: Optional[PlatformSkillRealization] = None
    applied: int = field(default=0)

    def __post_init__(self):
        if self.inner is None:
            self.inner = KinematicSkillRealization()
        for index, fraction in self.schedule.items():
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"Completion fraction for action {index} must be in "
                    f"[0, 1], got {fraction}."
                )

    def execute(
        self,
        request: ExecutionRequest,
        context: EvaluationContext,
        universe: ObjectUniverse,
    ) -> PlatformExecutionResult:
        index = self.applied
        self.applied += 1
        completion = self.schedule.get(index)
        connections = _articulated_connections(universe)
        before = {connection.dof.id: connection.position for connection in connections}
        result = self.inner.execute(request, context, universe)
        if completion is None:
            return result
        if result.status is not PlatformExecutionStatus.SUCCEEDED:
            return result
        for connection in connections:
            start = before[connection.dof.id]
            end = connection.position
            if start != end:
                connection.position = start + completion * (end - start)
        return result


@dataclass
class ExternalPerturbationRealization(PlatformSkillRealization):
    """
    After scheduled actions, an external disturbance mutates the world.
    """

    schedule: Mapping[int, Disturbance]
    """
    Apply-call index (0-based, counted by this wrapper) -> disturbance to apply right
    after the action ran.
    """

    inner: Optional[PlatformSkillRealization] = None
    applied: int = field(default=0)

    def __post_init__(self):
        if self.inner is None:
            self.inner = KinematicSkillRealization()

    def execute(
        self,
        request: ExecutionRequest,
        context: EvaluationContext,
        universe: ObjectUniverse,
    ) -> PlatformExecutionResult:
        index = self.applied
        self.applied += 1
        disturbance = self.schedule.get(index)
        result = self.inner.execute(request, context, universe)
        if (
            result.status is PlatformExecutionStatus.SUCCEEDED
            and disturbance is not None
        ):
            disturbance(context, universe)
        return result


# -- ready-made disturbances --------------------------------------------


def set_drawer_fraction(drawer_name: str, fraction: float) -> Disturbance:
    """
    Someone moves the drawer to an absolute joint fraction.
    """

    def disturb(context: EvaluationContext, universe: ObjectUniverse) -> None:
        set_joint_fraction(
            articulation_connection(universe[drawer_name]),
            fraction,
        )

    return disturb


def jitter_drawer_fraction(drawer_name: str, delta: float) -> Disturbance:
    """
    Sensor-noise-style jitter: the drawer joint shifts by a signed fraction of its
    range, clipped to the joint limits.
    """

    def disturb(context: EvaluationContext, universe: ObjectUniverse) -> None:
        connection = articulation_connection(universe[drawer_name])
        shifted = joint_fraction(connection) + delta
        set_joint_fraction(connection, min(1.0, max(0.0, shifted)))

    return disturb
