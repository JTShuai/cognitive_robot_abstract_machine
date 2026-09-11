"""What a concrete robot platform can actually do.

An :class:`EmbodimentProfile` declares the capability surface of one
embodiment: which truth procedures (evaluators) it implements, which
capabilities it can realize, and how its tool approaches a manipulation
target. The pipeline checks every selected symbol against the profile
*before* grounding or planning; a goal whose causal closure needs a
capability the platform lacks is an ``UNSUPPORTED_CAPABILITY`` failure,
not a library gap — no amount of model repair can add a motor.

Whether an embodiment is mobile is a fact of the robot description, not
of the profile: runtime code reads the drive off the robot
(``robot.drive``). A fixed arm simply lacks the witness-pose evaluator
and the navigation realization in its capability set.

This module is deliberately platform-free: profiles are declarative
data, so backends, the curator, and the experiment harness can derive
``available_capabilities`` from a profile without importing any robot stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from resym.platform.grounding_catalog import GroundingFactoryCatalog


class ToolOrientation(Enum):
    """How reachability targets orient the tool frame.

    The IK solver enforces the full 6D pose, so the target orientation
    must be one the arm can actually attain: a redundant mobile
    manipulator can align its tool with the base frame it chose to face
    the target, while a 5-DoF fixed arm can only point its tool axis
    along directions in the vertical plane of its base yaw.
    """

    BASE_ALIGNED = "base-aligned"
    """Tool frame aligned with the (chosen) base frame — the mobile case,
    where the base was already turned to face the target."""

    APPROACH_ALIGNED = "approach-aligned"
    """Tool z-axis along the horizontal direction from the arm base to the
    target — the natural grasp approach of a fixed arm, and the only
    attainable family of directions for a low-DoF wrist (a yaw-pitch-
    pitch-pitch-roll arm can point its tool axis nowhere else)."""


@dataclass(frozen=True)
class EmbodimentProfile:
    """The declared capability surface of one robot platform."""

    name: str
    """Identifier recorded in certificates and admission metadata."""

    evaluators: frozenset[str]
    """Predicate truth procedures this embodiment implements."""

    capabilities: frozenset[str]
    """Stable capability UIDs this embodiment can realize."""

    capability_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Native implementation evidence grouped by semantic capability UID."""

    tool_orientation: ToolOrientation = ToolOrientation.BASE_ALIGNED
    """How reachability targets orient the tool frame."""

    def missing_evaluators(
        self,
        selection,
        grounding_catalog: GroundingFactoryCatalog | None = None,
    ) -> tuple[str, ...]:
        """Evaluator bindings used by the selection but absent here."""
        missing = []
        for predicate in selection.predicates.values():
            plan = predicate.grounding_plan
            if plan is not None:
                if grounding_catalog is None:
                    missing.append(
                        f"predicate '{predicate.name}' needs grounding factory "
                        f"'{plan.factory_uid}', but no catalog is loaded"
                    )
                elif not grounding_catalog.supports(plan.factory_uid, self.name):
                    missing.append(
                        f"predicate '{predicate.name}' needs grounding factory "
                        f"'{plan.factory_uid}', which embodiment '{self.name}' "
                        "does not implement"
                    )
                continue
            implementation = predicate.implementation
            if implementation.evaluator_key not in self.evaluators:
                missing.append(
                    f"predicate '{predicate.name}' needs evaluator "
                    f"'{implementation.evaluator_key}', which embodiment "
                    f"'{self.name}' does not implement"
                )
        return tuple(missing)

    def missing_capabilities(self, selection) -> tuple[str, ...]:
        """Execution capabilities used by the selection but absent here."""
        missing = []
        for operator in selection.operators.values():
            capability_uid = operator.execution_binding.capability_ref.uid
            if capability_uid not in self.capabilities:
                missing.append(
                    f"operator '{operator.name}' needs capability "
                    f"'{capability_uid}', which embodiment "
                    f"'{self.name}' does not provide"
                )
        return tuple(missing)


class InvalidEvaluatorBindingError(Exception):
    """A selected predicate names an evaluator absent from the embodiment.

    This is first treated as a repairable model binding error.  The repair
    agent may still conclude that the evaluator is genuinely unavailable and
    choose the certificate's unsupported-capability alternative.
    """

    def __init__(self, missing: tuple[str, ...], goal: tuple = (), selection=None):
        super().__init__("Invalid evaluator binding: " + "; ".join(missing))
        self.missing = missing
        self.goal = goal
        self.selection = selection


class UnsupportedCapabilityError(Exception):
    """The goal's causal closure needs a capability the embodiment lacks.

    Raised by the pipeline before grounding; carries the goal and the
    selection so the caller can build the ``UNSUPPORTED_CAPABILITY``
    failure certificate.
    """

    def __init__(
        self,
        missing: tuple[str, ...],
        goal: tuple = (),
        selection=None,
    ):
        super().__init__("Unsupported capability: " + "; ".join(missing))
        self.missing = missing
        self.goal = goal
        self.selection = selection
