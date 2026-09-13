"""Select a growing set of concrete objects for one symbolic task."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Any, Protocol

from resym.core.symbols import Literal
from resym.platform.krrood_queries import KrroodObjectResolver
from resym.platform.universe import ObjectUniverse
from resym.planning.events import ObjectScopeExpansionReason
from resym.planning.selection import Selection

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% selection records


class ObjectInclusionReason(StrEnum):
    """Why an object participates in the projected problem."""

    GOAL = "goal"
    ROBOT = "robot"
    DEPENDENCY = "dependency"
    PARAMETER_CANDIDATE = "parameter_candidate"
    EXPANSION = "expansion"
    ADVISED = "advised"


@dataclass(frozen=True)
class ObjectInclusion:
    """The reason for keeping an object, optionally linked to another object."""

    reason: ObjectInclusionReason
    """Kind of selection evidence."""

    related_to: str | None = None
    """Object whose dependency introduced this object."""

    rationale: str | None = None
    """Explanation given by an advisor for recommending this object."""


@dataclass(frozen=True)
class ObjectRecommendation:
    """One object an advisor asks to keep for planning, with its justification."""

    name: str
    """Concrete object name from the world directory."""

    rationale: str
    """Why the advisor considers the object task-relevant."""


@dataclass(frozen=True)
class ObjectScopeAdvice:
    """Everything one advisor consultation produced."""

    recommendations: tuple[ObjectRecommendation, ...] = ()
    """Objects the advisor asks to add, in its preferred order."""

    trace: tuple[dict[str, Any], ...] = ()
    """Serializable record of how the advice was reached."""


@dataclass
class PlanningObjectScope:
    """Objects retained for planning, with an explanation for each inclusion."""

    inclusion_reasons: dict[str, ObjectInclusion] = field(default_factory=dict)
    """Selected object names and their selection evidence."""

    advice: tuple[ObjectScopeAdvice, ...] = ()
    """Advisor consultations that shaped this scope, oldest first."""

    @property
    def object_names(self) -> frozenset[str]:
        """Return the concrete object identities retained in this scope."""
        return frozenset(self.inclusion_reasons)

    def universe(self, world_objects: ObjectUniverse) -> ObjectUniverse:
        """Reference the selected objects without copying or mutating world entities."""
        return ObjectUniverse(
            {name: world_objects[name] for name in sorted(self.object_names)}
        )


# %% advice


@dataclass(frozen=True)
class ObjectScopeAdviceRequest:
    """The symbolic situation an advisor is asked to add objects for."""

    goal: tuple[Literal, ...]
    """Ground goal whose object identities are fixed."""

    selection: Selection
    """Goal-relevant predicates and operator schemas."""

    scope: PlanningObjectScope
    """Objects already retained, with their inclusion evidence."""

    world_objects: ObjectUniverse
    """Complete object directory available to queries."""

    eligible_names: frozenset[str]
    """Objects whose type lets them enter the projected problem."""

    expansion_reason: ObjectScopeExpansionReason | None = None
    """Why the current scope proved insufficient; ``None`` on first selection."""

    planner_message: str | None = None
    """Planner or execution evidence accompanying an expansion request."""


class ObjectScopeAdvisor(Protocol):
    """Recommends task-relevant objects; never supplies truth values or goals."""

    def advise(self, request: ObjectScopeAdviceRequest) -> ObjectScopeAdvice: ...


class InvalidObjectRecommendationError(Exception):
    """Raised when an advisor names an object the projected problem cannot use."""

    def __init__(self, name: str, reason: str):
        super().__init__(f"object recommendation '{name}' rejected: {reason}")
        self.name = name
        self.reason = reason


# %% deterministic selection


@dataclass
class PlanningObjectSelector:
    """Seed from the goal and native relationships, then widen typed candidates.

    Candidate ordering is deterministic, not an assertion of semantic relevance.
    An advisor may add objects ahead of typed growth; expansion eventually
    includes every object of a selected symbolic type either way.
    """

    world_objects: ObjectUniverse
    """Complete object directory visible to world queries."""

    selection: Selection
    """Goal-relevant predicates and operator schemas."""

    goal: tuple[Literal, ...]
    """Ground goals whose object identities must never be replaced."""

    robot: AbstractRobot | None
    """Native robot annotation, if available."""

    advisor: ObjectScopeAdvisor | None = None
    """Optional source of task-relevant object recommendations."""

    def initial(self) -> PlanningObjectScope:
        """Retain goals, the robot, dependencies, and candidates for missing roles."""
        scope = PlanningObjectScope()
        for literal in self.goal:
            for name in literal.arguments:
                if name not in self.world_objects.objects:
                    raise UnknownGoalObjectError(name)
                scope.inclusion_reasons[name] = ObjectInclusion(
                    ObjectInclusionReason.GOAL
                )
        for name, item in self.world_objects.objects.items():
            if self.robot is not None and item.semantic_entity is self.robot:
                scope.inclusion_reasons.setdefault(
                    name, ObjectInclusion(ObjectInclusionReason.ROBOT)
                )
        self._include_dependencies(scope)
        parameter_types = {
            symbol_type
            for operator in self.selection.operators.values()
            for _, symbol_type in operator.parameters
        }
        before_candidates = scope.object_names
        for symbol_type in sorted(parameter_types):
            candidates = sorted(
                self.world_objects.of_type(symbol_type), key=lambda item: item.name
            )
            if candidates and not any(
                item.name in scope.object_names for item in candidates
            ):
                scope.inclusion_reasons[candidates[0].name] = ObjectInclusion(
                    ObjectInclusionReason.PARAMETER_CANDIDATE
                )
        if scope.object_names != before_candidates:
            self._include_dependencies(scope)
        self._include_advice(scope, None, None)
        return scope

    def expand(
        self,
        scope: PlanningObjectScope,
        reason: ObjectScopeExpansionReason | None = None,
        planner_message: str | None = None,
    ) -> PlanningObjectScope | None:
        """Add advised objects, or grow each typed candidate set geometrically.

        Returns ``None`` once every typed candidate is already retained.
        """
        expanded = PlanningObjectScope(dict(scope.inclusion_reasons), scope.advice)
        self._include_advice(expanded, reason, planner_message)
        if expanded.object_names != scope.object_names:
            return expanded
        candidates = self.world_objects.for_task(self.selection, self.goal)
        for symbol_type in sorted(
            {item.symbol_type for item in candidates.objects.values()}
        ):
            typed = sorted(candidates.of_type(symbol_type), key=lambda item: item.name)
            retained = sum(item.name in scope.object_names for item in typed)
            remaining = [item for item in typed if item.name not in scope.object_names]
            for item in remaining[: max(1, retained)]:
                expanded.inclusion_reasons[item.name] = ObjectInclusion(
                    ObjectInclusionReason.EXPANSION
                )
        self._include_dependencies(expanded)
        return expanded if expanded.object_names != scope.object_names else None

    def _include_advice(
        self,
        scope: PlanningObjectScope,
        reason: ObjectScopeExpansionReason | None,
        planner_message: str | None,
    ) -> None:
        """Ask the advisor for objects and admit the ones the projection can use."""
        if self.advisor is None:
            return
        eligible = frozenset(
            self.world_objects.for_task(self.selection, self.goal).objects
        )
        advice = self.advisor.advise(
            ObjectScopeAdviceRequest(
                goal=self.goal,
                selection=self.selection,
                scope=PlanningObjectScope(dict(scope.inclusion_reasons), scope.advice),
                world_objects=self.world_objects,
                eligible_names=eligible,
                expansion_reason=reason,
                planner_message=planner_message,
            )
        )
        scope.advice = scope.advice + (advice,)
        before = scope.object_names
        for recommendation in advice.recommendations:
            if recommendation.name not in self.world_objects.objects:
                raise InvalidObjectRecommendationError(
                    recommendation.name, "not in the world directory"
                )
            if recommendation.name not in eligible:
                raise InvalidObjectRecommendationError(
                    recommendation.name, "its type is unused by the selected symbols"
                )
            scope.inclusion_reasons.setdefault(
                recommendation.name,
                ObjectInclusion(
                    ObjectInclusionReason.ADVISED, rationale=recommendation.rationale
                ),
            )
        if scope.object_names != before:
            self._include_dependencies(scope)

    def _include_dependencies(self, scope: PlanningObjectScope) -> None:
        """Follow native dependencies through intermediate non-planning annotations."""
        candidates = self.world_objects.for_task(self.selection, self.goal)
        resolver = KrroodObjectResolver(self.world_objects)
        required_types = {
            symbol_type
            for predicate in self.selection.predicates.values()
            for symbol_type in predicate.parameter_types
        } | {
            symbol_type
            for operator in self.selection.operators.values()
            for _, symbol_type in operator.parameters
        }
        dependencies = resolver.dependencies(scope.object_names, required_types)
        for dependency in dependencies:
            if dependency.object_name in candidates.objects:
                scope.inclusion_reasons.setdefault(
                    dependency.object_name,
                    ObjectInclusion(
                        ObjectInclusionReason.DEPENDENCY,
                        dependency.related_to,
                    ),
                )


class UnknownGoalObjectError(Exception):
    """Raised when a goal names an object absent from the world directory."""

    def __init__(self, name: str):
        super().__init__(f"goal object '{name}' is not in the world directory")
        self.name = name
