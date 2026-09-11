"""
Natural-language task understanding at the boundary of reSym.

The language model may select existing goal predicates, report a relation that the
current library cannot express, or request clarification. Programmatic validation
resolves every referenced object and checks every existing predicate before planning. A
model gap becomes a failure certificate for the repair agent; an ambiguous instruction
never enters planning or repair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import Optional

from resym.core.model import Literal, SymbolLibrary, is_symbol_subtype
from resym.llm.prompting import render_predicates, render_prompt
from resym.llm.schemas import GoalTranslation
from resym.llm.structured import StructuredCompleter
from resym.platform.evaluators import EvaluationContext
from resym.platform.universe import ObjectUniverse
from resym.platform.krrood_queries import (
    KrroodObjectResolver,
    ObjectQuerySpec,
)
from resym.repair.certificate import certify_unsolvable
from resym.repair.diagnosis import TaskDiagnosis, diagnose
from resym.planning.selection import Selection

AGENT_NAME = "goal-translator"
"""
Transcript identity of this agent.
"""

PREDICATE_NAME = re.compile(r"^[a-z][a-z0-9-]*$")


class TaskUnderstandingStatus(StrEnum):
    """
    Whether an instruction may proceed to planning.
    """

    READY = "ready"
    MODEL_GAP = "model_gap"
    CLARIFICATION_NEEDED = "clarification_needed"


@dataclass(frozen=True)
class UnresolvedGoal:
    """
    A desired relation absent from the current predicate library.
    """

    literal: Literal
    description: str


@dataclass(frozen=True)
class TaskUnderstanding:
    """
    Validated interpretation of one user instruction.
    """

    instruction: str
    status: TaskUnderstandingStatus
    known_goal: tuple[Literal, ...] = ()
    unresolved_goal: tuple[UnresolvedGoal, ...] = ()
    message: str = ""

    @property
    def goal(self) -> tuple[Literal, ...]:
        """
        Known and provisional literals used by diagnosis and retrieval.
        """
        return self.known_goal + tuple(item.literal for item in self.unresolved_goal)


@dataclass(frozen=True)
class NaturalLanguageTaskDiagnosis:
    """
    Task understanding plus the optional planning/repair diagnosis.
    """

    understanding: TaskUnderstanding
    diagnosis: Optional[TaskDiagnosis] = None

    @property
    def needs_clarification(self) -> bool:
        return self.understanding.status is TaskUnderstandingStatus.CLARIFICATION_NEEDED

    @property
    def triggers_curation(self) -> bool:
        return bool(
            self.diagnosis
            and self.diagnosis.certificate
            and self.diagnosis.certificate.triggers_curation
        )


class UntranslatableGoalError(Exception):
    """
    Raised when the model cannot produce a valid goal for an instruction.
    """

    def __init__(self, instruction: str, errors: list[str]):
        super().__init__(
            f"Could not translate '{instruction}' into a valid goal: "
            + "; ".join(errors)
        )


@dataclass
class GoalTranslator:
    """
    Interprets instructions against the persistent library and live world.
    """

    completer: StructuredCompleter
    """
    The structured language-model channel.
    """

    maximum_repair_rounds: int = 2
    """
    How often semantic validation errors are fed back to the model.
    """

    def translate(
        self,
        instruction: str,
        library: SymbolLibrary,
        universe: ObjectUniverse,
    ) -> tuple[Literal, ...]:
        """
        Return an existing-symbol goal, preserving the original API.

        Call :meth:`interpret` when model gaps and clarification requests must remain
        distinguishable.
        """
        understanding = self.interpret(instruction, library, universe)
        if understanding.status is TaskUnderstandingStatus.READY:
            return understanding.goal
        detail = understanding.message or understanding.status.value
        raise UntranslatableGoalError(instruction, [detail])

    def interpret(
        self,
        instruction: str,
        library: SymbolLibrary,
        universe: ObjectUniverse,
    ) -> TaskUnderstanding:
        """
        Return a validated goal, model gap, or clarification request.
        """
        base_prompt = render_prompt(
            "translate_goal",
            instruction=instruction,
            predicates=render_predicates(library),
            objects=_object_listing(universe),
        )
        prompt = base_prompt
        errors: list[str] = []
        for _ in range(self.maximum_repair_rounds + 1):
            translation = self.completer.complete(AGENT_NAME, prompt, GoalTranslation)
            translation, resolution_errors = _resolve_object_queries(
                translation, universe
            )
            errors = resolution_errors or _translation_errors(
                translation, library, universe
            )
            if not errors:
                return _to_understanding(instruction, translation)
            prompt = (
                f"{base_prompt}\n\n"
                f"Your previous goal was rejected:\n- "
                + "\n- ".join(errors)
                + "\nReply again with a corrected JSON document."
            )
        raise UntranslatableGoalError(instruction, errors)


def diagnose_instruction(
    translator: GoalTranslator,
    instruction: str,
    library: SymbolLibrary,
    universe: ObjectUniverse,
    context: EvaluationContext,
    working_directory: Path,
    **solve_arguments,
) -> NaturalLanguageTaskDiagnosis:
    """
    Understand an instruction and enter the existing task workflow.

    A clear missing relation produces a deterministic
    ``MISSING_PREDICATE_MODEL`` certificate without sending an invalid goal to
    the planner. A clarification request stops here. A ready goal uses the
    normal monitored planning and execution path.
    """
    understanding = translator.interpret(instruction, library, universe)
    if understanding.status is TaskUnderstandingStatus.CLARIFICATION_NEEDED:
        return NaturalLanguageTaskDiagnosis(understanding)
    if understanding.status is TaskUnderstandingStatus.MODEL_GAP:
        descriptions = tuple(item.description for item in understanding.unresolved_goal)
        object_types = tuple(
            (name, universe[name].symbol_type.python_type_ref)
            for name in dict.fromkeys(
                argument
                for literal in understanding.goal
                for argument in literal.arguments
            )
        )
        certificate = certify_unsolvable(
            understanding.goal,
            library,
            Selection(),
            planner_message="natural-language task requires an absent relation",
            task_instruction=instruction,
            goal_semantics=descriptions,
            task_object_types=object_types,
        )
        return NaturalLanguageTaskDiagnosis(
            understanding,
            TaskDiagnosis(succeeded=False, certificate=certificate),
        )
    return NaturalLanguageTaskDiagnosis(
        understanding,
        diagnose(
            library,
            universe,
            context,
            understanding.goal,
            working_directory,
            **solve_arguments,
        ),
    )


def _to_understanding(
    instruction: str, translation: GoalTranslation
) -> TaskUnderstanding:
    return TaskUnderstanding(
        instruction=instruction,
        status=TaskUnderstandingStatus(translation.status),
        known_goal=tuple(literal.to_literal() for literal in translation.literals),
        unresolved_goal=tuple(
            UnresolvedGoal(item.to_literal(), item.description.strip())
            for item in translation.unresolved_literals
        ),
        message=translation.message.strip(),
    )


def _translation_errors(
    translation: GoalTranslation,
    library: SymbolLibrary,
    universe: ObjectUniverse,
) -> list[str]:
    status = TaskUnderstandingStatus(translation.status)
    known_goal = tuple(literal.to_literal() for literal in translation.literals)
    errors: list[str] = []
    if status is TaskUnderstandingStatus.READY:
        if translation.unresolved_literals:
            errors.append("a ready task cannot contain unresolved literals")
        errors.extend(_semantic_errors(known_goal, library, universe))
    elif status is TaskUnderstandingStatus.MODEL_GAP:
        if not translation.unresolved_literals:
            errors.append("model_gap requires at least one unresolved literal")
        errors.extend(_semantic_errors(known_goal, library, universe, allow_empty=True))
        errors.extend(_unresolved_errors(translation, library, universe))
    else:
        if (
            translation.object_queries
            or translation.literals
            or translation.unresolved_literals
        ):
            errors.append(
                "clarification_needed cannot contain object queries or goal literals"
            )
        if not translation.message.strip():
            errors.append("clarification_needed requires a question in message")
    return errors


def _resolve_object_queries(
    translation: GoalTranslation,
    universe: ObjectUniverse,
) -> tuple[GoalTranslation, list[str]]:
    """
    Resolve model-chosen descriptions; no model-provided code is executed.
    """
    aliases: dict[str, str] = {}
    errors: list[str] = []
    resolver = KrroodObjectResolver(universe)
    for query in translation.object_queries:
        if query.reference in aliases:
            errors.append(f"duplicate object query reference '{query.reference}'")
            continue
        matches = resolver.resolve(
            ObjectQuerySpec(
                type_ref=query.type,
                color=query.color,
                name_contains=query.name_contains,
            )
        )
        if not matches:
            errors.append(f"object query '{query.reference}' matched no objects")
        elif len(matches) > 1:
            errors.append(
                f"object query '{query.reference}' is ambiguous; matched "
                + ", ".join(item.name for item in matches)
            )
        else:
            aliases[query.reference] = matches[0].name

    used_aliases = {
        argument
        for literal in (*translation.literals, *translation.unresolved_literals)
        for argument in literal.arguments
        if argument.startswith("$")
    }
    for reference in aliases.keys() - used_aliases:
        errors.append(f"object query '{reference}' is not used by the goal")
    for reference in used_aliases - aliases.keys():
        if not any(
            query.reference == reference for query in translation.object_queries
        ):
            errors.append(f"goal uses undeclared object query '{reference}'")
    if errors:
        return translation, errors

    def resolved(arguments: list[str]) -> list[str]:
        return [aliases.get(argument, argument) for argument in arguments]

    return (
        translation.model_copy(
            update={
                "literals": [
                    literal.model_copy(
                        update={"arguments": resolved(literal.arguments)}
                    )
                    for literal in translation.literals
                ],
                "unresolved_literals": [
                    literal.model_copy(
                        update={"arguments": resolved(literal.arguments)}
                    )
                    for literal in translation.unresolved_literals
                ],
            }
        ),
        [],
    )


def _semantic_errors(
    goal: tuple[Literal, ...],
    library: SymbolLibrary,
    universe: ObjectUniverse,
    *,
    allow_empty: bool = False,
) -> list[str]:
    """
    Everything wrong with a goal against the library and the universe.
    """
    errors = []
    if not goal and not allow_empty:
        errors.append("the goal is empty")
    for literal in goal:
        if literal.predicate not in library.predicates:
            errors.append(f"unknown predicate '{literal.predicate}'")
            continue
        predicate = library.predicates[literal.predicate]
        if len(literal.arguments) != len(predicate.parameter_types):
            errors.append(
                f"'{literal.predicate}' takes {len(predicate.parameter_types)} "
                f"arguments, got {len(literal.arguments)}"
            )
            continue
        for argument, expected_type in zip(
            literal.arguments, predicate.parameter_types
        ):
            if argument not in universe.objects:
                errors.append(f"unknown object '{argument}'")
            elif not is_symbol_subtype(universe[argument].symbol_type, expected_type):
                errors.append(
                    f"object '{argument}' has type "
                    f"'{universe[argument].symbol_type.python_type_ref}', "
                    f"'{literal.predicate}' expects '{expected_type.python_type_ref}'"
                )
    return errors


def _unresolved_errors(
    translation: GoalTranslation,
    library: SymbolLibrary,
    universe: ObjectUniverse,
) -> list[str]:
    errors: list[str] = []
    for item in translation.unresolved_literals:
        name = item.suggested_predicate
        if not PREDICATE_NAME.fullmatch(name):
            errors.append(
                f"suggested predicate '{name}' must be a lowercase PDDL identifier"
            )
        elif name in library.predicates:
            errors.append(
                f"predicate '{name}' already exists; use it as a normal literal"
            )
        if not item.description.strip():
            errors.append(f"unresolved predicate '{name}' needs a description")
        for argument in item.arguments:
            if argument not in universe.objects:
                errors.append(f"unknown object '{argument}'")
    return errors


def _object_listing(universe: ObjectUniverse) -> str:
    return "\n".join(
        f"- {grounded_object.name} ({grounded_object.symbol_type.python_type_ref})"
        for grounded_object in universe.objects.values()
    )
