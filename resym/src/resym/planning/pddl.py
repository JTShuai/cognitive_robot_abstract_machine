"""
Projection to PDDL and planning with Fast Downward.

The domain file carries the goal-relevant operator schemas, the problem file carries the
typed object universe and the computed initial state. Both are task-scoped, written
fresh per task, and thrown away afterwards.
"""

from __future__ import annotations

import os
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path

from resym.core.model import (
    Literal,
    Operator,
    SymbolType,
    is_symbol_subtype,
)
from resym import PROJECT_ROOT
from resym.planning.selection import Selection
from resym.platform.universe import ObjectUniverse

FAST_DOWNWARD_DIRECTORY_ENVIRONMENT_VARIABLE = "RESYM_FAST_DOWNWARD_DIRECTORY"
"""
Environment variable selecting the installed planner directory.
"""


def planner_directory() -> Path:
    """
    Return the planner directory selected by the runtime environment.
    """
    configured_directory = os.environ.get(FAST_DOWNWARD_DIRECTORY_ENVIRONMENT_VARIABLE)
    if configured_directory:
        return Path(configured_directory)
    return PROJECT_ROOT / "vendor" / "downward"


@dataclass(frozen=True)
class GroundAction:
    """
    One step of a plan: an operator applied to concrete objects.
    """

    operator: str
    """
    Name of the operator schema.
    """

    arguments: tuple[str, ...]
    """
    PDDL object names bound to the operator parameters, in order.
    """

    def binding(self, operator: Operator) -> dict[str, str]:
        """
        Map the operator's parameter variables to this action's objects.
        """
        return dict(zip(operator.parameter_names, self.arguments))


class PlanNotFoundError(Exception):
    """
    Raised when the planner proves the projected problem unsolvable or fails.
    """

    def __init__(self, details: str):
        super().__init__(f"Fast Downward found no plan: {details}")


class PlannerNotInstalledError(Exception):
    """
    Raised when the vendored Fast Downward build is missing.
    """

    def __init__(self):
        super().__init__(
            "Fast Downward is not installed. Run scripts/install_fast_downward.sh first."
        )


def write_domain(selection: Selection, name: str) -> str:
    """
    Emit STRIPS with CRAM Python types represented as static predicates.
    """
    symbol_types = _selection_types(selection)
    lines = [
        f"(define (domain {name})",
        "  (:requirements :strips :negative-preconditions)",
        "  (:predicates",
    ]
    for symbol_type in symbol_types:
        lines.append(f"    ({_type_predicate(symbol_type)} ?x)")
    for predicate in selection.predicates.values():
        parameters = " ".join(
            f"?{_parameter_letter(i)}" for i, _ in enumerate(predicate.parameter_types)
        )
        lines.append(f"    ({predicate.name} {parameters})")
    lines.append("  )")
    for operator in selection.operators.values():
        parameters = " ".join(f"?{v}" for v, _ in operator.parameters)
        type_conditions = tuple(
            Literal(_type_predicate(symbol_type), (variable,))
            for variable, symbol_type in operator.parameters
        )
        lines.append(f"  (:action {operator.name}")
        lines.append(f"    :parameters ({parameters})")
        lines.append(
            f"    :precondition (and {_literals(type_conditions + operator.preconditions)})"
        )
        effects = (
            _literals(operator.add_effects)
            + " "
            + _literals(operator.delete_effects, negate=True)
        )
        lines.append(f"    :effect (and {effects.strip()})")
        lines.append("  )")
    lines.append(")")
    return "\n".join(lines)


def write_problem(
    universe: ObjectUniverse,
    init_atoms: set[Literal],
    goal: tuple[Literal, ...],
    domain_name: str,
    name: str,
    selection: Selection,
) -> str:
    """
    Emit the task-scoped problem: objects from the world, the true initial atoms, and
    the Boolean goal.

    False atoms are represented by the PDDL closed-world assumption.
    """
    lines = [
        f"(define (problem {name})",
        f"  (:domain {domain_name})",
        "  (:objects",
    ]
    for item in universe.objects.values():
        lines.append(f"    {item.name}")
    lines.append("  )")
    lines.append("  (:init")
    for symbol_type in _selection_types(selection):
        for item in universe.objects.values():
            if is_symbol_subtype(item.symbol_type, symbol_type):
                lines.append(f"    ({_type_predicate(symbol_type)} {item.name})")
    for atom in sorted(init_atoms, key=lambda a: (a.predicate, a.arguments)):
        lines.append(f"    ({atom.predicate} {' '.join(atom.arguments)})")
    lines.append("  )")
    lines.append(f"  (:goal (and {_ground_literals(goal)}))")
    lines.append(")")
    return "\n".join(lines)


def _selection_types(selection: Selection) -> tuple[SymbolType, ...]:
    symbol_types = {
        symbol_type
        for predicate in selection.predicates.values()
        for symbol_type in predicate.parameter_types
    }
    symbol_types.update(
        symbol_type
        for operator in selection.operators.values()
        for _, symbol_type in operator.parameters
    )
    return tuple(sorted(symbol_types))


def _type_predicate(symbol_type: SymbolType) -> str:
    short_name = re.sub(r"(?<!^)(?=[A-Z])", "-", symbol_type.short_name).lower()
    return f"cram-type-{short_name}"


def plan(
    domain_text: str, problem_text: str, working_directory: Path
) -> list[GroundAction]:
    """
    Run Fast Downward on the projected task and parse the resulting plan.
    """
    driver = planner_directory() / "fast-downward.py"
    if not driver.exists():
        raise PlannerNotInstalledError()
    working_directory = working_directory.resolve()
    working_directory.mkdir(parents=True, exist_ok=True)
    domain_path = working_directory / "domain.pddl"
    problem_path = working_directory / "problem.pddl"
    plan_path = working_directory / "sas_plan"
    # A replanning round reuses its working directory.  Fast Downward does
    # not promise to remove an older plan when a later search produces none,
    # so never let existence of a stale artifact masquerade as fresh output.
    plan_path.unlink(missing_ok=True)
    domain_path.write_text(domain_text)
    problem_path.write_text(problem_text)
    completed = subprocess.run(
        [
            sys.executable,
            str(driver),
            "--plan-file",
            str(plan_path),
            str(domain_path),
            str(problem_path),
            "--search",
            "lazy_greedy([ff()], preferred=[ff()])",
        ],
        capture_output=True,
        text=True,
        cwd=working_directory,
    )
    if completed.returncode != 0 or not plan_path.exists():
        raise PlanNotFoundError(completed.stdout[-2000:] + completed.stderr[-500:])
    return _parse_plan(plan_path.read_text())


def _parse_plan(text: str) -> list[GroundAction]:
    actions = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("("):
            continue
        tokens = line.strip("()").split()
        actions.append(GroundAction(operator=tokens[0], arguments=tuple(tokens[1:])))
    return actions


def _literals(literals: tuple[Literal, ...], negate: bool = False) -> str:
    parts = []
    for literal in literals:
        atom = f"({literal.predicate} {' '.join('?' + a for a in literal.arguments)})"
        if literal.negated != negate:
            atom = f"(not {atom})"
        parts.append(atom)
    return " ".join(parts)


def _ground_literals(literals: tuple[Literal, ...]) -> str:
    parts = []
    for literal in literals:
        atom = f"({literal.predicate} {' '.join(literal.arguments)})"
        if literal.negated:
            atom = f"(not {atom})"
        parts.append(atom)
    return " ".join(parts)


def _parameter_letter(index: int) -> str:
    return chr(ord("a") + index)
