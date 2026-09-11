"""
Prompt templates and their rendering.

Templates are Markdown files next to this module using
``string.Template`` placeholders (``$name``), which keeps literal JSON
braces in the templates harmless. Rendering with a missing placeholder
raises.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from resym.core.model import (
    BindingSource,
    Literal,
    SymbolLibrary,
    SymbolType,
    is_symbol_subtype,
)

PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"
"""
Where the prompt template files live.
"""


def render_prompt(template_name: str, **arguments: str) -> str:
    """
    Load ``prompts/<template_name>.md`` and substitute all placeholders.
    """
    template_path = PROMPT_DIRECTORY / f"{template_name}.md"
    return Template(template_path.read_text()).substitute(**arguments)


def render_predicates(library: SymbolLibrary, *, mark_fluents: bool = False) -> str:
    """
    Render the library's predicate signatures for an LLM prompt.
    """
    return "\n".join(
        f"- {predicate.name}"
        f"({', '.join(kind.python_type_ref for kind in predicate.parameter_types)})"
        f"{' [fluent]' if mark_fluents and predicate.fluent else ''}"
        for predicate in library.predicates.values()
    )


def render_operators(library: SymbolLibrary) -> str:
    """
    Render complete operator schemas for a proposal prompt.
    """
    lines = []
    for operator in library.operators.values():
        parameters = ", ".join(
            f"{variable}: {kind.python_type_ref}"
            for variable, kind in operator.parameters
        )
        preconditions = ", ".join(
            render_literal(literal) for literal in operator.preconditions
        )
        effects = ", ".join(
            [render_literal(literal) for literal in operator.add_effects]
            + [f"not {render_literal(literal)}" for literal in operator.delete_effects]
        )
        binding = operator.execution_binding
        role_bindings = ", ".join(
            f"{role} <- "
            f"{'?' + value.value if value.source is BindingSource.PARAMETER else repr(value.value)}"
            for role, value in binding.role_bindings
        )
        lines.append(
            f"- {operator.name}({parameters}): {preconditions} -> {effects}; "
            f"binding capability_uid={binding.capability_ref.uid}, "
            f"capability_version={binding.capability_ref.version} "
            f"({role_bindings})"
        )
    return "\n".join(lines)


def render_symbol_types(symbol_types: tuple[SymbolType, ...]) -> str:
    """
    Render platform types and their Python-defined subtype relations.
    """
    ordered = tuple(sorted(symbol_types, key=lambda item: item.python_type_ref))
    lines = [f"- {symbol_type.python_type_ref}" for symbol_type in ordered]
    relations = [
        f"- {actual.python_type_ref} <: {expected.python_type_ref}"
        for actual in ordered
        for expected in ordered
        if actual != expected and is_symbol_subtype(actual, expected)
    ]
    if relations:
        lines.extend(("Subtype relations:", *relations))
    return "\n".join(lines) or "- none"


def render_capability_contracts(
    library: SymbolLibrary, supported_uids: set[str] | None = None
) -> str:
    """
    Render the contract information needed to construct a typed binding.
    """
    lines = []
    for contract in library.capability_contracts.values():
        if supported_uids is not None and contract.uid not in supported_uids:
            continue
        roles = []
        for role in contract.roles:
            if role.allowed_values:
                accepts = "constants {" + ", ".join(role.allowed_values) + "}"
            elif role.accepted_symbol_types:
                accepts = (
                    "parameter types {"
                    + ", ".join(
                        symbol_type.python_type_ref
                        for symbol_type in role.accepted_symbol_types
                    )
                    + "}"
                )
            else:
                accepts = "no declared value constraint"
            requirement = "required" if role.required else "optional"
            roles.append(f"{role.name}: {accepts}, {requirement}")
        effect_role_values = ", ".join(
            f"{effect} -> {role}={value}"
            for effect, role, value in contract.effect_role_values
        )
        lines.append(
            f"- capability_uid={contract.uid}, capability_version={contract.version} "
            f"({contract.label}); "
            f"roles [{'; '.join(roles)}]; success "
            f"{contract.success_relation}; verifiable effects "
            f"{{{', '.join(contract.verifiable_effect_names)}}}"
            + (
                f"; effect role values {{{effect_role_values}}}"
                if effect_role_values
                else ""
            )
        )
    return "\n".join(lines) or "- none"


def render_literal(literal: Literal) -> str:
    """
    Render one literal in the compact syntax used by prompts.
    """
    prefix = "not " if literal.negated else ""
    return f"{prefix}{literal.predicate}({', '.join(literal.arguments)})"
