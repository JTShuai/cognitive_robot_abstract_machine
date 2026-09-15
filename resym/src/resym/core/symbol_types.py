"""
References to CRAM semantic types.
"""

from __future__ import annotations

import re
import ast
import importlib.util
from pathlib import Path
from dataclasses import dataclass
from functools import cache

from krrood.ripple_down_rules.utils import get_type_from_string
from krrood.utils import get_full_class_name

ALLOWED_TYPE_REFERENCE = re.compile(
    r"(?:semantic_digital_twin|krrood)(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
)
"""
Type references may only name classes of the CRAM packages that define semantics.
"""


@dataclass(frozen=True, order=True)
class SymbolType:
    """
    Stable import reference to one CRAM Python semantic type, spelled the way krrood
    names types: ``module.Class``.
    """

    python_type_ref: str

    def __post_init__(self) -> None:
        if not ALLOWED_TYPE_REFERENCE.fullmatch(self.python_type_ref):
            raise ValueError(
                f"Symbol type '{self.python_type_ref}' is not an allowed CRAM Python type reference"
            )

    @property
    def short_name(self) -> str:
        return self.python_type_ref.rsplit(".", 1)[1]

    @classmethod
    def from_python_type(cls, python_type: type) -> SymbolType:
        return cls(get_full_class_name(python_type))


@cache
def resolve_symbol_type(symbol_type: SymbolType) -> type:
    value = get_type_from_string(symbol_type.python_type_ref)
    if not isinstance(value, type):
        raise TypeError(
            f"Symbol type '{symbol_type.python_type_ref}' does not resolve to a class"
        )
    return value


def is_symbol_subtype(actual: SymbolType, expected: SymbolType) -> bool:
    return issubclass(resolve_symbol_type(actual), resolve_symbol_type(expected))


@cache
def matching_symbol_types(reference: str) -> tuple[str, ...]:
    """Find real class locations for a reference with the wrong module path."""
    package_name, _, remainder = reference.partition(".")
    if package_name not in ("semantic_digital_twin", "krrood") or not remainder:
        return ()
    specification = importlib.util.find_spec(package_name)
    if specification is None or specification.submodule_search_locations is None:
        return ()
    class_name = reference.rsplit(".", 1)[-1]
    matches = []
    for location in specification.submodule_search_locations:
        root = Path(location)
        for path in root.rglob("*.py"):
            if path.name == "ormatic_interface.py":
                continue
            source = path.read_text()
            if f"class {class_name}" not in source:
                continue
            if not any(
                isinstance(node, ast.ClassDef) and node.name == class_name
                for node in ast.parse(source).body
            ):
                continue
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")
            matches.append(f"{package_name}.{module}.{class_name}")
    return tuple(sorted(matches))
