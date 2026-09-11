"""
References to CRAM semantic types and trusted evaluator signatures.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from functools import cache


@dataclass(frozen=True, order=True)
class SymbolType:
    """
    Stable import reference to one CRAM Python semantic type.
    """

    python_type_ref: str

    def __post_init__(self) -> None:
        if not re.fullmatch(
            r"(?:semantic_digital_twin|krrood)(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
            r":[A-Za-z_][A-Za-z0-9_.]*",
            self.python_type_ref,
        ):
            raise ValueError(
                f"Symbol type '{self.python_type_ref}' is not an allowed CRAM Python type reference"
            )

    @property
    def short_name(self) -> str:
        return self.python_type_ref.rsplit(":", 1)[1].rsplit(".", 1)[-1]

    @classmethod
    def from_python_type(cls, python_type: type) -> SymbolType:
        return cls(f"{python_type.__module__}:{python_type.__qualname__}")


@cache
def resolve_symbol_type(symbol_type: SymbolType) -> type:
    module_name, qualname = symbol_type.python_type_ref.split(":", 1)
    value: object = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    if not isinstance(value, type):
        raise TypeError(
            f"Symbol type '{symbol_type.python_type_ref}' does not resolve to a class"
        )
    return value


def is_symbol_subtype(actual: SymbolType, expected: SymbolType) -> bool:
    return issubclass(resolve_symbol_type(actual), resolve_symbol_type(expected))


@dataclass(frozen=True)
class EvaluatorSpec:
    """
    Trusted evaluator ABI: name and ordered object types.
    """

    name: str
    parameter_types: tuple[SymbolType, ...]
