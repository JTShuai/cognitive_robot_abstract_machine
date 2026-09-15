"""
Check grounded role types against trusted native query interfaces.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import sys
import typing
from dataclasses import dataclass, field
from types import UnionType
from typing import Any, Union, TypeVar, get_args, get_origin
from typing_extensions import TYPE_CHECKING, Iterable

from resym.core.grounding_model import GroundingFactoryCandidate
from resym.core.symbol_types import SymbolType, resolve_symbol_type
from resym.platform.universe import GroundedObject

if TYPE_CHECKING:
    from resym.platform.grounding_catalog import GroundingVocabulary

# %% native interface types


class GroundingTypeError(ValueError):
    """
    A role or attribute does not satisfy a native interface's type contract.
    """


def resolvable_type_hints(subject: Any) -> dict[str, Any]:
    """
    Type hints of a native function or class, resolved one annotation at a time so that
    a name the platform declares only for static checking hides just its own annotation.
    """
    owners = subject.__mro__ if isinstance(subject, type) else (subject,)
    hints: dict[str, Any] = {}
    for owner in reversed(owners):
        namespace = dict(vars(sys.modules[owner.__module__]))
        namespace.update(vars(typing))
        for name, annotation in inspect.get_annotations(owner).items():
            try:
                hints[name] = (
                    eval(annotation, namespace)
                    if isinstance(annotation, str)
                    else annotation
                )
            except (NameError, AttributeError, TypeError):
                hints.pop(name, None)
    return hints


def readable_member_type(owner: type, name: str) -> Any:
    """
    Declared type of a public field or property, with class type variables bound through
    the owner's generic bases.
    """
    descriptor = inspect.getattr_static(owner, name, None)
    result = (
        resolvable_type_hints(descriptor.fget).get("return")
        if isinstance(descriptor, property)
        else resolvable_type_hints(owner).get(name)
    )
    if isinstance(result, TypeVar):
        for ancestor in owner.__mro__:
            for base in ancestor.__dict__.get("__orig_bases__", ()):
                origin = get_origin(base)
                if origin is not None:
                    bindings = dict(
                        zip(origin.__dict__.get("__parameters__", ()), get_args(base))
                    )
                    if result in bindings:
                        result = bindings[result]
    return result


def accepted_role_types(
    vocabulary: GroundingVocabulary, references: Iterable[str]
) -> tuple[Any, ...]:
    """
    Parameter types of the cited queries and owner types of the cited readable members,
    as the types a role may take when the relation is implemented with exactly those
    interfaces.
    """
    entries = {entry.qualified_name: entry for entry in vocabulary.entries}
    accepted: list[Any] = []
    for reference in references:
        entry = entries[reference]
        if entry.owner_type_ref is not None:
            accepted.append(resolve_symbol_type(SymbolType(entry.owner_type_ref)))
            continue
        module = importlib.import_module(entry.module_name)
        subject = module.__dict__[entry.symbol_name]
        hints = resolvable_type_hints(subject)
        hints.pop("return", None)
        accepted.extend(hints.values())
    return tuple(accepted)


def compatible_types(actual: Any, expected: Any) -> bool:
    """
    Whether a value of the actual type satisfies the expected hint, looking through
    unions, type variables and parameterized container origins.
    """
    if expected is Any:
        return True
    if isinstance(expected, TypeVar) or isinstance(actual, TypeVar):
        return True
    if get_origin(actual) in (Union, UnionType):
        return all(compatible_types(option, expected) for option in get_args(actual))
    origin = get_origin(expected)
    if origin in (Union, UnionType):
        return any(compatible_types(actual, option) for option in get_args(expected))
    if origin is not None:
        expected = origin
    actual = get_origin(actual) or actual
    if expected is float and actual is int:
        return True
    return (
        isinstance(actual, type)
        and isinstance(expected, type)
        and issubclass(actual, expected)
    )


# %% type checking of candidate source


@dataclass
class GroundingTypeChecker:
    """
    Infer role-derived expressions without executing candidate source.
    """

    vocabulary: GroundingVocabulary
    """
    Trusted queries and readable attributes.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    """
    Types propagated through local assignments.
    """

    queries: dict[str, Any] = field(default_factory=dict)
    """
    Native callables imported by the candidate.
    """

    roles: tuple[type, ...] = ()
    """
    Concrete input types declared by the candidate.
    """

    def validate(self, candidate: GroundingFactoryCandidate) -> None:
        """
        Reject incompatible role arguments and untyped attribute access.
        """
        self.roles = tuple(
            (
                resolve_symbol_type(role.symbol_type)
                if candidate.native_arguments
                else GroundedObject
            )
            for role in candidate.roles
        )
        tree = ast.parse(candidate.source_code)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                module = importlib.import_module(node.module)
                for alias in node.names:
                    self.queries[alias.asname or alias.name] = module.__dict__[
                        alias.name
                    ]
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        self._statements(function.body)

    def _statements(self, statements: list[ast.stmt]) -> None:
        """
        Propagate types through straight-line code and both sides of branches.
        """
        for statement in statements:
            if isinstance(statement, ast.Assign):
                value_type = self._infer(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        self.variables[target.id] = value_type
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        if (
                            isinstance(statement.value, ast.Name)
                            and statement.value.id == "arguments"
                        ):
                            for index, name in enumerate(target.elts):
                                if isinstance(name, ast.Name) and index < len(
                                    self.roles
                                ):
                                    self.variables[name.id] = self.roles[index]
            elif isinstance(statement, ast.AnnAssign):
                value_type = self._infer(statement.value)
                if isinstance(statement.target, ast.Name):
                    self.variables[statement.target.id] = value_type
            elif isinstance(statement, ast.If):
                self._infer(statement.test)
                before = dict(self.variables)
                self._statements(statement.body)
                left = dict(self.variables)
                self.variables = dict(before)
                self._statements(statement.orelse)
                self.variables = {
                    name: value
                    for name, value in left.items()
                    if self.variables.get(name) == value
                }
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self._infer(child)

    def _infer(self, expression: ast.expr | None) -> Any:
        """
        Resolve expressions derived from roles or native query return annotations.
        """
        if expression is None:
            return None
        if isinstance(expression, ast.Name):
            return self.variables.get(expression.id)
        if isinstance(expression, ast.Constant):
            return type(expression.value)
        if isinstance(expression, ast.Subscript):
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id == "arguments"
                and isinstance(expression.slice, ast.Constant)
                and type(expression.slice.value) is int
            ):
                index = expression.slice.value
                if not -len(self.roles) <= index < len(self.roles):
                    raise GroundingTypeError(
                        f"Role index {index} is outside the declared arguments"
                    )
                return self.roles[index]
            self._infer(expression.value)
            return None
        if isinstance(expression, ast.Attribute):
            return self._attribute_type(self._infer(expression.value), expression.attr)
        if isinstance(expression, ast.Call):
            arguments = [self._infer(value) for value in expression.args]
            keywords = {
                value.arg: self._infer(value.value)
                for value in expression.keywords
                if value.arg
            }
            if (
                isinstance(expression.func, ast.Name)
                and expression.func.id in self.queries
            ):
                function = self.queries[expression.func.id]
                if not inspect.isfunction(function):
                    return None
                hints = resolvable_type_hints(function)
                try:
                    bound = inspect.signature(function).bind(*arguments, **keywords)
                except TypeError as error:
                    raise GroundingTypeError(
                        f"Cannot verify {expression.func.id}: {error}"
                    ) from error
                for name, actual in bound.arguments.items():
                    expected = hints.get(name)
                    if (
                        actual is not None
                        and expected is not None
                        and not compatible_types(actual, expected)
                    ):
                        raise GroundingTypeError(
                            f"{expression.func.id} parameter '{name}' expects {expected}, received {actual}"
                        )
                return hints.get("return")
            if isinstance(expression.func, ast.Attribute):
                self._infer(expression.func.value)
            return None
        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                self._infer(child)
        return None

    def _attribute_type(self, owner: Any, name: str) -> Any:
        """
        Allow only a scanned public member declared on the receiver or one of its
        ancestors.
        """
        ancestors = (
            {
                f"{ancestor.__module__}.{ancestor.__qualname__}"
                for ancestor in owner.__mro__
            }
            if isinstance(owner, type)
            else set()
        )
        for entry in self.vocabulary.entries:
            if entry.owner_type_ref not in ancestors or entry.symbol_name != name:
                continue
            result = readable_member_type(owner, name)
            if result is None:
                raise GroundingTypeError(
                    f"Cannot resolve type of {entry.qualified_name}"
                )
            return result
        raise GroundingTypeError(
            f"Attribute '{name}' is not a scanned readable member of {owner}"
        )
