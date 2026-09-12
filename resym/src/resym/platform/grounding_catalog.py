"""
Discovery, review, persistence, and loading of predicate-grounding factories.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from krrood.adapters.json_serializer import from_json, to_json

from resym.core.capabilities import CapabilityContract
from resym.core.grounding import (
    GroundingFactoryCandidate,
    GroundingFactoryOrigin,
    GroundingFactoryParameter,
    GroundingFactoryProcedure,
    GroundingFactoryReviewStatus,
    GroundingFactorySpec,
    text_checksum as _text_checksum,
)
from resym.platform.feasibility import (
    FEASIBILITY_FACTORY_NAMESPACE,
    capability_feasibility_factories,
)

LOCAL_FACTORY_PACKAGE = "resym_local_grounding_factories"


# %% Source discovery


class GroundingVocabularyKind(StrEnum):
    """
    Kind of reviewed EQL building block found in source code.
    """

    EQL_FACTORY = "eql-factory"
    SYMBOLIC_FUNCTION = "symbolic-function"
    PREDICATE = "predicate"
    QUERY_HELPER = "query-helper"


@dataclass(frozen=True)
class GroundingVocabularyEntry:
    """
    One source-backed EQL symbol available to factory candidates.
    """

    qualified_name: str
    """
    Importable Python identity.
    """

    kind: GroundingVocabularyKind
    """
    Role the symbol has in EQL construction.
    """

    signature: str
    """
    Source signature shown to agents and reviewers.
    """

    source_file: str
    """
    File from which the symbol was discovered.
    """

    source_checksum: str
    """
    Hash of the discovered source file.
    """

    documentation: str = ""
    """Source docstring shown during semantic review and Agent drafting."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GroundingVocabularyKind(self.kind))

    @property
    def module_name(self) -> str:
        """
        Import module containing the symbol.
        """
        return self.qualified_name.rsplit(".", 1)[0]

    @property
    def symbol_name(self) -> str:
        """
        Name imported from :attr:`module_name`.
        """
        return self.qualified_name.rsplit(".", 1)[1]


@dataclass(frozen=True)
class GroundingVocabulary:
    """
    Closed EQL vocabulary that an agent-authored candidate may import.
    """

    entries: tuple[GroundingVocabularyEntry, ...] = ()
    """
    Discovered source-backed EQL symbols.
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))

    def allows(self, module_name: str, symbol_name: str) -> bool:
        """
        Whether one explicit import belongs to this vocabulary.
        """
        return any(
            entry.module_name == module_name and entry.symbol_name == symbol_name
            for entry in self.entries
        )

    def render(self) -> str:
        """
        Compact menu suitable for an agent prompt or Viewer page.
        """
        return "\n".join(
            f"- {entry.qualified_name}{entry.signature}: {entry.kind.value}"
            + (
                f" — {entry.documentation.splitlines()[0]}"
                if entry.documentation
                else ""
            )
            for entry in self.entries
        )


@dataclass(frozen=True)
class GroundingVocabularyCandidate:
    """
    Review state of one source-discovered EQL building block.
    """

    entry: GroundingVocabularyEntry
    """
    Source-backed symbol discovered during initialization.
    """

    review_status: GroundingFactoryReviewStatus = (
        GroundingFactoryReviewStatus.PENDING_REVIEW
    )
    """
    Current human-review decision for this exact source checksum.
    """

    reviewed_by: str | None = None
    """
    Reviewer identity after a decision.
    """

    review_note: str | None = None
    """
    Optional explanation supplied with the decision.
    """

    discovery_scope: str = "default"
    """
    Scanner namespace that owns this record. Synchronizing one source must not
    erase review decisions produced by another source.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "review_status",
            GroundingFactoryReviewStatus(self.review_status),
        )


def discover_grounding_vocabulary(
    package_roots: Mapping[str, Path],
) -> GroundingVocabulary:
    """
    Discover public EQL factories, symbolic functions, and predicate classes.

    The scanner parses source without importing robot or ROS modules.
    """
    entries: dict[str, GroundingVocabularyEntry] = {}
    for root_module, root_path in sorted(package_roots.items()):
        for module_name, source_file in _python_modules(root_module, root_path):
            source = source_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(source_file))
            checksum = _text_checksum(source)
            for node in tree.body:
                kind = _vocabulary_kind(module_name, node)
                if kind is None:
                    continue
                entry = GroundingVocabularyEntry(
                    qualified_name=f"{module_name}.{node.name}",
                    kind=kind,
                    signature=_source_signature(node),
                    source_file=str(source_file),
                    source_checksum=checksum,
                    documentation=ast.get_docstring(node) or "",
                )
                entries[entry.qualified_name] = entry
    return GroundingVocabulary(
        tuple(sorted(entries.values(), key=lambda entry: entry.qualified_name))
    )


def helper_vocabulary(functions: Iterable[Callable]) -> GroundingVocabulary:
    """
    Vocabulary entries for explicitly chosen query-helper callables.

    Each entry pins the checksum of the helper's current module source, so the normal
    vocabulary review flow decides whether candidates may import it.
    """
    entries = []
    for function in functions:
        source_file = Path(inspect.getsourcefile(function))
        entries.append(
            GroundingVocabularyEntry(
                qualified_name=f"{function.__module__}.{function.__name__}",
                kind=GroundingVocabularyKind.QUERY_HELPER,
                signature=str(inspect.signature(function)),
                source_file=str(source_file),
                source_checksum=_text_checksum(source_file.read_text(encoding="utf-8")),
                documentation=inspect.getdoc(function) or "",
            )
        )
    return GroundingVocabulary(
        tuple(sorted(entries, key=lambda entry: entry.qualified_name))
    )


def discover_default_grounding_vocabulary() -> GroundingVocabulary:
    """
    Scan the installed kRrood and Semantic Digital Twin source packages.
    """
    roots = {}
    for package_name in ("krrood", "semantic_digital_twin"):
        specification = importlib.util.find_spec(package_name)
        if specification is None or not specification.submodule_search_locations:
            continue
        roots[package_name] = Path(next(iter(specification.submodule_search_locations)))
    return discover_grounding_vocabulary(roots)


class GroundingFactorySourceError(Exception):
    """
    Raised when candidate source exceeds the reviewed EQL source boundary.
    """


class DuplicateGroundingFactoryCandidateError(Exception):
    """
    Raised when a review workspace already contains the candidate identity.
    """


class UnknownGroundingFactoryCandidateError(Exception):
    """
    Raised when a review decision names no stored candidate.
    """


class GroundingFactoryUidConflictError(Exception):
    """
    Raised when a candidate claims a factory identity another source owns.
    """


class GroundingFactoryCatalogError(Exception):
    """
    Raised when an approved factory cannot be resolved without semantic drift.
    """


# %% Candidate source validation


@dataclass(frozen=True)
class GroundingFactorySourceValidator:
    """
    Static boundary for a native-EQL factory candidate.
    """

    vocabulary: GroundingVocabulary
    """
    Only symbols in this reviewed vocabulary may be imported.
    """

    maximum_tree_depth: int = 18
    """
    Maximum syntax-tree nesting accepted for one candidate.
    """

    def objections(self, source_code: str) -> tuple[str, ...]:
        """
        Return a deterministic static verdict suitable for an agent tool.
        """
        try:
            self.validate(source_code)
        except GroundingFactorySourceError as error:
            return (str(error),)
        return ()

    def candidate_objections(
        self, candidate: GroundingFactoryCandidate
    ) -> tuple[str, ...]:
        """
        Return source and interface objections for one complete candidate.
        """
        source_objections = self.objections(candidate.source_code)
        if source_objections:
            return source_objections
        try:
            _validate_candidate_interface(candidate)
        except GroundingFactorySourceError as error:
            return (str(error),)
        return ()

    def validate(self, source_code: str) -> None:
        """
        Raise when source is not a bounded, read-only EQL evaluator.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as error:
            raise GroundingFactorySourceError(str(error)) from error
        if _tree_depth(tree) > self.maximum_tree_depth:
            raise GroundingFactorySourceError("candidate syntax is too deeply nested")
        self._validate_top_level(tree)
        self._validate_imports(tree)
        self._validate_function(tree)
        self._validate_calls(tree)

    def _validate_top_level(self, tree: ast.Module) -> None:
        allowed = (ast.ImportFrom, ast.FunctionDef)
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node, allowed):
                raise GroundingFactorySourceError(
                    f"top-level {type(node).__name__} is not allowed"
                )

    def _validate_imports(self, tree: ast.Module) -> None:
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "__future__":
                if tuple(alias.name for alias in node.names) != ("annotations",):
                    raise GroundingFactorySourceError(
                        "only __future__.annotations may be imported"
                    )
                continue
            if node.level != 0 or node.module is None:
                raise GroundingFactorySourceError("relative imports are not allowed")
            for alias in node.names:
                if alias.name == "*" or not self.vocabulary.allows(
                    node.module, alias.name
                ):
                    raise GroundingFactorySourceError(
                        f"'{node.module}.{alias.name}' is outside the reviewed EQL vocabulary"
                    )

    def _validate_function(self, tree: ast.Module) -> None:
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1 or functions[0].name != "evaluate":
            raise GroundingFactorySourceError(
                "candidate must define exactly one function named 'evaluate'"
            )
        function = functions[0]
        if function.decorator_list:
            raise GroundingFactorySourceError("candidate decorators are not allowed")
        arguments = tuple(argument.arg for argument in function.args.args)
        if arguments != ("context", "universe", "arguments", "parameters"):
            raise GroundingFactorySourceError(
                "evaluate must accept (context, universe, arguments, parameters)"
            )
        forbidden = (
            ast.AsyncFunctionDef,
            ast.Await,
            ast.ClassDef,
            ast.Delete,
            ast.For,
            ast.Global,
            ast.Import,
            ast.Lambda,
            ast.Nonlocal,
            ast.Raise,
            ast.Try,
            ast.While,
            ast.With,
            ast.Yield,
            ast.YieldFrom,
        )
        for node in ast.walk(function):
            if isinstance(node, forbidden):
                raise GroundingFactorySourceError(
                    f"{type(node).__name__} is not allowed in a factory candidate"
                )
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise GroundingFactorySourceError("dunder access is not allowed")
        returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
        if not returns or any(
            not _is_boolean_expression(node.value) for node in returns
        ):
            raise GroundingFactorySourceError(
                "every factory return must be statically Boolean"
            )

    def _validate_calls(self, tree: ast.Module) -> None:
        imported_names = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module != "__future__"
            for alias in node.names
        }
        allowed_builtins = {"all", "any", "bool", "len", "list", "tuple"}
        allowed_methods = {
            "evaluate",
            "grouped_by",
            "having",
            "limit",
            "ordered_by",
            "where",
        }
        called_attributes = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and id(node) not in called_attributes:
                raise GroundingFactorySourceError(
                    f"attribute read '{node.attr}' is outside the reviewed EQL boundary"
                )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id not in imported_names | allowed_builtins:
                    raise GroundingFactorySourceError(
                        f"call to '{node.func.id}' is outside the reviewed EQL boundary"
                    )
                continue
            if isinstance(node.func, ast.Attribute):
                if node.func.attr not in allowed_methods:
                    raise GroundingFactorySourceError(
                        f"method '{node.func.attr}' is outside the reviewed EQL boundary"
                    )
                continue
            raise GroundingFactorySourceError("dynamic calls are not allowed")


# %% Local review workspace


@dataclass
class GroundingFactoryWorkspace:
    """
    Local review workspace for candidates and approved source modules.
    """

    root: Path
    """
    Directory containing the review queue, catalog, and approved package.
    """

    @property
    def candidate_directory(self) -> Path:
        """
        Non-importable pending-review directory.
        """
        return self.root / "candidates"

    @property
    def approved_package_directory(self) -> Path:
        """
        Importable package holding locally approved implementations.
        """
        return self.root / "approved" / LOCAL_FACTORY_PACKAGE

    @property
    def catalog_path(self) -> Path:
        """
        Current local factory catalog.
        """
        return self.root / "catalog.json"

    @property
    def review_log_path(self) -> Path:
        """
        Append-only local review history.
        """
        return self.root / "review_events.jsonl"

    @property
    def vocabulary_review_path(self) -> Path:
        """
        Review records for source-discovered EQL building blocks.
        """
        return self.root / "grounding_vocabulary.json"

    def synchronize_vocabulary(
        self,
        discovered: GroundingVocabulary,
        discovery_scope: str = "default",
    ) -> None:
        """
        Update the review queue while preserving unchanged decisions.
        """
        records = self.vocabulary_candidates()
        discovered_names = {entry.qualified_name for entry in discovered.entries}
        existing = {
            item.entry.qualified_name: item
            for item in records
            if item.discovery_scope == discovery_scope
            or item.entry.qualified_name in discovered_names
        }
        synchronized = [
            item
            for item in records
            if item.discovery_scope != discovery_scope
            and item.entry.qualified_name not in discovered_names
        ]
        for entry in discovered.entries:
            previous = existing.get(entry.qualified_name)
            if (
                previous is not None
                and previous.entry.source_checksum == entry.source_checksum
            ):
                synchronized.append(
                    replace(
                        previous,
                        entry=entry,
                        discovery_scope=discovery_scope,
                    )
                )
            else:
                synchronized.append(
                    GroundingVocabularyCandidate(
                        entry=entry,
                        discovery_scope=discovery_scope,
                    )
                )
        self.root.mkdir(parents=True, exist_ok=True)
        self.vocabulary_review_path.write_text(
            json.dumps(
                {"entries": [to_json(item) for item in synchronized]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def vocabulary_candidates(self) -> tuple[GroundingVocabularyCandidate, ...]:
        """
        Return every currently discovered vocabulary review record.
        """
        if not self.vocabulary_review_path.is_file():
            return ()
        data = json.loads(self.vocabulary_review_path.read_text(encoding="utf-8"))
        return tuple(from_json(item) for item in data.get("entries", ()))

    def reviewed_vocabulary(self) -> GroundingVocabulary:
        """
        Return only unchanged EQL symbols approved by a human.
        """
        return GroundingVocabulary(
            tuple(
                item.entry
                for item in self.vocabulary_candidates()
                if item.review_status == GroundingFactoryReviewStatus.APPROVED
            )
        )

    def approve_vocabulary(
        self,
        qualified_name: str,
        reviewer: str,
        review_note: str | None = None,
    ) -> GroundingVocabularyCandidate:
        """
        Approve one discovered symbol for bounded factory composition.
        """
        return self._review_vocabulary(
            qualified_name,
            GroundingFactoryReviewStatus.APPROVED,
            reviewer,
            review_note,
        )

    def reject_vocabulary(
        self,
        qualified_name: str,
        reviewer: str,
        review_note: str,
    ) -> GroundingVocabularyCandidate:
        """
        Reject one discovered symbol from bounded factory composition.
        """
        return self._review_vocabulary(
            qualified_name,
            GroundingFactoryReviewStatus.REJECTED,
            reviewer,
            review_note,
        )

    def _review_vocabulary(
        self,
        qualified_name: str,
        status: GroundingFactoryReviewStatus,
        reviewer: str,
        review_note: str | None,
    ) -> GroundingVocabularyCandidate:
        records = list(self.vocabulary_candidates())
        matches = [
            (index, item)
            for index, item in enumerate(records)
            if item.entry.qualified_name == qualified_name
        ]
        if not matches:
            raise GroundingFactoryCatalogError(
                f"No discovered EQL symbol under '{qualified_name}'."
            )
        index, current = matches[0]
        if current.review_status != GroundingFactoryReviewStatus.PENDING_REVIEW:
            raise GroundingFactoryCatalogError(
                f"EQL symbol '{qualified_name}' already has a review decision."
            )
        reviewed = replace(
            current,
            review_status=status,
            reviewed_by=reviewer,
            review_note=review_note,
        )
        records[index] = reviewed
        self.vocabulary_review_path.write_text(
            json.dumps(
                {"entries": [to_json(item) for item in records]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return reviewed

    def submit(self, candidate: GroundingFactoryCandidate) -> None:
        """
        Persist a candidate outside executable Python paths.
        """
        path = self._candidate_path(candidate.candidate_id)
        if path.exists():
            raise DuplicateGroundingFactoryCandidateError(candidate.candidate_id)
        self.candidate_directory.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_json(candidate), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def candidates(self) -> tuple[GroundingFactoryCandidate, ...]:
        """
        All candidate records, including completed reviews.
        """
        if not self.candidate_directory.is_dir():
            return ()
        return tuple(
            from_json(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.candidate_directory.glob("*.json"))
        )

    def specifications(self) -> tuple[GroundingFactorySpec, ...]:
        """
        Current locally approved factory specifications.
        """
        if not self.catalog_path.is_file():
            return ()
        data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return tuple(from_json(item) for item in data.get("factories", ()))

    def approve(
        self,
        candidate_id: str,
        reviewer: str,
        vocabulary: GroundingVocabulary,
        review_note: str | None = None,
    ) -> GroundingFactorySpec:
        """
        Validate and materialize one human-approved candidate.
        """
        candidate = self._candidate(candidate_id)
        if candidate.review_status != GroundingFactoryReviewStatus.PENDING_REVIEW:
            raise GroundingFactorySourceError("candidate already has a review decision")
        if candidate.proposed_uid.startswith(FEASIBILITY_FACTORY_NAMESPACE):
            raise GroundingFactoryUidConflictError(
                f"factory uid '{candidate.proposed_uid}' lies in the reserved "
                "capability-feasibility namespace"
            )
        GroundingFactorySourceValidator(vocabulary).validate(candidate.source_code)
        _validate_candidate_interface(candidate)
        self.approved_package_directory.mkdir(parents=True, exist_ok=True)
        package_file = self.approved_package_directory / "__init__.py"
        if not package_file.exists():
            package_file.write_text("", encoding="utf-8")
        module_stem = f"factory_{_text_checksum(candidate.proposed_uid)[:16]}"
        source_file = self.approved_package_directory / f"{module_stem}.py"
        implementation_ref = f"{LOCAL_FACTORY_PACKAGE}.{module_stem}:evaluate"
        with tempfile.TemporaryDirectory(dir=self.root) as staging_directory:
            staged_source = Path(staging_directory) / source_file.name
            staged_source.write_text(candidate.source_code, encoding="utf-8")
            _load_function(staged_source, implementation_ref)
            shutil.copy2(staged_source, source_file)

        existing = {item.uid: item for item in self.specifications()}
        previous = existing.get(candidate.proposed_uid)
        revision = 1 if previous is None else int(previous.active_revision_id[1:]) + 1
        specification = GroundingFactorySpec(
            uid=candidate.proposed_uid,
            semantic_name=candidate.semantic_name,
            implementation_ref=implementation_ref,
            implementation_checksum=_file_checksum(source_file),
            roles=candidate.roles,
            origin=GroundingFactoryOrigin.LOCAL,
            reviewed_by=reviewer,
            approved_at=datetime.now(UTC).isoformat(),
            active_revision_id=f"r{revision:04d}",
            parameters=candidate.parameters,
            dependency_checksums=_candidate_dependency_checksums(
                candidate.source_code, vocabulary
            ),
        )
        existing[specification.uid] = specification
        self._write_catalog(existing.values())
        self._record_candidate_decision(
            candidate,
            GroundingFactoryReviewStatus.APPROVED,
            reviewer,
            review_note,
        )
        self._record_review_event(
            "approved",
            candidate,
            reviewer,
            review_note,
            specification,
            previous,
        )
        return specification

    def reject(
        self, candidate_id: str, reviewer: str, review_note: str
    ) -> GroundingFactoryCandidate:
        """
        Record a human rejection without materializing executable code.
        """
        candidate = self._candidate(candidate_id)
        if candidate.review_status != GroundingFactoryReviewStatus.PENDING_REVIEW:
            raise GroundingFactorySourceError("candidate already has a review decision")
        rejected = self._record_candidate_decision(
            candidate,
            GroundingFactoryReviewStatus.REJECTED,
            reviewer,
            review_note,
        )
        self._record_review_event(
            "rejected", candidate, reviewer, review_note, None, None
        )
        return rejected

    def implementation_drift(self, specification: GroundingFactorySpec) -> str | None:
        """
        Describe how the materialized source diverged from approval, if it did.
        """
        source_file = self._approved_source_file(specification)
        if not source_file.is_file():
            return f"factory '{specification.uid}' source file is missing"
        if not specification.dependency_checksums and _source_imports_dependencies(
            source_file.read_text(encoding="utf-8")
        ):
            return (
                f"factory '{specification.uid}' predates dependency pinning and "
                "requires review"
            )
        actual_checksum = _file_checksum(source_file)
        if actual_checksum != specification.implementation_checksum:
            return (
                f"factory '{specification.uid}' source changed from "
                f"{specification.implementation_checksum} to {actual_checksum}"
            )
        for qualified_name, expected_checksum in specification.dependency_checksums:
            actual_checksum = _qualified_name_source_checksum(qualified_name)
            if actual_checksum is None:
                return (
                    f"factory '{specification.uid}' dependency "
                    f"'{qualified_name}' is missing"
                )
            if actual_checksum != expected_checksum:
                return (
                    f"factory '{specification.uid}' dependency '{qualified_name}' "
                    f"changed from {expected_checksum} to {actual_checksum}"
                )
        return None

    def load_procedure(
        self, specification: GroundingFactorySpec
    ) -> GroundingFactoryProcedure:
        """
        Load one approved-local callable after verifying its source hash.
        """
        drift = self.implementation_drift(specification)
        if drift is not None:
            raise GroundingFactoryCatalogError(drift)
        return _load_function(
            self._approved_source_file(specification),
            specification.implementation_ref,
        )

    def _approved_source_file(self, specification: GroundingFactorySpec) -> Path:
        module_name, _ = specification.implementation_ref.split(":", 1)
        return self.approved_package_directory / f"{module_name.rsplit('.', 1)[1]}.py"

    def _candidate(self, candidate_id: str) -> GroundingFactoryCandidate:
        path = self._candidate_path(candidate_id)
        if not path.is_file():
            raise UnknownGroundingFactoryCandidateError(candidate_id)
        return from_json(json.loads(path.read_text(encoding="utf-8")))

    def _candidate_path(self, candidate_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_id):
            raise ValueError(f"Invalid candidate id '{candidate_id}'.")
        return self.candidate_directory / f"{candidate_id}.json"

    def _record_candidate_decision(
        self,
        candidate: GroundingFactoryCandidate,
        status: GroundingFactoryReviewStatus,
        reviewer: str,
        review_note: str | None,
    ) -> GroundingFactoryCandidate:
        reviewed = replace(
            candidate,
            review_status=status,
            reviewed_by=reviewer,
            review_note=review_note,
        )
        self._candidate_path(candidate.candidate_id).write_text(
            json.dumps(to_json(reviewed), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return reviewed

    def _write_catalog(self, specifications: Iterable[GroundingFactorySpec]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "factories": [
                to_json(specification)
                for specification in sorted(specifications, key=lambda item: item.uid)
            ]
        }
        self.catalog_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _record_review_event(
        self,
        decision: str,
        candidate: GroundingFactoryCandidate,
        reviewer: str,
        review_note: str | None,
        specification: GroundingFactorySpec | None,
        previous: GroundingFactorySpec | None,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "decision": decision,
            "candidate_id": candidate.candidate_id,
            "factory_uid": candidate.proposed_uid,
            "reviewed_by": reviewer,
            "reviewed_at": datetime.now(UTC).isoformat(),
            "review_note": review_note,
            "implementation_checksum": (
                specification.implementation_checksum
                if specification is not None
                else None
            ),
            "supersedes_checksum": (
                previous.implementation_checksum if previous is not None else None
            ),
        }
        with self.review_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


# %% Runtime catalog


@dataclass
class GroundingFactoryCatalog:
    """
    Unified runtime view of platform, local, and shared factories.
    """

    _specifications: dict[str, GroundingFactorySpec]
    """
    One current approved specification per factory identity.
    """

    _procedures: dict[str, GroundingFactoryProcedure]
    """
    Verified callable corresponding to every specification.
    """

    _unavailable: dict[str, str] = field(default_factory=dict)
    """
    Reason each currently unresolvable factory identity cannot be used.
    """

    _workspace: GroundingFactoryWorkspace | None = field(default=None, repr=False)
    """
    Review workspace backing locally approved factories, when one was loaded.
    """

    @property
    def workspace(self) -> GroundingFactoryWorkspace | None:
        """Local review workspace available to an application repair loop."""
        return self._workspace

    def __iter__(self) -> Iterator[GroundingFactorySpec]:
        return iter(sorted(self._specifications.values(), key=lambda item: item.uid))

    @property
    def unavailable(self) -> Mapping[str, str]:
        """
        Unresolvable factory identities and why each one is unavailable.
        """
        return dict(self._unavailable)

    @classmethod
    def load(
        cls,
        workspace: GroundingFactoryWorkspace | None = None,
        shared_specifications: Iterable[GroundingFactorySpec] = (),
        capability_contracts: Iterable[CapabilityContract] = (),
        capability_feasibility_implementations: Mapping[str, Callable] | None = None,
    ) -> GroundingFactoryCatalog:
        """
        Load the unified catalog with duplicate identities rejected.

        The platform contributes no factories of its own: every entry is a
        locally approved implementation, a shared reviewed specification, or a
        feasibility factory derived from a reviewed capability contract that has a
        concrete platform feasibility implementation. A
        locally approved factory whose materialized source drifted from its
        reviewed checksum is marked unavailable instead of failing the load.
        """
        catalog = cls({}, {}, _workspace=workspace)
        for specification, procedure in capability_feasibility_factories(
            capability_contracts,
            capability_feasibility_implementations,
        ):
            catalog._add(specification, procedure)
        if workspace is not None:
            for specification in workspace.specifications():
                drift = workspace.implementation_drift(specification)
                if drift is not None:
                    catalog._unavailable[specification.uid] = drift
                    continue
                catalog._add(specification, workspace.load_procedure(specification))
        for specification in shared_specifications:
            procedure = _import_function(specification.implementation_ref)
            actual_checksum = _callable_source_checksum(procedure)
            if actual_checksum != specification.implementation_checksum:
                catalog._unavailable[specification.uid] = (
                    f"shared factory '{specification.uid}' source changed from "
                    f"{specification.implementation_checksum} to {actual_checksum}"
                )
                continue
            catalog._add(specification, procedure)
        return catalog

    def specification(self, uid: str) -> GroundingFactorySpec:
        """
        Current approved specification under one identity.
        """
        if uid in self._unavailable:
            raise GroundingFactoryCatalogError(self._unavailable[uid])
        try:
            return self._specifications[uid]
        except KeyError as error:
            raise GroundingFactoryCatalogError(
                f"No approved grounding factory under '{uid}'."
            ) from error

    def resolve(
        self, uid: str, expected_checksum: str | None = None
    ) -> GroundingFactoryProcedure:
        """
        Resolve a callable only when the approved implementation still matches.
        """
        specification = self.specification(uid)
        if (
            expected_checksum is not None
            and expected_checksum != specification.implementation_checksum
        ):
            raise GroundingFactoryCatalogError(
                f"factory '{uid}' changed from {expected_checksum} to "
                f"{specification.implementation_checksum}; its plan requires review"
            )
        return self._procedures[uid]

    def supports(self, uid: str, embodiment: str) -> bool:
        """
        Whether the current implementation supports an embodiment profile.
        """
        specification = self._specifications.get(uid)
        if specification is None:
            return False
        return (
            not specification.supported_embodiments
            or embodiment in specification.supported_embodiments
        )

    def to_json(self) -> dict:
        """
        JSON-ready catalog snapshot without executable callables.
        """
        return {
            "factories": [to_json(item) for item in self],
            "unavailable": dict(self._unavailable),
        }

    def render(self) -> str:
        """
        Compact, checksum-bearing menu for agents and human review.
        """
        lines = []
        for specification in self:
            roles = ", ".join(
                f"{role.name}: {role.symbol_type.python_type_ref}"
                for role in specification.roles
            )
            parameters = ", ".join(
                _render_parameter(parameter) for parameter in specification.parameters
            )
            lines.append(
                f"- {specification.uid} [{roles}] "
                f"parameters=[{parameters}] -> bool; "
                f"checksum={specification.implementation_checksum}; "
                f"origin={specification.origin.value}"
            )
        return "\n".join(lines)

    def _add(
        self,
        specification: GroundingFactorySpec,
        procedure: GroundingFactoryProcedure,
    ) -> None:
        if specification.uid in self._specifications:
            raise GroundingFactoryCatalogError(
                f"Duplicate current grounding factory '{specification.uid}'."
            )
        self._specifications[specification.uid] = specification
        self._procedures[specification.uid] = procedure


# %% Initialization and experiment snapshots


@dataclass(frozen=True)
class GroundingFactoryInitialization:
    """
    Complete predicate-grounding state produced during system startup.
    """

    workspace: GroundingFactoryWorkspace
    """
    Local persistence and review boundary.
    """

    discovered_vocabulary: GroundingVocabulary
    """
    Source symbols found in the current kRrood and SDT installation.
    """

    reviewed_vocabulary: GroundingVocabulary
    """
    Unchanged symbols explicitly approved for candidate composition.
    """

    catalog: GroundingFactoryCatalog
    """
    Unified current runtime catalog of approved factories.
    """

    @property
    def factory_specs(self) -> Mapping[str, GroundingFactorySpec]:
        """
        Specifications suitable for deterministic Curator checks.
        """
        return {specification.uid: specification for specification in self.catalog}

    def candidate_objections(
        self, candidate: GroundingFactoryCandidate
    ) -> tuple[str, ...]:
        """
        Validate an Agent candidate against the reviewed startup vocabulary.
        """
        return GroundingFactorySourceValidator(
            self.reviewed_vocabulary
        ).candidate_objections(candidate)

    def submit_candidate(self, candidate: GroundingFactoryCandidate) -> None:
        """
        Place an Agent candidate in the local human-review queue.
        """
        self.workspace.submit(candidate)


def initialize_grounding_factories(
    workspace_root: Path,
    package_roots: Mapping[str, Path] | None = None,
    capability_contracts: Iterable[CapabilityContract] = (),
    capability_feasibility_implementations: Mapping[str, Callable] | None = None,
) -> GroundingFactoryInitialization:
    """
    Scan query sources, update their review queue, and load approved factories.
    """
    discovered = (
        discover_default_grounding_vocabulary()
        if package_roots is None
        else discover_grounding_vocabulary(package_roots)
    )
    workspace = GroundingFactoryWorkspace(workspace_root)
    workspace.synchronize_vocabulary(discovered, discovery_scope="platform-default")
    return GroundingFactoryInitialization(
        workspace=workspace,
        discovered_vocabulary=discovered,
        reviewed_vocabulary=workspace.reviewed_vocabulary(),
        catalog=GroundingFactoryCatalog.load(
            workspace=workspace,
            capability_contracts=capability_contracts,
            capability_feasibility_implementations=(
                capability_feasibility_implementations
            ),
        ),
    )


@dataclass(frozen=True)
class GroundingFactorySnapshot:
    """
    Immutable experiment copy of local factory source and catalog metadata.
    """

    directory: Path
    """
    Snapshot root.
    """

    source_bundle: Path
    """
    Copied approved-local Python package.
    """

    manifest: Path
    """
    Checksummed experiment manifest.
    """

    catalog_checksum: str
    """
    Hash of the frozen catalog.
    """

    symbol_library_checksum: str
    """
    Hash of the frozen task-level symbol library.
    """


def freeze_grounding_factories(
    workspace: GroundingFactoryWorkspace,
    output_directory: Path,
    symbol_library: Path,
    capability_contracts: Iterable[CapabilityContract] = (),
    capability_feasibility_implementations: Mapping[str, Callable] | None = None,
    git_commits: tuple[tuple[str, str], ...] = (),
    package_versions: tuple[tuple[str, str], ...] = (),
    container_image_digest: str | None = None,
    experiment_configuration: tuple[tuple[str, str], ...] = (),
) -> GroundingFactorySnapshot:
    """
    Freeze local source, catalog, library, and environment identities for one run.
    """
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    source_bundle = output_directory / "factory_source_bundle"
    source_bundle.mkdir()
    if workspace.approved_package_directory.is_dir():
        shutil.copytree(
            workspace.approved_package_directory,
            source_bundle / LOCAL_FACTORY_PACKAGE,
        )
    review_metadata = output_directory / "review_metadata"
    review_metadata.mkdir()
    for source in (
        workspace.review_log_path,
        workspace.vocabulary_review_path,
    ):
        if source.is_file():
            shutil.copy2(source, review_metadata / source.name)
    frozen_catalog = output_directory / "grounding_factory_catalog.json"
    frozen_catalog.write_text(
        json.dumps(
            GroundingFactoryCatalog.load(
                workspace=workspace,
                capability_contracts=capability_contracts,
                capability_feasibility_implementations=(
                    capability_feasibility_implementations
                ),
            ).to_json(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    frozen_library = output_directory / "symbol_library.json"
    shutil.copy2(symbol_library, frozen_library)
    manifest = output_directory / "manifest.json"
    catalog_checksum = _file_checksum(frozen_catalog)
    symbol_library_checksum = _file_checksum(frozen_library)
    manifest.write_text(
        json.dumps(
            {
                "factory_source_bundle_checksum": _directory_checksum(source_bundle),
                "review_metadata_checksum": _directory_checksum(review_metadata),
                "catalog_checksum": catalog_checksum,
                "symbol_library_checksum": symbol_library_checksum,
                "git_commits": dict(git_commits),
                "package_versions": dict(package_versions),
                "container_image_digest": container_image_digest,
                "experiment_configuration": dict(experiment_configuration),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return GroundingFactorySnapshot(
        directory=output_directory,
        source_bundle=source_bundle,
        manifest=manifest,
        catalog_checksum=catalog_checksum,
        symbol_library_checksum=symbol_library_checksum,
    )


# %% Source and implementation helpers


def _python_modules(root_module: str, root_path: Path) -> Iterator[tuple[str, Path]]:
    if root_path.is_file():
        yield root_module, root_path
        return
    for source_file in sorted(root_path.rglob("*.py")):
        relative = source_file.relative_to(root_path)
        parts = relative.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        module_name = ".".join((root_module, *parts)) if parts else root_module
        yield module_name, source_file


def _vocabulary_kind(
    module_name: str, node: ast.stmt
) -> GroundingVocabularyKind | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        decorators = {_terminal_name(item) for item in node.decorator_list}
        if "overload" in decorators:
            return None
        if "symbolic_function" in decorators:
            return GroundingVocabularyKind.SYMBOLIC_FUNCTION
        if (
            module_name == "krrood.entity_query_language.factories"
            and not node.name.startswith("_")
        ):
            return GroundingVocabularyKind.EQL_FACTORY
    if isinstance(node, ast.ClassDef):
        bases = {_terminal_name(item) for item in node.bases}
        if "Predicate" in bases:
            return GroundingVocabularyKind.PREDICATE
        if "SymbolicFunction" in bases:
            return GroundingVocabularyKind.SYMBOLIC_FUNCTION
    return None


def _terminal_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _terminal_name(node.func)
    return ""


def _source_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> str:
    if isinstance(node, ast.ClassDef):
        return ""
    positional = list(node.args.posonlyargs) + list(node.args.args)
    default_offset = len(positional) - len(node.args.defaults)
    arguments = []
    for index, argument in enumerate(positional):
        rendered = argument.arg
        if argument.annotation is not None:
            rendered += f": {ast.unparse(argument.annotation)}"
        if index >= default_offset:
            rendered += f" = {ast.unparse(node.args.defaults[index - default_offset])}"
        arguments.append(rendered)
    if node.args.vararg is not None:
        arguments.append(f"*{node.args.vararg.arg}")
    arguments.extend(
        f"{argument.arg}"
        + (f" = {ast.unparse(default)}" if default is not None else "")
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
    )
    if node.args.kwarg is not None:
        arguments.append(f"**{node.args.kwarg.arg}")
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"({', '.join(arguments)}){returns}"


def _tree_depth(node: ast.AST) -> int:
    children = tuple(ast.iter_child_nodes(node))
    return 1 if not children else 1 + max(_tree_depth(child) for child in children)


def _is_boolean_expression(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is bool
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
    )


def _validate_candidate_interface(candidate: GroundingFactoryCandidate) -> None:
    """
    Reject duplicate names and invalid numeric constraints before admission.
    """
    role_names = [role.name for role in candidate.roles]
    if len(role_names) != len(set(role_names)):
        raise GroundingFactorySourceError("factory role names must be unique")
    parameter_names = [parameter.name for parameter in candidate.parameters]
    if len(parameter_names) != len(set(parameter_names)):
        raise GroundingFactorySourceError("factory parameter names must be unique")
    for parameter in candidate.parameters:
        if (
            parameter.minimum is not None
            and parameter.maximum is not None
            and parameter.minimum > parameter.maximum
        ):
            raise GroundingFactorySourceError(
                f"factory parameter '{parameter.name}' has an invalid range"
            )


def _render_parameter(parameter: GroundingFactoryParameter) -> str:
    """
    Render one factory parameter for the Agent's catalog menu.
    """
    lower = "" if parameter.minimum is None else str(parameter.minimum)
    upper = "" if parameter.maximum is None else str(parameter.maximum)
    bounds = (
        f"[{lower},{upper}]"
        if parameter.minimum is not None or parameter.maximum is not None
        else ""
    )
    required = "" if parameter.required else "?"
    return f"{parameter.name}{required}:{parameter.value_type.value}{bounds}"


def _candidate_dependency_checksums(
    source_code: str, vocabulary: GroundingVocabulary
) -> tuple[tuple[str, str], ...]:
    """Pin every reviewed symbol imported by a materialized candidate."""
    approved = {entry.qualified_name: entry for entry in vocabulary.entries}
    dependencies: dict[str, str] = {}
    tree = ast.parse(source_code)
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module in (None, "__future__"):
            continue
        for alias in node.names:
            qualified_name = f"{node.module}.{alias.name}"
            dependencies[qualified_name] = approved[qualified_name].source_checksum
    return tuple(sorted(dependencies.items()))


def _source_imports_dependencies(source_code: str) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and node.module not in (None, "__future__")
        for node in ast.parse(source_code).body
    )


def _load_function(
    source_file: Path, implementation_ref: str
) -> GroundingFactoryProcedure:
    module_name, function_name = implementation_ref.split(":", 1)
    specification = importlib.util.spec_from_file_location(module_name, source_file)
    if specification is None or specification.loader is None:
        raise GroundingFactoryCatalogError(f"Cannot load '{implementation_ref}'.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    procedure = vars(module).get(function_name)
    if not callable(procedure):
        raise GroundingFactoryCatalogError(
            f"'{implementation_ref}' does not resolve to a callable."
        )
    return procedure


def _import_function(implementation_ref: str) -> GroundingFactoryProcedure:
    module_name, function_name = implementation_ref.split(":", 1)
    module = importlib.import_module(module_name)
    procedure = vars(module).get(function_name)
    if not callable(procedure):
        raise GroundingFactoryCatalogError(
            f"'{implementation_ref}' does not resolve to a callable."
        )
    return procedure


def _qualified_name_source_checksum(qualified_name: str) -> str | None:
    module_name, _ = qualified_name.rsplit(".", 1)
    specification = importlib.util.find_spec(module_name)
    if specification is None or specification.origin is None:
        return None
    source_file = Path(specification.origin)
    return _file_checksum(source_file) if source_file.is_file() else None


def _callable_source_checksum(procedure: Callable) -> str:
    source_file = inspect.getsourcefile(procedure)
    if source_file is None:
        raise GroundingFactoryCatalogError(
            f"Cannot locate reviewed source for '{procedure}'."
        )
    return _file_checksum(Path(source_file))


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_checksum(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
