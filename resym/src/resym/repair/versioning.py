"""
Versioned library store: admission writes versions, never overwrites.

Every admission commits a new immutable version with its parent link,
admission metadata (capability request, backend, suite version,
evidence), and a content checksum. A false admission discovered after
deployment is quarantined and the head rolls back to the parent —
monotonic growth is not assumed. Version Recoverability (revised plan
§4.4): after a rollback, loading the head yields content byte-equal to
the parent version; per-task caches (witness poses, groundings) are
transient by design and never persisted, so they cannot survive a
rollback inconsistently.

Layout of a store directory::

    versions/v0001.json   {"meta": {...}, "library": {...}}
    HEAD                  version id
    quarantine.json       {version id: {"reason": ..., "at": ...}}
    events.jsonl          append-only commit/rollback/quarantine trail
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import Optional

from resym.core.model import SymbolLibrary


class UnknownVersionError(Exception):
    def __init__(self, version_id: str):
        super().__init__(f"No version '{version_id}' in this store.")


class CorruptVersionError(Exception):
    def __init__(self, version_id: str, expected: str, actual: str):
        super().__init__(
            f"Version '{version_id}' is corrupt: content hashes to {actual}, "
            f"metadata says {expected}."
        )


class NothingToRollBackError(Exception):
    def __init__(self, version_id: str):
        super().__init__(f"Version '{version_id}' has no parent to roll back to.")


class NonHeadRollbackError(Exception):
    """
    Raised when a caller tries to roll back an inactive version.
    """

    def __init__(self, version_id: str, head: str):
        super().__init__(
            f"Cannot roll back version '{version_id}': current head is '{head}'."
        )


class EmptyStoreError(Exception):
    def __init__(self):
        super().__init__("The store has no committed version yet.")


@dataclass(frozen=True)
class VersionInfo:
    """
    Metadata of one committed library version.
    """

    version_id: str
    parent: Optional[str]
    created_at: str
    checksum: str
    metadata: dict
    quarantined: bool = False


class VersionedLibraryStore:
    """
    Append-only versioned persistence for one symbol library.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        (directory / "versions").mkdir(parents=True, exist_ok=True)

    # -- committing -----------------------------------------------------

    def commit(self, library: SymbolLibrary, metadata: Optional[dict] = None) -> str:
        """
        Write a new version whose parent is the current head; returns the new version id
        and moves the head to it.
        """
        parent = self._read_head()
        version_id = f"v{self._next_index():04d}"
        content = library.to_json()
        record = {
            "meta": {
                "version_id": version_id,
                "parent": parent,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checksum": _checksum(content),
                "metadata": metadata or {},
            },
            "library": content,
        }
        self._version_path(version_id).write_text(json.dumps(record, indent=2))
        self._write_head(version_id)
        self._event("commit", version_id, parent=parent)
        return version_id

    # -- reading --------------------------------------------------------

    def head_version(self) -> str:
        head = self._read_head()
        if head is None:
            raise EmptyStoreError()
        return head

    def head(self) -> SymbolLibrary:
        return self.load(self.head_version())

    def load(self, version_id: str) -> SymbolLibrary:
        """
        Load one version, verifying its checksum.

        Quarantined versions stay loadable for audit; they are just never the head.
        """
        record = self._read_record(version_id)
        content = record["library"]
        expected = record["meta"]["checksum"]
        actual = _checksum(content)
        if actual != expected:
            raise CorruptVersionError(version_id, expected, actual)
        return SymbolLibrary.from_json(content)

    def info(self, version_id: str) -> VersionInfo:
        meta = self._read_record(version_id)["meta"]
        return VersionInfo(
            version_id=meta["version_id"],
            parent=meta["parent"],
            created_at=meta["created_at"],
            checksum=meta["checksum"],
            metadata=meta["metadata"],
            quarantined=version_id in self._quarantine(),
        )

    def versions(self) -> list[str]:
        return sorted(
            path.stem for path in (self.directory / "versions").glob("v*.json")
        )

    # -- quarantine and rollback ---------------------------------------

    def quarantine_and_roll_back(self, version_id: str, reason: str) -> str:
        """
        Mark a deployed-but-wrong version quarantined and move the head to its parent;
        returns the restored head version id.
        """
        head = self.head_version()
        if version_id != head:
            raise NonHeadRollbackError(version_id, head)
        info = self.info(version_id)
        if info.parent is None:
            raise NothingToRollBackError(version_id)
        quarantine = self._quarantine()
        quarantine[version_id] = {
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        (self.directory / "quarantine.json").write_text(
            json.dumps(quarantine, indent=2)
        )
        self._write_head(info.parent)
        self._event(
            "quarantine_and_roll_back",
            version_id,
            parent=info.parent,
            reason=reason,
        )
        return info.parent

    # -- internals ------------------------------------------------------

    def _version_path(self, version_id: str) -> Path:
        return self.directory / "versions" / f"{version_id}.json"

    def _read_record(self, version_id: str) -> dict:
        path = self._version_path(version_id)
        if not path.exists():
            raise UnknownVersionError(version_id)
        return json.loads(path.read_text())

    def _next_index(self) -> int:
        existing = self.versions()
        return int(existing[-1][1:]) + 1 if existing else 1

    def _read_head(self) -> Optional[str]:
        path = self.directory / "HEAD"
        return path.read_text().strip() if path.exists() else None

    def _write_head(self, version_id: str) -> None:
        (self.directory / "HEAD").write_text(version_id)

    def _quarantine(self) -> dict:
        path = self.directory / "quarantine.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def _event(self, kind: str, version_id: str, **details) -> None:
        with (self.directory / "events.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": kind,
                        "version_id": version_id,
                        "at": datetime.now(timezone.utc).isoformat(),
                        **details,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _checksum(content: dict) -> str:
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
