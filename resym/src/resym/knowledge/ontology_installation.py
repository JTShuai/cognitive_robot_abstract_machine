"""
Install pinned ontology data outside the source distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resym import AUGMENT_DATASET_DIRECTORY

ONTOLOGY_DIRECTORY = AUGMENT_DATASET_DIRECTORY / "ontology"
"""
Default installation directory of the pinned ontology files.
"""


@dataclass(frozen=True)
class OntologyDownloadFile:
    """
    One pinned ontology file.
    """

    path: str
    """
    Relative installation path.
    """

    source_url: str
    """
    Download location.
    """

    sha256: str
    """
    Expected SHA-256 checksum.
    """

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> OntologyDownloadFile:
        """
        Parse one file record from the download specification.
        """
        return cls(
            path=record["path"],
            source_url=record["source_url"],
            sha256=record["sha256"],
        )

    def to_json(self) -> dict[str, str]:
        """
        Return the runtime manifest representation.
        """
        return {
            "path": self.path,
            "source_url": self.source_url,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class OntologyDownloadSource:
    """
    Versioned source of ontology files.
    """

    source_id: str
    """
    Stable source identifier.
    """

    name: str
    """
    Human-readable source name.
    """

    provenance: str
    """
    Kind of upstream source.
    """

    license: str
    """
    Upstream license identifier.
    """

    release: str | None
    """
    Pinned release, when the source publishes releases.
    """

    git_commit: str | None
    """
    Pinned commit, when the source is revision-based.
    """

    files: tuple[OntologyDownloadFile, ...]
    """
    Files installed from this source.
    """

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> OntologyDownloadSource:
        """
        Parse one source from the download specification.
        """
        return cls(
            source_id=record["id"],
            name=record["name"],
            provenance=record["provenance"],
            license=record["license"],
            release=record.get("release"),
            git_commit=record.get("git_commit"),
            files=tuple(
                OntologyDownloadFile.from_json(file_record)
                for file_record in record["files"]
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """
        Return the runtime manifest representation.
        """
        record: dict[str, Any] = {
            "id": self.source_id,
            "name": self.name,
            "provenance": self.provenance,
            "license": self.license,
            "files": [file.to_json() for file in self.files],
        }
        if self.release is not None:
            record["release"] = self.release
        if self.git_commit is not None:
            record["git_commit"] = self.git_commit
        return record


@dataclass(frozen=True)
class OntologyDownloadSpecification:
    """
    Pinned ontology download manifest.
    """

    schema_version: int
    """
    Manifest schema version.
    """

    sources: tuple[OntologyDownloadSource, ...]
    """
    Ontology sources to install.
    """

    @classmethod
    def from_file(cls, path: Path) -> OntologyDownloadSpecification:
        """
        Load a download specification from JSON.
        """
        record = json.loads(path.read_text())
        return cls(
            schema_version=record["schema_version"],
            sources=tuple(
                OntologyDownloadSource.from_json(source) for source in record["sources"]
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """
        Return the runtime manifest representation.
        """
        return {
            "schema_version": self.schema_version,
            "sources": [source.to_json() for source in self.sources],
        }


@dataclass(frozen=True)
class OntologyInstaller:
    """
    Download and verify one ontology data installation.
    """

    specification_path: Path
    """
    Path to the pinned download specification.
    """

    target_directory: Path
    """
    Directory that receives ontology data.
    """

    def install(self) -> Path:
        """
        Install all files and write their verified runtime manifest.
        """
        specification = OntologyDownloadSpecification.from_file(self.specification_path)
        self.target_directory.mkdir(parents=True, exist_ok=True)
        for source in specification.sources:
            for file in source.files:
                self._install_file(file)

        manifest_path = self.target_directory / "manifest.json"
        temporary_manifest = self.target_directory / ".manifest.json.download"
        temporary_manifest.write_text(
            json.dumps(specification.to_json(), indent=2, sort_keys=True) + "\n"
        )
        temporary_manifest.replace(manifest_path)
        return manifest_path

    def _install_file(self, file: OntologyDownloadFile) -> None:
        destination = (self.target_directory / file.path).resolve()
        root = self.target_directory.resolve()
        if not destination.is_relative_to(root):
            raise InvalidOntologyDownloadPathError(file.path)
        if destination.is_file() and _checksum(destination.read_bytes()) == file.sha256:
            return

        with urllib.request.urlopen(file.source_url, timeout=60) as response:
            content = response.read()
        actual_checksum = _checksum(content)
        if actual_checksum != file.sha256:
            raise DownloadedOntologyHashError(
                file.source_url,
                file.sha256,
                actual_checksum,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_destination = destination.with_name(f".{destination.name}.download")
        temporary_destination.write_bytes(content)
        temporary_destination.replace(destination)


class DownloadedOntologyHashError(Exception):
    """
    A downloaded ontology file did not match its pinned checksum.
    """

    def __init__(self, source_url: str, expected: str, actual: str):
        super().__init__(
            f"Downloaded ontology {source_url} hashes to {actual}, expected {expected}."
        )


class InvalidOntologyDownloadPathError(Exception):
    """
    An ontology download path escapes its target directory.
    """

    def __init__(self, path: str):
        super().__init__(f"Ontology download path escapes its target directory: {path}")


def _checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    """
    Install the ontology data configured for this reSym release.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=ONTOLOGY_DIRECTORY,
        help="ontology installation directory",
    )
    arguments = parser.parse_args()
    specification = Path(__file__).with_name("ontology_sources.json")
    manifest = OntologyInstaller(specification, arguments.target).install()
    print(f"reSym ontologies installed at {manifest.parent}")


if __name__ == "__main__":
    main()
