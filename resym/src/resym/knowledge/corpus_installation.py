"""
Install the pinned UniDomain retrieval corpus outside the source distribution.

The installer downloads one checksummed archive of the raw dataset, extracts
it, and freezes it into the versioned retrieval release experiments read
(:mod:`resym.knowledge.freeze`). The frozen release must reproduce the pinned
fragment checksum, so every installation serves the exact corpus the
experiments were designed against.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resym import AUGMENT_DATASET_DIRECTORY
from resym.knowledge.freeze import build_release, sha256_of_file

UNIDOMAIN_DIRECTORY = AUGMENT_DATASET_DIRECTORY / "unidomain"
"""
Default directory of the extracted raw UniDomain dataset.
"""

CORPUS_RELEASE_DIRECTORY = AUGMENT_DATASET_DIRECTORY / "corpus_release" / "r1"
"""
Default directory of the frozen retrieval release experiments read.
"""

RAW_CORPUS_MANIFEST_NAME = "raw_manifest.json"
"""
Stamp recording which pinned archive an extracted raw corpus came from.
"""


@dataclass(frozen=True)
class CorpusArchive:
    """
    The pinned downloadable archive of the raw dataset.
    """

    source_url: str
    """
    Download location.
    """

    sha256: str
    """
    Expected SHA-256 checksum of the archive.
    """

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> CorpusArchive:
        """
        Parse the archive record from the source specification.
        """
        return cls(source_url=record["source_url"], sha256=record["sha256"])


@dataclass(frozen=True)
class CorpusSource:
    """
    Versioned upstream source of the raw corpus.
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

    repository: str
    """
    Upstream repository identifier.
    """

    revision: str
    """
    Pinned upstream revision.
    """

    license: str
    """
    Upstream license identifier.
    """

    archive: CorpusArchive
    """
    The pinned archive installed from this source.
    """

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> CorpusSource:
        """
        Parse the source from the source specification.
        """
        return cls(
            source_id=record["id"],
            name=record["name"],
            provenance=record["provenance"],
            repository=record["repository"],
            revision=record["revision"],
            license=record["license"],
            archive=CorpusArchive.from_json(record["archive"]),
        )


@dataclass(frozen=True)
class CorpusReleasePin:
    """
    The frozen release every installation must reproduce.
    """

    version: str
    """
    Release version identifier.
    """

    fragments_sha256: str
    """
    Expected SHA-256 checksum of the released ``fragments.jsonl``.
    """

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> CorpusReleasePin:
        """
        Parse the release pin from the source specification.
        """
        return cls(
            version=record["version"],
            fragments_sha256=record["fragments_sha256"],
        )


@dataclass(frozen=True)
class CorpusDownloadSpecification:
    """
    Pinned corpus download and freeze manifest.
    """

    schema_version: int
    """
    Manifest schema version.
    """

    source: CorpusSource
    """
    The raw corpus to install.
    """

    release: CorpusReleasePin
    """
    The frozen release to build and verify.
    """

    @classmethod
    def from_file(cls, path: Path) -> CorpusDownloadSpecification:
        """
        Load a download specification from JSON.
        """
        record = json.loads(path.read_text())
        return cls(
            schema_version=record["schema_version"],
            source=CorpusSource.from_json(record["source"]),
            release=CorpusReleasePin.from_json(record["release"]),
        )


@dataclass(frozen=True)
class CorpusInstaller:
    """
    Download, extract, and freeze one retrieval-corpus installation.
    """

    specification_path: Path
    """
    Path to the pinned download specification.
    """

    target_directory: Path
    """
    Directory that receives the raw corpus and the frozen release.
    """

    def install(self) -> Path:
        """
        Install the raw corpus and its frozen release; return the release directory.
        """
        specification = CorpusDownloadSpecification.from_file(self.specification_path)
        release_directory = (
            self.target_directory / "corpus_release" / specification.release.version
        )
        if self._release_is_installed(release_directory, specification.release):
            return release_directory

        raw_directory = self.target_directory / "unidomain"
        self._ensure_raw_corpus(raw_directory, specification.source)
        report = build_release(
            corpus_root=raw_directory,
            output_directory=release_directory,
            version=specification.release.version,
        )
        if report.release_checksum != specification.release.fragments_sha256:
            raise CorpusReleaseChecksumError(
                release_directory,
                specification.release.fragments_sha256,
                report.release_checksum,
            )
        return release_directory

    def _release_is_installed(
        self, release_directory: Path, release: CorpusReleasePin
    ) -> bool:
        fragments_path = release_directory / "fragments.jsonl"
        if not fragments_path.is_file():
            return False
        return sha256_of_file(fragments_path) == release.fragments_sha256

    def _ensure_raw_corpus(self, raw_directory: Path, source: CorpusSource) -> None:
        stamp_path = raw_directory / RAW_CORPUS_MANIFEST_NAME
        stamp = {
            "repository": source.repository,
            "revision": source.revision,
            "archive_sha256": source.archive.sha256,
            "license": source.license,
        }
        if stamp_path.is_file() and json.loads(stamp_path.read_text()) == stamp:
            return

        raw_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=self.target_directory, suffix=".tar.gz.download"
        ) as archive_file:
            with urllib.request.urlopen(
                source.archive.source_url, timeout=300
            ) as response:
                while chunk := response.read(1 << 20):
                    archive_file.write(chunk)
            archive_file.flush()
            archive_path = Path(archive_file.name)
            actual_checksum = sha256_of_file(archive_path)
            if actual_checksum != source.archive.sha256:
                raise DownloadedCorpusArchiveHashError(
                    source.archive.source_url,
                    source.archive.sha256,
                    actual_checksum,
                )
            with tarfile.open(archive_path, "r:gz") as archive:
                # The "data" filter rejects absolute members, parent-directory
                # escapes, and special files.
                archive.extractall(path=raw_directory, filter="data")
        stamp_path.write_text(json.dumps(stamp, indent=2, sort_keys=True) + "\n")


class DownloadedCorpusArchiveHashError(Exception):
    """
    A downloaded corpus archive did not match its pinned checksum.
    """

    def __init__(self, source_url: str, expected: str, actual: str):
        super().__init__(
            f"Downloaded corpus archive {source_url} hashes to {actual}, "
            f"expected {expected}."
        )


class CorpusReleaseChecksumError(Exception):
    """
    A freshly frozen release did not reproduce the pinned fragment checksum.
    """

    def __init__(self, release_directory: Path, expected: str, actual: str):
        super().__init__(
            f"Frozen release at {release_directory} hashes to {actual}, "
            f"expected {expected}; the raw corpus or the freeze code drifted "
            f"from the pinned release."
        )


def main() -> None:
    """
    Install the retrieval corpus configured for this reSym release.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=AUGMENT_DATASET_DIRECTORY,
        help="augment-dataset installation directory",
    )
    arguments = parser.parse_args()
    specification = Path(__file__).with_name("corpus_source.json")
    release = CorpusInstaller(specification, arguments.target).install()
    print(f"reSym retrieval corpus installed at {release}")
    print(
        "The optional dense index is built separately: "
        "uv run python -m resym.knowledge.dense"
    )


if __name__ == "__main__":
    main()
