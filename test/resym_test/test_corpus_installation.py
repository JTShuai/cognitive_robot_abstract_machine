"""
Retrieval-corpus installation behaviour: download, verify, extract, freeze.
"""

from __future__ import annotations

import json
import tarfile

import pytest

from resym.knowledge.corpus_installation import (
    CorpusInstaller,
    CorpusReleaseChecksumError,
    DownloadedCorpusArchiveHashError,
    RAW_CORPUS_MANIFEST_NAME,
)
from resym.knowledge.freeze import build_release, sha256_of_file


def _write_atomic_domain(
    directory, *, group: str, episode: str, predicate: str
) -> None:
    record_directory = directory / group / f"episode_{episode}"
    record_directory.mkdir(parents=True)
    (record_directory / "atomic_domain.json").write_text(
        json.dumps(
            {
                "predicates": {f"({predicate} ?d)": f"drawer ?d is {predicate}"},
                "operators": {
                    f"make_{predicate}": (
                        f"(:action make_{predicate}\n"
                        "  :parameters (?d)\n"
                        f"  :precondition (and (not ({predicate} ?d)))\n"
                        f"  :effect (and ({predicate} ?d)))"
                    )
                },
            }
        )
    )


def _archive_of(corpus_directory, archive_path) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for entry in sorted(corpus_directory.iterdir()):
            archive.add(entry, arcname=entry.name)


def _write_specification(
    path, *, source_url: str, archive_sha256: str, fragments_sha256: str
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "id": "test-corpus",
                    "name": "Test corpus",
                    "provenance": "test",
                    "repository": "test/corpus",
                    "revision": "deadbeef",
                    "license": "test-license",
                    "archive": {
                        "source_url": source_url,
                        "sha256": archive_sha256,
                    },
                },
                "release": {
                    "version": "r1",
                    "fragments_sha256": fragments_sha256,
                },
            }
        )
    )


@pytest.fixture
def pinned_corpus(tmp_path):
    """
    A miniature raw corpus, its archive, and the fragment checksum its freeze
    reproduces.
    """
    raw = tmp_path / "raw"
    _write_atomic_domain(raw, group="1", episode="1", predicate="opened")
    _write_atomic_domain(raw, group="2", episode="7", predicate="closed")
    archive_path = tmp_path / "corpus.tar.gz"
    _archive_of(raw, archive_path)
    reference = build_release(
        corpus_root=raw, output_directory=tmp_path / "reference", version="r1"
    )
    specification = tmp_path / "corpus_source.json"
    _write_specification(
        specification,
        source_url=archive_path.as_uri(),
        archive_sha256=sha256_of_file(archive_path),
        fragments_sha256=reference.release_checksum,
    )
    return specification, archive_path, reference.release_checksum


def test_installer_downloads_extracts_and_freezes_the_pinned_release(
    tmp_path, pinned_corpus
):
    specification, _, fragments_sha256 = pinned_corpus
    target = tmp_path / "installed"

    release = CorpusInstaller(specification, target).install()

    assert release == target / "corpus_release" / "r1"
    assert sha256_of_file(release / "fragments.jsonl") == fragments_sha256
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["release_checksum_sha256"] == fragments_sha256
    assert manifest["counts"]["released"] == 2
    raw_stamp = json.loads(
        (target / "unidomain" / RAW_CORPUS_MANIFEST_NAME).read_text()
    )
    assert raw_stamp["revision"] == "deadbeef"


def test_installer_rejects_an_archive_with_the_wrong_hash(tmp_path, pinned_corpus):
    specification, archive_path, fragments_sha256 = pinned_corpus
    _write_specification(
        specification,
        source_url=archive_path.as_uri(),
        archive_sha256=sha256_of_file(specification),
        fragments_sha256=fragments_sha256,
    )
    target = tmp_path / "installed"

    with pytest.raises(DownloadedCorpusArchiveHashError):
        CorpusInstaller(specification, target).install()

    assert not (target / "unidomain" / RAW_CORPUS_MANIFEST_NAME).exists()
    assert not (target / "corpus_release").exists()


def test_installer_rejects_a_release_that_misses_the_pin(tmp_path, pinned_corpus):
    specification, archive_path, _ = pinned_corpus
    _write_specification(
        specification,
        source_url=archive_path.as_uri(),
        archive_sha256=sha256_of_file(archive_path),
        fragments_sha256="0" * 64,
    )
    target = tmp_path / "installed"

    with pytest.raises(CorpusReleaseChecksumError):
        CorpusInstaller(specification, target).install()


def test_installer_skips_when_the_pinned_release_is_installed(tmp_path, pinned_corpus):
    specification, archive_path, _ = pinned_corpus
    target = tmp_path / "installed"
    installer = CorpusInstaller(specification, target)
    installer.install()
    archive_path.unlink()

    release = installer.install()

    assert (
        sha256_of_file(release / "fragments.jsonl")
        == json.loads(specification.read_text())["release"]["fragments_sha256"]
    )


def test_installer_rebuilds_from_the_stamped_raw_corpus_without_downloading(
    tmp_path, pinned_corpus
):
    specification, archive_path, fragments_sha256 = pinned_corpus
    target = tmp_path / "installed"
    installer = CorpusInstaller(specification, target)
    release = installer.install()
    archive_path.unlink()
    (release / "fragments.jsonl").unlink()

    rebuilt = installer.install()

    assert sha256_of_file(rebuilt / "fragments.jsonl") == fragments_sha256
