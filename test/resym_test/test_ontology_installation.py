"""
Ontology installation behaviour.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from resym.knowledge.ontology_installation import (
    DownloadedOntologyHashError,
    OntologyInstaller,
)


def _write_specification(path, *, source_url: str, checksum: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "id": "test-source",
                        "name": "Test source",
                        "provenance": "test",
                        "release": "1.0",
                        "license": "test-license",
                        "files": [
                            {
                                "path": "test/source.owl",
                                "source_url": source_url,
                                "sha256": checksum,
                            }
                        ],
                    }
                ],
            }
        )
    )


def test_installer_downloads_files_and_writes_runtime_manifest(tmp_path):
    content = b"ontology test payload"
    source = tmp_path / "source.owl"
    source.write_bytes(content)
    specification = tmp_path / "ontology_sources.json"
    _write_specification(
        specification,
        source_url=source.as_uri(),
        checksum=hashlib.sha256(content).hexdigest(),
    )
    target = tmp_path / "installed"

    manifest = OntologyInstaller(specification, target).install()

    assert (target / "test" / "source.owl").read_bytes() == content
    assert manifest == target / "manifest.json"
    assert json.loads(manifest.read_text()) == json.loads(specification.read_text())


def test_installer_rejects_a_download_with_the_wrong_hash(tmp_path):
    source = tmp_path / "source.owl"
    source.write_bytes(b"corrupt")
    specification = tmp_path / "ontology_sources.json"
    _write_specification(
        specification,
        source_url=source.as_uri(),
        checksum=hashlib.sha256(b"expected").hexdigest(),
    )
    target = tmp_path / "installed"

    with pytest.raises(DownloadedOntologyHashError):
        OntologyInstaller(specification, target).install()

    assert not (target / "test" / "source.owl").exists()
    assert not (target / "manifest.json").exists()


def test_installer_reuses_a_verified_file_without_downloading(tmp_path):
    content = b"ontology test payload"
    source = tmp_path / "source.owl"
    source.write_bytes(content)
    specification = tmp_path / "ontology_sources.json"
    _write_specification(
        specification,
        source_url=source.as_uri(),
        checksum=hashlib.sha256(content).hexdigest(),
    )
    target = tmp_path / "installed"
    installer = OntologyInstaller(specification, target)
    installer.install()
    source.unlink()

    installer.install()

    assert (target / "test" / "source.owl").read_bytes() == content
