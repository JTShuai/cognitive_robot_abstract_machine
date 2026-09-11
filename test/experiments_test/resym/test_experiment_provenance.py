"""
Formal experiment provenance is complete, stable, and secret-free.
"""

from __future__ import annotations

import json
import subprocess

from experiments.resym.icra.provenance import (
    experiment_provenance,
    repository_snapshot,
)


def _git(root, *arguments):
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


def test_repository_snapshot_identifies_dirty_content_without_ignored_files(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / "tracked.txt").write_text("one\n")
    _git(tmp_path, "add", ".gitignore", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "seed")
    (tmp_path / "tracked.txt").write_text("two\n")
    (tmp_path / "extra.txt").write_text("extra\n")
    (tmp_path / ".env").write_text("API_KEY=secret\n")

    snapshot, patch = repository_snapshot(tmp_path)

    assert not snapshot["clean"]
    assert "tracked.txt" in patch
    assert "extra.txt" in {item["path"] for item in snapshot["untracked_files"]}
    assert ".env" not in json.dumps(snapshot)
    assert "secret" not in patch


def test_e2_provenance_marks_external_knowledge_as_unused(tmp_path, monkeypatch):
    for relative in (
        "config/experiment_protocol.json",
        "config/splits.json",
        "pyproject.toml",
        "uv.lock",
        "Dockerfile",
        "ontology/manifest.json",
        "corpus_release/r1/manifest.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("" if relative == "uv.lock" else "{}")
    monkeypatch.setenv("RESYM_CONTAINER_IMAGE", "cram:test")
    monkeypatch.setenv("RESYM_CONTAINER_IMAGE_ID", "sha256:abc")

    provenance = experiment_provenance(
        tmp_path,
        repository={"commit": "abc", "clean": True},
        arguments={
            "skip_e1": True,
            "skip_e2": False,
            "unsafe_random_splits": False,
        },
        templates=("missing-open-operator",),
        case_counts={"proposal": 10, "admission": 8, "held_out": 7},
        split_manifest=tmp_path / "config" / "splits.json",
        correct_library_checksum="library-sha",
        policy_tiers=("single-witness", "full-suite"),
        llm_metadata=None,
    )

    assert provenance["knowledge_use"] == {
        "unidomain_corpus": False,
        "ontology_alignment": False,
    }
    assert provenance["runtime"]["container_image_id"] == "sha256:abc"
    assert provenance["experiment"]["formal_frozen_splits"]
