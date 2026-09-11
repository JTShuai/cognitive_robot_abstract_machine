"""
Reproducibility metadata for formal E1/E2 runs.

The snapshot deliberately excludes ignored files, so credentials in ``.env`` never enter
an artifact.  A dirty run remains identifiable through the tracked source patch and a
content hash over tracked and non-ignored untracked files.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    """
    Return the SHA-256 digest of one artifact.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_snapshot(root: Path) -> tuple[dict[str, Any], str]:
    """
    Capture the exact non-ignored source tree and its patch from HEAD.
    """
    commit = _git(root, "rev-parse", "HEAD").strip()
    branch = _git(root, "branch", "--show-current").strip() or None
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    patch = _git(root, "diff", "--binary", "HEAD", "--")
    listed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted(
        Path(value.decode("utf-8", errors="surrogateescape"))
        for value in listed.split(b"\0")
        if value
    )
    digest = hashlib.sha256()
    untracked = set(
        _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    )
    untracked_records = []
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        content_digest = sha256_file(path)
        digest.update(relative.as_posix().encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(content_digest.encode())
        digest.update(b"\0")
        if relative.as_posix() in untracked:
            untracked_records.append(
                {"path": relative.as_posix(), "sha256": content_digest}
            )
    return (
        {
            "commit": commit,
            "branch": branch,
            "clean": not bool(status.strip()),
            "status": status.splitlines(),
            "source_tree_sha256": digest.hexdigest(),
            "source_patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "untracked_files": untracked_records,
        },
        patch,
    )


def experiment_provenance(
    root: Path,
    *,
    repository: Mapping[str, Any],
    arguments: Mapping[str, Any],
    templates: Iterable[str],
    case_counts: Mapping[str, int],
    split_manifest: Path,
    correct_library_checksum: str,
    policy_tiers: Iterable[str],
    llm_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Build the non-secret provenance record embedded in every run.
    """
    e1_enabled = not bool(arguments["skip_e1"])
    return {
        "schema_version": 1,
        "repository": dict(repository),
        "invocation": {
            "argv": list(sys.argv),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in arguments.items()
            },
        },
        "experiment": {
            "scene": "apartment",
            "templates": list(templates),
            "case_counts": dict(case_counts),
            "correct_library_checksum": correct_library_checksum,
            "e1_enabled": e1_enabled,
            "e2_enabled": not bool(arguments["skip_e2"]),
            "e2_policy_tiers": list(policy_tiers),
            "curator_suite_version": "tracy-suite-1/<template-id>",
            "formal_frozen_splits": not bool(arguments["unsafe_random_splits"]),
        },
        "artifacts": _artifact_records(
            root,
            split_manifest=split_manifest,
            llm_config=_llm_config_path(root) if e1_enabled else None,
        ),
        "locked_git_dependencies": _locked_git_dependencies(root / "uv.lock"),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "container_image": os.environ.get("RESYM_CONTAINER_IMAGE"),
            "container_image_id": os.environ.get("RESYM_CONTAINER_IMAGE_ID"),
        },
        "llm_configuration": dict(llm_metadata) if llm_metadata is not None else None,
        "knowledge_use": {
            "unidomain_corpus": e1_enabled,
            "ontology_alignment": False,
        },
    }


def _artifact_records(
    root: Path, *, split_manifest: Path, llm_config: Path | None
) -> list[dict[str, Any]]:
    candidates = (
        (root / "config" / "experiment_protocol.json", True),
        (split_manifest, True),
        (root / "pyproject.toml", True),
        (root / "uv.lock", True),
        (root / "Dockerfile", True),
        (root / "augment_dataset" / "ontology" / "manifest.json", False),
        (root / "augment_dataset" / "corpus_release" / "r1" / "manifest.json", False),
        (llm_config, llm_config is not None),
    )
    records = []
    seen: set[Path] = set()
    for path, used in candidates:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        records.append(
            {
                "path": _display_path(path, root),
                "present": path.is_file(),
                "used_by_run": used,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return records


def _locked_git_dependencies(lock_path: Path) -> list[dict[str, str]]:
    if not lock_path.is_file():
        return []
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    records = []
    for package in lock.get("package", []):
        git_source = package.get("source", {}).get("git")
        if git_source:
            records.append({"name": package["name"], "source": git_source})
    return sorted(records, key=lambda record: record["name"])


def _llm_config_path(root: Path) -> Path:
    configured = os.environ.get("RESYM_LLM_CONFIG", "config/llm.json")
    path = Path(configured)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def write_json(path: Path, value: Any) -> None:
    """
    Write stable, human-readable JSON for a run artifact.
    """
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
