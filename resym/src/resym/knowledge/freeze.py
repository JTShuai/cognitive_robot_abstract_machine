"""
Freezing the declarative corpus (UniDomain Dataset) into a versioned, checksummed
release.

Usage::

    uv run python -m resym.knowledge.freeze \
        --corpus-root /path/to/UniDomain/data/unified_domain \
        --output augment_dataset/corpus_release/r1 --version r1 \
        [--leakage-config config/leakage_targets.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from typing_extensions import Optional

from krrood.adapters.json_serializer import from_json, to_json
from resym.knowledge.corpus import (
    DomainFragment,
    load_corpus,
)

UPSTREAM_LICENSE = "MIT (UniDomain repository, (c) 2025 RoboticSJTU)"
UPSTREAM_DATASET = "huggingface:SII-PrimoButterfly/UniDomain-Data (unified subset)"


@dataclass
class LeakageRule:
    """
    One evaluation-task family and the name patterns that identify its near-duplicate
    fragments.
    """

    family: str
    """
    E.g.

    ``D1-articulated``.
    """

    patterns: tuple[str, ...]
    """
    Case-insensitive word-level regexes matched against operator names.
    """

    def matches(self, fragment: DomainFragment) -> Optional[str]:
        """
        The first operator name this rule hits, or ``None``.
        """
        for operator in fragment.operators:
            haystack = operator.name.replace("_", " ")
            for pattern in self.patterns:
                if re.search(rf"\b{pattern}\b", haystack, flags=re.IGNORECASE):
                    return operator.name
        return None


@dataclass
class ReleaseReport:
    """
    What the freeze produced, for the manifest and for tests.
    """

    released: int = 0
    parse_failures: int = 0
    exact_duplicates: int = 0
    leakage_excluded: int = 0
    files_seen: int = 0
    release_checksum: str = ""
    output_directory: Optional[Path] = None
    excluded_by_family: dict[str, int] = field(default_factory=dict)


def load_leakage_rules(path: Optional[Path]) -> list[LeakageRule]:
    if path is None:
        return []
    data = json.loads(path.read_text())
    return [
        LeakageRule(family=entry["family"], patterns=tuple(entry["patterns"]))
        for entry in data["targets"]
    ]


def canonical_hash(fragment: DomainFragment) -> str:
    """
    Content identity for exact deduplication: signatures, glosses, and whitespace-
    normalized operator bodies, independent of file location.
    """
    payload = json.dumps(
        {
            "predicates": sorted((p.signature, p.gloss) for p in fragment.predicates),
            "operators": sorted(
                (o.name, " ".join(o.raw.split())) for o in fragment.operators
            ),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def build_release(
    corpus_root: Path,
    output_directory: Path,
    version: str,
    leakage_rules: Optional[list[LeakageRule]] = None,
) -> ReleaseReport:
    """
    Load, dedup, filter, and write one immutable corpus release.
    """
    leakage_rules = leakage_rules or []
    output_directory.mkdir(parents=True, exist_ok=True)
    report = ReleaseReport(output_directory=output_directory)

    fragments, failures = load_corpus(corpus_root)
    report.files_seen = len(fragments) + len(failures)
    report.parse_failures = len(failures)

    released: list[DomainFragment] = []
    excluded: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for fragment in fragments:
        content_hash = canonical_hash(fragment)
        if content_hash in seen_hashes:
            report.exact_duplicates += 1
            excluded.append(
                {
                    "fragment_id": fragment.fragment_id,
                    "reason": "exact_duplicate",
                    "duplicate_of": seen_hashes[content_hash],
                }
            )
            continue
        leak = _first_leakage_match(fragment, leakage_rules)
        if leak is not None:
            family, operator_name = leak
            report.leakage_excluded += 1
            report.excluded_by_family[family] = (
                report.excluded_by_family.get(family, 0) + 1
            )
            excluded.append(
                {
                    "fragment_id": fragment.fragment_id,
                    "reason": "leakage",
                    "family": family,
                    "matched_operator": operator_name,
                }
            )
            continue
        seen_hashes[content_hash] = fragment.fragment_id
        released.append(fragment)
    report.released = len(released)

    fragments_path = output_directory / "fragments.jsonl"
    with fragments_path.open("w") as handle:
        for fragment in released:
            handle.write(json.dumps(to_json(fragment), sort_keys=True) + "\n")
    report.release_checksum = sha256_of_file(fragments_path)

    _write_jsonl(output_directory / "excluded.jsonl", excluded)
    _write_jsonl(
        output_directory / "parse_failures.jsonl",
        [
            {"fragment_id": failure.fragment_id, "detail": failure.detail}
            for failure in failures
        ],
    )
    _write_jsonl(
        output_directory / "source_hashes.jsonl",
        [
            {
                "fragment_id": fragment.fragment_id,
                "source_path": fragment.source_path,
                "sha256": sha256_of_file(corpus_root / fragment.source_path),
            }
            for fragment in released
        ],
    )

    manifest = {
        "release_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(corpus_root),
        "upstream_license": UPSTREAM_LICENSE,
        "upstream_dataset": UPSTREAM_DATASET,
        "counts": {
            "files_seen": report.files_seen,
            "parse_failures": report.parse_failures,
            "exact_duplicates": report.exact_duplicates,
            "leakage_excluded": report.leakage_excluded,
            "released": report.released,
        },
        "leakage_rules": [
            {"family": rule.family, "patterns": list(rule.patterns)}
            for rule in leakage_rules
        ],
        "excluded_by_family": report.excluded_by_family,
        "release_checksum_sha256": report.release_checksum,
        "pending_audits": [
            "embedding-level near-duplicate pass (runs with the retrieval "
            "index build; do not use this release for held-out evaluation "
            "before it has run)"
        ],
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return report


def load_release(release_directory: Path) -> list[DomainFragment]:
    """
    Read the fragments of a frozen release, verifying its checksum.
    """
    manifest = json.loads((release_directory / "manifest.json").read_text())
    fragments_path = release_directory / "fragments.jsonl"
    actual = sha256_of_file(fragments_path)
    expected = manifest["release_checksum_sha256"]
    if actual != expected:
        raise CorruptReleaseError(release_directory, expected, actual)
    return [
        from_json(json.loads(line))
        for line in fragments_path.read_text().splitlines()
        if line.strip()
    ]


class CorruptReleaseError(Exception):
    """
    Raised when a frozen release no longer matches its manifest checksum.
    """

    def __init__(self, directory: Path, expected: str, actual: str):
        super().__init__(
            f"Release at {directory} is corrupt: fragments.jsonl hashes to "
            f"{actual}, manifest says {expected}."
        )


def _first_leakage_match(
    fragment: DomainFragment, rules: list[LeakageRule]
) -> Optional[tuple[str, str]]:
    for rule in rules:
        operator_name = rule.matches(fragment)
        if operator_name is not None:
            return rule.family, operator_name
    return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--leakage-config", type=Path, default=None)
    arguments = parser.parse_args()
    report = build_release(
        corpus_root=arguments.corpus_root,
        output_directory=arguments.output,
        version=arguments.version,
        leakage_rules=load_leakage_rules(arguments.leakage_config),
    )
    print(
        f"release {arguments.version}: {report.released} fragments "
        f"({report.exact_duplicates} duplicates, "
        f"{report.leakage_excluded} leakage-excluded, "
        f"{report.parse_failures} parse failures) -> {report.output_directory}"
    )


if __name__ == "__main__":
    main()
