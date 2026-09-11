"""
Corpus loading, parsing, and release freezing.

Stdlib-only; runs on the host. A synthetic miniature corpus exercises the mechanics; a
smoke test runs against the real UniDomain data only where that checkout exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resym.knowledge.corpus import (
    fragment_id_for,
    load_corpus,
    load_fragment,
)
from resym.knowledge.freeze import (
    CorruptReleaseError,
    LeakageRule,
    build_release,
    load_release,
)

REAL_CORPUS = Path(
    "/home/jiangtao/shuai_ws/develop/Bremen/UniDomain/data/unified_domain"
)

PICK_DOMAIN = {
    "predicates": {
        "(on_counter ?o ?c)": "object ?o is on counter ?c",
        "(holding ?o)": "the gripper holds ?o; grasp is stable; ready to move",
        "(hand_free)": "the hand is empty",
    },
    "operators": {
        "pick_from_counter": (
            "(:action pick_from_counter\n"
            " :parameters (?o ?c)\n"
            " :precondition (and (on_counter ?o ?c) (hand_free))\n"
            " :effect (and (holding ?o) (not (on_counter ?o ?c)) (not (hand_free))))"
        )
    },
}

DRAWER_DOMAIN = {
    "predicates": {"(drawer_open ?d)": "drawer ?d is open"},
    "operators": {
        "open_drawer": (
            "(:action open_drawer :parameters (?d) "
            ":precondition (not (drawer_open ?d)) "
            ":effect (drawer_open ?d))"
        )
    },
}


def write_domain(root: Path, group: str, episode: str, record: dict) -> Path:
    directory = root / group / f"episode_{episode}"
    directory.mkdir(parents=True)
    path = directory / "atomic_domain.json"
    path.write_text(json.dumps(record))
    return path


@pytest.fixture()
def miniature_corpus(tmp_path):
    root = tmp_path / "corpus"
    write_domain(root, "1", "100", PICK_DOMAIN)
    write_domain(root, "2", "200", DRAWER_DOMAIN)
    write_domain(root, "3", "300", PICK_DOMAIN)  # exact duplicate of 1/100
    broken = root / "4" / "episode_400"
    broken.mkdir(parents=True)
    (broken / "atomic_domain.json").write_text("not json")
    return root


def test_fragment_id_matches_source_tag_layout(miniature_corpus):
    path = miniature_corpus / "1" / "episode_100" / "atomic_domain.json"
    assert fragment_id_for(path) == "1/100"


def test_operator_structure_is_parsed(miniature_corpus):
    fragment = load_fragment(
        miniature_corpus / "1" / "episode_100" / "atomic_domain.json"
    )
    (operator,) = fragment.operators
    assert operator.parameters == ("?o", "?c")
    assert operator.positive_preconditions == ("on_counter", "hand_free")
    assert operator.add_predicates == ("holding",)
    assert set(operator.delete_predicates) == {"on_counter", "hand_free"}


def test_negated_precondition_is_kept_apart(miniature_corpus):
    fragment = load_fragment(
        miniature_corpus / "2" / "episode_200" / "atomic_domain.json"
    )
    (operator,) = fragment.operators
    assert operator.negative_preconditions == ("drawer_open",)
    assert operator.positive_preconditions == ()


def test_gloss_with_semicolons_survives(miniature_corpus):
    """
    The upstream .pddl parser breaks on glosses containing ';'; ours reads the JSON
    values and must not.
    """
    fragment = load_fragment(
        miniature_corpus / "1" / "episode_100" / "atomic_domain.json"
    )
    holding = next(p for p in fragment.predicates if p.name == "holding")
    assert "grasp is stable; ready to move" in holding.gloss


def test_load_corpus_collects_failures_instead_of_crashing(miniature_corpus):
    fragments, failures = load_corpus(miniature_corpus)
    assert len(fragments) == 3
    assert len(failures) == 1
    assert failures[0].fragment_id == "4/400"


def test_release_dedups_and_checksums(miniature_corpus, tmp_path):
    output = tmp_path / "release"
    report = build_release(miniature_corpus, output, version="test-r1")
    assert report.released == 2
    assert report.exact_duplicates == 1
    assert report.parse_failures == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"]["released"] == 2
    reloaded = load_release(output)
    assert {f.fragment_id for f in reloaded} == {"1/100", "2/200"}


def test_release_rebuild_is_deterministic(miniature_corpus, tmp_path):
    first = build_release(miniature_corpus, tmp_path / "a", version="r")
    second = build_release(miniature_corpus, tmp_path / "b", version="r")
    assert first.release_checksum == second.release_checksum


def test_tampered_release_is_refused(miniature_corpus, tmp_path):
    output = tmp_path / "release"
    build_release(miniature_corpus, output, version="r")
    with (output / "fragments.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(CorruptReleaseError):
        load_release(output)


def test_leakage_rule_excludes_by_operator_name(miniature_corpus, tmp_path):
    rules = [LeakageRule(family="D1", patterns=("drawer",))]
    report = build_release(
        miniature_corpus, tmp_path / "release", version="r", leakage_rules=rules
    )
    assert report.leakage_excluded == 1
    assert report.excluded_by_family == {"D1": 1}
    excluded = [
        json.loads(line)
        for line in (tmp_path / "release" / "excluded.jsonl").read_text().splitlines()
    ]
    leakage_rows = [row for row in excluded if row["reason"] == "leakage"]
    assert leakage_rows[0]["fragment_id"] == "2/200"
    assert leakage_rows[0]["matched_operator"] == "open_drawer"


@pytest.mark.skipif(not REAL_CORPUS.exists(), reason="UniDomain data not present")
def test_real_corpus_smoke():
    from resym.knowledge.corpus import (
        iter_atomic_domain_files,
    )

    paths = list(iter_atomic_domain_files(REAL_CORPUS))
    assert len(paths) > 10_000
    sample = load_fragment(paths[0])
    assert sample.predicates
    assert sample.operators
    assert sample.index_text()
