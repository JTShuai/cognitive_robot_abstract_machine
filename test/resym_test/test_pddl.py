"""
PDDL projection: emitted text and plan parsing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from resym.core.model import Literal
from resym.planning import pddl
from resym.planning.pddl import (
    FAST_DOWNWARD_DIRECTORY_ENVIRONMENT_VARIABLE,
    GroundAction,
    PlanNotFoundError,
    _parse_plan,
    planner_directory,
    plan,
    write_domain,
    _literals,
)
from resym.planning.selection import select_for_goal


def test_domain_contains_selected_schemas(library):
    selection = select_for_goal(library, (Literal("opened", ("d",)),))
    domain = write_domain(selection, "test-domain")
    assert "(:action navigate" in domain
    assert "(:action open-drawer" in domain
    assert "(:requirements :strips :negative-preconditions)" in domain
    assert "(ready-to-open ?a ?b)" in domain
    assert "(cram-type-abstract-robot " in domain


def test_negative_literal_is_emitted_directly() -> None:
    assert _literals((Literal("blocked", ("r",), negated=True),)) == (
        "(not (blocked ?r))"
    )


def test_plan_parsing():
    text = (
        "(navigate base cabinet10)\n(open-drawer base handle cabinet10)\n; cost = 2\n"
    )
    actions = _parse_plan(text)
    assert actions == [
        GroundAction("navigate", ("base", "cabinet10")),
        GroundAction("open-drawer", ("base", "handle", "cabinet10")),
    ]


def test_planner_directory_can_be_provided_by_the_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv(FAST_DOWNWARD_DIRECTORY_ENVIRONMENT_VARIABLE, str(tmp_path))

    assert planner_directory() == tmp_path


def test_planner_does_not_reuse_a_stale_plan(tmp_path, monkeypatch):
    driver = tmp_path / "planner" / "fast-downward.py"
    driver.parent.mkdir()
    driver.write_text("")
    work = tmp_path / "work"
    work.mkdir()
    (work / "sas_plan").write_text("(stale-action)\n")
    monkeypatch.setenv(FAST_DOWNWARD_DIRECTORY_ENVIRONMENT_VARIABLE, str(driver.parent))
    monkeypatch.setattr(
        pddl.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="no plan", stderr=""
        ),
    )

    with pytest.raises(PlanNotFoundError):
        plan("(domain)", "(problem)", work)

    assert not (work / "sas_plan").exists()


def test_planner_resolves_relative_working_directory(tmp_path, monkeypatch):
    driver = tmp_path / "planner" / "fast-downward.py"
    driver.parent.mkdir()
    driver.write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(FAST_DOWNWARD_DIRECTORY_ENVIRONMENT_VARIABLE, str(driver.parent))
    observed = {}

    def run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["cwd"] = kwargs["cwd"]
        Path(arguments[3]).write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pddl.subprocess, "run", run)

    assert plan("(domain)", "(problem)", Path("work")) == []
    assert observed["cwd"] == tmp_path / "work"
    assert all(Path(path).is_absolute() for path in observed["arguments"][3:6])
