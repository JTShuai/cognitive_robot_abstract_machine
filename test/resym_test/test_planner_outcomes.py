"""
Only a proved unsolvable problem is eligible for object-scope expansion.
"""

from types import SimpleNamespace

import pytest

from resym.planning import pddl
from resym.planning.pddl import (
    PlannerExitCode,
    PlannerTimeoutError,
    PlannerExecutionError,
    UnsolvableProblemError,
)

# %% native outcomes


@pytest.mark.parametrize(
    "returncode,exception",
    [
        (PlannerExitCode.TRANSLATE_UNSOLVABLE, UnsolvableProblemError),
        (PlannerExitCode.SEARCH_UNSOLVABLE, UnsolvableProblemError),
        (PlannerExitCode.SEARCH_OUT_OF_TIME, PlannerTimeoutError),
        (PlannerExitCode.TRANSLATE_OUT_OF_TIME, PlannerTimeoutError),
        (PlannerExitCode.SEARCH_UNSOLVED_INCOMPLETE, PlannerExecutionError),
        (PlannerExitCode.SEARCH_INPUT_ERROR, PlannerExecutionError),
        (PlannerExitCode.SUCCESS, PlannerExecutionError),
    ],
)
def test_native_failure_classification(tmp_path, monkeypatch, returncode, exception):
    driver = tmp_path / "fast-downward.py"
    driver.touch()
    monkeypatch.setattr(pddl, "planner_directory", lambda: tmp_path)
    monkeypatch.setattr(
        pddl.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode,
            stdout="",
            stderr="",
        ),
    )

    with pytest.raises(exception):
        pddl.plan("", "", tmp_path)
