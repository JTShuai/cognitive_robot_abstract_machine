"""
The live viewer explains object selection independently of execution.
"""

from resym.observability.viewer import _render_live_execution, _reduce_pipeline_events
from resym.planning.events import PipelineEvent
from resym.planning.pipeline import TaskResult
from resym.observability.runlog import summarize_task_result


def test_live_view_explains_scope_and_expansion():
    records = [
        {"event": PipelineEvent.TASK_STARTED},
        {"event": PipelineEvent.ROUND_STARTED, "round": 1},
        {
            "event": PipelineEvent.OBJECT_SCOPE_EXPANDED,
            "reason": "unsolvable_subset",
            "added_objects": ["support"],
        },
        {
            "event": PipelineEvent.TASK_OBJECTS_SELECTED,
            "total_objects": 100,
            "selected_objects": ["target", "support"],
            "attempt": 2,
            "inclusion_reasons": {
                "target": {"reason": "goal"},
                "support": {"reason": "dependency", "related_to": "target"},
            },
        },
    ]

    rendered = _render_live_execution(records, "test")

    assert "Planning objects: 2 / 100" in rendered
    assert "unsolvable_subset" in rendered
    assert "dependency" in rendered
    assert "support" in rendered


def test_exhausted_scope_is_not_shown_as_still_planning():
    state = _reduce_pipeline_events(
        [
            {"event": PipelineEvent.PLANNING_STARTED},
            {"event": PipelineEvent.OBJECT_SCOPE_EXHAUSTED},
        ]
    )
    assert state["status"] == "no plan in full candidate scope"


def test_run_summary_preserves_scope_metrics():
    result = TaskResult(scope_expansions=2, scope_seconds=0.125)
    summary = summarize_task_result(result)
    assert summary["scope_expansions"] == result.scope_expansions
    assert summary["scope_seconds"] == result.scope_seconds


def test_live_view_shows_advised_objects_and_the_consultation():
    records = [
        {"event": PipelineEvent.TASK_STARTED},
        {
            "event": PipelineEvent.OBJECT_SCOPE_ADVISED,
            "attempt": 1,
            "recommendations": [{"name": "door", "rationale": "gives access"}],
            "trace": [
                {
                    "event": "tool_result",
                    "tool": "query_candidate_objects",
                    "observation": "- door (Door) eligible",
                }
            ],
        },
        {
            "event": PipelineEvent.TASK_OBJECTS_SELECTED,
            "total_objects": 3,
            "selected_objects": ["door"],
            "attempt": 1,
            "inclusion_reasons": {
                "door": {"reason": "advised", "rationale": "gives access"}
            },
        },
    ]

    rendered = _render_live_execution(records, "test")

    assert "gives access" in rendered
    assert "Advisor consultation 1: 1 recommendation(s)" in rendered
    assert "query_candidate_objects - door (Door) eligible" in rendered
