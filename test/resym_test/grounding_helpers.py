"""
Grounding-plan helpers for tests that bind predicates to reviewed factories.
"""

from __future__ import annotations

from resym.core.grounding_model import PredicateGroundingPlan
from resym.platform.grounding_catalog import GroundingFactoryCatalog

from .dataset.task_model import articulation_state_plan

STUB_GROUNDING_PLAN = PredicateGroundingPlan(
    factory_uid="test:grounding/stub",
    approved_factory_checksum="stub-checksum",
)
"""
A syntactically complete plan for tests that never evaluate it.
"""


def joint_fraction_plan(
    catalog: GroundingFactoryCatalog,
    threshold: float = 0.4,
    negated: bool = False,
) -> PredicateGroundingPlan:
    """
    A reviewed articulation-state binding against the given catalog.
    """
    return articulation_state_plan(catalog, threshold, negated)
