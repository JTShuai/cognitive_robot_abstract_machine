"""
Grounding-plan helpers for tests that bind predicates to reviewed factories.
"""

from __future__ import annotations

from resym.core.grounding import PredicateGroundingPlan
from resym.platform.grounding_catalog import GroundingFactoryCatalog

from experiments.resym.grounding_initialization import JOINT_FRACTION_OPENED_UID
from experiments.resym.seed_library import build_grounding_plan

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
    return build_grounding_plan(
        catalog,
        JOINT_FRACTION_OPENED_UID,
        (("articulated_object", 0),),
        (("threshold", threshold),),
        negated,
    )
