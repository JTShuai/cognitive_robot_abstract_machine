"""
Fault templates through the real closed loop: representative defects of each
manifestation kind produce exactly the certificate class their ground truth expects, on
the Tracy scene with real grounding, planning, and monitored execution.

Container suite (shares the session-scoped Tracy fixtures).
"""

from __future__ import annotations

import pytest

from experiments.resym.scenes import Scene
from experiments.resym.seed_library import build_fixed_arm_library
from resym.repair.certificate import CURATION_TRIGGERS
from resym.repair.diagnosis import diagnose
from experiments.resym.icra.articulation.faults import (
    drawer_fault_templates,
    evaluate_diagnosis,
)
from resym.core.model import Literal
from resym.platform.articulation import articulation_connection
from resym.platform.universe import pddl_name
from resym.platform.kinematic import KinematicSkillRealization

GOAL_DRAWER = pddl_name(Scene.APARTMENT.goal_drawer_body)

REPRESENTATIVE = (
    "missing-close-operator",  # planning failure, curation trigger
    "inverted-precondition",  # causal model error, not an IK query failure
    "wrong-add-effect",  # binding-contract mapping exposes the wrong effect
    "wrong-parameter-type",  # deterministic signature validation
    "execution-binding-mismatch",  # caught by the postcondition check
    "unsupported-navigation-library",  # static refusal at the capability gate
    "broken-grounding-binding",  # repairable local grounding-factory mismatch
    "missing-delete-effect",  # latent: the probe task succeeds
)


def set_drawer_fraction(setup, universe, name: str, fraction: float) -> None:
    drawer = universe[name]
    connection = articulation_connection(drawer)
    limits = connection.dof.limits
    setup.world.state[connection.dof.id].position = limits.lower.position + fraction * (
        limits.upper.position - limits.lower.position
    )
    setup.world.notify_state_change()


def test_representative_templates_diagnose_as_expected(
    tracy_setup, tracy_universe, tracy_context, tmp_path, grounding_catalog, library
):
    correct = build_fixed_arm_library(grounding_catalog)
    by_id = {
        t.identifier: t for t in drawer_fault_templates(correct, grounding_catalog)
    }
    observations = []
    for identifier in REPRESENTATIVE:
        template = by_id[identifier]
        assert template.task.on_mounted_drawer  # this test targets the mounted drawer
        set_drawer_fraction(
            tracy_setup,
            tracy_universe,
            GOAL_DRAWER,
            0.9 if template.task.initially_open else 0.0,
        )
        tracy_context.witness_base_poses.clear()
        diagnosis = diagnose(
            template.apply(correct),
            tracy_universe,
            tracy_context,
            (Literal(template.task.goal_predicate, (GOAL_DRAWER,)),),
            tmp_path / identifier,
            realization=KinematicSkillRealization(),
        )
        expected = template.ground_truth.expected_certificate_classes
        if expected:
            assert not diagnosis.succeeded, identifier
            assert diagnosis.failure_class_value in expected, (
                identifier,
                diagnosis.failure_class_value,
            )
        else:
            assert diagnosis.succeeded, (identifier, diagnosis.failure_class_value)
        observations.append((template, diagnosis.failure_class_value))

    report = evaluate_diagnosis(observations)
    assert report.total == len(REPRESENTATIVE)
    assert report.accuracy == pytest.approx(1.0)

    # The curation gate: repairable model defects may start curation; the
    # misbound skill's postcondition failure must not.
    curation_observed = {
        template.identifier
        for (template, observed) in observations
        if observed is not None and any(c.value == observed for c in CURATION_TRIGGERS)
    }
    assert "missing-close-operator" in curation_observed
    assert "inverted-precondition" in curation_observed
    assert "wrong-add-effect" in curation_observed
    assert "wrong-parameter-type" in curation_observed
    assert "execution-binding-mismatch" not in curation_observed
    assert "unsupported-navigation-library" not in curation_observed
    assert "broken-grounding-binding" in curation_observed


def test_diagnosis_restores_a_clean_world(tracy_setup, tracy_universe):
    """
    Leave the shared scene closed for whoever runs next.
    """
    set_drawer_fraction(tracy_setup, tracy_universe, GOAL_DRAWER, 0.0)
    drawer = tracy_universe[GOAL_DRAWER]
    connection = articulation_connection(drawer)
    limits = connection.dof.limits
    assert tracy_setup.world.state[connection.dof.id].position == limits.lower.position
