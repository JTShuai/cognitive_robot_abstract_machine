"""
Fault templates: pure library transformations with known ground truth, split generation,
and the diagnosis scoring.

Host-runnable: templates operate on the plain fixed-arm library; nothing here touches
the CRAM stack.
"""

from __future__ import annotations

import pytest

from experiments.resym.seed_library import build_fixed_arm_library
from experiments.resym.icra.articulation.faults import (
    ADMISSION,
    HELD_OUT,
    PROPOSAL,
    FaultCase,
    Manifestation,
    SceneVariation,
    FrozenSplitManifest,
    SplitCalibrationError,
    calibrate_splits,
    drawer_fault_templates,
    evaluate_diagnosis,
    generate_splits,
)
from resym.core.model import Literal
from resym.planning.selection import select_for_goal


@pytest.fixture(scope="module")
def correct(grounding_catalog):
    return build_fixed_arm_library(grounding_catalog)


@pytest.fixture(scope="module")
def templates(correct, grounding_catalog):
    return drawer_fault_templates(correct, grounding_catalog)


def test_template_set_size_and_groups(templates):
    assert len(templates) >= 12
    assert {template.group for template in templates} == {"D1", "D2", "D3"}
    identifiers = [template.identifier for template in templates]
    assert len(set(identifiers)) == len(identifiers)


def test_transforms_are_pure_and_actually_change_the_library(correct, templates):
    reference = correct.to_json()
    for template in templates:
        faulted = template.apply(correct)
        assert correct.to_json() == reference, template.identifier
        assert faulted.to_json() != reference, template.identifier


def test_every_repair_restores_the_probe_capability(correct, templates, library):
    """
    Applying the ground-truth patch to the faulted library must make the probe goal
    selectable again with an achiever — the reference repair is a real repair, not a
    label.
    """
    for template in templates:
        repair = template.ground_truth.repair
        if repair is None:
            continue
        faulted = template.apply(correct)
        restored = repair.apply_to(faulted)
        goal = (Literal(template.task.goal_predicate, ("d1",)),)
        selection = select_for_goal(restored, goal)
        achievers = [
            operator
            for operator in selection.operators.values()
            if any(
                effect.predicate == template.task.goal_predicate
                for effect in operator.add_effects
            )
        ]
        assert achievers, template.identifier


def test_unsupported_templates_have_no_repair(templates):
    unsupported = [
        template for template in templates if template.ground_truth.unsupported
    ]
    assert unsupported
    for template in unsupported:
        assert template.ground_truth.repair is None
        assert template.ground_truth.expected_certificate_classes == (
            "unsupported_capability",
        )


def test_broken_grounding_binding_is_a_repairable_model_error(templates):
    template = next(
        item for item in templates if item.identifier == "broken-grounding-binding"
    )

    assert template.ground_truth.repair is not None
    assert template.ground_truth.expected_certificate_classes == (
        "predicate_implementation_error",
    )
    assert template.ground_truth.curation_expected


def test_curation_flag_only_on_repairable_model_classes(templates):
    for template in templates:
        classes = template.ground_truth.expected_certificate_classes
        if template.ground_truth.curation_expected:
            assert classes[0] in (
                "missing_predicate_model",
                "missing_operator_model",
                "operator_precondition_error",
                "operator_effect_error",
                "operator_signature_error",
                "predicate_implementation_error",
            )
        elif classes:
            assert classes[0] not in (
                "missing_predicate_model",
                "missing_operator_model",
                "operator_precondition_error",
                "operator_effect_error",
                "operator_signature_error",
                "predicate_implementation_error",
            )


def test_silent_templates_expect_no_certificate(templates):
    silent = [
        template
        for template in templates
        if template.manifestation is Manifestation.SILENT
    ]
    assert silent
    for template in silent:
        assert template.ground_truth.expected_certificate_classes == ()
    contract_fault = next(
        t for t in silent if t.identifier == "contract-effect-conflict"
    )
    assert contract_fault.ground_truth.repair is None
    assert contract_fault.task.goal_predicate == "opened"
    assert contract_fault.task.initially_open
    reachability_fault = next(
        t for t in silent if t.identifier == "missing-reachability-precondition"
    )
    assert reachability_fault.ground_truth.repair is not None
    assert not reachability_fault.task.on_mounted_drawer


def test_scene_variation_is_deterministic_in_its_seed():
    assert SceneVariation.from_seed(7) == SceneVariation.from_seed(7)
    assert SceneVariation.from_seed(7) != SceneVariation.from_seed(8)
    variation = SceneVariation.from_seed(7)
    assert abs(variation.front_offset_delta) <= 0.05
    assert abs(variation.side_offset_delta) <= 0.05


def test_splits_cover_every_template_and_stay_disjoint(templates):
    cases = generate_splits(templates, variations_per_template=10, seed=1)
    assert len(cases) == 10 * len(templates)
    for template in templates:
        template_cases = [c for c in cases if c.template_id == template.identifier]
        by_split = {
            split: [c for c in template_cases if c.split == split]
            for split in (PROPOSAL, ADMISSION, HELD_OUT)
        }
        assert len(by_split[PROPOSAL]) == 4
        assert len(by_split[ADMISSION]) == 3
        assert len(by_split[HELD_OUT]) == 3
        seeds_by_split = {
            split: {c.variation.seed for c in split_cases}
            for split, split_cases in by_split.items()
        }
        assert not seeds_by_split[PROPOSAL] & seeds_by_split[ADMISSION]
        assert not seeds_by_split[PROPOSAL] & seeds_by_split[HELD_OUT]
        assert not seeds_by_split[ADMISSION] & seeds_by_split[HELD_OUT]


def test_splits_are_deterministic(templates):
    assert generate_splits(templates, seed=3) == generate_splits(templates, seed=3)
    assert generate_splits(templates, seed=3) != generate_splits(templates, seed=4)


def test_calibrated_manifest_excludes_infeasible_cases_and_round_trips(
    correct, templates, tmp_path
):
    template = (templates[0],)
    rejected = set()

    def feasible(_, variation):
        # Deterministically reject a subset; calibration must keep sampling.
        reasons = ["geometry infeasible"] if variation.seed % 3 == 0 else []
        if reasons:
            rejected.add(variation.seed)
        return reasons

    manifest = calibrate_splits(
        template,
        feasible,
        correct,
        variations_per_template=6,
        seed=4,
    )
    assert len(manifest.cases) == 6
    assert not ({case.variation.seed for case in manifest.cases} & rejected)
    assert manifest.exclusions
    path = tmp_path / "splits.json"
    manifest.save(path)
    restored = FrozenSplitManifest.load(path)
    assert restored == manifest
    assert restored.validation_problems({template[0].identifier}) == []


def test_calibration_error_retains_rejection_evidence(correct, templates):
    template = templates[0]

    with pytest.raises(SplitCalibrationError) as caught:
        calibrate_splits(
            (template,),
            lambda _template, variation: [f"seed {variation.seed} rejected"],
            correct,
            variations_per_template=3,
            maximum_candidates_per_template=3,
        )

    error = caught.value
    assert error.template_id == template.identifier
    assert error.accepted_count == 0
    assert error.attempted_count == 3
    assert error.accepted_candidates == ()
    assert len(error.exclusions) == 3


def test_manifest_validation_rejects_missing_and_cross_split_cases(templates):
    template_id = templates[0].identifier
    cases = (
        FaultCase(template_id, SceneVariation.from_seed(1), PROPOSAL),
        FaultCase(template_id, SceneVariation.from_seed(1), HELD_OUT),
    )
    manifest = FrozenSplitManifest(
        calibration_seed=0,
        variations_per_template=3,
        correct_library_checksum="checksum",
        cases=cases,
    )

    problems = manifest.validation_problems({template_id})

    assert any("duplicate case" in problem for problem in problems)
    assert any("crosses splits" in problem for problem in problems)
    assert any("expected 3" in problem for problem in problems)
    assert any("lacks splits" in problem for problem in problems)


def test_diagnosis_scoring_counts_expected_sets_and_confusion(templates):
    by_id = {template.identifier: template for template in templates}
    observations = [
        (by_id["missing-close-operator"], "missing_operator_model"),  # correct
        (by_id["wrong-parameter-type"], "operator_signature_error"),
        (by_id["execution-binding-mismatch"], "grounding_failure"),  # wrong
        (by_id["missing-delete-effect"], None),  # silent, correct
        (by_id["contract-effect-conflict"], "postcondition_failure"),  # wrong
    ]
    report = evaluate_diagnosis(observations)
    assert report.total == 5
    assert report.correct == 3
    assert report.accuracy == pytest.approx(0.6)
    assert report.confusion[("missing_operator_model", "missing_operator_model")] == 1
    assert report.confusion[("postcondition_failure", "grounding_failure")] == 1
    assert report.confusion[("success", "postcondition_failure")] == 1
    assert "diagnosis accuracy: 3/5" in report.render()


def test_templates_reject_a_drifted_library(correct, grounding_catalog):
    from experiments.resym.icra.articulation.faults import UnknownTemplateSymbolError

    import copy

    drifted = copy.deepcopy(correct)
    del drifted.operators["close-drawer"]
    with pytest.raises(UnknownTemplateSymbolError):
        drawer_fault_templates(drifted, grounding_catalog)
