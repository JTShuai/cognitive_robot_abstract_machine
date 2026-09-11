"""
The E1/E2 harness on the real Tracy scene: real certificates, real admission probes (IK,
collisions, Fast Downward, monitored execution) on admission-split scene variations, and
the held-out battery on scenes nobody trained or validated on.

Container suite (shares the session-scoped Tracy fixtures). Case counts are trimmed to
one scene per split — the full grid is the actual experiment run, not a regression test.
"""

from __future__ import annotations

import json

import pytest

from experiments.resym.icra.articulation.tracy_bench import build_tracy_bench
from experiments.resym.seed_library import build_fixed_arm_library
from resym.repair.backends import EnumerationBackend, OutcomeStatus
from experiments.resym.icra.harness import (
    POLICY_TIERS,
    admission_candidates,
    run_e1,
    run_e2,
)
from experiments.resym.icra.articulation.faults import (
    ADMISSION,
    HELD_OUT,
    PROPOSAL,
    SceneVariation,
    drawer_fault_templates,
    generate_splits,
)
from resym.observability.runlog import RunRecorder
from resym.platform.articulation import (
    articulation_connection,
    is_articulated_object,
)


@pytest.fixture(scope="module")
def bench(tracy_setup, tracy_universe):
    return build_tracy_bench(tracy_setup, tracy_universe)


@pytest.fixture(scope="module", autouse=True)
def restore_scene_afterwards(tracy_setup, tracy_universe):
    """
    The bench moves the table mount and the drawer joints; put both back for whichever
    container tests run after this module.
    """
    yield
    from experiments.resym.icra.articulation.tracy_bench import TracyBench

    bench = TracyBench(tracy_setup, tracy_universe)
    bench.apply_variation(None, 0.0, bench.goal_drawer)


@pytest.fixture(scope="module")
def template():
    templates = drawer_fault_templates(build_fixed_arm_library())
    return next(t for t in templates if t.identifier == "missing-close-operator")


@pytest.fixture(scope="module")
def cases(template):
    """
    One scene variation per split keeps the runtime of this smoke test bounded; splits
    stay disjoint by construction.
    """
    generated = generate_splits((template,), seed=1)
    keep = {PROPOSAL: 1, ADMISSION: 1, HELD_OUT: 1}
    trimmed = []
    for case in generated:
        if keep.get(case.split, 0) > 0:
            trimmed.append(case)
            keep[case.split] -= 1
    return tuple(trimmed)


def test_e1_episode_endpoints_on_the_real_scene(bench, template, cases, tmp_path):
    """
    Oracle reference: admitted by the real suite and clean on the held-out battery.

    Typed enumeration: its first statically clean
    candidate is behaviorally wrong and the suite refuses it. No-repair:
    nothing proposed. All from one real proposal-scene certificate.
    """
    recorder = RunRecorder.create("e1-smoke", root=tmp_path / "runs")
    report = run_e1(
        bench,
        backends={"typed-enumeration": EnumerationBackend()},
        templates=[template],
        cases=cases,
        working_root=tmp_path / "work",
        recorder=recorder,
    )
    by_backend = {e.backend: e for e in report.episodes}

    oracle = by_backend["oracle-reference"]
    assert oracle.proposal_class == "missing_operator_model"
    assert oracle.admitted, oracle.admission_evidence
    assert oracle.held_out_passed, oracle.held_out_failures
    assert oracle.correct_repair and not oracle.false_admission

    enumeration = by_backend["typed-enumeration"]
    assert enumeration.status == OutcomeStatus.PATCH_PROPOSED.value
    assert not enumeration.admitted
    failed_groups = {
        t["group"] for t in enumeration.admission_evidence["tests"] if t["failures"]
    }
    assert failed_groups, enumeration.admission_evidence
    assert not enumeration.false_admission

    no_repair = by_backend["no-repair"]
    assert no_repair.status == OutcomeStatus.NO_CANDIDATE.value
    assert not no_repair.admitted
    records = [
        json.loads(line)
        for line in (recorder.directory / "D_experiment" / "episodes.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {r["backend"] for r in records} == set(by_backend)


def test_apply_variation_restores_non_drawer_world_state(bench):
    tracy_bench = bench.runner.__self__
    drawer_dofs = {
        articulation_connection(grounded).dof.id
        for grounded in tracy_bench.universe.objects.values()
        if is_articulated_object(grounded)
    }
    dof_id = next(
        identifier
        for identifier in tracy_bench.setup.world.state
        if identifier not in drawer_dofs
    )
    expected = tracy_bench._initial_world_state[dof_id].position
    tracy_bench.setup.world.state[dof_id].position = expected + 0.25

    tracy_bench.apply_variation(None, 0.0, tracy_bench.goal_drawer)

    assert tracy_bench.setup.world.state[dof_id].position == pytest.approx(expected)

    variation = SceneVariation(seed=1, initial_fraction_noise=0.03)
    tracy_bench.apply_variation(variation, 0.0, tracy_bench.goal_drawer)
    connection = articulation_connection(tracy_bench.universe[tracy_bench.goal_drawer])
    limits = connection.dof.limits
    position = tracy_bench.setup.world.state[connection.dof.id].position
    fraction = (position - limits.lower.position) / (
        limits.upper.position - limits.lower.position
    )
    assert fraction == pytest.approx(0.03)


def test_frozen_admission_scenes_are_order_independent(bench, tmp_path):
    templates = drawer_fault_templates(build_fixed_arm_library())
    template = next(
        item for item in templates if item.identifier == "missing-open-operator"
    )
    correct = build_fixed_arm_library()
    for index, seed in enumerate((384452587, 626401695)):
        outcome = bench.runner(
            correct,
            template.task,
            SceneVariation.from_seed(seed),
            tmp_path / str(index),
        )
        assert outcome.succeeded, (seed, outcome.failure_class_value)


def test_e2_single_witness_admits_what_a_negative_context_refuses(
    bench, template, cases, tmp_path
):
    """
    The §9.5 core claim on the real scene: the missing-precondition bait closes the
    drawer perfectly on the positive witness, so a single-witness curator admits it; the
    negative admission context (the out-of-reach drawer it also claims to close) refuses
    it; the held-out battery documents the single-witness false admission.
    """

    def pool(template, correct):
        return tuple(
            candidate
            for candidate in admission_candidates(template, correct)
            if candidate.name
            in (
                "reference",
                "missing-reachability-precondition",
            )
        )

    report = run_e2(
        bench,
        templates=[template],
        cases=cases,
        working_root=tmp_path,
        policies=(POLICY_TIERS[0], POLICY_TIERS[2]),
        candidate_pool=pool,
    )

    def decision(policy, candidate):
        return next(d for d in report.of_policy(policy) if d.candidate == candidate)

    assert decision("single-witness", "reference").admitted
    assert decision("positive-negative", "reference").admitted

    bait_single = decision("single-witness", "missing-reachability-precondition")
    assert bait_single.admitted and bait_single.false_admission
    assert bait_single.held_out_failures  # the gold battery exposes it

    bait_checked = decision("positive-negative", "missing-reachability-precondition")
    assert not bait_checked.admitted

    assert report.false_admission_rate("single-witness").rate == 1.0
    assert report.false_admission_rate("positive-negative").rate == 0.0
    assert report.mean_tests_run("single-witness") < report.mean_tests_run(
        "positive-negative"
    )
