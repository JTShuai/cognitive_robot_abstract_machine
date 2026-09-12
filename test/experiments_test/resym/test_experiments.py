"""
The E1/E2 harness over a symbolic fake bench: endpoint definitions, split handling,
reference baselines, validation-policy tiers, the §11 records, and the Gate P2 pre-
registered rule.

The fake bench simulates exactly the decision structure of the real closed loop
(achiever present? capability implemented? precondition honest about reach? truth
binding correct?) so the harness logic is exercised end-to-end without the CRAM stack.
Host-runnable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace

import pytest

from experiments.resym.seed_library import build_fixed_arm_library
from resym.repair.patch import ModelPatch
from resym.repair.backends import (
    AGENTIC_RAG_BACKEND_NAME,
    BudgetMeter,
    EnumerationBackend,
    OutcomeStatus,
    RepairBackend,
    RepairOutcome,
    RepairTask,
)
from resym.repair.curator import Curator
from experiments.resym.icra.validation import (
    AdmissionTest,
    BehaviouralCurator,
    MandatorySuite,
    SuiteGroup,
)
from experiments.resym.icra.harness import (
    POLICY_TIERS,
    E1Report,
    E2Case,
    E2Report,
    EpisodeResult,
    ExperimentBench,
    admission_candidates,
    agentic_gate,
    run_e1,
    run_e2,
)
from experiments.resym.icra.articulation.faults import (
    HELD_OUT,
    drawer_fault_templates,
    generate_splits,
)
from resym.core.model import Literal
from resym.platform.capabilities import ARTICULATION_CAPABILITY_UID
from resym.observability.runlog import RunRecorder
from resym.planning.selection import (
    UnknownPredicateError,
    select_for_goal,
)
from resym.repair.versioning import VersionedLibraryStore

# -- the symbolic drawer world -----------------------------------------

AVAILABLE_CAPABILITIES = frozenset({ARTICULATION_CAPABILITY_UID})
CORRECT_TARGET_STATE = {"opened": "OPEN", "closed": "CLOSED"}


def outcome(succeeded, failure_class=None, goal="closed", plan=()):
    return SimpleNamespace(
        succeeded=succeeded,
        failure_class_value=failure_class,
        certificate=(
            None
            if succeeded
            else SimpleNamespace(
                task_goal=(Literal(goal, ("d1",)),),
                failure_class=SimpleNamespace(value=failure_class),
                render=lambda: f"failure class: {failure_class}\ngoal: ({goal} d1)",
            )
        ),
        result=SimpleNamespace(plan=tuple(plan)),
    )


def simulate(catalog, library, goal, reachable=True):
    """
    The closed loop in miniature, mirroring the real semantics: the static capability
    gate over the goal-relevant selection closure, then a wrong truth binding claims the
    goal satisfied without acting, an operator without a reachability precondition
    'succeeds' in the simplified bench even out of reach, and a misbound request is
    caught by monitoring.
    """
    try:
        selection = select_for_goal(library, (Literal(goal, ("d1",)),))
    except UnknownPredicateError:
        return outcome(False, "missing_predicate_model", goal)
    if any(
        operator.execution_binding.capability_ref.uid not in AVAILABLE_CAPABILITIES
        for operator in selection.operators.values()
    ):
        return outcome(False, "unsupported_capability", goal)
    known_factory_uids = {specification.uid for specification in catalog}
    if any(
        predicate.grounding_plan.factory_uid not in known_factory_uids
        for predicate in selection.predicates.values()
    ):
        return outcome(False, "predicate_implementation_error", goal)
    predicate = library.predicates[goal]
    correct_plan = build_fixed_arm_library(catalog).predicates[goal].grounding_plan
    if predicate.grounding_plan != correct_plan:
        return outcome(True, goal=goal, plan=())  # claimed satisfied, no action
    achiever = next(
        (
            operator
            for operator in selection.operators.values()
            if any(e.predicate == goal for e in operator.add_effects)
        ),
        None,
    )
    if achiever is None:
        return outcome(False, "missing_operator_model", goal)
    honest_about_reach = any(
        literal.predicate == "ready-to-open" for literal in achiever.preconditions
    )
    if not reachable:
        if honest_about_reach:
            return outcome(False, "geometric_infeasibility", goal)
        return outcome(True, goal=goal, plan=(achiever.name,))  # false success
    target_state = achiever.execution_binding.role_map["target_state"].value
    if target_state != CORRECT_TARGET_STATE[goal]:
        return outcome(False, "postcondition_failure", goal)
    return outcome(True, goal=goal, plan=(achiever.name,))


def behavioural_tests(catalog, template, cases, vacuous_beyond_positive=False):
    """
    The four mandatory groups as symbolic probes; ``vacuous_beyond_positive`` models a
    box-ticking admission suite whose non-positive groups pass everything (coverage
    looks complete, behaviour is not checked).
    """
    goal = template.task.goal_predicate
    reverse_goal = "closed" if goal == "opened" else "opened"
    faulted = template.apply(build_fixed_arm_library(catalog))
    reverse_worked_before = reverse_goal in faulted.predicates and any(
        any(e.predicate == reverse_goal for e in operator.add_effects)
        for operator in faulted.operators.values()
    )

    def context(case):
        return f"{template.identifier}/scene-{case.variation.seed}"

    def positive(candidate):
        result = simulate(catalog, candidate, goal, reachable=True)
        return (
            []
            if result.succeeded
            else [f"repair task failed: {result.failure_class_value}"]
        )

    def negative(candidate):
        result = simulate(catalog, candidate, goal, reachable=False)
        return ["claimed success out of reach"] if result.succeeded else []

    def boundary(candidate):
        result = simulate(catalog, candidate, goal, reachable=True)
        if result.succeeded and not result.result.plan:
            return ["claimed satisfied at the boundary without acting"]
        return []

    def regression(candidate):
        if not reverse_worked_before:
            return []
        result = simulate(catalog, candidate, reverse_goal, reachable=True)
        return (
            []
            if result.succeeded
            else [f"reverse task now fails: {result.failure_class_value}"]
        )

    def vacuous(candidate):
        return []

    return (
        AdmissionTest(
            "positive-0", SuiteGroup.CAPABILITY_POSITIVE, context(cases[0]), positive
        ),
        AdmissionTest(
            "positive-1", SuiteGroup.CAPABILITY_POSITIVE, context(cases[-1]), positive
        ),
        AdmissionTest(
            "negative",
            SuiteGroup.CAPABILITY_NEGATIVE,
            context(cases[-1]),
            vacuous if vacuous_beyond_positive else negative,
        ),
        AdmissionTest(
            "boundary",
            SuiteGroup.BOUNDARY,
            context(cases[0]),
            vacuous if vacuous_beyond_positive else boundary,
        ),
        AdmissionTest(
            "regression",
            SuiteGroup.REGRESSION,
            context(cases[0]),
            vacuous if vacuous_beyond_positive else regression,
        ),
    )


def fake_bench(catalog, weak_admission=False):
    def curator_builder(template, cases, proposal_context_id, working_directory):
        return BehaviouralCurator(
            available_capabilities=AVAILABLE_CAPABILITIES,
            grounding_factory_specs={
                specification.uid: specification for specification in catalog
            },
            suite=MandatorySuite(
                version="fake-suite-1",
                tests=behavioural_tests(
                    catalog, template, cases, vacuous_beyond_positive=weak_admission
                ),
            ),
        )

    def held_out(template, cases, candidate, working_directory):
        failures = []
        for test in behavioural_tests(catalog, template, cases):
            failures.extend(
                f"[{test.group.value}] {failure}" for failure in test.run(candidate)
            )
        return failures

    def fake_runner(library, task, variation, working_directory):
        return simulate(
            catalog, library, task.goal_predicate, reachable=task.on_mounted_drawer
        )

    def fake_safety_runner(library, task, variation, working_directory):
        return simulate(catalog, library, task.goal_predicate, reachable=False)

    return ExperimentBench(
        correct_library=partial(build_fixed_arm_library, catalog),
        runner=fake_runner,
        curator_builder=curator_builder,
        held_out=held_out,
        safety_runner=fake_safety_runner,
        grounding_factory_listing="\n".join(
            sorted(specification.uid for specification in catalog)
        ),
        capability_listing="\n".join(f"- {n}" for n in sorted(AVAILABLE_CAPABILITIES)),
    )


@dataclass
class ScriptedPatchBackend(RepairBackend):
    """
    Submits one fixed patch — for injecting labelled candidates into E1.
    """

    patch: ModelPatch
    name: str = "scripted-patch"

    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        meter.spend_candidate()
        return RepairOutcome(
            backend=self.name,
            status=OutcomeStatus.PATCH_PROPOSED,
            patch=self.patch,
            budget=meter.snapshot(),
        )


@pytest.fixture(scope="module")
def templates(grounding_catalog):
    return {
        t.identifier: t
        for t in drawer_fault_templates(
            build_fixed_arm_library(grounding_catalog), grounding_catalog
        )
    }


@pytest.fixture(scope="module")
def cases(grounding_catalog):
    return generate_splits(
        drawer_fault_templates(
            build_fixed_arm_library(grounding_catalog), grounding_catalog
        ),
        seed=1,
    )


# -- E1 -----------------------------------------------------------------


def test_e1_refuses_to_reuse_proposal_contexts(
    templates, cases, tmp_path, grounding_catalog
):
    with pytest.raises(ValueError, match="only 4 independent proposal contexts"):
        run_e1(
            fake_bench(grounding_catalog),
            backends={},
            templates=[templates["missing-close-operator"]],
            cases=cases,
            working_root=tmp_path,
            episodes_per_template=5,
        )


def test_oracle_reference_is_a_correct_repair_and_no_repair_is_not(
    templates, cases, tmp_path, grounding_catalog
):
    report = run_e1(
        fake_bench(grounding_catalog),
        backends={},
        templates=[templates["missing-close-operator"]],
        cases=cases,
        working_root=tmp_path,
    )
    by_backend = {e.backend: e for e in report.episodes}
    oracle = by_backend["oracle-reference"]
    assert oracle.triggered
    assert oracle.proposal_class == "missing_operator_model"
    assert oracle.admitted and oracle.held_out_passed
    assert oracle.correct_repair and not oracle.false_admission
    no_repair = by_backend["no-repair"]
    assert no_repair.status == OutcomeStatus.NO_CANDIDATE.value
    assert not no_repair.admitted and not no_repair.correct_repair
    assert report.correct_repair_rate("oracle-reference").rate == 1.0
    assert report.false_admission_rate("oracle-reference").rate == 0.0

    # Admission left a version trail: faulted base plus the admitted
    # candidate, head at the admitted version, evidence in the metadata.
    assert oracle.base_version is not None and oracle.admitted_version is not None
    store = VersionedLibraryStore(
        tmp_path / "versions" / "missing-close-operator/episode-0/oracle-reference"
    )
    assert store.head_version() == oracle.admitted_version
    admitted_library = store.load(oracle.admitted_version)
    assert "close-drawer" in admitted_library.operators
    info = store.info(oracle.admitted_version)
    assert info.parent == oracle.base_version
    assert info.metadata["backend"] == "oracle-reference"
    assert info.metadata["admission_evidence"]["tests"]
    # A no-candidate episode commits nothing at all.
    assert no_repair.base_version is None and no_repair.admitted_version is None


def test_oracle_declares_unsupported_on_the_navigation_template(
    templates, cases, tmp_path, grounding_catalog, library
):
    report = run_e1(
        fake_bench(grounding_catalog),
        backends={},
        templates=[templates["unsupported-navigation-library"]],
        cases=cases,
        working_root=tmp_path,
    )
    oracle = next(e for e in report.episodes if e.backend == "oracle-reference")
    assert oracle.declared_unsupported and oracle.correct_unsupported
    assert not oracle.admitted
    assert report.unsupported_detection_rate("oracle-reference").rate == 1.0


def test_silent_template_records_no_failure_and_leaves_rates_alone(
    templates, cases, tmp_path, grounding_catalog
):
    report = run_e1(
        fake_bench(grounding_catalog),
        backends={},
        templates=[templates["missing-delete-effect"]],
        cases=cases,
        working_root=tmp_path,
    )
    assert all(not e.triggered for e in report.episodes)
    assert report.correct_repair_rate("oracle-reference").total == 0


def test_enumeration_first_static_clean_candidate_is_refused_behaviorally(
    templates, cases, tmp_path, grounding_catalog
):
    """
    The enumerator stops at the first statically clean candidate; the mandatory suite
    must catch that it is behaviorally wrong — this is the baseline/curator division of
    labour working as designed.
    """
    report = run_e1(
        fake_bench(grounding_catalog),
        backends={"typed-enumeration": EnumerationBackend()},
        templates=[templates["missing-close-operator"]],
        cases=cases,
        working_root=tmp_path,
        include_reference_baselines=False,
    )
    (episode,) = report.episodes
    assert episode.status == OutcomeStatus.PATCH_PROPOSED.value
    assert not episode.admitted
    failed = [t for t in episode.admission_evidence["tests"] if t["failures"]]
    assert failed, episode.admission_evidence
    assert not episode.false_admission and not episode.correct_repair
    # The refusal leaves the version trail at the faulted base.
    assert episode.base_version is not None
    assert episode.admitted_version is None
    store = VersionedLibraryStore(
        tmp_path / "versions" / "missing-close-operator/episode-0/typed-enumeration"
    )
    assert store.head_version() == episode.base_version


def test_weak_admission_suite_admits_a_bait_and_held_out_catches_it(
    templates, cases, tmp_path, grounding_catalog
):
    """
    The E1 false-admission endpoint: a box-ticking admission suite lets the missing-
    precondition bait through; the held-out battery flags it.
    """
    template = templates["missing-close-operator"]
    bait = next(
        c
        for c in admission_candidates(
            template, build_fixed_arm_library(grounding_catalog)
        )
        if c.name == "missing-reachability-precondition"
    )
    backend = ScriptedPatchBackend(patch=bait.patch)
    weak = run_e1(
        fake_bench(grounding_catalog, weak_admission=True),
        backends={"scripted-patch": backend},
        templates=[template],
        cases=cases,
        working_root=tmp_path / "weak",
        include_reference_baselines=False,
    )
    (episode,) = weak.episodes
    assert episode.admitted
    assert episode.held_out_failures
    assert episode.false_admission and not episode.correct_repair

    strict = run_e1(
        fake_bench(grounding_catalog),
        backends={"scripted-patch": backend},
        templates=[template],
        cases=cases,
        working_root=tmp_path / "strict",
        include_reference_baselines=False,
    )
    (episode,) = strict.episodes
    assert not episode.admitted and not episode.false_admission


def test_probe_hook_runs_the_repair_task_on_proposal_scenes(
    templates, cases, tmp_path, grounding_catalog
):
    template = templates["missing-close-operator"]
    captured = {}

    @dataclass
    class ProbingBackend(RepairBackend):
        name: str = "probing"

        def repair(self, task, meter):
            reference = template.ground_truth.repair
            probes = [line.lstrip("- ") for line in task.probe_listing.splitlines()]
            captured["probe"] = []
            for probe in probes:
                meter.spend_probe()
                captured["probe"].append(task.run_probe(probe, reference))
            captured["listing"] = task.probe_listing
            captured["required"] = task.required_probes
            return RepairOutcome(backend=self.name, status=OutcomeStatus.NO_CANDIDATE)

    run_e1(
        fake_bench(grounding_catalog),
        backends={"probing": ProbingBackend()},
        templates=[template],
        cases=cases,
        working_root=tmp_path,
        include_reference_baselines=False,
    )
    assert all(probe["succeeded"] for probe in captured["probe"])
    assert captured["probe"][0]["failure_class"] is None
    assert "repair-task-on-proposal-scene-" in captured["listing"]
    assert "safety-refusal-on-proposal-scene-" in captured["listing"]
    assert set(captured["required"]) == {
        probe.removeprefix("- ") for probe in captured["listing"].splitlines()
    }


def test_a_backend_killed_by_infrastructure_loses_the_episode_not_the_grid(
    templates, cases, tmp_path, grounding_catalog
):
    """A rate-limit storm outlasting the retry policy escapes the backend
    as an exception: the episode is recorded as infrastructure-error,
    excluded from every rate denominator, and the remaining roster keeps
    running."""
    from experiments.resym.icra.harness import INFRASTRUCTURE_STATUS
    from resym.llm.client import TransientInfrastructureError

    template = templates["missing-close-operator"]

    @dataclass
    class DeadEndpointBackend(RepairBackend):
        name: str = "dead-endpoint"

        def repair(self, task, meter):
            raise TransientInfrastructureError("LLM call failed after 10 attempts")

    report = run_e1(
        fake_bench(grounding_catalog),
        backends={"dead-endpoint": DeadEndpointBackend()},
        templates=[template],
        cases=cases,
        working_root=tmp_path,
    )
    dead = next(e for e in report.episodes if e.backend == "dead-endpoint")
    assert dead.status == INFRASTRUCTURE_STATUS
    assert not dead.triggered
    assert dead.events[0]["step"] == "infrastructure"
    assert "TransientInfrastructureError" in dead.events[0]["error"]
    assert report.correct_repair_rate("dead-endpoint").total == 0
    assert report.false_admission_rate("dead-endpoint").total == 0
    assert "episodes lost to infrastructure: 1" in report.render()
    # the rest of the roster was unaffected
    oracle = next(e for e in report.episodes if e.backend == "oracle-reference")
    assert oracle.correct_repair


def test_backend_programming_error_is_not_hidden_as_infrastructure(
    templates, cases, tmp_path, grounding_catalog
):
    template = templates["missing-close-operator"]

    @dataclass
    class BrokenBackend(RepairBackend):
        name: str = "broken"

        def repair(self, task, meter):
            raise RuntimeError("program defect")

    with pytest.raises(RuntimeError, match="program defect"):
        run_e1(
            fake_bench(grounding_catalog),
            backends={"broken": BrokenBackend()},
            templates=[template],
            cases=cases,
            working_root=tmp_path,
            include_reference_baselines=False,
        )


def test_gate_margins_must_be_rates():
    from experiments.resym.icra.harness import GateP2Margins

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        GateP2Margins(repair_noninferiority=1.01)


def test_episode_records_rebuild_an_equivalent_report(
    templates, cases, tmp_path, grounding_catalog
):
    """
    to_json -> from_json -> E1Report reproduces every pre-registered rate, so a sharded
    or crashed grid can be assembled from its episodes.jsonl records alone.
    """
    import json as json_module

    from experiments.resym.icra.harness import e1_report_from_records

    template = templates["missing-close-operator"]
    original = run_e1(
        fake_bench(grounding_catalog),
        backends={"typed-enumeration": EnumerationBackend()},
        templates=[template],
        cases=cases,
        working_root=tmp_path,
    )
    records = [
        json_module.loads(json_module.dumps(e.to_json())) for e in original.episodes
    ]
    rebuilt = e1_report_from_records(records)
    assert rebuilt.backends() == original.backends()
    for backend in original.backends():
        for rate in ("correct_repair_rate", "false_admission_rate"):
            assert getattr(rebuilt, rate)(backend) == getattr(original, rate)(backend)
    assert rebuilt.per_template_lines("oracle-reference") == (
        original.per_template_lines("oracle-reference")
    )


def test_a_crashing_probe_is_an_observation_not_an_abort(
    templates, cases, tmp_path, grounding_catalog, library
):
    """
    The probed patch is arbitrary model output; if it makes the runner blow up, the
    backend gets a structured error back and the episode continues.
    """
    from dataclasses import replace as dc_replace

    template = templates["missing-close-operator"]
    captured = {}
    bench = fake_bench(grounding_catalog)
    real_runner = bench.runner

    def exploding_runner(library, task, variation, working_directory):
        # The initial fault manifestation must still work; only the probe
        # of the (arbitrary) candidate blows up.
        if working_directory.name.startswith("probe-"):
            raise ValueError("too many values to unpack (expected 2)")
        return real_runner(library, task, variation, working_directory)

    @dataclass
    class ProbingBackend(RepairBackend):
        name: str = "probing"

        def repair(self, task, meter):
            meter.spend_probe()
            captured["probe"] = task.run_probe(
                task.probe_listing.splitlines()[0].lstrip("- "),
                template.ground_truth.repair,
            )
            return RepairOutcome(backend=self.name, status=OutcomeStatus.NO_CANDIDATE)

    run_e1(
        dc_replace(bench, runner=exploding_runner),
        backends={"probing": ProbingBackend()},
        templates=[template],
        cases=cases,
        working_root=tmp_path,
        include_reference_baselines=False,
    )
    assert captured["probe"]["succeeded"] is False
    assert (
        captured["probe"]["error"]
        == "probe crashed (ValueError): too many values to unpack (expected 2)"
    )


def test_run_e1_appends_episode_records_to_the_runlog(
    templates, cases, tmp_path, grounding_catalog
):
    recorder = RunRecorder.create("e1-test", root=tmp_path / "runs")
    run_e1(
        fake_bench(grounding_catalog),
        backends={},
        templates=[templates["missing-close-operator"]],
        cases=cases,
        working_root=tmp_path / "work",
        recorder=recorder,
    )
    lines = (
        (recorder.directory / "D_experiment" / "episodes.jsonl")
        .read_text()
        .splitlines()
    )
    records = [json.loads(line) for line in lines]
    assert {r["backend"] for r in records} == {"no-repair", "oracle-reference"}
    assert all("budget" in r and "false_admission" in r for r in records)


def test_missing_split_cases_are_rejected(
    templates, cases, tmp_path, grounding_catalog
):
    incomplete = [c for c in cases if c.split != HELD_OUT]
    with pytest.raises(ValueError, match="held-out"):
        run_e1(
            fake_bench(grounding_catalog),
            backends={},
            templates=[templates["missing-close-operator"]],
            cases=incomplete,
            working_root=tmp_path,
        )


# -- labelled candidates ------------------------------------------------


def test_admission_candidates_are_labelled_and_static_clean(
    templates, grounding_catalog
):
    correct = build_fixed_arm_library(grounding_catalog)
    template = templates["missing-close-operator"]
    candidates = {c.name: c for c in admission_candidates(template, correct)}
    assert candidates["reference"].behaviorally_correct
    assert not candidates["missing-reachability-precondition"].behaviorally_correct
    assert not candidates["misbound-execution-request"].behaviorally_correct
    unconditional = candidates["missing-reachability-precondition"].patch.operators[0]
    assert all(
        literal.predicate != "ready-to-open" for literal in unconditional.preconditions
    )
    misbound = candidates["misbound-execution-request"].patch.operators[0]
    assert misbound.execution_binding.role_map["target_state"].value == "OPEN"
    curator = Curator(
        available_capabilities=AVAILABLE_CAPABILITIES,
        grounding_factory_specs={
            specification.uid: specification for specification in grounding_catalog
        },
    )
    faulted = template.apply(correct)
    for candidate in candidates.values():
        assert curator.static_review(candidate.patch, faulted) == []


def test_unrepairable_templates_have_no_candidate_pool(
    templates, grounding_catalog, library
):
    assert (
        admission_candidates(
            templates["unsupported-navigation-library"],
            build_fixed_arm_library(grounding_catalog),
        )
        == ()
    )


# -- E2 -----------------------------------------------------------------


def test_e2_policy_tiers_separate_the_baits(
    templates, cases, tmp_path, grounding_catalog
):
    report = run_e2(
        fake_bench(grounding_catalog),
        templates=[templates["missing-close-operator"]],
        cases=cases,
        working_root=tmp_path,
    )
    assert report.policies() == tuple(p.name for p in POLICY_TIERS)

    def decision(policy, candidate):
        return next(d for d in report.of_policy(policy) if d.candidate == candidate)

    # The reference repair is admitted under every tier.
    for policy in report.policies():
        assert decision(policy, "reference").admitted
        assert report.missed_admission_rate(policy).rate == 0.0

    # The missing-precondition bait passes positive-only tiers and is caught the
    # moment the negative group runs.
    assert decision("single-witness", "missing-reachability-precondition").admitted
    assert decision(
        "multi-context-positive", "missing-reachability-precondition"
    ).admitted
    assert not decision(
        "positive-negative", "missing-reachability-precondition"
    ).admitted
    assert not decision("full-suite", "missing-reachability-precondition").admitted

    # The misbound skill fails even a single positive witness.
    assert not decision("single-witness", "misbound-execution-request").admitted

    # False admission is monotonically eliminated with suite strength.
    assert report.false_admission_rate("single-witness").successes == 1
    assert report.false_admission_rate("positive-negative").successes == 0

    # Admitted-but-wrong is exposed by the held-out battery.
    bait = decision("single-witness", "missing-reachability-precondition")
    assert bait.held_out_evaluated and bait.held_out_failures
    assert report.held_out_clean_rate("single-witness").rate < 1.0
    assert report.held_out_clean_rate("full-suite").rate == 1.0

    # Curator cost grows with the tier.
    costs = [report.mean_tests_run(p.name) for p in POLICY_TIERS]
    assert costs == sorted(costs)
    assert costs[0] == 1.0

    rendered = report.render()
    assert "single-witness" in rendered and "false admission" in rendered


def test_e2_record_roundtrip_preserves_held_out_evaluation():
    original = E2Case(
        template_id="t",
        candidate="reference",
        behaviorally_correct=True,
        policy="full-suite",
        admitted=True,
        tests_run=5,
        static_objections=0,
        held_out_evaluated=True,
    )

    restored = E2Case.from_json(original.to_json())

    assert restored.held_out_evaluated


def test_missing_held_out_evaluation_is_not_reported_clean():
    unevaluated = E2Case(
        template_id="t",
        candidate="reference",
        behaviorally_correct=True,
        policy="full-suite",
        admitted=True,
        tests_run=5,
        static_objections=0,
    )

    report = E2Report(decisions=(unevaluated,))

    assert report.held_out_clean_rate("full-suite").rate == 0.0


# -- Gate P2 ------------------------------------------------------------


def episode_for_gate(backend, unit, correct, false_admission=False, tokens=1000):
    result = EpisodeResult(
        backend=backend,
        template_id=f"t{unit}",
        group="D1",
        template_label="missing-operator",
        template_unsupported=False,
        template_repairable=True,
        episode_index=0,
        proposal_seed=unit,
        proposal_context_id=f"t{unit}/scene-{unit}",
        status="patch_proposed",
    )
    result.admitted = correct or false_admission
    result.held_out_evaluated = result.admitted
    result.held_out_failures = ("held-out failure",) if false_admission else ()
    result.budget = {
        "candidates_used": 1,
        "probes_used": 0,
        "estimated_tokens_used": tokens,
    }
    return result


def gate_report(agent_correct, baseline_correct, **kwargs):
    episodes = []
    for unit in range(20):
        episodes.append(
            episode_for_gate(
                AGENTIC_RAG_BACKEND_NAME,
                unit,
                correct=unit < agent_correct,
                false_admission=unit < kwargs.get("agent_false", 0)
                and not unit < agent_correct,
                tokens=kwargs.get("agent_tokens", 1000),
            )
        )
        episodes.append(
            episode_for_gate(
                "fixed-pipeline",
                unit,
                correct=unit < baseline_correct,
                tokens=kwargs.get("baseline_tokens", 1000),
            )
        )
    return E1Report(episodes=tuple(episodes))


class TestGateP2:
    def test_clear_repair_gain_keeps_the_agentic_claim(self):
        decision = agentic_gate(gate_report(18, 8), iterations=2000)
        assert decision.keep_agentic
        assert "higher correct repair" in decision.reason

    def test_equal_correctness_but_cheaper_keeps_the_claim(self):
        decision = agentic_gate(
            gate_report(10, 10, agent_tokens=500, baseline_tokens=2000),
            iterations=2000,
        )
        assert decision.keep_agentic
        assert "lower cost" in decision.reason
        assert "estimated_tokens_used" in decision.reason

    def test_no_gain_downgrades_the_claim(self):
        decision = agentic_gate(gate_report(8, 8), iterations=2000)
        assert not decision.keep_agentic
        assert "DOWNGRADE" in decision.render()

    def test_repair_gain_bought_with_false_admissions_does_not_count(self):
        decision = agentic_gate(gate_report(10, 8, agent_false=18), iterations=2000)
        assert not decision.keep_agentic


def test_correct_repair_pairing_excludes_unsupported_templates():
    repairable_a = episode_for_gate("a", 1, correct=True)
    repairable_b = episode_for_gate("b", 1, correct=False)
    unsupported_a = episode_for_gate("a", 2, correct=False)
    unsupported_b = episode_for_gate("b", 2, correct=False)
    for episode in (unsupported_a, unsupported_b):
        episode.template_repairable = False
        episode.template_unsupported = True

    report = E1Report(
        episodes=(repairable_a, repairable_b, unsupported_a, unsupported_b)
    )

    a, b = report.paired_metric("correct_repair", "a", "b")
    assert a == [1.0]
    assert b == [0.0]
