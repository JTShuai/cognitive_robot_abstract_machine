"""The E1/E2 experiment harness (revised plan §9.4, §9.5, §11).

E1 runs matched-budget repair episodes: one fault template on one
proposal scene produces a real failure certificate; every backend gets
the same certificate, library, menus, corpus, curator tools, and budget
meter; a proposed patch faces the independent curator's full admission
suite (real behavioural probes on *admission-split* scenes); an admitted
patch is finally judged by the held-out battery — the same behavioural
groups instantiated on *held-out* scenes the proposer and the curator
never touched. E2 fixes the candidate set (the reference repair plus
labelled plausible-but-wrong baits derived from it) and varies only the
curator's validation policy across the five pre-registered tiers.

Pre-registered endpoint definitions (denominators in parentheses):

- **correct repair** — patch admitted AND the held-out battery is clean
  AND the template is repairable (episodes where the fault manifested);
- **false admission** — patch admitted AND (the held-out battery fails
  OR the template's ground truth is *unsupported*) (same denominator);
- **unsupported detection** — the backend declared the gap unsupported
  (episodes on unsupported templates where the fault manifested).

Split isolation is physical: proposal probes (including backend
diagnostic probes) run only on proposal-split scenes under
``<root>/proposal/``, admission tests only on admission-split scenes
under ``<root>/admission/``, and the held-out battery runs only after
the admission decision, under ``<root>/held_out/``. No feedback flows
from admission or held-out runs back to any backend.

The platform enters through :class:`ExperimentBench` — the Tracy
binding lives in ``experiments.resym.icra.articulation.tracy_bench``; host tests use a
symbolic fake. This module itself never imports the CRAM chain.
"""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Callable, Mapping, Optional, Sequence

from resym.repair.patch import ModelPatch
from resym.repair.backends import (
    AGENTIC_RAG_BACKEND_NAME,
    Budget,
    BudgetMeter,
    OutcomeStatus,
    RepairBackend,
    RepairOutcome,
    RepairTask,
)
from experiments.resym.icra.validation import BehaviouralCurator, SuiteGroup
from experiments.resym.icra.articulation.faults import (
    ADMISSION,
    HELD_OUT,
    PROPOSAL,
    FaultCase,
    FaultTemplate,
    GroundTruth,
    _opposite_target_binding,
)
from resym.core.model import (
    CapabilityContract,
    GroundingFactoryCandidate,
    SymbolLibrary,
)
from resym.llm.client import TransientInfrastructureError
from resym.evaluation.stats import (
    PairedBootstrap,
    RateSummary,
    paired_bootstrap,
    rate_summary,
)
from resym.repair.versioning import VersionedLibraryStore

NO_FAILURE_STATUS = "no-failure"
"""Episode status when the probe task succeeds despite the fault (SILENT
templates): no certificate, no repair episode."""

INFRASTRUCTURE_STATUS = "infrastructure-error"
"""Episode status when the backend was killed by infrastructure (e.g. a
rate-limit storm outlasting the retry policy): not a property of the
backend, excluded from every rate denominator, reported as lost."""

CaseRunner = Callable[..., object]
"""``(library, task: FaultTask, variation: SceneVariation, working_directory)
-> outcome`` where the outcome is duck-typed like
:class:`~resym.repair.diagnosis.TaskDiagnosis`: ``succeeded``,
``failure_class_value``, ``certificate``, ``result``."""


@dataclass
class ExperimentBench:
    """Everything platform-specific the harness needs, behind callables.

    ``curator_builder`` receives ``(template, admission_cases,
    proposal_context_id, working_directory)`` and returns the trusted
    curator whose mandatory suite runs real behavioural probes on the
    admission-split scenes. ``held_out`` receives ``(template,
    held_out_cases, candidate_library, working_directory)`` and returns
    the observed failures of the full held-out battery (empty = clean).
    Keeping the two separate keeps the experimenters' gold evaluation
    independent of whatever suite the system admits with.
    """

    correct_library: Callable[[], SymbolLibrary]
    runner: CaseRunner
    curator_builder: Callable[
        [FaultTemplate, tuple[FaultCase, ...], str, Path], BehaviouralCurator
    ]
    held_out: Callable[
        [FaultTemplate, tuple[FaultCase, ...], SymbolLibrary, Path], list[str]
    ]
    grounding_factory_listing: str
    capability_listing: str
    grounding_vocabulary_listing: str = ""
    validate_grounding_candidate: Optional[
        Callable[[GroundingFactoryCandidate], tuple[str, ...]]
    ] = None
    grounding_candidate_sink: Optional[Callable[[GroundingFactoryCandidate], None]] = (
        None
    )
    capability_catalog: tuple[CapabilityContract, ...] = ()
    """Reviewed platform catalog, separate from the task's symbol library."""
    capability_draft_listing: str = ""
    capability_draft_ids: frozenset[str] = frozenset()
    safety_runner: Optional[CaseRunner] = None
    """Run the task against a proposal-split negative context. The harness
    interprets honest refusal as a passing diagnostic probe."""

    index: Optional[object] = None
    """The frozen corpus index (``FragmentIndex``); ``None`` = closed book."""

    retrieval_top_k: int = 5


# -- reference baselines (§9.4 conditions 1, 7) -------------------------


@dataclass
class NoRepairBackend(RepairBackend):
    """Condition 1: the seed library stays as it is; nothing is proposed."""

    name: str = "no-repair"

    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        return RepairOutcome(
            backend=self.name,
            status=OutcomeStatus.NO_CANDIDATE,
            budget=meter.snapshot(),
        )


@dataclass
class OracleReferenceBackend(RepairBackend):
    """Condition 7: the template's ground truth, submitted directly — the
    upper bound any automatic backend is measured against."""

    ground_truth: GroundTruth
    name: str = "oracle-reference"

    def repair(self, task: RepairTask, meter: BudgetMeter) -> RepairOutcome:
        if self.ground_truth.unsupported:
            return RepairOutcome(
                backend=self.name,
                status=OutcomeStatus.UNSUPPORTED_DECLARED,
                events=[{"step": "unsupported", "reason": "ground truth"}],
                budget=meter.snapshot(),
            )
        if self.ground_truth.repair is None:
            return RepairOutcome(
                backend=self.name,
                status=OutcomeStatus.NO_CANDIDATE,
                budget=meter.snapshot(),
            )
        meter.spend_candidate()
        return RepairOutcome(
            backend=self.name,
            status=OutcomeStatus.PATCH_PROPOSED,
            patch=self.ground_truth.repair,
            budget=meter.snapshot(),
        )


# -- E1: matched-budget executable repair -------------------------------


@dataclass
class EpisodeResult:
    """One (backend, template, proposal scene) repair episode, with the
    §11 record fields."""

    backend: str
    template_id: str
    group: str
    template_label: str
    template_unsupported: bool
    template_repairable: bool
    episode_index: int
    proposal_seed: int
    proposal_context_id: str
    status: str = NO_FAILURE_STATUS
    proposal_class: Optional[str] = None
    declared_unsupported: bool = False
    admitted: bool = False
    admission_evidence: Optional[dict] = None
    base_version: Optional[str] = None
    """The faulted library as committed before the episode ran."""

    admitted_version: Optional[str] = None
    """The version the curator committed on admission; ``None`` when the
    episode admitted nothing."""

    held_out_evaluated: bool = False
    held_out_failures: tuple[str, ...] = ()
    budget: dict = field(default_factory=dict)
    events: list = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        """The fault manifested and a repair episode actually ran (lost
        infrastructure episodes do not count as attempts)."""
        return self.status not in (NO_FAILURE_STATUS, INFRASTRUCTURE_STATUS)

    @property
    def held_out_passed(self) -> bool:
        return self.held_out_evaluated and not self.held_out_failures

    @property
    def correct_repair(self) -> bool:
        return self.admitted and self.held_out_passed and self.template_repairable

    @property
    def false_admission(self) -> bool:
        return self.admitted and (self.template_unsupported or not self.held_out_passed)

    @property
    def correct_unsupported(self) -> bool:
        return self.template_unsupported and self.declared_unsupported

    def to_json(self) -> dict:
        return {
            "backend": self.backend,
            "template_id": self.template_id,
            "group": self.group,
            "template_label": self.template_label,
            "template_unsupported": self.template_unsupported,
            "template_repairable": self.template_repairable,
            "episode_index": self.episode_index,
            "proposal_seed": self.proposal_seed,
            "proposal_context_id": self.proposal_context_id,
            "status": self.status,
            "proposal_class": self.proposal_class,
            "declared_unsupported": self.declared_unsupported,
            "admitted": self.admitted,
            "admission_evidence": self.admission_evidence,
            "base_version": self.base_version,
            "admitted_version": self.admitted_version,
            "held_out_evaluated": self.held_out_evaluated,
            "held_out_failures": list(self.held_out_failures),
            "correct_repair": self.correct_repair,
            "false_admission": self.false_admission,
            "budget": dict(self.budget),
            "events": list(self.events),
        }

    @classmethod
    def from_json(cls, record: dict) -> "EpisodeResult":
        """The inverse of :meth:`to_json` (derived rate fields are simply
        recomputed), so persisted episode records are as good as the
        in-memory episodes they came from."""
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {key: value for key, value in record.items() if key in names}
        kwargs["held_out_failures"] = tuple(kwargs.get("held_out_failures", ()))
        return cls(**kwargs)


def run_e1_episode(
    bench: ExperimentBench,
    backend: RepairBackend,
    template: FaultTemplate,
    proposal_cases: Sequence[FaultCase],
    admission_cases: Sequence[FaultCase],
    held_out_cases: Sequence[FaultCase],
    budget: Budget,
    episode_index: int,
    working_root: Path,
) -> EpisodeResult:
    """One complete episode: manifest the fault, repair under the meter,
    admit through the curator, judge on the held-out battery."""
    faulted = template.apply(bench.correct_library())
    proposal_case = proposal_cases[episode_index]
    episode_key = f"{template.identifier}/episode-{episode_index}/{backend.name}"
    proposal_directory = working_root / "proposal" / episode_key
    proposal_context_id = f"{template.identifier}/scene-{proposal_case.variation.seed}"
    result = EpisodeResult(
        backend=backend.name,
        template_id=template.identifier,
        group=template.group,
        template_label=template.ground_truth.label,
        template_unsupported=template.ground_truth.unsupported,
        template_repairable=template.ground_truth.repair is not None,
        episode_index=episode_index,
        proposal_seed=proposal_case.variation.seed,
        proposal_context_id=proposal_context_id,
    )

    initial = bench.runner(
        faulted,
        template.task,
        proposal_case.variation,
        proposal_directory / "initial",
    )
    if initial.succeeded:
        return result
    result.proposal_class = initial.failure_class_value

    curator = bench.curator_builder(
        template,
        tuple(admission_cases),
        proposal_context_id,
        working_root / "admission" / episode_key,
    )
    probe_hook, probe_listing, required_probes = _probe_hook(
        bench, template, faulted, (proposal_case,), proposal_directory / "probes"
    )
    repair_task = RepairTask(
        certificate=initial.certificate,
        library=faulted,
        grounding_factory_listing=bench.grounding_factory_listing,
        grounding_vocabulary_listing=bench.grounding_vocabulary_listing,
        validate_grounding_candidate=bench.validate_grounding_candidate,
        grounding_candidate_sink=bench.grounding_candidate_sink,
        capability_listing=bench.capability_listing,
        capability_catalog=bench.capability_catalog,
        capability_draft_listing=bench.capability_draft_listing,
        capability_draft_ids=bench.capability_draft_ids,
        index=bench.index,
        retrieval_top_k=bench.retrieval_top_k,
        check_patch=lambda patch: curator.static_review(patch, faulted),
        run_probe=probe_hook,
        probe_listing=probe_listing,
        required_probes=required_probes,
    )
    meter = BudgetMeter(budget=budget)
    try:
        outcome = backend.repair(repair_task, meter)
    except TransientInfrastructureError as error:
        # Only an explicitly classified, retry-exhausted provider outage is
        # excluded. Programming errors propagate and fail the run loudly.
        result.status = INFRASTRUCTURE_STATUS
        result.budget = meter.snapshot()
        result.events = [
            {"step": "infrastructure", "error": f"{type(error).__name__}: {error}"}
        ]
        return result
    result.status = outcome.status.value
    result.budget = meter.snapshot()
    result.events = list(outcome.events)

    if outcome.status in {
        OutcomeStatus.UNSUPPORTED_DECLARED,
        OutcomeStatus.MISSING_EXECUTION_CAPABILITY,
    }:
        result.declared_unsupported = True
    elif outcome.status is OutcomeStatus.PATCH_PROPOSED:
        store = VersionedLibraryStore(working_root / "versions" / episode_key)
        result.base_version = store.commit(
            faulted,
            metadata={
                "role": "faulted-base",
                "template_id": template.identifier,
                "backend": backend.name,
                "episode_index": episode_index,
                "proposal_context_id": proposal_context_id,
            },
        )
        report, admitted_version = curator.admit(
            outcome.patch,
            faulted,
            store,
            proposal_context_id,
            metadata={
                "role": "episode-admission",
                "template_id": template.identifier,
                "backend": backend.name,
                "episode_index": episode_index,
                "certificate_class": result.proposal_class,
            },
        )
        result.admission_evidence = report.evidence()
        result.admitted = report.admitted
        result.admitted_version = admitted_version
        if report.admitted:
            assert admitted_version is not None
            candidate = store.load(admitted_version)
            result.held_out_failures = tuple(
                bench.held_out(
                    template,
                    tuple(held_out_cases),
                    candidate,
                    working_root / "held_out" / episode_key,
                )
            )
            result.held_out_evaluated = True
    return result


def _probe_hook(
    bench: ExperimentBench,
    template: FaultTemplate,
    faulted: SymbolLibrary,
    proposal_cases: Sequence[FaultCase],
    working_directory: Path,
):
    """Backend diagnostic probes: the repair task on proposal-split scenes
    only — the mandatory suite and held-out scenes are not reachable."""
    counter = {"n": 0}
    by_name = {
        f"repair-task-on-proposal-scene-{case.variation.seed}": ("task", case)
        for case in proposal_cases
    }
    if bench.safety_runner is not None:
        by_name.update(
            {
                f"safety-refusal-on-proposal-scene-{case.variation.seed}": (
                    "safety",
                    case,
                )
                for case in proposal_cases
            }
        )
    listing = "\n".join(f"- {name}" for name in by_name)

    def run_probe(probe: str, patch: ModelPatch) -> dict:
        selected = by_name.get(probe)
        if selected is None:
            return {
                "probe": probe,
                "succeeded": False,
                "error": "unknown probe; choose an exact name from the probe menu",
            }
        kind, case = selected
        counter["n"] += 1
        try:
            runner = bench.runner if kind == "task" else bench.safety_runner
            outcome = runner(
                patch.apply_to(faulted),
                template.task,
                case.variation,
                working_directory / f"probe-{counter['n']}",
            )
        except Exception as error:  # noqa: BLE001 -- the probed patch is
            # arbitrary model output; a crash is an observation for the
            # backend, not an abort of the episode.
            return {
                "probe": probe,
                "scene_seed": case.variation.seed,
                "succeeded": False,
                "error": f"probe crashed ({type(error).__name__}): {error}",
            }
        return {
            "probe": probe,
            "kind": kind,
            "scene_seed": case.variation.seed,
            "succeeded": (
                outcome.succeeded if kind == "task" else not outcome.succeeded
            ),
            "candidate_claimed_goal": outcome.succeeded,
            "failure_class": outcome.failure_class_value,
        }

    return run_probe, listing, tuple(by_name)


def run_e1(
    bench: ExperimentBench,
    backends: Mapping[str, RepairBackend],
    templates: Sequence[FaultTemplate],
    cases: Sequence[FaultCase],
    working_root: Path,
    budget: Budget = Budget(),
    episodes_per_template: int = 1,
    include_reference_baselines: bool = True,
    recorder=None,
    transcript=None,
) -> "E1Report":
    """The full E1 grid: backends × templates × episodes, every episode
    under a fresh meter with the identical budget.

    ``transcript`` is the shared :class:`TranscriptRecorder`; when given,
    every LLM call is stamped with the episode that made it (backend,
    template, group, proposal context) — the raw call log is useless for
    attribution otherwise, because all one-shot backends speak through
    the same agent."""
    grouped = _cases_by_template_and_split(cases, templates)
    if episodes_per_template < 1:
        raise ValueError("episodes_per_template must be positive")
    for template in templates:
        proposal_count = len(grouped[template.identifier][PROPOSAL])
        if episodes_per_template > proposal_count:
            raise ValueError(
                f"Template '{template.identifier}' has only {proposal_count} "
                f"independent proposal contexts, but {episodes_per_template} "
                "episodes were requested. Recalibrate with more variations "
                "instead of reusing a proposal context."
            )
    episodes: list[EpisodeResult] = []
    for template in templates:
        splits = grouped[template.identifier]
        roster: dict[str, RepairBackend] = dict(backends)
        if include_reference_baselines:
            roster[NoRepairBackend.name] = NoRepairBackend()
            roster[OracleReferenceBackend.name] = OracleReferenceBackend(
                template.ground_truth
            )
        for episode_index in range(episodes_per_template):
            for backend in roster.values():
                proposal_seed = splits[PROPOSAL][episode_index].variation.seed
                scope = (
                    transcript.scoped(
                        backend=backend.name,
                        template_id=template.identifier,
                        group=template.group,
                        episode_index=episode_index,
                        proposal_context_id=(
                            f"{template.identifier}/scene-{proposal_seed}"
                        ),
                    )
                    if transcript is not None
                    else nullcontext()
                )
                with scope:
                    episode = run_e1_episode(
                        bench,
                        backend,
                        template,
                        splits[PROPOSAL],
                        splits[ADMISSION],
                        splits[HELD_OUT],
                        budget,
                        episode_index,
                        working_root,
                    )
                episodes.append(episode)
                if recorder is not None:
                    recorder.record_episode(episode.to_json())
    return E1Report(episodes=tuple(episodes))


def e1_report_from_records(records: Sequence[dict]) -> "E1Report":
    """Rebuild an :class:`E1Report` from persisted episode records
    (``D_experiment/episodes.jsonl``). This is what lets a long grid be
    sharded across container invocations — or a crashed run resumed —
    without losing episodes that already spent real API budget."""
    return E1Report(episodes=tuple(EpisodeResult.from_json(r) for r in records))


def e2_report_from_records(records: Sequence[dict]) -> "E2Report":
    """Rebuild an :class:`E2Report` from persisted decision records
    (``e2_decisions.jsonl``), the E2 counterpart of
    :func:`e1_report_from_records`."""
    return E2Report(decisions=tuple(E2Case.from_json(r) for r in records))


def _cases_by_template_and_split(
    cases: Sequence[FaultCase], templates: Sequence[FaultTemplate]
) -> dict[str, dict[str, tuple[FaultCase, ...]]]:
    grouped: dict[str, dict[str, list[FaultCase]]] = {}
    for case in cases:
        grouped.setdefault(case.template_id, {}).setdefault(case.split, []).append(case)
    result: dict[str, dict[str, tuple[FaultCase, ...]]] = {}
    for template in templates:
        splits = grouped.get(template.identifier, {})
        for split in (PROPOSAL, ADMISSION, HELD_OUT):
            if not splits.get(split):
                raise ValueError(
                    f"Template '{template.identifier}' has no {split} cases."
                )
        result[template.identifier] = {
            split: tuple(splits[split]) for split in (PROPOSAL, ADMISSION, HELD_OUT)
        }
    return result


@dataclass(frozen=True)
class E1Report:
    """All E1 episodes plus the pre-registered aggregations."""

    episodes: tuple[EpisodeResult, ...]

    def backends(self) -> tuple[str, ...]:
        seen = dict.fromkeys(episode.backend for episode in self.episodes)
        return tuple(seen)

    def of_backend(self, backend: str) -> tuple[EpisodeResult, ...]:
        return tuple(e for e in self.episodes if e.backend == backend)

    def correct_repair_rate(self, backend: str) -> RateSummary:
        """Primary endpoint 2: over triggered episodes on repairable
        templates."""
        relevant = [
            e for e in self.of_backend(backend) if e.triggered and e.template_repairable
        ]
        return rate_summary(sum(e.correct_repair for e in relevant), len(relevant))

    def false_admission_rate(self, backend: str) -> RateSummary:
        """Primary endpoint 1: over all triggered episodes."""
        relevant = [e for e in self.of_backend(backend) if e.triggered]
        return rate_summary(sum(e.false_admission for e in relevant), len(relevant))

    def unsupported_detection_rate(self, backend: str) -> RateSummary:
        relevant = [
            e
            for e in self.of_backend(backend)
            if e.triggered and e.template_unsupported
        ]
        return rate_summary(sum(e.correct_unsupported for e in relevant), len(relevant))

    def paired_metric(
        self, metric: str, backend_a: str, backend_b: str
    ) -> tuple[list[float], list[float]]:
        """Aligned per-unit values of a metric for two backends; units are
        (template, proposal seed, episode index) triples both backends ran.

        Correct-repair comparisons use the same repairable-template
        population as :meth:`correct_repair_rate`; otherwise unsupported
        templates add artificial zero-vs-zero pairs to the interval.
        """

        def values(backend: str) -> dict[tuple[str, int, int], float]:
            out = {}
            for episode in self.of_backend(backend):
                if not _eligible_for_metric(episode, metric):
                    continue
                key = (
                    episode.template_id,
                    episode.proposal_seed,
                    episode.episode_index,
                )
                if key in out:
                    raise ValueError(
                        f"Duplicate experimental unit {key!r} for backend '{backend}'."
                    )
                out[key] = _metric_value(episode, metric)
            return out

        a_values, b_values = values(backend_a), values(backend_b)
        keys = sorted(set(a_values) & set(b_values))
        if not keys:
            raise ValueError(
                f"No shared triggered units between '{backend_a}' and '{backend_b}'."
            )
        return [a_values[k] for k in keys], [b_values[k] for k in keys]

    def compare(
        self,
        backend_a: str,
        backend_b: str,
        metric: str,
        iterations: int = 10_000,
        seed: int = 0,
    ) -> PairedBootstrap:
        a, b = self.paired_metric(metric, backend_a, backend_b)
        return paired_bootstrap(a, b, iterations=iterations, seed=seed)

    def per_template_lines(self, backend: str) -> list[str]:
        lines = []
        by_template: dict[str, list[EpisodeResult]] = {}
        for episode in self.of_backend(backend):
            by_template.setdefault(episode.template_id, []).append(episode)
        for template_id in sorted(by_template):
            group = by_template[template_id]
            triggered = [e for e in group if e.triggered]
            lines.append(
                f"  {template_id}: "
                f"repaired {sum(e.correct_repair for e in triggered)}"
                f"/{len(triggered)}, "
                f"false-admitted {sum(e.false_admission for e in triggered)}, "
                f"unsupported-declared "
                f"{sum(e.declared_unsupported for e in triggered)}"
            )
        return lines

    def render(self) -> str:
        lines = ["E1: matched-budget executable repair"]
        for backend in self.backends():
            lines.append(f"{backend}:")
            lines.append(
                "  false admission " + self.false_admission_rate(backend).render()
            )
            lines.append(
                "  correct repair " + self.correct_repair_rate(backend).render()
            )
            lost = sum(
                1 for e in self.of_backend(backend) if e.status == INFRASTRUCTURE_STATUS
            )
            if lost:
                lines.append(f"  episodes lost to infrastructure: {lost}")
            lines.extend(self.per_template_lines(backend))
        return "\n".join(lines)


_METRICS = {
    "correct_repair": lambda e: float(e.correct_repair),
    "false_admission": lambda e: float(e.false_admission),
    "candidates_used": lambda e: float(e.budget.get("candidates_used", 0)),
    "probes_used": lambda e: float(e.budget.get("probes_used", 0)),
    "estimated_tokens_used": lambda e: float(e.budget.get("estimated_tokens_used", 0)),
}


def _metric_value(episode: EpisodeResult, metric: str) -> float:
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric '{metric}'; known: {sorted(_METRICS)}.")
    return _METRICS[metric](episode)


def _eligible_for_metric(episode: EpisodeResult, metric: str) -> bool:
    if not episode.triggered:
        return False
    if metric == "correct_repair":
        return episode.template_repairable
    return True


# -- Gate P2: the pre-registered agentic decision rule ------------------

COST_METRICS = ("candidates_used", "probes_used", "estimated_tokens_used")


@dataclass(frozen=True)
class GateP2Margins:
    """Pre-registered absolute rate-difference margins (agent - baseline)."""

    repair_noninferiority: float = 0.05
    false_admission_increase: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("repair_noninferiority", self.repair_noninferiority),
            ("false_admission_increase", self.false_admission_increase),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be an absolute rate in [0, 1]")


@dataclass(frozen=True)
class GateP2Decision:
    """The §9.4 pre-registered rule, decided from the E1 record alone.

    ``agentic`` stays in the title and contributions only if the agent
    (a) improves correct repair while the false-admission confidence bound
    stays inside its pre-registered tolerance, or (b) establishes
    noninferiority inside the pre-registered repair margin while reducing at
    least one cost dimension. Otherwise the claim downgrades to
    failure-driven executable curation.
    """

    keep_agentic: bool
    reason: str
    repair: PairedBootstrap
    false_admission: PairedBootstrap
    costs: dict[str, PairedBootstrap]
    margins: GateP2Margins

    def render(self) -> str:
        verdict = (
            "KEEP the agentic claim"
            if self.keep_agentic
            else "DOWNGRADE to failure-driven executable curation"
        )
        lines = [
            f"Gate P2: {verdict} — {self.reason}",
            f"  correct repair: {self.repair.render()}",
            f"  false admission: {self.false_admission.render()}",
            "  margins: correct-repair degradation <= "
            f"{self.margins.repair_noninferiority:.3f}, false-admission "
            f"increase <= {self.margins.false_admission_increase:.3f}",
        ]
        for name, comparison in self.costs.items():
            lines.append(f"  {name}: {comparison.render()}")
        return "\n".join(lines)


def agentic_gate(
    report: E1Report,
    agent: str = AGENTIC_RAG_BACKEND_NAME,
    baseline: str = "fixed-pipeline",
    iterations: int = 10_000,
    seed: int = 0,
    margins: GateP2Margins = GateP2Margins(),
) -> GateP2Decision:
    repair = report.compare(agent, baseline, "correct_repair", iterations, seed)
    false_admission = report.compare(
        agent, baseline, "false_admission", iterations, seed
    )
    costs = {
        metric: report.compare(agent, baseline, metric, iterations, seed)
        for metric in COST_METRICS
    }
    false_admission_safe = false_admission.increase_bounded_by(
        margins.false_admission_increase
    )
    if repair.significantly_positive and false_admission_safe:
        return GateP2Decision(
            keep_agentic=True,
            reason=(
                "higher correct repair and the confidence bound keeps the "
                "false-admission increase within its pre-registered margin"
            ),
            repair=repair,
            false_admission=false_admission,
            costs=costs,
            margins=margins,
        )
    cheaper = [
        name for name, comparison in costs.items() if comparison.significantly_negative
    ]
    if (
        repair.noninferior(margins.repair_noninferiority)
        and false_admission_safe
        and cheaper
    ):
        return GateP2Decision(
            keep_agentic=True,
            reason=(
                "correct repair is noninferior within the pre-registered "
                f"margin and has lower cost ({', '.join(cheaper)})"
            ),
            repair=repair,
            false_admission=false_admission,
            costs=costs,
            margins=margins,
        )
    return GateP2Decision(
        keep_agentic=False,
        reason=(
            "neither pre-registered condition holds: superiority did not "
            "clear the false-admission margin, and the lower-cost path did "
            "not establish noninferiority"
        ),
        repair=repair,
        false_admission=false_admission,
        costs=costs,
        margins=margins,
    )


# -- E2: multi-context admission ----------------------------------------


@dataclass(frozen=True)
class ValidationPolicy:
    """One curation validation tier: which mandatory-suite groups run."""

    name: str
    groups: frozenset[SuiteGroup]
    single_test: bool = False
    """Tier 1 only: exactly one positive witness, no second context."""

    require_full_coverage: bool = False
    """Tier 5 only: the suite's own coverage rules (all groups present,
    an admission context distinct from the proposal context) apply."""

    def tests_of(self, suite) -> tuple:
        tests = tuple(t for t in suite.tests if t.group in self.groups)
        return tests[:1] if self.single_test else tests


POLICY_TIERS: tuple[ValidationPolicy, ...] = (
    ValidationPolicy(
        "single-witness",
        frozenset({SuiteGroup.CAPABILITY_POSITIVE}),
        single_test=True,
    ),
    ValidationPolicy(
        "multi-context-positive", frozenset({SuiteGroup.CAPABILITY_POSITIVE})
    ),
    ValidationPolicy(
        "positive-negative",
        frozenset({SuiteGroup.CAPABILITY_POSITIVE, SuiteGroup.CAPABILITY_NEGATIVE}),
    ),
    ValidationPolicy(
        "positive-negative-boundary",
        frozenset(
            {
                SuiteGroup.CAPABILITY_POSITIVE,
                SuiteGroup.CAPABILITY_NEGATIVE,
                SuiteGroup.BOUNDARY,
            }
        ),
    ),
    ValidationPolicy(
        "full-suite",
        frozenset(SuiteGroup),
        require_full_coverage=True,
    ),
)


@dataclass(frozen=True)
class LabelledCandidate:
    """One proposal with known behavioural correctness — E2 measures the
    curator, so the proposer is replaced by a labelled candidate pool."""

    name: str
    patch: ModelPatch
    behaviorally_correct: bool
    description: str


def admission_candidates(
    template: FaultTemplate, correct: SymbolLibrary
) -> tuple[LabelledCandidate, ...]:
    """The labelled pool for one template: the reference repair plus the
    plausible-but-wrong baits derived from it. Every bait passes the
    static review by construction; only behaviour separates them."""
    reference = template.ground_truth.repair
    if reference is None or not reference.operators:
        return ()
    candidates = [
        LabelledCandidate(
            name="reference",
            patch=reference,
            behaviorally_correct=True,
            description="the ground-truth repair",
        )
    ]
    unconditional = tuple(
        dataclasses.replace(
            operator,
            preconditions=tuple(
                literal
                for literal in operator.preconditions
                if literal.predicate != "ready-to-open"
            ),
        )
        for operator in reference.operators
    )
    if unconditional != reference.operators:
        candidates.append(
            LabelledCandidate(
                name="missing-reachability-precondition",
                patch=dataclasses.replace(reference, operators=unconditional),
                behaviorally_correct=False,
                description=(
                    "reachability precondition removed: "
                    "claims success on out-of-reach targets"
                ),
            )
        )
    misbound = tuple(
        dataclasses.replace(
            operator,
            execution_binding=_opposite_target_binding(operator.execution_binding),
        )
        for operator in reference.operators
    )
    if misbound != reference.operators:
        candidates.append(
            LabelledCandidate(
                name="misbound-execution-request",
                patch=dataclasses.replace(reference, operators=misbound),
                behaviorally_correct=False,
                description="the operator requests the opposite target state",
            )
        )
    return tuple(candidates)


@dataclass
class E2Case:
    """One (template, candidate, policy) admission decision."""

    template_id: str
    candidate: str
    behaviorally_correct: bool
    policy: str
    admitted: bool
    tests_run: int
    static_objections: int
    held_out_evaluated: bool = False
    held_out_failures: tuple[str, ...] = ()

    @property
    def false_admission(self) -> bool:
        return self.admitted and not self.behaviorally_correct

    @property
    def missed_admission(self) -> bool:
        return not self.admitted and self.behaviorally_correct

    def to_json(self) -> dict:
        return {
            "template_id": self.template_id,
            "candidate": self.candidate,
            "behaviorally_correct": self.behaviorally_correct,
            "policy": self.policy,
            "admitted": self.admitted,
            "tests_run": self.tests_run,
            "static_objections": self.static_objections,
            "held_out_evaluated": self.held_out_evaluated,
            "held_out_failures": list(self.held_out_failures),
            "false_admission": self.false_admission,
            "missed_admission": self.missed_admission,
        }

    @classmethod
    def from_json(cls, record: dict) -> "E2Case":
        names = {f.name for f in dataclasses.fields(cls)}
        kwargs = {key: value for key, value in record.items() if key in names}
        kwargs["held_out_failures"] = tuple(kwargs.get("held_out_failures", ()))
        return cls(**kwargs)


def run_e2(
    bench: ExperimentBench,
    templates: Sequence[FaultTemplate],
    cases: Sequence[FaultCase],
    working_root: Path,
    policies: Sequence[ValidationPolicy] = POLICY_TIERS,
    candidate_pool: Optional[
        Callable[[FaultTemplate, SymbolLibrary], tuple[LabelledCandidate, ...]]
    ] = None,
    recorder=None,
) -> "E2Report":
    """The E2 grid: the labelled candidate pool of every repairable
    template, decided under every validation policy; every admitted
    candidate is exposed to the same held-out battery."""
    pool = candidate_pool or admission_candidates
    grouped = _cases_by_template_and_split(cases, templates)
    decisions: list[E2Case] = []
    for template in templates:
        correct = bench.correct_library()
        candidates = pool(template, correct)
        if not candidates:
            continue
        faulted = template.apply(correct)
        splits = grouped[template.identifier]
        proposal_context_id = (
            f"{template.identifier}/scene-{splits[PROPOSAL][0].variation.seed}"
        )
        held_out_cache: dict[str, tuple[str, ...]] = {}
        for policy in policies:
            curator = bench.curator_builder(
                template,
                splits[ADMISSION],
                proposal_context_id,
                working_root / "admission" / template.identifier / policy.name,
            )
            for candidate in candidates:
                decision = _decide_under_policy(
                    template.identifier,
                    curator,
                    policy,
                    candidate,
                    faulted,
                    proposal_context_id,
                )
                if decision.admitted:
                    if candidate.name not in held_out_cache:
                        held_out_cache[candidate.name] = tuple(
                            bench.held_out(
                                template,
                                splits[HELD_OUT],
                                candidate.patch.apply_to(faulted),
                                working_root
                                / "held_out"
                                / template.identifier
                                / candidate.name,
                            )
                        )
                    decision.held_out_evaluated = True
                    decision.held_out_failures = held_out_cache[candidate.name]
                decisions.append(decision)
                if recorder is not None:
                    recorder.record_episode(decision.to_json())
    return E2Report(decisions=tuple(decisions))


def _decide_under_policy(
    template_id: str,
    curator: BehaviouralCurator,
    policy: ValidationPolicy,
    candidate: LabelledCandidate,
    faulted: SymbolLibrary,
    proposal_context_id: str,
) -> E2Case:
    """Static review always runs (it is not the ablation target); the
    behavioural tier decides which suite groups follow."""
    objections = curator.static_review(candidate.patch, faulted)
    if objections:
        return E2Case(
            template_id=template_id,
            candidate=candidate.name,
            behaviorally_correct=candidate.behaviorally_correct,
            policy=policy.name,
            admitted=False,
            tests_run=0,
            static_objections=len(objections),
        )
    coverage = (
        curator.suite.coverage_problems(proposal_context_id)
        if policy.require_full_coverage
        else []
    )
    tests = policy.tests_of(curator.suite)
    candidate_library = candidate.patch.apply_to(faulted)
    failures = []
    for test in tests:
        try:
            failures.append(test.run(candidate_library))
        except Exception as error:  # noqa: BLE001 -- same fail-closed rule
            # as the curator's full review: a candidate that crashes a
            # suite test is refused with evidence, not a batch abort.
            failures.append([f"test crashed ({type(error).__name__}): {error}"])
    admitted = not coverage and all(not failure for failure in failures)
    return E2Case(
        template_id=template_id,
        candidate=candidate.name,
        behaviorally_correct=candidate.behaviorally_correct,
        policy=policy.name,
        admitted=admitted,
        tests_run=len(tests),
        static_objections=0,
    )


@dataclass(frozen=True)
class E2Report:
    """All E2 decisions plus the per-policy aggregations (§9.5)."""

    decisions: tuple[E2Case, ...]

    def policies(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(d.policy for d in self.decisions))

    def of_policy(self, policy: str) -> tuple[E2Case, ...]:
        return tuple(d for d in self.decisions if d.policy == policy)

    def false_admission_rate(self, policy: str) -> RateSummary:
        wrong = [d for d in self.of_policy(policy) if not d.behaviorally_correct]
        return rate_summary(sum(d.false_admission for d in wrong), len(wrong))

    def missed_admission_rate(self, policy: str) -> RateSummary:
        right = [d for d in self.of_policy(policy) if d.behaviorally_correct]
        return rate_summary(sum(d.missed_admission for d in right), len(right))

    def held_out_clean_rate(self, policy: str) -> RateSummary:
        admitted = [d for d in self.of_policy(policy) if d.admitted]
        return rate_summary(
            sum(d.held_out_evaluated and not d.held_out_failures for d in admitted),
            len(admitted),
        )

    def mean_tests_run(self, policy: str) -> float:
        decisions = self.of_policy(policy)
        if not decisions:
            return 0.0
        return sum(d.tests_run for d in decisions) / len(decisions)

    def render(self) -> str:
        lines = ["E2: multi-context admission"]
        for policy in self.policies():
            lines.append(f"{policy}:")
            lines.append(
                "  false admission " + self.false_admission_rate(policy).render()
            )
            lines.append(
                "  missed admission " + self.missed_admission_rate(policy).render()
            )
            lines.append(
                "  held-out clean among admitted "
                + self.held_out_clean_rate(policy).render()
            )
            lines.append(
                f"  mean curator tests per decision {self.mean_tests_run(policy):.1f}"
            )
        return "\n".join(lines)
