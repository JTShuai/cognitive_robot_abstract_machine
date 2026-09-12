"""
The Tracy drawer-domain binding of the ICRA experiment harness.

One loaded fixed-arm scene serves every experimental case: a
:class:`~experiments.resym.icra.articulation.faults.SceneVariation` is applied by
moving the table mount (the same placement derivation as scene loading)
and resetting the drawer joints — no reload, so a full E1/E2 grid stays
tractable. Every behavioural check is a *real* closed-loop run through
:func:`~resym.repair.diagnosis.diagnose`: real grounding
(inverse kinematics, collisions), real Fast Downward planning, monitored
kinematic execution, independent goal verification.

The mandatory admission suite instantiates the four required groups as
witness probes on admission-split scenes:

- *capability-positive*: the repair task succeeds on admission scenes;
- *capability-negative*: the same goal on the farthest drawer (beyond
  the UR10e's reach by construction) must NOT be reported achieved —
  an operator without a reachability precondition "succeeds" kinematically
  there and is caught here;
- *boundary*: from a joint state just outside the goal region near the
  opened-fraction threshold, the goal must not be claimed already
  satisfied (no empty-plan success) — a wrong or too-eager truth
  binding is caught here;
- *regression*: the reverse manipulation task, when the defective
  library could still do it, must still work with the patch applied.

The held-out battery is the same four groups instantiated on held-out
scenes; it runs only after the admission decision.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path

from typing_extensions import Optional

from experiments.resym.scenes import SceneSetup, _fixed_arm_mount_pose
from experiments.resym.seed_library import (
    OPENED_FRACTION_THRESHOLD,
    build_fixed_arm_library,
)
from experiments.resym.icra.validation import (
    AdmissionTest,
    BehaviouralCurator,
    MandatorySuite,
    SuiteGroup,
)
from resym.repair.diagnosis import diagnose
from resym.platform.grounding_catalog import (
    GroundingFactoryCatalog,
    GroundingFactorySourceValidator,
)
from resym.platform.kinematic import KinematicFeasibility
from resym.platform.grounding_context import EvaluationContext
from experiments.resym.icra.harness import ExperimentBench
from experiments.resym.icra.articulation.faults import (
    FaultCase,
    FaultTask,
    FaultTemplate,
    Manifestation,
    SceneVariation,
)
from resym.core.model import Literal, SymbolLibrary
from resym.llm.prompting import render_capability_contracts
from resym.platform.coraplex_catalog import (
    compatible_coraplex_action_ids,
    render_coraplex_capability_candidates,
)
from resym.platform.capabilities import capability_contracts
from resym.planning.selection import select_for_goal
from resym.platform.kinematic import (
    KinematicSkillRealization,
    OPEN_TARGET_FRACTION,
)
from resym.platform.articulation import (
    articulation_connection,
    is_articulated_object,
)
from resym.platform.universe import (
    ObjectUniverse,
    pddl_name,
    set_joint_fraction,
)

OPENED_FRACTION = OPEN_TARGET_FRACTION
"""
Joint fraction an 'initially open' drawer starts at — the same fraction the kinematic
backend drives a drawer to for ``OPEN``.
"""

BOUNDARY_MARGIN = 0.05
"""
Fraction distance from the opened threshold for the boundary probe.
"""


class TracyBench:
    """
    One loaded Tracy scene, reused across all experimental cases.

    ``backend_factory`` builds a fresh execution backend per closed-loop run — the hook
    the perturbation wrappers plug into (a fresh wrapper per run keeps its action-
    counter schedule deterministic).
    """

    def __init__(
        self,
        setup: SceneSetup,
        universe: ObjectUniverse,
        grounding_catalog: GroundingFactoryCatalog,
        backend_factory=None,
    ):
        self.setup = setup
        self.universe = universe
        self.grounding_catalog = grounding_catalog
        self.backend_factory = backend_factory or KinematicSkillRealization
        # A probe may touch more than drawer joints. Keep every calibration,
        # admission, and held-out run independent of the preceding run.
        self._initial_world_state = deepcopy(setup.world.state)
        self.goal_drawer = pddl_name(setup.scene.goal_drawer_body)
        self.unreachable_drawer = self._farthest_drawer()
        self.mount = self._mount_connection()
        self.robot_name = next(
            grounded.name
            for grounded in universe.objects.values()
            if grounded.semantic_entity is self.setup.robot
        )

    # -- scene bookkeeping ------------------------------------------------

    def _mount_connection(self):
        for connection in self.setup.world.connections:
            if connection.parent.name.name == "mount":
                return connection
        raise ValueError("No 'mount' connection in the fixed-arm world.")

    def _farthest_drawer(self) -> str:
        """
        The annotated drawer farthest from the goal drawer — beyond the fixed arm's
        reach by construction, the negative-context target.
        """
        goal_position = self.universe[self.goal_drawer].body.global_pose.to_np()[:3, 3]
        drawers = [
            grounded
            for grounded in self.universe.objects.values()
            if is_articulated_object(grounded) and grounded.name != self.goal_drawer
        ]
        if not drawers:
            raise ValueError("The scene has no second drawer.")

        def distance(grounded) -> float:
            position = grounded.body.global_pose.to_np()[:3, 3]
            return float(((position - goal_position) ** 2).sum())

        return max(drawers, key=distance).name

    def target_of(self, task: FaultTask) -> str:
        return self.goal_drawer if task.on_mounted_drawer else self.unreachable_drawer

    def _set_drawer_fraction(self, drawer_name: str, fraction: float) -> None:
        set_joint_fraction(
            articulation_connection(self.universe[drawer_name]),
            fraction,
        )

    def apply_variation(self, variation, initial_fraction: float, target: str):
        """
        Restore the world, then apply one scene variation.
        """
        self.setup.world.state.merge_state(self._initial_world_state)
        # _fixed_arm_mount_pose reads handle global poses. Refresh forward
        # kinematics after restoring DOFs so it cannot observe the preceding
        # run's opened drawer pose.
        self.setup.world.notify_state_change()
        self.mount.origin = _fixed_arm_mount_pose(
            self.setup.world, self.setup.scene, variation
        )
        for grounded in self.universe.objects.values():
            if is_articulated_object(grounded):
                self._set_drawer_fraction(grounded.name, 0.0)
        noise = variation.initial_fraction_noise if variation is not None else 0.0
        self._set_drawer_fraction(target, min(1.0, max(0.0, initial_fraction + noise)))
        self.setup.world.notify_state_change()

    def _fresh_context(self) -> EvaluationContext:
        return EvaluationContext(
            world=self.setup.world,
            robot=self.setup.robot,
            profile=self.setup.profile,
            grounding_catalog=self.grounding_catalog,
            capability_feasibility=KinematicFeasibility(),
        )

    # -- the closed-loop probe --------------------------------------------

    def run(
        self,
        library: SymbolLibrary,
        goal_predicate: str,
        target: str,
        variation,
        initial_fraction: float,
        working_directory: Path,
    ):
        self.apply_variation(variation, initial_fraction, target)
        return diagnose(
            library,
            self.universe,
            self._fresh_context(),
            (Literal(goal_predicate, (target,)),),
            working_directory,
            realization=self.backend_factory(),
        )

    def runner(self, library, task: FaultTask, variation, working_directory):
        """
        The harness :data:`CaseRunner`: the template's probe task.
        """
        return self.run(
            library,
            task.goal_predicate,
            self.target_of(task),
            variation,
            OPENED_FRACTION if task.initially_open else 0.0,
            working_directory,
        )

    def safety_runner(self, library, task: FaultTask, variation, working_directory):
        """
        Try the same goal on the proposal split's out-of-reach drawer.
        """
        return self.run(
            library,
            task.goal_predicate,
            self.unreachable_drawer,
            variation,
            OPENED_FRACTION if task.initially_open else 0.0,
            working_directory,
        )

    def calibration_failures(
        self,
        template: FaultTemplate,
        variation: SceneVariation,
        working_directory: Path,
    ) -> list[str]:
        """
        Why a variation is not a valid unit for this template.

        Calibration is gold-only and runs before split labels exist: the
        correct library must exhibit the intended positive/refusal behavior,
        its reverse capability must still work, and the injected fault must
        manifest with the registered class (or remain silent when declared).
        """
        failures = []
        correct = build_fixed_arm_library(self.grounding_catalog)
        task = template.task
        correct_outcome = self.runner(
            correct, task, variation, working_directory / "correct-primary"
        )
        if task.on_mounted_drawer:
            if not correct_outcome.succeeded:
                failures.append(
                    "correct library cannot perform primary task: "
                    f"{correct_outcome.failure_class_value}"
                )
        elif correct_outcome.succeeded:
            failures.append("correct library falsely succeeds out of reach")
        elif (
            template.ground_truth.expected_certificate_classes
            and correct_outcome.failure_class_value
            not in template.ground_truth.expected_certificate_classes
        ):
            failures.append(
                "correct library refuses out-of-reach task as "
                f"{correct_outcome.failure_class_value}, expected "
                f"{template.ground_truth.expected_certificate_classes}"
            )
        if failures:
            return failures

        reverse_goal = "closed" if task.goal_predicate == "opened" else "opened"
        reverse = self.run(
            correct,
            reverse_goal,
            self.goal_drawer,
            variation,
            OPENED_FRACTION if reverse_goal == "closed" else 0.0,
            working_directory / "correct-regression",
        )
        if not reverse.succeeded:
            failures.append(
                f"correct library regression '{reverse_goal}' fails: "
                f"{reverse.failure_class_value}"
            )
            return failures

        faulted = self.runner(
            template.apply(correct),
            task,
            variation,
            working_directory / "fault-manifestation",
        )
        if template.manifestation is Manifestation.SILENT:
            if not faulted.succeeded:
                failures.append(
                    "fault should be silent but failed as "
                    f"{faulted.failure_class_value}"
                )
        elif faulted.succeeded:
            failures.append("injected fault did not manifest")
        elif (
            template.ground_truth.expected_certificate_classes
            and faulted.failure_class_value
            not in template.ground_truth.expected_certificate_classes
        ):
            failures.append(
                f"fault manifested as {faulted.failure_class_value}, expected "
                f"{template.ground_truth.expected_certificate_classes}"
            )
        return failures

    # -- the behavioural groups -------------------------------------------

    def _behavioural_tests(
        self,
        template: FaultTemplate,
        cases: tuple[FaultCase, ...],
        working_directory: Path,
    ) -> tuple[AdmissionTest, ...]:
        """
        The four mandatory groups over the given (admission or held-out) scene cases.
        """
        task = template.task
        goal = task.goal_predicate
        threshold = OPENED_FRACTION_THRESHOLD
        boundary_fraction = (
            threshold - BOUNDARY_MARGIN
            if goal == "opened"
            else threshold + BOUNDARY_MARGIN
        )
        reverse_goal = "closed" if goal == "opened" else "opened"
        faulted = template.apply(build_fixed_arm_library(self.grounding_catalog))
        reverse_worked_before = _has_achiever(faulted, reverse_goal)
        # Templates whose probe task targets the out-of-reach drawer
        # (on_mounted_drawer=False) have honest refusal as the CORRECT
        # post-repair behavior: the capability-positive assertion is "no
        # false success, refused with one of the template's expected
        # certificate classes", not "task succeeds".
        expects_refusal = not task.on_mounted_drawer
        expected_refusals = tuple(
            template.ground_truth.expected_certificate_classes or ()
        )

        def context_id(case: FaultCase) -> str:
            return f"{template.identifier}/scene-{case.variation.seed}"

        def positive(case: FaultCase, index: int):
            def run(candidate: SymbolLibrary) -> list[str]:
                outcome = self.run(
                    candidate,
                    goal,
                    self.target_of(task),
                    case.variation,
                    OPENED_FRACTION if task.initially_open else 0.0,
                    working_directory / f"positive-{index}",
                )
                if expects_refusal:
                    if outcome.succeeded:
                        return [
                            f"claimed '{goal}' achieved on the out-of-reach "
                            f"target on scene {case.variation.seed}: the "
                            "correct repair must refuse honestly"
                        ]
                    if (
                        expected_refusals
                        and outcome.failure_class_value not in expected_refusals
                    ):
                        return [
                            f"refused with '{outcome.failure_class_value}' "
                            f"instead of one of {list(expected_refusals)} on "
                            f"scene {case.variation.seed}"
                        ]
                    return []
                if outcome.succeeded:
                    return []
                return [
                    f"repair task failed on scene {case.variation.seed}: "
                    f"{outcome.failure_class_value}"
                ]

            return run

        def negative(case: FaultCase):
            def run(candidate: SymbolLibrary) -> list[str]:
                outcome = self.run(
                    candidate,
                    goal,
                    self.unreachable_drawer,
                    case.variation,
                    OPENED_FRACTION if goal == "closed" else 0.0,
                    working_directory / "negative",
                )
                if outcome.succeeded:
                    return [
                        f"claimed '{goal}' achieved on the out-of-reach "
                        f"drawer '{self.unreachable_drawer}'"
                    ]
                return []

            return run

        def boundary(case: FaultCase):
            def run(candidate: SymbolLibrary) -> list[str]:
                outcome = self.run(
                    candidate,
                    goal,
                    self.target_of(task),
                    case.variation,
                    boundary_fraction,
                    working_directory / "boundary",
                )
                if outcome.succeeded and not outcome.result.plan:
                    return [
                        f"claimed '{goal}' already satisfied at joint "
                        f"fraction {boundary_fraction} (threshold "
                        f"{threshold}): the boundary state was projected "
                        "into the goal region without acting"
                    ]
                return []

            return run

        def regression(case: FaultCase):
            def run(candidate: SymbolLibrary) -> list[str]:
                if not reverse_worked_before:
                    return []  # nothing previously worked to regress
                outcome = self.run(
                    candidate,
                    reverse_goal,
                    self.goal_drawer,
                    case.variation,
                    OPENED_FRACTION if reverse_goal == "closed" else 0.0,
                    working_directory / "regression",
                )
                if outcome.succeeded:
                    return []
                return [
                    f"previously working '{reverse_goal}' task now fails: "
                    f"{outcome.failure_class_value}"
                ]

            return run

        positive_cases = cases[:2]
        positive_name = (
            "repair-task-honestly-refuses"
            if expects_refusal
            else "repair-task-succeeds"
        )
        tests = [
            AdmissionTest(
                name=f"{positive_name}-{index}",
                group=SuiteGroup.CAPABILITY_POSITIVE,
                context_id=context_id(case),
                run=positive(case, index),
            )
            for index, case in enumerate(positive_cases)
        ]
        tests.append(
            AdmissionTest(
                name="refuses-out-of-reach-drawer",
                group=SuiteGroup.CAPABILITY_NEGATIVE,
                context_id=context_id(cases[-1]),
                run=negative(cases[-1]),
            )
        )
        tests.append(
            AdmissionTest(
                name="no-claim-at-threshold-boundary",
                group=SuiteGroup.BOUNDARY,
                context_id=context_id(cases[0]),
                run=boundary(cases[0]),
            )
        )
        tests.append(
            AdmissionTest(
                name="reverse-task-still-works",
                group=SuiteGroup.REGRESSION,
                context_id=context_id(cases[0]),
                run=regression(cases[0]),
            )
        )
        if expects_refusal and _has_achiever(faulted, goal):
            # A refusal-type probe task never exercises the skill on a
            # reachable target, so a candidate with a misbound skill (or
            # any damage to the reachable capability) would sail through
            # every group above: the honest refusal happens before the
            # skill could betray itself. The faulted library could still
            # achieve the goal on the mounted drawer, and the repair must
            # not lose that.
            def mounted_regression(case: FaultCase):
                def run(candidate: SymbolLibrary) -> list[str]:
                    outcome = self.run(
                        candidate,
                        goal,
                        self.goal_drawer,
                        case.variation,
                        OPENED_FRACTION if goal == "closed" else 0.0,
                        working_directory / "mounted-regression",
                    )
                    if outcome.succeeded:
                        return []
                    return [
                        f"previously working '{goal}' on the mounted drawer "
                        f"now fails: {outcome.failure_class_value}"
                    ]

                return run

            tests.append(
                AdmissionTest(
                    name="mounted-target-still-works",
                    group=SuiteGroup.REGRESSION,
                    context_id=context_id(cases[-1]),
                    run=mounted_regression(cases[-1]),
                )
            )
        return tuple(tests)

    def curator_builder(
        self,
        template: FaultTemplate,
        admission_cases: tuple[FaultCase, ...],
        proposal_context_id: str,
        working_directory: Path,
    ) -> BehaviouralCurator:
        return BehaviouralCurator(
            available_capabilities=self.setup.profile.capabilities,
            capability_catalog=capability_contracts(),
            grounding_factory_specs={
                specification.uid: specification
                for specification in self.grounding_catalog
            },
            suite=MandatorySuite(
                version=f"tracy-suite-1/{template.identifier}",
                tests=self._behavioural_tests(
                    template, admission_cases, working_directory
                ),
            ),
            allowed_symbol_types=frozenset(
                item.symbol_type for item in self.universe.objects.values()
            ),
        )

    def held_out(
        self,
        template: FaultTemplate,
        held_out_cases: tuple[FaultCase, ...],
        candidate: SymbolLibrary,
        working_directory: Path,
    ) -> list[str]:
        """
        The gold evaluation: the full behavioural battery on scenes neither the proposer
        nor the curator ever touched.
        """
        failures: list[str] = []
        for test in self._behavioural_tests(
            template, held_out_cases, working_directory
        ):
            failures.extend(
                f"[{test.group.value}/{test.context_id}] {failure}"
                for failure in test.run(candidate)
            )
        return failures


def build_tracy_bench(
    setup: SceneSetup,
    universe: ObjectUniverse,
    grounding_catalog: GroundingFactoryCatalog,
    index: Optional[object] = None,
    backend_factory=None,
) -> ExperimentBench:
    """
    The harness bench over one loaded Tracy scene.
    """
    bench = TracyBench(
        setup, universe, grounding_catalog, backend_factory=backend_factory
    )
    profile = setup.profile
    workspace = grounding_catalog.workspace
    vocabulary = workspace.reviewed_vocabulary() if workspace is not None else None
    validator = (
        GroundingFactorySourceValidator(vocabulary) if vocabulary is not None else None
    )
    return ExperimentBench(
        correct_library=partial(build_fixed_arm_library, grounding_catalog),
        runner=bench.runner,
        curator_builder=bench.curator_builder,
        held_out=bench.held_out,
        safety_runner=bench.safety_runner,
        grounding_factory_listing=grounding_catalog.render(),
        grounding_vocabulary_listing=(
            vocabulary.render() if vocabulary is not None else ""
        ),
        validate_grounding_candidate=(
            validator.candidate_objections if validator is not None else None
        ),
        grounding_candidate_sink=(workspace.submit if workspace is not None else None),
        capability_listing=render_capability_contracts(
            build_fixed_arm_library(grounding_catalog), set(profile.capabilities)
        ),
        capability_catalog=capability_contracts(),
        capability_draft_listing=render_coraplex_capability_candidates(profile),
        capability_draft_ids=compatible_coraplex_action_ids(profile),
        index=index,
    )


def _has_achiever(library: SymbolLibrary, goal_predicate: str) -> bool:
    if goal_predicate not in library.predicates:
        return False
    try:
        selection = select_for_goal(library, (Literal(goal_predicate, ("_probe",)),))
    except Exception:
        return False
    return any(
        any(effect.predicate == goal_predicate for effect in operator.add_effects)
        for operator in selection.operators.values()
    )
