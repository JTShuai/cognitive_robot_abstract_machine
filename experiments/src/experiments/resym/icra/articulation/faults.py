"""Drawer-domain controlled faults for the ICRA evaluation.

The P2 experiments need repair tasks whose correct answer is *known*. A
:class:`FaultTemplate` is a pure transformation of the correct fixed-arm
library into a defective variant, together with the task that exposes
the defect, how the defect is expected to manifest in the closed loop,
and the ground truth — either the reference :class:`ModelPatch` that
repairs it, or the verdict that the capability is unsupported and the
only correct move is to declare so.

Expected certificate classes are recorded as *strings* (the
``FailureClass`` values) so this module stays free of the CRAM import
chain and host-testable; some templates deliberately manifest under a
different class than their true fault kind — the §5.3 diagnosis
evaluation measures exactly this confusion, so it is data, not a bug.

Scene randomization here covers what the kinematic world can vary
deterministically (mount placement jitter, initial drawer state, target
drawer); perception-noise and partial-action-success perturbations are
execution-backend wrappers and live with the experiment harness (W2.2).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from typing_extensions import Callable, Optional

from resym.repair.patch import ModelPatch
from resym.core.model import (
    BindingSource,
    CapabilityRef,
    Literal,
    Operator,
    OperatorExecutionBinding,
    PredicateSymbol,
    RoleBinding,
    SymbolLibrary,
    SymbolType,
)
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


class Manifestation(Enum):
    """Where in the closed loop a fault is expected to surface."""

    STATIC_REFUSAL = "static-refusal"
    """Refused by the embodiment capability gate before grounding."""

    PLANNING_FAILURE = "planning-failure"
    """The planner proves the compiled task unsolvable."""

    EXECUTION_FAILURE = "execution-failure"
    """Platform execution or the goal check refuses at run time."""

    SILENT = "silent"
    """The probe task succeeds despite the defect — a latent fault that
    only regression or contract review can catch."""


@dataclass(frozen=True)
class FaultTask:
    """The probe task that exposes (or fails to expose) the fault."""

    goal_predicate: str
    """Goal literal predicate over the target drawer."""

    on_mounted_drawer: bool = True
    """Whether the goal targets the drawer the arm was mounted for; a goal
    on any other drawer is out of reach by construction."""

    initially_open: bool = False
    """Whether the target drawer starts opened instead of closed."""


@dataclass(frozen=True)
class GroundTruth:
    """What is actually wrong, and what the correct answer is."""

    label: str
    """The true fault kind (this module's taxonomy, e.g. 'missing-operator',
    'wrong-effect') — what a perfect diagnosis would name."""

    expected_certificate_classes: tuple[str, ...]
    """``FailureClass`` values the pipeline is expected to emit for the
    probe task, most likely first; empty for ``SILENT`` faults."""

    repair: Optional[ModelPatch]
    """The reference repair; ``None`` when no repair exists."""

    unsupported: bool = False
    """The correct answer is to declare the capability unsupported."""

    curation_expected: bool = False
    """Whether the expected primary class triggers model curation."""


@dataclass(frozen=True)
class FaultTemplate:
    """One reproducible defect with known ground truth."""

    identifier: str
    group: str
    """Difficulty group: D1 missing model elements, D2 wrong model
    content, D3 combined defects and platform mismatches."""

    description: str
    task: FaultTask
    manifestation: Manifestation
    ground_truth: GroundTruth
    transform: Callable[[SymbolLibrary], SymbolLibrary]
    """Pure: returns a defective deep copy, never mutates the input."""

    def apply(self, correct: SymbolLibrary) -> SymbolLibrary:
        faulted = self.transform(copy.deepcopy(correct))
        return faulted


class UnknownTemplateSymbolError(Exception):
    """Raised when a template references a symbol the correct library does
    not contain — the template set and the library drifted apart."""

    def __init__(self, template: str, symbol: str):
        super().__init__(
            f"Fault template '{template}' references '{symbol}', which the "
            "correct library does not contain."
        )


def drawer_fault_templates(correct: SymbolLibrary) -> tuple[FaultTemplate, ...]:
    """The drawer-domain template set over the correct fixed-arm library.

    Ground-truth patches are built from the correct library itself, so a
    reference repair is by construction the removed or corrupted symbol.
    Containment-domain templates (delete place operator, missing
    gripper-state precondition, ...) follow once the containment scene
    and library exist.
    """
    for name in ("open-drawer", "close-drawer"):
        if name not in correct.operators:
            raise UnknownTemplateSymbolError("drawer set", name)
    for name in ("opened", "closed", "ready-to-open", "handle-of"):
        if name not in correct.predicates:
            raise UnknownTemplateSymbolError("drawer set", name)

    open_drawer = correct.operators["open-drawer"]
    close_drawer = correct.operators["close-drawer"]
    opened = correct.predicates["opened"]
    closed = correct.predicates["closed"]

    def without_operator(name: str):
        def transform(library: SymbolLibrary) -> SymbolLibrary:
            del library.operators[name]
            return library

        return transform

    def without_predicate(name: str):
        """Remove a predicate and every operator that mentions it — a
        library that never learned the concept has no operators over it."""

        def transform(library: SymbolLibrary) -> SymbolLibrary:
            del library.predicates[name]
            library.operators = {
                operator_name: operator
                for operator_name, operator in library.operators.items()
                if all(
                    literal.predicate != name
                    for literal in operator.preconditions
                    + operator.add_effects
                    + operator.delete_effects
                )
            }
            return library

        return transform

    def replace_operator(name: str, **changes):
        def transform(library: SymbolLibrary) -> SymbolLibrary:
            library.operators[name] = dataclasses.replace(
                library.operators[name], **changes
            )
            return library

        return transform

    def replace_predicate(name: str, **changes):
        def transform(library: SymbolLibrary) -> SymbolLibrary:
            library.predicates[name] = dataclasses.replace(
                library.predicates[name], **changes
            )
            return library

        return transform

    def compose(*transforms):
        def transform(library: SymbolLibrary) -> SymbolLibrary:
            for step in transforms:
                library = step(library)
            return library

        return transform

    def repair(**kwargs) -> ModelPatch:
        return ModelPatch(rationale="ground-truth reference repair", **kwargs)

    templates = [
        # -- D1: missing model elements -------------------------------
        FaultTemplate(
            identifier="missing-close-operator",
            group="D1",
            description="the library never learned how to close a drawer",
            task=FaultTask(goal_predicate="closed", initially_open=True),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="missing-operator",
                expected_certificate_classes=("missing_operator_model",),
                repair=repair(operators=(close_drawer,)),
                curation_expected=True,
            ),
            transform=without_operator("close-drawer"),
        ),
        FaultTemplate(
            identifier="missing-open-operator",
            group="D1",
            description="the library never learned how to open a drawer",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="missing-operator",
                expected_certificate_classes=("missing_operator_model",),
                repair=repair(operators=(open_drawer,)),
                curation_expected=True,
            ),
            transform=without_operator("open-drawer"),
        ),
        FaultTemplate(
            identifier="missing-opened-predicate",
            group="D1",
            description="the joint-state concept 'opened' and everything "
            "built on it are absent",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="missing-predicate",
                expected_certificate_classes=("missing_predicate_model",),
                repair=repair(predicates=(opened,), operators=(open_drawer,)),
                curation_expected=True,
            ),
            transform=without_predicate("opened"),
        ),
        FaultTemplate(
            identifier="missing-closed-predicate",
            group="D1",
            description="the joint-state concept 'closed' and everything "
            "built on it are absent",
            task=FaultTask(goal_predicate="closed", initially_open=True),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="missing-predicate",
                expected_certificate_classes=("missing_predicate_model",),
                repair=repair(predicates=(closed,), operators=(close_drawer,)),
                curation_expected=True,
            ),
            transform=without_predicate("closed"),
        ),
        # -- D2: wrong model content ----------------------------------
        FaultTemplate(
            identifier="inverted-precondition",
            group="D2",
            description="open-drawer demands the drawer be already opened",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="wrong-precondition",
                expected_certificate_classes=("operator_precondition_error",),
                repair=repair(operators=(open_drawer,)),
                curation_expected=True,
            ),
            transform=replace_operator(
                "open-drawer",
                preconditions=tuple(
                    (
                        Literal("opened", literal.arguments)
                        if literal.predicate == "closed"
                        else literal
                    )
                    for literal in open_drawer.preconditions
                ),
            ),
        ),
        FaultTemplate(
            identifier="wrong-add-effect",
            group="D2",
            description="open-drawer claims to achieve 'closed' instead of 'opened'",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="wrong-effect",
                expected_certificate_classes=("operator_effect_error",),
                repair=repair(operators=(open_drawer,)),
                curation_expected=True,
            ),
            transform=replace_operator(
                "open-drawer",
                add_effects=(Literal("closed", ("d",)),),
                delete_effects=(),
            ),
        ),
        FaultTemplate(
            identifier="missing-delete-effect",
            group="D2",
            description="open-drawer forgets that opening un-closes the "
            "drawer; the probe task still succeeds",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.SILENT,
            ground_truth=GroundTruth(
                label="wrong-effect",
                expected_certificate_classes=(),
                repair=repair(operators=(open_drawer,)),
            ),
            transform=replace_operator("open-drawer", delete_effects=()),
        ),
        FaultTemplate(
            identifier="execution-binding-mismatch",
            group="D2",
            description="open-drawer requests the CLOSED target state",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.EXECUTION_FAILURE,
            ground_truth=GroundTruth(
                label="wrong-execution-binding",
                expected_certificate_classes=("postcondition_failure",),
                repair=repair(operators=(open_drawer,)),
            ),
            transform=replace_operator(
                "open-drawer",
                execution_binding=_opposite_target_binding(
                    open_drawer.execution_binding
                ),
            ),
        ),
        FaultTemplate(
            identifier="missing-reachability-precondition",
            group="D2",
            description="open-drawer no longer requires reachability and "
            "falsely claims success on an out-of-reach drawer",
            task=FaultTask(goal_predicate="opened", on_mounted_drawer=False),
            manifestation=Manifestation.SILENT,
            ground_truth=GroundTruth(
                label="missing-precondition",
                expected_certificate_classes=(),
                repair=repair(operators=(open_drawer,)),
            ),
            transform=replace_operator(
                "open-drawer",
                preconditions=tuple(
                    literal
                    for literal in open_drawer.preconditions
                    if literal.predicate != "ready-to-open"
                ),
            ),
        ),
        FaultTemplate(
            identifier="wrong-parameter-type",
            group="D2",
            description="open-drawer types its handle parameter as a drawer",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="wrong-signature",
                expected_certificate_classes=("operator_signature_error",),
                repair=repair(operators=(open_drawer,)),
                curation_expected=True,
            ),
            transform=replace_operator(
                "open-drawer",
                parameters=tuple(
                    (variable, DRAWER_TYPE if variable == "h" else symbol_type)
                    for variable, symbol_type in open_drawer.parameters
                ),
            ),
        ),
        FaultTemplate(
            identifier="contract-effect-conflict",
            group="D2",
            description="the articulation capability contract no longer covers "
            "'opened', so close-drawer's delete effect is unverifiable; "
            "an already-satisfied probe task does not expose the defect",
            task=FaultTask(goal_predicate="opened", initially_open=True),
            manifestation=Manifestation.SILENT,
            ground_truth=GroundTruth(
                label="contract-violation",
                expected_certificate_classes=(),
                # A repair model cannot authorize a wider actuator contract.
                # This is a latent trusted-configuration defect; restoration
                # requires a separately reviewed implementation release.
                repair=None,
            ),
            transform=_replace_contract(
                ARTICULATION_CAPABILITY_UID, verifiable_effects=("closed",)
            ),
        ),
        # -- D3: combined defects and platform mismatches -------------
        FaultTemplate(
            identifier="unsupported-navigation-library",
            group="D3",
            description="the library models drawer opening the mobile way "
            "(navigate to a sampled witness pose first); this embodiment "
            "cannot reposition its base",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.STATIC_REFUSAL,
            ground_truth=GroundTruth(
                label="unsupported-capability",
                expected_certificate_classes=("unsupported_capability",),
                repair=None,
                unsupported=True,
            ),
            transform=_add_navigation_model,
        ),
        FaultTemplate(
            identifier="broken-evaluator-binding",
            group="D3",
            description="'opened' references an evaluator that no embodiment registers",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.STATIC_REFUSAL,
            ground_truth=GroundTruth(
                label="broken-evaluator-binding",
                expected_certificate_classes=("predicate_implementation_error",),
                repair=repair(predicates=(opened,)),
                curation_expected=True,
            ),
            transform=replace_predicate("opened", evaluator="drawer_opened_v2"),
        ),
        FaultTemplate(
            identifier="combined-missing-close-and-delete-effect",
            group="D3",
            description="no close operator, and open-drawer additionally "
            "forgets its delete effect",
            task=FaultTask(goal_predicate="closed", initially_open=True),
            manifestation=Manifestation.PLANNING_FAILURE,
            ground_truth=GroundTruth(
                label="missing-operator",
                expected_certificate_classes=("missing_operator_model",),
                repair=repair(operators=(close_drawer, open_drawer)),
                curation_expected=True,
            ),
            transform=compose(
                without_operator("close-drawer"),
                replace_operator("open-drawer", delete_effects=()),
            ),
        ),
        FaultTemplate(
            identifier="combined-misbinding-and-missing-close",
            group="D3",
            description="open-drawer requests CLOSED and the "
            "close operator is missing entirely",
            task=FaultTask(goal_predicate="opened"),
            manifestation=Manifestation.EXECUTION_FAILURE,
            ground_truth=GroundTruth(
                label="wrong-execution-binding",
                expected_certificate_classes=("postcondition_failure",),
                repair=repair(operators=(open_drawer, close_drawer)),
            ),
            transform=compose(
                without_operator("close-drawer"),
                replace_operator(
                    "open-drawer",
                    execution_binding=_opposite_target_binding(
                        open_drawer.execution_binding
                    ),
                ),
            ),
        ),
    ]
    identifiers = [template.identifier for template in templates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate fault template identifiers.")
    return tuple(templates)


def _replace_contract(capability_uid: str, **changes):
    def transform(library: SymbolLibrary) -> SymbolLibrary:
        current = library.capability_contracts[capability_uid]
        replacements = dict(changes)
        if (
            "verifiable_effects" in replacements
            and "effect_role_values" not in replacements
        ):
            allowed = set(replacements["verifiable_effects"])
            replacements["effect_role_values"] = tuple(
                item for item in current.effect_role_values if item[0] in allowed
            )
        library.capability_contracts[capability_uid] = dataclasses.replace(
            current, **replacements
        )
        return library

    return transform


def _add_navigation_model(library: SymbolLibrary) -> SymbolLibrary:
    """Turn the library into its mobile-embodiment shape: reachability is
    achieved by navigating to a sampled witness pose."""
    library.add(
        PredicateSymbol(
            name="openable",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            evaluator="openable",
            fluent=False,
        )
    )
    library.add(
        Operator(
            name="navigate",
            parameters=(("r", ROBOT_TYPE), ("d", DRAWER_TYPE)),
            preconditions=(Literal("openable", ("r", "d")),),
            add_effects=(Literal("ready-to-open", ("r", "d")),),
            delete_effects=(),
            execution_binding=OperatorExecutionBinding(
                CapabilityRef(NAVIGATION_CAPABILITY_UID),
                (
                    ("actor", RoleBinding.parameter("r")),
                    ("patient", RoleBinding.parameter("d")),
                ),
            ),
        )
    )
    return library


def _opposite_target_binding(
    binding: OperatorExecutionBinding,
) -> OperatorExecutionBinding:
    role_bindings = []
    for role, role_binding in binding.role_bindings:
        if role == "target_state" and role_binding.source is BindingSource.CONSTANT:
            opposite = {"OPEN": "CLOSED", "CLOSED": "OPEN"}.get(
                role_binding.value, role_binding.value
            )
            role_binding = RoleBinding.constant(opposite)
        role_bindings.append((role, role_binding))
    return dataclasses.replace(binding, role_bindings=tuple(role_bindings))


# -- scene randomization and splits -----------------------------------


@dataclass(frozen=True)
class SceneVariation:
    """Deterministic per-case variation of the kinematic scene."""

    seed: int
    front_offset_delta: float = 0.0
    """Jitter (meters) on the table's front standoff."""

    side_offset_delta: float = 0.0
    """Jitter (meters) on the table's side standoff."""

    initial_fraction_noise: float = 0.0
    """Additive noise on the initial drawer fraction, clipped to [0, 1]."""

    @classmethod
    def from_seed(cls, seed: int) -> SceneVariation:
        generator = random.Random(seed)
        return cls(
            seed=seed,
            front_offset_delta=generator.uniform(-0.05, 0.05),
            side_offset_delta=generator.uniform(-0.05, 0.05),
            initial_fraction_noise=generator.uniform(0.0, 0.05),
        )


PROPOSAL, ADMISSION, HELD_OUT = "proposal", "admission", "held-out"

SPLITS = (PROPOSAL, ADMISSION, HELD_OUT)


@dataclass(frozen=True)
class FaultCase:
    """One concrete experimental unit: a template on a varied scene."""

    template_id: str
    variation: SceneVariation
    split: str

    def to_json(self) -> dict:
        return {
            "template_id": self.template_id,
            "split": self.split,
            "variation": dataclasses.asdict(self.variation),
        }

    @classmethod
    def from_json(cls, data: dict) -> FaultCase:
        return cls(
            template_id=data["template_id"],
            split=data["split"],
            variation=SceneVariation(**data["variation"]),
        )


@dataclass(frozen=True)
class FrozenSplitManifest:
    """Platform-calibrated cases frozen before model experiments run."""

    calibration_seed: int
    variations_per_template: int
    correct_library_checksum: str
    cases: tuple[FaultCase, ...]
    exclusions: tuple[dict, ...] = ()
    schema_version: int = 1

    def to_json(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "calibration_seed": self.calibration_seed,
            "variations_per_template": self.variations_per_template,
            "correct_library_checksum": self.correct_library_checksum,
            "cases": [case.to_json() for case in self.cases],
            "exclusions": list(self.exclusions),
        }

    @classmethod
    def from_json(cls, data: dict) -> FrozenSplitManifest:
        if data.get("schema_version") != 1:
            raise ValueError("Unknown frozen split manifest schema")
        return cls(
            calibration_seed=data["calibration_seed"],
            variations_per_template=data["variations_per_template"],
            correct_library_checksum=data["correct_library_checksum"],
            cases=tuple(FaultCase.from_json(case) for case in data["cases"]),
            exclusions=tuple(data.get("exclusions", ())),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> FrozenSplitManifest:
        return cls.from_json(json.loads(path.read_text()))

    def validation_problems(self, template_ids: set[str] | None = None) -> list[str]:
        """Structural reasons this manifest is unsafe for a formal run."""
        selected = tuple(
            case
            for case in self.cases
            if template_ids is None or case.template_id in template_ids
        )
        problems = []
        seen_cases = set()
        seed_splits: dict[int, str] = {}
        for case in selected:
            key = (case.template_id, case.variation.seed)
            if key in seen_cases:
                problems.append(
                    f"duplicate case '{case.template_id}' seed {case.variation.seed}"
                )
            seen_cases.add(key)
            if case.split not in SPLITS:
                problems.append(
                    f"case '{case.template_id}' seed {case.variation.seed} has "
                    f"unknown split '{case.split}'"
                )
            previous_split = seed_splits.setdefault(case.variation.seed, case.split)
            if previous_split != case.split:
                problems.append(
                    f"scene seed {case.variation.seed} crosses splits "
                    f"'{previous_split}' and '{case.split}'"
                )

        identifiers = template_ids or {case.template_id for case in selected}
        for template_id in sorted(identifiers):
            cases = [case for case in selected if case.template_id == template_id]
            if len(cases) != self.variations_per_template:
                problems.append(
                    f"template '{template_id}' has {len(cases)} cases, expected "
                    f"{self.variations_per_template}"
                )
            missing = set(SPLITS) - {case.split for case in cases}
            if missing:
                problems.append(
                    f"template '{template_id}' lacks splits {sorted(missing)}"
                )
        return problems


class SplitCalibrationError(RuntimeError):
    """Candidate exhaustion with the rejection evidence kept for diagnosis."""

    def __init__(
        self,
        template_id: str,
        accepted_count: int,
        requested_count: int,
        attempted_count: int,
        accepted_candidates: tuple[dict, ...],
        exclusions: tuple[dict, ...],
    ) -> None:
        self.template_id = template_id
        self.accepted_count = accepted_count
        self.requested_count = requested_count
        self.attempted_count = attempted_count
        self.accepted_candidates = accepted_candidates
        self.exclusions = exclusions
        super().__init__(
            f"Calibration found only {accepted_count}/{requested_count} "
            f"feasible cases for '{template_id}' after {attempted_count} "
            "candidates"
        )


def library_checksum(library: SymbolLibrary) -> str:
    payload = json.dumps(
        library.to_json(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calibrate_splits(
    templates: tuple[FaultTemplate, ...],
    feasible: Callable[[FaultTemplate, SceneVariation], list[str]],
    correct_library: SymbolLibrary,
    variations_per_template: int = 10,
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    seed: int = 0,
    maximum_candidates_per_template: int = 200,
) -> FrozenSplitManifest:
    """Filter candidate variations through the real platform, then split.

    ``feasible`` returns rejection reasons. It must exercise the unmodified
    library's positive/regression tasks and the template's expected fault
    manifestation. Split labels are assigned only after enough cases pass,
    so excluded geometry can never leak into a formal experiment.
    """
    if variations_per_template < 3:
        raise ValueError("At least three variations are needed")
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Exactly three split fractions must sum to 1")
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions):
        raise ValueError("Every split fraction must be strictly between 0 and 1")
    proposal_count = max(1, round(fractions[0] * variations_per_template))
    admission_count = max(1, round(fractions[1] * variations_per_template))
    if proposal_count + admission_count >= variations_per_template:
        raise ValueError("Split fractions do not leave all three shares")
    if maximum_candidates_per_template < variations_per_template:
        raise ValueError("Maximum candidates must be at least variations per template")
    generator = random.Random(seed)
    sampled_seeds: set[int] = set()
    accepted: list[tuple[FaultTemplate, SceneVariation]] = []
    exclusions: list[dict] = []
    for template in templates:
        template_accepted = []
        attempted = 0
        while len(template_accepted) < variations_per_template:
            if attempted >= maximum_candidates_per_template:
                raise SplitCalibrationError(
                    template_id=template.identifier,
                    accepted_count=len(template_accepted),
                    requested_count=variations_per_template,
                    attempted_count=attempted,
                    accepted_candidates=tuple(
                        {
                            "template_id": accepted_template.identifier,
                            "variation": dataclasses.asdict(variation),
                        }
                        for accepted_template, variation in (
                            accepted + template_accepted
                        )
                    ),
                    exclusions=tuple(exclusions),
                )
            attempted += 1
            variation_seed = generator.randrange(1_000_000_000)
            while variation_seed in sampled_seeds:
                variation_seed = generator.randrange(1_000_000_000)
            sampled_seeds.add(variation_seed)
            variation = SceneVariation.from_seed(variation_seed)
            reasons = feasible(template, variation)
            if reasons:
                exclusions.append(
                    {
                        "template_id": template.identifier,
                        "variation_seed": variation.seed,
                        "reasons": reasons,
                    }
                )
            else:
                template_accepted.append((template, variation))
        accepted.extend(template_accepted)

    # Reuse the same deterministic count rule as the uncalibrated generator.
    cases = []
    for template in templates:
        selected = [
            item for item in accepted if item[0].identifier == template.identifier
        ]
        for index, (_, variation) in enumerate(selected):
            split = (
                PROPOSAL
                if index < proposal_count
                else ADMISSION if index < proposal_count + admission_count else HELD_OUT
            )
            cases.append(FaultCase(template.identifier, variation, split))
    return FrozenSplitManifest(
        calibration_seed=seed,
        variations_per_template=variations_per_template,
        correct_library_checksum=library_checksum(correct_library),
        cases=tuple(cases),
        exclusions=tuple(exclusions),
    )


def generate_splits(
    templates: tuple[FaultTemplate, ...],
    variations_per_template: int = 10,
    fractions: tuple[float, float, float] = (0.4, 0.3, 0.3),
    seed: int = 0,
) -> tuple[FaultCase, ...]:
    """Assign scene variations of every template to the three splits.

    Every template appears in all three splits (a repair episode needs a
    proposal context, admission tests, and a held-out evaluation), but
    the *scene variations* are disjoint across splits: what the proposer
    saw is never what admission tests run on, and held-out scenes are
    seen by neither. Deterministic in ``seed``.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")
    proposal_count = max(1, round(fractions[0] * variations_per_template))
    admission_count = max(1, round(fractions[1] * variations_per_template))
    if proposal_count + admission_count >= variations_per_template:
        raise ValueError(
            "Not enough variations per template to leave a held-out share."
        )
    generator = random.Random(seed)
    cases = []
    for template in templates:
        seeds = generator.sample(range(1_000_000), variations_per_template)
        for index, variation_seed in enumerate(seeds):
            if index < proposal_count:
                split = PROPOSAL
            elif index < proposal_count + admission_count:
                split = ADMISSION
            else:
                split = HELD_OUT
            cases.append(
                FaultCase(
                    template_id=template.identifier,
                    variation=SceneVariation.from_seed(variation_seed),
                    split=split,
                )
            )
    return tuple(cases)


# -- diagnosis evaluation (§5.3) --------------------------------------


@dataclass
class DiagnosisReport:
    """Diagnosis accuracy of the programmatic taxonomy over a template set."""

    total: int = 0
    correct: int = 0
    confusion: dict[tuple[str, str], int] = dataclasses.field(default_factory=dict)
    """(expected primary class, observed class) -> count; 'success' stands
    for a task that did not fail."""

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def render(self) -> str:
        lines = [f"diagnosis accuracy: {self.correct}/{self.total}"]
        for (expected, observed), count in sorted(self.confusion.items()):
            marker = "==" if observed == expected else "->"
            lines.append(f"  {expected} {marker} {observed}: {count}")
        return "\n".join(lines)


NO_FAILURE = "success"
"""Observed-class marker for a probe task that succeeded."""


def evaluate_diagnosis(
    observations: list[tuple[FaultTemplate, Optional[str]]],
) -> DiagnosisReport:
    """Score observed certificate classes against template ground truth.

    ``observations`` pairs each template with the ``FailureClass`` value
    the pipeline produced (``None`` for success). An observation counts
    as correct when it is among the template's expected classes — or,
    for ``SILENT`` templates, when the task indeed succeeded.
    """
    report = DiagnosisReport()
    for template, observed in observations:
        expected = template.ground_truth.expected_certificate_classes
        observed_key = observed if observed is not None else NO_FAILURE
        expected_key = expected[0] if expected else NO_FAILURE
        report.total += 1
        if (observed is None and not expected) or (
            observed is not None and observed in expected
        ):
            report.correct += 1
        key = (expected_key, observed_key)
        report.confusion[key] = report.confusion.get(key, 0) + 1
    return report
