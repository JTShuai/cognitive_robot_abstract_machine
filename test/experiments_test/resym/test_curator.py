"""
The trusted curator: static checklist, mandatory-suite coverage rules, full-suite
execution, and versioned admission.

Host-runnable, stdlib-only.
"""

from __future__ import annotations

import pytest

from resym.repair.patch import ModelPatch
from resym.repair.curator import Curator
from experiments.resym.icra.validation import (
    AdmissionTest,
    BehaviouralCurator,
    MandatorySuite,
    SuiteGroup,
)
from resym.core.model import (
    EvaluatorSpec,
    Literal,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
    SymbolType,
)
from resym.repair.versioning import VersionedLibraryStore
from .capability_helpers import capability_contract, execution_binding

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


KNOWN_EVALUATORS = frozenset({"drawer_opened", "drawer_closed"})
CAPABILITY_UID = "test:DrawerStateChange"
AVAILABLE_CAPABILITIES = frozenset({CAPABILITY_UID})


def base_library() -> SymbolLibrary:
    library = SymbolLibrary()
    for name, fluent in (("opened", True), ("closed", True), ("kind-of", False)):
        library.add(
            PredicateSymbol(
                name=name,
                parameter_types=(DRAWER_TYPE,),
                evaluator="drawer_opened",
                fluent=fluent,
            )
        )
    library.add_capability_contract(
        capability_contract(
            CAPABILITY_UID,
            (("patient", DRAWER_TYPE),),
            ("opened", "closed"),
        )
    )
    return library


def close_operator(**overrides) -> Operator:
    defaults = dict(
        name="close-drawer",
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(Literal("opened", ("d",)),),
        add_effects=(Literal("closed", ("d",)),),
        delete_effects=(Literal("opened", ("d",)),),
        execution_binding=execution_binding(CAPABILITY_UID, (("patient", "d"),)),
    )
    defaults.update(overrides)
    return Operator(**defaults)


def good_patch(**overrides) -> ModelPatch:
    return ModelPatch(operators=(close_operator(**overrides),))


def passing_suite(recorder: list[str] | None = None) -> MandatorySuite:
    def passing(name):
        def run(candidate):
            if recorder is not None:
                recorder.append(name)
            return []

        return run

    return MandatorySuite(
        version="suite-1",
        tests=(
            AdmissionTest(
                "opens",
                SuiteGroup.CAPABILITY_POSITIVE,
                "admission-ctx",
                passing("opens"),
            ),
            AdmissionTest(
                "refuses-unreachable",
                SuiteGroup.CAPABILITY_NEGATIVE,
                "admission-ctx",
                passing("refuses-unreachable"),
            ),
            AdmissionTest(
                "threshold-boundary",
                SuiteGroup.BOUNDARY,
                "admission-ctx",
                passing("threshold-unknown"),
            ),
            AdmissionTest(
                "old-open-still-works",
                SuiteGroup.REGRESSION,
                "proposal-ctx",
                passing("old-open-still-works"),
            ),
        ),
    )


def curator_with(suite: MandatorySuite) -> BehaviouralCurator:
    return BehaviouralCurator(
        known_evaluators=KNOWN_EVALUATORS,
        available_capabilities=AVAILABLE_CAPABILITIES,
        suite=suite,
    )


def typed_curator(suite: MandatorySuite) -> BehaviouralCurator:
    return BehaviouralCurator(
        known_evaluators=KNOWN_EVALUATORS,
        available_capabilities=AVAILABLE_CAPABILITIES,
        suite=suite,
        evaluator_specs={
            "drawer_opened": EvaluatorSpec(
                "drawer_opened",
                (DRAWER_TYPE,),
            ),
        },
        allowed_symbol_types=frozenset({ROBOT_TYPE}),
    )


# -- static checklist ---------------------------------------------------


@pytest.mark.parametrize(
    "patch, expected",
    [
        (
            good_patch(
                execution_binding=execution_binding(
                    "test:Teleport", (("patient", "d"),)
                )
            ),
            "not implemented by the current embodiment",
        ),
        (
            good_patch(add_effects=(Literal("levitating", ("d",)),)),
            "unknown predicate",
        ),
        (
            good_patch(preconditions=(Literal("opened", ("d", "d")),)),
            "takes 1 arguments",
        ),
        (
            good_patch(preconditions=(Literal("opened", ("cabinet10",)),)),
            "not an operator parameter",
        ),
        (
            good_patch(add_effects=(Literal("kind-of", ("d",)),)),
            "non-fluent",
        ),
        (
            good_patch(
                add_effects=(Literal("closed", ("d",)),),
                delete_effects=(Literal("closed", ("d",)), Literal("opened", ("d",))),
            ),
            "adds and deletes the same literal",
        ),
        (
            ModelPatch(
                operators=(close_operator(),),
                unresolved=("MISSING_IMPLEMENTATION: predicate 'locked'",),
            ),
            "unresolved requirements",
        ),
    ],
)
def test_static_checklist_refuses(patch, expected):
    curator = curator_with(passing_suite())
    objections = curator.static_review(patch, base_library())
    assert any(expected in objection for objection in objections), objections


def test_effect_contract_conflict_is_static():
    library = base_library()
    library.capability_contracts[CAPABILITY_UID] = capability_contract(
        CAPABILITY_UID,
        (("patient", DRAWER_TYPE),),
        ("closed",),
    )
    curator = curator_with(passing_suite())
    objections = curator.static_review(good_patch(), library)
    assert any("not verifiable through capability" in o for o in objections)


def test_clean_patch_passes_static_review():
    curator = Curator(
        known_evaluators=KNOWN_EVALUATORS,
        available_capabilities=AVAILABLE_CAPABILITIES,
    )
    assert curator.static_review(good_patch(), base_library()) == []


def test_core_curator_admits_without_a_behavioural_suite(tmp_path):
    curator = Curator(
        known_evaluators=KNOWN_EVALUATORS,
        available_capabilities=AVAILABLE_CAPABILITIES,
    )
    library = base_library()
    store = VersionedLibraryStore(tmp_path)
    store.commit(library)

    report, version_id = curator.admit(
        good_patch(), library, store, proposal_context_id="task-context"
    )

    assert report.admitted
    assert version_id is not None
    assert store.info(version_id).metadata["admission_evidence"] == {
        "static_objections": []
    }


def test_unknown_evaluator_is_rejected_by_the_platform_catalog():
    patch = ModelPatch(
        predicates=(
            PredicateSymbol(
                name="has-handle",
                parameter_types=(DRAWER_TYPE,),
                evaluator="not_registered",
                fluent=False,
            ),
        )
    )

    objections = curator_with(passing_suite()).static_review(patch, base_library())

    assert any(
        "unknown truth procedure 'not_registered'" in item for item in objections
    )


def test_unknown_operator_type_is_an_objection_with_a_local_suggestion():
    library = base_library()
    curator = curator_with(passing_suite())
    curator.allowed_symbol_types = frozenset({ROBOT_TYPE})
    misspelled = SymbolType("semantic_digital_twin.robots.robots_parts:AbstractRobot")

    objections = curator.static_review(
        good_patch(parameters=(("d", misspelled),)),
        library,
    )

    assert any(
        "unknown symbol type" in item
        and misspelled.python_type_ref in item
        and ROBOT_TYPE.python_type_ref in item
        for item in objections
    ), objections


def test_unknown_predicate_type_is_an_objection_not_an_exception():
    misspelled = SymbolType(
        "semantic_digital_twin.semantic_annotations.semantic_annotation:Drawer"
    )
    predicate = PredicateSymbol(
        name="bad",
        parameter_types=(misspelled,),
        evaluator="drawer_opened",
        fluent=True,
    )

    objections = typed_curator(passing_suite()).static_review(
        ModelPatch(predicates=(predicate,)),
        base_library(),
    )

    assert any(
        "unknown symbol type" in item
        and misspelled.python_type_ref in item
        and DRAWER_TYPE.python_type_ref in item
        for item in objections
    ), objections


def test_typed_catalog_rejects_evaluator_and_capability_role_mismatches():
    curator = typed_curator(passing_suite())
    wrong_predicate = PredicateSymbol(
        name="bad",
        parameter_types=(ROBOT_TYPE,),
        evaluator="drawer_opened",
        fluent=True,
    )
    objections = curator.static_review(
        ModelPatch(
            predicates=(wrong_predicate,),
            operators=(
                close_operator(
                    parameters=(("r", ROBOT_TYPE),),
                    execution_binding=execution_binding(
                        CAPABILITY_UID, (("patient", "r"),)
                    ),
                ),
            ),
        ),
        base_library(),
    )
    assert any("evaluator 'drawer_opened' expects" in item for item in objections)
    assert any(
        "parameter 'r'" in item and "role 'patient'" in item for item in objections
    )


def test_model_patch_cannot_self_authorize_contract_change():
    curator = typed_curator(passing_suite())
    patch = ModelPatch(
        capability_contracts=(
            capability_contract(
                CAPABILITY_UID,
                (("patient", DRAWER_TYPE),),
                ("anything",),
            ),
        )
    )
    objections = curator.static_review(patch, base_library())
    assert any(
        "cannot modify trusted capability contracts" in item for item in objections
    )


def test_binding_version_must_match_persisted_contract():
    library = base_library()
    patch = good_patch(
        execution_binding=execution_binding(
            CAPABILITY_UID, (("patient", "d"),), version="2"
        )
    )
    objections = typed_curator(passing_suite()).static_review(patch, library)
    assert any("not contract" in item for item in objections)


# -- mandatory suite ----------------------------------------------------


def test_suite_missing_a_group_cannot_admit():
    incomplete = MandatorySuite(
        version="suite-0",
        tests=tuple(
            test
            for test in passing_suite().tests
            if test.group is not SuiteGroup.CAPABILITY_NEGATIVE
        ),
    )
    report = curator_with(incomplete).review(
        good_patch(), base_library(), proposal_context_id="proposal-ctx"
    )
    assert not report.admitted
    assert any("capability_negative" in p for p in report.suite_problems)


def test_suite_needs_a_distinct_admission_context():
    same_context = MandatorySuite(
        version="suite-0",
        tests=tuple(
            AdmissionTest(t.name, t.group, "proposal-ctx", t.run)
            for t in passing_suite().tests
        ),
    )
    report = curator_with(same_context).review(
        good_patch(), base_library(), proposal_context_id="proposal-ctx"
    )
    assert not report.admitted
    assert any("admission context" in p for p in report.suite_problems)


def test_suite_runs_in_full_even_after_a_failure():
    executed: list[str] = []
    suite = passing_suite(executed)
    failing = AdmissionTest(
        "fails-first",
        SuiteGroup.CAPABILITY_POSITIVE,
        "admission-ctx",
        lambda candidate: (executed.append("fails-first"), ["did not close"])[1],
    )
    suite = MandatorySuite(version="suite-1", tests=(failing, *suite.tests))
    report = curator_with(suite).review(
        good_patch(), base_library(), proposal_context_id="proposal-ctx"
    )
    assert not report.admitted
    assert len(executed) == 5  # every test ran; no short-circuit


def test_a_crashing_suite_test_refuses_instead_of_raising():
    """
    Candidate content is arbitrary model output; a crash it provokes in the harness is
    evidence against the candidate, not a batch abort.
    """

    def crashing(candidate):
        raise ValueError("too many values to unpack (expected 2)")

    suite = passing_suite()
    suite = MandatorySuite(
        version="suite-1",
        tests=(
            AdmissionTest(
                "crashes", SuiteGroup.CAPABILITY_POSITIVE, "admission-ctx", crashing
            ),
            *suite.tests,
        ),
    )
    report = curator_with(suite).review(
        good_patch(), base_library(), proposal_context_id="proposal-ctx"
    )
    assert not report.admitted
    crashed = next(r for r in report.test_results if r.name == "crashes")
    assert crashed.failures == [
        "test crashed (ValueError): too many values to unpack (expected 2)"
    ]
    assert all(r.passed for r in report.test_results if r.name != "crashes")


def test_statically_dirty_patch_never_reaches_behaviour():
    executed: list[str] = []
    report = curator_with(passing_suite(executed)).review(
        good_patch(
            execution_binding=execution_binding("test:Teleport", (("patient", "d"),))
        ),
        base_library(),
        proposal_context_id="proposal-ctx",
    )
    assert not report.admitted
    assert executed == []


# -- admission ----------------------------------------------------------


def test_admission_commits_a_version_with_evidence(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    library = base_library()
    store.commit(library, metadata={"note": "seed"})
    curator = curator_with(passing_suite())
    report, version_id = curator.admit(
        good_patch(),
        library,
        store,
        proposal_context_id="proposal-ctx",
        metadata={"backend": "fixed-pipeline", "certificate_id": "cert-1"},
    )
    assert report.admitted
    assert version_id is not None
    admitted = store.load(version_id)
    assert "close-drawer" in admitted.operators
    info = store.info(version_id)
    assert info.metadata["backend"] == "fixed-pipeline"
    assert info.metadata["admission_evidence"]["suite_version"] == "suite-1"
    assert len(info.metadata["admission_evidence"]["tests"]) == 4


def test_admission_resolves_and_pins_an_existing_catalog_contract(tmp_path):
    catalog_uid = "test:CatalogDrawerStateChange"
    contract = capability_contract(
        catalog_uid,
        (("patient", DRAWER_TYPE),),
        ("opened", "closed"),
    )
    patch = good_patch(
        execution_binding=execution_binding(
            catalog_uid,
            (("patient", "d"),),
        )
    )
    suite = passing_suite()
    catalog_visible = AdmissionTest(
        "catalog-contract-visible",
        SuiteGroup.CAPABILITY_POSITIVE,
        "admission-ctx",
        lambda candidate: (
            []
            if candidate.capability_contracts.get(catalog_uid) == contract
            else ["catalog contract was not pinned before behavioural review"]
        ),
    )
    curator = BehaviouralCurator(
        known_evaluators=KNOWN_EVALUATORS,
        available_capabilities=frozenset({catalog_uid}),
        capability_catalog=(contract,),
        suite=MandatorySuite(
            version=suite.version,
            tests=(catalog_visible, *suite.tests[1:]),
        ),
    )
    library = base_library()
    store = VersionedLibraryStore(tmp_path)
    store.commit(library)

    report, version_id = curator.admit(
        patch,
        library,
        store,
        proposal_context_id="proposal-ctx",
    )

    assert report.admitted
    assert patch.capability_contracts == ()
    assert store.load(version_id).capability_contracts[catalog_uid] == contract


def test_refused_patch_commits_nothing(tmp_path):
    store = VersionedLibraryStore(tmp_path)
    library = base_library()
    seed_version = store.commit(library)
    curator = curator_with(passing_suite())
    report, version_id = curator.admit(
        good_patch(
            execution_binding=execution_binding("test:Teleport", (("patient", "d"),))
        ),
        library,
        store,
        proposal_context_id="proposal-ctx",
    )
    assert not report.admitted
    assert version_id is None
    assert store.head_version() == seed_version
