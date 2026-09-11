"""
Grounding-factory discovery, review, materialization, and runtime resolution.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from resym.core.grounding import (
    GroundingFactoryCandidate,
    GroundingFactoryOrigin,
    GroundingFactoryReviewStatus,
    GroundingFactoryRole,
    GroundingFactorySourceKind,
    PredicateGroundingPlan,
)
from resym.core.model import (
    GROUNDING_PLAN_EVALUATOR_KEY,
    PredicateSymbol,
    SymbolLibrary,
    SymbolType,
)
from resym.llm.schemas import PredicateProposal
from resym.planning.grounding import evaluate_predicate
from resym.platform.embodiment import EmbodimentProfile
from resym.platform.evaluators import EvaluationContext
from resym.platform.grounding_catalog import (
    GroundingFactoryCatalog,
    GroundingFactoryCatalogError,
    GroundingFactorySourceError,
    GroundingFactoryUidConflictError,
    GroundingFactoryWorkspace,
    GroundingVocabulary,
    GroundingVocabularyEntry,
    GroundingVocabularyKind,
    discover_grounding_vocabulary,
    freeze_grounding_factories,
    initialize_grounding_factories,
)
from resym.platform.universe import GroundedObject, ObjectUniverse
from resym.repair.curator import Curator
from resym.repair.patch import ModelPatch
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)
DRAWER_TYPE = SymbolType.from_python_type(Drawer)
DATASET = Path(__file__).parent / "dataset" / "grounding_factories"


def vocabulary() -> GroundingVocabulary:
    return GroundingVocabulary(
        entries=(
            GroundingVocabularyEntry(
                qualified_name="krrood.entity_query_language.factories.entity",
                kind=GroundingVocabularyKind.EQL_FACTORY,
                signature="entity(selected_variable)",
                source_file="factories.py",
                source_checksum="entity-checksum",
            ),
            GroundingVocabularyEntry(
                qualified_name="krrood.entity_query_language.factories.variable",
                kind=GroundingVocabularyKind.EQL_FACTORY,
                signature="variable(type_, domain=None)",
                source_file="factories.py",
                source_checksum="variable-checksum",
            ),
        )
    )


def candidate(source_file: str = "valid_factory.py") -> GroundingFactoryCandidate:
    return GroundingFactoryCandidate(
        candidate_id="inside-region-candidate",
        proposed_uid="resym:grounding/inside-region",
        semantic_name="inside-region",
        source_code=(DATASET / source_file).read_text(),
        roles=(GroundingFactoryRole("object", ROBOT_TYPE),),
        generated_by="test-agent",
        rationale="A reviewed EQL query can decide this relation.",
        evidence=("preview:fixture",),
        source_kind=GroundingFactorySourceKind.AGENT_DRAFT,
    )


def test_scanner_discovers_symbolic_functions_and_predicate_classes() -> None:
    catalog = discover_grounding_vocabulary(
        {"fixture_grounding": DATASET / "discovered_vocabulary.py"}
    )

    entries = {entry.qualified_name: entry for entry in catalog.entries}
    assert (
        entries["fixture_grounding.has_label"].kind
        is GroundingVocabularyKind.SYMBOLIC_FUNCTION
    )
    assert entries["fixture_grounding.Near"].kind is GroundingVocabularyKind.PREDICATE


def test_scanned_eql_vocabulary_requires_review_and_detects_source_drift(
    tmp_path,
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    discovered = vocabulary()

    workspace.synchronize_vocabulary(discovered)

    assert workspace.reviewed_vocabulary() == GroundingVocabulary()
    assert all(
        str(item.review_status) == "pending-review"
        for item in workspace.vocabulary_candidates()
    )

    selected = discovered.entries[0]
    workspace.approve_vocabulary(selected.qualified_name, reviewer="human-reviewer")
    assert workspace.reviewed_vocabulary().entries == (selected,)

    changed = GroundingVocabulary(
        entries=(
            GroundingVocabularyEntry(
                qualified_name=selected.qualified_name,
                kind=selected.kind,
                signature=selected.signature,
                source_file=selected.source_file,
                source_checksum="changed-source",
            ),
        )
    )
    workspace.synchronize_vocabulary(changed)
    assert workspace.reviewed_vocabulary() == GroundingVocabulary()
    assert str(workspace.vocabulary_candidates()[0].review_status) == "pending-review"


def test_initialization_scans_to_review_queue_and_loads_only_approved_factories(
    tmp_path,
) -> None:
    initialization = initialize_grounding_factories(
        tmp_path,
        package_roots={"fixture_grounding": DATASET / "discovered_vocabulary.py"},
    )

    assert len(initialization.discovered_vocabulary.entries) == 2
    assert len(initialization.workspace.vocabulary_candidates()) == 2
    assert initialization.reviewed_vocabulary == GroundingVocabulary()
    assert all(
        item.origin is GroundingFactoryOrigin.PLATFORM
        for item in initialization.catalog
    )


def test_agent_candidate_stays_non_executable_until_human_approval(tmp_path) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(candidate())

    assert workspace.candidates()[0].candidate_id == candidate().candidate_id
    assert workspace.specifications() == ()
    assert not tuple(workspace.approved_package_directory.glob("*.py"))

    specification = workspace.approve(
        candidate().candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )

    assert specification.origin is GroundingFactoryOrigin.LOCAL
    assert specification.reviewed_by == "human-reviewer"
    assert specification.implementation_ref.endswith(":evaluate")
    assert workspace.specifications() == (specification,)
    assert (workspace.approved_package_directory / "__init__.py").is_file()


def test_approved_local_factory_uses_the_same_runtime_catalog_as_platform_factories(
    tmp_path,
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(candidate())
    specification = workspace.approve(
        candidate().candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )
    catalog = GroundingFactoryCatalog.load(workspace=workspace)

    procedure = catalog.resolve(
        specification.uid,
        expected_checksum=specification.implementation_checksum,
    )

    assert procedure(None, ObjectUniverse(), (), {}) is True
    assert catalog.specification(specification.uid) == specification
    assert any(entry.origin is GroundingFactoryOrigin.PLATFORM for entry in catalog)


def test_approved_factory_replacement_invalidates_old_plan_checksum(tmp_path) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    first_candidate = candidate()
    workspace.submit(first_candidate)
    first = workspace.approve(
        first_candidate.candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )
    replacement = replace(
        candidate("valid_factory_v2.py"),
        candidate_id="inside-region-candidate-v2",
    )
    workspace.submit(replacement)
    second = workspace.approve(
        replacement.candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )

    catalog = GroundingFactoryCatalog.load(workspace=workspace)

    assert second.uid == first.uid
    assert second.active_revision_id == "r0002"
    with pytest.raises(GroundingFactoryCatalogError):
        catalog.resolve(first.uid, expected_checksum=first.implementation_checksum)


@pytest.mark.parametrize(
    "source_file", ("invalid_factory.py", "invalid_attribute_factory.py")
)
def test_materializer_rejects_source_outside_the_scanned_eql_vocabulary(
    tmp_path, source_file
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(candidate(source_file))

    with pytest.raises(GroundingFactorySourceError):
        workspace.approve(
            candidate().candidate_id,
            reviewer="human-reviewer",
            vocabulary=vocabulary(),
        )

    assert workspace.specifications() == ()


def test_predicate_plan_resolves_factory_and_applies_negation(tmp_path) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(candidate())
    specification = workspace.approve(
        candidate().candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )
    catalog = GroundingFactoryCatalog.load(workspace=workspace)
    predicate = PredicateSymbol(
        name="closed",
        parameter_types=(),
        evaluator="legacy-unused",
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid=specification.uid,
            approved_factory_checksum=specification.implementation_checksum,
            negated=True,
        ),
    )
    context = EvaluationContext(
        world=None,
        robot=None,
        profile=EmbodimentProfile("stub", frozenset(), frozenset()),
        grounding_catalog=catalog,
    )

    assert evaluate_predicate(predicate, (), ObjectUniverse(), context) is False


@pytest.fixture()
def opened_drawer() -> tuple[ObjectUniverse, GroundedObject]:
    """
    A drawer on a prismatic joint, opened to 0.75 of its range.
    """
    world = World()
    base = Body(name=PrefixedName("base"))
    drawer_body = Body(name=PrefixedName("drawer"))
    with world.modify_world():
        world.add_kinematic_structure_entity(base)
        world.add_kinematic_structure_entity(drawer_body)
        lower = DerivativeMap()
        lower.position = 0.0
        lower.velocity = -1.0
        upper = DerivativeMap()
        upper.position = 0.4
        upper.velocity = 1.0
        slide = DegreeOfFreedom(
            name=PrefixedName("slide"),
            limits=DegreeOfFreedomLimits(lower=lower, upper=upper),
        )
        world.add_degree_of_freedom(slide)
        connection = PrismaticConnection(
            parent=base,
            child=drawer_body,
            raw_dof=slide,
            axis=Vector3.X(reference_frame=base),
        )
        world.add_connection(connection)
    connection.position = 0.3
    grounded = GroundedObject(
        name="d1",
        symbol_type=DRAWER_TYPE,
        body=drawer_body,
        semantic_entity=Drawer(root=drawer_body),
    )
    universe = ObjectUniverse()
    universe.add(grounded)
    return universe, grounded


def test_platform_threshold_plan_evaluates_the_drawer_joint_fraction(
    opened_drawer,
) -> None:
    """
    A plan threshold replaces the budget threshold on the same articulation query.
    """
    universe, drawer = opened_drawer
    catalog = GroundingFactoryCatalog.load()
    specification = catalog.specification("resym:grounding/joint-fraction-opened")
    context = EvaluationContext(
        world=None,
        robot=None,
        profile=EmbodimentProfile("stub", frozenset(), frozenset()),
        grounding_catalog=catalog,
    )

    def opened(threshold: float) -> PredicateSymbol:
        return PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            evaluator=GROUNDING_PLAN_EVALUATOR_KEY,
            fluent=True,
            grounding_plan=PredicateGroundingPlan(
                factory_uid=specification.uid,
                approved_factory_checksum=specification.implementation_checksum,
                role_bindings=(("articulated_object", 0),),
                parameters=(("threshold", threshold),),
            ),
        )

    assert evaluate_predicate(opened(0.5), (drawer,), universe, context) is True
    assert evaluate_predicate(opened(0.9), (drawer,), universe, context) is False


def test_approving_a_candidate_with_an_already_owned_uid_is_rejected(
    tmp_path,
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    colliding = replace(
        candidate(), proposed_uid="resym:grounding/joint-fraction-opened"
    )
    workspace.submit(colliding)

    with pytest.raises(GroundingFactoryUidConflictError):
        workspace.approve(
            colliding.candidate_id,
            reviewer="human-reviewer",
            vocabulary=vocabulary(),
        )

    assert workspace.specifications() == ()
    assert (
        workspace.candidates()[0].review_status
        is GroundingFactoryReviewStatus.PENDING_REVIEW
    )
    catalog = GroundingFactoryCatalog.load(workspace=workspace)
    assert any(entry.origin is GroundingFactoryOrigin.PLATFORM for entry in catalog)


def test_local_source_drift_disables_the_factory_without_breaking_the_catalog(
    tmp_path,
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(candidate())
    specification = workspace.approve(
        candidate().candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )
    source_file = next(workspace.approved_package_directory.glob("factory_*.py"))
    source_file.write_text(source_file.read_text() + "\n# tampered\n")

    catalog = GroundingFactoryCatalog.load(workspace=workspace)

    assert specification.uid in catalog.unavailable
    assert specification.uid not in {entry.uid for entry in catalog}
    with pytest.raises(GroundingFactoryCatalogError):
        catalog.resolve(
            specification.uid,
            expected_checksum=specification.implementation_checksum,
        )
    platform_specification = catalog.specification(
        "resym:grounding/joint-fraction-opened"
    )
    assert callable(
        catalog.resolve(
            platform_specification.uid,
            expected_checksum=platform_specification.implementation_checksum,
        )
    )


def test_curator_checks_factory_identity_checksum_and_role_types(tmp_path) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(candidate())
    specification = workspace.approve(
        candidate().candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )
    predicate = PredicateSymbol(
        name="inside-region",
        parameter_types=(ROBOT_TYPE,),
        evaluator=GROUNDING_PLAN_EVALUATOR_KEY,
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid=specification.uid,
            approved_factory_checksum=specification.implementation_checksum,
            role_bindings=(("object", 0),),
        ),
    )
    curator = Curator(
        known_evaluators=frozenset(),
        available_capabilities=frozenset(),
        grounding_factory_specs={specification.uid: specification},
    )

    assert (
        curator.static_review(ModelPatch(predicates=(predicate,)), SymbolLibrary())
        == []
    )

    drifted = PredicateSymbol(
        name="inside-region",
        parameter_types=(ROBOT_TYPE,),
        evaluator=GROUNDING_PLAN_EVALUATOR_KEY,
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid=specification.uid,
            approved_factory_checksum="different-checksum",
            role_bindings=(("object", 0),),
        ),
    )
    objections = curator.static_review(
        ModelPatch(predicates=(drifted,)), SymbolLibrary()
    )
    assert any("checksum" in objection for objection in objections)


def test_closed_reuses_opened_factory_by_negation_with_explicit_threshold() -> None:
    catalog = GroundingFactoryCatalog.load()
    specification = catalog.specification("resym:grounding/joint-fraction-opened")
    assert "threshold:number[0.0,1.0]" in catalog.render()
    specs = {item.uid: item for item in catalog}
    curator = Curator(
        known_evaluators=frozenset(),
        available_capabilities=frozenset(),
        grounding_factory_specs=specs,
    )

    def closed(parameters):
        return PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            evaluator=GROUNDING_PLAN_EVALUATOR_KEY,
            fluent=True,
            grounding_plan=PredicateGroundingPlan(
                factory_uid=specification.uid,
                approved_factory_checksum=specification.implementation_checksum,
                role_bindings=(("articulated_object", 0),),
                parameters=parameters,
                negated=True,
            ),
        )

    assert (
        curator.static_review(
            ModelPatch(predicates=(closed((("threshold", 0.4),)),)),
            SymbolLibrary(),
        )
        == []
    )
    missing = curator.static_review(
        ModelPatch(predicates=(closed(()),)), SymbolLibrary()
    )
    out_of_range = curator.static_review(
        ModelPatch(predicates=(closed((("threshold", 1.1),)),)),
        SymbolLibrary(),
    )
    assert any("missing required" in objection for objection in missing)
    assert any("invalid value" in objection for objection in out_of_range)


def test_agent_predicate_proposal_can_reference_an_approved_grounding_plan() -> None:
    proposal = PredicateProposal.model_validate(
        {
            "name": "closed",
            "parameter_types": [ROBOT_TYPE.python_type_ref],
            "fluent": True,
            "grounding_plan": {
                "factory_uid": "resym:grounding/joint-fraction-opened",
                "approved_factory_checksum": "reviewed-checksum",
                "role_bindings": {"articulated_object": 0},
                "parameters": {"threshold": 0.9},
                "negated": True,
            },
        }
    )

    predicate = proposal.to_predicate_symbol()

    assert predicate.evaluator == GROUNDING_PLAN_EVALUATOR_KEY
    assert predicate.grounding_plan is not None
    assert predicate.grounding_plan.negated is True
    assert predicate.grounding_plan.parameters == (("threshold", 0.9),)


def test_symbol_library_round_trips_a_predicate_grounding_plan(tmp_path) -> None:
    predicate = PredicateProposal.model_validate(
        {
            "name": "closed",
            "parameter_types": [ROBOT_TYPE.python_type_ref],
            "fluent": True,
            "grounding_plan": {
                "factory_uid": "resym:grounding/joint-fraction-opened",
                "approved_factory_checksum": "reviewed-checksum",
                "role_bindings": {"articulated_object": 0},
                "negated": True,
            },
        }
    ).to_predicate_symbol()
    path = tmp_path / "library.json"
    SymbolLibrary(predicates={predicate.name: predicate}).save(path)

    restored = SymbolLibrary.load(path).predicates[predicate.name]

    assert restored == predicate


def test_formal_snapshot_includes_unpublished_local_source_and_catalog(
    tmp_path,
) -> None:
    workspace = GroundingFactoryWorkspace(tmp_path / "workspace")
    workspace.submit(candidate())
    workspace.approve(
        candidate().candidate_id,
        reviewer="human-reviewer",
        vocabulary=vocabulary(),
    )
    symbol_library = tmp_path / "library.json"
    symbol_library.write_text("{}")

    snapshot = freeze_grounding_factories(
        workspace=workspace,
        output_directory=tmp_path / "snapshot",
        symbol_library=symbol_library,
        git_commits=(("cram", "abc123"),),
        package_versions=(("krrood", "26.07.0"),),
        container_image_digest="sha256:image",
        experiment_configuration=(("experiment", "E1"),),
    )

    manifest = json.loads(snapshot.manifest.read_text())
    assert manifest["catalog_checksum"] == snapshot.catalog_checksum
    assert manifest["symbol_library_checksum"] == snapshot.symbol_library_checksum
    assert manifest["git_commits"] == {"cram": "abc123"}
    assert manifest["container_image_digest"] == "sha256:image"
    assert manifest["review_metadata_checksum"]
    assert tuple(snapshot.source_bundle.rglob("*.py"))
