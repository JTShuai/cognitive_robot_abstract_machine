"""
Native query roles and read-only attributes are checked against CRAM types.
"""

from dataclasses import replace
import inspect
from pathlib import Path

from resym.core.grounding_model import GroundingFactoryRole
from resym.core.symbol_types import SymbolType
from resym.platform.grounding_types import resolvable_type_hints
from resym.platform.grounding_catalog import (
    READABLE_ATTRIBUTE_PACKAGES,
    GroundingFactorySourceValidator,
    GroundingVocabulary,
    discover_default_grounding_vocabulary,
    discover_grounding_vocabulary,
    helper_vocabulary,
    GroundingFactoryWorkspace,
    GroundingFactoryCatalog,
)
from resym.core.grounding_model import PredicateGroundingPlan
from resym.core.symbols import PredicateSymbol
from resym.platform.grounding_context import EvaluationContext
from resym.planning.state_evaluation import evaluate_predicate
from semantic_digital_twin.reasoning.robot_predicates import robot_holds_body
from semantic_digital_twin.world_description import world_entity
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from .test_grounding_factory_catalog import candidate, DATASET, opened_drawer


def test_query_rejects_body_in_robot_role():
    proposal = replace(
        candidate("direct_boolean.py"),
        native_arguments=True,
        roles=(
            GroundingFactoryRole("robot", SymbolType.from_python_type(Body)),
            GroundingFactoryRole("body", SymbolType.from_python_type(Body)),
        ),
    )
    assert GroundingFactorySourceValidator(
        helper_vocabulary((robot_holds_body,))
    ).candidate_objections(proposal)


def test_query_accepts_compatible_roles():
    proposal = replace(
        candidate("direct_boolean.py"),
        native_arguments=True,
        roles=(
            GroundingFactoryRole("robot", SymbolType.from_python_type(AbstractRobot)),
            GroundingFactoryRole("body", SymbolType.from_python_type(Body)),
        ),
    )
    assert (
        GroundingFactorySourceValidator(
            helper_vocabulary((robot_holds_body,))
        ).candidate_objections(proposal)
        == ()
    )


def test_scanned_pose_property_is_readable_on_body():
    root = Path(inspect.getsourcefile(world_entity)).parent
    vocabulary = discover_grounding_vocabulary(
        {"semantic_digital_twin.world_description": root}
    )
    proposal = replace(
        candidate("body_pose.py"),
        native_arguments=True,
        roles=(GroundingFactoryRole("body", SymbolType.from_python_type(Body)),),
    )
    assert (
        GroundingFactorySourceValidator(vocabulary).candidate_objections(proposal) == ()
    )


def test_pose_property_cannot_be_read_on_wrong_owner():
    root = Path(inspect.getsourcefile(world_entity)).parent
    vocabulary = discover_grounding_vocabulary(
        {"semantic_digital_twin.world_description": root}
    )
    assert GroundingFactorySourceValidator(vocabulary).candidate_objections(
        candidate("body_pose.py")
    )


def test_runtime_passes_native_body_to_approved_factory(tmp_path, opened_drawer):
    universe, grounded = opened_drawer
    body_type = SymbolType.from_python_type(Body)
    vocabulary = discover_grounding_vocabulary(
        {
            "semantic_digital_twin.world_description": Path(
                inspect.getsourcefile(world_entity)
            ).parent
        }
    )
    proposal = replace(
        candidate("body_pose.py"),
        native_arguments=True,
        roles=(GroundingFactoryRole("body", body_type),),
    )
    workspace = GroundingFactoryWorkspace(tmp_path)
    workspace.submit(proposal)
    specification = workspace.approve(proposal.candidate_id, "test", vocabulary)
    assert specification.native_arguments is True
    assert specification.dependency_checksums
    catalog = GroundingFactoryCatalog.load(workspace=workspace)
    predicate = PredicateSymbol(
        name="pose-available",
        parameter_types=(body_type,),
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid=specification.uid,
            approved_factory_checksum=specification.implementation_checksum,
        ),
    )
    context = EvaluationContext(world=None, robot=None, grounding_catalog=catalog)
    assert evaluate_predicate(predicate, (grounded,), universe, context) is True


def test_review_notes_flag_reuse_and_unused_roles(tmp_path):
    workspace = GroundingFactoryWorkspace(tmp_path)
    original = candidate()
    workspace.submit(original)
    duplicate = replace(original, candidate_id="other", proposed_uid="other")
    notes = workspace.review_notes(duplicate)
    assert any(original.proposed_uid in note for note in notes)
    assert any(
        original.roles[0].name in note and "Unused roles" in note for note in notes
    )


# %% attribute rendering and scan scope


def test_attributes_render_through_declared_return_types():
    from semantic_digital_twin.spatial_types import spatial_types

    vocabulary = discover_grounding_vocabulary(
        {
            "semantic_digital_twin.world_description": Path(
                inspect.getsourcefile(world_entity)
            ).parent,
            "semantic_digital_twin.spatial_types": Path(
                inspect.getsourcefile(spatial_types)
            ).parent,
        }
    )
    rendered = vocabulary.render_attributes(
        (SymbolType.from_python_type(Body).python_type_ref,)
    )
    pose_position = (
        f"{spatial_types.Pose.__module__}.{spatial_types.Pose.__qualname__}.position"
    )
    assert pose_position in rendered


def test_attribute_scan_covers_only_world_model_packages():
    vocabulary = discover_default_grounding_vocabulary()
    owners = {
        entry.owner_type_ref
        for entry in vocabulary.entries
        if entry.owner_type_ref is not None
    }
    assert owners
    assert all(owner.startswith(READABLE_ATTRIBUTE_PACKAGES) for owner in owners)


# %% type hints of platform interfaces


def test_member_hints_resolve_independently_of_unresolvable_siblings():
    from semantic_digital_twin.robots.robot_parts import EndEffector

    assert resolvable_type_hints(EndEffector)["tool_frame"] is Body


def test_parameter_hints_skip_only_the_unresolvable_names():
    from semantic_digital_twin.reasoning.predicates import visible
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

    hints = resolvable_type_hints(visible)
    assert hints["obj"] is KinematicStructureEntity
    assert "camera" not in hints


# %% canonical member references


def world_model_vocabulary():
    """
    Readable members of the world description, robots and spatial types.
    """
    from semantic_digital_twin.robots import robot_parts
    from semantic_digital_twin.spatial_types import spatial_types

    return discover_grounding_vocabulary(
        {
            "semantic_digital_twin.world_description": Path(
                inspect.getsourcefile(world_entity)
            ).parent,
            "semantic_digital_twin.robots": Path(
                inspect.getsourcefile(robot_parts)
            ).parent,
            "semantic_digital_twin.spatial_types": Path(
                inspect.getsourcefile(spatial_types)
            ).parent,
        }
    )


def qualified(native_type: type, member: str) -> str:
    return f"{native_type.__module__}.{native_type.__qualname__}.{member}"


def test_attributes_render_along_the_whole_member_chain():
    from semantic_digital_twin.robots.robot_parts import EndEffector
    from semantic_digital_twin.spatial_types.spatial_types import Point

    rendered = world_model_vocabulary().render_attributes(
        (SymbolType.from_python_type(EndEffector).python_type_ref,)
    )
    assert qualified(Point, "x") in rendered


def test_member_cited_with_a_wrong_module_resolves_to_its_declaring_class():
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

    misplaced = qualified(KinematicStructureEntity, "global_pose").replace(
        "world_description.world_entity", "spatial_types.spatial_types"
    )
    assert world_model_vocabulary().canonical_reference(misplaced) == qualified(
        KinematicStructureEntity, "global_pose"
    )


def test_member_cited_on_a_subclass_resolves_to_the_declaring_class():
    from semantic_digital_twin.spatial_types.spatial_types import Point, Point3

    assert world_model_vocabulary().canonical_reference(
        qualified(Point3, "x")
    ) == qualified(Point, "x")


def test_member_cited_on_a_superclass_has_no_canonical_reference():
    from semantic_digital_twin.world_description.connections import (
        ActiveConnection1DOF,
        Connection,
    )

    vocabulary = world_model_vocabulary()
    assert vocabulary.canonical_reference(qualified(Connection, "position")) is None
    assert vocabulary.member_owners(qualified(Connection, "position")) == (
        qualified(ActiveConnection1DOF, "position"),
    )


# %% unrelated owners stay unimported


def test_member_lookup_never_imports_owners_outside_the_receiver_mro():
    root = Path(inspect.getsourcefile(world_entity)).parent
    scanned = discover_grounding_vocabulary(
        {"semantic_digital_twin.world_description": root}
    )
    unimportable = replace(
        scanned.entries[0],
        qualified_name="semantic_digital_twin.robots.missing_module.Gripper.global_pose",
        owner_type_ref="semantic_digital_twin.robots.missing_module.Gripper",
    )
    vocabulary = GroundingVocabulary(scanned.entries + (unimportable,))
    proposal = replace(
        candidate("body_pose.py"),
        native_arguments=True,
        roles=(GroundingFactoryRole("body", SymbolType.from_python_type(Body)),),
    )
    assert (
        GroundingFactorySourceValidator(vocabulary).candidate_objections(proposal) == ()
    )
