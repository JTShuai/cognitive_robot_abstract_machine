"""
Relation proposals lead to factory drafts without approving executable code.
"""

from __future__ import annotations

import json
import ast
from dataclasses import replace

import pytest

from resym.core.grounding_model import GroundingFactoryReviewStatus
from resym.interfaces.initialization import InitializationFile
from resym.interfaces.initialization_models import InitializationKind
from resym.platform.coraplex_catalog import installed_coraplex_root

from resym.core.symbol_types import SymbolType
from resym.interfaces.initialization import (
    Initialization,
    SELECTOR_PARAMETER_TYPES,
)
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.platform.grounding_catalog import GroundingVocabulary, helper_vocabulary
from semantic_digital_twin.reasoning.robot_predicates import robot_holds_body
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from .test_initialization import initialization
from .test_capability_drafting import PLACE_ACTION
from .test_grounding_drafting import completer_with, draft_response, request
from .test_grounding_factory_catalog import DRAWER_TYPE, ROBOT_TYPE, vocabulary
from .test_llm import OutputLimitedClient

# %% relation discovery and factory follow-up


def relation_response():
    """
    Propose a typed relation using a scanned query reference.
    """
    needed = request()
    return json.dumps(
        {
            "relations": [
                {
                    "request": {
                        "proposed_uid": needed.proposed_uid,
                        "semantic_name": needed.semantic_name,
                        "meaning": needed.meaning,
                        "roles": [
                            {
                                "name": role.name,
                                "symbol_type": role.symbol_type.python_type_ref,
                            }
                            for role in needed.roles
                        ],
                    },
                    "query_references": [vocabulary().entries[0].qualified_name],
                    "rationale": "Use the available entity query to inspect the requested relation.",
                }
            ],
            "limitations": "Geometry must be supplied by the deployment.",
        }
    )


def prepare_relations(initialization):
    """
    Export just the grounding stages to keep model replies focused.
    """
    (job,) = initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    assert job.kind is InitializationKind.RELATIONS
    return job


def test_empty_workspace_exports_relation_proposal(initialization):
    job = prepare_relations(initialization)
    assert vocabulary().entries[0].qualified_name in job.prompt
    assert initialization.grounding().workspace.candidates() == ()


def test_external_relation_import_creates_factory_jobs(initialization):
    job = prepare_relations(initialization)
    initialization.response_path(job).write_text(relation_response())
    report = initialization.import_responses(generated_by="external-author")
    assert report.failed == {}
    (factory_job,) = [
        job for job in initialization.jobs() if job.kind is InitializationKind.GROUNDING
    ]
    assert factory_job.grounding_request.proposed_uid == request().proposed_uid
    assert factory_job.job_id in report.prepared
    assert initialization.grounding().workspace.candidates() == ()
    requests = json.loads(
        (initialization.materials / InitializationFile.REQUESTS).read_text()
    )
    assert requests[0]["proposed_uid"] == request().proposed_uid


def test_api_drafts_relations_then_factory_once(initialization):
    prepare_relations(initialization)
    completer, client = completer_with(
        relation_response(), draft_response("valid_factory.py")
    )
    report = initialization.draft(completer)
    assert report.failed == {}
    assert len(client.received_prompts) == 2
    (candidate,) = initialization.grounding().workspace.candidates()
    assert candidate.review_status is GroundingFactoryReviewStatus.PENDING_REVIEW
    assert initialization.grounding().workspace.specifications() == ()
    initialization.draft(completer)
    assert len(client.received_prompts) == 2
    assert (
        initialization.prepare(
            kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
        )
        == ()
    )


def test_unscanned_query_is_rejected_before_requests_are_saved(initialization):
    job = prepare_relations(initialization)
    response = json.loads(relation_response())
    response["relations"][0]["query_references"] = ["unavailable.query"]
    initialization.response_path(job).write_text(json.dumps(response))
    report = initialization.import_responses()
    assert job.job_id in report.failed
    assert (
        json.loads((initialization.materials / InitializationFile.REQUESTS).read_text())
        == []
    )
    assert initialization.jobs() == (job,)


def test_corrected_proposal_cannot_overwrite_existing_request(initialization):
    job = prepare_relations(initialization)
    initialization.response_path(job).write_text(relation_response())
    initialization.import_responses()
    path = initialization.materials / InitializationFile.REQUESTS
    original = path.read_text()
    response = json.loads(relation_response())
    response["relations"][0]["request"]["meaning"] = "A different relation."
    initialization.response_path(job).write_text(json.dumps(response))
    report = initialization.import_responses()
    assert job.job_id in report.failed
    assert path.read_text() == original


def test_empty_proposal_is_persisted_without_repeated_discovery(initialization):
    job = prepare_relations(initialization)
    initialization.response_path(job).write_text(
        json.dumps({"relations": [], "limitations": "No supported relation found."})
    )
    report = initialization.import_responses()
    assert report.failed == {}
    assert (
        initialization.prepare(
            kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
        )
        == ()
    )


def test_explicit_empty_requests_skip_relation_discovery(initialization):
    assert initialization.prepare(requests=(), kind=InitializationKind.GROUNDING) == ()


def test_source_drift_refuses_relation_import(initialization, monkeypatch):
    job = prepare_relations(initialization)
    initialization.response_path(job).write_text(relation_response())
    original = vocabulary()
    changed = replace(
        original,
        entries=tuple(
            replace(item, source_checksum="changed") for item in original.entries
        ),
    )
    monkeypatch.setattr(
        "resym.platform.grounding_catalog.discover_default_grounding_vocabulary",
        lambda: changed,
    )
    report = initialization.import_responses()
    assert job.job_id in report.failed
    assert (
        json.loads((initialization.materials / InitializationFile.REQUESTS).read_text())
        == []
    )


@pytest.mark.parametrize(
    "symbol_type", ["os.PathLike", "semantic_digital_twin.world.MissingClass"]
)
def test_invalid_relation_role_type_is_rejected(initialization, symbol_type):
    job = prepare_relations(initialization)
    response = json.loads(relation_response())
    response["relations"][0]["request"]["roles"][0]["symbol_type"] = symbol_type
    initialization.response_path(job).write_text(json.dumps(response))
    report = initialization.import_responses()
    assert job.job_id in report.failed
    assert initialization.jobs() == (job,)


def test_unsupported_factory_is_saved_without_retries_or_admission(initialization):
    prepare_relations(initialization)
    completer, client = completer_with(
        relation_response(),
        json.dumps(
            {
                "source_code": "",
                "rationale": "No matching observation.",
                "unsupported_reason": "Required world observation is unavailable.",
            }
        ),
    )
    report = initialization.draft(completer)
    assert len(client.received_prompts) == 2
    assert report.failed == {}
    assert len(report.unsupported) == 1
    assert initialization.grounding().workspace.candidates() == ()
    repeated = initialization.draft(completer)
    assert repeated.unsupported == report.unsupported
    assert len(client.received_prompts) == 2


def test_api_corrects_invalid_relation_before_drafting_factory(initialization):
    prepare_relations(initialization)
    invalid = json.loads(relation_response())
    invalid["relations"][0]["query_references"] = ["unavailable.query"]
    completer, client = completer_with(
        json.dumps(invalid), relation_response(), draft_response("valid_factory.py")
    )
    report = initialization.draft(completer)
    assert report.failed == {}
    assert len(client.received_prompts) == 3
    assert "unavailable.query" in client.received_prompts[1]
    assert len(initialization.grounding().workspace.candidates()) == 1


# %% batching and resumability


def test_action_source_preserves_class_decorators(initialization):
    action = next(
        item for item in initialization.actions() if item.source_id == PLACE_ACTION
    )
    source = (installed_coraplex_root() / action.source_file).read_text()
    class_name = action.action_class.rsplit(".", 1)[-1]
    original = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    extracted = next(
        node
        for node in ast.parse(initialization._action_source(action)).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    assert [ast.dump(item) for item in extracted.decorator_list] == [
        ast.dump(item) for item in original.decorator_list
    ]


def test_relations_are_split_by_action(initialization):
    jobs = initialization.prepare(kind=InitializationKind.RELATIONS)
    assert {job.relation_action_ids for job in jobs} == {
        (action.source_id,) for action in initialization.actions()
    }


def test_prepare_retains_unfinished_batches_after_first_import(initialization):
    jobs = initialization.prepare(kind=InitializationKind.RELATIONS)
    first = jobs[0]
    initialization.response_path(first).write_text(relation_response())
    initialization.import_responses()
    remaining = initialization.prepare(kind=InitializationKind.RELATIONS)
    assert {job.job_id for job in remaining} == {job.job_id for job in jobs[1:]}


def test_malformed_replies_do_not_multiply_retry_budgets(initialization):
    prepare_relations(initialization)
    completer, client = completer_with("invalid", "invalid", "invalid")
    completer.maximum_attempts = 3
    report = initialization.draft(completer, maximum_attempts=3)
    assert len(client.received_prompts) == 3
    assert len(report.failed) == 1


def test_draft_uses_configured_attempt_count(initialization):
    prepare_relations(initialization)
    completer, client = completer_with("invalid", "invalid")
    completer.maximum_attempts = 2
    report = initialization.draft(completer)
    assert len(client.received_prompts) == completer.maximum_attempts
    assert len(report.failed) == 1


def test_truncated_batch_retries_with_larger_output_allowance(initialization):
    job = prepare_relations(initialization)
    client = OutputLimitedClient(
        json.dumps({"relations": [], "limitations": "None supported."})
    )
    report = initialization.draft(StructuredCompleter(client, TranscriptRecorder()))
    assert report.failed == {}
    assert client.limits == [None, client.initial_limit * 2]
    assert initialization.response_path(job).is_file()


def test_repeated_truncation_stays_failed_and_resumable(initialization):
    job = prepare_relations(initialization)
    client = OutputLimitedClient("", always_truncate=True)
    report = initialization.draft(StructuredCompleter(client, TranscriptRecorder()))
    assert client.limits == [None, client.initial_limit * 2]
    assert job.job_id in report.failed
    assert job.job_id in report.missing
    assert not initialization.response_path(job).exists()


def test_failed_batch_is_not_retried_during_factory_followup(initialization):
    actions = tuple(action.source_id for action in initialization.actions()[:2])
    first, second = initialization.prepare(
        action_ids=actions, kind=InitializationKind.RELATIONS
    )
    completer, client = completer_with(
        "invalid",
        "invalid",
        "invalid",
        relation_response(),
        draft_response("valid_factory.py"),
    )
    report = initialization.draft(completer)
    assert len(client.received_prompts) == 5
    assert set(report.failed) == {first.job_id}
    assert initialization.response_path(second).exists()


# %% relation-stage validation against the cited queries


@pytest.fixture
def typed_initialization(tmp_path, monkeypatch):
    """
    A workspace whose vocabulary carries one query with typed parameters.
    """
    typed = GroundingVocabulary(
        vocabulary().entries + helper_vocabulary((robot_holds_body,)).entries
    )
    monkeypatch.setattr(
        "resym.platform.grounding_catalog.discover_default_grounding_vocabulary",
        lambda: typed,
    )
    return Initialization(tmp_path)


def typed_relation_response(role_type: SymbolType) -> str:
    """
    Propose a one-role relation citing the typed query.
    """
    response = json.loads(relation_response())
    proposal = response["relations"][0]
    proposal["request"]["roles"] = [
        {"name": "object", "symbol_type": role_type.python_type_ref}
    ]
    proposal["query_references"] = [
        f"{robot_holds_body.__module__}.{robot_holds_body.__name__}"
    ]
    return json.dumps(response)


def test_reference_copied_with_its_rendered_signature_is_normalized(
    typed_initialization,
):
    (job,) = typed_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    (entry,) = helper_vocabulary((robot_holds_body,)).entries
    response = json.loads(typed_relation_response(ROBOT_TYPE))
    response["relations"][0]["query_references"] = [
        f"{entry.qualified_name}{entry.signature}: {entry.kind.value}"
    ]
    typed_initialization.response_path(job).write_text(json.dumps(response))
    report = typed_initialization.import_responses()
    assert report.failed == {}
    (record,) = typed_initialization._relation_records().values()
    assert record.response.relations[0].query_references == [entry.qualified_name]


def test_role_type_outside_every_cited_query_parameter_is_rejected(
    typed_initialization,
):
    (job,) = typed_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    typed_initialization.response_path(job).write_text(
        typed_relation_response(DRAWER_TYPE)
    )
    report = typed_initialization.import_responses()
    assert set(report.failed) == {job.job_id}
    assert AbstractRobot.__qualname__ in report.failed[job.job_id]
    assert Body.__qualname__ in report.failed[job.job_id]


def test_role_type_accepted_by_a_cited_query_parameter_is_imported(
    typed_initialization,
):
    (job,) = typed_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    typed_initialization.response_path(job).write_text(
        typed_relation_response(ROBOT_TYPE)
    )
    report = typed_initialization.import_responses()
    assert report.failed == {}
    assert report.proposed == [request().proposed_uid]


def test_selector_parameters_are_mapped_to_world_entity_types(initialization):
    job = prepare_relations(initialization)
    for reference in SELECTOR_PARAMETER_TYPES["Arms"]:
        assert reference.python_type_ref in job.prompt


def test_failed_batch_reports_the_total_attempts(initialization):
    job = prepare_relations(initialization)
    completer, _ = completer_with("invalid", "invalid", "invalid")
    completer.maximum_attempts = 3
    report = initialization.draft(completer)
    assert f"{completer.maximum_attempts} attempts" in report.failed[job.job_id]


def test_draft_report_lists_each_candidate_once(initialization):
    prepare_relations(initialization)
    completer, _ = completer_with(
        relation_response(), draft_response("valid_factory.py")
    )
    report = initialization.draft(completer)
    assert len(report.submitted) == 1
    assert set(report.submitted).isdisjoint(report.existing)


def test_uniquely_matching_class_corrects_the_module_path(typed_initialization):
    (job,) = typed_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    misplaced = SymbolType(
        f"semantic_digital_twin.robots.abstract_robot.{AbstractRobot.__qualname__}"
    )
    typed_initialization.response_path(job).write_text(
        typed_relation_response(misplaced)
    )
    report = typed_initialization.import_responses()
    assert report.failed == {}
    (saved,) = json.loads(
        (typed_initialization.materials / InitializationFile.REQUESTS).read_text()
    )
    assert saved["roles"][0]["symbol_type"] == ROBOT_TYPE.python_type_ref


def test_role_type_no_task_object_can_denote_is_rejected(typed_initialization):
    (job,) = typed_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    pose_type = SymbolType.from_python_type(Pose)
    typed_initialization.response_path(job).write_text(
        typed_relation_response(pose_type)
    )
    report = typed_initialization.import_responses()
    assert set(report.failed) == {job.job_id}
    assert pose_type.python_type_ref in report.failed[job.job_id]


# %% member references at the relation stage


@pytest.fixture
def attribute_initialization(tmp_path, monkeypatch):
    """
    A workspace whose vocabulary carries readable members of the world model.
    """
    from .test_grounding_types import world_model_vocabulary

    with_members = GroundingVocabulary(
        vocabulary().entries + world_model_vocabulary().entries
    )
    monkeypatch.setattr(
        "resym.platform.grounding_catalog.discover_default_grounding_vocabulary",
        lambda: with_members,
    )
    return Initialization(tmp_path)


def test_member_reference_is_saved_under_its_declaring_class(
    attribute_initialization,
):
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )

    (job,) = attribute_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    response = json.loads(relation_response())
    proposal = response["relations"][0]
    proposal["request"]["roles"] = [
        {
            "name": "body",
            "symbol_type": SymbolType.from_python_type(Body).python_type_ref,
        }
    ]
    declared = (
        f"{KinematicStructureEntity.__module__}."
        f"{KinematicStructureEntity.__qualname__}.global_pose"
    )
    proposal["query_references"] = [
        declared.replace(
            "world_description.world_entity", "spatial_types.spatial_types"
        )
    ]
    attribute_initialization.response_path(job).write_text(json.dumps(response))
    report = attribute_initialization.import_responses()
    assert report.failed == {}
    (record,) = attribute_initialization._relation_records().values()
    assert record.response.relations[0].query_references == [declared]


def test_member_cited_on_a_superclass_is_rejected_naming_the_owners(
    attribute_initialization,
):
    from semantic_digital_twin.world_description.connections import (
        ActiveConnection1DOF,
        Connection,
    )

    (job,) = attribute_initialization.prepare(
        kind=InitializationKind.GROUNDING, action_ids=(PLACE_ACTION,)
    )
    response = json.loads(relation_response())
    proposal = response["relations"][0]
    proposal["request"]["roles"] = [
        {
            "name": "body",
            "symbol_type": SymbolType.from_python_type(Body).python_type_ref,
        }
    ]
    proposal["query_references"] = [
        f"{Connection.__module__}.{Connection.__qualname__}.position"
    ]
    attribute_initialization.response_path(job).write_text(json.dumps(response))
    report = attribute_initialization.import_responses()
    assert set(report.failed) == {job.job_id}
    assert ActiveConnection1DOF.__qualname__ in report.failed[job.job_id]
