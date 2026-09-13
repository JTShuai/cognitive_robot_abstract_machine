"""
Relation proposals lead to factory drafts without approving executable code.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from resym.core.grounding_model import GroundingFactoryReviewStatus
from resym.interfaces.initialization import InitializationFile
from resym.interfaces.initialization_models import InitializationKind

from .test_initialization import initialization
from .test_grounding_drafting import completer_with, draft_response, request
from .test_grounding_factory_catalog import vocabulary

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
    (job,) = initialization.prepare(kind=InitializationKind.GROUNDING)
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
    assert initialization.prepare(kind=InitializationKind.GROUNDING) == ()


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
    assert initialization.prepare(kind=InitializationKind.GROUNDING) == ()


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
