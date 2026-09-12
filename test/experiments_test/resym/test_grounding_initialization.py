"""
The drawer domain's grounding assets against the platform review flow.
"""

from __future__ import annotations

from pathlib import Path

from resym.platform.grounding_catalog import (
    GroundingFactorySourceValidator,
    helper_vocabulary,
)

from experiments.resym.grounding_initialization import (
    GROUNDING_QUERY_HELPERS,
    drawer_grounding_requests,
    reference_candidate,
)


def test_reference_sources_satisfy_the_reviewed_sandbox() -> None:
    """
    Every relation the drawer domain requests has a reference implementation the source
    validator accepts against the helpers that domain asks to review.
    """
    validator = GroundingFactorySourceValidator(
        helper_vocabulary(GROUNDING_QUERY_HELPERS)
    )
    requests = drawer_grounding_requests()

    objections = {
        request.proposed_uid: validator.candidate_objections(
            reference_candidate(request)
        )
        for request in requests
    }

    assert requests
    assert objections == {request.proposed_uid: () for request in requests}


def test_every_request_has_a_reference_implementation() -> None:
    """
    The bootstrap can submit each request without a model in the loop.
    """
    for request in drawer_grounding_requests():
        candidate = reference_candidate(request)
        assert candidate.source_code
        assert candidate.roles == request.roles
        assert candidate.parameters == request.parameters


def test_reference_directory_holds_only_reviewed_sources() -> None:
    """
    Every module beside the references is one of the requested implementations.
    """
    from experiments.resym.grounding_initialization import REFERENCE_DIRECTORY

    stems = {
        request.semantic_name.replace("-", "_")
        for request in drawer_grounding_requests()
    }
    modules = {
        path.stem
        for path in Path(REFERENCE_DIRECTORY).glob("*.py")
        if path.stem != "__init__"
    }

    assert modules == stems
