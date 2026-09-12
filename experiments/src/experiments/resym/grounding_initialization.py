"""
Drawer-domain grounding assets, initialized through the platform review flow.

The platform ships no predicate semantics. This module declares what the drawer
experiments need: the reviewed query-helper vocabulary, the factory requests a model
drafts EQL candidates for, and a deterministic bootstrap that submits the pinned
reference implementations through the same candidate-review flow for benches and tests,
where no model or reviewer is in the loop.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from resym import PROJECT_ROOT
from resym.core.grounding import (
    GroundingFactoryCandidate,
    GroundingFactoryParameter,
    GroundingFactoryParameterType,
    GroundingFactoryRole,
    GroundingFactorySourceKind,
    text_checksum,
)
from resym.interfaces.grounding_drafting import GroundingFactoryRequest
from resym.platform.articulation import (
    articulation_connection,
    interaction_point_belongs_to,
)
from resym.platform.capabilities import (
    ARTICULATED_PART_TYPE,
    HANDLE_TYPE,
    capability_contracts,
)
from resym.platform.grounding_catalog import (
    GroundingFactoryCatalog,
    GroundingFactoryWorkspace,
    helper_vocabulary,
)
from resym.platform.kinematic import KINEMATIC_FEASIBILITY
from resym.platform.universe import joint_fraction

DEFAULT_WORKSPACE_ROOT = PROJECT_ROOT / "tmp" / "grounding_factory_workspace"
"""
Standard local review workspace of the drawer experiments.
"""

REFERENCE_DIRECTORY = Path(__file__).parent / "grounding_references"
"""
Pinned reference implementations of the drawer grounding requests.
"""

GROUNDING_QUERY_HELPERS = (
    articulation_connection,
    interaction_point_belongs_to,
    joint_fraction,
)
"""
Platform query helpers this domain asks to have in the EQL vocabulary.
"""

JOINT_FRACTION_OPENED_UID = "resym:grounding/joint-fraction-opened"
INTERACTION_POINT_OF_UID = "resym:grounding/interaction-point-of"


def drawer_grounding_requests() -> tuple[GroundingFactoryRequest, ...]:
    """
    The state-predicate relations the drawer domain needs drafted.
    """
    return (
        GroundingFactoryRequest(
            proposed_uid=JOINT_FRACTION_OPENED_UID,
            semantic_name="joint-fraction-opened",
            meaning=(
                "True iff the articulated object's normalized joint position "
                "is at least the 'threshold' parameter."
            ),
            roles=(GroundingFactoryRole("articulated_object", ARTICULATED_PART_TYPE),),
            parameters=(
                GroundingFactoryParameter(
                    name="threshold",
                    value_type=GroundingFactoryParameterType.NUMBER,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
        ),
        GroundingFactoryRequest(
            proposed_uid=INTERACTION_POINT_OF_UID,
            semantic_name="interaction-point-of",
            meaning=(
                "True iff the interaction point is the one mounted on the "
                "articulated object in the scene graph."
            ),
            roles=(
                GroundingFactoryRole("interaction_point", HANDLE_TYPE),
                GroundingFactoryRole("articulated_object", ARTICULATED_PART_TYPE),
            ),
        ),
    )


def reference_source(request: GroundingFactoryRequest) -> str:
    """
    Pinned reference implementation of one drawer grounding request.
    """
    stem = request.semantic_name.replace("-", "_")
    return (REFERENCE_DIRECTORY / f"{stem}.py").read_text(encoding="utf-8")


def reference_candidate(
    request: GroundingFactoryRequest,
) -> GroundingFactoryCandidate:
    """
    The reference implementation wrapped as a review candidate.
    """
    return GroundingFactoryCandidate(
        candidate_id=f"reference-{request.semantic_name}",
        proposed_uid=request.proposed_uid,
        semantic_name=request.semantic_name,
        source_code=reference_source(request),
        roles=request.roles,
        parameters=request.parameters,
        generated_by="drawer-reference",
        rationale=request.meaning,
        source_kind=GroundingFactorySourceKind.DISCOVERED,
    )


def bootstrap_drawer_grounding(
    workspace_root: Path,
    reviewer: str = "experiment-bootstrap",
) -> GroundingFactoryCatalog:
    """
    Approve the pinned reference factories and load the runtime catalog.

    Live deployments instead draft candidates with
    :func:`resym.interfaces.grounding_drafting.draft_grounding_factory_candidates`
    and decide them in the review Viewer; this bootstrap exists for benches
    and tests, which need the same catalog without a model or a human in the
    loop. Repeated calls on the same workspace are no-ops.
    """
    workspace = GroundingFactoryWorkspace(workspace_root)
    vocabulary = helper_vocabulary(GROUNDING_QUERY_HELPERS)
    workspace.synchronize_vocabulary(
        vocabulary,
        discovery_scope="drawer-experiment-helpers",
    )
    reviewed = {
        entry.qualified_name for entry in workspace.reviewed_vocabulary().entries
    }
    for entry in vocabulary.entries:
        if entry.qualified_name not in reviewed:
            workspace.approve_vocabulary(entry.qualified_name, reviewer=reviewer)
    approved = {
        specification.uid: specification for specification in workspace.specifications()
    }
    for request in drawer_grounding_requests():
        candidate = reference_candidate(request)
        previous = approved.get(request.proposed_uid)
        if previous is not None and workspace.implementation_drift(previous) is None:
            continue
        dependency_revision = text_checksum(
            candidate.source_code
            + "".join(
                entry.source_checksum
                for entry in workspace.reviewed_vocabulary().entries
            )
        )[:12]
        candidate = replace(
            candidate,
            candidate_id=f"{candidate.candidate_id}-{dependency_revision}",
        )
        workspace.submit(candidate)
        workspace.approve(
            candidate.candidate_id,
            reviewer=reviewer,
            vocabulary=workspace.reviewed_vocabulary(),
        )
    return GroundingFactoryCatalog.load(
        workspace=workspace,
        capability_contracts=capability_contracts(),
        capability_feasibility_implementations=KINEMATIC_FEASIBILITY,
    )


def default_drawer_grounding_catalog() -> GroundingFactoryCatalog:
    """
    Bootstrap the standard local workspace (idempotently) and load it.
    """
    return bootstrap_drawer_grounding(DEFAULT_WORKSPACE_ROOT)
