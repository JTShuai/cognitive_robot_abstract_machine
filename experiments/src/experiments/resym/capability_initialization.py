"""
Initialization of the experiment's capability contracts.

The reference contracts are a JSON artifact; benches and tests admit them through the
same candidate-review flow a live deployment drives from the review Viewer, so the
library holds identical contracts whether a human or this bootstrap approved them.
"""

from __future__ import annotations

import json
from pathlib import Path

from krrood.adapters.json_serializer import from_json
from typing_extensions import Iterable

from experiments import EXPERIMENTS_ROOT
from resym.core.capability_model import CapabilityContract
from resym.platform.capability_contract_review import (
    CapabilityContractCandidate,
    CapabilityContractWorkspace,
)
from resym.platform.coraplex_catalog import discover_coraplex_capability_contract_drafts

REFERENCE_CONTRACTS = Path(__file__).parent / "capability_references" / "contracts.json"
"""
The reviewed contracts of the experiment platform.
"""

DEFAULT_CONTRACT_WORKSPACE_ROOT = EXPERIMENTS_ROOT / "tmp" / "contract_workspace"
"""
Standard local workspace the experiment bootstraps its contracts into.
"""


def reference_contracts() -> tuple[CapabilityContract, ...]:
    """
    The reference contracts as recorded in the artifact.
    """
    data = json.loads(REFERENCE_CONTRACTS.read_text(encoding="utf-8"))
    return tuple(from_json(item) for item in data["contracts"])


def bootstrap_capability_contracts(
    workspace_root: Path,
    reviewer: str = "experiment-bootstrap",
    package_root: Path | None = None,
) -> tuple[CapabilityContract, ...]:
    """
    Approve the reference contracts into a workspace and return what it admits.

    Repeated calls on the same workspace are no-ops for contracts already admitted.
    """
    workspace = CapabilityContractWorkspace(workspace_root)
    drafts = discover_coraplex_capability_contract_drafts(package_root)
    admitted = {(item.uid, item.version) for item in workspace.approved_contracts()}
    for contract in reference_contracts():
        if (contract.uid, contract.version) in admitted:
            continue
        candidate_id = f"reference-{contract.uid.split(':', 1)[1]}-v{contract.version}"
        workspace.submit(
            CapabilityContractCandidate(
                candidate_id=candidate_id,
                contract=contract,
                action_source_ids=(),
                generated_by=reviewer,
                rationale="reference contract of the experiment platform",
            )
        )
        workspace.approve(candidate_id, reviewer, drafts)
    return workspace.approved_contracts()


def default_capability_contracts() -> tuple[CapabilityContract, ...]:
    """
    Bootstrap the standard local workspace (idempotently) and return its contracts.
    """
    return bootstrap_capability_contracts(DEFAULT_CONTRACT_WORKSPACE_ROOT)


def contracts_by_uid(
    contracts: Iterable[CapabilityContract],
) -> dict[str, CapabilityContract]:
    """
    Contracts keyed by their uid.
    """
    return {contract.uid: contract for contract in contracts}
