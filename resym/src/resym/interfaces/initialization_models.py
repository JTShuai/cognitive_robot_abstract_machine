"""JSON documents exchanged by initialization and external draft authors."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from resym.core.grounding_model import GroundingFactoryParameter, GroundingFactoryRole
from resym.core.symbol_types import SymbolType
from resym.interfaces.grounding_drafting import GroundingFactoryRequest
from resym.platform.coraplex_realizations import ParameterSourceKind
from resym.repair.agent_harness import (
    GroundingFactoryParameterArguments,
    GroundingFactoryRoleArguments,
)

# %% input and output schemas


class InitializationKind(StrEnum):
    """The review artifact a drafting job produces."""

    CONTRACT = "contract"
    REALIZATION = "realization"
    GROUNDING = "grounding"


class GroundingRequest(BaseModel):
    """A deployment's requested relation, with typed inputs and explicit parameters."""

    model_config = ConfigDict(extra="forbid")
    proposed_uid: str = Field(min_length=1)
    """Stable factory identity requested by the deployment."""
    semantic_name: str = Field(min_length=1)
    """Short relation name."""
    meaning: str = Field(min_length=1)
    """The condition under which the relation must return true."""
    roles: list[GroundingFactoryRoleArguments]
    """Positional factory inputs."""
    parameters: list[GroundingFactoryParameterArguments] = Field(default_factory=list)
    """Configurable inputs and their allowed ranges."""

    def request(self) -> GroundingFactoryRequest:
        """Convert the deployment document into the shared factory drafting request."""
        return GroundingFactoryRequest(
            self.proposed_uid,
            self.semantic_name,
            self.meaning,
            tuple(
                GroundingFactoryRole(role.name, SymbolType(role.symbol_type))
                for role in self.roles
            ),
            tuple(
                GroundingFactoryParameter(**item.model_dump())
                for item in self.parameters
            ),
        )


class NativeParameterSource(BaseModel):
    """One native constructor argument supplied by the reviewed mapping."""

    model_config = ConfigDict(extra="forbid")
    parameter: str
    """Native parameter name."""
    kind: ParameterSourceKind
    """Role, context value or constant."""
    value: str
    """Name or value of the selected source."""
    key_roles: list[str] = Field(default_factory=list)
    """Roles indexing a stored context witness."""


class RealizationDraft(BaseModel):
    """An author's proposed implementation of a previously approved contract."""

    model_config = ConfigDict(extra="forbid")
    parameter_sources: list[NativeParameterSource] = Field(min_length=1)
    """Bindings for the native action constructor."""
    required_resources: list[str] = Field(default_factory=list)
    """CRAM type references for robot resources the action needs."""
    condition_role: str | None = None
    """Optional role selecting this action variant."""
    condition_value: str | None = None
    """Optional required value of that role."""
    rationale: str = Field(min_length=1)
    """Why the mapping realizes the contract."""


class DraftJob(BaseModel):
    """Persisted authoring instructions and the evidence they refer to."""

    model_config = ConfigDict(extra="forbid")
    job_id: str = Field(pattern=r"^[a-z]+-[a-f0-9]{16}$")
    """Stable filename stem derived from the job inputs."""
    kind: InitializationKind
    """Response schema and destination review workspace."""
    prompt: str
    """Complete instructions, identical for API and external authors."""
    response_schema: dict
    """JSON schema for the response file."""
    source_checksum: str
    """Action source or query vocabulary checksum at preparation."""
    action_source_id: str | None = None
    """Scanned native action, for contracts and realizations."""
    contract_uid: str | None = None
    """Approved contract, for realization jobs."""
    contract_checksum: str | None = None
    """Exact approved contract used to prepare the realization job."""
    grounding_request: GroundingRequest | None = None
    """Declared relation for a grounding job."""
