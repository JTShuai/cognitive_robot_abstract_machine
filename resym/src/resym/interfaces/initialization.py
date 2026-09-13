"""Prepare initialization materials and import drafts into human review workspaces."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from krrood.adapters.json_serializer import to_json
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Any, Iterable

from resym import PROJECT_ROOT
from resym.core.grounding_model import (
    GroundingFactoryCandidate,
    GroundingFactoryReviewStatus,
    text_checksum,
)
from resym.core.symbol_types import SymbolType
from resym.interfaces.capability_drafting import (
    CapabilityContractDraftModel,
    capability_contract_candidate,
    capability_contract_prompt,
)
from resym.interfaces.grounding_drafting import (
    GroundingFactoryDraftModel,
    grounding_factory_candidate,
    grounding_factory_prompt,
)
from resym.interfaces.initialization_models import (
    DraftJob,
    GroundingRequest,
    InitializationKind,
    RealizationDraft,
)
from resym.llm.configuration import LanguageModelConfiguration, build_completion_client
from resym.llm.prompting import render_prompt
from resym.llm.structured import (
    StructuredCompleter,
    StructuredOutputRetriesExceededError,
)
from resym.llm.transcript import TranscriptRecorder
from resym.platform.capability_contract_review import (
    CapabilityContractCandidate,
    CapabilityContractWorkspace,
    contract_objections,
)
from resym.platform.coraplex_catalog import (
    CoraplexCapabilityContractDraft,
    discover_coraplex_capability_contract_drafts,
    installed_coraplex_root,
)
from resym.platform.coraplex_realizations import (
    ActionRealization,
    CapabilityReviewStatus,
    ContextValue,
    CoraplexRealizationWorkspace,
    ParameterSource,
    RealizationCandidate,
    RoleCondition,
    action_source_checksum,
    realization_objections,
)
from resym.platform.grounding_catalog import (
    GroundingFactoryInitialization,
    GroundingFactoryWorkspace,
    initialize_grounding_factories,
)

# %% persistent materials


class InitializationFile(StrEnum):
    """Names of initialization artifacts and review workspaces."""

    MATERIALS = "initialization"
    JOBS = "jobs.json"
    PLATFORM = "platform.json"
    RESPONSES = "responses"
    REQUESTS = "grounding_requests.json"
    REQUEST_SCHEMA = "grounding_requests.schema.json"
    INSTRUCTIONS = "instructions.md"
    REPORT = "import_report.json"
    TRANSCRIPT = "llm_transcript.jsonl"
    API_RESPONSES = "api_responses.json"
    CONTRACTS = "contract_workspace"
    REALIZATIONS = "realization_workspace"
    GROUNDING = "grounding_factory_workspace"


class InitializationCommand(StrEnum):
    """Manually invoked stages of initialization."""

    PREPARE = "prepare"
    DRAFT = "draft"
    IMPORT = "import"


class InvalidInitializationDraft(ValueError):
    """A response cannot enter the review queue under the current platform."""


@dataclass
class ImportReport:
    """Per-job results; accepted entries are pending human review."""

    submitted: list[str] = field(default_factory=list)
    """New candidate identifiers."""
    existing: list[str] = field(default_factory=list)
    """Identical candidates already in a review workspace."""
    missing: list[str] = field(default_factory=list)
    """Jobs without a saved response."""
    failed: dict[str, str] = field(default_factory=dict)
    """Invalid responses and their actionable objections."""


@dataclass
class Initialization:
    """A deployment's persistent initialization and human-review directories."""

    root: Path
    """Parent of the three review workspaces and drafting materials."""

    @property
    def materials(self) -> Path:
        """Directory holding portable authoring instructions and replies."""
        return self.root / InitializationFile.MATERIALS

    @property
    def contract_root(self) -> Path:
        """Contract review workspace path."""
        return self.root / InitializationFile.CONTRACTS

    @property
    def realization_root(self) -> Path:
        """Native mapping review workspace path."""
        return self.root / InitializationFile.REALIZATIONS

    def grounding(self) -> GroundingFactoryInitialization:
        """Refresh official vocabulary and load the local approved factories."""
        return initialize_grounding_factories(self.root / InitializationFile.GROUNDING)

    def actions(self) -> tuple[CoraplexCapabilityContractDraft, ...]:
        """Discover action signatures from installed Coraplex source."""
        return tuple(discover_coraplex_capability_contract_drafts())

    def response_path(self, job: DraftJob, directory: Path | None = None) -> Path:
        """The JSON reply expected for one exported job."""
        return (
            directory or self.materials / InitializationFile.RESPONSES
        ) / f"{job.job_id}.json"

    def jobs(self) -> tuple[DraftJob, ...]:
        """Read the last prepared set of drafting tasks."""
        return tuple(
            TypeAdapter(list[DraftJob]).validate_json(
                (self.materials / InitializationFile.JOBS).read_text()
            )
        )

    def prepare(
        self,
        action_ids: tuple[str, ...] = (),
        requests: tuple[GroundingRequest, ...] | None = None,
        kind: InitializationKind | None = None,
    ) -> tuple[DraftJob, ...]:
        """Export missing artifacts as authoring jobs without calling a model."""
        self.materials.mkdir(parents=True, exist_ok=True)
        (self.materials / InitializationFile.RESPONSES).mkdir(exist_ok=True)
        request_path = self.materials / InitializationFile.REQUESTS
        if requests is None:
            requests = (
                tuple(
                    TypeAdapter(list[GroundingRequest]).validate_json(
                        request_path.read_text()
                    )
                )
                if request_path.exists()
                else ()
            )
        _write_json(request_path, [item.model_dump(mode="json") for item in requests])
        _write_json(
            self.materials / InitializationFile.REQUEST_SCHEMA,
            TypeAdapter(list[GroundingRequest]).json_schema(),
        )
        grounding = self.grounding()
        actions = self.actions()
        unknown = set(action_ids) - {item.source_id for item in actions}
        if unknown:
            raise InvalidInitializationDraft(f"Unknown action ids: {sorted(unknown)}")
        contracts = CapabilityContractWorkspace(self.contract_root)
        approved = contracts.approved_contracts()
        selected = tuple(
            item for item in actions if not action_ids or item.source_id in action_ids
        )
        jobs = []
        if kind in (None, InitializationKind.CONTRACT):
            covered = {
                source_id
                for candidate in contracts.candidates()
                if candidate.review_status
                in (CapabilityReviewStatus.PENDING, CapabilityReviewStatus.APPROVED)
                for source_id in candidate.action_source_ids
            }
            for action in selected:
                if action.source_id in covered:
                    continue
                objections = tuple(
                    f"{item.candidate_id}: {item.review_note or 'Rejected by reviewer'}"
                    for item in contracts.candidates()
                    if item.review_status is CapabilityReviewStatus.REJECTED
                    and action.source_id in item.action_source_ids
                )
                prompt = capability_contract_prompt(action, approved, objections)
                prompt += (
                    "\n\nNative source:\n"
                    + (installed_coraplex_root() / action.source_file).read_text()
                )
                jobs.append(
                    _job(
                        InitializationKind.CONTRACT,
                        prompt,
                        action_source_checksum(
                            installed_coraplex_root(), action.source_file
                        ),
                        action_source_id=action.source_id,
                    )
                )
        if kind in (None, InitializationKind.REALIZATION):
            jobs.extend(self._realization_jobs(selected, contracts))
        if kind in (None, InitializationKind.GROUNDING):
            covered = {item.uid for item in grounding.catalog}
            covered.update(
                item.proposed_uid
                for item in grounding.workspace.candidates()
                if item.review_status is GroundingFactoryReviewStatus.PENDING_REVIEW
            )
            for needed in requests:
                if needed.proposed_uid not in covered:
                    objections = tuple(
                        f"{item.candidate_id}: {item.review_note or 'Rejected by reviewer'}"
                        for item in grounding.workspace.candidates()
                        if item.review_status is GroundingFactoryReviewStatus.REJECTED
                        and item.proposed_uid == needed.proposed_uid
                    )
                    prompt = grounding_factory_prompt(
                        needed.request(), grounding.reviewed_vocabulary, objections
                    )
                    prompt += (
                        "\n\nComplete parameter constraints:\n"
                        + needed.model_dump_json(indent=2)
                    )
                    jobs.append(
                        _job(
                            InitializationKind.GROUNDING,
                            prompt,
                            _checksum(to_json(grounding.reviewed_vocabulary)),
                            grounding_request=needed,
                        )
                    )
        _write_json(
            self.materials / InitializationFile.JOBS,
            [item.model_dump(mode="json") for item in jobs],
        )
        _write_json(
            self.materials / InitializationFile.PLATFORM,
            {
                "actions": [item.to_json() for item in actions],
                "vocabulary": to_json(grounding.reviewed_vocabulary),
                "approved_contracts": [to_json(item) for item in approved],
            },
        )
        (self.materials / InitializationFile.INSTRUCTIONS).write_text(
            Path(__file__).with_name("initialization_instructions.md").read_text(),
            encoding="utf-8",
        )
        return tuple(jobs)

    def _realization_jobs(
        self,
        actions: tuple[CoraplexCapabilityContractDraft, ...],
        contracts: CapabilityContractWorkspace,
    ) -> list[DraftJob]:
        """Prepare native mappings only for human-approved contract candidates."""
        workspace = CoraplexRealizationWorkspace(self.realization_root)
        covered = {
            (item.action_source_id, item.realization.capability_uid)
            for item in workspace.candidates()
            if item.review_status is CapabilityReviewStatus.PENDING
        }
        covered.update(
            (item.action_source_id, item.realization.capability_uid)
            for item in workspace.approved()
            if workspace.drift(item, installed_coraplex_root()) is None
        )
        by_action = {item.source_id: item for item in actions}
        approved = {
            (item.uid, item.version): item for item in contracts.approved_contracts()
        }
        jobs = []
        for candidate in contracts.candidates():
            if candidate.review_status is not CapabilityReviewStatus.APPROVED:
                continue
            contract = approved.get(
                (candidate.contract.uid, candidate.contract.version)
            )
            if contract is None:
                continue
            for action_id in candidate.action_source_ids:
                if action_id not in by_action or (action_id, contract.uid) in covered:
                    continue
                action = by_action[action_id]
                source = (installed_coraplex_root() / action.source_file).read_text()
                prompt = render_prompt(
                    "draft_realization",
                    contract=json.dumps(to_json(contract), indent=2),
                    action=action.render(),
                    source=source,
                    context_values=", ".join(item.value for item in ContextValue),
                )
                objections = tuple(
                    f"{item.candidate_id}: {item.review_note or 'Rejected by reviewer'}"
                    for item in workspace.candidates()
                    if item.review_status is CapabilityReviewStatus.REJECTED
                    and item.action_source_id == action_id
                    and item.realization.capability_uid == contract.uid
                )
                if objections:
                    prompt += "\n\nReviewer feedback:\n" + "\n".join(objections)
                jobs.append(
                    _job(
                        InitializationKind.REALIZATION,
                        prompt,
                        action_source_checksum(
                            installed_coraplex_root(), action.source_file
                        ),
                        action_source_id=action_id,
                        contract_uid=contract.uid,
                        contract_checksum=_checksum(to_json(contract)),
                    )
                )
        return jobs

    def import_responses(
        self, directory: Path | None = None, generated_by: str = "external-agent"
    ) -> ImportReport:
        """Validate saved replies and submit pending candidates; never execute source."""
        report = ImportReport()
        api_path = self.materials / InitializationFile.API_RESPONSES
        api_responses = json.loads(api_path.read_text()) if api_path.exists() else {}
        for job in self.jobs():
            path = self.response_path(job, directory)
            if not path.is_file():
                report.missing.append(job.job_id)
                continue
            try:
                response = _response_type(job.kind).model_validate_json(
                    path.read_text()
                )
                author = generated_by
                if api_responses.get(job.job_id) == text_checksum(
                    response.model_dump_json()
                ):
                    author = "initialization-api"
                candidate, workspace = self._candidate(job, response, author)
                existing = {item.candidate_id for item in workspace.candidates()}
                if candidate.candidate_id in existing:
                    report.existing.append(candidate.candidate_id)
                else:
                    workspace.submit(candidate)
                    report.submitted.append(candidate.candidate_id)
            except ValueError as error:
                report.failed[job.job_id] = str(error)
        _write_json(self.materials / InitializationFile.REPORT, to_json(report))
        return report

    def _candidate(
        self,
        job: DraftJob,
        response: BaseModel,
        generated_by: str,
    ) -> tuple[
        GroundingFactoryCandidate | CapabilityContractCandidate | RealizationCandidate,
        GroundingFactoryWorkspace
        | CapabilityContractWorkspace
        | CoraplexRealizationWorkspace,
    ]:
        """Construct one candidate through the same validators used by review."""
        candidate_id = f"{job.job_id}-{text_checksum(response.model_dump_json())[:12]}"
        if job.kind is InitializationKind.GROUNDING:
            grounding = self.grounding()
            if _checksum(to_json(grounding.reviewed_vocabulary)) != job.source_checksum:
                raise InvalidInitializationDraft(
                    "Query vocabulary changed; run prepare again."
                )
            candidate = grounding_factory_candidate(
                job.grounding_request.request(), response
            )
            _require_valid(grounding.candidate_objections(candidate))
            return (
                replace(
                    candidate, candidate_id=candidate_id, generated_by=generated_by
                ),
                grounding.workspace,
            )
        actions = {item.source_id: item for item in self.actions()}
        action = actions.get(job.action_source_id)
        if (
            action is None
            or action_source_checksum(installed_coraplex_root(), action.source_file)
            != job.source_checksum
        ):
            raise InvalidInitializationDraft(
                "Native action changed; run prepare again."
            )
        contracts = CapabilityContractWorkspace(self.contract_root)
        approved = contracts.approved_contracts()
        if job.kind is InitializationKind.CONTRACT:
            candidate, objections = capability_contract_candidate(action, response)
            _require_valid(objections)
            # A repeated import remains a no-op after its candidate was approved.
            prior = {item.candidate_id for item in contracts.candidates()}
            if candidate_id not in prior:
                _require_valid(
                    contract_objections(
                        candidate,
                        actions,
                        {(item.uid, item.version): item for item in approved},
                    )
                )
            return (
                replace(
                    candidate, candidate_id=candidate_id, generated_by=generated_by
                ),
                contracts,
            )
        by_uid = {item.uid: item for item in approved}
        contract = by_uid.get(job.contract_uid)
        if contract is None or _checksum(to_json(contract)) != job.contract_checksum:
            raise InvalidInitializationDraft(
                "Approved contract changed; run prepare again."
            )
        candidate = RealizationCandidate(
            candidate_id,
            action.source_id,
            ActionRealization(
                contract.uid,
                tuple(
                    ParameterSource(**item.model_dump())
                    for item in response.parameter_sources
                ),
                (
                    RoleCondition(response.condition_role, response.condition_value)
                    if response.condition_role
                    else None
                ),
                tuple(SymbolType(item) for item in response.required_resources),
            ),
            generated_by,
            response.rationale,
        )
        objections = list(realization_objections(candidate, actions, by_uid))
        parameters = {
            item.parameter for item in candidate.realization.parameter_sources
        }
        declared = {item.name for item in action.parameters}
        required = {item.name for item in action.parameters if item.required}
        if parameters - declared or required - parameters:
            objections.append(
                f"Native parameter mismatch: unknown={sorted(parameters - declared)}, missing={sorted(required - parameters)}"
            )
        if response.condition_value is not None and response.condition_role is None:
            objections.append("condition_value requires condition_role")
        _require_valid(objections)
        return candidate, CoraplexRealizationWorkspace(self.realization_root)

    def draft(
        self, completer: StructuredCompleter, maximum_attempts: int = 3
    ) -> ImportReport:
        """Draft unanswered jobs, feed back validation errors, then use common import."""
        failures = {}
        api_path = self.materials / InitializationFile.API_RESPONSES
        api_responses = json.loads(api_path.read_text()) if api_path.exists() else {}
        for job in self.jobs():
            if self.response_path(job).exists():
                continue
            prompt = job.prompt
            for _ in range(maximum_attempts):
                try:
                    response = completer.complete(
                        f"initialization-{job.kind.value}",
                        prompt,
                        _response_type(job.kind),
                    )
                    self._candidate(job, response, "initialization-api")
                    self.response_path(job).write_text(
                        response.model_dump_json(indent=2), encoding="utf-8"
                    )
                    api_responses[job.job_id] = text_checksum(
                        response.model_dump_json()
                    )
                    _write_json(api_path, api_responses)
                    break
                except (ValueError, StructuredOutputRetriesExceededError) as error:
                    failures[job.job_id] = str(error)
                    prompt = f"{job.prompt}\n\nValidation objections:\n{error}\nCorrect the response."
            else:
                continue
            failures.pop(job.job_id, None)
        report = self.import_responses()
        report.failed.update(failures)
        _write_json(self.materials / InitializationFile.REPORT, to_json(report))
        return report


# %% command line


def _response_type(kind: InitializationKind) -> type[BaseModel]:
    """The fixed output schema for one job kind."""
    return {
        InitializationKind.CONTRACT: CapabilityContractDraftModel,
        InitializationKind.GROUNDING: GroundingFactoryDraftModel,
        InitializationKind.REALIZATION: RealizationDraft,
    }[kind]


def _checksum(value: object) -> str:
    """Stable checksum of JSON evidence."""
    return text_checksum(json.dumps(value, sort_keys=True))


def _job(
    kind: InitializationKind, prompt: str, source_checksum: str, **fields: Any
) -> DraftJob:
    """Create a deterministic job identity from its complete authoring inputs."""
    schema = _response_type(kind).model_json_schema()
    prompt = f"{prompt}\n\nResponse JSON schema:\n{json.dumps(schema, indent=2)}"
    return DraftJob(
        job_id=f"{kind.value}-{text_checksum(prompt + source_checksum)[:16]}",
        kind=kind,
        prompt=prompt,
        response_schema=schema,
        source_checksum=source_checksum,
        **fields,
    )


def _require_valid(objections: Iterable[str]) -> None:
    """Keep invalid candidates outside the review queue."""
    objections = tuple(objections)
    if objections:
        raise InvalidInitializationDraft("; ".join(objections))


def _write_json(path: Path, data: object) -> None:
    """Persist a readable initialization document."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """Run one explicit initialization stage; only draft constructs an API client."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in InitializationCommand:
        subparser = subparsers.add_parser(command.value)
        subparser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "tmp")
        if command is InitializationCommand.PREPARE:
            subparser.add_argument(
                "--requests",
                type=Path,
                help="JSON array of grounding relations; schema exported by prepare",
            )
            subparser.add_argument(
                "--action",
                action="append",
                default=[],
                help="Limit action source ids (repeatable)",
            )
            subparser.add_argument(
                "--kind", type=InitializationKind, choices=list(InitializationKind)
            )
        elif command is InitializationCommand.DRAFT:
            subparser.add_argument("--config", type=Path, required=True)
        else:
            subparser.add_argument("responses", type=Path, nargs="?")
            subparser.add_argument("--generated-by", default="external-agent")
    args = parser.parse_args()
    initialization = Initialization(args.workspace.resolve())
    command = InitializationCommand(args.command)
    if command is InitializationCommand.PREPARE:
        requests = (
            None
            if args.requests is None
            else tuple(
                TypeAdapter(list[GroundingRequest]).validate_json(
                    args.requests.read_text()
                )
            )
        )
        jobs = initialization.prepare(tuple(args.action), requests, args.kind)
        print(
            f"Prepared {len(jobs)} jobs in {initialization.materials}. See instructions.md."
        )
        print(
            "Grounding jobs require declared relations in --requests; rerun prepare after reviewing contracts for realization jobs."
        )
        return
    if command is InitializationCommand.DRAFT:
        configuration = LanguageModelConfiguration.load(args.config)
        completer = StructuredCompleter(
            build_completion_client(configuration),
            TranscriptRecorder(
                initialization.materials / InitializationFile.TRANSCRIPT
            ),
            configuration.structured_maximum_attempts,
        )
        report = initialization.draft(completer)
    else:
        report = initialization.import_responses(args.responses, args.generated_by)
    print(json.dumps(to_json(report), indent=2))
    print(
        f"Pending candidates are ready for Viewer review. Workspaces: {initialization.root}"
    )
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
