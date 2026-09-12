"""
The trusted curator: deterministic static admission.

The repair agent explores; the curator decides. They are permission-separated: backends
may *consult* the static review as a read-only tool, but at submission the curator
independently re-runs every mandatory check on the patch — nothing an agent observed,
skipped, or phrased can substitute for that. The curator is deterministic program logic;
no language model participates in admission.

Static review (revised plan §6.6): typing and variable binding, registry whitelists,
fluent-only effects, no add∧delete of the same literal, effect–contract consistency,
embodiment support, no unresolved requirements. Persisted expressions cannot smuggle
scene state: operator literals may only reference operator parameters (checked here),
and every predicate must reference a reviewed platform query with a compatible
signature.

Task behaviour and historical regression are evaluated outside this generic runtime
component. The ICRA harness provides its own experimental validation layer; a new task
does not have to supply hand-written positive and negative examples merely to use the
curator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from typing_extensions import Mapping, Optional

from krrood.adapters.json_serializer import to_json
from resym.repair.patch import ModelPatch
from resym.core.model import (
    CapabilityContract,
    GroundingFactorySpec,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
    SymbolType,
    contract_violations,
    is_symbol_subtype,
    resolve_symbol_type,
)
from resym.repair.versioning import VersionedLibraryStore


@dataclass
class AdmissionReport:
    """
    The curator's deterministic static verdict.
    """

    static_objections: list[str] = field(default_factory=list)

    @property
    def admitted(self) -> bool:
        return not self.static_objections

    def evidence(self) -> dict:
        return {"static_objections": list(self.static_objections)}


@dataclass
class Curator:
    """
    Deterministic static admission authority over one library.
    """

    available_capabilities: frozenset[str]
    allowed_symbol_types: frozenset[SymbolType] = frozenset()
    capability_catalog: tuple[CapabilityContract, ...] = ()
    """
    Read-only platform contracts that an admitted binding may reference.
    """

    grounding_factory_specs: Mapping[str, GroundingFactorySpec] = field(
        default_factory=dict
    )
    """
    Current reviewed grounding factories accepted by predicate plans.
    """

    def static_review(self, patch: ModelPatch, library: SymbolLibrary) -> list[str]:
        """
        The full static checklist; also exposed to backends as their read-only
        ``check_patch`` tool.

        The curator never trusts a backend's claim of having run it.
        """
        return static_objections(
            patch,
            library,
            self.available_capabilities,
            self.allowed_symbol_types,
            self.capability_catalog,
            self.grounding_factory_specs,
        )

    def review(
        self,
        patch: ModelPatch,
        library: SymbolLibrary,
        proposal_context_id: str | None = None,
    ) -> AdmissionReport:
        """
        Return the complete static verdict; the context is audit metadata.
        """
        return AdmissionReport(static_objections=self.static_review(patch, library))

    def candidate(self, patch: ModelPatch, library: SymbolLibrary) -> SymbolLibrary:
        """
        Apply a statically accepted patch and pin catalog contracts it uses.
        """
        return _candidate_with_catalog_contracts(
            patch, library, self.capability_catalog
        )

    def admit(
        self,
        patch: ModelPatch,
        library: SymbolLibrary,
        store: VersionedLibraryStore,
        proposal_context_id: str,
        metadata: Optional[dict] = None,
    ) -> tuple[AdmissionReport, Optional[str]]:
        """
        Review and, on admission, commit the candidate as a new library version carrying
        the complete evidence.

        Returns the report and the new version id (``None`` when refused).
        """
        report = self.review(patch, library, proposal_context_id)
        if not report.admitted:
            return report, None
        candidate = self.candidate(patch, library)
        version_id = store.commit(
            candidate,
            metadata={
                **(metadata or {}),
                "proposal_context_id": proposal_context_id,
                "patch": to_json(patch),
                "admission_evidence": report.evidence(),
            },
        )
        return report, version_id


def static_objections(
    patch: ModelPatch,
    library: SymbolLibrary,
    available_capabilities: frozenset[str],
    allowed_symbol_types: frozenset[SymbolType] = frozenset(),
    capability_catalog: tuple[CapabilityContract, ...] = (),
    grounding_factory_specs: Mapping[str, GroundingFactorySpec] | None = None,
) -> list[str]:
    """
    Every static reason the patch may not be admitted; empty means clean.
    """
    objections: list[str] = []
    grounding_factory_specs = grounding_factory_specs or {}
    catalog = set(library.symbol_types)
    catalog.update(allowed_symbol_types)
    catalog.update(
        symbol_type
        for contract in capability_catalog
        for role in contract.roles
        for symbol_type in role.accepted_symbol_types
    )
    catalog.update(
        role.symbol_type
        for specification in grounding_factory_specs.values()
        for role in specification.roles
    )
    allowed_type_refs = frozenset(
        symbol_type.python_type_ref for symbol_type in catalog
    )

    if patch.unresolved:
        objections.append(
            "patch carries unresolved requirements and is not admission-ready: "
            + "; ".join(patch.unresolved)
        )

    if patch.capability_contracts:
        objections.append(
            "repair patches cannot modify trusted capability contracts; contract "
            "updates require a separately reviewed implementation release"
        )

    known_predicates = dict(library.predicates)
    seen_in_patch: set[str] = set()
    for predicate in patch.predicates:
        subject = f"predicate '{predicate.name}'"
        if predicate.name in seen_in_patch:
            objections.append(f"{subject}: declared twice in the patch")
        seen_in_patch.add(predicate.name)
        type_objections, invalid_types = _symbol_type_objections(
            subject, predicate.parameter_types, allowed_type_refs
        )
        objections.extend(type_objections)
        objections.extend(
            _grounding_plan_objections(
                predicate, grounding_factory_specs, invalid_types
            )
        )
        known_predicates[predicate.name] = predicate

    contracts = dict(library.capability_contracts)
    for contract in capability_catalog:
        existing = contracts.get(contract.uid)
        if existing is None:
            contracts[contract.uid] = contract
        elif existing.version == contract.version and existing != contract:
            objections.append(
                f"capability catalog conflicts with task contract "
                f"'{contract.uid}' v{contract.version}"
            )

    seen_operators: set[str] = set()
    for operator in patch.operators:
        subject = f"operator '{operator.name}'"
        if operator.name in seen_operators:
            objections.append(f"{subject}: declared twice in the patch")
        seen_operators.add(operator.name)
        objections.extend(
            _operator_objections(
                operator,
                subject,
                known_predicates,
                contracts,
                available_capabilities,
                allowed_type_refs,
            )
        )
    return objections


def _grounding_plan_objections(
    predicate: PredicateSymbol,
    specifications: Mapping[str, GroundingFactorySpec],
    invalid_types: bool,
) -> list[str]:
    """
    Validate one predicate-to-factory plan without executing its query.
    """
    plan = predicate.grounding_plan
    subject = f"predicate '{predicate.name}'"
    specification = specifications.get(plan.factory_uid)
    if specification is None:
        return [f"{subject}: unknown grounding factory '{plan.factory_uid}'"]
    objections: list[str] = []
    if plan.approved_factory_checksum != specification.implementation_checksum:
        objections.append(
            f"{subject}: grounding factory checksum does not match the approved "
            f"implementation of '{plan.factory_uid}'"
        )
    bindings = dict(plan.role_bindings)
    if len(bindings) != len(plan.role_bindings):
        objections.append(f"{subject}: grounding role is bound more than once")
    expected_roles = {role.name for role in specification.roles}
    if bindings and set(bindings) != expected_roles:
        objections.append(
            f"{subject}: grounding roles must be exactly {sorted(expected_roles)}"
        )
    if not bindings and len(predicate.parameter_types) != len(specification.roles):
        objections.append(
            f"{subject}: positional grounding expects {len(specification.roles)} "
            f"arguments, symbol declares {len(predicate.parameter_types)}"
        )
    if not invalid_types:
        for position, role in enumerate(specification.roles):
            argument_index = bindings.get(role.name, position)
            if argument_index < 0 or argument_index >= len(predicate.parameter_types):
                objections.append(
                    f"{subject}: role '{role.name}' maps to invalid argument "
                    f"index {argument_index}"
                )
                continue
            declared = predicate.parameter_types[argument_index]
            if not is_symbol_subtype(declared, role.symbol_type):
                objections.append(
                    f"{subject}: role '{role.name}' expects "
                    f"{role.symbol_type.python_type_ref}, symbol argument "
                    f"{argument_index} declares {declared.python_type_ref}"
                )
    supplied_parameters = dict(plan.parameters)
    if len(supplied_parameters) != len(plan.parameters):
        objections.append(f"{subject}: grounding parameter is set more than once")
    parameter_specs = {
        parameter.name: parameter for parameter in specification.parameters
    }
    unknown_parameters = sorted(set(supplied_parameters) - set(parameter_specs))
    if unknown_parameters:
        objections.append(
            f"{subject}: unknown grounding parameters {unknown_parameters}"
        )
    missing_parameters = sorted(
        name
        for name, parameter in parameter_specs.items()
        if parameter.required and name not in supplied_parameters
    )
    if missing_parameters:
        objections.append(
            f"{subject}: missing required grounding parameters {missing_parameters}"
        )
    for name, value in supplied_parameters.items():
        parameter = parameter_specs.get(name)
        if parameter is not None and not parameter.accepts(value):
            objections.append(
                f"{subject}: grounding parameter '{name}' has an invalid "
                f"value {value!r}"
            )
    return objections


def _candidate_with_catalog_contracts(
    patch: ModelPatch,
    library: SymbolLibrary,
    capability_catalog: tuple[CapabilityContract, ...],
) -> SymbolLibrary:
    """
    Apply a patch and pin each referenced, already-reviewed contract.
    """
    candidate = patch.apply_to(library)
    catalog = {contract.uid: contract for contract in capability_catalog}
    for operator in patch.operators:
        reference = operator.execution_binding.capability_ref
        if reference.uid in candidate.capability_contracts:
            continue
        contract = catalog.get(reference.uid)
        if contract is not None and contract.ref == reference:
            candidate.capability_contracts[contract.uid] = contract
    return candidate


def _operator_objections(
    operator: Operator,
    subject: str,
    known_predicates: dict,
    contracts: dict,
    available_capabilities: frozenset[str],
    allowed_type_refs: frozenset[str],
) -> list[str]:
    objections: list[str] = []
    parameter_types = dict(operator.parameters)
    type_objections, invalid_types = _symbol_type_objections(
        subject, tuple(parameter_types.values()), allowed_type_refs
    )
    objections.extend(type_objections)

    capability_uid = operator.execution_binding.capability_ref.uid
    if capability_uid not in available_capabilities:
        objections.append(
            f"{subject}: capability '{capability_uid}' is not implemented by the "
            "current embodiment"
        )
    contract = contracts.get(capability_uid)
    if contract is None:
        objections.append(
            f"{subject}: no contract declared for capability '{capability_uid}'"
        )
    elif not invalid_types:
        objections.extend(
            f"{subject}: {violation}"
            for violation in contract_violations(operator, contract, known_predicates)
        )
    for literal in (
        *operator.preconditions,
        *operator.add_effects,
        *operator.delete_effects,
    ):
        predicate = known_predicates.get(literal.predicate)
        if predicate is None:
            objections.append(f"{subject}: unknown predicate '{literal.predicate}'")
            continue
        if len(literal.arguments) != len(predicate.parameter_types):
            objections.append(
                f"{subject}: '{literal.predicate}' takes "
                f"{len(predicate.parameter_types)} arguments, got "
                f"{len(literal.arguments)}"
            )
            continue
        for argument, expected in zip(literal.arguments, predicate.parameter_types):
            if argument not in parameter_types:
                objections.append(
                    f"{subject}: literal argument '{argument}' is not an "
                    "operator parameter (instance names cannot enter the "
                    "persistent library)"
                )
            elif (
                parameter_types[argument] not in invalid_types
                and expected.python_type_ref in allowed_type_refs
                and not is_symbol_subtype(parameter_types[argument], expected)
            ):
                objections.append(
                    f"{subject}: variable '{argument}' is "
                    f"'{parameter_types[argument].python_type_ref}', "
                    f"'{literal.predicate}' expects '{expected.python_type_ref}'"
                )

    for effect in (*operator.add_effects, *operator.delete_effects):
        predicate = known_predicates.get(effect.predicate)
        if predicate is not None and not predicate.fluent:
            objections.append(
                f"{subject}: effect on non-fluent predicate '{effect.predicate}'"
            )
    added = {(literal.predicate, literal.arguments) for literal in operator.add_effects}
    deleted = {
        (literal.predicate, literal.arguments) for literal in operator.delete_effects
    }
    for atom in sorted(added & deleted):
        objections.append(
            f"{subject}: adds and deletes the same literal {atom[0]}{atom[1]}"
        )
    return objections


def _symbol_type_objections(
    subject: str,
    symbol_types: tuple[SymbolType, ...],
    allowed_type_refs: frozenset[str],
) -> tuple[list[str], frozenset[SymbolType]]:
    objections: list[str] = []
    invalid: set[SymbolType] = set()
    for symbol_type in dict.fromkeys(symbol_types):
        reference = symbol_type.python_type_ref
        if reference not in allowed_type_refs:
            suggestion = get_close_matches(
                reference, sorted(allowed_type_refs), n=1, cutoff=0.6
            )
            message = (
                f"{subject}: UNKNOWN_SYMBOL_TYPE: unknown symbol type '{reference}'"
            )
            if suggestion:
                message += f"; closest allowed type is '{suggestion[0]}'"
            objections.append(message)
            invalid.add(symbol_type)
            continue
        try:
            resolve_symbol_type(symbol_type)
        except (ImportError, AttributeError, TypeError) as error:
            objections.append(
                f"{subject}: allowed symbol type '{reference}' cannot be resolved "
                f"({type(error).__name__}: {error})"
            )
            invalid.add(symbol_type)
    return objections, frozenset(invalid)


def _types(types) -> str:
    return "(" + ", ".join(kind.python_type_ref for kind in types) + ")"
