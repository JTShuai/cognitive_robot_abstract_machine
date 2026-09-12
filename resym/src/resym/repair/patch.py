"""
Executable adaptation: turning untrusted fragments into typed patches.

A retrieved PDDL fragment never enters the library as text. The adapter constructs a
typed :class:`ModelPatch` from it under an explicit *alignment* — which fragment
predicate maps to which local predicate (or becomes a new one with a reviewed platform
query), how the operator's variables are typed, and which capability contract the
operator binds to. Choosing the alignment is intelligence (the repair agent's job, or
the deterministic name-matching heuristic below for the fixed pipeline); *validating and
constructing* is this module, and it is deterministic.

Adaptation failures are structured (revised plan §6.3): type mismatch, missing predicate
implementation, missing capability contract, effect–contract conflict, unsupported
embodiment capability, duplicate or semantically conflicting symbol. A fragment that
cannot be grounded executably yields issues and unresolved requirements — never a
fabricated implementation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Optional

from resym.knowledge.corpus import (
    DomainFragment,
    OperatorFragment,
    parse_operator_literals,
)
from resym.knowledge.retrieval import tokenize
from resym.core.model import (
    CapabilityContract,
    Literal,
    Operator,
    OperatorExecutionBinding,
    PredicateGroundingPlan,
    PredicateRef,
    PredicateSymbol,
    Provenance,
    RoleBinding,
    SymbolLibrary,
    SymbolType,
    contract_violations,
    is_symbol_subtype,
)


class AdaptationIssueKind(Enum):
    """
    Why a fragment element could not be grounded executably.
    """

    TYPE_MISMATCH = "type_mismatch"
    MISSING_PREDICATE_IMPLEMENTATION = "missing_predicate_implementation"
    MISSING_CAPABILITY_CONTRACT = "missing_capability_contract"
    EFFECT_CONTRACT_CONFLICT = "effect_contract_conflict"
    UNSUPPORTED_EMBODIMENT = "unsupported_embodiment_capability"
    DUPLICATE_OR_CONFLICTING_SYMBOL = "duplicate_or_conflicting_symbol"


@dataclass(frozen=True)
class AdaptationIssue:
    kind: AdaptationIssueKind
    element: str
    """
    Which fragment element (predicate/operator name) the issue is about.
    """

    detail: str

    def render(self) -> str:
        return f"[{self.kind.value}] {self.element}: {self.detail}"


@dataclass(frozen=True)
class ProvenanceEdge:
    """
    One imported element and where it came from — kept per edge so a composed patch can
    cite several fragments.
    """

    element: str
    """
    Local name of the imported predicate or operator.
    """

    fragment_id: str
    source_name: str
    """
    Name of the element inside the source fragment.
    """


@dataclass(frozen=True)
class PatchComplexity:
    """
    The lexicographic minimality objective J (revised plan §3.4).
    """

    modified_symbols: int
    new_symbols: int
    ast_nodes: int
    contract_changes: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.modified_symbols,
            self.new_symbols,
            self.ast_nodes,
            self.contract_changes,
        )


@dataclass(frozen=True)
class ModelPatch:
    """
    A pure increment on the library: the only thing a repair backend may output, and the
    only thing the curator accepts for review.
    """

    predicates: tuple[PredicateSymbol, ...] = ()
    """
    Added or (by name collision) modified predicates.
    """

    operators: tuple[Operator, ...] = ()
    """
    Added or (by name collision) modified operators.
    """

    capability_contracts: tuple[CapabilityContract, ...] = ()
    """
    Capability contract changes require a separate trusted release.
    """

    edges: tuple[ProvenanceEdge, ...] = ()
    """
    Provenance of every imported element.
    """

    unresolved: tuple[str, ...] = ()
    """
    Requirements the patch could not ground (MISSING_IMPLEMENTATION notes); a patch with
    unresolved requirements is a proposal record, not an admission candidate.
    """

    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "operators", tuple(self.operators))
        object.__setattr__(
            self, "capability_contracts", tuple(self.capability_contracts)
        )
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "unresolved", tuple(self.unresolved))

    def complexity(self, base: SymbolLibrary) -> PatchComplexity:
        modified = sum(1 for p in self.predicates if p.name in base.predicates) + sum(
            1 for o in self.operators if o.name in base.operators
        )
        new = len(self.predicates) + len(self.operators) - modified
        ast_nodes = 0
        contract_changes = len(self.capability_contracts)
        return PatchComplexity(modified, new, ast_nodes, contract_changes)

    def apply_to(self, base: SymbolLibrary) -> SymbolLibrary:
        """
        A deep-copied candidate library with the patch applied; the base is never
        mutated — admission commits the candidate, rejection drops it.
        """
        candidate = copy.deepcopy(base)
        for predicate in self.predicates:
            candidate.predicates[predicate.name] = predicate
        for operator in self.operators:
            candidate.operators[operator.name] = operator
        for contract in self.capability_contracts:
            candidate.capability_contracts[contract.uid] = contract
        return candidate


@dataclass(frozen=True)
class PredicateBinding:
    """
    How one fragment predicate lands locally: onto an existing predicate, or as a new
    one carrying a reviewed grounding plan.
    """

    fragment_predicate: str
    local_name: str
    parameter_types: Optional[tuple[SymbolType, ...]] = None
    """
    Required for a new predicate; must be None-compatible with an existing one.
    """

    grounding_plan: Optional[PredicateGroundingPlan] = None
    """Required factory binding when the alignment creates a predicate."""

    fluent: bool = True

    @property
    def creates_new(self) -> bool:
        return self.grounding_plan is not None


@dataclass(frozen=True)
class OperatorAlignment:
    """
    The full alignment the adapter validates: predicate bindings, variable typing, and
    the capability binding.
    """

    fragment_operator: str
    local_name: str
    variable_types: dict[str, SymbolType]
    """
    Type of every ``?var`` the fragment operator uses.
    """

    bindings: tuple[PredicateBinding, ...]
    execution_binding: OperatorExecutionBinding


@dataclass
class AdaptationResult:
    """
    Either a patch or the reasons there is none.
    """

    patch: Optional[ModelPatch] = None
    issues: list[AdaptationIssue] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.patch is not None and not self.issues


def adapt_operator(
    fragment: DomainFragment,
    alignment: OperatorAlignment,
    library: SymbolLibrary,
    available_capabilities: frozenset[str],
    provenance: Optional[Provenance] = None,
) -> AdaptationResult:
    """
    Validate one alignment and construct the typed patch.

    ``available_capabilities`` is what the current embodiment implements. All issue
    kinds are collected instead of short-circuited.
    """
    result = AdaptationResult()
    operator_fragment = _find_operator(fragment, alignment.fragment_operator)
    if operator_fragment is None:
        result.issues.append(
            AdaptationIssue(
                AdaptationIssueKind.DUPLICATE_OR_CONFLICTING_SYMBOL,
                alignment.fragment_operator,
                f"fragment {fragment.fragment_id} has no such operator",
            )
        )
        return result
    literals = parse_operator_literals(operator_fragment.raw)
    bindings = {b.fragment_predicate: b for b in alignment.bindings}

    new_predicates: list[PredicateSymbol] = []
    edges: list[ProvenanceEdge] = []
    unresolved: list[str] = []

    used_predicates = {
        literal.predicate
        for group in (
            literals.preconditions,
            literals.add_effects,
            literals.delete_effects,
        )
        for literal in group
    }
    for name in sorted(used_predicates):
        binding = bindings.get(name)
        if binding is None:
            result.issues.append(
                AdaptationIssue(
                    AdaptationIssueKind.MISSING_PREDICATE_IMPLEMENTATION,
                    name,
                    "no binding provided for this fragment predicate",
                )
            )
            unresolved.append(f"MISSING_IMPLEMENTATION: predicate '{name}'")
            continue
        issue = _validate_binding(binding, library)
        if issue is not None:
            result.issues.append(issue)
            if issue.kind is AdaptationIssueKind.MISSING_PREDICATE_IMPLEMENTATION:
                unresolved.append(f"MISSING_IMPLEMENTATION: predicate '{name}'")
            continue
        if binding.creates_new and binding.local_name not in library.predicates:
            new_predicates.append(
                PredicateSymbol(
                    name=binding.local_name,
                    parameter_types=binding.parameter_types,
                    fluent=binding.fluent,
                    grounding_plan=binding.grounding_plan,
                    provenance=provenance or Provenance(source="curation"),
                )
            )
            edges.append(
                ProvenanceEdge(
                    element=binding.local_name,
                    fragment_id=fragment.fragment_id,
                    source_name=name,
                )
            )

    local_literals = _LocalLiterals()
    for group_name, group in (
        ("precondition", literals.preconditions),
        ("add_effect", literals.add_effects),
        ("delete_effect", literals.delete_effects),
    ):
        for literal in group:
            binding = bindings.get(literal.predicate)
            if binding is None:
                continue  # already reported above
            local = _type_check_literal(
                literal,
                binding,
                alignment,
                library,
                new_predicates,
                result.issues,
            )
            if local is not None:
                local_literals.append(group_name, local)

    _validate_capability(alignment, library, available_capabilities, result.issues)

    if result.issues:
        return result

    parameters = tuple(
        (variable.lstrip("?"), symbol_type)
        for variable, symbol_type in sorted(alignment.variable_types.items())
    )
    operator = Operator(
        name=alignment.local_name,
        parameters=parameters,
        preconditions=tuple(local_literals.preconditions),
        add_effects=tuple(local_literals.add_effects),
        delete_effects=tuple(local_literals.delete_effects),
        execution_binding=alignment.execution_binding,
        provenance=provenance or Provenance(source="curation"),
    )
    capability_uid = alignment.execution_binding.capability_ref.uid
    contract = library.capability_contracts.get(capability_uid)
    if contract is not None:
        candidate_predicates = dict(library.predicates)
        candidate_predicates.update(
            {predicate.name: predicate for predicate in new_predicates}
        )
        for violation in contract_violations(operator, contract, candidate_predicates):
            result.issues.append(
                AdaptationIssue(
                    AdaptationIssueKind.EFFECT_CONTRACT_CONFLICT,
                    alignment.local_name,
                    violation,
                )
            )
    if result.issues:
        return result
    edges.append(
        ProvenanceEdge(
            element=alignment.local_name,
            fragment_id=fragment.fragment_id,
            source_name=alignment.fragment_operator,
        )
    )
    result.patch = ModelPatch(
        predicates=tuple(new_predicates),
        operators=(operator,),
        edges=tuple(edges),
        unresolved=tuple(unresolved),
    )
    return result


# -- deterministic alignment heuristic (the fixed pipeline's brain) -----


def suggest_alignment(
    fragment: DomainFragment,
    operator_name: str,
    library: SymbolLibrary,
    minimum_overlap: float = 0.5,
) -> Optional[OperatorAlignment]:
    """
    Name-matching alignment: map each fragment predicate to the local predicate with the
    highest token overlap (above the threshold), infer variable types from the matched
    signatures, and bind the capability whose contract covers the operator's effects.

    Returns ``None`` when no contract fits; unmatched predicates are left unbound (the
    adapter reports them as missing implementations).
    """
    operator_fragment = _find_operator(fragment, operator_name)
    if operator_fragment is None:
        return None
    literals = parse_operator_literals(operator_fragment.raw)

    bindings = []
    matched: dict[str, PredicateSymbol] = {}
    used = {
        literal.predicate
        for group in (
            literals.preconditions,
            literals.add_effects,
            literals.delete_effects,
        )
        for literal in group
    }
    for name in sorted(used):
        local = _best_name_match(name, library, minimum_overlap)
        if local is not None:
            matched[name] = local
            bindings.append(
                PredicateBinding(fragment_predicate=name, local_name=local.name)
            )

    variable_types: dict[str, SymbolType] = {}
    conflict = False
    for group in (
        literals.preconditions,
        literals.add_effects,
        literals.delete_effects,
    ):
        for literal in group:
            local = matched.get(literal.predicate)
            if local is None or len(literal.arguments) != len(local.parameter_types):
                continue
            for variable, symbol_type in zip(literal.arguments, local.parameter_types):
                if variable_types.setdefault(variable, symbol_type) != symbol_type:
                    conflict = True
    if conflict:
        return None

    effect_names = {
        matched[literal.predicate].name
        for literal in (*literals.add_effects, *literals.delete_effects)
        if literal.predicate in matched
    }
    added_effect_names = {
        matched[literal.predicate].name
        for literal in literals.add_effects
        if literal.predicate in matched
    }
    contract = _best_contract(effect_names, library)
    if contract is None:
        return None
    execution_binding = _infer_execution_binding(
        contract, variable_types, added_effect_names, library
    )
    if execution_binding is None:
        return None
    return OperatorAlignment(
        fragment_operator=operator_name,
        local_name=operator_name.replace("_", "-"),
        variable_types=variable_types,
        bindings=tuple(bindings),
        execution_binding=execution_binding,
    )


def contract_scorer_for(library: SymbolLibrary):
    """
    The retrieval rerank hook: how well a fragment's operator effects overlap some
    declared contract's verifiable effects, in [0, 1].
    """
    contract_tokens = [
        {token for name in contract.verifiable_effect_names for token in tokenize(name)}
        for contract in library.capability_contracts.values()
    ]

    def score(query, fragment: DomainFragment) -> float:
        if not contract_tokens or not fragment.operators:
            return 0.0
        best = 0.0
        for operator in fragment.operators:
            effect_tokens = {
                t
                for name in operator.add_predicates + operator.delete_predicates
                for t in tokenize(name)
            }
            if not effect_tokens:
                continue
            for tokens in contract_tokens:
                if not tokens:
                    continue
                overlap = len(effect_tokens & tokens) / len(effect_tokens | tokens)
                best = max(best, overlap)
        return best

    return score


# -- internals ----------------------------------------------------------


@dataclass
class _LocalLiterals:
    preconditions: list[Literal] = field(default_factory=list)
    add_effects: list[Literal] = field(default_factory=list)
    delete_effects: list[Literal] = field(default_factory=list)

    def append(self, group: str, literal: Literal) -> None:
        {
            "precondition": self.preconditions,
            "add_effect": self.add_effects,
            "delete_effect": self.delete_effects,
        }[group].append(literal)


def _find_operator(fragment: DomainFragment, name: str) -> Optional[OperatorFragment]:
    for operator in fragment.operators:
        if operator.name == name:
            return operator
    return None


def _validate_binding(
    binding: PredicateBinding,
    library: SymbolLibrary,
) -> Optional[AdaptationIssue]:
    existing = library.predicates.get(binding.local_name)
    if binding.creates_new:
        if existing is not None:
            return AdaptationIssue(
                AdaptationIssueKind.DUPLICATE_OR_CONFLICTING_SYMBOL,
                binding.fragment_predicate,
                f"binding creates '{binding.local_name}' but that predicate "
                "already exists",
            )
        if binding.parameter_types is None:
            return AdaptationIssue(
                AdaptationIssueKind.TYPE_MISMATCH,
                binding.fragment_predicate,
                "a new predicate needs parameter types",
            )
        return None
    if existing is None:
        return AdaptationIssue(
            AdaptationIssueKind.MISSING_PREDICATE_IMPLEMENTATION,
            binding.fragment_predicate,
            f"binding targets '{binding.local_name}' which does not exist and "
            "provides no implementation",
        )
    return None


def _type_check_literal(
    literal,
    binding: PredicateBinding,
    alignment: OperatorAlignment,
    library: SymbolLibrary,
    new_predicates: list[PredicateSymbol],
    issues: list[AdaptationIssue],
) -> Optional[Literal]:
    declared = library.predicates.get(binding.local_name) or next(
        (p for p in new_predicates if p.name == binding.local_name), None
    )
    if declared is None:
        return None  # binding already reported
    if len(literal.arguments) != len(declared.parameter_types):
        issues.append(
            AdaptationIssue(
                AdaptationIssueKind.TYPE_MISMATCH,
                literal.predicate,
                f"used with {len(literal.arguments)} arguments, "
                f"'{declared.name}' takes {len(declared.parameter_types)}",
            )
        )
        return None
    variables = []
    for position, (variable, expected) in enumerate(
        zip(literal.arguments, declared.parameter_types)
    ):
        actual = alignment.variable_types.get(variable)
        if actual is None:
            issues.append(
                AdaptationIssue(
                    AdaptationIssueKind.TYPE_MISMATCH,
                    literal.predicate,
                    f"variable '{variable}' has no declared type",
                )
            )
            return None
        if not is_symbol_subtype(actual, expected):
            issues.append(
                AdaptationIssue(
                    AdaptationIssueKind.TYPE_MISMATCH,
                    literal.predicate,
                    f"position {position}: variable '{variable}' is "
                    f"{actual.python_type_ref}, '{declared.name}' expects "
                    f"{expected.python_type_ref}",
                )
            )
            return None
        variables.append(variable.lstrip("?"))
    return Literal(
        predicate=declared.name,
        arguments=tuple(variables),
        negated=literal.negated,
    )


def _validate_capability(
    alignment: OperatorAlignment,
    library: SymbolLibrary,
    available_capabilities: frozenset[str],
    issues: list[AdaptationIssue],
) -> None:
    capability_uid = alignment.execution_binding.capability_ref.uid
    if capability_uid not in available_capabilities:
        issues.append(
            AdaptationIssue(
                AdaptationIssueKind.UNSUPPORTED_EMBODIMENT,
                alignment.local_name,
                f"capability '{capability_uid}' is not implemented by the current "
                "embodiment",
            )
        )
    if capability_uid not in library.capability_contracts:
        issues.append(
            AdaptationIssue(
                AdaptationIssueKind.MISSING_CAPABILITY_CONTRACT,
                alignment.local_name,
                f"no contract declared for capability '{capability_uid}'",
            )
        )


def _best_name_match(
    fragment_name: str, library: SymbolLibrary, minimum_overlap: float
) -> Optional[PredicateSymbol]:
    fragment_tokens = set(tokenize(fragment_name))
    if not fragment_tokens:
        return None
    best: Optional[PredicateSymbol] = None
    best_score = 0.0
    for predicate in library.predicates.values():
        local_tokens = set(tokenize(predicate.name))
        if not local_tokens:
            continue
        score = len(fragment_tokens & local_tokens) / len(
            fragment_tokens | local_tokens
        )
        if score > best_score:
            best, best_score = predicate, score
    return best if best_score >= minimum_overlap else None


def _best_contract(
    effect_names: set[str], library: SymbolLibrary
) -> Optional[CapabilityContract]:
    if not effect_names:
        return None
    effect_refs = {
        (
            library.predicates[name].ref
            if name in library.predicates
            else PredicateRef.from_name(name)
        )
        for name in effect_names
    }
    for _, contract in sorted(library.capability_contracts.items()):
        if effect_refs <= set(contract.verifiable_effects):
            return contract
    return None


def _infer_execution_binding(
    contract: CapabilityContract,
    variable_types: dict[str, SymbolType],
    added_effect_names: set[str],
    library: SymbolLibrary,
) -> Optional[OperatorExecutionBinding]:
    """
    Construct the only unambiguous role mapping supported by the local typed vocabulary;
    ambiguous mappings remain unresolved.
    """
    role_bindings: list[tuple[str, RoleBinding]] = []
    for role in contract.roles:
        candidates = [
            variable.lstrip("?")
            for variable, symbol_type in variable_types.items()
            if any(
                is_symbol_subtype(symbol_type, accepted)
                for accepted in role.accepted_symbol_types
            )
        ]
        if len(candidates) == 1:
            role_bindings.append((role.name, RoleBinding.parameter(candidates[0])))
            continue
        constant_value = contract.constant_role_value(
            role.name,
            {
                (
                    library.predicates[name].ref
                    if name in library.predicates
                    else PredicateRef.from_name(name)
                )
                for name in added_effect_names
            },
        )
        if constant_value is not None:
            role_bindings.append((role.name, RoleBinding.constant(constant_value)))
            continue
        if role.required:
            return None
    return OperatorExecutionBinding(contract.ref, tuple(role_bindings), "retrieval")
