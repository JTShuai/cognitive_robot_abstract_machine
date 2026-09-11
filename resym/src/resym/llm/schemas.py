"""
Pydantic schemas of everything a language model may hand back.

These are the only Pydantic types in the project: they exist to validate model
output at the boundary and convert immediately into the dataclass model of
:mod:`resym.core.model`. Conversion assumes the gate has already
checked names and types; it raises on anything a gate would have rejected.
"""

from __future__ import annotations

import re
from typing import Literal as TypingLiteral

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Optional

from resym.core.model import (
    GROUNDING_PLAN_EVALUATOR_KEY,
    BindingSource,
    CapabilityRef,
    Literal,
    Operator,
    OperatorExecutionBinding,
    PredicateGroundingPlan,
    PredicateSymbol,
    Provenance,
    RoleBinding,
    SymbolType,
)
from resym.platform.krrood_queries import named_colors


def _known_type(value: str) -> str:
    """
    Reject malformed identifiers; catalog membership is checked by Curator.
    """
    SymbolType(value)
    return value


class LiteralModel(BaseModel):
    """
    A predicate applied to variables or object names, possibly negated.
    """

    predicate: str
    arguments: list[str]
    negated: bool = False

    def to_literal(self) -> Literal:
        return Literal(
            predicate=self.predicate,
            arguments=tuple(self.arguments),
            negated=self.negated,
        )


class ParameterModel(BaseModel):
    """
    One typed operator parameter.
    """

    variable: str
    type: str

    @field_validator("type")
    @classmethod
    def _type_is_known(cls, value: str) -> str:
        return _known_type(value)


class LiteralEdits(BaseModel):
    """
    Literal-level changes to one existing operator field.
    """

    add: list[LiteralModel] = []
    remove: list[LiteralModel] = []

    @model_validator(mode="after")
    def additions_and_removals_do_not_overlap(self) -> LiteralEdits:
        additions = {literal.to_literal() for literal in self.add}
        removals = {literal.to_literal() for literal in self.remove}
        if additions & removals:
            raise ValueError("the same literal cannot be both added and removed")
        return self


class GroundingPlanModel(BaseModel):
    """
    Structured binding to one already-approved grounding factory.
    """

    model_config = ConfigDict(extra="forbid")

    factory_uid: str = Field(min_length=1)
    approved_factory_checksum: str = Field(min_length=1)
    role_bindings: dict[str, int] = Field(default_factory=dict)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    negated: bool = False
    version: str = "1"

    @field_validator("role_bindings")
    @classmethod
    def _argument_positions_are_non_negative(
        cls, value: dict[str, int]
    ) -> dict[str, int]:
        if any(position < 0 for position in value.values()):
            raise ValueError("grounding role positions must be non-negative")
        return value

    def to_plan(self) -> PredicateGroundingPlan:
        """
        Convert model output into the persistent reviewed-plan record.
        """
        return PredicateGroundingPlan(
            factory_uid=self.factory_uid,
            approved_factory_checksum=self.approved_factory_checksum,
            role_bindings=tuple(sorted(self.role_bindings.items())),
            parameters=tuple(sorted(self.parameters.items())),
            negated=self.negated,
            version=self.version,
        )


class PredicateProposal(BaseModel):
    """
    A predicate symbol the model wants to add to the library.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    parameter_types: list[str]
    evaluator: Optional[str] = None
    fluent: bool
    grounding_plan: Optional[GroundingPlanModel] = None
    uid: Optional[str] = None
    """
    Stable semantic identity when the predicate realizes a known ref (for example a
    contract's verifiable effect); derived from ``name`` when omitted.
    """

    version: str = "1"

    @model_validator(mode="after")
    def has_one_grounding_implementation(self) -> PredicateProposal:
        if self.evaluator is None and self.grounding_plan is None:
            raise ValueError("predicate needs an evaluator or grounding_plan")
        if self.evaluator is not None and self.grounding_plan is not None:
            raise ValueError(
                "predicate cannot use evaluator and grounding_plan together"
            )
        return self

    @field_validator("parameter_types")
    @classmethod
    def _types_are_known(cls, value: list[str]) -> list[str]:
        return [_known_type(item) for item in value]

    @field_validator("uid")
    @classmethod
    def _uid_is_compact(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not re.fullmatch(r"\S+", value):
            raise ValueError("uid must be a non-empty identifier without spaces")
        return value

    def to_predicate_symbol(self) -> PredicateSymbol:
        return PredicateSymbol(
            name=self.name,
            parameter_types=tuple(SymbolType(t) for t in self.parameter_types),
            evaluator=self.evaluator or GROUNDING_PLAN_EVALUATOR_KEY,
            fluent=self.fluent,
            uid=self.uid,
            version=self.version,
            grounding_plan=(
                self.grounding_plan.to_plan()
                if self.grounding_plan is not None
                else None
            ),
        )


class OperatorProposal(BaseModel):
    """
    A complete new operator or a partial update to an existing operator.
    """

    name: str
    parameters: Optional[list[ParameterModel]] = None
    parameter_type_edits: Optional[dict[str, str]] = None
    preconditions: Optional[list[LiteralModel]] = None
    add_effects: Optional[list[LiteralModel]] = None
    delete_effects: Optional[list[LiteralModel]] = None
    precondition_edits: Optional[LiteralEdits] = None
    add_effect_edits: Optional[LiteralEdits] = None
    delete_effect_edits: Optional[LiteralEdits] = None
    capability_uid: Optional[str] = None
    capability_version: Optional[str] = None
    role_bindings: Optional[dict[str, str]] = None
    constant_bindings: Optional[dict[str, str]] = None

    @model_validator(mode="after")
    def replacements_and_edits_are_exclusive(self) -> OperatorProposal:
        if self.parameters is not None and self.parameter_type_edits is not None:
            raise ValueError(
                "'parameters' cannot be combined with 'parameter_type_edits'"
            )
        pairs = (
            ("preconditions", self.preconditions, self.precondition_edits),
            ("add_effects", self.add_effects, self.add_effect_edits),
            ("delete_effects", self.delete_effects, self.delete_effect_edits),
        )
        for field_name, replacement, edits in pairs:
            if replacement is not None and edits is not None:
                raise ValueError(
                    f"'{field_name}' cannot be combined with literal-level edits"
                )
        return self

    def to_operator(self, existing: Optional[Operator] = None) -> Operator:
        """
        Overlay supplied fields on an existing operator.

        New operators still require a complete definition. The persisted
        :class:`ModelPatch` always receives the resulting complete operator.
        """
        required = {
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "add_effects": self.add_effects,
            "delete_effects": self.delete_effects,
            "capability_uid": self.capability_uid,
            "role_bindings": self.role_bindings,
        }
        if existing is None:
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"new operator '{self.name}' requires complete fields: "
                    + ", ".join(missing)
                )

        if self.parameters is not None:
            parameters = tuple(
                (parameter.variable, SymbolType(parameter.type))
                for parameter in self.parameters
            )
        elif self.parameter_type_edits is not None:
            existing_names = {name for name, _ in existing.parameters}
            unknown = sorted(set(self.parameter_type_edits) - existing_names)
            if unknown:
                raise ValueError(
                    "operator parameter type edits reference unknown parameters: "
                    + ", ".join(unknown)
                )
            edits = {
                name: SymbolType(symbol_type)
                for name, symbol_type in self.parameter_type_edits.items()
            }
            parameters = tuple(
                (name, edits.get(name, symbol_type))
                for name, symbol_type in existing.parameters
            )
        else:
            parameters = existing.parameters
        preconditions = _updated_literals(
            existing.preconditions if existing else (),
            self.preconditions,
            self.precondition_edits,
        )
        add_effects = _updated_literals(
            existing.add_effects if existing else (),
            self.add_effects,
            self.add_effect_edits,
        )
        delete_effects = _updated_literals(
            existing.delete_effects if existing else (),
            self.delete_effects,
            self.delete_effect_edits,
        )

        binding_changed = any(
            value is not None
            for value in (
                self.capability_uid,
                self.capability_version,
                self.role_bindings,
                self.constant_bindings,
            )
        )
        if not binding_changed and existing is not None:
            execution_binding = existing.execution_binding
        else:
            existing_binding = existing.execution_binding if existing else None
            capability_uid = self.capability_uid or (
                existing_binding.capability_ref.uid if existing_binding else None
            )
            if capability_uid is None:
                raise ValueError(f"new operator '{self.name}' requires capability_uid")
            capability_version = self.capability_version or (
                existing_binding.capability_ref.version if existing_binding else "1"
            )
            parameter_roles = {
                role: binding.value
                for role, binding in (
                    existing_binding.role_bindings if existing_binding else ()
                )
                if binding.source is BindingSource.PARAMETER
            }
            constant_roles = {
                role: binding.value
                for role, binding in (
                    existing_binding.role_bindings if existing_binding else ()
                )
                if binding.source is BindingSource.CONSTANT
            }
            if self.role_bindings is not None:
                parameter_roles = self.role_bindings
                for role in parameter_roles:
                    constant_roles.pop(role, None)
            if self.constant_bindings is not None:
                constant_roles = self.constant_bindings
                for role in constant_roles:
                    parameter_roles.pop(role, None)
            execution_binding = OperatorExecutionBinding(
                CapabilityRef(capability_uid, capability_version),
                tuple(
                    (role, RoleBinding(BindingSource.PARAMETER, parameter))
                    for role, parameter in parameter_roles.items()
                )
                + tuple(
                    (role, RoleBinding(BindingSource.CONSTANT, value))
                    for role, value in constant_roles.items()
                ),
                proposal_source="llm",
            )

        return Operator(
            name=self.name,
            parameters=parameters,
            preconditions=preconditions,
            add_effects=add_effects,
            delete_effects=delete_effects,
            execution_binding=execution_binding,
            provenance=existing.provenance if existing else Provenance(),
        )


def _updated_literals(
    current: tuple[Literal, ...],
    replacement: Optional[list[LiteralModel]],
    edits: Optional[LiteralEdits],
) -> tuple[Literal, ...]:
    if replacement is not None:
        return tuple(literal.to_literal() for literal in replacement)
    if edits is None:
        return current
    removals = {literal.to_literal() for literal in edits.remove}
    missing = removals - set(current)
    if missing:
        names = ", ".join(
            sorted(f"{literal.predicate}{literal.arguments}" for literal in missing)
        )
        raise ValueError(f"cannot remove literals not present in the operator: {names}")
    result = [literal for literal in current if literal not in removals]
    for literal_model in edits.add:
        literal = literal_model.to_literal()
        if literal not in result:
            result.append(literal)
    return tuple(result)


class LibraryProposal(BaseModel):
    """
    One Stage-B proposal: new symbols plus the model's reasoning.

    The symbol lists default to empty: models routinely omit a list they
    have nothing to put in (an operator-only patch, say), and that is a
    valid proposal, not a schema violation.
    """

    rationale: str
    predicates: list[PredicateProposal] = []
    operators: list[OperatorProposal] = []


class UnresolvedGoalModel(BaseModel):
    """
    A desired relation that the listed local predicates cannot express.
    """

    suggested_predicate: str
    arguments: list[str]
    description: str
    negated: bool = False

    def to_literal(self) -> Literal:
        return Literal(
            predicate=self.suggested_predicate,
            arguments=tuple(self.arguments),
            negated=self.negated,
        )


class ObjectQueryModel(BaseModel):
    """
    One symbolic object reference to resolve through the krrood query adapter.
    """

    model_config = ConfigDict(extra="forbid")

    reference: str
    type: Optional[str] = None
    color: Optional[str] = None
    name_contains: Optional[str] = None

    @field_validator("reference")
    @classmethod
    def _reference_is_an_alias(cls, value: str) -> str:
        if not re.fullmatch(r"\$[a-z][a-z0-9_-]*", value):
            raise ValueError("object query reference must look like '$target-object'")
        return value

    @field_validator("type")
    @classmethod
    def _optional_type_is_known(cls, value: Optional[str]) -> Optional[str]:
        return _known_type(value) if value is not None else None

    @field_validator("color")
    @classmethod
    def _optional_color_is_named(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.lower()
        if normalized not in named_colors():
            raise ValueError(
                "color must be one of: " + ", ".join(sorted(named_colors()))
            )
        return normalized


class GoalTranslation(BaseModel):
    """
    A natural-language instruction resolved against the local model.

    The default keeps the original ``{"literals": [...]}`` reply compatible.
    ``model_gap`` is reserved for a clear task whose required relation is not present in
    the predicate menu; ``clarification_needed`` must not enter the planner or repair
    loop.
    """

    status: TypingLiteral["ready", "model_gap", "clarification_needed"] = "ready"
    object_queries: list[ObjectQueryModel] = []
    literals: list[LiteralModel] = []
    unresolved_literals: list[UnresolvedGoalModel] = []
    message: str = ""


class OntologyAlignmentProposal(BaseModel):
    """
    An ontology candidate proposed by an LLM, before local admission.
    """

    candidate_iri: Optional[str] = None
    relation: TypingLiteral["EXACT_MATCH", "SPECIALIZATION", "RELATED", "NO_MATCH"]
    role_mapping: dict[str, str] = {}
    cited_classes: list[str] = []
    cited_properties: list[str] = []


class OntologyAgentAction(BaseModel):
    """
    One bounded ontology-alignment agent action.
    """

    tool: TypingLiteral["search", "inspect_entity", "submit_alignment"]
    query: Optional[str] = None
    iri: Optional[str] = None
    proposal: Optional[OntologyAlignmentProposal] = None
    rationale: Optional[str] = None


class AgentActionModel(BaseModel):
    """
    One planning-model agent step: exactly one whitelisted tool call.
    """

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1)
    arguments: dict = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")
