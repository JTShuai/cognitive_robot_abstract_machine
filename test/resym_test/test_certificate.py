"""
Programmatic failure classification: the taxonomy is decided by evidence, never by
wording.

Container suite (imports reach the CRAM stack through execution).
"""

from __future__ import annotations

from dataclasses import replace

from krrood.adapters.json_serializer import to_json

from .grounding_helpers import STUB_GROUNDING_PLAN
from resym.core.grounding import GroundingFailure, GroundingFailureCode
from resym.repair.certificate import (
    CURATION_TRIGGERS,
    FailureClass,
    certify_execution_failure,
    certify_grounding_factory_binding_error,
    certify_grounding_failure,
    certify_unsolvable,
    certify_unsupported_capability,
)
from resym.planning.execution.engine import (
    ExecutionReport,
    ExecutionViolation,
    PlatformExecutionResult,
)
from resym.planning.grounding import GroundingResult
from resym.core.model import (
    Literal,
    Operator,
    PredicateSymbol,
    SymbolLibrary,
)
from .capability_helpers import execution_binding
from resym.planning.pddl import GroundAction
from resym.planning.selection import Selection
from resym.planning.selection import select_for_goal
from resym.core.validation import (
    ModelIssueKind,
    selection_model_issues,
)

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


def seedlike_library() -> SymbolLibrary:
    library = SymbolLibrary()
    for name in ("opened", "closed", "openable"):
        library.add(
            PredicateSymbol(
                name=name,
                parameter_types=(DRAWER_TYPE,),
                fluent=True,
                grounding_plan=STUB_GROUNDING_PLAN,
            )
        )
    library.add(
        Operator(
            name="open-drawer",
            parameters=(("d", DRAWER_TYPE),),
            preconditions=(Literal("openable", ("d",)), Literal("closed", ("d",))),
            add_effects=(Literal("opened", ("d",)),),
            delete_effects=(Literal("closed", ("d",)),),
            execution_binding=execution_binding("test:Pull", (("patient", "d"),)),
        )
    )
    return library


def selection_for(library: SymbolLibrary) -> Selection:
    return Selection(
        predicates=dict(library.predicates), operators=dict(library.operators)
    )


def test_goal_predicate_without_achiever_is_missing_operator_model():
    """
    The grow-library scenario: goal (closed d), no operator adds closed.
    """
    library = seedlike_library()
    goal = (Literal("closed", ("d1",)),)
    certificate = certify_unsolvable(
        goal, library, selection_for(library), planner_message="unsolvable"
    )
    assert certificate.failure_class is FailureClass.MISSING_OPERATOR_MODEL
    assert certificate.triggers_curation
    assert "closed" in certificate.causal_neighborhood.unachievable_goal_predicates


def test_capability_target_exposes_wrong_operator_effect(fixed_arm_library):
    library = fixed_arm_library
    original = library.operators["open-drawer"]
    library.operators["open-drawer"] = replace(
        original,
        add_effects=(Literal("closed", ("d",)),),
        delete_effects=(),
    )
    goal = (Literal("opened", ("d1",)),)

    certificate = certify_unsolvable(
        goal,
        library,
        select_for_goal(library, goal),
        planner_message="unsolvable",
    )

    assert certificate.failure_class is FailureClass.OPERATOR_EFFECT_ERROR
    assert certificate.triggers_curation
    assert certificate.causal_neighborhood.operators == ("open-drawer",)
    assert "target_state" in certificate.model_issues[0]
    assert "opened" in certificate.model_issues[0]
    assert "causal operators: open-drawer" in certificate.render()


def test_unknown_goal_predicate_is_missing_predicate_model():
    library = seedlike_library()
    goal = (Literal("locked", ("d1",)),)
    certificate = certify_unsolvable(
        goal, library, selection_for(library), planner_message="unsolvable"
    )
    assert certificate.failure_class is FailureClass.MISSING_PREDICATE_MODEL
    assert FailureClass.GOAL_SPECIFICATION_ERROR in certificate.alternative_classes


def test_grounding_failure_is_not_misdiagnosed_as_model_gap():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    failure = GroundingFailure(
        GroundingFailureCode.RESOURCE_LIMIT,
        "IK budget exhausted",
        Literal("openable", ("d1",)),
    )
    certificate = certify_grounding_failure(
        goal,
        selection_for(library),
        failure,
    )
    assert certificate.failure_class is FailureClass.GROUNDING_FAILURE
    assert not certificate.triggers_curation
    assert certificate.grounding_failure_code == "resource_limit"
    assert certificate.violated_literal == Literal("openable", ("d1",))


def test_decided_but_unsolvable_is_geometric_infeasibility():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    certificate = certify_unsolvable(
        goal,
        library,
        selection_for(library),
        planner_message="unsolvable",
        grounding=GroundingResult(false_atoms={Literal("openable", ("d1",))}),
    )
    assert certificate.failure_class is FailureClass.GEOMETRIC_INFEASIBILITY
    assert not certificate.triggers_curation


def test_execution_violations_map_mechanically():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    action = GroundAction("open-drawer", ("d1",))
    cases = {
        ExecutionViolation.PRECONDITION_FALSE: FailureClass.PRECONDITION_INVALIDATED,
        ExecutionViolation.PRECONDITION_GROUNDING_FAILED: FailureClass.GROUNDING_FAILURE,
        ExecutionViolation.PLATFORM_UNSUPPORTED: FailureClass.UNSUPPORTED_CAPABILITY,
        ExecutionViolation.PLATFORM_REJECTED: FailureClass.GEOMETRIC_INFEASIBILITY,
        ExecutionViolation.PLATFORM_FAILED: FailureClass.PLATFORM_EXECUTION_FAILURE,
        ExecutionViolation.POSTCONDITION_FAILED: FailureClass.POSTCONDITION_FAILURE,
        ExecutionViolation.POSTCONDITION_GROUNDING_FAILED: FailureClass.GROUNDING_FAILURE,
    }
    for violation, expected in cases.items():
        report = ExecutionReport(violation=violation, violated_action=action)
        certificate = certify_execution_failure(
            goal, library, selection_for(library), report
        )
        assert certificate.failure_class is expected, violation


def test_platform_error_code_overrides_toward_platform_failure():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    report = ExecutionReport(
        violation=ExecutionViolation.POSTCONDITION_FAILED,
        violated_action=GroundAction("open-drawer", ("d1",)),
    )
    certificate = certify_execution_failure(
        goal,
        library,
        selection_for(library),
        report,
        platform_failure_code="GRIPPER_SLIP",
    )
    assert certificate.failure_class is FailureClass.PLATFORM_EXECUTION_FAILURE
    assert certificate.platform_failure_code == "GRIPPER_SLIP"


def test_native_platform_code_is_preserved_without_changing_primary_class():
    library = seedlike_library()
    report = ExecutionReport(
        violation=ExecutionViolation.PLATFORM_REJECTED,
        violated_action=GroundAction("open-drawer", ("d1",)),
        platform_results=[
            PlatformExecutionResult.rejected(
                "CORAPLEX_PRECONDITION_FAILED", "unreachable"
            )
        ],
    )
    certificate = certify_execution_failure(
        (Literal("opened", ("d1",)),),
        library,
        selection_for(library),
        report,
    )
    assert certificate.failure_class is FailureClass.GEOMETRIC_INFEASIBILITY
    assert certificate.platform_failure_code == "CORAPLEX_PRECONDITION_FAILED"


def test_unsupported_capability_stops_curation():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    certificate = certify_unsupported_capability(
        goal, selection_for(library), missing="no capability binding for 'pull'"
    )
    assert certificate.failure_class is FailureClass.UNSUPPORTED_CAPABILITY
    assert not certificate.triggers_curation


def test_missing_grounding_binding_triggers_curation_with_unsupported_alternative():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    certificate = certify_grounding_factory_binding_error(
        goal,
        selection_for(library),
        missing=(
            "predicate 'opened' needs grounding factory "
            "'resym:grounding/joint-fraction-opened-v2'"
        ),
    )

    assert certificate.failure_class is FailureClass.PREDICATE_IMPLEMENTATION_ERROR
    assert certificate.alternative_classes == (FailureClass.UNSUPPORTED_CAPABILITY,)
    assert certificate.triggers_curation


def test_only_repairable_model_defects_trigger_curation():
    assert CURATION_TRIGGERS == {
        FailureClass.MISSING_PREDICATE_MODEL,
        FailureClass.MISSING_OPERATOR_MODEL,
        FailureClass.OPERATOR_PRECONDITION_ERROR,
        FailureClass.OPERATOR_EFFECT_ERROR,
        FailureClass.OPERATOR_SIGNATURE_ERROR,
        FailureClass.PREDICATE_IMPLEMENTATION_ERROR,
    }


def test_binary_unsolvable_state_is_geometric_infeasibility():
    library = seedlike_library()
    goal = (Literal("opened", ("d1",)),)
    grounding = GroundingResult(
        false_atoms={Literal("openable", ("d1",))},
    )

    certificate = certify_unsolvable(
        goal,
        library,
        selection_for(library),
        planner_message="unsolvable",
        grounding=grounding,
    )

    assert certificate.failure_class is FailureClass.GEOMETRIC_INFEASIBILITY
    assert certificate.truth.false_atoms == (Literal("openable", ("d1",)),)


def test_self_dependent_goal_achiever_is_wrong_precondition():
    library = seedlike_library()
    original = library.operators["open-drawer"]
    library.operators["open-drawer"] = Operator(
        name=original.name,
        parameters=original.parameters,
        preconditions=(Literal("opened", ("d",)),),
        add_effects=original.add_effects,
        delete_effects=original.delete_effects,
        execution_binding=original.execution_binding,
    )

    certificate = certify_unsolvable(
        (Literal("opened", ("d1",)),),
        library,
        selection_for(library),
        planner_message="unsolvable",
        grounding=GroundingResult(),
    )

    assert certificate.failure_class is FailureClass.OPERATOR_PRECONDITION_ERROR
    assert certificate.triggers_curation


def test_existing_operator_signature_is_checked_before_repair():
    library = seedlike_library()
    original = library.operators["open-drawer"]
    library.operators["open-drawer"] = Operator(
        name=original.name,
        parameters=(("d", ROBOT_TYPE),),
        preconditions=original.preconditions,
        add_effects=original.add_effects,
        delete_effects=original.delete_effects,
        execution_binding=original.execution_binding,
    )

    issues = selection_model_issues(library, selection_for(library))

    assert issues
    assert {issue.kind for issue in issues} == {ModelIssueKind.OPERATOR_SIGNATURE}
    assert any("expects" in issue.detail for issue in issues)


def test_unresolvable_type_reference_is_reported_as_such():
    """
    A parameter type that no longer resolves to a class is a resolution defect, not a
    type mismatch.
    """
    library = seedlike_library()
    original = library.operators["open-drawer"]
    library.operators["open-drawer"] = Operator(
        name=original.name,
        parameters=(("d", SymbolType("semantic_digital_twin:NoSuchClass")),),
        preconditions=original.preconditions,
        add_effects=original.add_effects,
        delete_effects=original.delete_effects,
        execution_binding=original.execution_binding,
    )

    issues = selection_model_issues(library, selection_for(library))

    assert issues
    assert {issue.kind for issue in issues} == {ModelIssueKind.OPERATOR_SIGNATURE}
    assert any("cannot resolve type" in issue.detail for issue in issues)
    assert all("expects" not in issue.detail for issue in issues)


def test_certificate_serializes_and_renders():
    library = seedlike_library()
    goal = (Literal("closed", ("d1",)),)
    certificate = certify_unsolvable(
        goal, library, selection_for(library), planner_message="planner said no"
    )
    data = to_json(certificate)
    assert data["failure_class"]["name"] == FailureClass.MISSING_OPERATOR_MODEL.name
    assert data["task_goal"][0]["predicate"] == "closed"
    rendered = certificate.render()
    assert FailureClass.MISSING_OPERATOR_MODEL.value in rendered
    assert "planner said no" in rendered
