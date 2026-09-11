"""
Capability contracts, operator bindings, and grounded requests.
"""

import inspect

import pytest
from krrood.adapters.json_serializer import from_json, to_json

from resym.core.model import (
    CapabilityRole,
    CapabilityContract,
    CapabilityRef,
    ExecutionRequest,
    Literal,
    Operator,
    OperatorExecutionBinding,
    RoleBinding,
    contract_violations,
)
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    PLACE_CAPABILITY_UID,
    articulation_capability_contract,
    matching_capability_contracts,
)
from resym.planning.execution.coraplex import (
    CoraplexSkillRealization,
    default_coraplex_capability_handlers,
)
from resym.platform.coraplex_catalog import (
    CORAPLEX_ADAPTER_CAPABILITY_UIDS,
)
from resym.planning.execution.engine import (
    PlatformExecutionStatus,
    execution_request_for,
)
from resym.planning.pddl import GroundAction

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer, Handle
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
OBJECT_TYPE = SymbolType.from_python_type(SemanticAnnotation)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)



def articulation_operator(target_state: str) -> Operator:
    return Operator(
        name=f"set-{target_state.lower()}",
        parameters=(
            ("r", ROBOT_TYPE),
            ("h", HANDLE_TYPE),
            ("d", DRAWER_TYPE),
        ),
        preconditions=(),
        add_effects=(
            Literal("opened" if target_state == "OPEN" else "closed", ("d",)),
        ),
        delete_effects=(
            Literal("closed" if target_state == "OPEN" else "opened", ("d",)),
        ),
        execution_binding=OperatorExecutionBinding(
            CapabilityRef(ARTICULATION_CAPABILITY_UID),
            (
                ("actor", RoleBinding.parameter("r")),
                ("patient", RoleBinding.parameter("d")),
                ("interaction_point", RoleBinding.parameter("h")),
                ("target_state", RoleBinding.constant(target_state)),
            ),
        ),
    )


def test_capability_role_declares_exactly_one_value_kind():
    with pytest.raises(ValueError, match="exactly one"):
        CapabilityRole("ambiguous")
    with pytest.raises(ValueError, match="exactly one"):
        CapabilityRole(
            "ambiguous",
            accepted_symbol_types=(DRAWER_TYPE,),
            allowed_values=("OPEN",),
        )


def test_two_operators_reuse_one_capability_with_different_targets():
    opened = articulation_operator("OPEN")
    closed = articulation_operator("CLOSED")

    assert opened.execution_binding.capability_ref == (
        closed.execution_binding.capability_ref
    )
    assert opened.execution_binding.role_map["target_state"].value == "OPEN"
    assert closed.execution_binding.role_map["target_state"].value == "CLOSED"


def test_contract_carries_effect_to_constant_semantics():
    contract = articulation_capability_contract()

    assert contract.constant_role_value("target_state", {"opened"}) == "OPEN"
    assert contract.constant_role_value("target_state", {"closed"}) == "CLOSED"
    reloaded = from_json(to_json(contract))
    assert reloaded.effect_role_values == contract.effect_role_values


def test_complete_catalog_lookup_finds_contract_by_effect_and_typed_role():
    matches = matching_capability_contracts(
        {"placed-at"}, {"patient": OBJECT_TYPE}
    )

    assert [contract.uid for contract in matches] == [PLACE_CAPABILITY_UID]


def test_catalog_lookup_rejects_contract_with_wrong_role_name():
    assert not matching_capability_contracts(
        {"placed-at"}, {"articulated_part": OBJECT_TYPE}
    )


def test_grounded_operator_becomes_an_explicit_execution_request():
    operator = articulation_operator("OPEN")
    request = execution_request_for(
        operator,
        GroundAction("set-open", ("robot1", "knob6", "door4")),
        articulation_capability_contract(),
    )

    assert request.capability_ref.uid == ARTICULATION_CAPABILITY_UID
    assert request.argument_map == {
        "actor": "robot1",
        "patient": "door4",
        "interaction_point": "knob6",
        "target_state": "OPEN",
    }


def test_contract_rejects_wrong_type_and_unsupported_constant():
    contract = articulation_capability_contract()
    operator = articulation_operator("OPEN")
    broken = OperatorExecutionBinding(
        operator.execution_binding.capability_ref,
        (
            ("actor", RoleBinding.parameter("d")),
            ("patient", RoleBinding.parameter("d")),
            ("target_state", RoleBinding.constant("LOCKED")),
        ),
    )
    violations = contract_violations(
        Operator(
            name=operator.name,
            parameters=operator.parameters,
            preconditions=operator.preconditions,
            add_effects=operator.add_effects,
            delete_effects=operator.delete_effects,
            execution_binding=broken,
        ),
        contract,
    )

    assert any(
        "parameter 'd'" in item and "role 'actor'" in item for item in violations
    )
    assert any("unsupported value 'LOCKED'" in item for item in violations)


def test_coraplex_reports_an_unimplemented_capability_as_unsupported():
    result = CoraplexSkillRealization(plan_context=None).execute(
        ExecutionRequest(CapabilityRef("test:UnknownCapability"), ()),
        context=None,
        universe=None,
    )

    assert result.status is PlatformExecutionStatus.UNSUPPORTED
    assert result.code == "CORAPLEX_CAPABILITY_NOT_IMPLEMENTED"


def test_coraplex_accepts_a_new_capability_handler_without_class_changes():
    marker = object()
    request = ExecutionRequest(CapabilityRef("test:PickObject"), ())
    realization = CoraplexSkillRealization(
        plan_context=None,
        capability_handlers={
            "test:PickObject": lambda request, context, universe: marker,
        },
    )

    assert realization.realize(request, context=None, universe=None) is marker


def test_coraplex_adapter_registry_covers_every_reviewed_contract():
    assert set(default_coraplex_capability_handlers()) == set(
        CORAPLEX_ADAPTER_CAPABILITY_UIDS
    )


def test_coraplex_adapter_arguments_match_native_action_signatures():
    """
    Catch Coraplex API drift before a request reaches the robot.
    """
    from coraplex.robot_plans.actions.composite.tool_based import (
        CuttingAction,
        MixingAction,
        PouringAction,
        WipingAction,
    )
    from coraplex.robot_plans.actions.composite.transporting import (
        PickAndPlaceAction,
        TransportAction,
    )
    from coraplex.robot_plans.actions.core.misc import DetectAction
    from coraplex.robot_plans.actions.core.navigation import (
        ElevatorNavigation,
        LookAtAction,
        NavigateAction,
    )
    from coraplex.robot_plans.actions.core.pick_up import (
        GraspingAction,
        PickUpAction,
        ReachAction,
    )
    from coraplex.robot_plans.actions.core.placing import PlaceAction
    from coraplex.robot_plans.actions.core.robot_body import (
        CarryAction,
        FollowToolCenterPointPathAction,
        MoveTorsoAction,
        ParkArmsAction,
        SetGripperAction,
    )

    expected = {
        NavigateAction: {"target_location"},
        LookAtAction: {"target"},
        DetectAction: {"technique", "region", "object_sem_annotation"},
        ReachAction: {"target_pose", "arm", "grasp_description", "object_designator"},
        GraspingAction: {"object_designator", "arm", "grasp_description"},
        PickUpAction: {"object_designator", "arm", "grasp_description"},
        PlaceAction: {"object_designator", "target_location", "arm"},
        TransportAction: {
            "object_designator",
            "target_location",
            "arm",
            "grasp_description",
        },
        PickAndPlaceAction: {
            "object_designator",
            "target_location",
            "arm",
            "grasp_description",
        },
        SetGripperAction: {"gripper", "motion"},
        ParkArmsAction: {"arm"},
        MoveTorsoAction: {"torso_state"},
        CarryAction: {"arm"},
        FollowToolCenterPointPathAction: {"target_locations", "arm"},
        MixingAction: {"arm", "tool", "container"},
        PouringAction: {"arm", "source_container", "target_container"},
        CuttingAction: {"arm", "tool", "object_to_cut"},
        WipingAction: {"arm", "tool", "surface"},
        ElevatorNavigation: {"elevator", "target_floor"},
    }
    for action, names in expected.items():
        assert names.issubset(inspect.signature(action).parameters), action.__name__
