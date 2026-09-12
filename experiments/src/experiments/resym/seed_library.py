"""
Task model constructed from the reviewed platform catalogs for experiments.
"""

from __future__ import annotations

from resym.core.model import (
    CapabilityRef,
    Literal,
    Operator,
    OperatorExecutionBinding,
    PredicateGroundingPlan,
    PredicateSymbol,
    RoleBinding,
    SymbolLibrary,
)
from resym.platform.feasibility import feasibility_factory_uid
from resym.platform.grounding_catalog import GroundingFactoryCatalog
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    articulation_capability_contract,
    navigation_capability_contract,
)

from experiments.resym.grounding_initialization import (
    INTERACTION_POINT_OF_UID,
    JOINT_FRACTION_OPENED_UID,
)

from resym.core.model import SymbolType
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


OPENED_FRACTION_THRESHOLD = 0.4
"""
Task-model threshold for the binary opened/closed articulation state.
"""


def build_seed_library(catalog: GroundingFactoryCatalog) -> SymbolLibrary:
    """
    Construct the drawer-opening library: five predicates, two operators.
    """
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="handle-of",
            parameter_types=(HANDLE_TYPE, DRAWER_TYPE),
            fluent=False,
            grounding_plan=build_grounding_plan(
                catalog,
                INTERACTION_POINT_OF_UID,
                (("interaction_point", 0), ("articulated_object", 1)),
            ),
        )
    )
    library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            fluent=True,
            grounding_plan=_articulation_state_plan(catalog, negated=True),
        )
    )
    library.add(
        PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            fluent=True,
            grounding_plan=_articulation_state_plan(catalog),
        )
    )
    library.add(
        PredicateSymbol(
            name="ready-to-open",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            fluent=True,
            grounding_plan=build_grounding_plan(
                catalog,
                feasibility_factory_uid(ARTICULATION_CAPABILITY_UID),
                (("actor", 0), ("patient", 1)),
            ),
        )
    )
    library.add(
        PredicateSymbol(
            name="openable",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            fluent=False,
            grounding_plan=build_grounding_plan(
                catalog,
                feasibility_factory_uid(NAVIGATION_CAPABILITY_UID),
                (("actor", 0), ("patient", 1)),
            ),
        )
    )
    library.add(
        Operator(
            name="navigate",
            parameters=(("r", ROBOT_TYPE), ("d", DRAWER_TYPE)),
            preconditions=(Literal("openable", ("r", "d")),),
            add_effects=(Literal("ready-to-open", ("r", "d")),),
            delete_effects=(),
            execution_binding=_navigation_binding(),
        )
    )
    library.add(
        Operator(
            name="open-drawer",
            parameters=(
                ("r", ROBOT_TYPE),
                ("h", HANDLE_TYPE),
                ("d", DRAWER_TYPE),
            ),
            preconditions=(
                Literal("ready-to-open", ("r", "d")),
                Literal("handle-of", ("h", "d")),
                Literal("closed", ("d",)),
            ),
            add_effects=(Literal("opened", ("d",)),),
            delete_effects=(Literal("closed", ("d",)),),
            execution_binding=_articulation_binding("OPEN"),
        )
    )
    library.add_capability_contract(navigation_capability_contract())
    library.add_capability_contract(articulation_capability_contract())
    return library


def build_fixed_arm_library(catalog: GroundingFactoryCatalog) -> SymbolLibrary:
    """
    The correct library of the fixed-arm embodiment: no navigation and no witness-pose
    predicate — reachability from the mount is a grounding fact (``ready-to-open``), not
    something an action achieves.

    Both manipulation directions are modeled; the P2 fault templates delete or distort
    pieces of this library to create repair tasks with known ground truth.
    """
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="handle-of",
            parameter_types=(HANDLE_TYPE, DRAWER_TYPE),
            fluent=False,
            grounding_plan=build_grounding_plan(
                catalog,
                INTERACTION_POINT_OF_UID,
                (("interaction_point", 0), ("articulated_object", 1)),
            ),
        )
    )
    library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            fluent=True,
            grounding_plan=_articulation_state_plan(catalog, negated=True),
        )
    )
    library.add(
        PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            fluent=True,
            grounding_plan=_articulation_state_plan(catalog),
        )
    )
    library.add(
        PredicateSymbol(
            name="ready-to-open",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            fluent=True,
            grounding_plan=build_grounding_plan(
                catalog,
                feasibility_factory_uid(ARTICULATION_CAPABILITY_UID),
                (("actor", 0), ("patient", 1)),
            ),
        )
    )
    library.add(
        Operator(
            name="open-drawer",
            parameters=(
                ("r", ROBOT_TYPE),
                ("h", HANDLE_TYPE),
                ("d", DRAWER_TYPE),
            ),
            preconditions=(
                Literal("ready-to-open", ("r", "d")),
                Literal("handle-of", ("h", "d")),
                Literal("closed", ("d",)),
            ),
            add_effects=(Literal("opened", ("d",)),),
            delete_effects=(Literal("closed", ("d",)),),
            execution_binding=_articulation_binding("OPEN"),
        )
    )
    library.add(
        Operator(
            name="close-drawer",
            parameters=(
                ("r", ROBOT_TYPE),
                ("h", HANDLE_TYPE),
                ("d", DRAWER_TYPE),
            ),
            preconditions=(
                Literal("ready-to-open", ("r", "d")),
                Literal("handle-of", ("h", "d")),
                Literal("opened", ("d",)),
            ),
            add_effects=(Literal("closed", ("d",)),),
            delete_effects=(Literal("opened", ("d",)),),
            execution_binding=_articulation_binding("CLOSED"),
        )
    )
    library.add_capability_contract(articulation_capability_contract())
    return library


def _navigation_binding() -> OperatorExecutionBinding:
    return OperatorExecutionBinding(
        CapabilityRef(NAVIGATION_CAPABILITY_UID),
        (
            ("actor", RoleBinding.parameter("r")),
            ("patient", RoleBinding.parameter("d")),
        ),
    )


def build_grounding_plan(
    catalog: GroundingFactoryCatalog,
    factory_uid: str,
    role_bindings: tuple[tuple[str, int], ...],
    parameters: tuple[tuple[str, float], ...] = (),
    negated: bool = False,
) -> PredicateGroundingPlan:
    """
    Bind a task predicate to the current reviewed factory implementation.
    """
    specification = catalog.specification(factory_uid)
    return PredicateGroundingPlan(
        factory_uid=factory_uid,
        approved_factory_checksum=specification.implementation_checksum,
        role_bindings=role_bindings,
        parameters=parameters,
        negated=negated,
    )


def _articulation_state_plan(
    catalog: GroundingFactoryCatalog, negated: bool = False
) -> PredicateGroundingPlan:
    """
    Create the plan shared by the opened and closed predicates.
    """
    return build_grounding_plan(
        catalog,
        JOINT_FRACTION_OPENED_UID,
        (("articulated_object", 0),),
        (("threshold", OPENED_FRACTION_THRESHOLD),),
        negated,
    )


def _articulation_binding(target_state: str) -> OperatorExecutionBinding:
    return OperatorExecutionBinding(
        CapabilityRef(ARTICULATION_CAPABILITY_UID),
        (
            ("actor", RoleBinding.parameter("r")),
            ("patient", RoleBinding.parameter("d")),
            ("interaction_point", RoleBinding.parameter("h")),
            ("target_state", RoleBinding.constant(target_state)),
        ),
    )
