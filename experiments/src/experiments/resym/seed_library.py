"""
The seeded symbol library of the demo.

In the full design this library is grown by gated LLM proposals and refined by execution
failures (stage B); here it is seeded by hand once, saved to JSON, and reused unchanged
across both demo scenes. Nothing in it references any particular scene.
"""

from __future__ import annotations

from pathlib import Path

from resym import PROJECT_ROOT
from resym.core.model import (
    CapabilityRef,
    Literal,
    Operator,
    OperatorExecutionBinding,
    PredicateSymbol,
    RoleBinding,
    SymbolLibrary,
)
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    NAVIGATION_CAPABILITY_UID,
    articulation_capability_contract,
    navigation_capability_contract,
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


LIBRARY_PATH = PROJECT_ROOT / "library" / "seed_library.json"
"""
Where the persistent library artifact lives.
"""

FIXED_ARM_LIBRARY_PATH = PROJECT_ROOT / "library" / "fixed_arm_library.json"
"""
The correct fixed-arm library artifact — the baseline the P2 fault templates mutate.
"""


def build_seed_library() -> SymbolLibrary:
    """
    Construct the drawer-opening library: five predicates, two operators.
    """
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="handle-of",
            parameter_types=(HANDLE_TYPE, DRAWER_TYPE),
            evaluator="handle_of",
            fluent=False,
        )
    )
    library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            evaluator="drawer_closed",
            fluent=True,
        )
    )
    library.add(
        PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            evaluator="drawer_opened",
            fluent=True,
        )
    )
    library.add(
        PredicateSymbol(
            name="ready-to-open",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            evaluator="ready_to_open",
            fluent=True,
        )
    )
    library.add(
        PredicateSymbol(
            name="openable",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            evaluator="openable",
            fluent=False,
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


def build_fixed_arm_library() -> SymbolLibrary:
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
            evaluator="handle_of",
            fluent=False,
        )
    )
    library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            evaluator="drawer_closed",
            fluent=True,
        )
    )
    library.add(
        PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            evaluator="drawer_opened",
            fluent=True,
        )
    )
    library.add(
        PredicateSymbol(
            name="ready-to-open",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            evaluator="ready_to_open",
            fluent=True,
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


def save_seed_library() -> Path:
    """
    Build and persist the library artifact; returns its path.
    """
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_seed_library().save(LIBRARY_PATH)
    return LIBRARY_PATH


def save_fixed_arm_library() -> Path:
    """
    Build and persist the fixed-arm library artifact; returns its path.
    """
    FIXED_ARM_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_fixed_arm_library().save(FIXED_ARM_LIBRARY_PATH)
    return FIXED_ARM_LIBRARY_PATH


if __name__ == "__main__":
    print(f"library saved to {save_seed_library()}")
    print(f"fixed-arm library saved to {save_fixed_arm_library()}")
