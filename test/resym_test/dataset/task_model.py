"""
The example task model the platform suite plans, repairs and renders.

An articulated container the robot opens: enough structure to exercise selection,
planning, grounding, execution binding and review, owned by the suite so the platform
can be tested without any application package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from resym.core.capability_model import (
    CapabilityRef,
    OperatorExecutionBinding,
    RoleBinding,
)
from resym.core.grounding_model import (
    GroundingFactoryCandidate,
    GroundingFactoryParameter,
    GroundingFactoryParameterType,
    GroundingFactoryRole,
    GroundingFactorySourceKind,
    PredicateGroundingPlan,
)
from resym.core.symbols import Literal, Operator, PredicateSymbol, SymbolLibrary
from resym.core.symbol_types import SymbolType
from resym.platform.feasibility import CapabilityFeasibility, feasibility_factory_uid
from resym.platform.grounding_catalog import (
    GroundingFactoryCatalog,
    GroundingFactoryWorkspace,
    helper_vocabulary,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

from .articulated_state import (
    articulation_connection,
    interaction_point_belongs_to,
    joint_fraction,
)

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)

FACTORY_SOURCE_DIRECTORY = Path(__file__).parent / "task_factories"
"""
Reviewable EQL implementations of the task model's state predicates.
"""

GROUNDING_QUERY_HELPERS = (
    articulation_connection,
    interaction_point_belongs_to,
    joint_fraction,
)
"""
Query helpers this task model asks to have in the reviewed EQL vocabulary.
"""

JOINT_FRACTION_OPENED_UID = "resym:grounding/joint-fraction-opened"
INTERACTION_POINT_OF_UID = "resym:grounding/interaction-point-of"

OPENED_FRACTION_THRESHOLD = 0.4
"""
Task-model threshold for the binary opened/closed articulation state.
"""

# %% capability feasibility
from .capability_model import (
    ARTICULATED_PART_TYPE,
    ARTICULATION_CAPABILITY_UID,
    HANDLE_TYPE,
    INTERACTION_NAVIGATION_CAPABILITY_UID,
    INTERACTION_POINT_ROLE,
    OpenCloseState,
    articulation_capability_contract,
    capability_contracts,
    interaction_navigation_capability_contract,
)


@dataclass
class ConstantFeasibility(CapabilityFeasibility):
    """
    Answer every feasibility question with one fixed verdict.
    """

    verdict: bool = True

    def feasible(self, capability_uid, arguments, context) -> bool:
        return self.verdict


TASK_FEASIBILITY = {
    ARTICULATION_CAPABILITY_UID: ConstantFeasibility.feasible,
    INTERACTION_NAVIGATION_CAPABILITY_UID: ConstantFeasibility.feasible,
}
"""
Capabilities this embodiment answers feasibility for, keyed like a real oracle registry.
"""


# %% reviewed grounding factories


def factory_candidates() -> tuple[GroundingFactoryCandidate, ...]:
    """
    The state-predicate implementations submitted for review.
    """
    return (
        GroundingFactoryCandidate(
            candidate_id="task-joint-fraction-opened",
            proposed_uid=JOINT_FRACTION_OPENED_UID,
            semantic_name="joint-fraction-opened",
            source_code=_factory_source("joint_fraction_opened"),
            roles=(GroundingFactoryRole("articulated_object", ARTICULATED_PART_TYPE),),
            parameters=(
                GroundingFactoryParameter(
                    name="threshold",
                    value_type=GroundingFactoryParameterType.NUMBER,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
            generated_by="task-model",
            rationale="the normalized joint position decides the open state",
            source_kind=GroundingFactorySourceKind.DISCOVERED,
        ),
        GroundingFactoryCandidate(
            candidate_id="task-interaction-point-of",
            proposed_uid=INTERACTION_POINT_OF_UID,
            semantic_name="interaction-point-of",
            source_code=_factory_source("interaction_point_of"),
            roles=(
                GroundingFactoryRole(INTERACTION_POINT_ROLE, HANDLE_TYPE),
                GroundingFactoryRole("articulated_object", ARTICULATED_PART_TYPE),
            ),
            generated_by="task-model",
            rationale="the scene graph decides which part is mounted where",
            source_kind=GroundingFactorySourceKind.DISCOVERED,
        ),
    )


def _factory_source(stem: str) -> str:
    return (FACTORY_SOURCE_DIRECTORY / f"{stem}.py").read_text(encoding="utf-8")


def bootstrap_task_grounding(
    workspace_root: Path,
    reviewer: str = "suite-bootstrap",
) -> GroundingFactoryCatalog:
    """
    Approve the task model's factories through the real review flow and load the runtime
    catalog.
    """
    workspace = GroundingFactoryWorkspace(workspace_root)
    vocabulary = helper_vocabulary(GROUNDING_QUERY_HELPERS)
    workspace.synchronize_vocabulary(vocabulary, discovery_scope="task-model-helpers")
    for entry in vocabulary.entries:
        workspace.approve_vocabulary(entry.qualified_name, reviewer=reviewer)
    for candidate in factory_candidates():
        workspace.submit(candidate)
        workspace.approve(
            candidate.candidate_id,
            reviewer=reviewer,
            vocabulary=workspace.reviewed_vocabulary(),
        )
    return GroundingFactoryCatalog.load(
        workspace=workspace,
        capability_contracts=capability_contracts(),
        capability_feasibility_implementations=TASK_FEASIBILITY,
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


# %% symbol libraries


def build_seed_library(catalog: GroundingFactoryCatalog) -> SymbolLibrary:
    """
    The mobile-robot library: five predicates, a navigate and an open operator.
    """
    library = _state_predicates(catalog)
    library.add(
        PredicateSymbol(
            name="openable",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            fluent=False,
            grounding_plan=build_grounding_plan(
                catalog,
                feasibility_factory_uid(INTERACTION_NAVIGATION_CAPABILITY_UID),
                (("actor", 0), ("patient", 1)),
            ),
        )
    )
    library.add(
        Operator(
            name="navigate",
            parameters=(("r", ROBOT_TYPE), ("d", DRAWER_TYPE)),
            preconditions=(Literal("openable", ("r", "d")),),
            add_effects=(Literal("ready-to-interact", ("r", "d")),),
            delete_effects=(),
            execution_binding=OperatorExecutionBinding(
                CapabilityRef(INTERACTION_NAVIGATION_CAPABILITY_UID),
                (
                    ("actor", RoleBinding.parameter("r")),
                    ("patient", RoleBinding.parameter("d")),
                ),
            ),
        )
    )
    library.add(_articulation_operator("open-drawer", OpenCloseState.OPEN))
    library.add_capability_contract(interaction_navigation_capability_contract())
    library.add_capability_contract(articulation_capability_contract())
    return library


def build_fixed_arm_library(catalog: GroundingFactoryCatalog) -> SymbolLibrary:
    """
    The fixed-arm library: no navigation, and both manipulation directions.

    Reachability from the mount is a grounding fact (``ready-to-interact``) rather than
    something an action achieves.
    """
    library = _state_predicates(catalog)
    library.add(_articulation_operator("open-drawer", OpenCloseState.OPEN))
    library.add(_articulation_operator("close-drawer", OpenCloseState.CLOSED))
    library.add_capability_contract(articulation_capability_contract())
    return library


def _state_predicates(catalog: GroundingFactoryCatalog) -> SymbolLibrary:
    """
    The predicates both embodiments share.
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
                ((INTERACTION_POINT_ROLE, 0), ("articulated_object", 1)),
            ),
        )
    )
    library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            fluent=True,
            grounding_plan=articulation_state_plan(catalog, negated=True),
        )
    )
    library.add(
        PredicateSymbol(
            name="opened",
            parameter_types=(DRAWER_TYPE,),
            fluent=True,
            grounding_plan=articulation_state_plan(catalog),
        )
    )
    library.add(
        PredicateSymbol(
            name="ready-to-interact",
            parameter_types=(ROBOT_TYPE, DRAWER_TYPE),
            fluent=True,
            grounding_plan=build_grounding_plan(
                catalog,
                feasibility_factory_uid(ARTICULATION_CAPABILITY_UID),
                (("actor", 0), ("patient", 1)),
            ),
        )
    )
    return library


def articulation_state_plan(
    catalog: GroundingFactoryCatalog,
    threshold: float = OPENED_FRACTION_THRESHOLD,
    negated: bool = False,
) -> PredicateGroundingPlan:
    """
    The reviewed binding shared by the opened and closed predicates.
    """
    return build_grounding_plan(
        catalog,
        JOINT_FRACTION_OPENED_UID,
        (("articulated_object", 0),),
        (("threshold", threshold),),
        negated,
    )


def _articulation_operator(name: str, target_state: str) -> Operator:
    """
    An operator driving the container into one articulation state.
    """
    opened, closed = Literal("opened", ("d",)), Literal("closed", ("d",))
    reached, undone = (
        (opened, closed) if target_state is OpenCloseState.OPEN else (closed, opened)
    )
    return Operator(
        name=name,
        parameters=(
            ("r", ROBOT_TYPE),
            ("h", HANDLE_TYPE),
            ("d", DRAWER_TYPE),
        ),
        preconditions=(
            Literal("ready-to-interact", ("r", "d")),
            Literal("handle-of", ("h", "d")),
            undone,
        ),
        add_effects=(reached,),
        delete_effects=(undone,),
        execution_binding=OperatorExecutionBinding(
            CapabilityRef(ARTICULATION_CAPABILITY_UID),
            (
                ("actor", RoleBinding.parameter("r")),
                ("patient", RoleBinding.parameter("d")),
                (INTERACTION_POINT_ROLE, RoleBinding.parameter("h")),
                ("target_state", RoleBinding.constant(target_state)),
            ),
        ),
    )
