"""
Capability-derived feasibility grounding: the platform carries no predicate semantics of
its own; truth follows the reviewed capability catalog.
"""

from __future__ import annotations

import pytest

from resym.core.grounding import (
    GroundingFactoryOrigin,
    GroundingFailure,
    GroundingFailureCode,
    PredicateGroundingPlan,
)
from resym.core.model import PredicateSymbol, SymbolType
from resym.planning.grounding import evaluate_predicate
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    PLACE_CAPABILITY_UID,
    articulation_capability_contract,
    navigation_capability_contract,
)
from resym.platform.embodiment import EmbodimentProfile
from resym.platform.feasibility import (
    CapabilityFeasibility,
    feasibility_factory_uid,
)
from resym.platform.grounding_catalog import GroundingFactoryCatalog
from resym.platform.grounding_context import EvaluationContext
from resym.platform.kinematic import (
    KINEMATIC_FEASIBILITY,
    KINEMATIC_REALIZATIONS,
    KinematicFeasibility,
)
from resym.platform.universe import GroundedObject, ObjectUniverse
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import HasMechanicalJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent
from semantic_digital_twin.world_description.world_entity import Body

AGENT_TYPE = SymbolType.from_python_type(Agent)
ARTICULATED_PART_TYPE = SymbolType.from_python_type(HasMechanicalJoint)
ROBOT_TYPE = SymbolType.from_python_type(AbstractRobot)


class RecordingFeasibility(CapabilityFeasibility):
    """
    Answer a fixed verdict and record every question asked.
    """

    def __init__(self, verdict: bool = True):
        self.verdict = verdict
        self.questions: list[tuple[str, tuple[str, ...]]] = []

    def feasible(self, capability_uid, arguments, context) -> bool:
        self.questions.append((capability_uid, tuple(item.name for item in arguments)))
        return self.verdict


def grounded(name: str) -> GroundedObject:
    return GroundedObject(
        name=name,
        symbol_type=ROBOT_TYPE,
        body=Body(name=PrefixedName(name)),
    )


def feasibility_predicate(
    catalog: GroundingFactoryCatalog, negated: bool = False
) -> PredicateSymbol:
    uid = feasibility_factory_uid(ARTICULATION_CAPABILITY_UID)
    specification = catalog.specification(uid)
    return PredicateSymbol(
        name="ready-to-open",
        parameter_types=(ROBOT_TYPE, ARTICULATED_PART_TYPE),
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid=uid,
            approved_factory_checksum=specification.implementation_checksum,
            role_bindings=(("actor", 0), ("patient", 1)),
            negated=negated,
        ),
    )


def test_platform_alone_provides_no_grounding_factories() -> None:
    assert tuple(GroundingFactoryCatalog.load()) == ()


def test_capability_contracts_derive_feasibility_factories() -> None:
    contracts = (
        navigation_capability_contract(),
        articulation_capability_contract(),
    )
    catalog = GroundingFactoryCatalog.load(
        capability_contracts=contracts,
        capability_feasibility_implementations=KINEMATIC_FEASIBILITY,
    )

    specifications = {item.uid: item for item in catalog}
    assert set(specifications) == {
        feasibility_factory_uid(contract.uid) for contract in contracts
    }
    articulation = specifications[feasibility_factory_uid(ARTICULATION_CAPABILITY_UID)]
    assert articulation.origin is GroundingFactoryOrigin.PLATFORM
    assert tuple((role.name, role.symbol_type) for role in articulation.roles) == (
        ("actor", AGENT_TYPE),
        ("patient", ARTICULATED_PART_TYPE),
    )


def test_contract_without_feasibility_implementation_is_not_advertised() -> None:
    catalog = GroundingFactoryCatalog.load(
        capability_contracts=(articulation_capability_contract(),)
    )

    assert tuple(catalog) == ()


def test_feasibility_plan_delegates_to_the_embodiment_oracle() -> None:
    catalog = GroundingFactoryCatalog.load(
        capability_contracts=(articulation_capability_contract(),),
        capability_feasibility_implementations={
            ARTICULATION_CAPABILITY_UID: RecordingFeasibility.feasible
        },
    )
    oracle = RecordingFeasibility(verdict=True)
    context = EvaluationContext(
        world=None,
        robot=None,
        profile=EmbodimentProfile("stub", frozenset()),
        grounding_catalog=catalog,
        capability_feasibility=oracle,
    )
    robot, drawer = grounded("robot"), grounded("drawer")

    value = evaluate_predicate(
        feasibility_predicate(catalog), (robot, drawer), ObjectUniverse(), context
    )

    assert value is True
    assert oracle.questions == [(ARTICULATION_CAPABILITY_UID, ("robot", "drawer"))]


def test_kinematic_oracle_answers_exactly_the_capabilities_it_realizes() -> None:
    """
    The oracle and the execution registry cover the same capability set, and an
    unregistered capability is refused instead of silently answered.
    """
    assert set(KINEMATIC_FEASIBILITY) == set(KINEMATIC_REALIZATIONS)

    with pytest.raises(GroundingFailure) as error:
        KinematicFeasibility().feasible(
            PLACE_CAPABILITY_UID, (grounded("robot"),), context=None
        )

    assert error.value.code is GroundingFailureCode.UNSUPPORTED_QUERY


def test_feasibility_plan_without_an_oracle_is_a_grounding_failure() -> None:
    catalog = GroundingFactoryCatalog.load(
        capability_contracts=(articulation_capability_contract(),),
        capability_feasibility_implementations={
            ARTICULATION_CAPABILITY_UID: RecordingFeasibility.feasible
        },
    )
    context = EvaluationContext(
        world=None,
        robot=None,
        profile=EmbodimentProfile("stub", frozenset()),
        grounding_catalog=catalog,
    )

    with pytest.raises(GroundingFailure) as error:
        evaluate_predicate(
            feasibility_predicate(catalog),
            (grounded("robot"), grounded("drawer")),
            ObjectUniverse(),
            context,
        )

    assert error.value.code is GroundingFailureCode.UNSUPPORTED_QUERY
