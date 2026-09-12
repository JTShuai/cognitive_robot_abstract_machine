"""
Embodiment profiles: the declared capability surface a goal's causal closure is checked
against before anything is grounded or planned.

Host-runnable: profiles are declarative data and the check runs over plain library
slices.
"""

from __future__ import annotations

from dataclasses import replace

from resym.core.grounding import PredicateGroundingPlan
from resym.platform.embodiment import (
    EmbodimentProfile,
    ToolOrientation,
    UnsupportedCapabilityError,
)
from resym.platform.capabilities import (
    ARTICULATION_CAPABILITY_UID,
    articulation_capability_contract,
)
from resym.platform.feasibility import feasibility_factory_uid
from resym.platform.grounding_catalog import GroundingFactoryCatalog

from resym.core.model import (
    Literal,
    Operator,
    PredicateSymbol,
)
from resym.planning.selection import Selection
from .capability_helpers import execution_binding

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)


ARTICULATION_CAPABILITY = "test:ArticulationStateChange"
NAVIGATION_CAPABILITY = "test:ReachArticulationInteraction"


def articulation_feasibility_stub(*arguments) -> bool:
    return True


def fixed_profile(**overrides) -> EmbodimentProfile:
    keywords = dict(
        name="xarm5-fixed",
        capabilities=frozenset({ARTICULATION_CAPABILITY}),
        tool_orientation=ToolOrientation.APPROACH_ALIGNED,
    )
    keywords.update(overrides)
    return EmbodimentProfile(**keywords)


def predicate(name: str, grounding_plan: PredicateGroundingPlan) -> PredicateSymbol:
    return PredicateSymbol(
        name=name,
        parameter_types=(DRAWER_TYPE,),
        fluent=True,
        grounding_plan=grounding_plan,
    )


def reviewed_catalog() -> GroundingFactoryCatalog:
    """
    A catalog carrying only the articulation feasibility factory.
    """
    return GroundingFactoryCatalog.load(
        capability_contracts=(articulation_capability_contract(),),
        capability_feasibility_implementations={
            ARTICULATION_CAPABILITY_UID: articulation_feasibility_stub
        },
    )


def reviewed_plan() -> PredicateGroundingPlan:
    """
    A plan bound to the derived articulation feasibility factory.
    """
    uid = feasibility_factory_uid(ARTICULATION_CAPABILITY_UID)
    specification = reviewed_catalog().specification(uid)
    return PredicateGroundingPlan(
        factory_uid=uid,
        approved_factory_checksum=specification.implementation_checksum,
    )


def unreviewed_plan() -> PredicateGroundingPlan:
    """
    A plan naming a factory the reviewed catalog does not provide.
    """
    reviewed = reviewed_plan()
    return replace(reviewed, factory_uid=f"{reviewed.factory_uid}-v2")


def operator(name: str, capability: str) -> Operator:
    return Operator(
        name=name,
        parameters=(("d", DRAWER_TYPE),),
        preconditions=(),
        add_effects=(Literal("opened", ("d",)),),
        delete_effects=(),
        execution_binding=execution_binding(capability, (("patient", "d"),)),
    )


def test_supported_selection_reports_no_gaps():
    selection = Selection(
        predicates={"opened": predicate("opened", reviewed_plan())},
        operators={"open-drawer": operator("open-drawer", ARTICULATION_CAPABILITY)},
    )
    profile = fixed_profile()
    assert profile.missing_grounding_factories(selection, reviewed_catalog()) == ()
    assert profile.missing_capabilities(selection) == ()


def test_grounding_and_capability_gaps_are_reported_separately():
    plan = unreviewed_plan()
    selection = Selection(
        predicates={"openable": predicate("openable", plan)},
        operators={"navigate": operator("navigate", NAVIGATION_CAPABILITY)},
    )
    profile = fixed_profile()

    missing_factories = profile.missing_grounding_factories(
        selection, reviewed_catalog()
    )
    assert len(missing_factories) == 1
    assert plan.factory_uid in missing_factories[0]
    assert len(profile.missing_capabilities(selection)) == 1
    assert NAVIGATION_CAPABILITY in profile.missing_capabilities(selection)[0]
    assert all(
        "xarm5-fixed" in message
        for message in missing_factories + profile.missing_capabilities(selection)
    )


def test_error_carries_goal_and_selection_for_the_certificate():
    plan = unreviewed_plan()
    selection = Selection(
        predicates={"openable": predicate("openable", plan)}, operators={}
    )
    goal = (Literal("opened", ("d1",)),)
    missing = fixed_profile().missing_grounding_factories(selection, reviewed_catalog())
    error = UnsupportedCapabilityError(missing, goal=goal, selection=selection)
    assert error.goal == goal
    assert error.selection is selection
    assert plan.factory_uid in str(error)


def test_available_capabilities_derive_from_the_profile():
    """
    Admission uses the profile's capability set directly.
    """
    from resym.repair.curator import static_objections
    from resym.repair.patch import ModelPatch
    from resym.core.model import SymbolLibrary

    profile = fixed_profile()
    patch = ModelPatch(
        operators=(operator("open-drawer", NAVIGATION_CAPABILITY),),
        rationale="uses a capability this embodiment lacks",
    )
    objections = static_objections(
        patch,
        SymbolLibrary(),
        available_capabilities=profile.capabilities,
    )
    assert any(NAVIGATION_CAPABILITY in objection for objection in objections)
