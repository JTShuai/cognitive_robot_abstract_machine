"""
Embodiment profiles: the declared capability surface a goal's causal closure is checked
against before anything is grounded or planned.

Host-runnable: profiles are declarative data and the check runs over plain library
slices.
"""

from __future__ import annotations

from resym.platform.embodiment import (
    EmbodimentProfile,
    ToolOrientation,
    UnsupportedCapabilityError,
)
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


def fixed_profile(**overrides) -> EmbodimentProfile:
    keywords = dict(
        name="xarm5-fixed",
        evaluators=frozenset({"drawer_opened"}),
        capabilities=frozenset({ARTICULATION_CAPABILITY}),
        tool_orientation=ToolOrientation.APPROACH_ALIGNED,
    )
    keywords.update(overrides)
    return EmbodimentProfile(**keywords)


def predicate(name: str, evaluator: str) -> PredicateSymbol:
    return PredicateSymbol(
        name=name,
        parameter_types=(DRAWER_TYPE,),
        evaluator=evaluator,
        fluent=True,
    )


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
        predicates={"opened": predicate("opened", "drawer_opened")},
        operators={"open-drawer": operator("open-drawer", ARTICULATION_CAPABILITY)},
    )
    profile = fixed_profile()
    assert profile.missing_evaluators(selection) == ()
    assert profile.missing_capabilities(selection) == ()


def test_evaluator_and_capability_gaps_are_reported_separately():
    selection = Selection(
        predicates={"openable": predicate("openable", "openable")},
        operators={"navigate": operator("navigate", NAVIGATION_CAPABILITY)},
    )
    profile = fixed_profile()

    assert len(profile.missing_evaluators(selection)) == 1
    assert "evaluator 'openable'" in profile.missing_evaluators(selection)[0]
    assert len(profile.missing_capabilities(selection)) == 1
    assert NAVIGATION_CAPABILITY in profile.missing_capabilities(selection)[0]
    assert all(
        "xarm5-fixed" in message
        for message in profile.missing_evaluators(selection)
        + profile.missing_capabilities(selection)
    )


def test_error_carries_goal_and_selection_for_the_certificate():
    selection = Selection(
        predicates={"openable": predicate("openable", "openable")}, operators={}
    )
    goal = (Literal("opened", ("d1",)),)
    missing = fixed_profile().missing_evaluators(selection)
    error = UnsupportedCapabilityError(missing, goal=goal, selection=selection)
    assert error.goal == goal
    assert error.selection is selection
    assert "openable" in str(error)


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
        known_evaluators=profile.evaluators,
        available_capabilities=profile.capabilities,
    )
    assert any(NAVIGATION_CAPABILITY in objection for objection in objections)
