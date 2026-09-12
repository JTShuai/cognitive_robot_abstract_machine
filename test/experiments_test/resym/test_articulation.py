"""
Articulation helpers of the drawer experiments.
"""

from experiments.resym.articulation import interaction_point_belongs_to
from resym.core.symbol_types import SymbolType
from resym.platform.universe import GroundedObject
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
)
from semantic_digital_twin.world_description.world_entity import Body

DRAWER_TYPE = SymbolType.from_python_type(Drawer)
HANDLE_TYPE = SymbolType.from_python_type(Handle)


def grounded_handle(name: str) -> GroundedObject:
    body = Body(name=PrefixedName(name))
    return GroundedObject(name, HANDLE_TYPE, body, Handle(root=body))


def grounded_drawer(name: str, handle: GroundedObject | None) -> GroundedObject:
    body = Body(name=PrefixedName(name))
    drawer = Drawer(root=body)
    if handle is not None:
        drawer.handle = handle.semantic_entity
    return GroundedObject(name, DRAWER_TYPE, body, drawer)


def test_a_handle_belongs_to_the_drawer_it_is_mounted_on():
    handle = grounded_handle("handle_cab1")

    assert interaction_point_belongs_to(handle, grounded_drawer("cabinet1", handle))


def test_a_drawer_without_a_handle_has_no_interaction_point_rather_than_an_error():
    """
    Some annotated drawers carry no handle link; asking whether a handle is theirs is
    simply false, and must not abort grounding over the whole universe.
    """
    handle = grounded_handle("handle_cab1")

    assert not interaction_point_belongs_to(
        handle, grounded_drawer("coffee_table", None)
    )


def test_reachability_of_a_drawer_without_a_handle_is_false_not_an_error():
    from experiments.resym.drawer_kinematic_oracle import (
        DrawerExperimentFeasibility,
        interaction_reachable_from_current_base,
    )

    robot = GroundedObject("tracy", DRAWER_TYPE, Body(name=PrefixedName("tracy")))

    assert not interaction_reachable_from_current_base(
        DrawerExperimentFeasibility(),
        None,
        (robot, grounded_drawer("coffee_table", None)),
    )
