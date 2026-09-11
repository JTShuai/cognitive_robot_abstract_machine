"""
Shared fixtures for reSym scenes and experiments.
"""

from __future__ import annotations

from operator import attrgetter

import pytest

from experiments.resym.scenes import Scene, load_fixed_arm_scene, load_scene
from experiments.resym.seed_library import build_seed_library
from resym.platform.cram_objects import (
    RobotObjectExtractor,
    SemanticAnnotationObjectExtractor,
)
from resym.platform.evaluators import EvaluationContext
from resym.platform.universe import ObjectUniverse
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer


def drawer_universe_extractors():
    return (
        RobotObjectExtractor(),
        SemanticAnnotationObjectExtractor(
            Drawer,
            related_entities=(attrgetter("handle"),),
        ),
    )


@pytest.fixture(scope="session")
def apartment_setup():
    return load_scene(Scene.APARTMENT)


@pytest.fixture(scope="session")
def tracy_setup():
    return load_fixed_arm_scene(Scene.APARTMENT)


@pytest.fixture(scope="session")
def tracy_universe(tracy_setup):
    return ObjectUniverse.from_world(
        tracy_setup.world, tracy_setup.robot, drawer_universe_extractors()
    )


@pytest.fixture()
def tracy_context(tracy_setup):
    return EvaluationContext(
        world=tracy_setup.world,
        robot=tracy_setup.robot,
        profile=tracy_setup.profile,
    )


@pytest.fixture(scope="session")
def apartment_universe(apartment_setup):
    return ObjectUniverse.from_world(
        apartment_setup.world, apartment_setup.robot, drawer_universe_extractors()
    )


@pytest.fixture()
def apartment_context(apartment_setup):
    return EvaluationContext(
        world=apartment_setup.world,
        robot=apartment_setup.robot,
        profile=apartment_setup.profile,
    )


@pytest.fixture()
def library():
    return build_seed_library()
