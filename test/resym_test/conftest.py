"""
Shared fixtures for the reSym package tests.
"""

from __future__ import annotations

import pytest

from .dataset.capability_model import bootstrap_capability_realizations
from .dataset.task_model import (
    bootstrap_task_grounding,
    build_fixed_arm_library,
    build_seed_library,
)


@pytest.fixture(scope="session")
def grounding_catalog(tmp_path_factory):
    """
    The reviewed catalog of the example task model, bootstrapped once per session.
    """
    return bootstrap_task_grounding(tmp_path_factory.mktemp("grounding_workspace"))


@pytest.fixture()
def library(grounding_catalog):
    """
    Return the mobile-robot symbol library.
    """
    return build_seed_library(grounding_catalog)


@pytest.fixture()
def fixed_arm_library(grounding_catalog):
    """
    Return the fixed-arm symbol library.
    """
    return build_fixed_arm_library(grounding_catalog)


@pytest.fixture(scope="session")
def capability_initialization(tmp_path_factory):
    """
    The reviewed realizations of the example task model, bootstrapped once per session.
    """
    return bootstrap_capability_realizations(
        tmp_path_factory.mktemp("realization_workspace")
    )
