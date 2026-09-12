"""
Shared fixtures for the reSym package tests.
"""

from __future__ import annotations

import pytest

from experiments.resym.grounding_initialization import bootstrap_drawer_grounding
from experiments.resym.seed_library import (
    build_fixed_arm_library,
    build_seed_library,
)


@pytest.fixture(scope="session")
def grounding_catalog(tmp_path_factory):
    """
    The reviewed drawer grounding catalog, bootstrapped once per session.
    """
    return bootstrap_drawer_grounding(tmp_path_factory.mktemp("grounding_workspace"))


@pytest.fixture()
def library(grounding_catalog):
    """
    Return the task-independent seed symbol library.
    """
    return build_seed_library(grounding_catalog)


@pytest.fixture()
def fixed_arm_library(grounding_catalog):
    """
    Return the fixed-arm drawer symbol library.
    """
    return build_fixed_arm_library(grounding_catalog)
