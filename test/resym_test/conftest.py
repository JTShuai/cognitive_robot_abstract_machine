"""
Shared fixtures for the reSym package tests.
"""

from __future__ import annotations

import pytest

from .library_fixtures import build_seed_library


@pytest.fixture()
def library():
    """
    Return the task-independent seed symbol library.
    """
    return build_seed_library()
