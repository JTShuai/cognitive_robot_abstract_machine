"""
Library layer: persistence roundtrip and duplicate protection.
"""

from __future__ import annotations

import pytest

from resym import PROJECT_ROOT
from resym.core.model import (
    DuplicateSymbolError,
    PredicateSymbol,
    SymbolLibrary,
)
from experiments.resym.seed_library import (
    FIXED_ARM_LIBRARY_PATH,
    LIBRARY_PATH,
    build_seed_library,
)

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)


def test_json_roundtrip(tmp_path):
    library = build_seed_library()
    path = tmp_path / "library.json"
    library.save(path)
    reloaded = SymbolLibrary.load(path)
    assert reloaded.to_json() == library.to_json()


def test_seed_library_paths_follow_the_resym_workspace_package():
    assert LIBRARY_PATH == PROJECT_ROOT / "library" / "seed_library.json"
    assert FIXED_ARM_LIBRARY_PATH == PROJECT_ROOT / "library" / "fixed_arm_library.json"


def test_duplicate_symbol_rejected(library):
    duplicate = PredicateSymbol(
        name="opened",
        parameter_types=(DRAWER_TYPE,),
        evaluator="drawer_opened",
        fluent=True,
    )
    with pytest.raises(DuplicateSymbolError):
        library.add(duplicate)


def test_library_is_scene_free():
    """
    The persistent artifact must not mention any concrete world entity.
    """
    serialized = str(build_seed_library().to_json())
    for scene_specific in ("apartment", "kitchen", "cabinet", "pr2"):
        assert scene_specific not in serialized
