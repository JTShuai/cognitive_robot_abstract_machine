"""
Library layer: persistence roundtrip and duplicate protection.
"""

from __future__ import annotations

import pytest

from resym.core.grounding import PredicateGroundingPlan
from resym.core.model import (
    DuplicateSymbolError,
    PredicateSymbol,
    SymbolLibrary,
)

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)


def test_json_roundtrip(tmp_path, library):
    path = tmp_path / "library.json"
    library.save(path)
    reloaded = SymbolLibrary.load(path)
    assert reloaded.to_json() == library.to_json()


def test_duplicate_symbol_rejected(library):
    duplicate = PredicateSymbol(
        name="opened",
        parameter_types=(DRAWER_TYPE,),
        fluent=True,
        grounding_plan=PredicateGroundingPlan(
            factory_uid="test:grounding/stub",
            approved_factory_checksum="stub-checksum",
        ),
    )
    with pytest.raises(DuplicateSymbolError):
        library.add(duplicate)


def test_library_is_scene_free(library):
    """
    The persistent artifact must not mention any concrete world entity.
    """
    serialized = str(library.to_json())
    for scene_specific in ("apartment", "kitchen", "cabinet", "pr2"):
        assert scene_specific not in serialized
