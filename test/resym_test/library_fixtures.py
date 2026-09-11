"""
Symbol-library fixtures that do not depend on the experiments package.
"""

from __future__ import annotations

from pathlib import Path

from resym.core.model import SymbolLibrary

LIBRARY_DIRECTORY = Path(__file__).resolve().parents[2] / "resym" / "library"


def build_seed_library() -> SymbolLibrary:
    """
    Load a fresh mobile-robot seed library.
    """
    return SymbolLibrary.load(LIBRARY_DIRECTORY / "seed_library.json")


def build_fixed_arm_library() -> SymbolLibrary:
    """
    Load a fresh fixed-arm seed library.
    """
    return SymbolLibrary.load(LIBRARY_DIRECTORY / "fixed_arm_library.json")
