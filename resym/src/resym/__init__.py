from __future__ import annotations

from pathlib import Path

from resym._version import __version__


def _find_project_root(module_path: Path = Path(__file__)) -> Path:
    """
    Return the checkout root or the installed package directory.
    """
    resolved_module_path = module_path.resolve()
    for candidate in resolved_module_path.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "resym"
        ).is_dir():
            return candidate
    return resolved_module_path.parent


PROJECT_ROOT = _find_project_root()

AUGMENT_DATASET_DIRECTORY = PROJECT_ROOT / "augment_dataset"
"""
Installed external data the source distribution does not carry: pinned ontology files
and the UniDomain retrieval corpus with its frozen release.
"""
