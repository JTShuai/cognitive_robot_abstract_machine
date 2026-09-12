from pathlib import Path

from ._version import __version__


def _find_experiments_root(module_path: Path = Path(__file__)) -> Path:
    """
    Return the checkout root or the installed package directory.
    """
    resolved_module_path = module_path.resolve()
    for candidate in resolved_module_path.parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "experiments"
        ).is_dir():
            return candidate
    return resolved_module_path.parent


EXPERIMENTS_ROOT = _find_experiments_root()
"""
Root of the experiments package, holding its configuration and scratch directories.
"""
