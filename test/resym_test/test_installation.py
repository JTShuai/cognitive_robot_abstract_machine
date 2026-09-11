"""
Package installation behaviour.
"""

from __future__ import annotations

from resym import _find_project_root


def test_installed_package_does_not_require_a_source_checkout(tmp_path):
    package_directory = tmp_path / "site-packages" / "resym"
    module_path = package_directory / "__init__.py"

    assert _find_project_root(module_path) == package_directory
