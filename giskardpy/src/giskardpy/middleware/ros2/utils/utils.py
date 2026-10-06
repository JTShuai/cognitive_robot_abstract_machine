from __future__ import division

import os

import xacro
from typing_extensions import Dict, Optional

from semantic_digital_twin.adapters.package_resolver import CompositePathResolver


def load_xacro(path: str, mappings: Optional[Dict[str, str]] = None) -> str:
    """
    Expand a xacro file to a URDF string.

    :param path: The path to the xacro file, package URIs included.
    :param mappings: Extra xacro substitution arguments, for a parameterized description
        that needs them to select a model (e.g. ``XArm5.get_xacro_mappings``).
    """
    path = CompositePathResolver().resolve(path)
    doc = xacro.process_file(path, mappings={"radius": "0.9", **(mappings or {})})
    return doc.toprettyxml(indent="  ")


def is_in_github_workflow():
    return "GITHUB_WORKFLOW" in os.environ
