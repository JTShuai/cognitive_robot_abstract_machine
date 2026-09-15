"""
Source validation follows the advertised query return contracts.
"""

from dataclasses import replace

import pytest

from resym.platform.grounding_catalog import (
    GroundingFactorySourceValidator,
    GroundingFactorySourceError,
    GroundingVocabulary,
    GroundingVocabularyKind,
)
from .test_grounding_factory_catalog import DATASET, vocabulary


def boolean_vocabulary():
    """
    A query with an explicit Boolean return signature.
    """
    entry = replace(
        vocabulary().entries[0],
        qualified_name="semantic_digital_twin.reasoning.robot_predicates.robot_holds_body",
        kind=GroundingVocabularyKind.SYMBOLIC_FUNCTION,
        signature="(robot: AbstractRobot, body: Body) -> bool",
    )
    return GroundingVocabulary((entry,))


def test_direct_boolean_query_is_accepted():
    GroundingFactorySourceValidator(boolean_vocabulary()).validate(
        (DATASET / "direct_boolean.py").read_text()
    )


def test_constructor_is_not_a_boolean_query():
    entry = replace(
        boolean_vocabulary().entries[0], kind=GroundingVocabularyKind.PREDICATE
    )
    with pytest.raises(GroundingFactorySourceError):
        GroundingFactorySourceValidator(GroundingVocabulary((entry,))).validate(
            (DATASET / "direct_boolean.py").read_text()
        )


def test_parameter_get_is_read_only_and_accepted():
    GroundingFactorySourceValidator(vocabulary()).validate(
        (DATASET / "parameter_boolean.py").read_text()
    )


def test_query_rejects_undeclared_keyword():
    with pytest.raises(GroundingFactorySourceError):
        GroundingFactorySourceValidator(boolean_vocabulary()).validate(
            (DATASET / "wrong_query_keyword.py").read_text()
        )
