"""
True iff the articulated object's joint reached the opened-fraction threshold.
"""

from resym.platform.articulation import articulation_connection
from resym.platform.universe import joint_fraction


def evaluate(context, universe, arguments, parameters):
    (articulated_object,) = arguments
    return bool(
        joint_fraction(articulation_connection(articulated_object))
        >= parameters["threshold"]
    )
