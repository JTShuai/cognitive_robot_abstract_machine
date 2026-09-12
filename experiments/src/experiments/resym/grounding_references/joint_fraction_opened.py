"""
True iff the articulated object's joint reached the opened-fraction threshold.
"""

from experiments.resym.articulation import articulation_connection, joint_fraction


def evaluate(context, universe, arguments, parameters):
    (articulated_object,) = arguments
    return bool(
        joint_fraction(articulation_connection(articulated_object))
        >= parameters["threshold"]
    )
