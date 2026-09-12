"""
True iff the interaction point is the one mounted on the articulated object.
"""

from resym.platform.articulation import interaction_point_belongs_to


def evaluate(context, universe, arguments, parameters):
    interaction_point, articulated_object = arguments
    return bool(interaction_point_belongs_to(interaction_point, articulated_object))
