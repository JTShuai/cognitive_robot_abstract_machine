from semantic_digital_twin.reasoning.robot_predicates import robot_holds_body


def evaluate(context, universe, arguments, parameters):
    return robot_holds_body(arguments[0], arguments[1], threshold=0.5)
