def evaluate(context, universe, arguments, parameters):
    return parameters.get("threshold", 0.5) > 0
