def evaluate(context, universe, arguments, parameters):
    body = arguments[0]
    return body.global_pose is not None
