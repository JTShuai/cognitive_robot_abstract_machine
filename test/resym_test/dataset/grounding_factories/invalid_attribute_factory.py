from krrood.entity_query_language.factories import entity


def evaluate(context, universe, arguments, parameters):
    return bool(context.unreviewed_state)
