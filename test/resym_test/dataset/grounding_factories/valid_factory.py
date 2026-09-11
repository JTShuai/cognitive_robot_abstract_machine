from krrood.entity_query_language.factories import entity, variable


def evaluate(context, universe, arguments, parameters):
    candidate = variable(bool, domain=(True,))
    return bool(list(entity(candidate).where(candidate == True).evaluate()))
