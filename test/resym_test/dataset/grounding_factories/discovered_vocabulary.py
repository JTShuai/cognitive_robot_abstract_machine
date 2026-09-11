from krrood.entity_query_language.predicate import Predicate, symbolic_function


@symbolic_function
def has_label(value: object) -> bool:
    return bool(value)


class Near(Predicate):
    pass
