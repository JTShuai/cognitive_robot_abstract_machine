You draft one bounded, read-only grounding-factory candidate for human review.
The candidate decides one predicate over concrete world objects. It is NOT
executed until a human reviewer approves it.

## The relation to implement

- proposed factory uid: $proposed_uid
- semantic name: $semantic_name
- meaning: $meaning
- ordered roles (the positional `arguments` tuple): $roles
- typed parameters (available in `parameters`): $parameters

## Source contract

Write a single Python module containing exactly one function:

```
def evaluate(context, universe, arguments, parameters):
    ...
```

- `arguments` is the tuple of grounded objects, in the declared role order.
- `parameters` is a mapping of the declared typed parameters.
- Return a Python `bool`. Raise nothing; illegal states are review failures.
- Only `from <module> import <name>` imports of the reviewed vocabulary below
  are allowed. No other imports, no attribute reads outside called query
  methods, no loops, no try/except, at most one function.

## Reviewed vocabulary (the only importable symbols)

$vocabulary

## Prior review objections

$objections

Respond with JSON: {"source_code": "<the module source>", "rationale": "<one
sentence on why this decides the relation>"}.
