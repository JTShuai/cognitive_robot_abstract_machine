Propose reusable Boolean world relations for this deployment's task planning.
Use the scanned CRAM/kRrood query vocabulary and native Coraplex action source
below, including preconditions and postconditions where present.

Query vocabulary:
$vocabulary

Native actions and source:
$actions

Existing factories (reuse these instead of proposing duplicates):
$factories

For each proposed relation, supply a GroundingRequest describing its stable UID,
meaning, typed roles and optional bounded parameters. Cite exact qualified names
from the query vocabulary and explain how they support the meaning. Role types
must name real CRAM semantic classes using their full Python import paths.
Propose schema-level relations, never object-instance-specific predicates.
Keep configurable thresholds explicit in parameters. For complementary relations
such as closed = not opened, propose one factory; predicate plans can negate it.

Action names and signatures alone are not proof of a state query. Do not invent
observations, interfaces or execution capabilities. Read query signatures and
descriptions carefully; generic EQL composition requires available world data.
If no supported relations can be identified, return an empty relations list and
explain the limitation. Do not claim completeness for future tasks.

This response contains data only, not factory source. It requests a later factory
draft; it neither approves a factory nor inserts predicates into a symbol library.
Human reviewers will inspect the resulting implementation before it can execute.
