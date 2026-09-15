Propose reusable Boolean world relations for this deployment's task planning.
This batch covers only the supplied action. Propose relations needed to describe
its preconditions, postconditions or independently checkable effects, not every
relation expressible by the query vocabulary. Other actions have separate batches.
Keep explanations concise; return every supported relation needed by this action.
Use the scanned CRAM/kRrood query vocabulary and native Coraplex action source
below, including preconditions and postconditions where present.

Query vocabulary:
$vocabulary

Native actions and source:
$actions

Native selector parameters and the world entities they select (use these entity
types for the corresponding roles; a selector enum is never a role type):
$selectors

Existing factories (reuse these instead of proposing duplicates):
$factories

If an existing relation request already covers a meaning, reuse its UID and exact
definition. Do not redefine it or create a synonymous UID. Omit an already covered
relation from this response. Record unsupported requirements in limitations.

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
Every cited query must establish the relation's stated meaning, not just something
associated with it. A structural property cannot prove completion of a process.
If an observation is missing, put the requirement in limitations and omit that
relation. Do not propose an unsupported relation merely to cover an action effect.
Use capability-independent relation names; reuse identical meanings across actions.
Every role type must be accepted by a parameter of a cited query or own a cited
readable attribute; a mismatched role is rejected. Never replace an unavailable
arm type with a base or robot type merely to satisfy validation.

This response contains data only, not factory source. It requests a later factory
draft; it neither approves a factory nor inserts predicates into a symbol library.
Human reviewers will inspect the resulting implementation before it can execute.
