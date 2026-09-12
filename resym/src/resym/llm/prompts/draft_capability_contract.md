You draft one capability contract for human review. A contract is the semantic
interface a robot platform offers task planning: which semantic roles an action
takes, which effects on the world it promises, and when it counts as succeeded.
The contract is NOT admitted until a human reviewer approves it.

## The native action to describe

$action

Docstring: $summary

## Contracts already admitted (do not duplicate their uid or meaning)

$existing_contracts

## What to produce

- `uid`: `resym:<CapabilityName>` in CamelCase, naming the semantic goal, never
  the action class (e.g. `resym:ObjectPlacement`, not `resym:PlaceAction`).
- `label`: `<activity>.<verb-phrase>` in lower kebab case, e.g.
  `manipulation.place`, `navigation.move-base`, `material.pour`.
- `roles`: one per semantic participant. An object role lists
  `accepted_symbol_types` as CRAM class references spelled
  `<module>.<Class>` under `semantic_digital_twin` or `krrood`, e.g.
  `semantic_digital_twin.world_description.world_entity.SemanticAnnotation`.
  A constant role lists `allowed_values` instead. Every contract has an
  `actor` role typed `semantic_digital_twin.semantic_annotations.semantic_annotations.Agent`.
- `verifiable_effects`: lower-kebab-case effect names a predicate can check on
  the world afterwards, e.g. `placed-at`, `opened`.
- `effect_role_values`: for constant roles, `[effect, role, value]` triples
  saying which value of the role realizes which effect; otherwise empty.
- `success_relation`: a short relation over the role names, e.g.
  `placed_at(patient, destination)`.
- `rationale`: one sentence on why this action realizes this capability.

## Prior review objections

$objections

Respond with JSON matching these fields exactly.
