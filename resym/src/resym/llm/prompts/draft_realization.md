Draft an action-parameter mapping for a human reviewer. The capability contract
has already been approved. Do not change its meaning or invent a native action.

## Approved contract
$contract

## Native action
$action

## Native source
$source

## Available context values
$context_values

Each parameter source is one of:
- role: use an existing contract role;
- context: use one of the listed context values;
- constant: an explicit string or native enum member name.

For witness_base_pose, key_roles declares which request roles identify the
stored witness. Declare robot resource requirements using full CRAM type references.
Use condition_role and condition_value only when the action realizes one variant
of the contract, such as a particular target state. Bind every required native
parameter. Never invent context values or computations the adapter cannot perform.
If this interface cannot implement the action, explain the gap rather than guessing.

The current adapter converts object roles to semantic entities, bodies, object
poses, single-pose trajectories, annotation types, or native enum values as declared
by the native parameter type. Constants other than enums remain strings.

Respond with JSON matching the supplied RealizationDraft schema. This only
creates a candidate; it does not approve it or execute a robot action.
