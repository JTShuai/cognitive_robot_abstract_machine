# reSym

reSym builds on two CRAM layers: **Coraplex**, whose native actions are
the robot's executable capability interface, and **krrood** with the Semantic
Digital Twin, whose reviewed entity queries are the source of Boolean truth
about the world. On top of them, reSym maintains the predicates, operators,
and execution bindings that task planning needs as retrievable, repairable,
and versioned symbolic assets. Symbol gaps and execution failures are
diagnosed into structured certificates, a retrieval-augmented LLM agent
proposes candidate repairs, and a deterministic curator validates and decides
admission — so the same Coraplex action primitives adapt to new tasks, scenes
and robots without changes to their implementation code or to reSym's: which
capabilities a platform offers, and which native action realizes each, are
reviewed artifacts drafted at initialization, and nothing the LLM produces
enters the query or execution path unreviewed.


## Core loop

```text
natural-language task → structured goal (model gap or clarification if not expressible)
  → select relevant symbols and an object subset → Boolean grounding + PDDL planning
  → per-action precondition recheck
  → execution through capability bindings (Coraplex) → independent effect and goal checks
failure → programmatic failure certificate → retrieval-augmented repair agent
  (UniDomain corpus, tool loop) → minimal model patch → curator admission
  → version store → replan
```


## Main concepts

| Concept | Role |
|---|---|
| `SymbolType` | Stable reference to a CRAM semantic class; subtyping comes from the Semantic Digital Twin |
| `PredicateSymbol` | Task-level predicate: parameter types plus a reviewed world-query reference |
| `Operator` | Task-level action schema: preconditions and effects, projected to a PDDL action |
| `CapabilityContract` | Versioned declarative interface of a platform capability: semantic roles plus verifiable effect predicates |
| `OperatorExecutionBinding` | Maps operator parameters to contract roles, resolved into a platform-independent `ExecutionRequest` |

Task-level composition is done by the PDDL planner; platform-level actions are
expanded and executed by Coraplex against the current environment.

## Layout

```text
resym/src/resym/
  core/           symbolic model, stable references, grounding failures
  platform/       CRAM/Coraplex adapters, capability contracts, grounding review
  planning/       selection, grounding, PDDL, monitored execution
  repair/         failure certificates, repair agents, curator, versioning
  retrieval/      UniDomain corpus, frozen ontologies, retrieval
  llm/ interfaces/ observability/
experiments/src/experiments/resym/   scenes, grounding assets, task models
test/resym_test/                     core tests (host-runnable)
test/experiments_test/resym/         scene tests (container)
```

## First-time initialization

Run the commands below from the CRAM root. Choose Docker or a local installation.

### Option 1: Docker

The image contains ROS 2, CRAM, Python dependencies, Fast Downward, the external
datasets, and Viewer dependencies:

```bash
docker build -f resym/Dockerfile -t cram:jazzy-resym .
```

The runtime scripts mount the current checkout, so initialization results remain
on the host. No separate dataset download is needed for Docker.

Start the review Viewer in Docker:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/opt/cram" \
  -w /opt/cram/resym \
  -p 5000:5000 \
  cram:jazzy-resym \
  uv run --active --no-sync python -m resym.observability.viewer \
    runs \
    --grounding-workspace tmp/grounding_factory_workspace \
    --contract-workspace tmp/contract_workspace \
    --realization-workspace tmp/realization_workspace \
    --host 0.0.0.0
```

### Option 2: Local installation

Install ROS 2 Jazzy and `uv`, then prepare the ROS overlay and Python
workspace:

```bash
export OVERLAY_WS="${OVERLAY_WS:-$HOME/ros2_ws}"
./scripts/setup_ros_workspace.sh
source /opt/ros/jazzy/setup.bash
source "$OVERLAY_WS/install/setup.bash"
uv sync --package resym --extra dev
```

Install the external knowledge data and Fast Downward:

```bash
uv run --no-sync resym-install-ontologies
uv run --no-sync resym-install-corpus
./resym/scripts/install_fast_downward.sh
```

Start the review Viewer locally:

```bash
uv run --no-sync python -m resym.observability.viewer \
  resym/runs \
  --grounding-workspace resym/tmp/grounding_factory_workspace \
  --contract-workspace resym/tmp/contract_workspace \
  --realization-workspace resym/tmp/realization_workspace
```

Generated data and review workspaces remain outside Git.

### Configure the LLM

Configure this only when using an API model:

```bash
cp resym/.env.example resym/.env
cp resym/config/llm.example.json resym/config/llm.json
```

Set the provider credentials in `resym/.env` and the model in
`resym/config/llm.json`.

### Prepare initialization drafts

Initialization is a one-time setup operation:

```bash
uv run --no-sync resym-init prepare --workspace resym/tmp
```

With Docker, use the mounted-workspace wrapper instead:

```bash
./resym/scripts/run_in_docker.sh resym-init prepare --workspace tmp
```

This scans Coraplex and CRAM/kRrood interfaces and writes drafting jobs to
`resym/tmp/initialization/`. Nothing is approved automatically.

Grounding proposes relations in separate batches, one per Coraplex action.
The model uses the scanned query vocabulary and the action's source and dependencies
to propose reusable relations, their meanings, typed roles and bounded parameters.
Each proposal cites scanned query interfaces. The program checks references and
types; human review decides admission. A reference copied with its rendered
signature is reduced to the qualified name; a role type whose class name the
platform defines exactly once is corrected to that class's module, and a readable
member cited on a subclass or under a wrong module is saved under the class that
declares it. Every role
type must be one a task object can denote (a body or a semantic annotation) and
be accepted by a parameter of a cited query or own a cited readable attribute, so
a type mismatch is rejected before a Factory job is prepared. Native selector parameters such as
`Arms` are mapped to the world entity types they select and listed in the prompt.

Each completed batch is saved to `grounding_requests.json`. Existing definitions
are supplied to later batches for reuse. Each new relation gets its own Factory job.
Running `prepare` again preserves completed batches and prepares unfinished ones.

Run drafting with the configured API model:

```bash
uv run --env-file resym/.env --no-sync resym-init draft \
  --workspace resym/tmp --config resym/config/llm.json
```

In Docker, run the same command through `run_in_docker.sh` with
`--with-llm-credentials`, using `--workspace tmp --config config/llm.json`.
One `draft` invocation proposes relations and drafts their Factories.
Rerunning it skips saved replies. An invalid reply that reaches the output-token
limit gets one retry with twice that allowance, within the configured attempt
count. If it still fails, the job remains available for a later run. Token usage
and output limits are recorded in `tmp/initialization/llm_transcript.jsonl`.

Import checks source consistency and candidate structure; it never executes
Factory source.

Factory drafts use the cited query source and declared input signatures. Missing
query support is saved as `unsupported_reason`, without submitting executable code
or repeatedly retrying the same refusal. Static checks do not establish semantic
correctness: human review must still check that the query proves the stated relation.
The catalog also scans public typed fields and properties of the world-model
packages (`world_description`, `robots`, `semantic_annotations`, `spatial_types`).
Draft prompts show the members of the supplied types, their ancestors, and every
type reachable through the declared types of those members; access is read-only and checked
against the receiver type. Generated factories receive native CRAM entities resolved from
reSym object references (`native_arguments`). Factories approved before this flag,
such as the hand-written drawer factories, keep receiving object references.
The report separates `unsupported` from `failed`, and includes `review_notes` for
possible reuse and unused roles. These notes also appear in candidate review evidence.

### Review and approve

The initialization uses three persistent review workspaces:

| Workspace under `resym/tmp/` | Content | Viewer page |
|---|---|---|
| `grounding_factory_workspace/` | Query source records, factory candidates, `catalog.json`, approved Python modules under `approved/` | `/grounding-factories` |
| `contract_workspace/` | Contract candidates, approved `contracts.json`, review log | `/capabilities` |
| `realization_workspace/` | Action-parameter mapping candidates, approved `realizations.json`, review log | `/capabilities` |

1. Open <http://127.0.0.1:5000/grounding-factories> and approve Factory candidates.
2. Open <http://127.0.0.1:5000/capabilities> and approve Contracts.
3. Run `prepare` again, draft the realization jobs, and approve the mappings.

Approved artifacts remain in `resym/tmp/` and are loaded by later task runs.

## Running

`run_in_docker.sh` deploys the container and runs whatever command you give
it; the command is an argument, not a built-in mode:

```bash
# core tests
./resym/scripts/run_in_docker.sh pytest -q /opt/cram/test/resym_test
# the drawer demo
./resym/scripts/run_in_docker.sh python -m experiments.resym.open_drawer apartment
# anything that calls a real model needs the credentials forwarded
./resym/scripts/run_in_docker.sh --with-llm-credentials python -m <module> [arguments]
```

Run records (worlds, plans, patches, review evidence, full provenance) are
written to `resym/runs/` and stay out of version control.
