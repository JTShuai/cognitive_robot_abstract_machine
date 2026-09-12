# reSym

reSym builds on two fixed CRAM layers: **Coraplex**, whose native actions are
the robot's executable capability interface, and **krrood** with the Semantic
Digital Twin, whose reviewed entity queries are the source of Boolean truth
about the world. On top of them, reSym maintains the predicates, operators,
and execution bindings that task planning needs as retrievable, repairable,
and versioned symbolic assets. Symbol gaps and execution failures are
diagnosed into structured certificates, a retrieval-augmented LLM agent
proposes candidate repairs, and a deterministic curator validates and decides
admission — so the same Coraplex action primitives adapt to new tasks and
scenes without changes to their implementation code, and nothing the LLM
produces enters the query or execution path unreviewed.

## Trust boundary

- The symbolic model is data: predicates, operators, and execution bindings
  are typed dataclasses. The LLM proposes data-shaped patches, and may draft
  restricted world queries as review candidates — but nothing it produces
  executes unreviewed, and it never admits its own proposals or drives the
  robot.
- The curator is a deterministic program: it independently checks types,
  registry whitelists, and effect/contract consistency. Admission creates a
  new version; bad versions can be quarantined and rolled back.
- When the platform lacks a required skill, the agent emits a structured
  capability-gap report and stops. New skill implementations always enter the
  trusted surface through human review.

## Core loop

```text
natural-language task → structured goal (model gap or clarification if not expressible)
  → Boolean grounding + PDDL planning → per-action precondition recheck
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
  platform/       capability contracts, embodiment, grounding-factory review, feasibility
  planning/       selection, grounding, PDDL, monitored execution
  repair/         failure certificates, repair agents, curator, versioning
  knowledge/      UniDomain corpus, ontology, retrieval
  llm/ interfaces/ observability/ evaluation/
experiments/src/experiments/resym/   scenes, grounding assets, task models, ICRA protocol
test/resym_test/                     core tests (host-runnable)
test/experiments_test/resym/         scene and experiment tests (container)
```

## First-time initialization

Run the commands below from the CRAM monorepo root. Choose either Docker or a
local installation; they are independent setup paths.

### Option 1: Docker

Docker provides ROS 2, the CRAM workspace, Python dependencies, Fast Downward,
and visualization dependencies:

```bash
docker build -f resym/Dockerfile -t cram:jazzy-resym .
```

The runtime scripts bind-mount the current checkout. Populate its ignored
`resym/augment_dataset/` directory from the container so no local Python
environment is required:

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD:/opt/cram" \
  -w /opt/cram/resym \
  cram:jazzy-resym \
  bash -c 'uv run --active --no-sync resym-install-ontologies &&
           uv run --active --no-sync resym-install-corpus'
```

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
  --grounding-workspace resym/tmp/grounding_factory_workspace
```

The ontology installer downloads pinned SOMA and IEEE 1872 files. The corpus
installer downloads, verifies, preprocesses, and freezes the pinned UniDomain
release. Both use versioned manifests and SHA-256 checksums. The generated
data, local grounding workspace, and review decisions remain outside Git.

### Configure the LLM

This step is needed only for runs that use the repair agent. Credentials and
model settings stay outside version control:

```bash
cp resym/.env.example resym/.env
cp resym/config/llm.example.json resym/config/llm.json
```

Set `CLIENT_TYPE`, `API_KEY`, and `BASE_URL` in `resym/.env`. Set the model,
generation arguments, and retry policy in `resym/config/llm.json`. LLM calls
go through `llm-agent-kit`; `run_in_docker.sh` injects `resym/.env` only for
the `smoke` and `experiment` commands.

### Complete the first-start review

For either setup path, open
<http://127.0.0.1:5000/grounding-factories>. At startup, reSym scans the
installed kRrood and Semantic Digital Twin query primitives. New or changed
items enter the review queue. Enter a reviewer name and approve only the
vocabulary entries whose metadata and semantics are correct. Agent-authored
factory candidates appear on the same page; approval validates and
materializes them into the local workspace before they become executable.
Restarting with the same workspace preserves decisions when source checksums
have not changed. The workspace is local and ignored by Git.

The `/capabilities` page shows the Coraplex actions discovered at startup,
their reviewed `CapabilityContract` mappings, robot requirements, and any
unmapped actions that still need platform integration.

## Running

Common commands:

```bash
# core tests
./resym/scripts/run_in_docker.sh test -q /opt/cram/test/resym_test
# real-model smoke episode
./resym/scripts/run_in_docker.sh smoke --backends agentic-rag --template missing-close-operator --seed 1
# E1/E2 experiments (require the checksum-matched frozen splits)
./resym/scripts/run_in_docker.sh experiment
```

Run records (worlds, plans, patches, review evidence, full provenance) are
written to `resym/runs/` and stay out of version control.
