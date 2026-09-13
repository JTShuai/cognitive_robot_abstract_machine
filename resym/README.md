# reSym

reSym builds on two fixed CRAM layers: **Coraplex**, whose native actions are
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
  → select relevant symbols and an object subset → Boolean grounding + PDDL planning
  → per-action precondition recheck
  → execution through capability bindings (Coraplex) → independent effect and goal checks
failure → programmatic failure certificate → retrieval-augmented repair agent
  (UniDomain corpus, tool loop) → minimal model patch → curator admission
  → version store → replan
```

### Task-scoped objects

CRAM retains the full world. reSym starts planning with the goal objects and
robot, follows native annotation references and storage owners, and queries
SDT support/containment for the selected bodies where geometry is available.
Structural dependencies can be reached through intermediate annotations that
do not themselves need to enter PDDL. Looking up an owner does not pull in all
sibling parts, and selecting a container does not pull in all its occupants.
Missing operator parameter types receive an initial typed
candidate; ordering is deterministic, not learned relevance.

Only the selected objects participate in predicate argument enumeration and
PDDL `:objects`. Grounding factories, execution and final checks retain access
to the complete object directory and SDT: an excluded object can still make
a bowl nonempty or obstruct motion. All selected ground atoms are evaluated;
unevaluated atoms are not recorded as false by reSym.

If the planner proves the subset unsolvable, candidate sets grow until a plan
is found or the relevant types are exhausted. A repeated failed plan may also
trigger expansion before it is refused. Timeouts, invalid planner inputs,
unavailable capabilities and grounding failures are not scope evidence.
Expansion does not consume execution-replanning rounds. Each new execution
round re-reads relationships and truth values; object identities come from
the supplied world directory. This does not guarantee a globally optimal plan.

The live viewer displays selected/total object counts, inclusion reasons,
scope attempts and expansions. Each attempt's PDDL files are saved under
`round-N/scope-M/` in the task planning directory. Task results separately
count predicate evaluations, scope expansions, and selection/planning time.

An optional planning-object agent (`resym.interfaces.object_selection`) can
add objects ahead of typed growth. It reuses the repair tool loop with three
read-only tools: `query_candidate_objects` (type, colour, name fragment over
the complete directory), `inspect_object_relations` (references, storage
owners, support and containment hints) and `propose_planning_objects`
(ordered names with a rationale each). The program resolves every name,
rejects objects outside the typed bound, keeps goal objects and the robot
unconditionally, and runs the queries itself; the model never supplies truth
values. It is consulted once for the initial selection and again on every
expansion with the planner's evidence; when it adds nothing, typed growth
proceeds as before. Entering through `diagnose_instruction` binds the agent to
the natural-language instruction; a goal given directly is advised from the
goal literals alone. Consultations are recorded as `object_scope_advised`
events and shown with each object's rationale in the viewer. The open-drawer
experiment enables the agent when `RESYM_OBJECT_AGENT_CONFIG` names a
language-model configuration.

Goals with an interchangeable object ("any bowl") are not the selector's
concern: `GoalTranslator` currently turns an ambiguous object reference into a
clarification request, and keeping that existential reading would be a
goal-language feature. Neither the selector nor the agent ever replaces an
object already bound in the goal.

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

The ontology installer downloads pinned SOMA and IEEE 1872 files. The corpus
installer downloads, verifies, preprocesses, and freezes the pinned UniDomain
release. Both use versioned manifests and SHA-256 checksums. The generated
data, local review workspaces, and review decisions remain outside Git.

### Configure the LLM

Configure this before calling an LLM for task understanding, initialization
drafting, or repair. Browsing and reviewing existing candidates does not
require an LLM. Credentials and model settings stay outside version control:

```bash
cp resym/.env.example resym/.env
cp resym/config/llm.example.json resym/config/llm.json
```

Set `CLIENT_TYPE`, `API_KEY`, and `BASE_URL` in `resym/.env`. Set the model,
generation arguments, and retry policy in `resym/config/llm.json`. LLM calls
go through `llm-agent-kit`. For local LLM commands, use
`uv run --env-file resym/.env --no-sync ...` from the monorepo root to load
credentials. For container commands, pass `--with-llm-credentials` to
`run_in_docker.sh`; it forwards `resym/.env` into the container.
`resym-init draft --config ...` loads the model configuration and constructs
the client. Starting the Viewer does not call an LLM.

### Prepare initialization drafts

Initialization is an explicit setup operation. Daily task execution loads
approved artifacts without rerunning it. The commands below run from the
monorepo root after installation. For an existing environment that has not
installed the new entry point, replace `resym-init` with
`python -m resym.interfaces.initialization`.

```bash
uv run --no-sync resym-init prepare --workspace resym/tmp
```

This scans the official interfaces and writes authoring materials to
`resym/tmp/initialization/`: `instructions.md`, `platform.json`, `jobs.json`,
and a `responses/` directory. Each job contains its instructions and response
JSON schema. No credentials are needed, and nothing is approved automatically.
Use `--action <source_id>` to limit the scanned actions selected for drafting;
the IDs are listed in `platform.json`.

Grounding also needs a description of the relations the deployment requires.
Edit the generated `grounding_requests.json` using
`grounding_requests.schema.json`, then run `prepare` again. Each request gives
`proposed_uid`, `semantic_name`, `meaning`, typed `roles`, and optional bounded
`parameters`. Alternatively supply a JSON array with `--requests <file>`.
The scanner does not invent task relations; an empty requests list produces
no grounding jobs.

Choose either drafting route:

- **External assistant, without `.env`:** ask the assistant to read
  `resym/tmp/initialization/instructions.md` and write the job responses to
  `responses/<job_id>.json`. Then import them:

  ```bash
  uv run --no-sync resym-init import --workspace resym/tmp --generated-by codex
  ```

  To read replies from another directory, pass it after `import`.

- **Configured API model:** load credentials and run drafting. Valid replies
  go through the same import checks automatically:

  ```bash
  uv run --env-file resym/.env --no-sync resym-init draft \
    --workspace resym/tmp --config resym/config/llm.json
  ```

  For Docker, use the same initialization stages through the runtime script:

  ```bash
  ./resym/scripts/run_in_docker.sh resym-init prepare --workspace tmp
  ./resym/scripts/run_in_docker.sh --with-llm-credentials resym-init draft \
    --workspace tmp --config config/llm.json
  ./resym/scripts/run_in_docker.sh resym-init import --workspace tmp --generated-by codex
  ```

`import_report.json` records submitted, already present, missing and invalid
responses. API exchanges are saved in `llm_transcript.jsonl`. API drafting
skips saved responses; repeated imports do not duplicate identical candidates.
Invalid external responses can be corrected and imported again. Import checks
source consistency and candidate structure; it never executes factory source.

### Complete the first-start review

Both startup commands enable three persistent review workspaces:

| Workspace under `resym/tmp/` | Content | Viewer page |
|---|---|---|
| `grounding_factory_workspace/` | Query source records, factory candidates, `catalog.json`, approved Python modules under `approved/` | `/grounding-factories` |
| `contract_workspace/` | Contract candidates, approved `contracts.json`, review log | `/capabilities` |
| `realization_workspace/` | Action-parameter mapping candidates, approved `realizations.json`, review log | `/capabilities` |

1. Open <http://127.0.0.1:5000/grounding-factories>. Startup scans the installed
   kRrood and Semantic Digital Twin query vocabulary. These official interfaces
   are trusted and immediately available for factory drafting; no per-interface
   approval is required. The page displays their signatures and source checksums.
2. Review factory candidates imported by the initialization program on the
   same page. Approval validates the source and
   materializes a local Python module that predicate plans can reference.
3. Open <http://127.0.0.1:5000/capabilities> to inspect discovered Coraplex
   actions and review the submitted contracts.
4. After approving contracts, run `prepare` again with the same workspace.
   It now generates realization jobs for the approved contracts. Use either
   drafting route above, then review the resulting mappings on `/capabilities`.
   They bind contract roles, context values, or constants to native action
   parameters and declare required robot resources. Approving a contract
   alone does not make its action executable.

Each stage persists its output, so the initialization process can exit before
human review. Repeated preparation skips pending or approved artifacts.
After rejecting a candidate, run `prepare` again: the new job includes the
reviewer's feedback and keeps the old response for reference.
If a native source changed after preparation, prepare fresh jobs before
importing. Human review is the only step that approves generated assets.
The drawer demo independently uses reference artifacts from the experiments package.

Reuse the same workspace paths when loading catalogs for task execution.
Official query vocabulary refreshes automatically on startup. Generated factories
remain subject to source and dependency checksums; changes require factory review.
Approved factories and review records stay local and
outside Git; approval does not publish them to a remote repository.

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
