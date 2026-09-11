# reSym

reSym builds on : **Coraplex**, whose native actions are
the robot's executable capability interface, and **krrood** with the Semantic
Digital Twin, whose reviewed entity queries are the source of Boolean truth
about the world. On top of them, reSym maintains the predicates, operators,
and execution bindings that task planning needs as retrievable, repairable,
and versioned symbolic assets. Symbol gaps and execution failures are
diagnosed into structured certificates, a retrieval-augmented LLM agent
proposes candidate repairs, and a deterministic curator validates and decides
admission — so the same Coraplex action primitives adapt to new tasks and
scenes without changes to their implementation code, and no generated code
ever enters the query or execution path.

## Trust boundary

- The symbolic model is data: predicates, operators, and execution bindings
  are typed dataclasses. The LLM only proposes data-shaped patches; it never
  writes executable code, admits its own proposals, or drives the robot.
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
  platform/       capability contracts, embodiment, evaluators, Coraplex catalog
  planning/       selection, grounding, PDDL, monitored execution
  repair/         failure certificates, repair agents, curator, versioning
  knowledge/      UniDomain corpus, ontology, retrieval
  llm/ interfaces/ observability/ evaluation/
experiments/src/experiments/resym/   scenes, seed libraries, ICRA protocol
test/resym_test/                     core tests (host-runnable)
test/experiments_test/resym/         scene and experiment tests (container)
```

## Getting started

Full functionality needs ROS 2, CRAM, Coraplex, and Fast Downward; Docker is
recommended:

```bash
# from the CRAM monorepo root
docker build -f resym/Dockerfile -t cram:jazzy-resym .
```

External data (ontology files and the UniDomain retrieval corpus) is not kept
in Git; it installs into `resym/augment_dataset/`. The Docker build does both
automatically; locally, run from the CRAM root:

```bash
uv sync --extra dev --active
uv run --active --no-sync resym-install-ontologies
uv run --active --no-sync resym-install-corpus
```

- `resym-install-ontologies` downloads and verifies pinned SOMA and IEEE 1872
  OWL files into `augment_dataset/ontology/`.
- `resym-install-corpus` downloads the pinned UniDomain archive (Hugging Face
  `SII-PrimoButterfly/UniDomain-Data`, ~4 MB), verifies and extracts it into
  `augment_dataset/unidomain/`, then parses, deduplicates, and freezes it into
  the retrieval release `augment_dataset/corpus_release/r1/`, which must
  reproduce the pinned fragments checksum. The optional dense index is built
  separately (`uv run python -m resym.knowledge.dense`, needs
  sentence-transformers).

Both download manifests pin sources, versions, and SHA-256 checksums.

LLM calls go through `llm-agent-kit`; credentials stay out of the repository:

```bash
cp .env.example .env
cp config/llm.example.json config/llm.json
```

Common commands:

```bash
# core tests
./scripts/run_in_docker.sh test -q /opt/cram/test/resym_test
# real-model smoke episode
./scripts/run_in_docker.sh smoke --backends agentic-rag --template missing-close-operator --seed 1
# E1/E2 experiments (require the checksum-matched frozen splits)
./scripts/run_in_docker.sh experiment
```

Run records (worlds, plans, patches, review evidence, full provenance) are
written to `runs/` and stay out of version control.
