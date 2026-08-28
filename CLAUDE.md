# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Repository Shape

This is the **CRAM (Cognitive Robot Abstract Machine)** monorepo: a single `uv` workspace
(`[tool.uv.workspace]` in `pyproject.toml`) wiring together 9 editable subpackages. Despite the
README mentioning submodules, the packages are vendored directly into the tree (there is no
`.gitmodules`). The root package is a meta-package with no code of its own — all real code lives
under `<package>/src/<package>/`.

Python is pinned to **>=3.12,<3.13**. Several packages (`coraplex`, `semantic_digital_twin`,
`robokudo`, `giskardpy`) optionally depend on **ROS 2 (Jazzy)**; code guards ROS imports with
`try/except ImportError` and helpers like `rclpy_installed()` so the core runs without ROS.

## Common Commands

Per global preference, drive everything through `uv` — never call bare `python`/`pip`.

```bash
# Install the whole workspace (editable) with dev tooling
uv sync --extra dev --active

# Run a package's test suite (tests live centrally under test/, NOT inside each package)
uv run pytest test/coraplex_test
uv run pytest test/semantic_digital_twin_test

# Run a single test file / class / method
uv run pytest test/coraplex_test/test_some_file.py::TestClass::test_method

# Parallel run, as CI does it (pytest-xdist; conftest assigns a ROS_DOMAIN_ID per worker)
uv run pytest -n auto test/krrood_test

# Format (the only pre-commit hook is Black 25.x)
uv run black .
```

`pytest.ini` sets `addopts = -sv`, so output is verbose and unbuffered by default. CI runs each
library independently as a matrix (`.github/workflows/ci.yml`) inside a `:jazzy` ROS container.

## ORM / ORMatic (read this before touching persistence)

Several packages persist their dataclasses to a relational schema via **ORMatic** (in
`krrood/ormatic`). The generated mapping lives in `<package>/src/<package>/orm/ormatic_interface.py`
for `semantic_digital_twin`, `coraplex`, and `experiments`.

- **Never hand-edit `ormatic_interface.py` files** — they are generated.
- To change the schema, edit the source dataclasses (and the `generate_orm.py` registration if a
  class needs ignoring/mapping), then regenerate. The regeneration script uses paths relative to
  `scripts/`, so run it from there:

  ```bash
  cd scripts && uv run python regenerate_all_orm.py
  ```

  This clears and re-emits the three interface files by invoking each package's
  `scripts/generate_orm.py`. If regeneration does not resolve an ORM issue, consult a developer
  rather than editing the interface by hand.

## Architecture & Package Layering

Cross-package imports flow strictly from foundational to high-level (top of the list = no siblings,
bottom = depends on everything):

- **`random_events`** — foundational set/interval/sigma-algebra primitives for describing random
  variables and events. Has a C++ core (`src/random_events_lib`, built/tested with Bazel in CI).
- **`krrood`** — *Knowledge Representation & Reasoning through Object-Oriented Design*. The
  shared infrastructure layer: **ORMatic** (dataclass→SQLAlchemy ORM), **EQL** (Entity Query
  Language), `ripple_down_rules`, `symbol_graph`, `ontomatic`, class-diagram introspection. Depended
  on by nearly everything.
- **`probabilistic_model`** — unified API for probabilistic models (probabilistic circuits,
  Bayesian networks, distributions). Depends on `krrood`, `random_events`.
- **`semantic_digital_twin`** — the **world model**. `World` (`world.py`) is a kinematic scene graph
  (bodies, links, connections, degrees of freedom) built on `rustworkx`, with semantic annotations
  ("Views" like drawers/handles), collision checking, spatial types, and ORM persistence. This is
  the central shared state most other packages operate on. Depends on `krrood`, `random_events`,
  `probabilistic_model`, `giskardpy`.
- **`giskardpy`** — whole-body motion control via constraint/optimization-based task-space control
  (`qp`, `motion_statechart`, `tree`). Mutually coupled with `semantic_digital_twin`.
- **`coraplex`** — *Cognitive Orchestrated Reasoning Architecture Planning Executive*. The top-level
  planning/execution stack: **designators** (intent-based, late-bound specs for actions/motions/
  objects/locations), a composable **plan language** (`language.py`, `plan.py` — sequential/parallel/
  retry/repeat/monitor plan-node trees), motion execution, and ORM-backed execution traces. Depends
  on all of the above.
- **`robokudo`** — ROS 2 robot perception framework. Depends on `krrood`, `semantic_digital_twin`.
- **`physics_simulators`** — thin simulator wrappers (`mujoco_simulator.py`, `base_simulator.py`).
- **`experiments`** — research scripts/experiments consuming `coraplex`, `giskardpy`,
  `semantic_digital_twin`, `krrood` (top consumer; also has its own generated ORM).

When working on spatial types, transforms, or connections in `semantic_digital_twin`, follow the
homogeneous-transform naming convention (`root_T_tip`, `root_P_tip`, etc.) documented in
`semantic_digital_twin/doc/style_guide.md`.

## Testing Conventions

- Tests for every package live under the **central** `test/<package>_test/` directory, not inside
  the packages themselves.
- `test/conftest.py` is the single source of shared fixtures for the whole monorepo. It builds
  expensive `World`s (robots like PR2/HSRB/Tiago/Stretch, apartment/kitchen scenes) once per session
  and exposes them through a strict discipline documented at the top of the file:
  - Session-scoped `_..._setup` world fixtures and "merging" fixtures **must not be used directly**
    (mutating the shared instance leaks into other tests).
  - Use the **function-scoped** `*_state_reset` fixtures for tests that only change world *state*,
    and the `*_copy` (deepcopy) fixtures for tests that change the world *model*.
  - An autouse fixture rebuilds the `SymbolGraph` around each test, and a module-scoped guard fails
    the run if more than 30 `World`s are alive (leak detector). Reuse existing fixtures rather than
    constructing worlds from scratch.
