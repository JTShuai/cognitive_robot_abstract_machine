"""
The Flask log viewer (:mod:`resym.observability.viewer`).

Self-contained: a run directory is built with the real :class:`RunRecorder` (stdlib
only) and served through Flask's test client, so this needs neither the CRAM stack nor a
bound port. Skipped when the optional ``flask`` extra is absent.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("flask")

from resym.core.model import (  # noqa: E402
    CapabilityContract,
    CapabilityRef,
    CapabilityRole,
    Literal,
    OntologyAlignment,
    Operator,
    OperatorExecutionBinding,
    PredicateRef,
    PredicateSymbol,
    Provenance,
    RoleBinding,
    SymbolLibrary,
    TruthProcedureRef,
)
from resym.observability.viewer import create_app  # noqa: E402
from resym.observability.runlog import RunRecorder  # noqa: E402

from resym.core.model import SymbolType
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer

DRAWER_TYPE = SymbolType.from_python_type(Drawer)


def _seed_run(root):
    recorder = RunRecorder.create("smoke", root=root)

    class Universe:
        objects = {
            "d1": type(
                "O", (), {"name": "d1", "symbol_type": DRAWER_TYPE, "body": "b"}
            )()
        }

    recorder.record_world(Universe(), scene="kitchen")
    recorder.transcript_path.write_text(
        '{"agent_name":"symbol-proposer","model":"m","attempt":1,'
        '"prompt":"PROMPT-TEXT","response":"RESPONSE-TEXT","parse_error":null}\n',
        encoding="utf-8",
    )
    recorder.artifact("C_solve", "task/domain.pddl", "(define (domain d))")
    recorder.json(
        "C_solve", "task/result.json", {"plan": ["(a)"], "evaluation_count": 7}
    )
    sink = recorder.task_event_sink("task")
    sink("task_started", {"goal": [{"display": "opened(d1)"}]})
    sink("round_started", {"round": 1})
    sink(
        "plan_generated",
        {
            "actions": [
                {"operator": "open", "arguments": ["d1"], "display": "(open d1)"}
            ]
        },
    )
    sink(
        "action_started",
        {
            "action_index": 0,
            "action": {"operator": "open", "arguments": ["d1"], "display": "(open d1)"},
        },
    )
    sink(
        "execution_request_created",
        {
            "action_index": 0,
            "capability_ref": {"uid": "resym:Open", "version": "1"},
            "arguments": {"patient": "d1"},
            "role_bindings": {"patient": {"source": "parameter", "value": "d"}},
        },
    )
    sink(
        "platform_execution_started",
        {"action_index": 0, "realization": "CoraplexSkillRealization"},
    )
    sink("coraplex_action_selected", {"action": "OpenAction"})
    sink(
        "platform_result",
        {
            "action_index": 0,
            "status": "succeeded",
            "code": "CORAPLEX_EXECUTION_SUCCEEDED",
        },
    )
    sink(
        "effect_checked",
        {
            "action_index": 0,
            "literal": {"display": "opened(d1)"},
            "truth": "true",
            "required": "true",
            "satisfied": True,
            "reason": None,
        },
    )
    sink(
        "effect_checked",
        {
            "action_index": 0,
            "literal": {"display": "closed(d1)"},
            "truth": "false",
            "required": "false",
            "satisfied": True,
            "reason": None,
        },
    )
    sink("action_completed", {"action_index": 0})
    sink("task_succeeded", {"round": 1, "actions": 1})
    video = recorder.stage_dir("D_visualization") / "rviz_execution.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    recorder.finish({"backend": "repair-agent"})
    return recorder.directory.name


def _client(root):
    return create_app(root).test_client()


def test_index_lists_runs(tmp_path):
    name = _seed_run(tmp_path)
    response = _client(tmp_path).get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert name in body
    assert ">repair-agent</span>" in body


def test_run_page_renders_all_stages(tmp_path):
    name = _seed_run(tmp_path)
    body = _client(tmp_path).get(f"/run/{name}").get_data(as_text=True)
    assert "Stage A" in body and "repair process" in body and "Stage C" in body
    assert "domain.pddl" in body
    assert "(define (domain d))" in body
    # long transcript fields are collapsed into <details>, not inline text
    assert "symbol-proposer" in body
    assert "<details>" in body and "RESPONSE-TEXT" in body
    assert "live execution" in body
    assert "Stage D · visualization" in body
    assert "video recording" in body
    assert "rviz_execution.mp4" in body


def test_live_page_renders_actual_request_and_coraplex_action(tmp_path):
    name = _seed_run(tmp_path)
    client = _client(tmp_path)
    body = client.get(f"/run/{name}/live").get_data(as_text=True)

    assert "opened(d1)" in body
    assert "(open d1)" in body
    assert "resym:Open@1" in body
    assert "patient=d1" in body
    assert "OpenAction" in body
    assert "CoraplexSkillRealization" in body
    assert "CapabilityContract" in body
    assert "ExecutionRequest" in body
    assert "CORAPLEX_EXECUTION_SUCCEEDED" in body
    assert "succeeded" in body
    assert "Model evaluation" in body and "Plan ready" in body
    assert "Executing" in body and "Verifying" in body
    assert "resym:Open → OpenAction → succeeded" in body
    assert "closed(d1) · expected FALSE · observed FALSE" in body
    assert "✓ PASS" in body
    fragment = client.get(f"/run/{name}/live/fragment")
    assert fragment.status_code == 200
    assert "OpenAction" in fragment.get_data(as_text=True)


def test_live_page_holds_generated_plan_before_execution(tmp_path):
    recorder = RunRecorder.create("preview", root=tmp_path)
    sink = recorder.task_event_sink("task")
    sink("task_started", {"goal": [{"display": "opened(d1)"}]})
    sink("round_started", {"round": 1})
    sink(
        "grounding_completed",
        {"evaluations": 9, "true": 2, "false": 7},
    )
    sink(
        "plan_generated",
        {"actions": [{"operator": "open", "display": "(open d1)"}]},
    )
    sink("plan_preview_started", {"seconds": 6})

    body = (
        _client(tmp_path)
        .get(f"/run/{recorder.directory.name}/live")
        .get_data(as_text=True)
    )

    assert "reviewing plan (6s)" in body
    assert "predicate evaluations" in body
    assert "(open d1)" in body
    assert "phase-step active'>2 · Plan ready" in body


def test_latest_live_follows_newest_run(tmp_path):
    name = _seed_run(tmp_path)
    response = _client(tmp_path).get("/live")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert name in body
    assert "OpenAction" in body


def test_missing_run_is_404(tmp_path):
    _seed_run(tmp_path)
    assert _client(tmp_path).get("/run/nope").status_code == 404


def test_path_traversal_blocked(tmp_path):
    _seed_run(tmp_path)
    assert _client(tmp_path).get("/run/..%2f..").status_code == 404


def test_empty_runs_root(tmp_path):
    body = _client(tmp_path / "empty").get("/").get_data(as_text=True)
    assert "No runs" in body


def _seed_experiment_run(root):
    """
    A run directory shaped like experiments.resym.icra.run output.
    """
    import json

    recorder = RunRecorder.create("e1e2", root=root)
    recorder.record_episode(
        {
            "backend": "oracle-reference",
            "template_id": "missing-close-operator",
            "episode_index": 0,
            "status": "patch_proposed",
            "admitted": True,
            "correct_repair": True,
            "false_admission": False,
            "budget": {"candidates_used": 1, "estimated_tokens_used": 0},
            "admission_evidence": {"static_objections": []},
            "events": [],
        }
    )
    recorder.record_episode(
        {
            "backend": "repair-agent",
            "template_id": "missing-close-operator",
            "episode_index": 0,
            "status": "infrastructure-error",
            "admitted": False,
            "correct_repair": False,
            "false_admission": False,
            "budget": {},
            "events": [{"step": "infrastructure", "error": "RateLimitError: 429"}],
        }
    )
    directory = recorder.directory
    (directory / "e1_report.txt").write_text("E1: matched-budget executable repair\n")
    (directory / "gate_p2.txt").write_text("Gate P2: DOWNGRADE — example\n")
    (directory / "e1_summary.json").write_text(
        json.dumps(
            {
                "oracle-reference": {
                    "correct_repair": {
                        "successes": 1,
                        "total": 1,
                        "rate": 1.0,
                        "wilson_low": 0.207,
                        "wilson_high": 1.0,
                    }
                }
            }
        )
    )
    (directory / "e2_decisions.jsonl").write_text(
        '{"policy":"single-witness","template_id":"missing-close-operator",'
        '"candidate":"missing-reachability-precondition","behaviorally_correct":false,'
        '"admitted":true,"tests_run":1,"false_admission":true,'
        '"missed_admission":false,"held_out_failures":["[negative] claimed"]}\n'
    )
    # heavy intermediate trees must not be rendered
    (directory / "work" / "admission").mkdir(parents=True)
    (directory / "work" / "admission" / "scratch.json").write_text("{}")
    recorder.finish({})
    return directory.name


def test_experiment_files_use_generic_json_rendering(tmp_path):
    name = _seed_experiment_run(tmp_path)
    body = _client(tmp_path).get(f"/run/{name}").get_data(as_text=True)
    # top-level result files are rendered in a Results section
    assert "Results" in body
    assert "E1: matched-budget executable repair" in body
    assert "Gate P2: DOWNGRADE" in body
    # Experiment payloads remain visible without domain-specific interpretation.
    assert "wilson_low" in body
    assert "oracle-reference" in body
    assert "infrastructure-error" in body
    assert "single-witness" in body
    assert "missing-reachability-precondition" in body
    # the intermediate working tree is named, not rendered
    assert "not rendered" in body and "scratch.json" not in body


def test_experiment_summaries_render_with_rate_bars(tmp_path):
    name = _seed_experiment_run(tmp_path)
    body = _client(tmp_path).get(f"/run/{name}").get_data(as_text=True)
    # e1_summary.json becomes a backend comparison table with Wilson bars
    assert "Wilson" in body and "rate-bar" in body
    # episodes.jsonl becomes an aggregate, not a raw event dump
    assert "Outcome counts per repair backend" in body
    assert "patch_proposed" in body and "infrastructure-error" in body
    # the admission matrix carries a per-policy summary with hover hints
    assert "curation policy" in body and "mean tests" in body
    assert "data-tip=" in body


def test_index_rows_show_outcomes(tmp_path):
    import json as _json

    _seed_experiment_run(tmp_path)  # writes gate_p2.txt: DOWNGRADE
    task = RunRecorder.create("demo", root=tmp_path)
    task.finish({"backend": "coraplex", "succeeded": True})
    unfinished = tmp_path / "20990101-000000-crashed"
    unfinished.mkdir()
    (unfinished / "run.json").write_text(
        _json.dumps({"label": "crashed", "started_at": "2026-01-01T00:00:00"})
    )
    body = _client(tmp_path).get("/").get_data(as_text=True)
    assert "gate: DOWNGRADE" in body
    assert "&#10003; succeeded" in body
    assert "unfinished" in body
    # the label facet filters the list down to one experiment's runs
    assert "?label=e1e2" in body
    filtered = _client(tmp_path).get("/?label=demo").get_data(as_text=True)
    assert "-demo</b>" in filtered
    assert "e1e2</b>" not in filtered


def test_task_run_page_leads_with_verdict(tmp_path):
    recorder = RunRecorder.create("demo", root=tmp_path)
    recorder.json(
        "C_solve",
        "apartment_open/result.json",
        {
            "plan": ["(navigate r d)", "(open-drawer r h d)"],
            "execution": {"succeeded": True},
        },
    )
    recorder.finish({"backend": "coraplex", "succeeded": True})
    body = (
        _client(tmp_path).get(f"/run/{recorder.directory.name}").get_data(as_text=True)
    )
    assert "task outcome" in body
    assert "&#10003; succeeded" in body
    assert "<code>navigate</code> &rarr; <code>open-drawer</code>" in body
    assert "executed &amp; verified" in body


def test_run_meta_shows_stage_timeline(tmp_path):
    import json as _json

    run = tmp_path / "20260101-000000-timed"
    (run / "A_world").mkdir(parents=True)
    (run / "C_solve").mkdir()
    (run / "run.json").write_text(
        _json.dumps(
            {
                "label": "timed",
                "started_at": "2026-01-01T00:00:00",
                "ended_at": "2026-01-01T00:04:28",
                "summary": {"succeeded": True},
            }
        )
    )
    (run / "A_world" / "events.jsonl").write_text(
        '{"at":"2026-01-01T00:01:09","event":"world_loaded"}\n'
    )
    (run / "C_solve" / "events.jsonl").write_text(
        '{"at":"2026-01-01T00:04:28","event":"task"}\n'
    )
    body = _client(tmp_path).get(f"/run/{run.name}").get_data(as_text=True)
    assert "where the time went" in body
    assert "1m 09s" in body and "3m 19s" in body


def test_episode_browser_groups_cases_with_facets(tmp_path):
    recorder = RunRecorder.create("e1e2", root=tmp_path)
    for backend, correct in (("agentic-rag", True), ("no-repair", False)):
        for template, group, seed in (
            ("missing-close-operator", "D1", 11),
            ("inverted-precondition", "D2", 22),
        ):
            recorder.record_episode(
                {
                    "backend": backend,
                    "template_id": template,
                    "group": group,
                    "proposal_seed": seed,
                    "proposal_context_id": f"{template}/scene-{seed}",
                    "episode_index": 0,
                    "status": "patch_proposed" if correct else "no_candidate",
                    "admitted": correct,
                    "correct_repair": correct,
                    "false_admission": False,
                    "budget": {"estimated_tokens_used": 1234},
                    "events": [],
                }
            )
    recorder.transcript_path.write_text(
        '{"agent_name":"symbol-proposer","model":"m","attempt":1,'
        '"prompt":"P","response":"R","parse_error":null}\n',
        encoding="utf-8",
    )
    recorder.finish({})
    name = recorder.directory.name
    client = _client(tmp_path)

    # the run overview keeps the aggregate card, now with a browse link
    overview = client.get(f"/run/{name}").get_data(as_text=True)
    assert "browse the 2 cases (4 episodes)" in overview

    # the artifact page regroups episodes into per-case boxes
    body = client.get(f"/run/{name}/artifact/D_experiment/episodes.jsonl").get_data(
        as_text=True
    )
    assert "4 episodes = 2 cases &times; 2 backends" in body
    assert "fault: missing-close-operator" in body and "scene-11" in body
    assert "correct repair" in body and "no fix found" in body
    assert "1,234 tokens" in body
    # each attempted case links to its Stage B conversations
    assert "read this case's LLM conversations" in body
    assert "../B_repair/llm_transcript.jsonl?fault=missing-close-operator" in body
    # a scene filter narrows to the one case a cross link points at
    pinned = client.get(
        f"/run/{name}/artifact/D_experiment/episodes.jsonl"
        "?fault=missing-close-operator&scene=11"
    ).get_data(as_text=True)
    assert "showing 1 of 2 cases" in pinned

    # facets narrow cases (group) and rows inside them (backend)
    grouped = client.get(
        f"/run/{name}/artifact/D_experiment/episodes.jsonl?group=D1"
    ).get_data(as_text=True)
    assert "fault: inverted-precondition" not in grouped
    assert "showing 1 of 2 cases" in grouped
    by_backend = client.get(
        f"/run/{name}/artifact/D_experiment/episodes.jsonl?backend=no-repair"
    ).get_data(as_text=True)
    assert "<code>agentic-rag</code>" not in by_backend

    # the Chinese page translates the case chrome
    chinese = client.get(
        f"/run/{name}/artifact/D_experiment/episodes.jsonl?lang=zh&group=D1"
    ).get_data(as_text=True)
    assert "命中 1 个，共 2 个案例" in chinese
    assert "正确修复" in chinese


def test_index_does_not_interpret_experiment_episode_files(tmp_path):
    name = _seed_experiment_run(tmp_path)
    body = _client(tmp_path).get("/").get_data(as_text=True)
    assert name in body
    assert "episodes" not in body


def _fixture_library(extra_opened_predicate: bool = False) -> SymbolLibrary:
    """
    The miniature articulation library the viewer pages are exercised with.
    """
    library = SymbolLibrary()
    library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            evaluator="drawer_closed",
            fluent=True,
        )
    )
    if extra_opened_predicate:
        library.add(
            PredicateSymbol(
                name="opened",
                parameter_types=(DRAWER_TYPE,),
                evaluator="drawer_opened",
                fluent=True,
                provenance=Provenance(source="repair-agent"),
            )
        )
    library.add(
        Operator(
            name="open-drawer",
            parameters=(("d", DRAWER_TYPE),),
            preconditions=(Literal("closed", ("d",)),),
            add_effects=(),
            delete_effects=(),
            execution_binding=OperatorExecutionBinding(
                capability_ref=CapabilityRef("resym:Articulation"),
                role_bindings=(("patient", RoleBinding.parameter("d")),),
            ),
        )
    )
    library.add_capability_contract(
        CapabilityContract(
            uid="resym:Articulation",
            label="articulation.state-transition",
            roles=(
                CapabilityRole(
                    name="patient",
                    accepted_symbol_types=(DRAWER_TYPE,),
                ),
            ),
            success_relation="state(patient) == target",
            verifiable_effects=("opened",),
            ontology_alignment=OntologyAlignment(
                relation="SPECIALIZATION",
                target_iri="http://soma#StateTransition",
                source_version="soma@2.1.0",
                decided_by="llm-agent",
                rationale="narrower than StateTransition",
            ),
        )
    )
    return library


_LIBRARY = _fixture_library().to_json()


def test_system_library_page_renders_readably(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    (library_dir / "seed_library.json").write_text(
        json.dumps(_LIBRARY), encoding="utf-8"
    )
    client = create_app(runs, library_dir=library_dir).test_client()
    body = client.get("/library").get_data(as_text=True)
    assert "seed_library" in body  # tab label
    assert "drawer_closed" in body  # predicate truth procedure
    assert "articulation.state-transition" in body  # contract label
    assert "StateTransition" in body  # ontology alignment
    assert "open-drawer" in body and "resym:Articulation@1" in body  # binding
    # the interactive relation graph is rendered with typed nodes and edges
    assert 'data-node="pred:closed"' in body or "data-node='pred:closed'" in body
    assert "gedge pre" in body  # precondition edge
    assert "gedge bind" in body  # operator-to-capability binding edge
    assert "binding:open-drawer" in body
    assert "Operator execution bindings" in body


def test_capability_catalog_page_shows_contract_realization_action_hierarchy(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    body = create_app(runs).test_client().get("/capabilities").get_data(as_text=True)

    assert "Coraplex capability catalog" in body
    assert "contracts</b>: 20" in body
    assert "resym:ArticulationStateChange" in body
    assert "OpenAction" in body
    assert "task-required" in body
    assert "adapter ready</b>: 20" in body
    assert "CapabilityContract" in body
    assert "implemented by" in body  # native realization row on each card


def test_capability_catalog_separates_actions_awaiting_semantic_review():
    from resym.observability.viewer import _render_capability_catalog

    body = _render_capability_catalog(
        {
            "contracts": [],
            "pending_actions": [
                {
                    "source_id": "coraplex:robot_plans.actions.custom.custom-action",
                    "action_class": "coraplex.robot_plans.actions.custom.CustomAction",
                    "summary": "A newly scanned action",
                }
            ],
            "summary": {
                "contracts": 0,
                "actions": 1,
                "ready": 0,
                "built_in_verification": 0,
                "pending_review": 1,
            },
        }
    )

    assert "awaiting review</b>: 1" in body
    assert "Actions awaiting semantic review" in body
    assert "CustomAction" in body


def test_capability_catalog_page_explains_the_chain_and_anchors_contracts(tmp_path):
    """
    The anatomy strip names every concept from predicate to adapter, and each contract
    card carries a stable anchor for cross-page links.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    body = create_app(runs).test_client().get("/capabilities").get_data(as_text=True)

    assert "How the pieces connect" in body
    assert "ExecutionRequest" in body  # the run-time lane of the anatomy
    assert "design time" in body and "run time" in body
    assert "id='cap-resym:ArticulationStateChange'" in body
    assert "used by operators" in body
    assert "none in the shipped libraries" in body  # no library_dir configured
    # entries are folded one-line summaries so the page stays scannable;
    # a #cap-<uid> jump unfolds its target via the hash handler
    assert "<details class=capability-entry" in body
    assert "<details class=capability-entry open" not in body
    assert "resymOpenHash" in body


def test_system_library_page_shows_versioned_predicate_references(tmp_path):
    modern_library = SymbolLibrary()
    modern_library.add(
        PredicateSymbol(
            name="closed",
            parameter_types=(DRAWER_TYPE,),
            evaluator="drawer_closed",
            fluent=True,
            uid="soma:Closed",
            version="2",
            truth_procedure_ref=TruthProcedureRef(
                "resym:truth-procedure/registry/drawer_closed", "3"
            ),
        )
    )
    modern_library.add_capability_contract(
        CapabilityContract(
            uid="resym:Articulation",
            label="articulation.state-transition",
            roles=(
                CapabilityRole(
                    name="patient",
                    accepted_symbol_types=(DRAWER_TYPE,),
                ),
            ),
            success_relation="state(patient) == target",
            verifiable_effects=(PredicateRef("soma:Opened", "1", "opened"),),
        )
    )
    modern = modern_library.to_json()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    (library_dir / "modern.json").write_text(json.dumps(modern), encoding="utf-8")

    body = (
        create_app(tmp_path / "runs", library_dir=library_dir)
        .test_client()
        .get("/library")
        .get_data(as_text=True)
    )

    assert "soma:Closed@2" in body
    assert "resym:truth-procedure/registry/drawer_closed@3" in body
    assert "soma:Opened" in body and ">opened<" in body


def test_capability_catalog_groups_contracts_and_labels_every_field(tmp_path):
    """
    Cards are grouped by activity category and every field says what it is, so a reader
    never has to guess whether a string is a name or an id.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    body = create_app(runs).test_client().get("/capabilities").get_data(as_text=True)

    assert "Moving around" in body  # navigation.* group
    assert "Handling objects" in body  # manipulation.* group
    assert "Working with material" in body  # material.* group
    assert body.index("Moving around") < body.index("Handling objects")
    assert ">name<" in body and ">stable id<" in body
    assert ">what it does<" in body and ">success when<" in body
    assert ">implemented by<" in body and ">used by operators<" in body


def test_capability_catalog_page_lists_operators_bound_to_each_contract(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    library = json.loads(json.dumps(_LIBRARY))
    binding = library["operators"][0]["execution_binding"]
    binding["capability_ref"]["uid"] = "resym:ArticulationStateChange"
    (library_dir / "seed_library.json").write_text(
        json.dumps(library), encoding="utf-8"
    )
    client = create_app(runs, library_dir=library_dir).test_client()
    body = client.get("/capabilities").get_data(as_text=True)

    assert "open-drawer (seed_library)" in body


def test_contract_renders_as_signature_with_linked_role_chips(tmp_path):
    """
    Roles appear as colored chips both in the signature and inside the success relation,
    sharing one data key so hover can link them.
    """
    runs = tmp_path / "runs"
    runs.mkdir()
    library_dir = tmp_path / "library"
    library_dir.mkdir()
    (library_dir / "seed_library.json").write_text(
        json.dumps(_LIBRARY), encoding="utf-8"
    )
    client = create_app(runs, library_dir=library_dir).test_client()
    body = client.get("/library").get_data(as_text=True)

    # signature: the label verb plus one chip per role
    assert "state-transition(" in body
    assert body.count("data-rk='resym:Articulation:patient'") >= 3  # signature,
    # success relation, and role description row share the key
    assert "role-chip rc0" in body
    assert "success when" in body
    assert "view in the capability catalog →" in body
    assert "#cap-resym:Articulation" in body
    assert "How the pieces connect" in body


def test_run_libraries_page_diffs_version_store(tmp_path):
    recorder = RunRecorder.create("repair", root=tmp_path)
    recorder.record_library(_LIBRARY, "start")

    patched_library = _fixture_library(extra_opened_predicate=True)
    patched = patched_library.to_json()
    patched["operators"][0]["preconditions"] = []
    store = recorder.directory / "work" / "versions" / "ep1" / "versions"
    store.mkdir(parents=True)
    (store / "v0001.json").write_text(
        json.dumps({"meta": {"role": "faulted-base"}, "library": _LIBRARY})
    )
    (store / "v0002.json").write_text(
        json.dumps({"meta": {"role": "episode-admission"}, "library": patched})
    )
    body = (
        _client(tmp_path)
        .get(f"/run/{recorder.directory.name}/libraries")
        .get_data(as_text=True)
    )
    assert "library_start.json" in body  # the starting snapshot
    assert "faulted-base" in body and "episode-admission" in body
    assert "diff v0001 → v0002" in body
    assert "+ opened" in body  # added predicate
    assert "~ open-drawer" in body  # changed operator
    # the run page links to the libraries page
    run_body = (
        _client(tmp_path).get(f"/run/{recorder.directory.name}").get_data(as_text=True)
    )
    assert "libraries" in run_body


def test_library_diff_reports_removals(tmp_path):
    from resym.observability.viewer import _render_library_diff

    stripped = {"predicates": [], "operators": [], "capability_contracts": []}
    body = _render_library_diff(_LIBRARY, stripped)
    assert "− closed" in body
    assert "− open-drawer" in body
    assert "− resym:Articulation" in body
    assert (
        _render_library_diff(_LIBRARY, _LIBRARY) == "<p class=muted>no differences</p>"
    )


def test_heavy_transcript_summarizes_and_paginates(tmp_path):
    name = _seed_run(tmp_path)
    # 60 calls of the looping agent: turns of retrieve → propose → submit,
    # so every 3 calls close one conversation → 20 conversations.
    tools = ["retrieve_domain_fragments", "propose_patch", "submit_for_admission"]
    lines = "".join(
        json.dumps(
            {
                "agent_name": "planning-model-agent",
                "model": "m",
                "attempt": 1,
                "prompt": f"PROMPT-{i}",
                "response": json.dumps({"tool": tools[i % 3], "marker": f"R-{i}"}),
                "input_tokens": 10,
                "output_tokens": 2,
            }
        )
        + "\n"
        for i in range(60)
    )
    (tmp_path / name / "B_repair" / "llm_transcript.jsonl").write_text(
        lines, encoding="utf-8"
    )
    client = _client(tmp_path)
    body = client.get(f"/run/{name}").get_data(as_text=True)
    # the overview shows only a stats card with a link, never the exchanges
    assert "browse the 20 repair cases (20 conversations)" in body
    assert "R-59" not in body
    assert "600 in" in body  # token totals surface on the overview
    page = client.get(
        f"/run/{name}/artifact/B_repair/llm_transcript.jsonl?page=4"
    ).get_data(as_text=True)
    # numbered pager marks page 4 of 4 as current
    assert "class='on' href='?page=4'>4</a>" in page
    assert "?page=5" not in page
    # cases 16-20 hold calls 45-59, grouped into 3 turns each
    assert "repair case 16" in page
    assert "R-59" in page and "R-0" not in page
    assert "submitted a fix for review" in page
    assert "turn 3" in page
    filtered = client.get(
        f"/run/{name}/artifact/B_repair/llm_transcript.jsonl"
        "?agent=planning-model-agent"
    ).get_data(as_text=True)
    assert "planning-model-agent (20)" in filtered
    # traversal outside the run directory is refused
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    assert client.get(f"/run/{name}/artifact/..%2Fsecret.txt").status_code == 404


def test_gate_verdict_renders_as_headline_card(tmp_path):
    name = _seed_run(tmp_path)
    (tmp_path / name / "gate_p2.txt").write_text(
        "Gate P2: DOWNGRADE to plan B — neither condition holds\n"
        "  correct repair: mean diff -0.36 [-0.46, -0.26] over 110 pairs\n",
        encoding="utf-8",
    )
    body = _client(tmp_path).get(f"/run/{name}").get_data(as_text=True)
    assert "badge warn'>DOWNGRADE" in body
    assert "neither condition holds" in body
    assert "mean diff -0.36" in body


def test_tagged_transcript_labels_cases_and_backends(tmp_path):
    name = _seed_run(tmp_path)

    def call(backend, agent, episode, scene, response, group="D1"):
        return json.dumps(
            {
                "agent_name": agent,
                "model": "m",
                "attempt": 1,
                "prompt": "P",
                "response": response,
                "backend": backend,
                "template_id": "missing-close-operator",
                "group": group,
                "episode_index": episode,
                "proposal_context_id": f"missing-close-operator/scene-{scene}",
            }
        )

    submit = json.dumps({"tool": "submit_for_admission"})
    proposal = json.dumps({"rationale": "why", "operators": []})
    lines = "\n".join(
        [
            call("agentic-rag", "planning-model-agent", 0, 111, submit),
            call("closed-book", "symbol-proposer", 0, 111, proposal),
            call("rag-one-shot", "symbol-proposer", 0, 111, proposal),
            call("agentic-rag", "planning-model-agent", 1, 222, submit, group="D2"),
            call("fixed-pipeline", "symbol-proposer", 1, 222, proposal, group="D2"),
        ]
    )
    (tmp_path / name / "B_repair" / "llm_transcript.jsonl").write_text(
        lines + "\n", encoding="utf-8"
    )
    page = (
        _client(tmp_path)
        .get(f"/run/{name}/artifact/B_repair/llm_transcript.jsonl")
        .get_data(as_text=True)
    )
    # tag-based case split: episode 0 and episode 1 become two boxes
    assert "repair case 1" in page and "repair case 2" in page
    assert "fault: missing-close-operator" in page
    assert "group: D1" in page and "missing model elements" in page
    assert "scene-111" in page and "scene-222" in page
    # recorded backends replace the one-of-three fallback label
    assert "backend: closed-book" in page
    assert "backend: rag-one-shot" in page
    assert "backend: fixed-pipeline" in page
    assert "backend: agentic-rag" in page
    assert "closed-book / rag-one-shot / fixed-pipeline" not in page
    # the group facet narrows to matching cases only
    filtered = (
        _client(tmp_path)
        .get(f"/run/{name}/artifact/B_repair/llm_transcript.jsonl?group=D2")
        .get_data(as_text=True)
    )
    assert "repair case 2" in filtered and "repair case 1" not in filtered
    assert "scene-222" in filtered and "scene-111" not in filtered


def test_transcript_recorder_stamps_scoped_context(tmp_path):
    from resym.llm.transcript import (
        LanguageModelExchange,
        TranscriptRecorder,
    )

    path = tmp_path / "llm_transcript.jsonl"
    recorder = TranscriptRecorder(path=path)

    def exchange():
        return LanguageModelExchange(
            agent_name="symbol-proposer",
            model_description="m",
            prompt="P",
            response="R",
            attempt=1,
        )

    with recorder.scoped(
        backend="closed-book",
        template_id="missing-close-operator",
        group="D1",
        episode_index=0,
        proposal_context_id="missing-close-operator/scene-111",
    ):
        recorder.record(exchange())
    recorder.record(exchange())  # outside the scope: no stamps

    first, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert first["backend"] == "closed-book"
    assert first["group"] == "D1"
    assert first["proposal_context_id"] == "missing-close-operator/scene-111"
    assert "backend" not in second


def test_chinese_language_toggle_and_cookie(tmp_path):
    name = _seed_run(tmp_path)
    client = _client(tmp_path)
    # English is the default and stays untouched
    body = client.get("/").get_data(as_text=True)
    assert "reSym run viewer" in body and "运行记录" not in body
    # ?lang=zh translates the page and offers the toggle
    zh = client.get("/?lang=zh").get_data(as_text=True)
    assert "reSym 运行查看器" in zh
    assert "实时执行" in zh  # navcard
    assert "lang=en" in zh and "lang=zh" in zh  # toggle links
    # the choice sticks via cookie: subpages translate without ?lang
    run_page = client.get(f"/run/{name}").get_data(as_text=True)
    assert "阶段 A · 世界" in run_page
    assert "执行轨迹" in run_page  # trace.jsonl explainer
    live = client.get(f"/run/{name}/live").get_data(as_text=True)
    assert "接地后的计划" in live and "当前动作" in live
    # switching back to English works the same way
    client.get("/?lang=en")
    back = client.get(f"/run/{name}").get_data(as_text=True)
    assert "Stage A · world" in back and "阶段 A" not in back


def test_long_prompt_folds_into_sections_with_change_badges(tmp_path):
    name = _seed_run(tmp_path)
    boiler = "x" * 40
    prompt_1 = (
        "You are an agent.\n"
        f"## Failure certificate\ncert {boiler}\n"
        f"## Episode so far\n(no steps yet)\n"
    )
    prompt_2 = (
        "You are an agent.\n"
        f"## Failure certificate\ncert {boiler}\n"
        f"## Episode so far\nturn 1: retrieved fragments\n"
    )
    submit = json.dumps({"tool": "submit_for_admission"})
    lines = "\n".join(
        [
            json.dumps(
                {
                    "agent_name": "planning-model-agent",
                    "model": "m",
                    "attempt": 1,
                    "prompt": prompt_1,
                    "response": json.dumps({"tool": "retrieve_domain_fragments"}),
                }
            ),
            json.dumps(
                {
                    "agent_name": "planning-model-agent",
                    "model": "m",
                    "attempt": 1,
                    "prompt": prompt_2,
                    "response": submit,
                }
            ),
            # a format retry with the identical prompt collapses to one line
            json.dumps(
                {
                    "agent_name": "planning-model-agent",
                    "model": "m",
                    "attempt": 1,
                    "prompt": prompt_2,
                    "response": "garbage",
                }
            ),
        ]
    )
    (tmp_path / name / "B_repair" / "llm_transcript.jsonl").write_text(
        lines + "\n", encoding="utf-8"
    )
    page = (
        _client(tmp_path)
        .get(f"/run/{name}/artifact/B_repair/llm_transcript.jsonl")
        .get_data(as_text=True)
    )
    # sections render as their own folds with char counts
    assert "class=psec" in page
    assert "Failure certificate" in page and "Episode so far" in page
    assert "role &amp; instructions" in page
    # turn 2 marks the unchanged and changed sections, changed opens
    assert "same as last turn" in page
    assert "changed since last turn" in page
    assert "2 changed since the previous turn" not in page  # only 1 changed
    assert "1 changed since the previous turn" in page
    assert "full raw prompt" in page
