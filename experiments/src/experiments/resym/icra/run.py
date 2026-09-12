"""
The E1/E2 formal batch entry point on the Tracy drawer scene.

Same machinery as ``experiments.resym.icra.smoke_llm_episode`` (frozen corpus index
with the dense channel, real backends through llm-agent-kit, curator
admission on admission-split scenes, held-out battery, version trails),
scaled to the pre-registered grid and made shard-friendly: every episode
is appended to ``D_experiment/episodes.jsonl`` as it finishes, and
``--report-from`` rebuilds the combined report (rates, Gate P2) from any
set of run directories — so the 15-template grid can be split across
container invocations, and a crashed run loses nothing that was already
paid for.

Full grid (many hours, real API spend)::

    uv run --active --no-sync python -m experiments.resym.icra.run

Sharded, e.g. four templates per invocation::

    uv run --active --no-sync python -m experiments.resym.icra.run \
      --templates missing-close-operator,missing-open-operator,...

Assemble the pre-registered outputs from finished shards (host or
container, no scene load, no LLM)::

    uv run --active --no-sync python -m experiments.resym.icra.run \
      --report-from runs/<a> runs/<b> ...

Container invocation matches the smoke script's docstring (mount project
+ HF cache, ``--env-file .env``). Model/retry policy: ``RESYM_LLM_CONFIG``
(default ``config/llm.json``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.resym.scenes import Scene, load_fixed_arm_scene
from experiments.resym.seed_library import build_fixed_arm_library
from experiments.resym.icra.harness import (
    POLICY_TIERS,
    E1Report,
    GateP2Margins,
    agentic_gate,
    e1_report_from_records,
    e2_report_from_records,
    run_e1,
    run_e2,
)
from experiments.resym.icra.provenance import (
    experiment_provenance,
    repository_snapshot,
    write_json,
)
from experiments.resym.icra.articulation.faults import (
    FrozenSplitManifest,
    drawer_fault_templates,
    generate_splits,
    library_checksum,
)
from resym import PROJECT_ROOT
from resym.observability.runlog import RunRecorder
from resym.platform.capabilities import capability_contracts
from resym.platform.cram_objects import task_object_universe
from resym.platform.grounding_catalog import freeze_grounding_factories
from resym.platform.kinematic import KINEMATIC_FEASIBILITY
from resym.repair.backends import AGENTIC_RAG_BACKEND_NAME
from experiments.resym.grounding_initialization import default_drawer_grounding_catalog

DEFAULT_BACKENDS = (
    AGENTIC_RAG_BACKEND_NAME,
    "closed-book",
    "rag-one-shot",
    "fixed-pipeline",
    "typed-enumeration",
)


def select_templates(names: str):
    grounding_catalog = default_drawer_grounding_catalog()
    templates = drawer_fault_templates(
        build_fixed_arm_library(grounding_catalog), grounding_catalog
    )
    if names == "all":
        return templates
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_id = {t.identifier: t for t in templates}
    unknown = [name for name in wanted if name not in by_id]
    if unknown:
        raise SystemExit(f"unknown templates {unknown}; available: {sorted(by_id)}")
    return tuple(by_id[name] for name in wanted)


def write_e1_outputs(report: E1Report, directory: Path) -> None:
    """
    The pre-registered E1 artifacts: rendered report, per-backend rate summary, and the
    Gate P2 decision when both gate backends ran.
    """
    (directory / "e1_report.txt").write_text(report.render() + "\n")
    summary = {}
    for backend in report.backends():
        rates = {}
        for name, rate in (
            ("correct_repair", report.correct_repair_rate(backend)),
            ("false_admission", report.false_admission_rate(backend)),
            ("unsupported_detection", report.unsupported_detection_rate(backend)),
        ):
            rates[name] = {
                "successes": rate.successes,
                "total": rate.total,
                "rate": rate.rate,
                "wilson_low": rate.low,
                "wilson_high": rate.high,
            }
        summary[backend] = rates
    (directory / "e1_summary.json").write_text(json.dumps(summary, indent=1))
    protocol = json.loads(
        (PROJECT_ROOT / "config" / "experiment_protocol.json").read_text()
    )
    gate_config = protocol["gate_p2"]
    margins = GateP2Margins(
        repair_noninferiority=gate_config["repair_noninferiority_absolute_difference"],
        false_admission_increase=gate_config[
            "false_admission_maximum_absolute_increase"
        ],
    )
    try:
        gate = agentic_gate(
            report,
            iterations=gate_config["bootstrap_iterations"],
            seed=gate_config["bootstrap_seed"],
            margins=margins,
        )
    except ValueError as error:
        (directory / "gate_p2.txt").write_text(
            f"Gate P2 not decidable from this record: {error}\n"
        )
    else:
        (directory / "gate_p2.txt").write_text(gate.render() + "\n")
    print((directory / "e1_report.txt").read_text())
    print((directory / "gate_p2.txt").read_text())


def assemble(run_directories: list[Path], output_directory: Path) -> None:
    records = []
    e2_records = []
    for directory in run_directories:
        path = directory / "D_experiment" / "episodes.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                # E1 episode records carry a backend; E2 decision records
                # (written to e2_decisions.jsonl) never enter these files,
                # but stay defensive about mixed content.
                if "backend" in record:
                    records.append(record)
        e2_path = directory / "e2_decisions.jsonl"
        if e2_path.exists():
            e2_records.extend(
                json.loads(line) for line in e2_path.read_text().splitlines()
            )
    if not records and not e2_records:
        raise SystemExit("no E1/E2 records found in the given runs")
    output_directory.mkdir(parents=True, exist_ok=True)
    if records:
        print(f"[e1e2] assembled {len(records)} E1 episodes -> {output_directory}")
        write_e1_outputs(e1_report_from_records(records), output_directory)
    if e2_records:
        print(f"[e1e2] assembled {len(e2_records)} E2 decisions")
        e2_report = e2_report_from_records(e2_records)
        (output_directory / "e2_report.txt").write_text(e2_report.render() + "\n")
        print(e2_report.render())


def parse_arguments() -> argparse.Namespace:
    """
    Parse the batch and report-assembly command-line options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-from",
        nargs="+",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="assemble the combined E1 report + Gate P2 from finished run "
        "directories instead of running episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output directory for --report-from (default: runs/<ts>-e1e2-report)",
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"comma-separated backends (default: {','.join(DEFAULT_BACKENDS)})",
    )
    parser.add_argument(
        "--templates",
        default="all",
        help="comma-separated template identifiers, or 'all' (default)",
    )
    parser.add_argument(
        "--episodes-per-template",
        type=int,
        default=10,
        help="repair episodes per (template, backend) (default: 10)",
    )
    parser.add_argument(
        "--variations-per-template",
        type=int,
        default=10,
        help="scene variations per template across the three splits (default: 10)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="split-generation seed; keep identical across shards (default: 0)",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=PROJECT_ROOT / "config" / "tracy_frozen_splits.json",
        help="platform-calibrated frozen split manifest",
    )
    parser.add_argument(
        "--unsafe-random-splits",
        action="store_true",
        help="development only: bypass platform calibration (formal results "
        "must never use this flag)",
    )
    parser.add_argument("--skip-e1", action="store_true")
    parser.add_argument("--skip-e2", action="store_true")
    return parser.parse_args()


def load_cases(
    templates,
    split_manifest: Path,
    unsafe_random_splits: bool,
    variations_per_template: int,
    split_seed: int,
):
    """
    Load validated frozen cases, or development-only random cases.
    """
    if unsafe_random_splits:
        print("[e1e2] WARNING: using uncalibrated development splits")
        return generate_splits(
            templates,
            variations_per_template=variations_per_template,
            seed=split_seed,
        )

    if not split_manifest.exists():
        raise SystemExit(
            f"frozen split manifest not found: {split_manifest}; "
            "run `./scripts/run_in_docker.sh calibrate` first"
        )
    manifest = FrozenSplitManifest.load(split_manifest)
    expected_checksum = library_checksum(
        build_fixed_arm_library(default_drawer_grounding_catalog())
    )
    if manifest.correct_library_checksum != expected_checksum:
        raise SystemExit(
            "frozen split manifest was calibrated against a different "
            "correct library; recalibrate before running"
        )
    wanted = {template.identifier for template in templates}
    problems = manifest.validation_problems(wanted)
    if problems:
        raise SystemExit(
            "frozen split manifest is incomplete or inconsistent:\n- "
            + "\n- ".join(problems)
        )
    print(f"[e1e2] frozen splits: {split_manifest}")
    return tuple(case for case in manifest.cases if case.template_id in wanted)


def main() -> None:
    arguments = parse_arguments()

    if arguments.report_from is not None:
        recorder = None
        output = arguments.output
        if output is None:
            recorder = RunRecorder.create("e1e2-report", root=PROJECT_ROOT / "runs")
            output = recorder.directory
        assemble(list(arguments.report_from), output)
        return

    # Imported lazily: the assemble path must work without llm-agent-kit
    # or a loadable scene.
    from experiments.resym.icra.smoke_llm_episode import (
        build_frozen_index,
        load_llm_configuration,
        real_backends,
    )
    from resym.llm.configuration import build_completion_client
    from resym.llm.structured import StructuredCompleter
    from resym.llm.transcript import TranscriptRecorder

    templates = select_templates(arguments.templates)
    cases = load_cases(
        templates,
        split_manifest=arguments.split_manifest,
        unsafe_random_splits=arguments.unsafe_random_splits,
        variations_per_template=arguments.variations_per_template,
        split_seed=arguments.split_seed,
    )
    backend_names = [
        name.strip() for name in arguments.backends.split(",") if name.strip()
    ]
    needs_llm = any(name != "typed-enumeration" for name in backend_names)
    configuration = None
    if needs_llm and not arguments.skip_e1:
        configuration = load_llm_configuration()
    repository, source_patch = repository_snapshot(PROJECT_ROOT)
    case_counts: dict[str, int] = {}
    for case in cases:
        case_counts[case.split] = case_counts.get(case.split, 0) + 1
    correct_library_checksum = library_checksum(
        build_fixed_arm_library(default_drawer_grounding_catalog())
    )
    metadata = {
        "experiment_protocol": json.loads(
            (PROJECT_ROOT / "config" / "experiment_protocol.json").read_text()
        ),
        "provenance": experiment_provenance(
            PROJECT_ROOT,
            repository=repository,
            arguments=vars(arguments),
            templates=(template.identifier for template in templates),
            case_counts=case_counts,
            split_manifest=arguments.split_manifest,
            correct_library_checksum=correct_library_checksum,
            policy_tiers=(policy.name for policy in POLICY_TIERS),
            llm_metadata=(
                configuration.to_metadata() if configuration is not None else None
            ),
        ),
    }
    if configuration is not None:
        metadata["llm_configuration"] = configuration.to_metadata()
    recorder = RunRecorder.create("e1e2", root=PROJECT_ROOT / "runs", metadata=metadata)
    write_json(
        recorder.stage_dir("D_experiment") / "provenance.json",
        metadata["provenance"],
    )
    recorder.artifact("D_experiment", "source.patch", source_patch)
    grounding_catalog = default_drawer_grounding_catalog()
    frozen_library = recorder.stage_dir("D_experiment") / "symbol_library.json"
    build_fixed_arm_library(grounding_catalog).save(frozen_library)
    if grounding_catalog.workspace is None:
        raise RuntimeError("formal experiment requires a grounding review workspace")
    freeze_grounding_factories(
        workspace=grounding_catalog.workspace,
        output_directory=recorder.directory / "grounding_snapshot",
        symbol_library=frozen_library,
        capability_contracts=capability_contracts(),
        capability_feasibility_implementations=KINEMATIC_FEASIBILITY,
        git_commits=(("cram", str(repository["commit"])),),
        container_image_digest=os.environ.get("RESYM_CONTAINER_IMAGE_ID"),
        experiment_configuration=tuple(
            sorted((name, str(value)) for name, value in vars(arguments).items())
        ),
    )
    print(f"[e1e2] run directory: {recorder.directory}")
    print(
        f"[e1e2] templates ({len(templates)}): "
        + ", ".join(t.identifier for t in templates)
    )

    print("[e1e2] loading Tracy scene ...")
    setup = load_fixed_arm_scene(Scene.APARTMENT)
    universe = task_object_universe(setup.world, setup.robot)
    from experiments.resym.icra.articulation.tracy_bench import build_tracy_bench

    # E2 evaluates a fixed labelled candidate pool and never retrieves from
    # UniDomain. Keep the frozen corpus as an E1-only dependency so an E2-only
    # run does not download an embedding model or require corpus_release/r1.
    index = None if arguments.skip_e1 else build_frozen_index()
    bench = build_tracy_bench(setup, universe, grounding_catalog, index=index)
    completer = None
    if configuration is not None:
        completer = StructuredCompleter(
            client=build_completion_client(configuration),
            transcript=TranscriptRecorder(path=recorder.transcript_path),
            maximum_attempts=configuration.structured_maximum_attempts,
        )
        print(f"[e1e2] model: {completer.client.description}")

    if not arguments.skip_e1:
        backends = real_backends(backend_names, completer)
        print(
            f"[e1e2] E1: {arguments.episodes_per_template} episodes × "
            f"{len(templates)} templates × {len(backends)} backends "
            "(+ reference baselines)"
        )
        report = run_e1(
            bench,
            backends=backends,
            templates=list(templates),
            cases=cases,
            working_root=recorder.directory / "work",
            episodes_per_template=arguments.episodes_per_template,
            recorder=recorder,
            transcript=completer.transcript if completer is not None else None,
        )
        write_e1_outputs(report, recorder.directory)

    if not arguments.skip_e2:
        print(f"[e1e2] E2: {len(POLICY_TIERS)} policy tiers × labelled pools")
        e2_report = run_e2(
            bench,
            templates=list(templates),
            cases=cases,
            working_root=recorder.directory / "e2",
        )
        (recorder.directory / "e2_report.txt").write_text(e2_report.render() + "\n")
        with (recorder.directory / "e2_decisions.jsonl").open("w") as sink:
            for decision in e2_report.decisions:
                sink.write(json.dumps(decision.to_json()) + "\n")
        print(e2_report.render())

    recorder.finish(
        {
            "status": "succeeded",
            "e1_ran": not arguments.skip_e1,
            "e2_ran": not arguments.skip_e2,
        }
    )
    print(f"\n[e1e2] artifacts in {recorder.directory}")


if __name__ == "__main__":
    main()
