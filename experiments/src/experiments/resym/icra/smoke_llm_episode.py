"""
One real-language-model repair episode, end to end, on the Tracy scene.

The pre-flight for the E1/E2 formal runs: everything the grid will use is
exercised once with the real model — the frozen corpus index with the
dense channel at its frozen weight, the repair backend proposing through
llm-agent-kit, the curator's admission suite on admission-split scenes,
the held-out battery, the version trail, and the episode record. The
reference baselines (oracle, no-repair) run alongside for free, so the
printed report already has the shape of the E1 main table.

Run inside the container (the image must have llm-agent-kit and
sentence-transformers; credentials come from the project ``.env`` — note
that llm-agent-kit's ``load_dotenv()`` cannot find a mounted ``.env``
from ``/opt/ros/cram-env``, so it must be injected as environment
variables)::

    docker run --rm \
      -v "$(dirname "$(pwd)"):/work" -w /work/resym \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e PYTHONPATH=/work/resym/src \
      -e QT_QPA_PLATFORM=offscreen \
      --env-file .env \
      cram:jazzy-resym \
      bash -c "source /opt/ros/cram-env/bin/activate \
               && uv run --active --no-sync python \
                    -m experiments.resym.icra.smoke_llm_episode"

Model and retry policy come from ``RESYM_LLM_CONFIG`` (default
``config/llm.json``). ``--backends`` selects which methods run,
comma-separated among: agentic-rag, closed-book, rag-one-shot,
fixed-pipeline, typed-enumeration. ``repair-agent`` remains an input alias
for older commands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.resym.scenes import Scene, load_fixed_arm_scene
from experiments.resym.seed_library import build_fixed_arm_library
from experiments.resym.icra.articulation.faults import (
    ADMISSION,
    HELD_OUT,
    PROPOSAL,
    drawer_fault_templates,
    generate_splits,
)
from experiments.resym.icra.articulation.tracy_bench import build_tracy_bench
from experiments.resym.icra.harness import run_e1
from resym import PROJECT_ROOT
from resym.knowledge.corpus_installation import CORPUS_RELEASE_DIRECTORY
from resym.knowledge.freeze import load_release
from resym.knowledge.retrieval import FragmentIndex, RetrievalConfig
from resym.llm.configuration import (
    ExperimentConfiguration,
    build_completion_client,
)
from resym.llm.structured import StructuredCompleter
from resym.llm.transcript import TranscriptRecorder
from resym.observability.runlog import RunRecorder
from resym.platform.cram_objects import task_object_universe
from resym.repair.agent import RetrievalAugmentedPlanningAgent
from resym.repair.backends import (
    AGENTIC_RAG_BACKEND_NAME,
    ClosedBookBackend,
    EnumerationBackend,
    FixedPipelineBackend,
    RagOneShotBackend,
)


def load_llm_configuration() -> ExperimentConfiguration:
    path = Path(
        os.environ.get("RESYM_LLM_CONFIG", PROJECT_ROOT / "config" / "llm.json")
    )
    configuration = ExperimentConfiguration.load(path)
    if not configuration.agent.get("model"):
        raise SystemExit(
            f"{path} configures no model; a smoke test of the real model "
            "cannot fall back to the scripted client."
        )
    return configuration


def build_frozen_index() -> FragmentIndex:
    """
    The corpus index exactly as the formal runs will use it: lexical + structural
    rerank, dense channel at the frozen weight when the release ships the pre-encoded
    index and sentence-transformers is importable.
    """
    fragments = load_release(CORPUS_RELEASE_DIRECTORY)
    config = RetrievalConfig()
    from resym.knowledge.dense import (
        FROZEN_CONFIG_FILENAME,
        INDEX_FILENAME,
        DenseIndex,
        mpnet_encoder,
    )

    index_path = CORPUS_RELEASE_DIRECTORY / INDEX_FILENAME
    frozen_path = CORPUS_RELEASE_DIRECTORY / FROZEN_CONFIG_FILENAME
    if index_path.exists() and frozen_path.exists():
        try:
            encoder = mpnet_encoder()
        except ImportError as error:
            print(f"[smoke] dense channel OFF (sentence-transformers missing: {error})")
        else:
            dense = DenseIndex.load(index_path)
            frozen = json.loads(frozen_path.read_text())
            if frozen["index_checksum"] != dense.checksum():
                raise SystemExit("release dense index does not match its frozen config")
            config.dense_weight = frozen["dense_weight"]
            config.dense_scorer_by_id = dense.scorer(encoder)
            print(
                f"[smoke] dense channel ON: {frozen['model']}, "
                f"weight {frozen['dense_weight']}, "
                f"{len(dense.fragment_ids)} fragments"
            )
    else:
        print(
            f"[smoke] dense channel OFF (no pre-encoded index in {CORPUS_RELEASE_DIRECTORY})"
        )
    return FragmentIndex(fragments, config)


def real_backends(names: list[str], completer: StructuredCompleter) -> dict:
    available = {
        AGENTIC_RAG_BACKEND_NAME: lambda: RetrievalAugmentedPlanningAgent(
            completer=completer
        ),
        "closed-book": lambda: ClosedBookBackend(completer=completer),
        "rag-one-shot": lambda: RagOneShotBackend(completer=completer),
        "fixed-pipeline": lambda: FixedPipelineBackend(completer=completer),
        "typed-enumeration": lambda: EnumerationBackend(),
    }
    unknown = [name for name in names if name not in available]
    if unknown:
        raise SystemExit(f"unknown backends {unknown}; choose from {sorted(available)}")
    if len(names) != len(set(names)):
        raise SystemExit("the same backend was selected more than once")
    return {name: available[name]() for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        default=AGENTIC_RAG_BACKEND_NAME,
        help="comma-separated methods to run (default: agentic-rag)",
    )
    parser.add_argument(
        "--template",
        default="missing-close-operator",
        help="fault template identifier (default: missing-close-operator)",
    )
    parser.add_argument("--seed", type=int, default=1, help="split-generation seed")
    arguments = parser.parse_args()

    configuration = load_llm_configuration()
    recorder = RunRecorder.create(
        "llm-smoke",
        root=PROJECT_ROOT / "runs",
        metadata={"llm_configuration": configuration.to_metadata()},
    )
    print(f"[smoke] run directory: {recorder.directory}")

    print("[smoke] loading Tracy scene ...")
    setup = load_fixed_arm_scene(Scene.APARTMENT)
    universe = task_object_universe(setup.world, setup.robot)
    bench = build_tracy_bench(setup, universe, index=build_frozen_index())
    completer = StructuredCompleter(
        client=build_completion_client(configuration),
        transcript=TranscriptRecorder(path=recorder.transcript_path),
        maximum_attempts=configuration.structured_maximum_attempts,
    )
    print(f"[smoke] model: {completer.client.description}")

    templates = drawer_fault_templates(build_fixed_arm_library())
    template = next((t for t in templates if t.identifier == arguments.template), None)
    if template is None:
        raise SystemExit(
            f"unknown template '{arguments.template}'; available: "
            + ", ".join(t.identifier for t in templates)
        )

    # One scene variation per split bounds the runtime; the formal run
    # sweeps the full grid instead.
    generated = generate_splits((template,), seed=arguments.seed)
    keep = {PROPOSAL: 1, ADMISSION: 1, HELD_OUT: 1}
    cases = []
    for case in generated:
        if keep.get(case.split, 0) > 0:
            cases.append(case)
            keep[case.split] -= 1

    backends = real_backends(
        [name.strip() for name in arguments.backends.split(",") if name.strip()],
        completer,
    )
    print(f"[smoke] backends: {', '.join(backends)} (+ reference baselines)")

    report = run_e1(
        bench,
        backends=backends,
        templates=[template],
        cases=tuple(cases),
        working_root=recorder.directory / "work",
        recorder=recorder,
    )

    print("\n" + report.render())
    print("\n[smoke] per-episode outcomes:")
    for episode in report.episodes:
        held_out = (
            "held-out clean"
            if episode.admitted and episode.held_out_passed
            else (
                f"held-out failures: {episode.held_out_failures}"
                if episode.admitted
                else "not admitted"
            )
        )
        print(
            f"  {episode.backend:20s} status={episode.status:22s} "
            f"admitted={str(episode.admitted):5s} {held_out} "
            f"budget={episode.budget}"
        )
    print(
        f"\n[smoke] episode records: {recorder.directory / 'D_experiment' / 'episodes.jsonl'}"
    )
    print(f"[smoke] LLM transcript:  {recorder.transcript_path}")


if __name__ == "__main__":
    main()
