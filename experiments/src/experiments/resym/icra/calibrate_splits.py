"""
Create the frozen, platform-feasible split manifest for formal E1/E2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.resym.scenes import Scene, load_fixed_arm_scene
from experiments.resym.seed_library import build_fixed_arm_library
from experiments.resym.icra.articulation.faults import (
    SplitCalibrationError,
    calibrate_splits,
)
from experiments.resym.icra.articulation.tracy_bench import TracyBench
from experiments.resym.icra.run import PROJECT_ROOT, select_templates
from resym.platform.cram_objects import task_object_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "config" / "tracy_frozen_splits.json",
    )
    parser.add_argument("--templates", default="all")
    parser.add_argument("--variations-per-template", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-candidates-per-template", type=int, default=200)
    args = parser.parse_args()

    templates = select_templates(args.templates)
    setup = load_fixed_arm_scene(Scene.APARTMENT)
    universe = task_object_universe(setup.world, setup.robot)
    bench = TracyBench(setup, universe)
    root = args.output.parent / ".split-calibration-work"
    counter = {"value": 0}

    def feasible(template, variation):
        counter["value"] += 1
        print(
            f"[calibrate] {template.identifier} seed={variation.seed} "
            f"candidate={counter['value']}",
            flush=True,
        )
        failures = bench.calibration_failures(
            template,
            variation,
            root / template.identifier / str(variation.seed),
        )
        if failures:
            print(f"[calibrate] rejected: {'; '.join(failures)}", flush=True)
        else:
            print("[calibrate] accepted", flush=True)
        return failures

    try:
        manifest = calibrate_splits(
            templates,
            feasible,
            build_fixed_arm_library(),
            variations_per_template=args.variations_per_template,
            seed=args.seed,
            maximum_candidates_per_template=args.maximum_candidates_per_template,
        )
    except SplitCalibrationError as error:
        failure_path = args.output.with_suffix(".failed.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "template_id": error.template_id,
                    "accepted_count": error.accepted_count,
                    "requested_count": error.requested_count,
                    "attempted_count": error.attempted_count,
                    "accepted_candidates": list(error.accepted_candidates),
                    "exclusions": list(error.exclusions),
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(f"{error}; details written to {failure_path}") from error
    manifest.save(args.output)
    print(
        f"[calibrate] wrote {len(manifest.cases)} accepted cases and "
        f"{len(manifest.exclusions)} exclusions to {args.output}"
    )


if __name__ == "__main__":
    main()
