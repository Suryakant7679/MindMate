#!/usr/bin/env python3
"""Run missing Llama steering stages (13--19) in dependency order.

Existing complete stages are skipped by default.  Use ``--force`` only when
you deliberately want to recompute and overwrite a stage's output files.
The script writes a before/after audit to ``results/llama31_8b_steering``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llama31_8b_steering"


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    command: tuple[str, ...]
    required_outputs: tuple[str, ...]

    @property
    def directory(self) -> Path:
        return RESULTS / self.name


STAGES = (
    Stage(
        13,
        "13_single_joy_neuron_selection",
        ("13_select_single_joy_steering_neuron.py", "--batch-size", "4"),
        (
            "joy_candidate_activation_statistics.csv",
            "best_joy_neuron.json",
            "best_joy_neuron_residual_direction.npy",
            "best_joy_neuron_residual_direction_unit.npy",
        ),
    ),
    Stage(
        14,
        "14_single_joy_neuron_steering",
        ("14_single_joy_neuron_steering.py",),
        ("single_joy_neuron_alpha_sweep.csv", "single_joy_neuron_alpha_sweep.json"),
    ),
    Stage(
        15,
        "15_direction_and_logit_analysis",
        ("15_direction_and_logit_analysis.py", "--batch-size", "4"),
        (
            "direction_alignment.json",
            "joy_minus_sad_hidden_direction.npy",
            "token_level_logit_analysis.csv",
            "alpha_logit_summary.csv",
            "baseline_continuation.txt",
            "full_analysis.json",
        ),
    ),
    Stage(
        16,
        "16_steering_direction_visualization",
        ("16_visualize_steering_direction.py",),
        ("steering_direction_summary.csv", "steering_direction_summary.json"),
    ),
    Stage(
        17,
        "17_joy_actuator_ranking",
        ("17_rank_all_joy_actuator_neurons.py", "--batch-size", "4"),
        (
            "joy_actuator_ranking.csv",
            "joy_actuator_ranking.json",
            "best_joy_actuator_neuron.json",
            "best_joy_actuator_write_direction.npy",
            "best_joy_actuator_write_direction_unit.npy",
            "layer_direction_summary.csv",
            "joy_actuator_ranking.png",
        ),
    ),
    Stage(
        18,
        "18_all_top1pct_joy_actuators",
        ("18_rank_all_top1pct_joy_actuators.py", "--batch-size", "4"),
        (
            "all_top1pct_joy_actuator_ranking.csv",
            "all_top1pct_joy_actuator_ranking.json",
            "top_joy_actuators.csv",
            "best_joy_actuator_neuron.json",
            "best_joy_actuator_write_direction.npy",
            "best_joy_actuator_write_direction_unit.npy",
            "layer_direction_summary.csv",
            "joy_actuator_ranking.png",
        ),
    ),
    Stage(
        19,
        "19_top_joy_actuator_tests",
        ("19_test_top_joy_actuators.py",),
        ("actuator_test_summary.csv", "actuator_test_summary.json", "baseline.json"),
    ),
)


def missing_outputs(stage: Stage) -> list[str]:
    return [name for name in stage.required_outputs if not (stage.directory / name).is_file()]


def audit() -> list[dict[str, object]]:
    return [
        {
            "stage": stage.number,
            "folder": stage.name,
            "status": "complete" if not missing_outputs(stage) else "incomplete",
            "missing_outputs": missing_outputs(stage),
            "file_count": sum(1 for path in stage.directory.rglob("*") if path.is_file())
            if stage.directory.is_dir()
            else 0,
        }
        for stage in STAGES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Audit and print commands only.")
    parser.add_argument("--force", action="store_true", help="Recompute stages even if complete.")
    parser.add_argument(
        "--stages", type=int, nargs="+", choices=range(13, 20),
        help="Optional subset of stages. Dependencies must already be complete.",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    before = audit()
    selected = [stage for stage in STAGES if args.stages is None or stage.number in args.stages]
    print("Llama steering output audit:")
    for item in before:
        missing = ", ".join(item["missing_outputs"]) or "none"
        print(f"  {item['stage']:02d} {item['status']:10s} files={item['file_count']:3d} missing={missing}")

    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    for stage in selected:
        missing = missing_outputs(stage)
        if not args.force and not missing:
            print(f"Skipping stage {stage.number:02d}: outputs are complete.")
            continue
        command = [sys.executable, *stage.command]
        print(f"{'Would run' if args.dry_run else 'Running'} stage {stage.number:02d}: {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            still_missing = missing_outputs(stage)
            if still_missing:
                raise RuntimeError(
                    f"Stage {stage.number:02d} finished but did not create: {', '.join(still_missing)}"
                )

    report = {"before": before, "after": audit(), "dry_run": args.dry_run, "forced": args.force}
    report_path = RESULTS / "steering_results_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote audit: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
