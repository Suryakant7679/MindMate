#!/usr/bin/env python3
"""Create reproducible Llama selected-vs-random masking comparison tables.

This is a reporting-only script: it reads the completed Llama random-masking
control outputs and writes two tables without rerunning inference.

The selected-neuron intervention is deterministic and was run with seed 42.
For each (percentage, association degree) condition, the random control uses
ten layer-count-matched masks.  Their seeds are stored in the seed-level table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "results/llama31_8b_neuron_analysis/07_random_matched_control"
)


def metric(value: float) -> str:
    """Format proportions as percentages for the Markdown report."""
    return f"{100 * value:.2f}%"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a Markdown table without the optional ``tabulate`` package."""
    columns = [str(column) for column in frame.columns]

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    rows = [[cell(value) for value in row] for row in frame.itertuples(index=False)]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selected-seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = args.input_dir
    aggregate = pd.read_csv(input_dir / "random_masking_aggregate.csv")
    random_runs = pd.read_csv(input_dir / "random_masking_runs.csv")
    results = json.loads((input_dir / "random_masking_results.json").read_text())
    baseline = results["baseline"]

    keys = ["percentage", "masked_degree", "num_masked_neurons"]
    grouped = random_runs.groupby(keys, as_index=False).agg(
        random_seed_values=("random_seed", lambda x: ", ".join(map(str, x))),
        random_accuracy_mean=("random_accuracy", "mean"),
        random_accuracy_std=("random_accuracy", "std"),
        random_macro_f1_mean=("random_macro_f1", "mean"),
        random_macro_f1_std=("random_macro_f1", "std"),
        random_macro_f1_min=("random_macro_f1", "min"),
        random_macro_f1_max=("random_macro_f1", "max"),
    )
    comparison = aggregate.merge(grouped, on=keys, validate="one_to_one")

    # The aggregate stores selected deltas measured against this exact baseline.
    comparison.insert(3, "selected_seed", args.selected_seed)
    comparison.insert(4, "baseline_accuracy", baseline["accuracy"])
    comparison.insert(5, "baseline_macro_f1", baseline["macro_f1"])
    comparison["selected_accuracy"] = (
        comparison["baseline_accuracy"] + comparison["selected_delta_accuracy"]
    )
    comparison["selected_macro_f1"] = (
        comparison["baseline_macro_f1"] + comparison["selected_delta_macro_f1"]
    )
    comparison["random_seed_count"] = comparison["num_random_runs"]

    ordered = [
        "percentage", "masked_degree", "num_masked_neurons", "selected_seed",
        "baseline_accuracy", "baseline_macro_f1",
        "selected_accuracy", "selected_delta_accuracy",
        "selected_macro_f1", "selected_delta_macro_f1",
        "random_seed_count", "random_seed_values",
        "random_accuracy_mean", "random_accuracy_std",
        "random_macro_f1_mean", "random_macro_f1_std",
        "random_macro_f1_min", "random_macro_f1_max",
        "selected_minus_random_mean_f1", "f1_z_score",
        "empirical_p_lower_tail",
    ]
    comparison = comparison[ordered].sort_values(["percentage", "masked_degree"])

    summary_path = input_dir / "llama_masking_comparison_summary.csv"
    comparison.to_csv(summary_path, index=False, float_format="%.8f")

    seed_table = random_runs.merge(
        comparison[
            keys + ["selected_seed", "selected_accuracy", "selected_macro_f1"]
        ],
        on=keys,
        validate="many_to_one",
    )
    seed_table = seed_table.rename(
        columns={
            "random_accuracy": "random_seed_accuracy",
            "random_macro_f1": "random_seed_macro_f1",
            "delta_accuracy": "random_seed_delta_accuracy",
            "delta_macro_f1": "random_seed_delta_macro_f1",
        }
    )
    seed_columns = [
        "percentage", "masked_degree", "num_masked_neurons", "selected_seed",
        "baseline_accuracy", "baseline_macro_f1", "selected_accuracy",
        "selected_macro_f1", "random_seed_index", "random_seed",
        "random_seed_accuracy", "random_seed_delta_accuracy",
        "random_seed_macro_f1", "random_seed_delta_macro_f1",
    ]
    seed_path = input_dir / "llama_masking_comparison_by_seed.csv"
    seed_table[seed_columns].sort_values(
        ["percentage", "masked_degree", "random_seed_index"]
    ).to_csv(seed_path, index=False, float_format="%.8f")

    markdown = comparison.copy()
    for column in [
        "baseline_accuracy", "baseline_macro_f1", "selected_accuracy",
        "selected_delta_accuracy", "selected_macro_f1", "selected_delta_macro_f1",
        "random_accuracy_mean", "random_accuracy_std", "random_macro_f1_mean",
        "random_macro_f1_std", "random_macro_f1_min", "random_macro_f1_max",
        "selected_minus_random_mean_f1",
    ]:
        markdown[column] = markdown[column].map(metric)
    markdown = markdown.rename(
        columns={
            "percentage": "Mask %", "masked_degree": "Association degree",
            "num_masked_neurons": "Masked neurons", "selected_seed": "Selected seed",
            "baseline_accuracy": "Baseline accuracy", "baseline_macro_f1": "Baseline macro-F1",
            "selected_accuracy": "Selected accuracy", "selected_delta_accuracy": "Selected Δ accuracy",
            "selected_macro_f1": "Selected macro-F1", "selected_delta_macro_f1": "Selected Δ macro-F1",
            "random_seed_count": "Random runs", "random_seed_values": "Random seeds",
            "random_accuracy_mean": "Random mean accuracy", "random_accuracy_std": "Random accuracy SD",
            "random_macro_f1_mean": "Random mean macro-F1", "random_macro_f1_std": "Random macro-F1 SD",
            "random_macro_f1_min": "Random min macro-F1", "random_macro_f1_max": "Random max macro-F1",
            "selected_minus_random_mean_f1": "Selected − random mean F1",
            "f1_z_score": "F1 z-score", "empirical_p_lower_tail": "Empirical lower-tail p",
        }
    )
    report_path = input_dir / "llama_masking_comparison_summary.md"
    report_path.write_text(
        "# Llama selected-neuron versus random-masking comparison\n\n"
        f"Baseline: accuracy {metric(baseline['accuracy'])}; macro-F1 "
        f"{metric(baseline['macro_f1'])}. Selected-neuron masks use seed "
        f"{args.selected_seed}; each random control has 10 matched masks.\n\n"
        + dataframe_to_markdown(markdown)
        + "\n"
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {seed_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
