#!/usr/bin/env python3
"""Analyze emotion-wise single and shared Sarvam-1 neurons.

Input JSON structure:

selected_neurons
  -> "0.5%", "1%", "2%", "5%", "10%"
      -> layer_0 ... layer_27
          -> emotion
              -> [neuron_ids...]

A physical neuron is identified by:
    (layer_index, neuron_id)

Outputs:
1. detailed_neuron_membership.csv
2. single_emotion_summary.csv
3. single_emotion_neurons.csv
4. shared_emotion_summary.csv
5. shared_emotion_neurons.csv
6. shared_combination_summary.csv
7. single_emotion_counts.png
8. shared_emotion_counts_<percentage>.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def percentage_key(value: float) -> str:
    return f"{value:g}%"


def validate_input(data: dict) -> None:
    required = {
        "percentages",
        "num_layers",
        "neurons_per_layer",
        "labels",
        "selected_neurons",
    }

    missing = required - data.keys()

    if missing:
        raise ValueError(
            f"Input JSON is missing keys: {sorted(missing)}"
        )

    percentages = [
        float(v) for v in data["percentages"]
    ]

    num_layers = int(data["num_layers"])
    width = int(data["neurons_per_layer"])
    labels = list(data["labels"])

    if not labels:
        raise ValueError("No emotion labels found.")

    for percentage in percentages:
        key = percentage_key(percentage)

        if key not in data["selected_neurons"]:
            raise ValueError(
                f"Missing selection percentage: {key}"
            )

        by_layer = data["selected_neurons"][key]

        if len(by_layer) != num_layers:
            raise ValueError(
                f"{key}: expected {num_layers} layers, "
                f"found {len(by_layer)}"
            )

        for layer_index in range(num_layers):
            layer_name = f"layer_{layer_index}"

            if layer_name not in by_layer:
                raise ValueError(
                    f"{key}: missing {layer_name}"
                )

            by_emotion = by_layer[layer_name]

            for emotion in labels:
                if emotion not in by_emotion:
                    raise ValueError(
                        f"{key}/{layer_name}: "
                        f"missing emotion '{emotion}'"
                    )

                neuron_ids = by_emotion[emotion]

                # Check duplicates inside same layer/emotion
                if len(neuron_ids) != len(set(neuron_ids)):
                    raise ValueError(
                        f"Duplicate neuron ID in "
                        f"{key}/{layer_name}/{emotion}"
                    )

                # Check neuron IDs are valid
                for neuron in neuron_ids:
                    neuron = int(neuron)

                    if not 0 <= neuron < width:
                        raise ValueError(
                            f"Invalid neuron ID {neuron} in "
                            f"{key}/{layer_name}/{emotion}"
                        )


def build_associations(
    data: dict,
    percentage: float,
) -> dict[tuple[int, int], set[str]]:
    """Return:
    (layer, neuron_id) -> set of emotions
    """

    key = percentage_key(percentage)

    associations: dict[
        tuple[int, int],
        set[str]
    ] = {}

    for layer_name, by_emotion in data[
        "selected_neurons"
    ][key].items():

        layer = int(
            layer_name.removeprefix("layer_")
        )

        for emotion, neuron_ids in by_emotion.items():

            for neuron in neuron_ids:

                neuron = int(neuron)

                associations.setdefault(
                    (layer, neuron),
                    set()
                ).add(emotion)

    return associations


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def analyze(
    data: dict
) -> dict[str, list[dict]]:

    percentages = [
        float(v)
        for v in data["percentages"]
    ]

    labels = list(data["labels"])

    label_order = {
        label: index
        for index, label in enumerate(labels)
    }

    detailed_rows = []
    single_summary_rows = []
    single_neuron_rows = []
    shared_summary_rows = []
    shared_neuron_rows = []
    combination_rows = []

    for percentage in percentages:

        associations = build_associations(
            data,
            percentage
        )

        # =====================================================
        # Detailed membership
        # =====================================================

        for (
            layer,
            neuron_id
        ), emotion_set in sorted(
            associations.items()
        ):

            emotions = sorted(
                emotion_set,
                key=lambda emotion:
                label_order[emotion]
            )

            degree = len(emotions)

            detailed_rows.append(
                {
                    "percentage": percentage,
                    "layer": layer,
                    "neuron_id": neuron_id,
                    "association_degree": degree,
                    "emotions": ";".join(emotions),
                }
            )

            # =================================================
            # Single-emotion neurons
            # =================================================

            if degree == 1:

                single_neuron_rows.append(
                    {
                        "percentage": percentage,
                        "emotion": emotions[0],
                        "layer": layer,
                        "neuron_id": neuron_id,
                    }
                )

            # =================================================
            # Shared neurons
            # =================================================

            else:

                shared_neuron_rows.append(
                    {
                        "percentage": percentage,
                        "association_degree": degree,
                        "layer": layer,
                        "neuron_id": neuron_id,
                        "emotions": ";".join(
                            emotions
                        ),
                    }
                )

        # =====================================================
        # Single-emotion count by emotion
        # =====================================================

        single_counter = Counter()

        for emotion_set in associations.values():

            if len(emotion_set) == 1:

                emotion = next(
                    iter(emotion_set)
                )

                single_counter[emotion] += 1

        for emotion in labels:

            single_summary_rows.append(
                {
                    "percentage": percentage,
                    "emotion": emotion,
                    "single_emotion_neuron_count":
                    single_counter[emotion],
                }
            )

        # =====================================================
        # Shared count emotion-wise
        #
        # Example:
        # degree=2, sadness
        # -> number of degree-2 neurons containing sadness
        # =====================================================

        shared_counter = defaultdict(
            Counter
        )

        for emotion_set in associations.values():

            degree = len(emotion_set)

            if degree >= 2:

                for emotion in emotion_set:

                    shared_counter[
                        degree
                    ][emotion] += 1

        for degree in range(
            2,
            len(labels) + 1
        ):

            for emotion in labels:

                shared_summary_rows.append(
                    {
                        "percentage": percentage,
                        "association_degree": degree,
                        "emotion": emotion,
                        "shared_neuron_count":
                        shared_counter[
                            degree
                        ][emotion],
                    }
                )

        # =====================================================
        # Exact emotion combinations
        #
        # Examples:
        # fear;sadness
        # anger;disgust;fear
        # =====================================================

        combination_counter = Counter()

        for emotion_set in associations.values():

            degree = len(emotion_set)

            if degree >= 2:

                combination = tuple(
                    sorted(
                        emotion_set,
                        key=lambda emotion:
                        label_order[emotion]
                    )
                )

                combination_counter[
                    (
                        degree,
                        combination
                    )
                ] += 1

        for (
            degree,
            combination
        ), count in sorted(
            combination_counter.items(),
            key=lambda item: (
                item[0][0],
                tuple(
                    label_order[emotion]
                    for emotion
                    in item[0][1]
                )
            )
        ):

            combination_rows.append(
                {
                    "percentage": percentage,
                    "association_degree":
                    degree,
                    "emotion_combination":
                    ";".join(
                        combination
                    ),
                    "neuron_count":
                    count,
                }
            )

    return {
        "detailed":
        detailed_rows,

        "single_summary":
        single_summary_rows,

        "single_neurons":
        single_neuron_rows,

        "shared_summary":
        shared_summary_rows,

        "shared_neurons":
        shared_neuron_rows,

        "combinations":
        combination_rows,
    }


def plot_single_emotion_counts(
    rows: list[dict],
    labels: list[str],
    percentages: list[float],
    path: Path,
) -> None:

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    lookup = {
        (
            float(row["percentage"]),
            row["emotion"]
        ):
        int(
            row[
                "single_emotion_neuron_count"
            ]
        )
        for row in rows
    }

    x = np.arange(
        len(labels)
    )

    n = len(
        percentages
    )

    width = 0.8 / max(
        n,
        1
    )

    figure, axis = plt.subplots(
        figsize=(15, 6)
    )

    for i, percentage in enumerate(
        percentages
    ):

        values = [
            lookup[
                (
                    percentage,
                    emotion
                )
            ]
            for emotion in labels
        ]

        offset = (
            i - (n - 1) / 2
        ) * width

        axis.bar(
            x + offset,
            values,
            width=width,
            label=f"{percentage:g}%"
        )

    axis.set_xticks(
        x,
        labels,
        rotation=35,
        ha="right"
    )

    axis.set_xlabel(
        "Emotion"
    )

    axis.set_ylabel(
        "Single-emotion neuron count"
    )

    axis.set_title(
        "Emotion-wise single-emotion selected neurons"
    )

    axis.legend(
        title="Selection percentage"
    )

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


def plot_shared_emotion_counts(
    rows: list[dict],
    labels: list[str],
    percentages: list[float],
    output_dir: Path,
) -> None:

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    for percentage in percentages:

        subset = [
            row
            for row in rows
            if float(
                row["percentage"]
            ) == percentage
        ]

        max_degree = max(
            int(
                row[
                    "association_degree"
                ]
            )
            for row in subset
        )

        degrees = list(
            range(
                2,
                max_degree + 1
            )
        )

        matrix = np.zeros(
            (
                len(labels),
                len(degrees)
            ),
            dtype=int
        )

        emotion_to_index = {
            emotion: index
            for index, emotion
            in enumerate(labels)
        }

        degree_to_index = {
            degree: index
            for index, degree
            in enumerate(degrees)
        }

        for row in subset:

            emotion_index = (
                emotion_to_index[
                    row["emotion"]
                ]
            )

            degree_index = (
                degree_to_index[
                    int(
                        row[
                            "association_degree"
                        ]
                    )
                ]
            )

            matrix[
                emotion_index,
                degree_index
            ] = int(
                row[
                    "shared_neuron_count"
                ]
            )

        figure, axis = plt.subplots(
            figsize=(14, 7)
        )

        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="viridis"
        )

        axis.set_xticks(
            range(
                len(degrees)
            ),
            [
                f"{degree} emotions shared"
                for degree in degrees
            ],
            rotation=35,
            ha="right"
        )

        axis.set_yticks(
            range(
                len(labels)
            ),
            labels
        )

        axis.set_xlabel(
            "Association degree"
        )

        axis.set_ylabel(
            "Emotion included in shared neuron"
        )

        axis.set_title(
            f"Emotion-wise shared selected neurons — "
            f"top {percentage:g}%"
        )

        # Write count in every cell
        for row_index in range(
            matrix.shape[0]
        ):

            for column_index in range(
                matrix.shape[1]
            ):

                count = int(
                    matrix[
                        row_index,
                        column_index
                    ]
                )

                normalized = (
                    image.norm(
                        count
                    )
                )

                text_color = (
                    "white"
                    if normalized < 0.48
                    else "black"
                )

                axis.text(
                    column_index,
                    row_index,
                    f"{count:,}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color
                )

        colorbar = figure.colorbar(
            image,
            ax=axis
        )

        colorbar.set_label(
            "Neuron count"
        )

        figure.tight_layout()

        path = (
            output_dir
            /
            f"shared_emotion_counts_"
            f"{percentage:g}pct.png"
        )

        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(
            figure
        )


def print_summary(
    results: dict[str, list[dict]],
    labels: list[str],
    percentages: list[float],
) -> None:

    print()

    print("=" * 78)

    print(
        "SARVAM-1 EMOTION-WISE "
        "SINGLE / SHARED NEURON REPORT"
    )

    print("=" * 78)

    for percentage in percentages:

        print()

        print(
            f"Top {percentage:g}%"
        )

        print("-" * 78)

        print(
            "Single-emotion neurons:"
        )

        for emotion in labels:

            row = next(
                row
                for row
                in results[
                    "single_summary"
                ]
                if float(
                    row["percentage"]
                ) == percentage
                and row["emotion"]
                == emotion
            )

            print(
                f"  {emotion:<14} "
                f"{int(row['single_emotion_neuron_count']):>7,}"
            )

        print()

        print(
            "Shared neurons containing each emotion:"
        )

        for emotion in labels:

            values = []

            for degree in range(
                2,
                len(labels) + 1
            ):

                row = next(
                    row
                    for row
                    in results[
                        "shared_summary"
                    ]
                    if float(
                        row[
                            "percentage"
                        ]
                    ) == percentage
                    and int(
                        row[
                            "association_degree"
                        ]
                    ) == degree
                    and row[
                        "emotion"
                    ] == emotion
                )

                values.append(
                    f"d{degree}="
                    f"{int(row['shared_neuron_count']):,}"
                )

            print(
                f"  {emotion:<14}"
                + "  "
                + "  ".join(values)
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input-file",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "top_selectivity_neurons.json"
        )
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "emotion_membership_analysis"
        )
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_file
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    validate_input(
        data
    )

    labels = list(
        data["labels"]
    )

    percentages = [
        float(v)
        for v
        in data["percentages"]
    ]

    results = analyze(
        data
    )

    # =========================================================
    # CSV OUTPUTS
    # =========================================================

    write_csv(
        output_dir
        / "detailed_neuron_membership.csv",
        results["detailed"],
        [
            "percentage",
            "layer",
            "neuron_id",
            "association_degree",
            "emotions",
        ]
    )

    write_csv(
        output_dir
        / "single_emotion_summary.csv",
        results["single_summary"],
        [
            "percentage",
            "emotion",
            "single_emotion_neuron_count",
        ]
    )

    write_csv(
        output_dir
        / "single_emotion_neurons.csv",
        results["single_neurons"],
        [
            "percentage",
            "emotion",
            "layer",
            "neuron_id",
        ]
    )

    write_csv(
        output_dir
        / "shared_emotion_summary.csv",
        results["shared_summary"],
        [
            "percentage",
            "association_degree",
            "emotion",
            "shared_neuron_count",
        ]
    )

    write_csv(
        output_dir
        / "shared_emotion_neurons.csv",
        results["shared_neurons"],
        [
            "percentage",
            "association_degree",
            "layer",
            "neuron_id",
            "emotions",
        ]
    )

    write_csv(
        output_dir
        / "shared_combination_summary.csv",
        results["combinations"],
        [
            "percentage",
            "association_degree",
            "emotion_combination",
            "neuron_count",
        ]
    )

    # =========================================================
    # PNG OUTPUTS
    # =========================================================

    plot_single_emotion_counts(
        results[
            "single_summary"
        ],
        labels,
        percentages,
        output_dir
        / "single_emotion_counts.png"
    )

    plot_shared_emotion_counts(
        results[
            "shared_summary"
        ],
        labels,
        percentages,
        output_dir
    )

    # =========================================================
    # TERMINAL REPORT
    # =========================================================

    print_summary(
        results,
        labels,
        percentages
    )

    print()

    print("=" * 78)

    print(
        f"Saved reports in: "
        f"{output_dir}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
