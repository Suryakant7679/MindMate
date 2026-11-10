#!/usr/bin/env python3
"""Report single- and multi-emotion membership of selected neurons.

Reads the JSON created by emotion_neuron_statistics.py.

For each selection percentage, a physical neuron is identified by:

    (layer_index, neuron_id)

A neuron is associated with an emotion when it occurs in that emotion's
top-selectivity set within the same layer.

Association degree:
    0  -> not selected for any emotion
    1  -> selected for exactly one emotion
    2  -> selected for exactly two emotions
    ...
    10 -> selected for all ten emotions
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def build_rows(
    data: dict,
) -> tuple[list[dict], list[float], list[list[int]]]:

    percentages = [float(value) for value in data["percentages"]]

    num_layers = int(data["num_layers"])
    width = int(data["neurons_per_layer"])
    num_emotions = len(data["labels"])

    total_neurons = num_layers * width

    rows = []
    heatmap = []

    for percentage in percentages:

        percentage_key = f"{percentage:g}%"

        # ---------------------------------------------------------
        # Map physical neuron -> set of emotions
        #
        # Example:
        # (16, 8399) -> {"anger", "fear", "sadness"}
        # ---------------------------------------------------------
        associations: dict[tuple[int, int], set[str]] = {}

        selected_for_percentage = data["selected_neurons"][percentage_key]

        for layer_name, by_emotion in selected_for_percentage.items():

            layer = int(layer_name.removeprefix("layer_"))

            for emotion, neuron_ids in by_emotion.items():

                for neuron in neuron_ids:

                    neuron = int(neuron)

                    associations.setdefault(
                        (layer, neuron),
                        set(),
                    ).add(emotion)

        # ---------------------------------------------------------
        # counts[d] =
        # number of physical neurons associated with exactly d emotions
        # ---------------------------------------------------------

        counts = [0] * (num_emotions + 1)

        # Degree 0 = never selected for any emotion
        counts[0] = total_neurons - len(associations)

        # Degrees 1...10
        for emotions in associations.values():

            degree = len(emotions)

            if not 1 <= degree <= num_emotions:
                raise RuntimeError(
                    f"Invalid association degree: {degree}"
                )

            counts[degree] += 1

        distinct_selected_neurons = len(associations)

        # ---------------------------------------------------------
        # Write CSV rows
        # ---------------------------------------------------------

        for degree, count in enumerate(counts):

            if degree == 0:
                label = "not_selected"

            elif degree == 1:
                label = "single_emotion"

            else:
                label = f"shared_{degree}_emotions"

            percentage_of_all = (
                count / total_neurons * 100.0
            )

            if degree == 0 or distinct_selected_neurons == 0:
                percentage_of_selected = 0.0
            else:
                percentage_of_selected = (
                    count
                    / distinct_selected_neurons
                    * 100.0
                )

            rows.append(
                {
                    "percentage": percentage,
                    "association_degree": degree,
                    "association_label": label,
                    "neuron_count": count,
                    "percentage_of_all_neurons": percentage_of_all,
                    "percentage_of_selected_neurons": percentage_of_selected,
                }
            )

        # Do not display degree-0 neurons in heatmap.
        heatmap.append(counts[1:])

    return rows, percentages, heatmap


def write_png(
    percentages: list[float],
    counts: list[list[int]],
    path: Path,
) -> None:

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    values = np.asarray(counts)

    # -------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(17, 6.5)
    )

    image = axis.imshow(
        values,
        aspect="auto",
        cmap="viridis",
    )

    # -------------------------------------------------------------
    # Human-readable association labels
    # -------------------------------------------------------------

    association_labels = [
        "Single\nemotion",
        "2 emotions\nshared",
        "3 emotions\nshared",
        "4 emotions\nshared",
        "5 emotions\nshared",
        "6 emotions\nshared",
        "7 emotions\nshared",
        "8 emotions\nshared",
        "9 emotions\nshared",
        "10 emotions\nshared",
    ]

    axis.set_xticks(
        np.arange(len(association_labels))
    )

    axis.set_xticklabels(
        association_labels,
        fontsize=10,
    )

    # -------------------------------------------------------------
    # Selection percentages
    # -------------------------------------------------------------

    percentage_labels = [
        f"{percentage:g}%"
        for percentage in percentages
    ]

    axis.set_yticks(
        np.arange(len(percentages))
    )

    axis.set_yticklabels(
        percentage_labels,
        fontsize=11,
    )

    # -------------------------------------------------------------
    # Axis labels / title
    # -------------------------------------------------------------

    axis.set_xlabel(
        "Emotion-association category",
        fontsize=13,
        labelpad=12,
    )

    axis.set_ylabel(
        "Neuron selection percentage",
        fontsize=13,
        labelpad=12,
    )

    axis.set_title(
        "Emotion-association degree of selected Sarvam-1 neurons",
        fontsize=17,
        pad=16,
    )

    # -------------------------------------------------------------
    # Write values inside heatmap cells
    #
    # Adaptive text:
    # dark background  -> white
    # bright background -> black
    # -------------------------------------------------------------

    for row_index, row in enumerate(values):

        for column_index, count in enumerate(row):

            normalized_value = image.norm(count)

            # Viridis is dark at low values and bright at high values.
            if normalized_value < 0.48:
                text_color = "white"
            else:
                text_color = "black"

            axis.text(
                column_index,
                row_index,
                f"{int(count):,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    # -------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.025,
    )

    colorbar.set_label(
        "Number of distinct physical neurons",
        fontsize=12,
    )

    colorbar.ax.tick_params(
        labelsize=10
    )

    # -------------------------------------------------------------
    # Styling
    # -------------------------------------------------------------

    axis.tick_params(
        axis="x",
        bottom=True,
        top=False,
    )

    axis.tick_params(
        axis="y",
        left=True,
        right=False,
    )

    figure.tight_layout()

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def validate_input(data: dict) -> None:
    """Basic validation of the selected-neuron JSON."""

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

    num_layers = int(data["num_layers"])
    width = int(data["neurons_per_layer"])
    labels = list(data["labels"])

    if len(labels) == 0:
        raise ValueError("No emotion labels found.")

    # -------------------------------------------------------------
    # Validate every selection percentage
    # -------------------------------------------------------------

    for percentage in data["percentages"]:

        percentage = float(percentage)
        key = f"{percentage:g}%"

        if key not in data["selected_neurons"]:
            raise ValueError(
                f"Missing selected-neuron set for {key}"
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

            # Every emotion should exist.
            for emotion in labels:

                if emotion not in by_emotion:
                    raise ValueError(
                        f"{key}/{layer_name}: "
                        f"missing emotion '{emotion}'"
                    )

                neuron_ids = by_emotion[emotion]

                # No duplicates inside same emotion/layer.
                if len(neuron_ids) != len(set(neuron_ids)):
                    raise ValueError(
                        f"Duplicate neuron ID found in "
                        f"{key}/{layer_name}/{emotion}"
                    )

                # Validate neuron IDs.
                for neuron in neuron_ids:

                    neuron = int(neuron)

                    if neuron < 0 or neuron >= width:
                        raise ValueError(
                            f"Invalid neuron ID {neuron} in "
                            f"{key}/{layer_name}/{emotion}"
                        )


def print_summary(rows: list[dict]) -> None:

    percentages = sorted(
        set(row["percentage"] for row in rows)
    )

    print()
    print("=" * 70)
    print("SARVAM-1 EMOTION ASSOCIATION REPORT")
    print("=" * 70)

    for percentage in percentages:

        subset = [
            row
            for row in rows
            if row["percentage"] == percentage
        ]

        print()
        print(f"Selection percentage: {percentage:g}%")
        print("-" * 70)

        for row in subset:

            degree = row["association_degree"]

            if degree == 0:
                continue

            count = row["neuron_count"]

            if degree == 1:
                label = "Single emotion"
            else:
                label = f"{degree} emotions shared"

            print(
                f"{label:<22} : "
                f"{count:>8,}"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input-file",
        default=(
            "results/llama31_8b_neuron_analysis/"
            "top_selectivity_neurons.json"
        ),
    )

    parser.add_argument(
        "--output-csv",
        default=(
            "results/llama31_8b_neuron_analysis/"
            "emotion_association_degree_report.csv"
        ),
    )

    parser.add_argument(
        "--output-png",
        default=(
            "results/llama31_8b_neuron_analysis/"
            "emotion_association_degree_report.png"
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Load JSON
    # -------------------------------------------------------------

    input_path = Path(args.input_file)

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_input(data)

    # -------------------------------------------------------------
    # Build overlap report
    # -------------------------------------------------------------

    rows, percentages, counts = build_rows(
        data
    )

    # -------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------

    csv_path = Path(args.output_csv)
    png_path = Path(args.output_png)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------------------------------------------
    # Save PNG
    # -------------------------------------------------------------

    write_png(
        percentages,
        counts,
        png_path,
    )

    # -------------------------------------------------------------
    # Terminal report
    # -------------------------------------------------------------

    print_summary(rows)

    print()
    print("=" * 70)
    print(f"CSV saved to : {csv_path}")
    print(f"PNG saved to : {png_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
