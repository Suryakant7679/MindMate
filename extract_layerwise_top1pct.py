#!/usr/bin/env python3

"""
Extract the top 1% emotion-selective MLP neurons from Sarvam-1,
independently for every transformer layer and every emotion.

Expected input
--------------
results/llama31_8b_emotions_full/selectivity.npy

Shape:
    (32, 14336, 10)

Axis order:
    axis 0 -> transformer layer
    axis 1 -> MLP neuron
    axis 2 -> emotion

Output
------
results/llama31_8b_layerwise_top1pct/

    layer_01_top1pct.csv
    layer_02_top1pct.csv
    ...
    layer_28_top1pct.csv

    all_layers_top1pct.csv
    all_layers_top1pct.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SELECTIVITY_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_emotions_full"
    / "selectivity.npy"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

TOP_PERCENT = 1.0

EXPECTED_NUM_LAYERS = 32
EXPECTED_NEURONS_PER_LAYER = 14336

EMOTIONS = [
    "admiration",
    "anger",
    "disapproval",
    "disgust",
    "fear",
    "joy",
    "love",
    "neutral",
    "sadness",
    "surprise",
]

EXPECTED_NUM_EMOTIONS = len(EMOTIONS)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not SELECTIVITY_FILE.is_file():
    raise FileNotFoundError(
        "\nFull selectivity tensor was not found.\n\n"
        f"Expected file:\n{SELECTIVITY_FILE}\n\n"
        "The file must be a NumPy array with shape:\n"
        "(32, 14336, 10)\n\n"
        "Generate selectivity.npy first, then rerun this script."
    )


# ============================================================
# LOAD FULL SELECTIVITY TENSOR
# ============================================================

print("=" * 80)
print("SARVAM-1 LAYER-WISE TOP-1% NEURON EXTRACTION")
print("=" * 80)

print(f"\nLoading:\n{SELECTIVITY_FILE}")

# mmap_mode avoids unnecessarily copying the entire tensor
S = np.load(
    SELECTIVITY_FILE,
    mmap_mode="r"
)

print(f"\nLoaded shape : {S.shape}")
print(f"Data type    : {S.dtype}")


# ============================================================
# VALIDATE SHAPE
# ============================================================

expected_shape = (
    EXPECTED_NUM_LAYERS,
    EXPECTED_NEURONS_PER_LAYER,
    EXPECTED_NUM_EMOTIONS,
)

if S.shape != expected_shape:
    raise ValueError(
        "\nUnexpected selectivity tensor shape.\n"
        f"Found    : {S.shape}\n"
        f"Expected : {expected_shape}\n\n"
        "Required axis order:\n"
        "(layer, neuron, emotion)"
    )

num_layers, num_neurons, num_emotions = S.shape

print()
print(f"Number of layers        : {num_layers}")
print(f"MLP neurons per layer   : {num_neurons}")
print(f"Number of emotions      : {num_emotions}")

total_physical_neurons = num_layers * num_neurons

print(
    f"Total physical MLP dims : "
    f"{total_physical_neurons:,}"
)


# ============================================================
# COMPUTE TOP 1%
# ============================================================

top_k = int(
    num_neurons * TOP_PERCENT / 100.0
)

print()
print(f"Selection percentage : {TOP_PERCENT}%")
print(f"Top-k per emotion     : {top_k}")

if top_k != 143:
    raise ValueError(
        f"Expected top_k = 143 for 14336 neurons, "
        f"but calculated {top_k}."
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"\nOutput directory:\n{OUTPUT_DIR}")


# ============================================================
# STORAGE
# ============================================================

all_rows = []

json_output = {
    "model_id": MODEL_ID,
    "source_selectivity_file": str(SELECTIVITY_FILE),
    "selectivity_shape": list(S.shape),
    "axis_order": [
        "layer",
        "neuron",
        "emotion"
    ],
    "num_layers": num_layers,
    "neurons_per_layer": num_neurons,
    "num_emotions": num_emotions,
    "emotions": EMOTIONS,
    "top_percentage": TOP_PERCENT,
    "top_k_per_layer_per_emotion": top_k,
    "layers": {},
}


# ============================================================
# EXTRACT TOP 1%
# ============================================================

for layer_idx in range(num_layers):

    # Internal HF index: 0..27
    # Human-readable:   1..28
    layer_number = layer_idx + 1

    print()
    print("=" * 80)
    print(
        f"LAYER {layer_number:02d} "
        f"(model layer index = {layer_idx})"
    )
    print("=" * 80)

    layer_rows = []

    layer_key = f"layer_{layer_number:02d}"

    json_output["layers"][layer_key] = {}


    # --------------------------------------------------------
    # Process each emotion independently
    # --------------------------------------------------------

    for emotion_idx, emotion in enumerate(EMOTIONS):

        # All 11008 selectivity values for:
        # one layer + one emotion
        scores = np.asarray(
            S[layer_idx, :, emotion_idx],
            dtype=np.float64
        )

        if scores.shape != (num_neurons,):
            raise ValueError(
                f"Wrong score vector shape for "
                f"layer={layer_idx}, emotion={emotion}: "
                f"{scores.shape}"
            )

        # Ensure no NaN or Inf
        if not np.all(np.isfinite(scores)):
            bad_count = int(
                np.size(scores)
                - np.count_nonzero(np.isfinite(scores))
            )

            raise ValueError(
                f"{bad_count} NaN/Inf values detected in "
                f"layer={layer_idx}, emotion={emotion}"
            )


        # ====================================================
        # RANK ALL 11008 NEURONS
        # ====================================================
        #
        # Highest selectivity score first.
        #
        # stable sort ensures deterministic ordering if
        # two neurons have exactly equal scores.
        # ====================================================

        ranked_indices = np.argsort(
            -scores,
            kind="stable"
        )

        top_indices = ranked_indices[:top_k]


        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        if len(top_indices) != top_k:
            raise RuntimeError(
                f"Expected {top_k} selected neurons, "
                f"found {len(top_indices)}."
            )

        if len(np.unique(top_indices)) != top_k:
            raise RuntimeError(
                f"Duplicate neuron IDs detected in "
                f"Layer {layer_number}, emotion={emotion}"
            )


        # ----------------------------------------------------
        # Print concise status
        # ----------------------------------------------------

        best_score = float(
            scores[top_indices[0]]
        )

        cutoff_score = float(
            scores[top_indices[-1]]
        )

        print(
            f"{emotion:12s} "
            f"| selected={top_k:3d} "
            f"| max S={best_score:.8f} "
            f"| rank-110 S={cutoff_score:.8f}"
        )


        # ----------------------------------------------------
        # JSON storage
        # ----------------------------------------------------

        json_output["layers"][
            layer_key
        ][emotion] = []


        # ----------------------------------------------------
        # Store selected neurons
        # ----------------------------------------------------

        for rank, neuron_idx in enumerate(
            top_indices,
            start=1
        ):

            neuron_idx = int(neuron_idx)

            selectivity_score = float(
                scores[neuron_idx]
            )

            row = {
                # Human-readable layer number: 1..28
                "layer": layer_number,

                # Actual HF/Python index: 0..27
                "layer_index": layer_idx,

                "emotion": emotion,

                # Rank within this layer + emotion
                "rank": rank,

                # Actual 0-based MLP neuron index
                "neuron_index": neuron_idx,

                # Optional human-readable neuron number
                "neuron_number": neuron_idx + 1,

                "selectivity_score": selectivity_score,
            }

            layer_rows.append(row)
            all_rows.append(row)

            json_output["layers"][
                layer_key
            ][emotion].append(
                {
                    "rank": rank,
                    "neuron_index": neuron_idx,
                    "neuron_number": neuron_idx + 1,
                    "selectivity_score": selectivity_score,
                }
            )


    # ========================================================
    # VALIDATE CURRENT LAYER
    # ========================================================

    expected_layer_entries = (
        num_emotions * top_k
    )

    if len(layer_rows) != expected_layer_entries:
        raise RuntimeError(
            f"Layer {layer_number}: expected "
            f"{expected_layer_entries} entries, "
            f"found {len(layer_rows)}."
        )


    # ========================================================
    # SAVE ONE CSV PER LAYER
    # ========================================================

    layer_df = pd.DataFrame(layer_rows)

    layer_csv = (
        OUTPUT_DIR
        / f"layer_{layer_number:02d}_top1pct.csv"
    )

    layer_df.to_csv(
        layer_csv,
        index=False
    )

    print()
    print(
        f"Saved Layer {layer_number}: "
        f"{layer_csv.name}"
    )

    print(
        f"Entries in layer file: "
        f"{len(layer_df):,}"
    )


# ============================================================
# COMBINE ALL LAYERS
# ============================================================

all_df = pd.DataFrame(all_rows)


# ============================================================
# VALIDATE TOTAL NUMBER OF SELECTION ENTRIES
# ============================================================

expected_total_entries = (
    num_layers
    * num_emotions
    * top_k
)

if len(all_df) != expected_total_entries:
    raise RuntimeError(
        f"Expected {expected_total_entries:,} "
        f"total selection entries, "
        f"found {len(all_df):,}."
    )


# ============================================================
# COUNT UNIQUE PHYSICAL NEURONS
# ============================================================
#
# IMPORTANT:
#
# neuron 500 in layer 1 and neuron 500 in layer 2 are
# different physical neurons.
#
# Therefore physical identity = (layer_index, neuron_index)
# ============================================================

unique_physical_neurons = set(
    zip(
        all_df["layer_index"].astype(int),
        all_df["neuron_index"].astype(int),
    )
)

num_unique_physical_neurons = len(
    unique_physical_neurons
)


# ============================================================
# SAVE COMBINED CSV
# ============================================================

combined_csv = (
    OUTPUT_DIR
    / "all_layers_top1pct.csv"
)

all_df.to_csv(
    combined_csv,
    index=False
)


# ============================================================
# SAVE COMBINED JSON
# ============================================================

json_output[
    "total_selection_entries"
] = len(all_df)

json_output[
    "unique_physical_neurons_selected"
] = num_unique_physical_neurons

combined_json = (
    OUTPUT_DIR
    / "all_layers_top1pct.json"
)

with combined_json.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_output,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("EXTRACTION COMPLETED SUCCESSFULLY")
print("=" * 80)

print()
print(f"Model                    : {MODEL_ID}")
print(f"Layers                   : {num_layers}")
print(f"MLP neurons/layer        : {num_neurons:,}")
print(
    f"Total physical MLP dims  : "
    f"{total_physical_neurons:,}"
)

print()
print(f"Emotion classes          : {num_emotions}")
print(f"Top percentage           : {TOP_PERCENT}%")
print(
    f"Selected/layer/emotion   : "
    f"{top_k}"
)

print(
    f"Selections per layer     : "
    f"{top_k * num_emotions:,}"
)

print(
    f"Total selection entries  : "
    f"{len(all_df):,}"
)

print(
    f"Unique physical neurons  : "
    f"{num_unique_physical_neurons:,}"
)

print()
print("Combined CSV:")
print(combined_csv)

print()
print("Combined JSON:")
print(combined_json)

print()
print("Per-layer CSV files:")
print(
    f"{OUTPUT_DIR}/layer_01_top1pct.csv"
)
print("...")
print(
    f"{OUTPUT_DIR}/layer_28_top1pct.csv"
)

print()
print("=" * 80)
