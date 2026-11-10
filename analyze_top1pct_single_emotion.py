#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "all_layers_top1pct.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "single_emotion_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "layer",
    "layer_index",
    "emotion",
    "rank",
    "neuron_index",
    "neuron_number",
    "selectivity_score",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


print("=" * 80)
print("TOP-1% SINGLE-EMOTION NEURON ANALYSIS")
print("=" * 80)

print(f"Total selection entries : {len(df):,}")


# ============================================================
# PHYSICAL NEURON MEMBERSHIP
# ============================================================
#
# A physical neuron is:
#
#     (layer_index, neuron_index)
#
# We count how many DISTINCT emotions selected that neuron.
# ============================================================

membership = (
    df.groupby(
        ["layer_index", "neuron_index"]
    )
    .agg(
        layer=("layer", "first"),
        neuron_number=("neuron_number", "first"),
        num_emotions=("emotion", "nunique"),
        emotions=(
            "emotion",
            lambda x: sorted(set(x))
        ),
    )
    .reset_index()
)


print(
    f"Unique physical neurons : "
    f"{len(membership):,}"
)


# ============================================================
# DEGREE / MEMBERSHIP DISTRIBUTION
# ============================================================

degree_distribution = (
    membership["num_emotions"]
    .value_counts()
    .sort_index()
    .rename_axis("num_emotions")
    .reset_index(name="num_neurons")
)

print()
print("=" * 80)
print("EMOTION MEMBERSHIP DISTRIBUTION")
print("=" * 80)

print(
    degree_distribution.to_string(
        index=False
    )
)


# ============================================================
# EXACTLY ONE EMOTION
# ============================================================

single = membership[
    membership["num_emotions"] == 1
].copy()

single["emotion"] = single[
    "emotions"
].apply(
    lambda x: x[0]
)


print()
print("=" * 80)
print("SINGLE-EMOTION NEURONS")
print("=" * 80)

print(
    f"Exactly one emotion : "
    f"{len(single):,}"
)


# ============================================================
# ADD ORIGINAL RANK + SELECTIVITY SCORE
# ============================================================

single_details = single.merge(
    df[
        [
            "layer",
            "layer_index",
            "emotion",
            "rank",
            "neuron_index",
            "neuron_number",
            "selectivity_score",
        ]
    ],
    on=[
        "layer",
        "layer_index",
        "emotion",
        "neuron_index",
        "neuron_number",
    ],
    how="left",
)


# ============================================================
# SUMMARY BY EMOTION
# ============================================================

emotion_summary = (
    single_details
    .groupby("emotion")
    .size()
    .reindex(
        [
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
        ],
        fill_value=0,
    )
    .reset_index(name="single_emotion_neurons")
)


print()
print("=" * 80)
print("SINGLE-EMOTION NEURONS BY EMOTION")
print("=" * 80)

print(
    emotion_summary.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY BY LAYER
# ============================================================

layer_summary = (
    single_details
    .groupby("layer")
    .size()
    .reindex(
        range(1, 33),
        fill_value=0
    )
    .reset_index(name="single_emotion_neurons")
)


print()
print("=" * 80)
print("SINGLE-EMOTION NEURONS BY LAYER")
print("=" * 80)

print(
    layer_summary.to_string(
        index=False
    )
)


# ============================================================
# LAYER × EMOTION MATRIX
# ============================================================

layer_emotion_summary = (
    single_details
    .groupby(
        ["layer", "emotion"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

emotion_order = [
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

layer_emotion_summary = (
    layer_emotion_summary
    .reindex(
        index=range(1, 33),
        fill_value=0
    )
    .reindex(
        columns=emotion_order,
        fill_value=0
    )
)


print()
print("=" * 80)
print("LAYER × EMOTION SINGLE-EMOTION COUNTS")
print("=" * 80)

print(
    layer_emotion_summary.to_string()
)


# ============================================================
# SAVE RESULTS
# ============================================================

membership_file = (
    OUTPUT_DIR
    / "all_neuron_membership.csv"
)

single_file = (
    OUTPUT_DIR
    / "single_emotion_neurons.csv"
)

emotion_summary_file = (
    OUTPUT_DIR
    / "single_emotion_counts_by_emotion.csv"
)

layer_summary_file = (
    OUTPUT_DIR
    / "single_emotion_counts_by_layer.csv"
)

layer_emotion_file = (
    OUTPUT_DIR
    / "single_emotion_counts_layer_by_emotion.csv"
)

degree_file = (
    OUTPUT_DIR
    / "membership_degree_distribution.csv"
)


membership_to_save = membership.copy()

membership_to_save["emotions"] = (
    membership_to_save["emotions"]
    .apply(lambda x: ",".join(x))
)

membership_to_save.to_csv(
    membership_file,
    index=False
)

single_details[
    [
        "layer",
        "layer_index",
        "emotion",
        "rank",
        "neuron_index",
        "neuron_number",
        "selectivity_score",
        "num_emotions",
    ]
].sort_values(
    [
        "emotion",
        "layer",
        "rank",
    ]
).to_csv(
    single_file,
    index=False
)

emotion_summary.to_csv(
    emotion_summary_file,
    index=False
)

layer_summary.to_csv(
    layer_summary_file,
    index=False
)

layer_emotion_summary.to_csv(
    layer_emotion_file
)

degree_distribution.to_csv(
    degree_file,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

assert (
    degree_distribution["num_neurons"].sum()
    == len(membership)
)

assert (
    len(single)
    ==
    int(
        degree_distribution.loc[
            degree_distribution["num_emotions"] == 1,
            "num_neurons"
        ].iloc[0]
    )
    if (degree_distribution["num_emotions"] == 1).any()
    else len(single) == 0
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(f"Membership table:")
print(membership_file)

print()
print(f"Single-emotion neurons:")
print(single_file)

print()
print(f"Counts by emotion:")
print(emotion_summary_file)

print()
print(f"Counts by layer:")
print(layer_summary_file)

print()
print(f"Layer × emotion counts:")
print(layer_emotion_file)

print()
print(f"Degree distribution:")
print(degree_file)

print()
print("Done.")
