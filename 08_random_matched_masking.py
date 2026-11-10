#!/usr/bin/env python3

"""
Layer-count-matched RANDOM neuron masking control for Sarvam-1.

For every condition:

    percentage = 0.5, 1, 2, 5, 10
    degree     = 1, 2, ..., 10

we compare:

    emotion-sensitive masking
vs
    random masking

Random controls preserve EXACTLY the same number of masked neurons
in every transformer layer.

Degree meanings:
    degree 1  -> single-emotion neurons
    degree 2  -> shared by exactly 2 emotions
    ...
    degree 10 -> shared by all 10 emotions

The script uses:

    single_emotion_neurons.csv
    shared_emotion_neurons.csv

and your already-generated selected masking summaries:

    single_neuron_masking_summary.csv
    shared_masking_summary.csv

The random pool EXCLUDES all emotion-selected neurons for the same
percentage by default.

Outputs:

    random_masking_runs.csv
    random_masking_per_emotion.csv
    random_masking_aggregate.csv
    random_masking_results.json

Important:
    - pretrained Sarvam remains frozen
    - probe remains frozen
    - probe is NOT retrained
    - same test split is used
    - masking occurs at MLP down_proj input
    - only valid non-special token positions are masked
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


# ================================================================
# DATASET
# ================================================================

def prepare_dataframe(
    path,
    text_column="hinglish_genz_text",
    label_column="label",
):

    df = pd.read_csv(path)[
        [text_column, label_column]
    ].rename(
        columns={
            text_column: "text",
            label_column: "label",
        }
    )

    df["text"] = (
        df["text"]
        .astype("string")
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .astype("string")
        .str.strip()
    )

    df = (
        df
        .dropna()
        .loc[
            (df["text"] != "")
            & (df["label"] != "")
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    labels = sorted(
        df["label"]
        .unique()
        .tolist()
    )

    mapping = {
        label: i
        for i, label
        in enumerate(labels)
    }

    df["label_id"] = (
        df["label"]
        .map(mapping)
        .astype(int)
    )

    return df, labels


# ================================================================
# EXACT SAME SPLIT AS BASELINE
# ================================================================

def stratified_split(
    labels,
    seed=42,
):

    rng = np.random.default_rng(seed)

    groups = {
        int(y):
        np.where(labels == y)[0].tolist()

        for y in np.unique(labels)
    }

    train = []
    validation = []
    test = []

    for indices in groups.values():

        rng.shuffle(indices)

        n = len(indices)

        n_test = max(
            1,
            round(n * 0.15)
        )

        n_val = max(
            1,
            round(n * 0.15)
        )

        test += indices[:n_test]

        validation += indices[
            n_test:
            n_test + n_val
        ]

        train += indices[
            n_test + n_val:
        ]

    return (
        np.asarray(train),
        np.asarray(validation),
        np.asarray(test),
    )


# ================================================================
# VALID TOKEN MASK
# ================================================================

def valid_mask(
    input_ids,
    attention_mask,
    tokenizer,
):

    mask = (
        attention_mask
        .bool()
        .clone()
    )

    for token_id in set(
        tokenizer.all_special_ids
    ):

        mask &= input_ids.ne(
            token_id
        )

    return mask


# ================================================================
# METRICS
# ================================================================

def classification_metrics(
    y_true,
    y_pred,
    num_classes,
):

    cm = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=int,
    )

    for truth, pred in zip(
        y_true,
        y_pred
    ):

        cm[
            int(truth),
            int(pred)
        ] += 1

    precision = []
    recall = []
    f1 = []

    for c in range(
        num_classes
    ):

        tp = cm[c, c]

        fp = (
            cm[:, c].sum()
            - tp
        )

        fn = (
            cm[c, :].sum()
            - tp
        )

        p = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        r = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        score = (
            2 * p * r / (p + r)
            if p + r
            else 0.0
        )

        precision.append(
            float(p)
        )

        recall.append(
            float(r)
        )

        f1.append(
            float(score)
        )

    accuracy = float(
        np.trace(cm)
        / max(
            1,
            cm.sum()
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": float(
            np.mean(f1)
        ),
        "micro_f1": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


# ================================================================
# LOAD NEURON FILES
# ================================================================

def load_neuron_tables(
    single_path,
    shared_path,
):

    single_df = pd.read_csv(
        single_path
    )

    shared_df = pd.read_csv(
        shared_path
    )

    single_required = {
        "percentage",
        "emotion",
        "layer",
        "neuron_id",
    }

    shared_required = {
        "percentage",
        "association_degree",
        "layer",
        "neuron_id",
        "emotions",
    }

    missing = (
        single_required
        - set(single_df.columns)
    )

    if missing:

        raise ValueError(
            "Single-neuron file missing: "
            f"{sorted(missing)}"
        )

    missing = (
        shared_required
        - set(shared_df.columns)
    )

    if missing:

        raise ValueError(
            "Shared-neuron file missing: "
            f"{sorted(missing)}"
        )

    single_df["percentage"] = (
        single_df["percentage"]
        .astype(float)
    )

    single_df["layer"] = (
        single_df["layer"]
        .astype(int)
    )

    single_df["neuron_id"] = (
        single_df["neuron_id"]
        .astype(int)
    )

    shared_df["percentage"] = (
        shared_df["percentage"]
        .astype(float)
    )

    shared_df[
        "association_degree"
    ] = (
        shared_df[
            "association_degree"
        ]
        .astype(int)
    )

    shared_df["layer"] = (
        shared_df["layer"]
        .astype(int)
    )

    shared_df["neuron_id"] = (
        shared_df["neuron_id"]
        .astype(int)
    )

    return (
        single_df,
        shared_df,
    )


# ================================================================
# SELECT REAL EMOTION-SENSITIVE CONDITION
# ================================================================

def get_selected_condition(
    single_df,
    shared_df,
    percentage,
    degree,
):

    if degree == 1:

        subset = single_df[
            np.isclose(
                single_df[
                    "percentage"
                ],
                percentage,
            )
        ].copy()

    else:

        subset = shared_df[
            np.isclose(
                shared_df[
                    "percentage"
                ],
                percentage,
            )
            &
            (
                shared_df[
                    "association_degree"
                ]
                == degree
            )
        ].copy()

    neurons_by_layer = (
        defaultdict(set)
    )

    for row in subset.itertuples(
        index=False
    ):

        neurons_by_layer[
            int(row.layer)
        ].add(
            int(row.neuron_id)
        )

    return {
        layer: sorted(
            neurons
        )
        for layer, neurons
        in neurons_by_layer.items()
    }


# ================================================================
# ALL EMOTION-SENSITIVE NEURONS AT PERCENTAGE
#
# Used to EXCLUDE emotion-sensitive neurons from random pool.
# ================================================================

def get_all_selected_at_percentage(
    single_df,
    shared_df,
    percentage,
):

    selected_by_layer = (
        defaultdict(set)
    )

    single_subset = single_df[
        np.isclose(
            single_df[
                "percentage"
            ],
            percentage,
        )
    ]

    shared_subset = shared_df[
        np.isclose(
            shared_df[
                "percentage"
            ],
            percentage,
        )
    ]

    for row in single_subset.itertuples(
        index=False
    ):

        selected_by_layer[
            int(row.layer)
        ].add(
            int(row.neuron_id)
        )

    for row in shared_subset.itertuples(
        index=False
    ):

        selected_by_layer[
            int(row.layer)
        ].add(
            int(row.neuron_id)
        )

    return selected_by_layer


# ================================================================
# TRANSFORMER LAYERS
# ================================================================

def get_transformer_layers(
    model,
):

    if (
        hasattr(model, "model")
        and hasattr(
            model.model,
            "layers"
        )
    ):

        return model.model.layers

    if hasattr(
        model,
        "layers"
    ):

        return model.layers

    raise RuntimeError(
        "Could not locate transformer layers."
    )


# ================================================================
# GET MLP WIDTH
# ================================================================

def get_mlp_width(
    model,
):

    layers = get_transformer_layers(
        model
    )

    down_proj = (
        layers[0]
        .mlp
        .down_proj
    )

    if hasattr(
        down_proj,
        "in_features"
    ):

        return int(
            down_proj.in_features
        )

    return int(
        down_proj.weight.shape[1]
    )


# ================================================================
# LAYER-MATCHED RANDOM SAMPLE
# ================================================================

def sample_random_matched_neurons(
    model,
    selected_by_layer,
    excluded_by_layer,
    seed,
):

    rng = np.random.default_rng(
        seed
    )

    width = get_mlp_width(
        model
    )

    random_by_layer = {}

    for (
        layer,
        selected_neurons
    ) in selected_by_layer.items():

        required_count = len(
            selected_neurons
        )

        excluded = set(
            excluded_by_layer.get(
                layer,
                set()
            )
        )

        candidates = np.asarray(
            [
                neuron
                for neuron
                in range(width)
                if neuron
                not in excluded
            ],
            dtype=int,
        )

        if (
            len(candidates)
            < required_count
        ):

            raise RuntimeError(
                f"Layer {layer}: "
                f"need {required_count} "
                "random neurons but only "
                f"{len(candidates)} available."
            )

        sampled = rng.choice(
            candidates,
            size=required_count,
            replace=False,
        )

        random_by_layer[
            layer
        ] = sorted(
            sampled.tolist()
        )

        # Exact layer-count check
        assert (
            len(
                random_by_layer[
                    layer
                ]
            )
            ==
            required_count
        )

    return random_by_layer


# ================================================================
# MASK CONTROLLER
# ================================================================

class NeuronMaskController:

    def __init__(self):

        self.current_valid_mask = None


# ================================================================
# INSTALL MASKS
# ================================================================

def install_mask_hooks(
    model,
    neurons_by_layer,
    controller,
):

    layers = get_transformer_layers(
        model
    )

    handles = []

    for (
        layer_index,
        neuron_ids
    ) in neurons_by_layer.items():

        layer = layers[
            layer_index
        ]

        selected_ids = tuple(
            sorted(
                set(
                    int(x)
                    for x
                    in neuron_ids
                )
            )
        )

        def make_hook(
            ids,
            current_layer,
        ):

            index_cache = {}

            def hook(
                module,
                inputs,
            ):

                if not inputs:

                    return inputs

                x = inputs[0]

                if (
                    controller
                    .current_valid_mask
                    is None
                ):

                    raise RuntimeError(
                        "Valid-token mask "
                        "not initialized."
                    )

                if x.ndim != 3:

                    raise RuntimeError(
                        "Expected down_proj input "
                        "[batch, seq, mlp_width]."
                    )

                device_key = str(
                    x.device
                )

                if (
                    device_key
                    not in index_cache
                ):

                    index = torch.tensor(
                        ids,
                        dtype=torch.long,
                        device=x.device,
                    )

                    if (
                        index.numel() > 0
                        and
                        int(
                            index.max()
                        ) >= x.shape[-1]
                    ):

                        raise RuntimeError(
                            f"Invalid neuron index "
                            f"in layer {current_layer}"
                        )

                    index_cache[
                        device_key
                    ] = index

                index = index_cache[
                    device_key
                ]

                if (
                    index.numel()
                    == 0
                ):

                    return inputs

                valid = (
                    controller
                    .current_valid_mask
                    .to(
                        device=x.device,
                        dtype=torch.bool,
                    )
                )

                masked_x = x.clone()

                selected_values = (
                    masked_x
                    .index_select(
                        -1,
                        index,
                    )
                )

                selected_values = (
                    selected_values
                    .masked_fill(
                        valid.unsqueeze(-1),
                        0.0,
                    )
                )

                masked_x.index_copy_(
                    -1,
                    index,
                    selected_values,
                )

                if len(inputs) == 1:

                    return (
                        masked_x,
                    )

                return (
                    masked_x,
                    *inputs[1:]
                )

            return hook

        handles.append(
            layer
            .mlp
            .down_proj
            .register_forward_pre_hook(
                make_hook(
                    selected_ids,
                    layer_index,
                )
            )
        )

    return handles


def remove_hooks(
    handles,
):

    for handle in handles:

        handle.remove()


# ================================================================
# EVALUATE
# ================================================================

def evaluate(
    model,
    tokenizer,
    classifier,
    texts,
    labels,
    batch_size,
    max_length,
    controller=None,
):

    model.eval()
    classifier.eval()

    input_device = (
        model
        .get_input_embeddings()
        .weight
        .device
    )

    classifier_device = next(
        classifier.parameters()
    ).device

    predictions = []
    probabilities_all = []

    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                len(texts),
                batch_size
            ),
            leave=False,
        ):

            batch_texts = texts[
                start:
                start + batch_size
            ]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )

            enc = {
                k:
                v.to(
                    input_device
                )

                for k, v
                in enc.items()
            }

            mask = valid_mask(
                enc["input_ids"],
                enc["attention_mask"],
                tokenizer,
            )

            if controller is not None:

                controller.current_valid_mask = (
                    mask
                )

            output = model.model(
                **enc,
                output_hidden_states=False,
                use_cache=False,
            )

            hidden = (
                output
                .last_hidden_state
            )

            weights = (
                mask
                .to(
                    hidden.device
                )
                .unsqueeze(-1)
                .to(
                    hidden.dtype
                )
            )

            pooled = (
                (
                    hidden
                    * weights
                )
                .sum(dim=1)
                /
                weights
                .sum(dim=1)
                .clamp_min(1)
            )

            logits = classifier(
                pooled
                .float()
                .to(
                    classifier_device
                )
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            predictions.extend(
                logits
                .argmax(-1)
                .cpu()
                .numpy()
                .tolist()
            )

            probabilities_all.append(
                probabilities
                .float()
                .cpu()
            )

    if controller is not None:

        controller.current_valid_mask = None

    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    probabilities_all = (
        torch.cat(
            probabilities_all,
            dim=0,
        )
        .numpy()
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    metrics = classification_metrics(
        labels,
        predictions,
        probabilities_all.shape[1],
    )

    true_prob = (
        probabilities_all[
            np.arange(
                len(labels)
            ),
            labels,
        ]
    )

    metrics[
        "mean_true_class_probability"
    ] = float(
        np.mean(
            true_prob
        )
    )

    per_class_prob = []

    for class_index in range(
        probabilities_all.shape[1]
    ):

        mask_class = (
            labels
            == class_index
        )

        per_class_prob.append(
            float(
                np.mean(
                    probabilities_all[
                        mask_class,
                        class_index,
                    ]
                )
            )
        )

    metrics[
        "per_class_true_probability"
    ] = per_class_prob

    return metrics


# ================================================================
# READ EXISTING SELECTED-MASK RESULTS
# ================================================================

def load_selected_results(
    single_summary_path,
    shared_summary_path,
):

    single = pd.read_csv(
        single_summary_path
    )

    shared = pd.read_csv(
        shared_summary_path
    )

    results = {}

    for row in single.itertuples(
        index=False
    ):

        key = (
            float(
                row.percentage
            ),
            1,
        )

        results[key] = {
            "num_masked_neurons":
            int(
                row.num_masked_neurons
            ),

            "delta_accuracy":
            float(
                row.delta_accuracy
            ),

            "delta_macro_f1":
            float(
                row.delta_macro_f1
            ),

            "delta_true_probability":
            float(
                row.delta_true_probability
            ),
        }

    for row in shared.itertuples(
        index=False
    ):

        key = (
            float(
                row.percentage
            ),
            int(
                row.masked_degree
            ),
        )

        results[key] = {
            "num_masked_neurons":
            int(
                row.num_masked_neurons
            ),

            "delta_accuracy":
            float(
                row.delta_accuracy
            ),

            "delta_macro_f1":
            float(
                row.delta_macro_f1
            ),

            "delta_true_probability":
            float(
                row.delta_true_probability
            ),
        }

    return results


# ================================================================
# CSV HELPER
# ================================================================

def write_csv(
    path,
    rows,
):

    if not rows:

        return

    with Path(path).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# ================================================================
# MAIN
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )

    parser.add_argument(
        "--csv-path",
        default=(
            "data/"
            "mlt_hinghlish_dataset_clean.csv"
        ),
    )

    parser.add_argument(
        "--probe-path",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "04_probe/"
            "probe.pt"
        ),
    )

    parser.add_argument(
        "--single-neuron-csv",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "emotion_membership_analysis/"
            "single_emotion_neurons.csv"
        ),
    )

    parser.add_argument(
        "--shared-neuron-csv",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "emotion_membership_analysis/"
            "shared_emotion_neurons.csv"
        ),
    )

    parser.add_argument(
        "--single-selected-summary",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "05_single_neuron_masking/"
            "single_neuron_masking_summary.csv"
        ),
    )

    parser.add_argument(
        "--shared-selected-summary",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "06_shared_neuron_masking/"
            "shared_masking_summary.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "llama31_8b_neuron_analysis/"
            "07_random_matched_control"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-random-seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        default=[
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
    )

    parser.add_argument(
        "--degrees",
        nargs="+",
        type=int,
        default=list(
            range(
                1,
                11
            )
        ),
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # MODEL
    # ============================================================

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            args.model_id
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
    )

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad_(
            False
        )

    print(
        "MLP width:",
        get_mlp_width(
            model
        )
    )

    # ============================================================
    # DATA
    # ============================================================

    df, dataset_labels = (
        prepare_dataframe(
            args.csv_path
        )
    )

    y = (
        df[
            "label_id"
        ]
        .to_numpy()
    )

    _, _, test_idx = (
        stratified_split(
            y,
            args.seed,
        )
    )

    test_texts = (
        df.iloc[
            test_idx
        ]["text"]
        .tolist()
    )

    test_labels = (
        y[
            test_idx
        ]
    )

    print(
        "Test samples:",
        len(test_idx)
    )

    # ============================================================
    # PROBE
    # ============================================================

    probe_data = torch.load(
        args.probe_path,
        map_location="cpu",
    )

    probe_labels = list(
        probe_data[
            "labels"
        ]
    )

    if (
        probe_labels
        != dataset_labels
    ):

        raise RuntimeError(
            "Probe labels do not "
            "match dataset labels."
        )

    hidden_size = int(
        probe_data[
            "hidden_size"
        ]
    )

    probe_device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    classifier = torch.nn.Linear(
        hidden_size,
        len(probe_labels),
    ).to(
        probe_device
    )

    classifier.load_state_dict(
        probe_data[
            "state_dict"
        ]
    )

    classifier.eval()

    for parameter in (
        classifier.parameters()
    ):

        parameter.requires_grad_(
            False
        )

    # ============================================================
    # NEURON TABLES
    # ============================================================

    (
        single_df,
        shared_df,
    ) = load_neuron_tables(
        args.single_neuron_csv,
        args.shared_neuron_csv,
    )

    # ============================================================
    # EXISTING EMOTION-SENSITIVE RESULTS
    # ============================================================

    selected_results = (
        load_selected_results(
            args.single_selected_summary,
            args.shared_selected_summary,
        )
    )

    # ============================================================
    # BASELINE
    # ============================================================

    print()
    print("=" * 80)
    print("BASELINE")
    print("=" * 80)

    baseline = evaluate(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        texts=test_texts,
        labels=test_labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(
        "Accuracy:",
        baseline[
            "accuracy"
        ]
    )

    print(
        "Macro-F1:",
        baseline[
            "macro_f1"
        ]
    )

    print(
        "True probability:",
        baseline[
            "mean_true_class_probability"
        ]
    )

    # ============================================================
    # RANDOM RUNS
    # ============================================================

    random_rows = []
    per_emotion_rows = []

    full_results = {
        "baseline": baseline,
        "random_runs": {},
    }

    for percentage in (
        args.percentages
    ):

        # --------------------------------------------------------
        # IMPORTANT:
        # Exclude ALL emotion-associated neurons at this
        # percentage from the random pool.
        # --------------------------------------------------------

        excluded_by_layer = (
            get_all_selected_at_percentage(
                single_df,
                shared_df,
                percentage,
            )
        )

        for degree in (
            args.degrees
        ):

            condition = (
                percentage,
                degree,
            )

            if (
                condition
                not in selected_results
            ):

                print(
                    "No existing selected "
                    "result for",
                    condition,
                    "- skipping"
                )

                continue

            selected_by_layer = (
                get_selected_condition(
                    single_df,
                    shared_df,
                    percentage,
                    degree,
                )
            )

            if not selected_by_layer:

                continue

            selected_count = sum(
                len(x)
                for x
                in selected_by_layer.values()
            )

            expected_count = (
                selected_results[
                    condition
                ][
                    "num_masked_neurons"
                ]
            )

            if (
                selected_count
                != expected_count
            ):

                raise RuntimeError(
                    f"Neuron count mismatch "
                    f"for {percentage}% "
                    f"degree {degree}: "
                    f"{selected_count} vs "
                    f"{expected_count}"
                )

            print()
            print("=" * 80)

            print(
                f"TOP {percentage:g}% "
                f"| DEGREE {degree}"
            )

            print(
                "Selected neuron count:",
                selected_count
            )

            print(
                "Selected Δ Macro-F1:",
                selected_results[
                    condition
                ][
                    "delta_macro_f1"
                ]
            )

            print("=" * 80)

            condition_key = (
                f"{percentage:g}%_"
                f"degree_{degree}"
            )

            full_results[
                "random_runs"
            ][condition_key] = []

            for random_index in range(
                args.num_random_seeds
            ):

                random_seed = (
                    args.seed
                    + 100000
                    + int(
                        percentage
                        * 1000
                    )
                    + degree
                    * 1000
                    + random_index
                )

                random_by_layer = (
                    sample_random_matched_neurons(
                        model=model,
                        selected_by_layer=
                        selected_by_layer,
                        excluded_by_layer=
                        excluded_by_layer,
                        seed=random_seed,
                    )
                )

                random_count = sum(
                    len(x)
                    for x
                    in random_by_layer.values()
                )

                assert (
                    random_count
                    ==
                    selected_count
                )

                # Exact same count per layer
                for layer in (
                    selected_by_layer
                ):

                    assert (
                        len(
                            random_by_layer[
                                layer
                            ]
                        )
                        ==
                        len(
                            selected_by_layer[
                                layer
                            ]
                        )
                    )

                controller = (
                    NeuronMaskController()
                )

                handles = (
                    install_mask_hooks(
                        model,
                        random_by_layer,
                        controller,
                    )
                )

                try:

                    metrics = evaluate(
                        model=model,
                        tokenizer=tokenizer,
                        classifier=classifier,
                        texts=test_texts,
                        labels=test_labels,
                        batch_size=args.batch_size,
                        max_length=args.max_length,
                        controller=controller,
                    )

                finally:

                    remove_hooks(
                        handles
                    )

                delta_accuracy = (
                    metrics[
                        "accuracy"
                    ]
                    - baseline[
                        "accuracy"
                    ]
                )

                delta_macro_f1 = (
                    metrics[
                        "macro_f1"
                    ]
                    - baseline[
                        "macro_f1"
                    ]
                )

                delta_true_prob = (
                    metrics[
                        "mean_true_class_probability"
                    ]
                    - baseline[
                        "mean_true_class_probability"
                    ]
                )

                print(
                    f"seed {random_index + 1:02d}: "
                    f"ΔF1={delta_macro_f1:+.6f} "
                    f"ΔAcc={delta_accuracy:+.6f}"
                )

                row = {
                    "percentage":
                    percentage,

                    "masked_degree":
                    degree,

                    "random_seed_index":
                    random_index,

                    "random_seed":
                    random_seed,

                    "num_masked_neurons":
                    random_count,

                    "baseline_accuracy":
                    baseline[
                        "accuracy"
                    ],

                    "random_accuracy":
                    metrics[
                        "accuracy"
                    ],

                    "delta_accuracy":
                    delta_accuracy,

                    "baseline_macro_f1":
                    baseline[
                        "macro_f1"
                    ],

                    "random_macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                    "delta_macro_f1":
                    delta_macro_f1,

                    "baseline_true_probability":
                    baseline[
                        "mean_true_class_probability"
                    ],

                    "random_true_probability":
                    metrics[
                        "mean_true_class_probability"
                    ],

                    "delta_true_probability":
                    delta_true_prob,
                }

                random_rows.append(
                    row
                )

                for (
                    emotion_index,
                    emotion
                ) in enumerate(
                    probe_labels
                ):

                    per_emotion_rows.append(
                        {
                            "percentage":
                            percentage,

                            "masked_degree":
                            degree,

                            "random_seed":
                            random_seed,

                            "emotion":
                            emotion,

                            "num_masked_neurons":
                            random_count,

                            "baseline_f1":
                            baseline[
                                "f1"
                            ][
                                emotion_index
                            ],

                            "random_f1":
                            metrics[
                                "f1"
                            ][
                                emotion_index
                            ],

                            "delta_f1":
                            (
                                metrics[
                                    "f1"
                                ][
                                    emotion_index
                                ]
                                -
                                baseline[
                                    "f1"
                                ][
                                    emotion_index
                                ]
                            ),

                            "baseline_recall":
                            baseline[
                                "recall"
                            ][
                                emotion_index
                            ],

                            "random_recall":
                            metrics[
                                "recall"
                            ][
                                emotion_index
                            ],

                            "delta_recall":
                            (
                                metrics[
                                    "recall"
                                ][
                                    emotion_index
                                ]
                                -
                                baseline[
                                    "recall"
                                ][
                                    emotion_index
                                ]
                            ),

                            "baseline_true_probability":
                            baseline[
                                "per_class_true_probability"
                            ][
                                emotion_index
                            ],

                            "random_true_probability":
                            metrics[
                                "per_class_true_probability"
                            ][
                                emotion_index
                            ],

                            "delta_true_probability":
                            (
                                metrics[
                                    "per_class_true_probability"
                                ][
                                    emotion_index
                                ]
                                -
                                baseline[
                                    "per_class_true_probability"
                                ][
                                    emotion_index
                                ]
                            ),
                        }
                    )

                full_results[
                    "random_runs"
                ][condition_key].append(
                    {
                        "seed":
                        random_seed,

                        "metrics":
                        metrics,

                        "delta_accuracy":
                        delta_accuracy,

                        "delta_macro_f1":
                        delta_macro_f1,

                        "delta_true_probability":
                        delta_true_prob,
                    }
                )

                # Save continuously
                write_csv(
                    output_dir
                    / "random_masking_runs.csv",
                    random_rows,
                )

                write_csv(
                    output_dir
                    / "random_masking_per_emotion.csv",
                    per_emotion_rows,
                )

    # ============================================================
    # AGGREGATE RANDOM RESULTS
    # ============================================================

    random_df = pd.DataFrame(
        random_rows
    )

    aggregate_rows = []

    for (
        percentage,
        degree
    ), group in random_df.groupby(
        [
            "percentage",
            "masked_degree",
        ]
    ):

        condition = (
            float(percentage),
            int(degree),
        )

        selected = (
            selected_results[
                condition
            ]
        )

        random_f1 = (
            group[
                "delta_macro_f1"
            ]
            .to_numpy(
                dtype=float
            )
        )

        random_acc = (
            group[
                "delta_accuracy"
            ]
            .to_numpy(
                dtype=float
            )
        )

        random_prob = (
            group[
                "delta_true_probability"
            ]
            .to_numpy(
                dtype=float
            )
        )

        selected_delta_f1 = (
            selected[
                "delta_macro_f1"
            ]
        )

        # --------------------------------------------------------
        # One-sided empirical probability:
        #
        # How often random masking caused an effect
        # at least as damaging (as negative) as selected masking.
        #
        # Smaller = selected set is unusually damaging.
        # --------------------------------------------------------

        empirical_p = (
            1
            +
            np.sum(
                random_f1
                <=
                selected_delta_f1
            )
        ) / (
            len(
                random_f1
            )
            + 1
        )

        random_mean = float(
            np.mean(
                random_f1
            )
        )

        random_std = float(
            np.std(
                random_f1,
                ddof=1,
            )
        ) if (
            len(
                random_f1
            ) > 1
        ) else 0.0

        if random_std > 0:

            z_score = (
                selected_delta_f1
                - random_mean
            ) / random_std

        else:

            z_score = float(
                "nan"
            )

        aggregate_rows.append(
            {
                "percentage":
                percentage,

                "masked_degree":
                degree,

                "num_masked_neurons":
                selected[
                    "num_masked_neurons"
                ],

                "num_random_runs":
                len(
                    group
                ),

                # Selected emotion neurons
                "selected_delta_macro_f1":
                selected_delta_f1,

                # Random distribution
                "random_mean_delta_macro_f1":
                random_mean,

                "random_std_delta_macro_f1":
                random_std,

                "random_min_delta_macro_f1":
                float(
                    np.min(
                        random_f1
                    )
                ),

                "random_max_delta_macro_f1":
                float(
                    np.max(
                        random_f1
                    )
                ),

                # Negative means selected neurons
                # caused MORE damage than random.
                "selected_minus_random_mean_f1":
                (
                    selected_delta_f1
                    -
                    random_mean
                ),

                "f1_z_score":
                z_score,

                "empirical_p_lower_tail":
                float(
                    empirical_p
                ),

                "selected_delta_accuracy":
                selected[
                    "delta_accuracy"
                ],

                "random_mean_delta_accuracy":
                float(
                    np.mean(
                        random_acc
                    )
                ),

                "selected_delta_true_probability":
                selected[
                    "delta_true_probability"
                ],

                "random_mean_delta_true_probability":
                float(
                    np.mean(
                        random_prob
                    )
                ),
            }
        )

    # ============================================================
    # SAVE
    # ============================================================

    write_csv(
        output_dir
        / "random_masking_runs.csv",
        random_rows,
    )

    write_csv(
        output_dir
        / "random_masking_per_emotion.csv",
        per_emotion_rows,
    )

    write_csv(
        output_dir
        / "random_masking_aggregate.csv",
        aggregate_rows,
    )

    (
        output_dir
        / "random_masking_results.json"
    ).write_text(
        json.dumps(
            full_results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ============================================================
    # PRINT FINAL COMPARISON
    # ============================================================

    print()
    print("=" * 100)

    print(
        "FINAL SELECTED vs RANDOM COMPARISON"
    )

    print("=" * 100)

    for row in aggregate_rows:

        print(
            f"{row['percentage']:>4g}% "
            f"d={int(row['masked_degree']):>2} "
            f"n={int(row['num_masked_neurons']):>6} | "
            f"selected ΔF1="
            f"{row['selected_delta_macro_f1']:+.5f} | "
            f"random="
            f"{row['random_mean_delta_macro_f1']:+.5f}"
            f" ± "
            f"{row['random_std_delta_macro_f1']:.5f} | "
            f"diff="
            f"{row['selected_minus_random_mean_f1']:+.5f}"
        )

    print()
    print(
        "Saved to:",
        output_dir
    )


if __name__ == "__main__":

    main()
