#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "mlt_hinghlish_dataset_clean.csv"
)

PROBE_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_neuron_analysis"
    / "04_probe"
    / "probe.pt"
)

# This must be produced by the degree-1 analysis script.
SINGLE_EMOTION_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "single_emotion_analysis"
    / "single_emotion_neurons.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "09_layerwise_unique_masking"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

TEXT_COLUMN = "hinglish_genz_text"
LABEL_COLUMN = "label"

MAX_LENGTH = 256
BATCH_SIZE = 4

TEST_SIZE = 0.15
SPLIT_SEED = 42

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

LABEL_TO_ID = {
    emotion: i
    for i, emotion in enumerate(EMOTIONS)
}

NUM_CLASSES = len(EMOTIONS)


# ============================================================
# FILE CHECKS
# ============================================================

for path in [
    DATA_FILE,
    PROBE_FILE,
    SINGLE_EMOTION_FILE,
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    MODEL_DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

else:

    DEVICE = torch.device("cpu")
    MODEL_DTYPE = torch.float32


print("=" * 80)
print("SARVAM-1 LAYER-WISE DEGREE-1 NEURON MASKING")
print("=" * 80)

print(f"Device : {DEVICE}")
print(f"Dtype  : {MODEL_DTYPE}")


# ============================================================
# DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

df = df[
    [
        TEXT_COLUMN,
        LABEL_COLUMN,
    ]
].copy()

df = df.dropna()

df[TEXT_COLUMN] = (
    df[TEXT_COLUMN]
    .astype(str)
    .str.strip()
)

df[LABEL_COLUMN] = (
    df[LABEL_COLUMN]
    .astype(str)
    .str.strip()
)

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)

unknown = (
    set(df[LABEL_COLUMN].unique())
    - set(EMOTIONS)
)

if unknown:
    raise ValueError(
        f"Unknown labels: {unknown}"
    )

print(f"Dataset rows: {len(df):,}")


# ============================================================
# RECREATE HELD-OUT TEST SET
# ============================================================
#
# This reproduces the 15% stratified test split.
# Same seed as the original probe experiment.
# ============================================================

_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=df[LABEL_COLUMN],
)

test_df = test_df.reset_index(drop=True)

print(f"Test samples: {len(test_df):,}")

if len(test_df) != 3827:
    print(
        "WARNING: expected 3,827 test examples "
        f"from the previous probe run; got {len(test_df)}."
    )


# ============================================================
# LOAD SINGLE-EMOTION / DEGREE-1 NEURONS
# ============================================================

print()
print("=" * 80)
print("LOADING DEGREE-1 NEURONS")
print("=" * 80)

single_df = pd.read_csv(
    SINGLE_EMOTION_FILE
)

required = {
    "layer",
    "layer_index",
    "emotion",
    "neuron_index",
}

missing = required - set(single_df.columns)

if missing:
    raise ValueError(
        f"Missing columns in degree-1 file: {missing}"
    )


# Safety: every row should be degree 1 if column exists.
if "num_emotions" in single_df.columns:

    if not (
        single_df["num_emotions"] == 1
    ).all():

        raise ValueError(
            "Input file contains neurons whose "
            "num_emotions != 1."
        )


# Map:
#
# layer_index -> [neuron IDs]
#
neurons_by_layer = {}

for layer_idx in range(32):

    ids = (
        single_df.loc[
            single_df["layer_index"] == layer_idx,
            "neuron_index"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    neurons_by_layer[layer_idx] = sorted(ids)

    print(
        f"Layer {layer_idx + 1:02d}: "
        f"{len(ids):4d} exact-degree-1 neurons"
    )


print()
print(
    "Total degree-1 physical neurons:",
    sum(
        len(v)
        for v in neurons_by_layer.values()
    )
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# SARVAM-1
# ============================================================

print()
print("=" * 80)
print("LOADING SARVAM-1")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=MODEL_DTYPE,
    trust_remote_code=True,
)

model.to(DEVICE)

model.eval()

for parameter in model.parameters():
    parameter.requires_grad_(False)


# Sarvam model used in your extraction.
layers = model.model.layers

if len(layers) != 32:
    raise ValueError(
        f"Expected 32 layers; found {len(layers)}"
    )

for i, layer in enumerate(layers):

    if not hasattr(layer.mlp, "down_proj"):
        raise RuntimeError(
            f"Layer {i} missing mlp.down_proj"
        )


# Base model avoids computing vocabulary logits.
backbone = model.model


# ============================================================
# VALID TOKEN MASK
# ============================================================

def create_valid_token_mask(
    input_ids,
    attention_mask,
):
    """
    True only for real non-special content tokens.
    """

    valid = attention_mask.bool().clone()

    for token_id in tokenizer.all_special_ids:

        valid &= input_ids.ne(
            token_id
        )

    return valid


# ============================================================
# LOAD LINEAR PROBE
# ============================================================

def load_probe(path: Path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------
    # Case 1: complete nn.Module saved directly
    # --------------------------------------------

    if isinstance(checkpoint, nn.Module):

        probe = checkpoint

        probe.to(DEVICE)
        probe.eval()

        for p in probe.parameters():
            p.requires_grad_(False)

        return probe


    # --------------------------------------------
    # Case 2: state_dict stored in dictionary
    # --------------------------------------------

    if not isinstance(checkpoint, dict):

        raise TypeError(
            f"Unsupported probe checkpoint type: "
            f"{type(checkpoint)}"
        )


    state = None

    for key in [
        "state_dict",
        "model_state_dict",
        "probe_state_dict",
    ]:

        if key in checkpoint:

            state = checkpoint[key]
            break


    # Maybe the checkpoint itself IS the state dict.
    if state is None:

        state = checkpoint


    # Remove DataParallel prefix if present.
    clean_state = {}

    for key, value in state.items():

        if not torch.is_tensor(value):
            continue

        if key.startswith("module."):
            key = key[len("module."):]

        clean_state[key] = value


    # Find a classifier matrix shaped:
    #
    # [10, hidden_dim]
    #
    candidates = []

    for key, value in clean_state.items():

        if (
            value.ndim == 2
            and value.shape[0] == NUM_CLASSES
        ):

            candidates.append(
                (key, value)
            )


    if not candidates:

        raise RuntimeError(
            "Could not find a 10-class linear "
            "weight matrix in probe.pt."
        )


    # Prefer obvious classifier names.
    candidates.sort(
        key=lambda item: (
            0 if any(
                name in item[0].lower()
                for name in [
                    "classifier",
                    "linear",
                    "probe",
                    "weight",
                ]
            )
            else 1
        )
    )

    weight_key, weight = candidates[0]

    hidden_dim = int(
        weight.shape[1]
    )

    probe = nn.Linear(
        hidden_dim,
        NUM_CLASSES,
    )


    with torch.no_grad():

        probe.weight.copy_(
            weight.float()
        )

        prefix = (
            weight_key[:-len("weight")]
            if weight_key.endswith("weight")
            else ""
        )

        bias_key = (
            prefix + "bias"
        )

        if bias_key in clean_state:

            probe.bias.copy_(
                clean_state[
                    bias_key
                ].float()
            )

        else:

            probe.bias.zero_()


    probe.to(DEVICE)
    probe.eval()

    for p in probe.parameters():
        p.requires_grad_(False)

    print(
        f"Loaded linear probe: "
        f"{hidden_dim} -> {NUM_CLASSES}"
    )

    return probe


probe = load_probe(
    PROBE_FILE
)


# ============================================================
# MASK CONTEXT
# ============================================================

mask_context = {
    "valid_mask": None
}


# ============================================================
# CREATE MASK HOOK
# ============================================================

def create_mask_hook(
    neuron_indices,
):

    neuron_indices = list(
        map(int, neuron_indices)
    )

    def hook(
        module,
        inputs,
    ):

        if not inputs:
            return inputs

        x = inputs[0]

        if not neuron_indices:
            return inputs

        valid_mask = mask_context[
            "valid_mask"
        ]

        if valid_mask is None:
            raise RuntimeError(
                "Mask hook called without valid token mask."
            )


        idx = torch.tensor(
            neuron_indices,
            device=x.device,
            dtype=torch.long,
        )


        # Clone so original activation is not edited in place.
        masked = x.clone()


        # Selected dimensions:
        #
        # [batch, sequence, num_selected]
        selected_values = masked[
            :,
            :,
            idx
        ]


        # Zero ONLY content-token positions.
        #
        # Special/PAD tokens remain untouched.
        selected_values = (
            selected_values.masked_fill(
                valid_mask
                .to(x.device)
                .unsqueeze(-1),
                0.0,
            )
        )


        masked[
            :,
            :,
            idx
        ] = selected_values


        return (
            masked,
            *inputs[1:]
        )

    return hook


# ============================================================
# FORWARD + PROBE
# ============================================================

def run_evaluation(
    masked_layer_idx=None,
    masked_neurons=None,
):

    handle = None

    if (
        masked_layer_idx is not None
        and masked_neurons
    ):

        handle = (
            layers[
                masked_layer_idx
            ]
            .mlp
            .down_proj
            .register_forward_pre_hook(
                create_mask_hook(
                    masked_neurons
                )
            )
        )


    all_logits = []
    all_labels = []


    try:

        with torch.inference_mode():

            for start in tqdm(
                range(
                    0,
                    len(test_df),
                    BATCH_SIZE
                ),
                leave=False,
            ):

                batch = test_df.iloc[
                    start:
                    start + BATCH_SIZE
                ]


                texts = batch[
                    TEXT_COLUMN
                ].tolist()


                labels = torch.tensor(
                    [
                        LABEL_TO_ID[x]
                        for x in batch[
                            LABEL_COLUMN
                        ].tolist()
                    ],
                    dtype=torch.long,
                    device=DEVICE,
                )


                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )


                input_ids = encoded[
                    "input_ids"
                ].to(DEVICE)

                attention_mask = encoded[
                    "attention_mask"
                ].to(DEVICE)


                valid_mask = (
                    create_valid_token_mask(
                        input_ids,
                        attention_mask,
                    )
                )


                mask_context[
                    "valid_mask"
                ] = valid_mask


                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )


                hidden = (
                    outputs.last_hidden_state
                    if hasattr(
                        outputs,
                        "last_hidden_state"
                    )
                    else outputs[0]
                )


                # --------------------------------------------
                # Mean-pool only valid content tokens.
                # --------------------------------------------

                pool_mask = (
                    valid_mask
                    .unsqueeze(-1)
                    .to(hidden.dtype)
                )


                pooled = (
                    hidden * pool_mask
                ).sum(
                    dim=1
                )


                denominator = (
                    pool_mask
                    .sum(dim=1)
                    .clamp_min(1.0)
                )


                pooled = (
                    pooled
                    / denominator
                )


                # Linear probe normally operates float32.
                logits = probe(
                    pooled.float()
                )


                all_logits.append(
                    logits.cpu()
                )

                all_labels.append(
                    labels.cpu()
                )


    finally:

        if handle is not None:
            handle.remove()

        mask_context[
            "valid_mask"
        ] = None


    logits = torch.cat(
        all_logits,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    return logits, labels


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    labels,
    baseline_predictions=None,
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = logits.argmax(
        dim=1
    )


    y = labels.numpy()

    pred = predictions.numpy()


    accuracy = accuracy_score(
        y,
        pred
    )

    macro_f1 = f1_score(
        y,
        pred,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        y,
        pred,
        average="micro",
        zero_division=0,
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y,
            pred,
            labels=np.arange(
                NUM_CLASSES
            ),
            zero_division=0,
        )
    )


    row_idx = torch.arange(
        len(labels)
    )


    # Probability of ground-truth class.
    true_probability = probabilities[
        row_idx,
        labels
    ]


    # Maximum current prediction probability.
    max_confidence = (
        probabilities.max(
            dim=1
        ).values
    )


    result = {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "micro_f1": float(
            micro_f1
        ),

        "mean_true_class_probability":
            float(
                true_probability.mean()
            ),

        "mean_max_confidence":
            float(
                max_confidence.mean()
            ),

        "precision":
            precision.tolist(),

        "recall":
            recall.tolist(),

        "f1":
            f1.tolist(),

        "support":
            support.tolist(),

        "per_class_true_probability": [],
    }


    # --------------------------------------------
    # Probability of baseline prediction
    # --------------------------------------------

    if baseline_predictions is not None:

        baseline_prediction_probability = (
            probabilities[
                row_idx,
                baseline_predictions
            ]
        )

        result[
            "mean_baseline_prediction_probability"
        ] = float(
            baseline_prediction_probability.mean()
        )


    # --------------------------------------------
    # Per-class true-label probabilities
    # --------------------------------------------

    for class_id in range(
        NUM_CLASSES
    ):

        class_mask = (
            labels == class_id
        )

        value = (
            true_probability[
                class_mask
            ].mean()
        )

        result[
            "per_class_true_probability"
        ].append(
            float(value)
        )


    return (
        result,
        probabilities,
        predictions,
    )


# ============================================================
# BASELINE
# ============================================================

print()
print("=" * 80)
print("BASELINE — NO MASK")
print("=" * 80)

baseline_logits, labels = (
    run_evaluation()
)

(
    baseline_metrics,
    baseline_probabilities,
    baseline_predictions,
) = calculate_metrics(
    baseline_logits,
    labels,
)


print()
print(
    f"Baseline accuracy       : "
    f"{baseline_metrics['accuracy']:.6f}"
)

print(
    f"Baseline macro-F1       : "
    f"{baseline_metrics['macro_f1']:.6f}"
)

print(
    f"Baseline true-class prob: "
    f"{baseline_metrics['mean_true_class_probability']:.6f}"
)

print(
    f"Baseline max confidence : "
    f"{baseline_metrics['mean_max_confidence']:.6f}"
)


# ============================================================
# LAYER-WISE MASKING
# ============================================================

summary_rows = []
per_emotion_rows = []

results = {
    "experiment": (
        "Layer-wise masking of exact-degree-1 "
        "neurons from top-1% emotion selection"
    ),
    "baseline": baseline_metrics,
    "layers": {},
}


for layer_idx in range(32):

    layer_number = (
        layer_idx + 1
    )

    neuron_ids = (
        neurons_by_layer[
            layer_idx
        ]
    )


    print()
    print("=" * 80)
    print(
        f"MASKING LAYER {layer_number:02d}"
    )
    print("=" * 80)

    print(
        f"Neurons masked: "
        f"{len(neuron_ids)}"
    )


    if not neuron_ids:

        print(
            "No degree-1 neurons in this layer; skipping."
        )

        continue


    masked_logits, masked_labels = (
        run_evaluation(
            masked_layer_idx=layer_idx,
            masked_neurons=neuron_ids,
        )
    )


    (
        masked_metrics,
        masked_probabilities,
        masked_predictions,
    ) = calculate_metrics(
        masked_logits,
        masked_labels,
        baseline_predictions=
            baseline_predictions,
    )


    # ========================================================
    # DELTAS
    # ========================================================

    delta_accuracy = (
        masked_metrics["accuracy"]
        - baseline_metrics["accuracy"]
    )

    delta_macro_f1 = (
        masked_metrics["macro_f1"]
        - baseline_metrics["macro_f1"]
    )


    # Positive = confidence decreased.
    true_confidence_drop = (
        baseline_metrics[
            "mean_true_class_probability"
        ]
        - masked_metrics[
            "mean_true_class_probability"
        ]
    )


    max_confidence_drop = (
        baseline_metrics[
            "mean_max_confidence"
        ]
        - masked_metrics[
            "mean_max_confidence"
        ]
    )


    # Probability assigned, after masking,
    # to the class predicted by BASELINE.
    baseline_pred_prob_before = (
        baseline_probabilities[
            torch.arange(
                len(labels)
            ),
            baseline_predictions
        ]
        .mean()
        .item()
    )


    baseline_pred_prob_after = (
        masked_probabilities[
            torch.arange(
                len(labels)
            ),
            baseline_predictions
        ]
        .mean()
        .item()
    )


    baseline_prediction_confidence_drop = (
        baseline_pred_prob_before
        - baseline_pred_prob_after
    )


    # Percentage of samples whose predicted label changed.
    prediction_change_rate = (
        masked_predictions
        .ne(
            baseline_predictions
        )
        .float()
        .mean()
        .item()
    )


    summary_rows.append(
        {
            "layer": layer_number,
            "layer_index": layer_idx,
            "num_masked_neurons":
                len(neuron_ids),

            "baseline_accuracy":
                baseline_metrics["accuracy"],

            "masked_accuracy":
                masked_metrics["accuracy"],

            "delta_accuracy":
                delta_accuracy,

            "baseline_macro_f1":
                baseline_metrics["macro_f1"],

            "masked_macro_f1":
                masked_metrics["macro_f1"],

            "delta_macro_f1":
                delta_macro_f1,

            "baseline_true_probability":
                baseline_metrics[
                    "mean_true_class_probability"
                ],

            "masked_true_probability":
                masked_metrics[
                    "mean_true_class_probability"
                ],

            # Positive means drop.
            "true_confidence_drop":
                true_confidence_drop,

            "baseline_max_confidence":
                baseline_metrics[
                    "mean_max_confidence"
                ],

            "masked_max_confidence":
                masked_metrics[
                    "mean_max_confidence"
                ],

            "max_confidence_drop":
                max_confidence_drop,

            "baseline_prediction_probability_before":
                baseline_pred_prob_before,

            "baseline_prediction_probability_after":
                baseline_pred_prob_after,

            "baseline_prediction_confidence_drop":
                baseline_prediction_confidence_drop,

            "prediction_change_rate":
                prediction_change_rate,
        }
    )


    # ========================================================
    # PER-EMOTION EFFECTS
    # ========================================================

    for emotion_idx, emotion in enumerate(
        EMOTIONS
    ):

        base_f1 = (
            baseline_metrics["f1"][
                emotion_idx
            ]
        )

        masked_f1 = (
            masked_metrics["f1"][
                emotion_idx
            ]
        )


        base_recall = (
            baseline_metrics["recall"][
                emotion_idx
            ]
        )

        masked_recall = (
            masked_metrics["recall"][
                emotion_idx
            ]
        )


        base_prob = (
            baseline_metrics[
                "per_class_true_probability"
            ][emotion_idx]
        )

        masked_prob = (
            masked_metrics[
                "per_class_true_probability"
            ][emotion_idx]
        )


        per_emotion_rows.append(
            {
                "layer": layer_number,
                "layer_index": layer_idx,
                "emotion": emotion,
                "num_masked_neurons":
                    len(neuron_ids),

                "baseline_f1":
                    base_f1,

                "masked_f1":
                    masked_f1,

                "delta_f1":
                    masked_f1
                    - base_f1,

                "baseline_recall":
                    base_recall,

                "masked_recall":
                    masked_recall,

                "delta_recall":
                    masked_recall
                    - base_recall,

                "baseline_true_probability":
                    base_prob,

                "masked_true_probability":
                    masked_prob,

                # Positive = confidence loss.
                "true_confidence_drop":
                    base_prob
                    - masked_prob,
            }
        )


    results[
        "layers"
    ][str(layer_number)] = {
        "num_masked_neurons":
            len(neuron_ids),

        "metrics":
            masked_metrics,

        "delta_accuracy":
            float(delta_accuracy),

        "delta_macro_f1":
            float(delta_macro_f1),

        "true_confidence_drop":
            float(true_confidence_drop),

        "max_confidence_drop":
            float(max_confidence_drop),

        "baseline_prediction_confidence_drop":
            float(
                baseline_prediction_confidence_drop
            ),

        "prediction_change_rate":
            float(
                prediction_change_rate
            ),
    }


    print(
        f"Accuracy    : "
        f"{masked_metrics['accuracy']:.6f} "
        f"(Δ {delta_accuracy:+.6f})"
    )

    print(
        f"Macro-F1    : "
        f"{masked_metrics['macro_f1']:.6f} "
        f"(Δ {delta_macro_f1:+.6f})"
    )

    print(
        f"True-P drop : "
        f"{true_confidence_drop:+.6f}"
    )

    print(
        f"Base-pred confidence drop: "
        f"{baseline_prediction_confidence_drop:+.6f}"
    )

    print(
        f"Prediction change rate: "
        f"{prediction_change_rate:.4%}"
    )


# ============================================================
# SAVE
# ============================================================

summary_df = pd.DataFrame(
    summary_rows
)

per_emotion_df = pd.DataFrame(
    per_emotion_rows
)


summary_file = (
    OUTPUT_DIR
    / "layerwise_masking_summary.csv"
)

per_emotion_file = (
    OUTPUT_DIR
    / "layerwise_masking_per_emotion.csv"
)

json_file = (
    OUTPUT_DIR
    / "layerwise_masking_results.json"
)


summary_df.to_csv(
    summary_file,
    index=False,
)

per_emotion_df.to_csv(
    per_emotion_file,
    index=False,
)


with json_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=2,
    )


# ============================================================
# RANK LAYERS BY DAMAGE
# ============================================================

ranked = (
    summary_df
    .sort_values(
        "delta_macro_f1",
        ascending=True,
    )
)


print()
print("=" * 80)
print("LAYERS RANKED BY MACRO-F1 DAMAGE")
print("=" * 80)

print(
    ranked[
        [
            "layer",
            "num_masked_neurons",
            "delta_macro_f1",
            "true_confidence_drop",
            "baseline_prediction_confidence_drop",
            "prediction_change_rate",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(summary_file)
print(per_emotion_file)
print(json_file)
