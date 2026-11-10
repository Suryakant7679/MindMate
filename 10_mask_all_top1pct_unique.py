#!/usr/bin/env python3

"""
Mask ALL exact-degree-1 top-1% emotion-selective neurons
across ALL 28 Sarvam-1 transformer layers simultaneously.

Comparison:
    Baseline: no masking
    Intervention: all degree-1 neurons masked in their own layers

Outputs:
    overall metrics
    per-emotion metrics
    per-sample confidence changes
"""

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
    confusion_matrix,
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
    / "10_all_unique_masking"
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

NUM_CLASSES = len(EMOTIONS)

LABEL_TO_ID = {
    emotion: idx
    for idx, emotion in enumerate(EMOTIONS)
}


# ============================================================
# CHECK FILES
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
print("MASK ALL TOP-1% EXACT-DEGREE-1 NEURONS AT ONCE")
print("=" * 80)

print(f"Device : {DEVICE}")
print(f"Dtype  : {MODEL_DTYPE}")


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

df = pd.read_csv(
    DATA_FILE
)

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


unknown_labels = (
    set(df[LABEL_COLUMN].unique())
    - set(EMOTIONS)
)

if unknown_labels:

    raise ValueError(
        f"Unknown labels: {unknown_labels}"
    )


print(
    f"Dataset rows: {len(df):,}"
)


# ============================================================
# SAME TEST SPLIT AS LAYER-WISE EXPERIMENT
# ============================================================

_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=df[LABEL_COLUMN],
)

test_df = (
    test_df
    .reset_index(drop=True)
)


print(
    f"Test samples: {len(test_df):,}"
)


if len(test_df) != 3827:

    print(
        "WARNING: expected 3,827 test samples "
        f"but obtained {len(test_df):,}"
    )


# ============================================================
# LOAD DEGREE-1 NEURONS
# ============================================================

print()
print("=" * 80)
print("LOADING EXACT-DEGREE-1 NEURONS")
print("=" * 80)

single_df = pd.read_csv(
    SINGLE_EMOTION_FILE
)


required_columns = {
    "layer",
    "layer_index",
    "emotion",
    "neuron_index",
}

missing = (
    required_columns
    - set(single_df.columns)
)

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


if "num_emotions" in single_df.columns:

    if not (
        single_df["num_emotions"] == 1
    ).all():

        raise ValueError(
            "Input contains neurons with "
            "num_emotions != 1"
        )


# ============================================================
# CREATE:
#
# layer_index -> unique neuron IDs
# ============================================================

neurons_by_layer = {}

total_masked = 0


for layer_idx in range(32):

    neuron_ids = (
        single_df.loc[
            single_df["layer_index"]
            == layer_idx,
            "neuron_index",
        ]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    neurons_by_layer[
        layer_idx
    ] = neuron_ids

    total_masked += len(
        neuron_ids
    )

    print(
        f"Layer {layer_idx + 1:02d}: "
        f"{len(neuron_ids):4d} neurons"
    )


print()
print(
    f"TOTAL neurons masked across all layers: "
    f"{total_masked:,}"
)


# ============================================================
# SANITY CHECK PHYSICAL UNIQUENESS
# ============================================================

physical_neurons = set()

for layer_idx, neuron_ids in (
    neurons_by_layer.items()
):

    for neuron_idx in neuron_ids:

        physical_neurons.add(
            (
                layer_idx,
                neuron_idx,
            )
        )


if len(physical_neurons) != total_masked:

    raise RuntimeError(
        "Duplicate physical neuron identities found."
    )


print(
    f"Unique physical neurons: "
    f"{len(physical_neurons):,}"
)


# ============================================================
# EMOTION COMPOSITION
# ============================================================

emotion_counts = (
    single_df
    .groupby("emotion")
    .size()
    .reindex(
        EMOTIONS,
        fill_value=0,
    )
)


print()
print("Degree-1 neurons by emotion:")

for emotion in EMOTIONS:

    print(
        f"{emotion:12s}: "
        f"{int(emotion_counts[emotion]):4d}"
    )


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# LOAD SARVAM
# ============================================================

print()
print("=" * 80)
print("LOADING SARVAM-1")
print("=" * 80)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_ID,
        torch_dtype=MODEL_DTYPE,
        trust_remote_code=True,
    )
)

model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad_(
        False
    )


layers = model.model.layers


if len(layers) != 32:

    raise ValueError(
        f"Expected 32 layers, "
        f"found {len(layers)}"
    )


for layer_idx, layer in enumerate(
    layers
):

    if not hasattr(
        layer.mlp,
        "down_proj",
    ):

        raise RuntimeError(
            f"Layer {layer_idx} "
            "does not have mlp.down_proj"
        )


# Avoid vocabulary projection.
backbone = model.model


# ============================================================
# LOAD PROBE
# ============================================================

def load_probe(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


    # --------------------------------------------------------
    # Saved whole module
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        nn.Module,
    ):

        probe = checkpoint

        probe.to(
            DEVICE
        )

        probe.eval()

        for p in probe.parameters():

            p.requires_grad_(
                False
            )

        return probe


    # --------------------------------------------------------
    # State dictionary
    # --------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            f"Unsupported probe type: "
            f"{type(checkpoint)}"
        )


    state = None


    for key in [
        "state_dict",
        "model_state_dict",
        "probe_state_dict",
    ]:

        if key in checkpoint:

            state = checkpoint[
                key
            ]

            break


    if state is None:

        state = checkpoint


    clean_state = {}

    for key, value in state.items():

        if not torch.is_tensor(
            value
        ):
            continue

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        clean_state[
            key
        ] = value


    candidates = []

    for key, value in (
        clean_state.items()
    ):

        if (
            value.ndim == 2
            and value.shape[0]
            == NUM_CLASSES
        ):

            candidates.append(
                (
                    key,
                    value,
                )
            )


    if not candidates:

        raise RuntimeError(
            "Could not find "
            "10-class linear probe weights."
        )


    candidates.sort(
        key=lambda x:
        (
            0
            if any(
                name in x[0].lower()
                for name in [
                    "classifier",
                    "probe",
                    "linear",
                    "weight",
                ]
            )
            else 1
        )
    )


    weight_key, weight = (
        candidates[0]
    )


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


        if weight_key.endswith(
            "weight"
        ):

            prefix = (
                weight_key[
                    : -len("weight")
                ]
            )

        else:

            prefix = ""


        bias_key = (
            prefix
            + "bias"
        )


        if bias_key in clean_state:

            probe.bias.copy_(
                clean_state[
                    bias_key
                ].float()
            )

        else:

            probe.bias.zero_()


    probe.to(
        DEVICE
    )

    probe.eval()


    for p in probe.parameters():

        p.requires_grad_(
            False
        )


    print(
        f"Probe: "
        f"{hidden_dim} -> {NUM_CLASSES}"
    )

    return probe


probe = load_probe(
    PROBE_FILE
)


# ============================================================
# TOKEN MASK
# ============================================================

def create_valid_token_mask(
    input_ids,
    attention_mask,
):

    valid_mask = (
        attention_mask
        .bool()
        .clone()
    )


    for token_id in (
        tokenizer.all_special_ids
    ):

        valid_mask &= (
            input_ids
            .ne(token_id)
        )


    return valid_mask


# ============================================================
# GLOBAL BATCH CONTEXT
# ============================================================

mask_context = {
    "valid_mask": None
}


# ============================================================
# HOOK FACTORY
# ============================================================

def create_mask_hook(
    neuron_indices,
):

    neuron_indices = [
        int(x)
        for x in neuron_indices
    ]


    def hook(
        module,
        inputs,
    ):

        if not inputs:

            return inputs


        if not neuron_indices:

            return inputs


        x = inputs[0]


        valid_mask = (
            mask_context[
                "valid_mask"
            ]
        )


        if valid_mask is None:

            raise RuntimeError(
                "Mask hook called without "
                "valid token mask."
            )


        indices = torch.tensor(
            neuron_indices,
            dtype=torch.long,
            device=x.device,
        )


        # Clone MLP intermediate.
        masked = x.clone()


        # Selected dimensions:
        #
        # [batch, sequence, selected neurons]
        values = masked[
            :,
            :,
            indices
        ]


        # Mask selected neurons ONLY
        # on valid content tokens.
        values = (
            values
            .masked_fill(
                valid_mask
                .to(x.device)
                .unsqueeze(-1),
                0.0,
            )
        )


        masked[
            :,
            :,
            indices
        ] = values


        return (
            masked,
            *inputs[1:],
        )


    return hook


# ============================================================
# RUN MODEL
# ============================================================

def run_evaluation(
    mask_all=False,
):

    handles = []


    # ========================================================
    # IMPORTANT:
    #
    # Register ALL 28 masks BEFORE inference.
    #
    # This means all selected neurons in all layers
    # are simultaneously zeroed in the SAME forward pass.
    # ========================================================

    if mask_all:

        for layer_idx in range(
            28
        ):

            neuron_ids = (
                neurons_by_layer[
                    layer_idx
                ]
            )


            if not neuron_ids:

                continue


            handle = (
                layers[
                    layer_idx
                ]
                .mlp
                .down_proj
                .register_forward_pre_hook(
                    create_mask_hook(
                        neuron_ids
                    )
                )
            )


            handles.append(
                handle
            )


        print(
            f"Registered "
            f"{len(handles)} layer hooks."
        )


    all_logits = []
    all_labels = []


    try:

        with torch.inference_mode():

            for start in tqdm(
                range(
                    0,
                    len(test_df),
                    BATCH_SIZE,
                ),
                desc=(
                    "ALL-MASK"
                    if mask_all
                    else "BASELINE"
                ),
            ):

                batch = (
                    test_df.iloc[
                        start:
                        start + BATCH_SIZE
                    ]
                )


                texts = (
                    batch[
                        TEXT_COLUMN
                    ]
                    .tolist()
                )


                labels = torch.tensor(
                    [
                        LABEL_TO_ID[label]
                        for label in (
                            batch[
                                LABEL_COLUMN
                            ]
                            .tolist()
                        )
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


                input_ids = (
                    encoded[
                        "input_ids"
                    ]
                    .to(DEVICE)
                )


                attention_mask = (
                    encoded[
                        "attention_mask"
                    ]
                    .to(DEVICE)
                )


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


                if hasattr(
                    outputs,
                    "last_hidden_state",
                ):

                    hidden = (
                        outputs
                        .last_hidden_state
                    )

                else:

                    hidden = outputs[0]


                # =============================================
                # SAME MEAN POOLING
                # =============================================

                pool_mask = (
                    valid_mask
                    .unsqueeze(-1)
                    .to(hidden.dtype)
                )


                pooled = (
                    hidden
                    * pool_mask
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

        # Remove ALL hooks after masked evaluation.
        for handle in handles:

            handle.remove()


        mask_context[
            "valid_mask"
        ] = None


    logits = torch.cat(
        all_logits,
        dim=0,
    )


    labels = torch.cat(
        all_labels,
        dim=0,
    )


    return (
        logits,
        labels,
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def compute_metrics(
    logits,
    labels,
):

    probabilities = (
        torch.softmax(
            logits,
            dim=1,
        )
    )


    predictions = (
        probabilities
        .argmax(dim=1)
    )


    y_true = (
        labels.numpy()
    )

    y_pred = (
        predictions.numpy()
    )


    accuracy = (
        accuracy_score(
            y_true,
            y_pred,
        )
    )


    macro_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )


    micro_f1 = (
        f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0,
        )
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(
                NUM_CLASSES
            ),
            zero_division=0,
        )
    )


    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(
            NUM_CLASSES
        ),
    )


    row_indices = torch.arange(
        len(labels)
    )


    true_probabilities = (
        probabilities[
            row_indices,
            labels
        ]
    )


    max_probabilities = (
        probabilities
        .max(dim=1)
        .values
    )


    per_class_probability = []


    for class_idx in range(
        NUM_CLASSES
    ):

        mask = (
            labels
            == class_idx
        )


        per_class_probability.append(
            float(
                true_probabilities[
                    mask
                ].mean()
            )
        )


    metrics = {

        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "micro_f1":
            float(micro_f1),

        "mean_true_class_probability":
            float(
                true_probabilities.mean()
            ),

        "mean_max_confidence":
            float(
                max_probabilities.mean()
            ),

        "precision":
            precision.tolist(),

        "recall":
            recall.tolist(),

        "f1":
            f1.tolist(),

        "support":
            support.tolist(),

        "confusion_matrix":
            cm.tolist(),

        "per_class_true_probability":
            per_class_probability,
    }


    return (
        metrics,
        probabilities,
        predictions,
        true_probabilities,
        max_probabilities,
    )


# ============================================================
# BASELINE
# ============================================================

print()
print("=" * 80)
print("1. BASELINE — NO NEURON MASKING")
print("=" * 80)


baseline_logits, labels = (
    run_evaluation(
        mask_all=False
    )
)


(
    baseline_metrics,
    baseline_probs,
    baseline_predictions,
    baseline_true_probs,
    baseline_max_probs,
) = compute_metrics(
    baseline_logits,
    labels,
)


print()
print(
    f"Accuracy        : "
    f"{baseline_metrics['accuracy']:.6f}"
)

print(
    f"Macro-F1        : "
    f"{baseline_metrics['macro_f1']:.6f}"
)

print(
    f"True-class prob : "
    f"{baseline_metrics['mean_true_class_probability']:.6f}"
)

print(
    f"Max confidence  : "
    f"{baseline_metrics['mean_max_confidence']:.6f}"
)


# ============================================================
# MASK ALL DEGREE-1 NEURONS SIMULTANEOUSLY
# ============================================================

print()
print("=" * 80)
print(
    "2. MASKING ALL DEGREE-1 NEURONS "
    "IN ALL 28 LAYERS"
)
print("=" * 80)


masked_logits, masked_labels = (
    run_evaluation(
        mask_all=True
    )
)


if not torch.equal(
    labels,
    masked_labels,
):

    raise RuntimeError(
        "Baseline and masked labels differ."
    )


(
    masked_metrics,
    masked_probs,
    masked_predictions,
    masked_true_probs,
    masked_max_probs,
) = compute_metrics(
    masked_logits,
    masked_labels,
)


# ============================================================
# GLOBAL DELTAS
# ============================================================

delta_accuracy = (
    masked_metrics["accuracy"]
    - baseline_metrics["accuracy"]
)

delta_macro_f1 = (
    masked_metrics["macro_f1"]
    - baseline_metrics["macro_f1"]
)

delta_micro_f1 = (
    masked_metrics["micro_f1"]
    - baseline_metrics["micro_f1"]
)


# Positive = loss of confidence
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


# ============================================================
# BASELINE-PREDICTED CLASS CONFIDENCE
# ============================================================

row_indices = torch.arange(
    len(labels)
)


baseline_pred_prob_before = (
    baseline_probs[
        row_indices,
        baseline_predictions,
    ]
)


baseline_pred_prob_after = (
    masked_probs[
        row_indices,
        baseline_predictions,
    ]
)


baseline_prediction_confidence_drop = (
    baseline_pred_prob_before
    - baseline_pred_prob_after
)


mean_baseline_prediction_confidence_drop = float(
    baseline_prediction_confidence_drop.mean()
)


# ============================================================
# SAMPLE-LEVEL TRUE-CLASS CONFIDENCE DROP
# ============================================================

sample_true_confidence_drop = (
    baseline_true_probs
    - masked_true_probs
)


median_true_confidence_drop = float(
    sample_true_confidence_drop.median()
)


fraction_true_confidence_decreased = float(
    (
        sample_true_confidence_drop > 0
    )
    .float()
    .mean()
)


prediction_changed = (
    baseline_predictions
    != masked_predictions
)


prediction_change_rate = float(
    prediction_changed
    .float()
    .mean()
)


# ============================================================
# PER-EMOTION RESULT
# ============================================================

per_emotion_rows = []


for emotion_idx, emotion in enumerate(
    EMOTIONS
):

    baseline_f1 = (
        baseline_metrics[
            "f1"
        ][emotion_idx]
    )


    masked_f1 = (
        masked_metrics[
            "f1"
        ][emotion_idx]
    )


    baseline_recall = (
        baseline_metrics[
            "recall"
        ][emotion_idx]
    )


    masked_recall = (
        masked_metrics[
            "recall"
        ][emotion_idx]
    )


    baseline_prob = (
        baseline_metrics[
            "per_class_true_probability"
        ][emotion_idx]
    )


    masked_prob = (
        masked_metrics[
            "per_class_true_probability"
        ][emotion_idx]
    )


    class_mask = (
        labels
        == emotion_idx
    )


    class_prediction_change_rate = float(
        prediction_changed[
            class_mask
        ]
        .float()
        .mean()
    )


    per_emotion_rows.append(
        {
            "emotion":
                emotion,

            "num_unique_neurons":
                int(
                    emotion_counts[
                        emotion
                    ]
                ),

            "baseline_f1":
                baseline_f1,

            "masked_f1":
                masked_f1,

            "delta_f1":
                masked_f1
                - baseline_f1,

            "baseline_recall":
                baseline_recall,

            "masked_recall":
                masked_recall,

            "delta_recall":
                masked_recall
                - baseline_recall,

            "baseline_true_probability":
                baseline_prob,

            "masked_true_probability":
                masked_prob,

            # Positive = confidence drop
            "true_confidence_drop":
                baseline_prob
                - masked_prob,

            "prediction_change_rate":
                class_prediction_change_rate,
        }
    )


per_emotion_df = pd.DataFrame(
    per_emotion_rows
)


# ============================================================
# SAMPLE-LEVEL OUTPUT
# ============================================================

sample_rows = []


for i in range(
    len(test_df)
):

    true_id = int(
        labels[i]
    )


    base_pred = int(
        baseline_predictions[i]
    )


    masked_pred = int(
        masked_predictions[i]
    )


    sample_rows.append(
        {
            "sample_index":
                i,

            "text":
                test_df.iloc[i][
                    TEXT_COLUMN
                ],

            "true_emotion":
                EMOTIONS[
                    true_id
                ],

            "baseline_prediction":
                EMOTIONS[
                    base_pred
                ],

            "masked_prediction":
                EMOTIONS[
                    masked_pred
                ],

            "prediction_changed":
                bool(
                    base_pred
                    != masked_pred
                ),

            "baseline_true_probability":
                float(
                    baseline_true_probs[i]
                ),

            "masked_true_probability":
                float(
                    masked_true_probs[i]
                ),

            # Positive = confidence loss
            "true_confidence_drop":
                float(
                    sample_true_confidence_drop[
                        i
                    ]
                ),

            "baseline_max_confidence":
                float(
                    baseline_max_probs[i]
                ),

            "masked_max_confidence":
                float(
                    masked_max_probs[i]
                ),

            "baseline_pred_probability_before":
                float(
                    baseline_pred_prob_before[
                        i
                    ]
                ),

            "baseline_pred_probability_after":
                float(
                    baseline_pred_prob_after[
                        i
                    ]
                ),

            "baseline_pred_confidence_drop":
                float(
                    baseline_prediction_confidence_drop[
                        i
                    ]
                ),
        }
    )


sample_df = pd.DataFrame(
    sample_rows
)


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary = {

    "experiment":
        (
            "All exact-degree-1 top-1% "
            "emotion neurons masked "
            "simultaneously across all 28 layers"
        ),

    "num_layers_masked":
        int(
            sum(
                len(x) > 0
                for x in (
                    neurons_by_layer.values()
                )
            )
        ),

    "num_masked_neurons":
        int(total_masked),

    "baseline": {
        "accuracy":
            baseline_metrics[
                "accuracy"
            ],

        "macro_f1":
            baseline_metrics[
                "macro_f1"
            ],

        "micro_f1":
            baseline_metrics[
                "micro_f1"
            ],

        "mean_true_class_probability":
            baseline_metrics[
                "mean_true_class_probability"
            ],

        "mean_max_confidence":
            baseline_metrics[
                "mean_max_confidence"
            ],
    },

    "masked": {
        "accuracy":
            masked_metrics[
                "accuracy"
            ],

        "macro_f1":
            masked_metrics[
                "macro_f1"
            ],

        "micro_f1":
            masked_metrics[
                "micro_f1"
            ],

        "mean_true_class_probability":
            masked_metrics[
                "mean_true_class_probability"
            ],

        "mean_max_confidence":
            masked_metrics[
                "mean_max_confidence"
            ],
    },

    "delta": {

        # negative = performance loss
        "delta_accuracy":
            float(
                delta_accuracy
            ),

        "delta_macro_f1":
            float(
                delta_macro_f1
            ),

        "delta_micro_f1":
            float(
                delta_micro_f1
            ),

        # positive = confidence loss
        "true_confidence_drop":
            float(
                true_confidence_drop
            ),

        "median_true_confidence_drop":
            median_true_confidence_drop,

        "max_confidence_drop":
            float(
                max_confidence_drop
            ),

        "baseline_prediction_confidence_drop":
            mean_baseline_prediction_confidence_drop,

        "fraction_true_confidence_decreased":
            fraction_true_confidence_decreased,

        "prediction_change_rate":
            prediction_change_rate,
    },

    "per_class":
        per_emotion_rows,
}


# ============================================================
# SAVE FILES
# ============================================================

summary_json = (
    OUTPUT_DIR
    / "all_unique_masking_results.json"
)


per_emotion_file = (
    OUTPUT_DIR
    / "all_unique_masking_per_emotion.csv"
)


sample_file = (
    OUTPUT_DIR
    / "all_unique_masking_per_sample.csv"
)


with summary_json.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


per_emotion_df.to_csv(
    per_emotion_file,
    index=False,
)


sample_df.to_csv(
    sample_file,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL RESULT")
print("=" * 80)

print(
    f"Total neurons masked    : "
    f"{total_masked:,}"
)

print(
    f"Layers affected         : "
    f"{summary['num_layers_masked']}"
)

print()

print(
    f"Baseline accuracy       : "
    f"{baseline_metrics['accuracy']:.6f}"
)

print(
    f"Masked accuracy         : "
    f"{masked_metrics['accuracy']:.6f}"
)

print(
    f"Δ Accuracy              : "
    f"{delta_accuracy:+.6f}"
)

print()

print(
    f"Baseline macro-F1       : "
    f"{baseline_metrics['macro_f1']:.6f}"
)

print(
    f"Masked macro-F1         : "
    f"{masked_metrics['macro_f1']:.6f}"
)

print(
    f"Δ Macro-F1              : "
    f"{delta_macro_f1:+.6f}"
)

print()

print(
    f"Baseline true prob      : "
    f"{baseline_metrics['mean_true_class_probability']:.6f}"
)

print(
    f"Masked true prob        : "
    f"{masked_metrics['mean_true_class_probability']:.6f}"
)

print(
    f"TRUE CONFIDENCE DROP    : "
    f"{true_confidence_drop:+.6f}"
)

print()

print(
    f"Baseline max confidence : "
    f"{baseline_metrics['mean_max_confidence']:.6f}"
)

print(
    f"Masked max confidence   : "
    f"{masked_metrics['mean_max_confidence']:.6f}"
)

print(
    f"Max-confidence drop     : "
    f"{max_confidence_drop:+.6f}"
)

print()

print(
    "Baseline-pred confidence drop: "
    f"{mean_baseline_prediction_confidence_drop:+.6f}"
)

print(
    "Median true-confidence drop  : "
    f"{median_true_confidence_drop:+.6f}"
)

print(
    "Samples with confidence drop : "
    f"{fraction_true_confidence_decreased:.2%}"
)

print(
    "Prediction change rate       : "
    f"{prediction_change_rate:.2%}"
)


print()
print("=" * 80)
print("PER-EMOTION RESULTS")
print("=" * 80)

print(
    per_emotion_df[
        [
            "emotion",
            "num_unique_neurons",
            "delta_f1",
            "delta_recall",
            "true_confidence_drop",
            "prediction_change_rate",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(summary_json)
print(per_emotion_file)
print(sample_file)
