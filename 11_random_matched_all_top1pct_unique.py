#!/usr/bin/env python3

"""
Matched-random control for the all-layer exact-degree-1 top-1% experiment.

Selected intervention:
    782 exact-degree-1 neurons, distributed across 28 layers.

Random control:
    For each layer, sample exactly the same number of neurons as the
    selected degree-1 set in that layer.

Random pool:
    Excludes EVERY neuron appearing in ANY emotion's top-1% list
    in that layer.

Therefore the control is:
    - count matched
    - layer matched
    - non-selected
    - applied across all 28 layers simultaneously

Outputs:
    random_runs.csv
    random_control_summary.json
    layer_sampling_counts.csv
    random_masks.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score
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

# Exact-degree-1 neurons
SINGLE_EMOTION_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "single_emotion_analysis"
    / "single_emotion_neurons.csv"
)

# All top-1% selections.
# Used only to EXCLUDE selected neurons from random sampling.
ALL_TOP1_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "all_layers_top1pct.csv"
)

# Result from the selected 782-neuron intervention.
SELECTED_RESULT_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "10_all_unique_masking"
    / "all_unique_masking_results.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "11_random_matched_control"
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

NUM_LAYERS = 32
MLP_WIDTH = 14336

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
    emotion: i
    for i, emotion in enumerate(EMOTIONS)
}


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--runs",
    type=int,
    default=50,
    help="Number of matched-random masking runs",
)

parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="Base random seed",
)

args = parser.parse_args()

NUM_RANDOM_RUNS = args.runs
BASE_RANDOM_SEED = args.seed


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    DATA_FILE,
    PROBE_FILE,
    SINGLE_EMOTION_FILE,
    ALL_TOP1_FILE,
    SELECTED_RESULT_FILE,
]:

    if not path.is_file():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
print("MATCHED RANDOM CONTROL")
print("=" * 80)

print(f"Random runs : {NUM_RANDOM_RUNS}")
print(f"Base seed   : {BASE_RANDOM_SEED}")
print(f"Device      : {DEVICE}")
print(f"Dtype       : {MODEL_DTYPE}")


# ============================================================
# DATASET
# ============================================================

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


# ============================================================
# EXACT SAME TEST SET
# ============================================================

_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=df[LABEL_COLUMN],
)

test_df = test_df.reset_index(
    drop=True
)


print()
print(f"Dataset rows : {len(df):,}")
print(f"Test rows    : {len(test_df):,}")


# ============================================================
# LOAD DEGREE-1 SELECTED NEURONS
# ============================================================

single_df = pd.read_csv(
    SINGLE_EMOTION_FILE
)


selected_by_layer = {}

for layer_idx in range(
    NUM_LAYERS
):

    neuron_ids = (
        single_df.loc[
            single_df["layer_index"] == layer_idx,
            "neuron_index",
        ]
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_by_layer[
        layer_idx
    ] = neuron_ids


selected_total = sum(
    len(x)
    for x in selected_by_layer.values()
)


print()
print("=" * 80)
print("SELECTED DEGREE-1 COUNTS")
print("=" * 80)

for layer_idx in range(
    NUM_LAYERS
):

    print(
        f"Layer {layer_idx + 1:02d}: "
        f"{len(selected_by_layer[layer_idx]):4d}"
    )


print()
print(
    f"Total selected degree-1 neurons: "
    f"{selected_total:,}"
)


if selected_total != 782:

    print(
        "WARNING: expected 782 from the latest experiment, "
        f"but current file contains {selected_total}."
    )


# ============================================================
# LOAD ALL TOP-1% SELECTED NEURONS
# ============================================================
#
# Random pool excludes ANY neuron appearing in the
# top-1% list for ANY emotion.
#
# This prevents the random baseline from accidentally
# sampling high-selectivity candidate neurons.
# ============================================================

top1_df = pd.read_csv(
    ALL_TOP1_FILE
)


excluded_by_layer = {}

random_pool_by_layer = {}


all_neurons = np.arange(
    MLP_WIDTH,
    dtype=np.int64,
)


for layer_idx in range(
    NUM_LAYERS
):

    excluded = np.unique(
        top1_df.loc[
            top1_df["layer_index"] == layer_idx,
            "neuron_index",
        ]
        .astype(int)
        .to_numpy()
    )


    excluded_by_layer[
        layer_idx
    ] = excluded


    pool = np.setdiff1d(
        all_neurons,
        excluded,
        assume_unique=False,
    )


    random_pool_by_layer[
        layer_idx
    ] = pool


    required = len(
        selected_by_layer[
            layer_idx
        ]
    )


    if len(pool) < required:

        raise RuntimeError(
            f"Layer {layer_idx}: "
            f"random pool size={len(pool)}, "
            f"required={required}"
        )


# ============================================================
# SAVE LAYER COUNTS
# ============================================================

layer_count_rows = []


for layer_idx in range(
    NUM_LAYERS
):

    layer_count_rows.append(
        {
            "layer": layer_idx + 1,
            "layer_index": layer_idx,

            "selected_degree1_neurons":
                len(
                    selected_by_layer[
                        layer_idx
                    ]
                ),

            "all_top1_unique_neurons_excluded":
                len(
                    excluded_by_layer[
                        layer_idx
                    ]
                ),

            "random_pool_size":
                len(
                    random_pool_by_layer[
                        layer_idx
                    ]
                ),
        }
    )


pd.DataFrame(
    layer_count_rows
).to_csv(
    OUTPUT_DIR
    / "layer_sampling_counts.csv",
    index=False,
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
# MODEL
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

model.to(
    DEVICE
)

model.eval()


for parameter in model.parameters():

    parameter.requires_grad_(
        False
    )


layers = model.model.layers


if len(layers) != NUM_LAYERS:

    raise ValueError(
        f"Expected {NUM_LAYERS} layers, "
        f"got {len(layers)}"
    )


backbone = model.model


# ============================================================
# PROBE LOADER
# ============================================================

def load_probe(path):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


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


    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            f"Unsupported checkpoint type: "
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
            "Could not find 10-class probe weights."
        )


    candidates.sort(
        key=lambda x:
        (
            0
            if any(
                n in x[0].lower()
                for n in [
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

            prefix = weight_key[
                :-len("weight")
            ]

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


    return probe


probe = load_probe(
    PROBE_FILE
)


# ============================================================
# VALID TOKEN MASK
# ============================================================

def create_valid_token_mask(
    input_ids,
    attention_mask,
):

    mask = (
        attention_mask
        .bool()
        .clone()
    )


    for token_id in (
        tokenizer.all_special_ids
    ):

        mask &= input_ids.ne(
            token_id
        )


    return mask


# ============================================================
# HOOK CONTEXT
# ============================================================

mask_context = {
    "valid_mask": None
}


# ============================================================
# MASK HOOK
# ============================================================

def create_mask_hook(
    neuron_indices,
):

    neuron_indices = [
        int(x)
        for x in neuron_indices
    ]


    index_cache = {}


    def hook(
        module,
        inputs,
    ):

        if (
            not inputs
            or not neuron_indices
        ):

            return inputs


        x = inputs[0]


        valid_mask = (
            mask_context[
                "valid_mask"
            ]
        )


        if valid_mask is None:

            raise RuntimeError(
                "Missing valid token mask."
            )


        device_key = str(
            x.device
        )


        if device_key not in (
            index_cache
        ):

            index_cache[
                device_key
            ] = torch.tensor(
                neuron_indices,
                dtype=torch.long,
                device=x.device,
            )


        indices = (
            index_cache[
                device_key
            ]
        )


        masked = x.clone()


        selected = masked[
            :,
            :,
            indices
        ]


        selected = (
            selected.masked_fill(
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
        ] = selected


        return (
            masked,
            *inputs[1:]
        )


    return hook


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation(
    mask_by_layer=None,
    description="Evaluation",
):

    handles = []


    if mask_by_layer is not None:

        for layer_idx in range(
            NUM_LAYERS
        ):

            neuron_ids = (
                mask_by_layer.get(
                    layer_idx,
                    [],
                )
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
                desc=description,
                leave=False,
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
                        LABEL_TO_ID[x]
                        for x in (
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


                hidden = (
                    outputs.last_hidden_state
                    if hasattr(
                        outputs,
                        "last_hidden_state"
                    )
                    else outputs[0]
                )


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

        for handle in handles:

            handle.remove()


        mask_context[
            "valid_mask"
        ] = None


    return (
        torch.cat(
            all_logits,
            dim=0,
        ),
        torch.cat(
            all_labels,
            dim=0,
        ),
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    logits,
    labels,
    baseline_predictions=None,
):

    probabilities = torch.softmax(
        logits,
        dim=1,
    )


    predictions = probabilities.argmax(
        dim=1
    )


    y_true = labels.numpy()

    y_pred = predictions.numpy()


    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )


    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )


    rows = torch.arange(
        len(labels)
    )


    true_probs = probabilities[
        rows,
        labels
    ]


    max_confidence = (
        probabilities
        .max(dim=1)
        .values
    )


    result = {
        "accuracy":
            float(accuracy),

        "macro_f1":
            float(macro_f1),

        "micro_f1":
            float(micro_f1),

        "mean_true_class_probability":
            float(
                true_probs.mean()
            ),

        "mean_max_confidence":
            float(
                max_confidence.mean()
            ),
    }


    if baseline_predictions is not None:

        baseline_pred_probs = (
            probabilities[
                rows,
                baseline_predictions
            ]
        )


        result[
            "mean_baseline_prediction_probability"
        ] = float(
            baseline_pred_probs.mean()
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
print("BASELINE")
print("=" * 80)


baseline_logits, labels = (
    run_evaluation(
        mask_by_layer=None,
        description="Baseline",
    )
)


(
    baseline_metrics,
    baseline_probabilities,
    baseline_predictions,
) = compute_metrics(
    baseline_logits,
    labels,
)


rows = torch.arange(
    len(labels)
)


baseline_true_probs = (
    baseline_probabilities[
        rows,
        labels
    ]
)


baseline_original_pred_probs = (
    baseline_probabilities[
        rows,
        baseline_predictions
    ]
)


print()
print(
    f"Baseline accuracy  : "
    f"{baseline_metrics['accuracy']:.9f}"
)

print(
    f"Baseline macro-F1  : "
    f"{baseline_metrics['macro_f1']:.9f}"
)

print(
    f"Baseline true prob : "
    f"{baseline_metrics['mean_true_class_probability']:.9f}"
)


# ============================================================
# LOAD SELECTED RESULT
# ============================================================

with SELECTED_RESULT_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    selected_result = json.load(
        f
    )


stored_baseline = (
    selected_result[
        "baseline"
    ]
)


# ============================================================
# CHECK SAME BASELINE
# ============================================================

for metric in [
    "accuracy",
    "macro_f1",
    "mean_true_class_probability",
]:

    current = (
        baseline_metrics[
            metric
        ]
    )

    stored = (
        stored_baseline[
            metric
        ]
    )


    if abs(
        current - stored
    ) > 1e-6:

        raise RuntimeError(
            f"Baseline mismatch for {metric}:\n"
            f"Current = {current}\n"
            f"Stored  = {stored}\n"
            "Do not compare experiments using "
            "different evaluation pipelines."
        )


print()
print(
    "Baseline matches selected-neuron experiment."
)


# ============================================================
# SELECTED EFFECT
# ============================================================

selected_masked = (
    selected_result[
        "masked"
    ]
)


selected_delta = (
    selected_result[
        "delta"
    ]
)


selected_accuracy_damage = (
    baseline_metrics[
        "accuracy"
    ]
    - selected_masked[
        "accuracy"
    ]
)


selected_macro_f1_damage = (
    baseline_metrics[
        "macro_f1"
    ]
    - selected_masked[
        "macro_f1"
    ]
)


selected_true_confidence_drop = (
    selected_delta[
        "true_confidence_drop"
    ]
)


selected_baseline_pred_drop = (
    selected_delta[
        "baseline_prediction_confidence_drop"
    ]
)


selected_prediction_change_rate = (
    selected_delta[
        "prediction_change_rate"
    ]
)


# ============================================================
# RANDOM MASK GENERATOR
# ============================================================

def generate_random_mask(
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    mask = {}


    for layer_idx in range(
        NUM_LAYERS
    ):

        count = len(
            selected_by_layer[
                layer_idx
            ]
        )


        if count == 0:

            mask[
                layer_idx
            ] = []

            continue


        pool = (
            random_pool_by_layer[
                layer_idx
            ]
        )


        sampled = rng.choice(
            pool,
            size=count,
            replace=False,
        )


        mask[
            layer_idx
        ] = sorted(
            sampled
            .astype(int)
            .tolist()
        )


    total = sum(
        len(v)
        for v in mask.values()
    )


    if total != selected_total:

        raise RuntimeError(
            f"Random mask has {total} neurons; "
            f"expected {selected_total}."
        )


    return mask


# ============================================================
# RANDOM RUNS
# ============================================================

random_rows = []

saved_masks = {}


print()
print("=" * 80)
print(
    f"RUNNING {NUM_RANDOM_RUNS} MATCHED RANDOM CONTROLS"
)
print("=" * 80)


for run_idx in range(
    NUM_RANDOM_RUNS
):

    seed = (
        BASE_RANDOM_SEED
        + run_idx
    )


    random_mask = (
        generate_random_mask(
            seed
        )
    )


    saved_masks[
        str(seed)
    ] = {
        str(layer_idx):
            neuron_ids
        for (
            layer_idx,
            neuron_ids
        ) in random_mask.items()
    }


    print()
    print(
        f"Random run "
        f"{run_idx + 1}/{NUM_RANDOM_RUNS} "
        f"| seed={seed}"
    )


    logits, run_labels = (
        run_evaluation(
            mask_by_layer=
                random_mask,

            description=
                f"Random {run_idx + 1}",
        )
    )


    if not torch.equal(
        labels,
        run_labels,
    ):

        raise RuntimeError(
            "Label ordering changed."
        )


    (
        metrics,
        probabilities,
        predictions,
    ) = compute_metrics(
        logits,
        run_labels,
        baseline_predictions=
            baseline_predictions,
    )


    true_probs = (
        probabilities[
            rows,
            labels
        ]
    )


    original_pred_probs = (
        probabilities[
            rows,
            baseline_predictions
        ]
    )


    # Positive = performance degradation.
    accuracy_damage = (
        baseline_metrics[
            "accuracy"
        ]
        - metrics[
            "accuracy"
        ]
    )


    macro_f1_damage = (
        baseline_metrics[
            "macro_f1"
        ]
        - metrics[
            "macro_f1"
        ]
    )


    true_confidence_drop = float(
        (
            baseline_true_probs
            - true_probs
        )
        .mean()
    )


    baseline_pred_confidence_drop = float(
        (
            baseline_original_pred_probs
            - original_pred_probs
        )
        .mean()
    )


    prediction_change_rate = float(
        (
            predictions
            != baseline_predictions
        )
        .float()
        .mean()
    )


    random_rows.append(
        {
            "run":
                run_idx + 1,

            "seed":
                seed,

            "num_masked_neurons":
                selected_total,

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "micro_f1":
                metrics[
                    "micro_f1"
                ],

            # conventional delta:
            # masked - baseline
            "delta_accuracy":
                metrics[
                    "accuracy"
                ]
                - baseline_metrics[
                    "accuracy"
                ],

            "delta_macro_f1":
                metrics[
                    "macro_f1"
                ]
                - baseline_metrics[
                    "macro_f1"
                ],

            # Positive damage notation
            "accuracy_damage":
                accuracy_damage,

            "macro_f1_damage":
                macro_f1_damage,

            "mean_true_class_probability":
                metrics[
                    "mean_true_class_probability"
                ],

            "true_confidence_drop":
                true_confidence_drop,

            "baseline_prediction_confidence_drop":
                baseline_pred_confidence_drop,

            "prediction_change_rate":
                prediction_change_rate,
        }
    )


    print(
        f"Macro-F1={metrics['macro_f1']:.6f} "
        f"| damage={macro_f1_damage:+.6f} "
        f"| true-P drop="
        f"{true_confidence_drop:+.6f}"
    )


# ============================================================
# DATAFRAME
# ============================================================

random_df = pd.DataFrame(
    random_rows
)


random_csv = (
    OUTPUT_DIR
    / "random_runs.csv"
)


random_df.to_csv(
    random_csv,
    index=False,
)


# ============================================================
# AGGREGATE
# ============================================================

def stats(
    column,
):

    values = (
        random_df[
            column
        ]
        .to_numpy(
            dtype=float
        )
    )


    return {
        "mean":
            float(
                np.mean(values)
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1,
                )
            ),

        "median":
            float(
                np.median(values)
            ),

        "min":
            float(
                np.min(values)
            ),

        "max":
            float(
                np.max(values)
            ),

        "q025":
            float(
                np.quantile(
                    values,
                    0.025,
                )
            ),

        "q975":
            float(
                np.quantile(
                    values,
                    0.975,
                )
            ),
    }


random_macro_damage = (
    random_df[
        "macro_f1_damage"
    ]
    .to_numpy()
)


random_accuracy_damage = (
    random_df[
        "accuracy_damage"
    ]
    .to_numpy()
)


random_conf_drop = (
    random_df[
        "true_confidence_drop"
    ]
    .to_numpy()
)


# ============================================================
# ONE-SIDED EMPIRICAL P VALUES
# ============================================================
#
# Question:
# How often was RANDOM damage >= SELECTED damage?
#
# Smaller p = selected neurons are more damaging
# than matched random neurons.
# ============================================================

p_macro = (
    1
    + np.sum(
        random_macro_damage
        >= selected_macro_f1_damage
    )
) / (
    NUM_RANDOM_RUNS
    + 1
)


p_accuracy = (
    1
    + np.sum(
        random_accuracy_damage
        >= selected_accuracy_damage
    )
) / (
    NUM_RANDOM_RUNS
    + 1
)


p_confidence = (
    1
    + np.sum(
        random_conf_drop
        >= selected_true_confidence_drop
    )
) / (
    NUM_RANDOM_RUNS
    + 1
)


# ============================================================
# DESCRIPTIVE Z-SCORES
# ============================================================

def descriptive_z(
    selected,
    random_values,
):

    mean = np.mean(
        random_values
    )

    std = np.std(
        random_values,
        ddof=1,
    )


    if std == 0:

        return None


    return float(
        (
            selected - mean
        )
        / std
    )


z_macro = descriptive_z(
    selected_macro_f1_damage,
    random_macro_damage,
)


z_accuracy = descriptive_z(
    selected_accuracy_damage,
    random_accuracy_damage,
)


z_confidence = descriptive_z(
    selected_true_confidence_drop,
    random_conf_drop,
)


# ============================================================
# EXCESS EFFECT
# ============================================================

random_mean_macro_damage = float(
    random_macro_damage.mean()
)


random_mean_accuracy_damage = float(
    random_accuracy_damage.mean()
)


random_mean_conf_drop = float(
    random_conf_drop.mean()
)


excess_macro_damage = (
    selected_macro_f1_damage
    - random_mean_macro_damage
)


excess_accuracy_damage = (
    selected_accuracy_damage
    - random_mean_accuracy_damage
)


excess_confidence_drop = (
    selected_true_confidence_drop
    - random_mean_conf_drop
)


# ============================================================
# FINAL JSON
# ============================================================

summary = {

    "experiment":
        (
            "Layer-count-matched random control "
            "for all-layer exact-degree-1 "
            "top-1% masking"
        ),

    "random_runs":
        NUM_RANDOM_RUNS,

    "base_seed":
        BASE_RANDOM_SEED,

    "num_masked_neurons_per_run":
        selected_total,

    "random_pool":
        (
            "All neurons excluding every neuron "
            "appearing in any emotion's top-1% "
            "selection in the same layer"
        ),

    "matching":
        {
            "same_total_count":
                True,

            "same_layer_counts":
                True,

            "emotion_specific_matching":
                False,
        },

    "baseline":
        baseline_metrics,

    "selected":
        {
            "num_masked_neurons":
                selected_total,

            "accuracy":
                selected_masked[
                    "accuracy"
                ],

            "macro_f1":
                selected_masked[
                    "macro_f1"
                ],

            "accuracy_damage":
                selected_accuracy_damage,

            "macro_f1_damage":
                selected_macro_f1_damage,

            "true_confidence_drop":
                selected_true_confidence_drop,

            "baseline_prediction_confidence_drop":
                selected_baseline_pred_drop,

            "prediction_change_rate":
                selected_prediction_change_rate,
        },

    "random":
        {
            "accuracy_damage":
                stats(
                    "accuracy_damage"
                ),

            "macro_f1_damage":
                stats(
                    "macro_f1_damage"
                ),

            "true_confidence_drop":
                stats(
                    "true_confidence_drop"
                ),

            "baseline_prediction_confidence_drop":
                stats(
                    "baseline_prediction_confidence_drop"
                ),

            "prediction_change_rate":
                stats(
                    "prediction_change_rate"
                ),
        },

    "selected_minus_random_mean":
        {
            "excess_accuracy_damage":
                float(
                    excess_accuracy_damage
                ),

            "excess_macro_f1_damage":
                float(
                    excess_macro_damage
                ),

            "excess_true_confidence_drop":
                float(
                    excess_confidence_drop
                ),
        },

    "empirical_one_sided_p":
        {
            "accuracy_damage":
                float(
                    p_accuracy
                ),

            "macro_f1_damage":
                float(
                    p_macro
                ),

            "true_confidence_drop":
                float(
                    p_confidence
                ),
        },

    "descriptive_z_score":
        {
            "accuracy_damage":
                z_accuracy,

            "macro_f1_damage":
                z_macro,

            "true_confidence_drop":
                z_confidence,
        },
}


summary_file = (
    OUTPUT_DIR
    / "random_control_summary.json"
)


with summary_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# SAVE RANDOM MASKS
# ============================================================

mask_file = (
    OUTPUT_DIR
    / "random_masks.json"
)


with mask_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        saved_masks,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("SELECTED VS MATCHED RANDOM")
print("=" * 80)

print()
print(
    f"Masked neurons per condition : "
    f"{selected_total:,}"
)


print()
print("--- MACRO-F1 DAMAGE ---")

print(
    f"Selected damage      : "
    f"{selected_macro_f1_damage:+.6f}"
)

print(
    f"Random mean damage   : "
    f"{random_mean_macro_damage:+.6f}"
)

print(
    f"Random std           : "
    f"{random_df['macro_f1_damage'].std(ddof=1):.6f}"
)

print(
    f"Excess selected      : "
    f"{excess_macro_damage:+.6f}"
)

print(
    f"Empirical p          : "
    f"{p_macro:.6f}"
)

print(
    f"Descriptive z        : "
    f"{z_macro}"
)


print()
print("--- ACCURACY DAMAGE ---")

print(
    f"Selected damage      : "
    f"{selected_accuracy_damage:+.6f}"
)

print(
    f"Random mean damage   : "
    f"{random_mean_accuracy_damage:+.6f}"
)

print(
    f"Excess selected      : "
    f"{excess_accuracy_damage:+.6f}"
)

print(
    f"Empirical p          : "
    f"{p_accuracy:.6f}"
)


print()
print("--- TRUE-CLASS CONFIDENCE DROP ---")

print(
    f"Selected drop        : "
    f"{selected_true_confidence_drop:+.6f}"
)

print(
    f"Random mean drop     : "
    f"{random_mean_conf_drop:+.6f}"
)

print(
    f"Excess selected      : "
    f"{excess_confidence_drop:+.6f}"
)

print(
    f"Empirical p          : "
    f"{p_confidence:.6f}"
)

print(
    f"Descriptive z        : "
    f"{z_confidence}"
)


print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(random_csv)
print(summary_file)
print(mask_file)

print()
print("Done.")
