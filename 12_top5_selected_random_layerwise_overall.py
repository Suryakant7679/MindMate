#!/usr/bin/env python3

"""
Sarvam-1 top-5% neuron causal experiment.

Pipeline
--------
1. Load full selectivity tensor:
       S[layer, neuron, emotion]

2. For every layer and every emotion:
       select top 5% = 550 / 11008 neurons

3. Compute emotion-membership degree within each layer.

4. Keep exact degree-1 neurons:
       neuron appears in top-5% of exactly ONE emotion.

5. Evaluate:
       A. Baseline
       B. Selected layer-wise masking
       C. Selected all-layer masking
       D. Layer/count-matched random masking layer-wise
       E. Layer/count-matched random masking overall

Random control
--------------
By default random neurons are sampled from neurons NOT appearing in
ANY top-5% emotion list in that layer.

This matches the previous top-1% random-control methodology.

Outputs
-------
results/sarvam1_layerwise_top5pct/12_selected_random/

    top5_degree1_neurons.csv
    top5_degree1_counts_by_layer.csv
    top5_degree1_counts_by_emotion.csv
    top5_union_by_layer.json

    baseline.json

    selected_layerwise.csv
    selected_overall.json
    selected_overall_per_class.csv

    random_layerwise_runs.csv
    random_layerwise_summary.csv

    random_overall_runs.csv
    random_overall_summary.json
    random_overall_masks.json
"""

from __future__ import annotations

import argparse
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
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--layer-random-runs",
    type=int,
    default=10,
    help=(
        "Random repetitions PER LAYER. "
        "10 is reasonable initially; 50 is expensive."
    ),
)

parser.add_argument(
    "--overall-random-runs",
    type=int,
    default=50,
    help="Random repetitions for all-layer simultaneous masking.",
)

parser.add_argument(
    "--random-seed",
    type=int,
    default=12345,
)

parser.add_argument(
    "--random-pool",
    choices=[
        "non_top5",
        "all_except_degree1",
    ],
    default="non_top5",
    help=(
        "non_top5 = exclude every neuron appearing in any top-5%% list. "
        "all_except_degree1 = exclude only target degree-1 neurons."
    ),
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SELECTIVITY_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_emotions_full"
    / "selectivity.npy"
)

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

# Previous 1% result is used only to verify that the baseline
# remains identical.
BASELINE_REFERENCE_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "10_all_unique_masking"
    / "all_unique_masking_results.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top5pct"
    / "12_selected_random"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

NUM_LAYERS = 32
MLP_WIDTH = 14336

TOP_PERCENT = 5.0

# Same integer rule as previous extraction.
TOP_K = int(
    MLP_WIDTH
    * TOP_PERCENT
    / 100.0
)

if TOP_K != 716:

    raise RuntimeError(
        f"Expected top_k=716 for Llama-3.1-8B (14,336 MLP neurons), got {TOP_K}"
    )


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


TEXT_COLUMN = "hinglish_genz_text"
LABEL_COLUMN = "label"

MAX_LENGTH = 256
BATCH_SIZE = 4

TEST_SIZE = 0.15
SPLIT_SEED = 42


# ============================================================
# CHECK INPUTS
# ============================================================

for file_path in [
    SELECTIVITY_FILE,
    DATA_FILE,
    PROBE_FILE,
]:

    if not file_path.is_file():

        raise FileNotFoundError(
            f"Missing required file:\n{file_path}"
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


print("=" * 90)
print("LLAMA-3.1-8B TOP-5% SELECTED + RANDOM CAUSAL EXPERIMENT")
print("=" * 90)

print(f"Top percentage       : {TOP_PERCENT}%")
print(f"Top-k/layer/emotion  : {TOP_K}")
print(f"Layer random runs    : {args.layer_random_runs}")
print(f"Overall random runs  : {args.overall_random_runs}")
print(f"Random pool          : {args.random_pool}")
print(f"Device               : {DEVICE}")
print(f"Dtype                : {MODEL_DTYPE}")


# ============================================================
# PART 1
# EXTRACT TOP-5% AND EXACT DEGREE-1 NEURONS
# ============================================================

print()
print("=" * 90)
print("1. EXTRACTING TOP-5% EXACT-DEGREE-1 NEURONS")
print("=" * 90)


S = np.load(
    SELECTIVITY_FILE,
    mmap_mode="r",
)


expected_shape = (
    NUM_LAYERS,
    MLP_WIDTH,
    NUM_CLASSES,
)

if S.shape != expected_shape:

    raise ValueError(
        f"Unexpected selectivity shape: {S.shape}\n"
        f"Expected: {expected_shape}"
    )


selected_by_layer = {}

top5_union_by_layer = {}

degree1_rows = []


for layer_idx in range(NUM_LAYERS):

    membership_count = np.zeros(
        MLP_WIDTH,
        dtype=np.int16,
    )

    # Valid only for degree-1 neurons.
    owner_emotion = np.full(
        MLP_WIDTH,
        -1,
        dtype=np.int16,
    )

    owner_rank = np.full(
        MLP_WIDTH,
        -1,
        dtype=np.int32,
    )

    emotion_top_indices = []


    for emotion_idx, emotion in enumerate(
        EMOTIONS
    ):

        scores = np.asarray(
            S[
                layer_idx,
                :,
                emotion_idx,
            ],
            dtype=np.float64,
        )


        if not np.isfinite(scores).all():

            raise ValueError(
                f"Non-finite selectivity values: "
                f"layer={layer_idx}, emotion={emotion}"
            )


        ranked = np.argsort(
            -scores,
            kind="stable",
        )

        top_indices = ranked[
            :TOP_K
        ]


        emotion_top_indices.append(
            top_indices
        )


        membership_count[
            top_indices
        ] += 1


        # If neuron is degree 1, these values remain the correct owner.
        # Overlap neurons may be overwritten, but they are discarded later.
        owner_emotion[
            top_indices
        ] = emotion_idx

        owner_rank[
            top_indices
        ] = np.arange(
            1,
            TOP_K + 1,
            dtype=np.int32,
        )


    # --------------------------------------------------------
    # Union of every top-5% emotion neuron in this layer.
    # Used for matched random sampling.
    # --------------------------------------------------------

    top5_union = np.unique(
        np.concatenate(
            emotion_top_indices
        )
    )

    top5_union_by_layer[
        layer_idx
    ] = top5_union.astype(
        int
    ).tolist()


    # --------------------------------------------------------
    # Exact degree-1 neurons
    # --------------------------------------------------------

    degree1_indices = np.where(
        membership_count == 1
    )[0]


    selected_by_layer[
        layer_idx
    ] = (
        degree1_indices
        .astype(int)
        .tolist()
    )


    for neuron_idx in degree1_indices:

        emotion_idx = int(
            owner_emotion[
                neuron_idx
            ]
        )

        emotion = EMOTIONS[
            emotion_idx
        ]


        degree1_rows.append(
            {
                "layer":
                    layer_idx + 1,

                "layer_index":
                    layer_idx,

                "neuron_index":
                    int(neuron_idx),

                "neuron_number":
                    int(neuron_idx) + 1,

                "emotion":
                    emotion,

                "rank_within_emotion":
                    int(
                        owner_rank[
                            neuron_idx
                        ]
                    ),

                "selectivity_score":
                    float(
                        S[
                            layer_idx,
                            neuron_idx,
                            emotion_idx,
                        ]
                    ),

                "num_emotions":
                    1,
            }
        )


    print(
        f"Layer {layer_idx + 1:02d} | "
        f"top5 union={len(top5_union):4d} | "
        f"degree1={len(degree1_indices):4d}"
    )


degree1_df = pd.DataFrame(
    degree1_rows
)


TOTAL_SELECTED = len(
    degree1_df
)


print()
print(
    f"Total exact-degree-1 neurons at 5%: "
    f"{TOTAL_SELECTED:,}"
)


# ============================================================
# SAVE DEGREE-1 DATA
# ============================================================

degree1_file = (
    OUTPUT_DIR
    / "top5_degree1_neurons.csv"
)

degree1_df.to_csv(
    degree1_file,
    index=False,
)


layer_counts_df = (
    degree1_df
    .groupby("layer")
    .size()
    .reindex(
        range(1, NUM_LAYERS + 1),
        fill_value=0,
    )
    .reset_index(
        name="num_degree1_neurons"
    )
)


layer_counts_df.to_csv(
    OUTPUT_DIR
    / "top5_degree1_counts_by_layer.csv",
    index=False,
)


emotion_counts_df = (
    degree1_df
    .groupby("emotion")
    .size()
    .reindex(
        EMOTIONS,
        fill_value=0,
    )
    .reset_index(
        name="num_degree1_neurons"
    )
)


emotion_counts_df.to_csv(
    OUTPUT_DIR
    / "top5_degree1_counts_by_emotion.csv",
    index=False,
)


with (
    OUTPUT_DIR
    / "top5_union_by_layer.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            str(k): v
            for k, v
            in top5_union_by_layer.items()
        },
        f,
        indent=2,
    )


print()
print("Degree-1 neurons by emotion:")
print(
    emotion_counts_df.to_string(
        index=False
    )
)


# ============================================================
# PART 2
# DATASET / TEST SPLIT
# ============================================================

print()
print("=" * 90)
print("2. LOADING DATASET")
print("=" * 90)


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
    .reset_index(
        drop=True
    )
)


unknown_labels = (
    set(
        df[
            LABEL_COLUMN
        ].unique()
    )
    - set(EMOTIONS)
)


if unknown_labels:

    raise ValueError(
        f"Unknown labels: "
        f"{unknown_labels}"
    )


_, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=df[LABEL_COLUMN],
)


test_df = test_df.reset_index(
    drop=True
)


print(
    f"Dataset rows : {len(df):,}"
)

print(
    f"Test rows    : {len(test_df):,}"
)


# ============================================================
# PART 3
# LOAD TOKENIZER / MODEL
# ============================================================

print()
print("=" * 90)
print("3. LOADING MODEL")
print("=" * 90)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


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

    raise RuntimeError(
        f"Expected {NUM_LAYERS} layers, "
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
            "missing mlp.down_proj"
        )


backbone = model.model


# ============================================================
# PROBE LOADER
# ============================================================

def load_probe(path: Path):

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

        for parameter in probe.parameters():

            parameter.requires_grad_(
                False
            )

        return probe


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
            "Could not find the "
            "10-class probe weight matrix."
        )


    candidates.sort(
        key=lambda item:
        (
            0
            if any(
                name
                in item[0].lower()
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


    for parameter in probe.parameters():

        parameter.requires_grad_(
            False
        )


    print(
        f"Probe loaded: "
        f"{hidden_dim} -> {NUM_CLASSES}"
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

    valid_mask = (
        attention_mask
        .bool()
        .clone()
    )


    for special_id in (
        tokenizer.all_special_ids
    ):

        valid_mask &= (
            input_ids.ne(
                special_id
            )
        )


    return valid_mask


# ============================================================
# MASK CONTEXT
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
                "Hook invoked without "
                "valid-token mask."
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


        selected_values = (
            masked[
                :,
                :,
                indices,
            ]
        )


        selected_values = (
            selected_values
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
            indices,
        ] = selected_values


        return (
            masked,
            *inputs[1:],
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

                batch = test_df.iloc[
                    start:
                    start + BATCH_SIZE
                ]


                texts = (
                    batch[
                        TEXT_COLUMN
                    ]
                    .tolist()
                )


                labels = torch.tensor(
                    [
                        LABEL_TO_ID[
                            label
                        ]
                        for label in (
                            batch[
                                LABEL_COLUMN
                            ].tolist()
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
                        "last_hidden_state",
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


    row_indices = torch.arange(
        len(labels)
    )


    true_probabilities = (
        probabilities[
            row_indices,
            labels,
        ]
    )


    max_confidences = (
        probabilities
        .max(dim=1)
        .values
    )


    per_class_true_probability = []


    for class_idx in range(
        NUM_CLASSES
    ):

        class_mask = (
            labels == class_idx
        )


        per_class_true_probability.append(
            float(
                true_probabilities[
                    class_mask
                ].mean()
            )
        )


    metrics = {

        "accuracy":
            float(
                accuracy
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "micro_f1":
            float(
                micro_f1
            ),

        "mean_true_class_probability":
            float(
                true_probabilities.mean()
            ),

        "mean_max_confidence":
            float(
                max_confidences.mean()
            ),

        "precision":
            precision.tolist(),

        "recall":
            recall.tolist(),

        "f1":
            f1.tolist(),

        "support":
            support.tolist(),

        "per_class_true_probability":
            per_class_true_probability,
    }


    return (
        metrics,
        probabilities,
        predictions,
    )


# ============================================================
# COMPARISON FUNCTION
# ============================================================

def compare_to_baseline(
    baseline_metrics,
    baseline_probabilities,
    baseline_predictions,
    masked_metrics,
    masked_probabilities,
    masked_predictions,
    labels,
):

    rows = torch.arange(
        len(labels)
    )


    baseline_true = (
        baseline_probabilities[
            rows,
            labels,
        ]
    )


    masked_true = (
        masked_probabilities[
            rows,
            labels,
        ]
    )


    baseline_original_prediction_prob = (
        baseline_probabilities[
            rows,
            baseline_predictions,
        ]
    )


    masked_original_prediction_prob = (
        masked_probabilities[
            rows,
            baseline_predictions,
        ]
    )


    return {

        # masked - baseline
        "delta_accuracy":
            float(
                masked_metrics[
                    "accuracy"
                ]
                - baseline_metrics[
                    "accuracy"
                ]
            ),

        "delta_macro_f1":
            float(
                masked_metrics[
                    "macro_f1"
                ]
                - baseline_metrics[
                    "macro_f1"
                ]
            ),

        "delta_micro_f1":
            float(
                masked_metrics[
                    "micro_f1"
                ]
                - baseline_metrics[
                    "micro_f1"
                ]
            ),


        # Positive damage convention
        "accuracy_damage":
            float(
                baseline_metrics[
                    "accuracy"
                ]
                - masked_metrics[
                    "accuracy"
                ]
            ),

        "macro_f1_damage":
            float(
                baseline_metrics[
                    "macro_f1"
                ]
                - masked_metrics[
                    "macro_f1"
                ]
            ),


        # Positive = confidence loss
        "true_confidence_drop":
            float(
                (
                    baseline_true
                    - masked_true
                ).mean()
            ),

        "baseline_prediction_confidence_drop":
            float(
                (
                    baseline_original_prediction_prob
                    - masked_original_prediction_prob
                ).mean()
            ),

        "max_confidence_drop":
            float(
                baseline_metrics[
                    "mean_max_confidence"
                ]
                - masked_metrics[
                    "mean_max_confidence"
                ]
            ),

        "prediction_change_rate":
            float(
                (
                    baseline_predictions
                    != masked_predictions
                )
                .float()
                .mean()
            ),

        "fraction_true_confidence_decreased":
            float(
                (
                    (
                        baseline_true
                        - masked_true
                    ) > 0
                )
                .float()
                .mean()
            ),
    }


# ============================================================
# PART 4
# BASELINE
# ============================================================

print()
print("=" * 90)
print("4. BASELINE")
print("=" * 90)


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


print()
print(
    f"Accuracy        : "
    f"{baseline_metrics['accuracy']:.9f}"
)

print(
    f"Macro-F1        : "
    f"{baseline_metrics['macro_f1']:.9f}"
)

print(
    f"True-class prob : "
    f"{baseline_metrics['mean_true_class_probability']:.9f}"
)


# ============================================================
# VERIFY SAME BASELINE AS 1% EXPERIMENT
# ============================================================

if BASELINE_REFERENCE_FILE.is_file():

    with BASELINE_REFERENCE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        reference = json.load(
            f
        )


    old_baseline = reference[
        "baseline"
    ]


    for metric in [
        "accuracy",
        "macro_f1",
        "mean_true_class_probability",
    ]:

        difference = abs(
            baseline_metrics[
                metric
            ]
            - old_baseline[
                metric
            ]
        )


        if difference > 1e-6:

            raise RuntimeError(
                "\nBASELINE MISMATCH\n"
                f"Metric  : {metric}\n"
                f"Current : {baseline_metrics[metric]}\n"
                f"Old 1%  : {old_baseline[metric]}\n"
                "\nDo not compare 1% and 5% experiments "
                "until this is fixed."
            )


    print(
        "\nBaseline exactly matches "
        "the previous 1% experiment."
    )


with (
    OUTPUT_DIR
    / "baseline.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        baseline_metrics,
        f,
        indent=2,
    )


# ============================================================
# PART 5
# SELECTED LAYER-WISE MASKING
# ============================================================

print()
print("=" * 90)
print("5. SELECTED TOP-5% DEGREE-1 — LAYER-WISE")
print("=" * 90)


selected_layer_rows = []


for layer_idx in range(
    NUM_LAYERS
):

    neuron_ids = (
        selected_by_layer[
            layer_idx
        ]
    )


    print()
    print(
        f"Selected Layer "
        f"{layer_idx + 1:02d} | "
        f"neurons={len(neuron_ids)}"
    )


    if len(neuron_ids) == 0:

        selected_layer_rows.append(
            {
                "layer":
                    layer_idx + 1,

                "layer_index":
                    layer_idx,

                "num_masked_neurons":
                    0,

                "accuracy":
                    baseline_metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    baseline_metrics[
                        "macro_f1"
                    ],

                "mean_true_class_probability":
                    baseline_metrics[
                        "mean_true_class_probability"
                    ],

                "delta_accuracy":
                    0.0,

                "delta_macro_f1":
                    0.0,

                "accuracy_damage":
                    0.0,

                "macro_f1_damage":
                    0.0,

                "true_confidence_drop":
                    0.0,

                "baseline_prediction_confidence_drop":
                    0.0,

                "prediction_change_rate":
                    0.0,
            }
        )

        continue


    logits, run_labels = (
        run_evaluation(
            mask_by_layer={
                layer_idx:
                    neuron_ids
            },
            description=(
                f"Selected L{layer_idx + 1}"
            ),
        )
    )


    (
        metrics,
        probabilities,
        predictions,
    ) = compute_metrics(
        logits,
        run_labels,
    )


    effects = compare_to_baseline(
        baseline_metrics,
        baseline_probabilities,
        baseline_predictions,
        metrics,
        probabilities,
        predictions,
        labels,
    )


    row = {

        "layer":
            layer_idx + 1,

        "layer_index":
            layer_idx,

        "num_masked_neurons":
            len(neuron_ids),

        "accuracy":
            metrics[
                "accuracy"
            ],

        "macro_f1":
            metrics[
                "macro_f1"
            ],

        "mean_true_class_probability":
            metrics[
                "mean_true_class_probability"
            ],

        **effects,
    }


    selected_layer_rows.append(
        row
    )


    print(
        f"  ΔF1={effects['delta_macro_f1']:+.6f} "
        f"| F1 damage={effects['macro_f1_damage']:+.6f} "
        f"| true-P drop={effects['true_confidence_drop']:+.6f}"
    )


selected_layer_df = pd.DataFrame(
    selected_layer_rows
)


selected_layer_df.to_csv(
    OUTPUT_DIR
    / "selected_layerwise.csv",
    index=False,
)


# ============================================================
# PART 6
# SELECTED OVERALL MASKING
# ============================================================

print()
print("=" * 90)
print("6. SELECTED TOP-5% DEGREE-1 — ALL LAYERS AT ONCE")
print("=" * 90)


selected_overall_logits, selected_overall_labels = (
    run_evaluation(
        mask_by_layer=
            selected_by_layer,

        description=
            "Selected overall",
    )
)


(
    selected_overall_metrics,
    selected_overall_probabilities,
    selected_overall_predictions,
) = compute_metrics(
    selected_overall_logits,
    selected_overall_labels,
)


selected_overall_effects = (
    compare_to_baseline(
        baseline_metrics,
        baseline_probabilities,
        baseline_predictions,
        selected_overall_metrics,
        selected_overall_probabilities,
        selected_overall_predictions,
        labels,
    )
)


selected_overall_result = {

    "experiment":
        (
            "All exact-degree-1 top-5% "
            "neurons masked simultaneously"
        ),

    "top_percent":
        TOP_PERCENT,

    "top_k_per_layer_per_emotion":
        TOP_K,

    "num_masked_neurons":
        TOTAL_SELECTED,

    "num_layers":
        NUM_LAYERS,

    "baseline":
        baseline_metrics,

    "masked":
        selected_overall_metrics,

    "effect":
        selected_overall_effects,
}


with (
    OUTPUT_DIR
    / "selected_overall.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selected_overall_result,
        f,
        indent=2,
    )


# ============================================================
# SELECTED OVERALL PER CLASS
# ============================================================

per_class_rows = []


for emotion_idx, emotion in enumerate(
    EMOTIONS
):

    per_class_rows.append(
        {
            "emotion":
                emotion,

            "num_degree1_neurons":
                int(
                    (
                        degree1_df[
                            "emotion"
                        ]
                        == emotion
                    ).sum()
                ),

            "baseline_f1":
                baseline_metrics[
                    "f1"
                ][emotion_idx],

            "masked_f1":
                selected_overall_metrics[
                    "f1"
                ][emotion_idx],

            "delta_f1":
                (
                    selected_overall_metrics[
                        "f1"
                    ][emotion_idx]
                    - baseline_metrics[
                        "f1"
                    ][emotion_idx]
                ),

            "baseline_recall":
                baseline_metrics[
                    "recall"
                ][emotion_idx],

            "masked_recall":
                selected_overall_metrics[
                    "recall"
                ][emotion_idx],

            "delta_recall":
                (
                    selected_overall_metrics[
                        "recall"
                    ][emotion_idx]
                    - baseline_metrics[
                        "recall"
                    ][emotion_idx]
                ),

            "baseline_true_probability":
                baseline_metrics[
                    "per_class_true_probability"
                ][emotion_idx],

            "masked_true_probability":
                selected_overall_metrics[
                    "per_class_true_probability"
                ][emotion_idx],

            "true_confidence_drop":
                (
                    baseline_metrics[
                        "per_class_true_probability"
                    ][emotion_idx]
                    - selected_overall_metrics[
                        "per_class_true_probability"
                    ][emotion_idx]
                ),
        }
    )


pd.DataFrame(
    per_class_rows
).to_csv(
    OUTPUT_DIR
    / "selected_overall_per_class.csv",
    index=False,
)


print()
print(
    f"Selected neurons masked : "
    f"{TOTAL_SELECTED:,}"
)

print(
    f"Baseline Macro-F1       : "
    f"{baseline_metrics['macro_f1']:.6f}"
)

print(
    f"Masked Macro-F1         : "
    f"{selected_overall_metrics['macro_f1']:.6f}"
)

print(
    f"Δ Macro-F1              : "
    f"{selected_overall_effects['delta_macro_f1']:+.6f}"
)

print(
    f"True-confidence drop    : "
    f"{selected_overall_effects['true_confidence_drop']:+.6f}"
)


# ============================================================
# PART 7
# BUILD RANDOM POOLS
# ============================================================

print()
print("=" * 90)
print("7. BUILDING MATCHED RANDOM POOLS")
print("=" * 90)


all_neuron_ids = np.arange(
    MLP_WIDTH,
    dtype=np.int64,
)


random_pool_by_layer = {}


for layer_idx in range(
    NUM_LAYERS
):

    if args.random_pool == "non_top5":

        excluded = np.array(
            top5_union_by_layer[
                layer_idx
            ],
            dtype=np.int64,
        )

    else:

        excluded = np.array(
            selected_by_layer[
                layer_idx
            ],
            dtype=np.int64,
        )


    pool = np.setdiff1d(
        all_neuron_ids,
        excluded,
        assume_unique=False,
    )


    required = len(
        selected_by_layer[
            layer_idx
        ]
    )


    if len(pool) < required:

        raise RuntimeError(
            f"Layer {layer_idx + 1}: "
            f"pool={len(pool)}, "
            f"required={required}"
        )


    random_pool_by_layer[
        layer_idx
    ] = pool


    print(
        f"Layer {layer_idx + 1:02d}: "
        f"selected={required:4d} | "
        f"random pool={len(pool):5d}"
    )


# ============================================================
# RANDOM MASK GENERATOR
# ============================================================

def sample_random_mask_for_layer(
    layer_idx,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    count = len(
        selected_by_layer[
            layer_idx
        ]
    )


    if count == 0:

        return []


    sampled = rng.choice(
        random_pool_by_layer[
            layer_idx
        ],
        size=count,
        replace=False,
    )


    return sorted(
        sampled.astype(
            int
        ).tolist()
    )


def sample_random_overall_mask(
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    result = {}


    for layer_idx in range(
        NUM_LAYERS
    ):

        count = len(
            selected_by_layer[
                layer_idx
            ]
        )


        if count == 0:

            result[
                layer_idx
            ] = []

            continue


        sampled = rng.choice(
            random_pool_by_layer[
                layer_idx
            ],
            size=count,
            replace=False,
        )


        result[
            layer_idx
        ] = sorted(
            sampled.astype(
                int
            ).tolist()
        )


    return result


# ============================================================
# PART 8
# RANDOM LAYER-WISE CONTROL
# ============================================================

print()
print("=" * 90)
print(
    "8. RANDOM MATCHED CONTROL — LAYER-WISE"
)
print("=" * 90)


random_layer_rows = []


for layer_idx in range(
    NUM_LAYERS
):

    count = len(
        selected_by_layer[
            layer_idx
        ]
    )


    if count == 0:

        continue


    for run_idx in range(
        args.layer_random_runs
    ):

        seed = (
            args.random_seed
            + (
                layer_idx
                * 100000
            )
            + run_idx
        )


        random_ids = (
            sample_random_mask_for_layer(
                layer_idx,
                seed,
            )
        )


        print(
            f"Layer {layer_idx + 1:02d} "
            f"random {run_idx + 1}/"
            f"{args.layer_random_runs} "
            f"| n={count}"
        )


        logits, run_labels = (
            run_evaluation(
                mask_by_layer={
                    layer_idx:
                        random_ids
                },
                description=(
                    f"Rand L{layer_idx + 1} "
                    f"R{run_idx + 1}"
                ),
            )
        )


        (
            metrics,
            probabilities,
            predictions,
        ) = compute_metrics(
            logits,
            run_labels,
        )


        effects = compare_to_baseline(
            baseline_metrics,
            baseline_probabilities,
            baseline_predictions,
            metrics,
            probabilities,
            predictions,
            labels,
        )


        random_layer_rows.append(
            {
                "layer":
                    layer_idx + 1,

                "layer_index":
                    layer_idx,

                "run":
                    run_idx + 1,

                "seed":
                    seed,

                "num_masked_neurons":
                    count,

                "accuracy":
                    metrics[
                        "accuracy"
                    ],

                "macro_f1":
                    metrics[
                        "macro_f1"
                    ],

                "mean_true_class_probability":
                    metrics[
                        "mean_true_class_probability"
                    ],

                **effects,
            }
        )


random_layer_df = pd.DataFrame(
    random_layer_rows
)


random_layer_df.to_csv(
    OUTPUT_DIR
    / "random_layerwise_runs.csv",
    index=False,
)


# ============================================================
# RANDOM LAYER-WISE SUMMARY
# ============================================================

random_layer_summary_rows = []


for layer_idx in range(
    NUM_LAYERS
):

    selected_row = (
        selected_layer_df.loc[
            selected_layer_df[
                "layer_index"
            ] == layer_idx
        ]
    )


    if selected_row.empty:

        continue


    selected_row = (
        selected_row.iloc[0]
    )


    random_subset = (
        random_layer_df.loc[
            random_layer_df[
                "layer_index"
            ] == layer_idx
        ]
    )


    if random_subset.empty:

        continue


    selected_f1_damage = float(
        selected_row[
            "macro_f1_damage"
        ]
    )


    selected_conf_drop = float(
        selected_row[
            "true_confidence_drop"
        ]
    )


    random_f1_damage = (
        random_subset[
            "macro_f1_damage"
        ]
        .to_numpy(
            dtype=float
        )
    )


    random_conf_drop = (
        random_subset[
            "true_confidence_drop"
        ]
        .to_numpy(
            dtype=float
        )
    )


    p_f1 = (
        1
        + np.sum(
            random_f1_damage
            >= selected_f1_damage
        )
    ) / (
        len(
            random_f1_damage
        )
        + 1
    )


    p_conf = (
        1
        + np.sum(
            random_conf_drop
            >= selected_conf_drop
        )
    ) / (
        len(
            random_conf_drop
        )
        + 1
    )


    random_layer_summary_rows.append(
        {
            "layer":
                layer_idx + 1,

            "layer_index":
                layer_idx,

            "num_masked_neurons":
                int(
                    selected_row[
                        "num_masked_neurons"
                    ]
                ),

            "selected_macro_f1_damage":
                selected_f1_damage,

            "random_macro_f1_damage_mean":
                float(
                    random_f1_damage.mean()
                ),

            "random_macro_f1_damage_std":
                float(
                    random_f1_damage.std(
                        ddof=1
                    )
                )
                if len(
                    random_f1_damage
                ) > 1
                else 0.0,

            "excess_macro_f1_damage":
                float(
                    selected_f1_damage
                    - random_f1_damage.mean()
                ),

            "macro_f1_empirical_p":
                float(
                    p_f1
                ),

            "selected_true_confidence_drop":
                selected_conf_drop,

            "random_true_confidence_drop_mean":
                float(
                    random_conf_drop.mean()
                ),

            "random_true_confidence_drop_std":
                float(
                    random_conf_drop.std(
                        ddof=1
                    )
                )
                if len(
                    random_conf_drop
                ) > 1
                else 0.0,

            "excess_true_confidence_drop":
                float(
                    selected_conf_drop
                    - random_conf_drop.mean()
                ),

            "confidence_empirical_p":
                float(
                    p_conf
                ),
        }
    )


random_layer_summary_df = (
    pd.DataFrame(
        random_layer_summary_rows
    )
)


random_layer_summary_df.to_csv(
    OUTPUT_DIR
    / "random_layerwise_summary.csv",
    index=False,
)


# ============================================================
# PART 9
# RANDOM OVERALL CONTROL
# ============================================================

print()
print("=" * 90)
print(
    "9. RANDOM MATCHED CONTROL — ALL 28 LAYERS"
)
print("=" * 90)


random_overall_rows = []

random_overall_masks = {}


for run_idx in range(
    args.overall_random_runs
):

    seed = (
        args.random_seed
        + 10_000_000
        + run_idx
    )


    random_mask = (
        sample_random_overall_mask(
            seed
        )
    )


    random_overall_masks[
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
        f"Overall random "
        f"{run_idx + 1}/"
        f"{args.overall_random_runs} "
        f"| seed={seed}"
    )


    logits, run_labels = (
        run_evaluation(
            mask_by_layer=
                random_mask,

            description=(
                f"Overall rand "
                f"{run_idx + 1}"
            ),
        )
    )


    (
        metrics,
        probabilities,
        predictions,
    ) = compute_metrics(
        logits,
        run_labels,
    )


    effects = compare_to_baseline(
        baseline_metrics,
        baseline_probabilities,
        baseline_predictions,
        metrics,
        probabilities,
        predictions,
        labels,
    )


    random_overall_rows.append(
        {
            "run":
                run_idx + 1,

            "seed":
                seed,

            "num_masked_neurons":
                TOTAL_SELECTED,

            "accuracy":
                metrics[
                    "accuracy"
                ],

            "macro_f1":
                metrics[
                    "macro_f1"
                ],

            "mean_true_class_probability":
                metrics[
                    "mean_true_class_probability"
                ],

            **effects,
        }
    )


    print(
        f"  Macro-F1="
        f"{metrics['macro_f1']:.6f} "
        f"| damage="
        f"{effects['macro_f1_damage']:+.6f} "
        f"| true-P drop="
        f"{effects['true_confidence_drop']:+.6f}"
    )


random_overall_df = pd.DataFrame(
    random_overall_rows
)


random_overall_df.to_csv(
    OUTPUT_DIR
    / "random_overall_runs.csv",
    index=False,
)


with (
    OUTPUT_DIR
    / "random_overall_masks.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        random_overall_masks,
        f,
        indent=2,
    )


# ============================================================
# PART 10
# RANDOM OVERALL SUMMARY
# ============================================================

def stats(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )


    return {
        "mean":
            float(
                values.mean()
            ),

        "std":
            float(
                values.std(
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        "median":
            float(
                np.median(
                    values
                )
            ),

        "min":
            float(
                values.min()
            ),

        "max":
            float(
                values.max()
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


random_overall_f1_damage = (
    random_overall_df[
        "macro_f1_damage"
    ].to_numpy()
)


random_overall_accuracy_damage = (
    random_overall_df[
        "accuracy_damage"
    ].to_numpy()
)


random_overall_conf_drop = (
    random_overall_df[
        "true_confidence_drop"
    ].to_numpy()
)


selected_f1_damage = (
    selected_overall_effects[
        "macro_f1_damage"
    ]
)


selected_accuracy_damage = (
    selected_overall_effects[
        "accuracy_damage"
    ]
)


selected_conf_drop = (
    selected_overall_effects[
        "true_confidence_drop"
    ]
)


p_f1 = (
    1
    + np.sum(
        random_overall_f1_damage
        >= selected_f1_damage
    )
) / (
    args.overall_random_runs
    + 1
)


p_accuracy = (
    1
    + np.sum(
        random_overall_accuracy_damage
        >= selected_accuracy_damage
    )
) / (
    args.overall_random_runs
    + 1
)


p_conf = (
    1
    + np.sum(
        random_overall_conf_drop
        >= selected_conf_drop
    )
) / (
    args.overall_random_runs
    + 1
)


overall_random_summary = {

    "experiment":
        (
            "Top-5% exact-degree-1 selected neurons "
            "versus layer/count-matched random neurons"
        ),

    "top_percent":
        TOP_PERCENT,

    "top_k_per_layer_per_emotion":
        TOP_K,

    "num_selected_degree1_neurons":
        TOTAL_SELECTED,

    "random_pool":
        args.random_pool,

    "random_runs":
        args.overall_random_runs,

    "selected":
        {
            "accuracy_damage":
                selected_accuracy_damage,

            "macro_f1_damage":
                selected_f1_damage,

            "true_confidence_drop":
                selected_conf_drop,

            "prediction_change_rate":
                selected_overall_effects[
                    "prediction_change_rate"
                ],
        },

    "random": {

        "accuracy_damage":
            stats(
                random_overall_accuracy_damage
            ),

        "macro_f1_damage":
            stats(
                random_overall_f1_damage
            ),

        "true_confidence_drop":
            stats(
                random_overall_conf_drop
            ),

        "prediction_change_rate":
            stats(
                random_overall_df[
                    "prediction_change_rate"
                ].to_numpy()
            ),
    },

    "selected_minus_random_mean": {

        "accuracy_damage":
            float(
                selected_accuracy_damage
                - random_overall_accuracy_damage.mean()
            ),

        "macro_f1_damage":
            float(
                selected_f1_damage
                - random_overall_f1_damage.mean()
            ),

        "true_confidence_drop":
            float(
                selected_conf_drop
                - random_overall_conf_drop.mean()
            ),
    },

    "empirical_one_sided_p": {

        "accuracy_damage":
            float(
                p_accuracy
            ),

        "macro_f1_damage":
            float(
                p_f1
            ),

        "true_confidence_drop":
            float(
                p_conf
            ),
    },
}


with (
    OUTPUT_DIR
    / "random_overall_summary.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        overall_random_summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 90)
print("TOP-5% FINAL SUMMARY")
print("=" * 90)


print()
print(
    f"Top-k per layer/emotion       : "
    f"{TOP_K}"
)

print(
    f"Exact degree-1 neurons        : "
    f"{TOTAL_SELECTED:,}"
)


print()
print("--- SELECTED OVERALL ---")

print(
    f"Baseline Accuracy             : "
    f"{baseline_metrics['accuracy']:.6f}"
)

print(
    f"Selected Accuracy             : "
    f"{selected_overall_metrics['accuracy']:.6f}"
)

print(
    f"Accuracy damage               : "
    f"{selected_accuracy_damage:+.6f}"
)

print()

print(
    f"Baseline Macro-F1             : "
    f"{baseline_metrics['macro_f1']:.6f}"
)

print(
    f"Selected Macro-F1             : "
    f"{selected_overall_metrics['macro_f1']:.6f}"
)

print(
    f"Selected Macro-F1 damage      : "
    f"{selected_f1_damage:+.6f}"
)

print()

print(
    f"Selected true-confidence drop : "
    f"{selected_conf_drop:+.6f}"
)


print()
print("--- RANDOM OVERALL ---")

print(
    f"Random Macro-F1 damage mean   : "
    f"{random_overall_f1_damage.mean():+.6f}"
)

print(
    f"Random Macro-F1 damage std    : "
    f"{random_overall_f1_damage.std(ddof=1):.6f}"
)

print(
    f"Excess selected F1 damage     : "
    f"{selected_f1_damage - random_overall_f1_damage.mean():+.6f}"
)

print(
    f"Empirical F1 p                : "
    f"{p_f1:.6f}"
)

print()

print(
    f"Random confidence-drop mean   : "
    f"{random_overall_conf_drop.mean():+.6f}"
)

print(
    f"Selected confidence drop      : "
    f"{selected_conf_drop:+.6f}"
)

print(
    f"Excess selected conf. drop    : "
    f"{selected_conf_drop - random_overall_conf_drop.mean():+.6f}"
)

print(
    f"Empirical confidence p        : "
    f"{p_conf:.6f}"
)


print()
print("=" * 90)
print("SELECTED LAYER-WISE")
print("=" * 90)

print(
    selected_layer_df[
        [
            "layer",
            "num_masked_neurons",
            "macro_f1_damage",
            "true_confidence_drop",
            "prediction_change_rate",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("SELECTED VS RANDOM — LAYER-WISE")
print("=" * 90)

print(
    random_layer_summary_df[
        [
            "layer",
            "num_masked_neurons",
            "selected_macro_f1_damage",
            "random_macro_f1_damage_mean",
            "excess_macro_f1_damage",
            "macro_f1_empirical_p",
            "selected_true_confidence_drop",
            "random_true_confidence_drop_mean",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("OUTPUT DIRECTORY")
print("=" * 90)

print(
    OUTPUT_DIR
)

print()
print("Done.")
