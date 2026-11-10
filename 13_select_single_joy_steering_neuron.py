#!/usr/bin/env python3

"""
Select the best SINGLE joy steering neuron from the exact-degree-1
top-1% neuron set.

For every joy-specific degree-1 candidate:
    1. Capture continuous SwiGLU MLP activation m before down_proj.
    2. Measure activation on JOY examples.
    3. Measure activation on SADNESS examples.
    4. Compute:
           mean_joy
           mean_sadness
           pooled std
           standardized joy-vs-sadness contrast
    5. Rank candidates.
    6. Select the strongest positive candidate.
    7. Extract its residual-stream direction:
           W_down[:, neuron]

No steering is performed in this script.
This script only selects and characterizes ONE neuron.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
)

parser.add_argument(
    "--max-per-class",
    type=int,
    default=0,
    help=(
        "0 = use all joy and sadness examples. "
        "Set e.g. 500 for a quick test."
    ),
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "mlt_hinghlish_dataset_clean.csv"
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
    / "llama31_8b_steering"
    / "13_single_joy_neuron_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

TEXT_COLUMN = "hinglish_genz_text"
LABEL_COLUMN = "label"

TARGET_EMOTION = "joy"
SOURCE_EMOTION = "sadness"

NUM_LAYERS = 32
MLP_WIDTH = 14336

MAX_LENGTH = 256

EPS = 1e-12

SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

else:

    DEVICE = torch.device("cpu")
    DTYPE = torch.float32


print("=" * 90)
print("SINGLE JOY STEERING NEURON SELECTION")
print("=" * 90)

print(f"Model          : {MODEL_ID}")
print(f"Target emotion : {TARGET_EMOTION}")
print(f"Source emotion : {SOURCE_EMOTION}")
print(f"Device         : {DEVICE}")
print(f"Dtype          : {DTYPE}")


# ============================================================
# CHECK FILES
# ============================================================

for path in [
    DATA_FILE,
    SINGLE_EMOTION_FILE,
]:

    if not path.is_file():

        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


# ============================================================
# LOAD DEGREE-1 CANDIDATES
# ============================================================

print()
print("=" * 90)
print("1. LOADING JOY DEGREE-1 CANDIDATES")
print("=" * 90)


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


joy_candidates = (
    single_df.loc[
        single_df["emotion"]
        == TARGET_EMOTION
    ]
    .copy()
    .reset_index(drop=True)
)


if joy_candidates.empty:

    raise RuntimeError(
        "No joy degree-1 neurons were found."
    )


print(
    f"Joy degree-1 candidates: "
    f"{len(joy_candidates)}"
)


print()

display_columns = [
    "layer",
    "layer_index",
    "neuron_index",
]


if "selectivity_score" in (
    joy_candidates.columns
):

    display_columns.append(
        "selectivity_score"
    )


print(
    joy_candidates[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# GROUP CANDIDATES BY LAYER
# ============================================================

candidates_by_layer = {}


for layer_idx, group in (
    joy_candidates
    .groupby("layer_index")
):

    candidates_by_layer[
        int(layer_idx)
    ] = sorted(
        group[
            "neuron_index"
        ]
        .astype(int)
        .tolist()
    )


print()
print(
    f"Candidate layers: "
    f"{len(candidates_by_layer)}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 90)
print("2. LOADING JOY + SADNESS DATA")
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
    .reset_index(drop=True)
)


joy_df = (
    df.loc[
        df[LABEL_COLUMN]
        == TARGET_EMOTION
    ]
    .copy()
    .reset_index(drop=True)
)


sad_df = (
    df.loc[
        df[LABEL_COLUMN]
        == SOURCE_EMOTION
    ]
    .copy()
    .reset_index(drop=True)
)


# Optional fast-test subsampling.
if args.max_per_class > 0:

    n_joy = min(
        args.max_per_class,
        len(joy_df),
    )

    n_sad = min(
        args.max_per_class,
        len(sad_df),
    )


    joy_df = (
        joy_df.sample(
            n=n_joy,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )


    sad_df = (
        sad_df.sample(
            n=n_sad,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )


print(
    f"Joy examples     : "
    f"{len(joy_df):,}"
)

print(
    f"Sadness examples : "
    f"{len(sad_df):,}"
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 90)
print("3. LOADING TOKENIZER")
print("=" * 90)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    "Special token IDs excluded:",
    tokenizer.all_special_ids,
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 90)
print("4. LOADING SARVAM-1")
print("=" * 90)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
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


backbone = model.model


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
# CAPTURE CONTEXT
# ============================================================

capture_context = {

    # [B, T]
    "valid_mask": None,

    # Temporary captures for current batch.
    #
    # layer_idx ->
    # {
    #   "mean": [B, K],
    #   "positive_fraction": [B, K],
    # }
    "batch_capture": {},
}


# ============================================================
# HOOK FACTORY
# ============================================================

def make_capture_hook(
    layer_idx,
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

        if not inputs:

            return inputs


        x = inputs[0]

        # x:
        # [batch, sequence, 11008]
        #
        # This is the SwiGLU intermediate m
        # immediately BEFORE down_proj.


        valid_mask = (
            capture_context[
                "valid_mask"
            ]
        )


        if valid_mask is None:

            raise RuntimeError(
                "Capture hook called without "
                "valid token mask."
            )


        key = str(
            x.device
        )


        if key not in index_cache:

            index_cache[
                key
            ] = torch.tensor(
                neuron_indices,
                dtype=torch.long,
                device=x.device,
            )


        indices = index_cache[
            key
        ]


        values = x[
            :,
            :,
            indices,
        ].float()


        # [B,T,1]
        mask = (
            valid_mask
            .to(x.device)
            .unsqueeze(-1)
        )


        mask_float = mask.to(
            values.dtype
        )


        denominator = (
            mask_float
            .sum(dim=1)
            .clamp_min(1.0)
        )


        # ----------------------------------------------------
        # Mean continuous activation per example
        # ----------------------------------------------------

        example_mean = (
            (
                values
                * mask_float
            )
            .sum(dim=1)
            / denominator
        )


        # ----------------------------------------------------
        # Fraction of valid tokens where activation > 0
        #
        # Useful because original selectivity used:
        #     1[m > 0]
        # ----------------------------------------------------

        positive = (
            values > 0
        ).to(
            values.dtype
        )


        positive_fraction = (
            (
                positive
                * mask_float
            )
            .sum(dim=1)
            / denominator
        )


        capture_context[
            "batch_capture"
        ][layer_idx] = {

            "mean":
                example_mean
                .detach()
                .cpu(),

            "positive_fraction":
                positive_fraction
                .detach()
                .cpu(),
        }


        # DO NOT modify activation.
        return inputs


    return hook


# ============================================================
# REGISTER CANDIDATE-LAYER HOOKS
# ============================================================

handles = []


for (
    layer_idx,
    neuron_indices
) in candidates_by_layer.items():

    if layer_idx < 0 or layer_idx >= NUM_LAYERS:

        raise ValueError(
            f"Invalid layer index: "
            f"{layer_idx}"
        )


    handle = (
        layers[
            layer_idx
        ]
        .mlp
        .down_proj
        .register_forward_pre_hook(
            make_capture_hook(
                layer_idx,
                neuron_indices,
            )
        )
    )


    handles.append(
        handle
    )


print(
    f"Registered {len(handles)} "
    f"candidate-layer hooks."
)


# ============================================================
# STORAGE
# ============================================================

#
# key:
#     (layer_idx, neuron_idx)
#
# value:
# {
#     "joy_mean": [],
#     "sad_mean": [],
#     "joy_positive_fraction": [],
#     "sad_positive_fraction": []
# }
#

candidate_storage = {}


for _, row in (
    joy_candidates.iterrows()
):

    key = (
        int(
            row[
                "layer_index"
            ]
        ),
        int(
            row[
                "neuron_index"
            ]
        ),
    )


    candidate_storage[
        key
    ] = {

        "joy_mean": [],

        "sad_mean": [],

        "joy_positive_fraction": [],

        "sad_positive_fraction": [],
    }


# ============================================================
# DATA EXTRACTION FUNCTION
# ============================================================

def process_class(
    class_df,
    class_name,
):

    if class_name not in [
        TARGET_EMOTION,
        SOURCE_EMOTION,
    ]:

        raise ValueError(
            f"Unexpected class: "
            f"{class_name}"
        )


    if class_name == TARGET_EMOTION:

        mean_key = "joy_mean"

        positive_key = (
            "joy_positive_fraction"
        )

    else:

        mean_key = "sad_mean"

        positive_key = (
            "sad_positive_fraction"
        )


    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                len(class_df),
                args.batch_size,
            ),
            desc=class_name,
        ):

            batch = class_df.iloc[
                start:
                start + args.batch_size
            ]


            texts = (
                batch[
                    TEXT_COLUMN
                ]
                .tolist()
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


            capture_context[
                "valid_mask"
            ] = valid_mask


            capture_context[
                "batch_capture"
            ] = {}


            # Forward pass.
            #
            # Hooks collect all candidate activations.
            _ = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )


            # ------------------------------------------------
            # Read captures
            # ------------------------------------------------

            for (
                layer_idx,
                neuron_indices
            ) in candidates_by_layer.items():

                if (
                    layer_idx
                    not in capture_context[
                        "batch_capture"
                    ]
                ):

                    raise RuntimeError(
                        f"No capture for "
                        f"layer {layer_idx}"
                    )


                captured = (
                    capture_context[
                        "batch_capture"
                    ][layer_idx]
                )


                means = (
                    captured[
                        "mean"
                    ]
                    .numpy()
                )


                pos_fracs = (
                    captured[
                        "positive_fraction"
                    ]
                    .numpy()
                )


                if means.shape[1] != len(
                    neuron_indices
                ):

                    raise RuntimeError(
                        "Candidate dimension mismatch."
                    )


                for local_idx, neuron_idx in enumerate(
                    neuron_indices
                ):

                    key = (
                        layer_idx,
                        neuron_idx,
                    )


                    candidate_storage[
                        key
                    ][
                        mean_key
                    ].extend(
                        means[
                            :,
                            local_idx
                        ].tolist()
                    )


                    candidate_storage[
                        key
                    ][
                        positive_key
                    ].extend(
                        pos_fracs[
                            :,
                            local_idx
                        ].tolist()
                    )


# ============================================================
# EXTRACT BOTH CLASSES
# ============================================================

print()
print("=" * 90)
print("5. EXTRACTING CONTINUOUS ACTIVATIONS")
print("=" * 90)


try:

    process_class(
        joy_df,
        TARGET_EMOTION,
    )


    process_class(
        sad_df,
        SOURCE_EMOTION,
    )

finally:

    for handle in handles:

        handle.remove()


    capture_context[
        "valid_mask"
    ] = None


print()
print("Activation extraction complete.")


# ============================================================
# COMPUTE STATISTICS
# ============================================================

print()
print("=" * 90)
print("6. COMPUTING JOY-vs-SADNESS CONTRAST")
print("=" * 90)


result_rows = []


for _, candidate_row in (
    joy_candidates.iterrows()
):

    layer_idx = int(
        candidate_row[
            "layer_index"
        ]
    )

    neuron_idx = int(
        candidate_row[
            "neuron_index"
        ]
    )


    key = (
        layer_idx,
        neuron_idx,
    )


    data = candidate_storage[
        key
    ]


    joy_values = np.asarray(
        data[
            "joy_mean"
        ],
        dtype=np.float64,
    )


    sad_values = np.asarray(
        data[
            "sad_mean"
        ],
        dtype=np.float64,
    )


    joy_positive = np.asarray(
        data[
            "joy_positive_fraction"
        ],
        dtype=np.float64,
    )


    sad_positive = np.asarray(
        data[
            "sad_positive_fraction"
        ],
        dtype=np.float64,
    )


    if (
        len(joy_values) == 0
        or len(sad_values) == 0
    ):

        raise RuntimeError(
            f"No values for "
            f"layer={layer_idx}, "
            f"neuron={neuron_idx}"
        )


    # --------------------------------------------------------
    # Means
    # --------------------------------------------------------

    mean_joy = float(
        joy_values.mean()
    )

    mean_sad = float(
        sad_values.mean()
    )


    raw_difference = (
        mean_joy
        - mean_sad
    )


    # --------------------------------------------------------
    # Within-class SDs
    # --------------------------------------------------------

    sd_joy = float(
        joy_values.std(
            ddof=1
        )
    )


    sd_sad = float(
        sad_values.std(
            ddof=1
        )
    )


    n_joy = len(
        joy_values
    )

    n_sad = len(
        sad_values
    )


    # --------------------------------------------------------
    # Pooled within-class SD
    #
    # sqrt(
    #     ((nJ-1)sJ² + (nS-1)sS²)
    #     / (nJ+nS-2)
    # )
    # --------------------------------------------------------

    pooled_variance = (
        (
            (n_joy - 1)
            * (sd_joy ** 2)
        )
        +
        (
            (n_sad - 1)
            * (sd_sad ** 2)
        )
    ) / max(
        n_joy + n_sad - 2,
        1,
    )


    pooled_sd = float(
        np.sqrt(
            max(
                pooled_variance,
                0.0,
            )
        )
    )


    standardized_difference = (
        raw_difference
        / (
            pooled_sd
            + EPS
        )
    )


    # --------------------------------------------------------
    # Binary-positive statistics
    #
    # Helps connect continuous analysis back to original
    # selectivity definition.
    # --------------------------------------------------------

    joy_positive_rate = float(
        joy_positive.mean()
    )


    sad_positive_rate = float(
        sad_positive.mean()
    )


    positive_rate_difference = (
        joy_positive_rate
        - sad_positive_rate
    )


    result = {

        "layer":
            layer_idx + 1,

        "layer_index":
            layer_idx,

        "neuron_index":
            neuron_idx,

        "neuron_number":
            neuron_idx + 1,

        "n_joy":
            n_joy,

        "n_sadness":
            n_sad,

        "mean_activation_joy":
            mean_joy,

        "mean_activation_sadness":
            mean_sad,

        "raw_activation_difference":
            raw_difference,

        "sd_joy":
            sd_joy,

        "sd_sadness":
            sd_sad,

        "pooled_sd":
            pooled_sd,

        # Main selection statistic.
        "standardized_joy_minus_sadness":
            standardized_difference,

        "positive_rate_joy":
            joy_positive_rate,

        "positive_rate_sadness":
            sad_positive_rate,

        "positive_rate_difference":
            positive_rate_difference,
    }


    if (
        "selectivity_score"
        in candidate_row.index
    ):

        result[
            "original_selectivity_score"
        ] = float(
            candidate_row[
                "selectivity_score"
            ]
        )


    if "rank" in candidate_row.index:

        result[
            "original_rank"
        ] = int(
            candidate_row[
                "rank"
            ]
        )


    result_rows.append(
        result
    )


results_df = pd.DataFrame(
    result_rows
)


# ============================================================
# RANK
# ============================================================

results_df = (
    results_df
    .sort_values(
        by=[
            "standardized_joy_minus_sadness",
            "raw_activation_difference",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


results_df.insert(
    0,
    "steering_rank",
    np.arange(
        1,
        len(results_df) + 1,
    ),
)


# ============================================================
# SELECT BEST CANDIDATE
# ============================================================

positive_candidates = (
    results_df.loc[
        results_df[
            "standardized_joy_minus_sadness"
        ] > 0
    ]
)


if len(
    positive_candidates
) > 0:

    best = (
        positive_candidates
        .iloc[0]
    )

    selection_reason = (
        "largest positive standardized "
        "joy-minus-sadness activation contrast"
    )

else:

    best = (
        results_df.iloc[0]
    )

    selection_reason = (
        "no candidate had positive contrast; "
        "selected maximum standardized contrast"
    )


BEST_LAYER_IDX = int(
    best[
        "layer_index"
    ]
)


BEST_LAYER_NUMBER = int(
    best[
        "layer"
    ]
)


BEST_NEURON_IDX = int(
    best[
        "neuron_index"
    ]
)


BEST_NEURON_NUMBER = int(
    best[
        "neuron_number"
    ]
)


BEST_DIRECTION_SCALAR = float(
    best[
        "standardized_joy_minus_sadness"
    ]
)


# ============================================================
# EXTRACT DOWN-PROJECTION VECTOR
# ============================================================

print()
print("=" * 90)
print("7. EXTRACTING RESIDUAL-STREAM VECTOR DIRECTION")
print("=" * 90)


down_weight = (
    layers[
        BEST_LAYER_IDX
    ]
    .mlp
    .down_proj
    .weight
    .detach()
    .float()
    .cpu()
)


print(
    f"down_proj weight shape: "
    f"{tuple(down_weight.shape)}"
)


if (
    BEST_NEURON_IDX
    >= down_weight.shape[1]
):

    raise RuntimeError(
        "Neuron index exceeds "
        "down_proj input dimension."
    )


# W_down shape:
#
#     [hidden_size, intermediate_size]
#
# Therefore one MLP neuron contributes in the direction:
#
#     W_down[:, neuron]
#

residual_direction = (
    down_weight[
        :,
        BEST_NEURON_IDX
    ]
    .numpy()
    .astype(
        np.float32
    )
)


direction_norm = float(
    np.linalg.norm(
        residual_direction
    )
)


unit_direction = (
    residual_direction
    / (
        direction_norm
        + EPS
    )
).astype(
    np.float32
)


# ============================================================
# SAVE
# ============================================================

results_file = (
    OUTPUT_DIR
    / "joy_candidate_activation_statistics.csv"
)


results_df.to_csv(
    results_file,
    index=False,
)


direction_file = (
    OUTPUT_DIR
    / "best_joy_neuron_residual_direction.npy"
)


unit_direction_file = (
    OUTPUT_DIR
    / "best_joy_neuron_residual_direction_unit.npy"
)


np.save(
    direction_file,
    residual_direction,
)


np.save(
    unit_direction_file,
    unit_direction,
)


metadata = {

    "experiment":
        (
            "Select one top-1% exact-degree-1 joy neuron "
            "for single-neuron steering"
        ),

    "target_emotion":
        TARGET_EMOTION,

    "source_emotion":
        SOURCE_EMOTION,

    "num_candidates":
        int(
            len(
                joy_candidates
            )
        ),

    "num_joy_examples":
        int(
            len(
                joy_df
            )
        ),

    "num_sadness_examples":
        int(
            len(
                sad_df
            )
        ),

    "selection_reason":
        selection_reason,

    "best_neuron": {

        "layer":
            BEST_LAYER_NUMBER,

        "layer_index":
            BEST_LAYER_IDX,

        "neuron_index":
            BEST_NEURON_IDX,

        "neuron_number":
            BEST_NEURON_NUMBER,

        "mean_activation_joy":
            float(
                best[
                    "mean_activation_joy"
                ]
            ),

        "mean_activation_sadness":
            float(
                best[
                    "mean_activation_sadness"
                ]
            ),

        "raw_activation_difference":
            float(
                best[
                    "raw_activation_difference"
                ]
            ),

        "pooled_sd":
            float(
                best[
                    "pooled_sd"
                ]
            ),

        "standardized_joy_minus_sadness":
            BEST_DIRECTION_SCALAR,

        "positive_rate_joy":
            float(
                best[
                    "positive_rate_joy"
                ]
            ),

        "positive_rate_sadness":
            float(
                best[
                    "positive_rate_sadness"
                ]
            ),

        "positive_rate_difference":
            float(
                best[
                    "positive_rate_difference"
                ]
            ),

        "residual_direction_norm":
            direction_norm,

        "residual_direction_dimension":
            int(
                residual_direction.shape[0]
            ),
    },

    "steering_definition": {

        "activation_intervention":
            (
                "m' = m + alpha * d"
            ),

        "d":
            BEST_DIRECTION_SCALAR,

        "residual_effect":
            (
                "Delta_h = alpha * d * "
                "W_down[:, neuron_index]"
            ),
    },

    "files": {

        "candidate_statistics":
            str(
                results_file
            ),

        "residual_direction":
            str(
                direction_file
            ),

        "unit_residual_direction":
            str(
                unit_direction_file
            ),
    },
}


metadata_file = (
    OUTPUT_DIR
    / "best_joy_neuron.json"
)


with metadata_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 90)
print("JOY CANDIDATES RANKED FOR STEERING")
print("=" * 90)


columns_to_print = [

    "steering_rank",

    "layer",

    "neuron_index",

    "mean_activation_joy",

    "mean_activation_sadness",

    "raw_activation_difference",

    "pooled_sd",

    "standardized_joy_minus_sadness",

    "positive_rate_joy",

    "positive_rate_sadness",
]


print(
    results_df[
        columns_to_print
    ].to_string(
        index=False
    )
)


print()
print("=" * 90)
print("SELECTED SINGLE JOY NEURON")
print("=" * 90)


print(
    f"Layer                  : "
    f"{BEST_LAYER_NUMBER}"
)

print(
    f"Layer index            : "
    f"{BEST_LAYER_IDX}"
)

print(
    f"Neuron index           : "
    f"{BEST_NEURON_IDX}"
)

print(
    f"Neuron number          : "
    f"{BEST_NEURON_NUMBER}"
)

print()

print(
    f"Mean activation JOY    : "
    f"{best['mean_activation_joy']:+.8f}"
)

print(
    f"Mean activation SAD    : "
    f"{best['mean_activation_sadness']:+.8f}"
)

print(
    f"Raw JOY-SAD difference : "
    f"{best['raw_activation_difference']:+.8f}"
)

print(
    f"Pooled SD              : "
    f"{best['pooled_sd']:.8f}"
)

print()

print(
    "STANDARDIZED DIRECTION  : "
    f"{BEST_DIRECTION_SCALAR:+.8f}"
)

print()

print(
    f"P(m>0) JOY             : "
    f"{best['positive_rate_joy']:.6f}"
)

print(
    f"P(m>0) SADNESS         : "
    f"{best['positive_rate_sadness']:.6f}"
)

print(
    f"Positive-rate difference: "
    f"{best['positive_rate_difference']:+.6f}"
)

print()

print(
    f"W_down direction dim   : "
    f"{residual_direction.shape[0]}"
)

print(
    f"W_down direction norm  : "
    f"{direction_norm:.8f}"
)


print()
print("Steering rule for next experiment:")
print()

print(
    "    m' = m + alpha * d"
)

print()

print(
    f"where d = "
    f"{BEST_DIRECTION_SCALAR:+.8f}"
)

print()

print(
    "and the resulting residual change is:"
)

print()

print(
    "    Delta_h = alpha * d * "
    "W_down[:, neuron]"
)


print()
print("=" * 90)
print("SAVED")
print("=" * 90)

print(results_file)
print(metadata_file)
print(direction_file)
print(unit_direction_file)

print()
print("Done.")
