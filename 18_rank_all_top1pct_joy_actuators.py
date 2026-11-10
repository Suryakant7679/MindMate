#!/usr/bin/env python3

"""
Rank ALL layerwise top-1% Joy neurons as possible steering actuators.

We no longer restrict actuator discovery to exact-degree-1 neurons.

Candidate pool:
    all neurons appearing in the layerwise top-1% Joy selection.

For Sarvam-1:
    28 layers
    110 Joy-selected neurons/layer
    expected total Joy candidate entries = 28 * 110 = 3,080

For each candidate neuron u at layer l:

    Delta_mu_u
        = mean_activation_joy
        - mean_activation_sadness

    v_u
        = W_down_l[:, u]

    d_l
        = mean_residual_joy_l
        - mean_residual_sadness_l

    write_projection_u
        = v_u^T unit(d_l)

Primary actuator score:

    Q_u
        = Delta_mu_u * write_projection_u

Equivalently:

    Delta_h(alpha=1)
        = Delta_mu_u * v_u

    Q_u
        = Delta_h(alpha=1)^T unit(d_l)

Interpretation:

    Q_u > 0:
        positive steering moves toward Joy relative to Sadness.

    Q_u < 0:
        positive steering moves away from Joy.

    Q_u ~= 0:
        steering is approximately orthogonal to the Joy-Sadness axis.

The script also computes:

    - continuous Joy/Sadness activation means
    - SDs
    - pooled SD
    - standardized Joy-Sadness activation contrast
    - positive-activation rates
    - W_down norm
    - W_down cosine with Joy-Sadness direction
    - expected residual shift norm at alpha=1
    - actual intervention-direction cosine
    - top-1% cross-emotion membership degree

Outputs:

    results/sarvam1_steering/18_all_top1pct_joy_actuators/

        all_top1pct_joy_actuator_ranking.csv
        top_joy_actuators.csv
        all_top1pct_joy_actuator_ranking.json
        best_joy_actuator_neuron.json

        best_joy_actuator_write_direction.npy
        best_joy_actuator_write_direction_unit.npy

        layer_direction_summary.csv

        layer_directions/
            layer_01_joy_minus_sad.npy
            layer_01_joy_minus_sad_unit.npy
            ...
            layer_28_joy_minus_sad.npy
            layer_28_joy_minus_sad_unit.npy

        joy_actuator_ranking.png

IMPORTANT:
    This is an exploratory discovery analysis using all Joy/Sadness data.
    For publication-quality evaluation, candidate discovery should later
    be repeated using training data only and evaluated on held-out data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    "--max-length",
    type=int,
    default=256,
)


parser.add_argument(
    "--max-per-class",
    type=int,
    default=0,
    help=(
        "0 = use all Joy/Sadness examples. "
        "Use 200 for a quick sanity test."
    ),
)


parser.add_argument(
    "--top-k-next",
    type=int,
    default=20,
    help=(
        "Number of highest positive-Q candidates "
        "saved separately for later steering tests."
    ),
)


parser.add_argument(
    "--plot-top-n",
    type=int,
    default=30,
)


parser.add_argument(
    "--seed",
    type=int,
    default=42,
)


parser.add_argument(
    "--local-files-only",
    action="store_true",
    help=(
        "Use only locally cached Hugging Face files."
    ),
)


args = parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

TEXT_COLUMN = "hinglish_genz_text"

LABEL_COLUMN = "label"

TARGET_EMOTION = "joy"

SOURCE_EMOTION = "sadness"

EPS = 1e-12


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATA_FILE = (
    BASE_DIR
    / "data"
    / "mlt_hinghlish_dataset_clean.csv"
)


TOP1_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_layerwise_top1pct"
    / "all_layers_top1pct.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "18_all_top1pct_joy_actuators"
)


DIRECTION_DIR = (
    OUTPUT_DIR
    / "layer_directions"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DIRECTION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device(
        "cuda"
    )

    DTYPE = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

else:

    DEVICE = torch.device(
        "cpu"
    )

    DTYPE = torch.float32


print("=" * 105)
print("RANK ALL TOP-1% JOY NEURONS AS STEERING ACTUATORS")
print("=" * 105)

print(
    f"Device                  : {DEVICE}"
)

print(
    f"Dtype                   : {DTYPE}"
)

print(
    f"Model                   : {MODEL_ID}"
)


# ============================================================
# INPUT CHECKS
# ============================================================

if not DATA_FILE.is_file():

    raise FileNotFoundError(
        f"Dataset missing:\n"
        f"{DATA_FILE}"
    )


if not TOP1_FILE.is_file():

    raise FileNotFoundError(
        f"Top-1% selection file missing:\n"
        f"{TOP1_FILE}"
    )


# ============================================================
# HELPER: FIND OPTIONAL COLUMN
# ============================================================

def first_existing_column(
    dataframe,
    possibilities,
):

    for column in possibilities:

        if column in dataframe.columns:

            return column

    return None


# ============================================================
# LOAD ALL TOP-1% SELECTIONS
# ============================================================

print()
print("=" * 105)
print("LOADING TOP-1% SELECTIONS")
print("=" * 105)


all_top1 = pd.read_csv(
    TOP1_FILE
)


required_columns = [

    "layer_index",

    "neuron_index",

    "emotion",
]


for column in required_columns:

    if column not in all_top1.columns:

        raise RuntimeError(
            f"Missing required column "
            f"in top-1% CSV: {column}"
        )


# ------------------------------------------------------------
# Normalize important types.
# ------------------------------------------------------------

all_top1[
    "layer_index"
] = (
    all_top1[
        "layer_index"
    ]
    .astype(int)
)


all_top1[
    "neuron_index"
] = (
    all_top1[
        "neuron_index"
    ]
    .astype(int)
)


all_top1[
    "emotion"
] = (
    all_top1[
        "emotion"
    ]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Optional metadata columns.
# ------------------------------------------------------------

rank_column = first_existing_column(
    all_top1,
    [
        "rank",
        "emotion_rank",
        "rank_within_layer_emotion",
        "selection_rank",
    ],
)


score_column = first_existing_column(
    all_top1,
    [
        "selectivity_score",
        "selectivity",
        "score",
    ],
)


# ============================================================
# MEMBERSHIP DEGREE
# ============================================================

membership = (
    all_top1
    .groupby(
        [
            "layer_index",
            "neuron_index",
        ],
        as_index=False,
    )
    .agg(
        membership_degree_top1pct=(
            "emotion",
            "nunique",
        )
    )
)


# ============================================================
# FILTER JOY CANDIDATES
# ============================================================

joy_candidates = (
    all_top1.loc[
        all_top1[
            "emotion"
        ]
        == TARGET_EMOTION
    ]
    .copy()
)


# Remove any accidental duplicates.
joy_candidates = (
    joy_candidates
    .sort_values(
        [
            "layer_index",
            "neuron_index",
        ]
    )
    .drop_duplicates(
        subset=[
            "layer_index",
            "neuron_index",
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


joy_candidates = (
    joy_candidates
    .merge(
        membership,
        on=[
            "layer_index",
            "neuron_index",
        ],
        how="left",
    )
)


joy_candidates[
    "is_exact_degree1"
] = (
    joy_candidates[
        "membership_degree_top1pct"
    ]
    == 1
)


if "layer" not in joy_candidates.columns:

    joy_candidates[
        "layer"
    ] = (
        joy_candidates[
            "layer_index"
        ]
        + 1
    )


if "neuron_number" not in joy_candidates.columns:

    joy_candidates[
        "neuron_number"
    ] = (
        joy_candidates[
            "neuron_index"
        ]
        + 1
    )


if rank_column is not None:

    joy_candidates[
        "original_joy_rank"
    ] = (
        joy_candidates[
            rank_column
        ]
    )

else:

    joy_candidates[
        "original_joy_rank"
    ] = np.nan


if score_column is not None:

    joy_candidates[
        "original_selectivity_score"
    ] = (
        joy_candidates[
            score_column
        ]
    )

else:

    joy_candidates[
        "original_selectivity_score"
    ] = np.nan


print(
    f"All top-1% entries       : "
    f"{len(all_top1):,}"
)

print(
    f"Joy candidates           : "
    f"{len(joy_candidates):,}"
)

print(
    f"Exact-degree-1 Joy       : "
    f"{int(joy_candidates['is_exact_degree1'].sum()):,}"
)


if len(
    joy_candidates
) != 3080:

    print()
    print(
        "WARNING:"
    )

    print(
        "Expected 3,080 Joy entries for "
        "28 layers x 110 neurons."
    )

    print(
        f"Observed: {len(joy_candidates):,}"
    )


# ============================================================
# BUILD CANDIDATE TABLE PER LAYER
# ============================================================

candidate_tables = {}


for layer_idx in sorted(
    joy_candidates[
        "layer_index"
    ]
    .unique()
    .tolist()
):

    subset = (
        joy_candidates.loc[
            joy_candidates[
                "layer_index"
            ]
            == layer_idx
        ]
        .copy()
    )


    if rank_column is not None:

        subset = (
            subset
            .sort_values(
                "original_joy_rank"
            )
        )

    else:

        subset = (
            subset
            .sort_values(
                "neuron_index"
            )
        )


    subset = (
        subset
        .reset_index(
            drop=True
        )
    )


    candidate_tables[
        int(
            layer_idx
        )
    ] = subset


candidate_layer_indices = sorted(
    candidate_tables.keys()
)


print(
    f"Candidate layers         : "
    f"{len(candidate_layer_indices)}"
)


print(
    "Candidates/layer        : "
    + ", ".join(
        (
            f"L{layer_idx + 1}:"
            f"{len(candidate_tables[layer_idx])}"
        )

        for layer_idx
        in candidate_layer_indices
    )
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 105)
print("LOADING DATASET")
print("=" * 105)


df = pd.read_csv(
    DATA_FILE
)


for column in [
    TEXT_COLUMN,
    LABEL_COLUMN,
]:

    if column not in df.columns:

        raise RuntimeError(
            f"Dataset missing column: "
            f"{column}"
        )


df = (
    df[
        [
            TEXT_COLUMN,
            LABEL_COLUMN,
        ]
    ]
    .dropna()
    .copy()
)


df[
    TEXT_COLUMN
] = (
    df[
        TEXT_COLUMN
    ]
    .astype(str)
    .str.strip()
)


df[
    LABEL_COLUMN
] = (
    df[
        LABEL_COLUMN
    ]
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


joy_df = (
    df.loc[
        df[
            LABEL_COLUMN
        ]
        == TARGET_EMOTION
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


sad_df = (
    df.loc[
        df[
            LABEL_COLUMN
        ]
        == SOURCE_EMOTION
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


if args.max_per_class > 0:

    joy_df = (
        joy_df.sample(
            n=min(
                args.max_per_class,
                len(
                    joy_df
                ),
            ),
            random_state=
                args.seed,
        )
        .reset_index(
            drop=True
        )
    )


    sad_df = (
        sad_df.sample(
            n=min(
                args.max_per_class,
                len(
                    sad_df
                ),
            ),
            random_state=
                args.seed,
        )
        .reset_index(
            drop=True
        )
    )


print(
    f"Joy examples             : "
    f"{len(joy_df):,}"
)

print(
    f"Sadness examples         : "
    f"{len(sad_df):,}"
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 105)
print("LOADING TOKENIZER")
print("=" * 105)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,

    trust_remote_code=True,

    local_files_only=
        args.local_files_only,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    f"Special IDs              : "
    f"{tokenizer.all_special_ids}"
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 105)
print("LOADING SARVAM-1")
print("=" * 105)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,

    dtype=DTYPE,

    trust_remote_code=True,

    local_files_only=
        args.local_files_only,
)


model.to(
    DEVICE
)


model.eval()


for parameter in model.parameters():

    parameter.requires_grad_(
        False
    )


layers = (
    model.model.layers
)


backbone = (
    model.model
)


NUM_LAYERS = len(
    layers
)


HIDDEN_SIZE = int(
    layers[0]
    .mlp
    .down_proj
    .weight
    .shape[0]
)


MLP_WIDTH = int(
    layers[0]
    .mlp
    .down_proj
    .weight
    .shape[1]
)


print(
    f"Transformer layers       : "
    f"{NUM_LAYERS}"
)

print(
    f"Hidden size              : "
    f"{HIDDEN_SIZE}"
)

print(
    f"MLP width                : "
    f"{MLP_WIDTH}"
)


# ============================================================
# VALIDATE CANDIDATE LAYERS
# ============================================================

for layer_idx in (
    candidate_layer_indices
):

    if (
        layer_idx < 0
        or layer_idx >= NUM_LAYERS
    ):

        raise RuntimeError(
            f"Invalid candidate layer: "
            f"{layer_idx}"
        )


# ============================================================
# GPU CANDIDATE INDEX TENSORS
# ============================================================

candidate_index_tensors = {}


for layer_idx in (
    candidate_layer_indices
):

    indices = (
        candidate_tables[
            layer_idx
        ][
            "neuron_index"
        ]
        .astype(int)
        .to_numpy()
    )


    if (
        indices.min() < 0
        or indices.max()
        >= MLP_WIDTH
    ):

        raise RuntimeError(
            f"Invalid neuron index "
            f"in layer {layer_idx + 1}"
        )


    candidate_index_tensors[
        layer_idx
    ] = torch.tensor(
        indices,
        dtype=torch.long,
        device=DEVICE,
    )


# ============================================================
# TOKEN MASK
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


    for special_id in (
        tokenizer.all_special_ids
    ):

        mask &= (
            input_ids
            .ne(
                special_id
            )
        )


    return mask


# ============================================================
# SHARED HOOK CAPTURE
# ============================================================

capture = {

    "valid_mask":
        None,

    "activation_stats":
        {},

    "residual_sums":
        {},
}


# ============================================================
# MLP ACTIVATION HOOK
# ============================================================

def make_activation_hook(
    layer_idx,
):

    """
    Capture selected Joy candidate neurons at the
    input to down_proj.

    This is the SwiGLU MLP intermediate activation m.

    For each sentence:

        mean_activation =
            mean valid-token activation

        positive_fraction =
            fraction of valid tokens for which m > 0
    """

    candidate_indices = (
        candidate_index_tensors[
            layer_idx
        ]
    )


    def hook(
        module,
        inputs,
    ):

        if not inputs:

            raise RuntimeError(
                f"No down_proj input "
                f"at layer {layer_idx + 1}"
            )


        x = inputs[0]


        valid_mask = (
            capture[
                "valid_mask"
            ]
        )


        if valid_mask is None:

            raise RuntimeError(
                "valid_mask missing "
                f"in layer {layer_idx + 1}"
            )


        # -----------------------------------------------
        # B x T x K
        # K is normally 110.
        # -----------------------------------------------

        selected = (
            x.index_select(
                dim=-1,
                index=
                    candidate_indices,
            )
        )


        mask_bool = (
            valid_mask
            .unsqueeze(
                -1
            )
        )


        mask_float = (
            mask_bool
            .to(
                selected.dtype
            )
        )


        valid_counts = (
            mask_float
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        # -----------------------------------------------
        # One mean activation per sentence/candidate.
        # -----------------------------------------------

        sentence_means = (
            (
                selected
                * mask_float
            )
            .sum(
                dim=1
            )
            / valid_counts
        )


        # -----------------------------------------------
        # Fraction of valid tokens with activation > 0.
        # -----------------------------------------------

        positive_counts = (
            (
                selected.gt(
                    0
                )
                & mask_bool
            )
            .sum(
                dim=1
            )
            .float()
        )


        positive_fraction = (
            positive_counts
            /
            valid_counts.float()
        )


        sentence_means_cpu = (
            sentence_means
            .detach()
            .float()
            .double()
            .cpu()
        )


        positive_fraction_cpu = (
            positive_fraction
            .detach()
            .double()
            .cpu()
        )


        capture[
            "activation_stats"
        ][
            layer_idx
        ] = {

            "sum":
                sentence_means_cpu
                .sum(
                    dim=0
                ),

            "sumsq":
                (
                    sentence_means_cpu
                    ** 2
                )
                .sum(
                    dim=0
                ),

            "positive_sum":
                positive_fraction_cpu
                .sum(
                    dim=0
                ),
        }


    return hook


# ============================================================
# RESIDUAL OUTPUT HOOK
# ============================================================

def make_residual_hook(
    layer_idx,
):

    """
    Capture mean residual representation after layer l.

    Each sentence contributes one mean residual vector,
    so long sentences do not receive greater weight.
    """

    def hook(
        module,
        inputs,
        output,
    ):

        if torch.is_tensor(
            output
        ):

            hidden = output

        elif isinstance(
            output,
            (tuple, list),
        ):

            hidden = output[0]

        elif hasattr(
            output,
            "last_hidden_state",
        ):

            hidden = (
                output.last_hidden_state
            )

        else:

            raise RuntimeError(
                f"Unsupported layer output "
                f"at layer {layer_idx + 1}: "
                f"{type(output)}"
            )


        valid_mask = (
            capture[
                "valid_mask"
            ]
        )


        if valid_mask is None:

            raise RuntimeError(
                "valid_mask missing "
                f"inside residual hook "
                f"layer {layer_idx + 1}"
            )


        mask = (
            valid_mask
            .to(
                hidden.device
            )
            .unsqueeze(
                -1
            )
            .to(
                hidden.dtype
            )
        )


        valid_counts = (
            mask
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        sentence_means = (
            (
                hidden
                * mask
            )
            .sum(
                dim=1
            )
            /
            valid_counts
        )


        capture[
            "residual_sums"
        ][
            layer_idx
        ] = (
            sentence_means
            .detach()
            .float()
            .double()
            .sum(
                dim=0
            )
            .cpu()
        )


    return hook


# ============================================================
# REGISTER HOOKS
# ============================================================

hook_handles = []


for layer_idx in (
    candidate_layer_indices
):

    activation_handle = (
        layers[
            layer_idx
        ]
        .mlp
        .down_proj
        .register_forward_pre_hook(
            make_activation_hook(
                layer_idx
            )
        )
    )


    residual_handle = (
        layers[
            layer_idx
        ]
        .register_forward_hook(
            make_residual_hook(
                layer_idx
            )
        )
    )


    hook_handles.extend(
        [
            activation_handle,
            residual_handle,
        ]
    )


print()
print(
    f"Hooks registered          : "
    f"{len(hook_handles)}"
)


# ============================================================
# PROCESS ONE EMOTION CLASS
# ============================================================

def process_class(
    class_df,
    description,
):

    """
    Returns per-layer:

        activation sum
        activation sumsq
        positive-rate sum
        residual-vector sum

    Statistics are accumulated across sentence-level means.
    """

    activation_sum = {}

    activation_sumsq = {}

    positive_sum = {}

    residual_sum = {}


    for layer_idx in (
        candidate_layer_indices
    ):

        k = len(
            candidate_tables[
                layer_idx
            ]
        )


        activation_sum[
            layer_idx
        ] = torch.zeros(
            k,
            dtype=torch.float64,
        )


        activation_sumsq[
            layer_idx
        ] = torch.zeros(
            k,
            dtype=torch.float64,
        )


        positive_sum[
            layer_idx
        ] = torch.zeros(
            k,
            dtype=torch.float64,
        )


        residual_sum[
            layer_idx
        ] = torch.zeros(
            HIDDEN_SIZE,
            dtype=torch.float64,
        )


    total_sentences = 0


    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                len(
                    class_df
                ),
                args.batch_size,
            ),
            desc=
                description,
        ):

            batch = (
                class_df.iloc[
                    start:
                    start
                    + args.batch_size
                ]
            )


            texts = (
                batch[
                    TEXT_COLUMN
                ]
                .tolist()
            )


            encoded_batch = tokenizer(
                texts,

                padding=True,

                truncation=True,

                max_length=
                    args.max_length,

                return_tensors=
                    "pt",
            )


            input_ids = (
                encoded_batch[
                    "input_ids"
                ]
                .to(
                    DEVICE
                )
            )


            attention_mask = (
                encoded_batch[
                    "attention_mask"
                ]
                .to(
                    DEVICE
                )
            )


            capture[
                "valid_mask"
            ] = (
                create_valid_token_mask(
                    input_ids,
                    attention_mask,
                )
            )


            capture[
                "activation_stats"
            ] = {}


            capture[
                "residual_sums"
            ] = {}


            _ = backbone(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                use_cache=False,
            )


            # -----------------------------------------------
            # Verify every layer hook ran.
            # -----------------------------------------------

            for layer_idx in (
                candidate_layer_indices
            ):

                if (
                    layer_idx
                    not in capture[
                        "activation_stats"
                    ]
                ):

                    raise RuntimeError(
                        "Activation hook did "
                        f"not run for layer "
                        f"{layer_idx + 1}"
                    )


                if (
                    layer_idx
                    not in capture[
                        "residual_sums"
                    ]
                ):

                    raise RuntimeError(
                        "Residual hook did "
                        f"not run for layer "
                        f"{layer_idx + 1}"
                    )


                batch_stats = (
                    capture[
                        "activation_stats"
                    ][
                        layer_idx
                    ]
                )


                activation_sum[
                    layer_idx
                ] += (
                    batch_stats[
                        "sum"
                    ]
                )


                activation_sumsq[
                    layer_idx
                ] += (
                    batch_stats[
                        "sumsq"
                    ]
                )


                positive_sum[
                    layer_idx
                ] += (
                    batch_stats[
                        "positive_sum"
                    ]
                )


                residual_sum[
                    layer_idx
                ] += (
                    capture[
                        "residual_sums"
                    ][
                        layer_idx
                    ]
                )


            total_sentences += int(
                input_ids.shape[0]
            )


    if total_sentences <= 0:

        raise RuntimeError(
            f"No samples processed "
            f"for {description}"
        )


    return {

        "n":
            total_sentences,

        "activation_sum":
            activation_sum,

        "activation_sumsq":
            activation_sumsq,

        "positive_sum":
            positive_sum,

        "residual_sum":
            residual_sum,
    }


# ============================================================
# PROCESS JOY AND SADNESS
# ============================================================

print()
print("=" * 105)
print("EXTRACTING CONTINUOUS ACTIVATIONS + RESIDUAL REPRESENTATIONS")
print("=" * 105)


try:

    joy_stats = process_class(
        joy_df,
        "Joy",
    )


    sadness_stats = process_class(
        sad_df,
        "Sadness",
    )


finally:

    for handle in (
        hook_handles
    ):

        handle.remove()


    capture[
        "valid_mask"
    ] = None


    capture[
        "activation_stats"
    ] = {}


    capture[
        "residual_sums"
    ] = {}


# ============================================================
# SAMPLE STATISTICS
# ============================================================

def mean_and_sample_sd(
    total,
    total_sq,
    n,
):

    mean = (
        total
        / n
    )


    if n <= 1:

        variance = torch.zeros_like(
            mean
        )

    else:

        variance = (
            total_sq
            - (
                total
                * total
                / n
            )
        ) / (
            n - 1
        )


        variance = torch.clamp(
            variance,
            min=0.0,
        )


    sd = torch.sqrt(
        variance
    )


    return (
        mean.numpy(),
        sd.numpy(),
    )


# ============================================================
# BUILD LAYER-SPECIFIC RESIDUAL DIRECTIONS
# ============================================================

print()
print("=" * 105)
print("BUILDING JOY-SADNESS RESIDUAL DIRECTIONS FOR ALL LAYERS")
print("=" * 105)


layer_direction_data = {}

layer_summary_rows = []


for layer_idx in (
    candidate_layer_indices
):

    layer_number = (
        layer_idx + 1
    )


    mean_joy_residual = (
        joy_stats[
            "residual_sum"
        ][
            layer_idx
        ]
        / joy_stats[
            "n"
        ]
    ).numpy()


    mean_sad_residual = (
        sadness_stats[
            "residual_sum"
        ][
            layer_idx
        ]
        / sadness_stats[
            "n"
        ]
    ).numpy()


    direction = (
        mean_joy_residual
        - mean_sad_residual
    )


    norm = float(
        np.linalg.norm(
            direction
        )
    )


    if norm < EPS:

        raise RuntimeError(
            f"Layer {layer_number}: "
            "Joy-Sad residual direction "
            "has near-zero norm."
        )


    unit = (
        direction
        / norm
    )


    layer_direction_data[
        layer_idx
    ] = {

        "direction":
            direction,

        "unit":
            unit,

        "norm":
            norm,
    }


    np.save(
        DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"joy_minus_sad.npy"
        ),

        direction.astype(
            np.float32
        ),
    )


    np.save(
        DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"joy_minus_sad_unit.npy"
        ),

        unit.astype(
            np.float32
        ),
    )


    np.save(
        DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"mean_joy.npy"
        ),

        mean_joy_residual.astype(
            np.float32
        ),
    )


    np.save(
        DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"mean_sadness.npy"
        ),

        mean_sad_residual.astype(
            np.float32
        ),
    )


    layer_summary_rows.append(
        {

            "layer":
                layer_number,

            "layer_index":
                layer_idx,

            "num_joy_examples":
                joy_stats[
                    "n"
                ],

            "num_sadness_examples":
                sadness_stats[
                    "n"
                ],

            "joy_sad_direction_norm":
                norm,
        }
    )


    print(
        f"Layer {layer_number:02d} | "
        f"||Joy-Sad|| = "
        f"{norm:.8f}"
    )


layer_summary_df = pd.DataFrame(
    layer_summary_rows
)


layer_summary_file = (
    OUTPUT_DIR
    / "layer_direction_summary.csv"
)


layer_summary_df.to_csv(
    layer_summary_file,
    index=False,
)


# ============================================================
# SCORE EVERY JOY CANDIDATE
# ============================================================

print()
print("=" * 105)
print("COMPUTING ACTUATOR SCORES FOR ALL TOP-1% JOY NEURONS")
print("=" * 105)


scored_tables = []


for layer_idx in (
    candidate_layer_indices
):

    candidates_layer = (
        candidate_tables[
            layer_idx
        ]
        .copy()
    )


    candidate_indices = (
        candidates_layer[
            "neuron_index"
        ]
        .astype(int)
        .to_numpy()
    )


    k = len(
        candidate_indices
    )


    # --------------------------------------------------------
    # Continuous activation statistics.
    # --------------------------------------------------------

    (
        mu_joy,
        sd_joy,
    ) = mean_and_sample_sd(
        joy_stats[
            "activation_sum"
        ][
            layer_idx
        ],

        joy_stats[
            "activation_sumsq"
        ][
            layer_idx
        ],

        joy_stats[
            "n"
        ],
    )


    (
        mu_sad,
        sd_sad,
    ) = mean_and_sample_sd(
        sadness_stats[
            "activation_sum"
        ][
            layer_idx
        ],

        sadness_stats[
            "activation_sumsq"
        ][
            layer_idx
        ],

        sadness_stats[
            "n"
        ],
    )


    delta_mu = (
        mu_joy
        - mu_sad
    )


    n_joy = (
        joy_stats[
            "n"
        ]
    )


    n_sad = (
        sadness_stats[
            "n"
        ]
    )


    pooled_variance = (
        (
            n_joy - 1
        )
        * (
            sd_joy
            ** 2
        )
        +
        (
            n_sad - 1
        )
        * (
            sd_sad
            ** 2
        )
    ) / max(
        (
            n_joy
            + n_sad
            - 2
        ),
        1,
    )


    pooled_variance = np.maximum(
        pooled_variance,
        0.0,
    )


    pooled_sd = np.sqrt(
        pooled_variance
    )


    standardized_contrast = (
        delta_mu
        /
        (
            pooled_sd
            + EPS
        )
    )


    positive_rate_joy = (
        joy_stats[
            "positive_sum"
        ][
            layer_idx
        ]
        .numpy()
        / n_joy
    )


    positive_rate_sad = (
        sadness_stats[
            "positive_sum"
        ][
            layer_idx
        ]
        .numpy()
        / n_sad
    )


    positive_rate_difference = (
        positive_rate_joy
        - positive_rate_sad
    )


    # ========================================================
    # W_DOWN WRITE DIRECTIONS
    #
    # shape:
    #   [hidden_size, k]
    # ========================================================

    candidate_index_tensor = (
        candidate_index_tensors[
            layer_idx
        ]
    )


    write_matrix = (
        layers[
            layer_idx
        ]
        .mlp
        .down_proj
        .weight
        .index_select(
            dim=1,
            index=
                candidate_index_tensor,
        )
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    if write_matrix.shape != (
        HIDDEN_SIZE,
        k,
    ):

        raise RuntimeError(
            f"Unexpected W_down matrix shape "
            f"for layer {layer_idx + 1}: "
            f"{write_matrix.shape}"
        )


    write_norm = np.linalg.norm(
        write_matrix,
        axis=0,
    )


    emotion_unit = (
        layer_direction_data[
            layer_idx
        ][
            "unit"
        ]
    )


    emotion_norm = (
        layer_direction_data[
            layer_idx
        ][
            "norm"
        ]
    )


    # --------------------------------------------------------
    # Projection:
    #
    # v_u^T d_hat
    # --------------------------------------------------------

    write_projection = (
        emotion_unit
        @ write_matrix
    )


    # --------------------------------------------------------
    # Cosine of raw W_down direction.
    # --------------------------------------------------------

    write_cosine = np.divide(
        write_projection,
        write_norm,
        out=np.zeros_like(
            write_projection
        ),
        where=
            write_norm
            > EPS,
    )


    write_cosine = np.clip(
        write_cosine,
        -1.0,
        1.0,
    )


    write_angle = np.degrees(
        np.arccos(
            write_cosine
        )
    )


    # ========================================================
    # ALPHA=1 INTERVENTION
    #
    # Delta_h_u =
    #     Delta_mu_u * W_down[:,u]
    # ========================================================

    expected_shift_norm = (
        np.abs(
            delta_mu
        )
        * write_norm
    )


    # --------------------------------------------------------
    # PRIMARY ACTUATOR SCORE
    #
    # Q_u =
    # Delta_mu_u * projection
    # --------------------------------------------------------

    actuator_q = (
        delta_mu
        * write_projection
    )


    # --------------------------------------------------------
    # Cosine of ACTUAL alpha=1 intervention.
    #
    # Because Delta_mu may be negative, this may have
    # the opposite sign from the raw W_down cosine.
    # --------------------------------------------------------

    intervention_cosine = np.divide(
        actuator_q,
        expected_shift_norm,
        out=np.zeros_like(
            actuator_q
        ),
        where=
            expected_shift_norm
            > EPS,
    )


    intervention_cosine = np.clip(
        intervention_cosine,
        -1.0,
        1.0,
    )


    intervention_angle = np.degrees(
        np.arccos(
            intervention_cosine
        )
    )


    # ========================================================
    # STORE
    # ========================================================

    candidates_layer[
        "n_joy"
    ] = n_joy


    candidates_layer[
        "n_sadness"
    ] = n_sad


    candidates_layer[
        "mean_activation_joy"
    ] = mu_joy


    candidates_layer[
        "mean_activation_sadness"
    ] = mu_sad


    candidates_layer[
        "raw_activation_difference"
    ] = delta_mu


    candidates_layer[
        "sd_joy"
    ] = sd_joy


    candidates_layer[
        "sd_sadness"
    ] = sd_sad


    candidates_layer[
        "pooled_sd"
    ] = pooled_sd


    candidates_layer[
        "standardized_joy_minus_sadness"
    ] = standardized_contrast


    candidates_layer[
        "positive_rate_joy"
    ] = positive_rate_joy


    candidates_layer[
        "positive_rate_sadness"
    ] = positive_rate_sad


    candidates_layer[
        "positive_rate_difference"
    ] = positive_rate_difference


    candidates_layer[
        "joy_sad_residual_direction_norm"
    ] = emotion_norm


    candidates_layer[
        "write_direction_norm"
    ] = write_norm


    candidates_layer[
        "write_direction_cosine_joy_sad"
    ] = write_cosine


    candidates_layer[
        "write_direction_angle_joy_sad_deg"
    ] = write_angle


    candidates_layer[
        "write_projection_on_joy_axis"
    ] = write_projection


    candidates_layer[
        "expected_residual_shift_norm_alpha1"
    ] = expected_shift_norm


    candidates_layer[
        "intervention_direction_cosine_joy_sad"
    ] = intervention_cosine


    candidates_layer[
        "intervention_direction_angle_joy_sad_deg"
    ] = intervention_angle


    candidates_layer[
        "actuator_score_q"
    ] = actuator_q


    candidates_layer[
        "actuator_score_sign"
    ] = np.where(
        actuator_q > 0,
        "toward_joy",
        np.where(
            actuator_q < 0,
            "away_from_joy",
            "orthogonal",
        ),
    )


    scored_tables.append(
        candidates_layer
    )


# ============================================================
# COMBINE ALL 3,080 CANDIDATES
# ============================================================

ranking_df = pd.concat(
    scored_tables,
    ignore_index=True,
)


# ============================================================
# GLOBAL ACTUATOR RANK
# ============================================================

ranking_df[
    "actuator_rank"
] = (
    ranking_df[
        "actuator_score_q"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)


ranking_df = (
    ranking_df
    .sort_values(
        "actuator_rank"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# POSITIVE-Q RANK
# ============================================================

positive_mask = (
    ranking_df[
        "actuator_score_q"
    ]
    > 0
)


num_positive = int(
    positive_mask.sum()
)


ranking_df[
    "positive_actuator_rank"
] = np.nan


if num_positive > 0:

    ranking_df.loc[
        positive_mask,
        "positive_actuator_rank",
    ] = np.arange(
        1,
        num_positive + 1,
    )


# ============================================================
# RECOMMENDED TOP-K
# ============================================================

ranking_df[
    "recommended_for_next_steering_test"
] = False


positive_indices = (
    ranking_df.index[
        positive_mask
    ]
    .tolist()
)


num_recommended = min(
    args.top_k_next,
    len(
        positive_indices
    ),
)


if num_recommended > 0:

    ranking_df.loc[
        positive_indices[
            :num_recommended
        ],
        "recommended_for_next_steering_test",
    ] = True


# ============================================================
# REORDER IMPORTANT COLUMNS
# ============================================================

important_columns = [

    "actuator_rank",

    "positive_actuator_rank",

    "recommended_for_next_steering_test",

    "layer",

    "layer_index",

    "neuron_index",

    "neuron_number",

    "original_joy_rank",

    "original_selectivity_score",

    "membership_degree_top1pct",

    "is_exact_degree1",

    "n_joy",

    "n_sadness",

    "mean_activation_joy",

    "mean_activation_sadness",

    "raw_activation_difference",

    "sd_joy",

    "sd_sadness",

    "pooled_sd",

    "standardized_joy_minus_sadness",

    "positive_rate_joy",

    "positive_rate_sadness",

    "positive_rate_difference",

    "joy_sad_residual_direction_norm",

    "write_direction_norm",

    "write_direction_cosine_joy_sad",

    "write_direction_angle_joy_sad_deg",

    "write_projection_on_joy_axis",

    "expected_residual_shift_norm_alpha1",

    "intervention_direction_cosine_joy_sad",

    "intervention_direction_angle_joy_sad_deg",

    "actuator_score_q",

    "actuator_score_sign",
]


important_columns = [

    c

    for c in important_columns

    if c in ranking_df.columns
]


other_columns = [

    c

    for c in ranking_df.columns

    if c not in important_columns
]


ranking_df = ranking_df[
    important_columns
    + other_columns
]


# ============================================================
# SAVE FULL RANKING
# ============================================================

ranking_csv = (
    OUTPUT_DIR
    / "all_top1pct_joy_actuator_ranking.csv"
)


ranking_df.to_csv(
    ranking_csv,
    index=False,
)


# ============================================================
# TOP-K TABLE
# ============================================================

top_df = (
    ranking_df
    .head(
        args.top_k_next
    )
    .copy()
)


top_file = (
    OUTPUT_DIR
    / "top_joy_actuators.csv"
)


top_df.to_csv(
    top_file,
    index=False,
)


# ============================================================
# BEST POSITIVE ACTUATOR
# ============================================================

positive_df = (
    ranking_df.loc[
        ranking_df[
            "actuator_score_q"
        ]
        > 0
    ]
    .copy()
)


if positive_df.empty:

    best_row = None

else:

    best_row = (
        positive_df.iloc[0]
    )


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def dataframe_records(
    dataframe,
):

    return json.loads(
        dataframe.to_json(
            orient="records"
        )
    )


# ============================================================
# SAVE FULL JSON
# ============================================================

ranking_json = (
    OUTPUT_DIR
    / "all_top1pct_joy_actuator_ranking.json"
)


json_output = {

    "experiment":
        (
            "Rank all layerwise top-1pct Joy neurons "
            "as candidate causal actuators"
        ),

    "model":
        MODEL_ID,

    "target_emotion":
        TARGET_EMOTION,

    "source_emotion":
        SOURCE_EMOTION,

    "candidate_source":
        str(
            TOP1_FILE
        ),

    "num_candidates":
        int(
            len(
                ranking_df
            )
        ),

    "num_positive_q":
        num_positive,

    "num_exact_degree1":
        int(
            ranking_df[
                "is_exact_degree1"
            ]
            .sum()
        ),

    "num_joy_examples":
        int(
            joy_stats[
                "n"
            ]
        ),

    "num_sadness_examples":
        int(
            sadness_stats[
                "n"
            ]
        ),

    "actuator_definition":
        {

            "activation_shift":
                (
                    "Delta_mu = mean_activation_joy "
                    "- mean_activation_sadness"
                ),

            "write_direction":
                (
                    "v_u = W_down[layer][:, neuron]"
                ),

            "emotion_direction":
                (
                    "d_l = mean_residual_joy(layer) "
                    "- mean_residual_sadness(layer)"
                ),

            "alpha1_residual_shift":
                (
                    "Delta_h_u = Delta_mu_u * v_u"
                ),

            "primary_score":
                (
                    "Q_u = Delta_mu_u * "
                    "(v_u^T unit(d_l))"
                ),

            "interpretation":
                (
                    "Q_u > 0 means alpha=+1 moves "
                    "toward the layer-specific Joy-Sadness axis."
                ),
        },

    "exploratory_data_warning":
        (
            "This run uses all supplied Joy/Sadness examples "
            "for discovery. Publication-quality analysis should "
            "select candidates on training data and evaluate "
            "steering on held-out prompts/data."
        ),

    "top_candidates":
        dataframe_records(
            ranking_df.head(
                min(
                    100,
                    len(
                        ranking_df
                    ),
                )
            )
        ),
}


with ranking_json.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        json_output,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# BEST ACTUATOR JSON
# ============================================================

best_json_file = (
    OUTPUT_DIR
    / "best_joy_actuator_neuron.json"
)


if best_row is None:

    best_json = {

        "best_neuron":
            None,

        "reason":
            (
                "No candidate had positive actuator_score_q."
            ),
    }

else:

    best_record = (
        dataframe_records(
            pd.DataFrame(
                [
                    best_row
                ]
            )
        )[0]
    )


    best_json = {

        "experiment":
            (
                "Best Joy actuator selected "
                "from all top-1pct Joy neurons"
            ),

        "selection_rule":
            (
                "largest positive actuator_score_q"
            ),

        "best_neuron":
            best_record,
    }


with best_json_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        best_json,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE BEST WRITE DIRECTION
# ============================================================

if best_row is not None:

    best_layer_idx = int(
        best_row[
            "layer_index"
        ]
    )


    best_neuron_idx = int(
        best_row[
            "neuron_index"
        ]
    )


    best_write = (
        layers[
            best_layer_idx
        ]
        .mlp
        .down_proj
        .weight[
            :,
            best_neuron_idx
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


    best_write_norm = float(
        np.linalg.norm(
            best_write
        )
    )


    np.save(
        OUTPUT_DIR
        / "best_joy_actuator_write_direction.npy",

        best_write,
    )


    if best_write_norm > EPS:

        np.save(
            OUTPUT_DIR
            / "best_joy_actuator_write_direction_unit.npy",

            (
                best_write
                / best_write_norm
            ).astype(
                np.float32
            ),
        )


# ============================================================
# RANKING FIGURE
# ============================================================

print()
print("=" * 105)
print("GENERATING RANKING FIGURE")
print("=" * 105)


plot_top_n = min(
    args.plot_top_n,
    len(
        ranking_df
    ),
)


top_plot = (
    ranking_df
    .head(
        plot_top_n
    )
    .sort_values(
        "actuator_score_q",
        ascending=True,
    )
    .copy()
)


top_plot[
    "label"
] = [

    (
        f"L{int(row.layer)}"
        f"-N{int(row.neuron_index)}"
        f"-d{int(row.membership_degree_top1pct)}"
    )

    for row in (
        top_plot.itertuples()
    )
]


fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(
        22,
        9,
    ),
)


ax1, ax2, ax3 = axes


# ============================================================
# PANEL A — TOP Q
# ============================================================

bars = ax1.barh(
    top_plot[
        "label"
    ],
    top_plot[
        "actuator_score_q"
    ],
)


ax1.axvline(
    0,
    linewidth=1,
)


ax1.set_title(
    (
        f"A. Top {plot_top_n} Joy actuator scores\n"
        r"$Q_u=\Delta\mu_u"
        r"(W_{down,u}^{T}\hat d_{J-S})$"
    ),
    fontsize=12,
)


ax1.set_xlabel(
    "Expected alpha=1 Joy-axis displacement"
)


ax1.set_ylabel(
    "Layer-Neuron-membership degree"
)


ax1.grid(
    True,
    axis="x",
    alpha=0.2,
)


for bar, value in zip(
    bars,
    top_plot[
        "actuator_score_q"
    ],
):

    value = float(
        value
    )


    ax1.text(
        value,
        (
            bar.get_y()
            + bar.get_height()
            / 2
        ),
        f" {value:+.4f}",
        va="center",
        ha="left",
        fontsize=7,
    )


# ============================================================
# PANEL B — MARKER VS ACTUATOR
# ============================================================

scatter = ax2.scatter(
    ranking_df[
        "standardized_joy_minus_sadness"
    ],
    ranking_df[
        "actuator_score_q"
    ],
    s=16,
    alpha=0.50,
)


ax2.axhline(
    0,
    linewidth=1,
)


ax2.axvline(
    0,
    linewidth=1,
)


ax2.set_title(
    (
        "B. Activation marker strength "
        "vs actuator utility"
    ),
    fontsize=12,
)


ax2.set_xlabel(
    (
        "Standardized Joy-Sad activation contrast"
    )
)


ax2.set_ylabel(
    "Actuator score Q"
)


ax2.grid(
    True,
    alpha=0.18,
)


# Highlight previous neuron 10707.
previous = ranking_df.loc[
    (
        ranking_df[
            "layer"
        ]
        == 20
    )
    &
    (
        ranking_df[
            "neuron_index"
        ]
        == 10707
    )
]


if not previous.empty:

    old = previous.iloc[0]


    ax2.scatter(
        [
            old[
                "standardized_joy_minus_sadness"
            ]
        ],
        [
            old[
                "actuator_score_q"
            ]
        ],
        s=130,
        marker="x",
        linewidths=2.5,
    )


    ax2.annotate(
        "Previous L20-N10707",
        (
            old[
                "standardized_joy_minus_sadness"
            ],
            old[
                "actuator_score_q"
            ],
        ),
        xytext=(
            8,
            -18,
        ),
        textcoords=
            "offset points",
        fontsize=9,
    )


# Highlight best actuator.
if best_row is not None:

    ax2.scatter(
        [
            best_row[
                "standardized_joy_minus_sadness"
            ]
        ],
        [
            best_row[
                "actuator_score_q"
            ]
        ],
        s=140,
        marker="*",
    )


    ax2.annotate(
        (
            f"Best: L{int(best_row['layer'])}"
            f"-N{int(best_row['neuron_index'])}"
        ),
        (
            best_row[
                "standardized_joy_minus_sadness"
            ],
            best_row[
                "actuator_score_q"
            ],
        ),
        xytext=(
            10,
            12,
        ),
        textcoords=
            "offset points",
        fontsize=9,
        fontweight="bold",
    )


# ============================================================
# PANEL C — Q DISTRIBUTION
# ============================================================

ax3.hist(
    ranking_df[
        "actuator_score_q"
    ],
    bins=80,
)


ax3.axvline(
    0,
    linewidth=1.2,
)


ax3.set_title(
    "C. Distribution of actuator scores",
    fontsize=12,
)


ax3.set_xlabel(
    "Actuator score Q"
)


ax3.set_ylabel(
    "Number of Joy candidates"
)


ax3.grid(
    True,
    axis="y",
    alpha=0.18,
)


ax3.text(
    0.97,
    0.96,
    (
        f"Total = {len(ranking_df):,}\n"
        f"Q > 0 = {num_positive:,}\n"
        f"Degree-1 = "
        f"{int(ranking_df['is_exact_degree1'].sum()):,}"
    ),
    transform=
        ax3.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle=
            "round,pad=0.4",
        facecolor=
            "white",
        alpha=
            0.90,
    ),
)


fig.suptitle(
    (
        "All Layerwise Top-1% Joy Neurons: "
        "Marker vs Actuator Analysis\n"
        f"{joy_stats['n']:,} Joy examples, "
        f"{sadness_stats['n']:,} Sadness examples"
    ),
    fontsize=16,
    fontweight="bold",
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.93,
    ]
)


ranking_png = (
    OUTPUT_DIR
    / "joy_actuator_ranking.png"
)


fig.savefig(
    ranking_png,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)


plt.close(
    fig
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print()
print("=" * 105)
print("SUMMARY")
print("=" * 105)


print(
    f"Total Joy candidates      : "
    f"{len(ranking_df):,}"
)


print(
    f"Positive-Q candidates     : "
    f"{num_positive:,}"
)


print(
    f"Degree-1 candidates       : "
    f"{int(ranking_df['is_exact_degree1'].sum()):,}"
)


# ============================================================
# TOP 20
# ============================================================

print()
print("=" * 105)
print("TOP 20 JOY ACTUATORS")
print("=" * 105)


display_columns = [

    "actuator_rank",

    "layer",

    "neuron_index",

    "original_joy_rank",

    "membership_degree_top1pct",

    "raw_activation_difference",

    "standardized_joy_minus_sadness",

    "write_direction_cosine_joy_sad",

    "write_projection_on_joy_axis",

    "expected_residual_shift_norm_alpha1",

    "intervention_direction_cosine_joy_sad",

    "actuator_score_q",
]


print(
    ranking_df[
        display_columns
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# PREVIOUS NEURON
# ============================================================

print()
print("=" * 105)
print("PREVIOUS TESTED NEURON L20-N10707")
print("=" * 105)


previous = ranking_df.loc[
    (
        ranking_df[
            "layer"
        ]
        == 20
    )
    &
    (
        ranking_df[
            "neuron_index"
        ]
        == 10707
    )
]


if previous.empty:

    print(
        "L20-N10707 not found "
        "in current Joy top-1% pool."
    )

else:

    print(
        previous[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# BEST CANDIDATE
# ============================================================

print()
print("=" * 105)
print("BEST JOY ACTUATOR")
print("=" * 105)


if best_row is None:

    print(
        "No positive-Q candidate found."
    )

else:

    print(
        f"Layer                       : "
        f"{int(best_row['layer'])}"
    )


    print(
        f"Layer index                 : "
        f"{int(best_row['layer_index'])}"
    )


    print(
        f"Neuron index                : "
        f"{int(best_row['neuron_index'])}"
    )


    print(
        f"Neuron number               : "
        f"{int(best_row['neuron_number'])}"
    )


    print(
        f"Original Joy rank            : "
        f"{best_row['original_joy_rank']}"
    )


    print(
        f"Membership degree            : "
        f"{int(best_row['membership_degree_top1pct'])}"
    )


    print(
        f"Exact degree-1?              : "
        f"{bool(best_row['is_exact_degree1'])}"
    )


    print(
        f"Mean activation Joy          : "
        f"{best_row['mean_activation_joy']:+.8f}"
    )


    print(
        f"Mean activation Sadness      : "
        f"{best_row['mean_activation_sadness']:+.8f}"
    )


    print(
        f"Delta mu                     : "
        f"{best_row['raw_activation_difference']:+.8f}"
    )


    print(
        f"Standardized contrast        : "
        f"{best_row['standardized_joy_minus_sadness']:+.8f}"
    )


    print(
        f"W_down cosine                : "
        f"{best_row['write_direction_cosine_joy_sad']:+.8f}"
    )


    print(
        f"W_down Joy-axis projection   : "
        f"{best_row['write_projection_on_joy_axis']:+.8f}"
    )


    print(
        f"Expected ||Delta h|| a=1     : "
        f"{best_row['expected_residual_shift_norm_alpha1']:.8f}"
    )


    print(
        f"Intervention cosine          : "
        f"{best_row['intervention_direction_cosine_joy_sad']:+.8f}"
    )


    print(
        f"Intervention angle           : "
        f"{best_row['intervention_direction_angle_joy_sad_deg']:.4f} deg"
    )


    print(
        f"ACTUATOR SCORE Q             : "
        f"{best_row['actuator_score_q']:+.8f}"
    )


# ============================================================
# SAVED FILES
# ============================================================

print()
print("=" * 105)
print("SAVED OUTPUTS")
print("=" * 105)


print(
    f"Full ranking CSV:\n"
    f"{ranking_csv}"
)


print()
print(
    f"Top candidates CSV:\n"
    f"{top_file}"
)


print()
print(
    f"Ranking JSON:\n"
    f"{ranking_json}"
)


print()
print(
    f"Best actuator JSON:\n"
    f"{best_json_file}"
)


print()
print(
    f"Ranking PNG:\n"
    f"{ranking_png}"
)


print()
print(
    f"Layer direction summary:\n"
    f"{layer_summary_file}"
)


print()
print(
    f"Layer direction files:\n"
    f"{DIRECTION_DIR}"
)


print()
print("Done.")
