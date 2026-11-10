#!/usr/bin/env python3

"""
Rank all exact-degree-1 joy neurons by steering / actuator utility.

The previous candidate-selection experiment ranked joy neurons using
their scalar joy-vs-sadness activation separation:

    Delta_mu_u = mu_joy,u - mu_sadness,u

    d_u = Delta_mu_u / pooled_sd_u

That tells us whether a neuron is a good REPRESENTATION MARKER.

However, a neuron can activate more strongly for joy while its
down-projection write direction points somewhere unrelated to joy.

For steering, we therefore additionally examine:

    v_u = W_down[:, u]

and the empirical Joy-Sadness residual direction at the neuron's layer:

    d_l = mean_h_l(joy) - mean_h_l(sadness)

We calculate:

    cosine_write_u
        = cos(v_u, d_l)

    projection_write_u
        = v_u^T d_hat_l

The actual intervention at alpha = 1 is:

    Delta_h_u
        = Delta_mu_u * v_u

Therefore its expected displacement along the Joy-Sadness axis is:

    Q_u
        = Delta_mu_u * projection_write_u

        = Delta_h_u^T d_hat_l

This is the main actuator score used for ranking.

Interpretation
--------------

Q_u > 0
    alpha=+1 is expected to move the residual state toward Joy
    relative to Sadness.

Q_u < 0
    alpha=+1 is expected to move in the opposite direction.

Q_u ~= 0
    the intervention is largely orthogonal to the Joy-Sadness axis.

The script:

1. Loads the 12 exact-degree-1 joy candidates.
2. Finds the unique layers containing those candidates.
3. Uses ALL joy and sadness examples by default.
4. Computes a Joy-Sadness residual direction for each required layer.
5. Extracts W_down[:, neuron] for every candidate.
6. Computes activation and write-direction geometry.
7. Ranks candidates by Q_u.
8. Saves:
       joy_actuator_ranking.csv
       joy_actuator_ranking.json
       best_joy_actuator_neuron.json
       joy_actuator_ranking.png
       per-layer residual directions as .npy files

This does NOT perform generation steering yet.
It selects better candidates for the next steering experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

# Safe for headless Linux / remote server.
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
        "0 = use all joy and sadness examples. "
        "Use e.g. 200 for a quick sanity check."
    ),
)


parser.add_argument(
    "--local-files-only",
    action="store_true",
    help=(
        "Load Sarvam-1 only from local Hugging Face cache."
    ),
)


parser.add_argument(
    "--top-k-next",
    type=int,
    default=3,
    help=(
        "How many highest-positive-Q neurons to flag "
        "for the next single-neuron steering experiment."
    ),
)


parser.add_argument(
    "--seed",
    type=int,
    default=42,
)


args = parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

TARGET_EMOTION = "joy"

SOURCE_EMOTION = "sadness"

TEXT_COLUMN = "hinglish_genz_text"

LABEL_COLUMN = "label"

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


CANDIDATE_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "13_single_joy_neuron_selection"
    / "joy_candidate_activation_statistics.csv"
)


OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "17_joy_actuator_ranking"
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


print("=" * 100)
print("RANK ALL JOY ACTUATOR NEURONS")
print("=" * 100)

print(
    f"Device              : {DEVICE}"
)

print(
    f"Dtype               : {DTYPE}"
)

print(
    f"Model               : {MODEL_ID}"
)

print(
    f"Target emotion      : {TARGET_EMOTION}"
)

print(
    f"Source emotion      : {SOURCE_EMOTION}"
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not DATA_FILE.is_file():

    raise FileNotFoundError(
        f"Missing dataset:\n"
        f"{DATA_FILE}"
    )


if not CANDIDATE_FILE.is_file():

    raise FileNotFoundError(
        f"Missing candidate file:\n"
        f"{CANDIDATE_FILE}"
    )


# ============================================================
# LOAD CANDIDATES
# ============================================================

print()
print("=" * 100)
print("LOADING JOY CANDIDATES")
print("=" * 100)


candidates = pd.read_csv(
    CANDIDATE_FILE
)


required_candidate_columns = [

    "steering_rank",

    "layer",

    "layer_index",

    "neuron_index",

    "neuron_number",

    "mean_activation_joy",

    "mean_activation_sadness",

    "raw_activation_difference",

    "pooled_sd",

    "standardized_joy_minus_sadness",

    "positive_rate_joy",

    "positive_rate_sadness",

    "positive_rate_difference",

    "original_selectivity_score",

    "original_rank",
]


missing_columns = [

    column

    for column
    in required_candidate_columns

    if column
    not in candidates.columns
]


if missing_columns:

    raise RuntimeError(
        "Candidate CSV is missing columns:\n"
        + "\n".join(
            missing_columns
        )
    )


candidates = (
    candidates
    .sort_values(
        "steering_rank"
    )
    .reset_index(
        drop=True
    )
)


print(
    f"Candidates loaded     : "
    f"{len(candidates)}"
)


print()
print(
    candidates[
        [
            "steering_rank",
            "layer",
            "neuron_index",
            "raw_activation_difference",
            "standardized_joy_minus_sadness",
            "original_rank",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# UNIQUE CANDIDATE LAYERS
# ============================================================

candidate_layer_indices = sorted(
    candidates[
        "layer_index"
    ]
    .astype(int)
    .unique()
    .tolist()
)


candidate_layer_numbers = [

    layer_idx + 1

    for layer_idx
    in candidate_layer_indices
]


print()
print(
    "Required transformer layers : "
    + ", ".join(
        str(x)
        for x
        in candidate_layer_numbers
    )
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 100)
print("LOADING DATASET")
print("=" * 100)


df = pd.read_csv(
    DATA_FILE
)


required_dataset_columns = [

    TEXT_COLUMN,

    LABEL_COLUMN,
]


for column in required_dataset_columns:

    if column not in df.columns:

        raise RuntimeError(
            f"Dataset missing column: "
            f"{column}"
        )


df = (
    df[
        required_dataset_columns
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
        joy_df
        .sample(
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
        sad_df
        .sample(
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
    f"Joy examples          : "
    f"{len(joy_df):,}"
)

print(
    f"Sadness examples      : "
    f"{len(sad_df):,}"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("=" * 100)
print("LOADING TOKENIZER")
print("=" * 100)


tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_ID,

        trust_remote_code=True,

        local_files_only=
            args.local_files_only,
    )
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


print(
    f"Special token IDs     : "
    f"{tokenizer.all_special_ids}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING SARVAM-1")
print("=" * 100)


model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_ID,

        dtype=DTYPE,

        trust_remote_code=True,

        local_files_only=
            args.local_files_only,
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


layers = (
    model.model.layers
)


backbone = (
    model.model
)


num_layers = len(
    layers
)


hidden_size = int(
    layers[0]
    .mlp
    .down_proj
    .weight
    .shape[0]
)


mlp_width = int(
    layers[0]
    .mlp
    .down_proj
    .weight
    .shape[1]
)


print(
    f"Transformer layers    : "
    f"{num_layers}"
)

print(
    f"Hidden size           : "
    f"{hidden_size}"
)

print(
    f"MLP width             : "
    f"{mlp_width}"
)


for layer_idx in candidate_layer_indices:

    if (
        layer_idx < 0
        or layer_idx >= num_layers
    ):

        raise RuntimeError(
            f"Invalid candidate layer index: "
            f"{layer_idx}"
        )


# ============================================================
# VALID TOKEN MASK
# ============================================================

def create_valid_token_mask(
    input_ids,
    attention_mask,
):

    """
    Keep only actual content tokens.

    Remove:
        PAD
        BOS
        EOS
        and every tokenizer special token.
    """

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
# MULTI-LAYER RESIDUAL CAPTURE
# ============================================================

capture = {

    "valid_mask":
        None,

    "batch_sums":
        {},
}


def make_layer_capture_hook(
    layer_idx,
):

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
                f"Unsupported output "
                f"at layer {layer_idx}: "
                f"{type(output)}"
            )


        valid_mask = (
            capture[
                "valid_mask"
            ]
        )


        if valid_mask is None:

            raise RuntimeError(
                "Token mask missing "
                f"inside layer {layer_idx} hook."
            )


        if (
            valid_mask.shape[0]
            != hidden.shape[0]
            or
            valid_mask.shape[1]
            != hidden.shape[1]
        ):

            raise RuntimeError(
                "Token mask shape mismatch "
                f"at layer {layer_idx}. "
                f"Mask={valid_mask.shape}, "
                f"hidden={hidden.shape}"
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


        numerator = (
            hidden
            * mask
        ).sum(
            dim=1
        )


        denominator = (
            mask
            .sum(
                dim=1
            )
            .clamp_min(
                1.0
            )
        )


        # One mean residual vector per sentence.
        sentence_means = (
            numerator
            / denominator
        )


        # We only need the sum over the batch.
        # This avoids storing thousands of 2048-D vectors.
        batch_sum = (
            sentence_means
            .detach()
            .float()
            .double()
            .sum(
                dim=0
            )
            .cpu()
        )


        capture[
            "batch_sums"
        ][
            layer_idx
        ] = batch_sum


    return hook


# ============================================================
# REGISTER ONLY REQUIRED LAYERS
# ============================================================

capture_handles = []


for layer_idx in (
    candidate_layer_indices
):

    handle = (
        layers[
            layer_idx
        ]
        .register_forward_hook(
            make_layer_capture_hook(
                layer_idx
            )
        )
    )


    capture_handles.append(
        handle
    )


print()
print(
    f"Residual hooks registered : "
    f"{len(capture_handles)}"
)


# ============================================================
# CLASS-MEAN RESIDUAL REPRESENTATIONS
# ============================================================

def compute_class_mean_residuals(
    class_df,
    description,
):

    """
    For every required layer l, calculate:

        mean_h_l(class)

    Sentence weighting:

        Each sentence contributes one mean residual vector.

    Therefore long sentences do not automatically receive
    more weight than short sentences.
    """

    layer_sums = {

        layer_idx:
            torch.zeros(
                hidden_size,
                dtype=torch.float64,
            )

        for layer_idx
        in candidate_layer_indices
    }


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

            batch = class_df.iloc[
                start:
                start
                + args.batch_size
            ]


            texts = (
                batch[
                    TEXT_COLUMN
                ]
                .tolist()
            )


            encoded_batch = (
                tokenizer(
                    texts,

                    padding=True,

                    truncation=True,

                    max_length=
                        args.max_length,

                    return_tensors=
                        "pt",
                )
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


            valid_mask = (
                create_valid_token_mask(
                    input_ids,
                    attention_mask,
                )
            )


            capture[
                "valid_mask"
            ] = valid_mask


            capture[
                "batch_sums"
            ] = {}


            _ = backbone(
                input_ids=
                    input_ids,

                attention_mask=
                    attention_mask,

                use_cache=False,
            )


            for layer_idx in (
                candidate_layer_indices
            ):

                if (
                    layer_idx
                    not in capture[
                        "batch_sums"
                    ]
                ):

                    raise RuntimeError(
                        "Residual hook failed "
                        f"for layer "
                        f"{layer_idx + 1}."
                    )


                layer_sums[
                    layer_idx
                ] += (
                    capture[
                        "batch_sums"
                    ][
                        layer_idx
                    ]
                )


            total_sentences += int(
                input_ids.shape[0]
            )


    if total_sentences == 0:

        raise RuntimeError(
            f"No examples processed "
            f"for {description}"
        )


    means = {

        layer_idx:
            (
                layer_sums[
                    layer_idx
                ]
                / total_sentences
            )
            .numpy()

        for layer_idx
        in candidate_layer_indices
    }


    return (
        means,
        total_sentences,
    )


# ============================================================
# COMPUTE JOY AND SADNESS MEANS
# ============================================================

print()
print("=" * 100)
print("COMPUTING MULTI-LAYER JOY/SADNESS RESIDUAL DIRECTIONS")
print("=" * 100)


try:

    (
        joy_layer_means,
        n_joy,
    ) = compute_class_mean_residuals(
        joy_df,
        "Joy residual means",
    )


    (
        sad_layer_means,
        n_sadness,
    ) = compute_class_mean_residuals(
        sad_df,
        "Sadness residual means",
    )


finally:

    for handle in (
        capture_handles
    ):

        handle.remove()


    capture[
        "valid_mask"
    ] = None


    capture[
        "batch_sums"
    ] = {}


# ============================================================
# CONSTRUCT LAYER-SPECIFIC JOY-SAD DIRECTIONS
# ============================================================

print()
print("=" * 100)
print("LAYER-SPECIFIC JOY-SADNESS DIRECTIONS")
print("=" * 100)


layer_directions = {}

layer_direction_rows = []


for layer_idx in (
    candidate_layer_indices
):

    layer_number = (
        layer_idx + 1
    )


    mean_joy = (
        joy_layer_means[
            layer_idx
        ]
    )


    mean_sad = (
        sad_layer_means[
            layer_idx
        ]
    )


    direction = (
        mean_joy
        - mean_sad
    )


    direction_norm = float(
        np.linalg.norm(
            direction
        )
    )


    if direction_norm < EPS:

        raise RuntimeError(
            f"Joy-Sad direction "
            f"at layer {layer_number} "
            f"has near-zero norm."
        )


    direction_unit = (
        direction
        / direction_norm
    )


    layer_directions[
        layer_idx
    ] = {

        "direction":
            direction,

        "unit":
            direction_unit,

        "norm":
            direction_norm,

        "mean_joy":
            mean_joy,

        "mean_sadness":
            mean_sad,
    }


    # --------------------------------------------------------
    # Save directions individually.
    # --------------------------------------------------------

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

        direction_unit.astype(
            np.float32
        ),
    )


    np.save(
        DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"mean_joy.npy"
        ),

        mean_joy.astype(
            np.float32
        ),
    )


    np.save(
        DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"mean_sadness.npy"
        ),

        mean_sad.astype(
            np.float32
        ),
    )


    layer_direction_rows.append(
        {

            "layer":
                layer_number,

            "layer_index":
                layer_idx,

            "num_joy_examples":
                n_joy,

            "num_sadness_examples":
                n_sadness,

            "joy_sad_direction_norm":
                direction_norm,
        }
    )


    print(
        f"Layer {layer_number:02d} | "
        f"||Joy-Sad|| = "
        f"{direction_norm:.8f}"
    )


layer_direction_df = (
    pd.DataFrame(
        layer_direction_rows
    )
)


layer_direction_file = (
    OUTPUT_DIR
    / "layer_direction_summary.csv"
)


layer_direction_df.to_csv(
    layer_direction_file,
    index=False,
)


# ============================================================
# VECTOR UTILITIES
# ============================================================

def vector_norm(
    vector,
):

    return float(
        np.linalg.norm(
            np.asarray(
                vector,
                dtype=np.float64,
            )
        )
    )


def cosine_similarity(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=np.float64,
    )


    b = np.asarray(
        b,
        dtype=np.float64,
    )


    norm_a = (
        vector_norm(
            a
        )
    )


    norm_b = (
        vector_norm(
            b
        )
    )


    if (
        norm_a < EPS
        or norm_b < EPS
    ):

        return float(
            "nan"
        )


    cosine = float(
        np.dot(
            a,
            b,
        )
        /
        (
            norm_a
            * norm_b
        )
    )


    return float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )


def angle_degrees(
    a,
    b,
):

    cosine = (
        cosine_similarity(
            a,
            b,
        )
    )


    if not np.isfinite(
        cosine
    ):

        return float(
            "nan"
        )


    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


# ============================================================
# SCORE EVERY CANDIDATE
# ============================================================

print()
print("=" * 100)
print("COMPUTING ACTUATOR SCORES")
print("=" * 100)


candidate_results = []


for _, row in (
    candidates.iterrows()
):

    layer_number = int(
        row[
            "layer"
        ]
    )


    layer_idx = int(
        row[
            "layer_index"
        ]
    )


    neuron_idx = int(
        row[
            "neuron_index"
        ]
    )


    neuron_number = int(
        row[
            "neuron_number"
        ]
    )


    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if (
        layer_number
        != layer_idx + 1
    ):

        raise RuntimeError(
            "Layer numbering mismatch:\n"
            f"layer={layer_number}, "
            f"layer_index={layer_idx}"
        )


    if (
        neuron_idx < 0
        or neuron_idx
        >= mlp_width
    ):

        raise RuntimeError(
            f"Invalid neuron index "
            f"{neuron_idx}"
        )


    # --------------------------------------------------------
    # Activation contrast
    # --------------------------------------------------------

    mu_joy = float(
        row[
            "mean_activation_joy"
        ]
    )


    mu_sad = float(
        row[
            "mean_activation_sadness"
        ]
    )


    # Recalculate to avoid relying only
    # on the stored difference column.
    delta_mu = (
        mu_joy
        - mu_sad
    )


    stored_delta_mu = float(
        row[
            "raw_activation_difference"
        ]
    )


    if not np.isclose(
        delta_mu,
        stored_delta_mu,
        rtol=1e-6,
        atol=1e-10,
    ):

        print(
            "WARNING: activation difference "
            f"mismatch for layer {layer_number}, "
            f"neuron {neuron_idx}"
        )


    # --------------------------------------------------------
    # Layer-specific emotion direction
    # --------------------------------------------------------

    emotion_direction = (
        layer_directions[
            layer_idx
        ][
            "direction"
        ]
    )


    emotion_unit = (
        layer_directions[
            layer_idx
        ][
            "unit"
        ]
    )


    emotion_direction_norm = float(
        layer_directions[
            layer_idx
        ][
            "norm"
        ]
    )


    # --------------------------------------------------------
    # Physical neuron write direction
    #
    # PyTorch Linear:
    #
    # down_proj.weight shape:
    #
    # [hidden_size, intermediate_size]
    #
    # Neuron u contributes along:
    #
    # W_down[:, u]
    # --------------------------------------------------------

    write_vector = (
        layers[
            layer_idx
        ]
        .mlp
        .down_proj
        .weight[
            :,
            neuron_idx
        ]
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


    write_norm = (
        vector_norm(
            write_vector
        )
    )


    if write_norm < EPS:

        write_cosine = 0.0

        write_angle = 90.0

        write_projection = 0.0

    else:

        # ----------------------------------------------------
        # W_down alignment itself.
        # ----------------------------------------------------

        write_cosine = (
            cosine_similarity(
                write_vector,
                emotion_direction,
            )
        )


        write_angle = (
            angle_degrees(
                write_vector,
                emotion_direction,
            )
        )


        # ----------------------------------------------------
        # Scalar projection of W_down onto
        # unit Joy-Sad direction.
        #
        # Units:
        # residual-vector units per one unit
        # of neuron activation.
        # ----------------------------------------------------

        write_projection = float(
            np.dot(
                write_vector,
                emotion_unit,
            )
        )


    # --------------------------------------------------------
    # ACTUAL alpha=1 intervention:
    #
    # Delta_h =
    # Delta_mu * W_down[:,u]
    # --------------------------------------------------------

    expected_residual_shift = (
        delta_mu
        * write_vector
    )


    expected_residual_norm = (
        vector_norm(
            expected_residual_shift
        )
    )


    # --------------------------------------------------------
    # Main actuator score:
    #
    # Q =
    # Delta_h dot unit(Joy-Sad)
    #
    #   =
    # Delta_mu
    # *
    # W_down[:,u]^T d_hat
    #
    # Positive:
    # expected movement toward Joy.
    # --------------------------------------------------------

    actuator_score_q = float(
        np.dot(
            expected_residual_shift,
            emotion_unit,
        )
    )


    # Mathematically equivalent calculation.
    actuator_score_from_projection = (
        delta_mu
        * write_projection
    )


    if not np.isclose(
        actuator_score_q,
        actuator_score_from_projection,
        rtol=1e-6,
        atol=1e-10,
    ):

        raise RuntimeError(
            "Actuator score implementation "
            "consistency check failed."
        )


    # --------------------------------------------------------
    # Direction of the actual alpha=1 intervention.
    #
    # This already includes the sign of Delta_mu.
    # --------------------------------------------------------

    if expected_residual_norm > EPS:

        intervention_cosine = (
            cosine_similarity(
                expected_residual_shift,
                emotion_direction,
            )
        )


        intervention_angle = (
            angle_degrees(
                expected_residual_shift,
                emotion_direction,
            )
        )

    else:

        intervention_cosine = float(
            "nan"
        )

        intervention_angle = float(
            "nan"
        )


    # --------------------------------------------------------
    # Relative amount of alpha=1 perturbation
    # pointing along the desired emotion axis.
    #
    # This equals intervention cosine,
    # but keeping the column makes the interpretation clear.
    # --------------------------------------------------------

    if expected_residual_norm > EPS:

        joy_axis_fraction = (
            actuator_score_q
            / expected_residual_norm
        )

    else:

        joy_axis_fraction = float(
            "nan"
        )


    # --------------------------------------------------------
    # Preserve original candidate information.
    # --------------------------------------------------------

    result = row.to_dict()


    result.update(
        {

            "recomputed_activation_difference":
                delta_mu,

            "joy_sad_residual_direction_norm":
                emotion_direction_norm,

            "write_direction_norm":
                write_norm,

            "write_direction_cosine_joy_sad":
                write_cosine,

            "write_direction_angle_joy_sad_deg":
                write_angle,

            "write_projection_on_joy_axis":
                write_projection,

            "expected_residual_shift_norm_alpha1":
                expected_residual_norm,

            "intervention_direction_cosine_joy_sad":
                intervention_cosine,

            "intervention_direction_angle_joy_sad_deg":
                intervention_angle,

            "joy_axis_fraction_alpha1":
                joy_axis_fraction,

            "actuator_score_q":
                actuator_score_q,

            "actuator_score_sign":
                (
                    "toward_joy"
                    if actuator_score_q > 0

                    else (
                        "away_from_joy"
                        if actuator_score_q < 0

                        else "orthogonal"
                    )
                ),
        }
    )


    candidate_results.append(
        result
    )


# ============================================================
# RANK BY ACTUATOR SCORE
# ============================================================

ranking_df = pd.DataFrame(
    candidate_results
)


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

ranking_df[
    "positive_actuator_rank"
] = np.nan


positive_mask = (
    ranking_df[
        "actuator_score_q"
    ]
    > 0
)


num_positive = int(
    positive_mask.sum()
)


if num_positive > 0:

    ranking_df.loc[
        positive_mask,
        "positive_actuator_rank",
    ] = np.arange(
        1,
        num_positive + 1,
    )


# ============================================================
# FLAG TOP POSITIVE CANDIDATES FOR NEXT EXPERIMENT
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


num_to_select = min(
    args.top_k_next,
    len(
        positive_indices
    ),
)


if num_to_select > 0:

    selected_indices = (
        positive_indices[
            :num_to_select
        ]
    )


    ranking_df.loc[
        selected_indices,
        "recommended_for_next_steering_test",
    ] = True


# ============================================================
# REORDER MAIN COLUMNS
# ============================================================

priority_columns = [

    "actuator_rank",

    "positive_actuator_rank",

    "recommended_for_next_steering_test",

    "layer",

    "layer_index",

    "neuron_index",

    "neuron_number",

    "steering_rank",

    "original_rank",

    "original_selectivity_score",

    "mean_activation_joy",

    "mean_activation_sadness",

    "raw_activation_difference",

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

    "joy_axis_fraction_alpha1",

    "actuator_score_q",

    "actuator_score_sign",
]


remaining_columns = [

    column

    for column
    in ranking_df.columns

    if column
    not in priority_columns
]


ranking_df = ranking_df[
    priority_columns
    + remaining_columns
]


# ============================================================
# SAVE CSV
# ============================================================

ranking_csv = (
    OUTPUT_DIR
    / "joy_actuator_ranking.csv"
)


ranking_df.to_csv(
    ranking_csv,
    index=False,
)


# ============================================================
# BEST POSITIVE ACTUATOR
# ============================================================

positive_candidates = (
    ranking_df.loc[
        ranking_df[
            "actuator_score_q"
        ]
        > 0
    ]
    .copy()
)


if not positive_candidates.empty:

    best_row = (
        positive_candidates
        .iloc[0]
    )


    best_positive_actuator = {

        "actuator_rank":
            int(
                best_row[
                    "actuator_rank"
                ]
            ),

        "layer":
            int(
                best_row[
                    "layer"
                ]
            ),

        "layer_index":
            int(
                best_row[
                    "layer_index"
                ]
            ),

        "neuron_index":
            int(
                best_row[
                    "neuron_index"
                ]
            ),

        "neuron_number":
            int(
                best_row[
                    "neuron_number"
                ]
            ),

        "original_steering_rank":
            int(
                best_row[
                    "steering_rank"
                ]
            ),

        "original_selectivity_rank":
            int(
                best_row[
                    "original_rank"
                ]
            ),

        "mean_activation_joy":
            float(
                best_row[
                    "mean_activation_joy"
                ]
            ),

        "mean_activation_sadness":
            float(
                best_row[
                    "mean_activation_sadness"
                ]
            ),

        "raw_activation_difference":
            float(
                best_row[
                    "raw_activation_difference"
                ]
            ),

        "standardized_joy_minus_sadness":
            float(
                best_row[
                    "standardized_joy_minus_sadness"
                ]
            ),

        "write_direction_norm":
            float(
                best_row[
                    "write_direction_norm"
                ]
            ),

        "write_direction_cosine_joy_sad":
            float(
                best_row[
                    "write_direction_cosine_joy_sad"
                ]
            ),

        "write_projection_on_joy_axis":
            float(
                best_row[
                    "write_projection_on_joy_axis"
                ]
            ),

        "expected_residual_shift_norm_alpha1":
            float(
                best_row[
                    "expected_residual_shift_norm_alpha1"
                ]
            ),

        "intervention_direction_cosine_joy_sad":
            float(
                best_row[
                    "intervention_direction_cosine_joy_sad"
                ]
            ),

        "intervention_direction_angle_joy_sad_deg":
            float(
                best_row[
                    "intervention_direction_angle_joy_sad_deg"
                ]
            ),

        "actuator_score_q":
            float(
                best_row[
                    "actuator_score_q"
                ]
            ),

        "actuator_interpretation":
            (
                "Positive alpha=1 intervention is expected "
                "to move the residual representation toward "
                "the layer-specific Joy-Sadness direction."
            ),
    }

else:

    best_positive_actuator = None


# ============================================================
# TOP-K RECOMMENDED CANDIDATES
# ============================================================

recommended_df = (
    ranking_df.loc[
        ranking_df[
            "recommended_for_next_steering_test"
        ]
    ]
    .copy()
)


recommended_neurons = []


for _, row in (
    recommended_df.iterrows()
):

    recommended_neurons.append(
        {

            "actuator_rank":
                int(
                    row[
                        "actuator_rank"
                    ]
                ),

            "layer":
                int(
                    row[
                        "layer"
                    ]
                ),

            "layer_index":
                int(
                    row[
                        "layer_index"
                    ]
                ),

            "neuron_index":
                int(
                    row[
                        "neuron_index"
                    ]
                ),

            "neuron_number":
                int(
                    row[
                        "neuron_number"
                    ]
                ),

            "activation_difference":
                float(
                    row[
                        "raw_activation_difference"
                    ]
                ),

            "standardized_activation_contrast":
                float(
                    row[
                        "standardized_joy_minus_sadness"
                    ]
                ),

            "write_cosine":
                float(
                    row[
                        "write_direction_cosine_joy_sad"
                    ]
                ),

            "intervention_cosine":
                float(
                    row[
                        "intervention_direction_cosine_joy_sad"
                    ]
                ),

            "actuator_score_q":
                float(
                    row[
                        "actuator_score_q"
                    ]
                ),
        }
    )


# ============================================================
# SAVE JSON
# ============================================================

ranking_json = (
    OUTPUT_DIR
    / "joy_actuator_ranking.json"
)


json_output = {

    "experiment":
        (
            "Rank exact-degree-1 joy neurons "
            "by causal actuator geometry"
        ),

    "model":
        MODEL_ID,

    "target_emotion":
        TARGET_EMOTION,

    "source_emotion":
        SOURCE_EMOTION,

    "num_candidates":
        int(
            len(
                ranking_df
            )
        ),

    "num_positive_actuators":
        num_positive,

    "num_joy_examples":
        int(
            n_joy
        ),

    "num_sadness_examples":
        int(
            n_sadness
        ),

    "candidate_layers":
        candidate_layer_numbers,

    "selection_definition":
        {

            "activation_difference":
                (
                    "Delta_mu = "
                    "mean_activation_joy "
                    "- mean_activation_sadness"
                ),

            "write_direction":
                (
                    "v_u = "
                    "W_down[layer][:, neuron]"
                ),

            "emotion_direction":
                (
                    "d_l = "
                    "mean_residual_joy(layer) "
                    "- mean_residual_sadness(layer)"
                ),

            "actuator_score":
                (
                    "Q_u = Delta_mu * "
                    "(v_u^T unit(d_l))"
                ),

            "interpretation":
                (
                    "Q_u > 0 means alpha=+1 is expected "
                    "to move the residual representation "
                    "toward Joy relative to Sadness."
                ),
        },

    "best_positive_actuator":
        best_positive_actuator,

    "recommended_for_next_steering_test":
        recommended_neurons,

    "ranking":
        (
            ranking_df
            .replace(
                {
                    np.nan:
                        None,
                }
            )
            .to_dict(
                orient="records"
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
# SAVE BEST ACTUATOR JSON SEPARATELY
# ============================================================

best_file = (
    OUTPUT_DIR
    / "best_joy_actuator_neuron.json"
)


with best_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {

            "experiment":
                (
                    "Best positive-Q joy actuator neuron"
                ),

            "target_emotion":
                TARGET_EMOTION,

            "source_emotion":
                SOURCE_EMOTION,

            "selection_rule":
                (
                    "largest positive actuator_score_q"
                ),

            "best_neuron":
                best_positive_actuator,

            "recommended_top_candidates":
                recommended_neurons,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# SAVE BEST NEURON WRITE DIRECTION
# ============================================================

if best_positive_actuator is not None:

    best_layer_idx = int(
        best_positive_actuator[
            "layer_index"
        ]
    )


    best_neuron_idx = int(
        best_positive_actuator[
            "neuron_index"
        ]
    )


    best_write_direction = (
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
            best_write_direction
        )
    )


    np.save(
        OUTPUT_DIR
        / "best_joy_actuator_write_direction.npy",

        best_write_direction,
    )


    if best_write_norm > EPS:

        np.save(
            OUTPUT_DIR
            / "best_joy_actuator_write_direction_unit.npy",

            (
                best_write_direction
                / best_write_norm
            ).astype(
                np.float32
            ),
        )


# ============================================================
# RANKING FIGURE
# ============================================================

print()
print("=" * 100)
print("GENERATING RANKING FIGURE")
print("=" * 100)


plot_df = (
    ranking_df
    .sort_values(
        "actuator_score_q",
        ascending=True,
    )
    .copy()
)


labels = []


for _, row in (
    plot_df.iterrows()
):

    label = (
        f"L{int(row['layer'])}"
        f"-N{int(row['neuron_index'])}"
    )


    # Mark the neuron we tested first.
    if (
        int(
            row[
                "layer"
            ]
        ) == 20
        and
        int(
            row[
                "neuron_index"
            ]
        ) == 10707
    ):

        label += " [current]"


    labels.append(
        label
    )


plot_df[
    "plot_label"
] = labels


fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(
        16,
        8,
    ),
)


# ------------------------------------------------------------
# PANEL A:
# Main Q actuator score
# ------------------------------------------------------------

ax1 = axes[0]


bars1 = ax1.barh(
    plot_df[
        "plot_label"
    ],
    plot_df[
        "actuator_score_q"
    ],
)


ax1.axvline(
    0,
    linewidth=1,
)


ax1.set_title(
    (
        "A. Joy actuator score\n"
        r"$Q_u=\Delta\mu_u"
        r"(W_{down,u}^{T}\hat d_{J-S})$"
    )
)


ax1.set_xlabel(
    (
        "Expected alpha=1 displacement "
        "along Joy-Sadness axis"
    )
)


ax1.set_ylabel(
    "Candidate neuron"
)


ax1.grid(
    True,
    axis="x",
    alpha=0.20,
)


for bar, value in zip(
    bars1,
    plot_df[
        "actuator_score_q"
    ],
):

    value = float(
        value
    )


    x_position = (
        value
        + (
            0.002
            if value >= 0
            else -0.002
        )
    )


    horizontal_alignment = (
        "left"
        if value >= 0
        else "right"
    )


    ax1.text(
        x_position,
        (
            bar.get_y()
            + bar.get_height()
            / 2
        ),
        f"{value:+.5f}",
        va="center",
        ha=
            horizontal_alignment,
        fontsize=8,
    )


# ------------------------------------------------------------
# PANEL B:
# Actual intervention cosine
# ------------------------------------------------------------

ax2 = axes[1]


bars2 = ax2.barh(
    plot_df[
        "plot_label"
    ],
    plot_df[
        "intervention_direction_cosine_joy_sad"
    ],
)


ax2.axvline(
    0,
    linewidth=1,
)


ax2.set_xlim(
    -1.0,
    1.0,
)


ax2.set_title(
    (
        "B. Alignment of alpha=1 intervention\n"
        r"$\cos(\Delta h_u,d_{J-S})$"
    )
)


ax2.set_xlabel(
    (
        "Direction cosine "
        "(+1 toward Joy, -1 opposite)"
    )
)


ax2.set_ylabel(
    "Candidate neuron"
)


ax2.grid(
    True,
    axis="x",
    alpha=0.20,
)


for bar, value in zip(
    bars2,
    plot_df[
        "intervention_direction_cosine_joy_sad"
    ],
):

    value = float(
        value
    )


    x_position = (
        value
        + (
            0.025
            if value >= 0
            else -0.025
        )
    )


    horizontal_alignment = (
        "left"
        if value >= 0
        else "right"
    )


    ax2.text(
        x_position,
        (
            bar.get_y()
            + bar.get_height()
            / 2
        ),
        f"{value:+.3f}",
        va="center",
        ha=
            horizontal_alignment,
        fontsize=8,
    )


fig.suptitle(
    (
        "Exact-Degree-1 Joy Neurons: "
        "Marker vs Actuator Ranking\n"
        f"{n_joy:,} joy examples, "
        f"{n_sadness:,} sadness examples"
    ),
    fontsize=15,
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
# TERMINAL RESULT TABLE
# ============================================================

print()
print("=" * 100)
print("FINAL ACTUATOR RANKING")
print("=" * 100)


display_columns = [

    "actuator_rank",

    "layer",

    "neuron_index",

    "raw_activation_difference",

    "standardized_joy_minus_sadness",

    "write_direction_cosine_joy_sad",

    "write_projection_on_joy_axis",

    "expected_residual_shift_norm_alpha1",

    "intervention_direction_cosine_joy_sad",

    "actuator_score_q",

    "actuator_score_sign",
]


print(
    ranking_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# RECOMMENDED TOP CANDIDATES
# ============================================================

print()
print("=" * 100)
print("RECOMMENDED FOR NEXT STEERING TEST")
print("=" * 100)


if recommended_df.empty:

    print(
        "No positive-Q actuator neuron found."
    )

else:

    print(
        recommended_df[
            [
                "actuator_rank",
                "layer",
                "neuron_index",
                "raw_activation_difference",
                "standardized_joy_minus_sadness",
                "intervention_direction_cosine_joy_sad",
                "actuator_score_q",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# BEST ACTUATOR
# ============================================================

print()
print("=" * 100)
print("BEST POSITIVE ACTUATOR")
print("=" * 100)


if best_positive_actuator is None:

    print(
        "No candidate had positive actuator_score_q."
    )

else:

    print(
        f"Layer                  : "
        f"{best_positive_actuator['layer']}"
    )


    print(
        f"Layer index            : "
        f"{best_positive_actuator['layer_index']}"
    )


    print(
        f"Neuron index           : "
        f"{best_positive_actuator['neuron_index']}"
    )


    print(
        f"Neuron number          : "
        f"{best_positive_actuator['neuron_number']}"
    )


    print(
        f"Activation difference  : "
        f"{best_positive_actuator['raw_activation_difference']:+.8f}"
    )


    print(
        f"Standardized contrast  : "
        f"{best_positive_actuator['standardized_joy_minus_sadness']:+.8f}"
    )


    print(
        f"Write cosine           : "
        f"{best_positive_actuator['write_direction_cosine_joy_sad']:+.8f}"
    )


    print(
        f"Write projection       : "
        f"{best_positive_actuator['write_projection_on_joy_axis']:+.8f}"
    )


    print(
        f"Intervention cosine    : "
        f"{best_positive_actuator['intervention_direction_cosine_joy_sad']:+.8f}"
    )


    print(
        f"Intervention angle     : "
        f"{best_positive_actuator['intervention_direction_angle_joy_sad_deg']:.4f} deg"
    )


    print(
        f"Actuator score Q       : "
        f"{best_positive_actuator['actuator_score_q']:+.8f}"
    )


# ============================================================
# OUTPUTS
# ============================================================

print()
print("=" * 100)
print("SAVED OUTPUTS")
print("=" * 100)


print(
    f"Ranking CSV:\n"
    f"{ranking_csv}"
)


print()
print(
    f"Ranking JSON:\n"
    f"{ranking_json}"
)


print()
print(
    f"Best actuator JSON:\n"
    f"{best_file}"
)


print()
print(
    f"Ranking PNG:\n"
    f"{ranking_png}"
)


print()
print(
    f"Layer directions:\n"
    f"{DIRECTION_DIR}"
)


print()
print(
    f"Layer summary:\n"
    f"{layer_direction_file}"
)


print()
print("Done.")
