#!/usr/bin/env python3

"""
Mechanistic analysis of the selected single joy neuron.

Selected neuron:
    loaded from best_joy_neuron.json

PART A
------
Construct an empirical Layer-20 joy-vs-sadness residual direction:

    d_hidden = mean_h(joy) - mean_h(sadness)

where h is the transformer residual representation AFTER Layer 20.

Compare this with the neuron's own write direction:

    v_neuron = W_down[:, neuron]

using cosine similarity:

    cos(v_neuron, d_hidden)

PART B
------
Measure next-token distribution changes produced by steering.

Important:
    We first generate ONE unsteered greedy continuation.

    Then every alpha is evaluated while teacher-forcing exactly
    that same baseline continuation.

Therefore every alpha sees the SAME token sequence.

This isolates:

    intervention -> hidden state -> logits

instead of confounding the analysis with:

    intervention -> different token -> different context -> logits

Steering rule:

    m' = m + alpha * (mu_joy - mu_sadness)

The prompt forward pass is NOT steered.
Steering starts after the first generated token, matching the
previous steering experiment.

Outputs:
    direction_alignment.json
    joy_minus_sad_hidden_direction.npy
    token_level_logit_analysis.csv
    alpha_logit_summary.csv
    baseline_continuation.txt
    full_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    default=(
        "Aaj mera din bahut kharab gaya. "
        "Mujhe lag raha hai ki kuch bhi sahi nahi ho raha."
    ),
)

parser.add_argument(
    "--alphas",
    type=float,
    nargs="+",
    default=[
        -10.0,
        -5.0,
        -3.0,
        -1.0,
        0.0,
        1.0,
        3.0,
        5.0,
        10.0,
    ],
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=60,
)

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
        "0 = use all joy and sadness examples "
        "for residual-direction estimation. "
        "Use e.g. 300 for a quick test."
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

BEST_NEURON_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "13_single_joy_neuron_selection"
    / "best_joy_neuron.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "15_direction_and_logit_analysis"
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


print("=" * 100)
print("DIRECTION ALIGNMENT + TOKEN-LEVEL LOGIT ANALYSIS")
print("=" * 100)

print(f"Device : {DEVICE}")
print(f"Dtype  : {DTYPE}")


# ============================================================
# LOAD SELECTED NEURON METADATA
# ============================================================

if not BEST_NEURON_FILE.is_file():

    raise FileNotFoundError(
        f"Missing:\n{BEST_NEURON_FILE}"
    )


with BEST_NEURON_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    metadata = json.load(f)


best = metadata[
    "best_neuron"
]


LAYER_NUMBER = int(
    best["layer"]
)

LAYER_IDX = int(
    best["layer_index"]
)

NEURON_IDX = int(
    best["neuron_index"]
)


MU_JOY = float(
    best["mean_activation_joy"]
)

MU_SAD = float(
    best["mean_activation_sadness"]
)


RAW_DIRECTION = float(
    best["raw_activation_difference"]
)


POOLED_SD = float(
    best["pooled_sd"]
)


STANDARDIZED_EFFECT = float(
    best["standardized_joy_minus_sadness"]
)


print()
print("SELECTED NEURON")
print("-" * 100)

print(f"Layer number              : {LAYER_NUMBER}")
print(f"Layer index               : {LAYER_IDX}")
print(f"Neuron index              : {NEURON_IDX}")

print(f"Mean joy activation       : {MU_JOY:+.8f}")
print(f"Mean sadness activation   : {MU_SAD:+.8f}")

print(f"Raw joy-sad difference    : {RAW_DIRECTION:+.8f}")
print(f"Pooled SD                 : {POOLED_SD:.8f}")
print(f"Standardized effect       : {STANDARDIZED_EFFECT:+.8f}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("=" * 100)
print("LOADING TOKENIZER")
print("=" * 100)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING SARVAM-1")
print("=" * 100)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
)


model.to(DEVICE)
model.eval()


for parameter in model.parameters():

    parameter.requires_grad_(False)


layers = model.model.layers

backbone = model.model


if LAYER_IDX >= len(layers):

    raise RuntimeError(
        f"Invalid layer index {LAYER_IDX}"
    )


hidden_size = int(
    layers[
        LAYER_IDX
    ]
    .mlp
    .down_proj
    .weight
    .shape[0]
)


mlp_width = int(
    layers[
        LAYER_IDX
    ]
    .mlp
    .down_proj
    .weight
    .shape[1]
)


print(
    f"Hidden size              : {hidden_size}"
)

print(
    f"MLP width                : {mlp_width}"
)


# ============================================================
# EXTRACT NEURON WRITE DIRECTION
# ============================================================

down_weight = (
    layers[
        LAYER_IDX
    ]
    .mlp
    .down_proj
    .weight
    .detach()
    .float()
    .cpu()
)


neuron_write_direction = (
    down_weight[
        :,
        NEURON_IDX
    ]
    .numpy()
    .astype(np.float64)
)


neuron_write_norm = float(
    np.linalg.norm(
        neuron_write_direction
    )
)


neuron_write_unit = (
    neuron_write_direction
    / (
        neuron_write_norm
        + EPS
    )
)


print(
    f"W_down neuron norm       : {neuron_write_norm:.8f}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print()
print("=" * 100)
print("PART A — BUILDING LAYER-20 JOY-vs-SADNESS RESIDUAL DIRECTION")
print("=" * 100)


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


if args.max_per_class > 0:

    joy_df = (
        joy_df.sample(
            n=min(
                args.max_per_class,
                len(joy_df),
            ),
            random_state=SEED,
        )
        .reset_index(drop=True)
    )


    sad_df = (
        sad_df.sample(
            n=min(
                args.max_per_class,
                len(sad_df),
            ),
            random_state=SEED,
        )
        .reset_index(drop=True)
    )


print(
    f"Joy examples             : {len(joy_df):,}"
)

print(
    f"Sadness examples         : {len(sad_df):,}"
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


    for special_id in tokenizer.all_special_ids:

        mask &= input_ids.ne(
            special_id
        )


    return mask


# ============================================================
# DATASET LAYER CAPTURE
# ============================================================

dataset_capture = {

    "valid_mask": None,

    "sentence_means": None,
}


def dataset_layer_hook(
    module,
    inputs,
    output,
):

    if torch.is_tensor(output):

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

        hidden = output.last_hidden_state

    else:

        raise RuntimeError(
            f"Unsupported layer output: "
            f"{type(output)}"
        )


    valid_mask = dataset_capture[
        "valid_mask"
    ]


    if valid_mask is None:

        raise RuntimeError(
            "valid_mask missing."
        )


    mask = (
        valid_mask
        .to(hidden.device)
        .unsqueeze(-1)
        .to(hidden.dtype)
    )


    numerator = (
        hidden
        * mask
    ).sum(
        dim=1
    )


    denominator = (
        mask.sum(dim=1)
        .clamp_min(1.0)
    )


    sentence_means = (
        numerator
        / denominator
    )


    dataset_capture[
        "sentence_means"
    ] = (
        sentence_means
        .detach()
        .float()
        .cpu()
    )


dataset_handle = (
    layers[
        LAYER_IDX
    ]
    .register_forward_hook(
        dataset_layer_hook
    )
)


# ============================================================
# COMPUTE CLASS-MEAN RESIDUAL VECTOR
# ============================================================

def class_mean_hidden(
    class_df,
    description,
):

    hidden_sum = torch.zeros(
        hidden_size,
        dtype=torch.float64,
    )

    count = 0


    with torch.inference_mode():

        for start in tqdm(
            range(
                0,
                len(class_df),
                args.batch_size,
            ),
            desc=description,
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


            dataset_capture[
                "valid_mask"
            ] = (
                create_valid_token_mask(
                    input_ids,
                    attention_mask,
                )
            )


            dataset_capture[
                "sentence_means"
            ] = None


            _ = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )


            sentence_means = (
                dataset_capture[
                    "sentence_means"
                ]
            )


            if sentence_means is None:

                raise RuntimeError(
                    "Layer hook failed."
                )


            hidden_sum += (
                sentence_means
                .double()
                .sum(dim=0)
            )


            count += int(
                sentence_means.shape[0]
            )


    return (
        hidden_sum
        / max(
            count,
            1,
        )
    ).numpy()


try:

    mean_hidden_joy = (
        class_mean_hidden(
            joy_df,
            "Joy residuals",
        )
    )


    mean_hidden_sad = (
        class_mean_hidden(
            sad_df,
            "Sadness residuals",
        )
    )

finally:

    dataset_handle.remove()


dataset_capture[
    "valid_mask"
] = None


# ============================================================
# JOY-SAD HIDDEN DIRECTION
# ============================================================

joy_sad_hidden_direction = (
    mean_hidden_joy
    - mean_hidden_sad
)


joy_sad_hidden_norm = float(
    np.linalg.norm(
        joy_sad_hidden_direction
    )
)


if joy_sad_hidden_norm < EPS:

    raise RuntimeError(
        "Joy-sad hidden direction "
        "has near-zero norm."
    )


joy_sad_hidden_unit = (
    joy_sad_hidden_direction
    / joy_sad_hidden_norm
)


# ============================================================
# COSINE ALIGNMENT
# ============================================================

dot_product = float(
    np.dot(
        neuron_write_direction,
        joy_sad_hidden_direction,
    )
)


cosine_alignment = float(
    dot_product
    /
    (
        neuron_write_norm
        * joy_sad_hidden_norm
        + EPS
    )
)


cosine_alignment = float(
    np.clip(
        cosine_alignment,
        -1.0,
        1.0,
    )
)


angle_degrees = float(
    np.degrees(
        np.arccos(
            cosine_alignment
        )
    )
)


# Projection of neuron direction onto
# empirical joy-sad axis.

write_projection_on_joy_axis = float(
    np.dot(
        neuron_write_direction,
        joy_sad_hidden_unit,
    )
)


fraction_along_joy_axis = float(
    write_projection_on_joy_axis
    /
    (
        neuron_write_norm
        + EPS
    )
)


# Expected joy-axis movement for alpha=1.

alpha1_joy_axis_shift = float(
    RAW_DIRECTION
    * write_projection_on_joy_axis
)


alpha1_residual_norm = float(
    abs(
        RAW_DIRECTION
    )
    * neuron_write_norm
)


print()
print("=" * 100)
print("RESIDUAL-DIRECTION ALIGNMENT")
print("=" * 100)

print(
    f"||joy-sad residual direction|| : "
    f"{joy_sad_hidden_norm:.8f}"
)

print(
    f"||neuron W_down direction||    : "
    f"{neuron_write_norm:.8f}"
)

print(
    f"Cosine alignment               : "
    f"{cosine_alignment:+.8f}"
)

print(
    f"Angle                          : "
    f"{angle_degrees:.4f} degrees"
)

print(
    f"W_down projection on joy axis  : "
    f"{write_projection_on_joy_axis:+.8f}"
)

print(
    f"Fraction along joy axis         : "
    f"{fraction_along_joy_axis:+.8f}"
)

print(
    f"Expected joy-axis shift α=1     : "
    f"{alpha1_joy_axis_shift:+.8f}"
)

print(
    f"Expected ||Delta h|| α=1        : "
    f"{alpha1_residual_norm:.8f}"
)


# ============================================================
# SAVE DIRECTIONS
# ============================================================

np.save(
    OUTPUT_DIR
    / "joy_minus_sad_hidden_direction.npy",
    joy_sad_hidden_direction.astype(
        np.float32
    ),
)


np.save(
    OUTPUT_DIR
    / "joy_minus_sad_hidden_direction_unit.npy",
    joy_sad_hidden_unit.astype(
        np.float32
    ),
)


direction_alignment = {

    "layer":
        LAYER_NUMBER,

    "layer_index":
        LAYER_IDX,

    "neuron_index":
        NEURON_IDX,

    "num_joy_examples":
        len(joy_df),

    "num_sadness_examples":
        len(sad_df),

    "joy_sad_hidden_direction_norm":
        joy_sad_hidden_norm,

    "neuron_write_direction_norm":
        neuron_write_norm,

    "cosine_alignment":
        cosine_alignment,

    "angle_degrees":
        angle_degrees,

    "write_projection_on_joy_axis":
        write_projection_on_joy_axis,

    "fraction_of_neuron_direction_along_joy_axis":
        fraction_along_joy_axis,

    "raw_activation_steering_unit":
        RAW_DIRECTION,

    "expected_residual_norm_alpha1":
        alpha1_residual_norm,

    "expected_joy_axis_shift_alpha1":
        alpha1_joy_axis_shift,
}


with (
    OUTPUT_DIR
    / "direction_alignment.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        direction_alignment,
        f,
        indent=2,
    )


# ============================================================
# PART B
# GENERATION-TIME LOGIT ANALYSIS
# ============================================================

print()
print("=" * 100)
print("PART B — FIXED-TRAJECTORY NEXT-TOKEN LOGIT ANALYSIS")
print("=" * 100)

print()
print("Prompt:")
print(args.prompt)


# ============================================================
# GENERATION CAPTURE
# ============================================================

generation_capture = {

    "layer_output":
        None,
}


def generation_layer_hook(
    module,
    inputs,
    output,
):

    if torch.is_tensor(output):

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

        hidden = output.last_hidden_state

    else:

        raise RuntimeError(
            f"Unsupported layer output: "
            f"{type(output)}"
        )


    generation_capture[
        "layer_output"
    ] = (
        hidden[
            :,
            -1,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


generation_layer_handle = (
    layers[
        LAYER_IDX
    ]
    .register_forward_hook(
        generation_layer_hook
    )
)


# ============================================================
# STEERING STATE
# ============================================================

steering_state = {

    "enabled":
        False,

    "delta":
        0.0,
}


def steering_hook(
    module,
    inputs,
):

    if (
        not inputs
        or not steering_state[
            "enabled"
        ]
    ):

        return inputs


    x = inputs[0]


    modified = x.clone()


    modified[
        ...,
        NEURON_IDX
    ] = (
        modified[
            ...,
            NEURON_IDX
        ]
        + float(
            steering_state[
                "delta"
            ]
        )
    )


    return (
        modified,
        *inputs[1:],
    )


steering_handle = (
    layers[
        LAYER_IDX
    ]
    .mlp
    .down_proj
    .register_forward_pre_hook(
        steering_hook
    )
)


# ============================================================
# PROMPT TOKENIZATION
# ============================================================

prompt_encoding = tokenizer(
    args.prompt,
    return_tensors="pt",
)


prompt_ids = (
    prompt_encoding[
        "input_ids"
    ]
    .to(DEVICE)
)


prompt_attention_mask = (
    prompt_encoding[
        "attention_mask"
    ]
    .to(DEVICE)
)


# ============================================================
# BUILD UNSTEERED GREEDY BASELINE
# ============================================================

def build_baseline():

    steering_state[
        "enabled"
    ] = False


    generation_capture[
        "layer_output"
    ] = None


    with torch.inference_mode():

        output = model(
            input_ids=prompt_ids,
            attention_mask=
                prompt_attention_mask,
            use_cache=True,
        )


    past = output.past_key_values


    logits = (
        output.logits[
            0,
            -1,
            :
        ]
        .detach()
        .float()
        .cpu()
    )


    first_token = int(
        torch.argmax(
            logits
        ).item()
    )


    generated_ids = [
        first_token
    ]


    # Index j contains logits used to choose generated token j.
    baseline_logits = [
        logits
    ]


    # Index 0 has no comparable steered layer-state because
    # prompt forward is deliberately unsteered.
    baseline_layer_states = [
        None
    ]


    attention_mask = (
        prompt_attention_mask.clone()
    )


    current_token = torch.tensor(
        [
            [
                first_token
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )


    if (
        tokenizer.eos_token_id
        is not None
        and first_token
        == tokenizer.eos_token_id
    ):

        return (
            generated_ids,
            baseline_logits,
            baseline_layer_states,
        )


    for _ in range(
        args.max_new_tokens - 1
    ):

        extra_mask = torch.ones(
            (
                attention_mask.shape[0],
                1,
            ),
            dtype=attention_mask.dtype,
            device=DEVICE,
        )


        attention_mask = torch.cat(
            [
                attention_mask,
                extra_mask,
            ],
            dim=1,
        )


        generation_capture[
            "layer_output"
        ] = None


        with torch.inference_mode():

            output = model(
                input_ids=current_token,
                attention_mask=
                    attention_mask,
                past_key_values=past,
                use_cache=True,
            )


        past = output.past_key_values


        layer_state = (
            generation_capture[
                "layer_output"
            ]
        )


        if layer_state is None:

            raise RuntimeError(
                "Layer state capture failed "
                "during baseline."
            )


        current_logits = (
            output.logits[
                0,
                -1,
                :
            ]
            .detach()
            .float()
            .cpu()
        )


        baseline_logits.append(
            current_logits
        )


        baseline_layer_states.append(
            layer_state[
                0
            ].clone()
        )


        next_token = int(
            torch.argmax(
                current_logits
            ).item()
        )


        generated_ids.append(
            next_token
        )


        if (
            tokenizer.eos_token_id
            is not None
            and next_token
            == tokenizer.eos_token_id
        ):

            break


        current_token = torch.tensor(
            [
                [
                    next_token
                ]
            ],
            dtype=torch.long,
            device=DEVICE,
        )


    return (
        generated_ids,
        baseline_logits,
        baseline_layer_states,
    )


(
    baseline_ids,
    baseline_logits,
    baseline_layer_states,
) = build_baseline()


baseline_text = tokenizer.decode(
    baseline_ids,
    skip_special_tokens=True,
)


print()
print("=" * 100)
print("BASELINE GREEDY CONTINUATION")
print("=" * 100)

print(
    f"Generated tokens: "
    f"{len(baseline_ids)}"
)

print()
print(baseline_text)


with (
    OUTPUT_DIR
    / "baseline_continuation.txt"
).open(
    "w",
    encoding="utf-8",
) as f:

    f.write(
        baseline_text
    )


# ============================================================
# DISTRIBUTION METRICS
# ============================================================

def distribution_metrics(
    base_logits,
    steered_logits,
):

    base_logits = (
        base_logits
        .double()
    )


    steered_logits = (
        steered_logits
        .double()
    )


    base_logp = F.log_softmax(
        base_logits,
        dim=-1,
    )


    steer_logp = F.log_softmax(
        steered_logits,
        dim=-1,
    )


    base_p = torch.exp(
        base_logp
    )


    steer_p = torch.exp(
        steer_logp
    )


    # KL(P_alpha || P_0)
    kl_steer_to_base = float(
        torch.sum(
            steer_p
            * (
                steer_logp
                - base_logp
            )
        )
    )


    # KL(P_0 || P_alpha)
    kl_base_to_steer = float(
        torch.sum(
            base_p
            * (
                base_logp
                - steer_logp
            )
        )
    )


    mixture = (
        0.5
        * (
            base_p
            + steer_p
        )
    )


    log_mixture = torch.log(
        mixture.clamp_min(
            1e-300
        )
    )


    js_divergence = float(
        0.5
        * torch.sum(
            base_p
            * (
                base_logp
                - log_mixture
            )
        )
        +
        0.5
        * torch.sum(
            steer_p
            * (
                steer_logp
                - log_mixture
            )
        )
    )


    total_variation = float(
        0.5
        * torch.sum(
            torch.abs(
                base_p
                - steer_p
            )
        )
    )


    logit_difference = (
        steered_logits
        - base_logits
    )


    logit_l2 = float(
        torch.linalg.vector_norm(
            logit_difference
        )
    )


    logit_mean_abs = float(
        torch.mean(
            torch.abs(
                logit_difference
            )
        )
    )


    logit_max_abs = float(
        torch.max(
            torch.abs(
                logit_difference
            )
        )
    )


    base_argmax = int(
        torch.argmax(
            base_logits
        )
    )


    steer_argmax = int(
        torch.argmax(
            steered_logits
        )
    )


    base_top_prob = float(
        base_p[
            base_argmax
        ]
    )


    steer_prob_of_base_top = float(
        steer_p[
            base_argmax
        ]
    )


    steer_top_prob = float(
        steer_p[
            steer_argmax
        ]
    )


    base_top2 = torch.topk(
        base_logits,
        k=2,
    ).values


    steer_top2 = torch.topk(
        steered_logits,
        k=2,
    ).values


    base_logit_margin = float(
        base_top2[0]
        - base_top2[1]
    )


    steer_logit_margin = float(
        steer_top2[0]
        - steer_top2[1]
    )


    baseline_token_logit_shift = float(
        logit_difference[
            base_argmax
        ]
    )


    steered_rank_of_base_token = int(
        1
        + torch.sum(
            steered_logits
            >
            steered_logits[
                base_argmax
            ]
        ).item()
    )


    return {

        "kl_steer_to_base":
            kl_steer_to_base,

        "kl_base_to_steer":
            kl_base_to_steer,

        "js_divergence":
            js_divergence,

        "total_variation":
            total_variation,

        "logit_l2":
            logit_l2,

        "logit_mean_abs":
            logit_mean_abs,

        "logit_max_abs":
            logit_max_abs,

        "base_argmax_id":
            base_argmax,

        "steer_argmax_id":
            steer_argmax,

        "top_token_changed":
            int(
                base_argmax
                != steer_argmax
            ),

        "base_top_probability":
            base_top_prob,

        "steered_probability_of_base_top":
            steer_prob_of_base_top,

        "steered_top_probability":
            steer_top_prob,

        "base_logit_margin":
            base_logit_margin,

        "steered_logit_margin":
            steer_logit_margin,

        "baseline_top_token_logit_shift":
            baseline_token_logit_shift,

        "steered_rank_of_base_token":
            steered_rank_of_base_token,
    }


# ============================================================
# FIXED-TRAJECTORY STEERED REPLAY
# ============================================================

def replay_alpha(
    alpha,
):

    delta = (
        float(alpha)
        * RAW_DIRECTION
    )


    steering_state[
        "delta"
    ] = delta


    # --------------------------------------------------------
    # Prompt forward remains completely unsteered.
    # --------------------------------------------------------

    steering_state[
        "enabled"
    ] = False


    generation_capture[
        "layer_output"
    ] = None


    with torch.inference_mode():

        output = model(
            input_ids=prompt_ids,
            attention_mask=
                prompt_attention_mask,
            use_cache=True,
        )


    past = output.past_key_values


    attention_mask = (
        prompt_attention_mask.clone()
    )


    rows = []


    # First token was generated from unsteered prompt.
    # Analysis therefore begins with generated token index 1.
    #
    # At step j:
    #   input  = baseline token j-1
    #   logits = distribution predicting baseline token j

    for j in range(
        1,
        len(baseline_ids),
    ):

        previous_baseline_token = int(
            baseline_ids[
                j - 1
            ]
        )


        target_baseline_token = int(
            baseline_ids[
                j
            ]
        )


        current_token = torch.tensor(
            [
                [
                    previous_baseline_token
                ]
            ],
            dtype=torch.long,
            device=DEVICE,
        )


        extra_mask = torch.ones(
            (
                attention_mask.shape[0],
                1,
            ),
            dtype=attention_mask.dtype,
            device=DEVICE,
        )


        attention_mask = torch.cat(
            [
                attention_mask,
                extra_mask,
            ],
            dim=1,
        )


        generation_capture[
            "layer_output"
        ] = None


        steering_state[
            "enabled"
        ] = (
            abs(
                float(alpha)
            )
            > 0
        )


        with torch.inference_mode():

            output = model(
                input_ids=current_token,
                attention_mask=
                    attention_mask,
                past_key_values=past,
                use_cache=True,
            )


        past = output.past_key_values


        steering_state[
            "enabled"
        ] = False


        steered_logits = (
            output.logits[
                0,
                -1,
                :
            ]
            .detach()
            .float()
            .cpu()
        )


        steered_layer_state = (
            generation_capture[
                "layer_output"
            ]
        )


        if steered_layer_state is None:

            raise RuntimeError(
                "Layer capture failed."
            )


        steered_layer_state = (
            steered_layer_state[
                0
            ]
        )


        base_layer_state = (
            baseline_layer_states[
                j
            ]
        )


        if base_layer_state is None:

            raise RuntimeError(
                f"Missing baseline state "
                f"at step {j}."
            )


        actual_layer_delta = (
            steered_layer_state
            - base_layer_state
        )


        actual_layer_delta_norm = float(
            torch.linalg.vector_norm(
                actual_layer_delta
            )
        )


        actual_projection_joy_axis = float(
            torch.dot(
                actual_layer_delta.double(),
                torch.from_numpy(
                    joy_sad_hidden_unit
                ).double(),
            )
        )


        expected_residual_delta_norm = float(
            abs(
                delta
            )
            * neuron_write_norm
        )


        expected_joy_axis_projection = float(
            delta
            * write_projection_on_joy_axis
        )


        metrics = distribution_metrics(
            baseline_logits[
                j
            ],
            steered_logits,
        )


        base_argmax_id = metrics[
            "base_argmax_id"
        ]


        steer_argmax_id = metrics[
            "steer_argmax_id"
        ]


        base_argmax_piece = tokenizer.decode(
            [
                base_argmax_id
            ],
            clean_up_tokenization_spaces=False,
        )


        steer_argmax_piece = tokenizer.decode(
            [
                steer_argmax_id
            ],
            clean_up_tokenization_spaces=False,
        )


        target_piece = tokenizer.decode(
            [
                target_baseline_token
            ],
            clean_up_tokenization_spaces=False,
        )


        row = {

            "alpha":
                float(alpha),

            "activation_delta":
                float(delta),

            "step":
                int(j),

            "previous_baseline_token_id":
                previous_baseline_token,

            "target_baseline_token_id":
                target_baseline_token,

            "target_baseline_token_piece":
                target_piece,

            "base_argmax_token_piece":
                base_argmax_piece,

            "steered_argmax_token_piece":
                steer_argmax_piece,

            "expected_residual_delta_norm":
                expected_residual_delta_norm,

            "actual_layer20_delta_norm":
                actual_layer_delta_norm,

            "expected_joy_axis_projection":
                expected_joy_axis_projection,

            "actual_joy_axis_projection":
                actual_projection_joy_axis,

            **metrics,
        }


        rows.append(
            row
        )


    return rows


# ============================================================
# RUN ALL ALPHAS
# ============================================================

all_rows = []


try:

    for alpha in args.alphas:

        print()
        print(
            f"Teacher-forced logit analysis "
            f"alpha={alpha:+.2f}"
        )


        rows = replay_alpha(
            alpha
        )


        all_rows.extend(
            rows
        )

finally:

    steering_handle.remove()
    generation_layer_handle.remove()


token_df = pd.DataFrame(
    all_rows
)


token_file = (
    OUTPUT_DIR
    / "token_level_logit_analysis.csv"
)


token_df.to_csv(
    token_file,
    index=False,
)


# ============================================================
# ALPHA SUMMARY
# ============================================================

summary_rows = []


for alpha in args.alphas:

    subset = token_df.loc[
        token_df[
            "alpha"
        ]
        == float(alpha)
    ]


    if subset.empty:

        continue


    summary_rows.append(
        {

            "alpha":
                float(alpha),

            "activation_delta":
                float(
                    alpha
                    * RAW_DIRECTION
                ),

            "num_steps":
                int(
                    len(subset)
                ),

            "mean_kl_steer_to_base":
                float(
                    subset[
                        "kl_steer_to_base"
                    ].mean()
                ),

            "max_kl_steer_to_base":
                float(
                    subset[
                        "kl_steer_to_base"
                    ].max()
                ),

            "mean_kl_base_to_steer":
                float(
                    subset[
                        "kl_base_to_steer"
                    ].mean()
                ),

            "mean_js_divergence":
                float(
                    subset[
                        "js_divergence"
                    ].mean()
                ),

            "mean_total_variation":
                float(
                    subset[
                        "total_variation"
                    ].mean()
                ),

            "max_total_variation":
                float(
                    subset[
                        "total_variation"
                    ].max()
                ),

            "mean_logit_l2":
                float(
                    subset[
                        "logit_l2"
                    ].mean()
                ),

            "mean_logit_abs_shift":
                float(
                    subset[
                        "logit_mean_abs"
                    ].mean()
                ),

            "max_logit_abs_shift":
                float(
                    subset[
                        "logit_max_abs"
                    ].max()
                ),

            "top_token_change_rate":
                float(
                    subset[
                        "top_token_changed"
                    ].mean()
                ),

            "num_top_token_changes":
                int(
                    subset[
                        "top_token_changed"
                    ].sum()
                ),

            "mean_base_top_probability":
                float(
                    subset[
                        "base_top_probability"
                    ].mean()
                ),

            "mean_steered_probability_of_base_top":
                float(
                    subset[
                        "steered_probability_of_base_top"
                    ].mean()
                ),

            "mean_base_top_probability_change":
                float(
                    (
                        subset[
                            "steered_probability_of_base_top"
                        ]
                        - subset[
                            "base_top_probability"
                        ]
                    ).mean()
                ),

            "mean_actual_layer20_delta_norm":
                float(
                    subset[
                        "actual_layer20_delta_norm"
                    ].mean()
                ),

            "mean_expected_layer20_delta_norm":
                float(
                    subset[
                        "expected_residual_delta_norm"
                    ].mean()
                ),

            "mean_actual_joy_axis_projection":
                float(
                    subset[
                        "actual_joy_axis_projection"
                    ].mean()
                ),

            "mean_expected_joy_axis_projection":
                float(
                    subset[
                        "expected_joy_axis_projection"
                    ].mean()
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_file = (
    OUTPUT_DIR
    / "alpha_logit_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False,
)


# ============================================================
# FULL JSON
# ============================================================

full_result = {

    "experiment":
        (
            "Single-neuron residual-direction alignment "
            "and fixed-trajectory token-logit analysis"
        ),

    "prompt":
        args.prompt,

    "selected_neuron":
        {

            "layer":
                LAYER_NUMBER,

            "layer_index":
                LAYER_IDX,

            "neuron_index":
                NEURON_IDX,

            "mean_activation_joy":
                MU_JOY,

            "mean_activation_sadness":
                MU_SAD,

            "raw_activation_difference":
                RAW_DIRECTION,

            "pooled_sd":
                POOLED_SD,

            "standardized_effect":
                STANDARDIZED_EFFECT,
        },

    "steering_rule":
        (
            "m' = m + alpha * "
            "(mean_activation_joy "
            "- mean_activation_sadness)"
        ),

    "direction_alignment":
        direction_alignment,

    "baseline_continuation":
        baseline_text,

    "baseline_generated_token_count":
        len(
            baseline_ids
        ),

    "analysis_design":
        {

            "prompt_pass_steered":
                False,

            "first_generated_token_steered":
                False,

            "fixed_baseline_token_trajectory":
                True,

            "purpose":
                (
                    "Isolate the effect of the "
                    "activation intervention on hidden "
                    "states and next-token logits without "
                    "lexical trajectory confounding."
                ),
        },

    "alpha_summary":
        summary_rows,
}


full_file = (
    OUTPUT_DIR
    / "full_analysis.json"
)


with full_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        full_result,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 100)
print("FINAL DIRECTION RESULT")
print("=" * 100)

print(
    f"Cosine(W_down neuron, Joy-Sad residual) : "
    f"{cosine_alignment:+.8f}"
)

print(
    f"Angle                                  : "
    f"{angle_degrees:.3f} degrees"
)

print(
    f"W_down projection on Joy-Sad axis      : "
    f"{write_projection_on_joy_axis:+.8f}"
)

print(
    f"Expected Joy-Sad shift at alpha=1      : "
    f"{alpha1_joy_axis_shift:+.8f}"
)


print()
print("=" * 100)
print("TOKEN-LEVEL LOGIT SUMMARY")
print("=" * 100)


columns = [

    "alpha",

    "activation_delta",

    "mean_kl_steer_to_base",

    "max_kl_steer_to_base",

    "mean_total_variation",

    "mean_logit_l2",

    "top_token_change_rate",

    "num_top_token_changes",

    "mean_base_top_probability_change",

    "mean_actual_layer20_delta_norm",

    "mean_actual_joy_axis_projection",
]


print(
    summary_df[
        columns
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_DIR
    / "direction_alignment.json"
)

print(
    OUTPUT_DIR
    / "joy_minus_sad_hidden_direction.npy"
)

print(token_file)
print(summary_file)
print(full_file)

print()
print("Done.")
