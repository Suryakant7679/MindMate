#!/usr/bin/env python3

"""
19_test_top_joy_actuators.py

Test the top Joy actuator neurons discovered by:

    18_rank_all_top1pct_joy_actuators.py

This script performs TWO complementary experiments for each selected neuron
and each steering alpha.

======================================================================
A. FIXED-TRAJECTORY CAUSAL ANALYSIS
======================================================================

1. Generate ONE unsteered greedy baseline continuation.

2. Replay exactly the SAME generated tokens.

3. During replay, intervene only on the selected MLP neuron:

       m'_u = m_u + alpha * Delta_mu_u

   where

       Delta_mu_u
           = mean_activation_joy
           - mean_activation_sadness

4. Prompt prefill is NOT steered.

5. Measure:

       ||Delta h||
       angle(h_base, h_steered)
       cos(Delta h, Joy-Sadness)
       Joy-axis projection

       KL divergence
       total variation
       logit L2 change
       top-token change rate
       target-token log-probability change

Because the lexical token trajectory is fixed, the hidden-state comparison
is much easier to interpret causally.

======================================================================
B. FREE GREEDY GENERATION
======================================================================

Run autoregressive greedy generation under the same neuron intervention.

This tells us whether the internal actuator geometry actually changes
the generated response.

======================================================================
C. PNG VISUALIZATION
======================================================================

For every non-zero alpha, produce a clean three-panel PNG:

    A. baseline vs steered residual direction
    B. actual local residual endpoint displacement
    C. Delta-h vs empirical Joy-Sadness direction

All angles/cosines are calculated in the original hidden dimension.
The 2-D plots are constructed only for visualization.

======================================================================
INPUTS
======================================================================

Ranking CSV:

results/sarvam1_steering/18_all_top1pct_joy_actuators/
    all_top1pct_joy_actuator_ranking.csv

Layer-specific Joy-Sadness directions:

results/sarvam1_steering/18_all_top1pct_joy_actuators/
    layer_directions/
        layer_XX_joy_minus_sad.npy

======================================================================
DEFAULT CANDIDATES
======================================================================

Actuator ranks:

    1
    2
    5

You can change them using:

    --ranks 1 2 3 4 5

======================================================================
DEFAULT ALPHAS
======================================================================

A conservative grid is used by default:

    -1 -0.5 -0.25 0 0.25 0.5 1

This is intentional because the best rank-1 neuron has a very large
predicted alpha=1 residual perturbation.

A guard also skips conditions whose predicted linear residual perturbation
exceeds --max-expected-residual-norm.

Use:

    --max-expected-residual-norm 0

to disable this guard.

======================================================================
OUTPUTS
======================================================================

results/sarvam1_steering/19_top_joy_actuator_tests/

    baseline.json

    actuator_test_summary.csv
    actuator_test_summary.json

    candidate_rank_001_L28_N2768/
        candidate_metadata.json
        fixed_trajectory_token_metrics.csv
        alpha_summary.csv
        free_generation.csv
        png/
            alpha_m1.png
            alpha_m0p5.png
            ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

from transformers import AutoModelForCausalLM, AutoTokenizer


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
    "--ranks",
    type=int,
    nargs="+",
    default=[
        1,
        2,
        5,
    ],
    help=(
        "Global actuator ranks from script 18 ranking CSV."
    ),
)


parser.add_argument(
    "--alphas",
    type=float,
    nargs="+",
    default=[
        -1.0,
        -0.5,
        -0.25,
        0.0,
        0.25,
        0.5,
        1.0,
    ],
)


parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=30,
)


parser.add_argument(
    "--max-expected-residual-norm",
    type=float,
    default=20.0,
    help=(
        "Skip alpha values whose predicted linear "
        "|alpha| * ||Delta-h(alpha=1)|| exceeds this value. "
        "Set <=0 to disable."
    ),
)


parser.add_argument(
    "--local-files-only",
    action="store_true",
)


parser.add_argument(
    "--skip-png",
    action="store_true",
)


args = parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

EPS = 1e-12


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


RANKING_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "18_all_top1pct_joy_actuators"
)


RANKING_FILE = (
    RANKING_DIR
    / "all_top1pct_joy_actuator_ranking.csv"
)


LAYER_DIRECTION_DIR = (
    RANKING_DIR
    / "layer_directions"
)


OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "19_top_joy_actuator_tests"
)


OUTPUT_DIR.mkdir(
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


print("=" * 110)
print("TOP JOY ACTUATOR STEERING TEST")
print("=" * 110)

print(
    f"Device                 : {DEVICE}"
)

print(
    f"Dtype                  : {DTYPE}"
)

print(
    f"Requested ranks        : {args.ranks}"
)

print(
    f"Requested alphas       : {args.alphas}"
)

print(
    f"Max new tokens         : {args.max_new_tokens}"
)

print(
    f"Residual-norm guard    : "
    f"{args.max_expected_residual_norm}"
)


# ============================================================
# INPUT CHECK
# ============================================================

if not RANKING_FILE.is_file():

    raise FileNotFoundError(
        "\nRanking file not found:\n"
        f"{RANKING_FILE}\n\n"
        "Run 18_rank_all_top1pct_joy_actuators.py first."
    )


# ============================================================
# LOAD RANKING
# ============================================================

ranking_df = pd.read_csv(
    RANKING_FILE
)


if "actuator_rank" not in ranking_df.columns:

    raise RuntimeError(
        "Ranking CSV does not contain actuator_rank."
    )


ranking_df[
    "actuator_rank"
] = (
    ranking_df[
        "actuator_rank"
    ]
    .astype(int)
)


selected_rows = []


for requested_rank in args.ranks:

    match = ranking_df.loc[
        ranking_df[
            "actuator_rank"
        ]
        == requested_rank
    ]


    if len(match) != 1:

        raise RuntimeError(
            f"Expected exactly one candidate "
            f"for actuator rank {requested_rank}, "
            f"found {len(match)}."
        )


    selected_rows.append(
        match.iloc[0]
    )


print()
print("=" * 110)
print("SELECTED CANDIDATES")
print("=" * 110)


candidate_display_columns = [

    "actuator_rank",
    "layer",
    "layer_index",
    "neuron_index",
    "original_joy_rank",
    "membership_degree_top1pct",
    "raw_activation_difference",
    "standardized_joy_minus_sadness",
    "intervention_direction_cosine_joy_sad",
    "expected_residual_shift_norm_alpha1",
    "actuator_score_q",
]


print(
    pd.DataFrame(
        selected_rows
    )[
        candidate_display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 110)
print("LOADING TOKENIZER")
print("=" * 110)


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


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 110)
print("LOADING SARVAM-1")
print("=" * 110)


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
    f"Layers                 : {NUM_LAYERS}"
)

print(
    f"Hidden size            : {HIDDEN_SIZE}"
)

print(
    f"MLP width              : {MLP_WIDTH}"
)


# ============================================================
# TOKENIZE PROMPT
# ============================================================

encoded_prompt = tokenizer(
    args.prompt,
    return_tensors="pt",
)


PROMPT_IDS = (
    encoded_prompt[
        "input_ids"
    ]
    .to(
        DEVICE
    )
)


PROMPT_ATTENTION = (
    encoded_prompt[
        "attention_mask"
    ]
    .to(
        DEVICE
    )
)


# ============================================================
# VECTOR UTILITIES
# ============================================================

def vector_norm(
    vector,
):

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            vector
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

    norm_a = vector_norm(
        a
    )

    norm_b = vector_norm(
        b
    )

    if (
        norm_a < EPS
        or norm_b < EPS
    ):

        return float(
            "nan"
        )

    value = float(
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
            value,
            -1.0,
            1.0,
        )
    )


def angle_degrees(
    a,
    b,
):

    cosine = cosine_similarity(
        a,
        b,
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
# BASELINE GREEDY GENERATION
# ============================================================

def generate_unsteered_baseline():

    """
    Generate one deterministic greedy continuation.

    This token sequence will be reused for fixed-trajectory
    causal comparisons for every candidate.
    """

    with torch.inference_mode():

        output = model(
            input_ids=
                PROMPT_IDS,

            attention_mask=
                PROMPT_ATTENTION,

            use_cache=True,
        )


    past = (
        output.past_key_values
    )


    first_token = int(
        output.logits[
            0,
            -1,
            :
        ]
        .argmax()
        .item()
    )


    generated = [
        first_token
    ]


    if (
        tokenizer.eos_token_id
        is not None
        and first_token
        == tokenizer.eos_token_id
    ):

        return generated


    current_token = torch.tensor(
        [
            [
                first_token
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )


    attention_mask = (
        PROMPT_ATTENTION.clone()
    )


    for _ in range(
        args.max_new_tokens
        - 1
    ):

        attention_mask = torch.cat(
            [
                attention_mask,

                torch.ones(
                    (
                        attention_mask.shape[0],
                        1,
                    ),
                    dtype=
                        attention_mask.dtype,
                    device=
                        DEVICE,
                ),
            ],
            dim=1,
        )


        with torch.inference_mode():

            output = model(
                input_ids=
                    current_token,

                attention_mask=
                    attention_mask,

                past_key_values=
                    past,

                use_cache=True,
            )


        past = (
            output.past_key_values
        )


        next_token = int(
            output.logits[
                0,
                -1,
                :
            ]
            .argmax()
            .item()
        )


        generated.append(
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


    return generated


print()
print("=" * 110)
print("GENERATING SHARED BASELINE CONTINUATION")
print("=" * 110)


BASELINE_IDS = (
    generate_unsteered_baseline()
)


BASELINE_TEXT = tokenizer.decode(
    BASELINE_IDS,
    skip_special_tokens=True,
)


print()
print("PROMPT:")
print(args.prompt)


print()
print("BASELINE RESPONSE:")
print(BASELINE_TEXT)


print()
print(
    f"Generated tokens        : "
    f"{len(BASELINE_IDS)}"
)


baseline_json_file = (
    OUTPUT_DIR
    / "baseline.json"
)


with baseline_json_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {

            "model":
                MODEL_ID,

            "prompt":
                args.prompt,

            "baseline_generated_token_ids":
                BASELINE_IDS,

            "baseline_text":
                BASELINE_TEXT,

            "max_new_tokens":
                args.max_new_tokens,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# LOGIT METRICS
# ============================================================

def compare_logits(
    baseline_logits,
    steered_logits,
    target_token_id,
):

    """
    Compare two vocabulary distributions.

    KL convention:

        KL(p_steered || p_baseline)
    """

    base = (
        baseline_logits
        .float()
        .cpu()
    )


    steer = (
        steered_logits
        .float()
        .cpu()
    )


    base_logp = torch.log_softmax(
        base,
        dim=-1,
    )


    steer_logp = torch.log_softmax(
        steer,
        dim=-1,
    )


    base_p = torch.exp(
        base_logp
    )


    steer_p = torch.exp(
        steer_logp
    )


    kl = float(
        torch.sum(
            steer_p
            * (
                steer_logp
                - base_logp
            )
        ).item()
    )


    tv = float(
        0.5
        * torch.sum(
            torch.abs(
                steer_p
                - base_p
            )
        ).item()
    )


    logit_l2 = float(
        torch.linalg.vector_norm(
            steer
            - base
        ).item()
    )


    base_top_id = int(
        torch.argmax(
            base
        ).item()
    )


    steer_top_id = int(
        torch.argmax(
            steer
        ).item()
    )


    top_changed = int(
        base_top_id
        != steer_top_id
    )


    base_top_probability = float(
        base_p[
            base_top_id
        ].item()
    )


    steer_probability_of_base_top = float(
        steer_p[
            base_top_id
        ].item()
    )


    base_top_probability_change = (
        steer_probability_of_base_top
        - base_top_probability
    )


    target_token_id = int(
        target_token_id
    )


    target_logprob_change = float(
        (
            steer_logp[
                target_token_id
            ]
            - base_logp[
                target_token_id
            ]
        ).item()
    )


    return {

        "kl_steer_to_base":
            kl,

        "total_variation":
            tv,

        "logit_l2":
            logit_l2,

        "baseline_top_token_id":
            base_top_id,

        "steered_top_token_id":
            steer_top_id,

        "top_token_changed":
            top_changed,

        "base_top_probability":
            base_top_probability,

        "steer_probability_of_base_top":
            steer_probability_of_base_top,

        "base_top_probability_change":
            base_top_probability_change,

        "target_token_logprob_change":
            target_logprob_change,
    }


# ============================================================
# PLOT HELPERS
# ============================================================

def draw_arrow(
    ax,
    start,
    end,
    linewidth=3.0,
    mutation_scale=18,
    zorder=5,
):

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=
            mutation_scale,
        linewidth=
            linewidth,
        shrinkA=0,
        shrinkB=0,
        zorder=
            zorder,
    )


    ax.add_patch(
        patch
    )


    return patch


def build_base_delta_plane(
    h_base,
    delta_h,
    joy_direction,
):

    base_norm = vector_norm(
        h_base
    )


    if base_norm < EPS:

        raise RuntimeError(
            "Baseline state has near-zero norm."
        )


    e1 = (
        h_base
        / base_norm
    )


    delta_orthogonal = (
        delta_h
        - np.dot(
            delta_h,
            e1,
        )
        * e1
    )


    orthogonal_norm = vector_norm(
        delta_orthogonal
    )


    if orthogonal_norm > 1e-10:

        e2 = (
            delta_orthogonal
            / orthogonal_norm
        )

        return e1, e2


    joy_orthogonal = (
        joy_direction
        - np.dot(
            joy_direction,
            e1,
        )
        * e1
    )


    joy_norm = vector_norm(
        joy_orthogonal
    )


    if joy_norm > 1e-10:

        e2 = (
            joy_orthogonal
            / joy_norm
        )

        return e1, e2


    candidate = np.zeros_like(
        e1
    )


    candidate[
        np.argmin(
            np.abs(
                e1
            )
        )
    ] = 1.0


    candidate = (
        candidate
        - np.dot(
            candidate,
            e1,
        )
        * e1
    )


    e2 = (
        candidate
        /
        (
            vector_norm(
                candidate
            )
            + EPS
        )
    )


    return e1, e2


def project_to_plane(
    vector,
    e1,
    e2,
):

    return np.array(
        [
            np.dot(
                vector,
                e1,
            ),

            np.dot(
                vector,
                e2,
            ),
        ],
        dtype=np.float64,
    )


def build_delta_joy_plane(
    delta_h,
    joy_direction,
):

    delta_norm = vector_norm(
        delta_h
    )


    joy_norm = vector_norm(
        joy_direction
    )


    if (
        delta_norm < EPS
        or joy_norm < EPS
    ):

        raise RuntimeError(
            "Cannot construct Delta-h/Joy plane "
            "from zero-length vectors."
        )


    e1 = (
        delta_h
        / delta_norm
    )


    joy_unit = (
        joy_direction
        / joy_norm
    )


    joy_orthogonal = (
        joy_unit
        - np.dot(
            joy_unit,
            e1,
        )
        * e1
    )


    joy_orthogonal_norm = vector_norm(
        joy_orthogonal
    )


    if joy_orthogonal_norm > EPS:

        e2 = (
            joy_orthogonal
            / joy_orthogonal_norm
        )

    else:

        candidate = np.zeros_like(
            e1
        )


        candidate[
            np.argmin(
                np.abs(
                    e1
                )
            )
        ] = 1.0


        candidate = (
            candidate
            - np.dot(
                candidate,
                e1,
            )
            * e1
        )


        e2 = (
            candidate
            /
            (
                vector_norm(
                    candidate
                )
                + EPS
            )
        )


    delta_plot = np.array(
        [
            1.0,
            0.0,
        ]
    )


    joy_plot = np.array(
        [
            np.dot(
                joy_unit,
                e1,
            ),

            np.dot(
                joy_unit,
                e2,
            ),
        ]
    )


    return (
        e1,
        e2,
        delta_plot,
        joy_plot,
    )


def save_geometry_png(
    output_file,
    candidate,
    alpha,
    h_base,
    h_steered,
    joy_direction,
    activation_delta,
):

    h_base = np.asarray(
        h_base,
        dtype=np.float64,
    )


    h_steered = np.asarray(
        h_steered,
        dtype=np.float64,
    )


    joy_direction = np.asarray(
        joy_direction,
        dtype=np.float64,
    )


    delta_h = (
        h_steered
        - h_base
    )


    delta_norm = vector_norm(
        delta_h
    )


    if delta_norm < EPS:

        return


    baseline_norm = vector_norm(
        h_base
    )


    steered_norm = vector_norm(
        h_steered
    )


    base_steer_angle = angle_degrees(
        h_base,
        h_steered,
    )


    base_steer_cosine = cosine_similarity(
        h_base,
        h_steered,
    )


    delta_joy_cosine = cosine_similarity(
        delta_h,
        joy_direction,
    )


    delta_joy_angle = angle_degrees(
        delta_h,
        joy_direction,
    )


    joy_unit = (
        joy_direction
        /
        (
            vector_norm(
                joy_direction
            )
            + EPS
        )
    )


    joy_projection = float(
        np.dot(
            delta_h,
            joy_unit,
        )
    )


    # ========================================================
    # PANELS A/B BASIS
    # ========================================================

    e1, e2 = build_base_delta_plane(
        h_base,
        delta_h,
        joy_direction,
    )


    base_2d = project_to_plane(
        h_base,
        e1,
        e2,
    )


    steer_2d = project_to_plane(
        h_steered,
        e1,
        e2,
    )


    joy_2d = project_to_plane(
        joy_direction,
        e1,
        e2,
    )


    base_unit_2d = (
        base_2d
        /
        (
            vector_norm(
                base_2d
            )
            + EPS
        )
    )


    steer_unit_2d = (
        steer_2d
        /
        (
            vector_norm(
                steer_2d
            )
            + EPS
        )
    )


    joy_unit_2d = (
        joy_2d
        /
        (
            vector_norm(
                joy_2d
            )
            + EPS
        )
    )


    # ========================================================
    # PANEL C BASIS
    # ========================================================

    (
        _,
        _,
        delta_plot,
        joy_plot,
    ) = build_delta_joy_plane(
        delta_h,
        joy_direction,
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            19,
            7,
        ),
    )


    ax1, ax2, ax3 = axes


    fig.suptitle(
        (
            "Single-Neuron Joy Actuator Test\n"
            f"Actuator rank {int(candidate['actuator_rank'])}, "
            f"Layer {int(candidate['layer'])}, "
            f"Neuron {int(candidate['neuron_index'])}, "
            f"alpha={alpha:+g}"
        ),
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )


    # ========================================================
    # PANEL A
    # ========================================================

    ax1.set_title(
        "A. Baseline vs steered residual direction",
        fontsize=12,
        fontweight="bold",
    )


    circle = plt.Circle(
        (
            0,
            0,
        ),
        1.0,
        fill=False,
        linewidth=1.0,
        alpha=0.35,
    )


    ax1.add_patch(
        circle
    )


    draw_arrow(
        ax1,
        (
            0,
            0,
        ),
        (
            base_unit_2d[0],
            base_unit_2d[1],
        ),
        linewidth=4.0,
    )


    draw_arrow(
        ax1,
        (
            0,
            0,
        ),
        (
            steer_unit_2d[0],
            steer_unit_2d[1],
        ),
        linewidth=2.5,
        zorder=6,
    )


    draw_arrow(
        ax1,
        (
            0,
            0,
        ),
        (
            joy_unit_2d[0],
            joy_unit_2d[1],
        ),
        linewidth=2.5,
    )


    ax1.annotate(
        "Baseline",
        xy=(
            base_unit_2d[0],
            base_unit_2d[1],
        ),
        xytext=(
            -15,
            20,
        ),
        textcoords=
            "offset points",
        fontsize=10,
        fontweight="bold",
    )


    ax1.annotate(
        "Steered",
        xy=(
            steer_unit_2d[0],
            steer_unit_2d[1],
        ),
        xytext=(
            -15,
            -28,
        ),
        textcoords=
            "offset points",
        fontsize=10,
        fontweight="bold",
    )


    ax1.annotate(
        "Joy - Sadness",
        xy=(
            joy_unit_2d[0],
            joy_unit_2d[1],
        ),
        xytext=(
            8,
            6,
        ),
        textcoords=
            "offset points",
        fontsize=10,
        fontweight="bold",
    )


    ax1.text(
        0.03,
        0.04,
        (
            f"True hidden-space angle\n"
            f"base -> steered = "
            f"{base_steer_angle:.5f} deg\n\n"

            f"cos(base, steered)\n"
            f"= {base_steer_cosine:.8f}"
        ),
        transform=
            ax1.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.90,
        ),
    )


    ax1.axhline(
        0,
        linewidth=0.7,
        alpha=0.3,
    )


    ax1.axvline(
        0,
        linewidth=0.7,
        alpha=0.3,
    )


    ax1.set_xlim(
        -1.3,
        1.3,
    )


    ax1.set_ylim(
        -1.3,
        1.3,
    )


    ax1.set_aspect(
        "equal"
    )


    ax1.grid(
        True,
        alpha=0.12,
    )


    ax1.set_xlabel(
        "Residual-plane axis 1"
    )


    ax1.set_ylabel(
        "Residual-plane axis 2"
    )


    # ========================================================
    # PANEL B — ACTUAL ENDPOINT CHANGE
    # ========================================================

    ax2.set_title(
        "B. Local zoom of actual residual change",
        fontsize=12,
        fontweight="bold",
    )


    ax2.scatter(
        [
            base_2d[0]
        ],
        [
            base_2d[1]
        ],
        s=100,
        zorder=8,
    )


    ax2.scatter(
        [
            steer_2d[0]
        ],
        [
            steer_2d[1]
        ],
        s=100,
        zorder=8,
    )


    draw_arrow(
        ax2,
        (
            base_2d[0],
            base_2d[1],
        ),
        (
            steer_2d[0],
            steer_2d[1],
        ),
        linewidth=3.5,
        mutation_scale=20,
        zorder=7,
    )


    ax2.annotate(
        "Baseline endpoint",
        xy=(
            base_2d[0],
            base_2d[1],
        ),
        xytext=(
            -10,
            -25,
        ),
        textcoords=
            "offset points",
        fontsize=9,
        fontweight="bold",
        ha="right",
    )


    ax2.annotate(
        "Steered endpoint",
        xy=(
            steer_2d[0],
            steer_2d[1],
        ),
        xytext=(
            10,
            15,
        ),
        textcoords=
            "offset points",
        fontsize=9,
        fontweight="bold",
    )


    midpoint = (
        base_2d
        + steer_2d
    ) / 2.0


    ax2.annotate(
        (
            "Delta-h\n"
            f"||Delta-h||={delta_norm:.5f}"
        ),
        xy=(
            midpoint[0],
            midpoint[1],
        ),
        xytext=(
            15,
            0,
        ),
        textcoords=
            "offset points",
        fontsize=9,
        fontweight="bold",
    )


    center = (
        base_2d
        + steer_2d
    ) / 2.0


    endpoint_distance = vector_norm(
        steer_2d
        - base_2d
    )


    radius = max(
        endpoint_distance
        * 1.75,
        0.10,
    )


    ax2.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )


    ax2.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )


    ax2.set_aspect(
        "equal"
    )


    ax2.grid(
        True,
        alpha=0.15,
    )


    ax2.set_xlabel(
        "Raw residual-plane coordinate 1"
    )


    ax2.set_ylabel(
        "Raw residual-plane coordinate 2"
    )


    ax2.text(
        0.03,
        0.97,
        (
            "True local zoom\n"
            "No artificial angular scaling\n\n"
            f"||h_base||={baseline_norm:.3f}\n"
            f"||h_steered||={steered_norm:.3f}"
        ),
        transform=
            ax2.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.90,
        ),
    )


    # ========================================================
    # PANEL C
    # ========================================================

    ax3.set_title(
        "C. Steering displacement vs Joy-Sadness",
        fontsize=12,
        fontweight="bold",
    )


    circle = plt.Circle(
        (
            0,
            0,
        ),
        1.0,
        fill=False,
        linewidth=1.0,
        alpha=0.35,
    )


    ax3.add_patch(
        circle
    )


    draw_arrow(
        ax3,
        (
            0,
            0,
        ),
        (
            delta_plot[0],
            delta_plot[1],
        ),
        linewidth=3.5,
    )


    draw_arrow(
        ax3,
        (
            0,
            0,
        ),
        (
            joy_plot[0],
            joy_plot[1],
        ),
        linewidth=3.5,
    )


    ax3.annotate(
        "Delta-h",
        xy=(
            delta_plot[0],
            delta_plot[1],
        ),
        xytext=(
            -10,
            18,
        ),
        textcoords=
            "offset points",
        fontsize=10,
        fontweight="bold",
        ha="right",
    )


    ax3.annotate(
        "Joy - Sadness",
        xy=(
            joy_plot[0],
            joy_plot[1],
        ),
        xytext=(
            8,
            5,
        ),
        textcoords=
            "offset points",
        fontsize=10,
        fontweight="bold",
    )


    signed_angle = float(
        np.degrees(
            np.arctan2(
                joy_plot[1],
                joy_plot[0],
            )
        )
    )


    arc_radius = 0.42


    if signed_angle >= 0:

        theta1 = 0
        theta2 = signed_angle

    else:

        theta1 = signed_angle
        theta2 = 0


    arc = Arc(
        (
            0,
            0,
        ),
        width=
            2
            * arc_radius,
        height=
            2
            * arc_radius,
        angle=0,
        theta1=
            theta1,
        theta2=
            theta2,
        linewidth=2.0,
    )


    ax3.add_patch(
        arc
    )


    mid_signed = (
        signed_angle
        / 2.0
    )


    mid_rad = np.deg2rad(
        mid_signed
    )


    ax3.text(
        0.57
        * np.cos(
            mid_rad
        ),
        0.57
        * np.sin(
            mid_rad
        ),
        f"{delta_joy_angle:.2f} deg",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
    )


    ax3.text(
        0.03,
        0.04,
        (
            f"cos(Delta-h, Joy-Sad)\n"
            f"= {delta_joy_cosine:+.6f}\n\n"

            f"Joy-axis projection\n"
            f"= {joy_projection:+.6f}"
        ),
        transform=
            ax3.transAxes,
        fontsize=9,
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.90,
        ),
    )


    ax3.axhline(
        0,
        linewidth=0.7,
        alpha=0.3,
    )


    ax3.axvline(
        0,
        linewidth=0.7,
        alpha=0.3,
    )


    ax3.set_xlim(
        -1.3,
        1.3,
    )


    ax3.set_ylim(
        -1.3,
        1.3,
    )


    ax3.set_aspect(
        "equal"
    )


    ax3.grid(
        True,
        alpha=0.12,
    )


    ax3.set_xlabel(
        "Delta-h direction axis"
    )


    ax3.set_ylabel(
        "Orthogonal emotion component"
    )


    fig.text(
        0.5,
        0.025,
        (
            f"activation Delta-m={activation_delta:+.6f}"
            "     |     "
            f"||Delta-h||={delta_norm:.6f}"
            "     |     "
            f"cos(Delta-h, Joy-Sad)="
            f"{delta_joy_cosine:+.6f}"
            "     |     "
            f"Joy projection={joy_projection:+.6f}"
        ),
        ha="center",
        fontsize=9,
    )


    fig.subplots_adjust(
        left=0.05,
        right=0.985,
        bottom=0.12,
        top=0.83,
        wspace=0.28,
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


# ============================================================
# SAFE FILENAME
# ============================================================

def alpha_filename(
    alpha,
):

    text = f"{float(alpha):g}"

    text = text.replace(
        "-",
        "m",
    )

    text = text.replace(
        ".",
        "p",
    )

    return text


# ============================================================
# GLOBAL SUMMARY HOLDERS
# ============================================================

global_summary_rows = []

global_json_results = []


# ============================================================
# TEST EACH CANDIDATE
# ============================================================

for candidate_series in selected_rows:

    candidate = (
        candidate_series.to_dict()
    )


    actuator_rank = int(
        candidate[
            "actuator_rank"
        ]
    )


    layer_number = int(
        candidate[
            "layer"
        ]
    )


    layer_idx = int(
        candidate[
            "layer_index"
        ]
    )


    neuron_idx = int(
        candidate[
            "neuron_index"
        ]
    )


    neuron_number = int(
        candidate[
            "neuron_number"
        ]
    )


    delta_mu = float(
        candidate[
            "raw_activation_difference"
        ]
    )


    expected_norm_alpha1 = float(
        candidate[
            "expected_residual_shift_norm_alpha1"
        ]
    )


    static_actuator_q = float(
        candidate[
            "actuator_score_q"
        ]
    )


    static_intervention_cosine = float(
        candidate[
            "intervention_direction_cosine_joy_sad"
        ]
    )


    if (
        layer_idx < 0
        or layer_idx >= NUM_LAYERS
    ):

        raise RuntimeError(
            f"Invalid layer index "
            f"{layer_idx}"
        )


    if (
        neuron_idx < 0
        or neuron_idx >= MLP_WIDTH
    ):

        raise RuntimeError(
            f"Invalid neuron index "
            f"{neuron_idx}"
        )


    direction_file = (
        LAYER_DIRECTION_DIR
        / (
            f"layer_{layer_number:02d}_"
            f"joy_minus_sad.npy"
        )
    )


    if not direction_file.is_file():

        raise FileNotFoundError(
            f"Missing Joy-Sad direction:\n"
            f"{direction_file}"
        )


    joy_direction = np.load(
        direction_file
    ).astype(
        np.float64
    )


    if joy_direction.shape != (
        HIDDEN_SIZE,
    ):

        raise RuntimeError(
            f"Unexpected Joy direction "
            f"shape: {joy_direction.shape}"
        )


    joy_direction_norm = vector_norm(
        joy_direction
    )


    joy_unit = (
        joy_direction
        /
        (
            joy_direction_norm
            + EPS
        )
    )


    candidate_dir = (
        OUTPUT_DIR
        / (
            f"candidate_rank_{actuator_rank:03d}_"
            f"L{layer_number:02d}_"
            f"N{neuron_idx}"
        )
    )


    png_dir = (
        candidate_dir
        / "png"
    )


    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    png_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print()
    print("=" * 110)

    print(
        f"CANDIDATE RANK {actuator_rank}"
    )

    print("=" * 110)

    print(
        f"Layer                  : {layer_number}"
    )

    print(
        f"Layer index            : {layer_idx}"
    )

    print(
        f"Neuron index           : {neuron_idx}"
    )

    print(
        f"Neuron number          : {neuron_number}"
    )

    print(
        f"Delta mu               : {delta_mu:+.8f}"
    )

    print(
        f"Expected ||Dh|| a=1    : "
        f"{expected_norm_alpha1:.8f}"
    )

    print(
        f"Static actuator Q      : "
        f"{static_actuator_q:+.8f}"
    )

    print(
        f"Static cosine          : "
        f"{static_intervention_cosine:+.8f}"
    )


    # ========================================================
    # HOOK STATE
    # ========================================================

    steering_state = {

        "enabled":
            False,

        "delta":
            0.0,

        "collect_activation":
            False,

        "pre_activation":
            None,

        "post_activation":
            None,
    }


    capture_state = {

        "hidden":
            None,
    }


    # ========================================================
    # STEERING HOOK
    # ========================================================

    def steering_hook(
        module,
        inputs,
    ):

        if not inputs:

            raise RuntimeError(
                "down_proj hook received "
                "no input."
            )


        x = inputs[0]


        if steering_state[
            "collect_activation"
        ]:

            steering_state[
                "pre_activation"
            ] = float(
                x[
                    ...,
                    neuron_idx
                ]
                .detach()
                .float()
                .mean()
                .item()
            )


        if not steering_state[
            "enabled"
        ]:

            if steering_state[
                "collect_activation"
            ]:

                steering_state[
                    "post_activation"
                ] = (
                    steering_state[
                        "pre_activation"
                    ]
                )

            return inputs


        modified = (
            x.clone()
        )


        modified[
            ...,
            neuron_idx
        ] = (
            modified[
                ...,
                neuron_idx
            ]
            + float(
                steering_state[
                    "delta"
                ]
            )
        )


        if steering_state[
            "collect_activation"
        ]:

            steering_state[
                "post_activation"
            ] = float(
                modified[
                    ...,
                    neuron_idx
                ]
                .detach()
                .float()
                .mean()
                .item()
            )


        return (
            modified,
            *inputs[1:],
        )


    # ========================================================
    # LAYER OUTPUT HOOK
    # ========================================================

    def layer_output_hook(
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
                "Unsupported transformer "
                f"output type: {type(output)}"
            )


        capture_state[
            "hidden"
        ] = (
            hidden[
                :,
                -1,
                :
            ]
            .detach()
            .float()
            .cpu()
            .numpy()[0]
            .astype(
                np.float64
            )
        )


    steering_handle = (
        layers[
            layer_idx
        ]
        .mlp
        .down_proj
        .register_forward_pre_hook(
            steering_hook
        )
    )


    layer_handle = (
        layers[
            layer_idx
        ]
        .register_forward_hook(
            layer_output_hook
        )
    )


    # ========================================================
    # FIXED TRAJECTORY REPLAY
    # ========================================================

    def replay_fixed_trajectory(
        alpha,
    ):

        """
        Replay BASELINE_IDS exactly.

        The prompt pass is always unsteered.

        Steering begins only when generated tokens
        are fed through the cached decoder.
        """

        activation_delta = (
            float(alpha)
            * delta_mu
        )


        steering_state[
            "delta"
        ] = activation_delta


        steering_state[
            "enabled"
        ] = False


        steering_state[
            "collect_activation"
        ] = False


        capture_state[
            "hidden"
        ] = None


        # ----------------------------------------------------
        # Unsteered prompt prefill.
        # ----------------------------------------------------

        with torch.inference_mode():

            output = model(
                input_ids=
                    PROMPT_IDS,

                attention_mask=
                    PROMPT_ATTENTION,

                use_cache=True,
            )


        past = (
            output.past_key_values
        )


        attention_mask = (
            PROMPT_ATTENTION.clone()
        )


        states = []

        logits_list = []

        pre_activations = []

        post_activations = []


        # ----------------------------------------------------
        # BASELINE_IDS[j-1] predicts BASELINE_IDS[j].
        #
        # Thus number of causal analysis positions:
        #
        # len(BASELINE_IDS) - 1
        # ----------------------------------------------------

        for j in range(
            1,
            len(
                BASELINE_IDS
            ),
        ):

            previous_token_id = int(
                BASELINE_IDS[
                    j - 1
                ]
            )


            current_token = torch.tensor(
                [
                    [
                        previous_token_id
                    ]
                ],
                dtype=torch.long,
                device=DEVICE,
            )


            attention_mask = torch.cat(
                [
                    attention_mask,

                    torch.ones(
                        (
                            attention_mask.shape[0],
                            1,
                        ),
                        dtype=
                            attention_mask.dtype,
                        device=
                            DEVICE,
                    ),
                ],
                dim=1,
            )


            capture_state[
                "hidden"
            ] = None


            steering_state[
                "pre_activation"
            ] = None


            steering_state[
                "post_activation"
            ] = None


            steering_state[
                "collect_activation"
            ] = True


            steering_state[
                "enabled"
            ] = (
                abs(
                    activation_delta
                )
                > EPS
            )


            with torch.inference_mode():

                output = model(
                    input_ids=
                        current_token,

                    attention_mask=
                        attention_mask,

                    past_key_values=
                        past,

                    use_cache=True,
                )


            steering_state[
                "enabled"
            ] = False


            steering_state[
                "collect_activation"
            ] = False


            past = (
                output.past_key_values
            )


            if capture_state[
                "hidden"
            ] is None:

                raise RuntimeError(
                    "Layer output capture failed."
                )


            states.append(
                capture_state[
                    "hidden"
                ].copy()
            )


            logits_list.append(
                output.logits[
                    0,
                    -1,
                    :
                ]
                .detach()
                .float()
                .cpu()
            )


            pre_activations.append(
                float(
                    steering_state[
                        "pre_activation"
                    ]
                )
            )


            post_activations.append(
                float(
                    steering_state[
                        "post_activation"
                    ]
                )
            )


        return {

            "states":
                np.stack(
                    states,
                    axis=0,
                ),

            "logits":
                logits_list,

            "pre_activations":
                np.asarray(
                    pre_activations,
                    dtype=np.float64,
                ),

            "post_activations":
                np.asarray(
                    post_activations,
                    dtype=np.float64,
                ),

            "activation_delta":
                activation_delta,
        }


    # ========================================================
    # FREE GENERATION
    # ========================================================

    def generate_with_steering(
        alpha,
    ):

        """
        Greedy autoregressive generation.

        Prompt prefill is unsteered.

        The first generated token is therefore the same
        baseline first token.

        Steering begins from the subsequent generated-token
        decoding steps.
        """

        activation_delta = (
            float(alpha)
            * delta_mu
        )


        steering_state[
            "delta"
        ] = activation_delta


        steering_state[
            "enabled"
        ] = False


        steering_state[
            "collect_activation"
        ] = False


        with torch.inference_mode():

            output = model(
                input_ids=
                    PROMPT_IDS,

                attention_mask=
                    PROMPT_ATTENTION,

                use_cache=True,
            )


        past = (
            output.past_key_values
        )


        first_token = int(
            output.logits[
                0,
                -1,
                :
            ]
            .argmax()
            .item()
        )


        generated = [
            first_token
        ]


        if (
            tokenizer.eos_token_id
            is not None
            and first_token
            == tokenizer.eos_token_id
        ):

            text = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )

            return (
                generated,
                text,
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


        attention_mask = (
            PROMPT_ATTENTION.clone()
        )


        for _ in range(
            args.max_new_tokens
            - 1
        ):

            attention_mask = torch.cat(
                [
                    attention_mask,

                    torch.ones(
                        (
                            attention_mask.shape[0],
                            1,
                        ),
                        dtype=
                            attention_mask.dtype,
                        device=
                            DEVICE,
                    ),
                ],
                dim=1,
            )


            steering_state[
                "enabled"
            ] = (
                abs(
                    activation_delta
                )
                > EPS
            )


            with torch.inference_mode():

                output = model(
                    input_ids=
                        current_token,

                    attention_mask=
                        attention_mask,

                    past_key_values=
                        past,

                    use_cache=True,
                )


            steering_state[
                "enabled"
            ] = False


            past = (
                output.past_key_values
            )


            next_token = int(
                output.logits[
                    0,
                    -1,
                    :
                ]
                .argmax()
                .item()
            )


            generated.append(
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


        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )


        return (
            generated,
            text,
        )


    # ========================================================
    # BASELINE FIXED TRAJECTORY FOR THIS LAYER
    # ========================================================

    print()
    print(
        "Building candidate-layer baseline trajectory..."
    )


    baseline_replay = (
        replay_fixed_trajectory(
            alpha=0.0
        )
    )


    base_states = (
        baseline_replay[
            "states"
        ]
    )


    base_logits = (
        baseline_replay[
            "logits"
        ]
    )


    mean_base_state = (
        base_states.mean(
            axis=0
        )
    )


    # ========================================================
    # CANDIDATE METADATA
    # ========================================================

    candidate_metadata_file = (
        candidate_dir
        / "candidate_metadata.json"
    )


    with candidate_metadata_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {

                "actuator_rank":
                    actuator_rank,

                "layer":
                    layer_number,

                "layer_index":
                    layer_idx,

                "neuron_index":
                    neuron_idx,

                "neuron_number":
                    neuron_number,

                "raw_activation_difference":
                    delta_mu,

                "expected_residual_shift_norm_alpha1":
                    expected_norm_alpha1,

                "static_actuator_score_q":
                    static_actuator_q,

                "static_intervention_cosine":
                    static_intervention_cosine,

                "joy_sad_direction_norm":
                    joy_direction_norm,

                "ranking_row":
                    {
                        key:
                            (
                                value.item()
                                if isinstance(
                                    value,
                                    np.generic,
                                )
                                else value
                            )

                        for key, value
                        in candidate.items()
                    },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # ALPHA SWEEP
    # ========================================================

    candidate_alpha_rows = []

    candidate_token_rows = []

    candidate_generation_rows = []

    candidate_json_results = []


    for alpha in args.alphas:

        alpha = float(
            alpha
        )


        predicted_linear_norm = (
            abs(
                alpha
            )
            * expected_norm_alpha1
        )


        if (
            args.max_expected_residual_norm > 0
            and
            predicted_linear_norm
            > args.max_expected_residual_norm
        ):

            print()
            print(
                f"SKIPPING alpha={alpha:+g}: "
                f"predicted linear ||Delta-h|| "
                f"={predicted_linear_norm:.4f} "
                f"> guard "
                f"{args.max_expected_residual_norm:.4f}"
            )

            continue


        print()
        print("-" * 110)

        print(
            f"RANK {actuator_rank} | "
            f"L{layer_number} N{neuron_idx} | "
            f"ALPHA={alpha:+.3f}"
        )

        print("-" * 110)


        # ====================================================
        # FIXED TRAJECTORY
        # ====================================================

        replay = replay_fixed_trajectory(
            alpha
        )


        steer_states = (
            replay[
                "states"
            ]
        )


        steer_logits = (
            replay[
                "logits"
            ]
        )


        mean_steer_state = (
            steer_states.mean(
                axis=0
            )
        )


        mean_delta_h = (
            mean_steer_state
            - mean_base_state
        )


        mean_delta_norm = vector_norm(
            mean_delta_h
        )


        baseline_to_steered_angle = (
            angle_degrees(
                mean_base_state,
                mean_steer_state,
            )
        )


        baseline_to_steered_cosine = (
            cosine_similarity(
                mean_base_state,
                mean_steer_state,
            )
        )


        if mean_delta_norm > EPS:

            actual_delta_joy_cosine = (
                cosine_similarity(
                    mean_delta_h,
                    joy_direction,
                )
            )


            actual_delta_joy_angle = (
                angle_degrees(
                    mean_delta_h,
                    joy_direction,
                )
            )


            mean_joy_projection = float(
                np.dot(
                    mean_delta_h,
                    joy_unit,
                )
            )

        else:

            actual_delta_joy_cosine = float(
                "nan"
            )


            actual_delta_joy_angle = float(
                "nan"
            )


            mean_joy_projection = 0.0


        # ====================================================
        # TOKEN-LEVEL HIDDEN METRICS
        # ====================================================

        token_delta_states = (
            steer_states
            - base_states
        )


        token_delta_norms = (
            np.linalg.norm(
                token_delta_states,
                axis=1,
            )
        )


        token_joy_projections = (
            token_delta_states
            @ joy_unit
        )


        mean_token_delta_norm = float(
            np.mean(
                token_delta_norms
            )
        )


        max_token_delta_norm = float(
            np.max(
                token_delta_norms
            )
        )


        mean_token_joy_projection = float(
            np.mean(
                token_joy_projections
            )
        )


        # ====================================================
        # TOKEN-LEVEL LOGIT METRICS
        # ====================================================

        kl_values = []

        tv_values = []

        logit_l2_values = []

        top_change_values = []

        base_top_prob_change_values = []

        target_logprob_change_values = []


        for position in range(
            len(
                base_logits
            )
        ):

            target_token_id = int(
                BASELINE_IDS[
                    position + 1
                ]
            )


            metrics = compare_logits(
                base_logits[
                    position
                ],

                steer_logits[
                    position
                ],

                target_token_id,
            )


            kl_values.append(
                metrics[
                    "kl_steer_to_base"
                ]
            )


            tv_values.append(
                metrics[
                    "total_variation"
                ]
            )


            logit_l2_values.append(
                metrics[
                    "logit_l2"
                ]
            )


            top_change_values.append(
                metrics[
                    "top_token_changed"
                ]
            )


            base_top_prob_change_values.append(
                metrics[
                    "base_top_probability_change"
                ]
            )


            target_logprob_change_values.append(
                metrics[
                    "target_token_logprob_change"
                ]
            )


            token_delta = (
                token_delta_states[
                    position
                ]
            )


            token_delta_norm = float(
                token_delta_norms[
                    position
                ]
            )


            if token_delta_norm > EPS:

                token_delta_joy_cosine = (
                    cosine_similarity(
                        token_delta,
                        joy_direction,
                    )
                )

            else:

                token_delta_joy_cosine = float(
                    "nan"
                )


            candidate_token_rows.append(
                {

                    "actuator_rank":
                        actuator_rank,

                    "layer":
                        layer_number,

                    "layer_index":
                        layer_idx,

                    "neuron_index":
                        neuron_idx,

                    "alpha":
                        alpha,

                    "position":
                        position,

                    "input_generated_token_id":
                        int(
                            BASELINE_IDS[
                                position
                            ]
                        ),

                    "target_next_token_id":
                        target_token_id,

                    "input_generated_token":
                        tokenizer.decode(
                            [
                                int(
                                    BASELINE_IDS[
                                        position
                                    ]
                                )
                            ]
                        ),

                    "target_next_token":
                        tokenizer.decode(
                            [
                                target_token_id
                            ]
                        ),

                    "activation_pre":
                        float(
                            replay[
                                "pre_activations"
                            ][
                                position
                            ]
                        ),

                    "activation_post":
                        float(
                            replay[
                                "post_activations"
                            ][
                                position
                            ]
                        ),

                    "token_delta_h_norm":
                        token_delta_norm,

                    "token_joy_axis_projection":
                        float(
                            token_joy_projections[
                                position
                            ]
                        ),

                    "token_delta_joy_cosine":
                        token_delta_joy_cosine,

                    **metrics,
                }
            )


        mean_kl = float(
            np.mean(
                kl_values
            )
        )


        max_kl = float(
            np.max(
                kl_values
            )
        )


        mean_tv = float(
            np.mean(
                tv_values
            )
        )


        mean_logit_l2 = float(
            np.mean(
                logit_l2_values
            )
        )


        top_token_change_rate = float(
            np.mean(
                top_change_values
            )
        )


        num_top_token_changes = int(
            np.sum(
                top_change_values
            )
        )


        mean_base_top_probability_change = float(
            np.mean(
                base_top_prob_change_values
            )
        )


        mean_target_logprob_change = float(
            np.mean(
                target_logprob_change_values
            )
        )


        # ====================================================
        # FREE GENERATION
        # ====================================================

        (
            generated_ids,
            generated_text,
        ) = generate_with_steering(
            alpha
        )


        if generated_ids:

            unique_token_ratio = (
                len(
                    set(
                        generated_ids
                    )
                )
                /
                len(
                    generated_ids
                )
            )

        else:

            unique_token_ratio = float(
                "nan"
            )


        generation_changed_from_baseline = (
            generated_ids
            != BASELINE_IDS
        )


        # ====================================================
        # ACTIVATION METRICS
        # ====================================================

        mean_pre_activation = float(
            np.mean(
                replay[
                    "pre_activations"
                ]
            )
        )


        mean_post_activation = float(
            np.mean(
                replay[
                    "post_activations"
                ]
            )
        )


        observed_mean_activation_delta = (
            mean_post_activation
            - mean_pre_activation
        )


        # ====================================================
        # SAVE PNG
        # ====================================================

        png_path = None


        if (
            not args.skip_png
            and
            abs(
                alpha
            )
            > EPS
            and
            mean_delta_norm
            > EPS
        ):

            png_path = (
                png_dir
                / (
                    f"alpha_"
                    f"{alpha_filename(alpha)}.png"
                )
            )


            save_geometry_png(
                output_file=
                    png_path,

                candidate=
                    candidate,

                alpha=
                    alpha,

                h_base=
                    mean_base_state,

                h_steered=
                    mean_steer_state,

                joy_direction=
                    joy_direction,

                activation_delta=
                    replay[
                        "activation_delta"
                    ],
            )


        # ====================================================
        # SUMMARY ROW
        # ====================================================

        summary_row = {

            "actuator_rank":
                actuator_rank,

            "layer":
                layer_number,

            "layer_index":
                layer_idx,

            "neuron_index":
                neuron_idx,

            "neuron_number":
                neuron_number,

            "membership_degree_top1pct":
                int(
                    candidate[
                        "membership_degree_top1pct"
                    ]
                ),

            "original_joy_rank":
                int(
                    candidate[
                        "original_joy_rank"
                    ]
                ),

            "alpha":
                alpha,

            "raw_activation_difference":
                delta_mu,

            "activation_delta":
                float(
                    replay[
                        "activation_delta"
                    ]
                ),

            "predicted_linear_delta_norm":
                predicted_linear_norm,

            "static_actuator_score_q":
                static_actuator_q,

            "static_intervention_cosine":
                static_intervention_cosine,

            "mean_pre_activation":
                mean_pre_activation,

            "mean_post_activation":
                mean_post_activation,

            "observed_mean_activation_delta":
                observed_mean_activation_delta,

            "mean_delta_h_norm":
                mean_delta_norm,

            "mean_token_delta_h_norm":
                mean_token_delta_norm,

            "max_token_delta_h_norm":
                max_token_delta_norm,

            "baseline_to_steered_cosine":
                baseline_to_steered_cosine,

            "baseline_to_steered_angle_deg":
                baseline_to_steered_angle,

            "actual_delta_joy_cosine":
                actual_delta_joy_cosine,

            "actual_delta_joy_angle_deg":
                actual_delta_joy_angle,

            "mean_joy_axis_projection":
                mean_joy_projection,

            "mean_token_joy_axis_projection":
                mean_token_joy_projection,

            "mean_kl_steer_to_base":
                mean_kl,

            "max_kl_steer_to_base":
                max_kl,

            "mean_total_variation":
                mean_tv,

            "mean_logit_l2":
                mean_logit_l2,

            "top_token_change_rate":
                top_token_change_rate,

            "num_top_token_changes":
                num_top_token_changes,

            "mean_base_top_probability_change":
                mean_base_top_probability_change,

            "mean_target_token_logprob_change":
                mean_target_logprob_change,

            "free_generation_changed":
                bool(
                    generation_changed_from_baseline
                ),

            "generated_token_count":
                len(
                    generated_ids
                ),

            "unique_token_ratio":
                float(
                    unique_token_ratio
                ),

            "generated_text":
                generated_text,

            "png":
                (
                    str(
                        png_path
                    )
                    if png_path
                    is not None
                    else None
                ),
        }


        candidate_alpha_rows.append(
            summary_row
        )


        global_summary_rows.append(
            summary_row.copy()
        )


        candidate_generation_rows.append(
            {

                "actuator_rank":
                    actuator_rank,

                "layer":
                    layer_number,

                "neuron_index":
                    neuron_idx,

                "alpha":
                    alpha,

                "activation_delta":
                    float(
                        replay[
                            "activation_delta"
                        ]
                    ),

                "generated_token_count":
                    len(
                        generated_ids
                    ),

                "unique_token_ratio":
                    float(
                        unique_token_ratio
                    ),

                "changed_from_baseline":
                    bool(
                        generation_changed_from_baseline
                    ),

                "generated_text":
                    generated_text,
            }
        )


        candidate_json_results.append(
            summary_row.copy()
        )


        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print(
            f"Activation delta        : "
            f"{replay['activation_delta']:+.8f}"
        )


        print(
            f"Predicted ||Dh||        : "
            f"{predicted_linear_norm:.8f}"
        )


        print(
            f"Actual mean ||Dh||      : "
            f"{mean_delta_norm:.8f}"
        )


        print(
            f"Base->steered angle     : "
            f"{baseline_to_steered_angle:.6f} deg"
        )


        print(
            f"Actual Joy cosine       : "
            f"{actual_delta_joy_cosine:+.8f}"
        )


        print(
            f"Actual Joy angle        : "
            f"{actual_delta_joy_angle:.4f} deg"
        )


        print(
            f"Joy-axis projection     : "
            f"{mean_joy_projection:+.8f}"
        )


        print(
            f"Mean KL                 : "
            f"{mean_kl:.8f}"
        )


        print(
            f"Mean TV                 : "
            f"{mean_tv:.8f}"
        )


        print(
            f"Top-token change rate   : "
            f"{top_token_change_rate:.4%}"
        )


        print(
            f"Generation changed?     : "
            f"{generation_changed_from_baseline}"
        )


        print()
        print("RESPONSE:")
        print(generated_text)


    # ========================================================
    # REMOVE CANDIDATE HOOKS
    # ========================================================

    steering_state[
        "enabled"
    ] = False


    steering_handle.remove()


    layer_handle.remove()


    # ========================================================
    # SAVE PER-CANDIDATE FILES
    # ========================================================

    candidate_alpha_df = pd.DataFrame(
        candidate_alpha_rows
    )


    candidate_alpha_file = (
        candidate_dir
        / "alpha_summary.csv"
    )


    candidate_alpha_df.to_csv(
        candidate_alpha_file,
        index=False,
    )


    candidate_token_df = pd.DataFrame(
        candidate_token_rows
    )


    candidate_token_file = (
        candidate_dir
        / "fixed_trajectory_token_metrics.csv"
    )


    candidate_token_df.to_csv(
        candidate_token_file,
        index=False,
    )


    generation_df = pd.DataFrame(
        candidate_generation_rows
    )


    generation_file = (
        candidate_dir
        / "free_generation.csv"
    )


    generation_df.to_csv(
        generation_file,
        index=False,
    )


    candidate_result_json = (
        candidate_dir
        / "candidate_results.json"
    )


    with candidate_result_json.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {

                "actuator_rank":
                    actuator_rank,

                "layer":
                    layer_number,

                "layer_index":
                    layer_idx,

                "neuron_index":
                    neuron_idx,

                "raw_activation_difference":
                    delta_mu,

                "expected_residual_shift_norm_alpha1":
                    expected_norm_alpha1,

                "static_actuator_score_q":
                    static_actuator_q,

                "static_intervention_cosine":
                    static_intervention_cosine,

                "prompt":
                    args.prompt,

                "baseline_response":
                    BASELINE_TEXT,

                "results":
                    candidate_json_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


    global_json_results.append(
        {

            "actuator_rank":
                actuator_rank,

            "layer":
                layer_number,

            "layer_index":
                layer_idx,

            "neuron_index":
                neuron_idx,

            "raw_activation_difference":
                delta_mu,

            "static_actuator_score_q":
                static_actuator_q,

            "results":
                candidate_json_results,
        }
    )


# ============================================================
# GLOBAL SAVE
# ============================================================

global_summary_df = pd.DataFrame(
    global_summary_rows
)


global_csv = (
    OUTPUT_DIR
    / "actuator_test_summary.csv"
)


global_summary_df.to_csv(
    global_csv,
    index=False,
)


global_json = (
    OUTPUT_DIR
    / "actuator_test_summary.json"
)


with global_json.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {

            "experiment":
                (
                    "Top Joy actuator single-neuron "
                    "steering evaluation"
                ),

            "model":
                MODEL_ID,

            "prompt":
                args.prompt,

            "candidate_ranks":
                args.ranks,

            "alphas":
                args.alphas,

            "max_expected_residual_norm":
                args.max_expected_residual_norm,

            "baseline_response":
                BASELINE_TEXT,

            "fixed_token_trajectory":
                True,

            "prompt_prefill_steered":
                False,

            "results":
                global_json_results,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL TERMINAL TABLE
# ============================================================

print()
print()
print("=" * 110)
print("FINAL ACTUATOR TEST SUMMARY")
print("=" * 110)


if not global_summary_df.empty:

    final_columns = [

        "actuator_rank",

        "layer",

        "neuron_index",

        "alpha",

        "predicted_linear_delta_norm",

        "mean_delta_h_norm",

        "baseline_to_steered_angle_deg",

        "actual_delta_joy_cosine",

        "actual_delta_joy_angle_deg",

        "mean_joy_axis_projection",

        "mean_kl_steer_to_base",

        "mean_total_variation",

        "top_token_change_rate",

        "free_generation_changed",

        "unique_token_ratio",
    ]


    print(
        global_summary_df[
            final_columns
        ].to_string(
            index=False
        )
    )


print()
print("=" * 110)
print("SAVED")
print("=" * 110)


print()
print(
    f"Baseline:\n"
    f"{baseline_json_file}"
)


print()
print(
    f"Global CSV:\n"
    f"{global_csv}"
)


print()
print(
    f"Global JSON:\n"
    f"{global_json}"
)


print()
print(
    f"Candidate directories:\n"
    f"{OUTPUT_DIR}"
)


print()
print("Done.")
