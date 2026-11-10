#!/usr/bin/env python3

"""
Research-quality visualization of baseline vs steered residual-stream geometry.

For each steering coefficient alpha:

    h_base   = mean residual state over a fixed unsteered token trajectory
    h_steer  = mean residual state over the SAME token trajectory with steering
    Delta_h  = h_steer - h_base

Steering rule:

    m' = m + alpha * (mu_joy - mu_sadness)

One PNG is generated per alpha with three panels:

A. Full residual-direction geometry
   - Baseline residual direction
   - Steered residual direction
   - Joy-Sadness direction projected into the same plane

B. Local endpoint zoom
   - Actual endpoint of h_base
   - Actual endpoint of h_steer
   - Actual Delta_h connecting them
   - No artificial angular exaggeration

C. Steering-vs-emotion geometry
   - Delta_h direction
   - Joy-Sadness direction
   - Exact 2048-D angle between them

Geometry:

Panel A/B:
    plane = span(h_base, Delta_h)

Therefore both h_base and h_steer are represented exactly.

Panel C:
    plane = span(Delta_h, Joy-Sadness)

Therefore the angle between Delta_h and the Joy-Sadness direction
is represented exactly.

All reported cosines and angles are calculated from the original
2048-dimensional vectors, not from an arbitrary PCA projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

# Safe for Linux/server/headless execution.
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
    "--alphas",
    type=float,
    nargs="+",
    default=[
        -5.0,
        -3.0,
        -1.0,
        1.0,
        3.0,
        5.0,
    ],
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=30,
)

parser.add_argument(
    "--local-files-only",
    action="store_true",
    help=(
        "Load tokenizer/model only from local Hugging Face cache. "
        "Useful if the model is already downloaded."
    ),
)

args = parser.parse_args()


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

EPS = 1e-12


BEST_NEURON_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "13_single_joy_neuron_selection"
    / "best_joy_neuron.json"
)


JOY_DIRECTION_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "15_direction_and_logit_analysis"
    / "joy_minus_sad_hidden_direction.npy"
)


OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "16_steering_direction_visualization"
)


PNG_DIR = (
    OUTPUT_DIR
    / "png"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PNG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
print("STEERING-DIRECTION VISUALIZATION — 3-PANEL VERSION")
print("=" * 100)

print(f"Device : {DEVICE}")
print(f"Dtype  : {DTYPE}")


# ============================================================
# INPUT CHECKS
# ============================================================

if not BEST_NEURON_FILE.is_file():

    raise FileNotFoundError(
        f"Missing neuron metadata:\n"
        f"{BEST_NEURON_FILE}"
    )


if not JOY_DIRECTION_FILE.is_file():

    raise FileNotFoundError(
        "\nMissing Joy-Sad residual direction:\n"
        f"{JOY_DIRECTION_FILE}\n\n"
        "Run 15_direction_and_logit_analysis.py first."
    )


# ============================================================
# LOAD NEURON METADATA
# ============================================================

with BEST_NEURON_FILE.open(
    "r",
    encoding="utf-8",
) as f:

    neuron_metadata = json.load(f)


best = neuron_metadata[
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


RAW_DIRECTION = (
    MU_JOY
    - MU_SAD
)


print()
print("SELECTED NEURON")
print("-" * 100)

print(
    f"Layer             : {LAYER_NUMBER}"
)

print(
    f"Layer index       : {LAYER_IDX}"
)

print(
    f"Neuron index      : {NEURON_IDX}"
)

print(
    f"Joy mean          : {MU_JOY:+.8f}"
)

print(
    f"Sadness mean      : {MU_SAD:+.8f}"
)

print(
    f"Steering unit     : {RAW_DIRECTION:+.8f}"
)


# ============================================================
# LOAD JOY-SADNESS DIRECTION
# ============================================================

joy_direction = np.load(
    JOY_DIRECTION_FILE
).astype(
    np.float64
)


joy_direction_norm = float(
    np.linalg.norm(
        joy_direction
    )
)


if joy_direction_norm < EPS:

    raise RuntimeError(
        "Joy-Sadness direction has near-zero norm."
    )


joy_unit = (
    joy_direction
    / joy_direction_norm
)


print(
    f"Joy-Sad direction norm : "
    f"{joy_direction_norm:.8f}"
)


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
    local_files_only=
        args.local_files_only,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 100)
print("LOADING SARVAM-1")
print("=" * 100)


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

    parameter.requires_grad_(False)


layers = (
    model.model.layers
)


if (
    LAYER_IDX < 0
    or LAYER_IDX >= len(layers)
):

    raise RuntimeError(
        f"Invalid layer index: "
        f"{LAYER_IDX}"
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


if NEURON_IDX >= mlp_width:

    raise RuntimeError(
        f"Neuron index {NEURON_IDX} "
        f"outside MLP width {mlp_width}"
    )


if (
    joy_direction.shape[0]
    != hidden_size
):

    raise RuntimeError(
        f"Joy direction dimension "
        f"{joy_direction.shape[0]} "
        f"!= hidden size "
        f"{hidden_size}"
    )


print(
    f"Hidden size            : "
    f"{hidden_size}"
)

print(
    f"MLP width              : "
    f"{mlp_width}"
)


# ============================================================
# LAYER OUTPUT CAPTURE
# ============================================================

capture_state = {
    "hidden": None,
}


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
            f"layer output type: {type(output)}"
        )


    # During cached generation the sequence length
    # is normally 1. Capture the current token state.
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


layer_handle = (
    layers[
        LAYER_IDX
    ]
    .register_forward_hook(
        layer_output_hook
    )
)


# ============================================================
# STEERING HOOK
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


    modified = (
        x.clone()
    )


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
# TOKENIZE PROMPT
# ============================================================

encoded = tokenizer(
    args.prompt,
    return_tensors="pt",
)


prompt_ids = (
    encoded[
        "input_ids"
    ]
    .to(
        DEVICE
    )
)


prompt_attention = (
    encoded[
        "attention_mask"
    ]
    .to(
        DEVICE
    )
)


# ============================================================
# BASELINE TRAJECTORY
# ============================================================

def build_baseline():

    """
    Generate the normal greedy continuation.

    Also record the residual state at the selected
    transformer layer for every generated step.

    The prompt pass itself is not included because
    steering starts only after generation begins.
    """

    steering_state[
        "enabled"
    ] = False


    capture_state[
        "hidden"
    ] = None


    with torch.inference_mode():

        output = model(
            input_ids=
                prompt_ids,

            attention_mask=
                prompt_attention,

            use_cache=True,
        )


    past = (
        output.past_key_values
    )


    # First generated token comes entirely
    # from normal prompt processing.
    first_token = int(
        output.logits[
            0,
            -1,
            :
        ]
        .argmax()
        .item()
    )


    generated_ids = [
        first_token
    ]


    layer_states = []


    attention_mask = (
        prompt_attention.clone()
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

        raise RuntimeError(
            "Baseline immediately generated EOS. "
            "No trajectory available."
        )


    for _ in range(
        args.max_new_tokens
        - 1
    ):

        new_mask = torch.ones(
            (
                attention_mask.shape[0],
                1,
            ),
            dtype=
                attention_mask.dtype,
            device=
                DEVICE,
        )


        attention_mask = torch.cat(
            [
                attention_mask,
                new_mask,
            ],
            dim=1,
        )


        capture_state[
            "hidden"
        ] = None


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


        if capture_state[
            "hidden"
        ] is None:

            raise RuntimeError(
                "Baseline layer "
                "capture failed."
            )


        layer_states.append(
            capture_state[
                "hidden"
            ].copy()
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


    if not layer_states:

        raise RuntimeError(
            "No baseline residual "
            "states were captured."
        )


    return (
        generated_ids,
        np.stack(
            layer_states,
            axis=0,
        ),
    )


print()
print("=" * 100)
print("BUILDING BASELINE")
print("=" * 100)


(
    baseline_ids,
    baseline_states,
) = build_baseline()


baseline_text = tokenizer.decode(
    baseline_ids,
    skip_special_tokens=True,
)


print()
print("Baseline continuation:")
print(baseline_text)


print()

print(
    "Captured residual states : "
    f"{baseline_states.shape}"
)


# ============================================================
# FIXED-TRAJECTORY STEERED REPLAY
# ============================================================

def run_steered_trajectory(
    alpha,
):

    """
    Replay exactly the SAME baseline token sequence.

    Only the internal MLP neuron is modified.

    Therefore differences in hidden states are caused
    by the intervention rather than by comparing two
    different lexical sequences.
    """

    delta = (
        float(alpha)
        * RAW_DIRECTION
    )


    steering_state[
        "delta"
    ] = delta


    # --------------------------------------------------------
    # Prompt pass remains normal.
    # --------------------------------------------------------

    steering_state[
        "enabled"
    ] = False


    capture_state[
        "hidden"
    ] = None


    with torch.inference_mode():

        output = model(
            input_ids=
                prompt_ids,

            attention_mask=
                prompt_attention,

            use_cache=True,
        )


    past = (
        output.past_key_values
    )


    attention_mask = (
        prompt_attention.clone()
    )


    states = []


    # --------------------------------------------------------
    # Teacher-force baseline generated tokens.
    # --------------------------------------------------------

    for j in range(
        1,
        len(
            baseline_ids
        ),
    ):

        previous_token = int(
            baseline_ids[
                j - 1
            ]
        )


        current_token = (
            torch.tensor(
                [
                    [
                        previous_token
                    ]
                ],
                dtype=torch.long,
                device=DEVICE,
            )
        )


        new_mask = torch.ones(
            (
                attention_mask.shape[0],
                1,
            ),
            dtype=
                attention_mask.dtype,
            device=
                DEVICE,
        )


        attention_mask = torch.cat(
            [
                attention_mask,
                new_mask,
            ],
            dim=1,
        )


        capture_state[
            "hidden"
        ] = None


        steering_state[
            "enabled"
        ] = (
            abs(
                float(alpha)
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


        past = (
            output.past_key_values
        )


        steering_state[
            "enabled"
        ] = False


        if capture_state[
            "hidden"
        ] is None:

            raise RuntimeError(
                "Steered layer "
                "capture failed."
            )


        states.append(
            capture_state[
                "hidden"
            ].copy()
        )


    if not states:

        raise RuntimeError(
            f"No states captured "
            f"for alpha={alpha}"
        )


    return np.stack(
        states,
        axis=0,
    )


# ============================================================
# VECTOR UTILITIES
# ============================================================

def safe_norm(
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


    norm_a = safe_norm(
        a
    )


    norm_b = safe_norm(
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


def vector_angle_degrees(
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
# EXACT BASELINE / STEERING PLANE
# ============================================================

def build_visualization_basis(
    h_base,
    delta_h,
    joy_axis,
):

    """
    Construct:

        e1 = h_base / ||h_base||

        e2 = normalized component of Delta_h
             orthogonal to e1

    Therefore:

        span(e1,e2)
        =
        span(h_base, Delta_h)

    Since:

        h_steered = h_base + Delta_h

    baseline and steered vectors are represented
    exactly in this plane.
    """

    base_norm = (
        safe_norm(
            h_base
        )
    )


    if base_norm < EPS:

        raise RuntimeError(
            "Baseline residual vector "
            "has near-zero norm."
        )


    e1 = (
        h_base
        / base_norm
    )


    orthogonal_delta = (
        delta_h
        - np.dot(
            delta_h,
            e1,
        )
        * e1
    )


    orthogonal_norm = (
        safe_norm(
            orthogonal_delta
        )
    )


    if orthogonal_norm > 1e-10:

        e2 = (
            orthogonal_delta
            / orthogonal_norm
        )

        return e1, e2


    # --------------------------------------------------------
    # Fallback:
    # use Joy-Sad component orthogonal to h_base.
    # --------------------------------------------------------

    joy_orthogonal = (
        joy_axis
        - np.dot(
            joy_axis,
            e1,
        )
        * e1
    )


    joy_orthogonal_norm = (
        safe_norm(
            joy_orthogonal
        )
    )


    if joy_orthogonal_norm > 1e-10:

        e2 = (
            joy_orthogonal
            / joy_orthogonal_norm
        )

        return e1, e2


    # --------------------------------------------------------
    # Final numerical fallback.
    # --------------------------------------------------------

    candidate = np.zeros_like(
        e1
    )


    # Pick dimension where e1 is smallest.
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


    candidate_norm = (
        safe_norm(
            candidate
        )
    )


    if candidate_norm < EPS:

        raise RuntimeError(
            "Could not construct "
            "visualization basis."
        )


    e2 = (
        candidate
        / candidate_norm
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


# ============================================================
# EXACT DELTA-H / EMOTION PLANE
# ============================================================

def build_delta_emotion_plane(
    delta_h,
    emotion_direction,
):

    """
    Construct exact plane containing:

        Delta_h
        Joy-Sadness direction

    e1 points along Delta_h.

    e2 contains the orthogonal component of
    Joy-Sadness.

    Therefore the displayed angle is exactly
    the same as the original 2048-D angle.
    """

    delta_norm = (
        safe_norm(
            delta_h
        )
    )


    emotion_norm = (
        safe_norm(
            emotion_direction
        )
    )


    if delta_norm < EPS:

        raise RuntimeError(
            "Delta-h has near-zero norm."
        )


    if emotion_norm < EPS:

        raise RuntimeError(
            "Emotion direction "
            "has near-zero norm."
        )


    e1 = (
        delta_h
        / delta_norm
    )


    emotion_unit = (
        emotion_direction
        / emotion_norm
    )


    emotion_orthogonal = (
        emotion_unit
        - np.dot(
            emotion_unit,
            e1,
        )
        * e1
    )


    orthogonal_norm = (
        safe_norm(
            emotion_orthogonal
        )
    )


    if orthogonal_norm > EPS:

        e2 = (
            emotion_orthogonal
            / orthogonal_norm
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
                safe_norm(
                    candidate
                )
                + EPS
            )
        )


    delta_plot = np.array(
        [
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )


    emotion_plot = np.array(
        [
            np.dot(
                emotion_unit,
                e1,
            ),

            np.dot(
                emotion_unit,
                e2,
            ),
        ],
        dtype=np.float64,
    )


    return (
        e1,
        e2,
        delta_plot,
        emotion_plot,
    )


# ============================================================
# PLOT HELPER
# ============================================================

def draw_arrow(
    ax,
    start,
    end,
    color,
    linewidth=3.0,
    mutation_scale=18,
    zorder=5,
):

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=
            mutation_scale,
        linewidth=
            linewidth,
        color=
            color,
        shrinkA=0,
        shrinkB=0,
        zorder=
            zorder,
    )


    ax.add_patch(
        arrow
    )


    return arrow


# ============================================================
# SAVE THREE-PANEL PNG
# ============================================================

def save_direction_png(
    alpha,
    h_base,
    h_steered,
    joy_axis,
    output_file,
):

    h_base = np.asarray(
        h_base,
        dtype=np.float64,
    )


    h_steered = np.asarray(
        h_steered,
        dtype=np.float64,
    )


    joy_axis = np.asarray(
        joy_axis,
        dtype=np.float64,
    )


    # ========================================================
    # ORIGINAL 2048-D METRICS
    # ========================================================

    delta_h = (
        h_steered
        - h_base
    )


    baseline_norm = (
        safe_norm(
            h_base
        )
    )


    steered_norm = (
        safe_norm(
            h_steered
        )
    )


    delta_norm = (
        safe_norm(
            delta_h
        )
    )


    baseline_steered_cosine = (
        cosine_similarity(
            h_base,
            h_steered,
        )
    )


    baseline_steered_angle = (
        vector_angle_degrees(
            h_base,
            h_steered,
        )
    )


    delta_joy_cosine = (
        cosine_similarity(
            delta_h,
            joy_axis,
        )
    )


    delta_joy_angle = (
        vector_angle_degrees(
            delta_h,
            joy_axis,
        )
    )


    joy_axis_projection = float(
        np.dot(
            delta_h,
            joy_unit,
        )
    )


    activation_delta = (
        float(alpha)
        * RAW_DIRECTION
    )


    # ========================================================
    # PLANE A/B
    # span(h_base, Delta_h)
    # ========================================================

    e1, e2 = (
        build_visualization_basis(
            h_base=
                h_base,

            delta_h=
                delta_h,

            joy_axis=
                joy_axis,
        )
    )


    base_2d = project_to_plane(
        h_base,
        e1,
        e2,
    )


    steered_2d = project_to_plane(
        h_steered,
        e1,
        e2,
    )


    delta_2d = (
        steered_2d
        - base_2d
    )


    joy_2d = project_to_plane(
        joy_axis,
        e1,
        e2,
    )


    # --------------------------------------------------------
    # Unit-length direction display for Panel A only.
    # --------------------------------------------------------

    base_plot = (
        base_2d
        /
        (
            safe_norm(
                base_2d
            )
            + EPS
        )
    )


    steer_plot = (
        steered_2d
        /
        (
            safe_norm(
                steered_2d
            )
            + EPS
        )
    )


    joy_2d_norm = (
        safe_norm(
            joy_2d
        )
    )


    if joy_2d_norm > EPS:

        joy_plot = (
            joy_2d
            / joy_2d_norm
        )

    else:

        joy_plot = np.array(
            [
                1.0,
                0.0,
            ]
        )


    # ========================================================
    # PLANE C
    # span(Delta_h, Joy-Sadness)
    # ========================================================

    (
        _,
        _,
        delta_emotion_plot,
        joy_emotion_plot,
    ) = build_delta_emotion_plane(
        delta_h,
        joy_axis,
    )


    # ========================================================
    # COLORS
    # ========================================================

    BASE_COLOR = "#2563EB"

    STEER_COLOR = "#DC2626"

    DELTA_COLOR = "#F59E0B"

    JOY_COLOR = "#16A34A"

    GRID_COLOR = "#D1D5DB"

    TEXT_COLOR = "#111827"


    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            19,
            7,
        ),
    )


    ax1, ax2, ax3 = axes


    fig.suptitle(
        (
            "Single-Neuron Steering Geometry\n"
            f"Layer {LAYER_NUMBER}, "
            f"Neuron {NEURON_IDX}, "
            f"alpha = {alpha:+g}"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )


    # ========================================================
    # PANEL A
    # FULL RESIDUAL-DIRECTION GEOMETRY
    # ========================================================

    ax1.set_title(
        "A. Full residual-direction geometry",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )


    unit_circle = plt.Circle(
        (
            0,
            0,
        ),
        1.0,
        fill=False,
        linewidth=1.0,
        color=GRID_COLOR,
        alpha=0.8,
    )


    ax1.add_patch(
        unit_circle
    )


    draw_arrow(
        ax1,
        (
            0,
            0,
        ),
        (
            base_plot[0],
            base_plot[1],
        ),
        color=
            BASE_COLOR,
        linewidth=
            4.5,
        mutation_scale=
            20,
    )


    draw_arrow(
        ax1,
        (
            0,
            0,
        ),
        (
            steer_plot[0],
            steer_plot[1],
        ),
        color=
            STEER_COLOR,
        linewidth=
            2.5,
        mutation_scale=
            18,
        zorder=6,
    )


    draw_arrow(
        ax1,
        (
            0,
            0,
        ),
        (
            joy_plot[0],
            joy_plot[1],
        ),
        color=
            JOY_COLOR,
        linewidth=
            2.8,
        mutation_scale=
            18,
    )


    # --------------------------------------------------------
    # Labels separated from endpoints.
    # --------------------------------------------------------

    ax1.annotate(
        "Baseline",
        xy=(
            base_plot[0],
            base_plot[1],
        ),
        xytext=(
            -18,
            25,
        ),
        textcoords=
            "offset points",
        color=
            BASE_COLOR,
        fontsize=11,
        fontweight="bold",
        ha="right",
    )


    ax1.annotate(
        "Steered",
        xy=(
            steer_plot[0],
            steer_plot[1],
        ),
        xytext=(
            -18,
            -32,
        ),
        textcoords=
            "offset points",
        color=
            STEER_COLOR,
        fontsize=11,
        fontweight="bold",
        ha="right",
    )


    ax1.annotate(
        (
            "Joy - Sadness\n"
            "(projection)"
        ),
        xy=(
            joy_plot[0],
            joy_plot[1],
        ),
        xytext=(
            12,
            5,
        ),
        textcoords=
            "offset points",
        color=
            JOY_COLOR,
        fontsize=10,
        fontweight="bold",
    )


    ax1.text(
        0.03,
        0.04,
        (
            "Baseline and steered arrows may visually overlap\n"
            "because their measured angular difference is tiny.\n\n"
            f"True 2048-D angle\n"
            f"= {baseline_steered_angle:.6f} deg"
        ),
        transform=
            ax1.transAxes,
        fontsize=9.5,
        va="bottom",
        bbox=dict(
            boxstyle=
                "round,pad=0.45",
            facecolor=
                "white",
            edgecolor=
                "#9CA3AF",
            alpha=
                0.94,
        ),
    )


    ax1.axhline(
        0,
        linewidth=0.8,
        color=GRID_COLOR,
    )


    ax1.axvline(
        0,
        linewidth=0.8,
        color=GRID_COLOR,
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
        "equal",
        adjustable="box",
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
    # PANEL B
    # TRUE LOCAL ENDPOINT ZOOM
    # ========================================================

    ax2.set_title(
        "B. Local zoom of actual state change",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )


    ax2.scatter(
        base_2d[0],
        base_2d[1],
        s=130,
        color=BASE_COLOR,
        edgecolors="white",
        linewidths=1.5,
        zorder=8,
    )


    ax2.scatter(
        steered_2d[0],
        steered_2d[1],
        s=130,
        color=STEER_COLOR,
        edgecolors="white",
        linewidths=1.5,
        zorder=8,
    )


    # Actual displacement:
    #
    # h_base -> h_steered
    draw_arrow(
        ax2,
        (
            base_2d[0],
            base_2d[1],
        ),
        (
            steered_2d[0],
            steered_2d[1],
        ),
        color=
            DELTA_COLOR,
        linewidth=
            3.5,
        mutation_scale=
            22,
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
            -28,
        ),
        textcoords=
            "offset points",
        color=
            BASE_COLOR,
        fontsize=10,
        fontweight="bold",
        ha="right",
    )


    ax2.annotate(
        "Steered endpoint",
        xy=(
            steered_2d[0],
            steered_2d[1],
        ),
        xytext=(
            10,
            18,
        ),
        textcoords=
            "offset points",
        color=
            STEER_COLOR,
        fontsize=10,
        fontweight="bold",
        ha="left",
    )


    midpoint = (
        base_2d
        + steered_2d
    ) / 2.0


    ax2.annotate(
        (
            "Delta-h\n"
            f"||Delta-h|| = "
            f"{delta_norm:.6f}"
        ),
        xy=(
            midpoint[0],
            midpoint[1],
        ),
        xytext=(
            20,
            2,
        ),
        textcoords=
            "offset points",
        color=
            DELTA_COLOR,
        fontsize=10,
        fontweight="bold",
        ha="left",
    )


    # --------------------------------------------------------
    # Dynamic true zoom.
    #
    # Nothing is rotated/exaggerated.
    # We simply zoom into the real endpoints.
    # --------------------------------------------------------

    center = (
        base_2d
        + steered_2d
    ) / 2.0


    raw_distance = (
        safe_norm(
            steered_2d
            - base_2d
        )
    )


    local_radius = max(
        0.10,
        raw_distance
        * 2.0,
    )


    ax2.set_xlim(
        center[0]
        - local_radius,
        center[0]
        + local_radius,
    )


    ax2.set_ylim(
        center[1]
        - local_radius,
        center[1]
        + local_radius,
    )


    ax2.set_aspect(
        "equal",
        adjustable="box",
    )


    ax2.grid(
        True,
        alpha=0.20,
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
            "No angular exaggeration\n\n"
            f"angle(h_base, h_steered)\n"
            f"= {baseline_steered_angle:.6f} deg"
        ),
        transform=
            ax2.transAxes,
        va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle=
                "round,pad=0.45",
            facecolor=
                "white",
            edgecolor=
                "#9CA3AF",
            alpha=
                0.94,
        ),
    )


    # ========================================================
    # PANEL C
    # DELTA-H VS JOY-SADNESS
    # ========================================================

    ax3.set_title(
        "C. Steering direction vs emotion direction",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )


    emotion_circle = plt.Circle(
        (
            0,
            0,
        ),
        1.0,
        fill=False,
        linewidth=1.0,
        color=GRID_COLOR,
        alpha=0.8,
    )


    ax3.add_patch(
        emotion_circle
    )


    # Delta-h is +x by construction.
    draw_arrow(
        ax3,
        (
            0,
            0,
        ),
        (
            delta_emotion_plot[0],
            delta_emotion_plot[1],
        ),
        color=
            DELTA_COLOR,
        linewidth=
            3.7,
        mutation_scale=
            20,
    )


    # Joy-Sadness in exact same plane.
    draw_arrow(
        ax3,
        (
            0,
            0,
        ),
        (
            joy_emotion_plot[0],
            joy_emotion_plot[1],
        ),
        color=
            JOY_COLOR,
        linewidth=
            3.7,
        mutation_scale=
            20,
    )


    ax3.annotate(
        (
            "Steering displacement\n"
            "Delta-h"
        ),
        xy=(
            delta_emotion_plot[0],
            delta_emotion_plot[1],
        ),
        xytext=(
            -15,
            24,
        ),
        textcoords=
            "offset points",
        color=
            DELTA_COLOR,
        fontsize=10,
        fontweight="bold",
        ha="right",
    )


    ax3.annotate(
        "Joy - Sadness",
        xy=(
            joy_emotion_plot[0],
            joy_emotion_plot[1],
        ),
        xytext=(
            12,
            4,
        ),
        textcoords=
            "offset points",
        color=
            JOY_COLOR,
        fontsize=10,
        fontweight="bold",
    )


    # --------------------------------------------------------
    # Exact angle arc.
    # --------------------------------------------------------

    if np.isfinite(
        delta_joy_angle
    ):

        arc_radius = 0.42


        angle_arc = Arc(
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
            theta1=0,
            theta2=
                delta_joy_angle,
            linewidth=2.2,
            color=TEXT_COLOR,
        )


        ax3.add_patch(
            angle_arc
        )


        midpoint_angle_rad = (
            np.deg2rad(
                delta_joy_angle
                / 2.0
            )
        )


        ax3.text(
            0.56
            * np.cos(
                midpoint_angle_rad
            ),
            0.56
            * np.sin(
                midpoint_angle_rad
            ),
            (
                f"{delta_joy_angle:.3f} deg"
            ),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color=TEXT_COLOR,
        )


    ax3.text(
        0.03,
        0.04,
        (
            f"cos(Delta-h, Joy-Sad)\n"
            f"= {delta_joy_cosine:+.6f}\n\n"

            f"Joy-axis projection\n"
            f"= {joy_axis_projection:+.6f}"
        ),
        transform=
            ax3.transAxes,
        fontsize=9.8,
        va="bottom",
        bbox=dict(
            boxstyle=
                "round,pad=0.45",
            facecolor=
                "white",
            edgecolor=
                "#9CA3AF",
            alpha=
                0.94,
        ),
    )


    ax3.axhline(
        0,
        linewidth=0.8,
        color=GRID_COLOR,
    )


    ax3.axvline(
        0,
        linewidth=0.8,
        color=GRID_COLOR,
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
        "equal",
        adjustable="box",
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


    # ========================================================
    # BOTTOM METRICS
    # ========================================================

    bottom_text = (
        f"activation Delta-m = "
        f"{activation_delta:+.6f}"
        "     |     "

        f"||h_base|| = "
        f"{baseline_norm:.4f}"
        "     |     "

        f"||h_steered|| = "
        f"{steered_norm:.4f}"
        "     |     "

        f"||Delta-h|| = "
        f"{delta_norm:.6f}"
        "     |     "

        f"cos(base, steered) = "
        f"{baseline_steered_cosine:.8f}"
        "     |     "

        f"angle = "
        f"{baseline_steered_angle:.6f} deg"
    )


    fig.text(
        0.5,
        0.025,
        bottom_text,
        ha="center",
        va="center",
        fontsize=9.6,
        color=TEXT_COLOR,
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.subplots_adjust(
        left=0.05,
        right=0.985,
        bottom=0.12,
        top=0.83,
        wspace=0.28,
    )


    # ========================================================
    # SAVE
    # ========================================================

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


    plt.close(
        fig
    )


    # ========================================================
    # RETURN NUMERICAL RESULTS
    # ========================================================

    return {

        "alpha":
            float(
                alpha
            ),

        "activation_delta":
            float(
                activation_delta
            ),

        "baseline_norm":
            baseline_norm,

        "steered_norm":
            steered_norm,

        "delta_norm":
            delta_norm,

        "baseline_steered_cosine":
            baseline_steered_cosine,

        "baseline_steered_angle_deg":
            baseline_steered_angle,

        "delta_joy_cosine":
            delta_joy_cosine,

        "delta_joy_angle_deg":
            delta_joy_angle,

        "joy_axis_projection":
            joy_axis_projection,

        "png":
            str(
                output_file
            ),
    }


# ============================================================
# RUN ALL ALPHAS
# ============================================================

rows = []


try:

    for alpha in args.alphas:

        # ----------------------------------------------------
        # alpha=0 means steered == baseline.
        # There is no Delta-h direction to visualize.
        # ----------------------------------------------------

        if abs(
            float(alpha)
        ) < EPS:

            print()
            print(
                "Skipping alpha=0 because "
                "Delta-h = 0."
            )

            continue


        print()
        print("=" * 100)

        print(
            f"VISUALIZING "
            f"ALPHA {alpha:+.3f}"
        )

        print("=" * 100)


        steered_states = (
            run_steered_trajectory(
                alpha
            )
        )


        # ----------------------------------------------------
        # Compare exactly matching trajectory positions.
        # ----------------------------------------------------

        n_steps = min(
            len(
                baseline_states
            ),
            len(
                steered_states
            ),
        )


        if n_steps <= 0:

            raise RuntimeError(
                f"No common states "
                f"for alpha={alpha}"
            )


        base = (
            baseline_states[
                :n_steps
            ]
        )


        steer = (
            steered_states[
                :n_steps
            ]
        )


        # ----------------------------------------------------
        # Mean residual state across generated positions.
        # ----------------------------------------------------

        mean_base = (
            base.mean(
                axis=0
            )
        )


        mean_steered = (
            steer.mean(
                axis=0
            )
        )


        filename_alpha = (
            f"{float(alpha):g}"
            .replace(
                "-",
                "m"
            )
            .replace(
                ".",
                "p"
            )
        )


        png_file = (
            PNG_DIR
            / (
                "baseline_vs_steered_"
                f"alpha_{filename_alpha}.png"
            )
        )


        result = (
            save_direction_png(
                alpha=
                    alpha,

                h_base=
                    mean_base,

                h_steered=
                    mean_steered,

                joy_axis=
                    joy_direction,

                output_file=
                    png_file,
            )
        )


        rows.append(
            result
        )


        print(
            f"Saved PNG                  : "
            f"{png_file}"
        )


        print(
            f"Activation Delta-m         : "
            f"{result['activation_delta']:+.8f}"
        )


        print(
            f"||Delta-h||                : "
            f"{result['delta_norm']:.8f}"
        )


        print(
            f"Angle baseline -> steered  : "
            f"{result['baseline_steered_angle_deg']:.6f} deg"
        )


        print(
            f"cos(Delta-h, Joy-Sad)      : "
            f"{result['delta_joy_cosine']:+.6f}"
        )


        print(
            f"Angle Delta-h vs Joy-Sad   : "
            f"{result['delta_joy_angle_deg']:.3f} deg"
        )


        print(
            f"Joy-axis projection        : "
            f"{result['joy_axis_projection']:+.6f}"
        )


finally:

    steering_state[
        "enabled"
    ] = False


    layer_handle.remove()


    steering_handle.remove()


# ============================================================
# SAVE SUMMARY
# ============================================================

results_df = pd.DataFrame(
    rows
)


summary_csv = (
    OUTPUT_DIR
    / "steering_direction_summary.csv"
)


results_df.to_csv(
    summary_csv,
    index=False,
)


summary_json = (
    OUTPUT_DIR
    / "steering_direction_summary.json"
)


with summary_json.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {

            "experiment":
                (
                    "Fixed-trajectory "
                    "single-neuron steering "
                    "geometry visualization"
                ),

            "prompt":
                args.prompt,

            "model":
                MODEL_ID,

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

            "steering_unit":
                RAW_DIRECTION,

            "joy_sad_direction_norm":
                joy_direction_norm,

            "baseline_continuation":
                baseline_text,

            "num_baseline_states":
                int(
                    len(
                        baseline_states
                    )
                ),

            "steering_rule":
                (
                    "m' = m + alpha * "
                    "(mu_joy - mu_sadness)"
                ),

            "fixed_baseline_token_trajectory":
                True,

            "prompt_pass_steered":
                False,

            "results":
                rows,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 100)
print("FINAL VISUALIZATION SUMMARY")
print("=" * 100)


if not results_df.empty:

    columns = [

        "alpha",

        "activation_delta",

        "delta_norm",

        "baseline_steered_angle_deg",

        "delta_joy_cosine",

        "delta_joy_angle_deg",

        "joy_axis_projection",
    ]


    print(
        results_df[
            columns
        ].to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)


print()
print("PNG directory:")
print(PNG_DIR)


print()
print("Summary CSV:")
print(summary_csv)


print()
print("Summary JSON:")
print(summary_json)


print()
print("Done.")
