#!/usr/bin/env python3

"""
Single-neuron joy steering experiment for Sarvam-1.

Selected neuron:
    loaded automatically from best_joy_neuron.json

Steering rule:
    m' = m + alpha * (mu_joy - mu_sadness)

NOT:
    m' = m + alpha * standardized_effect_size

The standardized effect size was used only to SELECT the neuron.

For each alpha:
    - same prompt
    - deterministic greedy decoding
    - same model
    - same neuron
    - different steering intensity

We record:
    - generated response
    - activation shift
    - residual-vector displacement norm
    - joy/sadness probe probabilities
    - all emotion probabilities
    - prediction
    - base-model NLL/perplexity
    - repetition statistics
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    "--max-new-tokens",
    type=int,
    default=80,
)

parser.add_argument(
    "--alphas",
    type=float,
    nargs="+",
    default=[
        -2.0,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        2.0,
    ],
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BEST_NEURON_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "13_single_joy_neuron_selection"
    / "best_joy_neuron.json"
)

PROBE_FILE = (
    BASE_DIR
    / "results"
    / "llama31_8b_neuron_analysis"
    / "04_probe"
    / "probe.pt"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_steering"
    / "14_single_joy_neuron_steering"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

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

JOY_ID = EMOTIONS.index("joy")
SADNESS_ID = EMOTIONS.index("sadness")

MAX_SCORE_LENGTH = 256


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
print("SINGLE JOY-NEURON STEERING")
print("=" * 90)

print(f"Device : {DEVICE}")
print(f"Dtype  : {DTYPE}")


# ============================================================
# LOAD NEURON METADATA
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

W_DIRECTION_NORM = float(
    best["residual_direction_norm"]
)


print()
print("Selected neuron")
print("-" * 90)

print(
    f"Layer number                : {LAYER_NUMBER}"
)

print(
    f"Layer index                 : {LAYER_IDX}"
)

print(
    f"Neuron index                : {NEURON_IDX}"
)

print(
    f"Mean joy activation         : {MU_JOY:+.8f}"
)

print(
    f"Mean sadness activation     : {MU_SAD:+.8f}"
)

print(
    f"Raw joy-sad direction       : {RAW_DIRECTION:+.8f}"
)

print(
    f"Pooled SD                   : {POOLED_SD:.8f}"
)

print(
    f"Standardized effect         : {STANDARDIZED_EFFECT:+.8f}"
)

print(
    f"W_down direction norm       : {W_DIRECTION_NORM:.8f}"
)


print()
print(
    "IMPORTANT: standardized effect is used for neuron "
    "selection only."
)

print(
    "Actual steering unit = "
    f"mu_joy - mu_sad = {RAW_DIRECTION:+.8f}"
)


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 90)
print("LOADING TOKENIZER")
print("=" * 90)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 90)
print("LOADING SARVAM-1")
print("=" * 90)


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


if LAYER_IDX >= len(layers):

    raise RuntimeError(
        f"Invalid layer {LAYER_IDX}"
    )


backbone = model.model


# ============================================================
# LOAD FROZEN EMOTION PROBE
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
        probe.to(DEVICE)
        probe.eval()

        for p in probe.parameters():

            p.requires_grad_(False)

        return probe


    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            f"Unsupported probe checkpoint: "
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


    if state is None:

        state = checkpoint


    clean_state = {}


    for key, value in state.items():

        if not torch.is_tensor(value):

            continue


        if key.startswith("module."):

            key = key[
                len("module.") :
            ]


        clean_state[key] = value


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
            "Could not locate probe weights."
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


        if weight_key.endswith("weight"):

            prefix = weight_key[
                :-len("weight")
            ]

        else:

            prefix = ""


        bias_key = prefix + "bias"


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


    return probe


probe = load_probe(
    PROBE_FILE
)


# ============================================================
# STEERING STATE
# ============================================================

steering_state = {

    "enabled": False,

    "alpha": 0.0,

    "delta": 0.0,

    "pre_values": [],

    "post_values": [],

    "num_hook_calls": 0,
}


# ============================================================
# STEERING HOOK
# ============================================================

def steering_hook(
    module,
    inputs,
):

    if not inputs:

        return inputs


    if not steering_state[
        "enabled"
    ]:

        return inputs


    x = inputs[0]


    if (
        NEURON_IDX
        >= x.shape[-1]
    ):

        raise RuntimeError(
            f"Neuron {NEURON_IDX} "
            f"outside MLP width {x.shape[-1]}"
        )


    delta = float(
        steering_state[
            "delta"
        ]
    )


    modified = x.clone()


    # Record activation BEFORE intervention.
    pre = (
        modified[
            ...,
            NEURON_IDX
        ]
        .detach()
        .float()
        .mean()
        .item()
    )


    modified[
        ...,
        NEURON_IDX
    ] = (
        modified[
            ...,
            NEURON_IDX
        ]
        + delta
    )


    # Record activation AFTER intervention.
    post = (
        modified[
            ...,
            NEURON_IDX
        ]
        .detach()
        .float()
        .mean()
        .item()
    )


    steering_state[
        "pre_values"
    ].append(pre)


    steering_state[
        "post_values"
    ].append(post)


    steering_state[
        "num_hook_calls"
    ] += 1


    return (
        modified,
        *inputs[1:],
    )


handle = (
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
# GREEDY GENERATION
# ============================================================

def generate_with_alpha(
    prompt,
    alpha,
    max_new_tokens,
):

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
    )


    prompt_ids = (
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


    generated = []


    steering_state[
        "alpha"
    ] = float(alpha)


    # --------------------------------------------------------
    # IMPORTANT
    #
    # Steering amount is RAW activation-space difference:
    #
    # delta = alpha * (mu_joy - mu_sad)
    # --------------------------------------------------------

    delta = (
        float(alpha)
        * RAW_DIRECTION
    )


    steering_state[
        "delta"
    ] = delta


    steering_state[
        "pre_values"
    ] = []


    steering_state[
        "post_values"
    ] = []


    steering_state[
        "num_hook_calls"
    ] = 0


    # ========================================================
    # PROMPT PASS
    #
    # No steering.
    #
    # This preserves normal prompt processing.
    # ========================================================

    steering_state[
        "enabled"
    ] = False


    with torch.inference_mode():

        output = model(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )


    past = output.past_key_values


    next_token = (
        output.logits[
            :,
            -1,
            :
        ]
        .argmax(
            dim=-1,
            keepdim=True,
        )
    )


    generated.append(
        int(
            next_token.item()
        )
    )


    if (
        tokenizer.eos_token_id
        is not None
        and next_token.item()
        == tokenizer.eos_token_id
    ):

        response = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return (
            response,
            generated,
            delta,
        )


    # ========================================================
    # AUTOREGRESSIVE DECODE
    #
    # Now steer every generated-token forward pass.
    #
    # Note:
    # the FIRST generated token comes from the unsteered
    # prompt representation.
    # Steering affects generation from token 2 onward.
    # ========================================================

    steering_state[
        "enabled"
    ] = (
        abs(alpha) > 0
    )


    current_token = next_token


    for _ in range(
        max_new_tokens - 1
    ):

        new_attention = torch.ones(
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
                new_attention,
            ],
            dim=1,
        )


        with torch.inference_mode():

            output = model(
                input_ids=current_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )


        past = output.past_key_values


        next_token = (
            output.logits[
                :,
                -1,
                :
            ]
            .argmax(
                dim=-1,
                keepdim=True,
            )
        )


        token_id = int(
            next_token.item()
        )


        generated.append(
            token_id
        )


        if (
            tokenizer.eos_token_id
            is not None
            and token_id
            == tokenizer.eos_token_id
        ):

            break


        current_token = next_token


    steering_state[
        "enabled"
    ] = False


    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )


    return (
        response,
        generated,
        delta,
    )


# ============================================================
# VALID TOKEN MASK FOR PROBE
# ============================================================

def valid_token_mask(
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

        mask &= input_ids.ne(
            special_id
        )


    return mask


# ============================================================
# SCORE GENERATED RESPONSE WITH FROZEN EMOTION PROBE
# ============================================================

def emotion_probe_score(
    text,
):

    steering_state[
        "enabled"
    ] = False


    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_SCORE_LENGTH,
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


    mask = valid_token_mask(
        input_ids,
        attention_mask,
    )


    with torch.inference_mode():

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


        float_mask = (
            mask
            .unsqueeze(-1)
            .to(hidden.dtype)
        )


        pooled = (
            hidden
            * float_mask
        ).sum(
            dim=1
        )


        denominator = (
            float_mask
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


        probabilities = torch.softmax(
            logits,
            dim=-1,
        )[0]


    result = {
        emotion:
            float(
                probabilities[i]
            )
        for i, emotion
        in enumerate(EMOTIONS)
    }


    result[
        "predicted_emotion"
    ] = EMOTIONS[
        int(
            probabilities.argmax()
        )
    ]


    return result


# ============================================================
# BASE-MODEL NLL / PERPLEXITY
# ============================================================

def calculate_response_nll(
    prompt,
    generated_ids,
):

    """
    Evaluate the generated continuation under the ORIGINAL,
    unsteered model.

    Lower NLL / perplexity generally means the continuation
    is less surprising to the base model.

    This is only a fluency-style proxy, not a complete
    measure of response quality.
    """

    steering_state[
        "enabled"
    ] = False


    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
    )[
        "input_ids"
    ][0].to(
        DEVICE
    )


    response_ids = torch.tensor(
        generated_ids,
        dtype=torch.long,
        device=DEVICE,
    )


    if len(
        response_ids
    ) == 0:

        return None, None


    full_ids = torch.cat(
        [
            prompt_ids,
            response_ids,
        ],
        dim=0,
    ).unsqueeze(0)


    attention_mask = torch.ones_like(
        full_ids
    )


    with torch.inference_mode():

        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )


    logits = outputs.logits[0]


    prompt_length = len(
        prompt_ids
    )


    # Position prompt_length-1 predicts first response token.
    prediction_logits = logits[
        prompt_length - 1 :
        prompt_length
        + len(response_ids)
        - 1
    ]


    log_probs = F.log_softmax(
        prediction_logits.float(),
        dim=-1,
    )


    token_log_probs = (
        log_probs.gather(
            dim=-1,
            index=response_ids.unsqueeze(-1),
        )
        .squeeze(-1)
    )


    nll = float(
        -token_log_probs.mean()
    )


    perplexity = float(
        math.exp(
            min(
                nll,
                20.0,
            )
        )
    )


    return (
        nll,
        perplexity,
    )


# ============================================================
# RUN ALPHA SWEEP
# ============================================================

results = []


print()
print("=" * 90)
print("ALPHA SWEEP")
print("=" * 90)

print()
print("PROMPT:")
print(args.prompt)


try:

    for alpha in args.alphas:

        print()
        print("=" * 90)
        print(
            f"ALPHA = {alpha:+.2f}"
        )
        print("=" * 90)


        response, generated_ids, delta = (
            generate_with_alpha(
                prompt=args.prompt,
                alpha=alpha,
                max_new_tokens=
                    args.max_new_tokens,
            )
        )


        # ----------------------------------------------------
        # Hook statistics
        # ----------------------------------------------------

        pre_values = (
            steering_state[
                "pre_values"
            ]
        )


        post_values = (
            steering_state[
                "post_values"
            ]
        )


        mean_pre = (
            float(
                np.mean(
                    pre_values
                )
            )
            if pre_values
            else None
        )


        mean_post = (
            float(
                np.mean(
                    post_values
                )
            )
            if post_values
            else None
        )


        # ----------------------------------------------------
        # Residual-vector displacement
        #
        # ||Delta_h||
        # =
        # |alpha * raw_direction| * ||W_down[:,u]||
        # ----------------------------------------------------

        residual_delta_norm = (
            abs(delta)
            * W_DIRECTION_NORM
        )


        signed_residual_scale = (
            delta
            * W_DIRECTION_NORM
        )


        # ----------------------------------------------------
        # Emotion score
        # ----------------------------------------------------

        emotion_scores = (
            emotion_probe_score(
                response
            )
        )


        # ----------------------------------------------------
        # Base model quality proxy
        # ----------------------------------------------------

        nll, perplexity = (
            calculate_response_nll(
                args.prompt,
                generated_ids,
            )
        )


        # ----------------------------------------------------
        # Simple repetition metric
        # ----------------------------------------------------

        if generated_ids:

            unique_ratio = (
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

            unique_ratio = 0.0


        joy_probability = (
            emotion_scores[
                "joy"
            ]
        )


        sadness_probability = (
            emotion_scores[
                "sadness"
            ]
        )


        joy_sad_margin = (
            joy_probability
            - sadness_probability
        )


        row = {

            "alpha":
                float(
                    alpha
                ),

            "activation_delta":
                float(
                    delta
                ),

            "residual_delta_norm":
                float(
                    residual_delta_norm
                ),

            "signed_residual_scale":
                float(
                    signed_residual_scale
                ),

            "hook_calls":
                int(
                    steering_state[
                        "num_hook_calls"
                    ]
                ),

            "mean_pre_activation":
                mean_pre,

            "mean_post_activation":
                mean_post,

            "generated_tokens":
                len(
                    generated_ids
                ),

            "unique_token_ratio":
                float(
                    unique_ratio
                ),

            "base_model_nll":
                nll,

            "base_model_perplexity":
                perplexity,

            "predicted_emotion":
                emotion_scores[
                    "predicted_emotion"
                ],

            "joy_probability":
                joy_probability,

            "sadness_probability":
                sadness_probability,

            "joy_minus_sadness_probability":
                joy_sad_margin,

            "response":
                response,
        }


        for emotion in EMOTIONS:

            row[
                f"prob_{emotion}"
            ] = emotion_scores[
                emotion
            ]


        results.append(
            row
        )


        print(
            f"Activation delta     : "
            f"{delta:+.8f}"
        )

        print(
            f"Residual ||Delta h|| : "
            f"{residual_delta_norm:.8f}"
        )


        if mean_pre is not None:

            print(
                f"Mean m before        : "
                f"{mean_pre:+.8f}"
            )

            print(
                f"Mean m after         : "
                f"{mean_post:+.8f}"
            )


        print(
            f"Joy probability      : "
            f"{joy_probability:.6f}"
        )

        print(
            f"Sadness probability  : "
            f"{sadness_probability:.6f}"
        )

        print(
            f"Joy-Sad margin       : "
            f"{joy_sad_margin:+.6f}"
        )

        print(
            f"Predicted emotion    : "
            f"{emotion_scores['predicted_emotion']}"
        )

        print(
            f"Base-model PPL       : "
            f"{perplexity}"
        )

        print()
        print("RESPONSE:")
        print(response)


finally:

    handle.remove()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


csv_file = (
    OUTPUT_DIR
    / "single_joy_neuron_alpha_sweep.csv"
)


results_df.to_csv(
    csv_file,
    index=False,
)


json_file = (
    OUTPUT_DIR
    / "single_joy_neuron_alpha_sweep.json"
)


output_json = {

    "experiment":
        (
            "Single-neuron sadness-to-joy "
            "activation steering"
        ),

    "prompt":
        args.prompt,

    "neuron":
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

            "raw_direction":
                RAW_DIRECTION,

            "pooled_sd":
                POOLED_SD,

            "standardized_effect":
                STANDARDIZED_EFFECT,

            "residual_direction_norm":
                W_DIRECTION_NORM,
        },

    "steering_rule":
        (
            "m' = m + alpha * "
            "(mean_joy - mean_sadness)"
        ),

    "prompt_forward_pass_steered":
        False,

    "first_generated_token_steered":
        False,

    "subsequent_generated_tokens_steered":
        True,

    "results":
        results,
}


with json_file.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output_json,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 90)
print("FINAL STEERING SUMMARY")
print("=" * 90)


summary_columns = [

    "alpha",

    "activation_delta",

    "residual_delta_norm",

    "mean_pre_activation",

    "mean_post_activation",

    "joy_probability",

    "sadness_probability",

    "joy_minus_sadness_probability",

    "predicted_emotion",

    "base_model_perplexity",

    "unique_token_ratio",
]


print(
    results_df[
        summary_columns
    ].to_string(
        index=False
    )
)


print()
print("Saved:")
print(csv_file)
print(json_file)

print()
print("Done.")
