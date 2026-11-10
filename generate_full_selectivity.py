#!/usr/bin/env python3

"""
Generate the complete Llama 3.1 8B emotion-neuron selectivity tensor.

Output:
    results/llama31_8b_emotions_full/selectivity.npy

Expected selectivity shape:
    (32, 14336, 10)

Axis order:
    [layer, neuron, emotion]

Neuron activation definition:
    1[ SwiGLU MLP intermediate activation > 0 ]

The SwiGLU intermediate is captured at the INPUT of mlp.down_proj.

Dataset:
    data/mlt_hinghlish_dataset_clean.csv

Text column:
    hinglish_genz_text

Label column:
    label
"""

from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "mlt_hinghlish_dataset_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "llama31_8b_emotions_full"
)


# ============================================================
# MODEL / DATA CONFIG
# ============================================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

TEXT_COLUMN = "hinglish_genz_text"
LABEL_COLUMN = "label"

MAX_LENGTH = 256

# Start conservatively.
# RTX 6000 Ada 48 GB should normally tolerate this.
BATCH_SIZE = 4

EPS = 1e-12

EXPECTED_NUM_LAYERS = 32
EXPECTED_MLP_WIDTH = 14336

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

NUM_EMOTIONS = len(EMOTIONS)

LABEL_TO_ID = {
    emotion: i
    for i, emotion in enumerate(EMOTIONS)
}


# ============================================================
# BASIC CHECKS
# ============================================================

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_FILE}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD + CLEAN DATASET
# ============================================================

print("=" * 80)
print("LOADING EMOTION DATASET")
print("=" * 80)

print(f"Dataset: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required_columns = {
    TEXT_COLUMN,
    LABEL_COLUMN,
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset missing columns: {missing_columns}"
    )


# Keep only required columns.
df = df[
    [
        TEXT_COLUMN,
        LABEL_COLUMN,
    ]
].copy()


# Remove missing values.
df = df.dropna(
    subset=[
        TEXT_COLUMN,
        LABEL_COLUMN,
    ]
)


# Normalize strings.
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


# Remove empty text.
df = df[
    df[TEXT_COLUMN].str.len() > 0
]


# ============================================================
# IMPORTANT:
# Same deduplication used in the previous pipeline.
# ============================================================

before_dedup = len(df)

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)

after_dedup = len(df)


print()
print(f"Rows before dedup : {before_dedup:,}")
print(f"Rows after dedup  : {after_dedup:,}")


# Previous experiment should produce 25,508.
if after_dedup != 25508:
    print()
    print(
        "WARNING: previous experiment used 25,508 "
        f"samples, but current preprocessing produced "
        f"{after_dedup:,}."
    )


# Validate labels.
unknown_labels = (
    set(df[LABEL_COLUMN].unique())
    - set(EMOTIONS)
)

if unknown_labels:
    raise ValueError(
        f"Unknown emotion labels found: {unknown_labels}"
    )


print()
print("Class distribution:")

for emotion in EMOTIONS:
    n = int(
        (df[LABEL_COLUMN] == emotion).sum()
    )

    print(
        f"{emotion:12s}: {n:,}"
    )


texts = df[TEXT_COLUMN].tolist()

label_ids = np.asarray(
    [
        LABEL_TO_ID[label]
        for label in df[LABEL_COLUMN]
    ],
    dtype=np.int64,
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print()
print("=" * 80)
print("LOADING TOKENIZER")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)


# Some causal LMs do not define a padding token.
if tokenizer.pad_token_id is None:

    if tokenizer.eos_token_id is not None:

        tokenizer.pad_token = tokenizer.eos_token

    else:

        raise ValueError(
            "Tokenizer has neither pad_token_id "
            "nor eos_token_id."
        )


special_ids = sorted(
    set(tokenizer.all_special_ids)
)

print(
    f"Special token IDs excluded from counting: "
    f"{special_ids}"
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print()
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    gpu_gib = (
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3
    )

    print(
        f"GPU memory: {gpu_gib:.2f} GiB"
    )

else:

    device = torch.device("cpu")
    dtype = torch.float32

    print()
    print(
        "WARNING: CUDA unavailable. "
        "This experiment will be very slow on CPU."
    )


print(
    f"Model dtype: {dtype}"
)


# ============================================================
# LOAD FROZEN SARVAM-1
# ============================================================

print()
print("=" * 80)
print("LOADING SARVAM-1")
print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    trust_remote_code=True,
)

model.to(device)

model.eval()

for param in model.parameters():
    param.requires_grad_(False)


# ============================================================
# LOCATE TRANSFORMER LAYERS
# ============================================================

def get_transformer_layers(model):
    """
    Resolve the layer container for common HF decoder models.
    """

    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("transformer.h", lambda m: m.transformer.h),
    ]

    for name, getter in candidates:

        try:
            layers = getter(model)

            if layers is not None:
                print(
                    f"Transformer layers found at: {name}"
                )

                return layers

        except (AttributeError, TypeError):
            pass

    raise RuntimeError(
        "Could not locate transformer layers. "
        "Expected something like model.model.layers."
    )


layers = get_transformer_layers(model)

num_layers = len(layers)

print(
    f"Number of transformer layers: {num_layers}"
)

if num_layers != EXPECTED_NUM_LAYERS:

    raise ValueError(
        f"Expected {EXPECTED_NUM_LAYERS} layers, "
        f"found {num_layers}."
    )


# ============================================================
# VALIDATE MLP STRUCTURE
# ============================================================

for layer_idx, layer in enumerate(layers):

    if not hasattr(layer, "mlp"):
        raise RuntimeError(
            f"Layer {layer_idx} does not contain .mlp"
        )

    if not hasattr(layer.mlp, "down_proj"):
        raise RuntimeError(
            f"Layer {layer_idx}.mlp does not contain "
            f"down_proj"
        )


first_down_proj = layers[0].mlp.down_proj

if not hasattr(first_down_proj, "in_features"):

    raise RuntimeError(
        "Could not determine down_proj input dimension."
    )


mlp_width = int(
    first_down_proj.in_features
)

print(
    f"MLP intermediate width: {mlp_width}"
)

if mlp_width != EXPECTED_MLP_WIDTH:

    raise ValueError(
        f"Expected MLP width {EXPECTED_MLP_WIDTH}, "
        f"found {mlp_width}."
    )


# ============================================================
# DETERMINE BASE TRANSFORMER
# ============================================================

# Avoid computing the huge vocabulary logits if possible.
#
# Sarvam/Llama-style architecture generally uses model.model
# as the base decoder.

if hasattr(model, "model"):

    forward_model = model.model

    print(
        "Using base transformer forward: model.model"
    )

else:

    forward_model = model

    print(
        "Using full model forward."
    )


# ============================================================
# COUNTERS
# ============================================================
#
# C[layer, neuron, emotion]
#
# Number of valid token positions where:
#
#     MLP activation > 0
#
# Shape:
#     (28, 11008, 10)
# ============================================================

active_counts = torch.zeros(
    (
        num_layers,
        mlp_width,
        NUM_EMOTIONS,
    ),
    dtype=torch.int64,
    device=device,
)


# T[emotion]
#
# Total number of valid non-special token positions belonging
# to each emotion.
total_positions = torch.zeros(
    NUM_EMOTIONS,
    dtype=torch.int64,
    device=device,
)


# ============================================================
# HOOK CONTEXT
# ============================================================

hook_context = {
    "valid_mask": None,
    "emotion_ids": None,
}


# ============================================================
# FORWARD PRE-HOOK
# ============================================================

def make_down_proj_hook(layer_idx):

    def hook(module, inputs):

        if not inputs:
            raise RuntimeError(
                f"No input received by down_proj "
                f"for layer {layer_idx}"
            )

        # SwiGLU MLP intermediate:
        #
        # shape:
        #   [batch, sequence, 11008]
        #
        mlp_activation = inputs[0]

        if mlp_activation.ndim != 3:

            raise RuntimeError(
                f"Expected down_proj input to have 3 dims, "
                f"got {mlp_activation.shape}"
            )

        if mlp_activation.shape[-1] != mlp_width:

            raise RuntimeError(
                f"Layer {layer_idx}: expected width "
                f"{mlp_width}, got "
                f"{mlp_activation.shape[-1]}"
            )

        valid_mask = hook_context[
            "valid_mask"
        ]

        emotion_ids = hook_context[
            "emotion_ids"
        ]

        if valid_mask is None or emotion_ids is None:

            raise RuntimeError(
                "Hook called without current batch context."
            )


        # ----------------------------------------------------
        # Binary firing definition from original experiment:
        #
        #     1[m > 0]
        #
        # ----------------------------------------------------

        positive = (
            mlp_activation > 0
        )


        # Flatten batch and sequence:
        #
        # positive_flat:
        #     [(B*T), 11008]
        #
        positive_flat = positive.reshape(
            -1,
            mlp_width
        )

        valid_flat = valid_mask.reshape(-1)


        # Emotion ID for every token position.
        #
        # Example:
        #
        # sample labels = [fear, joy]
        #
        # each token from sample 0 gets fear ID,
        # each token from sample 1 gets joy ID.
        #
        token_emotions = (
            emotion_ids[:, None]
            .expand(
                -1,
                valid_mask.shape[1]
            )
            .reshape(-1)
        )


        # Only valid content tokens.
        valid_positive = positive_flat[
            valid_flat
        ]

        valid_emotions = token_emotions[
            valid_flat
        ]


        # ----------------------------------------------------
        # Update counts separately for every emotion present
        # in this batch.
        # ----------------------------------------------------

        unique_emotions = torch.unique(
            valid_emotions
        )

        for emotion_id_tensor in unique_emotions:

            emotion_id = int(
                emotion_id_tensor.item()
            )

            emotion_mask = (
                valid_emotions
                == emotion_id
            )

            # Sum over valid token positions.
            #
            # result shape:
            #     [11008]
            #
            counts = (
                valid_positive[
                    emotion_mask
                ]
                .sum(
                    dim=0,
                    dtype=torch.int64
                )
            )

            active_counts[
                layer_idx,
                :,
                emotion_id
            ] += counts

    return hook


# ============================================================
# REGISTER HOOKS
# ============================================================

print()
print("=" * 80)
print("REGISTERING MLP HOOKS")
print("=" * 80)

hook_handles = []

for layer_idx, layer in enumerate(layers):

    handle = (
        layer
        .mlp
        .down_proj
        .register_forward_pre_hook(
            make_down_proj_hook(
                layer_idx
            )
        )
    )

    hook_handles.append(handle)

print(
    f"Registered {len(hook_handles)} hooks."
)


# ============================================================
# SPECIAL TOKEN TENSOR
# ============================================================

special_ids_tensor = torch.tensor(
    special_ids,
    dtype=torch.long,
    device=device,
)


# ============================================================
# PROCESS DATASET
# ============================================================

print()
print("=" * 80)
print("EXTRACTING ACTIVATION STATISTICS")
print("=" * 80)

num_samples = len(texts)

num_batches = math.ceil(
    num_samples / BATCH_SIZE
)

start_time = time.time()


try:

    with torch.inference_mode():

        for batch_idx in range(num_batches):

            start = (
                batch_idx
                * BATCH_SIZE
            )

            end = min(
                start + BATCH_SIZE,
                num_samples
            )


            batch_texts = texts[
                start:end
            ]

            batch_labels_np = label_ids[
                start:end
            ]


            # =================================================
            # TOKENIZE
            # =================================================

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )


            input_ids = encoded[
                "input_ids"
            ].to(device)

            attention_mask = encoded[
                "attention_mask"
            ].to(device)


            batch_emotion_ids = torch.tensor(
                batch_labels_np,
                dtype=torch.long,
                device=device,
            )


            # =================================================
            # VALID TOKEN MASK
            # =================================================
            #
            # Start with normal attention mask.
            #
            valid_mask = (
                attention_mask.bool()
            )


            # Remove ALL tokenizer special tokens:
            #
            # PAD / BOS / EOS / etc.
            #
            if special_ids_tensor.numel() > 0:

                special_mask = torch.isin(
                    input_ids,
                    special_ids_tensor
                )

                valid_mask = (
                    valid_mask
                    & ~special_mask
                )


            # =================================================
            # COUNT TOTAL VALID POSITIONS PER EMOTION
            # =================================================

            valid_tokens_per_sample = (
                valid_mask
                .sum(dim=1)
                .to(torch.int64)
            )

            total_positions.scatter_add_(
                dim=0,
                index=batch_emotion_ids,
                src=valid_tokens_per_sample,
            )


            # =================================================
            # SET HOOK CONTEXT
            # =================================================

            hook_context[
                "valid_mask"
            ] = valid_mask

            hook_context[
                "emotion_ids"
            ] = batch_emotion_ids


            # =================================================
            # MODEL FORWARD
            # =================================================

            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }


            _ = forward_model(
                **model_inputs
            )


            # =================================================
            # PROGRESS
            # =================================================

            if (
                batch_idx == 0
                or (batch_idx + 1) % 50 == 0
                or batch_idx + 1 == num_batches
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                processed = end

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1:5d}/"
                    f"{num_batches:5d}"
                    f" | samples "
                    f"{processed:6d}/"
                    f"{num_samples:6d}"
                    f" | {rate:.2f} samples/s"
                )


finally:

    # Always remove hooks.
    for handle in hook_handles:
        handle.remove()


    hook_context[
        "valid_mask"
    ] = None

    hook_context[
        "emotion_ids"
    ] = None


# ============================================================
# MOVE COUNTS TO CPU
# ============================================================

print()
print("=" * 80)
print("COMPUTING SELECTIVITY")
print("=" * 80)


C = (
    active_counts
    .cpu()
    .numpy()
)

T = (
    total_positions
    .cpu()
    .numpy()
)


print(
    f"Active-count shape : {C.shape}"
)

print(
    f"Total-position shape: {T.shape}"
)


print()
print("Valid token positions per emotion:")

for emotion_idx, emotion in enumerate(
    EMOTIONS
):

    print(
        f"{emotion:12s}: "
        f"{int(T[emotion_idx]):,}"
    )


# ============================================================
# ACTIVATION PROBABILITY
# ============================================================

# Use float64 for the statistical computation.
C_float = C.astype(
    np.float64,
    copy=False
)

T_float = T.astype(
    np.float64,
    copy=False
)


P = (
    C_float
    /
    (
        T_float[
            None,
            None,
            :
        ]
        + EPS
    )
)


# ============================================================
# NORMALIZE ACROSS EMOTIONS
# ============================================================

emotion_sum = P.sum(
    axis=2,
    keepdims=True
)

P_normalized = (
    P
    /
    (
        emotion_sum
        + EPS
    )
)


# ============================================================
# NORMALIZED ENTROPY
# ============================================================

H = -(
    P_normalized
    * np.log(
        P_normalized
        + EPS
    )
).sum(
    axis=2
)

H = (
    H
    / np.log(NUM_EMOTIONS)
)


# ============================================================
# SELECTIVITY
# ============================================================

S = (
    P
    * (
        1.0
        - H[:, :, None]
    )
)


# ============================================================
# SANITY CHECKS
# ============================================================

expected_shape = (
    EXPECTED_NUM_LAYERS,
    EXPECTED_MLP_WIDTH,
    NUM_EMOTIONS,
)

if S.shape != expected_shape:

    raise RuntimeError(
        f"Wrong selectivity shape: {S.shape}. "
        f"Expected {expected_shape}."
    )


if not np.all(
    np.isfinite(S)
):

    raise RuntimeError(
        "NaN or Inf detected in selectivity tensor."
    )


print()
print(
    f"Selectivity shape: {S.shape}"
)

print(
    f"Selectivity min  : {S.min():.12f}"
)

print(
    f"Selectivity max  : {S.max():.12f}"
)


# ============================================================
# SAVE ALL FULL STATISTICS
# ============================================================

print()
print("=" * 80)
print("SAVING RESULTS")
print("=" * 80)


np.save(
    OUTPUT_DIR
    / "active_counts.npy",
    C,
)


np.save(
    OUTPUT_DIR
    / "total_positions.npy",
    T,
)


np.save(
    OUTPUT_DIR
    / "activation_probability.npy",
    P,
)


np.save(
    OUTPUT_DIR
    / "normalized_probability.npy",
    P_normalized,
)


np.save(
    OUTPUT_DIR
    / "entropy.npy",
    H,
)


# ============================================================
# THIS IS THE FILE NEEDED BY THE NEXT SCRIPT
# ============================================================

selectivity_file = (
    OUTPUT_DIR
    / "selectivity.npy"
)

np.save(
    selectivity_file,
    S,
)


# ============================================================
# METADATA
# ============================================================

metadata = {
    "model_id": MODEL_ID,

    "dataset": str(DATA_FILE),

    "num_samples": int(
        num_samples
    ),

    "num_layers": int(
        num_layers
    ),

    "neurons_per_layer": int(
        mlp_width
    ),

    "num_emotions": int(
        NUM_EMOTIONS
    ),

    "labels": EMOTIONS,

    "text_column": TEXT_COLUMN,

    "label_column": LABEL_COLUMN,

    "max_sequence_length": MAX_LENGTH,

    "batch_size": BATCH_SIZE,

    "activation_definition": (
        "1[MLP activation > 0.0]"
    ),

    "activation_location": (
        "input to layer.mlp.down_proj "
        "(SwiGLU intermediate)"
    ),

    "special_tokens_excluded": (
        special_ids
    ),

    "selectivity_formula": (
        "S[l,u,e] = P[l,u,e] * "
        "(1 - H[l,u])"
    ),

    "selectivity_shape": list(
        S.shape
    ),

    "axis_order": [
        "layer",
        "neuron",
        "emotion",
    ],

    "epsilon": EPS,
}


metadata_file = (
    OUTPUT_DIR
    / "metadata.json"
)

with metadata_file.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)

print(
    f"Samples processed     : "
    f"{num_samples:,}"
)

print(
    f"Layers                : "
    f"{num_layers}"
)

print(
    f"MLP neurons/layer     : "
    f"{mlp_width:,}"
)

print(
    f"Emotion classes       : "
    f"{NUM_EMOTIONS}"
)

print(
    f"Selectivity values    : "
    f"{S.size:,}"
)

print(
    f"Selectivity shape     : "
    f"{S.shape}"
)

print(
    f"Elapsed time          : "
    f"{elapsed / 60:.2f} min"
)

print()
print(
    "Required output:"
)

print(
    selectivity_file
)

print()
print(
    "Expected:"
)

print(
    "/data/mtech/surya_project/mental_health/"
    "results/sarvam1_emotions_full/selectivity.npy"
)

print()
print("Done.")
