#!/usr/bin/env python3
"""Extract top emotion-selective MLP neurons from a causal language model.

For every layer and emotion, this script ranks neurons by
S[l,u,e] = P[l,u,e] * (1 - H[l,u]) and writes only the selected neuron IDs.
It does not save activation counts, probabilities, entropy, or full scores.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


DEFAULT_PERCENTAGES = (0.5, 1.0, 2.0, 5.0, 10.0)


def prepare_dataframe(df: pd.DataFrame, text_column: str, label_column: str | None,
                      max_samples: int | None) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    if text_column not in df:
        raise ValueError(f"Missing text column: {text_column}")
    if label_column is None:
        label_column = next((name for name in ("emotion", "label", "labels") if name in df), None)
    if label_column is None:
        raise ValueError("Could not find an emotion label column; use --label-column.")
    data = df[[text_column, label_column]].rename(columns={text_column: "text", label_column: "emotion"}).copy()
    data["text"] = data["text"].astype("string").str.strip()
    data["emotion"] = data["emotion"].astype("string").str.strip()
    data = data.dropna().loc[(data.text != "") & (data.emotion != "")]
    data = data.drop_duplicates(subset=["text", "emotion"], keep="first").reset_index(drop=True)
    if max_samples is not None:
        data = data.iloc[:max_samples].reset_index(drop=True)
    if data.empty:
        raise ValueError("No non-empty rows remain after cleaning.")
    labels = sorted(data.emotion.unique().tolist())
    return data, labels, {label: index for index, label in enumerate(labels)}


def get_model_layers(model: Any) -> list[Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise AttributeError("Could not find model.model.layers; adapt get_model_layers for this checkpoint.")
    if any(not hasattr(getattr(layer, "mlp", None), "down_proj") for layer in layers):
        raise AttributeError("Each transformer layer must expose mlp.down_proj.")
    return list(layers)


def valid_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in set(getattr(tokenizer, "all_special_ids", [])):
        mask &= ~input_ids.eq(token_id)
    return mask


class ActivationCounter:
    def __init__(self, layers: list[Any], num_emotions: int, threshold: float):
        width = layers[0].mlp.down_proj.in_features
        # With device_map="auto", offloaded layers expose a meta-device weight.
        # Statistics must therefore live on CPU rather than beside the weight.
        self.counts = [torch.zeros((width, num_emotions), dtype=torch.int64) for _ in layers]
        self.positions = torch.zeros(num_emotions, dtype=torch.int64)
        self.threshold = threshold

    def update(self, activations: dict[int, torch.Tensor], mask: torch.Tensor,
               label_ids: torch.Tensor) -> None:
        weights = mask.sum(dim=1).cpu().to(torch.float32)
        self.positions += torch.bincount(label_ids.cpu(), weights=weights,
                                         minlength=self.positions.numel()).to(torch.int64)
        for layer_index, activation in activations.items():
            fired = (activation > self.threshold) & mask.to(activation.device).unsqueeze(-1)
            per_example = fired.sum(dim=1).to(torch.int64)
            for emotion_id in label_ids.unique().tolist():
                selected = label_ids.eq(emotion_id).to(per_example.device)
                self.counts[layer_index][:, emotion_id] += (
                    per_example[selected].sum(dim=0).cpu()
                )


def select_top_neurons(counts: list[torch.Tensor], positions: torch.Tensor, labels: list[str],
                       percentages: list[float]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Compute S in memory and retain only top neuron IDs per layer and emotion."""
    if (positions <= 0).any():
        raise ValueError("Every emotion must have at least one valid token position.")
    selections: dict[str, dict[str, dict[str, list[int]]]] = {}
    for layer_index, layer_counts in enumerate(counts):
        totals = positions.to(dtype=torch.float64)
        probability = layer_counts.to(torch.float64) / (totals.unsqueeze(0) + 1e-12)
        distribution = probability / (probability.sum(dim=1, keepdim=True) + 1e-12)
        entropy = -(distribution * torch.log(distribution + 1e-12)).sum(dim=1) / math.log(len(labels))
        selectivity = probability * (1.0 - entropy.unsqueeze(1))
        for percentage in percentages:
            key = f"{percentage:g}%"
            selections.setdefault(key, {})[f"layer_{layer_index}"] = {}
            # Match the reference extraction: retain int(percentage * layer width),
            # with at least one neuron for very small percentages.
            count = max(1, min(selectivity.shape[0], int(selectivity.shape[0] * percentage / 100.0)))
            for emotion_id, emotion in enumerate(labels):
                order = torch.argsort(selectivity[:, emotion_id], descending=True, stable=True)
                selections[key][f"layer_{layer_index}"][emotion] = order[:count].cpu().tolist()
    return selections


def select_from_saved_selectivity(selectivity: np.ndarray, labels: list[str],
                                  percentages: list[float]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Select IDs from the full tensor already created by the preceding stage."""
    if selectivity.ndim != 3 or selectivity.shape[2] != len(labels):
        raise ValueError(
            "Selectivity tensor must have shape [layer, neuron, emotion] with "
            f"{len(labels)} emotions; received {selectivity.shape}."
        )
    selections: dict[str, dict[str, dict[str, list[int]]]] = {}
    for layer_index in range(selectivity.shape[0]):
        for percentage in percentages:
            key = f"{percentage:g}%"
            by_emotion = selections.setdefault(key, {}).setdefault(f"layer_{layer_index}", {})
            count = max(1, min(selectivity.shape[1], int(selectivity.shape[1] * percentage / 100.0)))
            for emotion_id, emotion in enumerate(labels):
                order = np.argsort(-selectivity[layer_index, :, emotion_id], kind="stable")
                by_emotion[emotion] = order[:count].tolist()
    return selections


def write_selection_output(output_file: str, model_id: str, labels: list[str], num_layers: int,
                           width: int, num_samples: int, max_length: int, threshold: float,
                           percentages: list[float], selected: dict[str, Any]) -> None:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "model_id": model_id,
        "labels": labels,
        "num_layers": num_layers,
        "neurons_per_layer": width,
        "num_samples": num_samples,
        "max_sequence_length": max_length,
        "activation_definition": f"1[MLP activation > {threshold}]",
        "selection_score": "S[l,u,e] = P[l,u,e] * (1 - H[l,u])",
        "selection_method": "top percentage within each layer, independently for each emotion",
        "percentages": percentages,
        "neurons_per_layer_per_emotion": {f"{value:g}%": max(1, int(width * value / 100.0))
                                           for value in percentages},
        "selected_neurons": selected,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved selected neuron IDs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--csv-path", default="data/mlt_hinghlish_dataset_clean.csv")
    parser.add_argument("--text-column", default="hinglish_genz_text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--output-file", default="results/llama31_8b_neuron_analysis/top_selectivity_neurons.json")
    parser.add_argument("--percentages", type=float, nargs="+", default=DEFAULT_PERCENTAGES)
    # Llama's intermediate activations are 14,336-wide.  Four examples keep
    # the 32 hooked MLP tensors comfortably below the 48 GiB GPU budget.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument(
        "--selectivity-file",
        help="Reuse a saved [layer, neuron, emotion] selectivity .npy tensor instead of rerunning hooks.",
    )
    args = parser.parse_args()
    if not args.percentages or any(not 0 < value <= 100 for value in args.percentages):
        parser.error("--percentages values must be in (0, 100]")

    data, labels, label_to_id = prepare_dataframe(pd.read_csv(args.csv_path), args.text_column,
                                                   args.label_column, args.max_samples)
    percentages = sorted(set(args.percentages))
    if args.selectivity_file:
        selectivity_path = Path(args.selectivity_file)
        if not selectivity_path.is_file():
            raise FileNotFoundError(f"Selectivity tensor not found: {selectivity_path}")
        selectivity = np.load(selectivity_path, mmap_mode="r")
        selected = select_from_saved_selectivity(selectivity, labels, percentages)
        write_selection_output(args.output_file, args.model_id, labels, selectivity.shape[0],
                               selectivity.shape[1], len(data), args.max_length,
                               args.activation_threshold, percentages, selected)
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, trust_remote_code=True, device_map="auto", dtype=dtype
    )
    model.eval(); model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = get_model_layers(model)
    # We only need MLP intermediate activations. Calling the causal-LM wrapper
    # also builds [batch, sequence, vocab] logits (128,256 vocabulary items
    # for Llama 3.1), which can consume multiple GiB and causes an avoidable
    # CUDA OOM. model.model executes the identical decoder layers and hooks.
    backbone = model.model
    counter = ActivationCounter(layers, len(labels), args.activation_threshold)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(layers):
        def hook(module: Any, inputs: tuple[Any, ...], layer_index: int = index) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"Layer {layer_index} down_proj hook received no tensor input.")
            captured[layer_index] = inputs[0].detach()
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        for start in tqdm(range(0, len(data), args.batch_size), desc="Counting neuron firing"):
            batch = data.iloc[start:start + args.batch_size]
            encoded = tokenizer(batch.text.tolist(), return_tensors="pt", padding=True,
                                truncation=True, max_length=args.max_length, add_special_tokens=True)
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            mask = valid_token_mask(encoded["input_ids"], encoded["attention_mask"], tokenizer)
            label_ids = torch.tensor([label_to_id[label] for label in batch.emotion], dtype=torch.long)
            captured.clear()
            with torch.inference_mode():
                backbone(**encoded, use_cache=False)
            if len(captured) != len(layers):
                raise RuntimeError(f"Captured {len(captured)} MLP inputs for {len(layers)} layers.")
            counter.update(captured, mask, label_ids)
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()

    selected = select_top_neurons(counter.counts, counter.positions, labels, percentages)
    width = counter.counts[0].shape[0]
    write_selection_output(args.output_file, args.model_id, labels, len(layers), width, len(data),
                           args.max_length, args.activation_threshold, percentages, selected)


if __name__ == "__main__":
    main()
