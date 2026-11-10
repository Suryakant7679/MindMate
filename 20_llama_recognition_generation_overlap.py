#!/usr/bin/env python3
"""Test whether Llama MLP neurons are reused for emotion recognition and response generation.

R is derived from prompt/prefill activations and G from decoder-token
activations.  For both, each neuron's score is a target-vs-baseline pooled
z-score, not a binary activation count.  The experiment then masks R∩G only
during decoding and compares the target-emotion probability of generated text
against an equal-size, layer-matched random-neuron control.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_CSV = "data/mlt_hinghlish_dataset_clean.csv"
DEFAULT_PROBE = "results/llama31_8b_neuron_analysis/04_probe/probe.pt"
DEFAULT_OUTPUT = "results/llama31_8b_recognition_generation_overlap"


def valid_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokenizer: Any) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for token_id in tokenizer.all_special_ids:
        mask &= input_ids.ne(token_id)
    return mask


class Moments:
    """CPU moments for every [layer, MLP neuron], safe with device_map offload."""
    def __init__(self, widths: list[int]) -> None:
        self.total = [torch.zeros(width, dtype=torch.float64) for width in widths]
        self.squared = [torch.zeros(width, dtype=torch.float64) for width in widths]
        self.positions = 0

    def add(self, layer_index: int, activations: torch.Tensor, mask: torch.Tensor) -> None:
        values = activations[mask.to(activations.device)].detach().float().cpu()
        if values.numel() == 0:
            return
        self.total[layer_index] += values.sum(dim=0, dtype=torch.float64)
        self.squared[layer_index] += values.square().sum(dim=0, dtype=torch.float64)
        self.positions += int(values.shape[0]) if layer_index == 0 else 0

    def mean_and_variance(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.positions < 2:
            raise ValueError("Not enough valid positions to compute a standardized score.")
        mean = self.total[layer_index] / self.positions
        variance = (self.squared[layer_index] / self.positions - mean.square()).clamp_min(1e-12)
        return mean, variance


class MLPController:
    def __init__(self, layers: list[Any]) -> None:
        self.prompt_mask: torch.Tensor | None = None
        self.prompt_moments: Moments | None = None
        self.generated_moments: Moments | None = None
        self.mask_by_layer: dict[int, torch.Tensor] = {}
        self.record = False
        self.mask_generated = False
        self.handles = []
        for index, layer in enumerate(layers):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._hook(index)))

    def _hook(self, layer_index: int):
        def hook(_module: Any, inputs: tuple[Any, ...]):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"Layer {layer_index} down_proj hook did not receive activations.")
            activation = inputs[0]
            # generate() feeds the complete prompt once, then one decoder token at a time.
            is_prompt = self.prompt_mask is not None and activation.shape[1] == self.prompt_mask.shape[1]
            if self.record:
                if is_prompt and self.prompt_moments is not None:
                    self.prompt_moments.add(layer_index, activation, self.prompt_mask)
                elif not is_prompt and self.generated_moments is not None:
                    self.generated_moments.add(
                        layer_index, activation,
                        torch.ones(activation.shape[:2], dtype=torch.bool, device=activation.device),
                    )
            if self.mask_generated and not is_prompt and layer_index in self.mask_by_layer:
                masked = activation.clone()
                masked[..., self.mask_by_layer[layer_index].to(masked.device)] = 0
                return (masked, *inputs[1:])
            return None
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def z_scores(target: Moments, baseline: Moments, widths: list[int]) -> list[torch.Tensor]:
    scores = []
    for index, width in enumerate(widths):
        target_mean, target_var = target.mean_and_variance(index)
        baseline_mean, baseline_var = baseline.mean_and_variance(index)
        scores.append((target_mean - baseline_mean) / torch.sqrt((target_var + baseline_var) / 2 + 1e-12))
        if scores[-1].numel() != width:
            raise RuntimeError("Unexpected MLP score width.")
    return scores


def top_sets(scores: list[torch.Tensor], percentage: float) -> list[set[int]]:
    output = []
    for score in scores:
        count = max(1, int(score.numel() * percentage / 100.0))
        output.append(set(torch.argsort(score, descending=True, stable=True)[:count].tolist()))
    return output


def to_rows(recognition: list[torch.Tensor], generation: list[torch.Tensor], r_sets: list[set[int]],
            g_sets: list[set[int]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary, neurons = [], []
    for layer_index, (r_set, g_set) in enumerate(zip(r_sets, g_sets)):
        shared, r_only, g_only = r_set & g_set, r_set - g_set, g_set - r_set
        union = r_set | g_set
        summary.append({"layer": layer_index + 1, "layer_index": layer_index, "recognition_count": len(r_set),
                        "generation_count": len(g_set), "shared_count": len(shared),
                        "recognition_only_count": len(r_only), "generation_only_count": len(g_only),
                        "union_count": len(union), "jaccard": len(shared) / len(union) if union else 0.0})
        memberships = ([(n, "shared") for n in shared]
                       + [(n, "recognition_only") for n in r_only]
                       + [(n, "generation_only") for n in g_only])
        for neuron, membership in memberships:
            neurons.append({"layer": layer_index + 1, "layer_index": layer_index, "neuron_index": neuron,
                            "membership": membership, "recognition_z": float(recognition[layer_index][neuron]),
                            "generation_z": float(generation[layer_index][neuron])})
    return summary, neurons


def encode_prompt(tokenizer: Any, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": text + "\n\nRespond naturally and supportively."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {key: value.to(device) for key, value in tokenizer(prompt, return_tensors="pt", add_special_tokens=False).items()}


def generate_one(model: Any, tokenizer: Any, controller: MLPController, text: str, max_new_tokens: int,
                 *, record: bool = False, prompt_moments: Moments | None = None,
                 generated_moments: Moments | None = None, mask_by_layer: dict[int, torch.Tensor] | None = None) -> str:
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, text, device)
    controller.prompt_mask = valid_mask(encoded["input_ids"], encoded["attention_mask"], tokenizer)
    controller.prompt_moments, controller.generated_moments = prompt_moments, generated_moments
    controller.record = record
    controller.mask_by_layer = mask_by_layer or {}
    controller.mask_generated = bool(mask_by_layer)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                use_cache=True)
    controller.record = controller.mask_generated = False
    controller.prompt_mask = None
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def response_probabilities(model: Any, tokenizer: Any, classifier: torch.nn.Module, texts: list[str]) -> np.ndarray:
    device = next(model.parameters()).device
    classifier_device = next(classifier.parameters()).device
    rows = []
    with torch.inference_mode():
        for text in texts:
            encoded = tokenizer(text or " ", return_tensors="pt", truncation=True, max_length=256)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            mask = valid_mask(encoded["input_ids"], encoded["attention_mask"], tokenizer)
            hidden = model.model(**encoded, use_cache=False).last_hidden_state
            weights = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            rows.append(torch.softmax(classifier(pooled.float().to(classifier_device)), dim=-1).cpu().numpy()[0])
    return np.asarray(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--csv-path", default=DEFAULT_CSV)
    parser.add_argument("--text-column", default="hinglish_genz_text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--target-emotion", default="sadness")
    parser.add_argument("--baseline-emotion", default="neutral")
    parser.add_argument("--samples-per-group", type=int, default=64)
    parser.add_argument("--causal-samples", type=int, default=32)
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--probe-path", default=DEFAULT_PROBE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.top_percent <= 100:
        parser.error("--top-percent must be in (0, 100].")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.csv_path)[[args.text_column, args.label_column]].dropna().drop_duplicates()
    data[args.text_column] = data[args.text_column].astype(str).str.strip()
    data[args.label_column] = data[args.label_column].astype(str).str.strip()
    target = data.loc[data[args.label_column].eq(args.target_emotion), args.text_column].tolist()[:args.samples_per_group]
    baseline = data.loc[data[args.label_column].eq(args.baseline_emotion), args.text_column].tolist()[:args.samples_per_group]
    if len(target) < 2 or len(baseline) < 2:
        raise ValueError("Target and baseline groups each need at least two non-empty examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto", dtype=dtype, low_cpu_mem_usage=True)
    model.eval(); model.config.use_cache = True
    for parameter in model.parameters(): parameter.requires_grad_(False)
    layers = list(model.model.layers); widths = [int(layer.mlp.down_proj.in_features) for layer in layers]
    controller = MLPController(layers)
    try:
        groups = {name: (Moments(widths), Moments(widths)) for name in ("target", "baseline")}
        for group, texts in (("target", target), ("baseline", baseline)):
            print(f"Recording {group} prompt and generated-token MLP activations for {len(texts)} examples.")
            for index, text in enumerate(texts, 1):
                generate_one(model, tokenizer, controller, text, args.max_new_tokens, record=True,
                             prompt_moments=groups[group][0], generated_moments=groups[group][1])
                if index % 10 == 0 or index == len(texts): print(f"  {group}: {index}/{len(texts)}")

        recognition = z_scores(groups["target"][0], groups["baseline"][0], widths)
        generation = z_scores(groups["target"][1], groups["baseline"][1], widths)
        r_sets, g_sets = top_sets(recognition, args.top_percent), top_sets(generation, args.top_percent)
        summary, neurons = to_rows(recognition, generation, r_sets, g_sets)
        pd.DataFrame(summary).to_csv(output_dir / "layerwise_overlap.csv", index=False)
        pd.DataFrame(neurons).sort_values(["layer_index", "membership", "neuron_index"]).to_csv(output_dir / "neuron_ids.csv", index=False)

        shared = {index: sorted(r_set & g_set) for index, (r_set, g_set) in enumerate(zip(r_sets, g_sets)) if r_set & g_set}
        rng = np.random.default_rng(args.seed)
        random_mask = {layer: sorted(rng.choice(widths[layer], len(ids), replace=False).tolist()) for layer, ids in shared.items()}
        probe = torch.load(args.probe_path, map_location="cpu", weights_only=False)
        labels = probe["labels"]
        if args.target_emotion not in labels: raise ValueError(f"Probe does not contain {args.target_emotion!r}.")
        classifier = torch.nn.Linear(int(probe["hidden_size"]), len(labels)).to(next(model.parameters()).device)
        classifier.load_state_dict(probe["state_dict"]); classifier.eval()
        causal_texts = target[:min(args.causal_samples, len(target))]
        baseline_responses = [generate_one(model, tokenizer, controller, text, args.max_new_tokens) for text in causal_texts]
        shared_responses = [generate_one(model, tokenizer, controller, text, args.max_new_tokens,
                                         mask_by_layer={k: torch.tensor(v) for k, v in shared.items()}) for text in causal_texts]
        random_responses = [generate_one(model, tokenizer, controller, text, args.max_new_tokens,
                                         mask_by_layer={k: torch.tensor(v) for k, v in random_mask.items()}) for text in causal_texts]
        target_index = labels.index(args.target_emotion)
        scores = {"baseline": response_probabilities(model, tokenizer, classifier, baseline_responses)[:, target_index],
                  "shared_mask": response_probabilities(model, tokenizer, classifier, shared_responses)[:, target_index],
                  "random_mask": response_probabilities(model, tokenizer, classifier, random_responses)[:, target_index]}
        pd.DataFrame({"prompt": causal_texts, "baseline_response": baseline_responses, "shared_mask_response": shared_responses,
                      "random_mask_response": random_responses, "baseline_target_probability": scores["baseline"],
                      "shared_mask_target_probability": scores["shared_mask"], "random_mask_target_probability": scores["random_mask"]}).to_csv(output_dir / "causal_response_scores.csv", index=False)
        causal = {name: {"mean_target_emotion_probability": float(values.mean()), "std": float(values.std())} for name, values in scores.items()}
        causal["shared_minus_baseline"] = float((scores["shared_mask"] - scores["baseline"]).mean())
        causal["random_minus_baseline"] = float((scores["random_mask"] - scores["baseline"]).mean())
        (output_dir / "causal_summary.json").write_text(json.dumps(causal, indent=2), encoding="utf-8")
        (output_dir / "experiment_metadata.json").write_text(json.dumps({"model_id": args.model_id, "target_emotion": args.target_emotion,
            "baseline_emotion": args.baseline_emotion, "samples_per_group": len(target), "causal_samples": len(causal_texts),
            "top_percent": args.top_percent, "max_new_tokens": args.max_new_tokens, "shared_neurons_by_layer": shared,
            "random_control_neurons_by_layer": random_mask}, indent=2), encoding="utf-8")
        print(f"Saved overlap and causal-validation results to {output_dir}")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
