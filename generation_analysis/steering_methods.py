#!/usr/bin/env python3
"""Steering interventions compared in the version-3 study (experiments only).

All methods act on decode steps only (the prompt pass is left alone, as in the
deployed hook) and use the selected model's own panels and residual means.

  additive   deployed method: panel neurons (input of mlp.down_proj) += alpha * gain * dmu
  selective  additive, restricted to a band of the panel's layers
  bounded    additive shift scaled per token by ||h|| / typical ||h|| of that layer, capped at
             ``cap`` * ||h||, and the activation vector rescaled to its original norm
  residual   contrastive direction in the residual stream after one layer:
             h += alpha * ||h|| * unit(mean(emotion) - mean(reference)), then rescaled to ||h||
             (reference "neutral", or "others" = the other response emotions and neutral)

A *spec* is a dict: {"method": ..., "layers": [...], "cap": float, "reference": str}.
``generate_rows`` decodes one prompt under several (spec, alpha) rows at once.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np
import torch

from backend.config import STEERING_GAIN
from .common import NEUTRAL, RESULTS_DIR, read_json
from .steering_lab import TARGETS

PANEL_DIR = RESULTS_DIR / "steering_panels"


def spec_name(spec: dict[str, Any]) -> str:
    method = spec["method"]
    if method == "additive":
        return "additive"
    if method == "selective":
        return f"selective_L{min(spec['layers'])}-{max(spec['layers'])}"
    if method == "bounded":
        return f"bounded_cap{spec.get('cap', 0.5):g}"
    return f"residual_{spec.get('reference', 'neutral')}_L{spec['layers'][0]}"


def residual_directions(reference: str = "neutral") -> dict[str, np.ndarray]:
    """Unit directions [layers + 1, hidden] per response emotion from the saved residual condition means."""
    means = np.load(PANEL_DIR / "residual_condition_means.npy")
    labels = read_json(RESULTS_DIR / "capture_metadata.json")["conditions"]
    out = {}
    for emotion in TARGETS:
        if reference == "neutral":
            base = means[labels.index(NEUTRAL)]
        else:
            base = means[[labels.index(name) for name in (*TARGETS, NEUTRAL) if name != emotion]].mean(0)
        direction = means[labels.index(emotion)] - base
        out[emotion] = direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
    return out


def layer_norms(runtime: Any, prompts: Sequence[tuple[list[dict[str, str]], str]], layers: Sequence[int],
                max_new_tokens: int = 40) -> dict[int, float]:
    """Mean L2 norm of the down_proj input on decode steps, per layer (reference scale of ``bounded``)."""
    sums: dict[int, float] = {layer: 0.0 for layer in layers}
    counts: dict[int, int] = {layer: 0 for layer in layers}
    seen: dict[int, bool] = {}

    def hook_for(layer: int):
        def hook(_module, inputs):
            if not seen.get(layer):
                seen[layer] = True
                return
            sums[layer] += float(inputs[0][:, -1].float().norm(dim=-1).sum())
            counts[layer] += inputs[0].shape[0]
        return hook

    handles = [runtime.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook_for(layer)) for layer in layers]
    try:
        for messages, system in prompts:
            seen.clear()
            encoded = runtime._encode_conversation(messages, system, max_new_tokens)
            with runtime.lock, torch.inference_mode():
                runtime.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=runtime.tokenizer.pad_token_id)
    finally:
        for handle in handles:
            handle.remove()
    return {layer: sums[layer] / max(counts[layer], 1) for layer in layers}


class Interventions:
    """Hook factory bound to one runtime; holds the residual directions and the layer norms."""

    def __init__(self, runtime: Any, norms: dict[int, float] | None = None) -> None:
        self.runtime = runtime
        self.norms = norms or {}
        self._directions: dict[str, dict[str, np.ndarray]] = {}

    def directions(self, reference: str) -> dict[str, np.ndarray]:
        if reference not in self._directions:
            self._directions[reference] = residual_directions(reference)
        return self._directions[reference]

    def hooks(self, spec: dict[str, Any], emotion: str, alphas: Sequence[float]) -> list[Any]:
        """Register the hooks of one spec; row i of the batch uses ``alphas[i]`` (0 = untouched)."""
        model, method = self.runtime.model, spec["method"]
        handles = []
        if method == "residual":
            reference = spec.get("reference", "neutral")
            for layer in spec["layers"]:
                unit = self.directions(reference)[emotion][layer + 1]
                handles.append(model.model.layers[layer].register_forward_hook(self._residual_hook(unit, alphas)))
            return handles
        panel = self.runtime._emotion_panels[emotion]
        layers = [layer for layer in panel if method != "selective" or layer in set(spec["layers"])]
        for layer in layers:
            neurons, shifts = panel[layer]
            bound = None
            if method == "bounded":
                bound = (self.norms[layer], float(spec.get("cap", 0.5)))
            handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(
                self._panel_hook(neurons, shifts, alphas, bound)))
        return handles

    @staticmethod
    def _panel_hook(neurons, shifts, alphas, bound):
        state: dict[str, Any] = {"prompt": True}

        def hook(_module, inputs):
            if state["prompt"]:
                state["prompt"] = False
                return inputs
            hidden = inputs[0]
            if "index" not in state:
                state["index"] = torch.tensor(neurons, device=hidden.device, dtype=torch.long)
                base = torch.tensor(shifts, device=hidden.device, dtype=torch.float32) * STEERING_GAIN
                scale = torch.tensor([float(a) for a in alphas], device=hidden.device, dtype=torch.float32)
                state["delta"] = scale[:, None, None] * base                    # [rows, 1, neurons]
                state["active"] = (scale != 0)[:, None, None]
            delta = state["delta"]
            modified = hidden.clone()
            if bound is None:
                modified[..., state["index"]] += delta.to(hidden.dtype)
                return (modified, *inputs[1:])
            typical, cap = bound
            norm = hidden.float().norm(dim=-1, keepdim=True)                    # [rows, 1, 1]
            shift = delta * (norm / typical)
            limit = cap * norm
            size = shift.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            shift = shift * torch.clamp(limit / size, max=1.0)
            updated = hidden.float().clone()          # .float() is a no-op view for float32 inputs
            updated[..., state["index"]] += shift
            updated = updated * (norm / updated.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            modified = torch.where(state["active"], updated.to(hidden.dtype), hidden)
            return (modified, *inputs[1:])
        return hook

    @staticmethod
    def _residual_hook(unit, alphas):
        state: dict[str, Any] = {"prompt": True}

        def hook(_module, _inputs, output):
            if state["prompt"]:
                state["prompt"] = False
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if "unit" not in state:
                state["unit"] = torch.tensor(unit, device=hidden.device, dtype=torch.float32)
                state["scale"] = torch.tensor([float(a) for a in alphas], device=hidden.device,
                                              dtype=torch.float32)[:, None, None]
            values = hidden.float()
            norm = values.norm(dim=-1, keepdim=True)
            updated = values + state["scale"] * norm * state["unit"]
            updated = updated * (norm / updated.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            updated = torch.where(state["scale"] != 0, updated, values).to(hidden.dtype)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return hook

    def generate_rows(self, messages: list[dict[str, str]], system_prompt: str, emotion: str | None,
                      rows: Sequence[tuple[dict[str, Any] | None, float]], max_new_tokens: int = 120) -> list[str]:
        """Greedy replies to one prompt; row i is steered with ``rows[i] = (spec, alpha)`` (None / 0 = unsteered)."""
        runtime = self.runtime
        encoded = runtime._encode_conversation(messages, system_prompt, max_new_tokens)
        input_ids = encoded["input_ids"].repeat(len(rows), 1)
        attention = encoded["attention_mask"].repeat(len(rows), 1)
        groups: dict[str, tuple[dict[str, Any], list[float]]] = {}
        for index, (spec, alpha) in enumerate(rows):
            if spec is None or not alpha:
                continue
            key = json.dumps(spec, sort_keys=True)
            groups.setdefault(key, (spec, [0.0] * len(rows)))[1][index] = float(alpha)
        handles = []
        try:
            for spec, alphas in groups.values():
                handles += self.hooks(spec, emotion, alphas)
            settings = {"max_new_tokens": max_new_tokens, "do_sample": False,
                        "pad_token_id": runtime.tokenizer.pad_token_id, **runtime.generation_settings()}
            with runtime.lock, torch.inference_mode():
                output = runtime.model.generate(input_ids=input_ids, attention_mask=attention, **settings)
        finally:
            for handle in handles:
                handle.remove()
        stop = runtime.model.generation_config.eos_token_id
        stop = set(stop if isinstance(stop, (list, tuple)) else [stop]) | {runtime.tokenizer.eos_token_id}
        replies = []
        for row in range(len(rows)):
            tokens = output[row, input_ids.shape[1]:].tolist()
            length = next((index + 1 for index, token in enumerate(tokens) if token in stop), len(tokens))
            replies.append(runtime.decode_response(tokens[:length]))
        return replies
