"""Summarise hook measurements of a steered reply for display.

Everything here is measured by the runtime's forward hooks on the input of
``mlp.down_proj`` (see ``backend.runtime.HuggingFaceRuntime._emotion_steering``):
the panel neurons' mean activation over the reply's decode steps, in the
steered run and, when available, in an unsteered run of the same prompt.
"""
from __future__ import annotations

from typing import Any, Mapping


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_intervention(steered: Mapping[str, Any] | None, unsteered: Mapping[str, Any] | None = None,
                           *, top_neurons: int = 10) -> dict[str, Any]:
    """Per-layer and per-neuron view of one steering intervention.

    ``steered`` / ``unsteered`` are the trace dicts filled by the runtime.  The
    activation after steering is ``measured activation before the shift`` plus
    the ``applied shift``; both terms come from the hook.
    """
    if not steered or not steered.get("layers") or not steered.get("alpha"):
        return {"applied": False, "alpha": 0.0, "emotion": steered.get("emotion") if steered else None,
                "layers": [], "top_neurons": [], "neurons_total": 0}
    reference = {item["layer"]: item for item in (unsteered or {}).get("layers", [])}
    layers, neurons = [], []
    for item in steered["layers"]:
        before = item["mean_activation_before_shift"]
        after = [value + shift for value, shift in zip(before, item["applied_shift"])]
        base = reference.get(item["layer"], {}).get("mean_activation_before_shift")
        layers.append({
            "layer": item["layer"], "neuron_count": len(item["neurons"]), "neuron_ids": item["neurons"],
            "decode_positions": item["decode_positions"],
            "neurons_raised": sum(shift > 0 for shift in item["applied_shift"]),
            "neurons_lowered": sum(shift < 0 for shift in item["applied_shift"]),
            # Shifts are signed and largely cancel in a plain mean, so magnitudes are reported.
            "mean_abs_applied_shift": round(_mean([abs(shift) for shift in item["applied_shift"]]), 4),
            "mean_abs_measured_change": round(_mean([abs(a - b) for a, b in zip(after, base)]), 4) if base else None,
        })
        for index, neuron in enumerate(item["neurons"]):
            neurons.append({"layer": item["layer"], "neuron": neuron,
                            "applied_shift": round(item["applied_shift"][index], 4),
                            "steered_activation": round(after[index], 4),
                            "unsteered_activation": round(base[index], 4) if base else None})
    neurons.sort(key=lambda entry: -abs(entry["applied_shift"]))
    return {
        "applied": True, "emotion": steered["emotion"], "alpha": steered["alpha"], "gain": steered.get("gain"),
        "location": steered.get("location"), "neurons_total": sum(layer["neuron_count"] for layer in layers),
        "layers": layers, "top_neurons": neurons[:top_neurons],
    }
