"""Summarise hook measurements of a steered reply for display and for the per-neuron record.

Everything here is measured by the runtime's forward hooks on the steering site (the input of
``mlp.down_proj`` for Llama-style models, the shared expert's ``down_proj`` for Sarvam, see
``backend.runtime.HuggingFaceRuntime.steering_module``): the panel neurons' mean activation over
the reply's decode steps, in the steered run and, when available, in an unsteered run of the same
prompt decoded in the same batch.

Terms
-----
applied shift        what the hook added to a neuron: alpha x gain x the neuron's stored shift.
activation before    the neuron's own activation in that run, read before the hook adds its shift.
steered activation   activation before (steered run) + applied shift: the value the model used.
measured change      steered activation - the same neuron's activation in the unsteered run.
upstream change      activation before (steered run) - activation in the unsteered run: how much
                     steering at earlier layers and earlier tokens had already moved this neuron.
normalised shift     applied shift / RMS of the panel neurons' unsteered activation in that layer.
contribution         |applied shift| x the L2 norm of the neuron's write vector (its ``down_proj``
                     column): the size of what this neuron's shift adds to the residual stream.  It
                     ranks neurons by how hard they push, not by their effect on the reply; effects
                     on replies come from the controlled ablation experiments.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

HEATMAP_COLUMNS = 24


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def summarize_intervention(steered: Mapping[str, Any] | None, unsteered: Mapping[str, Any] | None = None,
                           *, top_neurons: int = 10, cards: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
                           detail: bool = False) -> dict[str, Any]:
    """Per-layer and per-neuron view of one steering intervention.

    ``steered`` / ``unsteered`` are the trace dicts filled by the runtime.  ``cards`` maps
    ``(layer, neuron)`` to that neuron's offline evidence (attribution rank, gradient estimate, ...),
    shown next to the most contributing neurons.  ``detail`` adds the record of every panel neuron
    (``neurons``), which is what the telemetry file stores.
    """
    if not steered or not steered.get("layers") or not steered.get("alpha"):
        return {"applied": False, "alpha": 0.0, "emotion": steered.get("emotion") if steered else None,
                "layers": [], "top_neurons": [], "neurons_total": 0}
    # A layer can appear twice for models with two kinds of site (neurons and the residual stream).
    reference = {(item["layer"], item.get("site")): item for item in (unsteered or {}).get("layers", [])}
    layers, neurons = [], []
    for item in steered["layers"]:
        site = item.get("site")
        before, shifts = item["mean_activation_before_shift"], item["applied_shift"]
        after = [value + shift for value, shift in zip(before, shifts)]
        base = reference.get((item["layer"], site), {}).get("mean_activation_before_shift")
        writes = item.get("write_norm")
        scale = _rms(base) if base else 0.0
        contributions = [abs(shift) * (writes[index] if writes else 1.0) for index, shift in enumerate(shifts)]
        layers.append({
            "layer": item["layer"], **({"site": site} if site else {}),
            "neuron_count": len(item["neurons"]), "neuron_ids": item["neurons"],
            "decode_positions": item["decode_positions"],
            "neurons_raised": sum(shift > 0 for shift in shifts),
            "neurons_lowered": sum(shift < 0 for shift in shifts),
            # Shifts are signed and largely cancel in a plain mean, so magnitudes are reported as well.
            "mean_abs_applied_shift": round(_mean([abs(shift) for shift in shifts]), 4),
            "mean_applied_shift": round(_mean(list(shifts)), 5),
            "mean_abs_measured_change": round(_mean([abs(a - b) for a, b in zip(after, base)]), 4) if base else None,
            "mean_measured_change": round(_mean([a - b for a, b in zip(after, base)]), 5) if base else None,
            "mean_abs_upstream_change": round(_mean([abs(a - b) for a, b in zip(before, base)]), 4) if base else None,
            "rms_unsteered_activation": round(scale, 4) if base else None,
            "normalized_shift": round(_mean([abs(shift) for shift in shifts]) / scale, 4) if scale else None,
            "residual_update_norm": item.get("residual_update_norm"),
            "contribution": round(sum(contributions), 5),
        })
        for index, neuron in enumerate(item["neurons"]):
            shift = shifts[index]
            entry = {"layer": item["layer"], **({"site": site} if site else {}), "neuron": neuron,
                     "applied_shift": round(shift, 4),
                     "direction": "raised" if shift > 0 else "lowered" if shift < 0 else "unchanged",
                     "steered_activation": round(after[index], 4),
                     "unsteered_activation": round(base[index], 4) if base else None,
                     "measured_change": round(after[index] - base[index], 4) if base else None,
                     "upstream_change": round(before[index] - base[index], 4) if base else None,
                     "normalized_shift": round(shift / scale, 4) if scale else None,
                     "write_norm": round(writes[index], 4) if writes else None,
                     "contribution": round(contributions[index], 5)}
            card = (cards or {}).get((item["layer"], neuron)) if not site else None
            if card:
                entry["evidence"] = dict(card)
            neurons.append(entry)
    total = sum(entry["contribution"] for entry in neurons) or 1.0
    layer_total = sum(layer["contribution"] for layer in layers) or 1.0
    for layer in layers:
        layer["contribution_share"] = round(layer["contribution"] / layer_total, 4)
    neurons.sort(key=lambda entry: -entry["contribution"])
    for rank, entry in enumerate(neurons, start=1):
        entry["rank"] = rank
        entry["contribution_share"] = round(entry["contribution"] / total, 5)
    has_writes = any(item.get("write_norm") for item in steered["layers"])
    # Heat map: per layer, the neurons with the largest applied shift, coloured by measured change.
    rows = []
    for item_index, layer in enumerate(layers):
        members = sorted((entry for entry in neurons if entry["layer"] == layer["layer"] and entry.get("site") == layer.get("site")),
                         key=lambda entry: -abs(entry["applied_shift"]))[:HEATMAP_COLUMNS]
        rows.append({"layer": layer["layer"], "label": f"L{layer['layer']}" + (f" ({layer['site']})" if layer.get("site") else ""),
                     "cells": [{"neuron": entry["neuron"], "shift": entry["applied_shift"],
                                "change": entry["measured_change"] if entry["measured_change"] is not None else entry["applied_shift"]}
                               for entry in members]})
    peak = max((abs(cell["change"]) for row in rows for cell in row["cells"]), default=0.0)
    summary = {
        "applied": True, "emotion": steered["emotion"], "alpha": steered["alpha"], "gain": steered.get("gain"),
        "location": steered.get("location"), "neurons_total": sum(layer["neuron_count"] for layer in layers),
        "direction": "toward the panel emotion" if steered["alpha"] > 0 else "away from the panel emotion",
        "decode_steps": max((layer["decode_positions"] for layer in layers), default=0),
        "layers": layers, "top_neurons": neurons[:top_neurons],
        "contribution_basis": ("|applied shift| x L2 norm of the neuron's write vector (down_proj column)" if has_writes
                               else "|applied shift| (write vectors were not available for this model)"),
        "heatmap": {"columns": HEATMAP_COLUMNS, "max_abs": round(peak, 4), "rows": rows,
                    "metric": ("measured activation change (steered - unsteered reply), the "
                               f"{HEATMAP_COLUMNS} most shifted neurons of each layer") if unsteered
                              else f"applied shift, the {HEATMAP_COLUMNS} most shifted neurons of each layer"},
    }
    if detail:
        summary["neurons"] = neurons
    return summary
