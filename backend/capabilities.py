"""What each registry model is and what its research artefacts support, read from disk (no weights, no inference).

* ``checkpoint_facts``: architecture, depth, widths and context window from the cached checkpoint's ``config.json``,
  plus the size of its weight files.
* ``steering_status``: whether emotion-neuron discovery and steering artefacts exist for exactly this checkpoint
  (panels are only used when they were built for this model id, as ``HuggingFaceRuntime`` enforces), and what the
  calibrated policy allows.  The decision to steer a reply is still made per turn by the Emotion Reasoner, the
  policy and the run-time gate (``EmotionSteeringService``); this module only describes it.  Nothing here changes a
  policy, a panel or a result file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ModelSpec

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
_CACHE: dict[tuple[str, ...], tuple[tuple[float, ...], Any]] = {}


def _snapshot(model_id: str) -> Path | None:
    if Path(model_id).is_dir():
        return Path(model_id)
    from .models import cache_dir
    root = cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
    for snapshot in sorted(root.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True):
        if (snapshot / "config.json").exists():
            return snapshot
    return None


def _mtimes(paths: list[Path]) -> tuple[float, ...]:
    return tuple(path.stat().st_mtime if path.exists() else -1.0 for path in paths)


def _cached(key: tuple[str, ...], paths: list[Path], compute):
    stamp = _mtimes(paths)
    hit = _CACHE.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    value = compute()
    _CACHE[key] = (stamp, value)
    return value


def checkpoint_facts(model_id: str) -> dict[str, Any]:
    """Architecture facts of a locally available checkpoint (empty when it is not on disk)."""
    snapshot = _snapshot(model_id)
    if snapshot is None:
        return {}

    def compute() -> dict[str, Any]:
        try:
            config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        weights = [*snapshot.glob("*.safetensors"), *snapshot.glob("*.bin")]
        size = 0
        for path in weights:
            try:
                size += os.path.getsize(path)
            except OSError:
                pass
        experts = config.get("num_experts") or config.get("n_routed_experts")
        return {
            "architecture": (config.get("architectures") or [None])[0],
            "model_type": config.get("model_type"),
            "layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "mixture_of_experts": bool(experts),
            "routed_experts": experts,
            "max_position_embeddings": config.get("max_position_embeddings"),
            "checkpoint_dtype": config.get("torch_dtype"),
            "weights_gib": round(size / 1024 ** 3, 1) if size else None,
        }
    return _cached(("facts", model_id), [snapshot / "config.json"], compute)


def _panel_summary(path: Path, model_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"present": False}
    panels = payload.get("panels") or {}
    summary: dict[str, Any] = {"present": True, "file": str(path.relative_to(ROOT)),
                               "built_for": payload.get("model_id"), "matches_checkpoint": payload.get("model_id") == model_id,
                               "panels": {}}
    for name, by_layer in panels.items():
        layers = sorted(int(layer) for layer in by_layer)
        summary["panels"][name] = {"neurons": sum(len(item.get("neurons", [])) for item in by_layer.values()),
                                   "layers": len(layers), "layer_range": [layers[0], layers[-1]] if layers else None}
    if payload.get("directions"):
        summary["directions"] = {name: {"layer": int(item["layer"])} for name, item in payload["directions"].items()}
    if payload.get("strategy_tone"):
        summary["strategy_panels"] = dict(payload["strategy_tone"])
    return summary


def _policy_summary(path: Path) -> dict[str, Any]:
    try:
        policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"present": False}
    cells = policy.get("cells") or {}
    steered = [{"context": key, "alpha": cell.get("alpha"), "panel": cell.get("panel")}
               for key, cell in cells.items() if cell.get("alpha")]
    enabled = policy.get("steering_enabled", True) is not False and bool(steered)
    return {"present": True, "file": str(path.relative_to(ROOT)), "version": policy.get("version"),
            "source": policy.get("source"), "cells": len(cells), "steered_cells": steered,
            "steers": enabled, "gate": (policy.get("gate") or {}).get("rule") or policy.get("rule"),
            "note": policy.get("note")}


def steering_status(spec: ModelSpec, model_id: str | None = None) -> dict[str, Any]:
    """Emotion-neuron research status of one registry model (for ``model_id``, default the registry checkpoint).

    ``status``:
      * ``policy_gated``             panels for this checkpoint and a calibrated policy that steers some contexts;
                                     replies are steered only in those contexts and only if the run-time gate passes
      * ``disabled_by_calibration``  panels exist but the calibrated policy steers nothing (replies are unsteered)
      * ``uncalibrated``             panels exist but no calibrated policy: never steered (no heuristic strengths)
      * ``unavailable``              no panels for this checkpoint: unsteered chat only
    """
    model_id = model_id or spec.config.model_id
    tag = spec.config.results_tag if model_id == spec.config.model_id else None
    if not tag:
        return {"status": "unavailable", "reason": "no research results belong to this checkpoint",
                "results_tag": None}
    generation = RESULTS / f"{tag}_generation_emotion"
    panel_path = generation / "steering_panels" / spec.config.panel_file
    policy_path = generation / "steering_policy" / "policy.json"
    discovery = [RESULTS / f"{tag}_neuron_analysis" / "top_selectivity_neurons.json",
                 generation / "identification_metadata.json"]
    probe = RESULTS / f"{tag}_neuron_analysis" / "04_probe" / "probe.pt"
    validation = generation / "neuron_validation" / "report.json"

    def compute() -> dict[str, Any]:
        panels = _panel_summary(panel_path, model_id) if panel_path.exists() else {"present": False}
        policy = _policy_summary(policy_path) if policy_path.exists() else {"present": False}
        usable = panels.get("present") and panels.get("matches_checkpoint")
        if not usable:
            status, reason = "unavailable", ("no steering panels for this checkpoint" if not panels.get("present")
                                             else "the panels were built for another checkpoint")
        elif not policy.get("present"):
            status, reason = "uncalibrated", "panels exist but no calibrated policy: replies are never steered"
        elif not policy.get("steers"):
            status, reason = "disabled_by_calibration", policy.get("note") or "the calibrated policy steers no context"
        else:
            status = "policy_gated"
            reason = ("steered only in the calibrated contexts (" + ", ".join(c["context"] for c in policy["steered_cells"])
                      + ") and only when the run-time gate accepts the steered reply")
        return {"status": status, "reason": reason, "results_tag": tag,
                "neuron_discovery": any(path.exists() for path in discovery), "emotion_probe": probe.exists(),
                "causal_validation": validation.exists(), "panels": panels, "policy": policy}
    return _cached(("steering", spec.key, model_id), [panel_path, policy_path, validation, *discovery], compute)
