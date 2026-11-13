"""Which model the experiment scripts analyse.

Every analysis script (recognition scripts in this folder and the
``generation_analysis`` package) reads its model id, result-folder prefix and
layer/width expectations from here.  The model is chosen with the environment
variable ``MINDMATE_MODEL`` (a key of ``backend.config.MODEL_REGISTRY``):

    MINDMATE_MODEL=llama2 python3 generate_full_selectivity.py

Unset, it is ``llama`` (Llama 3.1-8B), so existing commands and result folders
are unchanged.  ``run_model_pipeline.sh`` runs the whole sequence for one model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from backend.config import MODEL_REGISTRY
from backend.models import cache_dir, is_downloaded

EXPERIMENT_KEY = os.environ.get("MINDMATE_MODEL", "llama")
if EXPERIMENT_KEY not in MODEL_REGISTRY:
    raise SystemExit(f"MINDMATE_MODEL={EXPERIMENT_KEY!r} is not one of: {', '.join(MODEL_REGISTRY)}.")
EXPERIMENT_SPEC = MODEL_REGISTRY[EXPERIMENT_KEY]
EXPERIMENT_MODEL_ID = EXPERIMENT_SPEC.config.model_id
EXPERIMENT_TAG = EXPERIMENT_SPEC.config.results_tag

# (transformer layers, MLP intermediate width).  Used when the checkpoint's config.json is not on
# disk yet; when it is, the values are read from it instead.
_KNOWN_SHAPES = {"llama": (32, 14336), "llama2": (40, 13824), "emollama": (32, 11008)}


def _shape() -> tuple[int, int]:
    root = cache_dir() / ("models--" + EXPERIMENT_MODEL_ID.replace("/", "--")) / "snapshots"
    for path in [Path(EXPERIMENT_MODEL_ID) / "config.json", *root.glob("*/config.json")]:
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            config = config.get("text_config", config)
            if "num_hidden_layers" in config and "intermediate_size" in config:
                return int(config["num_hidden_layers"]), int(config["intermediate_size"])
    if EXPERIMENT_KEY in _KNOWN_SHAPES:
        return _KNOWN_SHAPES[EXPERIMENT_KEY]
    raise SystemExit(f"The layer count and MLP width of {EXPERIMENT_MODEL_ID} are unknown until it is downloaded.")


EXPERIMENT_NUM_LAYERS, EXPERIMENT_MLP_WIDTH = _shape()


def top_k(percent: float) -> int:
    """Neurons per layer for a top-percentage panel (same integer rule as the extraction scripts)."""
    return int(EXPERIMENT_MLP_WIDTH * percent / 100.0)


def require_downloaded() -> None:
    """Experiments never download weights implicitly."""
    if not is_downloaded(EXPERIMENT_MODEL_ID) and os.environ.get("MINDMATE_ALLOW_DOWNLOAD") != "1":
        raise SystemExit(f"{EXPERIMENT_MODEL_ID} is not downloaded. Run `hf download {EXPERIMENT_MODEL_ID}` first "
                         "(or set MINDMATE_ALLOW_DOWNLOAD=1).")
