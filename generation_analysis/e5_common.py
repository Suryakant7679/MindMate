"""Shared pieces of the E5 study (quality-constrained adaptive selective steering).

Paths, condition names, metrics, margins, and the *development table*: every message of the two earlier studies
(results/steering_modes + results/steering_kg, 226 messages; results/confirmatory, 301 messages) with its
knowledge-graph-only reply (E3), E4's steered candidate and sent reply, the gate classifier's scores, the risk screen
and both judges' ratings.  Those 527 messages are already used, so they serve E5 as development data only; the E5
held-out set is drawn fresh (see ``e5_data``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("results/e5_selective")
TAGS = ("llama31_8b", "emollama_chat_7b", "llama2_13b_chat", "sarvam_30b")
KEYS = {"llama31_8b": "llama", "emollama_chat_7b": "emollama", "llama2_13b_chat": "llama2", "sarvam_30b": "sarvam"}
TAG_OF = {key: tag for tag, key in KEYS.items()}

# Held-out conditions.  prompt variant (None = the chat's knowledge-graph prompt) and steering mode of the chat's own
# path; "selective" is E5-new (backend/selective_service.py).
CONDITIONS = {"E1": ("plain", "off"), "E2": ("plain", "always"), "E3": (None, "off"), "E4": (None, "always"),
              "E5old": (None, "policy"), "E5": (None, "selective")}
CONDITION_LABELS = {"E1": "baseline (MindMate prompt, no steering)", "E2": "steering only (MindMate prompt + steering)",
                    "E3": "KG-only", "E4": "KG + broad adaptive steering (mode always)",
                    "E5old": "KG + existing selective policy (mode policy)",
                    "E5": "KG + quality-constrained adaptive selective steering"}
# Control rows judged alongside the conditions: the unsteered reply decoded in the same batch as the steered
# candidate.  Batched and single-row greedy decoding differ numerically (E4's batched unsteered row equals the E3
# reply for only 60-70% of Llama-family and ~10% of Sarvam messages), so this row measures the batch-noise floor.
CONTROLS = ("E4_batch0", "E5_batch0")
PRIMARY_PAIRS = (("E5", "E3"), ("E5", "E4"), ("E5", "E5old"))
SECONDARY_PAIRS = (("E5", "E1"), ("E5", "E2"), ("E4", "E3"), ("E5old", "E3"))

TONE = "judge_emotion"
QUALITY = ("appropriateness", "empathy", "relevance", "fluency", "naturalness")       # quality constraints
SAFETY = "safety"                                                                      # separate constraint
EXTRA_RUBRICS = ("validation", "strategy_adherence")                                   # new in E5 (held-out)
HARM_DROP = 0.5
# Non-inferiority margins on the 1-5 judge scale.  Quality: 0.10, as registered in the confirmatory study (so the
# two studies are comparable), and about the inter-judge disagreement on per-message differences measured in the
# audit (see audit.json "judge_noise").  Safety: half of that, because safety must not be traded for tone.
MARGIN_QUALITY = 0.10
MARGIN_SAFETY = 0.05
SEED = 20261124


def existing_items() -> dict[str, pd.DataFrame]:
    out = {}
    for study, path in (("steering_modes", Path("results/steering_modes/items.csv")),
                        ("confirmatory", Path("results/confirmatory/items.csv"))):
        frame = pd.read_csv(path)
        frame["history"] = frame.history.map(json.loads)
        out[study] = frame.set_index("item_id")
    return out


def _records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    from .common import read_jsonl
    return {(row["item_id"], row["mode"]): row for row in read_jsonl(path)}


STUDIES = {  # study -> (scored folder, generation file, KG mode name in it, E4 mode name in it)
    "steering_modes": ("results/steering_kg/{tag}", "results/steering_modes/{tag}/generations.jsonl", "off", "always"),
    "confirmatory": ("results/confirmatory/{tag}", "results/confirmatory/{tag}/generations.jsonl", "kg", "kg_steer"),
}
RATED = ("judge_emotion", "appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness",
         "hinglish_naturalness")


def development_table(tag: str) -> pd.DataFrame:
    """One row per earlier message for model ``tag``: E3 reply, E4 candidate / sent reply, scores and ratings."""
    from chatbot_feature.emotion_reasoner.risk_screen import screen
    from chatbot_feature.emotion_reasoner.selective import language_of
    from . import steering_modes as sm
    items = existing_items()
    rows = []
    for study, (folder, generations, kg_mode, e4_mode) in STUDIES.items():
        table, preferences = sm.load_scores(Path(folder.format(tag=tag)))
        wide = {name: group.set_index("item_id") for name, group in table.groupby("system")}
        kg, e4 = wide["kg"], wide["kg_steer"]
        records = _records(Path(generations.format(tag=tag)))
        meta = items[study]
        pref = preferences["both"]
        for item_id in kg.index:
            record = records[(item_id, e4_mode)]
            info = meta.loc[item_id]
            target = kg.loc[item_id, "target"]
            unsteered_scores = record.get("unsteered_scores") or {}
            steered_scores = record.get("steered_scores") or {}
            risk = screen(info.text, info.history)
            row = {"study": study, "item_id": item_id, "group": info.group, "language": info.language,
                   "lang3": language_of(info.text), "turns": info.turns, "text": info.text, "history": info.history,
                   "source": info.source, "gold": info.gold, "user_emotion": info.user_emotion,
                   "intensity": float(info.intensity), "valence": float(info.valence),
                   "safety_flag": bool(info.safety_flag), "strategy": kg.loc[item_id, "strategy"], "target": target,
                   "risk_category": risk["category"], "state_level": record.get("state_level"),
                   "e4_planned": float(record.get("planned_alpha") or 0.0),
                   "e4_applied": float(record.get("applied_alpha") or 0.0), "e4_status": record.get("status"),
                   "e4_reason": record.get("reason"), "panel": record.get("panel"),
                   "kg_reply": kg.loc[item_id, "reply"], "e4_reply": e4.loc[item_id, "reply"],
                   "candidate": record.get("steered"), "batch0": record.get("unsteered"),
                   "s0_batch": float(unsteered_scores.get(target, np.nan)) if unsteered_scores else np.nan,
                   "cand_gate": float(steered_scores.get(target, np.nan)) if steered_scores else np.nan,
                   "s0_kg": float(kg.loc[item_id, "gate_target"]), "e4_gate": float(e4.loc[item_id, "gate_target"]),
                   "s0_kg_indep": float(kg.loc[item_id, "indep_target"]),
                   "e4_indep": float(e4.loc[item_id, "indep_target"]),
                   "kg_words": float(kg.loc[item_id, "reply_words"]), "e4_words": float(e4.loc[item_id, "reply_words"]),
                   "kg_degenerate": bool(kg.loc[item_id, "degenerate"]), "e4_degenerate": bool(e4.loc[item_id, "degenerate"]),
                   "pref_e4_over_kg": float(pref.loc[item_id, "kg_steer_over_kg"])}
            for metric in RATED:
                for source in ("judge1", "judge2", "both"):
                    column = f"{metric}_{source}"
                    if column in kg:
                        row[f"kg_{column}"] = float(kg.loc[item_id, column])
                        row[f"e4_{column}"] = float(e4.loc[item_id, column])
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["model"] = tag
    frame["batch0_is_kg"] = [isinstance(b, str) and b.strip() == str(k).strip()
                             for b, k in zip(frame.batch0, frame.kg_reply)]
    for source in ("judge1", "judge2", "both"):
        frame[f"kg_quality_{source}"] = frame[[f"kg_{m}_{source}" for m in QUALITY]].mean(axis=1)
        frame[f"e4_quality_{source}"] = frame[[f"e4_{m}_{source}" for m in QUALITY]].mean(axis=1)
    for metric in (*RATED, "quality"):
        for source in ("judge1", "judge2", "both"):
            if f"kg_{metric}_{source}" in frame:
                frame[f"d_{metric}_{source}"] = frame[f"e4_{metric}_{source}"] - frame[f"kg_{metric}_{source}"]
    checks = [f"d_{m}_both" for m in (*QUALITY, SAFETY)]
    frame["e4_harmed"] = (frame[checks].min(axis=1) < -HARM_DROP) & (frame.e4_applied > 0)
    return frame


def write_json(path: Path, payload: Any) -> None:
    from .common import write_json as _write
    _write(path, payload)


__all__ = ["ROOT", "TAGS", "KEYS", "TAG_OF", "CONDITIONS", "CONDITION_LABELS", "CONTROLS", "PRIMARY_PAIRS",
           "SECONDARY_PAIRS", "TONE", "QUALITY", "SAFETY", "EXTRA_RUBRICS", "HARM_DROP", "MARGIN_QUALITY",
           "MARGIN_SAFETY", "SEED", "development_table", "existing_items"]
