#!/usr/bin/env python3
"""Experimental two-judge consensus steering policy (calibration split only; never installed here).

The deployed v2 policy selects a strength per context from judge 1 (Qwen2.5-7B)
alone.  This policy keeps the same rule but requires *both* judges to pass it:

  safe       for each judge separately: no quality rating (empathy, relevance, fluency) drops by more than
             the tolerance, harm share <= 20%, pairwise preference >= 0.45
  effective  for each judge separately: the emotion rating gain has a 95% CI above 0, and the independent
             GoEmotions classifier (SamLowe/roberta-base-go_emotions) also rises
  admissible safe and effective for both judges, and the next stronger strength is safe for both
  selected   the admissible strength with the largest gain averaged over the two judges; otherwise 0

It reads the calibration replies already scored by both judges
(``recalibration/scores.csv`` and ``scores_judge2.csv``) for the deployed panel
of the selected model; nothing is generated, and no held-out data is read.
Output: ``causal_steering/consensus/policy_consensus.json`` (same format as the
deployed policy, with the frozen run-time gate) and ``selection.csv``.

  MINDMATE_MODEL=llama python3 -m generation_analysis.consensus_policy
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import recalibrate_policy
from .common import MODEL_KEY, write_json
from .freeze_v2 import frozen_policy, sha256
from .steering_lab import INDEPENDENT_SCORER, JUDGE_ID, LAB_DIR

RECAL = LAB_DIR / "recalibration"
OUT = LAB_DIR / "consensus"
JUDGE2_ID = "google/gemma-3-27b-it"
RULE = argparse.Namespace(tolerance=0.05, ci_margin=0.15, max_harm_share=0.20, min_preference=0.45, min_n=8,
                          min_strategy_n=10)        # identical to the deployed rule


def cells_of(steered: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {f"{target}|{language}": cell for (target, language), cell in steered.groupby(["target", "language"])}
    groups.update({f"strategy:{strategy}|{language}": cell
                   for (strategy, language), cell in steered.groupby(["strategy", "language"])
                   if cell.item_id.nunique() >= RULE.min_strategy_n})
    return groups


def combine(first: list[dict], second: list[dict], min_n: int = RULE.min_n) -> tuple[float, str, list[dict]]:
    """Consensus over two judges' sweeps of one context (rows aligned by strength)."""
    rows = []
    for a, b in zip(first, second):
        assert a["alpha"] == b["alpha"]
        rows.append({"alpha": a["alpha"], "n": a["n"],
                     "emotion_rating_gain": round((a["emotion_rating_gain"] + b["emotion_rating_gain"]) / 2, 3),
                     "independent_classifier_gain": a["independent_classifier_gain"],
                     "safe": bool(a["safe"] and b["safe"]), "effective": bool(a["effective"] and b["effective"]),
                     "judge1": a, "judge2": b})
    for index, row in enumerate(rows):
        row["admissible"] = bool(row["safe"] and row["effective"] and index + 1 < len(rows) and rows[index + 1]["safe"])
    if not rows or rows[0]["n"] < min_n:
        return 0.0, f"not steered: only {rows[0]['n'] if rows else 0} calibration messages for this context", rows
    admissible = [row for row in rows if row["admissible"]]
    if admissible:
        best = max(admissible, key=lambda row: (row["emotion_rating_gain"], -row["alpha"]))
        return best["alpha"], (
            f"both judges: quality-safe with a safety margin and a reliable gain (emotion rating "
            f"+{best['judge1']['emotion_rating_gain']:.2f} judge 1, +{best['judge2']['emotion_rating_gain']:.2f} judge 2, "
            f"{best['n']} calibration messages)"), rows
    if not any(row["safe"] for row in rows):
        failing = [name for name, key in (("judge 1", "judge1"), ("judge 2", "judge2"))
                   if not any(row[key]["safe"] for row in rows)]
        who = " and ".join(failing) if failing else "the two judges never at the same strength"
        return 0.0, f"not steered: no strength was quality-safe for both judges ({who})", rows
    if not any(row["safe"] and row["effective"] for row in rows):
        return 0.0, "not steered: no strength was both quality-safe and reliably effective for both judges", rows
    return 0.0, "not steered: the only consensus strength sits next to one that lowers quality", rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    v2 = frozen_policy(MODEL_KEY)
    panel = v2["panel"]
    scores = {"judge1": pd.read_csv(RECAL / "scores.csv"), "judge2": pd.read_csv(RECAL / "scores_judge2.csv")}
    sweeps = {}
    for name, table in scores.items():
        table = table[(table.condition != "steer") | (table.panel == panel)]
        if table[table.condition == "steer"].empty:
            raise SystemExit(f"{name} has no calibration scores for the deployed panel {panel}.")
        sweeps[name] = {key: recalibrate_policy.sweep(cell, RULE)
                        for key, cell in cells_of(recalibrate_policy.paired(table)).items()}
    cells, table = {}, []
    for key in sweeps["judge1"]:
        alpha, reason, rows = combine(sweeps["judge1"][key], sweeps["judge2"][key])
        cells[key] = {"alpha": alpha, "reason": reason, "sweep": rows}
        if key.startswith("strategy:"):
            cells[key]["target"] = v2["cells"][key]["target"]
        table.append({"context": key, "n": rows[0]["n"], "v2_alpha": v2["cells"][key]["alpha"],
                      "judge1_alone": recalibrate_policy.choose(sweeps["judge1"][key], RULE)[0],
                      "judge2_alone": recalibrate_policy.choose(sweeps["judge2"][key], RULE)[0],
                      "consensus_alpha": alpha, "reason": reason})
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "policy_consensus.json", {
        "source": "generation_analysis.consensus_policy", "version": "experimental-consensus-1", "experimental": True,
        "panel": panel, "split": "calibration",
        "rule": {**vars(RULE), "judges": [JUDGE_ID, JUDGE2_ID], "independent_classifier": INDEPENDENT_SCORER,
                 "consensus": "safe and effective for each judge separately; next stronger strength safe for both"},
        "gate": v2["gate"],
        "inputs": {name: sha256(RECAL / file) for name, file in (("judge1", "scores.csv"), ("judge2", "scores_judge2.csv"))},
        "cells": cells})
    table = pd.DataFrame(table)
    table.to_csv(OUT / "selection.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
