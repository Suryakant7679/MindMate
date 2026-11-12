#!/usr/bin/env python3
"""Turn the steering sweep into a calibrated steering policy and a report.

Reads ``steering_policy/scores.csv`` (from steering_policy_experiment) and, for
every context cell, selects the steering strength with the largest target-
emotion gain among the strengths that kept response quality intact.

Context cells: (strategy, language) when that cell has at least
``--min-strategy-n`` messages, otherwise (response emotion, language).

A strength is *safe* in a cell when, relative to the unsteered reply with the
same KG prompt, the mean judged empathy, relevance and fluency each drop by no
more than ``--tolerance`` and at most ``--max-harm-share`` of replies lose more
than 0.5 on any of them.  If no safe strength gains at least ``--min-gain``, or
the cell has fewer than ``--min-n`` messages, the policy is not to steer.

Run from the project root:  python3 -m generation_analysis.calibrate_steering_policy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RESULTS_DIR, write_json

QUALITY = ("empathy", "relevance", "fluency")


def paired(table: pd.DataFrame, condition: str, reference: str) -> pd.DataFrame:
    base = table[table.condition == reference].set_index("message_id")
    rows = table[table.condition == condition].copy()
    for metric in (*QUALITY, "target_probability"):
        rows[f"d_{metric}"] = rows[metric].to_numpy() - base.loc[rows.message_id, metric].to_numpy()
    rows["unsteered_target"] = base.loc[rows.message_id, "target_probability"].to_numpy()
    rows["harmed"] = rows[[f"d_{metric}" for metric in QUALITY]].min(axis=1) < -0.5
    rows["language"] = np.where(rows.hinglish, "hinglish", "english")
    return rows


def sweep(cell: pd.DataFrame, args) -> list[dict]:
    rows = []
    for alpha, group in cell.groupby("alpha"):
        entry = {"alpha": float(alpha), "n": int(len(group)),
                 "target_gain": round(float(group.d_target_probability.mean()), 4),
                 "share_negative_gain": round(float((group.d_target_probability < -0.02).mean()), 3),
                 **{f"{metric}_change": round(float(group[f"d_{metric}"].mean()), 3) for metric in QUALITY},
                 "harm_share": round(float(group.harmed.mean()), 3),
                 "semantic_preservation": round(float(group.similarity_to_unsteered.mean()), 3)}
        entry["safe"] = bool(all(entry[f"{metric}_change"] >= -args.tolerance for metric in QUALITY)
                             and entry["harm_share"] <= args.max_harm_share)
        rows.append(entry)
    return rows


def select(rows: list[dict], args) -> tuple[float, str]:
    if not rows or rows[0]["n"] < args.min_n:
        return 0.0, f"not steered: only {rows[0]['n'] if rows else 0} test messages for this context"
    safe = [row for row in rows if row["safe"] and row["target_gain"] >= args.min_gain]
    if not safe:
        if not any(row["safe"] for row in rows):
            return 0.0, "not steered: every tested strength lowered judged response quality"
        return 0.0, "not steered: no quality-safe strength raised the target emotion"
    best = max(safe, key=lambda row: row["target_gain"])
    return best["alpha"], (f"strongest quality-safe gain in the sweep: +{best['target_gain']:.2f} target emotion "
                           f"over {best['n']} test messages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(RESULTS_DIR / "steering_policy"))
    # 0.10 selected love at 1.0, which did not hold on held-out messages (relevance -0.13, fluency -0.15
    # for +0.04 gain); 0.05 keeps a margin from the strength where quality starts to fall.
    parser.add_argument("--tolerance", type=float, default=0.05, help="Max mean drop per judged metric (1-5 scale).")
    parser.add_argument("--max-harm-share", type=float, default=0.20)
    parser.add_argument("--min-gain", type=float, default=0.03)
    parser.add_argument("--min-n", type=int, default=4)
    parser.add_argument("--min-strategy-n", type=int, default=5)
    parser.add_argument("--ceiling", type=float, default=0.30,
                        help="Unsteered target score at or above which steering is skipped at run time.")
    args = parser.parse_args()
    folder = Path(args.dir)
    table = pd.read_csv(folder / "scores.csv")
    steered = paired(table, "kg_steer", "kg_prompt")

    cells = {}
    for (target, language), cell in steered.groupby(["target", "language"]):
        rows = sweep(cell, args)
        alpha, reason = select(rows, args)
        cells[f"{target}|{language}"] = {"alpha": alpha, "reason": reason, "sweep": rows}
    for (strategy, language), cell in steered.groupby(["strategy", "language"]):
        if cell.message_id.nunique() < args.min_strategy_n:
            continue
        rows = sweep(cell, args)
        alpha, reason = select(rows, args)
        cells[f"strategy:{strategy}|{language}"] = {"alpha": alpha, "reason": reason, "sweep": rows,
                                                    "target": cell.target.iloc[0]}

    low, high = steered[steered.unsteered_target < args.ceiling], steered[steered.unsteered_target >= args.ceiling]
    ceiling = {"threshold": args.ceiling,
               "mean_gain_below": round(float(low.d_target_probability.mean()), 3),
               "mean_gain_at_or_above": round(float(high.d_target_probability.mean()), 3),
               "share_negative_below": round(float((low.d_target_probability < -0.02).mean()), 3),
               "share_negative_at_or_above": round(float((high.d_target_probability < -0.02).mean()), 3),
               "messages_at_or_above": int(high.message_id.nunique())}
    write_json(folder / "policy.json", {
        "source": "generation_analysis.steering_policy_experiment + calibrate_steering_policy",
        "rule": {"tolerance": args.tolerance, "max_harm_share": args.max_harm_share, "min_gain": args.min_gain,
                 "min_n": args.min_n, "reference": "unsteered reply with the same KG strategy prompt",
                 "quality_metrics": list(QUALITY), "judge": "Qwen/Qwen2.5-7B-Instruct",
                 "target_metric": "suryakantmani/xlm-roberta-large-goemotions-hard"},
        "ceiling": ceiling, "cells": cells,
    })

    # ---- report tables
    metrics = ["target_probability", *QUALITY, "repeated_bigram_rate", "similarity_to_user", "similarity_to_unsteered"]
    table.groupby(["condition", "alpha"])[metrics].mean().round(3).to_csv(folder / "conditions_by_alpha.csv")
    table["language"] = np.where(table.hinglish, "hinglish", "english")

    def policy_alpha(row) -> float:
        cell = cells.get(f"strategy:{row.strategy}|{row.language}") or cells[f"{row.target}|{row.language}"]
        return cell["alpha"]
    chosen = table.apply(policy_alpha, axis=1)
    at_policy = table[((table.alpha == 0) & table.condition.isin(["baseline", "kg_prompt"]))
                      | ((table.alpha == chosen) & (chosen > 0))].copy()
    # Messages whose policy is "do not steer" keep their unsteered reply in the steered conditions.
    for condition, reference in (("steer", "baseline"), ("kg_steer", "kg_prompt")):
        missing = table[(table.condition == reference) & (chosen == 0)].assign(condition=condition)
        at_policy = pd.concat([at_policy, missing])
    four = at_policy.groupby("condition")[metrics].mean().round(3).reindex(["baseline", "kg_prompt", "steer", "kg_steer"])
    four.to_csv(folder / "four_conditions_at_policy.csv")
    selected = pd.DataFrame([{"context": key, "alpha": cell["alpha"], "reason": cell["reason"]}
                             for key, cell in cells.items()])
    selected.to_csv(folder / "selected_strengths.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 110)
    print(selected.to_string(index=False)); print(); print(four.to_string()); print(); print(ceiling)


if __name__ == "__main__":
    main()
