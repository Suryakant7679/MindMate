#!/usr/bin/env python3
"""Re-score the experiments with a second, independent judge model.

The first judge (Qwen2.5-7B-Instruct) selected the policy and scored the
held-out comparison.  This script repeats the judged measurements with another
model (default ``google/gemma-3-27b-it``, 4-bit) using the same rubrics and
prompts, without regenerating any reply:

heldout      all four systems on the test split: emotion rating, empathy,
             relevance, fluency, order-balanced pairwise preferences; the same
             paired contrasts as ``heldout_comparison``; agreement with judge 1.
calibration  the installed panel's sweep on the calibration split, and the
             selection rule re-run on judge-2 scores: would the second judge
             have selected the same strengths?

Judge 2 never influences which reply is sent or which policy is installed; it
is a check on judge 1.

  HF_HUB_OFFLINE=1 python3 -m generation_analysis.second_judge --load-in-4bit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import recalibrate_policy
from .common import write_json
from .steering_lab import (LAB_DIR, QUALITY, RUBRICS, Judge, emotion_rubric, holm, load_split, paired_summary)

HELDOUT = LAB_DIR / "heldout"
RECAL = LAB_DIR / "recalibration"
RATED = ("judge_emotion", *QUALITY)
PAIRS = (("v2", "v1"), ("v2", "kg_prompt"), ("v1", "kg_prompt"), ("v2", "baseline"), ("v1", "baseline"),
         ("kg_prompt", "baseline"))


def rate_all(judge: Judge, table: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Columns ``RATED`` re-scored by ``judge``; identical (context, reply, rubric) triples are scored once."""
    table = table.copy()
    contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
    replies = table.reply.fillna("").tolist()
    rubrics = {"judge_emotion": [emotion_rubric(target) for target in table.target],
               **{name: [text] * len(table) for name, text in RUBRICS.items()}}
    for name, rubric in rubrics.items():
        keys = list(zip(contexts, replies, rubric))
        unique = list(dict.fromkeys(keys))[:limit]
        values = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        table[name] = [float(values.get(key, np.nan)) for key in keys]
        print(f"rated {name}: {len(unique)} replies", flush=True)
    return table


def with_history(table: pd.DataFrame, split: str) -> pd.DataFrame:
    items = load_split(split).set_index("item_id")
    table = table.copy()
    table["history"] = [items.loc[item_id, "history"] for item_id in table.item_id]
    return table


def heldout(judge: Judge, tag: str, limit: int | None) -> None:
    first = pd.read_csv(HELDOUT / "scores.csv")
    second = rate_all(judge, with_history(first, "test"), limit)
    items = load_split("test")
    contexts = [Judge.context(row.text, row.history) for row in items.itertuples()]
    reply = {name: second[second.system == name].set_index("item_id").loc[items.item_id, "reply"].tolist()
             for name in ("baseline", "kg_prompt", "v1", "v2")}
    preferences = {}
    for a, b in PAIRS:
        differs = [i for i in range(len(items)) if reply[a][i] != reply[b][i]][:limit]
        values = np.full(len(items), 0.5)
        if differs:
            values[differs] = judge.prefer([contexts[i] for i in differs], [reply[a][i] for i in differs],
                                           [reply[b][i] for i in differs])
        preferences[f"{a}_over_{b}"] = values
    preferences = pd.DataFrame(preferences, index=items.item_id)
    second.drop(columns=["history"]).to_csv(HELDOUT / f"scores_{tag}.csv", index=False)
    preferences.to_csv(HELDOUT / f"preferences_{tag}.csv")

    first_pref = pd.read_csv(HELDOUT / "preferences.csv", index_col=0)
    wide = {judge_name: {name: group.set_index("item_id") for name, group in table.groupby("system")}
            for judge_name, table in (("judge1", first), (tag, second))}
    prefs = {"judge1": first_pref, tag: preferences}
    index = wide["judge1"]["baseline"].index
    steered = index[wide["judge1"]["v2"].loc[index, "applied_alpha"] > 0]
    rows = []
    for subset, ids in (("all", index), ("v2_steered", steered)):
        for a, b in PAIRS[:3]:
            for metric in (*RATED, "preference_minus_0.5"):
                entry = {"subset": subset, "contrast": f"{a} - {b}", "metric": metric}
                for judge_name in ("judge1", tag):
                    difference = (prefs[judge_name].loc[ids, f"{a}_over_{b}"] - 0.5 if metric.startswith("preference")
                                  else wide[judge_name][a].loc[ids, metric] - wide[judge_name][b].loc[ids, metric])
                    stats = paired_summary(difference)
                    entry.update({f"{judge_name}_mean": stats["mean"], f"{judge_name}_lo": stats["ci_low"],
                                  f"{judge_name}_hi": stats["ci_high"], f"{judge_name}_p": stats["p"]})
                # mean of the two judges per message, then the same paired test
                both = sum((prefs[j].loc[ids, f"{a}_over_{b}"] - 0.5 if metric.startswith("preference")
                            else wide[j][a].loc[ids, metric] - wide[j][b].loc[ids, metric]) for j in ("judge1", tag)) / 2
                stats = paired_summary(both)
                entry.update({"both_mean": stats["mean"], "both_lo": stats["ci_low"], "both_hi": stats["ci_high"],
                              "both_p": stats["p"]})
                entry["judges_agree_in_sign"] = bool(np.sign(entry["judge1_mean"]) == np.sign(entry[f"{tag}_mean"]))
                rows.append(entry)
    contrasts = pd.DataFrame(rows)
    primary = contrasts.subset == "all"
    for column in (f"{tag}_p", "both_p"):
        contrasts.loc[primary, column + "_holm"] = holm(contrasts.loc[primary, column])
    contrasts.to_csv(HELDOUT / f"contrasts_{tag}.csv", index=False)

    means = second.groupby("system")[list(RATED)].mean().reindex(["baseline", "kg_prompt", "v1", "v2"])
    means.to_csv(HELDOUT / f"system_means_{tag}.csv")
    agreement = []
    for metric in RATED:
        valid = second[metric].notna()
        rho = spearmanr(first.loc[valid, metric], second.loc[valid, metric]).statistic
        agreement.append({"metric": metric, "spearman_all_replies": rho,
                          "judge1_mean": first.loc[valid, metric].mean(), f"{tag}_mean": second.loc[valid, metric].mean(),
                          "mean_abs_difference": (first.loc[valid, metric] - second.loc[valid, metric]).abs().mean()})
    for a, b in PAIRS[:3]:
        column = f"{a}_over_{b}"
        differs = first_pref[column] != 0.5
        agreement.append({"metric": f"preference {a} over {b}",
                          "spearman_all_replies": spearmanr(first_pref.loc[differs, column], preferences.loc[differs, column]).statistic,
                          "same_winner_share": float(((first_pref.loc[differs, column] > 0.5)
                                                      == (preferences.loc[differs, column] > 0.5)).mean())})
    pd.DataFrame(agreement).to_csv(HELDOUT / f"judge_agreement_{tag}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(means.round(3).to_string()); print(pd.DataFrame(agreement).round(3).to_string(index=False))
    print(contrasts.round(4).to_string(index=False))


def calibration(judge: Judge, tag: str, panel: str, limit: int | None) -> None:
    first = pd.read_csv(RECAL / "scores.csv")
    first = first[(first.condition != "steer") | (first.panel == panel)].reset_index(drop=True)
    second = rate_all(judge, with_history(first, "calibration"), limit)
    kg = second[second.condition == "kg"].set_index("item_id")
    base = second[second.condition == "baseline"].set_index("item_id")
    reference = [base.loc[row.item_id, "reply"] if row.condition == "kg" else kg.loc[row.item_id, "reply"]
                 for row in second.itertuples()]
    contexts = [Judge.context(text, history) for text, history in zip(second.text, second.history)]
    rows = [i for i, row in enumerate(second.itertuples()) if row.condition != "baseline" and row.reply != reference[i]][:limit]
    preference = np.full(len(second), 0.5)
    preference[rows] = judge.prefer([contexts[i] for i in rows], [second.reply.iloc[i] for i in rows],
                                    [reference[i] for i in rows])
    second["preference_vs_reference"] = preference
    second.drop(columns=["history"]).to_csv(RECAL / f"scores_{tag}.csv", index=False)

    args = argparse.Namespace(tolerance=0.05, ci_margin=0.15, max_harm_share=0.20, min_preference=0.45, min_n=8,
                              min_strategy_n=10)
    table = []
    for judge_name, scores in (("judge1", first), (tag, second)):
        steered = recalibrate_policy.paired(scores)
        groups = [(f"{target}|{language}", cell) for (target, language), cell in steered.groupby(["target", "language"])]
        groups += [(f"strategy:{strategy}|{language}", cell)
                   for (strategy, language), cell in steered.groupby(["strategy", "language"])
                   if cell.item_id.nunique() >= args.min_strategy_n]
        for context, cell in groups:
            sweep = recalibrate_policy.sweep(cell, args)
            alpha, reason = recalibrate_policy.choose(sweep, args)
            for row in sweep:
                table.append({"judge": judge_name, "context": context, "selected_alpha": alpha, **row})
    table = pd.DataFrame(table)
    table.to_csv(RECAL / f"sweep_by_judge_{tag}.csv", index=False)
    chosen = table.drop_duplicates(["judge", "context"]).pivot(index="context", columns="judge", values="selected_alpha")
    chosen["same_decision"] = (chosen["judge1"] > 0) == (chosen[tag] > 0)
    chosen.to_csv(RECAL / f"selected_by_judge_{tag}.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(chosen.to_string())
    at = table[table.context.isin(["love|english", "strategy:compassionate_comfort|english",
                                   "strategy:grounding_reassurance|english"])]
    print(at[["judge", "context", "alpha", "n", "emotion_rating_gain", "emotion_rating_gain_lo", "empathy_change",
              "relevance_change", "fluency_change", "harm_share", "preference", "safe", "effective", "admissible"]]
          .round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", default="google/gemma-3-27b-it")
    parser.add_argument("--tag", default="judge2")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--stages", nargs="+", choices=("heldout", "calibration"), default=["heldout", "calibration"])
    parser.add_argument("--panel", default="cand:grad_global_half", help="Panel whose calibration sweep is re-scored.")
    parser.add_argument("--limit", type=int, help="Smoke test: score only the first N unique items per rubric.")
    args = parser.parse_args()
    judge = Judge(args.judge, load_in_4bit=args.load_in_4bit)
    for stage in args.stages:
        if stage == "heldout":
            heldout(judge, args.tag, args.limit)
        else:
            calibration(judge, args.tag, args.panel, args.limit)
    mass = float(np.mean(judge.answer_mass)) if judge.answer_mass else float("nan")
    write_json(LAB_DIR / f"{args.tag}_metadata.json", {
        "judge": args.judge, "load_in_4bit": args.load_in_4bit, "stages": args.stages, "limit": args.limit,
        "mean_probability_on_allowed_answers": mass, "rubrics": RUBRICS,
        "note": "same prompts and rubrics as judge 1; no reply was regenerated; judge 2 selects nothing"})
    print(f"mean probability on allowed answers: {mass:.3f}")
    judge.close()


if __name__ == "__main__":
    main()
