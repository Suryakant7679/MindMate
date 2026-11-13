#!/usr/bin/env python3
"""Why calibration rejected steering for most contexts, and how causal each model's panels are.

Reads only existing files of the selected model (calibration replies scored by
both judges, the dev-split causal experiment); the single new computation is the
28-label GoEmotions profile of the calibration replies.  Nothing is selected
here.  Outputs go to ``causal_steering/diagnosis/``:

  failed_criteria.csv    every context x strength x judge with each criterion of the selection rule
  binding_constraints.csv  per context and judge: which criteria fail at the strengths that raise the emotion
  label_shift.csv        which GoEmotions labels move when a target emotion is steered (independent classifier)
  language_drift.csv     reply language, length and repetition by strength and message language
  headroom.csv           how much of the target emotion the unsteered reply already has
  examples.json          least-preferred steered replies for the rejected contexts
  causal_summary.csv     dev-split interventions: panel vs random neurons, clamping, reversed shift, likelihood

  MINDMATE_MODEL=llama python3 -m generation_analysis.diagnose_calibration
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import recalibrate_policy
from .causal_validation import GOEMOTIONS
from .common import MODEL_KEY, write_json
from .consensus_policy import RECAL, RULE, cells_of
from .freeze_v2 import frozen_policy
from .steering_lab import (INDEPENDENT_SCORER, LAB_DIR, QUALITY, TARGETS, classifier_scores, free_classifiers,
                           is_hinglish, paired_summary)

OUT = LAB_DIR / "diagnosis"
CAUSAL = LAB_DIR / "panel_causality"


def criteria(steered: pd.DataFrame, judge: str) -> list[dict]:
    rows = []
    for context, cell in cells_of(steered).items():
        sweep = recalibrate_policy.sweep(cell, RULE)
        for index, entry in enumerate(sweep):
            group = cell[cell.alpha == entry["alpha"]]
            preference = paired_summary(group.preference_vs_reference)
            drops = [metric for metric in QUALITY if entry[f"{metric}_change"] < -RULE.tolerance
                     or entry[f"{metric}_change_lo"] < -RULE.ci_margin]
            rows.append({
                "judge": judge, "context": context, "alpha": entry["alpha"], "n": entry["n"],
                "emotion_gain": entry["emotion_rating_gain"], "emotion_gain_lo": entry["emotion_rating_gain_lo"],
                "independent_gain": entry["independent_classifier_gain"],
                **{f"{metric}_change": entry[f"{metric}_change"] for metric in QUALITY},
                "harm_share": entry["harm_share"], "preference": entry["preference"],
                "preference_lo": preference["ci_low"], "preference_hi": preference["ci_high"],
                "ok_emotion_ci": entry["emotion_rating_gain_lo"] > 0, "ok_independent": entry["independent_classifier_gain"] > 0,
                "ok_quality": not drops, "quality_failed": " ".join(drops), "ok_harm": entry["harm_share"] <= RULE.max_harm_share,
                "ok_preference": entry["preference"] >= RULE.min_preference,
                # a preference below the bar whose interval still includes 0.5 is not evidence of a worse reply
                "preference_clearly_worse": preference["ci_high"] < 0.5,
                "safe": entry["safe"], "effective": entry["effective"],
                "ok_next_stronger_safe": bool(index + 1 < len(sweep) and sweep[index + 1]["safe"]),
                "admissible": entry["admissible"]})
    return rows


def binding(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    names = {"ok_emotion_ci": "emotion gain not reliable", "ok_independent": "independent classifier did not rise",
             "ok_quality": "quality rating dropped", "ok_harm": "too many harmed replies",
             "ok_preference": "judge preferred the unsteered reply", "ok_next_stronger_safe": "next stronger strength unsafe"}
    for (judge, context), cell in table.groupby(["judge", "context"]):
        low = cell[cell.alpha <= 1.0]                      # the range a policy would realistically use
        entry = {"judge": judge, "context": context, "n": int(cell.n.iloc[0]),
                 "selected": float(cell.loc[cell.admissible, "alpha"].max()) if cell.admissible.any() else 0.0,
                 "strengths_with_reliable_gain": int(low.ok_emotion_ci.sum()),
                 "best_gain_up_to_1": float(low.emotion_gain.max())}
        for column, label in names.items():
            entry[label] = int((~low[column]).sum())
        entry["preference_fail_but_ci_includes_0.5"] = int((~low.ok_preference & ~low.preference_clearly_worse).sum())
        only = [label for column, label in names.items()
                if (~low[column]).any() and all(low.loc[~low[column], [c for c in names if c != column]].all(axis=1))]
        entry["sole_blocker_somewhere"] = "; ".join(only)
        rows.append(entry)
    return pd.DataFrame(rows)


def label_shift(scores: pd.DataFrame) -> pd.DataFrame:
    kg = scores[scores.condition == "kg"].set_index("item_id")
    steer = scores[(scores.condition == "steer") & scores.alpha.isin([0.5, 1.0])]
    reference = dict(zip(kg.index, classifier_scores(kg.reply.fillna("").tolist(), INDEPENDENT_SCORER)))
    profile = classifier_scores(steer.reply.fillna("").tolist(), INDEPENDENT_SCORER)
    free_classifiers()
    change = profile - np.stack([reference[item] for item in steer.item_id])
    rows = []
    for (target, language, alpha), index in steer.reset_index(drop=True).groupby(["target", "language", "alpha"]).groups.items():
        mean = change[list(index)].mean(0)
        order = np.argsort(-np.abs(mean))[:6]
        rows.append({"target": target, "language": language, "alpha": alpha, "n": len(index),
                     "target_label_change": float(mean[GOEMOTIONS.index(target)]),
                     "largest_changes": ", ".join(f"{GOEMOTIONS[i]} {mean[i]:+.3f}" for i in order)})
    return pd.DataFrame(rows)


def drift(scores: pd.DataFrame) -> pd.DataFrame:
    kg = scores[scores.condition == "kg"].set_index("item_id")
    rows = []
    for (language, alpha), group in pd.concat([scores[scores.condition == "kg"], scores[scores.condition == "steer"]]) \
            .groupby(["language", "alpha"]):
        rows.append({"language": language, "alpha": alpha, "replies": len(group),
                     "reply_in_hinglish_share": float(group.reply.fillna("").map(is_hinglish).mean()),
                     "reply_words": float(group.reply_words.mean()),
                     "length_vs_unsteered": float((group.reply_words / kg.loc[group.item_id, "reply_words"].to_numpy()).mean()),
                     "repeated_bigram_rate": float(group.repeated_bigram_rate.mean()),
                     "fluency": float(group.fluency.mean()), "relevance": float(group.relevance.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    panel = frozen_policy(MODEL_KEY)["panel"]
    OUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for judge, file in (("judge1", "scores.csv"), ("judge2", "scores_judge2.csv")):
        table = pd.read_csv(RECAL / file)
        tables[judge] = table[(table.condition != "steer") | (table.panel == panel)].reset_index(drop=True)
    failed = pd.DataFrame([row for judge, table in tables.items() for row in criteria(recalibrate_policy.paired(table), judge)])
    failed.to_csv(OUT / "failed_criteria.csv", index=False)
    blockers = binding(failed)
    blockers.to_csv(OUT / "binding_constraints.csv", index=False)

    first = tables["judge1"]
    shifts = label_shift(first)
    shifts.to_csv(OUT / "label_shift.csv", index=False)
    drifting = drift(first)
    drifting.to_csv(OUT / "language_drift.csv", index=False)

    kg = {judge: table[table.condition == "kg"] for judge, table in tables.items()}
    headroom = kg["judge1"].groupby(["target", "language"]).agg(
        n=("item_id", "size"), emotion_rating_judge1=("judge_emotion", "mean"), gate_score=("gate_target", "mean"),
        independent_score=("indep_target", "mean"),
        share_at_or_above_gate_ceiling=("gate_target", lambda values: float((values >= 0.30).mean()))).reset_index()
    headroom["emotion_rating_judge2"] = kg["judge2"].groupby(["target", "language"]).judge_emotion.mean().to_numpy()
    headroom.to_csv(OUT / "headroom.csv", index=False)

    steered = recalibrate_policy.paired(first)
    second = tables["judge2"].set_index(["item_id", "condition", "alpha"]).preference_vs_reference
    examples = {}
    for (target, language), cell in steered[steered.alpha == 0.5].groupby(["target", "language"]):
        cell = cell.assign(judge2_preference=[second.get((row.item_id, "steer", row.alpha), np.nan) for row in cell.itertuples()])
        cell = cell.assign(mean_preference=(cell.preference_vs_reference + cell.judge2_preference) / 2).nsmallest(3, "mean_preference")
        examples[f"{target}|{language}"] = [
            {"user": row.text, "unsteered": row.unsteered_reply, "steered_alpha_0.5": row.reply,
             "preference_judge1": round(row.preference_vs_reference, 2), "preference_judge2": round(row.judge2_preference, 2),
             "emotion_change_judge1": round(row.d_judge_emotion, 2)} for row in cell.itertuples()]
    write_json(OUT / "examples.json", examples)

    effects = pd.read_csv(CAUSAL / "effects.csv")
    keep = effects[effects.condition.str.match(r"^(full@|random|styled)")]
    causal = keep[["target", "condition", "alpha", "neurons", "judge_emotion", "judge_emotion_lo", "judge_emotion_hi",
                   "indep_target", "empathy", "relevance", "fluency"]]
    causal.to_csv(OUT / "causal_summary.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 120)
    print(blockers.to_string(index=False))
    print(headroom.round(3).to_string(index=False))
    print(drifting.round(3).to_string(index=False))
    print(shifts.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
