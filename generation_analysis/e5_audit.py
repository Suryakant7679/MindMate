#!/usr/bin/env python3
"""E5 audit: evidence from the already-used studies about E4, E5-old and the signals E5 could use.

CPU only.  Reads results/steering_modes, results/steering_kg and results/confirmatory (527 messages x 4 models, E3
and E4 replies rated by both judges) and writes results/e5_selective/audit/:

  policy_coverage.csv      why steering mode "policy" (E5-old) rarely steers, per model and decision category
  e4_attempts.csv          E4 attempted vs retained interventions, rejection reasons, alpha distribution
  batch_consistency.csv    how often E4's batched unsteered row equals the single-row E3 reply
  classifier_on_replies.csv  gate / independent classifier vs the judges' tone rating on assistant replies
  deficit_moderation.csv   E4 - E3 effects by the gate classifier's score of the unsteered reply (the deficit signal)
  cell_effects.csv         E4 - E3 effects per model x strategy x language
  acceptance_predictors.csv  which cheap candidate features predict a judged harm (AUC)
  markers.csv              risky-phrasing markers in E3 vs E4 replies and their association with safety
  judge_noise.csv          inter-judge agreement on per-message differences (justifies the margins)
  model_specific.json      Llama safety, EmoLLaMA degenerate replies, Llama-2 dose-response, Sarvam by language
  audit.json               the headline numbers of all of the above

  python3 -m generation_analysis.e5_audit
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .e5_common import QUALITY, ROOT, SAFETY, TAGS, development_table, write_json
from .steering_lab import paired_summary

OUT = ROOT / "audit"


def auc(scores: pd.Series, labels: pd.Series) -> float | None:
    """Mann-Whitney AUC of ``scores`` for ``labels`` (True = positive); None without both classes."""
    valid = scores.notna() & labels.notna()
    scores, labels = scores[valid].astype(float), labels[valid].astype(bool)
    if labels.all() or not labels.any():
        return None
    ranks = scores.rank()
    positives = labels.sum()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (~labels).sum()))


def spearman(a: pd.Series, b: pd.Series) -> float | None:
    from scipy.stats import spearmanr
    valid = a.notna() & b.notna()
    return float(spearmanr(a[valid], b[valid]).statistic) if valid.sum() > 5 else None


def summary(values: pd.Series) -> dict[str, float]:
    found = paired_summary(values.dropna().to_numpy(), resamples=2000)
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in found.items()}


# ------------------------------------------------------------------------------------------------ pieces

def policy_coverage() -> pd.DataFrame:
    from .common import read_jsonl
    rows = []
    for tag in TAGS:
        for record in read_jsonl(Path(f"results/steering_modes/{tag}/generations.jsonl")):
            if record["mode"] != "policy":
                continue
            reason = record.get("reason") or ""
            if record["status"] != "not_planned":
                category = record["status"]
            elif record.get("safety_flag"):
                category = "not planned: crisis / safety turn"
            elif "disabled for this model" in reason:
                category = "not planned: model has no calibrated policy"
            elif "no neuron panel" in reason:
                category = "not planned: no panel for the planned emotion"
            elif "only" in reason and "calibration messages" in reason:
                category = "not planned: calibration cell too small"
            elif reason.startswith("not steered") or "not steered" in reason:
                category = "not planned: calibrated cell set to alpha 0"
            else:
                category = "not planned: other"
            language = "hinglish" if "Hinglish" in reason else "english"
            rows.append({"model": tag, "category": category, "status": record["status"],
                         "strategy": record.get("strategy"), "reason": reason[:160], "language_in_reason": language})
    frame = pd.DataFrame(rows)
    table = frame.groupby(["model", "category"]).size().rename("n").reset_index()
    table["share"] = table.n / table.groupby("model").n.transform("sum")
    return table


def e4_attempts(dev: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for (tag, study), part in dev.groupby(["model", "study"]):
        attempted = part.e4_planned > 0
        retained = part.e4_applied > 0
        rejected = part.e4_status.eq("rejected")
        reasons = collections.Counter(re.sub(r"the steered reply was discarded because it ", "", r or "")[:60]
                                      for r in part.loc[rejected, "e4_reason"])
        applied = part.loc[retained, "e4_applied"]
        rows.append({"model": tag, "study": study, "n": len(part), "attempted": int(attempted.sum()),
                     "retained": int(retained.sum()), "rejected": int(rejected.sum()),
                     "attempted_share": attempted.mean(), "retained_share": retained.mean(),
                     "rejection_rate_of_attempts": rejected.sum() / max(attempted.sum(), 1),
                     "alpha_p10": applied.quantile(0.1) if len(applied) else None,
                     "alpha_median": applied.median() if len(applied) else None,
                     "alpha_p90": applied.quantile(0.9) if len(applied) else None,
                     "rejection_reasons": dict(reasons)})
    frame = pd.DataFrame(rows)
    note = {"steered_share_definition": "steered_share = mean(applied_alpha > 0); applied_alpha is set only when the "
                                        "candidate passed the run-time checks and was sent, so steered_share is the "
                                        "RETAINED rate. Attempts are planned_alpha > 0 (a candidate was generated)."}
    return frame, note


def batch_consistency(dev: pd.DataFrame) -> pd.DataFrame:
    part = dev[dev.batch0.notna()]
    return (part.groupby(["model", "study"]).batch0_is_kg.agg(["mean", "size"])
            .rename(columns={"mean": "batch0_equals_e3_share", "size": "n"}).reset_index())


def classifier_on_replies(dev: pd.DataFrame) -> pd.DataFrame:
    """Does the gate classifier read the target emotion of an assistant reply the way the judges do?"""
    rows = []
    long = pd.concat([
        dev.assign(reply_kind="E3", gate=dev.s0_kg, indep=dev.s0_kg_indep, tone=dev.kg_judge_emotion_both,
                   tone1=dev.kg_judge_emotion_judge1, tone2=dev.kg_judge_emotion_judge2),
        dev.assign(reply_kind="E4", gate=dev.e4_gate, indep=dev.e4_indep, tone=dev.e4_judge_emotion_both,
                   tone1=dev.e4_judge_emotion_judge1, tone2=dev.e4_judge_emotion_judge2)])
    for keys, part in long.groupby(["model", "lang3"]):
        rows.append({"model": keys[0], "language": keys[1], "replies": len(part),
                     "spearman_gate_vs_judges": spearman(part.gate, part.tone),
                     "spearman_indep_vs_judges": spearman(part.indep, part.tone),
                     "spearman_judge1_vs_judge2": spearman(part.tone1, part.tone2),
                     "auc_gate_for_tone_ge_3.5": auc(part.gate, part.tone >= 3.5),
                     "share_gate_ge_0.30": float((part.gate >= 0.30).mean()),
                     "median_gate": float(part.gate.median())})
        for target, sub in part.groupby("target"):
            if len(sub) >= 20:
                rows.append({"model": keys[0], "language": keys[1], "target": target, "replies": len(sub),
                             "spearman_gate_vs_judges": spearman(sub.gate, sub.tone),
                             "spearman_indep_vs_judges": spearman(sub.indep, sub.tone),
                             "spearman_judge1_vs_judge2": spearman(sub.tone1, sub.tone2),
                             "auc_gate_for_tone_ge_3.5": auc(sub.gate, sub.tone >= 3.5),
                             "share_gate_ge_0.30": float((sub.gate >= 0.30).mean()), "median_gate": float(sub.gate.median())})
    return pd.DataFrame(rows)


def deficit_moderation(dev: pd.DataFrame) -> pd.DataFrame:
    """E4 - E3 on retained interventions, by the gate score of the batched unsteered reply (what E5 sees)."""
    rows = []
    steered = dev[(dev.e4_applied > 0) & dev.s0_batch.notna()]
    bins = [(-0.01, 0.02, "<0.02"), (0.02, 0.10, "0.02-0.10"), (0.10, 0.30, "0.10-0.30"), (0.30, 1.01, ">=0.30")]
    for tag, part in steered.groupby("model"):
        for low, high, label in bins:
            sub = part[(part.s0_batch > low) & (part.s0_batch <= high)] if low >= 0 else part[part.s0_batch <= high]
            if len(sub) < 5:
                continue
            rows.append({"model": tag, "s0_bin": label, "n": len(sub),
                         **{f"d_{m}": sub[f"d_{m}_both"].mean() for m in ("judge_emotion", "quality", SAFETY)},
                         "harmed_share": sub.e4_harmed.mean(), "preference_e4": sub.pref_e4_over_kg.mean()})
        rows.append({"model": tag, "s0_bin": "spearman(s0, delta)", "n": len(part),
                     "d_judge_emotion": spearman(part.s0_batch, part.d_judge_emotion_both),
                     "d_quality": spearman(part.s0_batch, part.d_quality_both),
                     "d_safety": spearman(part.s0_batch, part.d_safety_both),
                     "harmed_share": auc(part.s0_batch, part.e4_harmed),
                     "preference_e4": spearman(part.s0_batch, part.pref_e4_over_kg)})
        # Oracle check: the judges' own tone of the E3 reply (not available at run time).
        rows.append({"model": tag, "s0_bin": "oracle: spearman(E3 judged tone, delta)", "n": len(part),
                     "d_judge_emotion": spearman(part.kg_judge_emotion_both, part.d_judge_emotion_both),
                     "d_quality": spearman(part.kg_judge_emotion_both, part.d_quality_both),
                     "d_safety": spearman(part.kg_judge_emotion_both, part.d_safety_both),
                     "harmed_share": auc(part.kg_judge_emotion_both, part.e4_harmed),
                     "preference_e4": spearman(part.kg_judge_emotion_both, part.pref_e4_over_kg)})
    return pd.DataFrame(rows)


def cell_effects(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordinary = dev[(dev.risk_category.isin(["none", "figurative"])) & ~dev.safety_flag]
    for keys, part in ordinary.groupby(["model", "lang3", "strategy"]):
        steered = part[part.e4_applied > 0]
        row = {"model": keys[0], "language": keys[1], "strategy": keys[2], "n": len(part),
               "retained": len(steered), "target": part.target.mode().iat[0]}
        for metric in ("judge_emotion", "quality", *QUALITY, SAFETY):
            row[f"d_{metric}"] = part[f"d_{metric}_both"].mean()
        tone, quality, safety = (summary(part[f"d_{m}_both"]) for m in ("judge_emotion", "quality", SAFETY))
        row.update(tone_lo=tone["ci_low"], tone_hi=tone["ci_high"], quality_lo=quality["ci_low"],
                   quality_hi=quality["ci_high"], safety_lo=safety["ci_low"], safety_hi=safety["ci_high"],
                   harmed_share=part.e4_harmed.mean(), preference_e4=part.pref_e4_over_kg.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def acceptance_predictors(dev: pd.DataFrame) -> pd.DataFrame:
    from chatbot_feature.emotion_reasoner.selective import RISKY, accept, SelectivePolicy, markers
    rows = []
    policy = SelectivePolicy(model="audit")
    for tag, part in dev[(dev.e4_applied > 0) & dev.candidate.notna()].groupby("model"):
        features = []
        for row in part.itertuples():
            unsteered = row.batch0 if isinstance(row.batch0, str) else row.kg_reply
            ok, reason, checks = accept(policy, unsteered, row.candidate, user_text=row.text)
            added = checks["added_markers"]
            features.append({"word_ratio": checks["word_ratio"], "length_ratio": checks["length_ratio"],
                             "repetition_increase": checks["repetition_increase"],
                             "language_change": float(checks["language_candidate"] != checks["language_unsteered"]),
                             "classifier_gain": row.cand_gate - row.s0_batch, "s0_batch": row.s0_batch,
                             "alpha": row.e4_applied, "intensity": row.intensity,
                             **{f"adds_{name}": float(name in added) for name in RISKY},
                             "e5_extra_rejects": float(not ok)})
        features = pd.DataFrame(features, index=part.index)
        harmed = part.e4_harmed
        safety_drop = part.d_safety_both < -0.25
        for column in features:
            rows.append({"model": tag, "feature": column, "n": len(part), "positive_rate": float(features[column].gt(0).mean())
                         if features[column].isin([0, 1]).all() else None,
                         "auc_harmed": auc(features[column], harmed), "auc_safety_drop": auc(features[column], safety_drop),
                         "mean_d_safety_if_present": float(part.loc[features[column] > 0, "d_safety_both"].mean())
                         if features[column].isin([0, 1]).all() and (features[column] > 0).any() else None,
                         "mean_d_safety_if_absent": float(part.loc[features[column] == 0, "d_safety_both"].mean())
                         if features[column].isin([0, 1]).all() else None})
    return pd.DataFrame(rows)


def marker_table(dev: pd.DataFrame) -> pd.DataFrame:
    from chatbot_feature.emotion_reasoner.selective import MARKERS, markers
    rows = []
    for tag, part in dev.groupby("model"):
        kg = [markers(text) for text in part.kg_reply.fillna("")]
        e4 = [markers(text) for text in part.e4_reply.fillna("")]
        for name in MARKERS:
            in_kg = np.array([name in found for found in kg])
            in_e4 = np.array([name in found for found in e4])
            added = in_e4 & ~in_kg
            rows.append({"model": tag, "marker": name, "share_e3": in_kg.mean(), "share_e4": in_e4.mean(),
                         "share_added_by_e4": added.mean(),
                         "d_safety_when_added": part.loc[added, "d_safety_both"].mean() if added.any() else None,
                         "d_safety_otherwise": part.loc[~added & (part.e4_applied > 0), "d_safety_both"].mean()})
    return pd.DataFrame(rows)


def judge_noise(dev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tag, part in dev[dev.e4_applied > 0].groupby("model"):
        for metric in ("judge_emotion", "quality", *QUALITY, SAFETY):
            a, b = part[f"d_{metric}_judge1"], part[f"d_{metric}_judge2"]
            valid = a.notna() & b.notna()
            a, b = a[valid], b[valid]
            moved = (a.abs() > 0.05) | (b.abs() > 0.05)
            rows.append({"model": tag, "metric": metric, "n": int(valid.sum()),
                         "spearman_judges_on_difference": spearman(a, b),
                         "sign_agreement_when_moved": float((np.sign(a[moved]) == np.sign(b[moved])).mean()) if moved.any() else None,
                         "mean_abs_judge_gap": float((a - b).abs().mean()),
                         "gap_of_means": float(abs(a.mean() - b.mean())),
                         "sd_of_difference": float(part[f"d_{metric}_both"].std())})
    return pd.DataFrame(rows)


def model_specific(dev: pd.DataFrame) -> dict[str, Any]:
    from chatbot_feature.emotion_reasoner.selective import markers
    from . import steering_modes as sm
    out: dict[str, Any] = {}
    llama = dev[(dev.model == "llama31_8b") & (dev.e4_applied > 0)]
    worst = llama.nsmallest(6, "d_safety_both")
    out["llama31_8b_safety"] = {
        "d_safety_by_language": llama.groupby("lang3").d_safety_both.mean().round(3).to_dict(),
        "d_safety_by_strategy": llama.groupby("strategy").d_safety_both.mean().round(3).to_dict(),
        "share_safety_drop_gt_0.5": float((llama.d_safety_both < -0.5).mean()),
        "word_increase_mean": float((llama.e4_words - llama.kg_words).mean()),
        "worst_examples": [{"item": row.item_id, "strategy": row.strategy, "language": row.lang3, "text": row.text,
                            "e3": row.kg_reply, "e4": row.e4_reply, "d_safety": round(row.d_safety_both, 2),
                            "e4_markers": markers(row.e4_reply)} for row in worst.itertuples()]}
    degenerate = {}
    for study, folder in (("steering_modes", "results/steering_kg/emollama_chat_7b"),
                          ("confirmatory", "results/confirmatory/emollama_chat_7b")):
        table = pd.read_csv(Path(folder) / "replies.csv")
        degenerate[study] = table.groupby("system").degenerate.mean().round(3).to_dict()
    emollama = dev[dev.model == "emollama_chat_7b"]
    out["emollama_chat_7b"] = {"degenerate_share_by_cell": degenerate,
                               "d_quality_e4_vs_e3_by_strategy": emollama.groupby("strategy").d_quality_both.mean().round(3).to_dict(),
                               "kg_degenerate_by_strategy": emollama.groupby("strategy").kg_degenerate.mean().round(3).to_dict()}
    kg_vs_none = {}
    for study, folder in (("steering_modes", "results/steering_kg/emollama_chat_7b"),
                          ("confirmatory", "results/confirmatory/emollama_chat_7b")):
        table, _ = sm.load_scores(Path(folder))
        for source in ("both",):
            table["quality"] = table[[f"{m}_{source}" for m in QUALITY]].mean(axis=1)
        wide = table.pivot_table(index="item_id", columns="system", values="quality")
        strategies = table.drop_duplicates("item_id").set_index("item_id").strategy
        kg_vs_none[study] = (wide["kg"] - wide["none"]).groupby(strategies).mean().round(3).to_dict()
    out["emollama_chat_7b"]["d_quality_kg_vs_none_by_strategy"] = kg_vs_none
    llama2 = dev[(dev.model == "llama2_13b_chat") & (dev.e4_applied > 0)].copy()
    llama2["alpha_bin"] = pd.cut(llama2.e4_applied, [0, 0.45, 0.55, 0.65, 1.0])
    out["llama2_13b_chat_dose_response"] = (llama2.groupby("alpha_bin", observed=True)
                                            .agg(n=("item_id", "size"), d_tone=("d_judge_emotion_both", "mean"),
                                                 d_quality=("d_quality_both", "mean"), d_safety=("d_safety_both", "mean"),
                                                 harmed=("e4_harmed", "mean"))
                                            .round(3).reset_index().astype({"alpha_bin": str}).to_dict("records"))
    sarvam = dev[dev.model == "sarvam_30b"]
    out["sarvam_30b_by_language"] = (sarvam.groupby("lang3")
                                     .agg(n=("item_id", "size"), retained=("e4_applied", lambda s: float((s > 0).mean())),
                                          d_tone=("d_judge_emotion_both", "mean"), d_quality=("d_quality_both", "mean"),
                                          d_safety=("d_safety_both", "mean"), preference=("pref_e4_over_kg", "mean"),
                                          batch0_equals_e3=("batch0_is_kg", "mean"))
                                     .round(3).reset_index().to_dict("records"))
    return out


# --------------------------------------------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.concat([development_table(tag) for tag in TAGS], ignore_index=True)
    dev.drop(columns=["history"]).to_csv(OUT / "development_table.csv", index=False)
    coverage = policy_coverage()
    coverage.to_csv(OUT / "policy_coverage.csv", index=False)
    attempts, attempt_note = e4_attempts(dev)
    attempts.to_csv(OUT / "e4_attempts.csv", index=False)
    batch = batch_consistency(dev)
    batch.to_csv(OUT / "batch_consistency.csv", index=False)
    reliability = classifier_on_replies(dev)
    reliability.to_csv(OUT / "classifier_on_replies.csv", index=False)
    moderation = deficit_moderation(dev)
    moderation.to_csv(OUT / "deficit_moderation.csv", index=False)
    cells = cell_effects(dev)
    cells.to_csv(OUT / "cell_effects.csv", index=False)
    predictors = acceptance_predictors(dev)
    predictors.to_csv(OUT / "acceptance_predictors.csv", index=False)
    marks = marker_table(dev)
    marks.to_csv(OUT / "markers.csv", index=False)
    noise = judge_noise(dev)
    noise.to_csv(OUT / "judge_noise.csv", index=False)
    specific = model_specific(dev)
    write_json(OUT / "model_specific.json", specific)
    pooled = {tag: {"tone": summary(part.d_judge_emotion_both), "quality": summary(part.d_quality_both),
                    "safety": summary(part.d_safety_both), "harmed_share_of_retained": float(part.e4_harmed.sum() / max((part.e4_applied > 0).sum(), 1)),
                    "preference_e4_over_e3": float(part.pref_e4_over_kg.mean())}
              for tag, part in dev.groupby("model")}
    write_json(OUT / "audit.json", {
        "messages_per_model": int(len(dev) / len(TAGS)),
        "e4_minus_e3_pooled_development": pooled,
        "policy_coverage": coverage.to_dict("records"),
        "e4_attempts": attempts.drop(columns=["rejection_reasons"]).round(4).to_dict("records"), **attempt_note,
        "batch_consistency": batch.round(4).to_dict("records"),
        "classifier_on_replies": reliability[reliability.get("target").isna()].round(3).to_dict("records")
        if "target" in reliability else reliability.round(3).to_dict("records"),
        "judge_noise": noise.round(3).to_dict("records"),
        "note": "Development evidence from already-used message sets (exploratory and confirmatory). Nothing here is "
                "a test of E5."})
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(coverage.to_string(index=False))
    print(attempts.drop(columns=["rejection_reasons"]).round(3).to_string(index=False))
    print(batch.round(3).to_string(index=False))
    print(reliability[reliability.get("target").isna()].drop(columns=["target"]).round(3).to_string(index=False))
    print(moderation.round(3).to_string(index=False))
    print(noise.round(3).to_string(index=False))
    print(json.dumps(pooled, indent=1)[:3000])


if __name__ == "__main__":
    main()
