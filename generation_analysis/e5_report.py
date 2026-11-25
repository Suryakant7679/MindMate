"""E5 held-out analysis: per-model report, cross-model verdict, figures and qualitative examples.

Every contrast is paired per message over the *whole* held-out population (conditions that do not steer a message
contribute their unsteered reply); eligible and steered subsets are reported separately.  Mean of the two judges
unless a column says otherwise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .e5_common import (CONDITIONS, CONDITION_LABELS, MARGIN_QUALITY, MARGIN_SAFETY, PRIMARY_PAIRS, QUALITY,
                        SAFETY, SECONDARY_PAIRS, TAGS, write_json)

METRICS = ("judge_emotion", "quality", *QUALITY, SAFETY, "validation", "strategy_adherence", "hinglish_naturalness",
           "hindi_naturalness")
AUTOMATIC = ("language_match", "indep_target", "reply_words", "degenerate")
JUDGE_NAMES = ("judge1", "judge2")
PRIMARY_METRICS = ("judge_emotion", "quality", SAFETY, "validation")


# ------------------------------------------------------------------------------------------------ loading

def load(folder: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    table = pd.read_csv(folder / "replies.csv")
    preferences = {}
    for name in JUDGE_NAMES:
        ratings = pd.read_csv(folder / f"ratings_{name}.csv")
        if len(ratings) != len(table):
            raise SystemExit(f"ratings_{name}.csv does not line up with replies.csv.")
        for column in ratings.columns:
            table[f"{column}_{name}"] = ratings[column].to_numpy()
        table[f"quality_{name}"] = table[[f"{m}_{name}" for m in QUALITY]].mean(axis=1)
        preferences[name] = pd.read_csv(folder / f"preferences_{name}.csv", index_col=0)
    for metric in METRICS:
        parts = [f"{metric}_{name}" for name in JUDGE_NAMES if f"{metric}_{name}" in table]
        if parts:
            table[f"{metric}_both"] = table[parts].mean(axis=1)
    preferences["both"] = sum(preferences[name] for name in JUDGE_NAMES) / len(JUDGE_NAMES)
    return table, preferences


def wide_of(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: group.set_index("item_id") for name, group in table.groupby("condition")}


# ------------------------------------------------------------------------------------------------ statistics

def estimate(values: np.ndarray, kind: str = "two-sided", margin: float = 0.0) -> dict[str, Any]:
    from scipy.stats import ttest_1samp
    from .steering_lab import paired_summary
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    summary = paired_summary(values)
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    out = {"n": summary["n"], "mean": summary["mean"], "ci_low": summary["ci_low"], "ci_high": summary["ci_high"],
           "wilcoxon_p": summary["p"], "dz": summary["mean"] / sd if sd and sd > 0 else float("nan")}
    if len(values) >= 3 and not np.allclose(values, values[0]):
        alternative = {"greater": "greater", "noninferior": "greater", "less": "less"}.get(kind, "two-sided")
        out["t_p"] = float(ttest_1samp(values, -margin if kind == "noninferior" else 0.0, alternative=alternative).pvalue)
    else:
        shift = (values[0] if len(values) else 0.0) + (margin if kind == "noninferior" else 0.0)
        out["t_p"] = 0.0 if (kind in ("greater", "noninferior") and shift > 0) else 1.0
    return out


def difference(wide: dict[str, pd.DataFrame], first: str, second: str, ids: pd.Index, column: str) -> np.ndarray:
    return (wide[first].loc[ids, column].astype(float) - wide[second].loc[ids, column].astype(float)).to_numpy()


# ---------------------------------------------------------------------------------------------------- report

def subsets(wide: dict[str, pd.DataFrame]) -> dict[str, pd.Index]:
    e3, e5 = wide["E3"], wide["E5"]
    index = e3.index
    out = {"all": index}
    for column in ("lang3", "group", "category", "risk_category"):
        for value in sorted(e3[column].dropna().unique()):
            out[f"{column}={value}"] = index[e3[column] == value]
    out["e5_eligible"] = index[e5.e5_eligible_stage == "eligible"]
    out["e5_attempted"] = index[e5.e5_attempted.astype(bool)]
    out["e5_steered"] = index[e5.applied_alpha > 0]
    out["e4_steered"] = index[wide["E4"].applied_alpha > 0]
    return out


def report(tag: str, base: Path, smoke: bool = False) -> dict[str, Any]:
    from . import e5_selective
    from .common import read_jsonl
    folder = base / "heldout" / tag
    table, preferences = load(folder)
    conditions = table[table.condition.isin(CONDITIONS)]
    wide = wide_of(conditions)
    controls = wide_of(table[~table.condition.isin(CONDITIONS)]) if (~table.condition.isin(CONDITIONS)).any() else {}
    groups = subsets(wide)
    index = groups["all"]

    means = []
    for subset, ids in groups.items():
        if not len(ids):
            continue
        for condition in CONDITIONS:
            frame = wide[condition].loc[ids]
            means.append({"subset": subset, "condition": condition, "n": len(ids),
                          **{metric: frame[f"{metric}_both"].mean() for metric in METRICS if f"{metric}_both" in frame},
                          **{column: frame[column].astype(float).mean() for column in AUTOMATIC if column in frame},
                          "steered_share": float((frame.applied_alpha > 0).mean()),
                          "attempted_share": float((frame.planned_alpha > 0).mean()),
                          "mean_applied_alpha": float(frame.loc[frame.applied_alpha > 0, "applied_alpha"].mean())
                          if (frame.applied_alpha > 0).any() else 0.0})
    means = pd.DataFrame(means)
    means.to_csv(folder / "condition_means.csv", index=False)

    min_n = 2 if smoke else 6                     # smoke runs exercise every code path on a handful of messages
    contrasts = []
    for subset, ids in groups.items():
        if len(ids) < min_n:
            continue
        for first, second in (*PRIMARY_PAIRS, *SECONDARY_PAIRS):
            for source in (*JUDGE_NAMES, "both"):
                for metric in METRICS:
                    column = f"{metric}_{source}"
                    if column not in wide[first]:
                        continue
                    values = difference(wide, first, second, ids, column)
                    if np.isnan(values).all():
                        continue
                    contrasts.append({"subset": subset, "contrast": f"{first} - {second}", "metric": metric,
                                      "source": source, **estimate(values)})
                pref = preferences[source].loc[ids, f"{first}_over_{second}"].to_numpy() - 0.5
                contrasts.append({"subset": subset, "contrast": f"{first} - {second}", "metric": "preference_minus_0.5",
                                  "source": source, **estimate(pref)})
            for column in AUTOMATIC:
                values = difference(wide, first, second, ids, column)
                contrasts.append({"subset": subset, "contrast": f"{first} - {second}", "metric": column,
                                  "source": "automatic", **estimate(values)})
    contrasts = pd.DataFrame(contrasts, columns=["subset", "contrast", "metric", "source", "n", "mean", "ci_low", "ci_high",
                                                 "wilcoxon_p", "dz", "t_p"] if not contrasts else None)
    primary = ((contrasts.subset == "all") & (contrasts.source == "both")
               & contrasts.contrast.isin([f"{a} - {b}" for a, b in PRIMARY_PAIRS]) & contrasts.metric.isin(PRIMARY_METRICS))
    from .steering_lab import holm
    contrasts.loc[primary, "p_holm_primary_family"] = holm(contrasts.loc[primary, "wilcoxon_p"])
    contrasts.to_csv(folder / "contrasts.csv", index=False)

    # Registered hypotheses (one-sided / non-inferiority paired t-tests over every message; within-model Holm).
    registered = []
    for hypothesis in e5_selective.hypotheses(tag):
        values = difference(wide, hypothesis["first"], hypothesis["second"], index, f"{hypothesis['metric']}_both")
        found = estimate(values, hypothesis["test"], hypothesis["margin"] or 0.0)
        registered.append({**hypothesis, **found, "p": found["t_p"]})
    for row, adjusted in zip(registered, holm([row["p"] for row in registered])):
        row["p_holm_within_model"] = adjusted

    # Non-inferiority per quality metric (E5 vs E3 and vs E5-old), safety with its own margin.
    noninferiority = []
    for first, second in (("E5", "E3"), ("E5", "E5old"), ("E5", "E4")):
        for metric in (*QUALITY, "validation", SAFETY):
            margin = MARGIN_SAFETY if metric == SAFETY else MARGIN_QUALITY
            found = estimate(difference(wide, first, second, index, f"{metric}_both"), "noninferior", margin)
            noninferiority.append({"contrast": f"{first} - {second}", "metric": metric, "margin": -margin, **found,
                                   "noninferior_ci": bool(found["ci_low"] > -margin)})

    # Coverage, decisions and alpha distribution.
    records = {(r["item_id"], r["condition"]): r for r in read_jsonl(folder / "generations.jsonl")}
    e5_logs = [records[(item, "E5")].get("e5") or {} for item in index]
    stages = pd.Series([log.get("stage") for log in e5_logs]).value_counts().to_dict()
    statuses = pd.Series([log.get("status") for log in e5_logs]).value_counts().to_dict()
    rejections = pd.Series([t["reason"] for log in e5_logs for t in (log.get("tried") or []) if not t.get("accepted")]).value_counts().to_dict()
    risk = pd.Series([log.get("risk_category") for log in e5_logs]).value_counts().to_dict()

    def coverage_of(condition: str) -> dict[str, Any]:
        frame = wide[condition]
        applied = frame.loc[frame.applied_alpha > 0, "applied_alpha"]
        attempted = frame.planned_alpha > 0
        return {"attempt_rate": float(attempted.mean()), "retained_rate": float((frame.applied_alpha > 0).mean()),
                "rejection_rate_of_attempts": float(((frame.applied_alpha == 0) & attempted).sum() / max(attempted.sum(), 1)),
                "alpha_quantiles": applied.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() if len(applied) else [],
                "alpha_values": applied.round(3).tolist()}
    coverage = {condition: coverage_of(condition) for condition in ("E2", "E4", "E5old", "E5")}
    coverage["E5"].update(stages=stages, statuses=statuses, rejection_reasons=rejections, risk_categories=risk)
    harmed = {}
    checks = [f"{m}_both" for m in (*QUALITY, SAFETY)]
    for condition in ("E2", "E4", "E5old", "E5"):
        steered = index[wide[condition].applied_alpha > 0]
        if len(steered):
            reference = "E1" if condition == "E2" else "E3"
            drops = (wide[condition].loc[steered, checks] - wide[reference].loc[steered, checks]).min(axis=1)
            harmed[condition] = {"n_steered": int(len(steered)), "harmed_share": float((drops < -0.5).mean()),
                                 "reference": reference}

    # Latency (generated replies only; reused ones copy their source's time).
    latency = []
    for condition in CONDITIONS:
        frame = wide[condition]
        own = frame[frame.reused_from.isna() & frame.latency_seconds.notna()]
        latency.append({"condition": condition, "generated": int(len(own)),
                        "median_seconds": float(own.latency_seconds.median()) if len(own) else None,
                        "p90_seconds": float(own.latency_seconds.quantile(0.9)) if len(own) else None})
    attempted = index[wide["E5"].e5_attempted.astype(bool)]
    overhead = None
    if len(attempted):
        e5_lat = wide["E5"].loc[attempted, "latency_seconds"].astype(float)
        e4_lat = wide["E4"].loc[attempted, "latency_seconds"].astype(float)
        e3_lat = wide["E3"].loc[attempted, "latency_seconds"].astype(float)
        candidates = [records[(item, "E5")].get("e5", {}).get("candidates_generated", 0) for item in attempted]
        overhead = {"attempted_turns": int(len(attempted)), "e5_median": float(e5_lat.median()),
                    "e4_median_same_turns": float(e4_lat.median()), "e3_median_same_turns": float(e3_lat.median()),
                    "e5_over_e3_median_ratio": float((e5_lat / e3_lat).median()),
                    "mean_candidates_per_attempt": float(np.mean(candidates))}
    pd.DataFrame(latency).to_csv(folder / "latency.csv", index=False)

    # Batch-noise control: the unsteered row of a steering batch against the single-row E3 reply.
    noise = {}
    for control in controls:
        ids = controls[control].index.intersection(index)
        if len(ids) >= min_n:
            noise[control] = {"n": int(len(ids)), "identical_to_E3": float((controls[control].loc[ids, "reply"] == wide["E3"].loc[ids, "reply"]).mean()),
                              **{metric: estimate((controls[control].loc[ids, f"{metric}_both"] - wide["E3"].loc[ids, f"{metric}_both"]).to_numpy())
                                 for metric in ("judge_emotion", "quality", SAFETY)}}

    # Subgroups: E5 - E3, E4 - E3, E5 - E4 by language, category, safety category, strategy, target, user emotion, intensity.
    e3 = wide["E3"]
    keys = {"lang3": e3.lang3, "category": e3.category, "risk_category": e3.risk_category, "strategy": e3.strategy,
            "target": e3.target, "user_emotion": e3.user_emotion,
            "intensity": pd.cut(e3.intensity.astype(float), [0, 0.5, 0.7, 1.01], labels=["low", "mid", "high"]).astype(str)}
    subgroup_rows = []
    for key, values in keys.items():
        for value, ids in values.groupby(values).groups.items():
            if len(ids) < (2 if smoke else 5):
                continue
            for first, second in (("E5", "E3"), ("E4", "E3"), ("E5", "E4")):
                row = {"by": key, "value": value, "contrast": f"{first} - {second}", "n": int(len(ids)),
                       "steered_share_first": float((wide[first].loc[ids, "applied_alpha"] > 0).mean())}
                for metric in ("judge_emotion", "quality", SAFETY, "validation"):
                    found = estimate(difference(wide, first, second, ids, f"{metric}_both"))
                    row.update({f"d_{metric}": found["mean"], f"d_{metric}_lo": found["ci_low"], f"d_{metric}_hi": found["ci_high"]})
                subgroup_rows.append(row)
    pd.DataFrame(subgroup_rows).to_csv(folder / "subgroups.csv", index=False)

    from . import steering_modes as sm
    agreement = sm.judge_agreement(conditions)
    summary = {
        "model": tag, "messages": int(len(index)), "smoke": smoke,
        "registered": registered, "noninferiority": noninferiority, "coverage": coverage, "harmed": harmed,
        "latency": latency, "latency_overhead": overhead, "batch_noise": noise,
        "primary_contrasts": contrasts[primary].round(4).to_dict("records"),
        "judge_agreement": agreement,
        "deviations": {"plan_mismatches": e5_selective.plan_mismatches(records.values()),
                       "failed_generations": int(sum(bool(r.get("failed")) for r in records.values()))},
        "condition_labels": CONDITION_LABELS,
    }
    write_json(folder / "summary.json", summary)
    write_json(folder / "coverage.json", {"coverage": coverage, "harmed": harmed})
    pd.set_option("display.width", 220)
    view = means[means.subset == "all"][["condition", "judge_emotion", "quality", SAFETY, "validation", "strategy_adherence",
                                         "language_match", "steered_share", "attempted_share"]]
    print(view.round(3).to_string(index=False))
    for row in registered:
        print(f"  {row['hypothesis']} {row['claim'][:64]:64} {row['mean']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] "
              f"p={row['p']:.2g} Holm(model)={row['p_holm_within_model']:.2g}")
    print("E5 coverage:", {k: v for k, v in coverage["E5"].items() if k != "alpha_values"})
    return summary


# ---------------------------------------------------------------------------------------------------- verdict

ALPHA = 0.05


def verdict(base: Path) -> dict[str, Any]:
    from .steering_lab import holm
    from . import e5_selective
    registration_path = base / "preregistration.json"
    registered = (json.loads(registration_path.read_text())["hypotheses"] if registration_path.exists()
                  else [h for tag in TAGS for h in e5_selective.hypotheses(tag)])
    summaries = {tag: json.loads((base / "heldout" / tag / "summary.json").read_text())
                 for tag in TAGS if (base / "heldout" / tag / "summary.json").exists()}
    results = {row["id"]: row for summary in summaries.values() for row in summary["registered"]}
    rows = []
    for hypothesis in registered:
        found = results.get(hypothesis["id"])
        rows.append({**hypothesis, **({k: found[k] for k in ("n", "mean", "ci_low", "ci_high", "p", "dz")} if found else
                                      {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "p": 1.0, "dz": None}),
                     "ran": bool(found)})
    for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = adjusted
        row["confirmed"] = bool(row["ran"] and adjusted < ALPHA)
    by_model: dict[str, dict[str, bool]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["hypothesis"]] = row["confirmed"]
    verdicts = {}
    for tag in TAGS:
        h = by_model.get(tag, {})
        summary = summaries.get(tag)
        steered = summary["coverage"]["E5"]["retained_rate"] if summary else 0.0
        improves_e3 = h.get("H1", False) and h.get("H2", False) and h.get("H3", False)
        improves_e4 = h.get("H4", False) or h.get("H5", False)
        improves_old = h.get("H6", False) and h.get("H7", False)
        if not summary:
            recommendation = "not run"
        elif steered == 0:
            recommendation = "disable (the frozen policy steers nothing for this model)"
        elif improves_e3:
            recommendation = "enable, restricted to the policy's calibrated contexts"
        elif h.get("H1"):
            recommendation = "restrict further or disable: tone rises but quality or safety non-inferiority was not shown"
        else:
            recommendation = "disable: no confirmed tone benefit over KG-only replies"
        harmed = (summary or {}).get("harmed", {}).get("E5", {}).get("harmed_share")
        if not summary or not h.get("H1"):
            e6 = "not justified by E5: a response critic cannot add value to steering that brings no confirmed benefit"
        elif not improves_e3 or (harmed is not None and harmed > 0.10):
            e6 = ("justified: steering helps tone but some steered replies lose quality/safety, and the audit showed the "
                  "lightweight checks cannot detect them (AUC about 0.5); a response critic is the remaining lever")
        else:
            e6 = "optional: E5 already keeps quality and safety; E6 would need its own benefit hypothesis"
        verdicts[tag] = {"improves_on_E3": improves_e3, "reduces_E4_tradeoff": improves_e4, "improves_on_E5old": improves_old,
                         "steering_beneficial": improves_e3, "recommendation": recommendation, "E6": e6,
                         "retained_rate": steered, "harmed_share_of_E5_steered": harmed}
    out = {"hypotheses": rows, "verdicts": verdicts, "models_not_run": [t for t in TAGS if t not in summaries]}
    write_json(base / "report.json", out)
    write_report_markdown(base, out, summaries)
    for tag, entry in verdicts.items():
        print(f"{tag}: {entry['recommendation']} | E6: {entry['E6'][:70]}")
    return out


def _fmt(value: Any, digits: int = 3, sign: bool = True) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def write_report_markdown(base: Path, out: dict[str, Any], summaries: dict[str, Any]) -> None:
    lines = ["# E5: quality-constrained adaptive selective steering - held-out results", "",
             "Holm-Bonferroni over every registered test of every model; confirmed when the adjusted p < 0.05. "
             "Differences are paired per message over the whole held-out population (mean of two judges, 1-5 scale).", "",
             "## Registered hypotheses", "", "| id | claim | difference [95% CI] | dz | Holm p | result |",
             "|---|---|---|---|---|---|"]
    for row in out["hypotheses"]:
        effect = f"{_fmt(row['mean'])} [{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]" if row["ran"] else "not run"
        lines.append(f"| {row['id']} | {row['claim']} | {effect} | {_fmt(row['dz'], 2)} | {row['p_holm']:.2g} | "
                     f"{'confirmed' if row['confirmed'] else 'not confirmed'} |")
    lines += ["", "## Coverage and decisions", "", "| model | E4 retained | E5-old retained | E5 attempted | E5 retained | "
              "E5 harmed share (of steered) | E5 median alpha |", "|---|---|---|---|---|---|---|"]
    for tag, summary in summaries.items():
        coverage = summary["coverage"]
        alpha = coverage["E5"]["alpha_quantiles"][2] if coverage["E5"]["alpha_quantiles"] else None
        lines.append(f"| {tag} | {coverage['E4']['retained_rate']:.2f} | {coverage['E5old']['retained_rate']:.2f} | "
                     f"{coverage['E5']['attempt_rate']:.2f} | {coverage['E5']['retained_rate']:.2f} | "
                     f"{_fmt(summary['harmed'].get('E5', {}).get('harmed_share'), 2, False)} | {_fmt(alpha, 2, False)} |")
    lines += ["", "## Verdicts", ""]
    for tag, entry in out["verdicts"].items():
        lines += [f"### {tag}", "", f"- Improves on E3 (tone up, quality and safety non-inferior): **{entry['improves_on_E3']}**",
                  f"- Reduces E4's quality/safety trade-off: **{entry['reduces_E4_tradeoff']}**",
                  f"- Improves on E5-old: **{entry['improves_on_E5old']}**",
                  f"- Recommendation: {entry['recommendation']}", f"- E6 (response critic): {entry['E6']}", ""]
    lines += ["Which strategies and languages benefit: see `heldout/<model>/subgroups.csv` (exploratory, not registered).", "",
              "## Deviations", ""]
    lines += [f"- Models not run: {out['models_not_run']}"] if out["models_not_run"] else []
    for tag, summary in summaries.items():
        deviations = summary["deviations"]
        lines.append(f"- {tag}: {len(deviations['plan_mismatches'])} plan mismatches, {deviations['failed_generations']} failed generations")
    (base / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------- examples

def examples(base: Path, per_kind: int = 3) -> None:
    from .common import read_jsonl
    lines = ["# E5 examples (held-out)", "",
             "Selected by rule, not by hand: successes = E5 steered with the largest tone gain whose quality and safety did "
             "not fall; failures = E5 steered with the largest quality/safety drop; rejected / deficit-skipped / "
             "safety-gated = the first such turns.", ""]
    for tag in TAGS:
        folder = base / "heldout" / tag
        if not (folder / "replies.csv").exists() or not (folder / "ratings_judge1.csv").exists():
            continue
        table, _ = load(folder)
        wide = wide_of(table[table.condition.isin(CONDITIONS)])
        records = {(r["item_id"], r["condition"]): r for r in read_jsonl(folder / "generations.jsonl")}
        e5, e3, e4 = wide["E5"], wide["E3"], wide["E4"]
        delta = pd.DataFrame({m: e5[f"{m}_both"] - e3[f"{m}_both"] for m in ("judge_emotion", "quality", SAFETY)})
        steered = e5.applied_alpha > 0
        groups = {
            "successes": delta[steered & (delta.quality >= 0) & (delta[SAFETY] >= 0)].nlargest(per_kind, "judge_emotion").index,
            "failures": delta[steered].assign(worst=delta[[SAFETY, "quality"]].min(axis=1)).nsmallest(per_kind, "worst").index,
            "rejected": e5.index[e5.status == "rejected"][:2], "deficit_skipped": e5.index[e5.status == "skipped"][:2],
            "safety_gated": e5.index[e5.e5_eligible_stage == "safety"][:2]}
        lines += [f"## {tag}", ""]
        for kind, ids in groups.items():
            lines += [f"### {kind.replace('_', ' ')}", ""]
            if not len(ids):
                lines += ["(none)", ""]
            for item in ids:
                log = records[(item, "E5")].get("e5") or {}
                lines += [f"**{item}** ({e3.loc[item, 'lang3']}, {e3.loc[item, 'strategy']}, target {e3.loc[item, 'target']}, "
                          f"risk {e3.loc[item, 'risk_category']})", "", f"> User: {e3.loc[item, 'text']}", "",
                          f"- E3 (KG-only): {e3.loc[item, 'reply']}", f"- E4 (broad, alpha {e4.loc[item, 'applied_alpha']:.2f}): {e4.loc[item, 'reply']}",
                          f"- E5 (alpha {e5.loc[item, 'applied_alpha']:.2f}, {log.get('status')}): {e5.loc[item, 'reply']}",
                          f"- E5 decision: {log.get('decision_reason') or log.get('eligibility_reason')}",
                          f"- E5 - E3: tone {_fmt(delta.loc[item, 'judge_emotion'], 2)}, quality {_fmt(delta.loc[item, 'quality'], 2)}, "
                          f"safety {_fmt(delta.loc[item, SAFETY], 2)}", ""]
    (base / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"examples -> {base / 'examples.md'}")


def plots(base: Path) -> None:
    from .e5_plots import draw_all
    draw_all(base)


__all__ = ["report", "verdict", "examples", "plots", "load", "estimate"]
