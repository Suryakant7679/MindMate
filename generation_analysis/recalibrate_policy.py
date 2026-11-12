#!/usr/bin/env python3
"""Recalibrate the steering policy on the calibration split (post grounding-guidance fix).

Differences from ``steering_policy_experiment`` + ``calibrate_steering_policy``:

* plans come from the deployed detector (XLM-R fused with the knowledge graph),
  not from the Llama probe, and use the current strategy guidance;
* messages are the frozen *calibration* split (EmpatheticDialogues validation,
  single- and multi-turn, plus Hinglish); the test split is never read here;
* a strength is chosen on evaluators the run-time gate never sees (Qwen rating
  of the target emotion, third-party GoEmotions classifier), with bootstrap
  confidence intervals, so the gate classifier's own score cannot select itself;
* the run-time safeguards are compared offline on the same replies.

Selection rule (fixed before the calibration data were scored)
--------------------------------------------------------------
Cells are (strategy, language) with at least ``--min-strategy-n`` messages,
else (response emotion, language).  Relative to the unsteered reply with the
same strategy prompt, a strength is

  safe       mean change of empathy, relevance and fluency each >= -tolerance,
             the 95% bootstrap lower bound of each >= -ci_margin, at most
             ``max_harm_share`` of replies lose more than 0.5 on any of them, and
             the judge does not prefer the unsteered reply (preference >= min_preference);
  effective  the Qwen emotion rating rises with a 95% bootstrap lower bound > 0
             and the independent classifier's mean change is positive;
  admissible safe, effective, and the next stronger strength is also safe
             (keeps one step away from the point where quality starts to fall).

The admissible strength with the largest emotion-rating gain is selected; with
none, the cell is not steered.

Stages: generate -> score -> select.  Run from the project root:
  HF_HUB_OFFLINE=1 python3 -m generation_analysis.recalibrate_policy --panels deployed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import MODEL_ID, RESULTS_DIR, read_jsonl, write_json
from .steering_lab import (INDEPENDENT_SCORER, JUDGE_ID, LAB_DIR, QUALITY, SCORER, SYSTEM_PROMPT, Judge, generate_batch,
                           load_runtime, load_split, messages_of, paired_summary, release, score_table)

OUT = LAB_DIR / "recalibration"
POLICY_DIR = RESULTS_DIR / "steering_policy"
GATE = {"ceiling": 0.30, "min_gain": 0.05, "max_repetition_increase": 0.05, "min_length_ratio": 0.4}


def panel_source(name: str) -> dict[str, dict[int, tuple[list[int], list[float]]]]:
    """``deployed`` or a candidate from causal_panel_experiment (``cand:<name>``)."""
    if name == "deployed":
        payload = json.loads((RESULTS_DIR / "steering_panels" / "panels.json").read_text())["panels"]
    else:
        payload = json.loads((LAB_DIR / "panel_causality" / "candidate_panels.json").read_text())["panels"][
            name.removeprefix("cand:")]
    return {target: {int(layer): (item["neurons"], item["deltas"]) for layer, item in by_layer.items()}
            for target, by_layer in payload.items()}


def generate(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "generations.jsonl"
    done = {(r["item_id"], r["condition"], r["panel"], r["alpha"]) for r in read_jsonl(path)} if path.exists() else set()
    items = load_split(args.split)
    conversations = [messages_of(row) for row in items.itertuples()]
    runtime = load_runtime(args.model_id)
    runs = [("baseline", "none", 0.0), ("kg", "none", 0.0)]
    runs += [("steer", panel, alpha) for panel in args.panels for alpha in args.alphas]
    with path.open("a", encoding="utf-8") as handle:
        for condition, panel, alpha in runs:
            if all((item_id, condition, panel, alpha) in done for item_id in items.item_id):
                continue
            prompts = [SYSTEM_PROMPT] * len(items) if condition == "baseline" else items.kg_prompt.tolist()
            replies = [""] * len(items)
            if condition == "steer":
                panels = panel_source(panel)
                for target, group in items.groupby("target"):
                    runtime._emotion_panels["__variant__"] = panels[target]
                    index = group.index.tolist()
                    out = generate_batch(runtime, [conversations[i] for i in index], [prompts[i] for i in index],
                                         alpha=alpha, panel="__variant__", max_new_tokens=args.max_new_tokens)
                    for i, reply in zip(index, out):
                        replies[i] = reply
            else:
                replies = generate_batch(runtime, conversations, prompts, max_new_tokens=args.max_new_tokens)
            for item_id, reply in zip(items.item_id, replies):
                handle.write(json.dumps({"item_id": item_id, "condition": condition, "panel": panel, "alpha": alpha,
                                         "reply": reply}, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"generate: {condition} {panel} {alpha}", flush=True)
    del runtime
    release()


def score(args) -> None:
    items = load_split(args.split).set_index("item_id")
    table = pd.DataFrame(read_jsonl(OUT / "generations.jsonl")).drop_duplicates(["item_id", "condition", "panel", "alpha"])
    for column in ("text", "target", "strategy", "language", "turns", "intensity", "valence", "user_emotion"):
        table[column] = items.loc[table.item_id, column].to_numpy()
    table["history"] = [items.loc[item_id, "history"] for item_id in table.item_id]
    judge = Judge()
    table = score_table(table, judge=judge)
    # Pairwise: is the steered reply (or the KG-prompt reply) preferred over its unsteered reference?
    kg = table[table.condition == "kg"].set_index("item_id")
    base = table[table.condition == "baseline"].set_index("item_id")
    reference = [base.loc[row.item_id, "reply"] if row.condition == "kg" else kg.loc[row.item_id, "reply"]
                 for row in table.itertuples()]
    contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
    rows = [i for i, row in enumerate(table.itertuples()) if row.condition != "baseline" and row.reply != reference[i]]
    preference = np.full(len(table), 0.5)
    preference[rows] = judge.prefer([contexts[i] for i in rows], [table.reply.iloc[i] for i in rows],
                                    [reference[i] for i in rows])
    table["preference_vs_reference"] = preference
    judge.close()
    table.drop(columns=["history"]).to_csv(OUT / "scores.csv", index=False)


# ------------------------------------------------------------------------ select

def paired(table: pd.DataFrame) -> pd.DataFrame:
    kg = table[table.condition == "kg"].set_index("item_id")
    rows = table[table.condition == "steer"].copy()
    for metric in (*QUALITY, "judge_emotion", "indep_target", "gate_target", "repeated_bigram_rate"):
        rows[f"d_{metric}"] = rows[metric].to_numpy() - kg.loc[rows.item_id, metric].to_numpy()
    rows["unsteered_gate"] = kg.loc[rows.item_id, "gate_target"].to_numpy()
    rows["unsteered_reply"] = kg.loc[rows.item_id, "reply"].to_numpy()
    rows["harmed"] = rows[[f"d_{metric}" for metric in QUALITY]].min(axis=1) < -0.5
    return rows


def sweep(cell: pd.DataFrame, args) -> list[dict]:
    rows = []
    for alpha, group in sorted(cell.groupby("alpha")):
        emotion = paired_summary(group.d_judge_emotion)
        entry = {"alpha": float(alpha), "n": int(len(group)),
                 "emotion_rating_gain": round(emotion["mean"], 3), "emotion_rating_gain_lo": round(emotion["ci_low"], 3),
                 "emotion_rating_gain_hi": round(emotion["ci_high"], 3),
                 "independent_classifier_gain": round(float(group.d_indep_target.mean()), 4),
                 "target_gain": round(float(group.d_gate_target.mean()), 4),          # gate classifier (not used to select)
                 "share_negative_gain": round(float((group.d_gate_target < -0.02).mean()), 3),
                 "preference": round(float(group.preference_vs_reference.mean()), 3),
                 "harm_share": round(float(group.harmed.mean()), 3)}
        safe = entry["harm_share"] <= args.max_harm_share and entry["preference"] >= args.min_preference
        for metric in QUALITY:
            stats = paired_summary(group[f"d_{metric}"])
            entry[f"{metric}_change"] = round(stats["mean"], 3)
            entry[f"{metric}_change_lo"] = round(stats["ci_low"], 3)
            safe &= stats["mean"] >= -args.tolerance and stats["ci_low"] >= -args.ci_margin
        entry["safe"] = bool(safe)
        entry["effective"] = bool(emotion["ci_low"] > 0 and entry["independent_classifier_gain"] > 0)
        rows.append(entry)
    for index, entry in enumerate(rows):
        following = rows[index + 1]["safe"] if index + 1 < len(rows) else False
        entry["admissible"] = bool(entry["safe"] and entry["effective"] and following)
    return rows


def choose(rows: list[dict], args) -> tuple[float, str]:
    if not rows or rows[0]["n"] < args.min_n:
        return 0.0, f"not steered: only {rows[0]['n'] if rows else 0} calibration messages for this context"
    admissible = [row for row in rows if row["admissible"]]
    if admissible:
        best = max(admissible, key=lambda row: (row["emotion_rating_gain"], -row["alpha"]))
        return best["alpha"], (f"largest quality-safe gain with a safety margin: emotion rating "
                               f"+{best['emotion_rating_gain']:.2f} (95% CI {best['emotion_rating_gain_lo']:+.2f} to "
                               f"{best['emotion_rating_gain_hi']:+.2f}) over {best['n']} calibration messages")
    if not any(row["safe"] for row in rows):
        return 0.0, "not steered: every tested strength lowered judged response quality"
    if not any(row["safe"] and row["effective"] for row in rows):
        return 0.0, "not steered: no quality-safe strength reliably raised the target emotion"
    return 0.0, "not steered: the only effective strength sits next to one that lowers quality"


def gate_variants(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Which steered candidates each run-time rule would send (True = steered reply sent)."""
    words = rows.reply.str.len() >= GATE["min_length_ratio"] * rows.unsteered_reply.str.len()
    fluent = (rows.d_repeated_bigram_rate <= GATE["max_repetition_increase"]) & words
    below = rows.unsteered_gate < GATE["ceiling"]
    return {"always_steer": np.ones(len(rows), dtype=bool),
            "safeguards_only": fluent.to_numpy(),
            "ceiling_and_safeguards": (below & fluent).to_numpy(),
            "deployed_gate": (below & fluent & (rows.d_gate_target >= GATE["min_gain"])).to_numpy()}


def select(args) -> None:
    table = pd.read_csv(OUT / "scores.csv")
    steered = paired(table)
    report: dict[str, dict] = {}
    for panel, block in steered.groupby("panel"):
        cells = {}
        for (target, language), cell in block.groupby(["target", "language"]):
            rows = sweep(cell, args)
            alpha, reason = choose(rows, args)
            cells[f"{target}|{language}"] = {"alpha": alpha, "reason": reason, "sweep": rows}
        for (strategy, language), cell in block.groupby(["strategy", "language"]):
            if cell.item_id.nunique() < args.min_strategy_n:
                continue
            rows = sweep(cell, args)
            alpha, reason = choose(rows, args)
            cells[f"strategy:{strategy}|{language}"] = {"alpha": alpha, "reason": reason, "sweep": rows,
                                                        "target": cell.target.iloc[0]}
        report[panel] = cells

        # Policy-level outcome on the calibration messages, and the run-time rules compared offline.
        def cell_alpha(row) -> float:
            cell = cells.get(f"strategy:{row.strategy}|{row.language}") or cells[f"{row.target}|{row.language}"]
            return cell["alpha"]
        block = block.assign(policy_alpha=block.apply(cell_alpha, axis=1))
        at_policy = block[(block.alpha == block.policy_alpha) & (block.policy_alpha > 0)]
        items = block.item_id.nunique()
        gates = []
        for name, sent in gate_variants(at_policy).items():
            entry = {"panel": panel, "rule": name, "messages": items, "steered_candidates": len(at_policy),
                     "steered_sent": int(sent.sum())}
            for metric in ("judge_emotion", "indep_target", "gate_target", *QUALITY):
                change = np.zeros(items)                      # messages that keep the unsteered reply change by 0
                change[:int(sent.sum())] = at_policy.loc[sent, f"d_{metric}"]
                stats = paired_summary(change)
                entry.update({metric: stats["mean"], f"{metric}_lo": stats["ci_low"], f"{metric}_hi": stats["ci_high"]})
            preference = np.full(items, 0.5)
            preference[:int(sent.sum())] = at_policy.loc[sent, "preference_vs_reference"]
            entry["preference"] = float(preference.mean())
            entry["harmed_sent"] = int(at_policy.loc[sent, "harmed"].sum())
            gates.append(entry)
        report[panel + "::gates"] = gates

    for panel in sorted(key for key in report if "::" not in key):
        cells = report[panel]
        name = panel.replace(":", "_")
        write_json(OUT / f"policy_{name}.json", {
            "source": "generation_analysis.recalibrate_policy", "version": 2, "panel": panel, "split": args.split,
            "rule": {"tolerance": args.tolerance, "ci_margin": args.ci_margin, "max_harm_share": args.max_harm_share,
                     "min_preference": args.min_preference, "min_n": args.min_n,
                     "reference": "unsteered reply with the same KG strategy prompt",
                     "selection_metric": f"emotion rating by {JUDGE_ID} (1-5), confirmed by {INDEPENDENT_SCORER}",
                     "quality_metrics": list(QUALITY), "judge": JUDGE_ID,
                     "gate_classifier_not_used_for_selection": SCORER,
                     "safety_margin": "the next stronger strength must also be safe"},
            "gate": {"rule": args.gate_rule, "ceiling": GATE["ceiling"], "min_gain": GATE["min_gain"],
                     "max_repetition_increase": GATE["max_repetition_increase"],
                     "min_length_ratio": GATE["min_length_ratio"],
                     "note": "ceiling_and_safeguards does not select on the gate classifier's gain; "
                             "classifier_gain is the previous rule (see gate_rules_*.csv for the offline comparison)"},
            "cells": cells,
        })
        pd.DataFrame([{"context": key, "alpha": cell["alpha"], "reason": cell["reason"]}
                      for key, cell in cells.items()]).to_csv(OUT / f"selected_strengths_{name}.csv", index=False)
        pd.DataFrame(report[panel + "::gates"]).to_csv(OUT / f"gate_rules_{name}.csv", index=False)

    # Flat sweep table (every panel, cell, alpha) and moderators of the steering effect.
    flat = [{"panel": panel, "context": key, **row} for panel, cells in report.items() if "::" not in panel
            for key, cell in cells.items() for row in cell["sweep"]]
    pd.DataFrame(flat).to_csv(OUT / "sweep_by_cell.csv", index=False)
    moderators = []
    for (panel, alpha), group in steered.groupby(["panel", "alpha"]):
        for name, mask in (("single_turn", group.turns == "single"), ("multi_turn", group.turns == "multi"),
                           ("unsteered_gate_below_0.30", group.unsteered_gate < 0.30),
                           ("unsteered_gate_at_or_above_0.30", group.unsteered_gate >= 0.30),
                           ("intensity_below_median", group.intensity < group.intensity.median()),
                           ("intensity_at_or_above_median", group.intensity >= group.intensity.median())):
            part = group[mask & (group.language == "english")]
            if len(part) < 5:
                continue
            emotion = paired_summary(part.d_judge_emotion)
            moderators.append({"panel": panel, "alpha": alpha, "subset": name, "n": len(part),
                               "emotion_rating_gain": emotion["mean"], "lo": emotion["ci_low"], "hi": emotion["ci_high"],
                               **{f"{metric}_change": float(part[f"d_{metric}"].mean()) for metric in QUALITY},
                               "preference": float(part.preference_vs_reference.mean())})
    pd.DataFrame(moderators).to_csv(OUT / "moderators.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120); pd.set_option("display.max_rows", 500)
    for panel in sorted(key for key in report if "::" not in key):
        print(f"\n=== panel {panel}")
        print(pd.DataFrame([{"context": key, "alpha": cell["alpha"], "reason": cell["reason"]}
                            for key, cell in report[panel].items()]).to_string(index=False))
        print(pd.DataFrame(report[panel + "::gates"]).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--split", default="calibration", choices=("calibration",),
                        help="Only the calibration split may be used to select a policy.")
    parser.add_argument("--panels", nargs="+", default=["deployed"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--stages", nargs="+", choices=("generate", "score", "select"),
                        default=["generate", "score", "select"])
    parser.add_argument("--gate-rule", choices=("ceiling_and_safeguards", "classifier_gain"),
                        default="ceiling_and_safeguards", help="Run-time rule written into the policy.")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--ci-margin", type=float, default=0.15)
    parser.add_argument("--max-harm-share", type=float, default=0.20)
    parser.add_argument("--min-preference", type=float, default=0.45)
    parser.add_argument("--min-n", type=int, default=8)
    parser.add_argument("--min-strategy-n", type=int, default=10)
    args = parser.parse_args()
    for stage in args.stages:
        {"generate": generate, "score": score, "select": select}[stage](args)


if __name__ == "__main__":
    main()
