#!/usr/bin/env python3
"""Steering-method comparison for the selected model (its own panels and residual means only).

  screen     development messages, judge 1 only: additive, bounded, selective-layer and residual
             (contrastive-direction) steering at a few strengths; one configuration per method
             family is kept (reliable emotion gain, then the highest pairwise preference)
  calibrate  calibration messages, both judges: a four-strength sweep of each kept configuration
  select     experimental policy: per context the method and strength that both judges find
             quality-safe and helpful, or no steering; run-time safeguards compared offline

Every steered reply is compared with the unsteered reply to the same prompt, decoded in the same
batch.  The prompt is the version chosen by ``v3_prompts`` (``--prompt``).  Nothing is installed.

  MINDMATE_MODEL=llama python3 -m generation_analysis.v3_methods --stages screen calibrate select --prompt kg_v3
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .common import MODEL_ID, MODEL_KEY, read_jsonl, write_json
from .freeze_v2 import frozen_policy
from .steering_lab import emotion_rubric, release
from .steering_methods import Interventions, layer_norms, spec_name
from .v3_common import (JUDGES, RATED, Judge, automatic_scores, judge_preferences, judge_ratings, load_items,
                        messages_of, paired_summary, results_dir, system_prompt, with_judges)

OUT = results_dir(MODEL_KEY) / "methods"
QUALITY_V3 = ("appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness")
GATE = {"ceiling": 0.30, "max_repetition_increase": 0.05, "min_length_ratio": 0.4, "max_length_ratio": 1.5}
RULE = {"tolerance": 0.05, "ci_margin": 0.15, "max_harm_share": 0.20, "min_preference_each": 0.45,
        "min_preference_mean": 0.50, "min_n": 8, "min_strategy_n": 10}


# ---------------------------------------------------------------- configurations

def screen_configs(runtime) -> list[tuple[dict, list[float]]]:
    layers = sorted(next(iter(runtime._emotion_panels.values())))
    third = max(len(layers) // 3, 1)
    bands = [layers[:third], layers[third:2 * third], layers[2 * third:]]
    depth = len(runtime.model.model.layers)
    configs: list[tuple[dict, list[float]]] = [({"method": "additive"}, [0.5, 1.0]),
                                                ({"method": "bounded", "cap": 0.1}, [1.0, 1.5, 2.0])]
    configs += [({"method": "selective", "layers": band}, [1.0, 2.0]) for band in bands]
    configs += [({"method": "residual", "layers": [round(depth * share)], "reference": "neutral"}, [0.3, 0.6])
                for share in (0.25, 0.375, 0.5)]
    configs.append(({"method": "residual", "layers": [round(depth * 0.375)], "reference": "others"}, [0.3, 0.6]))
    return configs


def family(spec: dict) -> str:
    return spec["method"]


def calibration_grid(spec: dict, alpha: float) -> list[float]:
    if spec["method"] == "additive":
        return [0.25, 0.5, 0.75, 1.0]
    return [round(alpha * factor, 3) for factor in (0.5, 0.75, 1.0, 1.5)]


# -------------------------------------------------------------------- generation

def run(stage: str, items: pd.DataFrame, configs: list[tuple[dict, list[float]]], args) -> None:
    from backend.models import create_runtime
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stage}_generations.jsonl"
    done = {row["item_id"] for row in read_jsonl(path)} if path.exists() else set()
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    configs = configs or screen_configs(runtime)
    norms_path = OUT / "layer_norms.json"
    if norms_path.exists():
        norms = {int(layer): value for layer, value in json.loads(norms_path.read_text()).items()}
    else:
        layers = sorted({layer for panel in runtime._emotion_panels.values() for layer in panel})
        development = load_items("dev").head(12)
        norms = layer_norms(runtime, [(messages_of(row), system_prompt(row, args.prompt)) for row in development.itertuples()],
                            layers)
        write_json(norms_path, {str(layer): value for layer, value in norms.items()})
    steer = Interventions(runtime, norms)
    rows = [(None, 0.0)] + [(spec, alpha) for spec, alphas in configs for alpha in alphas]
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            if row.item_id in done:
                continue
            replies = steer.generate_rows(messages_of(row), system_prompt(row, args.prompt), row.target, rows,
                                          args.max_new_tokens)
            record = {"item_id": row.item_id, "prompt": args.prompt, "replies": [
                {"config": spec_name(spec) if spec else "none", "spec": spec, "alpha": alpha, "reply": reply}
                for (spec, alpha), reply in zip(rows, replies)]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(stage, row.item_id, flush=True)
    del steer, runtime
    release()


def long_table(stage: str, split: str) -> pd.DataFrame:
    items = load_items(split).set_index("item_id")
    rows = []
    for record in read_jsonl(OUT / f"{stage}_generations.jsonl"):
        item = items.loc[record["item_id"]]
        unsteered = record["replies"][0]["reply"]
        for entry in record["replies"]:
            rows.append({"item_id": record["item_id"], "config": entry["config"], "alpha": entry["alpha"],
                         "family": entry["spec"]["method"] if entry["spec"] else "none",
                         "spec": json.dumps(entry["spec"], sort_keys=True), "reply": entry["reply"],
                         "unsteered_reply": unsteered, "text": item.text, "history": item.history, "target": item.target,
                         "strategy": item.strategy, "language": item.language, "turns": item.turns})
    return pd.DataFrame(rows)


def score_stage(stage: str, split: str, judges: tuple[str, ...]) -> None:
    table = automatic_scores(long_table(stage, split))
    table.drop(columns=["history"]).to_csv(OUT / f"{stage}_replies.csv", index=False)
    steered = table.config != "none"
    for name in judges:
        if (OUT / f"{stage}_ratings_{name}.csv").exists():
            continue
        model_id, four_bit = JUDGES[name]
        judge = Judge(model_id, load_in_4bit=four_bit)
        ratings = judge_ratings(judge, table, reference=[row.unsteered_reply if row.config != "none" else None
                                                         for row in table.itertuples()])
        contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
        keys = list(zip(contexts, table.reply.fillna(""), [emotion_rubric(target) for target in table.target]))
        unique = list(dict.fromkeys(keys))
        lookup = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings["judge_emotion"] = [float(lookup[key]) for key in keys]
        preference = np.full(len(table), 0.5)
        preference[steered.to_numpy()] = judge_preferences(judge, table[steered], table.reply[steered].tolist(),
                                                           table.unsteered_reply[steered].tolist())
        ratings["preference"] = preference
        ratings.to_csv(OUT / f"{stage}_ratings_{name}.csv", index=False)
        judge.close()


# ---------------------------------------------------------------------- analysis

def paired_table(stage: str, judges: tuple[str, ...]) -> pd.DataFrame:
    """Steered rows with ``d_<metric>_<source>`` = steered - unsteered, for each judge and their mean."""
    table = pd.read_csv(OUT / f"{stage}_replies.csv")
    metrics = ("judge_emotion", *RATED, "hinglish_naturalness")
    for name in judges:
        ratings = pd.read_csv(OUT / f"{stage}_ratings_{name}.csv")
        for column in (*metrics, "preference", "preservation"):
            table[f"{column}_{name}"] = ratings[column].to_numpy()
    for column in (*metrics, "preference", "preservation"):
        table[f"{column}_both"] = table[[f"{column}_{name}" for name in judges]].mean(axis=1)
    reference = table[table.config == "none"].set_index("item_id")
    rows = table[table.config != "none"].copy()
    for source in (*judges, "both"):
        for metric in metrics:
            rows[f"d_{metric}_{source}"] = rows[f"{metric}_{source}"].to_numpy() - reference.loc[rows.item_id, f"{metric}_{source}"].to_numpy()
    for column in ("indep_target", "gate_target", "repeated_bigram_rate", "questions"):
        rows[f"d_{column}"] = rows[column].to_numpy() - reference.loc[rows.item_id, column].to_numpy()
    rows["unsteered_gate"] = reference.loc[rows.item_id, "gate_target"].to_numpy()
    rows["length_ratio"] = rows.reply_words.to_numpy() / np.maximum(reference.loc[rows.item_id, "reply_words"].to_numpy(), 1)
    rows["unsteered_is_hinglish"] = reference.loc[rows.item_id, "reply_is_hinglish"].to_numpy()
    rows["unsteered_degenerate"] = reference.loc[rows.item_id, "degenerate"].to_numpy().astype(bool)
    return rows


def summarise(group: pd.DataFrame, source: str) -> dict:
    """Outcome of one configuration x strength on one set of messages, for one judge (or ``both``)."""
    emotion = paired_summary(group[f"d_judge_emotion_{source}"])
    quality = [metric for metric in QUALITY_V3]
    if group.language.eq("hinglish").all():
        quality.append("hinglish_naturalness")
    entry = {"n": int(len(group)), "emotion_gain": emotion["mean"], "emotion_gain_lo": emotion["ci_low"],
             "emotion_gain_hi": emotion["ci_high"], "independent_gain": float(group.d_indep_target.mean()),
             "preference": float(group[f"preference_{source}"].mean()),
             "preservation": float(group[f"preservation_{source}"].mean()),
             "length_ratio": float(group.length_ratio.mean()), "repetition_change": float(group.d_repeated_bigram_rate.mean())}
    drops, failed = [], []
    for metric in quality:
        stats = paired_summary(group[f"d_{metric}_{source}"])
        entry[f"{metric}_change"] = stats["mean"]
        entry[f"{metric}_change_lo"] = stats["ci_low"]
        if stats["mean"] < -RULE["tolerance"] or stats["ci_low"] < -RULE["ci_margin"]:
            failed.append(metric)
        drops.append(group[f"d_{metric}_{source}"])
    entry["harm_share"] = float((pd.concat(drops, axis=1).min(axis=1) < -0.5).mean())
    entry["quality_failed"] = " ".join(failed)
    entry["quality_ok"] = not failed and entry["harm_share"] <= RULE["max_harm_share"]
    return entry


def screen_report(args) -> None:
    rows = paired_table("screen", ("judge1",))
    summary = []
    for (config, alpha), group in rows.groupby(["config", "alpha"]):
        for language, part in (("all", group), ("english", group[group.language == "english"]),
                               ("hinglish", group[group.language == "hinglish"])):
            summary.append({"config": config, "family": group.family.iloc[0], "spec": group.spec.iloc[0], "alpha": alpha,
                            "language": language, **summarise(part, "judge1")})
    for target, block in rows[rows.language == "english"].groupby("target"):
        for (config, alpha), group in block.groupby(["config", "alpha"]):
            summary.append({"config": config, "family": group.family.iloc[0], "spec": group.spec.iloc[0], "alpha": alpha,
                            "language": f"english:{target}", **summarise(group, "judge1")})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "screen_summary.csv", index=False)
    english = summary[summary.language == "english"]
    chosen = {}
    for name, block in english.groupby("family"):
        reliable = block[block.emotion_gain_lo > 0]
        pool = reliable if len(reliable) else block
        best = pool.sort_values(["preference", "emotion_gain"], ascending=False).iloc[0]
        chosen[name] = {"config": best.config, "spec": json.loads(best.spec), "alpha": float(best.alpha),
                        "reliable_gain": bool(len(reliable)), "grid": calibration_grid(json.loads(best.spec), float(best.alpha))}
    write_json(OUT / "screen_selection.json", {
        "rule": "per method family: among strengths with a reliable emotion gain on English development messages "
                "(judge 1, 95% CI above 0), the one the judge prefers most over the unsteered reply",
        "prompt": args.prompt, "chosen": chosen})
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    columns = ["config", "alpha", "language", "n", "emotion_gain", "emotion_gain_lo", "appropriateness_change",
               "relevance_change", "fluency_change", "naturalness_change", "preference", "preservation", "length_ratio",
               "harm_share"]
    print(summary[summary.language.isin(["english", "hinglish"])][columns].round(2).to_string(index=False))
    print(json.dumps({name: (value["config"], value["alpha"]) for name, value in chosen.items()}, indent=1))


def cells_of(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {f"{target}|{language}": cell for (target, language), cell in rows.groupby(["target", "language"])}
    groups.update({f"strategy:{strategy}|{language}": cell for (strategy, language), cell in rows.groupby(["strategy", "language"])
                   if cell.item_id.nunique() >= RULE["min_strategy_n"]})
    return groups


def gate_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Which steered candidates each run-time rule would send."""
    old = ((rows.unsteered_gate < GATE["ceiling"]) & (rows.d_repeated_bigram_rate <= GATE["max_repetition_increase"])
           & (rows.length_ratio >= GATE["min_length_ratio"]))
    length = rows.length_ratio <= GATE["max_length_ratio"]
    language = ~(rows.unsteered_is_hinglish.astype(bool) & ~rows.reply_is_hinglish.astype(bool))
    usable = ~rows.degenerate.astype(bool)
    return {"always": np.ones(len(rows), dtype=bool), "v2_gate": old.to_numpy(),
            "v2_gate+length_cap": (old & length).to_numpy(), "v2_gate+language": (old & language).to_numpy(),
            "quality_gate": (old & length & language & usable).to_numpy()}


def select(args) -> None:
    judges = tuple(JUDGES)
    rows = paired_table("calibrate", judges)
    # A degenerate unsteered reply (a bare number, a fragment) is replaced by the plain-prompt reply at
    # run time and never steered, so those messages say nothing about steering a normal reply.
    excluded = int(rows[rows.unsteered_degenerate].item_id.nunique())
    rows = rows[~rows.unsteered_degenerate]
    print(f"messages left out because the unsteered reply was degenerate: {excluded}")
    sweep = []
    for context, cell in cells_of(rows).items():
        for (config, spec), block in cell.groupby(["config", "spec"]):
            alphas = sorted(block.alpha.unique())
            entries = []
            for alpha in alphas:
                group = block[block.alpha == alpha]
                per = {source: summarise(group, source) for source in (*judges, "both")}
                safe = all(per[name]["quality_ok"] and per[name]["preference"] >= RULE["min_preference_each"] for name in judges)
                effective = all(per[name]["emotion_gain_lo"] > 0 for name in judges) and per["both"]["independent_gain"] > 0
                helpful = (per["both"]["preference"] >= RULE["min_preference_mean"]
                           and per["both"]["appropriateness_change"] >= 0)
                entries.append({"context": context, "config": config, "family": block.family.iloc[0], "spec": spec,
                                "alpha": float(alpha), "safe": bool(safe), "effective": bool(effective), "helpful": bool(helpful),
                                **{f"{key}_{source}": value for source in (*judges, "both") for key, value in per[source].items()
                                   if key in ("emotion_gain", "emotion_gain_lo", "preference", "harm_share", "quality_failed",
                                              "appropriateness_change", "naturalness_change", "relevance_change",
                                              "fluency_change", "preservation")},
                                "n": per["both"]["n"], "independent_gain": per["both"]["independent_gain"],
                                "length_ratio": per["both"]["length_ratio"]})
            for index, entry in enumerate(entries):
                entry["admissible"] = bool(entry["safe"] and entry["effective"] and entry["helpful"]
                                           and index + 1 < len(entries) and entries[index + 1]["safe"]
                                           and entry["n"] >= RULE["min_n"])
            sweep += entries
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "calibration_sweep.csv", index=False)

    cells, table = {}, []
    v2 = frozen_policy(MODEL_KEY) if MODEL_KEY in ("llama", "emollama") else {"cells": {}}
    for context, block in sweep.groupby("context"):
        admissible = block[block.admissible]
        entry = {"alpha": 0.0, "method": None, "spec": None}
        if len(admissible):
            best = admissible.sort_values(["preference_both", "emotion_gain_both"], ascending=False).iloc[0]
            entry = {"alpha": float(best.alpha), "method": best.config, "spec": json.loads(best.spec),
                     "reason": (f"both judges: quality-safe, emotion gain +{best.emotion_gain_both:.2f}, preferred over the "
                                f"unsteered reply ({best.preference_both:.2f}), {int(best.n)} calibration messages")}
        else:
            blockers = [name for name, column in (("not quality-safe for both judges", "safe"),
                                                  ("no reliable emotion gain for both judges", "effective"),
                                                  ("not preferred or less appropriate", "helpful")) if not block[column].any()]
            entry["reason"] = "not steered: " + ("; ".join(blockers) if blockers else
                                                 "no strength passed every criterion with a safe stronger neighbour")
        cells[context] = entry
        table.append({"context": context, "n": int(block.n.max()), "v2_alpha": v2["cells"].get(context, {}).get("alpha"),
                      "v3_method": entry["method"], "v3_alpha": entry["alpha"], "reason": entry["reason"]})
    table = pd.DataFrame(table)
    table.to_csv(OUT / "selection.csv", index=False)

    # Method comparison at matched contexts: best admissible-or-not operating point per family on English messages.
    english = rows[rows.language == "english"]
    families = []
    for (config, alpha), group in rows.groupby(["config", "alpha"]):
        for language, part in (("english", group[group.language == "english"]), ("hinglish", group[group.language == "hinglish"])):
            both = summarise(part, "both")
            families.append({"config": config, "family": group.family.iloc[0], "alpha": alpha, "language": language,
                             **{key: both[key] for key in ("n", "emotion_gain", "emotion_gain_lo", "emotion_gain_hi",
                                                           "independent_gain", "preference", "preservation", "harm_share",
                                                           "length_ratio", "appropriateness_change", "empathy_change",
                                                           "relevance_change", "fluency_change", "safety_change",
                                                           "naturalness_change", "quality_ok")},
                             "hinglish_naturalness_change": both.get("hinglish_naturalness_change")})
    for target, block in english.groupby("target"):
        for (config, alpha), group in block.groupby(["config", "alpha"]):
            both = summarise(group, "both")
            families.append({"config": config, "family": group.family.iloc[0], "alpha": alpha, "language": f"english:{target}",
                             **{key: both[key] for key in ("n", "emotion_gain", "emotion_gain_lo", "emotion_gain_hi",
                                                           "independent_gain", "preference", "preservation", "harm_share",
                                                           "length_ratio", "appropriateness_change", "empathy_change",
                                                           "relevance_change", "fluency_change", "safety_change",
                                                           "naturalness_change", "quality_ok")}})
    families = pd.DataFrame(families)
    families.to_csv(OUT / "method_comparison.csv", index=False)

    # Run-time safeguards, offline: candidates the policy would generate, under each sending rule.
    def cell_of(row) -> dict:
        return cells.get(f"strategy:{row.strategy}|{row.language}") or cells[f"{row.target}|{row.language}"]
    planned = rows[[cell_of(row)["method"] == row.config and cell_of(row)["alpha"] == row.alpha for row in rows.itertuples()]]
    gates = []
    pools = {"policy_candidates": planned}
    for (config, alpha), group in rows.groupby(["config", "alpha"]):
        if group.family.iloc[0] == "additive" and alpha in (0.5, 1.0):
            pools[f"all_messages_{config}@{alpha:g}"] = group
    for pool_name, pool in pools.items():
        for name, mask in gate_masks(pool).items():
            sent = pool[mask]
            entry = {"pool": pool_name, "rule": name, "candidates": len(pool), "sent": int(mask.sum())}
            for metric in ("judge_emotion", *QUALITY_V3):
                entry[metric] = float(sent[f"d_{metric}_both"].mean()) if len(sent) else float("nan")
            entry["preference"] = float(sent.preference_both.mean()) if len(sent) else float("nan")
            entry["harmed"] = int((sent[[f"d_{metric}_both" for metric in QUALITY_V3]].min(axis=1) < -0.5).sum())
            gates.append(entry)
    gates = pd.DataFrame(gates)
    gates.to_csv(OUT / "gate_rules.csv", index=False)

    write_json(OUT / "policy_v3.json", {
        "source": "generation_analysis.v3_methods", "version": "experimental-v3", "experimental": True,
        "model": MODEL_ID, "prompt": args.prompt, "split": "calibration (+ extra Hinglish)", "rule": RULE,
        "selection": "per context: both judges quality-safe (appropriateness, empathy, relevance, fluency, safety, "
                     "naturalness; Hinglish naturalness for Hinglish), reliable emotion gain for both judges and the "
                     "independent classifier, preferred over the unsteered reply without loss of appropriateness, next "
                     "stronger strength safe; among those, the most preferred",
        "gate": {"rule": "quality_gate", **GATE, "language_consistency": True, "degenerate_fallback": True},
        "cells": cells})
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 90)
    print(table.to_string(index=False))
    shown = families[families.language.isin(["english", "hinglish"])]
    print(shown[["config", "alpha", "language", "n", "emotion_gain", "emotion_gain_lo", "preference", "preservation",
                 "appropriateness_change", "relevance_change", "fluency_change", "naturalness_change", "harm_share",
                 "length_ratio", "quality_ok"]].round(2).to_string(index=False))
    print(gates.round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--stages", nargs="+", required=True,
                        choices=("screen", "screen_score", "screen_report", "calibrate", "calibrate_score", "select"))
    parser.add_argument("--prompt", default="kg_v3", choices=("kg_v2", "kg_v3", "kg_style", "kg_tone"))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()
    for stage in args.stages:
        if stage == "screen":
            run("screen", load_items("dev"), None, args)
        elif stage == "screen_score":
            score_stage("screen", "dev", ("judge1",))
        elif stage == "screen_report":
            screen_report(args)
        elif stage == "calibrate":
            chosen = json.loads((OUT / "screen_selection.json").read_text())
            if chosen["prompt"] != args.prompt:
                raise SystemExit("The screening used a different prompt version.")
            run("calibrate", load_items("calibration"),
                [(value["spec"], value["grid"]) for value in chosen["chosen"].values()], args)
        elif stage == "calibrate_score":
            score_stage("calibrate", "calibration", tuple(JUDGES))
        else:
            select(args)


if __name__ == "__main__":
    main()
