#!/usr/bin/env python3
"""Final held-out comparison of the version-3 study (``holdout3``: 180 English + 120 Hinglish messages).

Systems, per model (its own panels only; decisions come from ``results/steering_v3/decisions.json``,
fixed on calibration data before this split is read):
  baseline   MindMate system prompt only
  kg_v2      deployed strategy prompt, no steering
  v2         kg_v2 + frozen deployed steering policy and gate          (Llama, EmoLLaMA)
  kg_v3      the prompt variant chosen for the model (+ degenerate-reply fallback), no steering
  v3         kg_v3 + experimental policy v3 and the gate chosen on calibration (only where a policy exists)

The unsteered reply and its steered candidate share one decoding batch.  Gates are applied from the
recorded candidates exactly as at run time (gate classifier, repetition, length, language, fallback).
Both judges rate every reply; steered replies are also rated for semantic preservation against
their own unsteered reply.

  MINDMATE_MODEL=llama python3 -m generation_analysis.holdout3 --stages generate score report
  python3 -m generation_analysis.holdout3 --stages compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import MODEL_REGISTRY
from chatbot_feature.emotion_reasoner.knowledge_graph import is_hinglish
from chatbot_feature.emotion_reasoner.tone import degenerate_reply
from .common import MODEL_ID, MODEL_KEY, read_jsonl, write_json
from .freeze_v2 import frozen_policy, sha256, verify
from .steering_lab import emotion_rubric, release, repeated_bigram_rate
from .v3_common import (JUDGES, RATED, SHARED, Judge, automatic_scores, generate_reply, holm, judge_preferences,
                        judge_ratings, load_items, messages_of, paired_summary, results_dir, system_prompt)

DECISIONS = SHARED / "decisions.json"
OUT = results_dir(MODEL_KEY) / "holdout3"
STEERED_MODELS = ("llama", "emollama")
PAIRS = (("kg_v2", "baseline"), ("kg_v3", "baseline"), ("kg_v3", "kg_v2"), ("v2", "kg_v2"), ("v3", "kg_v3"), ("v3", "v2"),
         ("v3", "baseline"))
METRICS = ("judge_emotion", *RATED, "hinglish_naturalness")
CHECKS = ("appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness")
V2_GATE = {"ceiling": 0.30, "max_repetition_increase": 0.05, "min_length_ratio": 0.4}


def decisions() -> dict:
    if not DECISIONS.exists():
        raise SystemExit(f"{DECISIONS} is missing: the calibration decisions must be written before the held-out run.")
    return json.loads(DECISIONS.read_text())[MODEL_KEY]


def systems() -> tuple[str, ...]:
    if THINK_SUBSET:
        return ("baseline", "kg_v3")
    return ("baseline", "kg_v2", "v2", "kg_v3", "v3") if MODEL_KEY in STEERED_MODELS else ("baseline", "kg_v2", "kg_v3")


def cell_for(policy: dict | None, row) -> dict:
    if not policy:
        return {"alpha": 0.0}
    language = "hinglish" if is_hinglish(row.text) else "english"
    return (policy["cells"].get(f"strategy:{row.strategy}|{language}") or policy["cells"].get(f"{row.target}|{language}")
            or {"alpha": 0.0})


def chosen_items(args) -> pd.DataFrame:
    items = load_items("holdout3")
    if THINK_SUBSET:                 # reasoning costs ~50 s a reply: the first 60 authored Hinglish and 40 English messages
        items = pd.concat([items[items.item_id.str.startswith("holdout3-hi")].head(60),
                           items[items.language == "english"].head(40)])
    return items.head(args.limit) if args.limit else items


# ---------------------------------------------------------------------- generate

def generate(args) -> None:
    from backend.models import create_runtime
    from .steering_methods import Interventions
    choice = decisions()
    OUT.mkdir(parents=True, exist_ok=True)
    policy_v2 = policy_v3 = None
    if MODEL_KEY in STEERED_MODELS and not THINK_SUBSET:
        verify(MODEL_KEY)
        policy_v2 = frozen_policy(MODEL_KEY)
        if choice.get("policy"):
            policy_v3 = json.loads(Path(choice["policy"]).read_text())
    lock = {"decisions": choice, "items": sha256(SHARED / "holdout3.csv"),
            "policy_v3": sha256(Path(choice["policy"])) if choice.get("policy") else None}
    if (OUT / "lock.json").exists() and json.loads((OUT / "lock.json").read_text()) != lock:
        raise SystemExit("Decisions or policy changed after the held-out generation started.")
    write_json(OUT / "lock.json", lock)
    path = OUT / "generations.jsonl"
    done = {row["item_id"] for row in read_jsonl(path)} if path.exists() else set()
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    steer = None
    if policy_v2 is not None:
        norms_path = results_dir(MODEL_KEY) / "methods" / "layer_norms.json"
        norms = {int(k): v for k, v in json.loads(norms_path.read_text()).items()} if norms_path.exists() else {}
        steer = Interventions(runtime, norms)
    think = THINK
    with path.open("a", encoding="utf-8") as handle:
        for row in chosen_items(args).itertuples():
            if row.item_id in done:
                continue
            messages = messages_of(row)
            record = {"item_id": row.item_id,
                      "baseline": generate_reply(runtime, messages, system_prompt(row, "baseline"), args.max_new_tokens, think)}
            prompts = {"kg_v2": system_prompt(row, "kg_v2"), "kg_v3": system_prompt(row, choice["prompt"])}
            if THINK_SUBSET:
                prompts.pop("kg_v2")
            cells = {"kg_v2": cell_for(policy_v2, row), "kg_v3": cell_for(policy_v3, row)}
            cache: dict[str, dict] = {}
            for name, prompt in prompts.items():
                cell = cells[name]
                if steer is None:
                    entry = {"unsteered": generate_reply(runtime, messages, prompt, args.max_new_tokens, think)}
                else:
                    wanted = [(None, 0.0)]
                    for other, other_prompt in prompts.items():          # identical prompts share one batch
                        if other_prompt == prompt and cells[other]["alpha"]:
                            spec = cells[other].get("spec") or {"method": "additive"}
                            wanted.append((spec, float(cells[other]["alpha"])))
                    key = prompt
                    if key not in cache:
                        cache[key] = {"rows": wanted, "replies": steer.generate_rows(messages, prompt, row.target, wanted,
                                                                                    args.max_new_tokens)}
                    batch = cache[key]
                    entry = {"unsteered": batch["replies"][0]}
                    if cell["alpha"]:
                        spec = cell.get("spec") or {"method": "additive"}
                        index = batch["rows"].index((spec, float(cell["alpha"])))
                        entry.update(candidate=batch["replies"][index], alpha=float(cell["alpha"]),
                                     method=cell.get("method") or "additive")
                record[name] = entry
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(row.item_id, {name: record[name].get("alpha", 0) for name in prompts}, flush=True)
    del steer, runtime
    release()


# ------------------------------------------------------------------------- score

def v2_gate(unsteered: str, steered: str, unsteered_gate: float) -> bool:
    """The deployed run-time rule (``ceiling_and_safeguards``)."""
    return bool(unsteered_gate < V2_GATE["ceiling"]
                and repeated_bigram_rate(steered) - repeated_bigram_rate(unsteered) <= V2_GATE["max_repetition_increase"]
                and len(steered) >= V2_GATE["min_length_ratio"] * len(unsteered))


def quality_gate(unsteered: str, steered: str, unsteered_gate: float) -> bool:
    """v2 rule + no reply more than 1.5x longer, no loss of Hinglish, no degenerate reply."""
    words = max(len(unsteered.split()), 1)
    return bool(v2_gate(unsteered, steered, unsteered_gate) and len(steered.split()) <= 1.5 * words
                and not (is_hinglish(unsteered) and not is_hinglish(steered)) and not degenerate_reply(steered))


def long_table(args) -> pd.DataFrame:
    from .steering_lab import SCORER, free_classifiers, target_scores
    choice = decisions()
    items = chosen_items(args).set_index("item_id")
    records = {row["item_id"]: row for row in read_jsonl(OUT / "generations.jsonl")}
    if set(records) != set(items.index):
        raise SystemExit("Held-out generation is incomplete.")
    order = list(items.index)
    gate = {}
    for name in ("kg_v2", "kg_v3"):
        if name in records[order[0]]:
            gate[name] = dict(zip(order, target_scores([records[i][name]["unsteered"] for i in order],
                                                       [items.loc[i, "target"] for i in order], SCORER)))
    free_classifiers()
    rows = []
    for item_id in order:
        record, item = records[item_id], items.loc[item_id]
        common = {"item_id": item_id, "text": item.text, "history": item.history, "target": item.target,
                  "strategy": item.strategy, "language": item.language, "turns": item.turns}

        def add(system: str, reply: str, status: str = "none", alpha: float = 0.0, reference: str | None = None):
            rows.append({**common, "system": system, "reply": reply, "status": status, "applied_alpha": alpha,
                         "reference": reference})
        add("baseline", record["baseline"])
        for prompt_name, steered_name in (("kg_v2", "v2"), ("kg_v3", "v3")):
            if prompt_name not in record:
                continue
            entry = record[prompt_name]
            unsteered = entry["unsteered"]
            fallback = prompt_name == "kg_v3" and choice.get("guard") and degenerate_reply(unsteered)
            plain = record["baseline"] if fallback else unsteered
            add(prompt_name, plain, status="fallback" if fallback else "none")
            if steered_name not in systems():
                continue
            if "candidate" not in entry or fallback:
                add(steered_name, plain, status="fallback" if fallback else "not_planned")
                continue
            rule = quality_gate if steered_name == "v3" and choice.get("gate") == "quality_gate" else v2_gate
            passed = rule(unsteered, entry["candidate"], gate[prompt_name][item_id])
            add(steered_name, entry["candidate"] if passed else plain, status="applied" if passed else "rejected",
                alpha=entry["alpha"] if passed else 0.0, reference=plain if passed else None)
    table = pd.DataFrame(rows)
    table["failed"] = table.reply.str.startswith("[generation failed")
    table.loc[table.failed, "reply"] = ""
    return table


def score(args) -> None:
    table = automatic_scores(long_table(args))
    table.drop(columns=["history"]).to_csv(OUT / "replies.csv", index=False)
    wide = table.pivot(index="item_id", columns="system", values="reply")
    first_rows = table.drop_duplicates("item_id").set_index("item_id").loc[wide.index].reset_index()
    for name, (model_id, four_bit) in JUDGES.items():
        if (OUT / f"ratings_{name}.csv").exists():
            continue
        judge = Judge(model_id, load_in_4bit=four_bit)
        ratings = judge_ratings(judge, table, reference=[value if isinstance(value, str) else None for value in table.reference])
        contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
        keys = list(zip(contexts, table.reply.fillna(""), [emotion_rubric(target) for target in table.target]))
        unique = list(dict.fromkeys(keys))
        lookup = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings["judge_emotion"] = [float(lookup[key]) for key in keys]
        preferences = pd.DataFrame(index=wide.index)
        for first, second in PAIRS:
            if first in wide and second in wide:
                preferences[f"{first}_over_{second}"] = judge_preferences(judge, first_rows, wide[first].tolist(),
                                                                          wide[second].tolist())
        ratings.to_csv(OUT / f"ratings_{name}.csv", index=False)
        preferences.to_csv(OUT / f"preferences_{name}.csv")
        judge.close()


# ------------------------------------------------------------------------ report

def load_scores(folder: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    table = pd.read_csv(folder / "replies.csv")
    preferences = {}
    for name in JUDGES:
        ratings = pd.read_csv(folder / f"ratings_{name}.csv")
        for column in (*METRICS, "preservation"):
            if column in ratings:
                table[f"{column}_{name}"] = ratings[column].to_numpy()
        preferences[name] = pd.read_csv(folder / f"preferences_{name}.csv", index_col=0)
    for column in (*METRICS, "preservation"):
        parts = [f"{column}_{name}" for name in JUDGES if f"{column}_{name}" in table]
        if parts:
            table[f"{column}_both"] = table[parts].mean(axis=1)
    preferences["both"] = sum(preferences[name] for name in JUDGES) / len(JUDGES)
    return table, preferences


def report(args) -> None:
    from scipy.stats import spearmanr
    table, preferences = load_scores(OUT)
    wide = {name: group.set_index("item_id") for name, group in table.groupby("system")}
    index = wide["baseline"].index
    subsets = {"all": index, "english": index[wide["baseline"].language == "english"],
               "hinglish": index[wide["baseline"].language == "hinglish"]}
    for name in ("v2", "v3"):
        if name in wide:
            subsets[f"{name}_steered"] = index[wide[name].applied_alpha > 0]
    automatic = ("indep_target", "reply_words", "repeated_bigram_rate", "reply_is_hinglish", "degenerate", "questions", "failed")
    means = pd.DataFrame([
        {"subset": subset, "system": name, "n": len(ids),
         **{metric: wide[name].loc[ids, f"{metric}_both"].mean() for metric in METRICS},
         **{column: wide[name].loc[ids, column].astype(float).mean() for column in automatic},
         "steered_share": float((wide[name].loc[ids, "applied_alpha"] > 0).mean())}
        for subset, ids in subsets.items() if len(ids) for name in wide])
    means.to_csv(OUT / "system_means.csv", index=False)

    rows = []
    for subset, ids in subsets.items():
        if len(ids) < 6:
            continue
        for first, second in PAIRS:
            if first not in wide or second not in wide:
                continue
            label = f"{first} - {second}"
            for source in (*JUDGES, "both"):
                for metric in METRICS:
                    a, b = wide[first].loc[ids, f"{metric}_{source}"], wide[second].loc[ids, f"{metric}_{source}"]
                    if a.notna().sum() >= 6:
                        rows.append({"subset": subset, "contrast": label, "metric": metric, "source": source,
                                     **paired_summary(a.to_numpy() - b.to_numpy())})
                column = f"{first}_over_{second}"
                rows.append({"subset": subset, "contrast": label, "metric": "preference_minus_0.5", "source": source,
                             **paired_summary(preferences[source].loc[ids, column] - 0.5)})
            for metric in ("indep_target", "reply_words"):
                rows.append({"subset": subset, "contrast": label, "metric": metric, "source": "automatic",
                             **paired_summary(wide[first].loc[ids, metric].to_numpy() - wide[second].loc[ids, metric].to_numpy())})
    contrasts = pd.DataFrame(rows)
    primary = ((contrasts.subset == "all") & contrasts.source.isin(["both"])
               & contrasts.contrast.isin(["kg_v3 - kg_v2", "v3 - kg_v3", "v3 - v2", "v2 - kg_v2"]))
    contrasts.loc[primary, "p_holm"] = holm(contrasts.loc[primary, "p"])
    contrasts.to_csv(OUT / "contrasts.csv", index=False)

    safety = []
    for name, frame in wide.items():
        entry = {"system": name, "messages": len(index), "steered": int((frame.applied_alpha > 0).sum()),
                 "steered_share": float((frame.applied_alpha > 0).mean()), "status": frame.status.value_counts().to_dict(),
                 "safety_below_3_share": float((frame.safety_both < 3).mean()), "min_safety": float(frame.safety_both.min())}
        reference = {"v2": "kg_v2", "v3": "kg_v3"}.get(name)
        ids = index[frame.applied_alpha > 0]
        if reference and len(ids):
            for source in (*JUDGES, "both"):
                drop = pd.concat([frame.loc[ids, f"{metric}_{source}"] - wide[reference].loc[ids, f"{metric}_{source}"]
                                  for metric in CHECKS], axis=1).min(axis=1)
                entry[f"harmed_share_{source}"] = float((drop < -0.5).mean())
            entry["preservation"] = float(frame.loc[ids, "preservation_both"].mean())
            entry["steered_by_strategy"] = frame.loc[ids, "strategy"].value_counts().to_dict()
        safety.append(entry)
    pd.DataFrame(safety).to_csv(OUT / "safety.csv", index=False)

    distinct = table.drop_duplicates(["item_id", "reply"])
    agreement = []
    for metric in METRICS:
        valid = distinct[f"{metric}_judge1"].notna() & distinct[f"{metric}_judge2"].notna()
        if valid.sum() > 5:
            agreement.append({"metric": metric, "replies": int(valid.sum()),
                              "spearman": spearmanr(distinct.loc[valid, f"{metric}_judge1"], distinct.loc[valid, f"{metric}_judge2"]).statistic,
                              "judge1_mean": distinct.loc[valid, f"{metric}_judge1"].mean(),
                              "judge2_mean": distinct.loc[valid, f"{metric}_judge2"].mean()})
    for column in preferences["judge1"]:
        differs = preferences["judge1"][column] != 0.5
        if differs.sum() > 5:
            a, b = preferences["judge1"].loc[differs, column], preferences["judge2"].loc[differs, column]
            agreement.append({"metric": f"preference {column}", "replies": int(differs.sum()), "spearman": spearmanr(a, b).statistic,
                              "judge1_mean": a.mean(), "judge2_mean": b.mean(),
                              "same_winner_share": float(((a > 0.5) == (b > 0.5)).mean())})
    pd.DataFrame(agreement).to_csv(OUT / "judge_agreement.csv", index=False)
    write_json(OUT / "metadata.json", {
        "model": MODEL_ID, "messages": int(len(index)), "decisions": decisions(), "systems": list(wide),
        "reasoning": {True: "on", False: "skipped (empty reasoning block)", None: "not applicable"}[THINK],
        "judges": {name: value[0] for name, value in JUDGES.items()}, "max_new_tokens": args.max_new_tokens,
        "uncertainty": "paired bootstrap over messages (10,000), Wilcoxon; Holm over the all-message two-judge contrasts "
                       "kg_v3 - kg_v2, v2 - kg_v2, v3 - kg_v3, v3 - v2",
        "harmed": "steered reply whose worst quality or safety rating change against its unsteered reply is below -0.5"})
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    print(means[means.subset.isin(["all", "english", "hinglish"])].round(3).to_string(index=False))
    print(pd.DataFrame(safety).round(3).to_string(index=False))
    view = contrasts[(contrasts.source == "both") & contrasts.subset.isin(["all", "english", "hinglish", "v2_steered", "v3_steered"])]
    view = view.assign(value=view.apply(lambda r: f"{r['mean']:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]", axis=1))
    print(view.pivot_table(index=["subset", "metric"], columns="contrast", values="value", aggfunc="first").to_string())
    print(contrasts[primary][["contrast", "metric", "n", "mean", "ci_low", "ci_high", "p", "p_holm"]].round(4).to_string(index=False))
    print(pd.DataFrame(agreement).round(3).to_string(index=False))


def compare(args) -> None:
    """All models on the same messages: baseline, deployed KG prompt and the chosen prompt, by language."""
    folder = args.out
    tables = {}
    for key in MODEL_REGISTRY:
        path = results_dir(key) / folder
        if (path / "replies.csv").exists() and all((path / f"ratings_{name}.csv").exists() for name in JUDGES):
            table, _ = load_scores(path)
            tables[key] = {name: group.set_index("item_id") for name, group in table.groupby("system")}
    if len(tables) < 2:
        raise SystemExit("Fewer than two models have held-out results.")
    keys = list(tables)
    index = tables[keys[0]]["baseline"].index
    languages = {"english": index[tables[keys[0]]["baseline"].language == "english"],
                 "hinglish": index[tables[keys[0]]["baseline"].language == "hinglish"]}
    automatic = ("reply_is_hinglish", "reply_words", "repeated_bigram_rate", "degenerate", "indep_target")
    levels, differences = [], []
    for language, ids in languages.items():
        for key in keys:
            for system in tables[key]:
                frame = tables[key][system].loc[ids]
                levels.append({"language": language, "model": key, "system": system, "n": len(ids),
                               **{metric: frame[f"{metric}_both"].mean() for metric in METRICS},
                               **{column: frame[column].astype(float).mean() for column in automatic}})
        best = {key: max((system for system in tables[key]), key=lambda system: tables[key][system].loc[ids, "appropriateness_both"].mean()
                         + tables[key][system].loc[ids, "naturalness_both"].mean()) for key in keys}
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                for system in ("baseline", "kg_v2", "kg_v3"):
                    if system not in tables[keys[a]] or system not in tables[keys[b]]:
                        continue
                    for metric in METRICS:
                        x = tables[keys[a]][system].loc[ids, f"{metric}_both"]
                        y = tables[keys[b]][system].loc[ids, f"{metric}_both"]
                        if x.notna().sum() >= 6:
                            differences.append({"language": language, "system": system, "contrast": f"{keys[a]} - {keys[b]}",
                                                "metric": metric, **paired_summary(x.to_numpy() - y.to_numpy())})
    out = SHARED / ("model_comparison" if folder == "holdout3" else f"model_comparison_{folder}")
    pd.DataFrame(levels).to_csv(f"{out}_levels.csv", index=False)
    pd.DataFrame(differences).to_csv(f"{out}_differences.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    print(pd.DataFrame(levels).round(3).to_string(index=False))
    view = pd.DataFrame(differences)
    view = view.assign(value=view.apply(lambda r: f"{r['mean']:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]", axis=1))
    print(view.pivot_table(index=["language", "system", "metric"], columns="contrast", values="value", aggfunc="first").to_string())


THINK: bool | None = None
THINK_SUBSET = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--stages", nargs="+", choices=("generate", "score", "report", "compare"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--think", choices=("on", "off"), help="Sarvam only: reasoning on (100-message subset), or skipped.")
    parser.add_argument("--out", default="holdout3")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    global OUT, THINK, THINK_SUBSET
    OUT = results_dir(MODEL_KEY) / args.out
    THINK = {"on": True, "off": False, None: None}[args.think]
    THINK_SUBSET = args.think == "on"
    for stage in args.stages:
        {"generate": generate, "score": score, "report": report, "compare": compare}[stage](args)


if __name__ == "__main__":
    main()
