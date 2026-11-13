#!/usr/bin/env python3
"""Before/after comparison on the held-out *test* split.

Every reply is produced by a real service object through the unbatched backend
path, exactly as the chat endpoint would:

  baseline   MindMate system prompt only (no knowledge graph, no steering)
  kg_prompt  strategy guidance, no steering (the unsteered candidate of both systems)
  v1         the system deployed before this work: frozen service copy
             (``deployed_v1_service``) + ``policy_v1_deployed.json`` + deployed panels
  v2         the current ``backend.emotion_service`` with the current policy

The test messages are not used anywhere else.  Scores come from evaluators that
neither system's run-time gate uses (Qwen ratings, third-party GoEmotions
classifier); the gate classifier's score is reported only for reference, since
both systems select replies with it.  Uncertainty: paired bootstrap over
messages and Wilcoxon signed-rank tests.

Stages:  v1 -> v2 -> score -> report.  ``v1`` and ``v2`` each load the model.
  HF_HUB_OFFLINE=1 python3 -m generation_analysis.heldout_comparison --stages v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import MODEL_ID, MODEL_KEY, RESULTS_DIR, read_jsonl, write_json
from .steering_lab import (LAB_DIR, QUALITY, SYSTEM_PROMPT, Judge, holm, load_split, messages_of, paired_summary,
                           release, score_table)

OUT = LAB_DIR / "heldout"
SYSTEMS = ("baseline", "kg_prompt", "v1", "v2")
METRICS = ("judge_emotion", "indep_target", *QUALITY)


def run_system(name: str, args) -> None:
    from backend.models import create_runtime
    if name == "v1" and MODEL_KEY != "llama":
        raise SystemExit("The frozen previous system (v1) exists only for Llama 3.1-8B. For other models run "
                         "--stages v2 score report: baseline, strategy prompt and the current system are compared.")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = {row["item_id"] for row in read_jsonl(path)} if path.exists() else set()
    items = load_split("test")
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    if name == "v1":
        from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
        from chatbot_feature.emotion_reasoner.agent import DETECTOR_WEIGHT
        from .deployed_v1_service import EmotionSteeringService
        policy = json.loads((RESULTS_DIR / "steering_policy" / "policy_v1_deployed.json").read_text())
        agent = EmotionReasonerAgent(available_panels=list(runtime._emotion_panels), policy=policy,
                                     detector_weight=DETECTOR_WEIGHT)
        service = EmotionSteeringService(runtime, agent=agent)
    else:
        from backend.emotion_service import EmotionSteeringService
        service = EmotionSteeringService(runtime)
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            if row.item_id in done:
                continue
            messages = messages_of(row)
            record = {"item_id": row.item_id}
            if name == "v1" or MODEL_KEY != "llama":   # the plain baseline is generated once, with the first system run
                started = time.perf_counter()
                record["baseline_reply"] = runtime.generate(
                    messages, max_new_tokens=args.max_new_tokens, deterministic=True, temperature=0.7,
                    system_prompt=SYSTEM_PROMPT)["response"]
                record["baseline_latency"] = round(time.perf_counter() - started, 3)
            started = time.perf_counter()
            turn_id, plan = service.begin_turn(messages)
            analysis = service.respond(turn_id, max_new_tokens=args.max_new_tokens)
            record.update({
                "latency": round(time.perf_counter() - started, 3), "strategy": plan["strategy"],
                "target": plan["response_emotion"], "user_emotion": plan["user_emotion"],
                "status": analysis["decision"]["status"], "planned_alpha": analysis["decision"]["planned_alpha"],
                "applied_alpha": analysis["decision"]["applied_alpha"], "reply": analysis["reply"],
                "unsteered_reply": analysis["effect"]["unsteered_reply"],
                "steered_reply": analysis["effect"]["steered_reply"],
                "first_token_latency": analysis.get("first_token_seconds"),
            })
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{name} {row.item_id}: {plan['strategy']} -> {record['status']} (alpha {record['applied_alpha']})",
                  flush=True)
    del service, runtime
    release()


def score(args) -> None:
    items = load_split("test").set_index("item_id")
    v2 = pd.DataFrame(read_jsonl(OUT / "v2.jsonl")).set_index("item_id")
    if (OUT / "v1.jsonl").exists():
        v1 = pd.DataFrame(read_jsonl(OUT / "v1.jsonl")).set_index("item_id")
    else:
        # No previous system for this model: its row repeats the unsteered strategy-prompt reply, so
        # "v1" contrasts are exactly zero and only baseline / kg_prompt / v2 carry information.
        v1 = v2.assign(reply=v2.unsteered_reply, status="none", applied_alpha=0.0, latency=np.nan)
    if set(v1.index) != set(items.index) or set(v2.index) != set(items.index):
        raise SystemExit("v1/v2 runs are incomplete.")
    rows = []
    for item_id, item in items.iterrows():
        common = {"item_id": item_id, "text": item.text, "history": item.history, "target": item.target,
                  "strategy": item.strategy, "language": item.language, "turns": item.turns}
        if v1.loc[item_id, "target"] != item.target or v2.loc[item_id, "target"] != item.target:
            raise SystemExit(f"Plan mismatch for {item_id}; the systems must share the knowledge-graph plan.")
        rows.append({**common, "system": "baseline", "reply": v1.loc[item_id, "baseline_reply"], "status": "none",
                     "applied_alpha": 0.0, "latency": v1.loc[item_id, "baseline_latency"]})
        rows.append({**common, "system": "kg_prompt", "reply": v1.loc[item_id, "unsteered_reply"], "status": "none",
                     "applied_alpha": 0.0, "latency": np.nan})
        for name, run in (("v1", v1), ("v2", v2)):
            rows.append({**common, "system": name, "reply": run.loc[item_id, "reply"],
                         "status": run.loc[item_id, "status"], "applied_alpha": run.loc[item_id, "applied_alpha"],
                         "latency": run.loc[item_id, "latency"],
                         "first_token_latency": run.loc[item_id].get("first_token_latency")})
    table = pd.DataFrame(rows)
    judge = Judge()
    table = score_table(table, judge=judge)
    # Order-balanced pairwise preferences between systems (0.5 when the two replies are identical).
    contexts = [Judge.context(item.text, item.history) for item in items.itertuples()]
    reply = {name: table[table.system == name].set_index("item_id").loc[items.index, "reply"].tolist()
             for name in SYSTEMS}
    preferences = {}
    for first, second in (("v2", "v1"), ("v2", "kg_prompt"), ("v1", "kg_prompt"), ("v2", "baseline"),
                          ("v1", "baseline"), ("kg_prompt", "baseline")):
        differs = [i for i in range(len(items)) if reply[first][i] != reply[second][i]]
        values = np.full(len(items), 0.5)
        if differs:
            values[differs] = judge.prefer([contexts[i] for i in differs], [reply[first][i] for i in differs],
                                           [reply[second][i] for i in differs])
        preferences[f"{first}_over_{second}"] = values
    judge.close()
    pd.DataFrame(preferences, index=items.index).to_csv(OUT / "preferences.csv")
    table.drop(columns=["history"]).to_csv(OUT / "scores.csv", index=False)


def report(args) -> None:
    table = pd.read_csv(OUT / "scores.csv")
    preferences = pd.read_csv(OUT / "preferences.csv", index_col=0)
    wide = {name: group.set_index("item_id") for name, group in table.groupby("system")}
    index = wide["baseline"].index
    subsets = {"all": index, "english": index[wide["baseline"].language == "english"],
               "hinglish": index[wide["baseline"].language == "hinglish"],
               "english_multi_turn": index[(wide["baseline"].language == "english") & (wide["baseline"].turns == "multi")]}
    for name in ("v1", "v2"):
        subsets[f"{name}_steered"] = index[wide[name].loc[index, "applied_alpha"] > 0]

    means = []
    for subset, ids in subsets.items():
        for name in SYSTEMS:
            part = wide[name].loc[ids]
            means.append({"subset": subset, "system": name, "n": len(part),
                          **{metric: part[metric].mean() for metric in (*METRICS, "gate_target")},
                          "steered_share": float((part.applied_alpha > 0).mean()),
                          "latency_mean": part.latency.mean(), "latency_median": part.latency.median(),
                          "first_token_median": part.first_token_latency.median()
                          if "first_token_latency" in part and part.first_token_latency.notna().any() else np.nan})
    pd.DataFrame(means).to_csv(OUT / "system_means.csv", index=False)

    contrasts = []
    for subset, ids in subsets.items():
        if not len(ids):
            continue
        for first, second in (("v2", "v1"), ("v2", "kg_prompt"), ("v1", "kg_prompt"), ("v2", "baseline"),
                              ("v1", "baseline"), ("kg_prompt", "baseline")):
            for metric in (*METRICS, "gate_target"):
                stats = paired_summary(wide[first].loc[ids, metric] - wide[second].loc[ids, metric])
                contrasts.append({"subset": subset, "contrast": f"{first} - {second}", "metric": metric, **stats})
            stats = paired_summary(preferences.loc[ids, f"{first}_over_{second}"] - 0.5)
            contrasts.append({"subset": subset, "contrast": f"{first} - {second}", "metric": "preference_minus_0.5",
                              **stats})
            if "latency" in wide[first] and wide[first].latency.notna().all() and wide[second].latency.notna().all():
                stats = paired_summary(wide[first].loc[ids, "latency"] - wide[second].loc[ids, "latency"])
                contrasts.append({"subset": subset, "contrast": f"{first} - {second}", "metric": "latency_seconds",
                                  **stats})
    contrasts = pd.DataFrame(contrasts)
    primary = (contrasts.subset == "all") & contrasts.metric.isin([*METRICS, "preference_minus_0.5"])
    contrasts.loc[primary, "p_holm"] = holm(contrasts.loc[primary, "p"])
    contrasts.to_csv(OUT / "contrasts.csv", index=False)

    status = pd.concat({name: wide[name].status.value_counts() for name in ("v1", "v2")}, axis=1).fillna(0).astype(int)
    status.to_csv(OUT / "decision_counts.csv")
    write_json(OUT / "metadata.json", {
        "split": "test (EmpatheticDialogues test + Hinglish odd rows); used only here",
        "messages": int(len(index)), "max_new_tokens": args.max_new_tokens, "decoding": "greedy, unbatched backend path",
        "independent_evaluators": ["judge_emotion (Qwen 1-5 rating of the target emotion)",
                                   "indep_target (SamLowe/roberta-base-go_emotions)"],
        "gate_metric": "gate_target (XLM-R; both systems select replies with it, so it is biased upward)",
        "uncertainty": "percentile bootstrap over messages, 10,000 resamples; Wilcoxon signed-rank; "
                       "Holm correction over the primary (all-message) contrasts",
    })
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(pd.DataFrame(means).round(3).to_string(index=False))
    print(status.to_string())
    shown = contrasts[contrasts.subset.isin(["all", "english", "v1_steered", "v2_steered"])
                      & contrasts.contrast.isin(["v2 - v1", "v2 - kg_prompt", "v1 - kg_prompt"])]
    print(shown.round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--stages", nargs="+", choices=("v1", "v2", "score", "report"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()
    for stage in args.stages:
        if stage in ("v1", "v2"):
            run_system(stage, args)
        else:
            {"score": score, "report": report}[stage](args)


if __name__ == "__main__":
    main()
