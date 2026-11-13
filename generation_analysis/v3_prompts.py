#!/usr/bin/env python3
"""Prompt ablation on the calibration messages (no steering): does the version-3 Reasoner prompt help?

Variants, each generated for every calibration message by the selected model:
  baseline   MindMate system prompt only
  kg_v2      + strategy guidance (deployed)
  kg_tone    + strategy guidance + response tone                      (ablation)
  kg_style   + strategy guidance + style and Hinglish rules           (ablation)
  kg_v3      + strategy guidance + tone + style and Hinglish rules
  kg_v3_inline   kg_v3 text placed inside the user turn instead of before the conversation
                 (EmoLLaMA only: its "Human:/Assistant:" format has no system role)
Derived without generation: ``<variant>_guard`` replaces a degenerate reply (a bare number, a
fragment) by the baseline reply, which is what a regenerate-with-the-plain-prompt fallback returns.

Both judges rate emotional appropriateness, empathy, relevance, fluency, safety, conversational
naturalness and (Hinglish messages) Hinglish naturalness, and compare variants pairwise.

  MINDMATE_MODEL=llama python3 -m generation_analysis.v3_prompts --stages generate score report
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from chatbot_feature.emotion_reasoner.tone import degenerate_reply
from .common import MODEL_ID, MODEL_KEY, read_jsonl, write_json
from .steering_lab import release
from .v3_common import (JUDGES, RATED, Judge, automatic_scores, generate_reply, contrast, holm, judge_preferences, judge_ratings,
                        load_items, messages_of, results_dir, system_prompt, with_judges)

OUT = results_dir(MODEL_KEY) / "prompts"
VARIANTS = ("baseline", "kg_v2", "kg_tone", "kg_style", "kg_v3")
SELECTED: tuple[str, ...] | None = None
THINK: bool | None = None              # Sarvam only: False = reasoning skipped, True = reasoning on
INLINE = "kg_v3_inline"
METRICS = (*RATED, "hinglish_naturalness")


def variants() -> tuple[str, ...]:
    if SELECTED:
        return SELECTED
    return (*VARIANTS, INLINE) if MODEL_KEY == "emollama" else VARIANTS


def chosen_items(args) -> pd.DataFrame:
    items = load_items(args.split)
    return items[items.language == args.language] if args.language else items


def reply_for(runtime, row, variant: str, max_new_tokens: int) -> str:
    messages = messages_of(row)
    if variant == INLINE:
        prompt = system_prompt(row, "kg_v3")
        messages = [*messages[:-1], {"role": "user", "content": f"{prompt}\n\nUser message: {row.text}"}]
        prompt = None
    else:
        prompt = system_prompt(row, variant)
    return generate_reply(runtime, messages, prompt, max_new_tokens, THINK)


def generate(args) -> None:
    from backend.models import create_runtime
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "generations.jsonl"
    done = {(row["item_id"], row["variant"]) for row in read_jsonl(path)} if path.exists() else set()
    items = chosen_items(args)
    if args.limit:
        items = items.head(args.limit)
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            for variant in variants():
                if (row.item_id, variant) in done:
                    continue
                reply = reply_for(runtime, row, variant, args.max_new_tokens)
                handle.write(json.dumps({"item_id": row.item_id, "variant": variant, "reply": reply}, ensure_ascii=False) + "\n")
                handle.flush()
            print(row.item_id, flush=True)
    del runtime
    release()


def table_of(args) -> pd.DataFrame:
    items = chosen_items(args).set_index("item_id")
    table = pd.DataFrame(read_jsonl(OUT / "generations.jsonl")).drop_duplicates(["item_id", "variant"])
    table = table[table.variant.isin(variants()) & table.item_id.isin(items.index)]
    for column in ("text", "history", "target", "strategy", "language", "turns"):
        table[column] = [items.loc[item, column] for item in table.item_id]
    table["failed"] = table.reply.str.startswith("[generation failed")
    table.loc[table.failed, "reply"] = ""
    return table.sort_values(["item_id", "variant"]).reset_index(drop=True)


def score(args) -> None:
    table = automatic_scores(table_of(args))
    table.drop(columns=["history"]).to_csv(OUT / "replies.csv", index=False)
    wide = table.pivot(index="item_id", columns="variant", values="reply")
    first_rows = table.drop_duplicates("item_id").set_index("item_id").loc[wide.index].reset_index()
    for name, (model_id, four_bit) in JUDGES.items():
        if (OUT / f"ratings_{name}.csv").exists():
            continue
        judge = Judge(model_id, load_in_4bit=four_bit)
        ratings = judge_ratings(judge, table)
        preferences = pd.DataFrame(index=wide.index)
        for variant in variants()[1:]:
            preferences[f"{variant}_over_baseline"] = judge_preferences(judge, first_rows, wide[variant].tolist(),
                                                                        wide["baseline"].tolist())
            if variant != "kg_v2":
                preferences[f"{variant}_over_kg_v2"] = judge_preferences(judge, first_rows, wide[variant].tolist(),
                                                                         wide["kg_v2"].tolist())
        ratings.to_csv(OUT / f"ratings_{name}.csv", index=False)
        preferences.to_csv(OUT / f"preferences_{name}.csv")
        judge.close()


def report(args) -> None:
    table = with_judges(pd.read_csv(OUT / "replies.csv"), OUT, "ratings", METRICS)
    preferences = sum(pd.read_csv(OUT / f"preferences_{name}.csv", index_col=0) for name in JUDGES) / len(JUDGES)
    wide = {name: group.set_index("item_id") for name, group in table.groupby("variant")}
    # Guarded variants: a degenerate reply falls back to the baseline reply (all its scores with it).
    for name in variants()[1:]:
        guarded = wide[name].copy()
        bad = guarded.degenerate.astype(bool)
        guarded.loc[bad] = wide["baseline"].loc[bad[bad].index].assign(variant=name).to_numpy()
        wide[f"{name}_guard"] = guarded
    index = wide["baseline"].index
    languages = {"all": index, "english": index[wide["baseline"].language == "english"],
                 "hinglish": index[wide["baseline"].language == "hinglish"]}
    extra = ("indep_target", "reply_words", "repeated_bigram_rate", "reply_is_hinglish", "degenerate", "questions")
    means = [{"language": language, "variant": name, "n": len(ids),
              **{metric: frame.loc[ids, f"{metric}_both"].mean() for metric in METRICS},
              **{column: frame.loc[ids, column].astype(float).mean() for column in extra}}
             for language, ids in languages.items() for name, frame in wide.items()]
    means = pd.DataFrame(means)
    means.to_csv(OUT / "variant_means.csv", index=False)

    rows = []
    for language, ids in languages.items():
        for name in wide:
            for reference in ("baseline", "kg_v2"):
                if name == reference or (reference == "kg_v2" and name == "baseline"):
                    continue
                for metric in METRICS:
                    values = wide[name].loc[ids, f"{metric}_both"], wide[reference].loc[ids, f"{metric}_both"]
                    if values[0].notna().sum() < 6:
                        continue
                    rows.append({"language": language, "contrast": f"{name} - {reference}", "metric": metric,
                                 **contrast(*values)})
                column = f"{name}_over_{reference}"
                if column in preferences:
                    stats = contrast(preferences.loc[ids, column], pd.Series(0.5, index=ids))
                    rows.append({"language": language, "contrast": f"{name} - {reference}",
                                 "metric": "preference_minus_0.5", **stats})
    contrasts = pd.DataFrame(rows)
    primary = (contrasts.language == "all") & contrasts.contrast.isin(["kg_v3 - kg_v2", "kg_v2 - baseline", "kg_v3 - baseline",
                                                                       "kg_hinglish - kg_v2"])
    contrasts.loc[primary, "p_holm"] = holm(contrasts.loc[primary, "p"])
    contrasts.to_csv(OUT / "contrasts.csv", index=False)
    write_json(OUT / "metadata.json", {"model": MODEL_ID, "split": args.split, "messages": int(len(index)),
                                       "max_new_tokens": args.max_new_tokens, "variants": list(variants()),
                                       "judges": {name: value[0] for name, value in JUDGES.items()},
                                       "scores": "mean of the two judges; paired bootstrap 95% CI; Holm over the "
                                                 "all-message contrasts of kg_v3 and kg_v2"})
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
    print(means.round(3).to_string(index=False))
    view = contrasts[contrasts.contrast.isin(["kg_v2 - baseline", "kg_tone - kg_v2", "kg_style - kg_v2", "kg_v3 - kg_v2",
                                              "kg_v3 - baseline", f"{INLINE} - kg_v2", f"{INLINE} - baseline",
                                              "kg_v3_guard - baseline", "kg_v2_guard - kg_v2", "kg_hinglish - kg_v2",
                                              "kg_hinglish - baseline"])]
    view = view.assign(value=view.apply(lambda r: f"{r['mean']:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]", axis=1))
    print(view.pivot_table(index=["language", "metric"], columns="contrast", values="value", aggfunc="first").to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--stages", nargs="+", choices=("generate", "score", "report"), required=True)
    parser.add_argument("--split", default="calibration", choices=("calibration", "dev"))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, help="Smoke test: only the first N messages.")
    parser.add_argument("--variants", nargs="+", help="Variants to run (first must be baseline, second kg_v2).")
    parser.add_argument("--language", choices=("english", "hinglish"), help="Only messages of this language.")
    parser.add_argument("--think", choices=("on", "off"), help="Sarvam only: reasoning on, or skipped.")
    parser.add_argument("--out", default="prompts", help="Output folder name below causal_steering/v3/.")
    args = parser.parse_args()
    global OUT, SELECTED, THINK
    THINK = {"on": True, "off": False, None: None}[args.think]
    OUT = results_dir(MODEL_KEY) / args.out
    SELECTED = tuple(args.variants) if args.variants else None
    for stage in args.stages:
        {"generate": generate, "score": score, "report": report}[stage](args)


if __name__ == "__main__":
    main()
