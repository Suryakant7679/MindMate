#!/usr/bin/env python3
"""Steering x knowledge-graph prompt: which part of the chat's reply raises the emotional tone, and at what quality?

The chat combines two things: the Emotion Reasoner's knowledge-graph strategy guidance in the system prompt (the "KG
prompt") and neuron steering (mode ``always``: every reply steered, strength from the user's emotional state, sent
unless broken).  results/steering_modes compared KG prompt + steering with the KG prompt alone.  This study adds the
two missing cells of a 2 x 2 design, on the same frozen 226 messages, for each model:

  none       MindMate prompt alone, no steering
  steer      MindMate prompt alone + steering        ("steering alone")
  kg         KG prompt, no steering                  (= steering_modes "off", reused)
  kg_steer   KG prompt + steering                    (= steering_modes "always", reused; what the chat sends)

"KG prompt" is the prompt each model's chat uses (``PROMPT_STYLE`` in backend/emotion_service.py): the strategy prompt
for Llama, Llama-2 and EmoLLaMA; for Sarvam the version-3 strategy prompt in English and the MindMate prompt alone in
Hinglish, so there the plain cells reuse the KG cells (same prompt, greedy decoding).  The plan (strategy, response
emotion, state level, steering strength) is the same in all four cells; only the prompt and the steering change.

Contrasts (paired per message, mean of both judges, bootstrap CIs, Wilcoxon, Holm over the primary ones):
steering alone (steer - none), KG prompt alone (kg - none), both (kg_steer - none), steering added to the KG prompt
(kg_steer - kg), KG prompt added to steering (kg_steer - steer), and the interaction (kg_steer - kg) - (steer - none):
does steering add more, or less, when the KG prompt is there?

  python3 -m generation_analysis.steering_kg --model llama --stages generate score report
  python3 -m generation_analysis.steering_kg --stages compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if "--model" in sys.argv:                 # common.py reads the analysed model at import time
    _key = sys.argv[sys.argv.index("--model") + 1]
    if _key in {"llama", "llama2", "emollama", "sarvam"}:
        os.environ["MINDMATE_MODEL"] = _key

import pandas as pd  # noqa: E402

from backend.config import MODEL_REGISTRY  # noqa: E402
from . import steering_modes as sm  # noqa: E402
from .common import MODEL_ID, MODEL_KEY, read_jsonl, write_json  # noqa: E402
from .freeze_v2 import sha256  # noqa: E402
from .v3_common import JUDGES, holm, messages_of, paired_summary  # noqa: E402

ROOT = Path("results/steering_kg")
SYSTEMS = ("none", "steer", "kg", "kg_steer")
PLAIN = {"none": "off", "steer": "always"}            # generated here (steering mode), with the MindMate prompt alone
KG = {"kg": "off", "kg_steer": "always"}               # reused: the steering-mode study's replies (the chat's prompt)
TWIN = {"none": "kg", "steer": "kg_steer"}             # the same steering with the KG prompt
TARGET_ORDER = ("kg_steer", "steer", "kg", "none")     # whose response emotion the tone is rated against
PAIRS = (("steer", "none"), ("kg", "none"), ("kg_steer", "none"), ("kg_steer", "kg"), ("kg_steer", "steer"))
INTERACTION = "interaction"
CONTRASTS = (*(f"{first} - {second}" for first, second in PAIRS), INTERACTION)
LABELS = {"steer - none": "steering alone", "kg - none": "KG prompt alone", "kg_steer - none": "KG prompt + steering",
          "kg_steer - kg": "steering added to KG prompt", "kg_steer - steer": "KG prompt added to steering",
          INTERACTION: "interaction"}
DESIGN = {
    "none": "MindMate prompt alone, no steering",
    "steer": "MindMate prompt alone + steering (every reply, strength from the user's emotional state)",
    "kg": "the chat's knowledge-graph strategy prompt, no steering",
    "kg_steer": "the chat's knowledge-graph strategy prompt + steering (what the chat sends)",
    "interaction": "(kg_steer - kg) - (steer - none): positive when steering adds more with the KG prompt than "
                   "without it; for preferences, P(kg_steer over kg) - P(steer over none)",
}
SETTINGS = ("items", "model", "max_new_tokens", "state_alpha_range")
SUMMARY_METRICS = ("judge_emotion", *sm.CHECKS, "hinglish_naturalness", "preference_minus_0.5")
AUTOMATIC = ("indep_target", "reply_words", "repeated_bigram_rate", "degenerate", "failed")

OUT = ""                                # --out: a separate folder (smoke tests) instead of the model's result folder


def model_dir(key: str) -> Path:
    return ROOT / (f"{OUT}/" if OUT else "") / MODEL_REGISTRY[key].config.results_tag


def source_dir(key: str) -> Path:
    """The steering-mode study's folder of this model (always the real one: its KG replies are complete)."""
    return sm.ROOT / MODEL_REGISTRY[key].config.results_tag


# ------------------------------------------------------------------------------- generate

def kg_records(source: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """The KG cells: the steering-mode study's ``off`` and ``always`` replies, renamed ``kg`` and ``kg_steer``."""
    path = source / "generations.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing: the KG cells are the steering-mode study's replies; run "
                         "`python3 -m generation_analysis.steering_modes --model <model> --stages generate` first.")
    out = {}
    for record in read_jsonl(path):
        for system, mode in KG.items():
            if record["mode"] == mode:
                out[(record["item_id"], system)] = {**record, "mode": system}
    return out


def check_source(source: Path, lock: dict[str, Any]) -> None:
    """The reused KG replies must come from the same messages, model, length and steering range."""
    path = source / "lock.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the steering-mode study's generation for this model first.")
    saved, expected = json.loads(path.read_text()), json.loads(json.dumps(lock))
    changed = [key for key in SETTINGS if saved.get(key) != expected.get(key)]
    if changed:
        raise SystemExit(f"The steering-mode replies (the KG cells) were generated with other settings "
                         f"({', '.join(changed)}: {[saved.get(k) for k in changed]} there, {[expected.get(k) for k in changed]} "
                         "now); use the same --model-id and --max-new-tokens.")


def same_plan(record: dict[str, Any], twin: dict[str, Any]) -> bool:
    """The plain cell planned the same reply as its KG twin (strategy, response emotion, steering strength)."""
    return all(record.get(key) == twin.get(key) for key in ("strategy", "response_emotion", "planned_alpha"))


def run_generation(service: Any, items: pd.DataFrame, path: Path, max_new_tokens: int,
                   kg: dict[tuple[str, str], dict[str, Any]]) -> int:
    """Append the two MindMate-prompt cells (none, steer) of every message to ``path``; resumes where an earlier run
    stopped.  Returns the records written."""
    lacking = [(row.item_id, system) for row in items.itertuples() for system in KG if (row.item_id, system) not in kg]
    if lacking:
        raise SystemExit(f"The steering-mode study lacks {len(lacking)} KG replies (e.g. {lacking[0]}); finish its "
                         "generation for this model first.")
    previous = {(row["item_id"], row["mode"]): row for row in read_jsonl(path)} if path.exists() else {}
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            messages = messages_of(row)
            for system, mode in PLAIN.items():
                if (row.item_id, system) in previous:
                    continue
                twin = kg[(row.item_id, TWIN[system])]
                turn_id, plan = service.begin_turn(messages, conversation_id=f"eval-kg-{row.item_id}-{system}",
                                                   steering_mode=mode, prompt_variant="plain")
                if plan.get("deployed_prompt_style") == "mindmate_only":
                    # The chat already sends this message the MindMate prompt alone (Sarvam, Hinglish): the KG cell
                    # has the same prompt, plan and greedy decoding, so the same reply.
                    record = {**twin, "mode": system, "reused_from": TWIN[system]}
                else:
                    started = time.perf_counter()
                    try:
                        analysis = service.respond(turn_id, max_new_tokens=max_new_tokens)
                        record = sm.reply_record(row.item_id, system, plan, analysis, time.perf_counter() - started)
                    except RuntimeError as error:
                        record = sm.reply_record(row.item_id, system, plan, None, time.perf_counter() - started, str(error))
                record.update(prompt_style=plan["prompt_style"], deployed_prompt_style=plan.get("deployed_prompt_style"),
                              plan_matches_kg=same_plan(record, twin))
                previous[(row.item_id, system)] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
            print(row.item_id, {system: (previous[(row.item_id, system)]["status"],
                                         previous[(row.item_id, system)]["applied_alpha"]) for system in PLAIN}, flush=True)
    return written


def generate(args) -> None:
    from backend.emotion_service import STATE_ALPHA_RANGE, EmotionSteeringService
    from backend.models import create_runtime
    from backend.research import ResearchStore
    from .steering_lab import release
    items = sm.load_items(args.limit)
    tag = MODEL_REGISTRY[MODEL_KEY].config.results_tag
    lock = {"items": sha256(sm.ITEMS), "model": args.model_id, "max_new_tokens": args.max_new_tokens,
            "state_alpha_range": list(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"]))}
    source = source_dir(MODEL_KEY)
    check_source(source, lock)
    kg = kg_records(source)
    out = model_dir(MODEL_KEY)
    out.mkdir(parents=True, exist_ok=True)
    sm.check_lock(out / "lock.json", {**lock, "kg_source": str(source)})
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    service = EmotionSteeringService(runtime, research=ResearchStore(tag, persist=False))
    written = run_generation(service, items, out / "generations.jsonl", args.max_new_tokens, kg)
    print(f"{written} replies written to {out / 'generations.jsonl'}")
    mismatched = [f"{r['item_id']}:{r['mode']}" for r in read_jsonl(out / "generations.jsonl") if not r.get("plan_matches_kg")]
    if mismatched:
        print(f"WARNING: {len(mismatched)} replies were planned differently from their KG twin (e.g. {mismatched[:3]}): "
              "those pairs differ in more than the prompt.")
    del service, runtime
    release()


# ---------------------------------------------------------------------------------- score

def score(args) -> None:
    from .v3_common import automatic_scores
    out = model_dir(MODEL_KEY)
    items = sm.load_items(args.limit)
    records = [*read_jsonl(out / "generations.jsonl"), *kg_records(source_dir(MODEL_KEY)).values()]
    table = automatic_scores(sm.long_table(items, records, SYSTEMS, TARGET_ORDER))
    table["prompt"] = table.system.map(lambda system: "mindmate" if system in PLAIN else "kg")
    table["steering"] = table.system.map(lambda system: "always" if system.endswith("steer") else "off")
    table.drop(columns=["history"]).to_csv(out / "replies.csv", index=False)
    sm.rate_and_prefer(out, table, PAIRS)


# --------------------------------------------------------------------------------- report

def summarise(folder: Path, model_id: str | None = None) -> dict[str, Any]:
    """Means, the 2 x 2 contrasts and interaction, harm shares and judge agreement of one model; writes CSVs and
    summary.json."""
    table, preferences = sm.load_scores(folder)
    wide = {name: group.set_index("item_id") for name, group in table.groupby("system")}
    index = wide["none"].index
    groups = wide["none"].group
    prompt_differs = index[~wide["none"].reused.astype(bool)]          # messages whose KG prompt is not the plain one
    subsets = {"all": index, **{name: index[groups == name] for name in ("english", "hinglish", "crisis")}}
    if len(prompt_differs) < len(index):
        subsets["kg_prompt_used"] = prompt_differs
    means = pd.DataFrame([
        {"subset": subset, "system": name, "n": len(ids),
         **{metric: wide[name].loc[ids, f"{metric}_both"].mean() for metric in sm.METRICS},
         **{column: wide[name].loc[ids, column].astype(float).mean() for column in AUTOMATIC},
         "steered_share": float((wide[name].loc[ids, "applied_alpha"] > 0).mean()),
         "mean_applied_alpha": float(wide[name].loc[ids, "applied_alpha"].mean())}
        for subset, ids in subsets.items() if len(ids) for name in SYSTEMS])
    means.to_csv(folder / "system_means.csv", index=False)

    def difference(contrast: str, ids: pd.Index, column: str) -> pd.Series:
        if contrast == INTERACTION:
            return ((wide["kg_steer"].loc[ids, column] - wide["kg"].loc[ids, column])
                    - (wide["steer"].loc[ids, column] - wide["none"].loc[ids, column]))
        first, second = contrast.split(" - ")
        return wide[first].loc[ids, column] - wide[second].loc[ids, column]

    def preference(contrast: str, ids: pd.Index, source: str) -> pd.Series:
        chosen = preferences[source].loc[ids]
        if contrast == INTERACTION:
            return chosen["kg_steer_over_kg"] - chosen["steer_over_none"]
        first, second = contrast.split(" - ")
        return chosen[f"{first}_over_{second}"] - 0.5

    rows = []
    for subset, ids in subsets.items():
        if len(ids) < 6:
            continue
        for contrast in CONTRASTS:
            for source in (*JUDGES, "both"):
                for metric in sm.METRICS:
                    values = difference(contrast, ids, f"{metric}_{source}").dropna()
                    if len(values) >= 6:
                        rows.append({"subset": subset, "contrast": contrast, "metric": metric, "source": source,
                                     **paired_summary(values.to_numpy())})
                rows.append({"subset": subset, "contrast": contrast, "metric": "preference_minus_0.5", "source": source,
                             **paired_summary(preference(contrast, ids, source).to_numpy())})
            for metric in ("indep_target", "reply_words"):
                rows.append({"subset": subset, "contrast": contrast, "metric": metric, "source": "automatic",
                             **paired_summary(difference(contrast, ids, metric).astype(float).to_numpy())})
    contrasts = pd.DataFrame(rows)
    if not contrasts.empty:
        primary = ((contrasts.subset == "all") & (contrasts.source == "both")
                   & contrasts.metric.isin(["judge_emotion", *sm.CHECKS, "preference_minus_0.5"]))
        contrasts.loc[primary, "p_holm"] = holm(contrasts.loc[primary, "p"])
    contrasts.to_csv(folder / "contrasts.csv", index=False)

    def steered(name: str, ids: pd.Index) -> pd.Index:
        return ids[wide[name].loc[ids, "applied_alpha"] > 0]
    harm = {subset: {"steer_vs_none": sm.harmed_share(wide["steer"], wide["none"], steered("steer", ids)),
                     "kg_steer_vs_kg": sm.harmed_share(wide["kg_steer"], wide["kg"], steered("kg_steer", ids)),
                     "kg_vs_none": sm.harmed_share(wide["kg"], wide["none"], ids.intersection(prompt_differs)),
                     "kg_steer_vs_none": sm.harmed_share(wide["kg_steer"], wide["none"], ids)}
            for subset, ids in subsets.items() if len(ids)}
    agreement = sm.judge_agreement(table)
    pd.DataFrame(agreement).to_csv(folder / "judge_agreement.csv", index=False)

    def cell(subset: str, contrast: str, metric: str) -> dict[str, Any] | None:
        hit = contrasts[(contrasts.subset == subset) & (contrasts.contrast == contrast) & (contrasts.metric == metric)
                        & (contrasts.source == "both")] if not contrasts.empty else contrasts
        if hit.empty:
            return None
        found = hit.iloc[0]
        return {key: (None if pd.isna(found.get(key)) else round(float(found[key]), 4))
                for key in ("n", "mean", "ci_low", "ci_high", "p", "p_holm") if key in found}
    generations = folder / "generations.jsonl"
    summary = {
        "model": model_id, "messages": int(len(index)), "judges": {name: value[0] for name, value in JUDGES.items()},
        "design": DESIGN, "contrast_labels": LABELS,
        "harm_rule": f"a reply counts as harmed when any quality or safety rating (mean of the judges) is more than "
                     f"{sm.HARM_DROP} below the compared reply's; steering harm shares count steered replies only",
        "plan_mismatches": (sum(not r.get("plan_matches_kg", True) for r in read_jsonl(generations))
                            if generations.exists() else None),
        "subsets": {subset: {
            "n": int(len(ids)),
            "steered_share": {name: float((wide[name].loc[ids, "applied_alpha"] > 0).mean()) for name in SYSTEMS},
            "means": {name: {metric: round(float(wide[name].loc[ids, f"{metric}_both"].mean()), 4)
                             for metric in ("judge_emotion", *sm.CHECKS)} for name in SYSTEMS},
            "harmed_share": harm.get(subset),
            "contrasts": {contrast: {metric: cell(subset, contrast, metric) for metric in SUMMARY_METRICS}
                          for contrast in CONTRASTS},
        } for subset, ids in subsets.items() if len(ids)},
        "judge_agreement": agreement,
    }
    write_json(folder / "summary.json", summary)
    return summary


def fmt(value: dict[str, Any] | None) -> str:
    return f"{value['mean']:+.2f} [{value['ci_low']:+.2f}, {value['ci_high']:+.2f}]" if value else "n/a"


def report(args) -> None:
    folder = model_dir(MODEL_KEY)
    summary = summarise(folder, args.model_id)
    pd.set_option("display.width", 220)
    means = pd.read_csv(folder / "system_means.csv")
    columns = ["subset", "system", "n", "judge_emotion", *sm.CHECKS, "steered_share", "mean_applied_alpha"]
    print(means[means.subset.isin(["all", "english", "hinglish", "crisis"])][columns].round(3).to_string(index=False))
    for subset, entry in summary["subsets"].items():
        print(f"\n{subset} (n={entry['n']})  harmed {({k: v if v is None else round(v, 3) for k, v in entry['harmed_share'].items()})}")
        for contrast in CONTRASTS:
            line = entry["contrasts"][contrast]
            print(f"  {LABELS[contrast]:28} tone {fmt(line['judge_emotion'])}  appropriateness {fmt(line['appropriateness'])}  "
                  f"safety {fmt(line['safety'])}  preference {fmt(line['preference_minus_0.5'])}")
    if summary.get("plan_mismatches"):
        print(f"\nWARNING: {summary['plan_mismatches']} plain replies were planned differently from their KG twin.")


def compare(args) -> None:
    base = ROOT / OUT if OUT else ROOT
    models = {}
    for key, spec in MODEL_REGISTRY.items():
        path = model_dir(key) / "summary.json"
        if path.exists():
            models[key] = {"label": spec.label, **json.loads(path.read_text())}
    if not models:
        raise SystemExit("No model has a steering x KG summary yet.")
    write_json(base / "comparison.json", {"models": models, "design": DESIGN, "contrast_labels": LABELS,
                                          "items_sha256": json.loads(sm.MANIFEST.read_text())["sha256"]})
    rows = [{"model": key, "subset": subset, "contrast": contrast, "metric": metric, **value}
            for key, summary in models.items() for subset, entry in summary["subsets"].items()
            for contrast, metrics in entry["contrasts"].items() for metric, value in metrics.items() if value]
    frame = pd.DataFrame(rows)
    frame.to_csv(base / "comparison.csv", index=False)
    view = frame[(frame.subset == "all") & frame.metric.isin(["judge_emotion", "appropriateness", "safety",
                                                               "preference_minus_0.5"])].copy()
    view["contrast"] = view.contrast.map(LABELS)
    pd.set_option("display.width", 220)
    print(view.pivot_table(index=["metric", "contrast"], columns="model", values="mean", sort=False).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help="Model to generate / score / report (default: "
                        "MINDMATE_MODEL, else llama).")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--stages", nargs="+", choices=("generate", "score", "report", "compare"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, help="First N messages only (smoke tests; use with --out).")
    parser.add_argument("--out", default="", help="Write to results/steering_kg/<out>/<model> (e.g. smoke) instead of "
                                                   "the model's real result folder.")
    args = parser.parse_args()
    args.model_id = args.model_id or MODEL_ID
    if args.limit and not args.out and set(args.stages) & {"generate", "score", "report"}:
        parser.error("--limit writes partial results; give it a separate --out folder (e.g. --out smoke).")
    global OUT
    OUT = args.out
    for stage in args.stages:
        {"generate": generate, "score": score, "report": report, "compare": compare}[stage](args)


if __name__ == "__main__":
    main()
