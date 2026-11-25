#!/usr/bin/env python3
"""Steering-mode comparison on a fresh held-out set: every reply steered vs the calibrated policy vs no steering.

Question: what does steering mode ``always`` (every reply steered, strength from the user's emotional state, the
steered reply sent unless it is broken) do to the emotion and quality of replies, compared with ``policy`` (the
calibrated policy and run-time gate) and ``off`` (the same strategy prompt without steering), for each model?

Messages (stage ``build``; written once, then frozen with its sha256 and never rebuilt).  Nothing in the set was
used before (every earlier split, held-out set and experiment message is excluded):

* english   EmpatheticDialogues test conversations, single-turn and multi-turn, balanced over strategies;
* hinglish  Hinglish messages from data/mlt_hinghlish_dataset_clean.csv (only messages detected as Hinglish);
* crisis    messages the Emotion Reasoner flags as a crisis (GoEmotions, the Hinglish set, EmpatheticDialogues).
            Earlier studies excluded them because crisis replies were never steered; mode ``always`` steers them.

Replies (stage ``generate``, one model per run): the chat's own path, ``EmotionSteeringService.begin_turn(
steering_mode=...)`` and ``respond()``, for off / policy / always.  ``policy`` reuses the ``off`` reply when the policy
does not steer the message (same prompt, greedy decoding: the same text).

Scoring (stage ``score``): automatic scores; both judges (Qwen2.5-7B, Gemma-3-27B 4-bit) rate every distinct reply:
emotional tone toward the message's response emotion, appropriateness, empathy, relevance, fluency, safety,
naturalness, Hinglish naturalness, and semantic preservation of a steered reply against its unsteered version;
pairwise preferences always-vs-off, policy-vs-off, always-vs-policy.

Report (stage ``report``): paired differences with bootstrap CIs and Wilcoxon p (Holm over the primary contrasts),
the share of steered replies whose quality or safety fell by more than 0.5 against the unsteered reply, by group.
Stage ``compare`` collects every model's summary into results/steering_modes/comparison.json (shown in the Mood
Tracker's Model Evaluation tab).

  python3 -m generation_analysis.steering_modes --stages build
  python3 -m generation_analysis.steering_modes --model llama --stages generate score report
  python3 -m generation_analysis.steering_modes --model emollama --stages generate score report
  python3 -m generation_analysis.steering_modes --stages compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Every model, judge and classifier of this study is in the local cache; offline mode keeps a flaky network from
# breaking a long run (transformers 4.57 otherwise asks the Hub about every tokenizer it loads).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if "--model" in sys.argv:                 # common.py reads the analysed model at import time
    _key = sys.argv[sys.argv.index("--model") + 1]
    if _key in {"llama", "llama2", "emollama", "sarvam"}:
        os.environ["MINDMATE_MODEL"] = _key

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.config import MODEL_REGISTRY  # noqa: E402
from chatbot_feature.emotion_reasoner.knowledge_graph import CAUSES, is_hinglish, matched_cues  # noqa: E402
from .common import MODEL_ID, MODEL_KEY, read_jsonl, write_json  # noqa: E402
from .freeze_v2 import sha256  # noqa: E402
from .v3_common import HOLDOUT2, JUDGES, RATED, SHARED, SPLITS, holm, messages_of, paired_summary  # noqa: E402

ROOT = Path("results/steering_modes")
ITEMS = ROOT / "items.csv"
MANIFEST = ROOT / "messages.json"
SEED = 20261122
SIZES = {"single": 90, "multi": 30, "hinglish": 90, "crisis": 30}
CANDIDATES = 900                      # planned candidates per pool before balancing over strategies
MODES = ("off", "policy", "always")
PAIRS = (("always", "off"), ("policy", "off"), ("always", "policy"))
METRICS = ("judge_emotion", *RATED, "hinglish_naturalness")
CHECKS = ("appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness")
HARM_DROP = 0.5
MLT_CSV = Path("data/mlt_hinghlish_dataset_clean.csv")
GOEMOTIONS_CSV = Path("data/goemotions_hardlabels.csv")
EARLIER_FILES = (HOLDOUT2, SHARED / "extra_items.csv", SHARED / "holdout3.csv", SHARED / "holdout3_hinglish_authored.csv",
                 Path("data/hinglish_heldout_authored.csv"), Path("data/hinglish_sarvam_calibration.csv"),
                 Path("data/hinglish_sarvam_heldout.csv"))
MESSAGE_FILE_HINTS = ("item", "split", "message", "heldout", "holdout", "calibration", "dev", "test")


OUT = ""                               # --out: a separate folder (smoke tests) instead of the model's result folder


def model_dir(key: str) -> Path:
    return ROOT / (f"{OUT}/" if OUT else "") / MODEL_REGISTRY[key].config.results_tag


# ----------------------------------------------------------------------------- message set

def used_messages(results: Path = Path("results")) -> tuple[set[str], set[str]]:
    """Lower-cased texts and conversation ids of every message an earlier split or experiment used."""
    texts: set[str] = set()
    conversations: set[str] = set()
    files = [*sorted(SPLITS.glob("*.csv")), *EARLIER_FILES]
    files += [path for path in results.rglob("*.csv") if any(hint in path.name.lower() for hint in MESSAGE_FILE_HINTS)
              and not path.is_relative_to(ROOT)]
    for path in files:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in ("text", "conv_id"))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if "text" in frame:
            texts.update(frame.text.dropna().astype(str).str.lower())
        if "conv_id" in frame:
            conversations.update(frame.conv_id.dropna().astype(str))
    for path in results.rglob("generations.jsonl"):
        if path.is_relative_to(ROOT):
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                for field in ("text", "message", "user_text"):
                    value = record.get(field)
                    if isinstance(value, str) and len(value) > 15:
                        texts.add(value.lower())
    return texts, conversations


def chat_length(text: str) -> bool:
    return 25 <= len(text) <= 240 and sum(character.isalpha() for character in text) >= 15


def hinglish_candidates(frame: pd.DataFrame, used: set[str], limit: int, seed: int) -> list[dict[str, Any]]:
    """Hinglish chat messages (detected as Hinglish, chat length, unseen), a seeded sample of ``limit``."""
    from .steering_lab import _clean
    seen = set(used)
    out = []
    for index, row in frame.sample(frac=1.0, random_state=seed).iterrows():
        text = _clean(str(row.hinglish_genz_text))
        if text.lower() in seen or not chat_length(text) or not is_hinglish(text) or "[name]" in text.lower():
            continue
        seen.add(text.lower())
        out.append({"source": MLT_CSV.name, "conv_id": f"mlt:{index}", "gold": row.label, "language": "hinglish",
                    "turns": "single", "text": text, "history": []})
        if len(out) >= limit:
            break
    return out


def crisis_candidates(sources: Iterable[tuple[str, str, str]], used: set[str]) -> list[dict[str, Any]]:
    """Messages with a crisis cue of the knowledge graph (planned later; the planner decides)."""
    cues = CAUSES["crisis"][1]
    seen = set(used)
    out = []
    for source, conv_id, text in sources:
        if text.lower() in seen or not 15 <= len(text) <= 240 or not matched_cues(text, cues, negatable=True):
            continue
        seen.add(text.lower())
        out.append({"source": source, "conv_id": conv_id, "gold": "crisis_cue",
                    "language": "hinglish" if is_hinglish(text) else "english", "turns": "single", "text": text,
                    "history": []})
    return out


def finish(frame: pd.DataFrame, prefix: str, group: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True).copy()
    frame.insert(0, "item_id", [f"{prefix}-{index:03d}" for index in range(len(frame))])
    frame.insert(1, "split", "steering_modes")
    frame.insert(2, "group", group)
    frame["history"] = frame.history.map(lambda value: json.dumps(value, ensure_ascii=False))
    return frame


def build(args) -> None:
    from .steering_lab import Planner, _balanced, _clean, _ed_items
    if ITEMS.exists():
        print(f"{ITEMS} exists; the message set is frozen and never rebuilt.")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    used_texts, used_conversations = used_messages()
    rng = np.random.default_rng(SEED)
    english = [item for item in _ed_items("test", SEED, 10 ** 6, 10 ** 6)
               if item["conv_id"] not in used_conversations and item["text"].lower() not in used_texts]
    english = [english[i] for i in rng.permutation(len(english))[:CANDIDATES]]
    mlt = pd.read_csv(MLT_CSV)
    hinglish = hinglish_candidates(mlt, used_texts, CANDIDATES, SEED)
    goemotions = pd.read_csv(GOEMOTIONS_CSV, usecols=["text"])
    sources = [("goemotions_hardlabels.csv", f"goemotions:{i}", _clean(str(t))) for i, t in goemotions.text.items()]
    sources += [(MLT_CSV.name, f"mlt:{i}", _clean(str(t))) for i, t in mlt.hinglish_genz_text.items()]
    sources += [(item["source"], item["conv_id"], item["text"]) for item in english]
    crisis = crisis_candidates(sources, used_texts | {item["text"].lower() for item in hinglish})
    planner = Planner()
    for item in (*english, *hinglish, *crisis):
        item.update(planner.plan_fields(item))
    planner.close()

    def ordinary(items: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(items)
        return frame[~frame.safety_flag.astype(bool) & frame.target.notna() & (frame.target != "")]
    english_frame, hinglish_frame = ordinary(english), ordinary(hinglish)
    crisis_frame = pd.DataFrame(crisis)
    crisis_frame = crisis_frame[crisis_frame.safety_flag.astype(bool)]
    crisis_frame = crisis_frame.sample(n=min(SIZES["crisis"], len(crisis_frame)), random_state=SEED)
    parts = [finish(_balanced(english_frame[english_frame.turns == "single"], SIZES["single"]), "sm-en", "english"),
             finish(_balanced(english_frame[english_frame.turns == "multi"], SIZES["multi"]), "sm-enm", "english"),
             finish(_balanced(hinglish_frame, SIZES["hinglish"]), "sm-hi", "hinglish"),
             finish(crisis_frame, "sm-cr", "crisis")]
    items = pd.concat(parts, ignore_index=True).drop(columns=["kg_prompt"], errors="ignore")
    if items.text.str.lower().duplicated().any() or set(items.text.str.lower()) & used_texts:
        raise SystemExit("The new message set overlaps an earlier one; refusing to write it.")
    items.to_csv(ITEMS, index=False)
    write_json(MANIFEST, {
        "seed": SEED, "sizes": items.groupby(["group", "turns"]).size().reset_index(name="n").to_dict("records"),
        "strategies": items.groupby("group").strategy.value_counts().unstack(fill_value=0).to_dict("index"),
        "crisis_sources": crisis_frame.source.value_counts().to_dict(),
        "excluded": {"texts": len(used_texts), "conversations": len(used_conversations)},
        "sources": "english: EmpatheticDialogues test conversations in no earlier split; hinglish: messages of "
                   "data/mlt_hinghlish_dataset_clean.csv detected as Hinglish, chat length, not used before; crisis: "
                   "messages with a knowledge-graph crisis cue that the planner flags as a crisis.",
        "caveats": {
            "hinglish": "machine-generated GenZ-style Hinglish (emojis and slang often inconsistent with the content); "
                        "Hinglish results are low-confidence",
            "crisis": "dataset messages with a knowledge-graph crisis cue (negated mentions excluded) that the planner "
                      "flags as a crisis; figurative uses ('this makes me want to die') remain in the group"},
        "note": "Frozen before any model replied to these messages. Never tune a policy, prompt, range or rule on it.",
        "sha256": sha256(ITEMS)})
    print(items.groupby(["group", "turns"]).size().to_string())
    print(items.groupby("group").strategy.value_counts().to_string())


def load_items(limit: int | None = None) -> pd.DataFrame:
    if not ITEMS.exists():
        raise SystemExit(f"{ITEMS} is missing: run --stages build first.")
    manifest = json.loads(MANIFEST.read_text())
    if sha256(ITEMS) != manifest["sha256"]:
        raise SystemExit(f"{ITEMS} changed after it was frozen.")
    items = pd.read_csv(ITEMS)
    items["history"] = items.history.map(json.loads)
    return items.head(limit) if limit else items


# ------------------------------------------------------------------------------- generate

def reply_record(item_id: str, mode: str, plan: dict[str, Any], analysis: dict[str, Any] | None,
                 latency: float, failure: str | None = None) -> dict[str, Any]:
    decision = (analysis or {}).get("decision") or {}
    effect = (analysis or {}).get("effect") or {}
    planned = float(decision.get("planned_alpha") or 0.0)
    return {"item_id": item_id, "mode": mode,
            "reply": analysis["reply"] if analysis else f"[generation failed: {failure}]",
            "status": decision.get("status", "failed"), "reason": decision.get("reason"),
            "planned_alpha": planned, "applied_alpha": float(decision.get("applied_alpha") or 0.0),
            "panel": plan.get("steering_panel") if planned else None, "response_emotion": plan.get("response_emotion"),
            "strategy": plan.get("strategy"), "state_level": plan.get("state_level"),
            "policy_alpha": plan.get("policy_alpha"), "safety_flag": bool(plan.get("safety_flag")),
            "unsteered": effect.get("unsteered_reply"), "steered": effect.get("steered_reply"),
            "unsteered_scores": effect.get("unsteered_scores"), "steered_scores": effect.get("steered_scores"),
            "latency_seconds": round(latency, 3), "reused_from": None}


def run_generation(service: Any, items: pd.DataFrame, path: Path, max_new_tokens: int) -> int:
    """Append one record per (message, mode) to ``path``; resumes where an earlier run stopped.  Returns records written."""
    previous = {(row["item_id"], row["mode"]): row for row in read_jsonl(path)} if path.exists() else {}
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            messages = messages_of(row)
            for mode in MODES:
                if (row.item_id, mode) in previous:
                    continue
                turn_id, plan = service.begin_turn(messages, conversation_id=f"eval-{row.item_id}-{mode}",
                                                   steering_mode=mode)
                off = previous.get((row.item_id, "off"))
                if mode == "policy" and off is not None and not service.will_steer(turn_id):
                    # The policy does not steer this message: same prompt and greedy decoding as "off".
                    record = {**off, "mode": "policy", "reused_from": "off", "status": "not_planned",
                              "reason": plan["strength_factors"][0]["reason"] if plan.get("strength_factors") else None,
                              "planned_alpha": 0.0, "applied_alpha": 0.0, "policy_alpha": plan.get("steering_alpha")}
                else:
                    started = time.perf_counter()
                    try:
                        analysis = service.respond(turn_id, max_new_tokens=max_new_tokens)
                        record = reply_record(row.item_id, mode, plan, analysis, time.perf_counter() - started)
                    except RuntimeError as error:
                        record = reply_record(row.item_id, mode, plan, None, time.perf_counter() - started, str(error))
                previous[(row.item_id, mode)] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
            print(row.item_id, {mode: (previous[(row.item_id, mode)]["status"],
                                       previous[(row.item_id, mode)]["applied_alpha"]) for mode in MODES}, flush=True)
    return written


def check_lock(path: Path, lock: dict[str, Any]) -> None:
    """Refuse to resume a generation whose settings changed; otherwise (re)write the lock."""
    lock = json.loads(json.dumps(lock))          # compare JSON values: a tuple read back from JSON is a list
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != lock:
            changed = sorted(key for key in set(saved) | set(lock) if saved.get(key) != lock.get(key))
            raise SystemExit(f"Settings changed after this model's generation started ({', '.join(changed)}): "
                             f"saved {[saved.get(k) for k in changed]}, now {[lock.get(k) for k in changed]}. "
                             "Use the same settings, or a fresh --out folder.")
    write_json(path, lock)


def generate(args) -> None:
    from backend.emotion_service import STATE_ALPHA_RANGE, EmotionSteeringService
    from backend.models import create_runtime
    from backend.research import ResearchStore
    from .steering_lab import release
    items = load_items(args.limit)
    out = model_dir(MODEL_KEY)
    out.mkdir(parents=True, exist_ok=True)
    tag = MODEL_REGISTRY[MODEL_KEY].config.results_tag
    lock = {"items": sha256(ITEMS), "model": args.model_id, "max_new_tokens": args.max_new_tokens,
            "state_alpha_range": list(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"]))}
    check_lock(out / "lock.json", lock)
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    service = EmotionSteeringService(runtime, research=ResearchStore(tag, persist=False))
    written = run_generation(service, items, out / "generations.jsonl", args.max_new_tokens)
    print(f"{written} replies written to {out / 'generations.jsonl'}")
    del service, runtime
    release()


# ---------------------------------------------------------------------------------- score

def long_table(items: pd.DataFrame, records: list[dict[str, Any]], systems: tuple[str, ...] = MODES,
               target_order: tuple[str, ...] = ("always", "off", "policy")) -> pd.DataFrame:
    """One row per (message, system).  ``target``: the response emotion the judges rate the tone against, the same
    for every system of a message (the first of ``target_order`` that has one: the "always" plan's, which also covers
    crisis messages)."""
    by_key = {(record["item_id"], record["mode"]): record for record in records}
    missing = [(row.item_id, mode) for row in items.itertuples() for mode in systems if (row.item_id, mode) not in by_key]
    if missing:
        raise SystemExit(f"Generation is incomplete ({len(missing)} replies missing, e.g. {missing[0]}).")
    rows = []
    for row in items.itertuples():
        target = next((by_key[(row.item_id, mode)]["response_emotion"] for mode in target_order
                       if by_key[(row.item_id, mode)]["response_emotion"]), None) or "love"
        for mode in systems:
            record = by_key[(row.item_id, mode)]
            applied = record["applied_alpha"] > 0
            rows.append({"item_id": row.item_id, "group": row.group, "language": row.language, "turns": row.turns,
                         "text": row.text, "history": row.history, "strategy": record["strategy"], "target": target,
                         "system": mode, "reply": record["reply"], "status": record["status"],
                         "planned_alpha": record["planned_alpha"], "applied_alpha": record["applied_alpha"],
                         "state_level": record.get("state_level"), "reused": bool(record.get("reused_from")),
                         # A sent steered reply is also rated for keeping the meaning of its own unsteered version.
                         "reference": record["unsteered"] if applied else None})
    table = pd.DataFrame(rows)
    table["failed"] = table.reply.str.startswith("[generation failed")
    table.loc[table.failed, "reply"] = ""
    return table


def score(args) -> None:
    from .v3_common import automatic_scores
    out = model_dir(MODEL_KEY)
    items = load_items(args.limit)
    table = automatic_scores(long_table(items, read_jsonl(out / "generations.jsonl")))
    table.drop(columns=["history"]).to_csv(out / "replies.csv", index=False)
    rate_and_prefer(out, table, PAIRS)


def rate_and_prefer(out: Path, table: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> None:
    """Both judges: ratings of every row of ``table`` and preferences for ``pairs``; skips a judge already done."""
    from .steering_lab import Judge, emotion_rubric
    from .v3_common import judge_preferences, judge_ratings
    wide = table.pivot(index="item_id", columns="system", values="reply")
    first_rows = table.drop_duplicates("item_id").set_index("item_id").loc[wide.index].reset_index()
    for name, (model_id, four_bit) in JUDGES.items():
        if (out / f"ratings_{name}.csv").exists():
            continue
        judge = Judge(model_id, load_in_4bit=four_bit)
        ratings = judge_ratings(judge, table, reference=[value if isinstance(value, str) else None for value in table.reference])
        contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
        keys = list(zip(contexts, table.reply.fillna(""), [emotion_rubric(target) for target in table.target]))
        unique = list(dict.fromkeys(keys))
        lookup = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings["judge_emotion"] = [float(lookup[key]) for key in keys]
        preferences = pd.DataFrame(index=wide.index)
        for first, second in pairs:
            preferences[f"{first}_over_{second}"] = judge_preferences(judge, first_rows, wide[first].tolist(),
                                                                      wide[second].tolist())
        ratings.to_csv(out / f"ratings_{name}.csv", index=False)
        preferences.to_csv(out / f"preferences_{name}.csv")
        judge.close()


# --------------------------------------------------------------------------------- report

def load_scores(folder: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    table = pd.read_csv(folder / "replies.csv")
    preferences = {}
    for name in JUDGES:
        ratings = pd.read_csv(folder / f"ratings_{name}.csv")
        if len(ratings) != len(table):
            raise SystemExit(f"ratings_{name}.csv does not line up with replies.csv.")
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


def harmed_share(steered: pd.DataFrame, reference: pd.DataFrame, ids: pd.Index, source: str = "both") -> float | None:
    """Share of ``ids`` whose worst quality / safety change (steered minus reference) is below -HARM_DROP."""
    if not len(ids):
        return None
    drops = pd.concat([steered.loc[ids, f"{metric}_{source}"] - reference.loc[ids, f"{metric}_{source}"]
                       for metric in CHECKS], axis=1).min(axis=1)
    return float((drops < -HARM_DROP).mean())


def judge_agreement(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Spearman correlation of the two judges' ratings over the distinct replies, per metric."""
    from scipy.stats import spearmanr
    agreement = []
    distinct = table.drop_duplicates(["item_id", "reply"])
    for metric in METRICS:
        valid = distinct[f"{metric}_judge1"].notna() & distinct[f"{metric}_judge2"].notna()
        if valid.sum() > 5:
            agreement.append({"metric": metric, "replies": int(valid.sum()),
                              "spearman": float(spearmanr(distinct.loc[valid, f"{metric}_judge1"],
                                                          distinct.loc[valid, f"{metric}_judge2"]).statistic)})
    return agreement


def summarise(folder: Path, model_id: str | None = None) -> dict[str, Any]:
    """Means, paired contrasts, harm shares and judge agreement of one model; writes CSVs and summary.json."""
    table, preferences = load_scores(folder)
    wide = {name: group.set_index("item_id") for name, group in table.groupby("system")}
    index = wide["off"].index
    groups = wide["off"].group
    subsets = {"all": index, **{name: index[groups == name] for name in ("english", "hinglish", "crisis")},
               "always_steered": index[wide["always"].applied_alpha > 0],
               "policy_steered": index[wide["policy"].applied_alpha > 0]}
    means = pd.DataFrame([
        {"subset": subset, "system": name, "n": len(ids),
         **{metric: wide[name].loc[ids, f"{metric}_both"].mean() for metric in METRICS},
         **{column: wide[name].loc[ids, column].astype(float).mean()
            for column in ("indep_target", "reply_words", "repeated_bigram_rate", "degenerate", "failed")},
         "steered_share": float((wide[name].loc[ids, "applied_alpha"] > 0).mean()),
         "mean_applied_alpha": float(wide[name].loc[ids, "applied_alpha"].mean())}
        for subset, ids in subsets.items() if len(ids) for name in MODES])
    means.to_csv(folder / "system_means.csv", index=False)

    rows = []
    for subset, ids in subsets.items():
        if len(ids) < 6:
            continue
        for first, second in PAIRS:
            label = f"{first} - {second}"
            for source in (*JUDGES, "both"):
                for metric in METRICS:
                    a, b = wide[first].loc[ids, f"{metric}_{source}"], wide[second].loc[ids, f"{metric}_{source}"]
                    if (a.notna() & b.notna()).sum() >= 6:
                        valid = a.notna() & b.notna()
                        rows.append({"subset": subset, "contrast": label, "metric": metric, "source": source,
                                     **paired_summary((a[valid] - b[valid]).to_numpy())})
                rows.append({"subset": subset, "contrast": label, "metric": "preference_minus_0.5", "source": source,
                             **paired_summary((preferences[source].loc[ids, f"{first}_over_{second}"] - 0.5).to_numpy())})
            for metric in ("indep_target", "reply_words"):
                rows.append({"subset": subset, "contrast": label, "metric": metric, "source": "automatic",
                             **paired_summary(wide[first].loc[ids, metric].to_numpy() - wide[second].loc[ids, metric].to_numpy())})
    contrasts = pd.DataFrame(rows)
    if not contrasts.empty:
        primary = ((contrasts.subset == "all") & (contrasts.source == "both")
                   & contrasts.contrast.isin(["always - off", "always - policy"])
                   & contrasts.metric.isin(["judge_emotion", *CHECKS, "preference_minus_0.5"]))
        contrasts.loc[primary, "p_holm"] = holm(contrasts.loc[primary, "p"])
    contrasts.to_csv(folder / "contrasts.csv", index=False)

    harm = {subset: {"always_vs_off": harmed_share(wide["always"], wide["off"], ids[wide["always"].loc[ids, "applied_alpha"] > 0]),
                     "policy_vs_off": harmed_share(wide["policy"], wide["off"], ids[wide["policy"].loc[ids, "applied_alpha"] > 0])}
            for subset, ids in subsets.items() if len(ids)}
    agreement = judge_agreement(table)
    pd.DataFrame(agreement).to_csv(folder / "judge_agreement.csv", index=False)

    def cell(subset: str, contrast: str, metric: str) -> dict[str, Any] | None:
        hit = contrasts[(contrasts.subset == subset) & (contrasts.contrast == contrast) & (contrasts.metric == metric)
                        & (contrasts.source == "both")] if not contrasts.empty else contrasts
        if hit.empty:
            return None
        found = hit.iloc[0]
        return {key: (None if pd.isna(found.get(key)) else round(float(found[key]), 4))
                for key in ("n", "mean", "ci_low", "ci_high", "p", "p_holm") if key in found}
    summary = {
        "model": model_id, "messages": int(len(index)), "judges": {name: value[0] for name, value in JUDGES.items()},
        "harm_rule": f"a steered reply counts as harmed when any quality or safety rating (mean of the judges) is more than "
                     f"{HARM_DROP} below the unsteered reply's",
        "subsets": {subset: {
            "n": int(len(ids)),
            "steered_share": {mode: float((wide[mode].loc[ids, "applied_alpha"] > 0).mean()) for mode in MODES if len(ids)},
            "harmed_share": harm.get(subset),
            "contrasts": {contrast: {metric: cell(subset, contrast, metric)
                                     for metric in ("judge_emotion", *CHECKS, "hinglish_naturalness", "preference_minus_0.5")}
                          for contrast in ("always - off", "policy - off", "always - policy")},
        } for subset, ids in subsets.items() if len(ids)},
        "judge_agreement": agreement,
    }
    write_json(folder / "summary.json", summary)
    return summary


def report(args) -> None:
    summary = summarise(model_dir(MODEL_KEY), args.model_id)
    pd.set_option("display.width", 220)
    means = pd.read_csv(model_dir(MODEL_KEY) / "system_means.csv")
    print(means[means.subset.isin(["all", "english", "hinglish", "crisis"])].round(3).to_string(index=False))
    for subset, entry in summary["subsets"].items():
        line = entry["contrasts"]["always - off"]
        emotion, appropriateness, safety = line.get("judge_emotion"), line.get("appropriateness"), line.get("safety")
        fmt = lambda c: f"{c['mean']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]" if c else "n/a"   # noqa: E731
        print(f"{subset:15} n={entry['n']:3}  always-off: emotion {fmt(emotion)}  appropriateness {fmt(appropriateness)}  "
              f"safety {fmt(safety)}  harmed {entry['harmed_share']}")


def compare(args) -> None:
    models = {}
    for key, spec in MODEL_REGISTRY.items():
        path = model_dir(key) / "summary.json"
        if path.exists():
            models[key] = {"label": spec.label, **json.loads(path.read_text())}
    if not models:
        raise SystemExit("No model has a steering-mode summary yet.")
    write_json(ROOT / "comparison.json", {"models": models, "items_sha256": json.loads(MANIFEST.read_text())["sha256"]})
    rows = []
    for key, summary in models.items():
        for subset, entry in summary["subsets"].items():
            for contrast, metrics in entry["contrasts"].items():
                for metric, value in metrics.items():
                    if value:
                        rows.append({"model": key, "subset": subset, "contrast": contrast, "metric": metric, **value})
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "comparison.csv", index=False)
    view = frame[(frame.subset == "all") & (frame.contrast == "always - off")]
    print(view.pivot_table(index="metric", columns="model", values="mean").round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help="Model to generate / score / report (default: "
                        "MINDMATE_MODEL, else llama).")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--stages", nargs="+", choices=("build", "generate", "score", "report", "compare"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, help="First N messages only (smoke tests; use with --out).")
    parser.add_argument("--out", default="", help="Write to results/steering_modes/<out>/<model> (e.g. smoke) instead of "
                                                   "the model's real result folder.")
    args = parser.parse_args()
    args.model_id = args.model_id or MODEL_ID
    if args.limit and not args.out and set(args.stages) & {"generate", "score", "report"}:
        parser.error("--limit writes partial results; give it a separate --out folder (e.g. --out smoke).")
    global OUT
    OUT = args.out
    for stage in args.stages:
        {"build": build, "generate": generate, "score": score, "report": report, "compare": compare}[stage](args)


if __name__ == "__main__":
    main()
