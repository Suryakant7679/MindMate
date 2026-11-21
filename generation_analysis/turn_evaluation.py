#!/usr/bin/env python3
"""Offline judging of recorded chatbot turns: the steered and unsteered reply to the same prompt.

Reads ``results/telemetry/<model tag>/turns.jsonl`` (written by the server when persistence is on) and, for every turn
with both replies, writes one line to ``judged.jsonl``: both judges' ratings of each reply (emotion, quality, Hinglish
naturalness, meaning kept), the pairwise preference of the steered reply, and the independent GoEmotions classifier's
score of the intended emotion.  The Mood Tracker reads that file and shows the judged verdict next to the automatic one.

Limits, stated in the output: the telemetry keeps the latest user message only, so judges see that message without
the earlier conversation; and a reply pair is judged once (re-running skips turns already judged, unless --force).

  python3 -m generation_analysis.turn_evaluation --model llama          # after some chats were recorded
  python3 -m generation_analysis.turn_evaluation --model sarvam --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "--model" in sys.argv:                 # the rubric helpers read the analysed model at import time
    _key = sys.argv[sys.argv.index("--model") + 1]
    if _key in {"llama", "llama2", "emollama", "sarvam"}:
        os.environ["MINDMATE_MODEL"] = _key

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .neuron_validation import MODELS, describe  # noqa: E402
from .steering_lab import release  # noqa: E402

TELEMETRY = Path("results/telemetry")
METRICS = ("emotion", "appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness", "hinglish_naturalness")


def rows_to_judge(path: Path, judged: set[str], key: str) -> list[dict]:
    """One row per reply: unsteered and steered reply of every recorded turn that has both."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        turn_id, replies, scores = record.get("turn_id"), record.get("replies") or {}, record.get("scores") or {}
        if not turn_id or turn_id in judged or not replies.get("steered") or not replies.get("unsteered"):
            continue
        plan = record.get("plan") or {}
        panel = plan.get("steering_panel") or plan.get("response_emotion") or scores.get("target")
        if not panel:
            continue
        try:
            target, description = describe(key, panel)
        except KeyError:
            continue                       # a panel without a rubric: not judged
        language = (record.get("user") or {}).get("language", "english")
        text = (record.get("user") or {}).get("text", "")
        for role, reply in (("unsteered", replies["unsteered"]), ("steered", replies["steered"])):
            rows.append({"turn_id": turn_id, "role": role, "text": text, "history": [], "reply": reply,
                         "reference": replies["unsteered"] if role == "steered" else None, "language": language,
                         "target": target, "description": description, "panel": panel})
    return rows


def judged_record(turn_id: str, judges: dict[str, str], ratings: dict[str, dict[str, dict[str, float | None]]],
                  preference: dict[str, float | None], independent: dict | None) -> dict:
    """The line written to judged.jsonl: ratings per reply, differences, and the preference of the steered reply."""
    names = list(judges)
    difference = {name: {metric: _diff(ratings["steered"][name].get(metric), ratings["unsteered"][name].get(metric))
                         for metric in METRICS + ("preservation",)} for name in names}
    difference["both"] = {metric: _diff(ratings["steered"]["both"].get(metric), ratings["unsteered"]["both"].get(metric))
                          for metric in METRICS + ("preservation",)}
    return {"turn_id": turn_id, "judges": judges, "ratings": ratings,
            "difference": difference, "preference": preference,
            "preservation": {name: ratings["steered"].get(name, {}).get("preservation") for name in (*names, "both")},
            "independent": independent}


def _diff(first, second):
    if first is None or second is None or (isinstance(first, float) and np.isnan(first)) or (isinstance(second, float) and np.isnan(second)):
        return None
    return round(float(first) - float(second), 4)


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)


def judge(args) -> None:
    from .v3_common import JUDGES, automatic_scores, judge_preferences, judge_ratings
    from .steering_lab import Judge
    key = args.model
    folder = TELEMETRY / MODELS[key]["tag"]
    turns = folder / "turns.jsonl"
    if not turns.exists():
        raise SystemExit(f"No recorded turns at {turns}. Chat with the model first (the server records them by default).")
    output = folder / "judged.jsonl"
    judged = set()
    if output.exists() and not args.force:
        judged = {json.loads(line)["turn_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    rows = rows_to_judge(turns, judged, key)
    if not rows:
        print("Nothing new to judge.", flush=True)
        return
    table = pd.DataFrame(rows).reset_index(drop=True)
    automatic = automatic_scores(table.assign(reply=table.reply))
    table["indep_target"] = automatic["indep_target"].to_numpy()
    scores = {name: {} for name in JUDGES}
    preferences, reference_text = {}, table.reference.tolist()
    for name, (model_id, four_bit) in JUDGES.items():
        judge_obj = Judge(model_id, load_in_4bit=four_bit)
        contexts = [Judge.context(text, []) for text in table.text]
        keys = list(zip(contexts, table.reply.tolist(), [f"Emotional tone: how strongly does the reply itself express {d}? "
                                                          "Judge only the tone of the reply, not whether it is a good reply. "
                                                          "1 = not at all. 5 = very strongly." for d in table.description]))
        unique = list(dict.fromkeys(keys))
        values = dict(zip(unique, judge_obj.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings = judge_ratings(judge_obj, table, reference=[r if isinstance(r, str) else None for r in reference_text])
        ratings["emotion"] = [float(values[k]) for k in keys]
        steered = table.role.eq("steered").to_numpy()
        preference = np.full(len(table), np.nan)
        part = table[steered].reset_index(drop=True)
        preference[steered] = judge_preferences(judge_obj, part, part.reply.tolist(), part.reference.tolist())
        preferences[name] = preference
        scores[name] = ratings
        judge_obj.close()
        print(f"judged by {name} ({model_id})", flush=True)
    release()
    records = []
    for turn_id, group in table.groupby("turn_id", sort=False):
        ratings_block = {"unsteered": {}, "steered": {}}
        for role in ("unsteered", "steered"):
            position = group.index[group.role == role][0]
            per_judge = {}
            for name in JUDGES:
                per_judge[name] = {metric: _clean(scores[name].loc[position, metric]) if metric in scores[name] else None
                                   for metric in METRICS + ("preservation",)}
            per_judge["both"] = {metric: (_clean(np.nanmean([per_judge[n][metric] for n in JUDGES if per_judge[n][metric] is not None]))
                                          if any(per_judge[n][metric] is not None for n in JUDGES) else None)
                                 for metric in METRICS + ("preservation",)}
            ratings_block[role] = per_judge
        steered_position = group.index[group.role == "steered"][0]
        preference = {name: _clean(preferences[name][steered_position]) for name in JUDGES}
        preference["both"] = _clean(np.nanmean([preference[n] for n in JUDGES])) if any(preference[n] is not None for n in JUDGES) else None
        independent = {"model": "SamLowe/roberta-base-go_emotions", "target": describe(key, group.panel.iloc[0])[0],
                       "unsteered": _clean(automatic.loc[group.index[group.role == "unsteered"][0], "indep_target"]),
                       "steered": _clean(automatic.loc[steered_position, "indep_target"])}
        independent["difference"] = _diff(independent["steered"], independent["unsteered"])
        records.append(judged_record(turn_id, {name: JUDGES[name][0] for name in JUDGES}, ratings_block, preference, independent))
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} judged turns to {output}; the ratings are offline and the Mood Tracker reads them", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODELS), default="llama")
    parser.add_argument("--force", action="store_true", help="Judge turns again even if they were judged before.")
    judge(parser.parse_args())


if __name__ == "__main__":
    main()
