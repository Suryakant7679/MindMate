#!/usr/bin/env python3
"""Confirmatory study: each model's locked setup and a "reply warmly" prompt baseline, on fresh messages, with the
hypotheses registered before any reply is generated.

Why: the exploratory studies (results/steering_modes, results/steering_kg) used one 226-message set both to measure
effects and to choose each model's setup, and they had no prompt baseline.  Here the setups are locked from that
evidence, the hypotheses and analysis are written down and fingerprinted first, and everything is tested once on
messages no earlier split, held-out set or study used.

Stages (in this order; each refuses to run out of order):
  build      draw and freeze the fresh message set (sha256 in results/confirmatory/messages.json)
  register   write the pre-registration (hypotheses, tests, correction, locked setups, prompts, strength ranges,
             judges, fingerprints of the code and panel/policy files, message-set hash) to
             results/confirmatory/preregistration.{json,md}.  Commit or upload the .md (e.g. OSF) before generating.
  generate   per model: the six cells through the chat's own path; refuses if registered code or settings changed
  score      both judges rate every reply; pairwise preferences for the registered pairs
  report     per model: registered hypotheses (unadjusted and within-model Holm), secondary, exploratory
  verdict    all models: Holm over every registered test of every model -> results/confirmatory/report.{json,md}

Cells (the plan, the steering target and strength are identical per message; only the prompt and steering change):
  none        MindMate prompt, no steering
  warm        MindMate prompt + "reply warmly" instruction naming the emotion steering targets, no steering
  steer       MindMate prompt + steering (mode always: strength from the user's state, sent unless broken)
  warm_steer  warmth instruction + steering
  kg          the chat's knowledge-graph strategy prompt, no steering
  kg_steer    the chat's knowledge-graph strategy prompt + steering

  python3 -m generation_analysis.confirmatory --stages build
  python3 -m generation_analysis.confirmatory --stages register
  python3 -m generation_analysis.confirmatory --model llama --stages generate score report     # per model
  python3 -m generation_analysis.confirmatory --stages verdict
  python3 -m generation_analysis.confirmatory --model llama --stages generate score report --smoke 3   # pipeline
             check on 3 messages of the already-used steering-mode set (the fresh set stays untouched)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.config import MODEL_REGISTRY  # noqa: E402
from . import steering_modes as sm  # noqa: E402
from .common import MODEL_ID, MODEL_KEY, read_jsonl, write_json  # noqa: E402
from .freeze_v2 import sha256  # noqa: E402
from .v3_common import JUDGES, holm, messages_of, paired_summary  # noqa: E402

ROOT = Path("results/confirmatory")
ITEMS = ROOT / "items.csv"
MANIFEST = ROOT / "messages.json"
REGISTRATION = ROOT / "preregistration.json"
REGISTRATION_MD = ROOT / "preregistration.md"
SEED = 20261123
SIZES = {"single": 150, "multi": 50, "hinglish": 100, "crisis": 30}
CANDIDATES = 1500
CELLS = {"none": ("plain", "off"), "warm": ("warm", "off"), "steer": ("plain", "always"),
         "warm_steer": ("warm", "always"), "kg": (None, "off"), "kg_steer": (None, "always")}
PLAIN_TWIN = {"kg": "none", "kg_steer": "steer"}     # same prompt where the chat itself sends the MindMate prompt alone
TARGET_ORDER = ("steer", "warm_steer", "kg_steer", "none", "warm", "kg")
STEERED = ("steer", "warm_steer", "kg_steer")
TAGS = ("llama31_8b", "emollama_chat_7b", "llama2_13b_chat", "sarvam_30b")
# Each model's setup, locked from the exploratory evidence (results/steering_kg, results/steering_modes and the
# earlier held-out studies) before this study.  Change it only before `register`.
LOCKED = {"llama31_8b": "kg_steer", "emollama_chat_7b": "steer", "llama2_13b_chat": "kg", "sarvam_30b": "kg_steer"}
LOCKED_REASONS = {
    "llama31_8b": "KG prompt + steering: highest tone (+0.85 vs none) with the KG prompt's quality and preference gain",
    "emollama_chat_7b": "MindMate prompt + steering: the KG prompt lowered EmoLLaMA's quality in every study; steering "
                        "raised tone and quality (fewer broken replies)",
    "llama2_13b_chat": "KG prompt, steering off: steering lowered quality and safety with and without the KG prompt, "
                       "increasingly with strength",
    "sarvam_30b": "KG prompt + steering (the chat's setup): tone up with quality kept",
}
MARGIN = 0.10                          # non-inferiority margin on the 1-5 judge scale
ALPHA = 0.05
QUALITY = ("appropriateness", "empathy", "relevance", "fluency", "safety")
PRIMARY_GROUPS = ("english", "hinglish")
BASE_PAIRS = (("steer", "none"), ("warm", "none"), ("steer", "warm"), ("warm_steer", "warm"), ("kg_steer", "kg"))
CODE_FILES = ("backend/emotion_service.py", "backend/config.py", "backend/runtime.py", "backend/llama.py",
              "backend/sarvam.py", "backend/models.py", "backend/generation_control.py",
              "chatbot_feature/emotion_reasoner/agent.py", "chatbot_feature/emotion_reasoner/knowledge_graph.py",
              "chatbot_feature/emotion_reasoner/tone.py", "chatbot_feature/emotion_reasoner/policy.py",
              "chatbot_feature/emotion_reasoner/__init__.py", "generation_analysis/confirmatory.py",
              "generation_analysis/steering_modes.py", "generation_analysis/steering_lab.py",
              "generation_analysis/v3_common.py", "generation_analysis/common.py")

SMOKE = 0                              # --smoke N: N messages of the steering-mode set, results/confirmatory/smoke/


def model_dir(key_or_tag: str) -> Path:
    tag = MODEL_REGISTRY[key_or_tag].config.results_tag if key_or_tag in MODEL_REGISTRY else key_or_tag
    return ROOT / ("smoke" if SMOKE else "") / tag


def tag_of(key: str) -> str:
    return MODEL_REGISTRY[key].config.results_tag


# ----------------------------------------------------------------------------- messages

def build(args) -> None:
    from .steering_lab import Planner, _balanced, _clean, _ed_items
    if ITEMS.exists():
        print(f"{ITEMS} exists; the message set is frozen and never rebuilt.")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    used_texts, used_conversations = sm.used_messages()
    earlier = pd.read_csv(sm.ITEMS)                    # the exploratory 226-message set (skipped by used_messages)
    used_texts |= set(earlier.text.str.lower())
    used_conversations |= set(earlier.conv_id.astype(str))
    rng = np.random.default_rng(SEED)
    english = [item for item in _ed_items("test", SEED, 10 ** 6, 10 ** 6)
               if item["conv_id"] not in used_conversations and item["text"].lower() not in used_texts]
    english = [english[i] for i in rng.permutation(len(english))[:CANDIDATES]]
    mlt = pd.read_csv(sm.MLT_CSV)
    hinglish = sm.hinglish_candidates(mlt, used_texts, CANDIDATES, SEED)
    goemotions = pd.read_csv(sm.GOEMOTIONS_CSV, usecols=["text"])
    sources = [("goemotions_hardlabels.csv", f"goemotions:{i}", _clean(str(t))) for i, t in goemotions.text.items()]
    sources += [(sm.MLT_CSV.name, f"mlt:{i}", _clean(str(t))) for i, t in mlt.hinglish_genz_text.items()]
    sources += [(item["source"], item["conv_id"], item["text"]) for item in english]
    crisis = sm.crisis_candidates(sources, used_texts | {item["text"].lower() for item in hinglish})
    planner = Planner()
    for item in (*english, *hinglish, *crisis):
        item.update(planner.plan_fields(item))
    planner.close()

    def ordinary(items: list[dict[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(items)
        return frame[~frame.safety_flag.astype(bool) & frame.target.notna() & (frame.target != "")]
    english_frame, hinglish_frame = ordinary(english), ordinary(hinglish)
    parts = [sm.finish(_balanced(english_frame[english_frame.turns == "single"], SIZES["single"]), "cf-en", "english"),
             sm.finish(_balanced(english_frame[english_frame.turns == "multi"], SIZES["multi"]), "cf-enm", "english"),
             sm.finish(_balanced(hinglish_frame, SIZES["hinglish"]), "cf-hi", "hinglish")]
    crisis_frame = pd.DataFrame(crisis)
    if len(crisis_frame):
        crisis_frame = crisis_frame[crisis_frame.safety_flag.astype(bool)]
        crisis_frame = crisis_frame.sample(n=min(SIZES["crisis"], len(crisis_frame)), random_state=SEED)
        if len(crisis_frame):
            parts.append(sm.finish(crisis_frame, "cf-cr", "crisis"))
    items = pd.concat(parts, ignore_index=True).drop(columns=["kg_prompt"], errors="ignore")
    items["split"] = "confirmatory"
    if items.text.str.lower().duplicated().any() or set(items.text.str.lower()) & used_texts:
        raise SystemExit("The new message set overlaps an earlier one; refusing to write it.")
    short = {group: (SIZES[key], int(n)) for group, key, n in (
        ("english single", "single", (items.group.eq("english") & items.turns.eq("single")).sum()),
        ("english multi", "multi", (items.group.eq("english") & items.turns.eq("multi")).sum()),
        ("hinglish", "hinglish", items.group.eq("hinglish").sum())) if n < SIZES[key]}
    items.to_csv(ITEMS, index=False)
    write_json(MANIFEST, {
        "seed": SEED, "sizes": items.groupby(["group", "turns"]).size().reset_index(name="n").to_dict("records"),
        "planned_sizes": SIZES, "short_of_plan": short,
        "strategies": items.groupby("group").strategy.value_counts().unstack(fill_value=0).to_dict("index"),
        "targets": items.groupby("group").target.value_counts().unstack(fill_value=0).to_dict("index"),
        "excluded": {"texts": len(used_texts), "conversations": len(used_conversations),
                     "includes": "every earlier split, held-out set and study, including results/steering_modes"},
        "sources": "english: EmpatheticDialogues test conversations in no earlier split or study; hinglish: messages of "
                   "data/mlt_hinghlish_dataset_clean.csv detected as Hinglish, chat length, not used before; crisis: "
                   "messages with a knowledge-graph crisis cue that the planner flags as a crisis.",
        "caveats": {"hinglish": "machine-generated GenZ-style Hinglish; Hinglish results are secondary",
                    "crisis": "few unused crisis messages remain; the crisis group is exploratory"},
        "note": "Frozen before registration and before any model replied to these messages. Used once.",
        "sha256": sha256(ITEMS)})
    print(items.groupby(["group", "turns"]).size().to_string())
    print(items.groupby("group").target.value_counts().to_string())
    if short:
        print(f"Fewer messages than planned: {short}")


def load_items(limit: int | None = None) -> pd.DataFrame:
    if SMOKE:
        return sm.load_items(SMOKE)
    if not ITEMS.exists():
        raise SystemExit(f"{ITEMS} is missing: run --stages build first.")
    if sha256(ITEMS) != json.loads(MANIFEST.read_text())["sha256"]:
        raise SystemExit(f"{ITEMS} changed after it was frozen.")
    items = pd.read_csv(ITEMS)
    items["history"] = items.history.map(json.loads)
    return items.head(limit) if limit else items


# ----------------------------------------------------------------------------- registration

def hypotheses(tag: str) -> list[dict[str, Any]]:
    """The registered hypotheses of one model (a duplicate of an earlier test is listed once)."""
    locked = LOCKED[tag]
    rows = [
        ("H1", "steering raises emotional tone", "steer", "none", "judge_emotion", "greater"),
        ("H2", "steering and the warmth instruction differ in emotional tone", "steer", "warm", "judge_emotion", "two-sided"),
        ("H3", "steering adds emotional tone on top of the warmth instruction", "warm_steer", "warm", "judge_emotion", "greater"),
        ("H4", "steering adds emotional tone on top of the KG prompt", "kg_steer", "kg", "judge_emotion", "greater"),
        ("H5", "the locked setup raises emotional tone over the plain model", locked, "none", "judge_emotion", "greater"),
        ("H6", f"the locked setup keeps reply quality (non-inferiority, margin {MARGIN})", locked, "none", "quality", "noninferior"),
        ("H7", f"the locked setup keeps safety (non-inferiority, margin {MARGIN})", locked, "none", "safety", "noninferior"),
        ("H8", "the locked setup and the warmth instruction differ in emotional tone", locked, "warm", "judge_emotion", "two-sided"),
        ("H9", "steering lowers reply quality", "steer", "none", "quality", "less") if tag == "llama2_13b_chat" else
        ("H9", f"steering alone keeps reply quality (non-inferiority, margin {MARGIN})", "steer", "none", "quality", "noninferior"),
    ]
    out, seen = [], {}
    for hid, claim, first, second, metric, kind in rows:
        key = (first, second, metric, kind)
        if key in seen:
            continue
        seen[key] = hid
        out.append({"id": f"{tag}:{hid}", "model": tag, "hypothesis": hid, "claim": claim, "first": first,
                    "second": second, "metric": metric, "test": kind, "margin": MARGIN if kind == "noninferior" else None})
    return out


def pairs_for(tag: str) -> list[tuple[str, str]]:
    pairs = list(BASE_PAIRS)
    for pair in ((LOCKED[tag], "none"), (LOCKED[tag], "warm")):
        if pair[0] != pair[1] and pair not in pairs:
            pairs.append(pair)
    return pairs


def fingerprints() -> dict[str, str]:
    files = [Path(name) for name in CODE_FILES]
    for tag in TAGS:
        base = Path("results") / f"{tag}_generation_emotion"
        files += sorted((base / "steering_panels").glob("*.json")) + sorted((base / "steering_policy").glob("*.json"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.exists()}


def settings() -> dict[str, Any]:
    """Everything that decides the replies, apart from the code and data fingerprints."""
    from backend.config import SYSTEM_PROMPT
    from backend.emotion_service import PROMPT_STYLE, STATE_ALPHA_RANGE, warmth_prompt
    return {"items_sha256": json.loads(MANIFEST.read_text())["sha256"] if MANIFEST.exists() else None,
            "models": {tag: next(spec.config.model_id for spec in MODEL_REGISTRY.values()
                                 if spec.config.results_tag == tag) for tag in TAGS},
            "max_new_tokens": 200, "decoding": "greedy", "state_alpha_range": {k: list(v) for k, v in STATE_ALPHA_RANGE.items()},
            "kg_prompt_style": PROMPT_STYLE, "locked_setups": LOCKED,
            "warmth_prompt_template": warmth_prompt("<MindMate system prompt>", "love"),
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "judges": {name: value[0] for name, value in JUDGES.items()}}


def git_state() -> dict[str, Any]:
    def run(*command: str) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"head": run("git", "rev-parse", "HEAD"), "uncommitted_changes": bool(run("git", "status", "--porcelain"))}


ANALYSIS = {
    "unit": "message; every cell of a message has the same plan, steering target and strength",
    "primary_population": "all English and Hinglish messages (the crisis group is exploratory)",
    "outcomes": {"judge_emotion": "emotional tone toward the steering target emotion, 1-5, mean of the two judges",
                 "quality": "mean of appropriateness, empathy, relevance, fluency and safety, 1-5, mean of the two judges",
                 "safety": "safety rating, 1-5, mean of the two judges"},
    "tests": {"greater": "one-sided paired t-test of the per-message difference (first - second) > 0",
              "less": "one-sided paired t-test of the per-message difference < 0",
              "two-sided": "two-sided paired t-test of the per-message difference = 0; the direction is reported",
              "noninferior": f"one-sided paired t-test of the per-message difference > -{MARGIN}"},
    "correction": f"Holm-Bonferroni over every registered test of every model; confirmed when the Holm-adjusted p < {ALPHA}",
    "estimates": "mean difference with a 95% percentile bootstrap CI (10,000 resamples); Wilcoxon signed-rank p as a check",
    "secondary": ["each registered hypothesis in English and in Hinglish separately",
                  "each judge separately", "pairwise preferences (both presentation orders averaged) for the registered pairs",
                  "the independent classifier's score for the target emotion",
                  "length-adjusted tone: intercept of the per-message tone difference regressed on the word-count difference",
                  "share of replies whose quality or safety falls by more than 0.5 against the compared cell"],
    "exploratory": ["the crisis group", "every contrast not registered", "per-strategy and per-emotion breakdowns"],
    "missing_data": "a failed generation is rated as an empty reply (kept, not dropped); a model that cannot be run "
                    "counts as not confirmed for every one of its hypotheses",
    "deviations": "any deviation from this registration is reported in results/confirmatory/report.md",
}


def register(args) -> None:
    if not ITEMS.exists():
        raise SystemExit(f"{ITEMS} is missing: run --stages build first.")
    load_items()                                        # checks the frozen hash
    if REGISTRATION.exists():
        generated = [path for path in ROOT.glob("*/generations.jsonl")]
        if not args.replace or generated:
            raise SystemExit(f"{REGISTRATION} exists" + (" and replies were generated after it; it can no longer change."
                             if generated else "; pass --replace to rewrite it (allowed only before any generation)."))
    registration = {
        "title": "MindMate confirmatory study: neuron steering, a knowledge-graph prompt and a warmth instruction",
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "messages": json.loads(MANIFEST.read_text()),
        "cells": {name: {"prompt": {"plain": "MindMate prompt", "warm": "MindMate prompt + warmth instruction",
                                    None: "the chat's knowledge-graph strategy prompt"}[variant],
                         "steering": {"off": "none", "always": "every reply, strength from the user's emotional state, "
                                                               "sent unless broken"}[mode]}
                  for name, (variant, mode) in CELLS.items()},
        "locked_setups": {tag: {"cell": LOCKED[tag], "reason": LOCKED_REASONS[tag]} for tag in TAGS},
        "hypotheses": [row for tag in TAGS for row in hypotheses(tag)],
        "preference_pairs": {tag: [f"{a} vs {b}" for a, b in pairs_for(tag)] for tag in TAGS},
        "analysis": ANALYSIS, "settings": settings(), "fingerprints": fingerprints(), "git": git_state(),
    }
    write_json(REGISTRATION, registration)
    REGISTRATION_MD.write_text(registration_markdown(registration), encoding="utf-8")
    digest = sha256(REGISTRATION_MD)
    print(f"Registered {len(registration['hypotheses'])} hypotheses for {len(TAGS)} models.")
    print(f"{REGISTRATION_MD}  sha256 {digest}")
    print("Timestamp it before generating, e.g. commit it to git or upload it to OSF.")


def registration_markdown(reg: dict[str, Any]) -> str:
    lines = [f"# {reg['title']}", "", f"Registered: {reg['registered_at_utc']}  ",
             f"Message set: `{ITEMS}` sha256 `{reg['messages']['sha256']}`  ",
             f"Code: git `{reg['git']['head'] or 'n/a'}` (uncommitted changes: {reg['git']['uncommitted_changes']}); "
             f"{len(reg['fingerprints'])} files fingerprinted in `{REGISTRATION}`", "",
             "## Messages", ""]
    lines += [f"- {row['group']} / {row['turns']}: {row['n']}" for row in reg["messages"]["sizes"]]
    lines += ["", f"Sources: {reg['messages']['sources']}", "",
              "None of these messages was used by any earlier split, held-out set or study (including the exploratory "
              "steering-mode and steering x KG studies). No model has replied to them before this registration.", "",
              "## Cells", "", "| cell | prompt | steering |", "|---|---|---|"]
    lines += [f"| `{name}` | {cell['prompt']} | {cell['steering']} |" for name, cell in reg["cells"].items()]
    lines += ["", f"Warmth instruction (appended to the MindMate prompt; the emotion is the one steering targets for "
              f"the message): \"{reg['settings']['warmth_prompt_template'].split(chr(10))[-1]}\"", "",
              "## Locked setup per model (chosen from the exploratory studies before this registration)", "",
              "| model | locked cell | reason |", "|---|---|---|"]
    lines += [f"| {tag} | `{row['cell']}` | {row['reason']} |" for tag, row in reg["locked_setups"].items()]
    lines += ["", "## Hypotheses (confirmatory)", "", "| id | claim | contrast | outcome | test |", "|---|---|---|---|---|"]
    lines += [f"| {h['id']} | {h['claim']} | `{h['first']}` - `{h['second']}` | {h['metric']} | {h['test']} |"
              for h in reg["hypotheses"]]
    analysis = reg["analysis"]
    lines += ["", "## Analysis", "", f"- Unit: {analysis['unit']}", f"- Primary population: {analysis['primary_population']}"]
    lines += [f"- Outcome `{name}`: {text}" for name, text in analysis["outcomes"].items()]
    lines += [f"- Test `{name}`: {text}" for name, text in analysis["tests"].items()]
    lines += [f"- Correction: {analysis['correction']}", f"- Estimates: {analysis['estimates']}",
              f"- Missing data: {analysis['missing_data']}", f"- Deviations: {analysis['deviations']}", "",
              "Secondary (reported, not confirmatory):", ""]
    lines += [f"- {text}" for text in analysis["secondary"]]
    lines += ["", "Exploratory:", ""] + [f"- {text}" for text in analysis["exploratory"]]
    lines += ["", "## Settings", "", "```json", json.dumps(reg["settings"], indent=2, ensure_ascii=False), "```", ""]
    return "\n".join(lines)


def check_registration() -> dict[str, Any]:
    """The registration of this run; refuses when the code, data or settings differ from it."""
    if not REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} is missing: run --stages register first (the hypotheses come before any reply).")
    registration = json.loads(REGISTRATION.read_text())
    now = fingerprints()
    changed = sorted(path for path in set(now) | set(registration["fingerprints"])
                     if now.get(path) != registration["fingerprints"].get(path))
    if changed:
        raise SystemExit(f"These files changed after registration: {changed}. Restore them (git diff) - the "
                         "registered study must run on the registered code.")
    current = json.loads(json.dumps(settings()))
    different = sorted(key for key in current if current[key] != registration["settings"].get(key))
    if different:
        raise SystemExit(f"Settings changed after registration: {different}.")
    return registration


# ------------------------------------------------------------------------------- generate

def run_generation(service: Any, items: pd.DataFrame, path: Path, max_new_tokens: int) -> int:
    """Append one record per (message, cell) to ``path``; resumes where an earlier run stopped.  Returns records written."""
    previous = {(row["item_id"], row["mode"]): row for row in read_jsonl(path)} if path.exists() else {}
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            messages = messages_of(row)
            for cell, (variant, mode) in CELLS.items():
                if (row.item_id, cell) in previous:
                    continue
                turn_id, plan = service.begin_turn(messages, conversation_id=f"confirm-{row.item_id}-{cell}",
                                                   steering_mode=mode, prompt_variant=variant)
                twin = PLAIN_TWIN.get(cell)
                if twin and plan["prompt_style"] == "mindmate_only" and (row.item_id, twin) in previous:
                    # The chat itself sends this message the MindMate prompt alone (Sarvam, Hinglish): same prompt,
                    # plan and greedy decoding as the plain cell, so the same reply.
                    record = {**previous[(row.item_id, twin)], "mode": cell, "reused_from": twin}
                else:
                    started = time.perf_counter()
                    try:
                        analysis = service.respond(turn_id, max_new_tokens=max_new_tokens)
                        record = sm.reply_record(row.item_id, cell, plan, analysis, time.perf_counter() - started)
                    except RuntimeError as error:
                        record = sm.reply_record(row.item_id, cell, plan, None, time.perf_counter() - started, str(error))
                record.update(prompt_style=plan["prompt_style"], warmth_emotion=plan.get("warmth_emotion"))
                previous[(row.item_id, cell)] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                written += 1
            print(row.item_id, {cell: previous[(row.item_id, cell)]["applied_alpha"] for cell in CELLS}, flush=True)
    return written


def plan_mismatches(records: list[dict[str, Any]]) -> list[str]:
    """Messages whose cells were planned differently (they must differ only in prompt and steering)."""
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_item.setdefault(record["item_id"], {})[record["mode"]] = record
    bad = []
    for item, cells in by_item.items():
        strategies = {cell["strategy"] for cell in cells.values()}
        steered = {(cells[name]["response_emotion"], cells[name]["planned_alpha"]) for name in STEERED if name in cells}
        warm = {cells[name].get("warmth_emotion") for name in ("warm", "warm_steer") if name in cells}
        steer = cells.get("steer", {})
        target = steer.get("response_emotion") if steer.get("planned_alpha") else None
        if len(strategies) > 1 or len(steered) > 1 or len(warm) > 1 or (target and warm and warm != {target}):
            bad.append(item)
    return bad


def generate(args) -> None:
    from backend.emotion_service import STATE_ALPHA_RANGE, EmotionSteeringService
    from backend.models import create_runtime
    from backend.research import ResearchStore
    from .steering_lab import release
    tag = tag_of(MODEL_KEY)
    if not SMOKE:
        check_registration()
        if args.max_new_tokens != 200 or args.model_id != MODEL_REGISTRY[MODEL_KEY].config.model_id:
            raise SystemExit("The registered study uses the registered model and --max-new-tokens 200.")
    items = load_items()
    out = model_dir(MODEL_KEY)
    out.mkdir(parents=True, exist_ok=True)
    source = ITEMS if not SMOKE else sm.ITEMS
    sm.check_lock(out / "lock.json", {"items": sha256(source), "model": args.model_id, "max_new_tokens": args.max_new_tokens,
                                      "state_alpha_range": list(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"])),
                                      "smoke": SMOKE})
    runtime = create_runtime(MODEL_KEY, model_id=args.model_id)
    service = EmotionSteeringService(runtime, research=ResearchStore(tag, persist=False))
    written = run_generation(service, items, out / "generations.jsonl", args.max_new_tokens)
    print(f"{written} replies written to {out / 'generations.jsonl'}")
    bad = plan_mismatches(read_jsonl(out / "generations.jsonl"))
    if bad:
        print(f"WARNING: {len(bad)} messages were planned differently across cells (e.g. {bad[:3]}).")
    del service, runtime
    release()


# ---------------------------------------------------------------------------------- score

def score(args) -> None:
    from .v3_common import automatic_scores
    if not SMOKE:
        check_registration()
    tag = tag_of(MODEL_KEY)
    out = model_dir(MODEL_KEY)
    table = automatic_scores(sm.long_table(load_items(), read_jsonl(out / "generations.jsonl"), tuple(CELLS), TARGET_ORDER))
    table.drop(columns=["history"]).to_csv(out / "replies.csv", index=False)
    sm.rate_and_prefer(out, table, pairs_for(tag))


# --------------------------------------------------------------------------------- report

def registered_test(difference: np.ndarray, kind: str, margin: float = MARGIN) -> float:
    """p-value of the registered paired t-test (see ANALYSIS["tests"])."""
    from scipy.stats import ttest_1samp
    values = difference[~np.isnan(difference)]
    popmean = -margin if kind == "noninferior" else 0.0
    alternative = {"greater": "greater", "noninferior": "greater", "less": "less", "two-sided": "two-sided"}[kind]
    if len(values) < 3:
        return 1.0
    if np.allclose(values, values[0]):                  # no variance: the t-test is undefined, the sign decides
        shift = float(values[0]) - popmean
        return 0.0 if {"greater": shift > 0, "less": shift < 0, "two-sided": shift != 0}[alternative] else 1.0
    return float(ttest_1samp(values, popmean, alternative=alternative).pvalue)


def load_table(folder: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    table, preferences = sm.load_scores(folder)
    for source in (*JUDGES, "both"):
        table[f"quality_{source}"] = table[[f"{metric}_{source}" for metric in QUALITY]].mean(axis=1)
    return {name: group.set_index("item_id") for name, group in table.groupby("system")}, preferences


def difference(wide: dict[str, pd.DataFrame], first: str, second: str, ids: pd.Index, column: str) -> pd.Series:
    return (wide[first].loc[ids, column] - wide[second].loc[ids, column]).astype(float)


def evaluate(hypothesis: dict[str, Any], wide: dict[str, pd.DataFrame], ids: pd.Index, source: str = "both") -> dict[str, Any]:
    values = difference(wide, hypothesis["first"], hypothesis["second"], ids, f"{hypothesis['metric']}_{source}").to_numpy()
    estimate = paired_summary(values)
    return {"n": estimate["n"], "mean": estimate["mean"], "ci_low": estimate["ci_low"], "ci_high": estimate["ci_high"],
            "wilcoxon_p": estimate["p"], "p": registered_test(values, hypothesis["test"])}


def length_adjusted(wide: dict[str, pd.DataFrame], first: str, second: str, ids: pd.Index, seed: int = 0) -> dict[str, float]:
    """Tone difference at equal reply length: intercept of tone difference ~ word-count difference, bootstrap CI."""
    tone = difference(wide, first, second, ids, "judge_emotion_both").to_numpy()
    words = difference(wide, first, second, ids, "reply_words").to_numpy()
    valid = ~(np.isnan(tone) | np.isnan(words))
    tone, words = tone[valid], words[valid]

    def intercept(index: np.ndarray) -> float:
        design = np.c_[np.ones(len(index)), words[index]]
        return float(np.linalg.lstsq(design, tone[index], rcond=None)[0][0])
    rng = np.random.default_rng(seed)
    draws = [intercept(rng.integers(0, len(tone), len(tone))) for _ in range(2000)]
    return {"intercept": intercept(np.arange(len(tone))), "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)), "n": int(len(tone))}


def summarise(folder: Path, tag: str) -> dict[str, Any]:
    wide, preferences = load_table(folder)
    index = wide["none"].index
    groups = wide["none"].group
    primary = index[groups.isin(PRIMARY_GROUPS)]
    registered = hypotheses(tag)
    confirmatory = [{**h, **evaluate(h, wide, primary)} for h in registered]
    within = holm([row["p"] for row in confirmatory])
    for row, adjusted in zip(confirmatory, within):
        row["p_holm_within_model"] = adjusted

    secondary = []
    for h in registered:
        for group in PRIMARY_GROUPS:
            ids = index[groups == group]
            if len(ids) >= 6:
                secondary.append({"id": h["id"], "analysis": f"{group} only", **evaluate(h, wide, ids)})
        for judge in JUDGES:
            secondary.append({"id": h["id"], "analysis": f"{judge} only", **evaluate(h, wide, primary, judge)})
        if h["metric"] == "judge_emotion":
            secondary.append({"id": h["id"], "analysis": "length-adjusted tone",
                              **length_adjusted(wide, h["first"], h["second"], primary)})
            classifier = paired_summary(difference(wide, h["first"], h["second"], primary, "indep_target").to_numpy())
            secondary.append({"id": h["id"], "analysis": "independent classifier (target emotion probability)", **classifier})
    preference_rows = []
    for first, second in pairs_for(tag):
        column = f"{first}_over_{second}"
        for source in (*JUDGES, "both"):
            if column in preferences[source]:
                preference_rows.append({"pair": f"{first} vs {second}", "source": source,
                                        **paired_summary((preferences[source].loc[primary, column] - 0.5).to_numpy())})
    crisis = index[groups == "crisis"]
    exploratory = {"crisis": [{**h, **evaluate(h, wide, crisis)} for h in registered] if len(crisis) >= 6 else [],
                   "harmed_share": {f"{first} vs {second}": sm.harmed_share(wide[first], wide[second], primary)
                                    for first, second in pairs_for(tag)}}
    means = pd.DataFrame([{"population": name, "cell": cell, "n": len(ids),
                           **{metric: float(wide[cell].loc[ids, f"{metric}_both"].mean())
                              for metric in ("judge_emotion", "quality", *QUALITY)},
                           "indep_target": float(wide[cell].loc[ids, "indep_target"].mean()),
                           "reply_words": float(wide[cell].loc[ids, "reply_words"].mean()),
                           "steered_share": float((wide[cell].loc[ids, "applied_alpha"] > 0).mean())}
                          for name, ids in (("primary", primary), *((g, index[groups == g]) for g in (*PRIMARY_GROUPS, "crisis")))
                          if len(ids) for cell in CELLS])
    means.to_csv(folder / "cell_means.csv", index=False)
    generations = folder / "generations.jsonl"
    records = read_jsonl(generations) if generations.exists() else []
    summary = {"model": tag, "locked_setup": LOCKED[tag], "n_primary": int(len(primary)), "n_crisis": int(len(crisis)),
               "confirmatory": confirmatory, "secondary": secondary, "preferences": preference_rows,
               "exploratory": exploratory, "judge_agreement": sm.judge_agreement(pd.concat(wide.values()).reset_index()),
               "deviations": {"plan_mismatches": plan_mismatches(records),
                              "failed_generations": int(sum(str(r.get("reply", "")).startswith("[generation failed")
                                                            for r in records))},
               "smoke": bool(SMOKE)}
    write_json(folder / "summary.json", summary)
    return summary


def fmt(row: dict[str, Any]) -> str:
    return f"{row['mean']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"


def report(args) -> None:
    tag = tag_of(MODEL_KEY)
    summary = summarise(model_dir(MODEL_KEY), tag)
    means = pd.read_csv(model_dir(MODEL_KEY) / "cell_means.csv")
    pd.set_option("display.width", 220)
    print(means[means.population == "primary"].round(3).to_string(index=False))
    print(f"\n{tag}: registered hypotheses (n={summary['n_primary']}; final Holm over all models in the verdict stage)")
    for row in summary["confirmatory"]:
        print(f"  {row['hypothesis']:3} {row['claim'][:70]:70} {fmt(row)}  p={row['p']:.2g}  "
              f"Holm(model)={row['p_holm_within_model']:.2g}")
    if summary["deviations"]["plan_mismatches"] or summary["deviations"]["failed_generations"]:
        print(f"Deviations: {summary['deviations']}")


def verdict(args) -> None:
    if not SMOKE and not REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} is missing: the verdict tests the registered hypotheses.")
    registration = None if SMOKE else json.loads(REGISTRATION.read_text())
    registered = registration["hypotheses"] if registration else [h for tag in TAGS for h in hypotheses(tag)]
    results = {}
    for tag in TAGS:
        path = model_dir(tag) / "summary.json"
        if path.exists():
            results.update({row["id"]: row for row in json.loads(path.read_text())["confirmatory"]})
    rows = []
    for h in registered:
        found = results.get(h["id"])
        rows.append({**h, **({key: found[key] for key in ("n", "mean", "ci_low", "ci_high", "p", "wilcoxon_p")}
                             if found else {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "p": 1.0,
                                            "wilcoxon_p": None}), "ran": bool(found)})
    for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = adjusted
        row["confirmed"] = bool(row["ran"] and adjusted < ALPHA)
        if row["test"] == "two-sided" and row["confirmed"]:
            row["direction"] = f"{row['first']} higher" if row["mean"] > 0 else f"{row['second']} higher"
    deviations = {tag: json.loads((model_dir(tag) / "summary.json").read_text())["deviations"]
                  for tag in TAGS if (model_dir(tag) / "summary.json").exists()}
    missing = [tag for tag in TAGS if not (model_dir(tag) / "summary.json").exists()]
    base = ROOT / ("smoke" if SMOKE else "")
    write_json(base / "report.json", {"registration_sha256": sha256(REGISTRATION_MD) if registration else None,
                                      "hypotheses": rows, "deviations": deviations, "models_not_run": missing})
    lines = ["# Confirmatory study: results", "",
             f"Registration: `{REGISTRATION_MD}`" + (f" (sha256 `{sha256(REGISTRATION_MD)}`)" if registration else " (smoke run)"),
             f"Holm-Bonferroni over all {len(rows)} registered tests; confirmed when the adjusted p < {ALPHA}.", "",
             "| id | claim | difference [95% CI] | p | Holm p | result |", "|---|---|---|---|---|---|"]
    for row in rows:
        effect = fmt(row) if row["ran"] else "not run"
        result = ("confirmed" + (f" ({row['direction']})" if row.get("direction") else "")) if row["confirmed"] else "not confirmed"
        lines.append(f"| {row['id']} | {row['claim']} | {effect} | {row['p']:.2g} | {row['p_holm']:.2g} | {result} |")
    lines += ["", "## Deviations from the registration", ""]
    lines += [f"- Models not run: {missing}"] if missing else []
    lines += [f"- {tag}: {len(dev['plan_mismatches'])} messages planned differently across cells, "
              f"{dev['failed_generations']} failed generations" for tag, dev in deviations.items()]
    (base / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help="Model to generate / score / report.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--stages", nargs="+", required=True,
                        choices=("build", "register", "generate", "score", "report", "verdict"))
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--replace", action="store_true", help="register: rewrite the registration (only before any "
                                                                 "generation).")
    parser.add_argument("--smoke", type=int, default=0, help="Pipeline check on N messages of the already-used "
                        "steering-mode set; writes results/confirmatory/smoke/ and needs no registration.")
    args = parser.parse_args()
    args.model_id = args.model_id or MODEL_ID
    global SMOKE
    SMOKE = args.smoke
    for stage in args.stages:
        {"build": build, "register": register, "generate": generate, "score": score, "report": report,
         "verdict": verdict}[stage](args)


if __name__ == "__main__":
    main()
