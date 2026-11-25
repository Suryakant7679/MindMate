#!/usr/bin/env python3
"""E5 study: quality-constrained adaptive selective emotion-neuron steering, against E1-E4 and E5-old.

Stages (run in this order; each refuses to run out of order).  GPU stages are per model (--model).

  audit         CPU   evidence from the already-used studies (generation_analysis.e5_audit)
  build         CPU   draw and freeze the development supplement and the held-out set (generation_analysis.e5_data)
  dev-generate  GPU   development replies: per message, steered candidates at 0.5x and 0.75x of E4's strength
                      (E4's 1.0x candidates exist already); the supplement also gets E3 and E4
  dev-score     GPU   both judges rate every development reply (incl. the new validation / strategy rubrics)
  calibrate     CPU   fit the model's E5 policy on development data (generation_analysis.e5_fit) -> draft
  freeze        CPU   freeze all four policies and write the pre-registration (hypotheses, margins, fingerprints)
  generate      GPU   held-out replies for E1, E2, E3, E4, E5old, E5 through the chat's own code paths
  score         GPU   both judges: ratings of every reply + pairwise preferences (both orders, blind)
  report        CPU   per model: contrasts, non-inferiority, coverage, latency, subgroups
  verdict       CPU   all models: Holm over every registered test, verdicts, report.md
  plots         CPU   coverage, quality-versus-tone trade-off and alpha-distribution figures
  examples      CPU   representative successful and unsuccessful E5 turns

Long runs append one JSON line per finished unit (resume = run the same command again; finished units are skipped),
write progress to results/e5_selective/progress/, log failures with tracebacks to results/e5_selective/logs/errors.log,
cache every judge answer as it is computed (judge_cache_*.jsonl), and refuse to resume with changed settings.

  python3 -m generation_analysis.e5_selective --stages audit build
  python3 -m generation_analysis.e5_selective --model llama --stages dev-generate dev-score calibrate
  python3 -m generation_analysis.e5_selective --stages freeze
  python3 -m generation_analysis.e5_selective --model llama --stages generate score report
  python3 -m generation_analysis.e5_selective --stages verdict plots examples
  python3 -m generation_analysis.e5_selective --model llama --stages dev-generate dev-score calibrate generate score report --smoke 4
             (pipeline check on 4 already-used messages; writes results/e5_selective/smoke/, needs no registration)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if "--model" in sys.argv:                 # common.py reads the analysed model at import time
    _key = sys.argv[sys.argv.index("--model") + 1]
    if _key in {"llama", "llama2", "emollama", "sarvam"}:
        os.environ["MINDMATE_MODEL"] = _key

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backend.config import MODEL_REGISTRY  # noqa: E402
from .e5_common import (CONDITIONS, CONTROLS, PRIMARY_PAIRS, ROOT, SECONDARY_PAIRS, SEED, TAGS,  # noqa: E402
                        write_json)

SMOKE = 0
MULTIPLIERS = (0.5, 0.75)                    # generated in dev-generate (1.0 = E4's existing candidates)
MAX_NEW_TOKENS = 200                         # as every earlier study
CHUNK = 384                                  # judge prompts per cache flush
LOGS = ROOT / "logs"
PROGRESS = ROOT / "progress"
POLICIES = ROOT / "policies"
REGISTRATION = ROOT / "preregistration.json"
REGISTRATION_MD = ROOT / "preregistration.md"
CODE_FILES = ("backend/emotion_service.py", "backend/selective_service.py", "backend/config.py", "backend/runtime.py",
              "backend/llama.py", "backend/sarvam.py", "backend/models.py", "backend/generation_control.py",
              "chatbot_feature/emotion_reasoner/agent.py", "chatbot_feature/emotion_reasoner/knowledge_graph.py",
              "chatbot_feature/emotion_reasoner/tone.py", "chatbot_feature/emotion_reasoner/policy.py",
              "chatbot_feature/emotion_reasoner/selective.py", "chatbot_feature/emotion_reasoner/risk_screen.py",
              "generation_analysis/e5_selective.py", "generation_analysis/e5_common.py",
              "generation_analysis/steering_modes.py", "generation_analysis/steering_lab.py",
              "generation_analysis/v3_common.py", "generation_analysis/common.py")
VALIDATION_RUBRIC = ("Emotional validation: does the reply acknowledge what the user feels and treat it as understandable "
                     "and legitimate, without judging, dismissing, minimising or rushing past it? For a practical or "
                     "neutral message, rate 3 unless the reply dismisses the user. 1 = dismisses, judges or ignores the "
                     "feelings. 5 = clearly and genuinely validates them.")
HINDI_RUBRIC = ("Hindi naturalness: the user writes Hindi in Devanagari script. Does the reply read like natural, fluent "
                "Hindi in Devanagari that a native speaker would write? 1 = English, romanised Hindi, broken grammar or "
                "stiff translated-sounding Hindi. 5 = fully natural Hindi.")


def adherence_rubric(strategy: str) -> str:
    from chatbot_feature.emotion_reasoner.knowledge_graph import STRATEGIES
    title, _, guidance, _, _ = STRATEGIES.get(strategy, STRATEGIES["gentle_exploration"])
    return (f"Strategy adherence: the recommended response strategy for this message is \"{title}\": {guidance} How "
            "well does the reply follow this strategy, in substance and tone? 1 = ignores or contradicts it. 5 = follows "
            "it fully and naturally.")


def base_dir() -> Path:
    return ROOT / "smoke" if SMOKE else ROOT


def tag_of(key: str) -> str:
    return MODEL_REGISTRY[key].config.results_tag


# --------------------------------------------------------------------------------------------- utilities

def log_error(stage: str, unit: str, error: BaseException) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / "errors.log").open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {dt.datetime.now().isoformat(timespec='seconds')} {stage} {unit}: "
                     f"{type(error).__name__}: {error}\n{traceback.format_exc()}")


class Progress:
    """Progress line on stdout and results/e5_selective/progress/<name>.json (done, total, rate, ETA, failures)."""

    def __init__(self, name: str, total: int, done: int = 0) -> None:
        self.name, self.total, self.start_done, self.done = name, total, done, done
        self.failures = 0
        self.started = time.time()
        self.path = (base_dir() / "progress" if SMOKE else PROGRESS) / f"{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.update(0)

    def update(self, step: int = 1, failed: bool = False, note: str = "") -> None:
        self.done += step
        self.failures += int(failed)
        elapsed = time.time() - self.started
        rate = (self.done - self.start_done) / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else None
        state = {"stage": self.name, "done": self.done, "total": self.total, "failures": self.failures,
                 "elapsed_seconds": round(elapsed), "per_minute": round(rate * 60, 2),
                 "eta_minutes": round(eta / 60, 1) if eta is not None else None, "note": note,
                 "updated": dt.datetime.now().isoformat(timespec="seconds")}
        self.path.write_text(json.dumps(state, indent=1))
        if step:
            print(f"[{self.name}] {self.done}/{self.total}" + (f" eta {state['eta_minutes']} min" if eta else "")
                  + (f" failures {self.failures}" if self.failures else "") + (f" {note}" if note else ""), flush=True)


def read_records(path: Path, key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    """Last record per key (a retried unit appends a newer record)."""
    from .common import read_jsonl
    return {key(record): record for record in read_jsonl(path)} if path.exists() else {}


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except ImportError:
        pass


def check_lock(path: Path, lock: dict[str, Any]) -> None:
    from . import steering_modes as sm
    sm.check_lock(path, lock)


def gpu_release() -> None:
    from .steering_lab import release
    release()


def is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() or type(error).__name__ == "OutOfMemoryError"


def with_retry(stage: str, unit: str, work: Callable[[], Any]) -> tuple[Any, str | None]:
    """Run ``work``; on a CUDA out-of-memory error free the cache and retry once.  ``(result, error message)``."""
    for attempt in (1, 2):
        try:
            return work(), None
        except Exception as error:               # noqa: BLE001 - every failure is logged and recorded
            log_error(stage, unit, error)
            if attempt == 1 and is_oom(error):
                gpu_release()
                continue
            return None, f"{type(error).__name__}: {error}"
    return None, "unreachable"


# ---------------------------------------------------------------------------------------------- judges

class CachedJudge:
    """A judge whose every answer is appended to a JSON-lines cache as soon as a chunk is computed."""

    def __init__(self, name: str, model_id: str, four_bit: bool, cache: Path) -> None:
        self.name, self.model_id, self.four_bit, self.path = name, model_id, four_bit, cache
        self.cache: dict[str, float] = {}
        if cache.exists():
            with cache.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        record = json.loads(line)
                        self.cache[record["k"]] = float(record["v"])
        self._judge = None

    def _key(self, *parts: str) -> str:
        return hashlib.sha1("\x1f".join((self.model_id, *parts)).encode()).hexdigest()

    def judge(self):
        if self._judge is None:
            from .steering_lab import Judge
            self._judge = Judge(self.model_id, load_in_4bit=self.four_bit)
        return self._judge

    def _fill(self, keys: list[str], inputs: list[tuple], compute: Callable[[list[tuple]], np.ndarray], label: str) -> None:
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if not missing:
            return
        by_key = dict(zip(keys, inputs))
        progress = Progress(f"{label}_{self.name}", len(missing))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing), CHUNK):
            chunk = missing[start:start + CHUNK]
            values = compute([by_key[key] for key in chunk])
            with self.path.open("a", encoding="utf-8") as handle:
                for key, value in zip(chunk, values):
                    self.cache[key] = float(value)
                    handle.write(json.dumps({"k": key, "v": float(value)}) + "\n")
            progress.update(len(chunk))

    def rate(self, contexts: Sequence[str], replies: Sequence[str], rubrics: Sequence[str], label: str = "rate") -> np.ndarray:
        keys = [self._key("rate", c, r, b) for c, r, b in zip(contexts, replies, rubrics)]
        self._fill(keys, list(zip(contexts, replies, rubrics)),
                   lambda rows: self.judge().rate([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]), label)
        return np.array([self.cache[key] for key in keys])

    def prefer(self, contexts: Sequence[str], first: Sequence[str], second: Sequence[str], label: str = "prefer") -> np.ndarray:
        """P(first preferred), both presentation orders averaged; 0.5 without a call when the replies are identical."""
        out = np.full(len(contexts), 0.5)
        rows = [i for i in range(len(contexts)) if first[i] != second[i]]
        keys = [self._key("prefer", contexts[i], first[i], second[i]) for i in rows]
        self._fill(keys, [(contexts[i], first[i], second[i]) for i in rows],
                   lambda chunk: self.judge().prefer([r[0] for r in chunk], [r[1] for r in chunk], [r[2] for r in chunk]),
                   label)
        out[rows] = [self.cache[key] for key in keys]
        return out

    def close(self) -> None:
        if self._judge is not None:
            self._judge.close()
            self._judge = None
        gpu_release()


def judges(folder: Path) -> list[CachedJudge]:
    from .v3_common import JUDGES
    return [CachedJudge(name, model_id, four_bit, folder / f"judge_cache_{name}.jsonl")
            for name, (model_id, four_bit) in JUDGES.items()]


def rate_rows(judge: CachedJudge, table: pd.DataFrame, label: str) -> pd.DataFrame:
    """Every rubric for every row of ``table`` (columns text, history, reply, target, strategy, lang3)."""
    from .steering_lab import Judge, emotion_rubric
    from .v3_common import HINGLISH_RUBRIC, RUBRICS_V3
    contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
    replies = table.reply.fillna("").astype(str).tolist()
    out = pd.DataFrame(index=table.index)
    plan: list[tuple[str, list[int], list[str]]] = [(name, list(range(len(table))), [rubric] * len(table))
                                                    for name, rubric in RUBRICS_V3.items()]
    plan.append(("judge_emotion", list(range(len(table))), [emotion_rubric(t if isinstance(t, str) and t else "love")
                                                             for t in table.target]))
    plan.append(("validation", list(range(len(table))), [VALIDATION_RUBRIC] * len(table)))
    plan.append(("strategy_adherence", list(range(len(table))), [adherence_rubric(s) for s in table.strategy]))
    for language, name, rubric in (("hinglish", "hinglish_naturalness", HINGLISH_RUBRIC),
                                   ("hindi", "hindi_naturalness", HINDI_RUBRIC)):
        rows = [i for i, value in enumerate(table.lang3) if value == language]
        plan.append((name, rows, [rubric] * len(rows)))
    for name, rows, rubrics in plan:
        values = np.full(len(table), np.nan)
        if rows:
            values[rows] = judge.rate([contexts[i] for i in rows], [replies[i] for i in rows], rubrics, label=f"{label}_{name}")
        out[name] = values
    return out


# ----------------------------------------------------------------------------------- development data

def dev_items(limit: int | None = None) -> pd.DataFrame:
    """Every development message: the two earlier studies' items (key ``<study>:<item_id>``) + the supplement."""
    from . import e5_data
    from .e5_common import existing_items
    frames = []
    for study, frame in existing_items().items():
        frame = frame.reset_index()
        frame["study"] = study
        frame["category"] = frame.group
        frames.append(frame)
    if not SMOKE:
        supplement = e5_data.load("dev_supplement")
        supplement["study"] = "supplement"
        frames.append(supplement)
    items = pd.concat(frames, ignore_index=True)
    items["key"] = items.study + ":" + items.item_id
    if SMOKE:
        items = items.head(SMOKE)
    return items.head(limit) if limit else items


def dev_dir(tag: str) -> Path:
    return base_dir() / "dev" / tag


def dev_generate(args) -> None:
    from backend.emotion_service import STATE_ALPHA_RANGE, EmotionSteeringService
    from backend.models import create_runtime
    from backend.research import ResearchStore
    from chatbot_feature.emotion_reasoner.risk_screen import ALLOWED, screen
    from .v3_common import messages_of
    seed_everything()
    tag = tag_of(args.model)
    out = dev_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    items = dev_items()
    check_lock(out / "lock.json", {"model": args.model_id, "max_new_tokens": MAX_NEW_TOKENS, "multipliers": list(MULTIPLIERS),
                                   "state_alpha_range": list(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"])),
                                   "items": hashlib.sha256("\n".join(items.key).encode()).hexdigest(), "smoke": SMOKE})
    path = out / "generations.jsonl"
    done = read_records(path, lambda r: (r["key"], r["kind"]))
    if args.retry_failed:
        done = {k: v for k, v in done.items() if not v.get("failed")}
    runtime = create_runtime(args.model, model_id=args.model_id)
    service = EmotionSteeringService(runtime, research=ResearchStore(tag, persist=False))
    progress = Progress(f"dev-generate_{tag}", len(items), sum(all((key, kind) in done for kind in ("plan",)) for key in items.key))
    with path.open("a", encoding="utf-8") as handle:
        def write(record: dict[str, Any]) -> None:
            done[(record["key"], record["kind"])] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        for row in items.itertuples():
            messages = messages_of(row)
            if (row.key, "plan") not in done:
                turn_id, plan = service.begin_turn(messages, conversation_id=f"e5dev-{row.key}", steering_mode="always")
                risk = screen(row.text, row.history)
                eligible = (risk["category"] in ALLOWED and not plan["safety_flag"] and plan["strategy"] != "crisis_support"
                            and bool(plan.get("steering_alpha")) and bool(plan.get("steering_panel")))
                write({"key": row.key, "kind": "plan", "strategy": plan["strategy"], "target": plan["response_emotion"],
                       "alpha": float(plan.get("steering_alpha") or 0.0), "panel": plan.get("steering_panel"),
                       "state_level": plan.get("state_level"), "safety_flag": bool(plan["safety_flag"]),
                       "risk_category": risk["category"], "eligible": eligible, "user_emotion": plan["user_emotion"],
                       "confidence": plan["confidence"], "intensity": plan["intensity"],
                       "system_prompt": service._turns[turn_id]["system_prompt"]})
            plan = done[(row.key, "plan")]
            kinds: list[tuple[str, Callable[[], dict[str, Any]]]] = []
            if row.study == "supplement" or SMOKE:
                for kind, mode in (("E3", "off"), ("E4", "always")):
                    def work(mode=mode, kind=kind):
                        from . import steering_modes as sm
                        turn_id, p = service.begin_turn(messages, conversation_id=f"e5dev-{row.key}-{kind}", steering_mode=mode)
                        started = time.perf_counter()
                        analysis = service.respond(turn_id, max_new_tokens=MAX_NEW_TOKENS)
                        return sm.reply_record(row.key, kind, p, analysis, time.perf_counter() - started)
                    kinds.append((kind, work))
            if plan["eligible"]:
                for multiplier in MULTIPLIERS:
                    def work(multiplier=multiplier):
                        alpha = round(multiplier * plan["alpha"], 4)
                        started = time.perf_counter()
                        outputs = runtime.generate_candidates(messages, max_new_tokens=MAX_NEW_TOKENS,
                                                              system_prompt=plan["system_prompt"],
                                                              steering_emotion=plan["panel"], alphas=[0.0, alpha])
                        return {"alpha": alpha, "multiplier": multiplier, "batch0": outputs[0]["response"],
                                "candidate": outputs[1]["response"],
                                "batch0_scores": service.scorer.score(outputs[0]["response"]),
                                "candidate_scores": service.scorer.score(outputs[1]["response"]),
                                "decode_positions": min((layer["decode_positions"] for layer in outputs[1]["trace"]["layers"]), default=0),
                                "latency_seconds": round(time.perf_counter() - started, 3)}
                    kinds.append((f"x{multiplier:.2f}", work))
            failed = False
            for kind, work in kinds:
                if (row.key, kind) in done:
                    continue
                result, error = with_retry("dev-generate", f"{tag} {row.key} {kind}", work)
                failed |= error is not None
                write({"key": row.key, "kind": kind, **(result or {}), "failed": error})
            progress.update(failed=failed, note=row.key)
    del service, runtime
    gpu_release()


def dev_replies(tag: str) -> pd.DataFrame:
    """Every development reply to rate: one row per (key, kind) - E3 / E4 (sent) / E4cand / E4b0 / x<m> / b<m>."""
    from .e5_common import development_table
    items = dev_items().set_index("key")
    records = read_records(dev_dir(tag) / "generations.jsonl", lambda r: (r["key"], r["kind"]))
    plans = {key: record for (key, kind), record in records.items() if kind == "plan"}
    rows = []

    def add(key: str, kind: str, reply: Any) -> None:
        if not isinstance(reply, str):
            return
        item, plan = items.loc[key], plans.get(key, {})
        rows.append({"key": key, "kind": kind, "reply": reply, "text": item.text, "history": item.history,
                     "strategy": plan.get("strategy", item.get("strategy")), "target": plan.get("target") or item.get("target") or "love",
                     "lang3": _lang3(item.text)})
    existing = development_table(tag)
    existing["key"] = existing.study + ":" + existing.item_id
    existing = existing.set_index("key")
    for key in items.index:
        if key in existing.index:
            row = existing.loc[key]
            add(key, "E3", row.kg_reply)
            add(key, "E4", row.e4_reply)
            add(key, "E4cand", row.candidate)
            add(key, "E4b0", row.batch0)
        for kind in ("E3", "E4"):
            record = records.get((key, kind))
            if record and not record.get("failed"):
                add(key, kind, record["reply"])
                if kind == "E4":
                    add(key, "E4cand", record.get("steered"))
                    add(key, "E4b0", record.get("unsteered"))
        for multiplier in MULTIPLIERS:
            record = records.get((key, f"x{multiplier:.2f}"))
            if record and not record.get("failed") and record.get("candidate") is not None:
                add(key, f"x{multiplier:.2f}", record["candidate"])
                add(key, f"b{multiplier:.2f}", record["batch0"])
    frame = pd.DataFrame(rows).drop_duplicates(["key", "kind"], keep="last").reset_index(drop=True)
    return frame


def _lang3(text: str) -> str:
    from chatbot_feature.emotion_reasoner.selective import language_of
    return language_of(text)


def dev_score(args) -> None:
    tag = tag_of(args.model)
    out = dev_dir(tag)
    table = dev_replies(tag)
    table.drop(columns=["history"]).to_csv(out / "dev_replies.csv", index=False)
    for judge in judges(out):
        ratings = rate_rows(judge, table, f"dev_{tag}")
        ratings.insert(0, "kind", table.kind)
        ratings.insert(0, "key", table.key)
        ratings.to_csv(out / f"dev_ratings_{judge.name}.csv", index=False)
        judge.close()
    print(f"{len(table)} development replies rated by both judges -> {out}")


def calibration_table(tag: str) -> pd.DataFrame:
    """One row per development message in the format of ``e5_fit.simulate`` (mean of both judges)."""
    from chatbot_feature.emotion_reasoner.risk_screen import ALLOWED, screen
    from .e5_common import development_table
    from .v3_common import JUDGES
    out = dev_dir(tag)
    items = dev_items().set_index("key")
    records = read_records(out / "generations.jsonl", lambda r: (r["key"], r["kind"]))
    replies = pd.read_csv(out / "dev_replies.csv")
    rated = None
    for name in JUDGES:
        ratings = pd.read_csv(out / f"dev_ratings_{name}.csv")
        metrics = [c for c in ratings.columns if c not in ("key", "kind")]
        ratings = ratings.set_index(["key", "kind"])[metrics]
        rated = ratings if rated is None else (rated + ratings)
    rated = rated / len(JUDGES)
    existing = development_table(tag)
    existing["key"] = existing.study + ":" + existing.item_id
    existing = existing.set_index("key")
    text_of = replies.set_index(["key", "kind"]).reply
    rows = []
    for key, item in items.iterrows():
        plan = records.get((key, "plan"), {})
        risk = screen(item.text, item.history)
        row = {"key": key, "study": item.study, "item_id": item.item_id, "group": item.get("group"),
               "category": item.get("category"), "text": item.text, "lang3": _lang3(item.text),
               "strategy": plan.get("strategy", item.get("strategy")), "target": plan.get("target", item.get("target")),
               "user_emotion": plan.get("user_emotion", item.get("user_emotion")), "intensity": plan.get("intensity", item.get("intensity")),
               "risk_category": risk["category"], "safety_flag": bool(plan.get("safety_flag", item.get("safety_flag"))),
               "eligible": bool(plan.get("eligible", False)) and risk["category"] in ALLOWED,
               "alpha_e4": plan.get("alpha")}
        e4 = records.get((key, "E4"))
        if key in existing.index:
            row["s0_batch"] = existing.loc[key, "s0_batch"]
            row["e4_applied"] = existing.loc[key, "e4_applied"]
            # The new candidates are comparable with the stored E4 data only if today's plan and batch reproduce it.
            stored = existing.loc[key]
            half = records.get((key, f"x{MULTIPLIERS[0]:.2f}")) or {}
            row["plan_matches_stored_e4"] = (abs(float(plan.get("alpha") or 0) - float(stored.e4_planned)) < 1e-6
                                             and plan.get("strategy") == stored.strategy)
            row["batch0_matches_stored_e4"] = (None if not half.get("batch0") or not isinstance(stored.batch0, str)
                                               else half["batch0"].strip() == stored.batch0.strip())
        elif e4 and e4.get("unsteered_scores"):
            row["s0_batch"] = e4["unsteered_scores"].get(row["target"], np.nan)
            row["e4_applied"] = e4.get("applied_alpha", 0.0)
        e4_sent = (row.get("e4_applied") or 0) > 0
        kinds = {"kg": "E3", "c100": "E4cand", "b100": "E4b0"}
        kinds.update({f"c{int(m * 100):03d}": f"x{m:.2f}" for m in MULTIPLIERS})
        kinds.update({f"b{int(m * 100):03d}": f"b{m:.2f}" for m in MULTIPLIERS})
        for prefix, kind in kinds.items():
            if (key, kind) not in text_of.index:
                continue
            row[f"{prefix}_text"] = text_of.loc[(key, kind)]
            if (key, kind) in rated.index:
                for metric, value in rated.loc[(key, kind)].items():
                    row[f"{prefix}_{metric}"] = value
        if not e4_sent and "c100_judge_emotion" in row:
            pass                                   # E4 rejected it; E5 (stricter) would too - kept for completeness
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("key")
    return frame


def calibrate(args) -> None:
    from backend.emotion_service import STATE_ALPHA_RANGE
    from . import e5_fit
    tag = tag_of(args.model)
    table = calibration_table(tag)
    out = base_dir() / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    table.drop(columns=[c for c in table.columns if c.endswith("_text")]).to_csv(out / f"{tag}_table.csv")
    alpha_range = STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"])
    consistency = {}
    if "plan_matches_stored_e4" in table:
        known = table.plan_matches_stored_e4.dropna().astype(bool)
        batch = table.batch0_matches_stored_e4.dropna().astype(bool)
        consistency = {"plan_matches_stored_e4": float(known.mean()) if len(known) else None,
                       "batch0_matches_stored_e4": float(batch.mean()) if len(batch) else None,
                       "plan_mismatches": table.index[~table.plan_matches_stored_e4.fillna(True).astype(bool)].tolist()[:50]}
        print(f"  consistency with the stored E4 data: plans {consistency['plan_matches_stored_e4']}, "
              f"unsteered batch rows {consistency['batch0_matches_stored_e4']}")
    source = {"table_sha256": e5_fit.table_fingerprint(table.reset_index()), "messages": int(len(table)),
              "studies": table.study.value_counts().to_dict(), "multipliers_rated": ["1.0", *[f"{m:g}" for m in MULTIPLIERS]],
              "consistency_with_stored_e4": consistency}
    policy, report = e5_fit.fit(table, tag, alpha_range, source=source)
    sensitivity = {}
    original = e5_fit.MIN_TONE_GAIN
    for value in (0.10, 0.20):
        e5_fit.MIN_TONE_GAIN = value
        chosen, evidence = e5_fit.fit_cells(table, report["settings"], tag)
        sensitivity[f"MIN_TONE_GAIN={value}"] = {"steered_cells": {k: list(v) for k, v in chosen.items() if v},
                                                 "in_sample": {k: evidence["population"][k] for k in ("tone_gain", "steered_share", "passes")}}
    e5_fit.MIN_TONE_GAIN = original
    report["sensitivity_not_used_for_selection"] = sensitivity
    POLICIES.mkdir(parents=True, exist_ok=True)
    policy.frozen = False
    target = (base_dir() / "policies") / f"{tag}.draft.json"
    digest = policy.save(target)
    write_json(out / f"{tag}_report.json", report)
    cells = pd.DataFrame([{"cell": cell, "option": option, **{k: v for k, v in notes.items() if not isinstance(v, list)},
                           "reasons": "; ".join(notes["reasons"])}
                          for cell, options in report["cells"].items() for option, notes in options.items()])
    cells.to_csv(out / f"{tag}_cells.csv", index=False)
    population = report["population_in_sample"]
    print(f"{tag}: draft policy {target} (sha256 {digest[:12]}); settings {report['settings']}")
    print(f"  steered cells: {report['steered_cells'] or 'none'}")
    print(f"  in-sample: tone {population['tone_gain']:+.3f}, steered {population['steered_share']:.2f}, "
          f"population check {'passes' if population['passes'] else 'FAILS'}")
    for row in report["grid"]:
        print(f"  cv {row['settings']}: tone {row['cv_tone_gain']:+.3f} steered {row['cv_steered_share']:.2f} "
              f"passes={row['cv_passes']}")


# ------------------------------------------------------------------------------------- freeze / register

def fingerprints() -> dict[str, str]:
    files = [Path(name) for name in CODE_FILES]
    for tag in TAGS:
        base = Path("results") / f"{tag}_generation_emotion"
        files += sorted((base / "steering_panels").glob("*.json")) + sorted((base / "steering_policy").glob("*.json"))
    files += sorted(POLICIES.glob("*.json"))
    from . import e5_data
    files += [e5_data.HELDOUT, e5_data.MANIFEST]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.exists()}


def hypotheses(tag: str) -> list[dict[str, Any]]:
    from .e5_common import MARGIN_QUALITY, MARGIN_SAFETY
    rows = [
        ("H1", "E5 raises emotional tone over KG-only (E3)", "E5", "E3", "judge_emotion", "greater", None),
        ("H2", f"E5 keeps reply quality vs E3 (non-inferiority, margin {MARGIN_QUALITY})", "E5", "E3", "quality", "noninferior", MARGIN_QUALITY),
        ("H3", f"E5 keeps safety vs E3 (non-inferiority, margin {MARGIN_SAFETY})", "E5", "E3", "safety", "noninferior", MARGIN_SAFETY),
        ("H4", "E5 has higher reply quality than E4", "E5", "E4", "quality", "greater", None),
        ("H5", "E5 has higher safety than E4", "E5", "E4", "safety", "greater", None),
        ("H6", "E5 raises emotional tone over E5-old", "E5", "E5old", "judge_emotion", "greater", None),
        ("H7", f"E5 keeps reply quality vs E5-old (non-inferiority, margin {MARGIN_QUALITY})", "E5", "E5old", "quality", "noninferior", MARGIN_QUALITY),
        ("H8", f"E5 keeps emotional validation vs E3 (non-inferiority, margin {MARGIN_QUALITY})", "E5", "E3", "validation", "noninferior", MARGIN_QUALITY),
    ]
    return [{"id": f"{tag}:{hid}", "model": tag, "hypothesis": hid, "claim": claim, "first": first, "second": second,
             "metric": metric, "test": test, "margin": margin} for hid, claim, first, second, metric, test, margin in rows]


def freeze(args) -> None:
    from chatbot_feature.emotion_reasoner.selective import SelectivePolicy
    from . import e5_data
    from .freeze_v2 import sha256
    if list((ROOT / "heldout").glob("*/generations.jsonl")):
        raise SystemExit("Held-out replies exist; the policies can no longer change.")
    drafts = {tag: POLICIES / f"{tag}.draft.json" for tag in TAGS}
    missing = [tag for tag, path in drafts.items() if not path.exists()]
    if missing and not args.allow_missing:
        raise SystemExit(f"No calibrated draft for {missing}: run calibrate for them, or pass --allow-missing to freeze a "
                         "policy that steers nothing for them (reported as a deviation).")
    e5_data.load("heldout")                                       # checks the frozen held-out hash
    policies = {}
    for tag in TAGS:
        if tag in missing:
            from backend.emotion_service import STATE_ALPHA_RANGE
            policy = SelectivePolicy(model=tag, version="e5-1-uncalibrated", frozen=True,
                                     alpha_range=tuple(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"])),
                                     provenance={"note": "no calibration was run for this model; nothing is steered"})
        else:
            policy = SelectivePolicy.load(drafts[tag])
            policy.frozen = True
        policies[tag] = {"sha256": policy.save(POLICIES / f"{tag}.json"), "steered_cells":
                         {k: list(c.multipliers) for k, c in policy.cells.items() if c.multipliers},
                         "deficit": policy.deficit, "acceptance": policy.acceptance, "version": policy.version}
    from . import confirmatory
    registration = {
        "title": "MindMate E5: quality-constrained adaptive selective emotion-neuron steering",
        "registered_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "conditions": {name: {"prompt": {"plain": "MindMate prompt", None: "the chat's KG strategy prompt"}[variant],
                              "steering": mode} for name, (variant, mode) in CONDITIONS.items()},
        "controls": list(CONTROLS), "policies": policies, "missing_calibration": missing,
        "hypotheses": [row for tag in TAGS for row in hypotheses(tag)],
        "preference_pairs": [f"{a} vs {b}" for a, b in (*PRIMARY_PAIRS, *SECONDARY_PAIRS)],
        "analysis": {
            "unit": "message; every condition of a message shares the plan (strategy, response emotion, state level)",
            "population": "every held-out message (English, Hinglish, Hindi, neutral, mixed, safety); subsets reported separately",
            "outcomes": {"judge_emotion": "tone toward the planned response emotion, 1-5, mean of two judges",
                         "quality": "mean of appropriateness, empathy, relevance, fluency, naturalness (1-5, two judges)",
                         "safety": "safety rating 1-5, mean of two judges (a separate constraint)",
                         "validation": "emotional-validation rating 1-5, mean of two judges"},
            "tests": confirmatory.ANALYSIS["tests"] | {"noninferior": "one-sided paired t-test of the per-message "
                                                                         "difference > -margin (margin per hypothesis)"},
            "correction": "Holm-Bonferroni over every registered test of every model; confirmed when adjusted p < 0.05",
            "estimates": "mean difference, 95% percentile bootstrap CI (10,000 resamples), Wilcoxon p, Cohen's dz",
            "secondary": ["each hypothesis by language, by safety category, by strategy", "each judge separately",
                          "pairwise preferences (both orders averaged)", "strategy adherence, language match",
                          "steering attempt / retained / rejected rates, alpha distribution, latency",
                          "batch-noise control: the batched unsteered rows vs E3"],
            "missing_data": "a failed generation is rated as an empty reply; a model that cannot be run counts as not "
                            "confirmed for each of its hypotheses",
            "note": "If a model's frozen E5 policy steers no context, E5 equals E3 by construction: H1/H6 cannot be "
                    "confirmed and that is the expected result, not a failure of the pipeline."},
        "settings": {"max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy", "judges": {j.name: j.model_id for j in judges(ROOT)},
                     "heldout_sha256": sha256(e5_data.HELDOUT)},
        "fingerprints": fingerprints(), "git": confirmatory.git_state(),
    }
    write_json(REGISTRATION, registration)
    REGISTRATION_MD.write_text(registration_markdown(registration), encoding="utf-8")
    print(f"Frozen {len(policies)} policies; registered {len(registration['hypotheses'])} hypotheses.")
    print(f"{REGISTRATION_MD} sha256 {sha256(REGISTRATION_MD)} - commit it (git) before generating held-out replies.")


def registration_markdown(reg: dict[str, Any]) -> str:
    lines = [f"# {reg['title']}", "", f"Registered: {reg['registered_at_utc']}  ",
             f"Held-out set sha256 `{reg['settings']['heldout_sha256']}`; git `{reg['git']['head']}` "
             f"(uncommitted changes: {reg['git']['uncommitted_changes']}); {len(reg['fingerprints'])} files fingerprinted.", "",
             "## Conditions", "", "| condition | prompt | steering |", "|---|---|---|"]
    lines += [f"| {name} | {cell['prompt']} | {cell['steering']} |" for name, cell in reg["conditions"].items()]
    lines += ["", "## Frozen E5 policies", "", "| model | steered cells (strength multipliers) | deficit gate | sha256 |",
              "|---|---|---|---|"]
    for tag, policy in reg["policies"].items():
        cells = ", ".join(f"{k} x{'/'.join(map(str, v))}" for k, v in policy["steered_cells"].items()) or "none"
        gate = f"English, classifier >= {policy['deficit']['threshold']}" if policy["deficit"].get("enabled") else "off"
        lines.append(f"| {tag} | {cells} | {gate} | `{policy['sha256'][:12]}` |")
    lines += ["", "## Hypotheses", "", "| id | claim | contrast | outcome | test |", "|---|---|---|---|---|"]
    lines += [f"| {h['id']} | {h['claim']} | {h['first']} - {h['second']} | {h['metric']} | {h['test']} |"
              for h in reg["hypotheses"]]
    lines += ["", "## Analysis", "", "```json", json.dumps(reg["analysis"], indent=2), "```", ""]
    return "\n".join(lines)


def check_registration() -> dict[str, Any]:
    if not REGISTRATION.exists():
        raise SystemExit(f"{REGISTRATION} is missing: run --stages freeze first (policies and hypotheses come first).")
    registration = json.loads(REGISTRATION.read_text())
    now = fingerprints()
    changed = sorted(path for path in set(now) | set(registration["fingerprints"])
                     if now.get(path) != registration["fingerprints"].get(path))
    if changed:
        raise SystemExit(f"These files changed after registration: {changed}. The registered study must run on the "
                         "registered code, policies and data.")
    return registration


# ------------------------------------------------------------------------------------------ held-out generation

def heldout_dir(tag: str) -> Path:
    return base_dir() / "heldout" / tag


def heldout_items() -> pd.DataFrame:
    from . import e5_data
    from . import steering_modes as sm
    if SMOKE:
        items = sm.load_items(SMOKE)
        items["category"] = items.group
        items["lang3"] = items.text.map(_lang3)
        return items
    return e5_data.load("heldout")


def smoke_policy(tag: str) -> Any:
    """Pipeline-check policy: every context steered at x1.0 with the deficit gate on (never used for results)."""
    from backend.emotion_service import STATE_ALPHA_RANGE
    from chatbot_feature.emotion_reasoner.selective import PolicyCell, SelectivePolicy
    return SelectivePolicy(model=tag, version="smoke", alpha_range=tuple(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"])),
                           cells={f"*|{language}": PolicyCell((1.0,), "smoke test only") for language in ("english", "hinglish", "hindi")},
                           deficit={"enabled": True, "signal": "gate_target", "threshold": 0.30, "languages": ["english"]})


def condition_record(item_id: str, condition: str, plan: dict[str, Any], analysis: dict[str, Any] | None,
                     latency: float, failure: str | None = None) -> dict[str, Any]:
    from . import steering_modes as sm
    record = sm.reply_record(item_id, condition, plan, analysis, latency, failure)
    record.update(condition=condition, prompt_style=plan.get("prompt_style"), user_emotion=plan.get("user_emotion"),
                  confidence=plan.get("confidence"), intensity=plan.get("intensity"), valence=plan.get("valence"),
                  failed=failure)
    if analysis and analysis.get("e5"):
        record["e5"] = analysis["e5"]
        record["candidates"] = (analysis.get("effect") or {}).get("candidates")
    steering = (analysis or {}).get("steering") or {}
    if steering.get("applied"):
        record["hooks"] = {"layers": len(steering.get("layers") or []), "alpha": steering.get("alpha")}
    return record


def run_heldout(service: Any, items: pd.DataFrame, path: Path, progress: Progress, retry_failed: bool) -> None:
    from .v3_common import messages_of
    from chatbot_feature.emotion_reasoner.risk_screen import screen
    done = read_records(path, lambda r: (r["item_id"], r["condition"]))
    if retry_failed:
        done = {k: v for k, v in done.items() if not v.get("failed")}
    with path.open("a", encoding="utf-8") as handle:
        def write(record: dict[str, Any]) -> None:
            done[(record["item_id"], record["condition"])] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        for row in items.itertuples():
            if all((row.item_id, condition) in done for condition in CONDITIONS):
                continue
            messages = messages_of(row)
            risk = screen(row.text, row.history)
            failed = False
            for condition, (variant, mode) in CONDITIONS.items():
                if (row.item_id, condition) in done:
                    continue
                unit = f"{service.tag} {row.item_id} {condition}"
                started = time.perf_counter()
                if condition == "E5":
                    turn_id, plan, pre = service.plan_selective(messages, conversation_id=f"e5-{row.item_id}-E5")
                    e3 = done.get((row.item_id, "E3"))
                    if not pre["eligible"] and e3 is not None and not e3.get("failed"):
                        # E5 does not generate candidates: the knowledge-graph-only path, i.e. E3's reply.
                        record = {**e3, "condition": "E5", "mode": "E5", "reused_from": "E3", "status": "not_eligible",
                                  "e5": {**service._turns[turn_id]["selective"]["pre"], "attempted": False, "retained": False,
                                         "status": "not_eligible", "decision_reason": pre["reason"],
                                         "risk_category": risk["category"], "applied_alpha": 0.0,
                                         "policy_version": service.selective_policy.version,
                                         "policy_fingerprint": service.policy_fingerprint}}
                    else:
                        analysis, error = with_retry("generate", unit, lambda: service.run_selective(turn_id, MAX_NEW_TOKENS))
                        record = condition_record(row.item_id, condition, plan, analysis, time.perf_counter() - started, error)
                else:
                    turn_id, plan = service.begin_turn(messages, conversation_id=f"e5-{row.item_id}-{condition}",
                                                       steering_mode=mode, prompt_variant=variant)
                    twin = {"E3": "E1", "E4": "E2"}.get(condition)
                    reuse = None
                    if twin and plan["prompt_style"] == "mindmate_only" and not (done.get((row.item_id, twin)) or {}).get("failed", True):
                        reuse = twin                      # the chat itself sends the MindMate prompt alone (Sarvam, Hinglish/Hindi)
                    elif condition == "E5old" and not service.will_steer(turn_id) and not (done.get((row.item_id, "E3")) or {}).get("failed", True):
                        reuse = "E3"                      # the policy does not steer: same prompt and greedy decoding as E3
                    if reuse:
                        record = {**done[(row.item_id, reuse)], "condition": condition, "mode": condition, "reused_from": reuse}
                        if condition == "E5old":
                            record.update(status="not_planned", planned_alpha=0.0, applied_alpha=0.0,
                                          reason=plan["strength_factors"][0]["reason"] if plan.get("strength_factors") else None)
                    else:
                        analysis, error = with_retry("generate", unit, lambda: service.respond(turn_id, max_new_tokens=MAX_NEW_TOKENS))
                        record = condition_record(row.item_id, condition, plan, analysis, time.perf_counter() - started, error)
                record["risk_category"] = risk["category"]
                failed |= bool(record.get("failed"))
                write(record)
            progress.update(failed=failed, note=row.item_id)


def plan_mismatches(records: Iterable[dict[str, Any]]) -> list[str]:
    by_item: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_item.setdefault(record["item_id"], {})[record["condition"]] = record
    bad = []
    for item, cells in by_item.items():
        strategies = {cell.get("strategy") for cell in cells.values() if not cell.get("failed")}
        levels = {round(cell["state_level"], 4) for cell in cells.values() if cell.get("state_level") is not None}
        if len(strategies) > 1 or len(levels) > 1:
            bad.append(item)
    return bad


def generate(args) -> None:
    from backend.emotion_service import STATE_ALPHA_RANGE
    from backend.models import create_runtime
    from backend.research import ResearchStore
    from backend.selective_service import SelectiveSteeringService, panel_manifest
    from chatbot_feature.emotion_reasoner.selective import SelectivePolicy
    from .freeze_v2 import sha256
    seed_everything()
    tag = tag_of(args.model)
    if not SMOKE:
        check_registration()
        if args.model_id != MODEL_REGISTRY[args.model].config.model_id:
            raise SystemExit("The registered study uses the registry's model checkpoint.")
        policy = SelectivePolicy.load(POLICIES / f"{tag}.json")
        if not policy.frozen:
            raise SystemExit(f"{POLICIES / f'{tag}.json'} is not frozen.")
    else:
        draft = base_dir() / "policies" / f"{tag}.draft.json"
        policy = SelectivePolicy.load(draft) if (draft.exists() and not args.smoke_permissive) else smoke_policy(tag)
    items = heldout_items()
    out = heldout_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    check_lock(out / "lock.json", {"items": sha256(Path("results/steering_modes/items.csv")) if SMOKE else
                                   json.loads(REGISTRATION.read_text())["settings"]["heldout_sha256"],
                                   "model": args.model_id, "max_new_tokens": MAX_NEW_TOKENS, "policy": policy.fingerprint(),
                                   "state_alpha_range": list(STATE_ALPHA_RANGE.get(tag, STATE_ALPHA_RANGE["default"])),
                                   "smoke": SMOKE})
    runtime = create_runtime(args.model, model_id=args.model_id)
    service = SelectiveSteeringService(runtime, policy, research=ResearchStore(tag, persist=False))
    write_json(out / "panel_manifest.json", panel_manifest(runtime))
    path = out / "generations.jsonl"
    done = read_records(path, lambda r: (r["item_id"], r["condition"]))
    finished = sum(all((item, c) in done for c in CONDITIONS) for item in items.item_id)
    progress = Progress(f"generate_{tag}", len(items), finished)
    run_heldout(service, items, path, progress, args.retry_failed)
    records = list(read_records(path, lambda r: (r["item_id"], r["condition"])).values())
    bad = plan_mismatches(records)
    failures = sum(bool(r.get("failed")) for r in records)
    print(f"{len(records)} held-out replies in {path}; {failures} failed; {len(bad)} plan mismatches")
    del service, runtime
    gpu_release()


# ------------------------------------------------------------------------------------------ held-out scoring

def long_table(tag: str) -> pd.DataFrame:
    """One row per (message, condition) plus the batch-noise control rows, with E5 decision fields."""
    items = heldout_items().set_index("item_id")
    records = read_records(heldout_dir(tag) / "generations.jsonl", lambda r: (r["item_id"], r["condition"]))
    missing = [(item, c) for item in items.index for c in CONDITIONS if (item, c) not in records]
    if missing:
        raise SystemExit(f"Generation is incomplete ({len(missing)} replies missing, e.g. {missing[0]}).")
    rows = []
    for item_id, item in items.iterrows():
        cells = {c: records[(item_id, c)] for c in CONDITIONS}
        target = next((cells[c].get("response_emotion") for c in ("E4", "E2", "E5", "E3") if cells[c].get("response_emotion")), None) or "love"
        base = {"item_id": item_id, "group": item.get("group"), "category": item.get("category"), "lang3": _lang3(item.text),
                "turns": item.get("turns"), "text": item.text, "history": item.history, "target": target,
                "strategy": cells["E3"].get("strategy"), "user_emotion": cells["E3"].get("user_emotion"),
                "confidence": cells["E3"].get("confidence"), "intensity": cells["E3"].get("intensity"),
                "risk_category": cells["E3"].get("risk_category"), "safety_flag": bool(cells["E3"].get("safety_flag")),
                "e5_attempted": bool((cells["E5"].get("e5") or {}).get("attempted")),
                "e5_eligible_stage": (cells["E5"].get("e5") or {}).get("stage")}
        for condition, record in cells.items():
            rows.append({**base, "condition": condition, "reply": record.get("reply") or "", "status": record.get("status"),
                         "planned_alpha": float(record.get("planned_alpha") or 0.0),
                         "applied_alpha": float(record.get("applied_alpha") or 0.0),
                         "reused_from": record.get("reused_from"), "failed": bool(record.get("failed")),
                         "latency_seconds": record.get("latency_seconds")})
        for condition, source in (("E4_batch0", "E4"), ("E5_batch0", "E5")):
            batch0 = cells[source].get("unsteered")
            if cells[source].get("planned_alpha") and isinstance(batch0, str) and batch0:
                rows.append({**base, "condition": condition, "reply": batch0, "status": "control", "planned_alpha": 0.0,
                             "applied_alpha": 0.0, "reused_from": None, "failed": False, "latency_seconds": None})
    table = pd.DataFrame(rows)
    table.loc[table.failed, "reply"] = ""
    return table


def score(args) -> None:
    from .v3_common import automatic_scores
    from .steering_lab import Judge
    from chatbot_feature.emotion_reasoner.selective import language_of
    if not SMOKE:
        check_registration()
    tag = tag_of(args.model)
    out = heldout_dir(tag)
    table = automatic_scores(long_table(tag))
    table["reply_language"] = table.reply.map(language_of)
    table["language_match"] = (table.reply_language == table.lang3).astype(float)
    table.drop(columns=["history"]).to_csv(out / "replies.csv", index=False)
    conditions = table[table.condition.isin(CONDITIONS)]
    wide = conditions.pivot(index="item_id", columns="condition", values="reply")
    first_rows = conditions.drop_duplicates("item_id").set_index("item_id").loc[wide.index]
    contexts = [Judge.context(text, history) for text, history in zip(first_rows.text, first_rows.history)]
    for judge in judges(out):
        ratings = rate_rows(judge, table, f"heldout_{tag}")
        ratings.to_csv(out / f"ratings_{judge.name}.csv", index=False)
        preferences = pd.DataFrame(index=wide.index)
        for first, second in (*PRIMARY_PAIRS, *SECONDARY_PAIRS):
            preferences[f"{first}_over_{second}"] = judge.prefer(contexts, wide[first].fillna("").tolist(),
                                                                 wide[second].fillna("").tolist(), label=f"pref_{tag}_{first}_{second}")
        preferences.to_csv(out / f"preferences_{judge.name}.csv")
        judge.close()
    print(f"{len(table)} held-out replies scored by both judges -> {out}")


# ------------------------------------------------------------------------------------------------- main

def main() -> None:
    from . import e5_report
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help="Model for the per-model stages.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--stages", nargs="+", required=True,
                        choices=("audit", "build", "dev-generate", "dev-score", "calibrate", "freeze", "generate", "score",
                                 "report", "verdict", "plots", "examples"))
    parser.add_argument("--smoke", type=int, default=0, help="Pipeline check on N already-used messages "
                                                             "(results/e5_selective/smoke/; no registration).")
    parser.add_argument("--smoke-permissive", action="store_true", help="smoke generate: steer every context (tests the "
                                                                        "candidate path even when the draft steers nothing)")
    parser.add_argument("--retry-failed", action="store_true", help="Re-run units whose earlier attempt failed.")
    parser.add_argument("--allow-missing", action="store_true", help="freeze: models without calibration get a policy "
                                                                     "that steers nothing (reported as a deviation).")
    args = parser.parse_args()
    global SMOKE
    SMOKE = args.smoke
    per_model = {"dev-generate", "dev-score", "calibrate", "generate", "score", "report"}
    if set(args.stages) & per_model and not args.model:
        parser.error(f"--model is required for {sorted(set(args.stages) & per_model)}")
    if args.model:
        args.model_id = args.model_id or MODEL_REGISTRY[args.model].config.model_id
    if SMOKE and set(args.stages) & {"build", "freeze", "verdict"}:
        parser.error("--smoke covers the per-model stages only.")
    stages = {"audit": lambda a: __import__("generation_analysis.e5_audit", fromlist=["main"]).main(),
              "build": build_stage, "dev-generate": dev_generate, "dev-score": dev_score, "calibrate": calibrate,
              "freeze": freeze, "generate": generate, "score": score,
              "report": lambda a: e5_report.report(tag_of(a.model), base_dir(), smoke=bool(SMOKE)),
              "verdict": lambda a: e5_report.verdict(base_dir()), "plots": lambda a: e5_report.plots(base_dir()),
              "examples": lambda a: e5_report.examples(base_dir())}
    for stage in args.stages:
        print(f"== stage {stage}" + (f" ({args.model})" if args.model else "") + (f" [smoke {SMOKE}]" if SMOKE else ""), flush=True)
        stages[stage](args)


def build_stage(args) -> None:
    from . import e5_data
    result = e5_data.build()
    manifest = result["manifest"]
    print(json.dumps({"heldout": manifest["heldout"]["n"], "dev_supplement": manifest["dev_supplement"]["n"],
                      "excluded_candidates": manifest["leakage"]["excluded_candidates"]}, indent=1))
    print(pd.DataFrame(manifest["heldout"]["by_group_category"]).to_string(index=False))


if __name__ == "__main__":
    main()
