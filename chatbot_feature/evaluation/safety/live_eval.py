"""Live GPU evaluation of the Safety & Risk Assessment Agent (Agent 1) with a real MindMate model.

The chat model of the chosen preset is loaded once (as the server loads it) and serves as the safety
assessor / validator through ``SafetyRuntimeModel``, exactly as in deployment.  Parts:

  classify  R0-R4 triage of every user turn: A (17-phrase cue list), B (rule screen, and the agent without
            context), C (full contextual agent) and ablations of C.  Per-class P/R/F1, macro-F1, confusion
            matrix, R3-R4 false-negative rate, R0 false-positive rate, per language / category, latency.
  outgoing  the outgoing-reply validator on curated safe and unsafe drafts (deterministic, model-only, hybrid)
            and whether unsafe drafts are replaced by the validated fixed template.
  pipeline  real HTTP requests to /api/chat and /api/chat/stream on an in-process server: routing, Emotion
            Reasoner and steering gating, template replies, stream ordering and leaks, fault injection.

Datasets: dev = cases.json (the set the screen was developed on; optimistic), heldout = benchmark.json.
Results: <out>/<dataset>/... and <out>/manifest.json.  No message text is written, only case ids and codes.

  python3 -m chatbot_feature.evaluation.safety.live_eval --model llama
  python3 -m chatbot_feature.evaluation.safety.live_eval --model sarvam --parts classify --ablations full context_free
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .harness import (ABLATION_TO_SYSTEM, DATASETS, LLM_SYSTEMS, SYSTEMS, RecordingModel, activation_checks,
                      load_dataset, outgoing_summary, run_classify, run_outgoing)
from .metrics import LABELS, latency, summarize


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        import transformers
        info.update(torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as exc:
        info["torch_error"] = type(exc).__name__
    return info


def load_model(args):
    from backend.models import create_runtime
    from chatbot_feature.safety.runtime import SafetyRuntimeModel
    started = time.perf_counter()
    runtime = create_runtime(args.model, model_id=args.model_id, load_in_4bit=args.load_in_4bit, allow_download=False)
    print(f"Loaded {runtime.model_id} in {time.perf_counter() - started:.0f} s", flush=True)
    safety_model = SafetyRuntimeModel(runtime)
    # Preflight: one real safety-model call.  If the call itself fails (not just its JSON), every assessment
    # would silently become an R1 fallback and the run would only measure the fallback path.
    started = time.perf_counter()
    try:
        raw = asyncio.run(safety_model.complete_safety('Return exactly this JSON object and nothing else: {"ok": true}'))
    except Exception as exc:
        raise SystemExit(f"PREFLIGHT FAILED: the safety model call raised {type(exc).__name__}: {exc}\n"
                         "Every assessment would fall back to R1; not running the evaluation.") from exc
    print(f"Preflight safety call OK ({time.perf_counter() - started:.1f} s, {len(raw)} chars)", flush=True)
    return runtime, safety_model


def classification_outputs(folder: Path, rows: list[dict], systems: list[str], calls: list[dict],
                           production_timeout: float) -> dict[str, Any]:
    write_csv(folder / "predictions_turns.csv", rows)
    final = [r for r in rows if r["final"]]
    write_csv(folder / "predictions_cases.csv", final)
    report: dict[str, Any] = {}
    failures = []
    for system in systems:
        cases = [r for r in final if r["system"] == system]
        turns = [r for r in rows if r["system"] == system and r["expected"]]
        entry = {"description": SYSTEMS[system], "case_level": summarize(cases), "turn_level": summarize(turns, ci=False),
                 "activation": activation_checks([r for r in rows if r["system"] == system])}
        if system in LLM_SYSTEMS or system == "C_no_llm":
            entry["latency_assessment"] = latency([r.get("assessment_ms") for r in turns])
        report[system] = entry
        matrix = entry["case_level"].get("confusion_matrix")
        if matrix:
            write_csv(folder / f"confusion_{system}.csv",
                      [{"expected": label, **{f"pred_{p}": matrix[i][j] for j, p in enumerate(LABELS)}}
                       for i, label in enumerate(LABELS)])
        for r in turns:
            if r["expected"] != r["predicted"]:
                gap = LABELS.index(r["predicted"]) - LABELS.index(r["expected"])
                failures.append({"system": system, "case_id": r["case_id"], "turn": r["turn"], "language": r["language"],
                                 "category": r["category"], "expected": r["expected"], "predicted": r["predicted"],
                                 "direction": "under_triage" if gap < 0 else "over_triage",
                                 "high_risk_miss": r["expected"] in ("R3", "R4") and r["predicted"] not in ("R3", "R4"),
                                 "fallback": r.get("fallback"), "reason": r.get("reason"), "evidence": r.get("evidence")})
    failures.sort(key=lambda f: (not f["high_risk_miss"], f["direction"] != "under_triage", f["system"], f["case_id"]))
    write_csv(folder / "failures.csv", failures)
    model_calls = [c for c in calls if c.get("part") == "classify"]
    fresh = [c for c in model_calls if not c["cached"]]
    report["_model_calls"] = {
        "calls": len(model_calls), "cached": len(model_calls) - len(fresh),
        "parse_status": {status: sum(c["parse"] == status for c in model_calls) for status in sorted({c["parse"] for c in model_calls})},
        "call_status": {status: sum(c["status"] == status for c in model_calls) for status in sorted({c["status"] for c in model_calls})},
        "latency_ms": latency([c["ms"] for c in model_calls if c["status"] == "ok"]),
        "production_timeout_s": production_timeout,
        "share_slower_than_production_timeout": (sum(c["ms"] > production_timeout * 1000 for c in model_calls if c["status"] == "ok")
                                                 / max(1, sum(c["status"] == "ok" for c in model_calls)))}
    return report


async def evaluate_async(args, model, dataset: dict, folder: Path, systems: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"dataset": {k: dataset[k] for k in ("name", "version", "sha256", "path", "source")},
                               "cases": len(dataset["cases"]), "user_turns": sum(len(c["messages"]) for c in dataset["cases"])}
    if "classify" in args.parts:
        print(f"[{dataset['name']}] classify: {', '.join(systems)}", flush=True)
        rows = await run_classify(dataset, systems, model, args.timeout)
        summary["classification"] = classification_outputs(folder, rows, systems, getattr(model, "calls", []),
                                                           args.production_timeout)
    if "outgoing" in args.parts and dataset.get("outgoing"):
        print(f"[{dataset['name']}] outgoing: {len(dataset['outgoing'])} drafts", flush=True)
        rows = await run_outgoing(dataset, model, args.timeout, args.country)
        write_csv(folder / "outgoing.csv", rows)
        summary["outgoing"] = outgoing_summary(rows)
    return summary


def main() -> None:
    from backend.config import MODEL_REGISTRY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), help="MindMate model preset to load (required "
                        "unless --no-model)")
    parser.add_argument("--model-id", help="Local checkpoint folder or HF id for this preset (default: the preset's)")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the preset's quantisation (Sarvam: 4-bit by default)")
    parser.add_argument("--datasets", nargs="+", default=["dev", "heldout"],
                        help="dev (cases.json), heldout (benchmark.json) or paths to JSON files in the same format")
    parser.add_argument("--parts", nargs="+", choices=("classify", "outgoing", "pipeline"),
                        default=["classify", "outgoing", "pipeline"])
    parser.add_argument("--ablations", nargs="+", choices=tuple(ABLATION_TO_SYSTEM),
                        default=["full", "context_free", "no_llm"],
                        help="Agent variants for 'classify' (A_keyword17 and B_rule_screen always run). "
                             "full=C, context_free=B_llm; no_prior, no_screen, no_normalization, no_llm = ablations")
    parser.add_argument("--endpoints", nargs="+", choices=("chat", "stream"), default=["chat", "stream"])
    parser.add_argument("--inject", type=int, default=None,
                        help="Fault injection in 'pipeline': how many of the dataset's unsafe normal-route drafts to force "
                             "as the chat reply, per endpoint (default: all; 0 = off)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Seconds the agent waits for the model in classify/outgoing (long, so accuracy is not "
                             "confounded by timeouts; see --production-timeout)")
    parser.add_argument("--production-timeout", type=float, default=float(os.getenv("MINDMATE_SAFETY_TIMEOUT", "30")),
                        help="Deployment timeout; classify reports the share of model calls slower than it, and the "
                             "pipeline part runs with it (as the server does)")
    parser.add_argument("--country", default="IN", help="Explicit ISO country for help resources ('' = none)")
    parser.add_argument("--limit", type=int, help="Use only the first N cases of each dataset (smoke test)")
    parser.add_argument("--out", type=Path, help="Output folder (default results/safety_eval/<model>)")
    parser.add_argument("--no-cache", action="store_true", help="Do not reuse cached safety-model outputs")
    parser.add_argument("--no-model", action="store_true",
                        help="Load no model: run only the model-free systems (A_keyword17, B_rule_screen, C_no_llm) "
                             "in 'classify'. Works without a GPU.")
    parser.add_argument("--save-raw", action="store_true",
                        help="Keep the first 400 characters of model outputs that fail the schema (synthetic data only)")
    args = parser.parse_args()
    args.country = args.country or None
    if not args.model and not args.no_model:
        parser.error("--model is required (or --no-model for the model-free baselines)")
    if args.no_model:
        args.parts = ["classify"]
        args.ablations = ["no_llm"]

    out = args.out or Path("results") / "safety_eval" / ("no_model" if args.no_model else args.model if not args.model_id
                                                       else f"{args.model}_{Path(args.model_id).name}")
    out.mkdir(parents=True, exist_ok=True)
    datasets = [load_dataset(DATASETS.get(name, Path(name))) for name in args.datasets]
    systems = ["A_keyword17", "B_rule_screen", *[ABLATION_TO_SYSTEM[a] for a in args.ablations]]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {"started": started, "model_preset": args.model, "model_id": args.model_id or (MODEL_REGISTRY[args.model].config.model_id if args.model else None),
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "git_commit": git_commit(), "environment": environment(), "systems": {s: SYSTEMS[s] for s in systems},
                "datasets": {d["name"]: {"sha256": d["sha256"], "cases": len(d["cases"]), "path": d["path"]} for d in datasets},
                "notes": ["Synthetic, not expert-annotated, not clinical data. Not a clinical validation.",
                          "dev = cases.json was used while the rule screen was written; its B/C scores are optimistic.",
                          "classify uses earlier user turns as history; 'pipeline' uses the real assistant replies.",
                          "A_keyword17 is binary; its hits are scored as R3 (a proxy, not a severity prediction).",
                          "Fault-injection rows force an unsafe chat reply to test the guard; they say nothing about "
                          "the model's own replies."]}
    write_json(out / "manifest.json", manifest)

    if args.no_model:
        runtime, model = None, RecordingModel(None)
        manifest["model_preset"] = manifest["model_id"] = None
    else:
        runtime, safety_model = load_model(args)
        model = RecordingModel(safety_model, None if args.no_cache else out / "llm_cache.jsonl", save_raw=args.save_raw)
    for dataset in datasets:
        if args.limit:
            dataset = {**dataset, "cases": dataset["cases"][:args.limit], "outgoing": dataset.get("outgoing", [])[:args.limit]}
        folder = out / dataset["name"]
        model.calls.clear()
        summary = asyncio.run(evaluate_async(args, model, dataset, folder, systems))
        errors = Counter(c["status"] for c in model.calls if c["status"] != "ok")
        if model.calls and sum(errors.values()) == len(model.calls):
            write_csv(folder / "llm_calls.csv", model.calls)
            raise SystemExit(f"ABORTED: all {len(model.calls)} safety-model calls failed ({dict(errors)}); "
                             f"the scores would measure only the R1 fallback. See {folder / 'llm_calls.csv'}.")
        if "pipeline" in args.parts:
            from .pipeline import pipeline_summary, run_pipeline
            os.environ["MINDMATE_SAFETY_TIMEOUT"] = str(args.production_timeout)
            ordinary = [c for c in dataset["cases"] if c["expected"] == "R0" and len(c["messages"]) == 1]
            unsafe = [d for d in dataset.get("outgoing", []) if not d["expected_approved"] and d.get("context", "R0") == "R0"]
            count = len(unsafe) if args.inject is None else min(args.inject, len(unsafe))
            print(f"[{dataset['name']}] pipeline: {len(dataset['cases'])} conversations x {args.endpoints}, "
                  f"{count} injected unsafe drafts per endpoint", flush=True)
            rows, injected = run_pipeline(runtime, args.model, dataset, endpoints=args.endpoints, country=args.country,
                                          request_timeout=max(900.0, args.production_timeout * 10),
                                          inject_drafts=unsafe[:count], inject_messages=ordinary)
            write_csv(folder / "pipeline.csv", rows)
            write_csv(folder / "fault_injection.csv", injected)
            summary["pipeline"] = pipeline_summary(rows, injected)
            summary["pipeline"]["latency_end_to_end"] = {
                endpoint: latency([r["latency_ms"] for r in rows if r["endpoint"] == endpoint]) for endpoint in args.endpoints}
        write_csv(folder / "llm_calls.csv", model.calls)
        summary["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(folder / "summary.json", summary)
        print_summary(dataset["name"], summary)
    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(out / "manifest.json", manifest)
    print(f"\nResults: {out}/  (build the report: python3 -m chatbot_feature.evaluation.safety.report)")


def print_summary(name: str, summary: dict[str, Any]) -> None:
    def fmt(value):
        return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    print(f"\n===== {name}: {summary['cases']} cases, {summary['user_turns']} user turns =====")
    classification = summary.get("classification", {})
    if classification:
        print(f"{'system':22} {'macroF1':>8} {'acc':>6} {'R3-4 FNR':>9} {'R0 FPR':>7} {'R4 rec':>7} {'fallback':>9}")
        for system, entry in classification.items():
            if system.startswith("_"):
                continue
            c = entry["case_level"]
            print(f"{system:22} {fmt(c.get('macro_f1')):>8} {fmt(c.get('accuracy')):>6} {fmt(c.get('high_risk_fnr')):>9} "
                  f"{fmt(c.get('r0_fpr')):>7} {fmt(c.get('r4_recall')):>7} {fmt(entry['activation'].get('fallback_rate')):>9}")
        calls = classification.get("_model_calls", {})
        print(f"model calls: {calls.get('calls')} (cached {calls.get('cached')}), parse {calls.get('parse_status')}, "
              f"latency {calls.get('latency_ms')}")
    if summary.get("outgoing"):
        o = summary["outgoing"]
        print(f"outgoing: unsafe escape hybrid {fmt(o['unsafe_escape_rate_hybrid'])} / rules {fmt(o['unsafe_escape_rate_deterministic_only'])} "
              f"/ model {fmt(o['unsafe_escape_rate_llm_only'])}; safe rejected {fmt(o['safe_rejection_rate_hybrid'])}; "
              f"override accuracy {fmt(o['safety_override_accuracy'])}")
    if summary.get("pipeline"):
        print("pipeline:", json.dumps(summary["pipeline"], indent=1)[:3000])


if __name__ == "__main__":
    sys.exit(main())
