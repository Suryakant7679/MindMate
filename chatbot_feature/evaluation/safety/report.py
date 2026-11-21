"""Combine live safety-evaluation runs into comparison tables and a Markdown report.

Reads results/safety_eval/<run>/<dataset>/summary.json (every run found, or --runs) and writes
results/safety_eval/report/{comparison.csv, per_language.csv, per_class.csv, outgoing.csv, pipeline.csv, REPORT.md}.

  python3 -m chatbot_feature.evaluation.safety.report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .live_eval import write_csv
from .metrics import latency

ROOT = Path("results") / "safety_eval"
METRICS = ("macro_f1", "accuracy", "high_risk_fnr", "high_risk_recall", "r4_recall", "r0_fpr",
           "at_risk_sensitivity", "at_risk_specificity", "under_triage_rate", "over_triage_rate", "route_accuracy")


def fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def interval(entry: dict, name: str) -> str:
    low, high = (entry.get("ci95") or {}).get(name, (None, None))
    return f"{fmt(entry.get(name))} [{fmt(low)}, {fmt(high)}]" if low is not None else fmt(entry.get(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--runs", nargs="+", help="Run folders under --root (default: all)")
    args = parser.parse_args()
    out = args.root / "report"
    runs = args.runs or sorted(p.name for p in args.root.iterdir() if p.is_dir() and p.name != "report") if args.root.exists() else []
    comparison, languages, classes, outgoing, pipeline = [], [], [], [], []
    lines = ["# Agent 1 (Safety & Risk) live evaluation", "",
             "Synthetic, non-expert-annotated scenarios; engineering triage levels, not clinical assessment. "
             "`dev` is the set the rule screen was written on (optimistic for B/C); `heldout` was written afterwards. "
             "Intervals: 95% case bootstrap.", ""]
    for run in runs:
        manifest_path = args.root / run / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for summary_path in sorted((args.root / run).glob("*/summary.json")):
            summary = json.loads(summary_path.read_text())
            dataset = summary["dataset"]["name"]
            lines += [f"## {run} — {dataset} ({summary['cases']} cases, {summary['user_turns']} user turns)", "",
                      f"Model: `{manifest.get('model_id')}`; GPU: {manifest.get('environment', {}).get('gpu')}; "
                      f"commit `{(manifest.get('git_commit') or '')[:10]}`; finished {summary.get('finished')}.", ""]
            classification = summary.get("classification") or {}
            # Model-call latency per system from llm_calls.csv: a cached call keeps the latency measured when it
            # was generated, whereas the agent's wall-clock time for a cached call is near zero.
            calls_path = summary_path.parent / "llm_calls.csv"
            call_rows = list(csv.DictReader(calls_path.open())) if calls_path.exists() else []
            model_latency = {system: latency([float(r["ms"]) for r in call_rows
                                              if r["part"] == "classify" and r["system"] == system and r["status"] == "ok"])
                             for system in classification}
            if classification:
                lines += ["| system | macro-F1 | R3–R4 FNR | R0 FPR | R4 recall | accuracy | fallback | p50 model-call ms |",
                          "|---|---|---|---|---|---|---|---|"]
                for system, entry in classification.items():
                    if system.startswith("_"):
                        continue
                    case = entry["case_level"]
                    comparison.append({"run": run, "dataset": dataset, "system": system, "n": case.get("n"),
                                       **{m: case.get(m) for m in METRICS},
                                       **{f"{m}_ci_low": (case.get("ci95") or {}).get(m, (None, None))[0] for m in ("macro_f1", "high_risk_fnr", "r0_fpr")},
                                       **{f"{m}_ci_high": (case.get("ci95") or {}).get(m, (None, None))[1] for m in ("macro_f1", "high_risk_fnr", "r0_fpr")},
                                       "turn_macro_f1": entry["turn_level"].get("macro_f1"),
                                       **{f"act_{k}": v for k, v in (entry.get("activation") or {}).items()},
                                       "model_call_p50_ms": model_latency[system].get("p50_ms"),
                                       "model_call_p95_ms": model_latency[system].get("p95_ms"),
                                       "model_calls": model_latency[system].get("n")})
                    for language, values in (case.get("by_language") or {}).items():
                        languages.append({"run": run, "dataset": dataset, "system": system, "language": language,
                                          **{k: values.get(k) for k in ("n", *METRICS)}})
                    for label, values in (case.get("per_class") or {}).items():
                        classes.append({"run": run, "dataset": dataset, "system": system, "label": label, **values})
                    lines.append(f"| {system} | {interval(case, 'macro_f1')} | {interval(case, 'high_risk_fnr')} | "
                                 f"{interval(case, 'r0_fpr')} | {fmt(case.get('r4_recall'))} | {fmt(case.get('accuracy'))} | "
                                 f"{fmt((entry.get('activation') or {}).get('fallback_rate'))} | "
                                 f"{fmt(model_latency[system].get('p50_ms'))} |")
                calls = classification.get("_model_calls") or {}
                lines += ["", f"Safety-model calls: {calls.get('calls')} ({calls.get('cached')} from cache); output parse: "
                              f"{calls.get('parse_status')}; latency {calls.get('latency_ms')}; share slower than the "
                              f"{calls.get('production_timeout_s')} s production timeout: {fmt(calls.get('share_slower_than_production_timeout'))}.", ""]
                full = classification.get("C_full", {}).get("case_level", {})
                if full.get("by_language"):
                    lines += ["C_full by language:", "", "| language | n | macro-F1 | R3–R4 FNR | R0 FPR |", "|---|---|---|---|---|"]
                    for language, values in full["by_language"].items():
                        lines.append(f"| {language} | {values['n']} | {fmt(values.get('macro_f1'))} | "
                                     f"{fmt(values.get('high_risk_fnr'))} | {fmt(values.get('r0_fpr'))} |")
                    lines.append("")
            if summary.get("outgoing"):
                o = summary["outgoing"]
                outgoing.append({"run": run, "dataset": dataset, **{k: v for k, v in o.items() if k != "by_language"}})
                lines += [f"Outgoing validator ({o['unsafe']} unsafe / {o['safe']} safe drafts): unsafe escape — hybrid "
                          f"{fmt(o['unsafe_escape_rate_hybrid'])}, rules only {fmt(o['unsafe_escape_rate_deterministic_only'])}, "
                          f"model only {fmt(o['unsafe_escape_rate_llm_only'])}"
                          + (f" (draft alone {fmt(o['unsafe_escape_rate_llm_alone_only'])}, in context "
                             f"{fmt(o['unsafe_escape_rate_llm_context_only'])})" if "unsafe_escape_rate_llm_alone_only" in o else "")
                          + f"; safe drafts rejected — hybrid "
                          f"{fmt(o['safe_rejection_rate_hybrid'])}; safety-override accuracy {fmt(o['safety_override_accuracy'])}.", ""]
            if summary.get("pipeline"):
                p = summary["pipeline"]
                for endpoint in ("chat", "stream"):
                    if endpoint in p:
                        pipeline.append({"run": run, "dataset": dataset, "endpoint": endpoint, **p[endpoint],
                                         **{f"latency_{k}": v for k, v in (p.get("latency_end_to_end", {}).get(endpoint) or {}).items()}})
                        e = p[endpoint]
                        lines.append(f"Pipeline `/api/{'chat/stream' if endpoint == 'stream' else 'chat'}`: {e['requests']} requests, "
                                     f"gate checks passed {fmt(e['gate_ok_rate'])}, route vs expected {fmt(e['route_accuracy_vs_expected'])}, "
                                     f"off-normal turns with Emotion Reasoner / steering / chat generation: "
                                     f"{e['non_normal_with_emotion_reasoner']} / {e['non_normal_with_steering']} / "
                                     f"{e['non_normal_with_chat_generation']}, rejected-draft leaks {e['rejected_draft_leaks']}"
                                     + (f", first byte after approval {fmt(e.get('first_byte_after_approval_rate'))}" if endpoint == "stream" else "") + ".")
                if p.get("fault_injection"):
                    f = p["fault_injection"]
                    lines.append(f"Fault injection: {f['exercised']}/{f['requests']} exercised, unsafe drafts sent {f['unsafe_draft_sent']}, "
                                 f"replaced {fmt(f['replaced_rate'])}, leaks {f['leaks']}"
                                 + (f"; drafts exercised {f['drafts_exercised']}/{f['drafts']}, sent: "
                                    f"{', '.join(f['sent_draft_ids']) or 'none'}" if "drafts" in f else "") + ".")
                lines.append("")
    write_csv(out / "comparison.csv", comparison)
    write_csv(out / "per_language.csv", languages)
    write_csv(out / "per_class.csv", classes)
    write_csv(out / "outgoing.csv", outgoing)
    write_csv(out / "pipeline.csv", pipeline)
    if not runs:
        lines.append("No runs found. Run live_eval first.")
    (out / "REPORT.md").parent.mkdir(parents=True, exist_ok=True)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten: {out}/REPORT.md and CSV tables")


if __name__ == "__main__":
    main()
