#!/usr/bin/env python3
"""Collect the offline causal and held-out results into one small file for the Mood Tracker.

Reads the outputs of ``causal_panel_experiment`` (dev split) and
``heldout_comparison`` (test split) and writes
``results/llama31_8b_generation_emotion/causal_steering/evidence.json``.  The
backend attaches the entry of the planned response emotion to each turn's
analysis; nothing here is computed at chat time.

Run from the project root:  python3 -m generation_analysis.export_steering_evidence
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from backend.config import STEERING_PANEL_FILE
from .common import RESULTS_DIR, write_json
from .steering_lab import LAB_DIR, TARGETS

CAUSAL = LAB_DIR / "panel_causality"
HELDOUT = LAB_DIR / "heldout"
SYSTEM_LABELS = {"baseline": "MindMate prompt only", "kg_prompt": "Strategy prompt, no steering",
                 "v1": "Previous system", "v2": "Current system"}
CONTRAST_LABELS = {"v2 - v1": "Current vs previous", "v2 - kg_prompt": "Current vs no steering",
                   "v1 - kg_prompt": "Previous vs no steering"}
METRIC_LABELS = {"judge_emotion": "Emotion rating (1-5)", "indep_target": "Independent classifier",
                 "empathy": "Empathy (1-5)", "relevance": "Relevance (1-5)", "fluency": "Fluency (1-5)",
                 "preference_minus_0.5": "Judge preference - 0.5", "latency_seconds": "Latency (s)"}


def number(value, digits: int = 3):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value), digits)


def effect(row, label: str) -> dict:
    return {"label": label, "alpha": number(row.alpha, 2), "neurons": int(row.neurons),
            "emotion_rating": number(row.judge_emotion), "lo": number(row.judge_emotion_lo),
            "hi": number(row.judge_emotion_hi), "fluency": number(row.fluency)}


def panel_evidence(target: str, effects: pd.DataFrame, layers: pd.DataFrame, likelihood: pd.DataFrame | None,
                   active: dict, variant: str) -> dict:
    """``variant`` is the condition prefix of the active panel in the causal experiment."""
    block = effects[effects.target == target].set_index("condition")
    deployed = active["panels"][target]
    neuron_count = sum(len(item["neurons"]) for item in deployed.values())
    prefix = "full" if variant == "deployed" else f"cand_{variant}"
    dose = [effect(row, f"Panel, α {row.alpha:g}") for name, row in block.iterrows() if name.startswith(prefix + "@")]
    dose.sort(key=lambda item: item["alpha"])
    controls = [effect(row, f"Random neurons, α {row.alpha:g}" + (f" (draw {name[6]})" if name[6] != "0" else ""))
                for name, row in block.iterrows() if name.startswith("random")]
    subsets = []
    names = {"top": "Top {}% of each layer by attribution", "rand": "Random {}% of each layer",
             "bottom": "Bottom {}% of each layer by attribution"}
    for name, row in block.iterrows():
        if row.family == "subset":
            kind = next(key for key in names if name.startswith(key))
            share = name[len(kind):].split("@")[0]
            subsets.append(effect(row, f"{names[kind].format(share)}, α {row.alpha:g}"))
        elif row.family == "band":
            subsets.append(effect(row, f"Layers {name.split('_')[2].split('@')[0]} only, α {row.alpha:g}"))
        elif row.family == "candidate":
            subsets.append(effect(row, f"{name[5:].split('@')[0].replace('_', ' ')} panel, α {row.alpha:g}"))
    ablation_labels = {"styled_clamp": "Panel neurons held at their unsteered mean",
                       "styled_clamp_random0": "Random neurons held at their mean (control)",
                       "styled_clamp_random1": "Random neurons held at their mean (control, draw 2)",
                       "styled_negative@-1.0": "Panel neurons shifted the opposite way (α -1)",
                       "styled_negative_random@-1.0": "Random neurons shifted the opposite way (control)"}
    ablation = [effect(block.loc[name], text) for name, text in ablation_labels.items() if name in block.index]
    layer_rows = [{"layer": int(row.layer), "loss_without": number(row.loss_without_judge_emotion),
                   "lo": number(row.loss_without_judge_emotion_lo), "hi": number(row.loss_without_judge_emotion_hi),
                   "alone": number(row.get("alone_judge_emotion")), "alone_lo": number(row.get("alone_judge_emotion_lo")),
                   "alone_hi": number(row.get("alone_judge_emotion_hi"))}
                  for row in layers[layers.target == target].itertuples(index=False)
                  for row in [pd.Series(row._asdict())]]
    check = None
    if likelihood is not None:
        rows = likelihood[(likelihood.target == target) & (likelihood.alpha == 1.0)].set_index("panel")
        if variant in rows.index and "random" in rows.index:
            check = {"alpha": 1.0, "panel": number(rows.loc[variant, "emotional_mean"]),
                     "panel_lo": number(rows.loc[variant, "emotional_ci_low"]),
                     "panel_hi": number(rows.loc[variant, "emotional_ci_high"]),
                     "random": number(rows.loc["random", "emotional_mean"]),
                     "neutral_cost": number(rows.loc[variant, "neutral_mean"])}
    # Quote the strongest dose that kept fluency within 0.1 of the unsteered reply, not the strongest dose overall.
    tolerable = [item for item in dose if item["fluency"] is None or item["fluency"] >= -0.1]
    strongest = max(tolerable, key=lambda item: item["emotion_rating"]) if tolerable else None
    control = max((item for item in controls if strongest and item["alpha"] >= strongest["alpha"]),
                  key=lambda item: item["emotion_rating"], default=None) or (max(controls, key=lambda item: item["emotion_rating"]) if controls else None)
    verdict = ("No tested dose raised the emotion without lowering fluency by more than 0.1, or no results exist; "
               "this panel is not steered by the current policy.")
    if strongest and control:
        causal = strongest["lo"] > 0 and strongest["lo"] > control["hi"]
        verdict = (f"Shifting these neurons {'causally raises' if causal else 'did not reliably raise'} the "
                   f"{target} expressed in replies: emotion rating {strongest['emotion_rating']:+.2f} "
                   f"(95% CI {strongest['lo']:+.2f} to {strongest['hi']:+.2f}) at α {strongest['alpha']:g} "
                   f"(fluency change {strongest['fluency']:+.2f}), against {control['emotion_rating']:+.2f} "
                   f"({control['lo']:+.2f} to {control['hi']:+.2f}) for random neurons at α {control['alpha']:g}. "
                   "Stronger doses raise the rating further but lower fluency. The layer ablations below were run on "
                   "the original attribution panel (layers 8-18); the evidence is that the effect is spread over "
                   "many layers, with no single layer necessary.")
    return {"emotion": target, "name": active.get("name", "Attribution panel"),
            "selection": active.get("selection", ""), "neurons": neuron_count,
            "layers": sorted(int(layer) for layer in deployed), "verdict": verdict, "likelihood": check,
            "dose_response": dose, "controls": controls, "ablation": ablation, "layer_effects": layer_rows,
            "subsets": subsets}


def held_out_evidence() -> dict | None:
    if not (HELDOUT / "system_means.csv").exists():
        return None
    means = pd.read_csv(HELDOUT / "system_means.csv")
    contrasts = pd.read_csv(HELDOUT / "contrasts.csv")
    overall = means[means.subset == "all"].set_index("system")
    systems = [{"system": name, "label": SYSTEM_LABELS[name], "emotion_rating": number(row.judge_emotion),
                "independent_classifier": number(row.indep_target), "empathy": number(row.empathy),
                "relevance": number(row.relevance), "fluency": number(row.fluency),
                "steered_share": number(row.steered_share), "latency_median": number(row.latency_median)}
               for name, row in overall.iterrows()]
    chosen = contrasts[(contrasts.subset == "all") & contrasts.contrast.isin(CONTRAST_LABELS)
                       & contrasts.metric.isin(METRIC_LABELS)]
    rows = [{"contrast": row.contrast, "label": CONTRAST_LABELS[row.contrast], "metric": METRIC_LABELS[row.metric],
             "mean": number(row["mean"], 4), "lo": number(row.ci_low, 4), "hi": number(row.ci_high, 4),
             "p": number(row.p, 4)} for _, row in chosen.iterrows()]
    def pick(contrast, metric):
        match = chosen[(chosen.contrast == contrast) & (chosen.metric == metric)]
        return match.iloc[0] if len(match) else None
    emotion, fluency = pick("v2 - kg_prompt", "judge_emotion"), pick("v2 - kg_prompt", "fluency")
    summary = "Held-out comparison of the current system with the previous one and with no steering."
    if emotion is not None and fluency is not None:
        summary = (f"Against the same replies without steering, the current system changed the emotion rating by "
                   f"{emotion['mean']:+.3f} (95% CI {emotion.ci_low:+.3f} to {emotion.ci_high:+.3f}) and fluency by "
                   f"{fluency['mean']:+.3f} ({fluency.ci_low:+.3f} to {fluency.ci_high:+.3f}), averaged over all "
                   "messages including those it chose not to steer.")
    return {"messages": int(overall.n.iloc[0]), "summary": summary, "systems": systems, "contrasts": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="deployed",
                        help="Condition family of the active panel in the causal experiment "
                             "('deployed' or a candidate name such as 'deployed_causal_half').")
    args = parser.parse_args()
    active = json.loads((RESULTS_DIR / "steering_panels" / STEERING_PANEL_FILE).read_text())
    effects = pd.read_csv(CAUSAL / "effects.csv")
    layers = pd.read_csv(CAUSAL / "layer_effects.csv")
    likelihood = pd.read_csv(CAUSAL / "likelihood_check.csv") if (CAUSAL / "likelihood_check.csv").exists() else None
    evidence = {
        "panels": {target: panel_evidence(target, effects, layers, likelihood, active, args.variant)
                   for target in TARGETS if target in active["panels"]},
        "held_out": held_out_evidence(),
        "sources": {"causal experiments": str(CAUSAL), "held-out comparison": str(HELDOUT),
                    "active panel": str(RESULTS_DIR / "steering_panels" / STEERING_PANEL_FILE)},
    }
    write_json(LAB_DIR / "evidence.json", evidence)
    print(f"Wrote {LAB_DIR / 'evidence.json'}")


if __name__ == "__main__":
    main()
