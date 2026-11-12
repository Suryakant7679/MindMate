#!/usr/bin/env python3
"""Which panel neurons causally change the emotional tone of generated replies?

Runs on the *dev* split only (EmpatheticDialogues train); nothing here touches
the calibration or test messages.  Selectivity is never used as evidence: every
claim rests on an intervention with a matched control.

Stages
------
attribution   First-order causal estimate for every MLP neuron in every layer.
              Teacher-forced on the saved matched responses (identification
              prompts 0-149): how much would a constant decode-step shift of
              neuron u raise the log-likelihood of the emotion-styled reply
              under the *neutral* prompt?  ``effect_u = dL/d(shift_u) * dmu_u``.
              Builds candidate panels ranked by this estimate.
likelihood    Classifier-free check of the estimate with real interventions on
              identification prompts 150-199 (not used for the ranking): change
              in log-likelihood of emotion-styled replies (wanted) and of
              neutral replies (cost) for each panel and a random control.
generate      Interventions on the dev messages with the MindMate prompt:
                full        dose-response of the deployed panel
                random      same layers, counts and shifts on random neurons
                loo_L       deployed panel without layer L       (necessity of a layer)
                only_L      layer L alone                        (sufficiency of a layer)
                band        layers 8-11 / 12-15 / 16-18 alone
                top/rand/bottom  fractions of each layer's panel by attribution rank
                cand        candidate panels from the attribution stage
              and, with an emotion-style prompt (the reply is emotional by itself):
                clamp       panel neurons held at their unsteered mean (mean ablation)
                clamp_random  the same for random neurons
                negative    panel shifted by -alpha
score         gate classifier, independent classifier, Qwen ratings.
analyze       paired effects with bootstrap CIs; writes the report tables.

Run from the project root:
  HF_HUB_OFFLINE=1 python3 -m generation_analysis.causal_panel_experiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import (MODEL_ID, NEUTRAL, RESULTS_DIR, condition_style, load_model, model_layers, prompt_token_ids,
                     read_json, read_jsonl, write_json)
from .steering_lab import (LAB_DIR, QUALITY, STEERING_GAIN, SYSTEM_PROMPT, TARGETS, generate_batch, holm, load_runtime,
                           load_split, messages_of, paired_summary, panel_size, random_panel, release, score_table,
                           subset_panel)

OUT = LAB_DIR / "panel_causality"
FIT_PROMPTS = 150            # identification prompts used for the attribution ranking; the rest check it
REFERENCE_ALPHA = 1.5
BANDS = {"early_08-11": range(8, 12), "mid_12-15": range(12, 16), "late_16-18": range(16, 19)}


# ------------------------------------------------------------------ attribution

def _sequences(tokenizer, rows: list[dict], reply_condition: str) -> list[tuple[list[int], list[int]]]:
    """(neutral prompt ids, reply ids) for the replies generated under ``reply_condition``."""
    return [(prompt_token_ids(tokenizer, row["user_text"], NEUTRAL), row["generated_ids"])
            for row in rows if row["condition"] == reply_condition and len(row["generated_ids"]) > 2]


def _batch(tokenizer, pairs: list[tuple[list[int], list[int]]], device):
    length = max(len(prompt) + len(reply) for prompt, reply in pairs)
    ids = torch.full((len(pairs), length), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    shifted = torch.zeros(ids.shape, dtype=torch.bool)     # positions that read a generated token (decode steps)
    scored = torch.zeros(ids.shape, dtype=torch.bool)      # positions whose next token is y_1.. (affected tokens)
    for row, (prompt, reply) in enumerate(pairs):
        size = len(prompt) + len(reply)
        ids[row, :size] = torch.as_tensor(prompt + reply)
        attention[row, :size] = 1
        shifted[row, len(prompt):size] = True
        scored[row, len(prompt):size - 1] = True
    return ids.to(device), attention.to(device), shifted.to(device), scored.to(device)


def _reply_logprob(model, ids, attention, scored) -> torch.Tensor:
    """Mean log-probability per sequence of the reply tokens that a decode-step shift can affect."""
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1].float()
    logprob = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    mask = scored[:, :-1]
    return (logprob * mask).sum(1) / mask.sum(1).clamp_min(1)


def attribution(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = read_json(RESULTS_DIR / "capture_metadata.json")
    labels = metadata["conditions"]
    means = np.load(RESULTS_DIR / "neuron_sets" / "decode_condition_means.npy")
    rows = [row for row in read_jsonl(RESULTS_DIR / "activations" / "responses.jsonl")
            if row["prompt_id"] < FIT_PROMPTS]
    model, tokenizer = load_model(args.model_id)
    layers = model_layers(model)
    width = layers[0].mlp.down_proj.in_features
    state: dict[str, torch.Tensor] = {}
    shifts = [torch.zeros(width, dtype=torch.float32, device=layer.mlp.down_proj.weight.device, requires_grad=True)
              for layer in layers]

    def hook_for(index: int):
        def hook(_module, inputs):
            hidden = inputs[0]
            extra = state["shifted"].to(hidden.device)[..., None] * shifts[index]
            return (hidden + extra.to(hidden.dtype), *inputs[1:])
        return hook
    handles = [layer.mlp.down_proj.register_forward_pre_hook(hook_for(i)) for i, layer in enumerate(layers)]
    gradients = np.zeros((len(TARGETS), len(layers), width), dtype=np.float32)
    device = next(model.parameters()).device
    for t, target in enumerate(TARGETS):
        pairs = _sequences(tokenizer, rows, target)
        for start in range(0, len(pairs), args.grad_batch):
            ids, attention, shifted, scored = _batch(tokenizer, pairs[start:start + args.grad_batch], device)
            state["shifted"] = shifted
            for shift in shifts:
                shift.grad = None
            _reply_logprob(model, ids, attention, scored).sum().backward()
            gradients[t] += np.stack([shift.grad.float().cpu().numpy() for shift in shifts])
        gradients[t] /= len(pairs)
        print(f"attribution: {target} over {len(pairs)} responses", flush=True)
    for handle in handles:
        handle.remove()
    dmu = np.stack([means[labels.index(target)] - means[labels.index(NEUTRAL)] for target in TARGETS])
    effect = gradients * dmu * STEERING_GAIN           # predicted change of mean reply log-prob per unit alpha
    np.save(OUT / "attribution_gradient.npy", gradients)
    np.save(OUT / "attribution_effect.npy", effect.astype(np.float32))

    deployed = json.loads((RESULTS_DIR / "steering_panels" / "panels.json").read_text())
    first, last = deployed["layers"]
    per_layer = deployed["neurons_per_layer"]
    total = per_layer * (last - first + 1)
    candidates: dict[str, dict] = {}
    summary = []
    for t, target in enumerate(TARGETS):
        current = {int(layer): set(item["neurons"]) for layer, item in deployed["panels"][target].items()}
        variants = {
            # same budget and layers as the deployed panel, ranked by the causal estimate
            "grad_layer": {layer: np.argsort(-effect[t, layer])[:per_layer] for layer in range(first, last + 1)},
        }
        flat = np.argsort(-effect[t].ravel())
        for name, count in (("grad_global", total), ("grad_global_half", total // 2), ("grad_global_quarter", total // 4)):
            layer_index, neuron_index = np.unravel_index(flat[:count], effect[t].shape)
            variants[name] = {int(layer): neuron_index[layer_index == layer] for layer in np.unique(layer_index)}
        # deployed panel re-ranked by the causal estimate, keeping only neurons predicted to help
        variants["deployed_causal_half"] = {
            layer: np.asarray(sorted(neurons, key=lambda u: -effect[t, layer, u])[:len(neurons) // 2])
            for layer, neurons in current.items()}
        for name, by_layer in variants.items():
            panel = {str(layer): {"neurons": [int(u) for u in neurons],
                                  "deltas": [round(float(dmu[t, layer, u]), 6) for u in neurons]}
                     for layer, neurons in sorted(by_layer.items()) if len(neurons)}
            candidates.setdefault(name, {})[target] = panel
            chosen = [(int(layer), int(u)) for layer, item in panel.items() for u in item["neurons"]]
            summary.append({"target": target, "panel": name, "neurons": len(chosen), "layers": len(panel),
                            "first_layer": min(layer for layer, _ in chosen), "last_layer": max(layer for layer, _ in chosen),
                            "overlap_with_deployed": sum(u in current.get(layer, ()) for layer, u in chosen),
                            "predicted_effect_per_alpha": float(sum(effect[t, layer, u] for layer, u in chosen))})
        deployed_pairs = [(layer, u) for layer, neurons in current.items() for u in neurons]
        predicted = np.asarray([effect[t, layer, u] for layer, u in deployed_pairs])
        summary.append({"target": target, "panel": "deployed", "neurons": len(deployed_pairs), "layers": len(current),
                        "first_layer": first, "last_layer": last, "overlap_with_deployed": len(deployed_pairs),
                        "predicted_effect_per_alpha": float(predicted.sum()),
                        "share_predicted_harmful": float((predicted < 0).mean())})
        layer_mass = np.clip(effect[t], 0, None).sum(1)
        for layer, mass in enumerate(layer_mass):
            summary.append({"target": target, "panel": "layer_positive_effect_mass", "layers": layer,
                            "predicted_effect_per_alpha": float(mass)})
    write_json(OUT / "candidate_panels.json", {"source": "causal_panel_experiment attribution stage",
                                               "fit_prompts": f"identification prompts 0-{FIT_PROMPTS - 1}",
                                               "panels": candidates})
    pd.DataFrame(summary).to_csv(OUT / "attribution_summary.csv", index=False)
    del model
    release()


def load_candidates() -> dict[str, dict[str, dict[int, tuple[list[int], list[float]]]]]:
    payload = json.loads((OUT / "candidate_panels.json").read_text())["panels"]
    return {name: {target: {int(layer): (item["neurons"], item["deltas"]) for layer, item in by_layer.items()}
                   for target, by_layer in by_target.items()} for name, by_target in payload.items()}


# ------------------------------------------------------------------- likelihood

def likelihood(args) -> None:
    """Real interventions, teacher-forced, on identification prompts held out from the ranking."""
    rows = [row for row in read_jsonl(RESULTS_DIR / "activations" / "responses.jsonl")
            if row["prompt_id"] >= FIT_PROMPTS]
    runtime = load_runtime(args.model_id)
    model, tokenizer = runtime.model, runtime.tokenizer
    width = model.model.layers[0].mlp.down_proj.in_features
    device = next(model.parameters()).device
    candidates = load_candidates()
    rng = np.random.default_rng(args.seed)
    state: dict[str, torch.Tensor] = {}

    def run(pairs, panel, alpha) -> np.ndarray:
        handles = []
        if panel and alpha:
            for layer, (neurons, deltas) in panel.items():
                index = torch.tensor(neurons, device=device)
                delta = torch.tensor(deltas, device=device) * alpha * STEERING_GAIN

                def hook(_module, inputs, index=index, delta=delta):
                    hidden = inputs[0].clone()
                    hidden[..., index] += state["shifted"][..., None] * delta.to(hidden.dtype)
                    return (hidden, *inputs[1:])
                handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        values = []
        with torch.inference_mode():
            for start in range(0, len(pairs), 16):
                ids, attention, shifted, scored = _batch(tokenizer, pairs[start:start + 16], device)
                state["shifted"] = shifted
                values.append(_reply_logprob(model, ids, attention, scored).cpu().numpy())
        for handle in handles:
            handle.remove()
        return np.concatenate(values)

    records = []
    for target in TARGETS:
        deployed = runtime._emotion_panels[target]
        panels = {"deployed": deployed, **{name: by_target[target] for name, by_target in candidates.items()},
                  "random": random_panel(deployed, width, rng)}
        emotional, neutral = _sequences(tokenizer, rows, target), _sequences(tokenizer, rows, NEUTRAL)
        base = {"emotional": run(emotional, None, 0.0), "neutral": run(neutral, None, 0.0)}
        for name, panel in panels.items():
            for alpha in args.likelihood_alphas:
                gain = run(emotional, panel, alpha) - base["emotional"]
                cost = run(neutral, panel, alpha) - base["neutral"]
                records.append({"target": target, "panel": name, "alpha": alpha, "neurons": panel_size(panel),
                                **{f"emotional_{k}": v for k, v in paired_summary(gain).items()},
                                **{f"neutral_{k}": v for k, v in paired_summary(cost).items()}})
        print(f"likelihood: {target}", flush=True)
    pd.DataFrame(records).to_csv(OUT / "likelihood_check.csv", index=False)
    del runtime
    release()


# --------------------------------------------------------------------- generate

def reference_means(runtime, conversations, union: dict[int, list[int]]) -> dict[int, dict[int, float]]:
    """Mean decode-step activation of the given neurons in unsteered MindMate replies (unbatched, exact)."""
    runtime._emotion_panels["__reference__"] = {layer: (neurons, [0.0] * len(neurons)) for layer, neurons in union.items()}
    sums = {layer: np.zeros(len(neurons)) for layer, neurons in union.items()}
    positions = dict.fromkeys(union, 0)
    for messages in conversations:
        trace: dict = {}
        runtime.generate(messages, max_new_tokens=120, deterministic=True, temperature=0.7,
                         system_prompt=SYSTEM_PROMPT, steering_alpha=0.0, steering_emotion="__reference__",
                         steering_trace=trace)
        for item in trace["layers"]:
            sums[item["layer"]] += np.asarray(item["mean_activation_before_shift"]) * item["decode_positions"]
            positions[item["layer"]] += item["decode_positions"]
    del runtime._emotion_panels["__reference__"]
    return {layer: dict(zip(union[layer], (sums[layer] / max(positions[layer], 1)).tolist())) for layer in union}


def generate(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "generations.jsonl"
    done = {(row["item_id"], row["target"], row["condition"]) for row in read_jsonl(path)} if path.exists() else set()
    dev = load_split("dev")
    conversations = [messages_of(row) for row in dev.itertuples()]
    runtime = load_runtime(args.model_id)
    width = runtime.model.model.layers[0].mlp.down_proj.in_features
    candidates = load_candidates() if (OUT / "candidate_panels.json").exists() else {}
    rng = np.random.default_rng(args.seed)

    plans: list[dict] = []          # one entry per condition: how to generate it
    randoms: dict[str, list[dict]] = {}
    for target in TARGETS:
        panel = runtime._emotion_panels[target]
        randoms[target] = [random_panel(panel, width, rng) for _ in range(2)]

        def add(condition, family, variant, alpha, *, system=SYSTEM_PROMPT, clamp=None, target=target):
            plans.append({"target": target, "condition": condition, "family": family, "alpha": alpha,
                          "variant": variant, "neurons": panel_size(variant) if variant else 0,
                          "system": system, "clamp": clamp})
        add("unsteered", "unsteered", None, 0.0)
        for alpha in (0.5, 1.0, 1.5, 2.0):
            add(f"full@{alpha}", "full", panel, alpha)
        for seed, control in enumerate(randoms[target]):
            add(f"random{seed}@{REFERENCE_ALPHA}", "random", control, REFERENCE_ALPHA)
        add("random0@2.0", "random", randoms[target][0], 2.0)
        for layer in sorted(panel):
            add(f"loo_L{layer}@{REFERENCE_ALPHA}", "loo", subset_panel(panel, layers=set(panel) - {layer}), REFERENCE_ALPHA)
            add(f"only_L{layer}@4.0", "only", subset_panel(panel, layers=[layer]), 4.0)
        for name, band in BANDS.items():
            for alpha in (REFERENCE_ALPHA, 3.0):
                add(f"band_{name}@{alpha}", "band", subset_panel(panel, layers=band), alpha)
        for fraction in (0.1, 0.25, 0.5):
            add(f"top{int(fraction * 100)}@{REFERENCE_ALPHA}", "subset", subset_panel(panel, fraction=fraction), REFERENCE_ALPHA)
            add(f"rand{int(fraction * 100)}@{REFERENCE_ALPHA}", "subset",
                subset_panel(panel, fraction=fraction, order="random", rng=rng), REFERENCE_ALPHA)
        add(f"bottom50@{REFERENCE_ALPHA}", "subset", subset_panel(panel, fraction=0.5, order="bottom"), REFERENCE_ALPHA)
        for fraction in (0.25, 0.5):
            add(f"top{int(fraction * 100)}@3.0", "subset", subset_panel(panel, fraction=fraction), 3.0)
        for name, by_target in candidates.items():
            for alpha in args.candidate_alphas:
                add(f"cand_{name}@{alpha}", "candidate", by_target[target], alpha)

    # Ablations need each neuron's unsteered mean in this prompt context.
    union: dict[int, list[int]] = {}
    for target in TARGETS:
        for source in (runtime._emotion_panels[target], *randoms[target]):
            for layer, (neurons, _) in source.items():
                union[layer] = sorted(set(union.get(layer, [])) | set(neurons))
    reference_path = OUT / "reference_means.json"
    if reference_path.exists():
        reference = {int(layer): {int(u): value for u, value in by_neuron.items()}
                     for layer, by_neuron in json.loads(reference_path.read_text()).items()}
    else:
        reference = reference_means(runtime, conversations, union)
        write_json(reference_path, reference)
    for target in TARGETS:
        panel = runtime._emotion_panels[target]
        styled = f"{SYSTEM_PROMPT} {condition_style(target)}"

        def clamp_values(source):
            return {layer: (neurons, [reference[layer][u] for u in neurons]) for layer, (neurons, _) in source.items()}
        common = {"target": target, "system": styled}
        plans.append({**common, "condition": "styled", "family": "styled", "alpha": 0.0, "variant": None,
                      "neurons": 0, "clamp": None})
        plans.append({**common, "condition": "styled_clamp", "family": "ablation", "alpha": 0.0, "variant": None,
                      "neurons": panel_size(panel), "clamp": clamp_values(panel)})
        for seed, control in enumerate(randoms[target]):
            plans.append({**common, "condition": f"styled_clamp_random{seed}", "family": "ablation", "alpha": 0.0,
                          "variant": None, "neurons": panel_size(control), "clamp": clamp_values(control)})
        plans.append({**common, "condition": "styled_negative@-1.0", "family": "ablation", "alpha": -1.0,
                      "variant": panel, "neurons": panel_size(panel), "clamp": None})
        plans.append({**common, "condition": "styled_negative_random@-1.0", "family": "ablation", "alpha": -1.0,
                      "variant": randoms[target][0], "neurons": panel_size(panel), "clamp": None})

    with path.open("a", encoding="utf-8") as handle:
        for number, plan in enumerate(plans):
            if all((item_id, plan["target"], plan["condition"]) in done for item_id in dev.item_id):
                continue
            if plan["condition"] == "unsteered" and plan["target"] != TARGETS[0]:
                source = replies_unsteered          # identical prompt: reuse instead of regenerating
            else:
                name = None
                if plan["variant"] is not None:
                    name = "__variant__"
                    runtime._emotion_panels[name] = plan["variant"]
                source = generate_batch(runtime, conversations, [plan["system"]] * len(conversations),
                                        alpha=plan["alpha"], panel=name, clamp=plan["clamp"],
                                        max_new_tokens=args.max_new_tokens)
                if plan["condition"] == "unsteered":
                    replies_unsteered = source
            for item_id, reply in zip(dev.item_id, source):
                handle.write(json.dumps({"item_id": item_id, "target": plan["target"], "condition": plan["condition"],
                                         "family": plan["family"], "alpha": plan["alpha"], "neurons": plan["neurons"],
                                         "reply": reply}, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 10 == 0:
                print(f"generate: {number + 1}/{len(plans)} conditions", flush=True)
    del runtime
    release()


# ------------------------------------------------------------------------ score

def score(args) -> None:
    table = pd.DataFrame(read_jsonl(OUT / "generations.jsonl")).drop_duplicates(["item_id", "target", "condition"])
    dev = load_split("dev").set_index("item_id")
    table["text"] = dev.loc[table.item_id, "text"].to_numpy()
    table["history"] = [dev.loc[item_id, "history"] for item_id in table.item_id]
    score_table(table).drop(columns=["history"]).to_csv(OUT / "scores.csv", index=False)


# ---------------------------------------------------------------------- analyze

METRICS = ("judge_emotion", "indep_target", "gate_target", *QUALITY)


def analyze(args) -> None:
    table = pd.read_csv(OUT / "scores.csv")
    rows = []
    for target, block in table.groupby("target"):
        wide = {condition: group.set_index("item_id") for condition, group in block.groupby("condition")}
        for condition, group in wide.items():
            if condition in ("unsteered", "styled"):
                continue
            reference = wide["styled" if condition.startswith("styled") else "unsteered"].loc[group.index]
            entry = {"target": target, "condition": condition, "family": group.family.iloc[0],
                     "alpha": group.alpha.iloc[0], "neurons": int(group.neurons.iloc[0]),
                     "changed_replies": float((group.reply != reference.reply).mean())}
            for metric in METRICS:
                stats = paired_summary(group[metric] - reference[metric])
                entry.update({f"{metric}": stats["mean"], f"{metric}_lo": stats["ci_low"],
                              f"{metric}_hi": stats["ci_high"], f"{metric}_p": stats["p"]})
            rows.append(entry)
    effects = pd.DataFrame(rows)
    effects["judge_emotion_p_holm"] = holm(effects.judge_emotion_p)
    effects.to_csv(OUT / "effects.csv", index=False)

    # Layer necessity: how much of the full panel's effect is lost without layer L (paired, same messages).
    layer_rows = []
    for target, block in table.groupby("target"):
        wide = {condition: group.set_index("item_id") for condition, group in block.groupby("condition")}
        full = wide[f"full@{REFERENCE_ALPHA}"]
        for condition, group in wide.items():
            if not condition.startswith("loo_L"):
                continue
            layer = int(condition.split("_L")[1].split("@")[0])
            entry = {"target": target, "layer": layer}
            for metric in ("judge_emotion", "indep_target", "fluency"):
                stats = paired_summary(full.loc[group.index, metric] - group[metric])
                entry.update({f"loss_without_{metric}": stats["mean"], f"loss_without_{metric}_lo": stats["ci_low"],
                              f"loss_without_{metric}_hi": stats["ci_high"]})
            only = effects[(effects.target == target) & (effects.condition == f"only_L{layer}@4.0")]
            if len(only):
                for metric in ("judge_emotion", "indep_target", "fluency"):
                    entry.update({f"alone_{metric}": only[metric].iloc[0], f"alone_{metric}_lo": only[f"{metric}_lo"].iloc[0],
                                  f"alone_{metric}_hi": only[f"{metric}_hi"].iloc[0]})
            layer_rows.append(entry)
    pd.DataFrame(layer_rows).sort_values(["target", "layer"]).to_csv(OUT / "layer_effects.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    columns = ["target", "condition", "neurons", "judge_emotion", "judge_emotion_lo", "judge_emotion_hi",
               "indep_target", "gate_target", "empathy", "relevance", "fluency"]
    print(effects[columns].round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--stages", nargs="+", default=["attribution", "likelihood", "generate", "score", "analyze"],
                        choices=("attribution", "likelihood", "generate", "score", "analyze"))
    parser.add_argument("--grad-batch", type=int, default=8)
    parser.add_argument("--likelihood-alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--candidate-alphas", type=float, nargs="+", default=[0.75, 1.5, 2.5])
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for stage in args.stages:
        {"attribution": attribution, "likelihood": likelihood, "generate": generate, "score": score,
         "analyze": analyze}[stage](args)
    write_json(OUT / "metadata.json", {
        "split": "dev (EmpatheticDialogues train); disjoint from calibration and test",
        "reference_alpha": REFERENCE_ALPHA, "fit_prompts": FIT_PROMPTS, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "targets": list(TARGETS), "decoding": "greedy, batched, left-padded",
        "effect_reference": "unsteered reply to the same message (styled reply for ablation conditions)",
        "uncertainty": "percentile bootstrap over messages, 10,000 resamples; two-sided Wilcoxon signed-rank",
    })


if __name__ == "__main__":
    main()
