#!/usr/bin/env python3
"""Stage 4: recognition- vs generation-derived neurons for steering and semantic preservation.

Held-out user messages (never used for neuron identification) are answered
under the *neutral* prompt while the neurons of one set receive an additive
shift on every decode step (prompt prefill is unchanged, as in the backend):

* ``mean_diff`` (default): delta[l,u] = alpha * (mu_e[l,u] - mu_neutral[l,u]),
  where mu are decode-step mean activations from the identification split.
  Every set uses the same activation statistics; only the neuron choice differs.
* ``selectivity``: alpha * 0.008311835 * selectivity / max(selectivity), the
  backend's panel steering rule.

With ``--delta-norm matched`` (default) each set's delta vector is rescaled to
the L2 norm of the generation set's vector for that emotion and alpha, so sets
are compared at an equal intervention budget.  ``random`` is a layer-matched
random-neuron control.

Steering success is scored with the frozen recognition probe on the response
text; semantic preservation by sentence-embedding similarity and content-word
F1 to the unsteered response, user-content recall, and fluency by the response
NLL under the unsteered model.

Run from the project root:  python3 -m generation_analysis.steering
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

from .common import (MODEL_ID, NEUTRAL, PROBE_PATH, RESULTS_DIR, SELECTIVITY_STEERING_SCALE, MLPController,
                     continuation_logprobs, greedy_decode, input_device, load_model, load_probe, model_layers,
                     probe_probabilities, prompt_token_ids, read_json, read_jsonl, special_token_ids,
                     steering_tensors, stop_token_ids, write_json)
from .lexicon import EMOTION_LEXICON, content_words, normalize_word

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
SET_CHOICES = ("recognition", "generation", "emotional_token", "random")


def set_deltas(neurons: dict[str, list[int]], scores: dict[tuple[int, int], float], emotion_mean: np.ndarray,
               neutral_mean: np.ndarray, alpha: float, mode: str) -> dict[int, tuple[list[int], list[float]]]:
    """Per-layer (neuron ids, additive shifts) for one set, emotion and alpha."""
    output = {}
    if mode == "selectivity":
        maximum = max(scores.values()) if scores else 1.0
    for layer, ids in neurons.items():
        layer, ids = int(layer), [int(neuron) for neuron in ids]
        if mode == "mean_diff":
            deltas = alpha * (emotion_mean[layer, ids] - neutral_mean[layer, ids])
        else:
            deltas = np.asarray([alpha * SELECTIVITY_STEERING_SCALE * scores[(layer, n)] / maximum for n in ids])
        output[layer] = (ids, [float(value) for value in deltas])
    return output


def delta_norm(deltas: dict[int, tuple[list[int], list[float]]]) -> float:
    return float(np.sqrt(sum(np.square(values).sum() for _, values in deltas.values())))


def rescale(deltas: dict[int, tuple[list[int], list[float]]], factor: float) -> dict[int, tuple[list[int], list[float]]]:
    return {layer: (ids, [value * factor for value in values]) for layer, (ids, values) in deltas.items()}


def content_f1(text: str, reference: str) -> float:
    a, b = content_words(text), content_words(reference)
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def lexicon_rate(text: str, emotion: str) -> float:
    words = [normalize_word(token) for token in text.split()]
    words = [word for word in words if word]
    return sum(word in EMOTION_LEXICON.get(emotion, ()) for word in words) / max(len(words), 1)


def repeated_bigram_rate(text: str) -> float:
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 2000) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (samples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


METRICS = ("target_probability_change", "target_is_argmax", "target_lexicon_rate", "embedding_similarity",
           "content_f1_to_baseline", "user_content_recall", "nll_change", "repeated_bigram_rate")


def summarize(table: pd.DataFrame, keys: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for key, group in table.groupby(keys, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["responses"] = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            low, high = bootstrap_ci(values, seed)
            row[metric] = float(np.nanmean(values)) if len(values) else np.nan
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = low, high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(table: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for alpha, group in table.groupby("alpha"):
        wide = {name: frame.set_index(["emotion", "prompt_id"]) for name, frame in group.groupby("set_type")}
        for first, second in pairs:
            if first not in wide or second not in wide:
                continue
            joined = wide[first].join(wide[second], lsuffix="_a", rsuffix="_b", how="inner")
            for metric in ("target_probability_change", "embedding_similarity", "content_f1_to_baseline", "nll_change"):
                difference = (joined[f"{metric}_a"] - joined[f"{metric}_b"]).dropna().to_numpy()
                p = float(wilcoxon(difference).pvalue) if len(difference) > 1 and np.any(difference) else 1.0
                rows.append({"alpha": alpha, "set_a": first, "set_b": second, "metric": metric,
                             "pairs": len(difference), "mean_difference_a_minus_b": float(difference.mean())
                             if len(difference) else np.nan, "wilcoxon_p": p})
    return pd.DataFrame(rows)


def plot_steering(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"recognition": "#2a6fdb", "generation": "#d1495b", "emotional_token": "#00a6a6", "random": "#8a8a8a"}
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for set_type, group in summary.groupby("set_type", sort=False):
        group = group.sort_values("alpha")
        color = colors.get(set_type, "#444444")
        for axis, metric, label in ((axes[0], "target_probability_change", "Change in target-emotion probability"),
                                    (axes[1], "embedding_similarity", "Embedding similarity to unsteered")):
            axis.errorbar(group.alpha, group[metric], yerr=[group[metric] - group[f"{metric}_ci_low"],
                                                            group[f"{metric}_ci_high"] - group[metric]],
                          color=color, marker="o", linewidth=2, capsize=3, label=set_type)
            axis.set_xlabel("alpha"); axis.set_ylabel(label)
        axes[2].plot(group.embedding_similarity, group.target_probability_change, color=color, marker="o",
                     linewidth=2, label=set_type)
        for _, row in group.iterrows():
            axes[2].annotate(f"{row.alpha:g}", (row.embedding_similarity, row.target_probability_change),
                             textcoords="offset points", xytext=(4, 4), fontsize=7, color=color)
    axes[2].set_xlabel("Embedding similarity to unsteered"); axes[2].set_ylabel("Change in target probability")
    axes[2].set_title("Steering vs semantic preservation", fontsize=10)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--set-types", nargs="+", choices=SET_CHOICES, default=list(SET_CHOICES))
    parser.add_argument("--emotions", nargs="+", help="Target emotions (default: every non-neutral label).")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--delta-mode", choices=("mean_diff", "selectivity"), default="mean_diff")
    parser.add_argument("--delta-norm", choices=("matched", "raw"), default="matched")
    parser.add_argument("--max-prompts", type=int, help="Limit the held-out steering prompts.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-path", default=str(PROBE_PATH))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="sentence-transformers model for semantic similarity, or 'none'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    steering_dir = output / "steering"; steering_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_json(output / "capture_metadata.json")
    if metadata["model_id"] != args.model_id:
        raise ValueError(f"Neurons were identified with {metadata['model_id']}, not {args.model_id}.")
    labels = metadata["conditions"]
    emotions = args.emotions or [label for label in labels if label != NEUTRAL]
    unknown = sorted(set(emotions) - set(labels) | ({NEUTRAL} & set(emotions)))
    if unknown:
        parser.error(f"Invalid target emotions: {unknown}")
    key = f"{args.top_percent:g}%"
    neuron_sets = read_json(output / "neuron_sets" / "neuron_sets.json")
    if key not in neuron_sets["sets"]:
        parser.error(f"{key} is not in neuron_sets.json; rerun identify_neurons with that percentage.")
    sets = neuron_sets["sets"][key]
    score_table = pd.read_csv(output / "neuron_sets" / f"neuron_sets_{key}.csv")
    scores = {(row.set_type, row.emotion): {} for row in score_table[["set_type", "emotion"]].drop_duplicates().itertuples()}
    for row in score_table.itertuples():
        scores[(row.set_type, row.emotion)][(row.layer_index, row.neuron_index)] = row.selectivity_score
    means = np.load(output / "neuron_sets" / "decode_condition_means.npy", mmap_mode="r")
    neutral_mean = np.asarray(means[labels.index(NEUTRAL)], dtype=np.float64)
    width = metadata["neurons_per_layer"]

    prompts = pd.read_csv(output / "prompts" / "steering_prompts.csv")
    if args.max_prompts:
        prompts = prompts.iloc[:args.max_prompts]
    model, tokenizer = load_model(args.model_id, allow_mismatch=args.allow_model_mismatch)
    layers = model_layers(model)
    special, stops = special_token_ids(tokenizer), stop_token_ids(model, tokenizer)
    controller = MLPController(layers)
    prompt_ids = [prompt_token_ids(tokenizer, text, NEUTRAL) for text in prompts.text]

    def generate(steer: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> list[dict[str, Any]]:
        controller.steer = steer
        rows = []
        try:
            for start in range(0, len(prompt_ids), args.batch_size):
                for result in greedy_decode(model, controller, prompt_ids[start:start + args.batch_size],
                                            args.max_new_tokens, pad_id=tokenizer.pad_token_id,
                                            stop_ids=stops, special_ids=special):
                    rows.append({"generated_ids": result.generated_ids,
                                 "text": tokenizer.decode(result.generated_ids, skip_special_tokens=True).strip()})
        finally:
            controller.steer = {}
        return rows

    responses_path = steering_dir / "steering_responses.jsonl"
    completed = {(r["emotion"], r["set_type"], r["alpha"]) for r in read_jsonl(responses_path)} \
        if args.resume and responses_path.exists() else set()
    if not args.resume and responses_path.exists():
        responses_path.unlink()
    norms = []
    try:
        baseline = generate({})
        with responses_path.open("a", encoding="utf-8") as handle:
            configurations = [(e, s, a) for e in emotions for a in args.alphas for s in args.set_types]
            for emotion, set_type, alpha in tqdm(configurations, desc="Steering"):
                emotion_mean = np.asarray(means[labels.index(emotion)], dtype=np.float64)
                generation_deltas = set_deltas(sets["generation"][emotion], scores[("generation", emotion)],
                                               emotion_mean, neutral_mean, alpha, args.delta_mode)
                if set_type == "random":
                    rng = np.random.default_rng([args.seed, labels.index(emotion)])
                    neurons = {int(layer): rng.choice(width, len(ids), replace=False).tolist()
                               for layer, ids in sets["generation"][emotion].items()}
                    if args.delta_mode == "mean_diff":
                        deltas = set_deltas(neurons, {}, emotion_mean, neutral_mean, alpha, "mean_diff")
                    else:   # the generation set's shifts, permuted onto random neurons
                        deltas = {layer: (neurons[layer], list(rng.permutation(generation_deltas[layer][1])))
                                  for layer in generation_deltas}
                else:
                    deltas = set_deltas(sets[set_type][emotion], scores[(set_type, emotion)], emotion_mean,
                                        neutral_mean, alpha, args.delta_mode)
                raw_norm, target_norm = delta_norm(deltas), delta_norm(generation_deltas)
                if args.delta_norm == "matched" and raw_norm > 0:
                    deltas = rescale(deltas, target_norm / raw_norm)
                norms.append({"emotion": emotion, "set_type": set_type, "alpha": alpha, "raw_delta_norm": raw_norm,
                              "applied_delta_norm": delta_norm(deltas),
                              "neurons": sum(len(ids) for ids, _ in deltas.values())})
                if (emotion, set_type, alpha) in completed:
                    continue
                for prompt_id, row in zip(prompts.prompt_id, generate(steering_tensors(deltas))):
                    handle.write(json.dumps({"emotion": emotion, "set_type": set_type, "alpha": alpha,
                                             "prompt_id": int(prompt_id), **row}, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        controller.close()

    # ---- scoring (no hooks are active any more)
    steered = pd.DataFrame(read_jsonl(responses_path))
    base = pd.DataFrame({"prompt_id": prompts.prompt_id.to_numpy(), "user_text": prompts.text.to_numpy(),
                         "baseline_text": [row["text"] for row in baseline],
                         "baseline_ids": [row["generated_ids"] for row in baseline]})
    classifier, probe_labels = load_probe(args.probe_path, input_device(model))
    base_probs = probe_probabilities(model, tokenizer, classifier, base.baseline_text.tolist(), special,
                                     args.batch_size)
    steered_probs = probe_probabilities(model, tokenizer, classifier, steered.text.tolist(), special, args.batch_size)
    position = {prompt_id: index for index, prompt_id in enumerate(base.prompt_id)}
    nll = lambda ids_list, prompt_list: [-float(np.mean(values)) if values else np.nan for values in
                                         continuation_logprobs(model, prompt_list, ids_list, tokenizer.pad_token_id)]
    base_nll, steered_nll = [], []
    for start in range(0, len(base), args.batch_size):
        base_nll += nll(base.baseline_ids.iloc[start:start + args.batch_size].tolist(),
                        prompt_ids[start:start + args.batch_size])
    for start in tqdm(range(0, len(steered), args.batch_size), desc="Fluency NLL"):
        chunk = steered.iloc[start:start + args.batch_size]
        steered_nll += nll(chunk.generated_ids.tolist(), [prompt_ids[position[p]] for p in chunk.prompt_id])
    if args.embedding_model.lower() != "none":
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(args.embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
        base_embeddings = encoder.encode(base.baseline_text.tolist(), normalize_embeddings=True)
        steered_embeddings = encoder.encode(steered.text.tolist(), normalize_embeddings=True)
        similarity = [float(steered_embeddings[i] @ base_embeddings[position[p]])
                      for i, p in enumerate(steered.prompt_id)]
    else:
        similarity = [np.nan] * len(steered)

    rows = []
    for index, row in enumerate(steered.itertuples()):
        reference = position[row.prompt_id]
        target = probe_labels.index(row.emotion)
        user_words = content_words(base.user_text.iloc[reference])
        rows.append({
            "emotion": row.emotion, "set_type": row.set_type, "alpha": row.alpha, "prompt_id": row.prompt_id,
            "user_text": base.user_text.iloc[reference], "baseline_text": base.baseline_text.iloc[reference],
            "steered_text": row.text,
            "baseline_target_probability": float(base_probs[reference, target]),
            "steered_target_probability": float(steered_probs[index, target]),
            "target_probability_change": float(steered_probs[index, target] - base_probs[reference, target]),
            "target_is_argmax": float(steered_probs[index].argmax() == target),
            "target_lexicon_rate": lexicon_rate(row.text, row.emotion),
            "embedding_similarity": similarity[index],
            "content_f1_to_baseline": content_f1(row.text, base.baseline_text.iloc[reference]),
            "user_content_recall": len(user_words & content_words(row.text)) / len(user_words) if user_words else np.nan,
            "nll_change": steered_nll[index] - base_nll[reference],
            "repeated_bigram_rate": repeated_bigram_rate(row.text),
        })
    table = pd.DataFrame(rows)
    table.to_csv(steering_dir / "steering_scores.csv", index=False)
    base.drop(columns="baseline_ids").assign(nll=base_nll).to_csv(steering_dir / "baseline_responses.csv", index=False)
    pd.DataFrame(norms).drop_duplicates(["emotion", "set_type", "alpha"]).to_csv(
        steering_dir / "intervention_norms.csv", index=False)
    overall = summarize(table, ["set_type", "alpha"], args.seed)
    overall.to_csv(steering_dir / "steering_summary.csv", index=False)
    summarize(table, ["emotion", "set_type", "alpha"], args.seed).to_csv(
        steering_dir / "steering_summary_by_emotion.csv", index=False)
    pairs = [("generation", "recognition"), ("generation", "random"), ("recognition", "random"),
             ("emotional_token", "generation"), ("emotional_token", "recognition")]
    paired_tests(table, pairs).to_csv(steering_dir / "paired_comparisons.csv", index=False)
    plot_steering(overall, steering_dir / "steering_tradeoff.png")
    write_json(steering_dir / "steering_metadata.json", {
        "model_id": args.model_id, "top_percent": args.top_percent, "set_types": args.set_types,
        "emotions": emotions, "alphas": args.alphas, "delta_mode": args.delta_mode, "delta_norm": args.delta_norm,
        "prompt_condition": NEUTRAL, "steering_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
        "intervention": "additive shift on selected down_proj inputs at every decode step; prefill unchanged",
        "probe_path": args.probe_path, "embedding_model": args.embedding_model, "seed": args.seed,
    })
    print(overall[["set_type", "alpha", "target_probability_change", "target_is_argmax",
                   "embedding_similarity", "content_f1_to_baseline", "nll_change"]].to_string(index=False))
    print(f"Saved steering comparison to {steering_dir}")


if __name__ == "__main__":
    main()
