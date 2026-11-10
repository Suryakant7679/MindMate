#!/usr/bin/env python3
"""Causal validation of shared, recognition-only and generation-only emotion neurons.

For every emotion e and layer l the existing neuron files define
R = recognition set and G = generation set (``neuron_sets/neuron_sets.json``;
selection methodology unchanged).  Groups:

* ``shared``           R ∩ G
* ``recognition_only`` R \\ G
* ``generation_only``  G \\ R
* ``random``           layer-matched neurons outside R ∪ G

Masking is **mean-ablation**: masked activations are replaced by the neuron's
natural mean (recognition: calibration dataset texts; generation: decode
steps of the identification responses), so every group, random included,
stays inside its own activation range.  With ``--size-matching matched``
(default) all groups use |R ∩ G| neurons in each layer (R \\ G and G \\ R are
seeded random subsamples), so intervention size is matched by neuron count;
the realised displacement norm is reported for every condition.

Tasks (run separately):
* recognition — frozen probe on held-out dataset texts; masking at all positions.
* generation — emotion-conditioned responses to held-out prompts; masking on
  decode steps only; scored with a GoEmotions classifier.

Positive controls shift *all* neurons by the full emotion-vs-neutral mean
difference: ``pc_remove`` subtracts it (recognition texts / emotion-prompted
generation), ``pc_add`` adds it to neutral-prompted generation.

Run from the project root:  python3 -m generation_analysis.causal_validation
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from .common import (MODEL_ID, NEUTRAL, PROBE_PATH, RESULTS_DIR, MLPController, continuation_logprobs,
                     greedy_decode, input_device, load_model, load_probe, model_layers, prompt_token_ids,
                     read_json, read_jsonl, special_token_ids, stop_token_ids, write_json)
from .steering import bootstrap_ci, repeated_bigram_rate

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
SCORER = "suryakantmani/xlm-roberta-large-goemotions-hard"
# Standard GoEmotions label order (the scorer's config only names LABEL_i).
GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
              "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
              "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
              "remorse", "sadness", "surprise", "neutral"]
GROUPS = ("shared", "recognition_only", "generation_only")
PC_REMOVE, PC_ADD = "pc_remove", "pc_add"


# ----------------------------------------------------------------------- groups

def build_groups(recognition: dict[str, list[int]], generation: dict[str, list[int]], width: int,
                 size_matching: str, rng: np.random.Generator) -> dict[str, dict[int, list[int]]]:
    """Per-layer neuron ids for each group and its layer-matched random control.

    ``matched``: every group (and ``random``) has |R ∩ G| neurons per layer.
    ``full``: groups keep their sizes and each gets its own ``random_<group>``.
    Random neurons are drawn from outside R ∪ G of the same emotion and layer.
    """
    groups: dict[str, dict[int, list[int]]] = {}
    for key in recognition:
        layer, r, g = int(key), set(recognition[key]), set(generation[key])
        members = {"shared": sorted(r & g), "recognition_only": sorted(r - g), "generation_only": sorted(g - r)}
        pool = np.asarray(sorted(set(range(width)) - (r | g)))
        if size_matching == "matched":
            k = len(members["shared"])
            for name in ("recognition_only", "generation_only"):
                members[name] = sorted(rng.choice(members[name], min(k, len(members[name])), replace=False).tolist())
            members["random"] = sorted(rng.choice(pool, k, replace=False).tolist())
        else:
            for name in GROUPS:
                members[f"random_{name}"] = sorted(rng.choice(pool, len(members[name]), replace=False).tolist())
        for name, ids in members.items():
            groups.setdefault(name, {})[layer] = [int(n) for n in ids]
    return groups


def parse_scopes(names: Sequence[str], bands: Sequence[str], layers: int) -> dict[str, list[int]]:
    scopes: dict[str, list[int]] = {}
    if "all" in names:
        scopes["all"] = list(range(layers))
    if "bands" in names:
        for band in bands:
            start, end = (int(value) for value in band.split("-"))
            scopes[f"layers_{start + 1:02d}-{end + 1:02d}"] = list(range(start, end + 1))
    if "layers" in names:
        scopes.update({f"layer_{layer + 1:02d}": [layer] for layer in range(layers)})
    return scopes


def random_name(group: str, size_matching: str) -> str:
    return "random" if size_matching == "matched" else f"random_{group}"


# ------------------------------------------------------------------ intervention

class InterventionController(MLPController):
    """down_proj input intervention: mean-replacement of neurons and/or a full-width shift.

    ``configure`` takes one ``{"replace": ..., "shift": ...}`` spec per phase
    ("prefill", "decode"), so each phase can use its own reference values.
    The realised intervention size is the mean per-position L2 norm (over all
    layers) of the change on valid positions, pooled over intervened phases.
    """

    def __init__(self, layers: Sequence[Any]) -> None:
        super().__init__(layers)
        self.configure()

    def configure(self, **by_phase: dict[str, Any]) -> None:
        self.by_phase = {}
        for phase, spec in by_phase.items():
            replace = {layer: (torch.as_tensor(ids, dtype=torch.long), torch.as_tensor(values, dtype=torch.float32))
                       for layer, (ids, values) in (spec.get("replace") or {}).items() if len(ids)}
            shift = {layer: torch.as_tensor(vector, dtype=torch.float32)
                     for layer, vector in (spec.get("shift") or {}).items()}
            if replace or shift:
                self.by_phase[phase] = (replace, shift, min([*replace, *shift]))
        self.squared, self.positions = 0.0, 0

    def realised_norm(self) -> float:
        return float(np.sqrt(self.squared / self.positions)) if self.positions else 0.0

    def _hook(self, layer_index: int):
        def hook(_module: Any, inputs: tuple[Any, ...]):
            if self.phase not in self.by_phase:
                return None
            replace, shift, first = self.by_phase[self.phase]
            if layer_index not in replace and layer_index not in shift:
                return None
            activation = inputs[0].clone()
            valid = (self.prefill_mask if self.phase == "prefill" else self.decode_active[:, None])
            valid = valid.to(activation.device)
            change = torch.zeros_like(activation, dtype=torch.float32)
            if layer_index in replace:
                ids, values = (t.to(activation.device) for t in replace[layer_index])
                change[..., ids] += values - activation[..., ids].float()
            if layer_index in shift:
                change += shift[layer_index].to(activation.device)
            # Special tokens (BOS attention sink, chat headers) and padding are left untouched.
            change = torch.where(valid.unsqueeze(-1), change, 0.0)
            activation = (activation.float() + change).to(activation.dtype)
            self.squared += float(change.square().sum(-1)[valid].sum())
            if layer_index == first:
                self.positions += int(valid.sum())
            return (activation, *inputs[1:])
        return hook


def replacement(groups: dict[int, list[int]], scope: list[int], reference: np.ndarray
                ) -> dict[int, tuple[list[int], np.ndarray]]:
    return {layer: (groups[layer], reference[layer, groups[layer]]) for layer in scope if groups.get(layer)}


# ------------------------------------------------------------------ recognition

def load_probe_split_helpers() -> Any:
    """Import 05_train_emotion_probe.py (digit-prefixed) to reuse its exact split."""
    path = Path(__file__).resolve().parent.parent / "05_train_emotion_probe.py"
    spec = importlib.util.spec_from_file_location("probe_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_sample(frame: pd.DataFrame, indices: np.ndarray, per_class: int, seed: int) -> pd.DataFrame:
    subset = frame.iloc[indices]
    return (subset.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(min(per_class, len(group)), random_state=seed))
            .reset_index(drop=False).rename(columns={"index": "text_id"}))


def recognition_pass(model: Any, tokenizer: Any, controller: InterventionController, classifier: torch.nn.Module,
                     texts: Sequence[str], special: set[int], batch_size: int,
                     collect_means: tuple[np.ndarray, np.ndarray, Sequence[int]] | None = None) -> np.ndarray:
    """Probe probabilities for texts (pooling identical to 05_train_emotion_probe.py).

    With ``collect_means=(sums [C,L,U], counts [C], class_ids)`` the unmodified
    down_proj inputs are also accumulated per class (calibration pass).
    """
    device, rows = input_device(model), []
    special_ids = torch.as_tensor(sorted(special), dtype=torch.long)
    probe_device = next(classifier.parameters()).device
    handles, captured = [], {}
    if collect_means is not None:
        for index, layer in enumerate(model_layers(model)):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(
                lambda _m, inputs, i=index: captured.__setitem__(i, inputs[0].detach())))
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(list(texts[start:start + batch_size]), return_tensors="pt", padding=True,
                                    truncation=True, max_length=256)
                mask = encoded["attention_mask"].bool() & ~torch.isin(encoded["input_ids"], special_ids)
                controller.phase, controller.prefill_mask = "prefill", mask.to(device)
                hidden = model.model(input_ids=encoded["input_ids"].to(device),
                                     attention_mask=encoded["attention_mask"].to(device),
                                     use_cache=False).last_hidden_state
                weights = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
                rows.append(torch.softmax(classifier(pooled.float().to(probe_device)), -1).cpu().numpy())
                if collect_means is not None:
                    sums, counts, class_ids = collect_means
                    batch_classes = class_ids[start:start + batch_size]
                    for row, class_id in enumerate(batch_classes):
                        valid = mask[row]
                        counts[class_id] += int(valid.sum())
                        for layer, activation in captured.items():
                            sums[class_id, layer] += activation[row][valid.to(activation.device)].float().sum(0).cpu().numpy()
    finally:
        controller.phase = None
        for handle in handles:
            handle.remove()
    return np.concatenate(rows)


def recognition_metrics(probabilities: np.ndarray, truth: np.ndarray, target: int) -> dict[str, float]:
    predicted = probabilities.argmax(1)
    target_rows = truth == target
    return {"target_probability": float(probabilities[target_rows, target].mean()),
            "target_recall": float((predicted[target_rows] == target).mean()),
            "accuracy": float((predicted == truth).mean()),
            "macro_f1": float(f1_score(truth, predicted, average="macro")),
            "target_f1": float(f1_score(truth == target, predicted == target))}


# ------------------------------------------------------------------- generation

def goemotion_scores(texts: Sequence[str], labels: Sequence[str], model_id: str, batch_size: int = 64) -> np.ndarray:
    """Sigmoid GoEmotions probabilities for the ten study labels."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    scorer = AutoModelForSequenceClassification.from_pretrained(model_id)
    if scorer.config.num_labels != len(GOEMOTIONS):
        raise ValueError(f"{model_id} has {scorer.config.num_labels} labels, expected the 28 GoEmotions labels.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = scorer.to(device).eval()
    columns = [GOEMOTIONS.index(label) for label in labels]
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer([text or " " for text in texts[start:start + batch_size]], return_tensors="pt",
                                padding=True, truncation=True, max_length=128).to(device)
            rows.append(torch.sigmoid(scorer(**encoded).logits.float())[:, columns].cpu().numpy())
    del scorer
    return np.concatenate(rows) if rows else np.zeros((0, len(labels)))


def token_change_rate(ids: Sequence[int], reference: Sequence[int]) -> float:
    """1 - (shared prefix length / longer length); 0 means identical output."""
    longest = max(len(ids), len(reference), 1)
    prefix = next((i for i, (a, b) in enumerate(zip(ids, reference)) if a != b), min(len(ids), len(reference)))
    return 1.0 - prefix / longest


# --------------------------------------------------------------------- analysis

def paired_p(first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    difference = difference[~np.isnan(difference)]
    return float(wilcoxon(difference).pvalue) if len(difference) > 1 and np.any(difference) else 1.0


def summarize(items: pd.DataFrame, keys: list[str], metrics: Sequence[str], seed: int) -> pd.DataFrame:
    rows = []
    for key, group in items.groupby(keys, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["items"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[metric] = float(np.nanmean(values))
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_ci(values, seed)
        rows.append(row)
    return pd.DataFrame(rows)


def hypothesis_table(effects: pd.DataFrame, size_matching: str) -> pd.DataFrame:
    """Group effects vs random and vs the other task, per scope (pooled over emotions).

    ``effects`` has one row per (task, emotion, scope, group, item) with column
    ``effect`` (drop in target-emotion probability) and ``normalized`` (effect
    divided by the task's positive-control effect for that emotion).
    """
    rows = []
    for scope, frame in effects.groupby("scope", sort=False):
        for group in GROUPS:
            control = random_name(group, size_matching)
            record = {"scope": scope, "group": group}
            for task in ("recognition", "generation"):
                own = frame[(frame.task == task) & (frame.group == group)].set_index(["emotion", "item"])
                rand = frame[(frame.task == task) & (frame.group == control)].set_index(["emotion", "item"])
                joined = own.join(rand, rsuffix="_random", how="inner")
                record[f"{task}_effect"] = float(joined.effect.mean()) if len(joined) else np.nan
                record[f"{task}_random_effect"] = float(joined.effect_random.mean()) if len(joined) else np.nan
                record[f"{task}_excess_over_random"] = record[f"{task}_effect"] - record[f"{task}_random_effect"]
                record[f"{task}_p_vs_random"] = paired_p(joined.effect.to_numpy(), joined.effect_random.to_numpy())
                record[f"{task}_normalized_excess"] = float((joined.normalized - joined.normalized_random).mean()) \
                    if joined.normalized.notna().any() else np.nan
                record[f"{task}_emotions_normalized"] = int(joined[joined.normalized.notna()]
                                                            .index.get_level_values("emotion").nunique())
            record["specificity_recognition_minus_generation"] = (
                record["recognition_normalized_excess"] - record["generation_normalized_excess"])
            rows.append(record)
    table = pd.DataFrame(rows)
    expectations = {"recognition_only": "recognition > generation", "generation_only": "generation > recognition",
                    "shared": "both > random"}
    table["hypothesis"] = table.group.map(expectations)
    def supported(row: pd.Series) -> str:
        rec_ok = row.recognition_excess_over_random > 0 and row.recognition_p_vs_random < 0.05
        gen_ok = row.generation_excess_over_random > 0 and row.generation_p_vs_random < 0.05
        if row.group == "shared":
            return "yes" if rec_ok and gen_ok else "no"
        specificity = row.specificity_recognition_minus_generation
        if np.isnan(specificity):
            return "not evaluable (a positive control failed, so task effects cannot be normalised)"
        if row.group == "recognition_only":
            return "yes" if rec_ok and specificity > 0 else "no"
        return "yes" if gen_ok and specificity < 0 else "no"
    table["supported"] = table.apply(supported, axis=1)
    return table


def plot_effects(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"shared": "#8e44ad", "recognition_only": "#2a6fdb", "generation_only": "#d1495b", "random": "#8a8a8a"}
    scopes = list(dict.fromkeys(summary.scope))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, sharey=False)
    for axis, task in zip(axes, ("recognition", "generation")):
        frame = summary[summary.task == task]
        names = [name for name in colors if name in set(frame.group)] + \
                [name for name in dict.fromkeys(frame.group) if name not in colors]
        width = 0.8 / max(len(names), 1)
        for offset, name in enumerate(names):
            values = frame[frame.group == name].set_index("scope").reindex(scopes)
            axis.bar(np.arange(len(scopes)) + offset * width, values.effect, width,
                     yerr=[values.effect - values.effect_ci_low, values.effect_ci_high - values.effect],
                     color=colors.get(name, "#bbbbbb"), label=name, capsize=2)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(np.arange(len(scopes)) + 0.4 - width / 2, scopes, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel("Drop in target-emotion probability")
        axis.set_title(task.capitalize(), fontsize=10)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


# ------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Generation-analysis results to read.")
    parser.add_argument("--output-dir", help="Default: <results-dir>/causal_validation")
    parser.add_argument("--top-percent", type=float, default=1.0)
    parser.add_argument("--emotions", nargs="+")
    parser.add_argument("--size-matching", choices=("matched", "full"), default="matched")
    parser.add_argument("--scopes", nargs="+", choices=("all", "bands", "layers"), default=["all", "bands"])
    parser.add_argument("--layer-bands", nargs="+", default=["0-7", "8-15", "16-23", "24-31"],
                        help="0-indexed inclusive layer ranges for the 'bands' scope.")
    parser.add_argument("--tasks", nargs="+", choices=("recognition", "generation"),
                        default=["recognition", "generation"])
    parser.add_argument("--csv-path", default="data/mlt_hinghlish_dataset_clean.csv")
    parser.add_argument("--probe-path", default=str(PROBE_PATH))
    parser.add_argument("--recognition-per-class", type=int, default=60, help="Held-out probe-test texts per class.")
    parser.add_argument("--calibration-per-class", type=int, default=100, help="Probe-train texts per class for means.")
    parser.add_argument("--generation-prompts", type=int, help="Limit the held-out generation prompts.")
    parser.add_argument("--generation-positions", choices=("decode", "all"), default="decode",
                        help="decode: intervene on generated-token steps only; all: also on prompt prefill, "
                             "using prefill means as the reference there.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scorer", default=SCORER)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--pc-min-effect", type=float, default=0.05,
                        help="Minimum mean target-probability change for a positive control to count as working.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results = Path(args.results_dir)
    output = Path(args.output_dir) if args.output_dir else results / "causal_validation"
    for name in ("groups", "reference", "recognition", "generation"):
        (output / name).mkdir(parents=True, exist_ok=True)
    metadata = read_json(results / "capture_metadata.json")
    if metadata["model_id"] != args.model_id:
        raise ValueError(f"Neuron sets come from {metadata['model_id']}, not {args.model_id}.")
    labels = metadata["conditions"]
    emotions = args.emotions or [label for label in labels if label != NEUTRAL]
    if set(emotions) - set(labels) or NEUTRAL in emotions:
        parser.error("--emotions must be non-neutral recognition labels.")
    sets = read_json(results / "neuron_sets" / "neuron_sets.json")["sets"][f"{args.top_percent:g}%"]
    width, num_layers = metadata["neurons_per_layer"], metadata["num_layers"]
    scopes = parse_scopes(args.scopes, args.layer_bands, num_layers)

    # ---- neuron groups (from the existing R and G files only)
    rng = np.random.default_rng(args.seed)
    groups = {e: build_groups(sets["recognition"][e], sets["generation"][e], width, args.size_matching, rng)
              for e in emotions}
    group_names = list(next(iter(groups.values())))
    size_rows = []
    for emotion in emotions:
        r_sets, g_sets = sets["recognition"][emotion], sets["generation"][emotion]
        for layer in range(num_layers):
            r, g = set(r_sets[str(layer)]), set(g_sets[str(layer)])
            size_rows.append({"emotion": emotion, "layer": layer + 1, "layer_index": layer, "R": len(r), "G": len(g),
                              "R_and_G": len(r & g), "R_minus_G": len(r - g), "G_minus_R": len(g - r),
                              **{f"used_{name}": len(groups[emotion][name].get(layer, [])) for name in group_names}})
    sizes = pd.DataFrame(size_rows)
    sizes.to_csv(output / "groups" / "group_sizes_by_layer.csv", index=False)
    write_json(output / "groups" / "neuron_groups.json", {
        "top_percent": args.top_percent, "size_matching": args.size_matching, "seed": args.seed,
        "groups": {e: {name: {str(l): ids for l, ids in by_layer.items()} for name, by_layer in g.items()}
                   for e, g in groups.items()}})

    model, tokenizer = load_model(args.model_id, allow_mismatch=args.allow_model_mismatch)
    layers = model_layers(model)
    special, stops = special_token_ids(tokenizer), stop_token_ids(model, tokenizer)
    controller = InterventionController(layers)
    classifier, probe_labels = load_probe(args.probe_path, input_device(model))
    if probe_labels != labels:
        raise ValueError("Probe label order differs from the neuron-set labels.")
    conditions = [(group, scope) for group in group_names for scope in scopes]
    effect_rows: list[dict[str, Any]] = []
    positive: dict[str, Any] = {}

    try:
        # ================================================================ recognition
        if "recognition" in args.tasks:
            helpers = load_probe_split_helpers()
            frame, probe_order = helpers.prepare_dataframe(args.csv_path)
            if probe_order != labels:
                raise ValueError("Dataset labels differ from the probe labels.")
            # Seed 42 reproduces the probe's own split, so evaluation texts are probe-test texts.
            train, _, test = helpers.stratified_split(frame.label_id.to_numpy(), 42)
            evaluation = balanced_sample(frame, test, args.recognition_per_class, args.seed)
            calibration = balanced_sample(frame, train, args.calibration_per_class, args.seed)
            evaluation.to_csv(output / "recognition" / "evaluation_texts.csv", index=False)
            means_path = output / "reference" / "recognition_class_means.npy"
            if args.resume and means_path.exists():
                class_means = np.load(means_path)
            else:
                sums = np.zeros((len(labels), num_layers, width), dtype=np.float64)
                counts = np.zeros(len(labels), dtype=np.int64)
                controller.configure()
                recognition_pass(model, tokenizer, controller, classifier, calibration.text.tolist(), special,
                                 args.batch_size, (sums, counts, calibration.label_id.tolist()))
                class_means = (sums / counts[:, None, None]).astype(np.float32)
                np.save(means_path, class_means)
                np.save(output / "reference" / "recognition_class_positions.npy", counts)
            reference = class_means.mean(axis=0)          # balanced over classes
            truth = evaluation.label_id.to_numpy()
            texts = evaluation.text.tolist()

            def run(name: str, emotion: str, scope: str, **intervention) -> None:
                controller.configure(prefill=intervention)
                probabilities = recognition_pass(model, tokenizer, controller, classifier, texts, special,
                                                 args.batch_size)
                target = labels.index(emotion)
                records.append({"emotion": emotion, "group": name, "scope": scope,
                                "realised_l2_norm": controller.realised_norm(),
                                "neurons": sum(len(ids) for ids, _ in intervention.get("replace", {}).values())
                                if "replace" in intervention else num_layers * width,
                                **recognition_metrics(probabilities, truth, target)})
                for row in np.where(truth == target)[0]:
                    items.append((emotion, name, scope, int(evaluation.text_id.iloc[row]),
                                  float(probabilities[row, target]), int(probabilities[row].argmax() == target)))

            records, items = [], []
            controller.configure()
            base_probabilities = recognition_pass(model, tokenizer, controller, classifier, texts, special,
                                                  args.batch_size)
            for emotion in tqdm(emotions, desc="Recognition masking"):
                target = labels.index(emotion)
                records.append({"emotion": emotion, "group": "baseline", "scope": "none", "realised_l2_norm": 0.0,
                                "neurons": 0, **recognition_metrics(base_probabilities, truth, target)})
                for row in np.where(truth == target)[0]:
                    items.append((emotion, "baseline", "none", int(evaluation.text_id.iloc[row]),
                                  float(base_probabilities[row, target]),
                                  int(base_probabilities[row].argmax() == target)))
                for group, scope in conditions:
                    run(group, emotion, scope, replace=replacement(groups[emotion][group], scopes[scope], reference))
                difference = class_means[target] - class_means[labels.index(NEUTRAL)]
                run(PC_REMOVE, emotion, "all", shift={layer: -difference[layer] for layer in range(num_layers)})
            summary = pd.DataFrame(records)
            baseline = summary[summary.group == "baseline"].set_index("emotion")
            for metric in ("target_probability", "target_recall", "accuracy", "macro_f1", "target_f1"):
                summary[f"{metric}_change"] = summary[metric] - summary.emotion.map(baseline[metric])
            summary.to_csv(output / "recognition" / "summary_by_emotion.csv", index=False)
            recognition_items = pd.DataFrame(items, columns=["emotion", "group", "scope", "item", "target_probability",
                                                             "correct"])
            recognition_items.to_csv(output / "recognition" / "item_scores.csv.gz", index=False)
            base_items = recognition_items[recognition_items.group == "baseline"].set_index(["emotion", "item"])
            joined = recognition_items[recognition_items.group != "baseline"].join(
                base_items.target_probability.rename("baseline"), on=["emotion", "item"])
            joined["effect"] = joined.baseline - joined.target_probability
            pc = joined[joined.group == PC_REMOVE].groupby("emotion").effect.mean()
            positive["recognition"] = {"pc_remove_mean_drop": float(pc.mean()), "by_emotion": pc.round(4).to_dict(),
                                       "p": paired_p(joined[joined.group == PC_REMOVE].baseline.to_numpy(),
                                                     joined[joined.group == PC_REMOVE].target_probability.to_numpy())}
            effect_rows += [{"task": "recognition", **row} for row in
                            joined[["emotion", "group", "scope", "item", "effect"]].to_dict("records")]

        # ================================================================= generation
        if "generation" in args.tasks:
            prompts = pd.read_csv(results / "prompts" / "steering_prompts.csv")
            if args.generation_prompts:
                prompts = prompts.iloc[:args.generation_prompts]
            decode_means = np.load(results / "neuron_sets" / "decode_condition_means.npy", mmap_mode="r")
            positions = np.load(results / "neuron_sets" / "decode_positions.npy").astype(np.float64)
            reference = np.tensordot(positions / positions.sum(), np.asarray(decode_means, dtype=np.float64), axes=1)
            np.save(output / "reference" / "generation_reference_mean.npy", reference.astype(np.float32))
            prefill_means = np.load(results / "neuron_sets" / "prefill_condition_means.npy", mmap_mode="r")
            prefill_positions = np.load(results / "neuron_sets" / "prefill_positions.npy").astype(np.float64)
            prefill_reference = np.tensordot(prefill_positions / prefill_positions.sum(),
                                             np.asarray(prefill_means, dtype=np.float64), axes=1)
            phases = {"decode": (decode_means, reference)}
            if args.generation_positions == "all":
                phases["prefill"] = (prefill_means, prefill_reference)

            def masking(neurons: dict[int, list[int]], scope: list[int]) -> dict[str, Any]:
                return {phase: {"replace": replacement(neurons, scope, ref)} for phase, (_, ref) in phases.items()}

            def full_shift(emotion: str, sign: float) -> dict[str, Any]:
                specs = {}
                for phase, (means, _) in phases.items():
                    difference = np.asarray(means[labels.index(emotion)] - means[labels.index(NEUTRAL)], np.float32)
                    specs[phase] = {"shift": {l: sign * difference[l] for l in range(num_layers)}}
                return specs
            prompt_ids = {condition: [prompt_token_ids(tokenizer, text, condition) for text in prompts.text]
                          for condition in [*emotions, NEUTRAL]}

            def generate(condition: str) -> list[dict[str, Any]]:
                rows = []
                for start in range(0, len(prompts), args.batch_size):
                    for result in greedy_decode(model, controller, prompt_ids[condition][start:start + args.batch_size],
                                                args.max_new_tokens, pad_id=tokenizer.pad_token_id, stop_ids=stops,
                                                special_ids=special):
                        rows.append({"generated_ids": result.generated_ids,
                                     "text": tokenizer.decode(result.generated_ids, skip_special_tokens=True).strip()})
                return rows

            path = output / "generation" / "responses.jsonl"
            done = {(r["emotion"], r["group"], r["scope"]) for r in read_jsonl(path)} \
                if args.resume and path.exists() else set()
            if not args.resume and path.exists():
                path.unlink()
            plan = [(e, "baseline", "none", e, {}) for e in emotions] + [(NEUTRAL, "baseline", "none", NEUTRAL, {})]
            for emotion in emotions:
                plan += [(emotion, group, scope, emotion, masking(groups[emotion][group], scopes[scope]))
                         for group, scope in conditions]
                plan.append((emotion, PC_REMOVE, "all", emotion, full_shift(emotion, -1.0)))
                plan.append((emotion, PC_ADD, "all", NEUTRAL, full_shift(emotion, 1.0)))
            with path.open("a", encoding="utf-8") as handle:
                for emotion, group, scope, condition, intervention in tqdm(plan, desc="Generation masking"):
                    if (emotion, group, scope) in done:
                        continue
                    controller.configure(**intervention)
                    rows = generate(condition)
                    norm = controller.realised_norm()
                    for prompt_id, row in zip(prompts.prompt_id, rows):
                        handle.write(json.dumps({"emotion": emotion, "group": group, "scope": scope,
                                                 "prompt_condition": condition, "prompt_id": int(prompt_id),
                                                 "realised_l2_norm": norm, **row}, ensure_ascii=False) + "\n")
                    handle.flush()
            controller.configure()

            # ---- scoring (no intervention active)
            table = pd.DataFrame(read_jsonl(path))
            position = {prompt_id: index for index, prompt_id in enumerate(prompts.prompt_id)}
            nll = []
            for start in tqdm(range(0, len(table), args.batch_size), desc="Fluency NLL"):
                chunk = table.iloc[start:start + args.batch_size]
                values = continuation_logprobs(model, [prompt_ids[c][position[p]] for c, p in
                                                       zip(chunk.prompt_condition, chunk.prompt_id)],
                                               chunk.generated_ids.tolist(), tokenizer.pad_token_id)
                nll += [-float(np.mean(v)) if v else np.nan for v in values]
            table["nll"] = nll
            del model
            torch.cuda.empty_cache()
            scores = goemotion_scores(table.text.tolist(), labels, args.scorer)
            table["target_probability"] = [scores[i, labels.index(e)] for i, e in enumerate(table.emotion)]
            table["target_is_top"] = [float(scores[i].argmax() == labels.index(e)) for i, e in enumerate(table.emotion)]
            table["repetition_rate"] = table.text.map(repeated_bigram_rate)
            # Baseline for every row: same emotion and prompt, same prompt condition, no intervention.
            base = table[table.group == "baseline"].set_index(["prompt_condition", "prompt_id"])
            key = list(zip(table.prompt_condition, table.prompt_id))
            for column in ("target_probability", "nll", "text", "generated_ids"):
                table[f"baseline_{column}"] = [base.at[k, column] for k in key]
            # For pc_add the baseline target probability is the neutral response's score for the target emotion.
            neutral_rows = table[(table.group == "baseline") & (table.prompt_condition == NEUTRAL)].index
            neutral_scores = {table.prompt_id[i]: scores[i] for i in neutral_rows}
            add_rows = table.group == PC_ADD
            table.loc[add_rows, "baseline_target_probability"] = [
                neutral_scores[p][labels.index(e)] for p, e in zip(table.prompt_id[add_rows], table.emotion[add_rows])]
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(args.embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")
            a = encoder.encode(table.text.tolist(), normalize_embeddings=True)
            b = encoder.encode(table.baseline_text.tolist(), normalize_embeddings=True)
            table["embedding_similarity"] = (a * b).sum(1)
            table["target_probability_change"] = table.target_probability - table.baseline_target_probability
            table["nll_change"] = table.nll - table.baseline_nll
            table["changed"] = [float(list(x) != list(y)) for x, y in zip(table.generated_ids, table.baseline_generated_ids)]
            table["token_change_rate"] = [token_change_rate(x, y) for x, y in zip(table.generated_ids,
                                                                                  table.baseline_generated_ids)]
            table = table[~((table.group == "baseline") & (table.prompt_condition == NEUTRAL))]
            table.drop(columns=["generated_ids", "baseline_generated_ids"]).to_csv(
                output / "generation" / "item_scores.csv.gz", index=False)
            metrics = ["target_probability", "target_probability_change", "target_is_top", "nll_change",
                       "embedding_similarity", "repetition_rate", "changed", "token_change_rate", "realised_l2_norm"]
            summarize(table, ["emotion", "group", "scope"], metrics, args.seed).to_csv(
                output / "generation" / "summary_by_emotion.csv", index=False)
            summarize(table, ["group", "scope"], metrics, args.seed).to_csv(
                output / "generation" / "summary_by_group.csv", index=False)
            for name in (PC_REMOVE, PC_ADD):
                rows = table[table.group == name]
                change = rows.target_probability_change.to_numpy()
                positive[f"generation_{name}"] = {
                    "mean_target_probability_change": float(change.mean()),
                    "by_emotion": rows.groupby("emotion").target_probability_change.mean().round(4).to_dict(),
                    "p": paired_p(rows.target_probability.to_numpy(), rows.baseline_target_probability.to_numpy()),
                    "mean_embedding_similarity": float(rows.embedding_similarity.mean()),
                    "mean_nll_change": float(rows.nll_change.mean())}
            masked = table[~table.group.isin(["baseline", PC_REMOVE, PC_ADD])]
            effect_rows += [{"task": "generation", "emotion": r.emotion, "group": r.group, "scope": r.scope,
                             "item": r.prompt_id, "effect": -r.target_probability_change}
                            for r in masked.itertuples()]
            for name, sign in ((PC_REMOVE, -1.0), (PC_ADD, 1.0)):
                effect_rows += [{"task": "generation", "emotion": r.emotion, "group": name, "scope": "all",
                                 "item": r.prompt_id, "effect": sign * r.target_probability_change}
                                for r in table[table.group == name].itertuples()]
    finally:
        controller.close()

    # ==================================================================== combined
    effects = pd.DataFrame(effect_rows)
    controls = []
    for (task, control, emotion), frame in effects[effects.group.isin([PC_REMOVE, PC_ADD])].groupby(
            ["task", "group", "emotion"]):
        values = frame.effect.to_numpy()          # positive = change in the intended direction
        mean, p = float(values.mean()), paired_p(values, np.zeros_like(values))
        controls.append({"task": task, "control": control, "emotion": emotion, "items": len(values),
                         "mean_effect": mean, "wilcoxon_p": p, "works": bool(mean >= args.pc_min_effect and p < 0.05)})
    controls = pd.DataFrame(controls)
    controls.to_csv(output / "positive_control_by_emotion.csv", index=False)
    verdict = {}
    for task in sorted(set(controls.task)) if len(controls) else []:
        frame = controls[controls.task == task]
        working = sorted(set(frame[frame.works].emotion))
        failing = sorted(set(frame.emotion) - set(working))
        verdict[task] = {"emotions_with_working_control": working, "emotions_without": failing}
        if task == "generation":
            verdict[task]["statement"] = (
                "The steering setup itself is insufficient: shifting all neurons by the full emotion-vs-neutral "
                "activation difference did not change generated emotion for any emotion, so null generation "
                "effects cannot be read as absence of a causal role." if not working else
                "The positive control changes generated emotion for " + ", ".join(working) + "." +
                (" It fails for " + ", ".join(failing) + "; generation effects for those emotions are not "
                 "interpretable and are excluded from normalised comparisons." if failing else ""))
    positive["verdict"] = verdict
    write_json(output / "positive_control.json", positive)

    if not effects.empty:
        # Normalise by the task's pc_remove effect, only for emotions whose control works.
        removal = controls[(controls.control == PC_REMOVE) & controls.works].set_index(["task", "emotion"]).mean_effect
        scale = np.asarray([removal.get(key, np.nan) for key in zip(effects.task, effects.emotion)], dtype=float)
        effects["normalized"] = effects.effect / scale
        effects.to_csv(output / "item_effects.csv.gz", index=False)
        masked = effects[~effects.group.isin([PC_REMOVE, PC_ADD])]
        summary = summarize(masked, ["task", "group", "scope"], ["effect", "normalized"], args.seed)
        summary.to_csv(output / "effects_by_group_and_scope.csv", index=False)
        summarize(masked, ["task", "emotion", "group", "scope"], ["effect", "normalized"], args.seed).to_csv(
            output / "effects_by_emotion.csv", index=False)
        if {"recognition", "generation"} <= set(effects.task):
            hypotheses = hypothesis_table(masked, args.size_matching)
            hypotheses.to_csv(output / "hypothesis_tests.csv", index=False)
            print(hypotheses[["scope", "group", "recognition_excess_over_random", "recognition_p_vs_random",
                              "generation_excess_over_random", "generation_p_vs_random",
                              "specificity_recognition_minus_generation", "generation_emotions_normalized",
                              "supported"]].round(4).to_string(index=False))
        plot_effects(summary, output / "effects_by_group.png")
    write_json(output / "experiment_metadata.json", {
        "model_id": args.model_id, "neuron_sets": str(results / "neuron_sets" / "neuron_sets.json"),
        "top_percent": args.top_percent, "emotions": emotions, "size_matching": args.size_matching,
        "scopes": scopes, "masking": "mean-ablation of down_proj inputs",
        "recognition_reference": "class-balanced mean over calibration texts (probe-train split)",
        "generation_reference": "position-weighted mean decode activation over identification responses",
        "recognition_intervention_positions": "all non-padded positions (single forward)",
        "generation_intervention_positions": "decode steps only (prompt prefill unchanged)",
        "effect_definition": "baseline minus intervened target-emotion probability (positive = impairment)",
        "normalized_effect": "effect / mean pc_remove effect of the same task and emotion",
        "recognition_scorer": args.probe_path, "generation_scorer": args.scorer,
        "pc_min_effect": args.pc_min_effect, "seed": args.seed, "tasks": args.tasks,
    })
    print(json.dumps(positive["verdict"], indent=2))
    print(f"Saved causal validation results to {output}")


if __name__ == "__main__":
    main()
