#!/usr/bin/env python3
"""Stage 3: generation, emotional-token and semantic-token neuron sets, and their overlap.

No model is loaded; this reads stage 1/2 outputs and the (read-only)
recognition selectivity tensor.

* Generation set G_e: decode-step firing counts grouped by response condition,
  scored with the recognition formula S = P * (1 - H) over the same ten labels.
* Token sets: decode-step firing patterns of individual generated tokens in
  emotional responses, grouped into classes ``emotional_<e>`` (one per
  non-neutral emotion) and ``semantic``, scored with the same formula.
  E_e = top neurons of ``emotional_<e>``; Sem = top neurons of ``semantic``.
* Recognition set R_e: top neurons of results/llama31_8b_emotions_full/selectivity.npy.

Layer-wise Jaccard is reported for every pair of R, G, E and Sem, together
with the Jaccard expected for random sets of the same sizes and a
hypergeometric overlap p-value.

Run from the project root:  python3 -m generation_analysis.identify_neurons
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from tqdm.auto import tqdm

from .common import (NEUTRAL, RECOGNITION_SELECTIVITY, RESULTS_DIR, emission_slot, expected_random_jaccard,
                     jaccard, read_json, read_jsonl, recognition_metadata, selectivity_scores, top_k,
                     top_neurons, write_json)
from .lexicon import content_words, is_content_word, is_emotion_word, normalize_word

SET_TYPES = ("recognition", "generation", "emotional_token", "semantic")
SEMANTIC = "semantic"
ALL = "all"


# ------------------------------------------------------------ token categories

def group_words(pieces: Sequence[str], special: Sequence[bool]) -> list[int]:
    """Word index for every token piece; -1 for punctuation, emoji bytes and specials."""
    word_ids, current, previous_word = [], -1, False
    for piece, is_special in zip(pieces, special):
        is_word = not is_special and any(character.isalpha() for character in piece)
        if not is_word:
            word_ids.append(-1)
            previous_word = False
            continue
        if not previous_word or piece[:1].isspace():
            current += 1
        word_ids.append(current)
        previous_word = True
    return word_ids


def categorize_tokens(pieces: Sequence[str], special: Sequence[bool], deltas: Sequence[float],
                      shared_words: set[str] | None, *, emotional_delta: float, semantic_delta: float,
                      criterion: str) -> tuple[list[str], list[str]]:
    """Return (category, word) per token: 'emotional', 'semantic' or 'other'.

    A word is emotional if it matches the emotion lexicon and/or its summed
    matched-prompt log-ratio is at least ``emotional_delta`` (``criterion``).
    A word is semantic if it is a non-emotion content word whose log-ratio is
    within ``semantic_delta`` of zero and, when ``shared_words`` is given, it
    also occurs in the user message or the matched neutral response.
    """
    word_ids = group_words(pieces, special)
    words: dict[int, list[int]] = {}
    for index, word_id in enumerate(word_ids):
        if word_id >= 0:
            words.setdefault(word_id, []).append(index)
    categories, texts = ["other"] * len(pieces), [""] * len(pieces)
    for indices in words.values():
        word = normalize_word("".join(pieces[index] for index in indices).strip())
        delta = float(np.nansum([deltas[index] for index in indices]))
        lexical, shifted = is_emotion_word(word), delta >= emotional_delta
        emotional = {"lexicon": lexical, "delta": shifted, "either": lexical or shifted,
                     "both": lexical and shifted}[criterion]
        semantic = (not lexical and is_content_word(word) and abs(delta) <= semantic_delta
                    and (shared_words is None or word in shared_words))
        category = "emotional" if emotional and word else "semantic" if semantic else "other"
        for index in indices:
            categories[index], texts[index] = category, word
    return categories, texts


# ------------------------------------------------------------------ aggregation

def condition_totals(counts: np.ndarray, means: np.ndarray, steps: np.ndarray, rows_by_class: list[list[int]],
                     chunk: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Firing counts [L,U,C], positions [C] and token-weighted mean activation [C,L,U]."""
    _, layers, width = counts.shape
    totals = np.zeros((layers, width, len(rows_by_class)), dtype=np.int64)
    activation = np.zeros((len(rows_by_class), layers, width), dtype=np.float64)
    positions = np.zeros(len(rows_by_class), dtype=np.int64)
    for class_index, rows in enumerate(rows_by_class):
        rows = sorted(rows)
        for start in range(0, len(rows), chunk):
            block = rows[start:start + chunk]
            weight = steps[block].astype(np.float64)[:, None, None]
            totals[:, :, class_index] += counts[block].astype(np.int64).sum(axis=0)
            activation[class_index] += (means[block].astype(np.float64) * weight).sum(axis=0)
        positions[class_index] = int(steps[rows].sum()) if rows else 0
        activation[class_index] /= max(positions[class_index], 1)
    return totals, positions, activation


def token_class_counts(bits: np.ndarray, slots_by_class: list[list[tuple[int, int]]], width: int,
                       chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Firing counts [L,U,K] from bit-packed decode slots grouped by token class."""
    layers = bits.shape[2]
    totals = np.zeros((layers, width, len(slots_by_class)), dtype=np.int64)
    for class_index, slots in enumerate(slots_by_class):
        slots = sorted(slots)
        for start in tqdm(range(0, len(slots), chunk), desc=f"Token class {class_index}", leave=False):
            responses, steps = map(np.asarray, zip(*slots[start:start + chunk]))
            fired = np.unpackbits(bits[responses, steps], axis=-1)[..., :width]
            totals[:, :, class_index] += fired.sum(axis=0, dtype=np.int64)
    return totals, np.asarray([len(slots) for slots in slots_by_class], dtype=np.int64)


# ---------------------------------------------------------------------- overlap

def overlap_rows(sets: dict[str, dict[str, list[set[int]]]], emotions: Sequence[str], width: int,
                 percentage: float) -> list[dict[str, Any]]:
    rows = []
    def add(layer: int, emotion: str, name_a: str, name_b: str, a: set[int], b: set[int]) -> None:
        shared = len(a & b)
        expected = len(a) * len(b) / width
        rows.append({"percentage": percentage, "layer": layer + 1, "layer_index": layer, "emotion": emotion,
                     "set_a": name_a, "set_b": name_b, "size_a": len(a), "size_b": len(b),
                     "intersection": shared, "union": len(a | b), "jaccard": jaccard(a, b),
                     "expected_random_jaccard": expected_random_jaccard(len(a), len(b), width),
                     "enrichment": shared / expected if expected else np.nan,
                     "hypergeom_p": float(hypergeom.sf(shared - 1, width, len(a), len(b)))})
    layers = len(next(iter(sets["recognition"].values())))
    for layer in range(layers):
        for name_a, name_b in combinations(SET_TYPES, 2):
            union_a, union_b = set(), set()
            for emotion in emotions:
                a = sets[name_a].get(ALL if name_a == SEMANTIC else emotion)
                b = sets[name_b].get(ALL if name_b == SEMANTIC else emotion)
                if a is None or b is None:
                    continue
                if emotion != NEUTRAL:
                    union_a |= a[layer]; union_b |= b[layer]
                if SEMANTIC not in (name_a, name_b) or emotion != NEUTRAL:
                    add(layer, emotion, name_a, name_b, a[layer], b[layer])
            add(layer, "union_of_emotions", name_a, name_b, union_a, union_b)
    return rows


def plot_overlap(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    per_emotion = table[table.emotion != "union_of_emotions"]
    pairs = per_emotion.groupby(["set_a", "set_b"], sort=False)
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    colors = ["#2a6fdb", "#d1495b", "#00a6a6", "#edae49", "#66a182", "#8e6c8a"]
    for color, ((name_a, name_b), group) in zip(colors, pairs):
        layer_mean = group.groupby("layer").agg(jaccard=("jaccard", "mean"),
                                                expected=("expected_random_jaccard", "mean"))
        axis.plot(layer_mean.index, layer_mean.jaccard, color=color, linewidth=2, label=f"{name_a} / {name_b}")
    axis.axhline(per_emotion.expected_random_jaccard.mean(), color="#777777", linestyle="--", linewidth=1,
                 label="random sets of equal size")
    axis.set_xlabel("Layer"); axis.set_ylabel("Jaccard (mean over emotions)")
    axis.set_title(f"Layer-wise neuron-set overlap, top {table.percentage.iloc[0]:g}% per layer")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


# ------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--recognition-selectivity", default=str(RECOGNITION_SELECTIVITY))
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--percentages", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--adherence", choices=("matched", "none"), default="matched",
                        help="matched: keep an emotional response only if the frozen probe gives its emotion "
                             "a higher probability than for the matched neutral response.")
    parser.add_argument("--token-alignment", choices=("emit", "read"), default="emit",
                        help="emit: activation of the decode step that produced the token; "
                             "read: activation of the step that consumed it.")
    parser.add_argument("--emotional-criterion", choices=("either", "both", "lexicon", "delta"), default="either")
    parser.add_argument("--emotional-delta", type=float, default=1.0,
                        help="Minimum summed log p(word|emotion prompt) - log p(word|neutral prompt), in nats.")
    parser.add_argument("--semantic-delta", type=float, default=0.5)
    parser.add_argument("--semantic-source", choices=("emotional", "all"), default="emotional",
                        help="Responses contributing semantic tokens (emotional: same responses as emotional tokens).")
    parser.add_argument("--no-shared-semantic", action="store_true",
                        help="Do not require semantic words to occur in the user message or neutral twin.")
    args = parser.parse_args()
    if any(not 0 < value <= 100 for value in args.percentages):
        parser.error("--percentages must be in (0, 100].")

    output = Path(args.output_dir)
    metadata = read_json(output / "capture_metadata.json")
    recognition = recognition_metadata()
    labels = list(recognition["labels"])
    if metadata["conditions"] != labels:
        raise ValueError("Generation conditions differ from the recognition label order.")
    if metadata["model_id"] != recognition["model_id"] and not args.allow_model_mismatch:
        raise ValueError(f"Generation used {metadata['model_id']}; recognition used {recognition['model_id']}.")
    emotions = [label for label in labels if label != NEUTRAL]
    activation_dir, analysis = output / "activations", output / "token_analysis"
    responses = read_jsonl(activation_dir / "responses.jsonl")
    if len(responses) != metadata["responses"]:
        raise ValueError("Stage 1 is incomplete.")
    scores = pd.read_csv(analysis / "response_scores.csv").set_index("response_id")
    twin = {r["prompt_id"]: r["response_id"] for r in responses if r["condition"] == NEUTRAL}

    kept = []
    for response in responses:
        keep = response["condition"] == NEUTRAL or args.adherence == "none"
        if not keep:
            column = f"probe_{response['condition']}"
            keep = scores.at[response["response_id"], column] > scores.at[twin[response["prompt_id"]], column]
        kept.append(bool(keep) and response["decode_steps"] > 0)
    adherence = pd.DataFrame({"condition": [r["condition"] for r in responses], "kept": kept})
    adherence = adherence.groupby("condition").kept.agg(["sum", "count"]).rename(
        columns={"sum": "kept", "count": "generated"})
    adherence["kept_fraction"] = adherence.kept / adherence.generated

    # ---- generation (decode) and prefill selectivity
    width, num_layers = metadata["neurons_per_layer"], metadata["num_layers"]
    rows_by_condition = [[r["response_id"] for r, keep in zip(responses, kept) if keep and r["condition"] == label]
                         for label in labels]
    sets_dir = output / "neuron_sets"; sets_dir.mkdir(parents=True, exist_ok=True)
    selectivity = {}
    for phase, step_key in (("decode", "decode_steps"), ("prefill", "prefill_tokens")):
        steps = np.asarray([r[step_key] for r in responses], dtype=np.int64)
        counts, positions, means = condition_totals(
            np.load(activation_dir / f"{phase}_firing_counts.npy", mmap_mode="r"),
            np.load(activation_dir / f"{phase}_activation_mean.npy", mmap_mode="r"), steps, rows_by_condition)
        selectivity[phase] = selectivity_scores(counts, positions).astype(np.float32)
        np.save(sets_dir / f"{phase}_firing_counts.npy", counts)
        np.save(sets_dir / f"{phase}_positions.npy", positions)
        np.save(sets_dir / f"{phase}_condition_means.npy", means.astype(np.float32))
        np.save(sets_dir / f"{'generation' if phase == 'decode' else phase}_selectivity.npy", selectivity[phase])

    # ---- token categories
    tokens = pd.read_csv(analysis / "token_scores.csv.gz", keep_default_na=False,
                         dtype={"token_text": str}, na_values={"logp_neutral": [""], "delta": [""]})
    tokens["special"] = tokens.special.astype(str).eq("True")
    by_response = {key: group.sort_values("token_index") for key, group in tokens.groupby("response_id")}
    class_names = [f"emotional_{emotion}" for emotion in emotions] + [SEMANTIC]
    slots: list[list[tuple[int, int]]] = [[] for _ in class_names]
    category_rows = []
    texts = {r["response_id"]: r["response_text"] for r in responses}
    for response, keep in zip(responses, kept):
        is_neutral = response["condition"] == NEUTRAL
        if not keep or (is_neutral and args.semantic_source == "emotional"):
            continue
        group = by_response[response["response_id"]]
        shared = None
        if not args.no_shared_semantic:
            shared = content_words(response["user_text"])
            if not is_neutral:
                shared |= content_words(texts[twin[response["prompt_id"]]])
        categories, words = categorize_tokens(
            group.token_text.tolist(), group.special.tolist(), group.delta.fillna(0.0).tolist(), shared,
            emotional_delta=args.emotional_delta, semantic_delta=args.semantic_delta,
            criterion=args.emotional_criterion)
        special_flags = group.special.tolist()
        for index, (category, word) in enumerate(zip(categories, words)):
            slot = emission_slot(index, args.token_alignment)
            valid = (slot is not None and slot < response["read_count"] and not special_flags[slot]
                     and not special_flags[index])
            if category == "emotional" and is_neutral:
                category = "other"          # neutral responses contribute semantic tokens only
            name = SEMANTIC if category == "semantic" else f"emotional_{response['condition']}"
            used = valid and category != "other"
            if used:
                slots[class_names.index(name)].append((response["response_id"], slot))
            category_rows.append((response["response_id"], response["condition"], index,
                                  group.token_text.iloc[index], word, category, slot if valid else -1, used))
    pd.DataFrame(category_rows, columns=["response_id", "condition", "token_index", "token_text", "word",
                                         "category", "slot", "used"]).to_csv(
        analysis / "token_categories.csv.gz", index=False)
    token_counts, token_positions = token_class_counts(
        np.load(activation_dir / "decode_token_firing_bits.npy", mmap_mode="r"), slots, width)
    missing = [name for name, count in zip(class_names, token_positions) if count == 0]
    if missing:
        raise ValueError(f"No tokens were assigned to {missing}; relax the token criteria or add prompts.")
    token_selectivity = selectivity_scores(token_counts, token_positions).astype(np.float32)
    np.save(sets_dir / "token_firing_counts.npy", token_counts)
    np.save(sets_dir / "token_positions.npy", token_positions)
    np.save(sets_dir / "token_selectivity.npy", token_selectivity)

    # ---- neuron sets
    recognition_scores = np.load(args.recognition_selectivity, mmap_mode="r")
    if recognition_scores.shape != (num_layers, width, len(labels)):
        raise ValueError(f"Recognition selectivity has shape {recognition_scores.shape}.")
    sources = {"recognition": (recognition_scores, labels), "generation": (selectivity["decode"], labels)}
    all_sets: dict[str, Any] = {}
    overlap: list[dict[str, Any]] = []
    for percentage in sorted(set(args.percentages)):
        key = f"{percentage:g}%"
        sets: dict[str, dict[str, list[set[int]]]] = {name: {} for name in SET_TYPES}
        csv_rows = []
        def register(set_type: str, emotion: str, scores_2d: np.ndarray) -> None:
            ids = top_neurons(scores_2d, percentage)
            sets[set_type][emotion] = [set(layer_ids) for layer_ids in ids]
            for layer, layer_ids in enumerate(ids):
                csv_rows.extend((set_type, emotion, layer, rank + 1, neuron, float(scores_2d[layer, neuron]))
                                for rank, neuron in enumerate(layer_ids))
        for set_type, (scores_3d, class_labels) in sources.items():
            for index, emotion in enumerate(class_labels):
                register(set_type, emotion, np.asarray(scores_3d[:, :, index]))
        for index, emotion in enumerate(emotions):
            register("emotional_token", emotion, token_selectivity[:, :, index])
        register(SEMANTIC, ALL, token_selectivity[:, :, class_names.index(SEMANTIC)])
        pd.DataFrame(csv_rows, columns=["set_type", "emotion", "layer_index", "rank", "neuron_index",
                                        "selectivity_score"]).to_csv(sets_dir / f"neuron_sets_{key}.csv", index=False)
        all_sets[key] = {set_type: {emotion: {str(layer): sorted(ids) for layer, ids in enumerate(layers)}
                                    for emotion, layers in by_emotion.items()}
                         for set_type, by_emotion in sets.items()}
        overlap.extend(overlap_rows(sets, labels, width, percentage))
    write_json(sets_dir / "neuron_sets.json", {
        "model_id": metadata["model_id"], "labels": labels, "percentages": sorted(set(args.percentages)),
        "neurons_per_layer_per_set": {f"{p:g}%": top_k(width, p) for p in sorted(set(args.percentages))},
        "selection_score": "S[l,u,c] = P[l,u,c] * (1 - H[l,u]); top percentage within each layer",
        "set_definitions": {
            "recognition": "dataset-text forward passes (results/llama31_8b_emotions_full/selectivity.npy)",
            "generation": "decode-step activations grouped by response condition (ten labels)",
            "emotional_token": f"decode slots of emotional words, {args.token_alignment} alignment; "
                               "classes emotional_<e> and semantic",
            "semantic": "decode slots of shared, non-emotional content words (class 'semantic')"},
        "token_classes": class_names, "token_class_positions": dict(zip(class_names, token_positions.tolist())),
        "sets": all_sets,
    })

    overlap_dir = output / "overlap"; overlap_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(overlap)
    table.to_csv(overlap_dir / "layerwise_jaccard.csv", index=False)
    per_emotion = table[table.emotion != "union_of_emotions"]
    summary = (per_emotion.groupby(["percentage", "set_a", "set_b"])
               .agg(mean_jaccard=("jaccard", "mean"), expected_random_jaccard=("expected_random_jaccard", "mean"),
                    mean_enrichment=("enrichment", "mean"),
                    rows=("jaccard", "size"),
                    rows_significant=("hypergeom_p", lambda p: int((p < 0.05 / p.size).sum())))
               .reset_index())
    summary.to_csv(overlap_dir / "overlap_summary.csv", index=False)
    for percentage, group in table.groupby("percentage"):
        plot_overlap(group, overlap_dir / f"layerwise_jaccard_{percentage:g}pct.png")
    adherence.to_csv(analysis / "adherence.csv")
    write_json(output / "identification_metadata.json", {
        "adherence_filter": args.adherence, "token_alignment": args.token_alignment,
        "emotional_criterion": args.emotional_criterion, "emotional_delta": args.emotional_delta,
        "semantic_delta": args.semantic_delta, "semantic_source": args.semantic_source,
        "semantic_requires_shared_word": not args.no_shared_semantic,
        "kept_responses": int(sum(kept)), "generated_responses": len(responses),
        "decode_positions": dict(zip(labels, np.load(sets_dir / "decode_positions.npy").tolist())),
        "token_class_positions": dict(zip(class_names, token_positions.tolist())),
        "significance": "rows_significant: (layer, emotion) rows with hypergeometric p < 0.05 / rows, "
                        "Bonferroni within each percentage and set pair; union_of_emotions excludes neutral",
    })
    print(summary.to_string(index=False))
    print(f"Saved neuron sets to {sets_dir} and overlap to {overlap_dir}")


if __name__ == "__main__":
    main()
