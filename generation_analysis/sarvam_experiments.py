#!/usr/bin/env python3
"""Sarvam-30B steering experiments: causal validation (dev), calibration (both judges), held-out test.

  probe      a few development messages at several strengths, printed (to find the usable range)
  causal     development messages, judge 1: dose-response of the tone panel, reversed shift, matched random
             neurons, layer bands, all-neuron shift, the Hinglish language panel, and neuron ablation
             (panel neurons held at their neutral mean while the prompt asks for the tone)
  calibrate  calibration messages, both judges: tone panel, language panel and both together, four strengths
  select     experimental policy: per strategy and language the intervention both judges find safe and
             helpful, or none
  heldout    untouched messages: MindMate prompt only / knowledge-graph prompt / KG + steering policy

Every steered reply shares a decoding batch with the unsteered reply to the same prompt.
  python3 -m generation_analysis.sarvam_experiments causal causal_score causal_report --alphas 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import SYSTEM_PROMPT
from chatbot_feature.emotion_reasoner.knowledge_graph import STRATEGIES, is_hinglish
from chatbot_feature.emotion_reasoner.agent import strategy_prompt
from .sarvam_lab import (LAB, LANGUAGE_PANEL, PANEL_FILE, ROOT, STRATEGY_TONE, TONES, Panel, attribution_ranking,
                         generate_rows, load_items, load_panels, load_runtime, messages_of, sites)
from .steering_lab import release
from .v3_common import (JUDGES, RATED, Judge, automatic_scores, holm, judge_preferences, judge_ratings, paired_summary)

QUALITY = ("appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness")
RULE = {"tolerance": 0.05, "ci_margin": 0.15, "max_harm_share": 0.20, "min_preference_each": 0.45,
        "min_preference_mean": 0.50, "min_n": 8}
GATE = {"rule": "ceiling_and_safeguards", "ceiling": 0.30, "min_gain": 0.05, "max_repetition_increase": 0.05,
        "min_length_ratio": 0.4}


def tone_rubric(tone: str) -> str:
    return (f"Tone: how strongly does the reply itself express {TONES[tone]}? Judge only the tone of the reply, "
            "not whether it is a good reply. 1 = not at all. 5 = very strongly.")


def kg_prompt(row) -> str:
    title, _, guidance, _, _ = STRATEGIES[row.strategy]
    return strategy_prompt(SYSTEM_PROMPT, {"strategy_title": title, "guidance": guidance})


def subset(panel: Panel, layers) -> Panel:
    return {layer: value for layer, value in panel.items() if layer in set(layers)}


def random_panel(panel: Panel, widths: dict[int, int], rng: np.random.Generator) -> Panel:
    """Same layers, neuron counts and shift magnitudes, on randomly chosen neurons."""
    return {layer: (sorted(rng.choice(widths[layer], len(neurons), replace=False).tolist()), list(rng.permutation(deltas)))
            for layer, (neurons, deltas) in panel.items()}


def combined(first: Panel, second: Panel) -> Panel:
    out: Panel = {}
    for layer in sorted(set(first) | set(second)):
        values: dict[int, float] = {}
        for panel in (first, second):
            for neuron, delta in zip(*panel.get(layer, ([], []))):
                values[neuron] = values.get(neuron, 0.0) + delta
        out[layer] = (sorted(values), [values[neuron] for neuron in sorted(values)])
    return out


# ------------------------------------------------------------------ generation

def run(stage: str, items: pd.DataFrame, builder, args) -> None:
    out = LAB / stage
    out.mkdir(parents=True, exist_ok=True)
    path = out / "generations.jsonl"
    done = {json.loads(line)["item_id"] for line in path.open()} if path.exists() else set()
    runtime = load_runtime()
    context = {"runtime": runtime, "panels": load_panels(), "full": load_panels(full=True), "widths": sites(runtime),
               "neutral": np.load(LAB / "discovery" / "neutral_means.npy"), "args": args,
               "directions": dict(np.load(LAB / "residual_directions.npz"))}
    if stage == "causal3":
        context["rankings"] = {tone: attribution_ranking(tone) for tone in TONES}
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            if row.item_id in done or not isinstance(row.tone, str):
                continue
            record = {"item_id": row.item_id, "replies": []}
            for prompt, rows in builder(row, context):
                replies = generate_rows(runtime, messages_of(row), prompt, [spec for _, spec in rows], args.max_new_tokens)
                record["replies"] += [{"name": name, "reply": reply} for (name, _), reply in zip(rows, replies)]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(stage, row.item_id, flush=True)
    del runtime
    release()


def causal_rows(row, context):
    a = context["args"].alphas
    widths, tone = context["widths"], context["panels"][row.tone]
    language, full = context["panels"][LANGUAGE_PANEL], context["full"][row.tone]
    rng = np.random.default_rng(zlib.crc32(row.item_id.encode()))
    randoms = [random_panel(tone, widths, rng) for _ in range(2)]
    layers = sorted(tone)
    third = len(layers) // 3
    bands = {"low": layers[:third], "mid": layers[third:2 * third], "high": layers[2 * third:]}
    rows = [("none", {})]
    rows += [(f"panel@{alpha:g}", {"shift": [(tone, alpha)]}) for alpha in a]
    rows += [(f"reverse@{-a[1]:g}", {"shift": [(tone, -a[1])]}),
             (f"random0@{a[1]:g}", {"shift": [(randoms[0], a[1])]}), (f"random0@{a[2]:g}", {"shift": [(randoms[0], a[2])]}),
             (f"random1@{a[2]:g}", {"shift": [(randoms[1], a[2])]})]
    rows += [(f"band_{name}@{a[2]:g}", {"shift": [(subset(tone, band), a[2])]}) for name, band in bands.items()]
    rows += [(f"full@{a[0]:g}", {"shift": [(full, a[0])]}), (f"full@{a[1]:g}", {"shift": [(full, a[1])]}),
             *[(f"language@{value:g}", {"shift": [(language, value)]}) for value in context["args"].language_alphas],
             (f"panel+language@{a[1]:g}", {"shift": [(tone, a[1]), (language, context["args"].combo_language)]})]
    offsets, start = {}, 0
    for layer in sorted(widths):
        offsets[layer] = start
        start += widths[layer]
    neutral = context["neutral"]

    def clamp(panel: Panel) -> dict:
        return {layer: (neurons, [float(neutral[offsets[layer] + n]) for n in neurons]) for layer, (neurons, _) in panel.items()}
    styled = kg_prompt(row) + f" Make the reply clearly express {TONES[row.tone]}."
    styled_rows = [("styled", {}), ("styled_clamp", {"clamp": clamp(tone)}), ("styled_clamp_random", {"clamp": clamp(randoms[0])}),
                   (f"styled_reverse@{-a[1]:g}", {"shift": [(tone, -a[1])]}),
                   (f"styled_reverse_random@{-a[1]:g}", {"shift": [(randoms[0], -a[1])]})]
    return [(kg_prompt(row), rows), (styled, styled_rows)]


def causal2_rows(row, context):
    """Second development test: residual-stream directions, the late band at higher strength, finer language strengths."""
    tone, language = context["panels"][row.tone], context["panels"][LANGUAGE_PANEL]
    directions = context["directions"]
    late = subset(tone, [layer for layer in tone if layer >= 13])
    rows = [("none", {})]
    rows += [(f"residual_L{layer}@{alpha:g}", {"residual": [(layer, directions[row.tone][layer], alpha)]})
             for layer in (6, 10, 14) for alpha in (0.1, 0.2)]
    rows += [(f"residual_language_L10@{alpha:g}", {"residual": [(10, directions[LANGUAGE_PANEL][10], alpha)]}) for alpha in (0.1, 0.2)]
    rows += [(f"late@{alpha:g}", {"shift": [(late, alpha)]}) for alpha in (4.0, 5.0)]
    rows += [(f"language@{alpha:g}", {"shift": [(language, alpha)]}) for alpha in (0.75,)]
    return [(kg_prompt(row), rows)]


DIRECTION_LAYER = 10            # chosen on development messages (causal2): tone gain without quality loss
DIRECTION_ALPHAS = (0.1, 0.2, 0.3, 0.4)
TOP_K = (10, 50, 200)


def top_k_panel(panel: Panel, ranking: dict[int, list[int]], k: int) -> Panel:
    """The ``k`` most attributed neurons of the whole panel (across layers), with their own shifts."""
    scored = [(position, layer, neuron) for layer, order in ranking.items() for position, neuron in enumerate(order)]
    # Interleave layers by rank position so the subset is spread the way the attribution ranks it.
    chosen = sorted(scored)[:k]
    out: Panel = {}
    for _, layer, neuron in chosen:
        neurons, deltas = panel[layer]
        out.setdefault(layer, ([], []))
        out[layer][0].append(neuron)
        out[layer][1].append(deltas[neurons.index(neuron)])
    return out


def panel_subset(panel: Panel, k: int, rng: np.random.Generator) -> Panel:
    """``k`` neurons drawn at random from the panel itself (same shifts)."""
    flat = [(layer, position) for layer, (neurons, _) in panel.items() for position in range(len(neurons))]
    picks = [flat[i] for i in rng.choice(len(flat), min(k, len(flat)), replace=False)]
    out: Panel = {}
    for layer, position in sorted(picks):
        out.setdefault(layer, ([], []))
        out[layer][0].append(panel[layer][0][position])
        out[layer][1].append(panel[layer][1][position])
    return out


def causal3_rows(row, context):
    """Third development test (both judges): sham and wrong-panel controls, a third matched random panel, partial
    panels (top-k by attribution against random subsets of the panel, and a concentrated top-10 shift) for
    panel-level against individual-neuron causality, the layer-10 direction against random and reversed
    directions, and ablations (neutral clamp, zero clamp, late-band clamp, top-200 clamp) under a styled prompt."""
    a = context["args"].alphas
    tone_name = row.tone
    widths, tone, panels = context["widths"], context["panels"][tone_name], context["panels"]
    rng = np.random.default_rng(zlib.crc32(("causal3:" + row.item_id).encode()))
    random2 = random_panel(tone, widths, rng)
    other_name = [name for name in TONES if name != tone_name][rng.integers(len(TONES) - 1)]
    ranking = context["rankings"][tone_name]
    unit = context["directions"][tone_name][DIRECTION_LAYER]
    random_units = []
    for _ in range(2):
        vector = rng.standard_normal(len(unit))
        random_units.append(vector / np.linalg.norm(vector))
    rows = [("none", {}), ("sham@0", {"sham": True}),
            (f"panel@{a[2]:g}", {"shift": [(tone, a[2])]}),
            (f"random2@{a[2]:g}", {"shift": [(random2, a[2])]}),
            (f"other_{other_name}@{a[2]:g}", {"shift": [(panels[other_name], a[2])]})]
    rows += [(f"top{k}@{a[2]:g}", {"shift": [(top_k_panel(tone, ranking, k), a[2])]}) for k in TOP_K]
    rows += [(f"subset{k}@{a[2]:g}", {"shift": [(panel_subset(tone, k, rng), a[2])]}) for k in TOP_K[1:]]
    rows += [(f"top10@{a[2] * 10:g}", {"shift": [(top_k_panel(tone, ranking, 10), a[2] * 10)]})]
    rows += [("direction@0.2", {"residual": [(DIRECTION_LAYER, unit, 0.2)]}),
             ("direction_reverse@-0.2", {"residual": [(DIRECTION_LAYER, unit, -0.2)]}),
             ("direction_random0@0.2", {"residual": [(DIRECTION_LAYER, random_units[0], 0.2)]}),
             ("direction_random1@0.2", {"residual": [(DIRECTION_LAYER, random_units[1], 0.2)]})]
    offsets, start = {}, 0
    for layer in sorted(widths):
        offsets[layer] = start
        start += widths[layer]
    neutral = context["neutral"]

    def clamp(panel: Panel, value=None) -> dict:
        return {layer: (neurons, [float(neutral[offsets[layer] + n]) if value is None else value for n in neurons])
                for layer, (neurons, _) in panel.items()}
    late = subset(tone, [layer for layer in tone if layer >= 13])
    styled = kg_prompt(row) + f" Make the reply clearly express {TONES[tone_name]}."
    styled_rows = [("styled", {}), ("styled_clamp", {"clamp": clamp(tone)}), ("styled_zero", {"clamp": clamp(tone, 0.0)}),
                   ("styled_clamp_late", {"clamp": clamp(late)}), ("styled_clamp_top200", {"clamp": clamp(top_k_panel(tone, ranking, 200))}),
                   ("styled_clamp_random", {"clamp": clamp(random2)})]
    return [(kg_prompt(row), rows), (styled, styled_rows)]


def calibration_rows(row, context):
    """Tone neuron panel, tone residual direction, and for Hinglish users the language panel alone and with the direction."""
    args = context["args"]
    tone, language = context["panels"][row.tone], context["panels"][LANGUAGE_PANEL]
    unit = context["directions"][row.tone][DIRECTION_LAYER]
    rows = [("none", {})] + [(f"tone@{alpha:g}", {"shift": [(tone, alpha)]}) for alpha in args.alphas]
    rows += [(f"direction@{alpha:g}", {"residual": [(DIRECTION_LAYER, unit, alpha)]}) for alpha in DIRECTION_ALPHAS]
    if row.language == "hinglish":               # the language panel is only ever used for Hinglish users
        rows += [(f"language@{value:g}", {"shift": [(language, value)]}) for value in args.language_alphas]
        rows += [(f"direction+language@{alpha:g}", {"residual": [(DIRECTION_LAYER, unit, alpha)],
                                                    "shift": [(language, args.combo_language)]}) for alpha in DIRECTION_ALPHAS[:3]]
    return [(kg_prompt(row), rows)]


# ----------------------------------------------------------------------- scoring

def long_table(stage: str, split: str) -> pd.DataFrame:
    items = load_items(split).set_index("item_id")
    rows = []
    for line in (LAB / stage / "generations.jsonl").open():
        record = json.loads(line)
        item = items.loc[record["item_id"]]
        replies = {entry["name"]: entry["reply"] for entry in record["replies"]}
        for name, reply in replies.items():
            reference = None if name in ("none", "styled", "baseline") else replies["styled" if name.startswith("styled") else "none"]
            rows.append({"item_id": record["item_id"], "name": name, "reply": reply, "reference": reference, "text": item.text,
                         "history": item.history, "target": item.target, "strategy": item.strategy, "tone": item.tone,
                         "language": item.language})
    return pd.DataFrame(rows)


def score(stage: str, split: str, judges: tuple[str, ...]) -> None:
    out = LAB / stage
    table = automatic_scores(long_table(stage, split))
    table.drop(columns=["history"]).to_csv(out / "replies.csv", index=False)
    steered = table.reference.notna().to_numpy()
    for name in judges:
        if (out / f"ratings_{name}.csv").exists():
            continue
        model_id, four_bit = JUDGES[name]
        judge = Judge(model_id, load_in_4bit=four_bit)
        ratings = judge_ratings(judge, table, reference=[value if isinstance(value, str) else None for value in table.reference])
        contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
        keys = list(zip(contexts, table.reply.fillna(""), [tone_rubric(tone) for tone in table.tone]))
        unique = list(dict.fromkeys(keys))
        lookup = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings["tone"] = [float(lookup[key]) for key in keys]
        preference = np.full(len(table), 0.5)
        if steered.any():
            preference[steered] = judge_preferences(judge, table[steered], table.reply[steered].tolist(),
                                                    table.reference[steered].tolist())
        ratings["preference"] = preference
        ratings.to_csv(out / f"ratings_{name}.csv", index=False)
        judge.close()


def paired_table(stage: str, judges: tuple[str, ...]) -> pd.DataFrame:
    out = LAB / stage
    table = pd.read_csv(out / "replies.csv")
    metrics = ("tone", *RATED, "hinglish_naturalness")
    for name in judges:
        ratings = pd.read_csv(out / f"ratings_{name}.csv")
        for column in (*metrics, "preference", "preservation"):
            table[f"{column}_{name}"] = ratings[column].to_numpy() if column in ratings else np.nan
    for column in (*metrics, "preference", "preservation"):
        table[f"{column}_both"] = table[[f"{column}_{name}" for name in judges]].mean(axis=1)
    table["reference_name"] = np.where(table.name.str.startswith("styled"), "styled", "none")
    base = table[table.name.isin(["none", "styled"])].set_index(["item_id", "name"])
    rows = table[~table.name.isin(["none", "styled"])].copy()
    key = list(zip(rows.item_id, rows.reference_name))
    for source in (*judges, "both"):
        for metric in metrics:
            rows[f"d_{metric}_{source}"] = rows[f"{metric}_{source}"].to_numpy() - base.loc[key, f"{metric}_{source}"].to_numpy()
    for column in ("indep_target", "gate_target", "repeated_bigram_rate"):
        rows[f"d_{column}"] = rows[column].to_numpy() - base.loc[key, column].to_numpy()
    rows["unsteered_gate"] = base.loc[key, "gate_target"].to_numpy()
    rows["length_ratio"] = rows.reply_words.to_numpy() / np.maximum(base.loc[key, "reply_words"].to_numpy(), 1)
    rows["unsteered_is_hinglish"] = base.loc[key, "reply_is_hinglish"].to_numpy().astype(bool)
    rows["d_is_hinglish"] = rows.reply_is_hinglish.astype(float).to_numpy() - rows.unsteered_is_hinglish.astype(float)
    rows["changed"] = rows.reply.to_numpy() != base.loc[key, "reply"].to_numpy()
    return rows


def summarise(group: pd.DataFrame, source: str) -> dict:
    tone = paired_summary(group[f"d_tone_{source}"])
    quality = list(QUALITY) + (["hinglish_naturalness"] if group.language.eq("hinglish").all() else [])
    entry = {"n": int(len(group)), "changed_share": float(group.changed.mean()), "tone_gain": tone["mean"],
             "tone_gain_lo": tone["ci_low"], "tone_gain_hi": tone["ci_high"], "tone_p": tone["p"],
             "independent_gain": float(group.d_indep_target.mean()), "preference": float(group[f"preference_{source}"].mean()),
             "preservation": float(group[f"preservation_{source}"].mean()), "length_ratio": float(group.length_ratio.mean()),
             "repetition_change": float(group.d_repeated_bigram_rate.mean()),
             "hinglish_reply_share": float(group.reply_is_hinglish.mean()), "hinglish_share_change": float(group.d_is_hinglish.mean())}
    drops, failed = [], []
    for metric in quality:
        stats = paired_summary(group[f"d_{metric}_{source}"])
        entry[f"{metric}_change"], entry[f"{metric}_change_lo"] = stats["mean"], stats["ci_low"]
        if stats["mean"] < -RULE["tolerance"] or stats["ci_low"] < -RULE["ci_margin"]:
            failed.append(metric)
        drops.append(group[f"d_{metric}_{source}"])
    entry["harm_share"] = float((pd.concat(drops, axis=1).min(axis=1) < -0.5).mean())
    entry["quality_failed"] = " ".join(failed)
    entry["quality_ok"] = not failed and entry["harm_share"] <= RULE["max_harm_share"]
    return entry


SHOWN = ["name", "subset", "n", "changed_share", "tone_gain", "tone_gain_lo", "tone_gain_hi", "appropriateness_change",
         "relevance_change", "fluency_change", "naturalness_change", "preference", "preservation", "harm_share",
         "length_ratio", "hinglish_share_change"]


def causal_report(args) -> None:
    rows = paired_table("causal", ("judge1",))
    table = []
    subsets = {"all": rows.index == rows.index, "hinglish": rows.language == "hinglish", "english": rows.language == "english"}
    subsets.update({f"tone:{tone}": rows.tone == tone for tone in sorted(rows.tone.unique())})
    for subset_name, mask in subsets.items():
        for name, group in rows[mask].groupby("name"):
            if len(group) >= 5:
                table.append({"name": name, "subset": subset_name, **summarise(group, "judge1")})
    table = pd.DataFrame(table)
    table.to_csv(LAB / "causal" / "effects.csv", index=False)
    # Panel against matched random neurons at the same strength, paired by message.
    wide = rows.pivot(index="item_id", columns="name", values="d_tone_judge1")
    a = args.alphas
    tests = []
    for first, second in ((f"panel@{a[1]:g}", f"random0@{a[1]:g}"), (f"panel@{a[2]:g}", f"random0@{a[2]:g}"),
                          (f"panel@{a[2]:g}", f"random1@{a[2]:g}"), ("styled_clamp", "styled_clamp_random"),
                          (f"styled_reverse@{-a[1]:g}", f"styled_reverse_random@{-a[1]:g}")):
        if first in wide and second in wide:
            tests.append({"contrast": f"{first} - {second}", **paired_summary(wide[first] - wide[second])})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p)
    tests.to_csv(LAB / "causal" / "panel_vs_random.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    print(table[table.subset.isin(["all", "hinglish", "english"])][SHOWN].round(2).to_string(index=False))
    print(table[table.subset.str.startswith("tone:") & table.name.str.match(r"^(panel@|random0|reverse|styled_clamp$)")][SHOWN]
          .round(2).to_string(index=False))
    print(tests.round(4).to_string(index=False))


def failures(group: pd.DataFrame, source: str) -> dict:
    """Consistency of the tone effect and failure rates of one intervention over paired replies."""
    gain = group[f"d_tone_{source}"]
    english_switch = group.unsteered_is_hinglish & ~group.reply_is_hinglish.astype(bool)
    return {"consistency_up": float((gain > 0).mean()), "consistency_down": float((gain < 0).mean()),
            "degenerate_share": float(group.degenerate.astype(bool).mean()),
            "english_switch_share": float(english_switch.mean()),
            "over_length_share": float((group.length_ratio > 2.0).mean()),
            "repetition_share": float((group.d_repeated_bigram_rate > 0.1).mean())}


def agreement(stage: str, out: Path | None = None) -> pd.DataFrame:
    """Judge agreement on the replies of one stage: Spearman per rubric and same-winner share of preferences."""
    from scipy.stats import spearmanr
    out = out or LAB / stage
    table = pd.read_csv(out / "replies.csv")
    one, two = pd.read_csv(out / "ratings_judge1.csv"), pd.read_csv(out / "ratings_judge2.csv")
    rows = []
    for metric in ("tone", *RATED, "hinglish_naturalness"):
        if metric not in one or metric not in two:
            continue
        valid = one[metric].notna() & two[metric].notna()
        distinct = ~table.reply.duplicated()
        valid &= distinct
        if valid.sum() > 5:
            rows.append({"metric": metric, "replies": int(valid.sum()),
                         "spearman": float(spearmanr(one.loc[valid, metric], two.loc[valid, metric]).statistic),
                         "mean_abs_difference": float((one.loc[valid, metric] - two.loc[valid, metric]).abs().mean())})
    steered = table.reference.notna()
    if steered.sum() > 5:
        a, b = one.loc[steered, "preference"], two.loc[steered, "preference"]
        rows.append({"metric": "preference", "replies": int(steered.sum()), "spearman": float(spearmanr(a, b).statistic),
                     "same_winner_share": float(((a > 0.5) == (b > 0.5)).mean())})
    result = pd.DataFrame(rows)
    result.to_csv(out / "judge_agreement.csv", index=False)
    return result


CONTRASTS3 = [("panel", "sham"), ("panel", "random2"), ("panel", "other"), ("panel", "top200"), ("top200", "subset200"),
              ("top50", "subset50"), ("top10", "panel"), ("top10x", "panel"), ("direction", "direction_random0"),
              ("direction", "direction_random1"), ("direction", "direction_reverse"), ("direction", "panel"),
              ("styled_clamp", "styled_clamp_random"), ("styled_zero", "styled_clamp_random"),
              ("styled_clamp_late", "styled_clamp_random"), ("styled_clamp_top200", "styled_clamp_random")]


def causal3_report(args) -> None:
    judges = tuple(JUDGES)
    rows = paired_table("causal3", judges)
    a = args.alphas
    # Family names: the wrong-panel control carries the other tone's name, top10 at ten times the strength is "top10x".
    rows["family"] = [("other" if name.startswith("other_") else "top10x" if name == f"top10@{a[2] * 10:g}" else name.split("@")[0])
                      for name in rows.name]
    table = []
    subsets = {"all": rows.index == rows.index, "hinglish": rows.language == "hinglish", "english": rows.language == "english"}
    subsets.update({f"tone:{tone}": rows.tone == tone for tone in sorted(rows.tone.unique())})
    for subset_name, mask in subsets.items():
        for family, group in rows[mask].groupby("family"):
            if len(group) >= 5:
                for source in (*judges, "both"):
                    table.append({"name": family, "subset": subset_name, "source": source, **summarise(group, source),
                                  **failures(group, source)})
    table = pd.DataFrame(table)
    table.to_csv(LAB / "causal3" / "effects.csv", index=False)
    # Sham: the hooks with a zero shift must reproduce the unsteered reply.
    sham = rows[rows.family == "sham"]
    sham_identical = float((~sham.changed).mean()) if len(sham) else float("nan")
    tests = []
    for source in (*judges, "both"):
        wide = rows.pivot(index="item_id", columns="family", values=f"d_tone_{source}")
        quality = {metric: rows.pivot(index="item_id", columns="family", values=f"d_{metric}_{source}") for metric in QUALITY}
        for first, second in CONTRASTS3:
            if first in wide and second in wide:
                entry = {"contrast": f"{first} - {second}", "source": source, **paired_summary(wide[first] - wide[second])}
                for metric in ("appropriateness", "naturalness"):
                    entry[f"{metric}_difference"] = float((quality[metric][first] - quality[metric][second]).mean())
                tests.append(entry)
    tests = pd.DataFrame(tests)
    both = tests.source == "both"
    tests.loc[both, "p_holm"] = holm(tests.loc[both, "p"])
    tests.to_csv(LAB / "causal3" / "contrasts.csv", index=False)
    agree = agreement("causal3")
    summary = {"messages": int(rows.item_id.nunique()), "sham_identical_share": sham_identical,
               "families": sorted(rows.family.unique())}
    (LAB / "causal3" / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500)
    shown = SHOWN + ["consistency_up", "degenerate_share", "english_switch_share", "over_length_share"]
    print(f"sham replies identical to unsteered: {sham_identical:.2%}")
    print(table[(table.source == "both") & table.subset.isin(["all", "hinglish", "english"])][shown].round(2).to_string(index=False))
    print(table[(table.source == "both") & table.subset.str.startswith("tone:")
                & table.name.isin(["panel", "direction", "random2", "other", "top200", "styled_clamp"])][shown].round(2).to_string(index=False))
    print(tests[tests.source == "both"].round(4).to_string(index=False))
    print(agree.round(2).to_string(index=False))


def compare(args) -> None:
    """Neuron panel against the layer-10 direction under the same prompts, strengths and judges (calibration messages)."""
    judges = tuple(JUDGES)
    rows = paired_table("calibrate", judges)
    rows[["kind", "alpha"]] = [candidate_of(name) for name in rows.name]
    table = []
    for (kind, alpha), group in rows.groupby(["kind", "alpha"]):
        parts = {"all": group, **{language: part for language, part in group.groupby("language")},
                 **{f"tone:{tone}": part for tone, part in group.groupby("tone")}}
        for subset_name, part in parts.items():
            if len(part) < 5:
                continue
            for source in (*judges, "both"):
                table.append({"kind": kind, "alpha": float(alpha), "subset": subset_name, "source": source,
                              **summarise(part, source), **failures(part, source)})
    table = pd.DataFrame(table)
    table.to_csv(LAB / "calibrate" / "method_comparison.csv", index=False)
    # Paired contrast of the two methods on the same messages: panel at a[1] (the usual working strength) against
    # the direction at 0.2, and the matched "weak" pair (panel a[0], direction 0.1).
    a = args.alphas
    tests = []
    for first, second in ((f"tone@{a[1]:g}", "direction@0.2"), (f"tone@{a[0]:g}", "direction@0.1"), (f"tone@{a[2]:g}", "direction@0.3")):
        for language in ("all", "hinglish", "english"):
            part = rows if language == "all" else rows[rows.language == language]
            for source in (*judges, "both"):
                wide = part.pivot(index="item_id", columns="name", values=f"d_tone_{source}")
                if first in wide and second in wide:
                    entry = {"contrast": f"{first} - {second}", "language": language, "source": source,
                             **paired_summary(wide[first] - wide[second])}
                    for metric in QUALITY:
                        q = part.pivot(index="item_id", columns="name", values=f"d_{metric}_{source}")
                        entry[f"{metric}_difference"] = float((q[first] - q[second]).mean())
                    tests.append(entry)
    tests = pd.DataFrame(tests)
    tests.to_csv(LAB / "calibrate" / "method_contrasts.csv", index=False)
    agree = agreement("calibrate")
    pd.set_option("display.width", 300); pd.set_option("display.max_rows", 500)
    shown = ["kind", "alpha", "subset", "n", "tone_gain", "tone_gain_lo", "tone_gain_hi", "appropriateness_change",
             "naturalness_change", "preference", "harm_share", "consistency_up", "degenerate_share", "english_switch_share",
             "over_length_share", "length_ratio", "hinglish_share_change", "quality_ok"]
    print(table[(table.source == "both") & table.subset.isin(["all", "hinglish", "english"])][shown].round(2).to_string(index=False))
    print(table[(table.source == "both") & table.subset.str.startswith("tone:") & table.kind.isin(["tone", "direction"])][shown]
          .round(2).to_string(index=False))
    print(tests[tests.source == "both"].round(3).to_string(index=False))
    print(agree.round(2).to_string(index=False))


def candidate_of(name: str) -> tuple[str, float]:
    kind, alpha = name.split("@")
    return kind, float(alpha)


def select(args) -> None:
    if (LAB / "heldout" / "lock.json").exists():
        raise SystemExit("Held-out evaluation has started; policy reselection is forbidden.")
    judges = tuple(JUDGES)
    rows = paired_table("calibrate", judges)
    rows[["kind", "alpha"]] = [candidate_of(name) for name in rows.name]
    sweep = []
    cells = {f"strategy:{strategy}|{language}": cell for (strategy, language), cell in rows.groupby(["strategy", "language"])}
    for context, cell in cells.items():
        for kind, block in cell.groupby("kind"):
            entries = []
            for alpha in sorted(block.alpha.unique()):
                group = block[block.alpha == alpha]
                per = {source: summarise(group, source) for source in (*judges, "both")}
                safe = all(per[name]["quality_ok"] and per[name]["preference"] >= RULE["min_preference_each"] for name in judges)
                safe = safe and per["both"]["hinglish_share_change"] >= 0          # never trades Hinglish for English
                # The language panel alone is selected separately, as a repair for English replies (below).
                effective = kind != "language" and all(per[name]["tone_gain_lo"] > 0 for name in judges)
                helpful = per["both"]["preference"] >= RULE["min_preference_mean"] and per["both"]["appropriateness_change"] >= 0
                entries.append({"context": context, "kind": kind, "alpha": float(alpha), "n": per["both"]["n"],
                                "safe": bool(safe), "effective": bool(effective), "helpful": bool(helpful),
                                **{f"{key}_{source}": per[source][key] for source in (*judges, "both")
                                   for key in ("tone_gain", "tone_gain_lo", "tone_gain_hi", "empathy_change", "preference",
                                               "harm_share", "quality_failed",
                                               "appropriateness_change", "naturalness_change", "relevance_change",
                                               "fluency_change", "preservation")},
                                "hinglish_share_change": per["both"]["hinglish_share_change"],
                                "hinglish_naturalness_change": per["both"].get("hinglish_naturalness_change"),
                                "length_ratio": per["both"]["length_ratio"], "independent_gain": per["both"]["independent_gain"]})
            for index, entry in enumerate(entries):
                entry["admissible"] = bool(entry["safe"] and entry["effective"] and entry["helpful"] and entry["n"] >= RULE["min_n"]
                                           and index + 1 < len(entries) and entries[index + 1]["safe"])
            sweep += entries
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(LAB / "calibrate" / "sweep.csv", index=False)

    policy_cells, table = {}, []
    for context, block in sweep.groupby("context"):
        strategy = context.split(":")[1].split("|")[0]
        tone = STRATEGY_TONE[strategy]
        admissible = block[block.admissible]
        entry = {"alpha": 0.0, "panel": None, "target": STRATEGIES[strategy][3]}
        if len(admissible):
            best = admissible.sort_values(["preference_both", "tone_gain_both"], ascending=False).iloc[0]
            panel = {"tone": tone, "direction": f"{tone}:direction",
                     "direction+language": f"{tone}:direction+{LANGUAGE_PANEL}"}[best.kind]
            entry.update(alpha=float(best.alpha), panel=panel, reason=(
                f"both judges: quality-safe, tone gain +{best.tone_gain_both:.2f}, preferred over the unsteered reply "
                f"({best.preference_both:.2f}), Hinglish replies {best.hinglish_share_change:+.0%}, {int(best.n)} calibration messages"))
        else:
            blockers = [text for text, column in (("not quality-safe for both judges", "safe"),
                                                  ("no reliable gain for both judges", "effective"),
                                                  ("not preferred or less appropriate", "helpful")) if not block[column].any()]
            entry["reason"] = "not steered: " + ("; ".join(blockers) or "no strength passed every criterion with a safe stronger neighbour")
        # Sweep of the selected intervention (or the tone panel) in the form the Mood Tracker displays.
        shown = block[block.kind == (best.kind if len(admissible) else "tone")]
        entry["sweep"] = [{"alpha": float(r.alpha), "n": int(r.n), "emotion_rating_gain": round(r.tone_gain_both, 3),
                           "emotion_rating_gain_lo": round(min(r.tone_gain_lo_judge1, r.tone_gain_lo_judge2), 3),
                           "emotion_rating_gain_hi": round(max(r.tone_gain_hi_judge1, r.tone_gain_hi_judge2), 3),
                           "target_gain": round(r.independent_gain, 4), "independent_classifier_gain": round(r.independent_gain, 4),
                           "empathy_change": round(r.empathy_change_both, 3), "relevance_change": round(r.relevance_change_both, 3),
                           "fluency_change": round(r.fluency_change_both, 3), "harm_share": round(r.harm_share_both, 3),
                           "preference": round(r.preference_both, 3), "safe": bool(r.safe), "effective": bool(r.effective),
                           "admissible": bool(r.admissible), "hinglish_share_change": round(r.hinglish_share_change, 3)}
                          for r in shown.itertuples()]
        policy_cells[context] = entry
        table.append({"context": context, "n": int(block.n.max()), "panel": entry["panel"], "alpha": entry["alpha"],
                      "reason": entry["reason"]})
    # Language repair: Hinglish user, unsteered reply in English.  The candidate is sent only if it is in Hinglish.
    repair_rows = rows[(rows.kind == "language") & (rows.language == "hinglish") & ~rows.unsteered_is_hinglish]
    repair, entries = None, []
    for alpha in sorted(repair_rows.alpha.unique()):
        group = repair_rows[repair_rows.alpha == alpha]
        sent = group[group.reply_is_hinglish.astype(bool)]
        if len(sent) < RULE["min_n"]:
            entries.append({"alpha": float(alpha), "english_replies": int(len(group)), "sent": int(len(sent)), "safe": False,
                            "helpful": False})
            continue
        per = {source: summarise(sent, source) for source in (*judges, "both")}
        natural = paired_summary(sent["d_hinglish_naturalness_both"])
        safe = all(per[name]["quality_ok"] and per[name]["preference"] >= RULE["min_preference_each"] for name in judges)
        helpful = natural["ci_low"] > 0 and per["both"]["preference"] >= RULE["min_preference_mean"]
        entries.append({"alpha": float(alpha), "english_replies": int(len(group)), "sent": int(len(sent)),
                        "sent_share": float(len(sent) / len(group)), "safe": bool(safe), "helpful": bool(helpful),
                        "hinglish_naturalness_gain": natural["mean"], "hinglish_naturalness_gain_lo": natural["ci_low"],
                        **{f"{key}_{source}": per[source][key] for source in (*judges, "both")
                           for key in ("preference", "harm_share", "quality_failed", "appropriateness_change",
                                       "naturalness_change", "relevance_change", "fluency_change", "empathy_change")},
                        "preservation": per["both"]["preservation"], "length_ratio": per["both"]["length_ratio"]})
    for position, entry in enumerate(entries):
        entry["admissible"] = bool(entry["safe"] and entry["helpful"] and position + 1 < len(entries) and entries[position + 1]["safe"])
    pd.DataFrame(entries).to_csv(LAB / "calibrate" / "language_repair.csv", index=False)
    chosen = [entry for entry in entries if entry["admissible"]]
    if chosen:
        repair = max(chosen, key=lambda entry: entry["preference_both"])
        for context, entry in policy_cells.items():
            if context.endswith("|hinglish") and not entry["alpha"]:
                entry.update(alpha=repair["alpha"], panel=LANGUAGE_PANEL, reason=(
                    f"no tone steering; language panel used only when the unsteered reply is in English "
                    f"(Hinglish naturalness +{repair['hinglish_naturalness_gain']:.2f}, preferred {repair['preference_both']:.2f}, "
                    f"{repair['sent']} calibration replies)"))
    for emotion in ("love", "admiration", "joy"):                       # contexts without their own cell are not steered
        for language in ("english", "hinglish"):
            policy_cells[f"{emotion}|{language}"] = {"alpha": 0.0, "panel": None, "sweep": [],
                                                    "reason": "not steered: no calibration for this context"}
    table = pd.DataFrame(table)
    table.to_csv(LAB / "calibrate" / "selection.csv", index=False)
    summary = []
    for (kind, alpha), group in rows.groupby(["kind", "alpha"]):
        for language, part in group.groupby("language"):
            both = summarise(part, "both")
            summary.append({"kind": kind, "alpha": alpha, "language": language, **{k: v for k, v in both.items() if k != "quality_failed"}})
    summary = pd.DataFrame(summary)
    summary.to_csv(LAB / "calibrate" / "intervention_summary.csv", index=False)
    register_composites(policy_cells, args.combo_language)
    (LAB / "policy_candidate.json").write_text(json.dumps({
        "source": "generation_analysis.sarvam_experiments", "version": "sarvam-1", "model": "sarvamai/sarvam-30b",
        "panel": PANEL_FILE.name, "split": "calibration (Sarvam)", "rule": RULE,
        "selection": "per strategy and language: both judges quality-safe (appropriateness, empathy, relevance, fluency, "
                     "safety, naturalness; Hinglish naturalness for Hinglish), no loss of Hinglish replies, reliable tone "
                     "gain for both judges (language panel: more Hinglish replies), preferred over the unsteered reply "
                     "without loss of appropriateness, next stronger strength safe; among those, the most preferred",
        "gate": {**GATE, "language_panel": LANGUAGE_PANEL}, "language_repair": repair,
        "direction_layer": DIRECTION_LAYER, "combo_language": args.combo_language, "cells": policy_cells}, indent=1))
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500); pd.set_option("display.max_colwidth", 110)
    print(table.to_string(index=False))
    print(pd.DataFrame(entries).round(3).to_string(index=False))
    print(summary[["kind", "alpha", "language", "n", "tone_gain", "tone_gain_lo", "preference", "preservation",
                   "appropriateness_change", "relevance_change", "fluency_change", "naturalness_change", "harm_share",
                   "length_ratio", "hinglish_reply_share", "hinglish_share_change", "quality_ok"]].round(2).to_string(index=False))


def register_composites(cells: dict, language_alpha: float) -> None:
    """Write the composite panels a policy uses into the panel file the chat backend loads.

    ``<tone>:direction``            residual direction of the tone after DIRECTION_LAYER
    ``<tone>:direction+hinglish``   the same direction plus the Hinglish language neurons at their fixed strength
    ``<tone>+hinglish``             tone neurons plus language neurons
    Language shifts are stored per unit of the cell's alpha, so the runtime needs one strength only.
    """
    payload = json.loads(PANEL_FILE.read_text())
    panels = load_panels()
    directions = dict(np.load(LAB / "residual_directions.npz"))
    payload.setdefault("directions", {})
    for cell in cells.values():
        name = cell.get("panel")
        if not name or not cell["alpha"] or name in panels:
            continue
        tone = name.split(":")[0].split("+")[0]
        language = {layer: (neurons, [delta * language_alpha / cell["alpha"] for delta in deltas])
                    for layer, (neurons, deltas) in panels[LANGUAGE_PANEL].items()}
        neuron_part: Panel = {}
        if ":direction" in name:
            payload["directions"][name] = {"layer": DIRECTION_LAYER, "unit": [float(v) for v in directions[tone][DIRECTION_LAYER]]}
            neuron_part = language if name.endswith(f"+{LANGUAGE_PANEL}") else {}
        elif name.endswith(f"+{LANGUAGE_PANEL}"):
            neuron_part = combined(panels[tone], language)
        if neuron_part:
            payload["panels"][name] = {str(layer): {"neurons": neurons, "deltas": deltas}
                                       for layer, (neurons, deltas) in neuron_part.items()}
    PANEL_FILE.write_text(json.dumps(payload))


# ----------------------------------------------------------------------- held-out

HELD = LAB / "heldout"
SYSTEMS = ("baseline", "kg", "steered")
PAIRS = (("kg", "baseline"), ("steered", "kg"), ("steered", "baseline"))
# Language-panel validation arm (Hinglish users only, not part of the policy): the KG prompt with the Hinglish
# panel at the calibrated repair strength, decoded in one batch with its own unsteered reference ("kg_batch").
LANGUAGE_SYSTEMS = ("kg_batch", "language")
LANGUAGE_PAIRS = (("language", "kg_batch"),)


def heldout_generate(args) -> None:
    """Untouched messages through the real steering service with the candidate policy (as the chatbot would)."""
    from backend.emotion_service import EmotionSteeringService
    from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
    from chatbot_feature.emotion_reasoner.agent import DETECTOR_WEIGHT
    from .freeze_v2 import sha256
    HELD.mkdir(parents=True, exist_ok=True)
    policy_path = LAB / "policy_candidate.json"
    lock = {"policy": sha256(policy_path), "panels": sha256(PANEL_FILE), "items": sha256(LAB / "data" / "heldout.csv")}
    if (HELD / "lock.json").exists() and json.loads((HELD / "lock.json").read_text()) != lock:
        raise SystemExit("Policy, panels or messages changed after the held-out generation started.")
    (HELD / "lock.json").write_text(json.dumps(lock, indent=1))
    path = HELD / "generations.jsonl"
    done = {json.loads(line)["item_id"] for line in path.open()} if path.exists() else set()
    runtime = load_runtime()                                  # loads the panel file, including composite panels
    policy = json.loads(policy_path.read_text())
    agent = EmotionReasonerAgent(available_panels=list(runtime._emotion_panels), policy=policy, detector_weight=DETECTOR_WEIGHT)
    service = EmotionSteeringService(runtime, agent=agent, evidence={})
    with path.open("a", encoding="utf-8") as handle:
        for row in load_items("heldout").itertuples():
            if row.item_id in done:
                continue
            messages = messages_of(row)
            baseline = runtime.generate(messages, max_new_tokens=args.max_new_tokens, deterministic=True, temperature=0.7,
                                        system_prompt=SYSTEM_PROMPT)["response"]
            turn_id, plan = service.begin_turn(messages)
            if plan["strategy"] != row.strategy:
                raise SystemExit(f"Plan mismatch for {row.item_id}.")
            analysis = service.respond(turn_id, max_new_tokens=args.max_new_tokens)
            layers = analysis["steering"].get("layers") or []
            record = {"item_id": row.item_id, "baseline": baseline, "kg": analysis["effect"]["unsteered_reply"],
                      "steered": analysis["reply"], "candidate": analysis["effect"]["steered_reply"],
                      "status": analysis["decision"]["status"], "reason": analysis["decision"]["reason"],
                      "planned_alpha": analysis["decision"]["planned_alpha"], "applied_alpha": analysis["decision"]["applied_alpha"],
                      "panel": plan.get("steering_panel"), "steered_layers": len(layers),
                      "steered_neurons": analysis["steering"].get("neurons_total", 0),
                      "mean_abs_measured_change": float(np.mean([layer["mean_abs_measured_change"] for layer in layers
                                                                 if layer.get("mean_abs_measured_change") is not None]))
                      if layers else None}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print("heldout", row.item_id, row.strategy, record["status"], record["panel"], record["applied_alpha"], flush=True)
    del service, runtime
    release()


def heldout_score(args) -> None:
    items = load_items("heldout").set_index("item_id")
    rows = []
    for line in (HELD / "generations.jsonl").open():
        record = json.loads(line)
        item = items.loc[record["item_id"]]
        for system in SYSTEMS:
            steered = system == "steered" and record["applied_alpha"] > 0
            rows.append({"item_id": record["item_id"], "system": system, "reply": record[system], "text": item.text,
                         "history": item.history, "target": item.target, "strategy": item.strategy, "tone": item.tone,
                         "language": item.language, "status": record["status"] if system == "steered" else "none",
                         "applied_alpha": record["applied_alpha"] if system == "steered" else 0.0,
                         "panel": record["panel"] if steered else None, "reference": record["kg"] if steered else None})
    table = pd.DataFrame(rows)
    if table.item_id.nunique() != len(items):
        raise SystemExit("Held-out generation is incomplete.")
    table = automatic_scores(table)
    table.drop(columns=["history"]).to_csv(HELD / "replies.csv", index=False)
    wide = table.pivot(index="item_id", columns="system", values="reply")
    first = table.drop_duplicates("item_id").set_index("item_id").loc[wide.index].reset_index()
    for name, (model_id, four_bit) in JUDGES.items():
        if (HELD / f"ratings_{name}.csv").exists():
            continue
        judge = Judge(model_id, load_in_4bit=four_bit)
        ratings = judge_ratings(judge, table, reference=[value if isinstance(value, str) else None for value in table.reference])
        contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
        keys = list(zip(contexts, table.reply.fillna(""), [tone_rubric(tone) for tone in table.tone]))
        unique = list(dict.fromkeys(keys))
        lookup = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings["tone"] = [float(lookup[key]) for key in keys]
        preferences = pd.DataFrame({f"{a}_over_{b}": judge_preferences(judge, first, wide[a].tolist(), wide[b].tolist())
                                    for a, b in PAIRS}, index=wide.index)
        ratings.to_csv(HELD / f"ratings_{name}.csv", index=False)
        preferences.to_csv(HELD / f"preferences_{name}.csv")
        judge.close()


def heldout_report(args) -> None:
    from scipy.stats import spearmanr
    table = pd.read_csv(HELD / "replies.csv")
    metrics = ("tone", *RATED, "hinglish_naturalness")
    preferences = {}
    for name in JUDGES:
        ratings = pd.read_csv(HELD / f"ratings_{name}.csv")
        for column in (*metrics, "preservation"):
            table[f"{column}_{name}"] = ratings[column].to_numpy() if column in ratings else np.nan
        preferences[name] = pd.read_csv(HELD / f"preferences_{name}.csv", index_col=0)
    for column in (*metrics, "preservation"):
        table[f"{column}_both"] = table[[f"{column}_{name}" for name in JUDGES]].mean(axis=1)
    preferences["both"] = sum(preferences[name] for name in JUDGES) / len(JUDGES)
    wide = {name: group.set_index("item_id") for name, group in table.groupby("system")}
    index = wide["baseline"].index
    subsets = {"all": index, "hinglish": index[wide["baseline"].language == "hinglish"],
               "english": index[wide["baseline"].language == "english"], "steered": index[wide["steered"].applied_alpha > 0]}
    automatic = ("indep_target", "reply_words", "repeated_bigram_rate", "reply_is_hinglish", "degenerate")
    means = pd.DataFrame([{"subset": subset, "system": name, "n": len(ids),
                           **{metric: wide[name].loc[ids, f"{metric}_both"].mean() for metric in metrics},
                           **{column: wide[name].loc[ids, column].astype(float).mean() for column in automatic},
                           "steered_share": float((wide[name].loc[ids, "applied_alpha"] > 0).mean())}
                          for subset, ids in subsets.items() if len(ids) for name in SYSTEMS])
    means.to_csv(HELD / "system_means.csv", index=False)
    rows = []
    for subset, ids in subsets.items():
        if len(ids) < 6:
            continue
        for a, b in PAIRS:
            for source in (*JUDGES, "both"):
                for metric in metrics:
                    x, y = wide[a].loc[ids, f"{metric}_{source}"], wide[b].loc[ids, f"{metric}_{source}"]
                    if x.notna().sum() >= 6:
                        rows.append({"subset": subset, "contrast": f"{a} - {b}", "metric": metric, "source": source,
                                     **paired_summary(x.to_numpy() - y.to_numpy())})
                rows.append({"subset": subset, "contrast": f"{a} - {b}", "metric": "preference_minus_0.5", "source": source,
                             **paired_summary(preferences[source].loc[ids, f"{a}_over_{b}"] - 0.5)})
            for metric in ("indep_target", "reply_is_hinglish", "reply_words"):
                rows.append({"subset": subset, "contrast": f"{a} - {b}", "metric": metric, "source": "automatic",
                             **paired_summary(wide[a].loc[ids, metric].astype(float).to_numpy() - wide[b].loc[ids, metric].astype(float).to_numpy())})
    contrasts = pd.DataFrame(rows)
    primary = (contrasts.subset.isin(["all", "hinglish"]) & (contrasts.source == "both") & (contrasts.contrast == "steered - kg"))
    contrasts.loc[primary, "p_holm"] = holm(contrasts.loc[primary, "p"])
    contrasts.to_csv(HELD / "contrasts.csv", index=False)
    steered = subsets["steered"]
    safety = {"messages": int(len(index)), "decisions": wide["steered"].status.value_counts().to_dict(),
              "steered": int(len(steered)), "steered_share": float(len(steered) / len(index)),
              "steered_by_strategy": wide["steered"].loc[steered, "strategy"].value_counts().to_dict(),
              "steered_by_language": wide["steered"].loc[steered, "language"].value_counts().to_dict(),
              "panels": wide["steered"].loc[steered, "panel"].value_counts().to_dict()}
    for source in (*JUDGES, "both"):
        if len(steered):
            drop = pd.concat([wide["steered"].loc[steered, f"{m}_{source}"] - wide["kg"].loc[steered, f"{m}_{source}"]
                              for m in QUALITY], axis=1).min(axis=1)
            safety[f"harmed_share_{source}"] = float((drop < -0.5).mean())
    for name in SYSTEMS:
        safety[f"safety_below_3_share_{name}"] = float((wide[name].safety_both < 3).mean())
    if len(steered):
        safety["preservation"] = float(wide["steered"].loc[steered, "preservation_both"].mean())
        safety["hinglish_kept"] = float((wide["steered"].loc[steered, "reply_is_hinglish"].astype(bool)
                                         >= wide["kg"].loc[steered, "reply_is_hinglish"].astype(bool)).mean())
    distinct = table.drop_duplicates(["item_id", "reply"])
    agreement = []
    for metric in metrics:
        valid = distinct[f"{metric}_judge1"].notna() & distinct[f"{metric}_judge2"].notna()
        if valid.sum() > 5:
            agreement.append({"metric": metric, "replies": int(valid.sum()),
                              "spearman": float(spearmanr(distinct.loc[valid, f"{metric}_judge1"], distinct.loc[valid, f"{metric}_judge2"]).statistic)})
    for column in preferences["judge1"]:
        differs = preferences["judge1"][column] != 0.5
        if differs.sum() > 5:
            a, b = preferences["judge1"].loc[differs, column], preferences["judge2"].loc[differs, column]
            agreement.append({"metric": f"preference {column}", "replies": int(differs.sum()),
                              "spearman": float(spearmanr(a, b).statistic), "same_winner_share": float(((a > 0.5) == (b > 0.5)).mean())})
    pd.DataFrame(agreement).to_csv(HELD / "judge_agreement.csv", index=False)
    (HELD / "summary.json").write_text(json.dumps(safety, indent=1))
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
    print(means.round(3).to_string(index=False))
    print(json.dumps(safety, indent=1))
    view = contrasts[contrasts.source.isin(["both", "automatic"])]
    view = view.assign(value=view.apply(lambda r: f"{r['mean']:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]", axis=1))
    print(view.pivot_table(index=["subset", "metric"], columns="contrast", values="value", aggfunc="first").to_string())
    print(contrasts[primary][["subset", "metric", "n", "mean", "ci_low", "ci_high", "p", "p_holm"]].round(4).to_string(index=False))
    print(pd.DataFrame(agreement).round(2).to_string(index=False))


# --------------------------------------------- held-out check of the language panel

HELD_LANGUAGE = LAB / "heldout_language"
LANGUAGE_CHECK_ALPHA = 1.0          # the strength behind the development claim (65% -> 91% Hinglish replies)


def heldout_language_generate(args) -> None:
    """Untouched Hinglish messages: KG prompt alone and with the Hinglish language panel, decoded in one batch.

    This validates the language-panel claim on held-out data.  It is not a policy arm: the policy, the
    policy held-out files and their lock are not touched, and nothing here feeds back into selection."""
    from .freeze_v2 import sha256
    HELD_LANGUAGE.mkdir(parents=True, exist_ok=True)
    lock = {"panels": sha256(PANEL_FILE), "items": sha256(LAB / "data" / "heldout.csv"), "alpha": LANGUAGE_CHECK_ALPHA}
    if (HELD_LANGUAGE / "lock.json").exists() and json.loads((HELD_LANGUAGE / "lock.json").read_text()) != lock:
        raise SystemExit("Panels, messages or strength changed after the language check started.")
    (HELD_LANGUAGE / "lock.json").write_text(json.dumps(lock, indent=1))
    path = HELD_LANGUAGE / "generations.jsonl"
    done = {json.loads(line)["item_id"] for line in path.open()} if path.exists() else set()
    runtime = load_runtime()
    language = load_panels()[LANGUAGE_PANEL]
    items = load_items("heldout")
    items = items[items.language == "hinglish"]
    with path.open("a", encoding="utf-8") as handle:
        for row in items.itertuples():
            if row.item_id in done:
                continue
            rows = [("none", {}), (f"language@{LANGUAGE_CHECK_ALPHA:g}", {"shift": [(language, LANGUAGE_CHECK_ALPHA)]})]
            replies = generate_rows(runtime, messages_of(row), kg_prompt(row), [spec for _, spec in rows], args.max_new_tokens)
            handle.write(json.dumps({"item_id": row.item_id, "replies": [{"name": name, "reply": reply}
                                                                           for (name, _), reply in zip(rows, replies)]},
                                    ensure_ascii=False) + "\n")
            handle.flush()
            print("heldout_language", row.item_id, flush=True)
    del runtime
    release()


def heldout_language_score(args) -> None:
    table = long_table("heldout_language", "heldout")
    expected = (load_items("heldout").language == "hinglish").sum()
    if table.item_id.nunique() != expected:
        raise SystemExit(f"Language check incomplete: {table.item_id.nunique()} of {expected} Hinglish messages.")
    score("heldout_language", "heldout", tuple(JUDGES))


def heldout_language_report(args) -> None:
    """Hinglish-reply share with CIs, language and quality ratings of both judges, and the repair subset
    (unsteered reply in English, steered reply in Hinglish), on the untouched held-out Hinglish messages."""
    judges = tuple(JUDGES)
    rows = paired_table("heldout_language", judges)
    report = []
    subsets = {"all hinglish users": rows, "unsteered reply in English": rows[~rows.unsteered_is_hinglish],
               "repair (English -> Hinglish)": rows[~rows.unsteered_is_hinglish & rows.reply_is_hinglish.astype(bool)],
               "unsteered reply already Hinglish": rows[rows.unsteered_is_hinglish]}
    for subset_name, group in subsets.items():
        if len(group) < 5:
            continue
        share = paired_summary(group.d_is_hinglish)
        entry = {"subset": subset_name, "n": int(len(group)), "unsteered_hinglish_share": float(group.unsteered_is_hinglish.mean()),
                 "steered_hinglish_share": float(group.reply_is_hinglish.astype(float).mean()),
                 "hinglish_share_change": share["mean"], "hinglish_share_change_lo": share["ci_low"],
                 "hinglish_share_change_hi": share["ci_high"]}
        for source in (*judges, "both"):
            per = summarise(group, source)
            natural = paired_summary(group[f"d_hinglish_naturalness_{source}"])
            entry.update({f"{key}_{source}": per[key] for key in ("preference", "harm_share", "quality_failed", "quality_ok",
                                                                  "appropriateness_change", "appropriateness_change_lo",
                                                                  "empathy_change", "empathy_change_lo", "relevance_change",
                                                                  "relevance_change_lo", "fluency_change", "fluency_change_lo",
                                                                  "naturalness_change", "naturalness_change_lo", "preservation")})
            entry[f"hinglish_naturalness_change_{source}"] = natural["mean"]
            entry[f"hinglish_naturalness_change_lo_{source}"] = natural["ci_low"]
            entry[f"hinglish_naturalness_change_hi_{source}"] = natural["ci_high"]
        entry.update(failures(group, "both"))
        entry["length_ratio"] = float(group.length_ratio.mean())
        report.append(entry)
    report = pd.DataFrame(report)
    report.to_csv(HELD_LANGUAGE / "report.csv", index=False)
    agree = agreement("heldout_language")
    pd.set_option("display.width", 320); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 60)
    shown = ["subset", "n", "unsteered_hinglish_share", "steered_hinglish_share", "hinglish_share_change", "hinglish_share_change_lo",
             "hinglish_share_change_hi", "hinglish_naturalness_change_both", "hinglish_naturalness_change_lo_both",
             "appropriateness_change_both", "empathy_change_both", "relevance_change_both", "fluency_change_both",
             "naturalness_change_both", "preference_both", "harm_share_judge1", "harm_share_judge2", "quality_ok_judge1",
             "quality_ok_judge2", "degenerate_share", "over_length_share", "length_ratio"]
    print(report[shown].round(3).T.to_string())
    print(agree.round(2).to_string(index=False))


def causal_agreement(args) -> None:
    """Second-judge ratings of the first two development tests, for judge agreement on those results."""
    for stage in ("causal", "causal2"):
        score(stage, "dev", tuple(JUDGES))
        print(stage)
        print(agreement(stage).round(2).to_string(index=False))


def smoke(args) -> None:
    """Real-model check of the chat runtime's steered generation and its Mood Tracker trace (nothing installed).

    Runs ``SarvamRuntime.generate_candidates`` for a neuron panel, the layer-10 direction and the direction plus
    language neurons on two development messages, and prints the replies and the measured per-layer trace."""
    runtime = load_runtime()
    directions = dict(np.load(LAB / "residual_directions.npz"))
    runtime._emotion_directions = {f"{tone}:direction": (DIRECTION_LAYER, [float(v) for v in directions[tone][DIRECTION_LAYER]])
                                   for tone in TONES}
    language = load_panels()[LANGUAGE_PANEL]
    runtime._emotion_panels["warmth:direction+hinglish"] = {layer: (neurons, [delta * 0.75 / 0.2 for delta in deltas])
                                                            for layer, (neurons, deltas) in language.items()}
    runtime._emotion_directions["warmth:direction+hinglish"] = runtime._emotion_directions["warmth:direction"]
    items = load_items("dev")
    picks = pd.concat([items[(items.language == "hinglish") & (items.tone == "warmth")].head(1),
                       items[(items.language == "english") & (items.tone == "joy")].head(1)])
    checks = []
    for row in picks.itertuples():
        for panel, alpha in ((row.tone, 2.0), (f"{row.tone}:direction", 0.2), ("warmth:direction+hinglish", 0.2)):
            if panel == "warmth:direction+hinglish" and row.language != "hinglish":
                continue
            results = runtime.generate_candidates(messages_of(row), args.max_new_tokens, kg_prompt(row), panel, [0.0, alpha])
            base, steered = results[0], results[1]
            trace = steered["trace"]
            layers = trace["layers"]
            print(f"\n### [{row.language}/{row.tone}] panel={panel} alpha={alpha}\n  user: {row.text[:160]}")
            print(f"  unsteered: {base['response'][:260]!r}\n  steered:   {steered['response'][:260]!r}")
            print(f"  location: {trace['location'][:160]}")
            print(f"  layers: {len(layers)}  neurons: {sum(len(l['neurons']) for l in layers)}  "
                  f"sites: {sorted({l.get('site', 'mlp') for l in layers})}")
            for layer in layers[:2] + layers[-1:]:
                print(f"    L{layer['layer']:<3d} {layer.get('site', 'mlp input'):16s} neurons={len(layer['neurons'])} "
                      f"decode_positions={layer['decode_positions']} first shift={layer['applied_shift'][:3]} "
                      f"before={layer['mean_activation_before_shift'][:3]}")
            checks.append({"item_id": row.item_id, "panel": panel, "alpha": alpha, "changed": base["response"] != steered["response"],
                           "layers": len(layers), "neurons": sum(len(l["neurons"]) for l in layers),
                           "decode_positions": max((l["decode_positions"] for l in layers), default=0),
                           "unsteered": base["response"], "steered": steered["response"]})
    (LAB / "smoke.json").write_text(json.dumps(checks, indent=1, ensure_ascii=False))
    print(f"\nwrote {LAB / 'smoke.json'}: {sum(c['changed'] for c in checks)} of {len(checks)} steered replies differ from unsteered")
    del runtime
    release()


# ------------------------------------------------------------ evidence and install

def _number(value, digits: int = 3):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def language_verdict(policy: dict) -> str:
    """What the calibration and, if run, the held-out check say about the Hinglish language panel."""
    text = ("These neurons control the reply language, not its emotion: on development messages strength 1 raised the share "
            "of Hinglish replies from 65% to 91% (one judge).")
    repair = LAB / "calibrate" / "language_repair.csv"
    if repair.exists():
        table = pd.read_csv(repair)
        row = table.iloc[(table.alpha - 1.0).abs().argmin()]
        text += (f" Calibration (both judges, {int(row.english_replies)} English replies to Hinglish users): strength {row.alpha:g} "
                 f"turned {row.sent_share:.0%} of them into Hinglish, but {row.harm_share_judge1:.0%} / {row.harm_share_judge2:.0%} "
                 f"of those replies lost at least half a point on some quality rating and judges preferred them only "
                 f"{row.preference_both:.2f} of the time, so the panel is not used.")
    held = HELD_LANGUAGE / "report.csv"
    if held.exists():
        table = pd.read_csv(held).set_index("subset")
        if "all hinglish users" in table.index:
            row = table.loc["all hinglish users"]
            text += (f" Held-out Hinglish messages (n={int(row.n)}): Hinglish replies {row.unsteered_hinglish_share:.0%} -> "
                     f"{row.steered_hinglish_share:.0%} ({row.hinglish_share_change:+.2f}, CI {row.hinglish_share_change_lo:+.2f} to "
                     f"{row.hinglish_share_change_hi:+.2f}); appropriateness {row.appropriateness_change_both:+.2f}, naturalness "
                     f"{row.naturalness_change_both:+.2f}, preference {row.preference_both:.2f}.")
    used = LANGUAGE_PANEL in {cell.get("panel") for cell in policy["cells"].values()}
    return text + (" Used only when the unsteered reply is in English." if used else " Not used by the installed policy.")


def export_evidence(args) -> dict:
    """Causal (development) and held-out results in the form the Mood Tracker shows, keyed by panel name."""
    effects = pd.read_csv(LAB / "causal" / "effects.csv")
    second = pd.read_csv(LAB / "causal2" / "effects.csv") if (LAB / "causal2" / "effects.csv").exists() else pd.DataFrame()
    third = pd.read_csv(LAB / "causal3" / "effects.csv") if (LAB / "causal3" / "effects.csv").exists() else pd.DataFrame()
    summary = pd.read_csv(LAB / "panel_summary.csv")
    policy = json.loads((LAB / "policy_candidate.json").read_text())

    def effect(row, label: str) -> dict:
        alpha = float(row["name"].split("@")[1]) if "@" in row["name"] else 0.0
        return {"label": label, "alpha": alpha, "neurons": int(summary[summary.panel == row.get("tone", "warmth")].selected.sum()) or 1328,
                "emotion_rating": _number(row.tone_gain), "lo": _number(row.tone_gain_lo), "hi": _number(row.tone_gain_hi),
                "fluency": _number(row.fluency_change)}
    panels = {}
    for tone in TONES:
        block = effects[effects.subset == f"tone:{tone}"].assign(tone=tone)
        if block.empty:
            continue
        rows = {row["name"]: row for _, row in block.iterrows()}
        dose = sorted((effect(row, f"Tone neurons, α {name.split('@')[1]}") for name, row in rows.items() if name.startswith("panel@")),
                      key=lambda item: item["alpha"])
        controls = [effect(row, f"Random neurons, same shifts, α {name.split('@')[1]}") for name, row in rows.items()
                    if name.startswith("random")]
        labels = {"styled_clamp": "Tone neurons held at their neutral mean (prompt asks for the tone)",
                  "styled_clamp_random": "Random neurons held at their neutral mean (control)"}
        ablation = [effect(rows[name], text) for name, text in labels.items() if name in rows]
        ablation += [effect(row, f"Tone neurons shifted the opposite way, α {name.split('@')[1]}") for name, row in rows.items()
                     if name.startswith("reverse@") or name.startswith("styled_reverse@")]
        subsets = [effect(row, f"Layers {name.split('_')[1].split('@')[0]} third only, α {name.split('@')[1]}")
                   for name, row in rows.items() if name.startswith("band_")]
        best = max((item for item in dose if item["fluency"] is None or item["fluency"] >= -0.1),
                   key=lambda item: item["emotion_rating"], default=None)
        control = max(controls, key=lambda item: item["emotion_rating"], default=None)
        verdict = "No tested strength raised this tone without lowering fluency."
        if best and control:
            reliable = best["lo"] is not None and best["lo"] > 0
            verdict = (f"Shifting these Sarvam neurons {'raised' if reliable else 'did not reliably raise'} the {tone} of replies on "
                       f"development messages (one judge): {best['emotion_rating']:+.2f} (95% CI {best['lo']:+.2f} to {best['hi']:+.2f}) "
                       f"at α {best['alpha']:g}; the same shifts on random neurons gave {control['emotion_rating']:+.2f} and damaged "
                       "the reply.")
        # Third development test, both judges: controls that separate the panel from the hook mechanics, from other
        # tones and from its own parts, and the ablations.
        if not third.empty:
            block3 = third[(third.subset == f"tone:{tone}") & (third.source == "both")].set_index("name")
            if "panel" in block3.index:
                def entry3(name: str, label: str) -> dict | None:
                    if name not in block3.index:
                        return None
                    r = block3.loc[name]
                    return {"label": label, "alpha": float(args.alphas[2]) if name not in ("direction", "direction_random0", "direction_reverse") else 0.2,
                            "neurons": {"top10": 10, "top50": 50, "top200": 200, "subset50": 50, "subset200": 200}.get(name, int(summary[summary.panel == tone].selected.sum())),
                            "emotion_rating": _number(r.tone_gain), "lo": _number(r.tone_gain_lo), "hi": _number(r.tone_gain_hi),
                            "fluency": _number(r.fluency_change)}
                controls += [item for item in (entry3("sham", "Hooks installed with a zero shift (sham)"),
                                               entry3("random2", "Random neurons, same shifts (third draw)"),
                                               entry3("other", "Another tone's panel, same strength")) if item]
                ablation += [item for item in (entry3("styled_zero", "Tone neurons set to zero (prompt asks for the tone)"),
                                               entry3("styled_clamp_late", "Late-layer tone neurons held at neutral"),
                                               entry3("styled_clamp_top200", "Top-200 tone neurons held at neutral")) if item]
                subsets += [item for item in (entry3("top10", "Top 10 neurons by attribution"), entry3("top50", "Top 50 neurons"),
                                              entry3("top200", "Top 200 neurons"), entry3("subset200", "Random 200 of the panel"),
                                              entry3("direction", "Layer-10 residual direction, α 0.2"),
                                              entry3("direction_random0", "Random residual direction, α 0.2"),
                                              entry3("direction_reverse", "Reversed residual direction, α -0.2")) if item]
                panel3 = block3.loc["panel"]
                verdict += (f" Two-judge replication: the whole panel {panel3.tone_gain:+.2f} (CI {panel3.tone_gain_lo:+.2f} to "
                            f"{panel3.tone_gain_hi:+.2f}), a sham hook 0.00, another tone's panel "
                            f"{block3.loc['other'].tone_gain:+.2f}" if "other" in block3.index else "")
                verdict += (". The effect is a property of the whole panel: its top 200 neurons by attribution gave "
                            f"{block3.loc['top200'].tone_gain:+.2f} and did no better than 200 random panel neurons; holding the "
                            "neurons at neutral or zero under a tone-requesting prompt did not lower the tone. So these neurons "
                            "are jointly sufficient to raise the tone but not necessary, and no individual neuron encodes it."
                            if "top200" in block3.index else "")
        layers = sorted(int(layer) for layer in summary[(summary.panel == tone) & (summary.selected > 0)].layer)
        entry = {"emotion": tone, "name": "Sarvam shared-expert tone neurons (top 5% per site)",
                 "selection": "paired |t| >= 3 over 170 matched prompts (120 Hinglish), then top attribution per layer",
                 "neurons": int(summary[summary.panel == tone].selected.sum()), "layers": layers, "verdict": verdict,
                 "likelihood": None, "dose_response": dose, "controls": controls, "ablation": ablation, "layer_effects": [],
                 "subsets": subsets}
        panels[tone] = entry
        for name in {cell.get("panel") for cell in policy["cells"].values() if cell.get("panel")}:
            if name.split(":")[0].split("+")[0] == tone and name != tone:
                panels[name] = {**entry, "emotion": name}
    if True:                       # the language panel is described whether or not the policy uses it
        block = effects[(effects.subset == "hinglish") & effects.name.str.startswith("language@")]
        dose = [{"label": f"Language neurons, α {row['name'].split('@')[1]} (Hinglish replies {row.hinglish_share_change:+.0%})",
                 "alpha": float(row["name"].split("@")[1]), "neurons": int(summary[summary.panel == LANGUAGE_PANEL].selected.sum()),
                 "emotion_rating": _number(row.tone_gain), "lo": _number(row.tone_gain_lo), "hi": _number(row.tone_gain_hi),
                 "fluency": _number(row.fluency_change)} for _, row in block.iterrows()]
        panels[LANGUAGE_PANEL] = {
            "emotion": LANGUAGE_PANEL, "name": "Sarvam Hinglish language neurons", "neurons": int(summary[summary.panel == LANGUAGE_PANEL].selected.sum()),
            "selection": "neutral reply in Hinglish minus the same reply in English, 75 matched pairs",
            "layers": sorted(int(layer) for layer in summary[summary.panel == LANGUAGE_PANEL].layer),
            "verdict": language_verdict(policy),
            "likelihood": None, "dose_response": dose, "controls": [], "ablation": [], "layer_effects": [], "subsets": []}
    held = None
    if (HELD / "contrasts.csv").exists():
        means = pd.read_csv(HELD / "system_means.csv")
        contrasts = pd.read_csv(HELD / "contrasts.csv")
        labels = {"baseline": "MindMate prompt only", "kg": "Strategy prompt, no steering", "steered": "Strategy prompt + steering"}
        overall = means[means.subset == "all"].set_index("system")
        systems = [{"system": name, "label": labels[name], "emotion_rating": _number(row.tone),
                    "independent_classifier": _number(row.indep_target), "empathy": _number(row.empathy),
                    "relevance": _number(row.relevance), "fluency": _number(row.fluency),
                    "steered_share": _number(row.steered_share), "latency_median": None} for name, row in overall.iterrows()]
        shown = {"tone": "Tone rating (1-5)", "empathy": "Empathy (1-5)", "relevance": "Relevance (1-5)", "fluency": "Fluency (1-5)",
                 "hinglish_naturalness": "Hinglish naturalness (1-5)", "preference_minus_0.5": "Judge preference - 0.5"}
        chosen = contrasts[(contrasts.subset == "all") & (contrasts.source == "both") & contrasts.metric.isin(shown)
                           & contrasts.contrast.isin(["steered - kg", "kg - baseline"])]
        rows = [{"contrast": row.contrast, "label": {"steered - kg": "Steering vs no steering", "kg - baseline": "Strategy prompt vs plain"}[row.contrast],
                 "metric": shown[row.metric], "mean": _number(row["mean"], 4), "lo": _number(row.ci_low, 4), "hi": _number(row.ci_high, 4),
                 "p": _number(row.p, 4)} for _, row in chosen.iterrows()]
        tone = chosen[(chosen.contrast == "steered - kg") & (chosen.metric == "tone")]
        held = {"messages": int(overall.n.iloc[0]), "systems": systems, "contrasts": rows,
                "summary": (f"Held-out Sarvam messages (two judges): steering changed the tone rating by {tone['mean'].iloc[0]:+.3f} "
                            f"(95% CI {tone.ci_low.iloc[0]:+.3f} to {tone.ci_high.iloc[0]:+.3f}) over all messages, including "
                            "those it chose not to steer.") if len(tone) else "Held-out comparison for Sarvam."}
    evidence = {"panels": panels, "held_out": held,
                "sources": {"causal experiments": str(LAB / "causal"), "held-out comparison": str(HELD), "active panel": str(PANEL_FILE)}}
    (LAB / "evidence.json").write_text(json.dumps(evidence, indent=1))
    print(f"Wrote {LAB / 'evidence.json'} ({len(panels)} panels)")
    return evidence


def install(args) -> None:
    """Install the calibrated Sarvam policy and evidence for the chatbot, only when the held-out test supports it.

    Every policy cell that steers must be backed by a held-out tone gain whose CI excludes zero for both judges and
    no quality metric whose CI lower bound is below -0.15, on the steered subset.  A policy that steers nothing is
    installed as an explicit, evidence-backed "steering disabled" state: without it the chat service would fall back
    to heuristic strengths on the Sarvam panels, which were never validated."""
    policy_path, evidence_path = LAB / "policy_candidate.json", LAB / "evidence.json"
    for required in (policy_path, evidence_path, HELD / "contrasts.csv", HELD / "lock.json", HELD / "summary.json"):
        if not required.exists():
            raise SystemExit(f"Missing {required}: run the held-out stages and 'evidence' first.")
    policy = json.loads(policy_path.read_text())
    steering = {name: cell for name, cell in policy["cells"].items() if cell.get("alpha")}
    if steering:
        contrasts = pd.read_csv(HELD / "contrasts.csv")
        steered = contrasts[(contrasts.subset == "steered") & (contrasts.contrast == "steered - kg")]
        if steered.empty:
            raise SystemExit("Policy steers but the held-out test has no steered subset; not installed.")
        gains = steered[steered.metric == "tone"]
        reliable = all(gains[gains.source == judge].ci_low.gt(0).all() for judge in JUDGES)
        quality = steered[steered.metric.isin(QUALITY) & (steered.source != "automatic")]
        safe = quality.ci_low.ge(-RULE["ci_margin"]).all()
        if not (reliable and safe):
            raise SystemExit("Held-out test does not show a reliable, quality-preserving benefit for both judges; not installed. "
                             f"Cells that would steer: {sorted(steering)}")
    target = ROOT / "steering_policy" / "policy.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    installed = {**policy, "installed_from": str(policy_path), "held_out": str(HELD),
                 "steering_enabled": bool(steering),
                 "note": ("Steering disabled by calibration: no strategy or language passed the two-judge safety and benefit "
                          "criteria. The Emotion Reasoner still selects the strategy and prompt; the Mood Tracker shows the "
                          "calibration evidence behind the decision." if not steering else
                          "Steering limited to the cells the two-judge calibration and held-out test support.")}
    target.write_text(json.dumps(installed, indent=1))
    evidence_target = ROOT / "causal_steering" / "evidence.json"
    evidence_target.parent.mkdir(parents=True, exist_ok=True)
    evidence_target.write_text(evidence_path.read_text())
    print(f"Installed policy -> {target} (steering {'enabled for ' + ', '.join(sorted(steering)) if steering else 'disabled'})")
    print(f"Installed evidence -> {evidence_target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=("probe", "causal", "causal_score", "causal_report", "causal2", "calibrate",
                                                      "calibrate_score", "select", "heldout", "heldout_score",
                                                      "heldout_report", "evidence", "install", "causal3", "causal3_score",
                                                      "causal3_report", "compare", "heldout_language", "heldout_language_score",
                                                      "heldout_language_report", "causal_agreement", "smoke"))
    parser.add_argument("--alphas", type=float, nargs=4, default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--language-alphas", type=float, nargs="+", default=[0.5, 1.0, 1.5],
                        help="Strengths of the Hinglish language panel (it is far stronger than a tone panel).")
    parser.add_argument("--combo-language", type=float, default=1.0,
                        help="Language-panel strength used together with a tone panel.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    for stage in args.stages:
        if stage == "probe":
            runtime = load_runtime()
            panels, full = load_panels(), load_panels(full=True)
            items = load_items("dev")
            picks = pd.concat([items[items.language == "hinglish"].drop_duplicates("tone").head(5), items[items.language == "english"].head(2)])
            for row in picks.itertuples():
                rows = [("none", {})] + [(f"panel@{a:g}", {"shift": [(panels[row.tone], a)]}) for a in (1, 2, 4, 8, 16)]
                rows += [(f"full@{a:g}", {"shift": [(full[row.tone], a)]}) for a in (1, 2, 4)]
                rows += [(f"language@{a:g}", {"shift": [(panels[LANGUAGE_PANEL], a)]}) for a in (2, 4, 8)]
                replies = generate_rows(runtime, messages_of(row), kg_prompt(row), [spec for _, spec in rows], 70)
                print(f"\n### [{row.tone}] {row.text}")
                for (name, _), reply in zip(rows, replies):
                    print(f"  {name:14s} {reply[:230]!r}")
        elif stage == "causal":
            run("causal", load_items("dev"), causal_rows, args)
        elif stage == "causal2":
            run("causal2", load_items("dev"), causal2_rows, args)
            score("causal2", "dev", ("judge1",))
            rows = paired_table("causal2", ("judge1",))
            table = pd.DataFrame([{"name": name, "subset": language, **summarise(group, "judge1")}
                                  for language, part in (("all", rows), ("hinglish", rows[rows.language == "hinglish"]),
                                                         ("english", rows[rows.language == "english"]))
                                  for name, group in part.groupby("name")])
            table.to_csv(LAB / "causal2" / "effects.csv", index=False)
            pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
            print(table[SHOWN + ["hinglish_reply_share"]].round(2).to_string(index=False))
        elif stage == "causal_score":
            score("causal", "dev", ("judge1",))
        elif stage == "causal_report":
            causal_report(args)
        elif stage == "calibrate":
            run("calibrate", load_items("calibration"), calibration_rows, args)
        elif stage == "calibrate_score":
            score("calibrate", "calibration", tuple(JUDGES))
        elif stage == "causal3":
            run("causal3", load_items("dev"), causal3_rows, args)
        elif stage == "causal3_score":
            score("causal3", "dev", tuple(JUDGES))
        elif stage == "causal3_report":
            causal3_report(args)
        elif stage == "compare":
            compare(args)
        elif stage == "select":
            select(args)
        else:
            {"heldout": heldout_generate, "heldout_score": heldout_score, "heldout_report": heldout_report,
             "heldout_language": heldout_language_generate, "heldout_language_score": heldout_language_score,
             "heldout_language_report": heldout_language_report, "causal_agreement": causal_agreement, "smoke": smoke,
             "evidence": export_evidence, "install": install}[stage](args)


if __name__ == "__main__":
    main()
