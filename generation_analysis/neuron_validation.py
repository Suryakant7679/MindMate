#!/usr/bin/env python3
"""Unified causal validation of every steering panel of one model.

Question: are the selected neurons causally relevant to the emotional tone of generated replies, or would the same
effects arise from random neurons, from any hook at all, or from unrelated features?  Each claim rests on an
intervention compared with a matched control.  Selectivity is recorded but never used as evidence on its own.

Stages (per model; ``summary`` reads all models afterwards):

  prepare    neutral activation means of every steering site (the value clamp ablations hold neurons at)
  generate   replies to the development messages under the knowledge-graph prompt (A: dose-response, reversed shift,
             sham hooks, two matched random draws, another emotion's panel, three layer bands, ranked subsets) and
             under the same prompt asking for the emotion (B: neutral, zero and random clamps, late-band clamp,
             reversed shift); plus sampled-decoding replicates with fixed seeds (panel, reference, random draw)
  score      two judges (Qwen2.5-7B-Instruct, Gemma 3-27B-IT; one loaded at a time) rate every reply for the intended
             emotion and the quality rubrics and judge the steered reply against its reference; the independent
             GoEmotions classifier scores the intended emotion
  analyse    paired differences against each reference reply, bootstrap CIs, Wilcoxon p-values with Holm correction,
             verdicts per question, language slices, judge agreement, failure cases, per-neuron evidence cards
  summary    cross-model table (results/research/summary.json)

Matched random neurons keep, per layer, the panel's neuron count and shift values, and draw neurons from the same
quantile band of the neutral activation magnitude.  Ranked subsets use attribution for Sarvam and the stored shift
magnitude for the Llama-family panels (their build did not store an attribution rank).

  python3 -m generation_analysis.neuron_validation --model llama all
  python3 -m generation_analysis.neuron_validation --model sarvam --limit 40 prepare generate score analyse
  python3 -m generation_analysis.neuron_validation summary
"""
from __future__ import annotations

import os
import sys

# The experiment helpers read the analysed model from MINDMATE_MODEL at import time, so it is set first.
if "--model" in sys.argv:
    _key = sys.argv[sys.argv.index("--model") + 1]
    if _key in {"llama", "llama2", "emollama", "sarvam"}:
        os.environ["MINDMATE_MODEL"] = _key

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import zlib  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .steering_lab import TARGETS, holm, paired_summary, release  # noqa: E402

RESULTS = Path("results")
MODELS = {
    "llama": {"tag": "llama31_8b", "source": "llama", "panel_file": "panels_causal.json", "reference": 1.5,
              "dose": (0.5, 1.0, 1.5, 2.0), "label": "Llama 3.1-8B"},
    "llama2": {"tag": "llama2_13b_chat", "source": "llama", "panel_file": "panels.json", "reference": 1.5,
               "dose": (0.5, 1.0, 1.5, 2.0), "label": "Llama-2-13B-chat"},
    "emollama": {"tag": "emollama_chat_7b", "source": "llama", "panel_file": "panels.json", "reference": 1.5,
                 "dose": (0.5, 1.0, 1.5, 2.0), "label": "EmoLLaMA-chat-7B"},
    "sarvam": {"tag": "sarvam_30b", "source": "sarvam", "panel_file": "panels.json", "reference": 3.0,
               "dose": (1.0, 2.0, 3.0, 4.0), "label": "Sarvam-30B"},
}
SAMPLE_SEEDS = (11, 23, 37)
SAMPLED_MESSAGES = 24
SAMPLE_TEMPERATURE, SAMPLE_TOP_P = 0.7, 0.9
RANDOM_DRAWS = 2
FRACTIONS = (0.1, 0.5)
# Sarvam's panels are named after response tones; the nearest GoEmotions label is the independent target.
TONE_TARGET = {"warmth": "love", "reassurance": "love", "encouragement": "admiration", "joy": "joy", "admiration": "admiration"}
QUALITY = ("appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness", "hinglish_naturalness")
METRICS = ("emotion", *QUALITY, "preservation")
HARM_DROP = -0.5          # a quality rating lower by more than this (1-5 scale) counts as harm
HARM_SHARE = 0.20         # the calibration rule's maximum harmed share
CONTRASTS = {
    "panel_vs_sham": ("Panel vs sham hooks", "Does the panel change the reply more than installing the hooks alone?"),
    "panel_vs_random": ("Panel vs matched random neurons (mean of draws)", "Is the effect specific to these neurons?"),
    "panel_vs_other": ("Panel vs another emotion's panel", "Is the effect specific to this emotion's panel?"),
    "reverse_vs_panel": ("Reversed shift vs panel", "Does reversing the shift reverse the effect?"),
    "dose_max_vs_min": ("Strongest vs weakest strength", "Does the effect grow with strength?"),
    "top_vs_random_subset": ("Top-ranked share vs random share (same size)", "Are the top-ranked neurons the ones that matter?"),
    "clamp_vs_random": ("Panel held at neutral vs random neurons held at neutral", "Are these neurons necessary beyond any neurons?"),
}
NOTES = [
    "Emotion change is the change of the mean of the two judges' 1-5 ratings of the intended emotion, against the reference "
    "reply decoded in the same batch. Quality changes use the same pairing.",
    "Intervals are 95% bootstrap CIs over messages; p-values are two-sided Wilcoxon signed-rank tests, Holm-corrected within each panel.",
    "The sham installs the hooks on every steering site with a zero shift; it must reproduce the unsteered reply.",
    "Matched random neurons are drawn per message (two draws); the 'random mean' averages them.",
    "Replicates use sampled decoding with fixed seeds on the first messages only.",
]


def model_paths(key: str) -> dict[str, Path]:
    root = RESULTS / f"{MODELS[key]['tag']}_generation_emotion" / "neuron_validation"
    return {"root": root, "neutral": root / "neutral_means.npy", "manifest": root / "manifest.json",
            "generations": root / "generations.jsonl", "scores": root / "scores.csv", "report": root / "report.json",
            "cards": root / "cards.json"}


def describe(key: str, panel: str) -> tuple[str, str]:
    """(GoEmotions label for the independent classifier, description the judges rate)."""
    from .sarvam_lab import TONES
    if MODELS[key]["source"] == "sarvam":
        return TONE_TARGET[panel], TONES[panel]
    from .common import EMOTION_DESCRIPTIONS
    return panel, EMOTION_DESCRIPTIONS[panel]


def load_runtime(key: str):
    if MODELS[key]["source"] == "sarvam":
        from .sarvam_lab import load_runtime as load_sarvam
        return load_sarvam()
    from backend.models import create_runtime
    return create_runtime(key)


def panel_names(key: str, runtime) -> list[str]:
    from .sarvam_lab import TONES
    available = set(getattr(runtime, "_emotion_panels", {}) or {})
    wanted = list(TONES) if MODELS[key]["source"] == "sarvam" else list(TARGETS)
    return [name for name in wanted if name in available]


def load_messages(key: str, limit: int | None) -> pd.DataFrame:
    if MODELS[key]["source"] == "sarvam":
        from .sarvam_lab import load_items
        frame = load_items("dev")
    else:
        if os.environ.get("MINDMATE_VALIDATION_HINGLISH") == "1":
            from .v3_common import load_items            # the 48 English dev messages plus 24 Hinglish ones
            frame = load_items("dev")
        else:
            from .steering_lab import load_split
            frame = load_split("dev")
    frame = frame.reset_index(drop=True)
    return frame.head(limit).reset_index(drop=True) if limit else frame


def messages_of(row: Any) -> list[dict[str, str]]:
    return [*row.history, {"role": "user", "content": row.text}]


def styled_prompt(row: Any, description: str) -> str:
    return row.kg_prompt + f" Make the reply clearly express {description}."


def neutral_prompt(row: Any) -> str:
    return row.kg_prompt + " Keep the reply emotionally neutral: calm, factual and free of emotional words."


def site_offsets(widths: dict[int, int]) -> dict[int, int]:
    out, start = {}, 0
    for layer in sorted(widths):
        out[layer], start = start, start + widths[layer]
    return out


Panel = dict[int, tuple[list[int], list[float]]]


def panel_layers(key: str, panel: str) -> list[int]:
    payload = json.loads((RESULTS / f"{MODELS[key]['tag']}_generation_emotion" / "steering_panels" / MODELS[key]["panel_file"]).read_text())
    return sorted(int(layer) for layer in payload["panels"][panel])


def ranking(key: str, name: str, panel: Panel) -> dict[int, dict[int, float]]:
    """Per layer and neuron, the score a panel is ranked by: attribution for Sarvam, |stored shift| otherwise."""
    if MODELS[key]["source"] == "sarvam":
        from .sarvam_lab import attribution_scores
        scores, _, _ = attribution_scores(name, statistics=True)
        return {layer: {neuron: float(scores[layer][neuron]) for neuron in neurons} for layer, (neurons, _) in panel.items()}
    return {layer: {neuron: abs(float(delta)) for neuron, delta in zip(neurons, deltas)} for layer, (neurons, deltas) in panel.items()}


def top_share(panel: Panel, scores: dict[int, dict[int, float]], fraction: float, order: str,
              rng: np.random.Generator | None = None) -> Panel:
    out: Panel = {}
    for layer, (neurons, deltas) in panel.items():
        count = max(1, round(fraction * len(neurons)))
        if order == "random":
            picked = sorted(rng.choice(len(neurons), count, replace=False).tolist())
        else:
            ranked = sorted(range(len(neurons)), key=lambda i: -scores[layer][neurons[i]])
            picked = sorted(ranked[:count] if order == "top" else ranked[-count:])
        out[layer] = ([neurons[i] for i in picked], [deltas[i] for i in picked])
    return out


def restrict(panel: Panel, layers) -> Panel:
    keep = set(layers)
    return {layer: value for layer, value in panel.items() if layer in keep}


def matched_random(panel: Panel, widths: dict[int, int], neutral: np.ndarray, start: dict[int, int],
                   rng: np.random.Generator) -> Panel:
    """Random neurons per layer: same count and shift values, neutral activation magnitude matched by quantile."""
    out: Panel = {}
    for layer, (neurons, deltas) in panel.items():
        magnitude = np.abs(neutral[start[layer]:start[layer] + widths[layer]])
        pool = np.setdiff1d(np.arange(widths[layer]), np.array(neurons, dtype=np.int64))
        pool = pool[np.argsort(magnitude[pool], kind="stable")]
        used: set[int] = set()
        chosen = [0] * len(neurons)
        window = max(5, len(pool) // 20)
        for index in np.argsort(magnitude[np.array(neurons)], kind="stable"):
            centre = int(np.searchsorted(magnitude[pool], magnitude[neurons[index]]))
            low, high = max(0, centre - window), min(len(pool), centre + window + 1)
            candidates = [c for c in range(low, high) if int(pool[c]) not in used]
            if not candidates:
                candidates = [c for c in range(len(pool)) if int(pool[c]) not in used]
            position = int(candidates[int(rng.integers(len(candidates)))])
            used.add(int(pool[position]))
            chosen[index] = int(pool[position])
        out[layer] = (chosen, [float(value) for value in rng.permutation(deltas)])
    return out


def clamp_to(panel: Panel, start: dict[int, int], neutral: np.ndarray) -> dict[int, tuple[list[int], list[float]]]:
    return {layer: (neurons, [float(neutral[start[layer] + n]) for n in neurons]) for layer, (neurons, _) in panel.items()}


def size(panel: Panel) -> int:
    return sum(len(neurons) for neurons, _ in panel.values())


# --------------------------------------------------------------------------- prepare

def prepare(args) -> None:
    from .sarvam_lab import Recorder, sites
    import torch
    key = args.model
    paths = model_paths(key)
    paths["root"].mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(key)
    widths = sites(runtime)
    names = panel_names(key, runtime)
    messages = load_messages(key, args.limit)
    accumulated = np.zeros(sum(widths.values()), dtype=np.float64)
    for row in messages.itertuples():
        encoded = runtime._encode_conversation(messages_of(row), neutral_prompt(row), args.max_new_tokens)
        recorder = Recorder(runtime)
        with recorder, runtime.lock, torch.inference_mode():
            runtime.model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                   max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=runtime.tokenizer.pad_token_id, **runtime.generation_settings())
        accumulated += recorder.means()[0].astype(np.float64)
    neutral = (accumulated / max(len(messages), 1)).astype(np.float32)
    np.save(paths["neutral"], neutral)
    manifest = {"model": runtime.model_id, "model_key": key, "tag": MODELS[key]["tag"], "panels": names,
                "messages": int(len(messages)), "split": "dev", "limit": args.limit,
                "sites": {str(k): v for k, v in widths.items()},
                "steering_location": getattr(runtime, "steering_location", None),
                "reference": MODELS[key]["reference"], "dose": list(MODELS[key]["dose"]), "sample_seeds": list(SAMPLE_SEEDS)}
    paths["manifest"].write_text(json.dumps(manifest, indent=1))
    print(f"prepared {key}: {len(messages)} messages, panels {names}, {sum(widths.values()):,} sites", flush=True)
    del runtime
    release()


# --------------------------------------------------------------------------- generate

def label_of(family: str, alpha: float | None, group: str) -> str:
    text = {"none": "Unsteered", "styled": "Unsteered, prompted for the emotion", "dose": "Panel",
            "reverse": "Reversed panel", "sham": "Sham hooks", "matched_random": "Matched random neurons",
            "other_panel": "Another emotion's panel", "band_only": "Layer band alone", "band_out": "Panel without layer band",
            "subset_top": "Top-ranked share", "subset_random": "Random share", "subset_bottom": "Lowest-ranked share",
            "clamp": "Panel held at neutral", "clamp_zero": "Panel held at zero",
            "clamp_random": "Random neurons held at neutral", "styled_reverse": "Reversed panel (prompted)",
            "styled_reverse_random": "Reversed random neurons (prompted)", "sampled": "Sampled decoding"}.get(family, family)
    if group:
        text += f" ({group})"
    if alpha is not None and family not in ("clamp", "clamp_zero", "clamp_random", "styled", "none"):
        text += f", α {alpha:g}"
    return text


def plan_conditions(key: str, panel: str, panels: dict[str, Panel], widths: dict[int, int], start: dict[int, int],
                    neutral: np.ndarray, scores: dict[int, dict[int, float]], rng: np.random.Generator,
                    other: str | None) -> list[dict]:
    reference, dose = MODELS[key]["reference"], MODELS[key]["dose"]
    base = panels[panel]
    randoms = [matched_random(base, widths, neutral, start, rng) for _ in range(RANDOM_DRAWS)]
    layers = sorted(base)
    thirds = {"early": layers[:len(layers) // 3], "mid": layers[len(layers) // 3:2 * len(layers) // 3],
              "late": layers[2 * len(layers) // 3:]}
    out: list[dict] = []

    def add(name, family, group, alpha, spec, prompt, reference_name, neurons, seed=None):
        out.append({"name": name, "family": family, "group": group, "alpha": alpha, "spec": spec, "prompt": prompt,
                    "reference": reference_name, "neurons": neurons, "seed": seed, "label": label_of(family, alpha, group)})

    add("none", "none", "", None, {}, "A", None, 0)
    for alpha in dose:
        add(f"dose_{alpha:g}", "dose", "", alpha, {"shift": [(base, alpha)]}, "A", "none", size(base))
    add(f"reverse_{reference:g}", "reverse", "", -reference, {"shift": [(base, -reference)]}, "A", "none", size(base))
    add("sham", "sham", "", 0.0, {"sham": True}, "A", "none", 0)
    for index, draw in enumerate(randoms):
        add(f"random_{index}", "matched_random", str(index), reference, {"shift": [(draw, reference)]}, "A", "none", size(draw), seed=index)
    if other and other in panels:
        add(f"other_{other}", "other_panel", other, reference, {"shift": [(panels[other], reference)]}, "A", "none", size(panels[other]))
    for band, members in thirds.items():
        if not members:
            continue
        only = restrict(base, members)
        rest = restrict(base, [layer for layer in layers if layer not in members])
        add(f"band_only_{band}", "band_only", band, reference, {"shift": [(only, reference)]}, "A", "none", size(only))
        if rest:
            add(f"band_out_{band}", "band_out", band, reference, {"shift": [(rest, reference)]}, "A", "none", size(rest))
    for fraction in FRACTIONS:
        group = f"{fraction:g}"
        top, bottom = top_share(base, scores, fraction, "top"), top_share(base, scores, fraction, "bottom")
        shuffled = top_share(base, scores, fraction, "random", rng)
        add(f"subset_top_{group}", "subset_top", group, reference, {"shift": [(top, reference)]}, "A", "none", size(top))
        add(f"subset_random_{group}", "subset_random", group, reference, {"shift": [(shuffled, reference)]}, "A", "none", size(shuffled))
        add(f"subset_bottom_{group}", "subset_bottom", group, reference, {"shift": [(bottom, reference)]}, "A", "none", size(bottom))
    add("styled", "styled", "", None, {}, "B", None, 0)
    add("clamp", "clamp", "", None, {"clamp": clamp_to(base, start, neutral)}, "B", "styled", size(base))
    add("clamp_zero", "clamp_zero", "", None, {"clamp": {layer: (n, [0.0] * len(n)) for layer, (n, _) in base.items()}},
        "B", "styled", size(base))
    add("clamp_random", "clamp_random", "", None, {"clamp": clamp_to(randoms[0], start, neutral)}, "B", "styled", size(randoms[0]))
    late = restrict(base, thirds["late"])
    if late:
        add("clamp_late", "clamp", "late", None, {"clamp": clamp_to(late, start, neutral)}, "B", "styled", size(late))
    add(f"styled_reverse_{reference:g}", "styled_reverse", "", -reference, {"shift": [(base, -reference)]}, "B", "styled", size(base))
    add(f"styled_reverse_random_{reference:g}", "styled_reverse_random", "", -reference,
        {"shift": [(randoms[0], -reference)]}, "B", "styled", size(randoms[0]))
    return out


def generate(args) -> None:
    from .sarvam_lab import generate_rows, sites
    paths = model_paths(args.model)
    if not paths["neutral"].exists():
        raise SystemExit("Run the prepare stage first.")
    key = args.model
    runtime = load_runtime(key)
    widths = sites(runtime)
    start = site_offsets(widths)
    neutral = np.load(paths["neutral"])
    names = json.loads(paths["manifest"].read_text())["panels"]
    panels = {name: runtime._emotion_panels[name] for name in names}
    scores = {name: ranking(key, name, panels[name]) for name in names}
    messages = load_messages(key, args.limit)
    done = set()
    if paths["generations"].exists():
        for line in paths["generations"].open(encoding="utf-8"):
            record = json.loads(line)
            done.add((record["item_id"], record["panel"], record["prompt"]))
    with paths["generations"].open("a", encoding="utf-8") as handle:
        for position, row in enumerate(messages.itertuples()):
            for panel in names:
                _, description = describe(key, panel)
                other = next((name for name in names if name != panel), None)
                rng = np.random.default_rng(zlib.crc32(f"{row.item_id}|{panel}".encode()))
                plan = plan_conditions(key, panel, panels, widths, start, neutral, scores[panel], rng, other)
                target = describe(key, panel)[0]
                for prompt_key, prompt in (("A", row.kg_prompt), ("B", styled_prompt(row, description))):
                    if (row.item_id, panel, prompt_key) in done:
                        continue
                    group = [item for item in plan if item["prompt"] == prompt_key]
                    replies = generate_rows(runtime, messages_of(row), prompt, [item["spec"] for item in group], args.max_new_tokens)
                    for item, reply in zip(group, replies):
                        handle.write(json.dumps({"item_id": row.item_id, "panel": panel, "prompt": prompt_key, "name": item["name"],
                                                 "family": item["family"], "group": item["group"], "alpha": item["alpha"],
                                                 "reference": item["reference"], "seed": item["seed"], "label": item["label"],
                                                 "neurons": item["neurons"], "sample_seed": None, "text": row.text,
                                                 "language": row.language, "history": list(row.history), "target": target,
                                                 "description": description, "reply": reply}, ensure_ascii=False) + "\n")
                    handle.flush()
                if position < SAMPLED_MESSAGES and (row.item_id, panel, "S") not in done:
                    base = panels[panel]
                    draw = plan[[item["name"] for item in plan].index("random_0")]["spec"]["shift"][0][0]
                    rows = [("none", {}, None), ("panel", {"shift": [(base, MODELS[key]["reference"])]}, "none"),
                            ("random_0", {"shift": [(draw, MODELS[key]["reference"])]}, "none")]
                    for seed in SAMPLE_SEEDS:
                        replies = generate_rows(runtime, messages_of(row), row.kg_prompt, [spec for _, spec, _ in rows],
                                                args.max_new_tokens, sample={"seed": seed, "temperature": SAMPLE_TEMPERATURE,
                                                                             "top_p": SAMPLE_TOP_P})
                        for (name, spec, reference), reply in zip(rows, replies):
                            handle.write(json.dumps({"item_id": row.item_id, "panel": panel, "prompt": "S",
                                                     "name": f"sampled_{name}_{seed}", "family": "sampled", "group": str(seed),
                                                     "alpha": MODELS[key]["reference"] if name != "none" else None,
                                                     "reference": f"sampled_none_{seed}" if reference else None,
                                                     "seed": seed, "label": label_of("sampled", None, str(seed)),
                                                     "neurons": size(base) if name != "none" else 0, "sample_seed": seed,
                                                     "text": row.text, "language": row.language, "history": list(row.history),
                                                     "target": target, "description": description, "reply": reply},
                                                    ensure_ascii=False) + "\n")
                        handle.flush()
            print(f"generated {row.item_id} ({position + 1}/{len(messages)})", flush=True)
    del runtime
    release()


# --------------------------------------------------------------------------- gradients (Sarvam)

def gradients(args) -> None:
    """First-order causal estimate for every Sarvam site neuron and tone.

    Teacher-forced on the saved discovery replies: under the *neutral* prompt, how much would a constant shift of
    neuron u on the decode positions raise the mean log-likelihood of the tone-styled reply?
    ``effect_u = dL/d(shift_u) x shift_u`` with shift_u the neuron's stored tone-minus-neutral difference.
    An estimate from one backward pass per reply, not an intervention; the battery's interventions test the panel.
    """
    import torch
    from .sarvam_lab import DATA, LAB, NEUTRAL, TONES, load_panels, sites, style_prompt
    key = args.model
    if MODELS[key]["source"] != "sarvam":
        raise SystemExit("Llama-family gradient estimates already exist (causal_panel_experiment attribution stage).")
    paths = model_paths(key)
    runtime = load_runtime(key)
    widths = sites(runtime)
    start = site_offsets(widths)
    total = sum(widths.values())
    index = pd.read_csv(LAB / "discovery" / "index.csv")
    items = pd.read_csv(DATA / "discovery.csv").set_index("item_id")
    wanted = list(index.item_id.unique())[:args.gradient_prompts]
    full = load_panels(full=True)
    grads = np.zeros((len(TONES), total), dtype=np.float64)
    counts = np.zeros(len(TONES), dtype=np.int64)
    device = next(runtime.model.parameters()).device
    for position, item_id in enumerate(wanted):
        row = items.loc[item_id]
        encoded = runtime._encode_conversation([{"role": "user", "content": row.text}], style_prompt(NEUTRAL, row.language), 80)
        prompt = encoded["input_ids"][0]
        for tone_index, tone in enumerate(TONES):
            found = index[(index.item_id == item_id) & (index.condition == tone)]
            if found.empty or not isinstance(found.reply.iloc[0], str):
                continue
            reply = torch.tensor(runtime.tokenizer(found.reply.iloc[0], add_special_tokens=False)["input_ids"], device=device)
            if len(reply) < 3:
                continue
            ids = torch.cat([prompt, reply])[None]
            length = len(prompt)
            shifts = {layer: torch.zeros(width, device=device, dtype=torch.float32, requires_grad=True) for layer, width in widths.items()}

            def hook_for(layer: int):
                def hook(_module, inputs):
                    hidden = inputs[0]
                    if hidden.dim() != 3 or hidden.shape[1] != ids.shape[1]:
                        return inputs
                    mask = torch.zeros(hidden.shape[1], 1, device=hidden.device, dtype=hidden.dtype)
                    mask[length:] = 1                          # decode positions: those that read a generated token
                    return (hidden + mask * shifts[layer].to(hidden.dtype), *inputs[1:])
                return hook
            handles = [runtime.steering_module(layer).register_forward_pre_hook(hook_for(layer)) for layer in widths]
            try:
                with runtime.lock, torch.enable_grad():
                    logits = runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float()
                    logp = torch.log_softmax(logits[length:-1], dim=-1)       # predicts reply tokens 1..
                    score = logp.gather(1, reply[1:, None]).mean()
                    score.backward()
            finally:
                for handle in handles:
                    handle.remove()
            for layer in widths:
                grads[tone_index, start[layer]:start[layer] + widths[layer]] += shifts[layer].grad.double().cpu().numpy()
            counts[tone_index] += 1
            del logits, logp, score, shifts
        if position % 5 == 4 or position == len(wanted) - 1:
            torch.cuda.empty_cache()
            print(f"gradients {position + 1}/{len(wanted)} prompts, replies per tone {counts.tolist()}", flush=True)
    grads /= np.maximum(counts, 1)[:, None]
    effect = np.zeros_like(grads)
    for tone_index, tone in enumerate(TONES):
        for layer, (neurons, deltas) in full[tone].items():
            effect[tone_index, start[layer] + np.array(neurons)] = grads[tone_index, start[layer] + np.array(neurons)] * np.array(deltas)
    np.save(paths["root"] / "gradient.npy", grads.astype(np.float32))
    np.save(paths["root"] / "gradient_effect.npy", effect.astype(np.float32))
    (paths["root"] / "gradient_meta.json").write_text(json.dumps({
        "tones": list(TONES), "prompts": len(wanted), "replies_per_tone": counts.tolist(),
        "estimate": "d(mean log-likelihood of the tone reply under the neutral prompt)/d(shift) x stored shift",
        "sites": {str(k): v for k, v in widths.items()}}, indent=1))
    print(f"wrote {paths['root'] / 'gradient_effect.npy'}", flush=True)
    del runtime
    release()


# --------------------------------------------------------------------------- score

def emotion_rubric(description: str) -> str:
    return (f"Emotional tone: how strongly does the reply itself express {description}? Judge only the tone of the reply, "
            "not whether it is a good reply. 1 = not at all. 5 = very strongly.")


def score(args) -> None:
    score_files(model_paths(args.model), args.force)


def score_files(paths: dict[str, Path], force: bool = False) -> None:
    """Judge and classify every reply in ``paths["generations"]`` (ratings are appended for new rows only)."""
    from .steering_lab import Judge
    from .v3_common import JUDGES, automatic_scores, judge_preferences, judge_ratings
    frame = pd.DataFrame([json.loads(line) for line in paths["generations"].open(encoding="utf-8")])
    replies = frame.set_index(["item_id", "panel", "name"])["reply"].to_dict()
    frame["reference_reply"] = [replies.get((row.item_id, row.panel, row.reference)) if isinstance(row.reference, str) else None
                                for row in frame.itertuples()]
    table = frame.reset_index(drop=True)
    automatic = automatic_scores(table)
    columns = ["indep_target", "degenerate", "repeated_bigram_rate", "reply_words", "reply_is_hinglish"]
    pd.concat([table.drop(columns=["history"]), automatic[columns]], axis=1).to_csv(paths["scores"], index=False)
    release()
    full_table = table
    for judge_name, (model_id, four_bit) in JUDGES.items():
        target = paths["root"] / f"ratings_{judge_name}.csv"
        earlier = pd.read_csv(target) if target.exists() and not force else None
        if earlier is not None and len(earlier) >= len(full_table):
            continue
        # Generations are appended, so rows already rated keep their position; only the new tail is judged.
        table = full_table.iloc[len(earlier):].reset_index(drop=True) if earlier is not None else full_table
        judge = Judge(model_id, load_in_4bit=four_bit)
        contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
        keys = list(zip(contexts, table.reply.fillna("").tolist(), [emotion_rubric(d) for d in table.description]))
        unique = list(dict.fromkeys(keys))
        values = dict(zip(unique, judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])))
        ratings = judge_ratings(judge, table, reference=[value if isinstance(value, str) else None for value in table.reference_reply])
        ratings["emotion"] = [float(values[k]) for k in keys]
        preference = np.full(len(table), np.nan)
        steered = table.reference_reply.notna().to_numpy()
        if steered.any():
            part = table[steered].reset_index(drop=True)
            preference[steered] = judge_preferences(judge, part, part.reply.tolist(), part.reference_reply.tolist())
        ratings["preference"] = preference
        if earlier is not None:
            ratings = pd.concat([earlier, ratings], ignore_index=True)
        ratings.to_csv(target, index=False)
        judge.close()
        print(f"scored by {judge_name} ({model_id}): {len(table)} new replies, {len(ratings)} in all", flush=True)


# --------------------------------------------------------------------------- analyse

def _num(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 5)


def stat(values) -> dict[str, Any]:
    values = pd.Series(values, dtype=float).dropna()
    if not len(values):
        return {"n": 0, "mean": None, "lo": None, "hi": None, "p": None, "dz": None}
    summary = paired_summary(values.to_numpy())
    spread = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    dz = float(values.mean()) / spread if spread and not math.isnan(spread) and spread > 0 else float("nan")
    return {"n": int(summary["n"]), "mean": _num(summary["mean"]), "lo": _num(summary["ci_low"]),
            "hi": _num(summary["ci_high"]), "p": _num(summary["p"]), "dz": _num(dz)}


def load_scores(key: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    return load_score_files(model_paths(key))


def load_score_files(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    table = pd.read_csv(paths["scores"], dtype={"item_id": str})
    judges = {name: pd.read_csv(paths["root"] / f"ratings_{name}.csv") for name in ("judge1", "judge2")}
    for name, ratings in judges.items():
        for column in METRICS + ("preference",):
            if column in ratings:
                table[f"{column}_{name}"] = ratings[column].to_numpy()
    for column in METRICS + ("preference",):
        parts = [f"{column}_{name}" for name in judges if f"{column}_{name}" in table]
        if parts:
            table[f"{column}_both"] = table[parts].mean(axis=1)
    table["reference"] = table["reference"].fillna("")
    table["language"] = table["language"].fillna("english")
    return table, judges


def wide(part: pd.DataFrame, column: str) -> pd.DataFrame:
    # dropna=False keeps conditions whose column is entirely missing (references have no preference).
    return part.pivot_table(index="item_id", columns="name", values=column, aggfunc="first", dropna=False)


def references(part: pd.DataFrame) -> dict[str, str]:
    refs = part.drop_duplicates("name").set_index("name")["reference"].to_dict()
    return {name: ref for name, ref in refs.items() if isinstance(ref, str) and ref}


def diffs(frame: pd.DataFrame, refs: dict[str, str], members: list[str]) -> pd.Series:
    """Per message: the mean over ``members`` of (condition - its reference)."""
    return pd.concat([frame[name] - frame[refs[name]] for name in members], axis=1).mean(axis=1)


def summarise(part: pd.DataFrame, reference: str = "dose_1.5", judges: tuple[str, ...] = ("judge1", "judge2")) -> dict[str, Any]:
    """Conditions and contrasts of one panel (or one language slice): every condition against its reference."""
    refs = references(part)
    meta = part.drop_duplicates("name").set_index("name")
    frames = {column: wide(part, column) for column in
              ["emotion_both", "indep_target", "degenerate", "reply_words", "preference_both", "reply",
               *[f"emotion_{j}" for j in judges], *[f"{m}_both" for m in QUALITY]] if column in part}
    draws = [name for name in refs if meta.loc[name, "family"] == "matched_random"]
    groups = [(name, [name]) for name in refs if name not in draws]
    if draws:
        groups.append(("random_mean", draws))
    rows, series, qualities = [], {}, {}
    for name, members in groups:
        emotion = diffs(frames["emotion_both"], refs, members)
        series[name] = emotion
        qualities[name] = {metric: diffs(frames[f"{metric}_both"], refs, members) for metric in QUALITY
                           if f"{metric}_both" in frames}
        scored = pd.concat(list(qualities[name].values()), axis=1).min(axis=1) if qualities[name] else pd.Series(dtype=float)
        head = meta.loc[members[0]]
        family = "matched_random_mean" if name == "random_mean" else str(head["family"])
        rows.append({
            "name": name, "family": family,
            "group": "" if name == "random_mean" or pd.isna(head["group"]) else str(head["group"]),
            "alpha": None if pd.isna(head["alpha"]) else float(head["alpha"]),
            "label": (f"Matched random neurons (mean of {len(members)} draws)" if name == "random_mean" else str(head["label"])),
            "neurons": None if pd.isna(head["neurons"]) else int(head["neurons"]),
            "seed": None if name == "random_mean" or pd.isna(head["seed"]) else int(head["seed"]),
            "n": int(emotion.notna().sum()), "reference": refs[members[0]] if name != "random_mean" else refs[members[0]],
            "changed_share": _num(np.mean([(frames["reply"][m] != frames["reply"][refs[m]]).mean() for m in members])),
            "emotion": stat(emotion),
            "emotion_judges": {j: stat(diffs(frames[f"emotion_{j}"], refs, members)) for j in judges if f"emotion_{j}" in frames},
            "independent": stat(diffs(frames["indep_target"], refs, members)),
            "quality": {metric: stat(values) for metric, values in qualities[name].items()},
            "preference": _num(frames["preference_both"][members].stack().mean() if name != "random_mean" and "preference_both" in frames else None),
            "harm_share": _num((scored < HARM_DROP).mean()) if len(scored) else None,
            "degenerate_share": _num(np.mean([frames["degenerate"][m].astype(float).mean() for m in members])),
            "length_ratio": _num(np.mean([(frames["reply_words"][m] / frames["reply_words"][refs[m]].clip(lower=1)).mean() for m in members])),
        })
    holm_p = holm([row["emotion"]["p"] if row["emotion"]["p"] is not None else 1.0 for row in rows]) if rows else []
    for row, adjusted in zip(rows, holm_p):
        row["emotion"]["p_holm"] = _num(adjusted)
    return {"conditions": rows, "contrasts": contrasts(series, rows, refs, reference), "series": series}


def contrasts(series: dict[str, pd.Series], rows: list[dict], refs: dict[str, str], reference: str) -> list[dict[str, Any]]:
    family = {row["name"]: row["family"] for row in rows}
    doses = sorted([name for name, fam in family.items() if fam == "dose"], key=lambda n: next(r["alpha"] for r in rows if r["name"] == n))
    if not doses or reference not in series:
        return []
    reference_dose = reference
    pairs = {
        "panel_vs_sham": (reference_dose, "sham"),
        "panel_vs_random": (reference_dose, "random_mean"),
        "panel_vs_other": (reference_dose, next((n for n in series if n.startswith("other_")), None)),
        "reverse_vs_panel": (next((n for n in series if n.startswith("reverse_")), None), reference_dose),
        "dose_max_vs_min": (doses[-1], doses[0]),
        "top_vs_random_subset": (f"subset_top_{FRACTIONS[1]:g}", f"subset_random_{FRACTIONS[1]:g}"),
        "clamp_vs_random": ("clamp", "clamp_random"),
    }
    out = []
    for name, (first, second) in pairs.items():
        if not first or not second or first not in series or second not in series:
            continue
        difference = (series[first] - series[second]).dropna()
        label, question = CONTRASTS[name]
        out.append({"name": name, "label": label, "question": question, "first": first, "second": second,
                    "stat": stat(difference), "supports": None})
    return out


def verdict_of(row: dict[str, Any] | None, *, positive: bool = True) -> str:
    if not row or row.get("lo") is None:
        return "inconclusive"
    if positive:
        if row["lo"] > 0:
            return "supported"
        return "not_supported" if row["hi"] <= 0 else "inconclusive"
    if row["hi"] < 0:
        return "supported"
    return "not_supported" if row["lo"] >= 0 else "inconclusive"


def _ci(row: dict[str, Any] | None) -> str:
    if not row or row.get("mean") is None:
        return "not estimated"
    return f"{row['mean']:+.2f} (95% CI {row['lo']:+.2f} to {row['hi']:+.2f})"


def verdicts(block: dict[str, Any], reference: str) -> dict[str, Any]:
    conditions = {row["name"]: row for row in block["conditions"]}
    found = {row["name"]: row["stat"] for row in block["contrasts"]}
    reverse_name = next((n for n in conditions if n.startswith("reverse_")), None)
    bands = [row for row in block["conditions"] if row["family"] == "band_only"]
    carrying = [f"{row['group']} third ({_ci(row['emotion'])})" for row in bands
                if row["emotion"]["lo"] is not None and row["emotion"]["lo"] > 0]
    layer_status = "supported" if carrying else ("not_supported" if bands and all(
        row["emotion"]["hi"] is not None and row["emotion"]["hi"] <= 0 for row in bands) else "inconclusive")
    result = {
        "panel_level": {"status": verdict_of(conditions.get(reference, {}).get("emotion")),
                        "text": f"At the reference strength the panel changes the judges' emotion rating by {_ci(conditions.get(reference, {}).get('emotion'))}."},
        "specificity": {"status": verdict_of(found.get("panel_vs_random")),
                        "text": f"Panel minus matched random neurons: {_ci(found.get('panel_vs_random'))}."},
        "reversibility": {"status": verdict_of(conditions.get(reverse_name, {}).get("emotion"), positive=False),
                          "text": f"Reversing the shift changes the rating by {_ci(conditions.get(reverse_name, {}).get('emotion'))}."},
        "dose_response": {"status": verdict_of(found.get("dose_max_vs_min")),
                          "text": f"Strongest minus weakest strength: {_ci(found.get('dose_max_vs_min'))}."},
        "necessity": {"status": verdict_of(found.get("clamp_vs_random"), positive=False),
                      "text": f"Holding the panel at neutral minus holding random neurons at neutral: {_ci(found.get('clamp_vs_random'))}."},
        "layer_level": {"status": layer_status,
                        "text": "Layer bands that carry the effect alone: " + ("; ".join(carrying) if carrying else "none with a CI above zero.")},
        "subset_level": {"status": verdict_of(found.get("top_vs_random_subset")),
                         "text": f"Top-ranked half minus random half: {_ci(found.get('top_vs_random_subset'))}."},
        "neuron_level": {"status": "not_tested",
                         "text": "Not tested. Single-neuron effects need one generation per neuron and message; the finest tested "
                                 "units are layer bands and ranked subsets, so no individual neuron is claimed."},
    }
    doses = sorted((row for row in block["conditions"] if row["family"] == "dose"), key=lambda row: row["alpha"])
    usable = [row for row in doses if row["harm_share"] is not None and row["harm_share"] <= HARM_SHARE
              and row["emotion"]["lo"] is not None and row["emotion"]["lo"] > 0]
    if usable:
        best = usable[-1]
        result["quality"] = {"status": "supported", "alpha": best["alpha"],
                             "text": (f"Strongest strength that raises the emotion while at most {HARM_SHARE:.0%} of replies lose quality: "
                                      f"α {best['alpha']:g}, emotion {_ci(best['emotion'])}, {best['harm_share']:.0%} of replies harmed, "
                                      f"judges prefer the steered reply {best['preference']:.2f} of the time.")}
    else:
        worst = doses[0] if doses else None
        result["quality"] = {"status": "not_supported", "alpha": None,
                             "text": ("No tested strength raises the emotion without harming more than "
                                      f"{HARM_SHARE:.0%} of replies"
                                      + (f" (weakest tested, α {worst['alpha']:g}: {worst['harm_share']:.0%} harmed, emotion {_ci(worst['emotion'])})."
                                         if worst and worst["harm_share"] is not None else "."))}
    reference_row = conditions.get(reference)
    if reference_row and reference_row.get("harm_share") is not None:
        result["panel_level"]["text"] += (f" At that strength {reference_row['harm_share']:.0%} of replies lose more than "
                                          f"{-HARM_DROP:g} points on a quality rating.")
    core = [result[key]["status"] for key in ("panel_level", "specificity", "reversibility")]
    if all(status == "supported" for status in core):
        status, text = "supported", ("Panel-level causal effect established: the panel raises the emotion of replies, beats matched "
                                     "random neurons, and reversing it lowers the rating. This says the panel as a whole is causally "
                                     "relevant; it says nothing about single neurons, and it is not a statement that steering is useful: "
                                     + result["quality"]["text"])
    elif "not_supported" in core[:2]:
        status, text = "not_supported", "The panel does not raise the emotion reliably beyond its controls."
    else:
        status, text = "inconclusive", "The evidence for a causal effect of this panel is incomplete."
    result["summary"] = {"status": status, "text": text}
    return result


def sampling_block(table: pd.DataFrame) -> dict[str, Any] | None:
    sampled = table[table.family == "sampled"].copy()
    if sampled.empty:
        return None
    sampled["item_id"] = sampled.item_id + "|" + sampled.panel        # one row per message and panel
    rows = []
    for seed, part in sampled.groupby("seed"):
        seed = int(seed)
        values = wide(part, "emotion_both")
        none, panel, draw = (values[f"sampled_none_{seed}"], values[f"sampled_panel_{seed}"], values[f"sampled_random_0_{seed}"])
        rows.append({"seed": int(seed), "panel": stat(panel - none), "matched_random": stat(draw - none),
                     "difference": stat((panel - none) - (draw - none))})
    return {"seeds": sorted(int(s) for s in sampled.seed.unique()),
            "messages": int(sampled.item_id.str.split("|").str[0].nunique()),
            "temperature": SAMPLE_TEMPERATURE, "rows": rows,
            "note": ("Sampled decoding with fixed seeds replicates the comparison with other draws of the reply. Each seed "
                     "compares the sampled panel and sampled matched random neurons with the sampled unsteered reply of the same seed.")}


def judge_agreement(judges: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    from scipy.stats import spearmanr
    rows = []
    for metric in METRICS:
        a, b = judges["judge1"].get(metric), judges["judge2"].get(metric)
        if a is None or b is None:
            continue
        valid = a.notna() & b.notna()
        if valid.sum() > 5:
            rows.append({"metric": metric, "replies": int(valid.sum()),
                         "spearman": _num(spearmanr(a[valid], b[valid]).statistic),
                         "mean_abs_difference": _num((a[valid] - b[valid]).abs().mean())})
    both = pd.DataFrame({"a": judges["judge1"]["preference"], "b": judges["judge2"]["preference"]}).dropna()
    if len(both) > 5:
        rows.append({"metric": "preference", "replies": int(len(both)), "spearman": _num(spearmanr(both.a, both.b).statistic),
                     "same_winner_share": _num(((both.a > 0.5) == (both.b > 0.5)).mean())})
    return rows


def failures(panels: dict[str, Any]) -> list[dict[str, str]]:
    found = []
    for name, block in panels.items():
        for row in block["conditions"]:
            if row.get("harm_share") is not None and row["harm_share"] > HARM_SHARE:
                found.append({"name": f"{name}:{row['name']}", "label": f"{name}: {row['label']}",
                              "issue": f"{row['harm_share']:.0%} of replies lost more than {-HARM_DROP:g} points on a quality rating."})
            if row["family"] == "sham" and row.get("changed_share"):
                found.append({"name": f"{name}:sham", "label": f"{name}: sham hooks",
                              "issue": f"{row['changed_share']:.0%} of replies changed with a zero shift (a hook artefact)."})
        if block["verdict"]["panel_level"]["status"] != "supported":
            found.append({"name": f"{name}:panel_level", "label": f"{name}: panel",
                          "issue": "The panel does not raise the intended emotion beyond the unsteered reply at the reference strength."})
    return found


def analyse(args) -> None:
    from .v3_common import JUDGES
    key = args.model
    paths = model_paths(key)
    manifest = json.loads(paths["manifest"].read_text())
    everything, judges = load_scores(key)
    table = everything[everything.family != "sampled"]           # replicates are reported separately
    reference = f"dose_{manifest['reference']:g}"
    languages = sorted(table.language.unique())
    panels, pooled_parts = {}, []
    for panel in manifest["panels"]:
        part = table[table.panel == panel]
        full = summarise(part, reference)
        reference_row = next(row for row in full["conditions"] if row["name"] == reference)
        block = {"panel": panel, "target": describe(key, panel)[0], "neurons": reference_row["neurons"],
                 "layers": panel_layers(key, panel), "messages": int(part.item_id.nunique()),
                 "languages": {lang: int(part[part.language == lang].item_id.nunique()) for lang in languages},
                 "selection": ("ranked by attribution (shift x write-vector alignment); paired |t| >= 3 over matched prompts; top 5% per site"
                               if MODELS[key]["source"] == "sarvam" else
                               "deployed panel; ranked by the stored shift magnitude |delta| within each layer"),
                 "conditions": full["conditions"], "contrasts": full["contrasts"], "by_language": {}}
        for lang in languages:
            slice_part = part[part.language == lang]
            if slice_part.item_id.nunique() >= 5:
                sliced = summarise(slice_part, reference)
                block["by_language"][lang] = {"messages": int(slice_part.item_id.nunique()),
                                              "conditions": sliced["conditions"], "contrasts": sliced["contrasts"]}
        block["verdict"] = verdicts(block, reference)
        panels[panel] = block
        pooled_parts.append(part.assign(item_id=part.item_id + "|" + panel))
    pooled_full = summarise(pd.concat(pooled_parts, ignore_index=True), reference)
    pooled = {"panel": "pooled", "target": "all panels", "neurons": None, "layers": [], "messages": int(table.item_id.nunique()),
              "languages": {lang: int(table[table.language == lang].item_id.nunique()) for lang in languages},
              "selection": "pooled over panels; each message counts once per panel", "conditions": pooled_full["conditions"],
              "contrasts": pooled_full["contrasts"], "by_language": {}}
    pooled["verdict"] = verdicts(pooled, reference)
    report = {"model": manifest["model"], "model_key": key, "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
              "split": "dev (disjoint from calibration and held-out)", "messages": int(table.item_id.nunique()),
              "judges": {name: model_id for name, (model_id, _) in JUDGES.items()},
              "reference_alpha": manifest["reference"], "alphas": manifest["dose"],
              "site": manifest.get("steering_location") or "", "seeds": list(SAMPLE_SEEDS), "panels": panels, "pooled": pooled,
              "judge_agreement": judge_agreement(judges), "sampling": sampling_block(everything),
              "failures": failures(panels), "notes": NOTES}
    paths["report"].write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    write_cards(key, paths)
    print(f"wrote {paths['report']}: {len(panels)} panels, {report['messages']} messages", flush=True)


def write_cards(key: str, paths: dict[str, Path]) -> None:
    """Per-neuron evidence: rank within the layer, stored shift and, for Sarvam, attribution and selectivity t."""
    payload = json.loads((RESULTS / f"{MODELS[key]['tag']}_generation_emotion" / "steering_panels" / MODELS[key]["panel_file"]).read_text())
    cards: dict[str, Any] = {"model_key": key, "panels": {}}
    # Llama-family models: first-order causal estimate of every neuron from the earlier attribution stage
    # (gradient of the emotional reply's log-likelihood with respect to the neuron's shift, times the shift).
    gradient_dir = RESULTS / f"{MODELS[key]['tag']}_generation_emotion" / "causal_steering" / "panel_causality"
    gradients, gradient_targets = None, []
    if (gradient_dir / "attribution_effect.npy").exists():
        gradients = np.load(gradient_dir / "attribution_effect.npy")
        gradient_targets = json.loads((gradient_dir / "metadata.json").read_text())["targets"]
    cards["gradient_source"] = ("dL/d(shift) x shift on matched replies (causal_panel_experiment attribution stage)"
                                if gradients is not None else None)
    site_gradients, site_tones, site_start, site_widths = None, [], {}, {}
    if (paths["root"] / "gradient_effect.npy").exists():            # Sarvam: this module's gradients stage
        site_gradients = np.load(paths["root"] / "gradient_effect.npy")
        meta = json.loads((paths["root"] / "gradient_meta.json").read_text())
        site_tones = meta["tones"]
        site_widths = {int(k): v for k, v in meta["sites"].items()}
        site_start = site_offsets(site_widths)
        cards["gradient_source"] = f"{meta['estimate']}; {meta['prompts']} discovery prompts"
    for panel, by_layer in payload["panels"].items():
        if MODELS[key]["source"] == "sarvam":
            from .sarvam_lab import attribution_scores
            scores, t_values, _ = attribution_scores(panel, statistics=True)
        neurons = {}
        for layer, item in by_layer.items():
            layer_index, ids, deltas = int(layer), item["neurons"], item["deltas"]
            if MODELS[key]["source"] == "sarvam":
                order = sorted(range(len(ids)), key=lambda i: -scores[layer_index][ids[i]])
            else:
                order = sorted(range(len(ids)), key=lambda i: -abs(deltas[i]))
            for rank, position in enumerate(order, start=1):
                neuron = int(ids[position])
                card = {"attribution_rank": rank, "delta": round(float(deltas[position]), 6)}
                if MODELS[key]["source"] == "sarvam":
                    card["attribution"] = round(float(scores[layer_index][neuron]), 6)
                    card["selectivity_t"] = round(float(t_values[layer_index][neuron]), 3)
                if gradients is not None and panel in gradient_targets:
                    effects = gradients[gradient_targets.index(panel), layer_index]
                    card["gradient_effect"] = round(float(effects[neuron]), 6)
                    # Rank of this neuron among all neurons of its layer by the estimate (1 = largest).
                    card["gradient_rank"] = int((effects > effects[neuron]).sum()) + 1
                if site_gradients is not None and panel in site_tones:
                    effects = site_gradients[site_tones.index(panel), site_start[layer_index]:site_start[layer_index] + site_widths[layer_index]]
                    card["gradient_effect"] = round(float(effects[neuron]), 6)
                    card["gradient_rank"] = int((effects > effects[neuron]).sum()) + 1
                neurons[f"{layer_index}:{neuron}"] = card
        cards["panels"][panel] = {"neurons": neurons}
    cards["panels"] = {name: {**item, "gradient_source": cards["gradient_source"]} for name, item in cards["panels"].items()}
    single = paths["root"] / "single" / "report.json"
    if single.exists():                                   # results of generation_analysis.single_neuron
        for panel, block in json.loads(single.read_text())["panels"].items():
            for row in block["neurons"]:
                card = cards["panels"].get(panel, {}).get("neurons", {}).get(f"{row['layer']}:{row['neuron']}")
                if card is not None and row["role"] == "panel":
                    card["single"] = {"status": row["status"], "screen": row["screen"]["mean"], "judged": (row.get("judged") or {}).get("mean")}
    paths["cards"].write_text(json.dumps(cards))
    print(f"wrote {paths['cards']}", flush=True)


# --------------------------------------------------------------------------- summary

def summary(args) -> None:
    rows, notes = [], []
    for key in MODELS:
        paths = model_paths(key)
        if not paths["report"].exists():
            notes.append(f"{MODELS[key]['label']}: validation battery not run.")
            continue
        report = json.loads(paths["report"].read_text())
        reference = f"dose_{report['reference_alpha']:g}"
        for language in ("all", "english", "hinglish"):
            if language == "all":
                blocks = [report["pooled"]]
            else:
                blocks = [block["by_language"][language] for block in report["panels"].values() if language in block["by_language"]]
            conditions = [row for block in blocks for row in block["conditions"] if row["name"] == reference]
            if not conditions:
                continue
            if language == "all":
                emotion = conditions[0]["emotion"]
                gain, low, high = emotion["mean"], emotion["lo"], emotion["hi"]
            else:
                gain = float(np.mean([c["emotion"]["mean"] for c in conditions]))
                low = float(np.mean([c["emotion"]["lo"] for c in conditions]))
                high = float(np.mean([c["emotion"]["hi"] for c in conditions]))
            quality = [np.mean([c["quality"][m]["mean"] for m in ("appropriateness", "empathy", "relevance", "fluency")
                                if c["quality"].get(m, {}).get("mean") is not None]) for c in conditions]
            rows.append({"model": MODELS[key]["label"], "model_key": key, "language": language,
                         "source": "validation battery (dev)", "n": int(sum(c["n"] for c in conditions)),
                         "steered_share": _num(np.mean([c["changed_share"] or 0 for c in conditions])),
                         "emotion_gain": _num(gain), "emotion_gain_lo": _num(low), "emotion_gain_hi": _num(high),
                         "quality_change": _num(np.mean(quality)), "harmed_share": _num(np.mean([c["harm_share"] or 0 for c in conditions])),
                         "policy": "validation battery: no policy applied"})
    out = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"), "rows": rows,
           "notes": notes + ["Each row compares the reference-strength panel with the unsteered reply of the same prompt.",
                             "Per-language rows average the panels' intervals (not a pooled interval)."]}
    target = RESULTS / "research" / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(f"wrote {target} ({len(rows)} rows)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=("prepare", "generate", "score", "analyse", "all", "summary", "gradients"))
    parser.add_argument("--gradient-prompts", type=int, default=40, help="Discovery prompts used for the Sarvam gradient estimate.")
    parser.add_argument("--model", choices=tuple(MODELS), default="llama")
    parser.add_argument("--limit", type=int, default=None, help="Development messages to use (default: all).")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Re-score even if ratings exist.")
    args = parser.parse_args()
    stages = ["prepare", "generate", "score", "analyse"] if "all" in args.stages else args.stages
    for stage in stages:
        {"prepare": prepare, "generate": generate, "score": score, "analyse": analyse, "summary": summary,
         "gradients": gradients}[stage](args)


if __name__ == "__main__":
    main()
