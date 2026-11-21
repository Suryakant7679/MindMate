#!/usr/bin/env python3
"""Sarvam-30B steering: message sets, neuron discovery, panels and the batched intervention generator.

Sarvam-30B is a 19-layer mixture-of-experts model.  Layer 0 has an ordinary MLP (8,192 neurons);
layers 1-18 have 128 routed experts (6 per token) and one *shared expert* (1,024 neurons) that runs
on every token.  The steerable "neurons" are the ones that are always active -- the layer-0 MLP and
the shared experts (26,624 in all) -- read and shifted at the input of their ``down_proj``
(``runtime.steering_module``).  No Llama neuron id, layer band or panel is used anywhere.

Method (the one developed for Llama 3.1-8B): matched prompts in which only the style sentence of
the system prompt varies; a neuron's shift is its mean decode-step activation under a response tone
minus its mean under the neutral style, paired over prompts; neurons are ranked by attribution
(shift x alignment of the neuron's write vector with the residual-stream direction of the tone).

Stages (run from the project root):
  MINDMATE_MODEL=llama python3 -m generation_analysis.sarvam_lab data      # plans need the detector only
  python3 -m generation_analysis.sarvam_lab discover panels
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from backend.config import STEERING_GAIN
from chatbot_feature.emotion_reasoner.knowledge_graph import is_hinglish

MODEL_KEY = "sarvam"
ROOT = Path("results/sarvam_30b_generation_emotion")
LAB = ROOT / "steering"
DATA = LAB / "data"
PANEL_FILE = ROOT / "steering_panels" / "panels.json"
SEED = 20261115

# Response tones (panel names) and the style sentence used to elicit them.
TONES = {
    "warmth": "compassionate warmth: gentle care, tenderness and sympathy for what the user is going through",
    "reassurance": "reassurance: calm, steady confidence that the user is safe and that this can be handled",
    "encouragement": "encouragement: belief in the user and recognition of their effort and strengths",
    "joy": "joy: happiness, delight and shared excitement",
    "admiration": "admiration: respect, praise and appreciation",
}
NEUTRAL = "neutral"
ENGLISH_ONLY = "neutral_english"            # neutral style, reply forced to English (language panel reference)
LANGUAGE_PANEL = "hinglish"
# Which tone panel a knowledge-graph strategy may use (calibration decides whether it is used at all).
STRATEGY_TONE = {"compassionate_comfort": "warmth", "warm_connection": "warmth", "reflective_validation": "warmth",
                 "gentle_exploration": "warmth", "grounding_reassurance": "reassurance",
                 "strengths_affirmation": "encouragement", "celebratory_sharing": "joy",
                 "appreciative_mirroring": "admiration"}
TEMPLATE = "You are a conversational assistant. Reply to the user's message in two or three sentences. {language} {style}"
LANGUAGE_RULE = {"hinglish": "Reply in natural Hinglish (Hindi in Roman letters mixed with English), the way the user writes.",
                 "english": "Reply in English.",
                 "english_only": "Reply in English only, with no Hindi words."}


def style_prompt(condition: str, language: str) -> str:
    """Matched system prompts: only the style sentence (and, for the language panel, the language rule) varies."""
    if condition == ENGLISH_ONLY:
        return TEMPLATE.format(language=LANGUAGE_RULE["english_only"],
                               style="Keep the reply emotionally neutral: calm, factual and free of emotional words.")
    style = ("Keep the reply emotionally neutral: calm, factual and free of emotional words." if condition == NEUTRAL
             else f"Make the reply clearly express {TONES[condition]}.")
    return TEMPLATE.format(language=LANGUAGE_RULE[language], style=style)


# ------------------------------------------------------------------ message sets

def _clean(text: str) -> str:
    emoji = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")
    return re.sub(r"\s+", " ", emoji.sub("", str(text)).replace("_comma_", ",")).strip()


def load_items(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f"{name}.csv")
    if "history" in frame:
        frame["history"] = frame.history.map(json.loads)
    return frame


def messages_of(row: Any) -> list[dict[str, str]]:
    return [*getattr(row, "history", []), {"role": "user", "content": row.text}]


def build_data() -> None:
    """Discovery, development, calibration and held-out messages for Sarvam (never rebuilt)."""
    from .steering_lab import Planner, _ed_items
    if (DATA / "heldout.csv").exists():
        print("Sarvam message sets exist; they are never rebuilt.")
        return
    DATA.mkdir(parents=True, exist_ok=True)
    v3, frozen = Path("results/steering_v3"), Path("results/frozen_v2/llama31_8b/causal_steering/splits")
    earlier = pd.concat([pd.read_csv(path) for path in (*sorted(frozen.glob("*.csv")), Path("results/steering_holdout2/items.csv"),
                                                        v3 / "extra_items.csv", v3 / "holdout3.csv",
                                                        v3 / "holdout3_hinglish_authored.csv")])
    used_texts, used_conversations = set(earlier.text.str.lower()), set(earlier.conv_id)

    # Discovery: Hinglish messages no experiment has used (synthetic file, then the Hinglish emotion corpus), plus English.
    synthetic = pd.read_csv("data/synthetic_hinglish_emotion_1000.csv")
    synthetic["text"] = synthetic.text.map(_clean)
    synthetic = synthetic[synthetic.text.map(is_hinglish) & ~synthetic.text.str.lower().isin(used_texts)]
    synthetic = synthetic.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 12), random_state=SEED))
    corpus = pd.read_csv("data/mlt_hinghlish_dataset_clean.csv").rename(columns={"hinglish_genz_text": "text"})
    corpus["text"] = corpus.text.map(_clean)
    corpus = corpus[corpus.text.map(is_hinglish) & corpus.text.str.len().between(35, 180)
                    & ~corpus.text.str.contains(r"\[name\]", case=False)].drop_duplicates("text")
    corpus = corpus.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 6), random_state=SEED))
    hinglish = pd.concat([synthetic[["text", "label"]], corpus[["text", "label"]]]).drop_duplicates("text").head(120)
    english = [item for item in _ed_items("train", SEED, 10 ** 6, 0) if item["conv_id"] not in used_conversations
               and item["text"].lower() not in used_texts][:50]
    discovery = pd.concat([hinglish.assign(language="hinglish"),
                           pd.DataFrame({"text": [item["text"] for item in english],
                                         "label": [item["gold"] for item in english], "language": "english"})],
                          ignore_index=True)
    discovery.insert(0, "item_id", [f"disc-{index:03d}" for index in range(len(discovery))])
    discovery.to_csv(DATA / "discovery.csv", index=False)
    used_texts |= set(discovery.text.str.lower())

    planner = Planner()

    def planned(texts: Sequence[str], labels: Sequence[str], source: str, prefix: str) -> pd.DataFrame:
        rows = []
        for index, (text, label) in enumerate(zip(texts, labels)):
            item = {"source": source, "conv_id": f"{prefix}:{index}", "gold": label, "language": "hinglish",
                    "turns": "single", "text": _clean(text), "history": []}
            item.update(planner.plan_fields(item))
            rows.append(item)
        frame = pd.DataFrame(rows)
        return frame[~frame.safety_flag.astype(bool) & frame.target.notna() & (frame.target != "")].drop(columns=["safety_flag"])

    authored = pd.read_csv("data/hinglish_sarvam_calibration.csv").sample(frac=1.0, random_state=SEED)
    authored = planned(authored.text, authored.label, "hinglish_sarvam_calibration.csv", "sarvam-cal")
    held_hinglish = pd.read_csv("data/hinglish_sarvam_heldout.csv")
    held_hinglish = planned(held_hinglish.text, held_hinglish.label, "hinglish_sarvam_heldout.csv", "sarvam-held")
    english_pool = [item for item in _ed_items("test", SEED, 10 ** 6, 0) if item["conv_id"] not in used_conversations
                    and item["text"].lower() not in used_texts]
    for item in english_pool[:400]:
        item.update(planner.plan_fields(item))
    planner.close()
    from .steering_lab import _balanced
    pool = pd.DataFrame(english_pool[:400])
    held_english = _balanced(pool[~pool.safety_flag.astype(bool) & pool.target.notna() & (pool.target != "")], 60) \
        .drop(columns=["safety_flag"])

    extra = pd.read_csv(v3 / "extra_items.csv")
    old = {name: pd.read_csv(frozen / f"{name}.csv") for name in ("dev", "calibration")}
    sets = {
        "dev": pd.concat([extra[extra.split == "dev_hinglish"], authored.iloc[:30], old["dev"].iloc[::2]]),
        "calibration": pd.concat([old["calibration"][old["calibration"].language == "hinglish"],
                                  extra[extra.split == "calibration_hinglish"], authored.iloc[30:],
                                  old["calibration"][old["calibration"].language == "english"].iloc[::2]]),
        "heldout": pd.concat([held_hinglish, held_english]),
    }
    seen: set[str] = set(discovery.text.str.lower())
    for name, frame in sets.items():
        frame = frame.reset_index(drop=True).copy()
        frame["history"] = frame.history.map(lambda value: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
        frame["item_id"] = [f"sarvam-{name}-{index:03d}" for index in range(len(frame))]
        frame["split"] = name
        frame["tone"] = frame.strategy.map(STRATEGY_TONE)
        texts = set(frame.text.str.lower())
        if texts & seen or frame.text.str.lower().duplicated().any():
            raise SystemExit(f"{name} overlaps another Sarvam message set.")
        if name == "heldout" and texts & used_texts:
            raise SystemExit("A held-out message was used in an earlier experiment.")
        seen |= texts
        frame.to_csv(DATA / f"{name}.csv", index=False)
        print(name, frame.groupby("language").size().to_dict(), frame.strategy.value_counts().to_dict())
    print("discovery", discovery.groupby("language").size().to_dict())


# ------------------------------------------------------------- model and hooks

def load_runtime():
    from backend.models import create_runtime
    runtime = create_runtime(MODEL_KEY)
    runtime.set_reasoning(False)                 # answers only: decode steps are answer tokens
    return runtime


def sites(runtime: Any) -> dict[int, int]:
    """Layer -> number of always-active neurons at its steering site."""
    return {layer: runtime.steering_module(layer).in_features for layer in range(len(runtime.model.model.layers))}


class Recorder:
    """Mean activation of every site neuron and of the residual stream over the decode steps of one generation."""

    def __init__(self, runtime: Any) -> None:
        self.runtime, self.handles = runtime, []
        self.widths = sites(runtime)
        self.reset()

    def reset(self) -> None:
        self.sums: dict[int, Any] = {}
        self.residual: dict[int, Any] = {}
        self.steps: dict[int, int] = {}
        self.prompt: dict[Any, bool] = {}

    def __enter__(self) -> "Recorder":
        def neuron_hook(layer: int):
            def hook(_module, inputs):
                if not self.prompt.get(("n", layer)):
                    self.prompt[("n", layer)] = True
                    return
                values = inputs[0][0, -1].float()
                self.sums[layer] = self.sums.get(layer, 0) + values
                self.steps[layer] = self.steps.get(layer, 0) + 1
            return hook

        def residual_hook(layer: int):
            def hook(_module, _inputs, output):
                if not self.prompt.get(("r", layer)):
                    self.prompt[("r", layer)] = True
                    return
                hidden = output[0] if isinstance(output, tuple) else output
                self.residual[layer] = self.residual.get(layer, 0) + hidden[0, -1].float()
            return hook

        for layer in self.widths:
            self.handles.append(self.runtime.steering_module(layer).register_forward_pre_hook(neuron_hook(layer)))
            self.handles.append(self.runtime.model.model.layers[layer].register_forward_hook(residual_hook(layer)))
        return self

    def __exit__(self, *_exc) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def means(self) -> tuple[np.ndarray, np.ndarray]:
        steps = max(next(iter(self.steps.values()), 0), 1)
        neurons = torch.cat([self.sums[layer] / steps for layer in sorted(self.widths)]).cpu().numpy().astype(np.float16)
        residual = torch.stack([self.residual[layer] / steps for layer in sorted(self.widths)]).cpu().numpy().astype(np.float16)
        return neurons, residual


def offsets(widths: dict[int, int]) -> dict[int, int]:
    start, out = 0, {}
    for layer in sorted(widths):
        out[layer] = start
        start += widths[layer]
    return out


Panel = dict[int, tuple[list[int], list[float]]]


def generate_rows(runtime: Any, messages: list[dict[str, str]], system_prompt: str,
                  rows: Sequence[dict[str, Any]], max_new_tokens: int = 100,
                  sample: dict[str, Any] | None = None) -> list[str]:
    """Greedy replies to one prompt under several interventions, decoded together.

    Each row is a dict: ``{"shift": [(panel, alpha), ...]}`` adds alpha * gain * delta to the panel
    neurons (several panels add up; negative alpha reverses); ``{"clamp": {layer: (neurons, values)}}``
    holds neurons at fixed values (ablation).  An empty dict is the unsteered reply.  Hooks act on
    decode steps only and are removed afterwards; model weights are never changed.

    The function works for any runtime with ``steering_module`` (Llama-style MLP sites as well as Sarvam's
    shared experts).  ``sample={"seed": s, "temperature": t, "top_p": p}`` replaces greedy decoding by
    sampling with a fixed seed (each row draws its own tokens).
    """
    widths = sites(runtime)
    encoded = runtime._encode_conversation(messages, system_prompt, max_new_tokens)
    input_ids = encoded["input_ids"].repeat(len(rows), 1)
    attention = encoded["attention_mask"].repeat(len(rows), 1)
    device = input_ids.device
    delta = {layer: torch.zeros(len(rows), widths[layer]) for layer in widths}
    clamp_mask = {layer: torch.zeros(len(rows), widths[layer], dtype=torch.bool) for layer in widths}
    clamp_value = {layer: torch.zeros(len(rows), widths[layer]) for layer in widths}
    for index, row in enumerate(rows):
        for panel, alpha in row.get("shift", []):
            for layer, (neurons, shifts) in panel.items():
                delta[layer][index, neurons] += float(alpha) * STEERING_GAIN * torch.tensor(shifts)
        for layer, (neurons, values) in row.get("clamp", {}).items():
            clamp_mask[layer][index, neurons] = True
            clamp_value[layer][index, neurons] = torch.tensor(values)
    # ``{"sham": True}`` installs the hooks on every site with a zero shift (control for the hook mechanics).
    sham = any(row.get("sham") for row in rows)
    active = [layer for layer in widths if sham or bool(delta[layer].any()) or bool(clamp_mask[layer].any())]
    seen: dict[int, bool] = {}
    # Residual-stream steering: {"residual": [(layer, unit direction, alpha)]} adds alpha * ||h|| * direction to the
    # output of that layer and rescales to the original norm.
    residual: dict[int, Any] = {}
    for index, row in enumerate(rows):
        for layer, unit, alpha in row.get("residual", []):
            if layer not in residual:
                residual[layer] = torch.zeros(len(rows), len(unit))
            residual[layer][index] += float(alpha) * torch.as_tensor(unit, dtype=torch.float32)

    def residual_hook(layer: int):
        state: dict[str, Any] = {}

        def hook(_module, _inputs, output):
            if not state.get("prompt"):
                state["prompt"] = True
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            values = hidden.float()
            norm = values.norm(dim=-1, keepdim=True)
            shift = residual[layer].to(values.device)[:, None, :]
            updated = values + norm * shift
            updated = updated * (norm / updated.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            updated = torch.where(shift.abs().sum(-1, keepdim=True) > 0, updated, values).to(hidden.dtype)
            return (updated, *output[1:]) if isinstance(output, tuple) else updated
        return hook

    def hook_for(layer: int):
        cache: dict[str, Any] = {}

        def hook(_module, inputs):
            if not seen.get(layer):
                seen[layer] = True
                return inputs
            hidden = inputs[0]
            if not cache:
                cache["delta"] = delta[layer].to(device=hidden.device, dtype=hidden.dtype)[:, None, :]
                cache["mask"] = clamp_mask[layer].to(hidden.device)[:, None, :]
                cache["value"] = clamp_value[layer].to(device=hidden.device, dtype=hidden.dtype)[:, None, :]
            modified = torch.where(cache["mask"], cache["value"], hidden) + cache["delta"]
            return (modified, *inputs[1:])
        return hook

    handles = [runtime.steering_module(layer).register_forward_pre_hook(hook_for(layer)) for layer in active]
    handles += [runtime.model.model.layers[layer].register_forward_hook(residual_hook(layer)) for layer in residual]
    decoding: dict[str, Any] = {"do_sample": False}
    if sample:
        torch.manual_seed(int(sample["seed"]))
        decoding = {"do_sample": True, "temperature": float(sample.get("temperature", 0.7)),
                    "top_p": float(sample.get("top_p", 0.9))}
    try:
        with runtime.lock, torch.inference_mode():
            output = runtime.model.generate(input_ids=input_ids, attention_mask=attention, max_new_tokens=max_new_tokens,
                                            pad_token_id=runtime.tokenizer.pad_token_id, **decoding,
                                            **runtime.generation_settings())
    finally:
        for handle in handles:
            handle.remove()
    stop = set(runtime.generation_settings().get("eos_token_id", [])) | {runtime.tokenizer.eos_token_id}
    configured = getattr(getattr(runtime.model, "generation_config", None), "eos_token_id", None)
    stop |= {token for token in (configured if isinstance(configured, (list, tuple)) else [configured]) if isinstance(token, int)}
    replies = []
    for index in range(len(rows)):
        tokens = output[index, input_ids.shape[1]:]
        ends = [position for position, token in enumerate(tokens.tolist()) if token in stop]
        replies.append(runtime.decode_response(tokens[:ends[0]] if ends else tokens))
    return replies


# ------------------------------------------------------------------- discovery

def discover(args) -> None:
    """Matched-prompt replies and their mean decode-step activations (one generation per prompt and condition)."""
    out = LAB / "discovery"
    out.mkdir(parents=True, exist_ok=True)
    items = load_items("discovery")
    done = pd.read_csv(out / "index.csv") if (out / "index.csv").exists() else pd.DataFrame(columns=["item_id", "condition"])
    finished = set(zip(done.item_id, done.condition))
    runtime = load_runtime()
    index_rows = done.to_dict("records")
    neurons = list(np.load(out / "neurons.npy")) if (out / "neurons.npy").exists() else []
    residuals = list(np.load(out / "residual.npy")) if (out / "residual.npy").exists() else []
    recorder = Recorder(runtime)
    for count, row in enumerate(items.itertuples()):
        conditions = [NEUTRAL, *TONES] + ([ENGLISH_ONLY] if row.language == "hinglish" else [])
        for condition in conditions:
            if (row.item_id, condition) in finished:
                continue
            recorder.reset()
            encoded = runtime._encode_conversation([{"role": "user", "content": row.text}],
                                                   style_prompt(condition, row.language), args.max_new_tokens)
            with recorder, runtime.lock, torch.inference_mode():
                output = runtime.model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False,
                                                pad_token_id=runtime.tokenizer.pad_token_id, **runtime.generation_settings())
            reply = runtime.decode_response(output[0, encoded["input_ids"].shape[1]:])
            vector, residual = recorder.means()
            neurons.append(vector)
            residuals.append(residual)
            index_rows.append({"item_id": row.item_id, "condition": condition, "language": row.language, "reply": reply,
                               "reply_is_hinglish": is_hinglish(reply), "tokens": int(output.shape[1] - encoded["input_ids"].shape[1])})
        if count % 10 == 9 or count == len(items) - 1:          # checkpoint
            np.save(out / "neurons.npy", np.stack(neurons))
            np.save(out / "residual.npy", np.stack(residuals))
            pd.DataFrame(index_rows).to_csv(out / "index.csv", index=False)
            print(f"discovery {count + 1}/{len(items)}", flush=True)
    (out / "sites.json").write_text(json.dumps({str(k): v for k, v in sites(runtime).items()}))
    # Write vectors of every site neuron (the dequantised down_proj columns), for the attribution score.
    writes = []
    with torch.inference_mode():
        for layer, width in sorted(sites(runtime).items()):
            module = runtime.steering_module(layer)
            eye = torch.eye(width, device=next(module.parameters()).device, dtype=torch.bfloat16)
            writes.append(torch.cat([module(eye[start:start + 1024]).float().cpu() for start in range(0, width, 1024)]))
    torch.save(writes, out / "write_vectors.pt")


def build_panels(args) -> None:
    """Paired shifts, attribution ranking and the panel file the chat backend loads."""
    out = LAB / "discovery"
    index = pd.read_csv(out / "index.csv")
    neurons = np.load(out / "neurons.npy").astype(np.float32)
    residual = np.load(out / "residual.npy").astype(np.float32)
    widths = {int(k): v for k, v in json.loads((out / "sites.json").read_text()).items()}
    start = offsets(widths)
    writes = torch.load(out / "write_vectors.pt")
    row_of = {(row.item_id, row.condition): position for position, row in enumerate(index.itertuples())}
    panels: dict[str, Panel] = {}
    full: dict[str, Panel] = {}
    directions: dict[str, np.ndarray] = {}          # unit residual-stream direction after each layer
    summary = []

    def paired(condition: str, reference: str, keep) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        pairs = [(row_of[(item, condition)], row_of[(item, reference)]) for item in index.item_id.unique()
                 if (item, condition) in row_of and (item, reference) in row_of and keep(item)]
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        difference = neurons[a] - neurons[b]
        shift = difference.mean(0)
        t = shift / (difference.std(0, ddof=1) / np.sqrt(len(pairs)) + 1e-8)
        direction = (residual[a] - residual[b]).mean(0)                 # [layers, hidden]
        return shift, t, direction, len(pairs)

    language = dict(zip(index.item_id, index.language))
    hinglish_reply = {(row.item_id, row.condition): bool(row.reply_is_hinglish) for row in index.itertuples()}
    specs = {tone: (tone, NEUTRAL, lambda item: True) for tone in TONES}
    # Language panel: the same neutral reply written in Hinglish minus written in English.
    specs[LANGUAGE_PANEL] = (NEUTRAL, ENGLISH_ONLY, lambda item: language[item] == "hinglish"
                             and hinglish_reply[(item, NEUTRAL)] and not hinglish_reply[(item, ENGLISH_ONLY)])
    for name, (condition, reference, keep) in specs.items():
        shift, t, direction, pairs = paired(condition, reference, keep)
        panel, everything = {}, {}
        for layer, width in sorted(widths.items()):
            block = slice(start[layer], start[layer] + width)
            unit = direction[layer] / (np.linalg.norm(direction[layer]) + 1e-8)
            alignment = (writes[layer].numpy() @ unit)                  # write vector . tone direction after this layer
            score = shift[block] * alignment
            # Reliable neurons only (paired |t| >= 3), ranked by attribution; top share per layer.
            reliable = np.abs(t[block]) >= 3.0
            ranked = [int(i) for i in np.argsort(-score) if reliable[i] and score[i] > 0]
            chosen = sorted(ranked[:max(1, round(width * args.top_percent / 100))])
            panel[layer] = (chosen, [float(shift[block][i]) for i in chosen])
            everything[layer] = (list(range(width)), [float(value) for value in shift[block]])
            summary.append({"panel": name, "layer": layer, "site_neurons": width, "pairs": pairs, "selected": len(chosen),
                            "reliable_neurons": int(reliable.sum()), "shift_norm": float(np.linalg.norm(shift[block])),
                            "selected_shift_norm": float(np.linalg.norm([shift[block][i] for i in chosen])),
                            "residual_direction_norm": float(np.linalg.norm(direction[layer])),
                            "attribution_selected": float(sum(score[i] for i in chosen)),
                            "attribution_all_positive": float(score[score > 0].sum())})
        panels[name], full[name] = panel, everything
        directions[name] = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
    pd.DataFrame(summary).to_csv(LAB / "panel_summary.csv", index=False)
    np.savez(LAB / "residual_directions.npz", **directions)
    # Mean activation of every site neuron in the neutral style (the value an ablation holds a neuron at).
    np.save(out / "neutral_means.npy", neurons[(index.condition == NEUTRAL).to_numpy()].mean(0))
    replies = index.groupby(["language", "condition"]).agg(n=("reply", "size"), hinglish_share=("reply_is_hinglish", "mean"),
                                                           tokens=("tokens", "mean")).reset_index()
    replies.to_csv(out / "condition_summary.csv", index=False)

    def as_json(panel: Panel) -> dict:
        return {str(layer): {"neurons": neurons_, "deltas": deltas} for layer, (neurons_, deltas) in panel.items() if neurons_}
    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": "sarvamai/sarvam-30b", "layers": [0, max(widths)], "top_percent": args.top_percent,
               "site": "layer 0: MLP intermediate (8,192 neurons); layers 1-18: shared-expert intermediate (1,024 neurons)",
               "selection": "paired |t| >= 3 over matched prompts, then top attribution (shift x write-vector alignment "
                            "with the tone's residual direction) per layer",
               "delta": "decode-step mean activation(tone) - mean activation(neutral), paired over prompts; "
                        "hinglish: neutral reply in Hinglish - neutral reply in English",
               "intervention": "add alpha * gain * delta to the input of the site's down_proj on decode steps",
               "discovery": {"prompts": int(index.item_id.nunique()), "hinglish_prompts": int((index.drop_duplicates("item_id").language == "hinglish").sum())},
               "strategy_tone": STRATEGY_TONE, "panels": {name: as_json(panel) for name, panel in panels.items()}}
    PANEL_FILE.write_text(json.dumps(payload))
    (LAB / "panels_full.json").write_text(json.dumps({name: as_json(panel) for name, panel in full.items()}))
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print(replies.round(2).to_string(index=False))
    table = pd.DataFrame(summary)
    print(table.groupby("panel").agg(pairs=("pairs", "first"), selected=("selected", "sum"), reliable=("reliable_neurons", "sum"),
                                     shift_norm=("shift_norm", lambda v: float(np.sqrt((v ** 2).sum())))).round(2).to_string())
    print(table.pivot(index="layer", columns="panel", values="selected").to_string())


def attribution_ranking(name: str) -> dict[int, list[int]]:
    """Panel neurons of ``name`` per layer, most attributed first (the ranking ``build_panels`` selected them by)."""
    scores = attribution_scores(name)
    return {layer: sorted(neurons, key=lambda neuron: -scores[layer][neuron])
            for layer, (neurons, _) in load_panels()[name].items()}


def attribution_scores(name: str, statistics: bool = False):
    """Attribution score of every site neuron for one panel: paired shift x alignment of the neuron's write
    vector with the panel's residual direction, per layer.  With ``statistics`` also the paired t value of
    the activation difference (the selectivity evidence) and the shift itself."""
    out = LAB / "discovery"
    index = pd.read_csv(out / "index.csv")
    neurons = np.load(out / "neurons.npy").astype(np.float32)
    residual = np.load(out / "residual.npy").astype(np.float32)
    widths = {int(k): v for k, v in json.loads((out / "sites.json").read_text()).items()}
    start = offsets(widths)
    writes = torch.load(out / "write_vectors.pt")
    row_of = {(row.item_id, row.condition): position for position, row in enumerate(index.itertuples())}
    language = dict(zip(index.item_id, index.language))
    hinglish_reply = {(row.item_id, row.condition): bool(row.reply_is_hinglish) for row in index.itertuples()}
    if name == LANGUAGE_PANEL:
        condition, reference = NEUTRAL, ENGLISH_ONLY
        keep = lambda item: language[item] == "hinglish" and hinglish_reply[(item, NEUTRAL)] and not hinglish_reply[(item, ENGLISH_ONLY)]
    else:
        condition, reference, keep = name, NEUTRAL, (lambda item: True)
    pairs = [(row_of[(item, condition)], row_of[(item, reference)]) for item in index.item_id.unique()
             if (item, condition) in row_of and (item, reference) in row_of and keep(item)]
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    difference = neurons[a] - neurons[b]
    shift = difference.mean(0)
    t = shift / (difference.std(0, ddof=1) / np.sqrt(len(pairs)) + 1e-8)
    direction = (residual[a] - residual[b]).mean(0)
    scores, t_values, shifts = {}, {}, {}
    for layer, width in sorted(widths.items()):
        block = slice(start[layer], start[layer] + width)
        unit = direction[layer] / (np.linalg.norm(direction[layer]) + 1e-8)
        scores[layer] = shift[block] * (writes[layer].numpy() @ unit)
        t_values[layer], shifts[layer] = t[block], shift[block]
    return (scores, t_values, shifts) if statistics else scores


def load_panels(full: bool = False) -> dict[str, Panel]:
    payload = json.loads((LAB / "panels_full.json").read_text()) if full else json.loads(PANEL_FILE.read_text())["panels"]
    return {name: {int(layer): (item["neurons"], item["deltas"]) for layer, item in by_layer.items()}
            for name, by_layer in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=("data", "discover", "panels"))
    parser.add_argument("--max-new-tokens", type=int, default=70)
    parser.add_argument("--top-percent", type=float, default=5.0, help="Share of each site's neurons kept per panel.")
    args = parser.parse_args()
    for stage in args.stages:
        {"data": build_data, "discover": lambda: discover(args), "panels": lambda: build_panels(args)}[stage]()


if __name__ == "__main__":
    main()
