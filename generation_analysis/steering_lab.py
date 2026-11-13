"""Shared infrastructure for the causal steering experiments.

Used by ``causal_panel_experiment``, ``recalibrate_policy`` and
``heldout_comparison``.  Everything here is deliberately independent of the
deployed run-time gate so that evaluation cannot inherit its selection bias:

data      three strictly separated message sets, built once and frozen
            dev          EmpatheticDialogues *train*       neuron / panel experiments
            calibration  EmpatheticDialogues *validation*  steering-policy selection
            test         EmpatheticDialogues *test*        final comparison, read once
          plus a Hinglish stratum per policy split (see ``build_splits``).
generate  batched greedy decoding through the real backend runtime and its
          steering hook, plus a clamp hook for ablations.
score     gate classifier (XLM-R, the model the deployed service selects with),
          two evaluators the gate never sees (a third-party GoEmotions RoBERTa
          and Qwen rubric ratings of the target emotion), Qwen quality rubrics
          and an order-balanced pairwise preference.
stats     paired bootstrap confidence intervals and Wilcoxon tests.
"""
from __future__ import annotations

import gc
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from backend.config import STEERING_GAIN, SYSTEM_PROMPT
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, strategy_prompt
from chatbot_feature.emotion_reasoner.knowledge_graph import is_hinglish

from .causal_validation import GOEMOTIONS, SCORER
from .common import EMOTION_DESCRIPTIONS, MODEL_ID, RESULTS_DIR, write_json

LAB_DIR = RESULTS_DIR / "causal_steering"
SPLIT_DIR = LAB_DIR / "splits"
ED_DIR = Path("data/external/empathetic_dialogues/default")
HINGLISH_CSV = Path("data/synthetic_hinglish_emotion_1000.csv")
INDEPENDENT_SCORER = "SamLowe/roberta-base-go_emotions"
JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDY_LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness",
                "surprise")
TARGETS = ("love", "admiration", "joy")          # response emotions the knowledge graph can ask for
QUALITY = ("empathy", "relevance", "fluency")
RUBRICS = {
    "empathy": "Empathy: does the reply show that it understands and cares about the user's feelings? "
               "1 = cold, dismissive or self-centred. 5 = deeply understanding and caring.",
    "relevance": "Contextual relevance: does the reply respond to what the user actually said, in a way that suits "
                 "their situation? 1 = off-topic, contradictory or inappropriate. 5 = fully on point and appropriate.",
    "fluency": "Fluency: is the reply natural, coherent and grammatical in its own language (English or romanised "
               "Hindi)? 1 = garbled, repetitive or nonsensical. 5 = perfectly natural.",
}


def emotion_rubric(target: str) -> str:
    return (f"Emotional tone: how strongly does the reply itself express {EMOTION_DESCRIPTIONS[target]}? "
            "Judge only the tone of the reply, not whether it is a good reply. "
            "1 = not at all. 5 = very strongly.")


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------ splits

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).replace("_comma_", ",")).strip()


def _ed_items(split: str, seed: int, single: int, multi: int) -> list[dict[str, Any]]:
    """Candidate items from one EmpatheticDialogues split (more than needed; stratified later)."""
    frame = pd.read_parquet(ED_DIR / split / "0000.parquet").sort_values(["conv_id", "utterance_idx"])
    rng = np.random.default_rng(seed)
    conversations = [group for _, group in frame.groupby("conv_id", sort=True)]
    order = rng.permutation(len(conversations))
    items: list[dict[str, Any]] = []
    wanted = {"single": single, "multi": multi}
    for position in order:
        turns = conversations[position]
        texts = [_clean(text) for text in turns.utterance]
        if any(not 25 <= len(text) <= 240 for text in texts[:3]):
            continue
        kind = "multi" if len(texts) >= 3 and wanted["multi"] > 0 and rng.random() < 0.35 else "single"
        if wanted[kind] <= 0:
            continue
        wanted[kind] -= 1
        history = ([{"role": "user", "content": texts[0]}, {"role": "assistant", "content": texts[1]}]
                   if kind == "multi" else [])
        items.append({"source": f"empathetic_dialogues/{split}", "conv_id": turns.conv_id.iloc[0],
                      "gold": turns.context.iloc[0], "language": "english", "turns": kind,
                      "text": texts[2] if kind == "multi" else texts[0], "history": history})
        if not wanted["single"] and not wanted["multi"]:
            break
    return items


def _hinglish_items(part: int, count: int, seed: int) -> list[dict[str, Any]]:
    """Romanised-Hindi messages; even rows feed calibration, odd rows the test set."""
    frame = pd.read_csv(HINGLISH_CSV)
    frame = frame[(frame.index % 2 == part) & frame.text.map(is_hinglish) & (frame.label != "disgust")]
    frame = frame.sample(n=min(count, len(frame)), random_state=seed)
    return [{"source": HINGLISH_CSV.name, "conv_id": f"synthetic:{index}", "gold": row.label,
             "language": "hinglish", "turns": "single", "text": _clean(row.text), "history": []}
            for index, row in frame.iterrows()]


def build_splits(sizes: dict[str, dict[str, int]] | None = None, seed: int = 20261111) -> dict[str, pd.DataFrame]:
    """Create and freeze dev / calibration / test.  Existing files are never rebuilt.

    Candidates are planned by the deployed detector + Emotion Reasoner and then
    balanced over the planned response strategy, which uses no outcome data.
    Messages that trigger the crisis path are left out: they are never steered
    (covered by unit tests instead).
    """
    sizes = sizes or {"dev": {"single": 48, "multi": 0, "hinglish": 0},
                      "calibration": {"single": 96, "multi": 32, "hinglish": 32},
                      "test": {"single": 96, "multi": 32, "hinglish": 32}}
    if all((SPLIT_DIR / f"{name}.csv").exists() for name in sizes):
        return {name: load_split(name) for name in sizes}
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    planner = Planner()
    result = {}
    ed_name = {"dev": "train", "calibration": "validation", "test": "test"}
    for offset, (name, size) in enumerate(sizes.items()):
        pool = _ed_items(ed_name[name], seed + offset, 6 * size["single"], 6 * size["multi"])
        if size["hinglish"]:
            pool += _hinglish_items(0 if name == "calibration" else 1, 4 * size["hinglish"], seed + offset)
        for item in pool:
            item.update(planner.plan_fields(item))
        frame = pd.DataFrame([item for item in pool if not item["safety_flag"] and item["target"]])
        chosen = []
        for stratum, quota in (("single", size["single"]), ("multi", size["multi"]), ("hinglish", size["hinglish"])):
            part = frame[(frame.language == "hinglish") if stratum == "hinglish"
                         else ((frame.language == "english") & (frame.turns == stratum))]
            chosen.append(_balanced(part, quota))
        frame = pd.concat(chosen).reset_index(drop=True)
        frame.insert(0, "item_id", [f"{name}-{index:03d}" for index in range(len(frame))])
        frame.insert(1, "split", name)
        frame["history"] = frame.history.map(lambda value: json.dumps(value, ensure_ascii=False))
        frame.drop(columns=["safety_flag"]).to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        result[name] = load_split(name)
    overlap = set(result["calibration"].conv_id) & set(result["test"].conv_id) | \
        set(result["dev"].conv_id) & (set(result["calibration"].conv_id) | set(result["test"].conv_id))
    texts = pd.concat(result.values()).text.str.lower()
    if overlap or texts.duplicated().any():
        raise SystemExit("Splits overlap; refusing to continue.")
    write_json(SPLIT_DIR / "metadata.json", {
        "seed": seed, "sizes": sizes, "detector": planner.detector,
        "sources": {"dev": "EmpatheticDialogues train", "calibration": "EmpatheticDialogues validation + "
                    f"{HINGLISH_CSV.name} even rows", "test": f"EmpatheticDialogues test + {HINGLISH_CSV.name} odd rows"},
        "balancing": "round-robin over the planned response strategy within each stratum",
        "excluded": "messages on the crisis path (never steered)",
        "counts": {name: frame.groupby(["language", "turns", "strategy"]).size().reset_index(name="n")
                   .to_dict("records") for name, frame in result.items()},
    })
    planner.close()
    return result


def _balanced(frame: pd.DataFrame, quota: int) -> pd.DataFrame:
    """Round-robin over strategies so that rare strategies are represented."""
    groups = {key: list(group.index) for key, group in frame.groupby("strategy", sort=True)}
    picked: list[int] = []
    while len(picked) < quota and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(picked) < quota:
                picked.append(groups[key].pop(0))
    return frame.loc[picked]


def load_split(name: str) -> pd.DataFrame:
    frame = pd.read_csv(SPLIT_DIR / f"{name}.csv")
    frame["history"] = frame.history.map(json.loads)
    return frame


def messages_of(row: Any) -> list[dict[str, str]]:
    return [*row.history, {"role": "user", "content": row.text}]


class Planner:
    """The deployed planning path (XLM-R detector fused with the knowledge graph) without the LLM."""

    def __init__(self, policy: dict[str, Any] | None = None) -> None:
        from backend.emotion_service import HISTORY_TURNS, ReplyEmotionScorer

        class _Torch:               # the scorer only needs ``runtime.torch``
            pass
        holder = _Torch()
        holder.torch = torch
        self.scorer = ReplyEmotionScorer(holder)
        if not self.scorer.uses_classifier:
            raise SystemExit("The gate classifier could not be loaded; plans would not match deployment.")
        from .common import EXPERIMENT_SPEC
        panels = json.loads((RESULTS_DIR / "steering_panels" / EXPERIMENT_SPEC.config.panel_file).read_text())["panels"]
        self.agent = EmotionReasonerAgent(available_panels=list(panels), policy=policy)
        self.history_turns = HISTORY_TURNS
        self.detector = f"{self.scorer.model_id} fused with knowledge-graph cues"

    def scores(self, text: str) -> dict[str, float]:
        values = self.scorer.score(text)
        total = sum(values.values()) or 1.0
        return {label: value / total for label, value in values.items()}

    def plan(self, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        users = [message["content"] for message in messages if message["role"] == "user"]
        history = []
        for text in users[:-1][-self.history_turns:]:
            state = self.agent.understand(text, self.scores(text))
            history.append({"emotion": state["emotion"], "valence": state["valence"]})
        return self.agent.plan(users[-1], self.scores(users[-1]), history).to_dict()

    def plan_fields(self, item: dict[str, Any]) -> dict[str, Any]:
        plan = self.plan([*item["history"], {"role": "user", "content": item["text"]}])
        return {"user_emotion": plan["user_emotion"], "intensity": plan["intensity"], "valence": plan["valence"],
                "strategy": plan["strategy"], "target": plan["response_emotion"], "safety_flag": plan["safety_flag"],
                "kg_prompt": strategy_prompt(SYSTEM_PROMPT, plan)}

    def close(self) -> None:
        self.scorer._model = None
        release()


# -------------------------------------------------------------------- generation

def load_runtime(model_id: str = MODEL_ID):
    from backend.models import create_runtime
    from .common import MODEL_KEY
    runtime = create_runtime(MODEL_KEY, model_id=model_id)      # the backend's own loader; never downloads
    if not runtime._emotion_panels:
        raise SystemExit("No steering panels. Run: python3 -m generation_analysis.build_steering_panels")
    runtime.tokenizer.padding_side = "left"
    return runtime


def panel_scores_order(panel: dict[int, tuple[list[int], list[float]]]) -> dict[int, tuple[list[int], list[float]]]:
    """Panels are stored in descending attribution order (``topk``); this is an explicit alias."""
    return panel


def subset_panel(panel: dict[int, tuple[list[int], list[float]]], *, layers: Iterable[int] | None = None,
                 fraction: float = 1.0, order: str = "top", rng: np.random.Generator | None = None
                 ) -> dict[int, tuple[list[int], list[float]]]:
    """Part of a panel: some layers and/or a fraction of each layer's neurons.

    ``order``: "top" / "bottom" by the panel's attribution rank, or "random".
    """
    keep = set(panel) if layers is None else set(layers)
    result = {}
    for layer, (neurons, shifts) in panel.items():
        if layer not in keep:
            continue
        count = max(1, round(fraction * len(neurons)))
        if order == "top":
            index = list(range(count))
        elif order == "bottom":
            index = list(range(len(neurons) - count, len(neurons)))
        else:
            index = sorted(rng.choice(len(neurons), count, replace=False).tolist())
        result[layer] = ([neurons[i] for i in index], [shifts[i] for i in index])
    return result


def random_panel(panel: dict[int, tuple[list[int], list[float]]], width: int, rng: np.random.Generator,
                 exclude: dict[int, set[int]] | None = None) -> dict[int, tuple[list[int], list[float]]]:
    """Same layers, counts and shift values on randomly chosen neurons outside the panel."""
    result = {}
    for layer, (neurons, shifts) in panel.items():
        banned = set(neurons) | (exclude or {}).get(layer, set())
        candidates = np.setdiff1d(np.arange(width), np.fromiter(banned, dtype=np.int64))
        result[layer] = (rng.choice(candidates, len(neurons), replace=False).tolist(),
                         [float(value) for value in rng.permutation(shifts)])
    return result


def panel_size(panel: dict[int, tuple[list[int], list[float]]]) -> int:
    return sum(len(neurons) for neurons, _ in panel.values())


@contextmanager
def clamp_neurons(runtime: Any, values: dict[int, tuple[list[int], list[float]]]):
    """Ablation hook: on decode steps set the given neurons to fixed values (their reference mean).

    Same location and phase convention as the steering hook (input of
    ``mlp.down_proj``; the first call of each layer is the prompt and is left alone).
    """
    handles, seen = [], {}

    def hook_for(layer: int, neurons: list[int], targets: list[float]):
        cache: dict[str, Any] = {}

        def hook(_module, inputs):
            if not seen.get(layer):
                seen[layer] = True
                return inputs
            hidden = inputs[0]
            if not cache:
                cache["index"] = torch.tensor(neurons, device=hidden.device, dtype=torch.long)
                cache["value"] = torch.tensor(targets, device=hidden.device, dtype=hidden.dtype)
            modified = hidden.clone()
            modified[..., cache["index"]] = cache["value"]
            return (modified, *inputs[1:])
        return hook

    try:
        for layer, (neurons, targets) in values.items():
            module = runtime.model.model.layers[layer].mlp.down_proj
            handles.append(module.register_forward_pre_hook(hook_for(layer, neurons, targets)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def generate_batch(runtime: Any, conversations: Sequence[Sequence[dict[str, str]]], system_prompts: Sequence[str],
                   *, alpha: float = 0.0, panel: str | None = None,
                   clamp: dict[int, tuple[list[int], list[float]]] | None = None,
                   max_new_tokens: int = 120, batch_size: int = 24) -> list[str]:
    """Greedy replies for many conversations through the backend runtime and its steering hook.

    ``panel`` names an entry of ``runtime._emotion_panels`` (experiments register
    variants there).  Batches are left-padded and sorted by length; the result
    order matches the input order.
    """
    tokenizer = runtime.tokenizer
    prompts = [tokenizer.apply_chat_template([{"role": "system", "content": system}, *messages], tokenize=False,
                                             add_generation_prompt=True)
               for messages, system in zip(conversations, system_prompts)]
    order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
    device = next(runtime.model.parameters()).device
    replies: list[str] = [""] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        encoded = tokenizer([prompts[index] for index in chunk], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
        steering = runtime._emotion_steering(alpha, panel) if alpha and panel else _noop()
        ablation = clamp_neurons(runtime, clamp) if clamp else _noop()
        with runtime.lock, torch.inference_mode(), steering, ablation:
            output = runtime.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                            pad_token_id=tokenizer.pad_token_id)
        for row, index in enumerate(chunk):
            replies[index] = runtime.decode_response(output[row, encoded["input_ids"].shape[1]:])
    return replies


@contextmanager
def _noop():
    yield


# ----------------------------------------------------------------------- scoring

_CLASSIFIERS: dict[str, Any] = {}


def classifier_scores(texts: Sequence[str], model_id: str) -> np.ndarray:
    """Sigmoid scores [n, 28] in GoEmotions order from a multi-label classifier."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if model_id not in _CLASSIFIERS:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id).to("cuda").eval()
        names = [model.config.id2label[i] for i in range(model.config.num_labels)]
        if names[0].startswith("LABEL_"):
            column = list(range(len(GOEMOTIONS)))
        else:
            column = [names.index(label) for label in GOEMOTIONS]
        _CLASSIFIERS[model_id] = (tokenizer, model, column)
    tokenizer, model, column = _CLASSIFIERS[model_id]
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            encoded = tokenizer([text or " " for text in texts[start:start + 64]], return_tensors="pt", padding=True,
                                truncation=True, max_length=256).to("cuda")
            rows.append(torch.sigmoid(model(**encoded).logits.float()).cpu().numpy()[:, column])
    return np.concatenate(rows) if rows else np.zeros((0, len(GOEMOTIONS)))


def target_scores(replies: Sequence[str], targets: Sequence[str], model_id: str) -> np.ndarray:
    scores = classifier_scores(replies, model_id)
    return np.asarray([scores[i, GOEMOTIONS.index(target)] for i, target in enumerate(targets)])


def free_classifiers() -> None:
    _CLASSIFIERS.clear()
    release()


class Judge:
    """Qwen rubric ratings (expected value over the digits 1-5) and pairwise preferences."""

    SYSTEM = ("You are a strict, careful evaluator of chatbot replies in a supportive-conversation app. "
              "Answer with a single digit from 1 to 5.")
    PAIR_SYSTEM = ("You are a strict, careful evaluator of chatbot replies in a supportive-conversation app. "
                   "Answer with a single letter, A or B.")

    def __init__(self, model_id: str = JUDGE_ID, load_in_4bit: bool = False) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.digits = [self.tokenizer.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]
        self.letters = [self.tokenizer.encode(letter, add_special_tokens=False)[-1] for letter in ("A", "B")]
        # Share of next-token probability that falls on the allowed answers; near 1 when the judge
        # really answers with a digit / letter.  Reported so a judge that ignores the format is noticed.
        self.answer_mass: list[float] = []

    @staticmethod
    def context(user: str, history: Sequence[dict[str, str]] = ()) -> str:
        if not history:
            return f"User message:\n{user}"
        earlier = "\n".join(f"{'User' if turn['role'] == 'user' else 'Chatbot'}: {turn['content']}" for turn in history)
        return f"Earlier conversation:\n{earlier}\n\nLatest user message:\n{user}"

    def _last_logits(self, prompts: list[str], tokens: list[int], batch_size: int) -> torch.Tensor:
        order = sorted(range(len(prompts)), key=lambda index: len(prompts[index]))
        out = torch.zeros(len(prompts), len(tokens))
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                chunk = order[start:start + batch_size]
                encoded = self.tokenizer([prompts[i] for i in chunk], return_tensors="pt", padding=True).to("cuda")
                logits = self.model(**encoded).logits[:, -1].float()
                self.answer_mass += torch.softmax(logits, -1)[:, tokens].sum(-1).cpu().tolist()
                out[chunk] = logits[:, tokens].cpu()
        return out

    def rate(self, contexts: Sequence[str], replies: Sequence[str], rubrics: Sequence[str],
             batch_size: int = 24) -> np.ndarray:
        prompts = [self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": f"{context}\n\nChatbot reply:\n{reply}\n\n{rubric}\n\nScore:"},
        ], tokenize=False, add_generation_prompt=True) for context, reply, rubric in zip(contexts, replies, rubrics)]
        probabilities = torch.softmax(self._last_logits(prompts, self.digits, batch_size), -1)
        return (probabilities @ torch.arange(1, 6, dtype=torch.float32)).numpy()

    def prefer(self, contexts: Sequence[str], first: Sequence[str], second: Sequence[str],
               batch_size: int = 16) -> np.ndarray:
        """P(first reply is the better supportive reply), averaged over both presentation orders."""
        question = ("Which reply is the better response to this user in a supportive-conversation app: more "
                    "understanding, more relevant to what the user said, and natural? Answer A or B.")

        def prompts(a: Sequence[str], b: Sequence[str]) -> list[str]:
            return [self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.PAIR_SYSTEM},
                {"role": "user", "content": f"{context}\n\nReply A:\n{x}\n\nReply B:\n{y}\n\n{question}\n\nAnswer:"},
            ], tokenize=False, add_generation_prompt=True) for context, x, y in zip(contexts, a, b)]
        forward = torch.softmax(self._last_logits(prompts(first, second), self.letters, batch_size), -1)[:, 0]
        backward = torch.softmax(self._last_logits(prompts(second, first), self.letters, batch_size), -1)[:, 1]
        return ((forward + backward) / 2).numpy()

    def close(self) -> None:
        self.model = None
        release()


def repeated_bigram_rate(text: str) -> float:
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def score_table(table: pd.DataFrame, *, quality: bool = True, judge: Judge | None = None) -> pd.DataFrame:
    """Add all automatic scores to a table with columns text, history, reply, target.

    gate_target      XLM-R score of the target emotion (the deployed gate's metric)
    indep_target     third-party RoBERTa GoEmotions score of the target emotion
    judge_emotion    Qwen 1-5 rating of how strongly the reply expresses the target emotion
    empathy / relevance / fluency   Qwen 1-5 rubric ratings
    Identical (context, reply, rubric) triples are scored once.
    """
    table = table.copy()
    replies = table.reply.fillna("").tolist()
    table["gate_target"] = target_scores(replies, table.target.tolist(), SCORER)
    table["indep_target"] = target_scores(replies, table.target.tolist(), INDEPENDENT_SCORER)
    free_classifiers()
    table["repeated_bigram_rate"] = [repeated_bigram_rate(reply) for reply in replies]
    table["reply_words"] = [len(reply.split()) for reply in replies]
    own_judge = judge is None
    judge = judge or Judge()
    contexts = [Judge.context(text, history) for text, history in zip(table.text, table.history)]
    rubrics = {"judge_emotion": [emotion_rubric(target) for target in table.target]}
    if quality:
        rubrics.update({name: [text] * len(table) for name, text in RUBRICS.items()})
    for name, rubric in rubrics.items():
        keys = list(zip(contexts, replies, rubric))
        unique = list(dict.fromkeys(keys))
        values = judge.rate([k[0] for k in unique], [k[1] for k in unique], [k[2] for k in unique])
        lookup = dict(zip(unique, values))
        table[name] = [float(lookup[key]) for key in keys]
    if own_judge:
        judge.close()
    return table


# ------------------------------------------------------------------------- stats

def paired_summary(difference: Sequence[float], *, seed: int = 0, resamples: int = 10000) -> dict[str, float]:
    """Mean paired difference with a percentile bootstrap 95% CI and a two-sided Wilcoxon p-value."""
    from scipy.stats import wilcoxon
    values = np.asarray(difference, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    p = float(wilcoxon(values).pvalue) if np.any(values) and len(values) > 5 else float("nan")
    return {"n": int(len(values)), "mean": float(values.mean()), "ci_low": float(np.quantile(means, 0.025)),
            "ci_high": float(np.quantile(means, 0.975)), "p": p}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (NaN entries are passed through)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


__all__ = ["LAB_DIR", "SPLIT_DIR", "TARGETS", "QUALITY", "RUBRICS", "STEERING_GAIN", "SYSTEM_PROMPT",
           "INDEPENDENT_SCORER", "SCORER", "JUDGE_ID", "Judge", "Planner", "build_splits", "load_split",
           "messages_of", "load_runtime", "generate_batch", "subset_panel", "random_panel", "panel_size",
           "clamp_neurons", "score_table", "paired_summary", "holm", "release", "free_classifiers",
           "emotion_rubric", "repeated_bigram_rate"]
