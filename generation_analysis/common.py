"""Shared model, decoding, hook and selectivity code for the generation analysis.

Phase convention (used by every script in this package)
-------------------------------------------------------
``greedy_decode`` runs prefill and decoding as separate forward calls and sets
``MLPController.phase`` before each one, so prompt and generated-token
activations are separated by construction rather than inferred from shapes.

* Prefill: one forward over the whole prompt.  Its last position emits y_0.
* Decode step s (s = 1..M): one forward over the single token y_{s-1}; it
  *reads* y_{s-1} and *emits* y_s.  Step M only reads y_{M-1} (budget reached),
  so every generated, non-special token is read by exactly one decode step.

Decode-step activations are stored in slot ``s - 1``.  Hence for generated
token y_i, the "read" activation is slot i and the "emit" activation is slot
i - 1 (y_0 has no decode-phase emit activation; it comes from prefill).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
RESULTS_DIR = Path("results/llama31_8b_generation_emotion")
RECOGNITION_DIR = Path("results/llama31_8b_emotions_full")
RECOGNITION_SELECTIVITY = RECOGNITION_DIR / "selectivity.npy"
RECOGNITION_METADATA = RECOGNITION_DIR / "metadata.json"
PROBE_PATH = Path("results/llama31_8b_neuron_analysis/04_probe/probe.pt")
DEFAULT_CSV = "data/mlt_hinghlish_dataset_clean.csv"
NEUTRAL = "neutral"
# Raw alpha=1 scale of the original single-joy-neuron experiment (backend/runtime.py).
SELECTIVITY_STEERING_SCALE = 0.008311835062595477

EMOTION_DESCRIPTIONS = {
    "admiration": "admiration: respect, praise and appreciation",
    "anger": "anger: irritation, frustration and outrage",
    "disapproval": "disapproval: disagreement and criticism",
    "disgust": "disgust: revulsion and distaste",
    "fear": "fear: worry, nervousness and alarm",
    "joy": "joy: happiness, delight and excitement",
    "love": "love: warmth, affection and care",
    "sadness": "sadness: sorrow, disappointment and hurt",
    "surprise": "surprise: astonishment and disbelief",
}
SYSTEM_TEMPLATE = ("You are a conversational assistant. Reply to the user's message in two or three "
                   "sentences, in the same language style as the user. {style}")


def condition_style(condition: str) -> str:
    if condition == NEUTRAL:
        return "Keep the reply emotionally neutral: calm, factual and free of emotional words."
    return f"Make the reply clearly express {EMOTION_DESCRIPTIONS.get(condition, condition)}."


def chat_messages(user_text: str, condition: str) -> list[dict[str, str]]:
    """Matched prompts: only the style sentence of the system prompt varies."""
    return [{"role": "system", "content": SYSTEM_TEMPLATE.format(style=condition_style(condition))},
            {"role": "user", "content": user_text}]


def prompt_token_ids(tokenizer: Any, user_text: str, condition: str) -> list[int]:
    ids = tokenizer.apply_chat_template(chat_messages(user_text, condition), tokenize=True,
                                        add_generation_prompt=True)
    return list(ids["input_ids"] if isinstance(ids, dict) else ids)


# --------------------------------------------------------------------------- I/O

def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recognition_metadata() -> dict[str, Any]:
    return read_json(RECOGNITION_METADATA)


# ------------------------------------------------------------------------ model

def assert_original_checkpoint(model_id: str, *, allow_mismatch: bool = False) -> None:
    """Refuse LoRA adapters and checkpoints other than the recognition model."""
    path = Path(model_id)
    if path.is_dir() and (path / "adapter_config.json").exists():
        raise SystemExit(f"{model_id} is a PEFT/LoRA adapter. Use the original pretrained checkpoint.")
    expected = recognition_metadata().get("model_id")
    if model_id != expected and not allow_mismatch:
        raise SystemExit(f"Recognition neurons were extracted from {expected!r}, not {model_id!r}. "
                         "Pass --allow-model-mismatch only for a deliberate comparison.")


def load_model(model_id: str, *, allow_mismatch: bool = False) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert_original_checkpoint(model_id, allow_mismatch=allow_mismatch)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", dtype=dtype,
                                                 low_cpu_mem_usage=True)
    if hasattr(model, "peft_config"):
        raise SystemExit("A PEFT adapter was attached to the model; use the original checkpoint.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def model_layers(model: Any) -> list[Any]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or any(not hasattr(getattr(layer, "mlp", None), "down_proj") for layer in layers):
        raise AttributeError("Expected model.model.layers[*].mlp.down_proj (Llama architecture).")
    return list(layers)


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def special_token_ids(tokenizer: Any) -> set[int]:
    """BOS/EOS plus every added special token (chat-template headers etc.)."""
    ids = set(getattr(tokenizer, "all_special_ids", []))
    ids |= {index for index, token in getattr(tokenizer, "added_tokens_decoder", {}).items()
            if getattr(token, "special", False)}
    return ids


def stop_token_ids(model: Any, tokenizer: Any) -> set[int]:
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = set(configured if isinstance(configured, (list, tuple)) else [configured])
    ids.add(tokenizer.eos_token_id)
    return {int(index) for index in ids if index is not None}


# ---------------------------------------------------------------------- hooking

def pack_bits(fired: torch.Tensor) -> torch.Tensor:
    """Pack a [..., U] bool tensor into [..., U/8] uint8, identical to np.packbits."""
    if fired.shape[-1] % 8:
        raise ValueError("MLP width must be divisible by 8 for bit packing.")
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=fired.device)
    grouped = fired.reshape(*fired.shape[:-1], fired.shape[-1] // 8, 8).to(torch.uint8)
    return (grouped * weights).sum(dim=-1, dtype=torch.uint8)


@dataclass
class BatchRecord:
    """Per-row activation statistics for one decoding batch (one entry per layer)."""
    prefill_counts: list[torch.Tensor] = field(default_factory=list)
    prefill_sums: list[torch.Tensor] = field(default_factory=list)
    decode_counts: list[torch.Tensor] = field(default_factory=list)
    decode_sums: list[torch.Tensor] = field(default_factory=list)
    decode_bits: list[torch.Tensor] = field(default_factory=list)
    raw_decode: list[list[torch.Tensor]] = field(default_factory=list)


class MLPController:
    """Forward-pre hooks on every ``layer.mlp.down_proj`` (input = SwiGLU intermediate).

    ``phase`` is set by ``greedy_decode``.  With ``record`` the hook stores
    prefill and decode statistics separately; ``steer`` maps a layer index to
    ``(neuron_indices, deltas)`` added to the activation on decode steps only,
    matching the backend's intervention (prompt processing is unchanged).
    """

    def __init__(self, layers: Sequence[Any], threshold: float = 0.0) -> None:
        self.num_layers = len(layers)
        self.width = int(layers[0].mlp.down_proj.in_features)
        self.threshold = threshold
        self.phase: str | None = None
        self.record = False
        self.keep_raw = False
        self.prefill_mask: torch.Tensor | None = None
        self.decode_active: torch.Tensor | None = None
        self.step = 0
        self.batch: BatchRecord | None = None
        self.steer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = [layer.mlp.down_proj.register_forward_pre_hook(self._hook(index))
                        for index, layer in enumerate(layers)]

    def begin(self, batch_size: int, max_steps: int, prefill_mask: torch.Tensor) -> None:
        self.prefill_mask = prefill_mask
        self.batch_size, self.max_steps = batch_size, max_steps
        self.batch = BatchRecord()
        self.batch.raw_decode = [[] for _ in range(self.num_layers)]
        self._allocated: set[int] = set()

    def _allocate(self, layer_index: int, device: torch.device) -> None:
        # Buffers live beside the activation, which is always on a real device
        # even when device_map="auto" offloads the weight itself.
        b, u, record = self.batch_size, self.width, self.batch
        for name, dtype, shape in (("prefill_counts", torch.int32, (b, u)),
                                   ("prefill_sums", torch.float32, (b, u)),
                                   ("decode_counts", torch.int32, (b, u)),
                                   ("decode_sums", torch.float32, (b, u)),
                                   ("decode_bits", torch.uint8, (b, self.max_steps, u // 8))):
            buffers = getattr(record, name)
            while len(buffers) <= layer_index:
                buffers.append(None)
            buffers[layer_index] = torch.zeros(shape, dtype=dtype, device=device)
        self._allocated.add(layer_index)

    def _record(self, layer_index: int, activation: torch.Tensor) -> None:
        if layer_index not in self._allocated:
            self._allocate(layer_index, activation.device)
        record = self.batch
        if self.phase == "prefill":
            mask = self.prefill_mask.to(activation.device).unsqueeze(-1)
            values = activation.float()
            fired = (values > self.threshold) & mask
            record.prefill_counts[layer_index] += fired.sum(dim=1, dtype=torch.int32)
            # torch.where, not multiplication: padded positions may hold NaN.
            record.prefill_sums[layer_index] += torch.where(mask, values, 0.0).sum(dim=1)
        elif self.phase == "decode":
            values = activation[:, -1].float()
            active = self.decode_active.to(activation.device).unsqueeze(-1)
            fired = (values > self.threshold) & active
            record.decode_counts[layer_index] += fired.to(torch.int32)
            record.decode_sums[layer_index] += torch.where(active, values, 0.0)
            record.decode_bits[layer_index][:, self.step - 1] = pack_bits(fired)
            if self.keep_raw:
                record.raw_decode[layer_index].append(values.cpu())

    def _hook(self, layer_index: int):
        def hook(_module: Any, inputs: tuple[Any, ...]):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError(f"Layer {layer_index} down_proj hook received no activation tensor.")
            activation, modified = inputs[0], False
            if self.phase == "decode" and layer_index in self.steer:
                indices, deltas = self.steer[layer_index]
                activation = activation.clone()
                activation[..., indices.to(activation.device)] += deltas.to(activation.device, activation.dtype)
                modified = True
            if self.record and self.phase is not None:
                self._record(layer_index, activation)
            return (activation, *inputs[1:]) if modified else None
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def steering_tensors(by_layer: dict[int, tuple[Sequence[int], Sequence[float]]]
                     ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {int(layer): (torch.as_tensor(list(ids), dtype=torch.long),
                         torch.as_tensor(list(deltas), dtype=torch.float32))
            for layer, (ids, deltas) in by_layer.items() if len(ids)}


# --------------------------------------------------------------------- decoding

@dataclass
class DecodeResult:
    prompt_ids: list[int]
    generated_ids: list[int]          # y_0..y_{n-1}, including a final stop token if emitted
    logprobs: list[float]             # log p(y_i | prompt, y_<i) under the decoding prompt
    read_count: int                   # decode steps run for this row; slot i read y_i
    decode_steps: int                 # active slots = generated non-special tokens read
    prefill_tokens: int               # non-special prompt positions recorded
    stopped: bool


def greedy_decode(model: Any, controller: MLPController, prompts: Sequence[Sequence[int]],
                  max_new_tokens: int, *, pad_id: int, stop_ids: Iterable[int],
                  special_ids: Iterable[int], record: bool = False, keep_raw: bool = False
                  ) -> list[DecodeResult]:
    """Batched greedy decoding with left padding and explicit prefill/decode phases."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    device = input_device(model)
    stop_ids, special_ids = set(stop_ids), set(special_ids)
    batch, length = len(prompts), max(len(prompt) for prompt in prompts)
    input_ids = torch.full((batch, length), pad_id, dtype=torch.long)
    attention = torch.zeros((batch, length), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        input_ids[row, length - len(prompt):] = torch.as_tensor(prompt, dtype=torch.long)
        attention[row, length - len(prompt):] = 1
    special = torch.as_tensor(sorted(special_ids), dtype=torch.long)
    prefill_mask = attention.bool() & ~torch.isin(input_ids, special)
    input_ids, attention = input_ids.to(device), attention.to(device)
    positions = (attention.cumsum(-1) - 1).clamp_min(0)

    controller.record, controller.keep_raw = record, keep_raw
    controller.begin(batch, max_new_tokens, prefill_mask.to(device))
    generated: list[list[int]] = [[] for _ in range(batch)]
    logprobs: list[list[float]] = [[] for _ in range(batch)]
    finished = [False] * batch
    decode_steps = [0] * batch
    read_count = [0] * batch
    try:
        with torch.inference_mode():
            controller.phase = "prefill"
            output = model(input_ids=input_ids, attention_mask=attention, position_ids=positions,
                           use_cache=True, logits_to_keep=1)
            for step in range(1, max_new_tokens + 1):
                log_probs = torch.log_softmax(output.logits[:, -1].float(), dim=-1)
                next_ids = log_probs.argmax(dim=-1)
                chosen = log_probs.gather(1, next_ids[:, None])[:, 0]
                # Emission of y_{step-1} by the previous forward call.
                active = []
                for row in range(batch):
                    if finished[row]:
                        active.append(False)
                        continue
                    token = int(next_ids[row])
                    generated[row].append(token)
                    logprobs[row].append(float(chosen[row]))
                    finished[row] = token in stop_ids
                    active.append(not finished[row] and token not in special_ids)
                    decode_steps[row] += int(active[-1])
                    read_count[row] += int(not finished[row])
                if all(finished):
                    break
                # Decode step `step` reads y_{step-1}; at step == M it only reads.
                attention = torch.cat([attention, attention.new_ones((batch, 1))], dim=1)
                positions = positions[:, -1:] + 1
                controller.phase, controller.step = "decode", step
                controller.decode_active = torch.as_tensor(active, dtype=torch.bool)
                output = model(input_ids=next_ids[:, None].to(device), attention_mask=attention,
                               position_ids=positions, past_key_values=output.past_key_values,
                               use_cache=True, logits_to_keep=1)
                if step == max_new_tokens:
                    break
    finally:
        controller.phase = None
        controller.record = controller.keep_raw = False
    return [DecodeResult(prompt_ids=list(prompts[row]), generated_ids=generated[row],
                         logprobs=logprobs[row], read_count=read_count[row],
                         decode_steps=decode_steps[row],
                         prefill_tokens=int(prefill_mask[row].sum()), stopped=finished[row])
            for row in range(batch)]


def emission_slot(token_index: int, alignment: str) -> int | None:
    """Decode slot holding the activation associated with generated token y_i."""
    if alignment == "read":
        return token_index
    if alignment == "emit":
        return token_index - 1 if token_index >= 1 else None
    raise ValueError("alignment must be 'emit' or 'read'.")


def verify_alignment(model: Any, layers: Sequence[Any], result: DecodeResult,
                     raw_decode: list[torch.Tensor], special_ids: set[int], threshold: float = 0.0
                     ) -> dict[str, float]:
    """Compare KV-cache decode activations with one teacher-forced forward pass.

    ``raw_decode[layer]`` is a [steps, U] tensor of decode activations for this
    row.  Slot s-1 must equal the full-sequence activation at prompt_len + s - 1.
    The check also confirms that greedy argmax reproduces each generated token.
    """
    fed = result.generated_ids[:result.read_count]
    # Only active slots (non-special read tokens) are recorded statistics.
    active = torch.as_tensor([token not in special_ids for token in fed], dtype=torch.bool)
    sequence = torch.as_tensor([result.prompt_ids + fed], dtype=torch.long, device=input_device(model))
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module, inputs, layer_index=index):
            captured[layer_index] = inputs[0][0].detach().float().cpu()
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            logits = model(input_ids=sequence, use_cache=False).logits[0].float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    start, steps = len(result.prompt_ids), result.read_count
    max_abs, scale, agree, total = 0.0, 0.0, 0, 0
    for index in range(len(layers)):
        teacher = captured[index][start:start + steps][active]
        cached = raw_decode[index][:steps][active]
        if not len(teacher):
            continue
        max_abs = max(max_abs, float((teacher - cached).abs().max()))
        scale = max(scale, float(teacher.abs().max()))
        agree += int(((teacher > threshold) == (cached > threshold)).sum())
        total += teacher.numel()
    targets = torch.as_tensor(result.generated_ids[:steps + 1])
    predicted = logits[start - 1:start - 1 + len(targets)].argmax(-1)
    return {"decode_steps": int(active.sum()), "max_abs_difference": max_abs, "max_abs_activation": scale,
            "firing_agreement": agree / max(total, 1),
            "greedy_token_agreement": float((predicted == targets).float().mean()) if len(targets) else 1.0}


# ------------------------------------------------------------------ selectivity

def selectivity_scores(counts: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """S[l,u,c] = P[l,u,c] * (1 - H[l,u]), exactly as in emotion_neuron_extraction.py.

    ``counts`` is [layer, neuron, class] firing counts and ``positions`` the
    number of valid token positions per class.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if counts.shape[-1] != positions.shape[0] or (positions <= 0).any():
        raise ValueError("Every class needs at least one valid token position.")
    probability = counts.astype(np.float64) / (positions + 1e-12)
    distribution = probability / (probability.sum(axis=-1, keepdims=True) + 1e-12)
    entropy = -(distribution * np.log(distribution + 1e-12)).sum(axis=-1) / math.log(positions.shape[0])
    return probability * (1.0 - entropy[..., None])


def top_k(width: int, percentage: float) -> int:
    return max(1, min(width, int(width * percentage / 100.0)))


def top_neurons(scores: np.ndarray, percentage: float) -> list[list[int]]:
    """Per layer, the stable top percentage of a [layer, neuron] score array."""
    count = top_k(scores.shape[1], percentage)
    return [np.argsort(-scores[layer], kind="stable")[:count].tolist() for layer in range(scores.shape[0])]


def jaccard(first: set[int], second: set[int]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def expected_random_jaccard(size_a: int, size_b: int, width: int) -> float:
    shared = size_a * size_b / width
    union = size_a + size_b - shared
    return shared / union if union else 0.0


# ------------------------------------------------------------------- read-outs

def load_probe(path: str | Path, device: torch.device) -> tuple[torch.nn.Linear, list[str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    labels = list(checkpoint["labels"])
    classifier = torch.nn.Linear(int(checkpoint["hidden_size"]), len(labels))
    classifier.load_state_dict(checkpoint["state_dict"])
    return classifier.to(device).eval(), labels


def probe_probabilities(model: Any, tokenizer: Any, classifier: torch.nn.Module, texts: Sequence[str],
                        special_ids: set[int], batch_size: int = 8) -> np.ndarray:
    """Frozen-probe distribution for standalone texts (same pooling as 05_train_emotion_probe)."""
    device, rows = input_device(model), []
    special = torch.as_tensor(sorted(special_ids), dtype=torch.long)
    probe_device = next(classifier.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = [text or " " for text in texts[start:start + batch_size]]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
            mask = encoded["attention_mask"].bool() & ~torch.isin(encoded["input_ids"], special)
            hidden = model.model(input_ids=encoded["input_ids"].to(device),
                                 attention_mask=encoded["attention_mask"].to(device),
                                 use_cache=False).last_hidden_state
            weights = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            rows.append(torch.softmax(classifier(pooled.float().to(probe_device)), dim=-1).cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, 0))


def continuation_logprobs(model: Any, prompts: Sequence[Sequence[int]], continuations: Sequence[Sequence[int]],
                          pad_id: int) -> list[list[float]]:
    """Teacher-forced log p(continuation_i | prompt, continuation_<i) for a batch."""
    device = input_device(model)
    sequences = [list(p) + list(c) for p, c in zip(prompts, continuations)]
    length = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for row, sequence in enumerate(sequences):
        ids[row, length - len(sequence):] = torch.as_tensor(sequence)
        attention[row, length - len(sequence):] = 1
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    with torch.inference_mode():
        hidden = model.model(input_ids=ids.to(device), attention_mask=attention.to(device),
                             position_ids=positions.to(device), use_cache=False).last_hidden_state
        output = []
        for row, (prompt, continuation) in enumerate(zip(prompts, continuations)):
            if not continuation:
                output.append([])
                continue
            # Positions predicting each continuation token, in the left-padded row.
            first = length - len(continuation) - 1
            states = hidden[row, first:first + len(continuation)]
            log_probs = torch.log_softmax(model.lm_head(states).float(), dim=-1)  # accelerate moves inputs
            target = torch.as_tensor(continuation, device=log_probs.device)
            output.append(log_probs.gather(1, target[:, None])[:, 0].cpu().tolist())
    return output
