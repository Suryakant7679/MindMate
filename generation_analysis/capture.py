#!/usr/bin/env python3
"""Stage 1: generate matched emotional/neutral responses and capture MLP activations.

Every neutral-content user message is answered once per condition (the ten
recognition labels, neutral included).  Conditions differ only in one style
sentence of the system prompt, so each emotional response has a neutral twin
with identical user content.

For each response this stores, separately for prompt prefill and generated-
token decoding, the per-neuron firing count (activation > threshold) and mean
activation at the input of every ``layer.mlp.down_proj``.  It also stores the
bit-packed firing pattern of every decode step for token-level analysis.  See
``generation_analysis/common.py`` for the slot/token alignment convention.

Run from the project root:  python3 -m generation_analysis.capture
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .common import (DEFAULT_CSV, MODEL_ID, NEUTRAL, RESULTS_DIR, SYSTEM_TEMPLATE, MLPController,
                     condition_style, greedy_decode, load_model, model_layers, prompt_token_ids,
                     read_jsonl, recognition_metadata, special_token_ids, stop_token_ids,
                     verify_alignment, write_json)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def select_prompts(csv_path: str, text_column: str, label_column: str, content_label: str,
                   min_words: int, identification: int, steering: int, seed: int
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Disjoint identification and steering prompt pools of neutral-content texts."""
    data = pd.read_csv(csv_path)[[text_column, label_column]].dropna()
    data[text_column] = data[text_column].astype(str).str.strip()
    texts = data.loc[data[label_column].astype(str).str.strip().eq(content_label), text_column]
    texts = texts[texts.str.split().str.len() >= min_words].drop_duplicates().tolist()
    if len(texts) < identification + steering:
        raise ValueError(f"Only {len(texts)} eligible {content_label!r} texts for "
                         f"{identification + steering} requested prompts.")
    order = np.random.default_rng(seed).permutation(len(texts))
    chosen = [texts[index] for index in order[:identification + steering]]
    ident = pd.DataFrame({"prompt_id": range(identification), "text": chosen[:identification]})
    steer = pd.DataFrame({"prompt_id": range(steering), "text": chosen[identification:]})
    return ident, steer


def open_array(path: Path, dtype: str, shape: tuple[int, ...], resume: bool) -> np.memmap:
    if resume and path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{path} has shape {array.shape}/{array.dtype}, expected {shape}/{dtype}.")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--csv-path", default=DEFAULT_CSV)
    parser.add_argument("--text-column", default="hinglish_genz_text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--content-label", default=NEUTRAL,
                        help="Dataset label whose texts serve as emotion-free user content.")
    parser.add_argument("--identification-prompts", type=int, default=200)
    parser.add_argument("--steering-prompts", type=int, default=32)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--activation-threshold", type=float, default=0.0)
    parser.add_argument("--verify-samples", type=int, default=4,
                        help="Responses re-run with one teacher-forced pass to check alignment.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    prompt_dir, activation_dir = output / "prompts", output / "activations"
    prompt_dir.mkdir(parents=True, exist_ok=True); activation_dir.mkdir(parents=True, exist_ok=True)
    ident, steer = select_prompts(args.csv_path, args.text_column, args.label_column, args.content_label,
                                  args.min_words, args.identification_prompts, args.steering_prompts, args.seed)
    ident_path, steer_path = prompt_dir / "identification_prompts.csv", prompt_dir / "steering_prompts.csv"
    if args.resume and ident_path.exists():
        if not (pd.read_csv(ident_path).text.tolist() == ident.text.tolist()
                and pd.read_csv(steer_path).text.tolist() == steer.text.tolist()):
            raise ValueError("Saved prompt pools differ from the requested ones; refusing to resume.")
    ident.to_csv(ident_path, index=False); steer.to_csv(steer_path, index=False)

    recognition = recognition_metadata()
    conditions = list(recognition["labels"])
    model, tokenizer = load_model(args.model_id, allow_mismatch=args.allow_model_mismatch)
    layers = model_layers(model)
    width = int(layers[0].mlp.down_proj.in_features)
    if (len(layers), width) != (recognition["num_layers"], recognition["neurons_per_layer"]):
        raise ValueError(f"Model has {len(layers)}x{width} MLP neurons; recognition used "
                         f"{recognition['num_layers']}x{recognition['neurons_per_layer']}.")
    special, stops = special_token_ids(tokenizer), stop_token_ids(model, tokenizer)
    controller = MLPController(layers, args.activation_threshold)

    jobs = [(row.prompt_id, row.text, condition) for row in ident.itertuples() for condition in conditions]
    n, shape = len(jobs), (len(jobs), len(layers), width)
    arrays = {
        "decode_firing_counts": open_array(activation_dir / "decode_firing_counts.npy", "uint16", shape, args.resume),
        "decode_activation_mean": open_array(activation_dir / "decode_activation_mean.npy", "float16", shape, args.resume),
        "prefill_firing_counts": open_array(activation_dir / "prefill_firing_counts.npy", "uint16", shape, args.resume),
        "prefill_activation_mean": open_array(activation_dir / "prefill_activation_mean.npy", "float16", shape, args.resume),
        "decode_token_firing_bits": open_array(activation_dir / "decode_token_firing_bits.npy", "uint8",
                                               (n, args.max_new_tokens, len(layers), width // 8), args.resume),
    }
    responses_path = activation_dir / "responses.jsonl"
    done = len(read_jsonl(responses_path)) if args.resume and responses_path.exists() else 0
    if not args.resume and responses_path.exists():
        responses_path.unlink()
    write_json(output / "capture_metadata.json", {
        "model_id": args.model_id, "conditions": conditions, "content_label": args.content_label,
        "system_template": SYSTEM_TEMPLATE, "condition_styles": {c: condition_style(c) for c in conditions},
        "identification_prompts": len(ident), "steering_prompts": len(steer), "responses": n,
        "max_new_tokens": args.max_new_tokens, "decoding": "greedy, KV cache, left-padded batches",
        "activation_location": "input to layer.mlp.down_proj (SwiGLU intermediate)",
        "activation_definition": f"1[MLP activation > {args.activation_threshold}]",
        "activation_threshold": args.activation_threshold, "num_layers": len(layers), "neurons_per_layer": width,
        "special_token_ids_excluded": sorted(special), "stop_token_ids": sorted(stops),
        "phases": {"prefill": "single forward over the prompt; non-special prompt positions are recorded",
                   "decode": "step s reads generated token y_{s-1} and emits y_s; stored in slot s-1; "
                             "only non-special read tokens are recorded"},
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()},
        "seed": args.seed,
    })

    checks = []
    try:
        with responses_path.open("a", encoding="utf-8") as handle:
            for start in tqdm(range(done, n, args.batch_size), desc="Generating", initial=done // args.batch_size,
                              total=(n + args.batch_size - 1) // args.batch_size):
                batch = jobs[start:start + args.batch_size]
                verify = start == 0 and args.verify_samples > 0
                results = greedy_decode(model, controller, [prompt_token_ids(tokenizer, text, condition)
                                                            for _, text, condition in batch],
                                        args.max_new_tokens, pad_id=tokenizer.pad_token_id, stop_ids=stops,
                                        special_ids=special, record=True, keep_raw=verify)
                record = controller.batch
                for name, sums, counts, steps in (
                        ("prefill", record.prefill_sums, record.prefill_counts, [r.prefill_tokens for r in results]),
                        ("decode", record.decode_sums, record.decode_counts, [r.decode_steps for r in results])):
                    denominator = torch.as_tensor(steps, dtype=torch.float32).clamp_min(1)[:, None, None]
                    stacked_counts = torch.stack([c.cpu() for c in counts], dim=1)
                    stacked_means = torch.stack([s.cpu() for s in sums], dim=1) / denominator
                    arrays[f"{name}_firing_counts"][start:start + len(batch)] = stacked_counts.numpy().astype(np.uint16)
                    arrays[f"{name}_activation_mean"][start:start + len(batch)] = stacked_means.numpy().astype(np.float16)
                bits = torch.stack([b.cpu() for b in record.decode_bits], dim=2)  # [B, steps, L, U/8]
                arrays["decode_token_firing_bits"][start:start + len(batch)] = bits.numpy()
                if verify:
                    for row, result in enumerate(results[:args.verify_samples]):
                        raw = [torch.stack([step[row] for step in record.raw_decode[layer]])
                               for layer in range(len(layers))]
                        checks.append({"response_id": start + row, "condition": batch[row][2],
                                       **verify_alignment(model, layers, result, raw, special,
                                                          args.activation_threshold)})
                    write_json(activation_dir / "alignment_check.json", {
                        "description": "KV-cache decode activations vs one teacher-forced forward pass "
                                       "over prompt + generated tokens (same positions).", "checks": checks})
                    print(json.dumps(checks, indent=2))
                for array in arrays.values():
                    array.flush()
                for offset, ((prompt_id, text, condition), result) in enumerate(zip(batch, results)):
                    handle.write(json.dumps({
                        "response_id": start + offset, "prompt_id": int(prompt_id), "condition": condition,
                        "user_text": text,
                        "response_text": tokenizer.decode(result.generated_ids, skip_special_tokens=True).strip(),
                        "prompt_length": len(result.prompt_ids), "generated_ids": result.generated_ids,
                        "logprobs": [round(value, 6) for value in result.logprobs],
                        "read_count": result.read_count, "decode_steps": result.decode_steps,
                        "prefill_tokens": result.prefill_tokens, "stopped": result.stopped,
                    }, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        controller.close()
    print(f"Saved {n} responses and activation statistics to {activation_dir}")


if __name__ == "__main__":
    main()
