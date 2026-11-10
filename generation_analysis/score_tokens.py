#!/usr/bin/env python3
"""Stage 2: matched-prompt token scores and response-level emotion probe scores.

For every emotional response R_e, each generated token y_i is re-scored under
the *neutral* prompt for the same user message with the identical prefix y_<i
(teacher forcing).  delta_i = log p(y_i | emotion prompt) - log p(y_i | neutral
prompt) measures how much the emotion instruction, rather than the shared user
content, drove that token.  The frozen recognition probe also scores every
response text so stage 3 can check that responses express their condition.

Run from the project root:  python3 -m generation_analysis.score_tokens
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .common import (MODEL_ID, NEUTRAL, PROBE_PATH, RESULTS_DIR, continuation_logprobs, input_device,
                     load_model, load_probe, probe_probabilities, prompt_token_ids, read_json, read_jsonl,
                     special_token_ids, write_json)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allow-model-mismatch", action="store_true")
    parser.add_argument("--probe-path", default=str(PROBE_PATH))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    output = Path(args.output_dir)
    metadata = read_json(output / "capture_metadata.json")
    if metadata["model_id"] != args.model_id:
        raise ValueError(f"Responses were generated with {metadata['model_id']}, not {args.model_id}.")
    responses = read_jsonl(output / "activations" / "responses.jsonl")
    if len(responses) != metadata["responses"]:
        raise ValueError("Stage 1 is incomplete; rerun capture with --resume.")
    model, tokenizer = load_model(args.model_id, allow_mismatch=args.allow_model_mismatch)
    special = special_token_ids(tokenizer)

    token_rows = []
    emotional = [response for response in responses if response["condition"] != NEUTRAL]
    for start in tqdm(range(0, len(emotional), args.batch_size), desc="Scoring tokens under neutral prompts"):
        batch = emotional[start:start + args.batch_size]
        neutral_logprobs = continuation_logprobs(
            model, [prompt_token_ids(tokenizer, r["user_text"], NEUTRAL) for r in batch],
            [r["generated_ids"] for r in batch], tokenizer.pad_token_id)
        for response, neutral in zip(batch, neutral_logprobs):
            for index, (token, own, base) in enumerate(zip(response["generated_ids"], response["logprobs"], neutral)):
                token_rows.append((response["response_id"], index, token, tokenizer.decode([token]),
                                   token in special, own, base, own - base))
    for response in responses:
        if response["condition"] == NEUTRAL:
            for index, (token, own) in enumerate(zip(response["generated_ids"], response["logprobs"])):
                token_rows.append((response["response_id"], index, token, tokenizer.decode([token]),
                                   token in special, own, np.nan, np.nan))
    tokens = pd.DataFrame(token_rows, columns=["response_id", "token_index", "token_id", "token_text", "special",
                                               "logp_condition", "logp_neutral", "delta"])
    analysis = output / "token_analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    tokens.sort_values(["response_id", "token_index"]).to_csv(analysis / "token_scores.csv.gz", index=False)

    classifier, labels = load_probe(args.probe_path, input_device(model))
    probabilities = probe_probabilities(model, tokenizer, classifier, [r["response_text"] for r in responses],
                                        special, args.batch_size)
    scores = pd.DataFrame(probabilities, columns=[f"probe_{label}" for label in labels])
    scores.insert(0, "condition", [r["condition"] for r in responses])
    scores.insert(0, "prompt_id", [r["prompt_id"] for r in responses])
    scores.insert(0, "response_id", [r["response_id"] for r in responses])
    scores["probe_prediction"] = [labels[index] for index in probabilities.argmax(1)]
    scores["probe_condition_probability"] = [row[f"probe_{row.condition}"] for _, row in scores.iterrows()]
    scores.to_csv(analysis / "response_scores.csv", index=False)
    write_json(analysis / "score_metadata.json", {
        "model_id": args.model_id, "probe_path": args.probe_path, "probe_labels": labels,
        "delta_definition": "log p(y_i | emotion prompt, y_<i) - log p(y_i | neutral prompt, y_<i)",
        "probe_input": "response text alone, mean-pooled final hidden state (05_train_emotion_probe.py)",
        "probe_argmax_matches_condition": float((scores.probe_prediction == scores.condition).mean()),
    })
    print(f"Saved token and response scores to {analysis}")


if __name__ == "__main__":
    main()
