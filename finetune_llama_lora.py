#!/usr/bin/env python3
"""LoRA/QLoRA fine-tune Llama 3.1 8B for Hinglish emotion classification.

The model learns to answer with exactly one label for a Hinglish text.  The
source CSV must contain ``hinglish_genz_text`` and ``label`` columns.  Splits
are deterministic and stratified: 85% train, 10% validation, 5% test.

Install once:
    python3 -m pip install -r backend/requirements.txt peft datasets pandas

Example (CUDA + QLoRA):
    python3 finetune_llama_lora.py --load-in-4bit --epochs 3

This writes an adapter only, not another copy of the 8B base model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = TaskType = get_peft_model = prepare_model_for_kbit_training = None
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


SYSTEM_PROMPT = (
    "Classify the emotion in the user's Hinglish message. "
    "Reply with exactly one label from: {labels}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=Path("data/mlt_hinghlish_dataset_clean.csv"))
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/llama31_8b_lora_emotion"))
    parser.add_argument("--text-column", default="hinglish_genz_text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 uses every training row.")
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(args.csv_path, usecols=[args.text_column, args.label_column])
    frame = frame.rename(columns={args.text_column: "text", args.label_column: "label"})
    frame["text"] = frame["text"].astype("string").str.strip()
    frame["label"] = frame["label"].astype("string").str.strip()
    frame = frame.dropna().loc[(frame["text"] != "") & (frame["label"] != "")]
    frame = frame.drop_duplicates().reset_index(drop=True)
    labels = sorted(frame["label"].unique().tolist())
    if len(labels) != 10:
        raise ValueError(f"Expected 10 emotion labels; found {len(labels)}: {labels}")
    return frame, labels


def stratified_split(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Create exact per-class 85/10/5 splits with every class represented."""
    rng = random.Random(seed)
    split_indices: dict[str, list[int]] = defaultdict(list)
    for _, group in frame.groupby("label", sort=True):
        indices = group.index.tolist()
        rng.shuffle(indices)
        size = len(indices)
        test_size = max(1, round(size * 0.05))
        validation_size = max(1, round(size * 0.10))
        split_indices["test"].extend(indices[:test_size])
        split_indices["validation"].extend(indices[test_size:test_size + validation_size])
        split_indices["train"].extend(indices[test_size + validation_size:])
    return {
        name: frame.loc[indices].sample(frac=1, random_state=seed).reset_index(drop=True)
        for name, indices in split_indices.items()
    }


def make_examples(
    frame: pd.DataFrame,
    tokenizer: Any,
    labels: list[str],
    max_length: int,
) -> Dataset:
    instruction = SYSTEM_PROMPT.format(labels=", ".join(labels))

    def encode(row: dict[str, str]) -> dict[str, list[int]]:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": row["text"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        # Preserve the complete label and EOS, trimming only the input text side.
        prompt_ids = prompt_ids[: max_length - len(target_ids)]
        input_ids = prompt_ids + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }

    return Dataset.from_pandas(frame[["text", "label"]], preserve_index=False).map(
        encode,
        remove_columns=["text", "label"],
        desc="Tokenizing",
    )


class CausalClassificationCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = width - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {name: torch.tensor(values, dtype=torch.long) for name, values in batch.items()}


def classification_metrics(truth: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    index = {label: i for i, label in enumerate(labels)}
    for actual, guessed in zip(truth, predicted):
        matrix[index[actual], index[guessed]] += 1
    f1 = []
    for position in range(len(labels)):
        tp = matrix[position, position]
        fp = matrix[:, position].sum() - tp
        fn = matrix[position, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "accuracy": round(float(np.trace(matrix) / matrix.sum()), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "f1_by_label": {label: round(float(score), 4) for label, score in zip(labels, f1)},
        "confusion_matrix": matrix.tolist(),
    }


def predict_labels(model: Any, tokenizer: Any, frame: pd.DataFrame, labels: list[str], max_length: int) -> list[str]:
    """Greedily generate labels for the held-out splits."""
    device = next(model.parameters()).device
    instruction = SYSTEM_PROMPT.format(labels=", ".join(labels))
    predictions = []
    model.eval()
    with torch.inference_mode():
        for text in frame["text"].tolist():
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": instruction}, {"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=max_length)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            answer = tokenizer.decode(generated[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            predictions.append(answer if answer in labels else "neutral")
    return predictions


def main() -> None:
    args = parse_args()
    if LoraConfig is None:
        raise SystemExit("Missing LoRA dependency. Install it with: python3 -m pip install peft")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit("--load-in-4bit requires CUDA. Use --no-load-in-4bit for a CPU smoke test.")

    frame, labels = load_data(args)
    splits = stratified_split(frame, args.seed)
    if args.max_train_samples:
        splits["train"] = splits["train"].iloc[:args.max_train_samples].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(args.output_dir / f"{name}.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    # Trainer cannot backpropagate through Hugging Face's CPU/disk offload map.
    # Keep the full model on one CUDA device for both LoRA and QLoRA training.
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["device_map"] = {"": local_rank}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    # With gradient checkpointing, LoRA needs an embedding-output gradient so
    # the recomputed transformer blocks retain an autograd graph.
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    ))
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_data = make_examples(splits["train"], tokenizer, labels, args.max_length)
    validation_data = make_examples(splits["validation"], tokenizer, labels, args.max_length)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=CausalClassificationCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")

    metrics = {"model_id": args.model_id, "labels": labels, "split_sizes": {name: len(split) for name, split in splits.items()}}
    for name in ("validation", "test"):
        predicted = predict_labels(model, tokenizer, splits[name], labels, args.max_length)
        metrics[name] = classification_metrics(splits[name]["label"].tolist(), predicted, labels)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved LoRA adapter to: {args.output_dir / 'adapter'}")


if __name__ == "__main__":
    main()
