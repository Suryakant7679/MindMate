#!/usr/bin/env python3
"""Build and evaluate a 1,000-example synthetic Hinglish emotion benchmark.

The benchmark is held out: it is never used by ``finetune_llama_lora.py``.
It contains 100 examples for each of the ten labels, split per label into
40 easy, 35 moderate, and 25 hard examples.  Base and LoRA-adapter models are
loaded one at a time to reduce GPU memory use.

Example:
    python3 evaluate_synthetic_emotions.py --build-benchmark --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = [
    "admiration", "anger", "disapproval", "disgust", "fear",
    "joy", "love", "neutral", "sadness", "surprise",
]
SYSTEM = "Classify the emotion in the user's Hinglish message. Reply with exactly one label from: " + ", ".join(LABELS) + "."

# Hand-authored lexical components prevent the benchmark from being generated
# by either model under test. The hard tier uses indirect/colloquial wording.
PARTS: dict[str, dict[str, list[str]]] = {
    "admiration": {
        "easy": ["{p} is my role model, unka kaam truly inspiring hai", "wow {p}, tumne kamaal kar diya, huge respect", "main {p} ki talent dekh ke genuinely impressed hoon"],
        "moderate": ["{p} handled that mess with such grace, respect badh gaya", "unke discipline se bahut seekhne ko milta hai", "{p} ka work dekhkar lagta hai standards aur high karne chahiye"],
        "hard": ["aisa composure sabke bas ki baat nahi, {p} alag level pe hain", "credit where it is due, {p} ne scene sambhal liya", "not many people could pull that off the way {p} did"],
    },
    "anger": {
        "easy": ["mujhe bahut gussa aa raha hai, {p} ne bilkul galat kiya", "this is so unfair, I am furious at {p}", "gussa control nahi ho raha after what {p} did"],
        "moderate": ["{p} ne phir mera time waste kiya, dimag kharab ho gaya", "bas karo yaar, this is seriously irritating", "itna careless kaise ho sakte ho, mujhe bohot gussa hai"],
        "hard": ["wah, {p} ne exactly wahi kiya jo mana kiya tha", "slow clap for this absolute mess", "mera patience aaj officially khatam ho gaya"],
    },
    "disapproval": {
        "easy": ["mujhe {p} ka behavior bilkul pasand nahi aaya", "that decision was wrong and I disapprove", "{p} ko aisa nahi karna chahiye tha"],
        "moderate": ["honestly, this approach makes no sense to me", "{p} ka choice kaafi disappointing tha", "main is plan ke against hoon, it is not okay"],
        "hard": ["interesting choice... not one I would stand behind", "yeh dekh ke bas head shake kar sakta hoon", "I expected better judgement from {p}"],
    },
    "disgust": {
        "easy": ["eww, that is disgusting, mujhe ghin aa rahi hai", "{p} ka mess dekh ke nausea ho raha hai", "this smells awful, totally gross"],
        "moderate": ["please isse door rakho, stomach turn ho raha hai", "{p} ne jo kiya woh dekhna bhi mushkil tha", "itna unhygienic kaise ho sakta hai"],
        "hard": ["my appetite just left the chat", "thanks, now I cannot unsee that", "that visual was deeply unpleasant"],
    },
    "fear": {
        "easy": ["mujhe bahut darr lag raha hai, I do not feel safe", "I am scared that {p} might happen", "dar ke maare neend nahi aa rahi"],
        "moderate": ["something feels wrong and I am getting anxious", "agar {p} hua toh kya hoga, I am worried", "raat ko woh sound suna and I panicked"],
        "hard": ["I keep checking the lock again and again", "heart is racing, bad feeling aa rahi hai", "please stay with me, I do not want to be alone right now"],
    },
    "joy": {
        "easy": ["main bahut khush hoon, today is amazing", "yay! {p} finally happened, so happy", "this made my day, I am smiling so much"],
        "moderate": ["aaj mood ekdum mast hai, everything feels good", "{p} worked out and I cannot stop grinning", "feeling blessed and excited today"],
        "hard": ["not me dancing around my room right now", "life is giving main-character energy today", "okay this is the kind of news I needed"],
    },
    "love": {
        "easy": ["I love {p} so much, they mean everything to me", "tum mere liye bahut special ho", "I feel so much love for my family"],
        "moderate": ["{p} ke saath hona feels like home", "bas unki smile dekh ke dil warm ho jata hai", "I care about {p} more than I can explain"],
        "hard": ["they are my favorite human, no contest", "unke bina days thode less bright lagte hain", "my heart is very soft for {p}"],
    },
    "neutral": {
        "easy": ["meeting kal 3 baje hai", "I submitted the form and received the email", "train 10 minutes late hai"],
        "moderate": ["{p} ke baad report upload kar dena", "I moved the files into the shared folder", "aaj weather cloudy hai aur office open hai"],
        "hard": ["noted, I will check and reply by evening", "the package arrived at the usual address", "we can discuss the schedule after lunch"],
    },
    "sadness": {
        "easy": ["main bahut udaas hoon aur rona aa raha hai", "I feel sad because I miss {p}", "dil heavy lag raha hai, nothing feels okay"],
        "moderate": ["{p} chala gaya and the house feels empty", "aaj bas bed se uthne ka mann nahi hua", "I keep thinking about what I lost and feel low"],
        "hard": ["everything feels quieter since {p} left", "I smiled today but it did not reach my eyes", "some days just feel heavy for no clear reason"],
    },
    "surprise": {
        "easy": ["what! I cannot believe {p} happened", "oh my god, I am so surprised", "wait, seriously? that was unexpected"],
        "moderate": ["I did not see that coming at all", "{p} just happened and I am still processing it", "that plot twist caught me completely off guard"],
        "hard": ["well, that was not on my bingo card", "I need a minute because my brain just paused", "did the universe really just do that"],
    },
}
NAMES = ["Aman", "Riya", "my friend", "my manager", "my brother", "the team"]
EVENTS = ["the interview went well", "the plan failed", "the lights went out", "they changed the decision", "the call came late"]
CONTEXTS = ["", " at home", " after class", " this morning", " on my way back", " during lunch", " in the group chat", " at work", " before dinner", " last night"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", type=Path, default=Path("data/synthetic_hinglish_emotion_1000.csv"))
    parser.add_argument("--build-benchmark", action="store_true")
    parser.add_argument("--build-only", action="store_true", help="Create the benchmark without loading either model.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=Path("results/llama31_8b_lora_emotion/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/llama31_8b_lora_emotion/synthetic_evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261104)
    return parser.parse_args()


def build_benchmark(path: Path, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    per_difficulty = {"easy": 40, "moderate": 35, "hard": 25}
    for label in LABELS:
        for difficulty, count in per_difficulty.items():
            seen = set()
            while len(seen) < count:
                text = rng.choice(PARTS[label][difficulty]).format(p=rng.choice(NAMES))
                # Add a non-emotion-bearing context fragment to diversify forms.
                suffix = rng.choice(["", ". yaar", ". honestly", ". aaj", ". fr fr", " after " + rng.choice(EVENTS)])
                candidate = text + suffix + rng.choice(CONTEXTS)
                if candidate in seen:
                    continue
                seen.add(candidate)
                rows.append({"text": candidate, "label": label, "difficulty": difficulty, "source": "hand_authored_template"})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(frame) != 1000 or frame.duplicated("text").any():
        raise RuntimeError("Synthetic benchmark construction failed uniqueness checks.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def prompt(tokenizer: Any, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_label(answer: str) -> str:
    answer = answer.lower().strip()
    return next((label for label in LABELS if label == answer or label in answer.split()), "invalid")


def predict(model: Any, tokenizer: Any, texts: list[str], batch_size: int, max_length: int) -> list[str]:
    device = next(model.parameters()).device
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            prompts = [prompt(tokenizer, text) for text in texts[start:start + batch_size]]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            generated = model.generate(**encoded, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            length = encoded["input_ids"].shape[1]
            output.extend(parse_label(tokenizer.decode(row[length:], skip_special_tokens=True)) for row in generated)
    return output


def metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    labels_with_invalid = LABELS + ["invalid"]
    def calculate(part: pd.DataFrame) -> dict[str, Any]:
        matrix = np.zeros((len(LABELS), len(labels_with_invalid)), dtype=int)
        index = {label: position for position, label in enumerate(LABELS)}
        prediction_index = {label: position for position, label in enumerate(labels_with_invalid)}
        for row in part.itertuples(index=False):
            matrix[index[row.label], prediction_index[getattr(row, prediction_column)]] += 1
        f1 = []
        for position in range(len(LABELS)):
            tp = matrix[position, position]
            fp = matrix[:, position].sum() - tp
            fn = matrix[position, :].sum() - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return {
            "examples": len(part),
            "accuracy": round(float((part["label"] == part[prediction_column]).mean()), 4),
            "macro_f1": round(float(np.mean(f1)), 4),
            "invalid_predictions": int((part[prediction_column] == "invalid").sum()),
            "f1_by_label": {label: round(float(value), 4) for label, value in zip(LABELS, f1)},
        }
    return {"overall": calculate(frame), **{difficulty: calculate(frame[frame["difficulty"] == difficulty]) for difficulty in ("easy", "moderate", "hard")}}


def load_model(model_id: str, adapter_path: Path | None) -> tuple[Any, Any]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this 8B benchmark evaluation.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": 0})
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def release(model: Any) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.build_benchmark or not args.benchmark_path.exists():
        frame = build_benchmark(args.benchmark_path, args.seed)
        print(f"Wrote {len(frame)} held-out synthetic examples to {args.benchmark_path}")
    else:
        frame = pd.read_csv(args.benchmark_path)
    required = {"text", "label", "difficulty"}
    if len(frame) != 1000 or not required.issubset(frame.columns):
        raise ValueError("Benchmark must contain exactly 1,000 rows with text, label, and difficulty columns.")
    if set(frame["label"]) != set(LABELS) or set(frame["difficulty"]) != {"easy", "moderate", "hard"}:
        raise ValueError("Benchmark labels or difficulty tiers are invalid.")
    if args.build_only:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base, tokenizer = load_model(args.model_id, None)
    frame["base_prediction"] = predict(base, tokenizer, frame["text"].tolist(), args.batch_size, args.max_length)
    release(base)
    tuned, tokenizer = load_model(args.model_id, args.adapter_path)
    frame["finetuned_prediction"] = predict(tuned, tokenizer, frame["text"].tolist(), args.batch_size, args.max_length)
    release(tuned)
    report = {
        "benchmark": str(args.benchmark_path),
        "examples": len(frame),
        "base": metrics(frame, "base_prediction"),
        "finetuned": metrics(frame, "finetuned_prediction"),
    }
    frame.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved predictions and report to {args.output_dir}")


if __name__ == "__main__":
    main()
