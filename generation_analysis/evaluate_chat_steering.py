#!/usr/bin/env python3
"""Is the chatbot's reply more emotional with neuron steering than without?

Uses the real backend path (``backend.llama.LlamaRuntime.generate`` with the
MindMate system prompt and the panels from ``build_steering_panels``).  Every
test message is answered unsteered (alpha 0), steered at each ``--alphas``
value, and with a random-neuron control: the same layers, the same number of
neurons and the same shift values, placed on randomly chosen neurons.

Replies are scored with the GoEmotions classifier used in causal_validation:
probability of the target emotion and of ``neutral``.  Fluency is the reply's
mean negative log-likelihood under the unsteered model plus the repeated-bigram
rate.  Steered and unsteered replies are paired by message (Wilcoxon).

Run from the project root:  python3 -m generation_analysis.evaluate_chat_steering
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from backend.config import SYSTEM_PROMPT
from backend.llama import LlamaRuntime

from .causal_validation import GOEMOTIONS, SCORER, goemotion_scores
from .common import MODEL_ID, RESULTS_DIR, write_json

MESSAGES = [
    "I failed my exam again and I feel like such a disappointment to my parents.",
    "My best friend stopped talking to me and I don't know why.",
    "yaar aaj bahut akela feel ho raha hai, kisi se baat karne ka mann nahi hai",
    "I have a presentation tomorrow and I can't stop shaking.",
    "My manager took credit for my work in front of everyone.",
    "mummy ki tabiyat theek nahi hai aur main ghar se door hoon",
    "I just don't feel anything these days. Everything is the same.",
    "I had a fight with my partner and said things I regret.",
    "I can't sleep at night, my mind keeps racing.",
    "My dog passed away yesterday.",
    "college mein sab mujhse aage nikal gaye, main peeche reh gaya",
    "Nobody at home listens to me, it's like I'm invisible.",
    "placement season chal raha hai aur mujhe ek bhi offer nahi mila, bahut tension hai",
    "I keep comparing myself to people on Instagram and it makes me feel worthless.",
    "I finally got the job offer I was waiting for!",
    "aaj mera birthday hai aur sab dost surprise dene aaye!",
    "I started going to the gym this week.",
    "My sister just had a baby and I got to hold her today.",
    "I spoke up in class for the first time today.",
    "What should I cook for dinner tonight?",
    "Can you tell me how to organise my study schedule?",
    "kal se naya semester shuru ho raha hai",
    "I moved to a new city last month for work.",
    "I've been thinking about calling my dad after a long time.",
]


def repeated_bigram_rate(text: str) -> float:
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def reply_nll(runtime: LlamaRuntime, user_text: str, reply: str) -> float:
    """Mean NLL of the reply tokens under the unsteered model and the chat prompt."""
    tokenizer = runtime.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    reply_ids = tokenizer(reply, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if reply_ids.shape[1] == 0:
        return float("nan")
    device = next(runtime.model.parameters()).device
    ids = torch.cat([prompt_ids, reply_ids], dim=1).to(device)
    with torch.inference_mode():
        logits = runtime.model(input_ids=ids).logits[0, prompt_ids.shape[1] - 1:-1].float()
    return float(torch.nn.functional.cross_entropy(logits, ids[0, prompt_ids.shape[1]:]))


def random_panel(panel: dict, width: int, rng: np.random.Generator) -> dict:
    return {layer: (rng.choice(width, len(neurons), replace=False).tolist(), list(rng.permutation(shifts)))
            for layer, (neurons, shifts) in panel.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "chat_steering_eval"))
    parser.add_argument("--targets", nargs="+", default=["auto"],
                        help='Steering targets: "auto", "mirror" or emotion labels (default: auto).')
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--control-alpha", type=float, default=1.0, help="Alpha of the random-neuron control.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--messages-csv", help="Optional CSV with a 'text' column replacing the built-in messages.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    messages = pd.read_csv(args.messages_csv).text.tolist() if args.messages_csv else MESSAGES
    runtime = LlamaRuntime(args.model_id)
    if not runtime._emotion_panels:
        raise SystemExit("No steering panels. Run: python3 -m generation_analysis.build_steering_panels")
    width = runtime.model.model.layers[0].mlp.down_proj.in_features
    rng = np.random.default_rng(args.seed)
    for emotion in list(runtime._emotion_panels):
        runtime._emotion_panels[f"random:{emotion}"] = random_panel(runtime._emotion_panels[emotion], width, rng)

    def reply(text: str, alpha: float, emotion: str | None) -> str:
        return runtime.generate([{"role": "user", "content": text}], max_new_tokens=args.max_new_tokens,
                                deterministic=True, temperature=0.7, steering_alpha=alpha,
                                steering_emotion=emotion)["response"]

    rows = []
    for index, text in enumerate(messages):
        detected = runtime.detect_emotion(text)["emotion"]
        baseline = reply(text, 0.0, None)
        baseline_nll = reply_nll(runtime, text, baseline)
        for requested in args.targets:
            target = runtime.steering_target(detected, requested)
            common = {"message_id": index, "user_text": text, "detected": detected, "requested": requested,
                      "target": target}
            rows.append({**common, "condition": "baseline", "alpha": 0.0, "reply": baseline, "nll": baseline_nll})
            for alpha in args.alphas:
                steered = reply(text, alpha, target)
                rows.append({**common, "condition": "steered", "alpha": alpha, "reply": steered,
                             "nll": reply_nll(runtime, text, steered)})
            control = reply(text, args.control_alpha, f"random:{target}")
            rows.append({**common, "condition": "random", "alpha": args.control_alpha, "reply": control,
                         "nll": reply_nll(runtime, text, control)})
        print(f"[{index + 1}/{len(messages)}] {detected}: {text[:60]}", flush=True)
    del runtime
    torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    scores = goemotion_scores(table.reply.tolist(), GOEMOTIONS, SCORER)
    table["target_probability"] = [scores[i, GOEMOTIONS.index(t)] for i, t in enumerate(table.target)]
    table["neutral_probability"] = scores[:, GOEMOTIONS.index("neutral")]
    table["repeated_bigram_rate"] = table.reply.map(repeated_bigram_rate)
    table.to_csv(output / "replies.csv", index=False)

    base = table[table.condition == "baseline"].set_index(["message_id", "requested"])
    summary = []
    for (requested, condition, alpha), group in table[table.condition != "baseline"].groupby(
            ["requested", "condition", "alpha"], sort=False):
        group = group.set_index(["message_id", "requested"])
        reference = base.loc[group.index]
        gain = (group.target_probability - reference.target_probability).to_numpy()
        summary.append({
            "requested": requested, "condition": condition, "alpha": alpha, "messages": len(group),
            "baseline_target_probability": float(reference.target_probability.mean()),
            "target_probability": float(group.target_probability.mean()),
            "mean_gain": float(gain.mean()),
            "messages_more_emotional": float((gain > 0).mean()),
            "wilcoxon_p": float(wilcoxon(gain, alternative="greater").pvalue) if np.any(gain) else 1.0,
            "neutral_probability_change": float((group.neutral_probability - reference.neutral_probability).mean()),
            "nll_change": float((group.nll - reference.nll).mean()),
            "repeated_bigram_rate": float(group.repeated_bigram_rate.mean()),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "summary.csv", index=False)
    write_json(output / "metadata.json", {
        "model_id": args.model_id, "targets": args.targets, "alphas": args.alphas,
        "control": "random neurons, same layers, counts and shift values", "control_alpha": args.control_alpha,
        "max_new_tokens": args.max_new_tokens, "scorer": SCORER, "messages": len(messages), "seed": args.seed,
        "system_prompt": "backend.config.SYSTEM_PROMPT",
    })
    pd.set_option("display.width", 220)
    print(summary.round(3).to_string(index=False))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
