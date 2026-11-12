#!/usr/bin/env python3
"""Controlled steering-strength sweep for the Emotion Reasoner + neuron steering.

For every test message the Emotion Reasoner plans a strategy and a response
emotion.  Replies are then generated with fixed greedy decoding under four
conditions:

  baseline        MindMate system prompt, no steering
  kg_prompt       system prompt + KG strategy guidance, no steering
  steer           MindMate system prompt, panel steered at each --alphas value
  kg_steer        strategy guidance + panel steered at each --alphas value

Scoring uses models that are independent of the generator and of each other:

  target emotion   GoEmotions XLM-R classifier, probability of the planned emotion
  empathy, contextual relevance, fluency
                   1-5 rubric judgements by Qwen2.5-7B-Instruct (expected value
                   over the digit logits, so the judgement is deterministic)
  fluency (2nd)    repeated-bigram rate; reply NLL under Qwen
  semantic preservation
                   MiniLM cosine similarity to the unsteered reply of the same prompt
  relevance (2nd)  MiniLM cosine similarity to the user message

It also records, for the gold-labelled messages and a labelled dataset, the
frozen probe's prediction, the XLM-R prediction and the KG-fused prediction,
to measure how reliable user-emotion detection is.

Stages (each resumable):  generate -> score.  Run from the project root:
  python3 -m generation_analysis.steering_policy_experiment
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backend.config import SYSTEM_PROMPT
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, strategy_prompt
from chatbot_feature.emotion_reasoner.knowledge_graph import is_hinglish

from .causal_validation import GOEMOTIONS, SCORER
from .common import MODEL_ID, RESULTS_DIR, read_jsonl, write_json

JUDGE_ID = "Qwen/Qwen2.5-7B-Instruct"
STUDY_LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness",
                "surprise")
# (user message, gold user emotion).  Gold labels are the author's reading of the message.
MESSAGES = [
    # grief / sadness
    ("My dog passed away yesterday.", "sadness"),
    ("My grandmother died last week and I couldn't even go to the funeral.", "sadness"),
    ("papa nahi rahe, ek mahina ho gaya aur abhi bhi yakeen nahi hota", "sadness"),
    ("I cried all evening and I don't even know why.", "sadness"),
    ("sab kuch khatam sa lag raha hai, bahut dukh ho raha hai", "sadness"),
    # loneliness
    ("My best friend stopped talking to me and I feel so alone.", "sadness"),
    ("yaar aaj bahut akela feel ho raha hai, kisi se baat karne ka mann nahi hai", "sadness"),
    ("Nobody at home listens to me, it's like I'm invisible.", "sadness"),
    ("I moved to a new city and I have no one to talk to.", "sadness"),
    # anger / disapproval
    ("My manager took credit for my work in front of everyone.", "anger"),
    ("My roommate keeps eating my food and lying about it, I'm so done.", "anger"),
    ("boss ne sabke saamne meri insult kar di, bahut gussa aa raha hai", "anger"),
    ("I think the way my college treats students is completely wrong.", "disapproval"),
    ("My partner cheated on me and now acts like nothing happened.", "anger"),
    # fear / anxiety
    ("I have a presentation tomorrow and I can't stop shaking.", "fear"),
    ("I can't sleep at night, my mind keeps racing.", "fear"),
    ("kal interview hai aur mujhe bahut darr lag raha hai", "fear"),
    ("What if I never find a job? I'm scared I'm falling behind.", "fear"),
    ("mummy ki tabiyat theek nahi hai aur main ghar se door hoon", "fear"),
    # self-worth / academic
    ("I failed my exam again and I feel like such a disappointment to my parents.", "sadness"),
    ("I keep comparing myself to people on Instagram and it makes me feel worthless.", "sadness"),
    ("college mein sab mujhse aage nikal gaye, main peeche reh gaya", "sadness"),
    ("placement season chal raha hai aur mujhe ek bhi offer nahi mila, bahut tension hai", "fear"),
    ("I studied for weeks and still got bad grades. Maybe I'm just not good enough.", "sadness"),
    # joy / achievement
    ("I finally got the job offer I was waiting for!", "joy"),
    ("aaj mera birthday hai aur sab dost surprise dene aaye!", "joy"),
    ("I passed my driving test on the first try!", "joy"),
    ("My sister just had a baby and I got to hold her today.", "joy"),
    ("mera result aa gaya aur main pass ho gaya, bahut khush hoon!", "joy"),
    ("I spoke up in class for the first time today.", "joy"),
    # admiration / love
    ("My teacher stayed late to help me, she is honestly the best.", "admiration"),
    ("I really admire how my mom handled everything on her own.", "admiration"),
    ("I love my friends so much, they always show up for me.", "love"),
    ("meri behen ne mere liye itna kuch kiya, I love her so much", "love"),
    # neutral / numb / everyday
    ("I just don't feel anything these days. Everything is the same.", "neutral"),
    ("What should I cook for dinner tonight?", "neutral"),
    ("Can you tell me how to organise my study schedule?", "neutral"),
    ("kal se naya semester shuru ho raha hai", "neutral"),
    ("I've been thinking about calling my dad after a long time.", "neutral"),
    ("I started going to the gym this week.", "neutral"),
    # surprise
    ("I can't believe it, my old friend just showed up at my door after five years!", "surprise"),
    ("Wait, they cancelled the exam an hour before it started?!", "surprise"),
]

RUBRICS = {
    "empathy": "Empathy: does the reply show that it understands and cares about the user's feelings? "
               "1 = cold, dismissive or self-centred. 5 = deeply understanding and caring.",
    "relevance": "Contextual relevance: does the reply respond to what the user actually said, in a way that suits "
                 "their situation? 1 = off-topic, contradictory or inappropriate. 5 = fully on point and appropriate.",
    "fluency": "Fluency: is the reply natural, coherent and grammatical in its own language (English or romanised "
               "Hindi)? 1 = garbled, repetitive or nonsensical. 5 = perfectly natural.",
}


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------- generation

def generate(args, output: Path) -> None:
    from backend.llama import LlamaRuntime
    runtime = LlamaRuntime(args.model_id)
    agent = EmotionReasonerAgent(available_panels=list(runtime._emotion_panels))
    path = output / "generations.jsonl"
    done = {(row["message_id"], row["condition"], row["alpha"]) for row in read_jsonl(path)} if path.exists() else set()

    def reply(text: str, prompt: str, alpha: float, emotion: str) -> str:
        return runtime.generate([{"role": "user", "content": text}], max_new_tokens=args.max_new_tokens,
                                deterministic=True, temperature=0.7, system_prompt=prompt, steering_alpha=alpha,
                                steering_emotion=emotion)["response"]

    plans = []
    with path.open("a", encoding="utf-8") as handle:
        for index, (text, gold) in enumerate(MESSAGES):
            probe = runtime.detect_emotion(text)["scores"]
            plan = agent.plan(text, probe).to_dict()
            plans.append({"message_id": index, "text": text, "gold": gold, "hinglish": is_hinglish(text),
                          "probe_scores": probe, "probe_emotion": max(probe, key=probe.get),
                          "fused_emotion": plan["user_emotion"], "strategy": plan["strategy"],
                          "target": plan["response_emotion"], "policy_alpha": plan["steering_alpha"]})
            target = plan["response_emotion"]
            if target is None:
                continue
            kg_prompt = strategy_prompt(SYSTEM_PROMPT, plan)
            runs = [("baseline", SYSTEM_PROMPT, 0.0), ("kg_prompt", kg_prompt, 0.0)]
            runs += [("steer", SYSTEM_PROMPT, alpha) for alpha in args.alphas]
            runs += [("kg_steer", kg_prompt, alpha) for alpha in args.alphas]
            for condition, prompt, alpha in runs:
                if (index, condition, alpha) in done:
                    continue
                handle.write(json.dumps({"message_id": index, "condition": condition, "alpha": alpha,
                                         "target": target, "reply": reply(text, prompt, alpha, target)},
                                        ensure_ascii=False) + "\n")
                handle.flush()
            print(f"[{index + 1}/{len(MESSAGES)}] {plan['strategy']} -> {target}: {text[:50]}", flush=True)
    write_json(output / "plans.json", plans)

    # Probe reliability on a labelled set.
    if args.labelled_csv:
        data = pd.read_csv(args.labelled_csv)
        data = data[data.label.isin(STUDY_LABELS)].reset_index(drop=True)
        rows = []
        for row in data.itertuples():
            probe = runtime.detect_emotion(row.text)["scores"]
            state = agent.understand(row.text, probe)
            rows.append({"text": row.text, "gold": row.label, "probe_emotion": max(probe, key=probe.get),
                         "probe_confidence": max(probe.values()), "fused_emotion": state["emotion"],
                         "cues": bool(state["causes"])})
        pd.DataFrame(rows).to_csv(output / "detection_labelled.csv", index=False)
    del runtime
    release()


# ---------------------------------------------------------------------- scoring

def classifier_scores(texts: list[str]) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SCORER)
    model = AutoModelForSequenceClassification.from_pretrained(SCORER).to("cuda").eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            encoded = tokenizer([t or " " for t in texts[start:start + 64]], return_tensors="pt", padding=True,
                                truncation=True, max_length=256).to("cuda")
            rows.append(torch.sigmoid(model(**encoded).logits.float()).cpu().numpy())
    del model
    release()
    return np.concatenate(rows)


def judge_scores(pairs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Qwen rubric judgements (expected value over digits 1-5) and reply NLL."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_ID)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(JUDGE_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    digits = [tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(1, 6)]
    values = torch.arange(1, 6, dtype=torch.float32)
    results: dict[str, list[float]] = {name: [] for name in RUBRICS}
    results["judge_nll"] = []
    with torch.inference_mode():
        for name, rubric in RUBRICS.items():
            prompts = [tokenizer.apply_chat_template([
                {"role": "system", "content": "You are a strict, careful evaluator of chatbot replies in a "
                                              "supportive-conversation app. Answer with a single digit from 1 to 5."},
                {"role": "user", "content": f"User message:\n{user}\n\nChatbot reply:\n{reply}\n\n{rubric}\n\nScore:"},
            ], tokenize=False, add_generation_prompt=True) for user, reply in pairs]
            for start in range(0, len(prompts), 16):
                encoded = tokenizer(prompts[start:start + 16], return_tensors="pt", padding=True).to("cuda")
                logits = model(**encoded).logits[:, -1, digits].float()
                results[name] += (torch.softmax(logits, -1).cpu() @ values).tolist()
        for _, reply in pairs:
            ids = tokenizer("Reply: " + (reply or " "), return_tensors="pt").to("cuda")
            results["judge_nll"].append(float(model(**ids, labels=ids["input_ids"]).loss))
    del model
    release()
    return {name: np.asarray(scores) for name, scores in results.items()}


def repeated_bigram_rate(text: str) -> float:
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def score(output: Path) -> None:
    table = pd.DataFrame(read_jsonl(output / "generations.jsonl"))
    plans = pd.DataFrame(read_json_list(output / "plans.json")).set_index("message_id")
    table = table.join(plans[["text", "gold", "hinglish", "strategy"]], on="message_id")
    probabilities = classifier_scores(table.reply.tolist())
    table["target_probability"] = [probabilities[i, GOEMOTIONS.index(t)] for i, t in enumerate(table.target)]
    for label in ("neutral", "caring"):
        table[f"{label}_probability"] = probabilities[:, GOEMOTIONS.index(label)]
    judged = judge_scores(list(zip(table.text, table.reply)))
    for name, values in judged.items():
        table[name] = values
    table["repeated_bigram_rate"] = table.reply.map(repeated_bigram_rate)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    replies = encoder.encode(table.reply.tolist(), normalize_embeddings=True)
    users = encoder.encode(table.text.tolist(), normalize_embeddings=True)
    table["similarity_to_user"] = (replies * users).sum(1)
    # Semantic preservation: similarity to the unsteered reply generated with the same prompt.
    reference = {"steer": "baseline", "kg_steer": "kg_prompt", "baseline": "baseline", "kg_prompt": "kg_prompt"}
    position = {(row.message_id, row.condition): i for i, row in enumerate(table.itertuples()) if row.alpha == 0}
    table["similarity_to_unsteered"] = [float(replies[i] @ replies[position[(row.message_id, reference[row.condition])]])
                                        for i, row in enumerate(table.itertuples())]
    table.to_csv(output / "scores.csv", index=False)

    # User-emotion detection by the independent classifier, for the reliability analysis.
    study = [GOEMOTIONS.index(label) for label in STUDY_LABELS]
    plans = plans.reset_index()
    plans["classifier_emotion"] = [STUDY_LABELS[i] for i in classifier_scores(plans.text.tolist())[:, study].argmax(1)]
    plans.drop(columns=["probe_scores"]).to_csv(output / "detection_messages.csv", index=False)
    labelled = output / "detection_labelled.csv"
    if labelled.exists():
        data = pd.read_csv(labelled)
        data["classifier_emotion"] = [STUDY_LABELS[i] for i in classifier_scores(data.text.tolist())[:, study].argmax(1)]
        data.to_csv(labelled, index=False)
    print(f"Saved scores to {output / 'scores.csv'}")


def read_json_list(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR / "steering_policy"))
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--labelled-csv", default="data/synthetic_hinglish_emotion_1000.csv")
    parser.add_argument("--stages", nargs="+", choices=("generate", "score"), default=["generate", "score"])
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if "generate" in args.stages:
        generate(args, output)
    if "score" in args.stages:
        score(output)
    write_json(output / "experiment_metadata.json", {
        "model_id": args.model_id, "alphas": args.alphas, "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy (deterministic)", "messages": len(MESSAGES), "emotion_scorer": SCORER,
        "judge": JUDGE_ID, "rubrics": RUBRICS, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "conditions": ["baseline", "kg_prompt", "steer", "kg_steer"],
    })


if __name__ == "__main__":
    main()
