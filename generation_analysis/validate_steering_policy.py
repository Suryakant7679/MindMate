#!/usr/bin/env python3
"""Held-out check of the deployed pipeline (reasoner + calibrated policy + send-only-if-better).

Messages here were not used for calibration.  For each one the real service
produces its reply; that reply is compared with the plain baseline reply and
the KG-prompt-only reply using the same independent scorers as the sweep.

Run from the project root:  python3 -m generation_analysis.validate_steering_policy
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import SYSTEM_PROMPT
from backend.emotion_service import EmotionSteeringService

from .causal_validation import GOEMOTIONS
from .common import MODEL_ID, RESULTS_DIR
from .steering_policy_experiment import classifier_scores, judge_scores, release

HELD_OUT = [
    "My cat has been missing for three days and I can't stop crying.",
    "I didn't get into the college I wanted and everyone else did.",
    "Everyone in my group made plans without me again.",
    "My landlord is refusing to return my deposit for no reason.",
    "My brother read my private diary and laughed about it.",
    "I have my final exam tomorrow and I feel like I'm going to throw up.",
    "I'm scared my parents will find out about my grades.",
    "I messed up a big project at work and I feel so stupid.",
    "No matter how hard I try, I'm never as good as my sister.",
    "I got promoted today!",
    "I ran my first 5k this morning and didn't stop once!",
    "My best friend drove four hours just to see me, she is amazing.",
    "I'm so grateful for my grandfather, he taught me everything.",
    "I don't really know what to say today.",
    "Can you suggest a good way to start journaling?",
    "My flight got cancelled and now I'm stuck here, unbelievable.",
    "I feel like I'm a burden to everyone around me.",
    "It's been a year since mom passed and today is really hard.",
    "aaj office mein sabne mera mazaak udaya, bahut bura laga",
    "meri best friend ki shaadi hai aur main bahut khush hoon!",
    "exam ka result kal aayega aur mujhe neend nahi aa rahi",
    "ghar ki bahut yaad aa rahi hai, yahan koi apna nahi hai",
    "maine itni mehnat ki phir bhi select nahi hua, main kisi kaam ka nahi",
    "aaj kuch khaas nahi hua, bas normal din tha",
]


def main() -> None:
    from backend.llama import LlamaRuntime
    output = Path(RESULTS_DIR / "steering_policy")
    runtime = LlamaRuntime(MODEL_ID)
    service = EmotionSteeringService(runtime)
    rows = []
    for index, text in enumerate(HELD_OUT):
        messages = [{"role": "user", "content": text}]
        baseline = runtime.generate(messages, max_new_tokens=120, deterministic=True, temperature=0.7,
                                    system_prompt=SYSTEM_PROMPT)["response"]
        turn_id, plan = service.begin_turn(messages)
        analysis = service.respond(turn_id, max_new_tokens=120)
        common = {"message_id": index, "text": text, "strategy": plan["strategy"], "target": plan["response_emotion"],
                  "user_emotion": plan["user_emotion"], "status": analysis["decision"]["status"],
                  "applied_alpha": analysis["decision"]["applied_alpha"]}
        rows.append({**common, "condition": "baseline", "reply": baseline})
        rows.append({**common, "condition": "kg_prompt", "reply": analysis["effect"]["unsteered_reply"]})
        rows.append({**common, "condition": "deployed", "reply": analysis["reply"]})
        print(f"[{index + 1}/{len(HELD_OUT)}] {plan['user_emotion']} -> {plan['strategy']} -> "
              f"{analysis['decision']['status']} (alpha {analysis['decision']['applied_alpha']})", flush=True)
    del service, runtime
    release()
    table = pd.DataFrame(rows)
    probabilities = classifier_scores(table.reply.tolist())
    table["target_probability"] = [probabilities[i, GOEMOTIONS.index(t)] if isinstance(t, str) else float("nan")
                                   for i, t in enumerate(table.target)]
    for name, values in judge_scores(list(zip(table.text, table.reply))).items():
        table[name] = values
    table.to_csv(output / "held_out_validation.csv", index=False)
    metrics = ["target_probability", "empathy", "relevance", "fluency"]
    order = ["baseline", "kg_prompt", "deployed"]
    pd.set_option("display.width", 200)
    print(table.groupby("condition")[metrics].mean().round(3).reindex(order).to_string())
    print(table[table.condition == "deployed"].status.value_counts().to_string())
    steered = table[table.status == "applied"]
    print("\nTurns where steering was applied:")
    print(steered.groupby("condition")[metrics].mean().round(3).reindex(order).to_string())


if __name__ == "__main__":
    main()
