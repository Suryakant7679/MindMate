#!/usr/bin/env python3
"""How far can the detected user emotion (and its confidence) be trusted? Checked on labelled messages.

The detector is the one the chatbot uses: the GoEmotions classifier's scores on the user's message, fused with
knowledge-graph cue evidence by ``EmotionReasonerAgent.understand``.  Each language is evaluated separately:

  English   data/goemotions_hardlabels.csv: rows with exactly one of the ten study labels (others dropped)
  Hinglish  data/synthetic_hinglish_emotion_1000.csv: hand-authored and template Romanised Hinglish (labelled)

Reported per language: label accuracy (Wilson 95% CI), accuracy of the classifier alone, polarity AUC (positive vs
negative labels, neutral and surprise excluded), Spearman correlation of the estimated valence with the labels'
valence, expected calibration error (10 equal-width bins) and reliability bins.  Not a validation of users' feelings:
the labels describe the message text, and the Hinglish set is synthetic.

  python3 -m generation_analysis.trajectory_validation          # writes results/emotion_trajectory/validation.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
from chatbot_feature.emotion_reasoner.knowledge_graph import EMOTIONS
from chatbot_feature.emotion_reasoner.trajectory import normalised
from .steering_lab import GOEMOTIONS, STUDY_LABELS, classifier_scores, release

ROOT = Path("results/emotion_trajectory")
ENGLISH = Path("data/goemotions_hardlabels.csv")
HINGLISH = Path("data/synthetic_hinglish_emotion_1000.csv")
POSITIVE = {"joy", "love", "admiration"}
NEGATIVE = {"anger", "disapproval", "disgust", "fear", "sadness"}
BINS = 10


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def english_rows(limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(ENGLISH)
    rows = []
    for record in frame.itertuples(index=False):
        labels = [label for label in STUDY_LABELS if getattr(record, label, 0) == 1]
        if len(labels) == 1:
            rows.append({"text": str(record.text).strip(), "label": labels[0], "language": "english"})
    out = pd.DataFrame(rows)
    return out.sample(n=min(limit, len(out)), random_state=7).reset_index(drop=True) if limit else out


def hinglish_rows(limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(HINGLISH)
    out = pd.DataFrame({"text": frame.text.astype(str).str.strip(), "label": frame.label, "language": "hinglish"})
    out = out[out.label.isin(STUDY_LABELS)]
    return out.sample(n=min(limit, len(out)), random_state=7).reset_index(drop=True) if limit else out


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count half)."""
    pos, neg = scores[positive], scores[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    greater = sum(float((p > neg).sum()) + 0.5 * float((p == neg).sum()) for p in pos)
    return greater / (len(pos) * len(neg))


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 4:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(first, second).statistic)


def evaluate(frame: pd.DataFrame, probes: np.ndarray, agent: EmotionReasonerAgent, language: str) -> tuple[dict, list[dict]]:
    fused_labels, confidence, valence, alone_labels, expected = [], [], [], [], []
    for text, probe in zip(frame.text, probes):
        scores = {label: float(probe[i]) for i, label in enumerate(STUDY_LABELS)}
        understood = agent.understand(text, normalised(scores))
        fused_labels.append(understood["emotion"])
        confidence.append(float(understood["confidence"]))
        distribution = normalised(understood["fused"])
        expected.append(sum(distribution[label] * EMOTIONS[label][0] for label in distribution))
        alone_labels.append(max(scores, key=scores.get))
    labels = frame.label.to_numpy()
    correct = np.array([a == b for a, b in zip(fused_labels, labels)], dtype=float)
    alone = np.array([a == b for a, b in zip(alone_labels, labels)], dtype=float)
    confidence, expected = np.array(confidence), np.array(expected)
    truth_valence = np.array([EMOTIONS[label][0] for label in labels])
    mask = np.array([label in POSITIVE | NEGATIVE for label in labels])
    positive = np.array([label in POSITIVE for label in labels])
    low, high = wilson(int(correct.sum()), len(correct))
    edges = np.linspace(0, 1, BINS + 1)
    bins, ece = [], 0.0
    for index in range(BINS):
        inside = (confidence >= edges[index]) & ((confidence < edges[index + 1]) if index < BINS - 1 else (confidence <= 1))
        if inside.any():
            gap = abs(confidence[inside].mean() - correct[inside].mean())
            ece += inside.mean() * gap
            bins.append({"low": round(float(edges[index]), 2), "high": round(float(edges[index + 1]), 2), "n": int(inside.sum()),
                         "confidence": round(float(confidence[inside].mean()), 4), "accuracy": round(float(correct[inside].mean()), 4)})
    summary = {
        "language": language, "n": int(len(frame)), "accuracy": round(float(correct.mean()), 4),
        "accuracy_lo": round(low, 4), "accuracy_hi": round(high, 4),
        "fused_accuracy": round(float(correct.mean()), 4), "classifier_only_accuracy": round(float(alone.mean()), 4),
        "polarity_auc": round(auc(expected[mask], positive[mask]), 4) if mask.any() else None,
        "polarity_accuracy": round(float(((expected[mask] > 0) == positive[mask]).mean()), 4) if mask.any() else None,
        "valence_correlation": round(spearman(expected, truth_valence), 4),
        "ece": round(float(ece), 4), "mean_confidence": round(float(confidence.mean()), 4),
        "uncertain_share": round(float(np.mean([bool(value < 0.45) for value in confidence])), 4),
    }
    return summary, bins


def run(args) -> dict:
    agent = EmotionReasonerAgent()
    by_language, reliability, notes = [], [], []
    for language, frame in (("english", english_rows(args.limit)), ("hinglish", hinglish_rows(args.limit))):
        if not len(frame):
            notes.append(f"No labelled {language} messages found.")
            continue
        columns = [GOEMOTIONS.index(label) for label in STUDY_LABELS]
        probes = classifier_scores(frame.text.tolist(), _gate_scorer())[:, columns]
        summary, bins = evaluate(frame, probes, agent, language)
        by_language.append(summary)
        reliability.append({"language": language, "bins": bins})
    release()
    notes += [
        "The detector is the chatbot's: GoEmotions classifier scores fused with knowledge-graph cue evidence.",
        "Labels describe the message text. The Hinglish set is synthetic (hand-authored and template sentences), "
        "so its accuracy is an upper bound for real Hinglish messages.",
        "Accuracy is measured against one label per message; mixed feelings are scored as errors.",
        "A small expected calibration error is needed before the detector's confidence is read as a probability.",
    ]
    payload = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
               "detector": f"{_gate_scorer()} fused with knowledge-graph cues (EmotionReasonerAgent.understand)",
               "messages": int(sum(item["n"] for item in by_language)), "by_language": by_language,
               "reliability": reliability, "notes": notes}
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "validation.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(pd.DataFrame(by_language).to_string(index=False))
    return payload


def _gate_scorer() -> str:
    """The run-time detector's classifier (the same model the chatbot loads)."""
    from backend.emotion_service import SCORER_ID
    return SCORER_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Messages per language (default: all).")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
