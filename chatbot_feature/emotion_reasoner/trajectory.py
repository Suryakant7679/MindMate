"""Emotion trajectory of a conversation: what was detected for the user, turn by turn.

Everything here is *observational*.  The numbers describe what the emotion
detector (GoEmotions classifier fused with knowledge-graph cues) estimated
for each user message, how sure it was, and how those estimates moved across
turns.  They are not calibrated mood measurements and they say nothing about
what caused a change: a later message can read as more positive because the
topic moved on, because distress regresses toward the user's usual level, or
because of the reply.  Treatment effects of steering are established only by
the controlled offline experiments (same prompt, steered against unsteered).

Mood estimate
-------------
``expected_valence`` of a turn is the probability-weighted valence of the
fused emotion distribution, ``sum_e p(e) * valence(e)``, with the valence of
each of the ten labels taken from the knowledge graph (-1 .. +1).  It uses the
whole distribution, so a turn the detector is unsure about sits near the
middle instead of jumping to the valence of a barely-winning label.  The
conversation mood is the mean over turns; an exponential moving average
(weight 0.5 on the newest turn) shows the recent level.  ``generation_analysis
.trajectory_validation`` checks this estimate against labelled messages per
language and reports how (mis)calibrated the detector's confidence is.

The module has no model dependency: it receives the per-turn output of
``EmotionReasonerAgent.understand`` and the detector's own scores.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .knowledge_graph import EMOTIONS, HINGLISH_WORDS

LABELS = tuple(EMOTIONS)
EMA_WEIGHT = 0.5                # weight of the newest turn in the moving average
UNCERTAIN_CONFIDENCE = 0.45     # top fused probability below which a turn is flagged uncertain
UNCERTAIN_MARGIN = 0.10         # ... or when the two best labels are closer than this
MIN_TREND_TURNS = 5             # a slope over fewer turns is reported but marked not interpretable
RATING_SCALE = (1, 5)           # voluntary self-rating: 1 = very low ... 5 = very good
CAVEATS = (
    "Detector estimates, not calibrated mood measurements: a confidence of 0.8 does not mean an 80% chance of being right.",
    "Observed changes between turns are not treatment effects: topic changes, regression toward the user's usual "
    "level and the reply all move the next message, and steering is applied to selected turns only.",
    "Valence per label is a fixed knowledge-graph value; mixed feelings and sarcasm are not represented.",
)


def language_of(text: str) -> str:
    """``hindi`` (Devanagari), ``hinglish`` (Romanised Hindi mixed with English) or ``english``."""
    letters = [character for character in text if character.isalpha()]
    if letters and sum("ऀ" <= character <= "ॿ" for character in letters) / len(letters) >= 0.3:
        return "hindi"
    words = re.findall(r"[a-z]+", text.lower())
    if re.search(r"[ऀ-ॿ]", text) or (words and sum(word in HINGLISH_WORDS for word in words) / len(words) >= 0.2):
        return "hinglish"
    return "english"


def normalised(scores: Mapping[str, float]) -> dict[str, float]:
    """Distribution over the ten study labels (missing labels count as zero)."""
    values = {label: max(float(scores.get(label, 0.0)), 0.0) for label in LABELS}
    total = sum(values.values())
    return {label: (value / total if total else 1.0 / len(LABELS)) for label, value in values.items()}


def expected_valence(distribution: Mapping[str, float]) -> float:
    return sum(float(distribution.get(label, 0.0)) * EMOTIONS[label][0] for label in LABELS)


def expected_arousal(distribution: Mapping[str, float]) -> float:
    return sum(float(distribution.get(label, 0.0)) * EMOTIONS[label][1] for label in LABELS)


def entropy(distribution: Mapping[str, float]) -> float:
    """Shannon entropy divided by its maximum (0 = one label certain, 1 = all labels equally likely)."""
    values = [value for value in distribution.values() if value > 0]
    return -sum(value * math.log(value) for value in values) / math.log(len(LABELS)) if values else 1.0


def js_divergence(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence (base 2, 0..1) between two emotion distributions."""
    total = 0.0
    for label in LABELS:
        p, q = float(first.get(label, 0.0)), float(second.get(label, 0.0))
        middle = (p + q) / 2
        if p > 0:
            total += 0.5 * p * math.log2(p / middle)
        if q > 0:
            total += 0.5 * q * math.log2(q / middle)
    return max(0.0, min(1.0, total))


def turn_state(text: str, understood: Mapping[str, Any], detector_scores: Mapping[str, float]) -> dict[str, Any]:
    """Per-turn record from ``EmotionReasonerAgent.understand`` and the detector's own scores."""
    distribution = normalised(understood["fused"])
    detector = normalised(detector_scores)
    ranked = sorted(distribution.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    detector_emotion = max(detector, key=detector.get)
    return {
        "text": text,
        "language": language_of(text),
        "emotion": understood["emotion"],
        "confidence": round(float(understood["confidence"]), 4),
        "intensity": round(float(understood["intensity"]), 4),
        "label_valence": round(float(understood["valence"]), 4),
        "distribution": {label: round(value, 4) for label, value in distribution.items()},
        "detector_distribution": {label: round(value, 4) for label, value in detector.items()},
        "detector_emotion": detector_emotion,
        "detector_agrees": detector_emotion == understood["emotion"],
        "expected_valence": round(expected_valence(distribution), 4),
        "expected_arousal": round(expected_arousal(distribution), 4),
        "entropy": round(entropy(distribution), 4),
        "margin": round(margin, 4),
        "uncertain": bool(ranked[0] < UNCERTAIN_CONFIDENCE or margin < UNCERTAIN_MARGIN),
        "causes": [cause["cause"] for cause in understood.get("causes", [])],
    }


def reply_state(text: str, scores: Mapping[str, float], target: str | None = None) -> dict[str, Any]:
    """Emotion estimate of a chatbot reply (classifier on the reply text; describes the text, not the user)."""
    distribution = normalised(scores)
    return {
        "emotion": max(distribution, key=distribution.get),
        "distribution": {label: round(value, 4) for label, value in distribution.items()},
        "expected_valence": round(expected_valence(distribution), 4),
        "target": target,
        "target_score": round(float(scores[target]), 4) if target and target in scores else None,
        "language": language_of(text),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _slope(values: Sequence[float]) -> float | None:
    """Least-squares change per turn."""
    if len(values) < 2:
        return None
    xs = range(len(values))
    mean_x, mean_y = (len(values) - 1) / 2, sum(values) / len(values)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values)) / denominator


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) < 4 or len(set(first)) < 2 or len(set(second)) < 2:
        return None
    a, b = _ranks(first), _ranks(second)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    spread = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return covariance / spread if spread else None


def rating_valence(rating: float) -> float:
    """Self-rating 1..5 mapped linearly to -1..+1."""
    low, high = RATING_SCALE
    return (float(rating) - (low + high) / 2) / ((high - low) / 2)


def self_report(turns: Sequence[Mapping[str, Any]], ratings: Mapping[int, float]) -> dict[str, Any]:
    """Voluntary user ratings against the detector's estimate for the same turns."""
    pairs = []
    for index, rating in sorted(ratings.items()):
        if 0 <= index < len(turns):
            reported, estimated = rating_valence(rating), turns[index]["expected_valence"]
            pairs.append({"turn": index + 1, "rating": rating, "reported_valence": round(reported, 4),
                          "estimated_valence": estimated, "error": round(estimated - reported, 4),
                          "same_sign": (reported > 0) == (estimated > 0) if reported and estimated else None})
    signed = [pair["same_sign"] for pair in pairs if pair["same_sign"] is not None]
    correlation = spearman([pair["reported_valence"] for pair in pairs], [pair["estimated_valence"] for pair in pairs])
    return {
        "n": len(pairs), "pairs": pairs,
        "mean_abs_error": round(_mean([abs(pair["error"]) for pair in pairs]), 4) if pairs else None,
        "mean_error": round(_mean([pair["error"] for pair in pairs]), 4) if pairs else None,
        "sign_agreement": round(sum(signed) / len(signed), 4) if signed else None,
        "spearman": round(correlation, 4) if correlation is not None else None,
        "note": ("Self-ratings (1 = very low, 5 = very good) are mapped linearly to -1..+1 and compared with the "
                 "detector's expected valence for the same turn. A rank correlation needs at least four rated turns."),
    }


def summarize(turns: Sequence[Mapping[str, Any]], ratings: Mapping[int, float] | None = None) -> dict[str, Any]:
    """Conversation-level view of the per-turn states (oldest turn first)."""
    rows, moving = [], None
    for index, turn in enumerate(turns):
        previous = turns[index - 1] if index else None
        moving = turn["expected_valence"] if moving is None else EMA_WEIGHT * turn["expected_valence"] + (1 - EMA_WEIGHT) * moving
        rows.append({
            **turn, "turn": index + 1, "ema_valence": round(moving, 4),
            "delta_valence": round(turn["expected_valence"] - previous["expected_valence"], 4) if previous else None,
            "js_divergence": round(js_divergence(turn["distribution"], previous["distribution"]), 4) if previous else None,
            "label_changed": turn["emotion"] != previous["emotion"] if previous else None,
            "self_rating": (ratings or {}).get(index),
        })
    count = len(rows)
    mean_distribution = {label: round(sum(row["distribution"][label] for row in rows) / count, 4) if count else 0.0
                         for label in LABELS}
    spread = {label: round(_std([row["distribution"][label] for row in rows]) or 0.0, 4) for label in LABELS}
    frequency = {label: sum(row["emotion"] == label for row in rows) for label in LABELS}
    valences = [row["expected_valence"] for row in rows]
    changes = [abs(row["delta_valence"]) for row in rows if row["delta_valence"] is not None]
    switches = [row["label_changed"] for row in rows if row["label_changed"] is not None]
    half = count // 2

    def dominant(part: Sequence[Mapping[str, Any]]) -> str | None:
        if not part:
            return None
        totals = {label: sum(row["distribution"][label] for row in part) for label in LABELS}
        return max(totals, key=totals.get)
    slope = _slope(valences)
    languages: dict[str, int] = {}
    for row in rows:
        languages[row["language"]] = languages.get(row["language"], 0) + 1
    return {
        "turns": rows,
        "labels": list(LABELS),
        "mean_distribution": dict(sorted(mean_distribution.items(), key=lambda item: -item[1])),
        "distribution_std": spread,
        "frequency": {label: value for label, value in sorted(frequency.items(), key=lambda item: -item[1]) if value},
        "dominant_emotion": dominant(rows),
        "dominant_first_half": dominant(rows[:half]) if half else None,
        "dominant_second_half": dominant(rows[half:]) if half else None,
        "mood": {
            "mean_expected_valence": round(_mean(valences), 4) if valences else None,
            "latest_ema": rows[-1]["ema_valence"] if rows else None,
            "trend_per_turn": round(slope, 4) if slope is not None else None,
            "trend_interpretable": count >= MIN_TREND_TURNS,
            "std": round(_std(valences), 4) if _std(valences) is not None else None,
            "mean_abs_change": round(_mean(changes), 4) if changes else None,
            "label_switch_rate": round(sum(switches) / len(switches), 4) if switches else None,
            "uncertain_share": round(sum(row["uncertain"] for row in rows) / count, 4) if count else None,
            "detector_disagreement_share": round(sum(not row["detector_agrees"] for row in rows) / count, 4) if count else None,
            "mean_confidence": round(_mean([row["confidence"] for row in rows]), 4) if count else None,
            "mean_entropy": round(_mean([row["entropy"] for row in rows]), 4) if count else None,
        },
        "languages": languages,
        "self_report": self_report(rows, ratings or {}),
        "method": {
            "detector": "GoEmotions classifier scores fused with knowledge-graph cue evidence (ten labels)",
            "mood": "expected valence = sum over labels of fused probability x knowledge-graph valence; "
                    f"moving average with weight {EMA_WEIGHT} on the newest turn",
            "uncertainty": f"normalised entropy of the fused distribution; a turn is flagged when its top probability is below "
                           f"{UNCERTAIN_CONFIDENCE} or the two best labels are within {UNCERTAIN_MARGIN}",
            "change": "difference of expected valence and Jensen-Shannon divergence between consecutive turns",
            "trend": f"least-squares slope of expected valence per turn; not interpretable below {MIN_TREND_TURNS} turns",
        },
        "caveats": list(CAVEATS),
    }


def after_response(turns: Sequence[Mapping[str, Any]], replies: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """How the detected user emotion moved after each reply, grouped by whether that reply was steered.

    Observational only.  ``replies[i]`` describes the reply to user turn ``i`` (None when unknown) and may
    carry ``delivered`` ("steered" / "unsteered").  The groups are not comparable as a treatment and a
    control: the policy steers particular strategies and emotions, so the groups differ before the reply.
    """
    rows = []
    for index in range(len(turns) - 1):
        reply = replies[index] if index < len(replies) else None
        rows.append({
            "turn": index + 1, "user_emotion": turns[index]["emotion"], "next_user_emotion": turns[index + 1]["emotion"],
            "valence_before": turns[index]["expected_valence"], "valence_after": turns[index + 1]["expected_valence"],
            "delta_valence": round(turns[index + 1]["expected_valence"] - turns[index]["expected_valence"], 4),
            "reply_emotion": reply.get("emotion") if reply else None,
            "reply_valence": reply.get("expected_valence") if reply else None,
            "delivered": reply.get("delivered") if reply else None,
        })
    groups = {}
    for name in ("steered", "unsteered"):
        values = [row["delta_valence"] for row in rows if row["delivered"] == name]
        groups[name] = {"n": len(values), "mean_delta_valence": round(_mean(values), 4) if values else None}
    values = [row["delta_valence"] for row in rows]
    return {
        "rows": rows, "n": len(rows),
        "mean_delta_valence": round(_mean(values), 4) if values else None,
        "improved_share": round(sum(value > 0.05 for value in values) / len(values), 4) if values else None,
        "worsened_share": round(sum(value < -0.05 for value in values) / len(values), 4) if values else None,
        "by_delivery": groups,
        "note": ("Observed change of the detected user emotion from one message to the next. This is not the effect of the "
                 "reply or of steering: it is not randomised, steered turns are a selected subset, and negative states "
                 "tend to drift back toward neutral on their own. Causal effects of steering on the reply are measured "
                 "in the controlled offline experiments; effects on users would need a randomised study with user ratings."),
    }
