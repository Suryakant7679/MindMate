"""Steered against unsteered reply to the same prompt: what changed, and was it better?

Three kinds of evidence are kept apart:

``automatic``  computed at run time from the two texts and the reply-emotion classifier: the
               classifier's score for the intended emotion, repetition, length, lexical overlap and
               reply language.  Cheap and immediate, but an estimate: the classifier reads surface
               wording and none of these numbers measures empathy or relevance.
``judged``     ratings of both replies by the two judge models (Qwen2.5-7B-Instruct and Gemma
               3-27B-IT) and an independent emotion classifier, added offline by
               ``generation_analysis.turn_evaluation``.  Absent until that has been run.
``neural``     measured inside the model (see ``intervention.py``); not part of this module.

The verdict (``improved`` / ``harmed`` / ``no_meaningful_change``) names its basis.  Its thresholds
are the ones the run-time safeguards and the calibration rule already use; they are conventions for
sorting single replies, not validated cut-offs.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .knowledge_graph import is_hinglish
from .tone import degenerate_reply
from .trajectory import language_of

MIN_GAIN = 0.05                 # classifier target-score change counted as a gain / loss
MAX_REPETITION_INCREASE = 0.05
MIN_LENGTH_RATIO = 0.4
MAX_LENGTH_RATIO = 2.0
JUDGED_HARM = 0.5               # a quality rating lower by more than this (1-5 scale, mean of judges) is harm
JUDGED_GAIN = 0.1               # each judge must raise the emotion rating by at least this
QUALITY = ("appropriateness", "empathy", "relevance", "fluency", "safety", "naturalness")


def repeated_bigram_rate(text: str) -> float:
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def lexical_overlap(first: str, second: str) -> float:
    """F1 of the longest common word subsequence (ROUGE-L); wording kept, not meaning."""
    a, b = re.findall(r"\w+", first.lower()), re.findall(r"\w+", second.lower())
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for word in a:
        current = [0]
        for index, other in enumerate(b):
            current.append(previous[index] + 1 if word == other else max(previous[index + 1], current[-1]))
        previous = current
    common = previous[-1]
    if not common:
        return 0.0
    precision, recall = common / len(b), common / len(a)
    return 2 * precision * recall / (precision + recall)


def describe(text: str, scores: Mapping[str, float], target: str | None) -> dict[str, Any]:
    top = max(scores, key=scores.get) if scores else None
    return {"target_score": round(float(scores[target]), 4) if target and target in scores else None,
            "top_emotion": top, "words": len(text.split()), "repetition": round(repeated_bigram_rate(text), 4),
            "language": language_of(text), "degenerate": degenerate_reply(text), "questions": text.count("?")}


def compare_replies(user_text: str, unsteered: str, steered: str | None, unsteered_scores: Mapping[str, float],
                    steered_scores: Mapping[str, float] | None, target: str | None, *, delivered: str,
                    shadow: bool = False) -> dict[str, Any]:
    """Run-time comparison of the two replies to one prompt.

    ``delivered`` is "steered" or "unsteered"; ``shadow`` marks a steered candidate generated only for
    this comparison (research mode) and never eligible to be sent.
    """
    before = describe(unsteered, unsteered_scores, target)
    if steered is None or steered_scores is None:
        return {"available": False, "delivered": delivered, "shadow": False, "unsteered": before, "steered": None,
                "automatic": None, "judged": None,
                "verdict": {"label": "not_compared", "basis": "none",
                            "reasons": ["no steered reply was generated for this turn"]}}
    after = describe(steered, steered_scores, target)
    user_hinglish = is_hinglish(user_text)
    gain = (after["target_score"] - before["target_score"]
            if after["target_score"] is not None and before["target_score"] is not None else None)
    automatic = {
        "identical": steered.strip() == unsteered.strip(),
        "target_gain": round(gain, 4) if gain is not None else None,
        "repetition_change": round(after["repetition"] - before["repetition"], 4),
        "length_ratio": round(after["words"] / max(before["words"], 1), 4),
        "lexical_overlap": round(lexical_overlap(unsteered, steered), 4),
        "user_language": language_of(user_text),
        "hinglish_kept": (is_hinglish(steered) or not is_hinglish(unsteered)) if user_hinglish else None,
        "hinglish_gained": (is_hinglish(steered) and not is_hinglish(unsteered)) if user_hinglish else None,
        "score_changes": {label: round(float(steered_scores[label]) - float(unsteered_scores[label]), 4)
                          for label in unsteered_scores if label in steered_scores},
    }
    harms, gains = [], []
    if after["degenerate"] and not before["degenerate"]:
        harms.append("the steered reply is not a conversational answer")
    if automatic["repetition_change"] > MAX_REPETITION_INCREASE:
        harms.append(f"more repetitive ({automatic['repetition_change']:+.2f} repeated-bigram rate)")
    if automatic["length_ratio"] < MIN_LENGTH_RATIO:
        harms.append(f"much shorter ({automatic['length_ratio']:.2f}x the unsteered length)")
    if automatic["length_ratio"] > MAX_LENGTH_RATIO:
        harms.append(f"much longer ({automatic['length_ratio']:.2f}x the unsteered length)")
    if automatic["hinglish_kept"] is False:
        harms.append("left the user's Hinglish for English")
    if gain is not None and gain <= -MIN_GAIN:
        harms.append(f"lower {target} score ({gain:+.2f})")
    if gain is not None and gain >= MIN_GAIN:
        gains.append(f"higher {target} score ({gain:+.2f})")
    if automatic["hinglish_gained"]:
        gains.append("reply returned to the user's Hinglish")
    if automatic["identical"]:
        label, reasons = "no_meaningful_change", ["the two replies are identical"]
    elif harms:
        label, reasons = "harmed", harms
    elif gains:
        label, reasons = "improved", gains
    else:
        label, reasons = "no_meaningful_change", [
            f"{target} score changed by {gain:+.2f}, below the {MIN_GAIN:.2f} threshold" if gain is not None
            else "no automatic measure moved beyond its threshold"]
    return {"available": True, "delivered": delivered, "shadow": shadow, "unsteered": before, "steered": after,
            "automatic": automatic, "judged": None,
            "verdict": {"label": label, "basis": "automatic", "reasons": reasons,
                        "note": "Automatic estimate from the classifier score, repetition, length and language. "
                                "Empathy, relevance, fluency and meaning are judged offline."}}


def judged_verdict(judged: Mapping[str, Any]) -> dict[str, Any]:
    """Verdict from offline judge ratings (``judged["difference"]`` holds steered minus unsteered per metric)."""
    difference = judged.get("difference", {})
    both, per_judge = difference.get("both", {}), {name: difference[name] for name in difference if name != "both"}
    quality = {metric: both[metric] for metric in (*QUALITY, "hinglish_naturalness") if both.get(metric) is not None}
    worst = min(quality.items(), key=lambda item: item[1]) if quality else None
    preference = judged.get("preference", {}).get("both")
    emotion_gains = [values.get("emotion") for values in per_judge.values() if values.get("emotion") is not None]
    reasons = []
    if worst and worst[1] < -JUDGED_HARM:
        label = "harmed"
        reasons.append(f"{worst[0].replace('_', ' ')} fell by {abs(worst[1]):.2f} points (mean of the judges)")
    elif (emotion_gains and all(gain >= JUDGED_GAIN for gain in emotion_gains)
          and (preference is None or preference >= 0.5)):
        label = "improved"
        reasons.append(f"every judge rated the intended emotion higher (mean {both.get('emotion', 0):+.2f})"
                       + (f" and the steered reply was preferred ({preference:.2f})" if preference is not None else ""))
    else:
        label = "no_meaningful_change"
        reasons.append("the judges did not agree on a gain and no quality rating fell by more than "
                       f"{JUDGED_HARM} points")
    return {"label": label, "basis": "judged", "reasons": reasons,
            "note": "From the two offline judge models. Single-reply ratings are noisy; the judges agree only "
                    "moderately, so read this together with the aggregate experiments."}


def with_judged(comparison: Mapping[str, Any], judged: Mapping[str, Any] | None) -> dict[str, Any]:
    """Attach offline judge results to a run-time comparison; the judged verdict then takes precedence."""
    if not judged or not comparison.get("available"):
        return dict(comparison)
    return {**comparison, "judged": dict(judged), "automatic_verdict": comparison["verdict"],
            "verdict": judged_verdict(judged)}


def tally(comparisons: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Counts of verdicts over the compared turns of a conversation (shares of compared turns)."""
    compared = [item for item in comparisons if item and item.get("available")]
    counts = {label: sum(item["verdict"]["label"] == label for item in compared)
              for label in ("improved", "harmed", "no_meaningful_change")}
    total = len(compared)
    return {"compared": total, "counts": counts,
            "shares": {label: round(value / total, 4) if total else None for label, value in counts.items()},
            "judged": sum(item["verdict"]["basis"] == "judged" for item in compared),
            "delivered_steered": sum(item.get("delivered") == "steered" for item in compared),
            "shadow": sum(bool(item.get("shadow")) for item in compared)}
