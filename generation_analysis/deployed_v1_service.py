 """FROZEN COPY of backend/emotion_service.py as deployed on 2026-11-11 (before the causal
steering work).  Kept only so that ``heldout_comparison`` can reproduce the "before"
system exactly; do not edit and do not import from application code.  Pair it with
``steering_policy/policy_v1_deployed.json`` and ``steering_panels/panels.json``.

Original module docstring follows.

Connects the Emotion Reasoner Agent to the Llama neuron-steering runtime.

Per chat turn the service

1. detects the user's emotion (GoEmotions classifier; the Llama probe is only a
   fallback, it was right on 38% of test messages against 76%),
2. asks the agent for a plan: strategy, response emotion, calibrated strength,
3. generates the reply without steering, and, if the plan steers, a steered
   candidate, and sends the steered one only when it is verifiably better:
   steering is skipped when the unsteered reply already expresses the target
   emotion, and a steered candidate is rejected when it does not raise the
   target score or shows a quality warning sign.

The turn's analysis for the Mood Tracker separates what is measured (panel
neuron activations from the model hooks) from what is estimated (emotion
scores of the reply texts from a classifier).
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any

import time

from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, load_policy, strategy_prompt
from chatbot_feature.emotion_reasoner.agent import DETECTOR_WEIGHT, WEAK_DETECTOR_WEIGHT
from chatbot_feature.emotion_reasoner.intervention import summarize_intervention

from backend.config import SYSTEM_PROMPT

SCORER_ID = "suryakantmani/xlm-roberta-large-goemotions-hard"
# Standard GoEmotions label order (the scorer's config only names LABEL_i).
GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
              "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
              "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
              "remorse", "sadness", "surprise", "neutral"]
STUDY_LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness",
                "surprise")
HISTORY_TURNS = 3
DEFAULT_CEILING = 0.30          # unsteered target score at which steering is skipped
MIN_VERIFIED_GAIN = 0.05        # target-score gain a steered reply must show to be sent
MAX_REPETITION_INCREASE = 0.05  # repeated-bigram rate a steered reply may add
MIN_LENGTH_RATIO = 0.4          # a steered reply much shorter than the unsteered one has collapsed
MAX_STORED_TURNS = 200


class ReplyEmotionScorer:
    """Estimates the emotion expressed by a reply text.

    Uses the GoEmotions XLM-R classifier when it can be loaded and otherwise
    the frozen Llama probe.  Either way the numbers are classifier estimates,
    not measurements of the model's internals.
    """

    def __init__(self, runtime: Any, model_id: str = SCORER_ID) -> None:
        self.runtime, self.model_id = runtime, model_id
        self._lock = threading.Lock()
        self._loaded = False
        self._tokenizer = self._model = None
        self.method = "frozen Llama emotion probe (softmax over 10 labels)"

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            if model.config.num_labels != len(GOEMOTIONS):
                raise ValueError("unexpected label count")
            torch = self.runtime.torch
            self._model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self.method = f"{self.model_id} (independent sigmoid score per label)"
        except Exception as exc:   # fall back to the probe; the tracker names the method used
            print(f"Reply emotion classifier unavailable ({type(exc).__name__}); using the Llama probe.")
            self._model = None

    @property
    def uses_classifier(self) -> bool:
        with self._lock:
            self._load()
            return self._model is not None

    def score(self, text: str) -> dict[str, float]:
        with self._lock:
            self._load()
            if self._model is None:
                return self.runtime.detect_emotion(text or " ")["scores"]
            torch = self.runtime.torch
            encoded = self._tokenizer([text or " "], return_tensors="pt", truncation=True, max_length=256)
            encoded = encoded.to(self._model.device)
            with torch.inference_mode():
                values = torch.sigmoid(self._model(**encoded).logits.float())[0].cpu().tolist()
            return {label: round(values[GOEMOTIONS.index(label)], 4) for label in STUDY_LABELS}


def repeated_bigram_rate(text: str) -> float:
    words = text.lower().split()
    bigrams = list(zip(words, words[1:]))
    return 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0.0


class EmotionSteeringService:
    def __init__(self, runtime: Any, agent: EmotionReasonerAgent | None = None,
                 scorer: ReplyEmotionScorer | None = None) -> None:
        self.runtime = runtime
        self.scorer = scorer or ReplyEmotionScorer(runtime)
        self.policy = agent.policy if agent is not None else load_policy()
        if self.policy is None:
            print("No calibrated steering policy found; using heuristic strengths. "
                  "Run generation_analysis.steering_policy_experiment and calibrate_steering_policy.")
        self.agent = agent or EmotionReasonerAgent(
            available_panels=list(runtime._emotion_panels), policy=self.policy,
            detector_weight=DETECTOR_WEIGHT if self.scorer.uses_classifier else WEAK_DETECTOR_WEIGHT)
        self.detector = (f"{self.scorer.model_id} fused with knowledge-graph cues" if self.scorer.uses_classifier
                         else "frozen Llama probe fused with knowledge-graph cues")
        self._turns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def probe_scores(self, text: str) -> dict[str, float]:
        """Detector distribution over the ten emotion labels for a user message."""
        scores = self.scorer.score(text)
        total = sum(scores.values()) or 1.0
        return {label: value / total for label, value in scores.items()}

    def begin_turn(self, messages: list[dict[str, str]], *, alpha_override: float | None = None,
                   emotion_override: str | None = None) -> tuple[str, dict[str, Any]]:
        """Plan the reply to the last user message; returns ``(turn_id, plan)``.

        The overrides exist for experiments through the API (they also bypass
        the send-only-if-better check); the chat UI never sends them.
        """
        users = [message["content"] for message in messages if message["role"] == "user"]
        history = []
        for text in users[:-1][-HISTORY_TURNS:]:
            state = self.agent.understand(text, self.probe_scores(text))
            history.append({"emotion": state["emotion"], "valence": state["valence"]})
        plan = self.agent.plan(users[-1], self.probe_scores(users[-1]), history).to_dict()
        plan["detector"] = self.detector
        manual = False
        if emotion_override and emotion_override in self.runtime._emotion_panels and not plan["safety_flag"]:
            plan["response_emotion"], manual = emotion_override, True
            plan["strength_factors"].append({"factor": "manual", "effect": f"emotion {emotion_override}",
                                             "reason": "response emotion set by the API request"})
        if alpha_override is not None and not plan["safety_flag"] and plan["response_emotion"]:
            plan["steering_alpha"], manual = alpha_override, True
            plan["strength_factors"].append({"factor": "manual", "effect": f"alpha {alpha_override:.2f}",
                                             "reason": "steering strength set by the API request"})
        turn_id = uuid.uuid4().hex
        with self._lock:
            self._turns[turn_id] = {"plan": plan, "messages": list(messages), "manual": manual, "analysis": None,
                                    "system_prompt": strategy_prompt(SYSTEM_PROMPT, plan)}
            while len(self._turns) > MAX_STORED_TURNS:
                self._turns.popitem(last=False)
        return turn_id, plan

    def respond(self, turn_id: str, max_new_tokens: int) -> dict[str, Any]:
        """Generate the reply for a planned turn and decide whether steering is used.

        Returns the turn analysis; ``analysis["reply"]`` is the text to send.
        """
        turn = self._turns[turn_id]
        plan, target = turn["plan"], turn["plan"]["response_emotion"]
        planned = float(plan["steering_alpha"]) if target else 0.0
        started = time.perf_counter()

        def generate(alpha: float, trace: dict[str, Any] | None) -> str:
            return self.runtime.generate(
                turn["messages"], max_new_tokens=max_new_tokens, deterministic=True, temperature=0.7,
                system_prompt=turn["system_prompt"], steering_alpha=alpha, steering_emotion=target,
                steering_trace=trace)["response"]

        reference: dict[str, Any] | None = {} if planned else None
        unsteered = generate(0.0, reference)
        unsteered_scores = self.scorer.score(unsteered)
        effect: dict[str, Any] = {"method": self.scorer.method, "target": target, "unsteered_reply": unsteered,
                                  "unsteered_scores": unsteered_scores, "steered_reply": None,
                                  "steered_scores": None, "sent": "unsteered"}
        decision: dict[str, Any] = {"planned_alpha": planned, "applied_alpha": 0.0}
        trace: dict[str, Any] = {}
        ceiling = ((self.policy or {}).get("ceiling") or {}).get("threshold", DEFAULT_CEILING)
        if not planned:
            decision.update(status="not_planned", reason=plan["strength_factors"][0]["reason"])
        elif not turn["manual"] and unsteered_scores[target] >= ceiling:
            decision.update(status="skipped", reason=(
                f"the unsteered reply already expresses {target} (score {unsteered_scores[target]:.2f}, "
                f"threshold {ceiling:.2f}); steering such replies gained nothing on average in the calibration sweep"))
        else:
            steered = generate(planned, trace)
            steered_scores = self.scorer.score(steered)
            effect.update(steered_reply=steered, steered_scores=steered_scores)
            gain = steered_scores[target] - unsteered_scores[target]
            repetition = repeated_bigram_rate(steered) - repeated_bigram_rate(unsteered)
            if turn["manual"]:
                problem = None
            elif gain < MIN_VERIFIED_GAIN:
                problem = (f"it did not raise the {target} score enough to justify steering "
                           f"({unsteered_scores[target]:.2f} to {steered_scores[target]:.2f}, "
                           f"needs +{MIN_VERIFIED_GAIN:.2f})")
            elif repetition > MAX_REPETITION_INCREASE:
                problem = "it was more repetitive than the unsteered reply"
            elif len(steered) < MIN_LENGTH_RATIO * len(unsteered):
                problem = "it was much shorter than the unsteered reply"
            else:
                problem = None
            if problem:
                decision.update(status="rejected", reason=f"the steered reply was discarded because {problem}")
            else:
                effect["sent"] = "steered"
                decision.update(status="applied", applied_alpha=planned,
                                reason=f"the steered reply raised the {target} score by {gain:+.2f} "
                                       "and passed the repetition and length checks"
                                if not turn["manual"] else "steering forced by the API request")
        steering = summarize_intervention(trace, reference)
        steering["sent"] = effect["sent"] == "steered"
        analysis = {"turn_id": turn_id, "plan": plan, "decision": decision, "steering": steering, "effect": effect,
                    "reply": effect["steered_reply"] if effect["sent"] == "steered" else unsteered,
                    "latency_seconds": round(time.perf_counter() - started, 3)}
        turn["analysis"] = analysis
        return analysis

    def analysis(self, turn_id: str) -> dict[str, Any] | None:
        """Plan, decision, measured intervention and emotion estimates for a finished turn."""
        with self._lock:
            turn = self._turns.get(turn_id)
        return turn["analysis"] if turn else None
