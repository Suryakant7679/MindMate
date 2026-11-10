"""Connects the Emotion Reasoner Agent to the Llama neuron-steering runtime.

Per chat turn the service asks the agent for a plan (strategy, response
emotion, steering strength), hands the runtime the matching generation options,
and afterwards produces the turn's analysis for the Emotion Tracker:

* measured: panel-neuron activations from the model hooks, in the steered reply
  and in an unsteered reply to the same prompt;
* estimated: emotion scores of both replies from a text classifier.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any

from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, strategy_prompt
from chatbot_feature.emotion_reasoner.intervention import summarize_intervention

from .config import SYSTEM_PROMPT

SCORER_ID = "suryakantmani/xlm-roberta-large-goemotions-hard"
# Standard GoEmotions label order (the scorer's config only names LABEL_i).
GOEMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
              "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
              "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
              "remorse", "sadness", "surprise", "neutral"]
STUDY_LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness",
                "surprise")
HISTORY_TURNS = 3
BASELINE_MAX_NEW_TOKENS = 512
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


class EmotionSteeringService:
    def __init__(self, runtime: Any, agent: EmotionReasonerAgent | None = None,
                 scorer: ReplyEmotionScorer | None = None) -> None:
        self.runtime = runtime
        self.agent = agent or EmotionReasonerAgent(available_panels=list(runtime._emotion_panels))
        self.scorer = scorer or ReplyEmotionScorer(runtime)
        self._turns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def probe_scores(self, text: str) -> dict[str, float]:
        return self.runtime.detect_emotion(text)["scores"]

    def begin_turn(self, messages: list[dict[str, str]], *, alpha_override: float | None = None,
                   emotion_override: str | None = None) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Plan the reply to the last user message.

        Returns ``(turn_id, plan, generation_options)``.  The overrides exist
        for experiments through the API; the chat UI never sends them.
        """
        users = [message["content"] for message in messages if message["role"] == "user"]
        history = []
        for text in users[:-1][-HISTORY_TURNS:]:
            state = self.agent.understand(text, self.probe_scores(text))
            history.append({"emotion": state["emotion"], "valence": state["valence"]})
        plan = self.agent.plan(users[-1], self.probe_scores(users[-1]), history).to_dict()
        if emotion_override and emotion_override in self.runtime._emotion_panels and not plan["safety_flag"]:
            plan["response_emotion"] = emotion_override
            plan["strength_factors"].append({"factor": "manual", "effect": f"emotion {emotion_override}",
                                             "reason": "response emotion set by the API request"})
        if alpha_override is not None and not plan["safety_flag"] and plan["response_emotion"]:
            plan["steering_alpha"] = alpha_override
            plan["strength_factors"].append({"factor": "manual", "effect": f"alpha {alpha_override:.2f}",
                                             "reason": "steering strength set by the API request"})
        trace: dict[str, Any] = {}
        options = {"steering_alpha": plan["steering_alpha"], "steering_emotion": plan["response_emotion"],
                   "system_prompt": strategy_prompt(SYSTEM_PROMPT, plan), "steering_trace": trace}
        turn_id = uuid.uuid4().hex
        with self._lock:
            self._turns[turn_id] = {"plan": plan, "messages": list(messages), "options": options, "trace": trace,
                                    "reply": None, "analysis": None, "lock": threading.Lock()}
            while len(self._turns) > MAX_STORED_TURNS:
                self._turns.popitem(last=False)
        return turn_id, plan, options

    def complete_turn(self, turn_id: str, reply: str) -> None:
        with self._lock:
            turn = self._turns.get(turn_id)
        if turn is not None:
            turn["reply"] = reply

    def analysis(self, turn_id: str) -> dict[str, Any] | None:
        """Plan, measured intervention and before/after emotion estimate for a finished turn."""
        with self._lock:
            turn = self._turns.get(turn_id)
        if turn is None or not turn["reply"]:
            return None
        with turn["lock"]:
            if turn["analysis"] is None:
                turn["analysis"] = self._analyse(turn_id, turn)
        return turn["analysis"]

    def _analyse(self, turn_id: str, turn: dict[str, Any]) -> dict[str, Any]:
        plan, options, reply = turn["plan"], turn["options"], turn["reply"]
        steered = bool(options["steering_alpha"]) and bool(turn["trace"].get("layers"))
        after = self.scorer.score(reply)
        target = plan["response_emotion"]
        effect: dict[str, Any] = {"method": self.scorer.method, "target": target, "after": after, "before": None,
                                  "steered_reply": reply, "unsteered_reply": None}
        reference = None
        if steered:
            reference = {}
            baseline = self.runtime.generate(
                turn["messages"], max_new_tokens=BASELINE_MAX_NEW_TOKENS, deterministic=True, temperature=0.7,
                system_prompt=options["system_prompt"], steering_alpha=0.0, steering_emotion=target,
                steering_trace=reference)["response"]
            effect.update(before=self.scorer.score(baseline), unsteered_reply=baseline)
            effect.update(target_before=effect["before"][target], target_after=after[target])
        return {"turn_id": turn_id, "plan": plan, "steering": summarize_intervention(turn["trace"], reference),
                "effect": effect}
