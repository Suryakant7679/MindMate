"""Connects the Emotion Reasoner Agent to the Llama neuron-steering runtime.

One code path serves the direct chat endpoints and the LangGraph/MCP agent
mode (``respond_with_plan``), so both apply the same steering selection and
the same quality safeguards.  Per reply the service

1. takes the agent's plan: strategy, response emotion, calibrated strength
   (the user's emotion comes from the GoEmotions classifier fused with the
   knowledge graph; the Llama probe is only a fallback),
2. if no steering is planned, generates the reply once and streams it token by
   token; otherwise decodes the unsteered reply and the steered candidate
   together in one batched pass,
3. sends the steered candidate only if it passes the run-time safeguards named
   in the policy (``policy["gate"]``): never when the unsteered reply already
   expresses the target emotion, never when it is more repetitive or much
   shorter than the unsteered reply, and, if the policy asks for it, only when
   it raises the classifier's target score.

The turn analysis for the Mood Tracker separates what is measured (panel neuron
activations from the model hooks), what is estimated (classifier scores of the
reply texts), and what was established offline (causal panel experiments and
held-out effectiveness, ``steering_evidence``).
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any

import json
import re
import time
from pathlib import Path
from typing import Callable

from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, load_policy, strategy_prompt
from chatbot_feature.emotion_reasoner.agent import DETECTOR_WEIGHT, WEAK_DETECTOR_WEIGHT
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
DEFAULT_CEILING = 0.30          # unsteered target score at which steering is skipped
MIN_VERIFIED_GAIN = 0.05        # target-score gain a steered reply must show to be sent
MAX_REPETITION_INCREASE = 0.05  # repeated-bigram rate a steered reply may add
MIN_LENGTH_RATIO = 0.4          # a steered reply much shorter than the unsteered one has collapsed
MAX_STORED_TURNS = 200
# Run-time rule used when the policy does not name one (policies written before the gate comparison).
DEFAULT_GATE = {"rule": "classifier_gain", "ceiling": DEFAULT_CEILING, "min_gain": MIN_VERIFIED_GAIN,
                "max_repetition_increase": MAX_REPETITION_INCREASE, "min_length_ratio": MIN_LENGTH_RATIO}
EVIDENCE_PATH = (Path(__file__).resolve().parent.parent / "results" / "llama31_8b_generation_emotion"
                 / "causal_steering" / "evidence.json")


def load_evidence(path: str | Path = EVIDENCE_PATH) -> dict[str, Any]:
    """Offline causal and held-out results shown next to each turn (empty when not yet exported)."""
    try:
        evidence = json.loads(Path(path).read_text(encoding="utf-8"))
        return evidence if isinstance(evidence, dict) else {}
    except (OSError, ValueError):
        return {}


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
                 scorer: ReplyEmotionScorer | None = None, evidence: dict[str, Any] | None = None) -> None:
        self.runtime = runtime
        self.scorer = scorer or ReplyEmotionScorer(runtime)
        # Policy and evidence live in the loaded model's own result folder.
        results = (Path(__file__).resolve().parent.parent / "results"
                   / f"{getattr(runtime, 'results_tag', None) or 'llama31_8b'}_generation_emotion")
        self.policy = agent.policy if agent is not None else load_policy(results / "steering_policy" / "policy.json")
        if self.policy is None:
            print("No calibrated steering policy found; using heuristic strengths. "
                  "Run generation_analysis.recalibrate_policy.")
        self.agent = agent or EmotionReasonerAgent(
            available_panels=list(runtime._emotion_panels), policy=self.policy,
            detector_weight=DETECTOR_WEIGHT if self.scorer.uses_classifier else WEAK_DETECTOR_WEIGHT)
        self.detector = (f"{self.scorer.model_id} fused with knowledge-graph cues" if self.scorer.uses_classifier
                         else "frozen Llama probe fused with knowledge-graph cues")
        self.gate = {**DEFAULT_GATE, **((self.policy or {}).get("gate") or {})}
        legacy_ceiling = ((self.policy or {}).get("ceiling") or {}).get("threshold")
        if legacy_ceiling is not None and "gate" not in (self.policy or {}):
            self.gate["ceiling"] = legacy_ceiling
        self.evidence = load_evidence(results / "causal_steering" / "evidence.json") if evidence is None else evidence
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
        the run-time safeguards); the chat UI never sends them.
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

    def will_steer(self, turn_id: str) -> bool:
        """Whether a steered candidate will be generated (False: the reply can be streamed live)."""
        plan = self._turns[turn_id]["plan"]
        return bool(plan["response_emotion"] and plan["steering_alpha"])

    def respond(self, turn_id: str, max_new_tokens: int, *,
                on_decision: Callable[[dict[str, Any]], None] | None = None,
                on_text: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Generate the reply for a planned turn; ``analysis["reply"]`` is the text to send.

        ``on_decision`` is called once, before any reply text, with the steering
        decision; ``on_text`` receives the reply text as it becomes available
        (token by token when no steering is planned).
        """
        turn = self._turns[turn_id]
        analysis = self.respond_with_plan(turn["messages"], turn["system_prompt"], turn["plan"], max_new_tokens,
                                          manual=turn["manual"], on_decision=on_decision, on_text=on_text)
        analysis["turn_id"] = turn_id
        turn["analysis"] = analysis
        return analysis

    # ----------------------------------------------------------------- shared core

    def _check(self, target: str, unsteered: str, steered: str, unsteered_scores: dict[str, float],
               steered_scores: dict[str, float]) -> tuple[str, str]:
        """Run-time safeguards for a steered candidate: ``(status, reason)``."""
        gate = self.gate
        before, after = unsteered_scores[target], steered_scores[target]
        if before >= gate["ceiling"]:
            return "skipped", (f"the unsteered reply already expresses {target} (score {before:.2f}, threshold "
                               f"{gate['ceiling']:.2f}); steering such replies gained nothing in calibration")
        if repeated_bigram_rate(steered) - repeated_bigram_rate(unsteered) > gate["max_repetition_increase"]:
            return "rejected", "the steered reply was discarded because it was more repetitive than the unsteered reply"
        if len(steered) < gate["min_length_ratio"] * len(unsteered):
            return "rejected", "the steered reply was discarded because it was much shorter than the unsteered reply"
        if gate["rule"] == "classifier_gain" and after - before < gate["min_gain"]:
            return "rejected", (f"the steered reply was discarded because it did not raise the {target} score enough "
                                f"to justify steering ({before:.2f} to {after:.2f}, needs +{gate['min_gain']:.2f})")
        passed = ("raised the " + target + f" score by {after - before:+.2f} and passed"
                  if gate["rule"] == "classifier_gain" else "passed")
        return "applied", f"the steered reply {passed} the repetition and length checks"

    def respond_with_plan(self, messages: list[dict[str, str]], system_prompt: str | None, plan: dict[str, Any],
                          max_new_tokens: int, *, manual: bool = False,
                          on_decision: Callable[[dict[str, Any]], None] | None = None,
                          on_text: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Steering selection, generation and safeguards for one reply (direct chat and agent mode)."""
        target = plan["response_emotion"]
        planned = float(plan["steering_alpha"]) if target else 0.0
        started = time.perf_counter()
        timing: dict[str, float | None] = {"first_text": None}

        def emit(text: str) -> None:
            if timing["first_text"] is None:
                timing["first_text"] = time.perf_counter() - started
            if on_text and text:
                on_text(text)

        decision: dict[str, Any] = {"planned_alpha": planned, "applied_alpha": 0.0}
        effect: dict[str, Any] = {"method": self.scorer.method, "target": target, "steered_reply": None,
                                  "steered_scores": None, "sent": "unsteered"}
        trace: dict[str, Any] = {}
        reference: dict[str, Any] | None = None
        if not planned:
            decision.update(status="not_planned", reason=plan["strength_factors"][0]["reason"])
            if on_decision:
                on_decision(decision)
            if on_text and hasattr(self.runtime, "stream"):
                pieces = []
                for piece in self.runtime.stream(messages, max_new_tokens=max_new_tokens, deterministic=True,
                                                 temperature=0.7, system_prompt=system_prompt):
                    piece = piece if pieces else piece.lstrip()
                    pieces.append(piece)
                    emit(piece)
                unsteered = "".join(pieces).strip()
                if not unsteered:
                    raise RuntimeError("The model returned an empty response.")
            else:
                unsteered = self.runtime.generate(
                    messages, max_new_tokens=max_new_tokens, deterministic=True, temperature=0.7,
                    system_prompt=system_prompt)["response"]
                emit(unsteered)
            effect.update(unsteered_reply=unsteered, unsteered_scores=self.scorer.score(unsteered))
            reply = unsteered
        else:
            if hasattr(self.runtime, "generate_candidates"):
                plain, candidate = self.runtime.generate_candidates(
                    messages, max_new_tokens=max_new_tokens, system_prompt=system_prompt, steering_emotion=target,
                    alphas=[0.0, planned])
                unsteered, reference = plain["response"], plain["trace"]
                steered, trace = candidate["response"], candidate["trace"]
            else:                       # runtimes without batched candidates: two sequential generations
                def generate(alpha: float, hook_trace: dict[str, Any]) -> str:
                    return self.runtime.generate(
                        messages, max_new_tokens=max_new_tokens, deterministic=True, temperature=0.7,
                        system_prompt=system_prompt, steering_alpha=alpha, steering_emotion=target,
                        steering_trace=hook_trace)["response"]
                reference = {}
                unsteered, steered = generate(0.0, reference), generate(planned, trace)
            unsteered_scores, steered_scores = self.scorer.score(unsteered), self.scorer.score(steered)
            effect.update(unsteered_reply=unsteered, unsteered_scores=unsteered_scores, steered_reply=steered,
                          steered_scores=steered_scores)
            if manual:
                status, reason = "applied", "steering forced by the API request"
            else:
                status, reason = self._check(target, unsteered, steered, unsteered_scores, steered_scores)
            decision.update(status=status, reason=reason)
            if status == "applied":
                decision["applied_alpha"] = planned
                effect["sent"] = "steered"
            reply = steered if status == "applied" else unsteered
            if on_decision:
                on_decision(decision)
            for piece in re.findall(r"\S+\s*|\s+", reply):
                emit(piece)
        steering = summarize_intervention(trace, reference)
        steering["sent"] = effect["sent"] == "steered"
        return {"plan": plan, "decision": decision, "steering": steering, "effect": effect, "reply": reply,
                "evidence": self.evidence_for(plan),
                "first_token_seconds": round(timing["first_text"], 3) if timing["first_text"] is not None else None,
                "latency_seconds": round(time.perf_counter() - started, 3)}

    def evidence_for(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        """Offline evidence for the planned response emotion: causal panel results and held-out effect."""
        target = plan.get("response_emotion")
        panels = self.evidence.get("panels") or {}
        if not target or target not in panels:
            return None
        return {"panel": panels[target], "held_out": self.evidence.get("held_out"),
                "sources": self.evidence.get("sources")}

    def analysis(self, turn_id: str) -> dict[str, Any] | None:
        """Plan, decision, measured intervention and emotion estimates for a finished turn."""
        with self._lock:
            turn = self._turns.get(turn_id)
        return turn["analysis"] if turn else None
