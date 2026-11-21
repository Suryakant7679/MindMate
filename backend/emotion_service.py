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

Research instrumentation (all optional, none of it changes which reply is sent):

* every finished turn is recorded with its conversation (``ResearchStore``) and the
  conversation's emotion trajectory is attached to the analysis;
* ``research_compare`` generates, for turns the policy does not steer, a *shadow*
  steered candidate in the same decoding batch as the reply.  It is compared with
  the unsteered reply and never sent; it is not generated on safety turns;
* API strength / emotion overrides are ignored unless the server was started with
  them enabled, and even then the candidate must pass the repetition, length and
  language safeguards and is never generated on a safety turn.
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

from chatbot_feature.emotion_reasoner.knowledge_graph import is_hinglish
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, load_policy, strategy_prompt
from chatbot_feature.emotion_reasoner.agent import DETECTOR_WEIGHT, WEAK_DETECTOR_WEIGHT
from chatbot_feature.emotion_reasoner.comparison import compare_replies, tally, with_judged
from chatbot_feature.emotion_reasoner.intervention import summarize_intervention
from chatbot_feature.emotion_reasoner.tone import degenerate_reply
from chatbot_feature.emotion_reasoner.trajectory import after_response, reply_state, summarize, turn_state

from .config import SYSTEM_PROMPT
from .research import ResearchStore, load_cards, load_json

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
# Diagnostic strength of the shadow candidate in research-compare mode: a strength each model's own
# calibration found quality-safe in at least one context (Llama and EmoLLaMA 0.5; Sarvam tone panel 2).
SHADOW_ALPHA = {"llama31_8b": 0.5, "llama2_13b_chat": 0.5, "emollama_chat_7b": 0.5, "sarvam_30b": 2.0}
MAX_CACHED_TEXTS = 2000
SEPARATION = (
    "Chatbot emotion expression (what the reply text expresses) is estimated by a classifier and, offline, by two judge models.",
    "User emotion is what the detector estimated from the user's own messages; it is not a calibrated mood measurement.",
    "A change in the user's detected emotion after a reply is an observation, not an effect of that reply or of steering.",
    "Neural measurements (activations, shifts) show what the intervention did inside the model, not whether the reply is better.",
)


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
                 scorer: ReplyEmotionScorer | None = None, evidence: dict[str, Any] | None = None, *,
                 research: ResearchStore | None = None, allow_overrides: bool = False,
                 research_compare: bool = False) -> None:
        self.runtime = runtime
        self.scorer = scorer or ReplyEmotionScorer(runtime)
        # Policy and evidence live in the loaded model's own result folder.
        self.tag = getattr(runtime, "results_tag", None) or "llama31_8b"
        results = Path(__file__).resolve().parent.parent / "results" / f"{self.tag}_generation_emotion"
        self.results = results
        self.research = research or ResearchStore(self.tag, persist=False)
        # Experimental strength / emotion overrides through the API: off unless the server enables them.
        self.allow_overrides = allow_overrides
        # Research mode: also generate a (never sent) steered candidate on turns the policy does not steer.
        self.research_compare = research_compare
        self.shadow_alpha = float(SHADOW_ALPHA.get(self.tag, 0.5))
        # Offline per-neuron evidence and validation report (written by generation_analysis.neuron_validation).
        self.cards = load_cards(results / "neuron_validation" / "cards.json") if agent is None else {}
        self._states: OrderedDict[str, tuple[dict[str, Any], dict[str, Any]]] = OrderedDict()
        self._probes: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._replies: OrderedDict[str, dict[str, float]] = OrderedDict()
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

    @staticmethod
    def _cached(cache: OrderedDict, key: str, compute: Callable[[], Any]) -> Any:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        value = cache[key] = compute()
        while len(cache) > MAX_CACHED_TEXTS:
            cache.popitem(last=False)
        return value

    def probe_scores(self, text: str) -> dict[str, float]:
        """Detector distribution over the ten emotion labels for a user message."""
        def compute() -> dict[str, float]:
            scores = self.scorer.score(text)
            total = sum(scores.values()) or 1.0
            return {label: value / total for label, value in scores.items()}
        return dict(self._cached(self._probes, text, compute))

    def user_state(self, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Trajectory record of one user message and the agent's understanding of it."""
        def compute() -> tuple[dict[str, Any], dict[str, Any]]:
            probe = self.probe_scores(text)
            understood = self.agent.understand(text, probe)
            return turn_state(text, understood, probe), understood
        return self._cached(self._states, text, compute)

    def reply_scores(self, text: str) -> dict[str, float]:
        return dict(self._cached(self._replies, text, lambda: self.scorer.score(text)))

    def begin_turn(self, messages: list[dict[str, str]], *, alpha_override: float | None = None,
                   emotion_override: str | None = None, conversation_id: str | None = None,
                   mode: str = "direct") -> tuple[str, dict[str, Any]]:
        """Plan the reply to the last user message; returns ``(turn_id, plan)``.

        The overrides exist for experiments through the API.  They are ignored unless the service was
        created with ``allow_overrides``, are never applied on a safety turn or for a model whose policy
        names its own panels, and the forced candidate still has to pass the repetition, length and
        language safeguards.  The chat UI never sends them.
        """
        ignored = None
        if alpha_override is not None or emotion_override:
            if self.agent._policy_names_panels():
                # Panel and strength of this model come from its calibrated policy only; no manual control.
                ignored = "this model's panel and strength come from its calibrated policy only"
            elif not self.allow_overrides:
                ignored = "experimental overrides are disabled on this server"
        if ignored:
            alpha_override = emotion_override = None
        users = [message["content"] for message in messages if message["role"] == "user"]
        history = []
        for text in users[:-1][-HISTORY_TURNS:]:
            state = self.user_state(text)[1]
            history.append({"emotion": state["emotion"], "valence": state["valence"]})
        plan = self.agent.plan(users[-1], self.probe_scores(users[-1]), history).to_dict()
        plan["detector"] = self.detector
        manual = False
        if ignored:
            plan["strength_factors"].append({"factor": "manual", "effect": "ignored",
                                             "reason": f"the strength requested through the API was ignored: {ignored}"})
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
                                    "system_prompt": strategy_prompt(SYSTEM_PROMPT, plan),
                                    "conversation_id": conversation_id, "mode": mode}
            while len(self._turns) > MAX_STORED_TURNS:
                self._turns.popitem(last=False)
        return turn_id, plan

    def will_steer(self, turn_id: str) -> bool:
        """Whether a steered candidate will be generated (False: the reply can be streamed live)."""
        plan = self._turns[turn_id]["plan"]
        return bool(plan["response_emotion"] and plan["steering_alpha"])

    def respond(self, turn_id: str, max_new_tokens: int, *,
                safety_filter: Callable[[str], str] | None = None,
                on_decision: Callable[[dict[str, Any]], None] | None = None,
                on_text: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Generate the reply for a planned turn; ``analysis["reply"]`` is the text to send.

        ``on_decision`` is called once, before any reply text, with the steering
        decision; ``on_text`` receives the reply text as it becomes available
        (token by token when no steering is planned).
        """
        turn = self._turns[turn_id]
        analysis = self.respond_with_plan(turn["messages"], turn["system_prompt"], turn["plan"], max_new_tokens,
                                          manual=turn["manual"], on_decision=None if safety_filter else on_decision,
                                          on_text=None if safety_filter else on_text,
                                          turn_id=turn_id)
        if safety_filter is not None:
            approved = safety_filter(analysis["reply"])
            if approved != analysis["reply"]:
                analysis["reply"] = approved
                analysis["effect"] = {"sent": "safety_replacement", "method": "safety", "target": None,
                                      "unsteered_reply": approved, "unsteered_scores": {}}
                analysis["comparison"] = None
                analysis["decision"].update(applied_alpha=0.0, status="safety_replacement")
            for candidate in ("unsteered_reply", "steered_reply"):
                text = analysis["effect"].get(candidate)
                if text and text != approved and safety_filter(text) != text:
                    analysis["effect"][candidate] = "[Safety-blocked research draft]"
                    analysis["comparison"] = None
            if on_decision:
                on_decision(analysis["decision"])
            if on_text:
                on_text(approved)
        analysis["turn_id"] = turn_id
        self.finalize_turn(turn["messages"], turn["plan"], analysis, conversation_id=turn.get("conversation_id"),
                           turn_id=turn_id, mode=turn.get("mode", "direct"))
        turn["analysis"] = analysis
        return analysis

    # ----------------------------------------------------------------- shared core

    def _check(self, target: str, unsteered: str, steered: str, unsteered_scores: dict[str, float],
               steered_scores: dict[str, float], panel: str | None = None, manual: bool = False) -> tuple[str, str]:
        """Run-time safeguards for a steered candidate: ``(status, reason)``.

        ``manual`` (an enabled API override) skips only the rules that decide *whether steering is worth it*
        (ceiling, classifier gain, language repair).  The rules that protect the reply itself always apply.
        """
        gate = self.gate
        before, after = unsteered_scores[target], steered_scores[target]
        if manual:
            if degenerate_reply(steered) and not degenerate_reply(unsteered):
                return "rejected", "the forced steered reply was discarded because it was not a conversational answer"
            if repeated_bigram_rate(steered) - repeated_bigram_rate(unsteered) > gate["max_repetition_increase"]:
                return "rejected", "the forced steered reply was discarded because it was more repetitive than the unsteered reply"
            if len(steered) < gate["min_length_ratio"] * len(unsteered):
                return "rejected", "the forced steered reply was discarded because it was much shorter than the unsteered reply"
            if is_hinglish(unsteered) and not is_hinglish(steered):
                return "rejected", "the forced steered reply was discarded because it left the user's Hinglish for English"
            return "applied", "steering forced by the API request; the reply passed the repetition, length and language checks"
        # A pure language panel (named by the policy's gate) repairs the reply language only: it is used
        # when the unsteered reply left the user's Hinglish, and its candidate must be in Hinglish.
        language_repair = panel is not None and panel == gate.get("language_panel")
        if language_repair and is_hinglish(unsteered):
            return "skipped", "the unsteered reply is already in Hinglish, so the language panel is not needed"
        if language_repair and not is_hinglish(steered):
            return "rejected", "the steered reply was discarded because it was still not in Hinglish"
        if not language_repair and before >= gate["ceiling"]:
            return "skipped", (f"the unsteered reply already expresses {target} (score {before:.2f}, threshold "
                               f"{gate['ceiling']:.2f}); steering such replies gained nothing in calibration")
        if repeated_bigram_rate(steered) - repeated_bigram_rate(unsteered) > gate["max_repetition_increase"]:
            return "rejected", "the steered reply was discarded because it was more repetitive than the unsteered reply"
        if len(steered) < gate["min_length_ratio"] * len(unsteered):
            return "rejected", "the steered reply was discarded because it was much shorter than the unsteered reply"
        # Optional safeguards a model's policy can switch on (off unless its calibration supports them).
        if gate.get("max_length_ratio") and len(steered.split()) > gate["max_length_ratio"] * max(len(unsteered.split()), 1):
            return "rejected", "the steered reply was discarded because it was much longer than the unsteered reply"
        if gate.get("keep_hinglish") and is_hinglish(unsteered) and not is_hinglish(steered):
            return "rejected", "the steered reply was discarded because it left the user's Hinglish for English"
        if gate["rule"] == "classifier_gain" and after - before < gate["min_gain"]:
            return "rejected", (f"the steered reply was discarded because it did not raise the {target} score enough "
                                f"to justify steering ({before:.2f} to {after:.2f}, needs +{gate['min_gain']:.2f})")
        passed = ("raised the " + target + f" score by {after - before:+.2f} and passed"
                  if gate["rule"] == "classifier_gain" else "passed")
        return "applied", f"the steered reply {passed} the repetition and length checks"

    def _shadow(self, plan: dict[str, Any]) -> tuple[str, float] | None:
        """Panel and strength of the comparison-only candidate for a turn the policy does not steer."""
        if plan.get("safety_flag") or plan.get("strategy") == "crisis_support" or not plan.get("response_emotion"):
            return None
        panels = getattr(self.runtime, "_emotion_panels", {}) or {}
        directions = getattr(self.runtime, "_emotion_directions", {}) or {}
        for name in ((plan.get("calibration") or {}).get("panel"),
                     (getattr(self.runtime, "strategy_panels", {}) or {}).get(plan.get("strategy")),
                     plan["response_emotion"]):
            if name and (panels.get(name) or name in directions):
                return name, self.shadow_alpha
        return None

    @staticmethod
    def _disabled_by(plan: dict[str, Any]) -> str:
        if plan.get("safety_flag"):
            return "safety"
        factor = (plan.get("strength_factors") or [{}])[0].get("factor")
        if factor == "panel" or not plan.get("response_emotion"):
            return "no_panel"
        return "policy" if plan.get("calibration") else "strength_rule"

    def respond_with_plan(self, messages: list[dict[str, str]], system_prompt: str | None, plan: dict[str, Any],
                          max_new_tokens: int, *, manual: bool = False,
                          on_decision: Callable[[dict[str, Any]], None] | None = None,
                          on_text: Callable[[str], None] | None = None, turn_id: str | None = None,
                          user_text: str | None = None) -> dict[str, Any]:
        """Steering selection, generation and safeguards for one reply (direct chat and agent mode)."""
        target = plan["response_emotion"]
        planned = float(plan["steering_alpha"]) if target else 0.0
        turn_id = turn_id or uuid.uuid4().hex
        user_text = user_text or plan.get("user_text") or next(
            (message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        started = time.perf_counter()
        timing: dict[str, float | None] = {"first_text": None}

        def emit(text: str) -> None:
            if timing["first_text"] is None:
                timing["first_text"] = time.perf_counter() - started
            if on_text and text:
                on_text(text)

        def candidates(panel: str, alpha: float) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
            """Unsteered reply and steered candidate to the same prompt (one batch where the runtime can)."""
            if hasattr(self.runtime, "generate_candidates"):
                plain, candidate = self.runtime.generate_candidates(
                    messages, max_new_tokens=max_new_tokens, system_prompt=system_prompt, steering_emotion=panel,
                    alphas=[0.0, alpha])
                return plain["response"], plain["trace"], candidate["response"], candidate["trace"]

            def generate(strength: float, hook_trace: dict[str, Any]) -> str:   # two sequential generations
                return self.runtime.generate(
                    messages, max_new_tokens=max_new_tokens, deterministic=True, temperature=0.7,
                    system_prompt=system_prompt, steering_alpha=strength, steering_emotion=panel,
                    steering_trace=hook_trace)["response"]
            base_trace: dict[str, Any] = {}
            steered_trace: dict[str, Any] = {}
            return generate(0.0, base_trace), base_trace, generate(alpha, steered_trace), steered_trace

        decision: dict[str, Any] = {"planned_alpha": planned, "applied_alpha": 0.0}
        effect: dict[str, Any] = {"method": self.scorer.method, "target": target, "steered_reply": None,
                                  "steered_scores": None, "sent": "unsteered", "shadow": False}
        trace: dict[str, Any] = {}
        reference: dict[str, Any] | None = None
        panel_name = plan.get("steering_panel") or target
        shadow = self._shadow(plan) if (not planned and self.research_compare and not manual) else None
        if not planned:
            decision.update(status="not_planned", reason=plan["strength_factors"][0]["reason"],
                            disabled_by=self._disabled_by(plan))
            if on_decision:
                on_decision(decision)
        if not planned and shadow:
            # Research mode: the reply is the unsteered one; the steered candidate is only compared.
            panel_name, shadow_alpha = shadow
            unsteered, reference, steered, trace = candidates(panel_name, shadow_alpha)
            decision.update(shadow_alpha=shadow_alpha, shadow_panel=panel_name)
            effect.update(unsteered_reply=unsteered, unsteered_scores=self.scorer.score(unsteered), steered_reply=steered,
                          steered_scores=self.scorer.score(steered), shadow=True)
            reply = unsteered
            for piece in re.findall(r"\S+\s*|\s+", reply):
                emit(piece)
        elif not planned:
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
            unsteered, reference, steered, trace = candidates(panel_name, planned)
            unsteered_scores, steered_scores = self.scorer.score(unsteered), self.scorer.score(steered)
            effect.update(unsteered_reply=unsteered, unsteered_scores=unsteered_scores, steered_reply=steered,
                          steered_scores=steered_scores)
            status, reason = self._check(target, unsteered, steered, unsteered_scores, steered_scores,
                                         panel=plan.get("steering_panel"), manual=manual)
            decision.update(status=status, reason=reason, **({"manual": True} if manual else {}))
            if status == "applied":
                decision["applied_alpha"] = planned
                effect["sent"] = "steered"
            reply = steered if status == "applied" else unsteered
            if on_decision:
                on_decision(decision)
            for piece in re.findall(r"\S+\s*|\s+", reply):
                emit(piece)
        steering = summarize_intervention(trace, reference, cards=self.cards.get(panel_name or "", {}), detail=True)
        records = steering.pop("neurons", None)
        steering["sent"] = effect["sent"] == "steered"
        steering["shadow"] = effect["shadow"]
        comparison = compare_replies(user_text, effect["unsteered_reply"], effect["steered_reply"],
                                     effect["unsteered_scores"], effect["steered_scores"], target,
                                     delivered=effect["sent"], shadow=effect["shadow"])
        if records:
            self.research.save_neurons(turn_id, {
                "schema": 1, "turn_id": turn_id, "model": getattr(self.runtime, "model_id", None), "panel": panel_name,
                "target": target, "alpha": steering["alpha"], "gain": steering.get("gain"),
                "location": steering.get("location"), "delivered": steering["sent"], "shadow": steering["shadow"],
                "decode_steps": steering.get("decode_steps"), "contribution_basis": steering.get("contribution_basis"),
                "layers": [{key: value for key, value in layer.items() if key != "neuron_ids"} for layer in steering["layers"]],
                "neurons": records})
        return {"plan": plan, "decision": decision, "steering": steering, "effect": effect, "reply": reply,
                "comparison": comparison, "turn_id": turn_id,
                "evidence": self.evidence_for(plan),
                "first_token_seconds": round(timing["first_text"], 3) if timing["first_text"] is not None else None,
                "latency_seconds": round(time.perf_counter() - started, 3)}

    def evidence_for(self, plan: dict[str, Any]) -> dict[str, Any] | None:
        """Offline evidence for the planned response emotion: causal panel results and held-out effect."""
        panels = self.evidence.get("panels") or {}
        strategy_panel = (getattr(self.runtime, "strategy_panels", {}) or {}).get(plan.get("strategy"))
        target = next((name for name in (plan.get("steering_panel"), strategy_panel, plan.get("response_emotion"))
                       if name in panels), None)
        if not target:
            return None
        return {"panel": panels[target], "held_out": self.evidence.get("held_out"),
                "sources": self.evidence.get("sources")}

    def analysis(self, turn_id: str) -> dict[str, Any] | None:
        """Plan, decision, measured intervention and emotion estimates for a finished turn."""
        with self._lock:
            turn = self._turns.get(turn_id)
        return turn["analysis"] if turn else None

    def trace(self, turn_id: str) -> dict[str, Any] | None:
        """The turn analysis as the Mood Tracker reads it: with offline judge results for this turn, if
        they exist by now, and the conversation's current trajectory (later ratings included)."""
        analysis = self.analysis(turn_id)
        if analysis is None:
            return None
        view = dict(analysis)
        view["comparison"] = with_judged(analysis.get("comparison") or {}, self.research.judged(turn_id))
        if analysis.get("conversation_id"):
            current = self.conversation_view(analysis["conversation_id"])
            if len(current["trajectory"]["turns"]) > analysis.get("turn_index", 0):
                view.update(trajectory=current["trajectory"], summary=current["summary"])
        return view

    # ------------------------------------------------------- conversation research

    def _record(self, user_text: str, plan: dict[str, Any] | None, analysis: dict[str, Any], turn_id: str,
                mode: str) -> dict[str, Any]:
        effect, steering = analysis["effect"], analysis.get("steering") or {}
        calibration = (plan or {}).get("calibration") or {}
        return {
            "turn_id": turn_id, "mode": mode, "model": getattr(self.runtime, "model_id", None), "model_tag": self.tag,
            "user": self.user_state(user_text)[0],
            "plan": ({**{key: plan.get(key) for key in ("strategy", "strategy_title", "response_emotion", "steering_panel",
                                                         "steering_alpha", "safety_flag")},
                      "calibration_context": calibration.get("context"), "calibration_reason": calibration.get("reason")}
                     if plan else None),
            "decision": analysis["decision"],
            "replies": {"text": analysis["reply"], "unsteered": effect["unsteered_reply"],
                        "steered": effect.get("steered_reply"), "delivered": effect["sent"],
                        "shadow": bool(effect.get("shadow"))},
            "scores": {"method": effect["method"], "target": effect["target"], "unsteered": effect["unsteered_scores"],
                       "steered": effect.get("steered_scores")},
            "comparison": analysis.get("comparison"),
            "steering": ({**{key: value for key, value in steering.items() if key not in ("top_neurons", "heatmap", "layers")},
                          "layers": [{key: value for key, value in layer.items() if key != "neuron_ids"}
                                     for layer in steering.get("layers", [])]} if steering.get("applied") else {"applied": False}),
            "latency_seconds": analysis.get("latency_seconds"), "first_token_seconds": analysis.get("first_token_seconds"),
        }

    def _reconstructed(self, user_text: str, reply: str | None) -> dict[str, Any]:
        """An earlier turn this store did not see (API use without a conversation id, or a restarted server)."""
        return {"turn_id": None, "mode": "reconstructed", "model": None, "model_tag": self.tag,
                "user": self.user_state(user_text)[0], "plan": None, "decision": None,
                "replies": {"text": reply, "unsteered": None, "steered": None, "delivered": None, "shadow": False},
                "scores": {"method": self.scorer.method, "target": None,
                           "unsteered": self.reply_scores(reply) if reply else None, "steered": None},
                "comparison": None, "steering": {"applied": False}}

    def finalize_turn(self, messages: list[dict[str, str]], plan: dict[str, Any] | None, analysis: dict[str, Any], *,
                      conversation_id: str | None = None, turn_id: str | None = None,
                      mode: str = "direct") -> dict[str, Any]:
        """Record a finished turn with its conversation and attach the conversation's trajectory and summary."""
        users = [message["content"] for message in messages if message["role"] == "user"]
        assistants = [message["content"] for message in messages if message["role"] == "assistant"]
        index = len(users) - 1
        turn_id = turn_id or analysis.get("turn_id") or uuid.uuid4().hex
        conversation = conversation_id or f"turn-{turn_id}"
        for earlier in range(index):
            if not self.research.has_turn(conversation, earlier, users[earlier]):
                self.research.record(conversation, earlier, self._reconstructed(
                    users[earlier], assistants[earlier] if earlier < len(assistants) else None), persist=False)
        self.research.record(conversation, index, self._record(users[-1], plan, analysis, turn_id, mode))
        view = self.conversation_view(conversation)
        analysis.update(turn_id=turn_id, conversation_id=conversation, turn_index=index,
                        trajectory=view["trajectory"], summary=view["summary"])
        return analysis

    def track_agent_turn(self, messages: list[dict[str, str]], result: dict[str, Any],
                         conversation_id: str) -> dict[str, Any]:
        """Record a LangGraph agent reply with its conversation, as ``respond`` does for direct chat.

        ``result`` is the agent service's answer.  A conversational reply carries the shared steering
        analysis (``result["steering"]``); an answer built from tool results, or one without an emotion
        plan, was generated unsteered and is recorded as such.
        """
        plan, steering, reply = result.get("emotion_plan"), result.get("steering") or {}, result["response"]
        if steering.get("effect"):
            extra = ("decision", "effect", "evidence", "comparison", "turn_id", "run_id")
            analysis = {"reply": reply, "decision": steering["decision"], "effect": steering["effect"],
                        "steering": {key: value for key, value in steering.items() if key not in extra},
                        "comparison": steering.get("comparison"), "turn_id": steering.get("turn_id")}
        else:
            target = (plan or {}).get("response_emotion")
            scores = self.reply_scores(reply)
            user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            analysis = {"reply": reply, "steering": {"applied": False},
                        "decision": {"status": "not_planned", "planned_alpha": 0.0, "applied_alpha": 0.0, "disabled_by": "agent",
                                     "reason": "the agent answered from tool results or without an emotion plan; "
                                               "such answers are never steered"},
                        "effect": {"method": self.scorer.method, "target": target, "unsteered_reply": reply,
                                   "unsteered_scores": scores, "steered_reply": None, "steered_scores": None,
                                   "sent": "unsteered", "shadow": False},
                        "comparison": compare_replies(user_text, reply, None, scores, None, target, delivered="unsteered")}
        self.finalize_turn(messages, plan, analysis, conversation_id=conversation_id, mode="agent")
        return analysis

    def conversation_view(self, conversation_id: str) -> dict[str, Any]:
        """Trajectory, steering history and research summary of a recorded conversation."""
        turns = self.research.turns(conversation_id)
        mood, helpful = self.research.ratings(conversation_id)
        states = [turn["user"] for turn in turns]
        trajectory = summarize(states, mood)
        responses, comparisons = [], []
        for index, turn in enumerate(turns):
            replies, scores = turn["replies"], turn["scores"]
            delivered_scores = scores.get("steered") if replies.get("delivered") == "steered" else scores.get("unsteered")
            text = replies.get("text")
            plan, decision = turn.get("plan") or {}, turn.get("decision") or {}
            info = reply_state(text, delivered_scores, plan.get("response_emotion")) if text and delivered_scores else {}
            comparison = turn.get("comparison")
            if comparison and turn.get("turn_id"):
                comparison = with_judged(comparison, self.research.judged(turn["turn_id"]))
            comparisons.append(comparison)
            automatic = (comparison or {}).get("automatic") or {}
            responses.append({
                "turn": index + 1, "turn_id": turn.get("turn_id"), "mode": turn.get("mode"), "text": text, **info,
                "delivered": replies.get("delivered"), "status": decision.get("status"),
                "reason": decision.get("reason"), "disabled_by": decision.get("disabled_by"),
                "planned_alpha": decision.get("planned_alpha"), "applied_alpha": decision.get("applied_alpha"),
                "shadow_alpha": decision.get("shadow_alpha"), "strategy": plan.get("strategy"),
                "strategy_title": plan.get("strategy_title"),
                "panel": plan.get("steering_panel") or decision.get("shadow_panel"),
                "compared": bool(comparison and comparison.get("available")), "shadow": bool(replies.get("shadow")),
                "verdict": (comparison or {}).get("verdict", {}).get("label") if comparison else None,
                "verdict_basis": (comparison or {}).get("verdict", {}).get("basis") if comparison else None,
                "target_gain": automatic.get("target_gain"),
                "unsteered_target_score": ((comparison or {}).get("unsteered") or {}).get("target_score"),
                "steered_target_score": ((comparison or {}).get("steered") or {}).get("target_score"),
                "helpfulness": helpful.get(index), "user_language": turn["user"]["language"],
            })
        trajectory["responses"] = responses
        trajectory["after_response"] = after_response(states, responses)
        return {"conversation_id": conversation_id, "model": getattr(self.runtime, "model_id", None), "model_tag": self.tag,
                "trajectory": trajectory, "summary": self._summary(turns, responses, comparisons, trajectory, helpful)}

    def _summary(self, turns: list[dict[str, Any]], responses: list[dict[str, Any]],
                 comparisons: list[dict[str, Any] | None], trajectory: dict[str, Any],
                 helpful: dict[int, int]) -> dict[str, Any]:
        tracked = [row for row in responses if row["status"]]
        statuses = {name: sum(row["status"] == name for row in tracked)
                    for name in ("applied", "rejected", "skipped", "not_planned")}
        gains = [row["target_gain"] for row in responses if row["compared"] and row["target_gain"] is not None]
        delivered = [row["target_gain"] for row in responses
                     if row["delivered"] == "steered" and row["target_gain"] is not None]
        languages: dict[str, dict[str, int]] = {}
        for row in responses:
            entry = languages.setdefault(row["user_language"], {"turns": 0, "steered_delivered": 0, "compared": 0,
                                                                "improved": 0, "harmed": 0, "no_meaningful_change": 0})
            entry["turns"] += 1
            entry["steered_delivered"] += row["delivered"] == "steered"
            entry["compared"] += row["compared"]
            if row["compared"] and row["verdict"] in entry:
                entry[row["verdict"]] += 1
        reply_valences = [row["expected_valence"] for row in responses if row.get("expected_valence") is not None]
        target_scores = [row["target_score"] for row in responses if row.get("target_score") is not None]
        ratings = list(helpful.values())
        mean = lambda values: round(sum(values) / len(values), 4) if values else None   # noqa: E731
        return {
            "turns": len(turns), "tracked_turns": len(tracked), "decisions": statuses,
            "steered_delivered": statuses["applied"],
            "steered_delivered_share": round(statuses["applied"] / len(tracked), 4) if tracked else None,
            "comparison": tally(comparisons),
            "mean_target_gain_compared": mean(gains), "mean_target_gain_delivered": mean(delivered),
            "by_language": languages,
            "user": {"dominant_emotion": trajectory["dominant_emotion"],
                     "mean_expected_valence": trajectory["mood"]["mean_expected_valence"],
                     "trend_per_turn": trajectory["mood"]["trend_per_turn"],
                     "trend_interpretable": trajectory["mood"]["trend_interpretable"],
                     "mean_distribution": trajectory["mean_distribution"],
                     "uncertain_share": trajectory["mood"]["uncertain_share"],
                     "self_reports": trajectory["self_report"]["n"]},
            "response": {"mean_expected_valence": mean(reply_valences), "mean_target_score": mean(target_scores)},
            "helpfulness": {"n": len(ratings), "mean": mean(ratings)},
            "research_compare": self.research_compare,
            "separation": list(SEPARATION),
        }

    def add_feedback(self, conversation_id: str, turn_index: int, *, mood_rating: int | None = None,
                     helpfulness: int | None = None) -> dict[str, Any]:
        self.research.feedback(conversation_id, turn_index, mood_rating=mood_rating, helpfulness=helpfulness)
        return self.conversation_view(conversation_id)

    def neuron_record(self, turn_id: str) -> dict[str, Any] | None:
        return self.research.neurons(turn_id)

    def architecture(self) -> dict[str, Any]:
        """Facts about the loaded model that locate its steering sites (read from the model object, no inference)."""
        model = getattr(self.runtime, "model", None)
        config = getattr(model, "config", None)
        info: dict[str, Any] = {"model": getattr(self.runtime, "model_id", None), "class": type(model).__name__ if model is not None else None}
        try:
            layers = model.model.layers
            widths = [int(self.runtime.steering_module(index).in_features) for index in range(len(layers))]
            info.update(layers=len(layers), hidden_size=getattr(config, "hidden_size", None), site_widths=widths,
                        steerable_neurons=sum(widths),
                        mixture_of_experts=any(hasattr(layer.mlp, "shared_experts") for layer in layers),
                        quantized=bool(getattr(model, "is_loaded_in_4bit", False)))
        except Exception:                      # noqa: BLE001 - description only
            pass
        return info

    def panel_view(self, name: str) -> dict[str, Any] | None:
        """Every neuron of one steering panel with its stored shift and offline evidence (no inference)."""
        panel = (getattr(self.runtime, "_emotion_panels", {}) or {}).get(name)
        if not panel:
            return None
        cards = self.cards.get(name, {})
        neurons = [{"layer": layer, "neuron": neuron, "delta": round(float(delta), 6), **cards.get((layer, neuron), {})}
                   for layer, (ids, deltas) in sorted(panel.items()) for neuron, delta in zip(ids, deltas)]
        offline = (self.evidence.get("panels") or {}).get(name) or {}
        return {"panel": name, "model": getattr(self.runtime, "model_id", None), "neurons_total": len(neurons),
                "layers": sorted(panel), "per_layer": {str(layer): len(ids) for layer, (ids, _) in sorted(panel.items())},
                "location": getattr(self.runtime, "steering_location", None), "selection": offline.get("selection"),
                "has_cards": bool(cards), "neurons": neurons}

    def overview(self) -> dict[str, Any]:
        """Model-level research material for the Mood Tracker: settings, policy, offline evidence and validation."""
        policy = self.policy or {}
        cells = [{"context": key, "alpha": cell.get("alpha", 0.0), "panel": cell.get("panel"), "reason": cell.get("reason")}
                 for key, cell in (policy.get("cells") or {}).items()]
        root = self.results.parent
        return {
            "model": getattr(self.runtime, "model_id", None), "model_tag": self.tag,
            "steering_location": getattr(self.runtime, "steering_location", None),
            "architecture": self.architecture(),
            "panels": {name: sum(len(neurons) for neurons, _ in by_layer.values())
                       for name, by_layer in (getattr(self.runtime, "_emotion_panels", {}) or {}).items()},
            "settings": {"research_compare": self.research_compare, "shadow_alpha": self.shadow_alpha,
                         "allow_overrides": self.allow_overrides, "telemetry": self.research.counts()},
            "policy": {"source": policy.get("source"), "version": policy.get("version"), "split": policy.get("split"),
                       "steering_enabled": any(cell["alpha"] for cell in cells), "note": policy.get("note"),
                       "gate": self.gate, "cells": cells} if self.policy else None,
            "evidence": self.evidence or None,
            "validation": load_json(self.results / "neuron_validation" / "report.json"),
            "single_neuron": load_json(self.results / "neuron_validation" / "single" / "report.json"),
            "trajectory_validation": load_json(root / "emotion_trajectory" / "validation.json"),
            "cross_model": load_json(root / "research" / "summary.json"),
        }
