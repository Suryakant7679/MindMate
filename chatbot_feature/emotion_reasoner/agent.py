"""Emotion Reasoner Agent: understand the user's emotion and plan the reply.

For each user turn the agent

1. fuses the frozen probe's emotion probabilities with knowledge-graph
   evidence (cue phrases -> cause -> emotion),
2. walks the graph ``emotion/cause -> need -> strategy -> response emotion``
   and picks the best-supported, therapist-inspired response strategy,
3. sets the neuron-steering strength from the strategy, the intensity of the
   emotion, the conversation history, the language, and how sensitive the
   chosen neuron panel is.

The result is an :class:`EmotionPlan`: fully deterministic and explainable as
graph paths.  The agent has no model dependency; it receives probe scores and
returns a plan, so the same instance serves the direct chat endpoints and the
LangGraph workflow.  It is a supportive conversational aid, not a therapist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .knowledge_graph import INTENSIFIERS, build_graph, is_hinglish, matched_cues

# How strongly each neuron panel reacts per unit of alpha, from
# generation_analysis.evaluate_chat_steering (love is the reference).
PANEL_SENSITIVITY = {"love": 1.0, "admiration": 0.8, "joy": 1.35, "surprise": 1.2}
MAX_ALPHA = 1.2            # beyond this replies lose fluency
MAX_ALPHA_HINGLISH = 0.7   # Hinglish replies degrade earlier
PROBE_WEIGHT, GRAPH_WEIGHT = 0.45, 0.55   # cue-backed graph evidence outranks the probe alone


@dataclass
class EmotionPlan:
    user_emotion: str
    confidence: float
    intensity: float
    valence: float
    arousal: float
    probe_scores: dict[str, float]
    fused_scores: dict[str, float]
    causes: list[dict[str, Any]]
    needs: list[dict[str, Any]]
    strategy: str
    strategy_title: str
    technique: str
    guidance: str
    response_emotion: str | None
    steering_alpha: float
    strength_factors: list[dict[str, Any]]
    reasoning_paths: list[dict[str, Any]]
    history: list[dict[str, Any]] = field(default_factory=list)
    safety_flag: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EmotionReasonerAgent:
    """Knowledge-graph reasoner that turns a detected emotion into a response plan."""

    name = "emotion_reasoner"

    def __init__(self, available_panels: Sequence[str] | None = None) -> None:
        self.graph = build_graph()
        # Response emotions that actually have a neuron panel; None means "all in the graph".
        self.available_panels = set(available_panels) if available_panels is not None else None

    # ------------------------------------------------------------------ understanding

    def _causes(self, text: str) -> list[dict[str, Any]]:
        found = []
        for node, data in self.graph.nodes(data=True):
            if data["kind"] != "cause":
                continue
            cues = matched_cues(text, data["cues"])
            if cues:
                found.append({"cause": data["label"], "description": data["description"], "cues": cues,
                              "weight": min(1.0, 0.6 + 0.2 * len(cues))})
        return sorted(found, key=lambda item: -item["weight"])

    def _fuse(self, probe: Mapping[str, float], causes: list[dict[str, Any]]) -> dict[str, float]:
        labels = [data["label"] for _, data in self.graph.nodes(data=True) if data["kind"] == "emotion"]
        evidence = dict.fromkeys(labels, 0.0)
        for cause in causes:
            for _, emotion, edge in self.graph.out_edges(f"cause:{cause['cause']}", data=True):
                if edge["relation"] == "evokes":
                    label = self.graph.nodes[emotion]["label"]
                    evidence[label] += cause["weight"] * edge["weight"]
        total = sum(evidence.values())
        if not total:
            fused = {label: float(probe.get(label, 0.0)) for label in labels}
        else:
            fused = {label: PROBE_WEIGHT * float(probe.get(label, 0.0)) + GRAPH_WEIGHT * evidence[label] / total
                     for label in labels}
        norm = sum(fused.values()) or 1.0
        return {label: round(value / norm, 4) for label, value in fused.items()}

    def understand(self, text: str, probe_scores: Mapping[str, float]) -> dict[str, Any]:
        """Emotion, causes and intensity of one user message."""
        causes = self._causes(text)
        fused = self._fuse(probe_scores, causes)
        emotion = max(fused, key=fused.get)
        node = self.graph.nodes[f"emotion:{emotion}"]
        emphasis = sum(text.lower().count(word) for word in INTENSIFIERS) + (text.isupper() and len(text) > 8)
        intensity = min(1.0, 0.35 + 0.5 * fused[emotion] + 0.08 * emphasis + 0.1 * bool(causes))
        return {"emotion": emotion, "confidence": fused[emotion], "intensity": round(intensity, 3),
                "valence": node["valence"], "arousal": node["arousal"], "causes": causes, "fused": fused}

    # ---------------------------------------------------------------------- reasoning

    def _strategy(self, emotion: str, confidence: float, causes: list[dict[str, Any]]
                  ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        """Score every ``source -> need -> strategy`` path; return the best strategy and its support."""
        sources = [(f"emotion:{emotion}", max(confidence, 0.3))]
        sources += [(f"cause:{cause['cause']}", cause["weight"]) for cause in causes]
        scores: dict[str, float] = {}
        paths: list[dict[str, Any]] = []
        need_scores: dict[str, float] = {}
        for source, source_weight in sources:
            for _, need, first in self.graph.out_edges(source, data=True):
                if first["relation"] != "needs":
                    continue
                need_label = self.graph.nodes[need]["label"]
                need_scores[need_label] = need_scores.get(need_label, 0.0) + source_weight * first["weight"]
                for _, strategy, second in self.graph.out_edges(need, data=True):
                    label = self.graph.nodes[strategy]["label"]
                    score = source_weight * first["weight"] * second["weight"]
                    scores[label] = scores.get(label, 0.0) + score
                    paths.append({"source": source, "need": need_label, "strategy": label,
                                  "score": round(score, 3)})
        if any(cause["cause"] == "crisis" for cause in causes):
            best = "crisis_support"            # safety outranks every other consideration
        else:
            best = max(scores, key=scores.get)
        needs = [{"need": need, "description": self.graph.nodes[f"need:{need}"]["description"],
                  "score": round(score, 3)} for need, score in sorted(need_scores.items(), key=lambda kv: -kv[1])]
        support = sorted((path for path in paths if path["strategy"] == best), key=lambda path: -path["score"])
        return best, needs, support

    def _strength(self, strategy: Mapping[str, Any], response: str | None, intensity: float, valence: float,
                  history: Sequence[Mapping[str, Any]], text: str) -> tuple[float, list[dict[str, Any]]]:
        """Steering alpha plus the factors that produced it."""
        if response is None:
            return 0.0, [{"factor": "safety", "effect": "steering disabled",
                          "reason": "no activation steering while the user may be at risk"}]
        base = float(strategy["base_strength"])
        factors = [{"factor": "strategy", "effect": f"base {base:.2f}",
                    "reason": f"{strategy['title']} calls for this much emotional expression"}]
        scale = 0.8 + 0.4 * intensity
        alpha = base * scale
        factors.append({"factor": "intensity", "effect": f"x{scale:.2f}",
                        "reason": f"user emotion intensity {intensity:.2f}"})
        sensitivity = PANEL_SENSITIVITY.get(response, 1.0)
        if sensitivity != 1.0:
            alpha *= sensitivity
            factors.append({"factor": "panel", "effect": f"x{sensitivity:.2f}",
                            "reason": f"the {response} neuron panel is {'less' if sensitivity > 1 else 'more'} "
                                      "responsive than the reference panel"})
        negative = [turn for turn in history if turn.get("valence", 0) < -0.3]
        if valence < -0.3 and len(history) >= 1 and len(negative) == len(history):
            alpha += 0.15
            factors.append({"factor": "history", "effect": "+0.15",
                            "reason": f"distress has persisted for {len(history) + 1} turns"})
        elif history and valence - history[-1].get("valence", 0) > 0.4:
            alpha -= 0.1
            factors.append({"factor": "history", "effect": "-0.10",
                            "reason": "the user's mood has lifted since the previous turn"})
        limit = MAX_ALPHA_HINGLISH if is_hinglish(text) else MAX_ALPHA
        if alpha > limit:
            factors.append({"factor": "fluency limit", "effect": f"capped at {limit:.2f}",
                            "reason": "Hinglish replies lose fluency sooner under steering"
                            if limit == MAX_ALPHA_HINGLISH else "stronger steering costs fluency"})
            alpha = limit
        return round(max(alpha, 0.0), 2), factors

    def plan(self, text: str, probe_scores: Mapping[str, float],
             history: Sequence[Mapping[str, Any]] = ()) -> EmotionPlan:
        """Full plan for the latest user message.

        ``history`` holds earlier user turns, oldest first, each a mapping with
        at least ``emotion`` and ``valence`` (as returned by :meth:`understand`).
        """
        state = self.understand(text, probe_scores)
        emotion, causes = state["emotion"], state["causes"]
        strategy_key, needs, support = self._strategy(emotion, state["confidence"], causes)
        strategy = self.graph.nodes[f"strategy:{strategy_key}"]
        response = next((self.graph.nodes[node]["label"] for _, node, edge in
                         self.graph.out_edges(f"strategy:{strategy_key}", data=True)
                         if edge["relation"] == "expresses"), None)
        if response is not None and self.available_panels is not None and response not in self.available_panels:
            response = None
        alpha, factors = self._strength(strategy, response, state["intensity"], state["valence"], history, text)
        if response is None and strategy_key != "crisis_support":
            factors = [{"factor": "panel", "effect": "steering disabled",
                        "reason": "no neuron panel is available for the planned response emotion"}]
        paths = []
        for path in support[:4]:
            kind, label = path["source"].split(":", 1)
            hops = [{"from": label, "from_kind": kind, "relation": "needs", "to": path["need"], "to_kind": "need"},
                    {"from": path["need"], "from_kind": "need", "relation": "addressed_by", "to": strategy_key,
                     "to_kind": "strategy"}]
            if response:
                hops.append({"from": strategy_key, "from_kind": "strategy", "relation": "expresses",
                             "to": response, "to_kind": "response"})
            paths.append({"score": path["score"], "hops": hops})
        return EmotionPlan(
            user_emotion=emotion, confidence=state["confidence"], intensity=state["intensity"],
            valence=state["valence"], arousal=state["arousal"],
            probe_scores={label: round(float(value), 4) for label, value in probe_scores.items()},
            fused_scores=state["fused"], causes=causes, needs=needs[:4], strategy=strategy_key,
            strategy_title=strategy["title"], technique=strategy["technique"], guidance=strategy["guidance"],
            response_emotion=response, steering_alpha=alpha if response else 0.0, strength_factors=factors,
            reasoning_paths=paths,
            history=[{"emotion": turn.get("emotion"), "valence": turn.get("valence")} for turn in history],
            safety_flag=strategy_key == "crisis_support",
        )


def strategy_prompt(base_prompt: str | None, plan: EmotionPlan | Mapping[str, Any]) -> str:
    """System prompt extended with the planned response strategy."""
    data = plan.to_dict() if isinstance(plan, EmotionPlan) else plan
    guidance = f"Response strategy for this turn ({data['strategy_title']}): {data['guidance']}"
    return f"{base_prompt} {guidance}" if base_prompt else guidance
