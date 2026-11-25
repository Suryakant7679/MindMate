"""E5 run-time path: quality-constrained adaptive selective steering on top of the existing steering service.

``SelectiveSteeringService`` is an ``EmotionSteeringService`` with one extra path, ``run_selective``.  Every existing
path (E1-E4 and E5-old: ``begin_turn`` / ``respond`` with steering modes off / always / policy and the prompt
variants) is inherited unchanged.  The decision logic lives in ``chatbot_feature.emotion_reasoner.selective`` and the
frozen, model-specific policy in ``results/e5_selective/policies/<tag>.json``.

Per turn (see ``selective.py`` for the reasoning behind each step):

1. the Emotion Reasoner plans the turn exactly as steering mode ``always`` does (same strategy, response emotion,
   panel and state-scaled strength, so E4 and E5 differ only in the decisions below);
2. safety and context gates decide, before generation, whether candidates are generated at all; if not, the reply is
   generated once and unsteered - the same code path and text as the knowledge-graph-only reply (E3);
3. otherwise the unsteered reply and one candidate per policy strength are decoded in one batch (as E4 does);
4. the deficit gate and the acceptance checks pick the reply to send.

Every decision is logged in ``analysis["e5"]`` (eligibility, reasons, risk category, deficit estimate, panel, layers,
neuron count and fingerprint, whether the hooks ran, alphas tried, rejection reasons, latency).  This path is for the
E5 experiment; the chat API does not expose it.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from chatbot_feature.emotion_reasoner.intervention import summarize_intervention
from chatbot_feature.emotion_reasoner.risk_screen import screen
from chatbot_feature.emotion_reasoner.selective import (SelectivePolicy, choose, deficit_decision, language_of,
                                                        pre_decision)
from chatbot_feature.emotion_reasoner.comparison import compare_replies

from .emotion_service import EmotionSteeringService


def panel_fingerprint(panel: dict[int, tuple[list[int], list[float]]]) -> str:
    payload = json.dumps({str(layer): [list(map(int, ids)), [round(float(s), 6) for s in shifts]]
                          for layer, (ids, shifts) in sorted(panel.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def panel_manifest(runtime: Any) -> dict[str, Any]:
    """Every steering panel of the loaded model: layers, neuron ids and a fingerprint (written once per run)."""
    panels = getattr(runtime, "_emotion_panels", {}) or {}
    return {name: {"fingerprint": panel_fingerprint(panel), "neurons": sum(len(ids) for ids, _ in panel.values()),
                   "layers": {str(layer): list(map(int, ids)) for layer, (ids, _) in sorted(panel.items())}}
            for name, panel in panels.items() if panel}


def hook_summary(trace: dict[str, Any] | None) -> dict[str, Any]:
    """Whether the steering hooks actually ran on decode steps, from the runtime's trace."""
    layers = (trace or {}).get("layers") or []
    positions = [int(layer.get("decode_positions") or 0) for layer in layers]
    applied = [any(abs(float(shift)) > 0 for shift in layer.get("applied_shift") or []) for layer in layers]
    return {"layers": [int(layer["layer"]) for layer in layers], "neurons": sum(len(layer.get("neurons") or []) for layer in layers),
            "decode_positions_min": min(positions) if positions else 0,
            "decode_positions_max": max(positions) if positions else 0,
            "hooks_executed": bool(positions) and min(positions) > 0, "shift_applied": any(applied),
            "alpha": float((trace or {}).get("alpha") or 0.0), "location": (trace or {}).get("location")}


class SelectiveSteeringService(EmotionSteeringService):
    """The steering service plus E5's selective path (``run_selective``)."""

    def __init__(self, runtime: Any, policy: SelectivePolicy, **kwargs: Any) -> None:
        super().__init__(runtime, **kwargs)
        if policy.model != self.tag:
            raise ValueError(f"The E5 policy is for {policy.model}, but the loaded model is {self.tag}.")
        self.selective_policy = policy
        self.policy_fingerprint = policy.fingerprint()
        panels = getattr(runtime, "_emotion_panels", {}) or {}
        self.panel_fingerprints = {name: panel_fingerprint(panel) for name, panel in panels.items() if panel}

    # ----------------------------------------------------------------------------------- planning
    def plan_selective(self, messages: list[dict[str, str]], *, conversation_id: str | None = None
                       ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Plan as mode ``always`` (E4's proposal), then apply E5's safety and context gates (steps 1-2)."""
        turn_id, plan = self.begin_turn(messages, conversation_id=conversation_id, steering_mode="always", mode="e5")
        users = [message["content"] for message in messages if message["role"] == "user"]
        user_text = users[-1]
        risk = screen(user_text, messages[:-1])
        proposal = {"alpha": float(plan.get("steering_alpha") or 0.0), "panel": plan.get("steering_panel"),
                    "response_emotion": plan.get("response_emotion"), "state_level": plan.get("state_level")}
        panel = plan.get("steering_panel") if proposal["alpha"] else None
        pre = pre_decision(self.selective_policy, plan, user_text, risk, panel=panel)
        plan["steering_mode"] = "selective"
        plan["e4_proposal"] = proposal
        if not pre["eligible"]:
            plan["steering_alpha"] = 0.0
            plan["strength_factors"].insert(0, {"factor": "e5", "effect": "alpha 0", "reason": f"E5 {pre['stage']}: {pre['reason']}"})
        with self._lock:
            self._turns[turn_id]["selective"] = {"risk": risk, "pre": pre, "proposal": proposal, "user_text": user_text}
        return turn_id, plan, pre

    def will_generate_candidates(self, turn_id: str) -> bool:
        return bool(self._turns[turn_id]["selective"]["pre"]["eligible"])

    # ----------------------------------------------------------------------------------- reply
    def run_selective(self, turn_id: str, max_new_tokens: int) -> dict[str, Any]:
        """Generate and choose the reply for a turn planned by ``plan_selective`` (steps 3-4)."""
        turn = self._turns[turn_id]
        info = turn["selective"]
        plan, pre = turn["plan"], info["pre"]
        started = time.perf_counter()
        base_log = {"policy_version": self.selective_policy.version, "policy_fingerprint": self.policy_fingerprint,
                    "risk_category": info["risk"]["category"], "risk_cues": info["risk"]["cues"],
                    "risk_source": info["risk"].get("source"), "language": pre["language"],
                    "eligible": pre["eligible"], "stage": pre["stage"], "eligibility_reason": pre["reason"],
                    "cell": pre["cell"], "multipliers": pre["multipliers"], "alphas_planned": pre["alphas"],
                    "alpha_e4": pre["alpha_e4"], "e4_proposal": info["proposal"]}
        if not pre["eligible"]:
            analysis = self.respond(turn_id, max_new_tokens)          # the knowledge-graph-only path (E3's code)
            analysis["e5"] = {**base_log, "attempted": False, "retained": False, "status": "not_eligible",
                              "decision_reason": pre["reason"], "deficit": None, "tried": [], "applied_alpha": 0.0,
                              "panel": None, "panel_fingerprint": None, "hooks": None, "candidates_generated": 0,
                              "latency_seconds": round(time.perf_counter() - started, 3)}
            return analysis

        messages, system_prompt = turn["messages"], turn["system_prompt"]
        panel_name = plan["steering_panel"]
        target = plan["response_emotion"]
        alphas = [0.0, *pre["alphas"]]
        generation_started = time.perf_counter()
        outputs = self.runtime.generate_candidates(messages, max_new_tokens=max_new_tokens, system_prompt=system_prompt,
                                                   steering_emotion=panel_name, alphas=alphas)
        generation_seconds = time.perf_counter() - generation_started
        unsteered = outputs[0]["response"]
        scoring_started = time.perf_counter()
        unsteered_scores = self.scorer.score(unsteered)
        candidate_scores = [self.scorer.score(output["response"]) for output in outputs[1:]]
        scoring_seconds = time.perf_counter() - scoring_started
        deficit = deficit_decision(self.selective_policy, target, unsteered, unsteered_scores, language=pre["language"])
        if deficit["steer"]:
            choice = choose(self.selective_policy, unsteered,
                            [(alpha, output["response"]) for alpha, output in zip(alphas[1:], outputs[1:])],
                            user_text=info["user_text"])
        else:
            choice = {"sent": "unsteered", "alpha": 0.0, "reply": unsteered, "tried": []}
        if choice["sent"] == "steered":
            index = alphas.index(choice["alpha"])
            status, reason = "applied", f"sent the alpha {choice['alpha']:.2f} candidate: it passed every safeguard"
        elif not deficit["steer"]:
            index, status, reason = 0, "skipped", deficit["reason"]
        else:
            index, status = 0, "rejected"
            reason = "every candidate was rejected: " + "; ".join(f"alpha {t['alpha']:.2f} {t['reason']}" for t in choice["tried"])
        reply = choice["reply"]
        sent_trace = outputs[index]["trace"]
        steering = summarize_intervention(sent_trace if index else None, outputs[0]["trace"],
                                          cards=self.cards.get(panel_name or "", {}), detail=False)
        steering["sent"] = index > 0
        steering["shadow"] = False
        decision = {"planned_alpha": max(pre["alphas"]), "applied_alpha": float(choice["alpha"]), "mode": "selective",
                    "status": status, "reason": reason}
        effect = {"method": self.scorer.method, "target": target, "unsteered_reply": unsteered,
                  "unsteered_scores": unsteered_scores, "steered_reply": outputs[1]["response"],
                  "steered_scores": candidate_scores[0], "sent": "steered" if index else "unsteered", "shadow": False,
                  "candidates": [{"alpha": alpha, "reply": output["response"], "scores": scores}
                                 for alpha, output, scores in zip(alphas[1:], outputs[1:], candidate_scores)]}
        comparison = compare_replies(info["user_text"], unsteered, reply if index else outputs[1]["response"],
                                     unsteered_scores, candidate_scores[max(index - 1, 0)], target,
                                     delivered=effect["sent"], shadow=False)
        analysis = {"plan": plan, "decision": decision, "steering": steering, "effect": effect, "reply": reply,
                    "comparison": comparison, "turn_id": turn_id, "evidence": self.evidence_for(plan),
                    "first_token_seconds": None, "latency_seconds": round(time.perf_counter() - started, 3)}
        analysis["e5"] = {**base_log, "attempted": True, "retained": index > 0, "status": status, "decision_reason": reason,
                          "deficit": deficit, "tried": choice["tried"], "applied_alpha": float(choice["alpha"]),
                          "panel": panel_name, "panel_fingerprint": self.panel_fingerprints.get(panel_name),
                          "hooks": hook_summary(outputs[index]["trace"] if index else outputs[1]["trace"]),
                          "hooks_of": "sent candidate" if index else "first (unsent) candidate",
                          "candidates_generated": len(alphas) - 1, "generation_seconds": round(generation_seconds, 3),
                          "scoring_seconds": round(scoring_seconds, 3),
                          "latency_seconds": analysis["latency_seconds"]}
        self.finalize_turn(messages, plan, analysis, conversation_id=turn.get("conversation_id"), turn_id=turn_id,
                           mode="e5")
        turn["analysis"] = analysis
        return analysis

    def respond_selective(self, messages: list[dict[str, str]], max_new_tokens: int, *,
                          conversation_id: str | None = None) -> dict[str, Any]:
        """Plan and answer one turn on the E5 path."""
        turn_id, _, _ = self.plan_selective(messages, conversation_id=conversation_id or f"e5-{uuid.uuid4().hex[:8]}")
        return self.run_selective(turn_id, max_new_tokens)


__all__ = ["SelectiveSteeringService", "hook_summary", "panel_fingerprint", "panel_manifest", "language_of"]
