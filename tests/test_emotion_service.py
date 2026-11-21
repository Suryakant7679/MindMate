from chatbot_feature.safety import SafetyAgent
from chatbot_feature.safety.session import SessionStore
from chatbot_feature.tests.safety_fakes import SafetyModel
"""Steering decision, safeguards and streaming of the shared emotion service,
and their consistency between direct chat and LangGraph/MCP agent mode."""
import asyncio
import io
import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage

from backend.emotion_service import EmotionSteeringService
from backend.server import RequestHandler
from chatbot_feature.agents import EmotionReasonerNode, LocalRuntimeModel
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
from chatbot_feature.graph import build_agent_graph
from chatbot_feature.mcp_connector.manager import MCPManager

LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness", "surprise")
SWEEP = [{"alpha": 0.75, "n": 8, "target_gain": 0.3, "safe": True}]
CELLS = {"love|english": {"alpha": 0.75, "reason": "safe gain", "sweep": SWEEP},
         "love|hinglish": {"alpha": 0.0, "reason": "not steered: quality fell", "sweep": SWEEP}}
POLICY = {"rule": {}, "ceiling": {"threshold": 0.3}, "cells": CELLS}                       # classifier-gain rule
SAFEGUARD_POLICY = {"rule": {}, "cells": CELLS, "gate": {"rule": "safeguards", "ceiling": 0.3}}
PLAIN = "plain reply that listens to you today"
STEERED = "steered reply with warmth and care for you today"
SAD = "Everyone forgot about me today and it hurts"
HINGLISH = "yaar aaj bahut akela feel ho raha hai, koi nahi hai mere saath"
CRISIS = "I want to end my life"


def trace(emotion, alpha):
    return {"emotion": emotion, "alpha": alpha, "gain": 1.75, "layers": [
        {"layer": 8, "neurons": [1, 2], "decode_positions": 4, "applied_shift": [0.2 * alpha, 0.0],
         "mean_activation_before_shift": [0.1, 0.1]}]}


class SequentialRuntime:
    """A runtime with only ``generate`` (two sequential generations per steered turn)."""
    _emotion_panels = {"love": {}}

    def __init__(self, steered_text=STEERED):
        self.calls, self.prompts, self.steered_text = [], [], steered_text

    def generate(self, messages, max_new_tokens, deterministic, temperature, system_prompt=None,
                 steering_alpha=0.0, steering_emotion=None, steering_trace=None):
        self.calls.append(steering_alpha)
        self.prompts.append(system_prompt)
        if steering_trace is not None:
            steering_trace.update(trace(steering_emotion, steering_alpha))
        return {"response": self.steered_text if steering_alpha else PLAIN}


class Runtime(SequentialRuntime):
    """Like the Llama runtime: batched candidates and token streaming."""

    def generate_candidates(self, messages, max_new_tokens, system_prompt, steering_emotion, alphas, panel=None):
        self.calls.append(list(alphas))
        self.prompts.append(system_prompt)
        return [{"response": self.steered_text if alpha else PLAIN, "trace": trace(steering_emotion, alpha)}
                for alpha in alphas]

    def stream(self, messages, max_new_tokens, deterministic, temperature, system_prompt=None, **_):
        self.calls.append("stream")
        self.prompts.append(system_prompt)
        yield from (" plain", " reply", " that", " listens", " to", " you", " today")


class Scorer:
    model_id, method, uses_classifier = "fake", "fake classifier", True

    def __init__(self, love_plain, love_steered):
        self.love = {"plain": love_plain, "steered": love_steered}

    def score(self, text):
        scores = dict.fromkeys(LABELS, 0.01)
        if text.startswith("plain") or text.startswith("steered"):
            scores["love"] = self.love[text.split()[0]]
        else:                                   # a user message: read as sad
            scores["sadness"] = 0.9
        return scores


def service(love_plain, love_steered, *, policy=POLICY, runtime_class=Runtime, options=None, **runtime_kwargs):
    runtime = runtime_class(**runtime_kwargs)
    return EmotionSteeringService(runtime, EmotionReasonerAgent(["love"], policy=policy),
                                  Scorer(love_plain, love_steered), evidence={}, **(options or {})), runtime


def run(svc, text=SAD, **overrides):
    turn_id, plan = svc.begin_turn([{"role": "user", "content": text}], **overrides)
    return plan, svc.respond(turn_id, 64)


class DecisionTests(unittest.TestCase):
    def test_steered_reply_sent_only_when_it_raises_the_target(self):
        svc, runtime = service(0.05, 0.40)
        plan, analysis = run(svc)
        self.assertEqual(plan["steering_alpha"], 0.75)
        self.assertEqual(analysis["decision"]["status"], "applied")
        self.assertEqual(analysis["reply"], STEERED)
        self.assertEqual(runtime.calls, [[0.0, 0.75]])          # one batched pass for both candidates
        self.assertTrue(analysis["steering"]["sent"])
        self.assertIsNotNone(analysis["steering"]["layers"][0]["mean_abs_measured_change"])
        self.assertIs(svc.analysis(analysis["turn_id"]), analysis)

    def test_insufficient_or_negative_gain_is_rejected(self):
        for steered_score in (0.07, 0.01):
            svc, _ = service(0.05, steered_score)
            _, analysis = run(svc)
            self.assertEqual(analysis["decision"]["status"], "rejected")
            self.assertEqual(analysis["decision"]["applied_alpha"], 0.0)
            self.assertEqual(analysis["reply"], PLAIN)
            self.assertEqual(analysis["effect"]["sent"], "unsteered")
            self.assertFalse(analysis["steering"]["sent"])

    def test_safeguard_rule_does_not_select_on_the_classifier_gain(self):
        svc, _ = service(0.05, 0.01, policy=SAFEGUARD_POLICY)
        self.assertEqual(run(svc)[1]["decision"]["status"], "applied")
        svc, _ = service(0.45, 0.90, policy=SAFEGUARD_POLICY)       # the ceiling still applies
        self.assertEqual(run(svc)[1]["decision"]["status"], "skipped")

    def test_repetitive_or_collapsed_steered_reply_is_rejected(self):
        for policy in (POLICY, SAFEGUARD_POLICY):
            for text in ("steered so so so so so so so so so so much", "steered ok"):
                svc, _ = service(0.05, 0.60, policy=policy, steered_text=text)
                _, analysis = run(svc)
                self.assertEqual(analysis["decision"]["status"], "rejected")
                self.assertEqual(analysis["reply"], PLAIN)

    def test_steering_skipped_when_reply_already_expresses_target(self):
        svc, _ = service(0.45, 0.90)
        _, analysis = run(svc)
        self.assertEqual(analysis["decision"]["status"], "skipped")
        self.assertEqual(analysis["reply"], PLAIN)
        self.assertEqual(analysis["effect"]["sent"], "unsteered")

    def test_policy_without_safe_strength_and_crisis_do_not_steer(self):
        svc, runtime = service(0.05, 0.90)
        plan, analysis = run(svc, HINGLISH)
        self.assertEqual((plan["steering_alpha"], analysis["decision"]["status"]), (0.0, "not_planned"))
        plan, analysis = run(svc, CRISIS)
        self.assertTrue(plan["safety_flag"])
        self.assertEqual(analysis["decision"]["status"], "not_planned")
        self.assertEqual(runtime.calls, [0.0, 0.0])              # never a steered candidate
        self.assertIsNone(analysis["effect"]["steered_reply"])

    def test_api_override_forces_steering_only_when_the_server_enables_it(self):
        svc, _ = service(0.45, 0.10, options={"allow_overrides": True})
        _, analysis = run(svc, alpha_override=1.5)
        self.assertEqual(analysis["decision"]["status"], "applied")
        self.assertEqual(analysis["decision"]["applied_alpha"], 1.5)
        svc, _ = service(0.45, 0.10)                                  # default: overrides are ignored
        plan, analysis = run(svc, alpha_override=1.5)
        self.assertEqual(plan["steering_alpha"], 0.75)
        self.assertEqual(analysis["decision"]["status"], "skipped")
        self.assertEqual(plan["strength_factors"][-1]["effect"], "ignored")

    def test_forced_steering_still_passes_the_reply_safeguards_and_never_runs_on_a_crisis_turn(self):
        svc, _ = service(0.05, 0.90, options={"allow_overrides": True},
                         steered_text="steered so so so so so so so so so so much")
        _, analysis = run(svc, alpha_override=1.5)
        self.assertEqual(analysis["decision"]["status"], "rejected")
        self.assertEqual(analysis["reply"], PLAIN)
        svc, runtime = service(0.05, 0.90, options={"allow_overrides": True})
        plan, analysis = run(svc, CRISIS, alpha_override=1.5, emotion_override="love")
        self.assertTrue(plan["safety_flag"])
        self.assertEqual(analysis["decision"]["status"], "not_planned")
        self.assertEqual(analysis["decision"]["disabled_by"], "safety")
        self.assertIsNone(analysis["effect"]["steered_reply"])

    def test_runtime_without_batched_candidates_gives_the_same_decisions(self):
        for scores, expected in (((0.05, 0.40), "applied"), ((0.05, 0.07), "rejected"), ((0.45, 0.90), "skipped")):
            batched, _ = service(*scores)
            sequential, runtime = service(*scores, runtime_class=SequentialRuntime)
            first, second = run(batched)[1], run(sequential)[1]
            self.assertEqual(first["decision"], second["decision"])
            self.assertEqual(first["decision"]["status"], expected)
            self.assertEqual(first["reply"], second["reply"])
            self.assertEqual(runtime.calls, [0.0, 0.75])

    def test_offline_evidence_is_attached_for_the_planned_emotion(self):
        evidence = {"panels": {"love": {"neurons": 3}}, "held_out": {"messages": 160}, "sources": {}}
        svc = EmotionSteeringService(Runtime(), EmotionReasonerAgent(["love"], policy=POLICY), Scorer(0.05, 0.4),
                                     evidence=evidence)
        self.assertEqual(run(svc)[1]["evidence"]["panel"], {"neurons": 3})
        self.assertIsNone(run(svc, CRISIS)[1]["evidence"])


class StreamingTests(unittest.TestCase):
    def collect(self, svc, text):
        events = []
        turn_id, _ = svc.begin_turn([{"role": "user", "content": text}])
        analysis = svc.respond(turn_id, 64, on_decision=lambda decision: events.append(("decision", decision["status"])),
                               on_text=lambda piece: events.append(("text", piece)))
        return events, analysis

    def test_unsteered_turn_streams_tokens_as_they_are_generated(self):
        svc, runtime = service(0.05, 0.90)
        events, analysis = self.collect(svc, HINGLISH)
        self.assertEqual(runtime.calls, ["stream"])
        self.assertEqual(events[0], ("decision", "not_planned"))
        self.assertEqual(len(events), 8)                         # the decision, then one event per token
        self.assertEqual("".join(piece for _, piece in events[1:]), PLAIN)
        self.assertEqual(analysis["reply"], PLAIN)
        self.assertIsNotNone(analysis["first_token_seconds"])
        self.assertLessEqual(analysis["first_token_seconds"], analysis["latency_seconds"])

    def test_steered_turn_sends_text_only_after_the_safeguards(self):
        for scores, status, reply in (((0.05, 0.40), "applied", STEERED), ((0.05, 0.07), "rejected", PLAIN)):
            svc, runtime = service(*scores)
            events, analysis = self.collect(svc, SAD)
            self.assertEqual(runtime.calls, [[0.0, 0.75]])
            self.assertEqual(events[0], ("decision", status))    # nothing is shown before the decision
            self.assertEqual("".join(piece for _, piece in events[1:]), reply)

    def http(self, svc, text):
        handler = RequestHandler.__new__(RequestHandler)
        body = json.dumps({"messages": [{"role": "user", "content": text}]}).encode()
        handler.path = "/api/chat/stream"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.server = Mock(agent_service=None)
        handler.server.session_secret = b"test-secret"
        handler.server.safety_sessions = SessionStore()
        handler.server.resource_owners = SessionStore()
        handler._emotion_service = lambda: svc
        handler.send_response, handler.send_header, handler.end_headers = Mock(), Mock(), Mock()
        with patch("backend.server.SafetyRuntimeModel", return_value=SafetyModel()):
            handler.do_POST()
        headers = {call.args[0]: call.args[1] for call in handler.send_header.call_args_list}
        return headers, handler.wfile.getvalue().decode()

    def test_stream_endpoint_reports_the_decision_in_headers(self):
        svc, _ = service(0.05, 0.40)
        headers, body = self.http(svc, SAD)
        self.assertEqual((headers["X-Steering-Status"], headers["X-Steering-Alpha"]), ("applied", "0.75"))
        self.assertEqual(body, STEERED)
        self.assertIsNotNone(svc.analysis(headers["X-Turn-Id"]))
        headers, body = self.http(svc, CRISIS)
        self.assertEqual(headers["X-Steering-Alpha"], "0")
        self.assertEqual(headers["X-Safety-Route"], "safety")
        self.assertIn("safe right now", body)


class AgentModeConsistencyTests(unittest.TestCase):
    """LangGraph/MCP agent replies use the same selection and safeguards as direct chat."""

    def agent_state(self, svc, text):
        node = EmotionReasonerNode(svc.agent, svc.probe_scores)
        graph = build_agent_graph(LocalRuntimeModel(svc.runtime, emotion_service=svc),
                                  MCPManager({}, adapter_factory=lambda _: None), emotion_reasoner=node, safety_agent=SafetyAgent(SafetyModel()))
        return asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=text)], "task": text, "user_id": "u",
                                          "run_id": "r1"}))

    def test_same_plan_decision_and_reply_in_both_modes(self):
        cases = (((0.05, 0.40), POLICY, SAD, "applied"), ((0.05, 0.07), POLICY, SAD, "rejected"),
                 ((0.45, 0.90), POLICY, SAD, "skipped"), ((0.05, 0.07), SAFEGUARD_POLICY, SAD, "applied"),
                 ((0.05, 0.90), POLICY, HINGLISH, "not_planned"), ((0.05, 0.90), POLICY, CRISIS, "not_planned"))
        for scores, policy, text, expected in cases:
            with self.subTest(scores=scores, text=text, rule=policy.get("gate")):
                direct_service, _ = service(*scores, policy=policy)
                plan, direct = run(direct_service, text)
                agent_service, runtime = service(*scores, policy=policy)
                state = self.agent_state(agent_service, text)
                self.assertEqual(direct["decision"]["status"], expected)
                if text == CRISIS:
                    self.assertEqual(state["safety"]["policy"]["route"], "safety")
                    self.assertEqual(state["emotion_plan"], {})
                    self.assertEqual(runtime.calls, [])
                    continue
                for key in ("strategy", "response_emotion", "steering_alpha", "safety_flag"):
                    self.assertEqual(state["emotion_plan"][key], plan[key])
                self.assertEqual(state["steering"]["decision"], direct["decision"])
                self.assertEqual(state["final_response"], direct["reply"])
                self.assertEqual(state["steering"]["effect"]["sent"], direct["effect"]["sent"])
                self.assertEqual(state["steering"]["sent"], direct["steering"]["sent"])
                if expected == "not_planned":
                    self.assertNotIn([0.0, 0.75], runtime.calls)

    def test_agent_model_without_the_service_never_steers(self):
        runtime = Runtime()
        model = LocalRuntimeModel(runtime)
        self.assertFalse(hasattr(model, "complete_planned"))
        node = EmotionReasonerNode(EmotionReasonerAgent(["love"], policy=POLICY), lambda text: Scorer(0, 0).score(text))
        graph = build_agent_graph(model, MCPManager({}, adapter_factory=lambda _: None), emotion_reasoner=node, safety_agent=SafetyAgent(SafetyModel()))
        state = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=SAD)], "task": SAD, "user_id": "u",
                                           "run_id": "r1"}))
        self.assertEqual(state["final_response"], PLAIN)
        self.assertEqual(runtime.calls, [0.0])
        self.assertEqual(state["steering"], {})


if __name__ == "__main__":
    unittest.main()
