from __future__ import annotations

import asyncio
import unittest

from langchain_core.messages import HumanMessage

from chatbot_feature.agents import EmotionReasonerNode
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent, build_graph, strategy_prompt
from chatbot_feature.emotion_reasoner.agent import MAX_ALPHA, MAX_ALPHA_HINGLISH
from chatbot_feature.emotion_reasoner.intervention import summarize_intervention
from chatbot_feature.graph import build_agent_graph
from chatbot_feature.mcp_connector.manager import MCPManager

LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness", "surprise")


def probe(top: str, confidence: float = 0.7) -> dict[str, float]:
    rest = (1 - confidence) / (len(LABELS) - 1)
    return {label: confidence if label == top else rest for label in LABELS}


class ReasonerTests(unittest.TestCase):
    def setUp(self):
        self.agent = EmotionReasonerAgent()

    def test_graph_links_every_emotion_to_a_strategy(self):
        graph = build_graph()
        for label in LABELS:
            needs = [n for _, n in graph.out_edges(f"emotion:{label}")]
            self.assertTrue(needs, label)
            self.assertTrue(any(graph.out_degree(need) for need in needs), label)

    def test_sadness_is_comforted_not_mirrored(self):
        plan = self.agent.plan("My dog passed away yesterday.", probe("sadness"))
        self.assertEqual(plan.user_emotion, "sadness")
        self.assertEqual(plan.strategy, "compassionate_comfort")
        self.assertEqual(plan.response_emotion, "love")
        self.assertIn("loss_grief", [cause["cause"] for cause in plan.causes])
        self.assertGreater(plan.steering_alpha, 0)
        self.assertEqual(plan.reasoning_paths[0]["hops"][-1]["to"], "love")

    def test_graph_evidence_corrects_a_weak_probe(self):
        plan = self.agent.plan("I finally got the job offer I was waiting for!", probe("love", 0.3))
        self.assertEqual(plan.user_emotion, "joy")
        self.assertEqual(plan.strategy, "celebratory_sharing")
        self.assertEqual(plan.response_emotion, "joy")

    def test_cues_match_whole_phrases_and_everyday_questions_stay_light(self):
        grief = self.agent.plan("My dog passed away yesterday.", probe("sadness"))
        self.assertNotIn("achievement", [cause["cause"] for cause in grief.causes])
        question = self.agent.plan("What should I cook for dinner tonight?", probe("anger", 0.98))
        self.assertEqual(question.user_emotion, "neutral")
        self.assertEqual(question.strategy, "gentle_exploration")
        self.assertLess(question.steering_alpha, grief.steering_alpha)

    def test_crisis_disables_steering(self):
        plan = self.agent.plan("I want to die, there is no reason to live", probe("sadness"))
        self.assertTrue(plan.safety_flag)
        self.assertEqual(plan.strategy, "crisis_support")
        self.assertIsNone(plan.response_emotion)
        self.assertEqual(plan.steering_alpha, 0.0)
        self.assertIn("emergency", plan.guidance)

    def test_strength_adapts_to_history_and_language(self):
        first = self.agent.plan("I feel so alone", probe("sadness"))
        sad = [{"emotion": "sadness", "valence": -0.8}] * 2
        later = self.agent.plan("I feel so alone", probe("sadness"), sad)
        self.assertGreater(later.steering_alpha, first.steering_alpha)
        self.assertIn("history", [factor["factor"] for factor in later.strength_factors])
        hinglish = self.agent.plan("yaar aaj bahut akela feel ho raha hai, bahut bahut dukh hai", probe("sadness", 0.95), sad)
        self.assertLessEqual(hinglish.steering_alpha, MAX_ALPHA_HINGLISH)
        self.assertLessEqual(later.steering_alpha, MAX_ALPHA)

    def test_missing_panel_disables_steering(self):
        agent = EmotionReasonerAgent(available_panels=["joy"])
        plan = agent.plan("My dog passed away yesterday.", probe("sadness"))
        self.assertIsNone(plan.response_emotion)
        self.assertEqual(plan.steering_alpha, 0.0)

    def test_strategy_prompt_and_intervention_summary(self):
        plan = self.agent.plan("I have a presentation tomorrow and I am scared", probe("fear"))
        self.assertIn(plan.strategy_title, strategy_prompt("Base.", plan))
        steered = {"emotion": "love", "alpha": 1.0, "gain": 1.75, "layers": [
            {"layer": 8, "neurons": [3, 9], "decode_positions": 5, "applied_shift": [0.5, 0.1],
             "mean_activation_before_shift": [0.2, 0.0]}]}
        unsteered = {"layers": [{"layer": 8, "neurons": [3, 9], "decode_positions": 6, "applied_shift": [0, 0],
                                 "mean_activation_before_shift": [0.1, 0.1]}]}
        summary = summarize_intervention(steered, unsteered)
        self.assertTrue(summary["applied"])
        self.assertEqual(summary["neurons_total"], 2)
        self.assertAlmostEqual(summary["layers"][0]["mean_abs_applied_shift"], 0.3)
        self.assertAlmostEqual(summary["layers"][0]["mean_abs_measured_change"], 0.3)
        self.assertEqual(summary["layers"][0]["neurons_raised"], 2)
        self.assertEqual(summary["top_neurons"][0]["neuron"], 3)
        self.assertFalse(summarize_intervention({}, None)["applied"])


class Model:
    def __init__(self):
        self.steered = []

    async def complete(self, prompt):
        return "plain"

    async def complete_steered(self, prompt, plan):
        self.steered.append((prompt, plan))
        return "steered", {"emotion": plan["response_emotion"], "alpha": plan["steering_alpha"], "layers": [
            {"layer": 8, "neurons": [1], "decode_positions": 2, "applied_shift": [0.3],
             "mean_activation_before_shift": [0.1]}]}


class GraphIntegrationTests(unittest.TestCase):
    def run_graph(self, message, model):
        manager = MCPManager({}, adapter_factory=lambda _: None)
        node = EmotionReasonerNode(EmotionReasonerAgent(), lambda text: probe("sadness"))
        graph = build_agent_graph(model, manager, emotion_reasoner=node)
        return asyncio.run(graph.ainvoke({"messages": [HumanMessage(content=message)], "task": message,
                                          "user_id": "u", "run_id": "r1"}))

    def test_reasoner_plans_and_response_is_steered(self):
        model = Model()
        state = self.run_graph("My best friend stopped talking to me and I feel alone", model)
        self.assertEqual(state["final_response"], "steered")
        self.assertEqual(state["emotion_plan"]["response_emotion"], "love")
        self.assertEqual(state["emotion_history"][0]["emotion"], "sadness")
        self.assertTrue(state["steering"]["applied"])
        self.assertIn("Response strategy", model.steered[0][0])

    def test_crisis_reply_is_not_steered(self):
        model = Model()
        state = self.run_graph("I want to end my life", model)
        self.assertEqual(state["final_response"], "plain")
        self.assertEqual(model.steered, [])
        self.assertTrue(state["emotion_plan"]["safety_flag"])


if __name__ == "__main__":
    unittest.main()
