"""Steering mode "always": every reply steered, strength from the user's emotional state, before/after tone."""
import unittest

from backend.chat_engine import TurnEngine, tone_record
from backend.config import MODEL_REGISTRY
from backend.emotion_service import STATE_ALPHA_RANGE, EmotionSteeringService
from backend.generation_control import GenerationControl
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
from chatbot_feature.emotion_reasoner.agent import CRISIS_WARMTH
from chatbot_feature.emotion_reasoner.knowledge_graph import EMOTIONS
from chatbot_feature.session import SessionStore
from tests.test_emotion_service import CRISIS, HINGLISH, PLAIN, POLICY, SAD, STEERED, Runtime, Scorer, service

PANELS = {"love": {8: ([1, 2], [0.1, 0.2])}, "admiration": {8: ([3], [0.1])}, "joy": {8: ([4], [0.1])}}


def always(svc, text, history=()):
    messages = []
    for earlier in history:
        messages += [{"role": "user", "content": earlier}, {"role": "assistant", "content": "ok"}]
    turn_id, plan = svc.begin_turn(messages + [{"role": "user", "content": text}], steering_mode="always")
    return plan, svc.respond(turn_id, 64)


def steering_service(love_plain=0.05, love_steered=0.40, **kwargs):
    svc, runtime = service(love_plain, love_steered, **kwargs)
    runtime._emotion_panels = PANELS
    return svc, runtime


class StateStrengthTests(unittest.TestCase):
    def setUp(self):
        self.agent = EmotionReasonerAgent()
        self.flat = {label: 1 / len(EMOTIONS) for label in EMOTIONS}

    def level(self, text, scores=None, history=()):
        plan = self.agent.plan(text, scores or self.flat, history).to_dict()
        return self.agent.state_strength(plan, text)

    def test_stronger_emotion_gets_a_higher_level(self):
        mild = self.level("I feel a bit sad today", {**self.flat, "sadness": 0.3})
        strong = self.level("I feel really really sad, so much, I can't stop crying!!", {**self.flat, "sadness": 0.9})
        self.assertLess(mild[1], strong[1])

    def test_persistent_distress_raises_and_a_lifting_mood_lowers_the_level(self):
        text, sad = "I feel really sad today", {**self.flat, "sadness": 0.8}
        alone = self.level(text, sad)[1]
        persisting = self.level(text, sad, [{"emotion": "sadness", "valence": -0.8}, {"emotion": "fear", "valence": -0.7}])
        self.assertGreater(persisting[1], alone)
        self.assertIn("history", [factor["factor"] for factor in persisting[2]])
        lifted = self.level("I feel much better and happy now", {**self.flat, "joy": 0.8}, [{"emotion": "sadness", "valence": -0.8}])
        self.assertIn("-0.10", [factor["effect"] for factor in lifted[2]])

    def test_hinglish_is_capped_lower(self):
        english = self.level("I feel so alone, nobody talks to me", {**self.flat, "sadness": 0.8})
        hinglish = self.level(HINGLISH, {**self.flat, "sadness": 0.8})
        self.assertLess(hinglish[1], english[1])
        self.assertIn("fluency limit", [factor["factor"] for factor in hinglish[2]])

    def test_crisis_is_steered_toward_warmth(self):
        response, level, factors = self.level(CRISIS)
        self.assertEqual(response, CRISIS_WARMTH[0])
        self.assertGreater(level, 0.7)
        self.assertEqual(factors[0]["factor"], "crisis")


class AlwaysModeTests(unittest.TestCase):
    def test_contexts_the_policy_does_not_steer_are_steered(self):
        svc, runtime = steering_service()
        plan, analysis = always(svc, HINGLISH)              # the policy cell for Hinglish has alpha 0
        low, high = STATE_ALPHA_RANGE["default"]
        self.assertEqual(plan["policy_alpha"], 0.0)
        self.assertTrue(low <= plan["steering_alpha"] <= high)
        self.assertEqual(analysis["decision"]["status"], "applied")
        self.assertEqual(analysis["decision"]["mode"], "always")
        self.assertEqual(analysis["reply"], STEERED)
        self.assertEqual(plan["strength_factors"][0]["factor"], "mode")

    def test_the_steered_reply_is_sent_even_without_a_classifier_gain(self):
        svc, _ = steering_service(love_plain=0.45, love_steered=0.40)   # policy mode would skip or reject this
        _, analysis = always(svc, SAD)
        self.assertEqual((analysis["decision"]["status"], analysis["reply"]), ("applied", STEERED))

    def test_a_broken_steered_reply_falls_back_to_the_unsteered_one(self):
        svc, _ = steering_service(steered_text="steered so so so so so so so so so so so much")
        _, analysis = always(svc, SAD)
        self.assertEqual(analysis["decision"]["status"], "rejected")
        self.assertIn("more repetitive", analysis["decision"]["reason"])
        self.assertEqual(analysis["reply"], PLAIN)

    def test_crisis_turn_is_steered_with_the_supportive_strategy(self):
        svc, runtime = steering_service()
        plan, analysis = always(svc, CRISIS)
        self.assertTrue(plan["safety_flag"])
        self.assertEqual((plan["strategy"], plan["response_emotion"], plan["steering_panel"]), ("crisis_support", "love", "love"))
        self.assertEqual(analysis["decision"]["status"], "applied")
        self.assertIn("encourage them not to act on these thoughts", runtime.prompts[-1])

    def test_alpha_follows_the_emotional_state(self):
        svc, _ = steering_service()
        mild, _ = always(svc, "I feel a bit low today")
        strong, _ = always(svc, "Everyone forgot about me today and it really really hurts so much, I can't stop crying!!")
        self.assertLess(mild["steering_alpha"], strong["steering_alpha"])

    def test_model_without_a_matching_panel_is_not_steered_and_says_why(self):
        svc, runtime = service(0.05, 0.40)
        runtime._emotion_panels = {"joy": {8: ([1], [0.1])}}
        plan, analysis = always(svc, SAD)
        self.assertEqual(plan["steering_alpha"], 0.0)
        self.assertIn("no neuron panel", plan["strength_factors"][0]["reason"])

    def test_tone_panel_models_use_their_strategy_panel_and_range(self):
        class SarvamLike(Runtime):
            strategy_panels = {"compassionate_comfort": "warmth", "warm_connection": "warmth"}
        runtime = SarvamLike()
        runtime._emotion_panels = {"warmth": {0: ([1], [0.1])}}
        runtime.results_tag = "sarvam_30b"
        sarvam_policy = {"names_panels": True, "steering_enabled": False, "cells": {}}
        svc = EmotionSteeringService(runtime, EmotionReasonerAgent(list(runtime._emotion_panels), policy=sarvam_policy),
                                     Scorer(0.05, 0.40), evidence={})
        for text in (SAD, CRISIS):
            plan, analysis = always(svc, text)
            low, high = STATE_ALPHA_RANGE["sarvam_30b"]
            self.assertEqual(plan["steering_panel"], "warmth")
            self.assertTrue(low <= plan["steering_alpha"] <= high)
            self.assertEqual(analysis["decision"]["status"], "applied")

    def test_policy_mode_is_unchanged(self):
        svc, _ = steering_service()
        turn_id, plan = svc.begin_turn([{"role": "user", "content": HINGLISH}])
        self.assertEqual((plan["steering_mode"], plan["steering_alpha"]), ("policy", 0.0))


class ToneRecordTests(unittest.TestCase):
    def test_before_and_after_tone_are_recorded_for_every_steered_reply(self):
        svc, runtime = steering_service()
        result = TurnEngine(SessionStore()).run(
            runtime=runtime, service=svc, spec=MODEL_REGISTRY["llama"], owner="o", conversation_key="c",
            messages=[{"role": "user", "content": HINGLISH}], steering_mode="always", control=GenerationControl(),
            emit=lambda *_: None)
        tone = result["tone"]
        self.assertEqual((tone["target"], tone["sent"]), ("love", "steered"))
        self.assertEqual((tone["before"]["target_score"], tone["after"]["target_score"]), (0.05, 0.40))   # raw scores
        self.assertEqual(tone["change"]["target_score"], round(tone["after"]["target_score"] - tone["before"]["target_score"], 4))
        self.assertEqual(len(tone["before"]["top"]), 3)
        self.assertIn("expected_valence", tone["after"])
        self.assertEqual(result["steering"]["factors"][0]["factor"], "mode")

    def test_unsteered_reply_has_before_only(self):
        self.assertIsNone(tone_record({}))
        record = tone_record({"effect": {"target": "love", "unsteered_reply": "plain reply", "steered_reply": None,
                                         "unsteered_scores": {label: 0.1 for label in EMOTIONS}, "sent": "unsteered"}})
        self.assertIsNotNone(record["before"])
        self.assertIsNone(record["after"])
        self.assertIsNone(record["change"])


if __name__ == "__main__":
    unittest.main()
