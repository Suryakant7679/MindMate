"""Sarvam-30B steering integration without loading the model: hook sites, reasoning switch,
policy-named panels, no manual control, optional Hinglish safeguards, panel arithmetic."""
import types
import unittest

import numpy as np
import torch

from backend.emotion_service import EmotionSteeringService
from backend.runtime import HuggingFaceRuntime
from backend.sarvam import SarvamRuntime
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
from generation_analysis import sarvam_experiments as experiments
from generation_analysis.sarvam_lab import STRATEGY_TONE, TONES, generate_rows, style_prompt
from tests.test_emotion_service import HINGLISH, PLAIN, SAD, STEERED, Runtime, Scorer, run

SWEEP = [{"alpha": 2.0, "n": 20, "target_gain": 0.1, "safe": True}]


def policy(alpha_english=2.0, alpha_hinglish=2.0, gate=None):
    cells = {"strategy:compassionate_comfort|english": {"alpha": alpha_english, "panel": "warmth", "reason": "ok", "sweep": SWEEP},
             "strategy:compassionate_comfort|hinglish": {"alpha": alpha_hinglish, "panel": "warmth+hinglish", "reason": "ok", "sweep": SWEEP},
             "love|english": {"alpha": 0.0, "panel": None, "reason": "not steered", "sweep": []},
             "love|hinglish": {"alpha": 0.0, "panel": None, "reason": "not steered", "sweep": []}}
    return {"rule": {}, "cells": cells, "gate": {"rule": "ceiling_and_safeguards", "ceiling": 0.3, **(gate or {})}}


class SarvamPanels(Runtime):
    _emotion_panels = {"warmth": {}, "warmth+hinglish": {}, "hinglish": {}}


def service(runtime, policy_, love_plain=0.05, love_steered=0.4):
    agent = EmotionReasonerAgent(list(runtime._emotion_panels), policy=policy_)
    return EmotionSteeringService(runtime, agent, Scorer(love_plain, love_steered), evidence={})


class SiteTests(unittest.TestCase):
    def runtime(self):
        dense = types.SimpleNamespace(down_proj="dense-down")
        moe = types.SimpleNamespace(down_proj=None, shared_experts=types.SimpleNamespace(down_proj="shared-down"), experts=[])
        runtime = SarvamRuntime.__new__(SarvamRuntime)
        runtime.model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=dense),
                                                                                 types.SimpleNamespace(mlp=moe)]))
        return runtime

    def test_sarvam_steers_the_dense_mlp_of_layer_0_and_the_shared_expert_elsewhere(self):
        runtime = self.runtime()
        self.assertEqual(runtime.steering_module(0), "dense-down")
        self.assertEqual(runtime.steering_module(1), "shared-down")
        self.assertIn("shared expert", runtime.steering_location)
        self.assertNotEqual(SarvamRuntime.steering_location, HuggingFaceRuntime.steering_location)   # Llama's site is untouched

    def test_reasoning_off_prefills_an_empty_reasoning_block(self):
        runtime = self.runtime()
        runtime.set_reasoning(False)
        self.assertEqual(runtime.prompt_suffix(), "<think>\n</think>\n")
        runtime.set_reasoning(True)
        self.assertEqual(runtime.prompt_suffix(), "")
        self.assertEqual(HuggingFaceRuntime.prompt_suffix(runtime), "")


class PolicyPanelTests(unittest.TestCase):
    def test_policy_names_the_panel_and_the_language_variant(self):
        runtime = SarvamPanels()
        svc = service(runtime, policy())
        plan, analysis = run(svc, SAD)
        self.assertEqual((plan["strategy"], plan["response_emotion"], plan["steering_panel"]),
                         ("compassionate_comfort", "love", "warmth"))            # "love" needs no panel of its own name
        self.assertEqual(analysis["steering"]["emotion"], "warmth")              # the hook used the tone panel
        self.assertEqual(analysis["decision"]["applied_alpha"], 2.0)
        plan, analysis = run(svc, "yaar aaj bahut dukh ho raha hai, meri dadi guzar gayi aur main bahut akela hoon")
        self.assertEqual(plan["steering_panel"], "warmth+hinglish")

    def test_unsteered_contexts_and_missing_panels_are_not_steered(self):
        svc = service(SarvamPanels(), policy(alpha_english=0.0))
        plan, analysis = run(svc, SAD)
        self.assertIsNone(plan["steering_panel"])
        self.assertEqual(analysis["decision"]["status"], "not_planned")

        class OnlyLanguage(Runtime):
            _emotion_panels = {"hinglish": {}}
        plan, analysis = run(service(OnlyLanguage(), policy()), SAD)
        self.assertEqual((plan["steering_alpha"], analysis["decision"]["status"]), (0.0, "not_planned"))

    def test_no_manual_strength_or_panel_control(self):
        runtime = SarvamPanels()
        plan, analysis = run(service(runtime, policy()), SAD, alpha_override=9.0, emotion_override="hinglish")
        self.assertEqual((plan["steering_alpha"], plan["steering_panel"]), (2.0, "warmth"))
        self.assertEqual(runtime.calls, [[0.0, 2.0]])

    def test_optional_safeguards_reject_english_and_overlong_replies(self):
        hinglish_plain = "plain yaar main samajh sakta hoon ki tum bahut akela feel kar rahe ho aaj"
        runtime = SarvamPanels(steered_text=STEERED)                              # the steered candidate is English

        class HinglishPlain(SarvamPanels):
            def generate_candidates(self, *args, **kwargs):
                rows = super().generate_candidates(*args, **kwargs)
                rows[0]["response"] = hinglish_plain
                return rows
        message = "yaar aaj bahut dukh ho raha hai, meri dadi guzar gayi aur main bahut akela hoon"
        _, analysis = run(service(HinglishPlain(), policy(gate={"keep_hinglish": True})), message)
        self.assertEqual(analysis["decision"]["status"], "rejected")
        self.assertIn("Hinglish", analysis["decision"]["reason"])
        self.assertEqual(analysis["reply"], hinglish_plain)
        _, analysis = run(service(HinglishPlain(), policy()), message)            # safeguard off by default
        self.assertEqual(analysis["decision"]["status"], "applied")
        long_reply = "steered " + "very long reply " * 30
        _, analysis = run(service(SarvamPanels(steered_text=long_reply), policy(gate={"max_length_ratio": 1.5})), SAD)
        self.assertEqual(analysis["decision"]["status"], "rejected")


class LanguageRepairTests(unittest.TestCase):
    MESSAGE = "yaar aaj bahut dukh ho raha hai, meri dadi guzar gayi aur main bahut akela hoon"
    HINGLISH_REPLY = "steered yaar main samajh sakta hoon ki tum bahut akela feel kar rahe ho aaj"

    def policy(self):
        value = policy(gate={"language_panel": "hinglish"})
        value["cells"]["strategy:compassionate_comfort|hinglish"] = {"alpha": 0.75, "panel": "hinglish", "reason": "ok",
                                                                    "sweep": SWEEP}
        return value

    def test_language_panel_brings_an_english_reply_back_to_hinglish(self):
        runtime = SarvamPanels(steered_text=self.HINGLISH_REPLY)          # unsteered PLAIN is English
        plan, analysis = run(service(runtime, self.policy(), love_plain=0.9, love_steered=0.9), self.MESSAGE)
        self.assertEqual(plan["steering_panel"], "hinglish")
        self.assertEqual(analysis["decision"]["status"], "applied")       # the emotion ceiling does not apply to it
        self.assertEqual(analysis["reply"], self.HINGLISH_REPLY)

    def test_language_panel_is_skipped_when_the_reply_is_already_hinglish_or_stays_english(self):
        class HinglishPlain(SarvamPanels):
            def generate_candidates(self, *args, **kwargs):
                rows = super().generate_candidates(*args, **kwargs)
                rows[0]["response"] = "plain yaar main sun raha hoon, tum akela feel kar rahe ho aur yeh mushkil hai"
                return rows
        _, analysis = run(service(HinglishPlain(steered_text=self.HINGLISH_REPLY), self.policy()), self.MESSAGE)
        self.assertEqual(analysis["decision"]["status"], "skipped")
        _, analysis = run(service(SarvamPanels(steered_text=STEERED), self.policy()), self.MESSAGE)   # candidate still English
        self.assertEqual(analysis["decision"]["status"], "rejected")
        self.assertEqual(analysis["reply"], PLAIN)


class PanelArithmeticTests(unittest.TestCase):
    def test_tones_cover_every_steerable_strategy_and_prompts_differ_only_in_style(self):
        self.assertTrue(set(STRATEGY_TONE.values()) <= set(TONES))
        self.assertNotIn("crisis_support", STRATEGY_TONE)
        neutral, warm = style_prompt("neutral", "hinglish"), style_prompt("warmth", "hinglish")
        self.assertEqual(neutral.split("Hinglish")[0], warm.split("Hinglish")[0])
        self.assertIn("Hinglish", warm)
        self.assertIn("English only", style_prompt("neutral_english", "hinglish"))

    def test_combined_random_and_band_panels(self):
        tone = {1: ([0, 3], [1.0, -2.0]), 2: ([5], [0.5])}
        language = {1: ([3, 4], [10.0, 1.0])}
        both = experiments.combined(tone, language)
        self.assertEqual(both[1], ([0, 3, 4], [1.0, 8.0, 1.0]))
        self.assertEqual(both[2], ([5], [0.5]))
        random = experiments.random_panel(tone, {1: 64, 2: 64}, np.random.default_rng(0))
        self.assertEqual({layer: len(neurons) for layer, (neurons, _) in random.items()}, {1: 2, 2: 1})
        self.assertEqual(sorted(random[1][1]), [-2.0, 1.0])                       # same shift magnitudes
        self.assertEqual(list(experiments.subset(tone, [2])), [2])

    def test_generate_rows_shifts_and_clamps_decode_steps_only(self):
        """A one-layer stand-in: the hook must leave the prompt pass alone and act per row afterwards."""
        seen = []

        class Site(torch.nn.Module):
            in_features = 4

            def forward(self, hidden):
                seen.append(hidden.clone())
                return hidden

        site = Site()

        class Model:
            generation_config = types.SimpleNamespace(eos_token_id=9)

            def generate(self, input_ids, attention_mask, **_):
                site(torch.zeros(input_ids.shape[0], 3, 4))                       # prompt pass
                site(torch.ones(input_ids.shape[0], 1, 4))                        # one decode step
                return torch.cat([input_ids, torch.full((input_ids.shape[0], 1), 9)], dim=1)

        import threading
        runtime = types.SimpleNamespace(
            model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[None]), generate=Model().generate),
            steering_module=lambda layer: site, lock=threading.Lock(),
            tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=9),
            generation_settings=lambda: {}, decode_response=lambda tokens: "",
            _encode_conversation=lambda messages, prompt, n: {"input_ids": torch.ones(1, 2, dtype=torch.long),
                                                              "attention_mask": torch.ones(1, 2, dtype=torch.long)})
        rows = [{}, {"shift": [({0: ([1], [2.0])}, 1.0)]}, {"shift": [({0: ([1], [2.0])}, -1.0)]},
                {"clamp": {0: ([2], [7.0])}}]
        generate_rows(runtime, [{"role": "user", "content": "hi"}], "system", rows, 4)
        prompt, step = seen
        self.assertTrue(torch.equal(prompt, torch.zeros(4, 3, 4)))                # prompt untouched
        self.assertTrue(torch.equal(step[0], torch.ones(1, 4)))                   # unsteered row
        self.assertAlmostEqual(float(step[1, 0, 1]), 1 + 2.0 * 1.75)              # alpha * gain * delta
        self.assertAlmostEqual(float(step[2, 0, 1]), 1 - 2.0 * 1.75)              # reversed
        self.assertEqual(float(step[3, 0, 2]), 7.0)                               # held at the given value
        self.assertEqual(len(site._forward_pre_hooks), 0)                         # hooks removed, weights untouched


if __name__ == "__main__":
    unittest.main()
