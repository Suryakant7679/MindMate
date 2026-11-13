"""Version-3 study pieces that need no model: tone prompt, fallback, gates, policy lookup, hook arithmetic."""
import types
import unittest

import torch

from backend.config import SYSTEM_PROMPT
from chatbot_feature.emotion_reasoner.knowledge_graph import STRATEGIES
from chatbot_feature.emotion_reasoner.tone import HINGLISH, STYLE, TONES, degenerate_reply, response_tone, strategy_prompt_v3
from generation_analysis import holdout3
from generation_analysis.steering_methods import Interventions, spec_name

HINGLISH_TEXT = "Yaar aaj mera mood bahut kharab hai, kuch accha nahi lag raha hai"


def plan(strategy="compassionate_comfort", intensity=0.5, valence=-0.6):
    title, _, guidance, emotion, _ = STRATEGIES[strategy]
    return {"strategy": strategy, "strategy_title": title, "guidance": guidance, "intensity": intensity,
            "valence": valence, "response_emotion": emotion}


class TonePromptTests(unittest.TestCase):
    def test_every_strategy_has_its_own_tone_and_none_is_called_love(self):
        self.assertEqual(set(TONES), set(STRATEGIES))
        labels = [label for label, _ in TONES.values()]
        self.assertEqual(len(set(labels)), len(labels))
        self.assertFalse(any("love" in label for label in labels))

    def test_ablation_switches(self):
        base = strategy_prompt_v3(SYSTEM_PROMPT, plan(), "I feel low", tone=False, style=False)
        self.assertNotIn("Tone:", base)
        self.assertNotIn(STYLE, base)
        self.assertTrue(base.startswith(SYSTEM_PROMPT))                       # version-2 prompt
        full = strategy_prompt_v3(SYSTEM_PROMPT, plan(), "I feel low")
        self.assertIn("Tone:", full)
        self.assertIn(STYLE, full)
        self.assertNotIn(HINGLISH, full)                                      # English message
        self.assertIn(HINGLISH, strategy_prompt_v3(SYSTEM_PROMPT, plan(), HINGLISH_TEXT))
        only_language = strategy_prompt_v3(SYSTEM_PROMPT, plan(), HINGLISH_TEXT, tone=False, style=False, language=True)
        self.assertIn(HINGLISH, only_language)
        self.assertNotIn(STYLE, only_language)

    def test_tone_follows_intensity_and_crisis_gets_no_style_rules(self):
        self.assertIn("short and steady", response_tone(plan(intensity=0.9))["text"])
        self.assertIn("light", response_tone(plan(intensity=0.2))["text"])
        crisis = strategy_prompt_v3(SYSTEM_PROMPT, plan("crisis_support"), "help")
        self.assertNotIn(STYLE, crisis)

    def test_degenerate_replies(self):
        for reply in ("0.896", "", "frustration", "Oh,"):
            self.assertTrue(degenerate_reply(reply), reply)
        self.assertFalse(degenerate_reply("That sounds really hard. What happened today?"))
        self.assertFalse(degenerate_reply("Yaar yeh sach mein mushkil hai, kya hua aaj?"))


class GateAndPolicyTests(unittest.TestCase):
    def test_gates(self):
        plain = "That sounds hard. I am here with you. What happened today at work?"
        warm = "That sounds really hard, and I am here with you. What happened today at work?"
        self.assertTrue(holdout3.v2_gate(plain, warm, 0.1))
        self.assertFalse(holdout3.v2_gate(plain, warm, 0.5))                       # reply already expresses the emotion
        self.assertFalse(holdout3.v2_gate(plain, "Yes.", 0.1))                     # much shorter
        self.assertFalse(holdout3.v2_gate(plain, "so sorry so sorry so sorry so sorry so sorry so sorry", 0.1))
        self.assertTrue(holdout3.quality_gate(plain, warm, 0.1))
        self.assertFalse(holdout3.quality_gate(plain, warm + " " + warm + " " + warm, 0.1))   # far longer
        hinglish = "Yeh sach mein mushkil hai yaar, main tumhare saath hoon. Kya hua aaj?"
        self.assertFalse(holdout3.quality_gate(hinglish, warm, 0.1))               # lost the user's language

    def test_most_specific_cell_wins_and_missing_policy_means_no_steering(self):
        policy = {"cells": {"strategy:grounding_reassurance|english": {"alpha": 0.5, "method": "additive"},
                            "strategy:compassionate_comfort|english": {"alpha": 0.0},
                            "love|english": {"alpha": 1.0, "method": "selective_L22-31"}}}
        row = types.SimpleNamespace(text="I am worried about tomorrow", strategy="grounding_reassurance", target="love")
        self.assertEqual(holdout3.cell_for(policy, row)["alpha"], 0.5)
        row.strategy = "compassionate_comfort"
        self.assertEqual(holdout3.cell_for(policy, row)["alpha"], 0.0)             # its own cell says no
        row.strategy = "warm_connection"
        self.assertEqual(holdout3.cell_for(policy, row)["alpha"], 1.0)             # falls back to the emotion cell
        row.text = HINGLISH_TEXT
        self.assertEqual(holdout3.cell_for(policy, row)["alpha"], 0.0)             # no Hinglish cell
        self.assertEqual(holdout3.cell_for(None, row)["alpha"], 0.0)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.randn(3, 1, 16)

    def test_additive_hook_shifts_only_the_steered_rows_after_the_prompt(self):
        hook = Interventions._panel_hook([1, 4], [0.5, -0.25], [0.0, 1.0, 2.0], None)
        self.assertIs(hook(None, (self.hidden,))[0], self.hidden)                 # prompt pass untouched
        out = hook(None, (self.hidden,))[0]
        self.assertTrue(torch.equal(out[0], self.hidden[0]))
        self.assertTrue(torch.allclose(out[1, 0, 1] - self.hidden[1, 0, 1], torch.tensor(0.5 * 1.75)))
        self.assertTrue(torch.allclose(out[2, 0, 4] - self.hidden[2, 0, 4], torch.tensor(-0.25 * 1.75 * 2)))

    def test_bounded_hook_keeps_the_norm_and_caps_the_shift(self):
        hook = Interventions._panel_hook([1, 4], [50.0, -50.0], [0.0, 1.0, 3.0], (float(self.hidden.norm(dim=-1).mean()), 0.1))
        hook(None, (self.hidden,))
        out = hook(None, (self.hidden,))[0]
        self.assertTrue(torch.equal(out[0], self.hidden[0]))
        self.assertTrue(torch.allclose(out.norm(dim=-1), self.hidden.norm(dim=-1), atol=1e-4))
        change = (out - self.hidden).norm(dim=-1) / self.hidden.norm(dim=-1)
        self.assertTrue((change[1:] <= 0.1 + 1e-3).all())                         # never more than the cap
        same = self.hidden[:1].repeat(3, 1, 1)                                    # identical rows: only strength differs
        hook = Interventions._panel_hook([1, 4], [50.0, -50.0], [0.0, 1.0, 3.0], (float(same.norm(dim=-1).mean()), 0.1))
        hook(None, (same,))
        out = hook(None, (same,))[0]
        self.assertTrue(torch.allclose(out[1], out[2], atol=1e-4))                 # both capped: strength no longer matters
        self.assertTrue(torch.equal(out[0], same[0]))

    def test_residual_hook_rotates_towards_the_direction_at_constant_norm(self):
        unit = torch.zeros(16)
        unit[0] = 1.0
        hook = Interventions._residual_hook(unit.numpy(), [0.0, 0.5, 0.5])
        hook(None, None, (self.hidden,))
        out = hook(None, None, (self.hidden,))[0]
        self.assertTrue(torch.equal(out[0], self.hidden[0]))
        self.assertTrue(torch.allclose(out.norm(dim=-1), self.hidden.norm(dim=-1), atol=1e-4))
        self.assertTrue((out[1:, 0, 0] > self.hidden[1:, 0, 0]).all())

    def test_spec_names(self):
        self.assertEqual(spec_name({"method": "selective", "layers": [22, 23, 31]}), "selective_L22-31")
        self.assertEqual(spec_name({"method": "residual", "layers": [12], "reference": "neutral"}), "residual_neutral_L12")


if __name__ == "__main__":
    unittest.main()
