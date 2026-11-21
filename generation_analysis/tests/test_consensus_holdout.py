"""Consensus rule, v2 freeze and held-out leakage checks (no model is loaded)."""
import json
import unittest
from pathlib import Path

import pandas as pd

from generation_analysis import consensus_policy, freeze_v2

SHARED = Path("results/steering_holdout2")


def row(alpha, safe, effective, gain=0.3):
    return {"alpha": alpha, "n": 20, "emotion_rating_gain": gain, "independent_classifier_gain": 0.01,
            "safe": safe, "effective": effective}


class ConsensusRuleTests(unittest.TestCase):
    def test_strength_needs_both_judges_and_a_safe_next_strength(self):
        first = [row(0.25, True, True, 0.2), row(0.5, True, True, 0.4), row(0.75, True, True, 0.6), row(1.0, False, True)]
        second = [row(0.25, True, True, 0.2), row(0.5, True, True, 0.5), row(0.75, False, True, 0.7), row(1.0, False, True)]
        alpha, reason, rows = consensus_policy.combine(first, second)
        self.assertEqual(alpha, 0.25)                       # 0.5 is safe for both, but 0.75 is not safe for judge 2
        self.assertEqual([entry["admissible"] for entry in rows], [True, False, False, False])
        self.assertIn("both judges", reason)

    def test_one_judge_alone_never_enables_steering(self):
        first = [row(0.25, True, True), row(0.5, True, True), row(0.75, True, True)]
        second = [row(0.25, True, False), row(0.5, True, False), row(0.75, True, False)]
        self.assertEqual(consensus_policy.combine(first, second)[0], 0.0)
        unsafe = [row(0.25, False, True), row(0.5, False, True), row(0.75, False, True)]
        alpha, reason, _ = consensus_policy.combine(first, unsafe)
        self.assertEqual(alpha, 0.0)
        self.assertIn("judge 2", reason)

    def test_small_cells_are_not_steered(self):
        small = [dict(row(0.25, True, True), n=5), dict(row(0.5, True, True), n=5)]
        self.assertEqual(consensus_policy.combine(small, small)[0], 0.0)


@unittest.skipUnless((freeze_v2.FROZEN / "llama31_8b" / "manifest.json").exists(), "v2 has not been frozen")
class FreezeTests(unittest.TestCase):
    def test_deployed_configuration_matches_the_freeze(self):
        for key in freeze_v2.frozen_models():
            manifest = freeze_v2.verify(key)                # raises SystemExit on any difference
            self.assertEqual(set(manifest["deployed"]), {"panels", "policy", "evidence"})

    def test_consensus_policy_is_not_installed_and_never_adds_steering(self):
        for key in freeze_v2.frozen_models():
            tag = freeze_v2.MODEL_REGISTRY[key].config.results_tag
            path = Path("results") / f"{tag}_generation_emotion" / "causal_steering" / "consensus" / "policy_consensus.json"
            if not path.exists():
                self.skipTest("consensus policy not built")
            consensus, deployed = json.loads(path.read_text()), freeze_v2.frozen_policy(key)
            self.assertTrue(consensus["experimental"])
            self.assertEqual(consensus["panel"], deployed["panel"])          # the model's own panel
            self.assertEqual(consensus["gate"], deployed["gate"])
            for name, cell in consensus["cells"].items():
                self.assertLessEqual(cell["alpha"], deployed["cells"][name]["alpha"], name)


@unittest.skipUnless((SHARED / "items.csv").exists(), "holdout2 has not been built")
class LeakageTests(unittest.TestCase):
    def test_new_messages_are_in_no_earlier_split(self):
        from generation_analysis.holdout2 import load_items
        items = pd.read_csv(SHARED / "items.csv")
        # Models frozen when the split was built: none of the shared messages is in their splits.
        built_with = [key for key in ("llama", "emollama") if key in freeze_v2.frozen_models()]
        earlier = pd.concat([pd.read_csv(path) for key in built_with
                             for path in (freeze_v2.frozen_dir(key) / "causal_steering" / "splits").glob("*.csv")])
        self.assertFalse(set(items.conv_id) & set(earlier.conv_id))
        self.assertFalse(set(items.text.str.lower()) & set(earlier.text.str.lower()))
        # Every model (including ones added later) evaluates only messages outside its own splits.
        for key in freeze_v2.frozen_models():
            own = pd.concat([pd.read_csv(path) for path in (freeze_v2.frozen_dir(key) / "causal_steering" / "splits").glob("*.csv")])
            evaluated = load_items(key)
            self.assertFalse(set(evaluated.conv_id) & set(own.conv_id), key)
            self.assertFalse(set(evaluated.text.str.lower()) & set(own.text.str.lower()), key)
        self.assertFalse(items.text.str.lower().duplicated().any())
        self.assertFalse((items.strategy == "crisis_support").any())


if __name__ == "__main__":
    unittest.main()
