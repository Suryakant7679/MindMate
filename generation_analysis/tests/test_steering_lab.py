"""CPU-only tests of the causal steering experiment helpers (no model is loaded)."""
import argparse
import unittest

import numpy as np
import pandas as pd

from generation_analysis import recalibrate_policy
from generation_analysis.steering_lab import holm, paired_summary, panel_size, random_panel, subset_panel

PANEL = {8: ([5, 3, 9, 1], [0.4, 0.3, -0.2, 0.1]), 9: ([7, 2], [0.5, -0.1])}     # stored in attribution order


class PanelTests(unittest.TestCase):
    def test_subsets_follow_attribution_rank_and_layers(self):
        self.assertEqual(subset_panel(PANEL, fraction=0.5)[8], ([5, 3], [0.4, 0.3]))
        self.assertEqual(subset_panel(PANEL, fraction=0.5, order="bottom")[8], ([9, 1], [-0.2, 0.1]))
        self.assertEqual(list(subset_panel(PANEL, layers=[9])), [9])
        chosen = subset_panel(PANEL, fraction=0.5, order="random", rng=np.random.default_rng(0))
        self.assertEqual(panel_size(chosen), 3)
        for layer, (neurons, shifts) in chosen.items():      # each neuron keeps its own shift
            for neuron, shift in zip(neurons, shifts):
                self.assertEqual(shift, PANEL[layer][1][PANEL[layer][0].index(neuron)])

    def test_random_control_matches_counts_and_shifts_outside_the_panel(self):
        control = random_panel(PANEL, 64, np.random.default_rng(1))
        for layer, (neurons, shifts) in control.items():
            self.assertEqual(len(neurons), len(PANEL[layer][0]))
            self.assertFalse(set(neurons) & set(PANEL[layer][0]))
            self.assertEqual(sorted(shifts), sorted(PANEL[layer][1]))


class StatsTests(unittest.TestCase):
    def test_paired_summary_interval_covers_the_mean(self):
        values = np.random.default_rng(0).normal(0.3, 0.2, 60)
        stats = paired_summary(values)
        self.assertLess(stats["ci_low"], stats["mean"])
        self.assertGreater(stats["ci_high"], stats["mean"])
        self.assertGreater(stats["ci_low"], 0)
        self.assertLess(stats["p"], 0.001)
        self.assertEqual(paired_summary([])["n"], 0)

    def test_holm_is_monotone_and_keeps_nan(self):
        adjusted = holm([0.01, 0.04, float("nan"), 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[3], 0.06)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertTrue(np.isnan(adjusted[2]))


def sweep_cell(emotion_gain, quality_drop, n=12):
    """One cell: per alpha, constant paired changes (plus small noise so the CIs are finite)."""
    rng = np.random.default_rng(0)
    rows = []
    for alpha, gain in emotion_gain.items():
        for item in range(n):
            noise = rng.normal(0, 0.01)
            drop = quality_drop.get(alpha, 0.0)
            rows.append({"alpha": alpha, "item_id": item, "d_judge_emotion": gain + noise, "d_indep_target": gain / 10,
                         "d_gate_target": gain / 5, "preference_vs_reference": 0.5, "harmed": drop > 0.5,
                         "d_empathy": -drop + noise, "d_relevance": noise, "d_fluency": -drop + noise})
    return pd.DataFrame(rows)


class SelectionRuleTests(unittest.TestCase):
    args = argparse.Namespace(tolerance=0.05, ci_margin=0.15, max_harm_share=0.2, min_preference=0.45, min_n=8)

    def select(self, emotion_gain, quality_drop, **kwargs):
        rows = recalibrate_policy.sweep(sweep_cell(emotion_gain, quality_drop, **kwargs), self.args)
        return recalibrate_policy.choose(rows, self.args), rows

    def test_largest_safe_gain_with_a_safe_neighbour_is_selected(self):
        (alpha, reason), rows = self.select({0.25: 0.1, 0.5: 0.3, 0.75: 0.5, 1.0: 0.8}, {1.0: 0.4})
        self.assertEqual(alpha, 0.5)              # 0.75 gains more but sits next to the unsafe 1.0
        self.assertIn("safety margin", reason)
        self.assertEqual([row["admissible"] for row in rows], [True, True, False, False])

    def test_no_steering_without_a_reliable_gain_or_with_quality_loss(self):
        (alpha, reason), _ = self.select({0.25: 0.0, 0.5: 0.0, 0.75: 0.0}, {})
        self.assertEqual(alpha, 0.0)
        self.assertIn("reliably", reason)
        (alpha, reason), _ = self.select({0.25: 0.4, 0.5: 0.6}, {0.25: 0.3, 0.5: 0.6})
        self.assertEqual(alpha, 0.0)
        self.assertIn("quality", reason)
        (alpha, reason), _ = self.select({0.25: 0.4, 0.5: 0.6}, {}, n=4)
        self.assertEqual(alpha, 0.0)
        self.assertIn("only 4", reason)

    def test_gate_rules_differ_only_in_what_they_claim(self):
        rows = pd.DataFrame({"reply": ["a much longer steered reply"] * 3, "unsteered_reply": ["an unsteered reply"] * 3,
                             "d_repeated_bigram_rate": [0.0, 0.2, 0.0], "unsteered_gate": [0.1, 0.1, 0.5],
                             "d_gate_target": [0.01, 0.4, 0.4]})
        sent = recalibrate_policy.gate_variants(rows)
        self.assertEqual(sent["always_steer"].tolist(), [True, True, True])
        self.assertEqual(sent["safeguards_only"].tolist(), [True, False, True])
        self.assertEqual(sent["ceiling_and_safeguards"].tolist(), [True, False, False])
        self.assertEqual(sent["deployed_gate"].tolist(), [False, False, False])


if __name__ == "__main__":
    unittest.main()
