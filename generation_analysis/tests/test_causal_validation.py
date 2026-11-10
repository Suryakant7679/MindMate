"""Tests for the R∩G / R\\G / G\\R causal-validation experiment (CPU, tiny random Llama)."""
import unittest

import numpy as np
import pandas as pd
import torch

from generation_analysis.causal_validation import (InterventionController, build_groups, hypothesis_table,
                                                   parse_scopes, replacement, token_change_rate)
from generation_analysis.common import greedy_decode, model_layers
from generation_analysis.tests.test_generation_analysis import BOS, EOS, PAD, SPECIAL, tiny_model


class GroupTests(unittest.TestCase):
    def setUp(self):
        self.r = {"0": [0, 1, 2, 3, 4], "1": [10, 11, 12]}
        self.g = {"0": [3, 4, 5, 6, 7], "1": [20, 21, 22]}

    def test_matched_groups(self):
        groups = build_groups(self.r, self.g, 40, "matched", np.random.default_rng(0))
        self.assertEqual(groups["shared"], {0: [3, 4], 1: []})
        for name in ("recognition_only", "generation_only", "random"):
            self.assertEqual([len(groups[name][layer]) for layer in (0, 1)], [2, 0])
        self.assertTrue(set(groups["recognition_only"][0]) <= {0, 1, 2})
        self.assertTrue(set(groups["generation_only"][0]) <= {5, 6, 7})
        self.assertFalse(set(groups["random"][0]) & set(range(8)))       # outside R ∪ G

    def test_full_groups_have_own_random_controls(self):
        groups = build_groups(self.r, self.g, 40, "full", np.random.default_rng(0))
        self.assertEqual(groups["recognition_only"][0], [0, 1, 2])
        self.assertEqual(len(groups["random_recognition_only"][0]), 3)
        self.assertEqual(len(groups["random_generation_only"][1]), 3)
        self.assertFalse(set(groups["random_generation_only"][1]) & {10, 11, 12, 20, 21, 22})

    def test_scopes(self):
        scopes = parse_scopes(["all", "bands"], ["0-1", "2-3"], 4)
        self.assertEqual(scopes, {"all": [0, 1, 2, 3], "layers_01-02": [0, 1], "layers_03-04": [2, 3]})

    def test_token_change_rate(self):
        self.assertEqual(token_change_rate([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertAlmostEqual(token_change_rate([1, 2, 9, 9], [1, 2, 3]), 0.5)


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.layers = model_layers(self.model)
        self.controller = InterventionController(self.layers)
        self.seen = {}
        self.probe = self.layers[1].mlp.down_proj.register_forward_hook(
            lambda _m, inputs, _o: self.seen.setdefault("x", []).append(inputs[0].detach().clone()))

    def tearDown(self):
        self.probe.remove()
        self.controller.close()

    def decode(self):
        self.seen.clear()
        return greedy_decode(self.model, self.controller, [[BOS, 5, 6, 7]], 6, pad_id=PAD, stop_ids={EOS},
                             special_ids=SPECIAL)[0]

    def test_mean_replacement_only_on_decode_steps(self):
        reference = np.full((3, 48), 0.25, dtype=np.float32)
        self.controller.configure(decode={"replace": replacement({1: [2, 7]}, [0, 1, 2], reference)})
        self.decode()
        prefill, decode = self.seen["x"][0], self.seen["x"][1:]
        self.assertFalse(torch.allclose(prefill[0, :, [2, 7]], torch.full((4, 2), 0.25)))
        for step in decode:
            self.assertTrue(torch.allclose(step[..., [2, 7]], torch.tensor(0.25)))
        self.assertGreater(self.controller.realised_norm(), 0)

    def test_scope_excludes_other_layers_and_no_op_matches_baseline(self):
        base = self.decode()
        self.controller.configure(decode={"replace": replacement({1: [2, 7]}, [0], np.zeros((3, 48), np.float32))})
        self.assertEqual(self.decode().generated_ids, base.generated_ids)
        self.assertEqual(self.controller.realised_norm(), 0.0)

    def test_full_shift_norm(self):
        shift = {layer: np.full(48, 0.5, np.float32) for layer in range(3)}
        self.controller.configure(decode={"shift": shift})
        self.decode()
        self.assertAlmostEqual(self.controller.realised_norm(), float(np.sqrt(3 * 48 * 0.25)), places=4)

    def test_phase_specific_references(self):
        self.controller.configure(prefill={"replace": replacement({1: [3]}, [1], np.full((3, 48), -1.0, np.float32))},
                                  decode={"replace": replacement({1: [3]}, [1], np.full((3, 48), 2.0, np.float32))})
        self.decode()
        self.assertTrue(torch.allclose(self.seen["x"][0][0, 1:, 3], torch.tensor(-1.0)))   # prompt after BOS
        self.assertNotAlmostEqual(float(self.seen["x"][0][0, 0, 3]), -1.0)                 # BOS untouched
        for step in self.seen["x"][1:]:
            self.assertTrue(torch.allclose(step[..., 3], torch.tensor(2.0)))


class HypothesisTests(unittest.TestCase):
    def test_recognition_only_supported(self):
        rows = []
        rng = np.random.default_rng(0)
        for task, effects in (("recognition", {"recognition_only": 0.3, "generation_only": 0.0, "shared": 0.3}),
                              ("generation", {"recognition_only": 0.0, "generation_only": 0.3, "shared": 0.3})):
            for group, size in {**effects, "random": 0.0}.items():
                for item in range(20):
                    value = size + rng.normal(0, 0.01)
                    rows.append({"task": task, "emotion": "joy", "group": group, "scope": "all", "item": item,
                                 "effect": value, "normalized": value / 0.5})
        table = hypothesis_table(pd.DataFrame(rows), "matched").set_index("group")
        self.assertEqual(table.at["recognition_only", "supported"], "yes")
        self.assertEqual(table.at["generation_only", "supported"], "yes")
        self.assertEqual(table.at["shared", "supported"], "yes")


if __name__ == "__main__":
    unittest.main()
