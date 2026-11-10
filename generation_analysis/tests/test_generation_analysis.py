"""Unit tests for the generation-time neuron analysis (CPU, tiny random Llama).

Run from the project root:  python3 -m unittest discover -s generation_analysis/tests -v
"""
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from emotion_neuron_extraction import select_top_neurons
from generation_analysis.common import (MLPController, emission_slot, expected_random_jaccard, greedy_decode,
                                        jaccard, model_layers, pack_bits, selectivity_scores, steering_tensors,
                                        top_neurons, verify_alignment)
from generation_analysis.identify_neurons import categorize_tokens, group_words, token_class_counts

BOS, EOS, PAD = 1, 2, 0
SPECIAL = {BOS, EOS}


def tiny_model(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    return LlamaForCausalLM(config).eval()


class DecodeAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.layers = model_layers(self.model)
        self.controller = MLPController(self.layers)
        self.prompts = [[BOS, 5, 6, 7, 8, 9, 10], [BOS, 11, 12], [BOS, 13, 14, 15, 16]]

    def tearDown(self):
        self.controller.close()

    def decode(self, prompts, **kwargs):
        return greedy_decode(self.model, self.controller, prompts, 10, pad_id=PAD, stop_ids={EOS},
                             special_ids=SPECIAL, **kwargs)

    def test_decode_activations_match_teacher_forcing(self):
        results = self.decode(self.prompts, record=True, keep_raw=True)
        raw = self.controller.batch.raw_decode
        for row, result in enumerate(results):
            check = verify_alignment(self.model, self.layers, result,
                                     [torch.stack([step[row] for step in raw[layer]]) for layer in range(3)],
                                     SPECIAL)
            self.assertLess(check["max_abs_difference"], 1e-5)
            self.assertEqual(check["firing_agreement"], 1.0)
            self.assertEqual(check["greedy_token_agreement"], 1.0)

    def test_left_padded_batch_matches_single_rows(self):
        batched = self.decode(self.prompts, record=True)
        counts = [c.clone() for c in self.controller.batch.decode_counts]
        bits = [b.clone() for b in self.controller.batch.decode_bits]
        prefill = [c.clone() for c in self.controller.batch.prefill_counts]
        for row, prompt in enumerate(self.prompts):
            single = self.decode([prompt], record=True)[0]
            self.assertEqual(single.generated_ids, batched[row].generated_ids)
            for layer in range(3):
                self.assertTrue(torch.equal(self.controller.batch.decode_counts[layer][0], counts[layer][row]))
                self.assertTrue(torch.equal(self.controller.batch.decode_bits[layer][0], bits[layer][row]))
                self.assertTrue(torch.equal(self.controller.batch.prefill_counts[layer][0], prefill[layer][row]))

    def test_prefill_statistics_exclude_special_and_generated_tokens(self):
        prompt = self.prompts[0]
        self.decode([prompt], record=True)
        captured = {}
        handle = self.layers[1].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inputs: captured.setdefault("x", inputs[0][0].detach()))
        with torch.inference_mode():
            self.model(input_ids=torch.tensor([prompt]))
        handle.remove()
        expected = (captured["x"][1:] > 0).sum(0).to(torch.int32)      # position 0 is BOS
        self.assertTrue(torch.equal(self.controller.batch.prefill_counts[1][0], expected))

    def test_decode_counts_equal_sum_of_token_bits(self):
        results = self.decode(self.prompts, record=True)
        record = self.controller.batch
        for layer in range(3):
            unpacked = np.unpackbits(record.decode_bits[layer].numpy(), axis=-1)
            np.testing.assert_array_equal(unpacked.sum(1), record.decode_counts[layer].numpy())
        self.assertTrue(all(r.decode_steps <= r.read_count <= len(r.generated_ids) for r in results))

    def test_steering_changes_decode_but_not_prefill(self):
        base = self.decode(self.prompts[:1], record=True)[0]
        prefill = [c.clone() for c in self.controller.batch.prefill_counts]
        self.controller.steer = steering_tensors({0: (list(range(48)), [5.0] * 48)})
        steered = self.decode(self.prompts[:1], record=True)[0]
        self.controller.steer = {}
        for layer in range(3):
            self.assertTrue(torch.equal(prefill[layer], self.controller.batch.prefill_counts[layer]))
        self.assertEqual(base.generated_ids[0], steered.generated_ids[0])   # y_0 comes from prefill
        self.assertNotEqual(base.generated_ids, steered.generated_ids)


class SelectionTests(unittest.TestCase):
    def test_selectivity_matches_recognition_extraction(self):
        rng = np.random.default_rng(0)
        counts = rng.integers(0, 50, size=(2, 40, 4))
        positions = np.array([60, 70, 80, 90])
        labels = ["a", "b", "c", "d"]
        reference = select_top_neurons([torch.from_numpy(c) for c in counts], torch.from_numpy(positions),
                                       labels, [10.0])["10%"]
        scores = selectivity_scores(counts, positions)
        for e, label in enumerate(labels):
            ids = top_neurons(scores[:, :, e], 10.0)
            for layer in range(2):
                self.assertEqual(ids[layer], reference[f"layer_{layer}"][label])

    def test_pack_bits_matches_numpy(self):
        fired = torch.rand(3, 5, 48) > 0.5
        np.testing.assert_array_equal(pack_bits(fired).numpy(), np.packbits(fired.numpy(), axis=-1))

    def test_token_class_counts(self):
        fired = np.random.default_rng(1).random((2, 4, 3, 16)) > 0.5      # [resp, slot, layer, U]
        bits = np.packbits(fired, axis=-1)
        slots = [[(0, 1), (1, 3)], [(1, 0)]]
        counts, positions = token_class_counts(bits, slots, 16)
        np.testing.assert_array_equal(counts[:, :, 0], fired[0, 1].astype(int) + fired[1, 3])
        np.testing.assert_array_equal(counts[:, :, 1], fired[1, 0].astype(int))
        self.assertEqual(positions.tolist(), [2, 1])

    def test_jaccard_and_random_expectation(self):
        self.assertAlmostEqual(jaccard({1, 2, 3}, {2, 3, 4}), 0.5)
        self.assertEqual(jaccard(set(), set()), 0.0)
        a, b, width = 143, 143, 14336
        rng = np.random.default_rng(0)
        simulated = np.mean([jaccard(set(rng.choice(width, a, replace=False)), set(rng.choice(width, b, replace=False)))
                             for _ in range(300)])
        self.assertAlmostEqual(simulated, expected_random_jaccard(a, b, width), delta=0.002)


class TokenCategoryTests(unittest.TestCase):
    def test_slot_alignment(self):
        self.assertEqual(emission_slot(0, "read"), 0)
        self.assertIsNone(emission_slot(0, "emit"))
        self.assertEqual(emission_slot(3, "emit"), 2)

    def test_word_grouping(self):
        pieces = ["I", " am", " so", " happ", "ily", " here", "!", " 😊", "<eot>"]
        special = [False] * 8 + [True]
        self.assertEqual(group_words(pieces, special), [0, 1, 2, 3, 3, 4, -1, -1, -1])

    def test_categories(self):
        pieces = [" The", " project", " deadline", " makes", " me", " so", " happy", "!"]
        special = [False] * 8
        deltas = [0.0, 0.1, -0.2, 0.0, 0.1, 1.5, 2.0, 0.3]
        categories, words = categorize_tokens(pieces, special, deltas, {"project", "deadline"},
                                              emotional_delta=1.0, semantic_delta=0.5, criterion="either")
        self.assertEqual(categories, ["other", "semantic", "semantic", "other", "other", "emotional",
                                      "emotional", "other"])
        self.assertEqual(words[6], "happy")
        categories, _ = categorize_tokens(pieces, special, deltas, None, emotional_delta=1.0,
                                          semantic_delta=0.5, criterion="lexicon")
        self.assertEqual(categories[5], "other")            # 'so' is not a lexicon word
        self.assertEqual(categories[3], "semantic")         # 'makes' is a content word without the shared filter


if __name__ == "__main__":
    unittest.main()
