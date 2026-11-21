"""Causal validation battery: matched controls, condition plans, paired statistics and verdicts (no model needed)."""
import unittest

import numpy as np
import pandas as pd

from generation_analysis import neuron_validation as nv


def fake_panel(rng, layers=(0, 1, 2), width=40, per_layer=8):
    panel = {}
    for layer in layers:
        neurons = sorted(rng.choice(width, per_layer, replace=False).tolist())
        panel[layer] = (neurons, [float(v) for v in rng.uniform(0.1, 0.5, per_layer)])
    return panel


class ControlTests(unittest.TestCase):
    def test_matched_random_keeps_count_layer_and_shift_values(self):
        rng = np.random.default_rng(0)
        widths = {0: 40, 1: 40, 2: 40}
        start = nv.site_offsets(widths)
        neutral = rng.normal(size=sum(widths.values())).astype(np.float32)
        panel = fake_panel(rng)
        draw = nv.matched_random(panel, widths, neutral, start, rng)
        self.assertEqual(sorted(draw), sorted(panel))
        for layer, (neurons, deltas) in panel.items():
            random_neurons, random_deltas = draw[layer]
            self.assertEqual(len(random_neurons), len(neurons))
            self.assertTrue(set(random_neurons).isdisjoint(neurons))
            self.assertEqual(sorted(random_deltas), sorted(deltas))
            self.assertEqual(len(set(random_neurons)), len(random_neurons))

    def test_matched_random_follows_neutral_magnitude(self):
        rng = np.random.default_rng(1)
        widths = {0: 400}
        start = {0: 0}
        neutral = np.concatenate([np.linspace(0.01, 1.0, 400)]).astype(np.float32)
        panel = {0: (list(range(0, 400, 40)), [0.2] * 10)}      # spread across the magnitude range
        draw = nv.matched_random(panel, widths, neutral, start, rng)
        panel_mag = np.abs(neutral[panel[0][0]]).mean()
        random_mag = np.abs(neutral[draw[0][0]]).mean()
        self.assertLess(abs(panel_mag - random_mag), 0.15)

    def test_clamp_holds_neutral_values(self):
        neutral = np.arange(10, dtype=np.float32)
        clamp = nv.clamp_to({1: ([2, 5], [0.3, 0.4])}, {1: 4}, neutral)
        self.assertEqual(clamp[1], ([2, 5], [6.0, 9.0]))


class PlanTests(unittest.TestCase):
    def test_plan_has_every_control_with_its_reference(self):
        rng = np.random.default_rng(2)
        widths = {0: 40, 1: 40, 2: 40}
        start = nv.site_offsets(widths)
        neutral = rng.normal(size=sum(widths.values())).astype(np.float32)
        panels = {"love": fake_panel(rng), "joy": fake_panel(rng)}
        scores = {layer: {n: float(abs(v)) for n, v in zip(*panels["love"][layer])} for layer in panels["love"]}
        plan = nv.plan_conditions("llama", "love", panels, widths, start, neutral, scores, rng, "joy")
        names = [item["name"] for item in plan]
        for expected in ("none", "dose_1.5", "reverse_1.5", "sham", "random_0", "random_1", "other_joy", "band_only_early",
                         "subset_top_0.5", "subset_random_0.5", "styled", "clamp", "clamp_zero", "clamp_random", "styled_reverse_1.5"):
            self.assertIn(expected, names)
        by_name = {item["name"]: item for item in plan}
        self.assertEqual(by_name["dose_1.5"]["reference"], "none")
        self.assertEqual(by_name["clamp"]["reference"], "styled")
        self.assertEqual(by_name["clamp"]["prompt"], "B")
        self.assertEqual(by_name["sham"]["spec"], {"sham": True})
        self.assertEqual(by_name["reverse_1.5"]["spec"]["shift"][0][1], -1.5)
        self.assertEqual(by_name["subset_top_0.5"]["neurons"], sum(max(1, round(0.5 * 8)) for _ in panels["love"]))
        for item in plan:                                       # every intervention names something it can be compared with
            if item["name"] not in ("none", "styled"):
                self.assertIsNotNone(item["reference"])


def synthetic_part(effect=0.4, control=0.0, judge_noise=0.05, items=30, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(items):
        base = rng.normal(3, 0.2)
        language = "hinglish" if item % 3 == 0 else "english"
        specs = [("none", "", "none", None, 0, "A", 0), ("dose_1.5", "dose", "none", 1.5, 8, "A", 0),
                 ("dose_0.5", "dose", "none", 0.5, 8, "A", 0), ("dose_1", "dose", "none", 1.0, 8, "A", 0),
                 ("dose_2", "dose", "none", 2.0, 8, "A", 0),
                 ("reverse_1.5", "reverse", "none", -1.5, 8, "A", 0), ("sham", "sham", "none", 0.0, 0, "A", 0),
                 ("random_0", "matched_random", "none", 1.5, 8, "A", 0), ("random_1", "matched_random", "none", 1.5, 8, "A", 1),
                 ("styled", "styled", None, None, 0, "B", 0), ("clamp", "clamp", "styled", None, 8, "B", 0),
                 ("clamp_random", "clamp_random", "styled", None, 8, "B", 0),
                 ("subset_top_0.5", "subset_top", "none", 1.5, 4, "A", 0),
                 ("subset_random_0.5", "subset_random", "none", 1.5, 4, "A", 0)]
        shifts = {"dose_0.5": 0.3, "dose_1": 0.6, "dose_1.5": 1.0, "dose_2": 1.2, "reverse_1.5": -0.8, "sham": 0.0,
                  "random_0": control, "random_1": control, "subset_top_0.5": 0.8, "subset_random_0.5": 0.2,
                  "clamp": -0.6, "clamp_random": -0.05, "styled": 0.0}
        for name, label, reference, alpha, neurons, prompt, seed_ in specs:
            value = base + effect * shifts.get(name, 0.0) + rng.normal(0, judge_noise)
            rows.append({"item_id": f"m{item:03d}", "panel": "love", "name": name, "family": label, "group": "",
                         "alpha": alpha, "reference": reference or "", "language": language, "seed": seed_,
                         "label": name, "neurons": neurons, "emotion_both": value, "emotion_judge1": value,
                         "emotion_judge2": value, "indep_target": 0.3 + 0.1 * shifts.get(name, 0.0) + rng.normal(0, 0.02),
                         "reply": f"{name} reply {item} {rng.integers(10**6)}", "reply_words": 40 + int(rng.integers(5)),
                         "degenerate": False, "preference_both": 0.5 + 0.2 * np.sign(shifts.get(name, 0.0)) * 0.5})
            for metric in nv.QUALITY:
                rows[-1][f"{metric}_both"] = 4.0 + rng.normal(0, 0.1)
    frame = pd.DataFrame(rows)
    frame.loc[frame.name.isin(["none", "styled"]), "preference_both"] = np.nan
    return frame


class StatisticsTests(unittest.TestCase):
    def test_paired_statistics_and_verdicts_on_a_known_effect(self):
        part = synthetic_part(effect=0.5)
        block = nv.summarise(part)
        by_name = {row["name"]: row for row in block["conditions"]}
        dose = by_name["dose_1.5"]["emotion"]
        self.assertGreater(dose["mean"], 0.3)          # true effect 0.5 with judge noise
        self.assertGreater(dose["lo"], 0)
        self.assertLess(by_name["sham"]["emotion"]["mean"], 0.2)
        contrast = {row["name"]: row["stat"] for row in block["contrasts"]}
        self.assertGreater(contrast["panel_vs_random"]["mean"], 0.3)
        self.assertGreater(contrast["dose_max_vs_min"]["mean"], 0)
        self.assertGreater(contrast["top_vs_random_subset"]["mean"], 0)
        verdict = nv.verdicts({"conditions": block["conditions"], "contrasts": block["contrasts"]}, "dose_1.5")
        self.assertEqual(verdict["panel_level"]["status"], "supported")
        self.assertEqual(verdict["specificity"]["status"], "supported")
        self.assertEqual(verdict["reversibility"]["status"], "supported")
        self.assertEqual(verdict["neuron_level"]["status"], "not_tested")
        self.assertEqual(verdict["summary"]["status"], "supported")

    def test_no_effect_is_not_reported_as_causal(self):
        part = synthetic_part(effect=0.0, control=0.0, items=30, seed=5)
        block = nv.summarise(part)
        verdict = nv.verdicts({"conditions": block["conditions"], "contrasts": block["contrasts"]}, "dose_1.5")
        self.assertNotEqual(verdict["panel_level"]["status"], "supported")
        self.assertNotEqual(verdict["summary"]["status"], "supported")

    def test_holm_correction_is_applied_across_conditions(self):
        part = synthetic_part(effect=0.5)
        rows = nv.summarise(part)["conditions"]
        for row in rows:
            self.assertIn("p_holm", row["emotion"])
            if row["emotion"]["p"] is not None:
                self.assertGreaterEqual(row["emotion"]["p_holm"], row["emotion"]["p"] - 1e-9)

    def test_random_draws_are_averaged_for_the_control_contrast(self):
        part = synthetic_part(effect=0.5, control=0.7)
        names = [row["name"] for row in nv.summarise(part)["conditions"]]
        self.assertIn("random_mean", names)
        self.assertNotIn("random_0", names)              # individual draws are used through their mean
        mean_row = next(row for row in nv.summarise(part)["conditions"] if row["name"] == "random_mean")
        self.assertAlmostEqual(mean_row["emotion"]["mean"], 0.5 * 0.7, delta=0.15)   # effect x control shift

    def test_language_slices_and_harm_share(self):
        part = synthetic_part(effect=0.5)
        block = nv.summarise(part[part.language == "hinglish"])
        self.assertTrue(all(row["n"] > 0 for row in block["conditions"]))
        for row in nv.summarise(part)["conditions"]:
            if row["harm_share"] is not None:
                self.assertGreaterEqual(row["harm_share"], 0)
                self.assertLessEqual(row["harm_share"], 1)

    def test_contrast_statistics_are_paired_differences(self):
        series = {"a": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), "b": pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])}
        rows = [{"name": "a", "family": "dose", "alpha": 1.0}, {"name": "b", "family": "dose", "alpha": 0.5}]
        found = nv.contrasts({**series, "dose_1": series["a"], "sham": series["b"]}, rows, {"a": "none", "b": "none"}, "a")
        self.assertTrue(found)
        for row in found:
            self.assertIn("mean", row["stat"])


class VerdictEdgeTests(unittest.TestCase):
    def test_verdict_directions(self):
        self.assertEqual(nv.verdict_of({"lo": 0.1, "hi": 0.5, "mean": 0.3}), "supported")
        self.assertEqual(nv.verdict_of({"lo": -0.5, "hi": -0.1, "mean": -0.3}), "not_supported")
        self.assertEqual(nv.verdict_of({"lo": -0.5, "hi": -0.1, "mean": -0.3}, positive=False), "supported")
        self.assertEqual(nv.verdict_of({"lo": -0.2, "hi": 0.2, "mean": 0.0}), "inconclusive")
        self.assertEqual(nv.verdict_of(None), "inconclusive")



class TurnJudgingTests(unittest.TestCase):
    def test_rows_skip_turns_without_a_steered_reply_and_already_judged_turns(self):
        import json
        import tempfile
        from pathlib import Path
        from generation_analysis import turn_evaluation as te
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "turns.jsonl"
            path.write_text(json.dumps({"turn_id": "a" * 32, "plan": {"response_emotion": "love"},
                                        "user": {"text": "I feel alone", "language": "english"},
                                        "replies": {"unsteered": "plain", "steered": "steered"}, "scores": {}}) + "\n"
                            + json.dumps({"turn_id": "b" * 32, "plan": {"response_emotion": "love"},
                                          "user": {"text": "x", "language": "english"},
                                          "replies": {"unsteered": "plain", "steered": None}, "scores": {}}) + "\n")
            rows = te.rows_to_judge(path, set(), "llama")
            self.assertEqual([row["role"] for row in rows], ["unsteered", "steered"])
            self.assertEqual(rows[1]["reference"], "plain")
            self.assertEqual(te.rows_to_judge(path, {"a" * 32}, "llama"), [])

    def test_judged_record_differences_and_missing_values(self):
        from generation_analysis import turn_evaluation as te
        ratings = {"unsteered": {"judge1": {"emotion": 2.0, "empathy": 3.0, "preservation": None},
                                 "judge2": {"emotion": 2.2, "empathy": 3.2, "preservation": None},
                                 "both": {"emotion": 2.1, "empathy": 3.1, "preservation": None}},
                   "steered": {"judge1": {"emotion": 3.0, "empathy": 3.0, "preservation": 4.0},
                               "judge2": {"emotion": 3.2, "empathy": 3.4, "preservation": 4.0},
                               "both": {"emotion": 3.1, "empathy": 3.2, "preservation": 4.0}}}
        record = te.judged_record("x", {"judge1": "q", "judge2": "g"}, ratings, {"judge1": 0.6, "both": 0.6},
                                  {"target": "love", "unsteered": 0.1, "steered": 0.4})
        self.assertAlmostEqual(record["difference"]["both"]["emotion"], 1.0)
        self.assertIsNone(record["difference"]["both"]["preservation"])        # no reference rating for the unsteered reply
        self.assertEqual(record["preservation"]["both"], 4.0)


class SingleNeuronTests(unittest.TestCase):
    def test_candidates_and_matched_controls(self):
        from generation_analysis import single_neuron as sn
        rng = np.random.default_rng(0)
        widths = {0: 60, 1: 60}
        start = nv.site_offsets(widths)
        neutral = rng.normal(size=120).astype(np.float32)
        panel = {0: ([1, 5, 9], [0.1, 0.5, 0.2]), 1: ([3, 7], [0.4, 0.05])}
        scores = {layer: {n: abs(d) for n, d in zip(*panel[layer])} for layer in panel}
        chosen = sn.choose(panel, scores, 3)
        self.assertEqual([(layer, neuron) for layer, neuron, _ in chosen], [(0, 5), (1, 3), (0, 9)])
        control = sn.controls(chosen, {0: {1, 5, 9}, 1: {3, 7}}, widths, neutral, start, rng)
        for (layer, neuron, delta), (c_layer, c_neuron, c_delta) in zip(chosen, control):
            self.assertEqual((layer, delta), (c_layer, c_delta))           # same layer and shift value
            self.assertNotIn(c_neuron, panel[layer][0])                    # outside the panel
        self.assertEqual(sn.without(panel, 0, 5)[0], ([1, 9], [0.1, 0.2]))
        self.assertEqual(sn.one(1, 3, 0.4), {1: ([3], [0.4])})

    def frame(self, strong=0.6, seed=1):
        """Screen records: neuron (0, 5) moves the emotional reply, the others and the controls do not."""
        rng = np.random.default_rng(seed)
        rows = []
        for item in range(20):
            def add(condition, role, layer, neuron, dose, emotional, plain):
                rows.append({"item_id": f"m{item}", "panel": "love", "language": "english", "condition": condition, "role": role,
                             "layer": layer, "neuron": neuron, "dose": dose, "emotional": emotional + rng.normal(0, 0.02),
                             "plain": plain + rng.normal(0, 0.02)})
            add("none", "", -1, -1, None, 0.0, 0.0)
            add("full", "", -1, -1, None, 1.0, 0.1)
            for role, neurons in (("panel", [(0, 5), (0, 9), (1, 3)]), ("control", [(0, 40), (0, 41), (1, 42)])):
                for layer, neuron in neurons:
                    for dose in (1.0, 8.0, 32.0):
                        effect = strong * dose / 32 if (layer, neuron) == (0, 5) else 0.0
                        add("solo", role, layer, neuron, dose, effect, 0.0)
                    if role == "panel":
                        add("loo", role, layer, neuron, None, 1.0 - (0.3 if (layer, neuron) == (0, 5) else 0.0), 0.1)
        return pd.DataFrame(rows)

    def test_screen_separates_an_effective_neuron_from_controls(self):
        from generation_analysis import single_neuron as sn
        block = sn.screen_summary(self.frame())["love"]
        by_id = {(entry["layer"], entry["neuron"]): entry for entry in block["neurons"]}
        self.assertEqual(by_id[(0, 5)]["screen_status"], "supported")
        self.assertEqual(by_id[(0, 9)]["screen_status"], "not_supported")
        self.assertEqual(by_id[(0, 40)]["screen_status"], "control")
        self.assertGreater(by_id[(0, 5)]["doses"]["32"]["mean"], by_id[(0, 5)]["doses"]["1"]["mean"])
        self.assertAlmostEqual(by_id[(0, 5)]["leave_out"]["mean"], 0.3, delta=0.05)     # the panel loses effect without it
        self.assertAlmostEqual(by_id[(0, 9)]["leave_out"]["mean"], 0.0, delta=0.05)
        self.assertIn("p_holm", by_id[(0, 5)]["screen"])

    def test_no_neuron_is_supported_without_an_effect(self):
        from generation_analysis import single_neuron as sn
        block = sn.screen_summary(self.frame(strong=0.0, seed=5))["love"]
        self.assertFalse(any(entry["screen_status"] == "supported" for entry in block["neurons"]))

    def test_a_neuron_needs_screen_and_judged_evidence(self):
        from generation_analysis import single_neuron as sn
        good = {"lo": 0.1, "hi": 0.5, "mean": 0.3, "vs_control": {"lo": 0.05, "hi": 0.4, "mean": 0.2}}
        weak = {"lo": -0.1, "hi": 0.3, "mean": 0.1, "vs_control": {"lo": -0.2, "hi": 0.2, "mean": 0.0}}
        self.assertEqual(sn.final_status("supported", good)[0], "supported")
        self.assertEqual(sn.final_status("supported", weak)[0], "inconclusive")       # likelihood only
        self.assertEqual(sn.final_status("supported", None)[0], "inconclusive")       # not generated
        self.assertEqual(sn.final_status("not_supported", weak)[0], "not_supported")
        self.assertEqual(sn.final_status("not_supported", good)[0], "inconclusive")
        self.assertEqual(sn.verdict({"tested": 5, "supported": 0, "screen_supported": 2, "generated": 5})["status"], "not_supported")
        self.assertEqual(sn.verdict({"tested": 5, "supported": 1, "screen_supported": 2, "generated": 5})["status"], "supported")
        self.assertEqual(sn.verdict({"tested": 5, "supported": 0, "screen_supported": 2, "generated": 0})["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
