"""Steering-mode experiment: message selection, generation through the chat's own path, and the report statistics."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generation_analysis import steering_modes as sm
from tests.test_always_steering import steering_service
from tests.test_emotion_service import HINGLISH, SAD


def items(rows):
    frame = pd.DataFrame(rows)
    frame["history"] = [[] for _ in range(len(frame))]
    return frame


class SelectionTests(unittest.TestCase):
    def test_hinglish_candidates_keep_unseen_hinglish_chat_messages(self):
        frame = pd.DataFrame({"hinglish_genz_text": [
            "yaar aaj bahut akela feel ho raha hai, koi nahi hai mere saath",      # kept
            "I had a really lovely day at the beach with my friends today",       # English
            "haan",                                                               # too short
            "[NAME] ne mujhe bahut bura feel karaya aaj office mein yaar",        # placeholder
            "mera dil bahut udaas hai aaj, kuch accha nahi lag raha hai yaar"],   # seen before
            "label": ["sadness"] * 5})
        out = sm.hinglish_candidates(frame, {"mera dil bahut udaas hai aaj, kuch accha nahi lag raha hai yaar"}, 10, 1)
        self.assertEqual([item["text"] for item in out], ["yaar aaj bahut akela feel ho raha hai, koi nahi hai mere saath"])

    def test_crisis_candidates_need_a_cue_and_are_unseen(self):
        sources = [("a", "1", "I want to end my life, nothing matters anymore"), ("a", "2", "What a lovely sunny day"),
                   ("a", "3", "I want to end my life, nothing matters anymore"), ("b", "4", "mujhe jeena nahi hai ab")]
        out = sm.crisis_candidates(sources, set())
        self.assertEqual([(item["conv_id"], item["language"]) for item in out], [("1", "english"), ("4", "hinglish")])


class GenerationTests(unittest.TestCase):
    def test_modes_through_the_chat_path_with_reuse_and_resume(self):
        svc, runtime = steering_service()
        frame = items([{"item_id": "e1", "text": SAD}, {"item_id": "h1", "text": HINGLISH}])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generations.jsonl"
            self.assertEqual(sm.run_generation(svc, frame, path, 64), 6)
            records = {(r["item_id"], r["mode"]): r for r in map(json.loads, path.read_text().splitlines())}
            self.assertEqual(records[("e1", "off")]["applied_alpha"], 0.0)
            self.assertEqual(records[("e1", "policy")]["planned_alpha"], 0.75)     # the policy steers English love
            self.assertIsNone(records[("e1", "policy")]["reused_from"])
            self.assertEqual(records[("h1", "policy")]["reused_from"], "off")      # not steered: the off reply
            self.assertEqual(records[("h1", "policy")]["reply"], records[("h1", "off")]["reply"])
            for item in ("e1", "h1"):
                self.assertEqual(records[(item, "always")]["status"], "applied")
                self.assertIsNotNone(records[(item, "always")]["state_level"])
                self.assertIsNotNone(records[(item, "always")]["unsteered"])
            self.assertEqual(sm.run_generation(svc, frame, path, 64), 0)           # resumes: nothing left to do

    def test_long_table_uses_one_target_per_message_and_flags_failures(self):
        frame = items([{"item_id": "c1", "text": "I want to end my life", "group": "crisis", "language": "english",
                        "turns": "single"}])
        records = [{"item_id": "c1", "mode": mode, "reply": "[generation failed: x]" if mode == "off" else "reply",
                    "status": "applied" if mode == "always" else "not_planned", "planned_alpha": 0.7 if mode == "always" else 0.0,
                    "applied_alpha": 0.7 if mode == "always" else 0.0, "strategy": "crisis_support",
                    "response_emotion": "love" if mode == "always" else None, "unsteered": "plain", "state_level": 0.9}
                   for mode in sm.MODES]
        table = sm.long_table(frame, records)
        self.assertEqual(set(table.target), {"love"})
        self.assertTrue(table.set_index("system").loc["off", "failed"])
        self.assertEqual(table.set_index("system").loc["always", "reference"], "plain")
        with self.assertRaises(SystemExit):
            sm.long_table(frame, records[:2])


class LockTests(unittest.TestCase):
    def test_rerun_with_the_same_settings_resumes_and_a_change_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lock.json"
            lock = {"items": "abc", "model": "m", "max_new_tokens": 200, "state_alpha_range": (0.25, 0.75)}
            sm.check_lock(path, lock)
            sm.check_lock(path, lock)                       # a tuple written as a JSON list still matches
            with self.assertRaises(SystemExit) as raised:
                sm.check_lock(path, {**lock, "max_new_tokens": 120})
            self.assertIn("max_new_tokens", str(raised.exception))


class ReportTests(unittest.TestCase):
    def write_scores(self, folder: Path, n: int = 12):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(n):
            group = "english" if i < 8 else "crisis"
            for system in sm.MODES:
                rows.append({"item_id": f"m{i}", "group": group, "language": "english", "system": system,
                             "applied_alpha": 0.6 if system == "always" or (system == "policy" and i < 4) else 0.0,
                             "indep_target": 0.1, "reply_words": 30, "repeated_bigram_rate": 0.0, "degenerate": False,
                             "failed": False, "reply": f"{system}-{i}"})
        table = pd.DataFrame(rows)
        table.to_csv(folder / "replies.csv", index=False)
        for name in sm.JUDGES:
            ratings = pd.DataFrame({metric: 3.0 + rng.normal(0, 0.01, len(table)) for metric in (*sm.METRICS, "preservation")})
            ratings.loc[table.system == "always", "judge_emotion"] += 1.0          # steering raises the tone by one point
            harmed = (table.system == "always") & (table.item_id.isin(["m0", "m1", "m2"]))
            ratings.loc[harmed, "safety"] -= 1.0                                      # three of twelve lose safety
            ratings["hinglish_naturalness"] = np.nan
            ratings.to_csv(folder / f"ratings_{name}.csv", index=False)
            preferences = pd.DataFrame({f"{a}_over_{b}": 0.5 for a, b in sm.PAIRS}, index=[f"m{i}" for i in range(n)])
            preferences["always_over_off"] = 0.7
            preferences.to_csv(folder / f"preferences_{name}.csv")

    def test_summary_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.write_scores(folder)
            summary = sm.summarise(folder, "fake-model")
            overall = summary["subsets"]["all"]
            emotion = overall["contrasts"]["always - off"]["judge_emotion"]
            self.assertAlmostEqual(emotion["mean"], 1.0, places=1)
            self.assertLess(emotion["ci_low"], emotion["mean"])
            self.assertIsNotNone(emotion["p_holm"])                               # a primary contrast
            self.assertAlmostEqual(overall["contrasts"]["always - off"]["preference_minus_0.5"]["mean"], 0.2, places=3)
            self.assertAlmostEqual(overall["harmed_share"]["always_vs_off"], 3 / 12)
            self.assertEqual(overall["steered_share"]["always"], 1.0)
            self.assertAlmostEqual(overall["steered_share"]["policy"], 4 / 12)
            self.assertEqual(summary["subsets"]["crisis"]["n"], 4)
            self.assertTrue((folder / "contrasts.csv").exists() and (folder / "system_means.csv").exists())


if __name__ == "__main__":
    unittest.main()
