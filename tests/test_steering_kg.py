"""Steering x knowledge-graph prompt study: the plain-prompt option, generation of the two new cells, the statistics."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import SYSTEM_PROMPT
from generation_analysis import steering_kg as sk
from generation_analysis import steering_modes as sm
from tests.test_always_steering import steering_service
from tests.test_emotion_service import HINGLISH, SAD
from tests.test_steering_modes import items


class PlainPromptTests(unittest.TestCase):
    def test_plain_prompt_changes_only_the_system_prompt(self):
        svc, runtime = steering_service()
        messages = [{"role": "user", "content": SAD}]
        deployed_id, deployed = svc.begin_turn(messages, steering_mode="always")
        plain_id, plain = svc.begin_turn(messages, steering_mode="always", prompt_variant="plain")
        self.assertEqual(plain["prompt_style"], "mindmate_only")
        self.assertEqual(plain["deployed_prompt_style"], deployed["prompt_style"])
        self.assertNotIn("deployed_prompt_style", deployed)
        for key in ("strategy", "response_emotion", "steering_alpha", "state_level"):
            self.assertEqual(plain[key], deployed[key])
        svc.respond(deployed_id, 64)
        self.assertNotEqual(runtime.prompts[-1], SYSTEM_PROMPT)       # the chat's prompt carries the KG strategy
        svc.respond(plain_id, 64)
        self.assertEqual(runtime.prompts[-1], SYSTEM_PROMPT)


class GenerationTests(unittest.TestCase):
    def run_both(self, svc, frame, folder: Path):
        source = folder / "modes"
        source.mkdir()
        sm.run_generation(svc, frame, source / "generations.jsonl", 64)
        kg = sk.kg_records(source)
        path = folder / "generations.jsonl"
        written = sk.run_generation(svc, frame, path, 64, kg)
        records = {(r["item_id"], r["mode"]): r for r in map(json.loads, path.read_text().splitlines())}
        return kg, path, written, records

    def test_new_cells_use_the_plain_prompt_with_the_same_plan(self):
        svc, runtime = steering_service()
        frame = items([{"item_id": "e1", "text": SAD, "group": "english", "language": "english", "turns": "single"},
                       {"item_id": "h1", "text": HINGLISH, "group": "hinglish", "language": "hinglish", "turns": "single"}])
        with tempfile.TemporaryDirectory() as folder:
            kg, path, written, records = self.run_both(svc, frame, Path(folder))
            self.assertEqual(written, 4)
            for item in ("e1", "h1"):
                self.assertEqual(records[(item, "none")]["applied_alpha"], 0.0)
                self.assertEqual(records[(item, "steer")]["status"], "applied")
                self.assertEqual(records[(item, "steer")]["planned_alpha"], kg[(item, "kg_steer")]["planned_alpha"])
                for system in ("none", "steer"):
                    self.assertEqual(records[(item, system)]["prompt_style"], "mindmate_only")
                    self.assertTrue(records[(item, system)]["plan_matches_kg"])
                    self.assertIsNone(records[(item, system)]["reused_from"])
            self.assertEqual(sk.run_generation(svc, frame, path, 64, kg), 0)          # resumes: nothing left
            table = sm.long_table(frame, [*records.values(), *kg.values()], sk.SYSTEMS, sk.TARGET_ORDER)
            self.assertEqual(len(table), 8)
            self.assertEqual(table.groupby("item_id").target.nunique().max(), 1)        # one tone target per message

    def test_a_message_the_chat_already_sends_the_plain_prompt_reuses_the_kg_reply(self):
        svc, runtime = steering_service()
        svc.tag = "sarvam_30b"                     # Sarvam: MindMate prompt alone for Hinglish, v3 prompt for English
        frame = items([{"item_id": "e1", "text": SAD}, {"item_id": "h1", "text": HINGLISH}])
        with tempfile.TemporaryDirectory() as folder:
            kg, _, written, records = self.run_both(svc, frame, Path(folder))
            self.assertEqual(records[("h1", "none")]["reused_from"], "kg")
            self.assertEqual(records[("h1", "steer")]["reply"], kg[("h1", "kg_steer")]["reply"])
            self.assertIsNone(records[("e1", "steer")]["reused_from"])

    def test_missing_kg_replies_or_other_settings_are_refused(self):
        svc, _ = steering_service()
        frame = items([{"item_id": "e1", "text": SAD}])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(SystemExit):
                sk.run_generation(svc, frame, Path(folder) / "g.jsonl", 64, {})
            lock = {"items": "abc", "model": "m", "max_new_tokens": 200, "state_alpha_range": [0.25, 0.75]}
            (Path(folder) / "lock.json").write_text(json.dumps(lock))
            sk.check_source(Path(folder), {**lock, "state_alpha_range": (0.25, 0.75)})
            with self.assertRaises(SystemExit) as raised:
                sk.check_source(Path(folder), {**lock, "max_new_tokens": 120})
            self.assertIn("max_new_tokens", str(raised.exception))


class ReportTests(unittest.TestCase):
    TONE = {"none": 3.0, "steer": 3.2, "kg": 3.5, "kg_steer": 3.6}

    def write_scores(self, folder: Path, n: int = 12, reused: tuple[str, ...] = ()):
        rng = np.random.default_rng(0)
        rows = [{"item_id": f"m{i}", "group": "english" if i < 8 else "crisis", "language": "english", "system": system,
                 "applied_alpha": 0.5 if system.endswith("steer") else 0.0,
                 "reused": system in sk.PLAIN and f"m{i}" in reused,
                 "indep_target": 0.1, "reply_words": 30, "repeated_bigram_rate": 0.0, "degenerate": False,
                 "failed": False, "reply": f"{system}-{i}"} for i in range(n) for system in sk.SYSTEMS]
        table = pd.DataFrame(rows)
        table.to_csv(folder / "replies.csv", index=False)
        for name in sm.JUDGES:
            ratings = pd.DataFrame({metric: 4.0 + rng.normal(0, 0.01, len(table)) for metric in (*sm.METRICS, "preservation")})
            ratings["judge_emotion"] += table.system.map(self.TONE).to_numpy() - 4.0
            harmed = (table.system == "steer") & table.item_id.isin(["m0", "m1", "m2"])
            ratings.loc[harmed, "safety"] -= 1.0                        # steering alone costs safety on 3 of 12
            ratings["hinglish_naturalness"] = np.nan
            ratings.to_csv(folder / f"ratings_{name}.csv", index=False)
            preferences = pd.DataFrame({f"{a}_over_{b}": 0.5 for a, b in sk.PAIRS}, index=[f"m{i}" for i in range(n)])
            preferences["steer_over_none"] = 0.6
            preferences["kg_steer_over_kg"] = 0.55
            preferences.to_csv(folder / f"preferences_{name}.csv")

    def test_factorial_contrasts_interaction_and_harm(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.write_scores(folder)
            summary = sk.summarise(folder, "fake-model")
            overall = summary["subsets"]["all"]
            tone = {contrast: overall["contrasts"][contrast]["judge_emotion"]["mean"] for contrast in sk.CONTRASTS}
            self.assertAlmostEqual(tone["steer - none"], 0.2, delta=0.02)
            self.assertAlmostEqual(tone["kg - none"], 0.5, delta=0.02)
            self.assertAlmostEqual(tone["kg_steer - kg"], 0.1, delta=0.02)
            self.assertAlmostEqual(tone["interaction"], -0.1, delta=0.02)      # steering adds less on top of the KG
            self.assertIsNotNone(overall["contrasts"]["interaction"]["judge_emotion"]["p_holm"])
            preference = overall["contrasts"]["interaction"]["preference_minus_0.5"]["mean"]
            self.assertAlmostEqual(preference, -0.05, places=6)
            self.assertAlmostEqual(overall["harmed_share"]["steer_vs_none"], 3 / 12)
            self.assertEqual(overall["harmed_share"]["kg_steer_vs_kg"], 0.0)
            self.assertAlmostEqual(overall["means"]["kg_steer"]["judge_emotion"], 3.6, delta=0.02)
            self.assertNotIn("kg_prompt_used", summary["subsets"])
            self.assertTrue((folder / "contrasts.csv").exists() and (folder / "system_means.csv").exists())

    def test_messages_without_a_kg_prompt_get_their_own_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            self.write_scores(folder, reused=("m10", "m11"))
            summary = sk.summarise(folder)
            self.assertEqual(summary["subsets"]["kg_prompt_used"]["n"], 10)


if __name__ == "__main__":
    unittest.main()
