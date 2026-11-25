"""Confirmatory study: the warmth-prompt baseline, six-cell generation, registration and the registered tests."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.config import SYSTEM_PROMPT
from backend.emotion_service import WARMTH_WORDS
from generation_analysis import confirmatory as cf
from generation_analysis import steering_modes as sm
from generation_analysis.common import EMOTION_DESCRIPTIONS
from tests.test_always_steering import steering_service
from tests.test_emotion_service import HINGLISH, SAD
from tests.test_steering_modes import items


def frame():
    return items([{"item_id": "e1", "text": SAD, "group": "english", "language": "english", "turns": "single"},
                  {"item_id": "h1", "text": HINGLISH, "group": "hinglish", "language": "hinglish", "turns": "single"}])


class WarmthPromptTests(unittest.TestCase):
    def test_warm_prompt_names_the_emotion_steering_targets(self):
        svc, runtime = steering_service()
        messages = [{"role": "user", "content": SAD}]
        _, steered = svc.begin_turn(messages, steering_mode="always")
        turn_id, warm = svc.begin_turn(messages, steering_mode="off", prompt_variant="warm")
        self.assertEqual(warm["warmth_emotion"], steered["response_emotion"])
        self.assertEqual(warm["steering_alpha"], 0.0)
        self.assertEqual(warm["prompt_style"], "warmth_instruction")
        svc.respond(turn_id, 64)
        prompt = runtime.prompts[-1]
        self.assertTrue(prompt.startswith(SYSTEM_PROMPT))
        self.assertIn("Reply warmly", prompt)
        self.assertIn(WARMTH_WORDS[warm["warmth_emotion"]], prompt)

    def test_warmth_words_are_the_judges_rubric_words_and_bad_variants_fail(self):
        for emotion, words in WARMTH_WORDS.items():
            self.assertEqual(EMOTION_DESCRIPTIONS[emotion].split(": ", 1)[1], words)
        svc, _ = steering_service()
        with self.assertRaises(ValueError):
            svc.begin_turn([{"role": "user", "content": SAD}], prompt_variant="shout")


class GenerationTests(unittest.TestCase):
    def test_six_cells_with_one_plan_per_message_and_resume(self):
        svc, runtime = steering_service()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generations.jsonl"
            self.assertEqual(cf.run_generation(svc, frame(), path, 64), 12)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            by = {(r["item_id"], r["mode"]): r for r in records}
            self.assertEqual(cf.plan_mismatches(records), [])
            for item in ("e1", "h1"):
                for cell in cf.STEERED:
                    self.assertGreater(by[(item, cell)]["applied_alpha"], 0)
                for cell in ("none", "warm", "kg"):
                    self.assertEqual(by[(item, cell)]["applied_alpha"], 0.0)
                self.assertEqual(by[(item, "warm")]["prompt_style"], "warmth_instruction")
                self.assertEqual(by[(item, "none")]["prompt_style"], "mindmate_only")
                self.assertEqual(by[(item, "kg")]["prompt_style"], "strategy")
            self.assertEqual(cf.run_generation(svc, frame(), path, 64), 0)
            table = sm.long_table(frame(), records, tuple(cf.CELLS), cf.TARGET_ORDER)
            self.assertEqual(len(table), 12)

    def test_kg_cells_reuse_the_plain_reply_where_the_chat_sends_the_plain_prompt(self):
        svc, _ = steering_service()
        svc.tag = "sarvam_30b"                     # Sarvam: MindMate prompt alone for Hinglish
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "generations.jsonl"
            cf.run_generation(svc, frame(), path, 64)
            by = {(r["item_id"], r["mode"]): r for r in map(json.loads, path.read_text().splitlines())}
            self.assertEqual(by[("h1", "kg_steer")]["reused_from"], "steer")
            self.assertEqual(by[("h1", "kg")]["reply"], by[("h1", "none")]["reply"])
            self.assertIsNone(by[("e1", "kg")]["reused_from"])


class HypothesisTests(unittest.TestCase):
    def test_duplicates_are_registered_once_and_llama2_predicts_harm(self):
        emollama = cf.hypotheses("emollama_chat_7b")              # locked setup = steer: H5, H8, H9 repeat H1, H2, H6
        self.assertEqual([h["hypothesis"] for h in emollama], ["H1", "H2", "H3", "H4", "H6", "H7"])
        llama2 = {h["hypothesis"]: h for h in cf.hypotheses("llama2_13b_chat")}
        self.assertEqual((llama2["H9"]["test"], llama2["H5"]["first"]), ("less", "kg"))
        self.assertEqual(len(cf.hypotheses("llama31_8b")), 9)
        self.assertIn(("kg", "warm"), cf.pairs_for("llama2_13b_chat"))

    def test_registered_tests(self):
        rng = np.random.default_rng(0)
        up = 0.3 + rng.normal(0, 0.5, 200)
        self.assertLess(cf.registered_test(up, "greater"), 0.001)
        self.assertGreater(cf.registered_test(up, "less"), 0.99)
        self.assertLess(cf.registered_test(up, "two-sided"), 0.001)
        slight_drop = -0.02 + rng.normal(0, 0.3, 300)            # within the 0.10 margin
        self.assertLess(cf.registered_test(slight_drop, "noninferior"), 0.001)
        self.assertGreater(cf.registered_test(-0.3 + rng.normal(0, 0.3, 300), "noninferior"), 0.99)
        self.assertEqual(cf.registered_test(np.zeros(10), "greater"), 1.0)
        self.assertEqual(cf.registered_test(np.zeros(10), "noninferior"), 0.0)   # identical replies are not worse


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        paths = {"ROOT": root, "ITEMS": root / "items.csv", "MANIFEST": root / "messages.json",
                 "REGISTRATION": root / "preregistration.json", "REGISTRATION_MD": root / "preregistration.md"}
        for name, value in paths.items():
            patcher = patch.object(cf, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.folder.cleanup)
        items = frame()
        items["history"] = items.history.map(json.dumps)
        items.to_csv(paths["ITEMS"], index=False)
        paths["MANIFEST"].write_text(json.dumps({"sha256": sm.sha256(paths["ITEMS"]), "sizes": [
            {"group": "english", "turns": "single", "n": 1}], "sources": "test"}))

    def test_register_once_then_refuse_changes(self):
        cf.register(argparse.Namespace(replace=False))
        registration = cf.check_registration()
        self.assertEqual(len(registration["hypotheses"]), 33)
        self.assertIn("| llama2_13b_chat:H9 |", cf.REGISTRATION_MD.read_text())
        with self.assertRaises(SystemExit):
            cf.register(argparse.Namespace(replace=False))        # already registered
        changed = {**registration["fingerprints"], "backend/emotion_service.py": "0" * 64}
        with patch.object(cf, "fingerprints", return_value=changed), self.assertRaises(SystemExit) as raised:
            cf.check_registration()
        self.assertIn("emotion_service.py", str(raised.exception))
        model = cf.ROOT / "llama31_8b"
        model.mkdir()
        (model / "generations.jsonl").write_text("{}\n")
        with self.assertRaises(SystemExit):
            cf.register(argparse.Namespace(replace=True))         # replies exist: the registration is final

    def test_summary_and_verdict_on_known_effects(self):
        cf.register(argparse.Namespace(replace=False))
        folder = cf.ROOT / "llama31_8b"
        folder.mkdir()
        tone = {"none": 3.0, "warm": 3.6, "steer": 3.4, "warm_steer": 3.8, "kg": 3.5, "kg_steer": 3.9}
        n, rng = 40, np.random.default_rng(1)
        rows = [{"item_id": f"m{i}", "group": "english" if i < 30 else "hinglish", "language": "english",
                 "system": cell, "applied_alpha": 0.5 if cell in cf.STEERED else 0.0, "indep_target": 0.1,
                 "reply_words": 30 + i % 5, "repeated_bigram_rate": 0.0, "degenerate": False, "failed": False,
                 "reply": f"{cell}-{i}"} for i in range(n) for cell in cf.CELLS]
        table = pd.DataFrame(rows)
        table.to_csv(folder / "replies.csv", index=False)
        for name in sm.JUDGES:
            ratings = pd.DataFrame({metric: 4.5 + rng.normal(0, 0.2, len(table)) for metric in (*sm.METRICS, "preservation")})
            ratings["judge_emotion"] = table.system.map(tone).to_numpy() + rng.normal(0, 0.2, len(table))
            ratings["hinglish_naturalness"] = np.nan
            ratings.to_csv(folder / f"ratings_{name}.csv", index=False)
            pd.DataFrame({f"{a}_over_{b}": 0.5 for a, b in cf.pairs_for("llama31_8b")},
                         index=[f"m{i}" for i in range(n)]).to_csv(folder / f"preferences_{name}.csv")
        summary = cf.summarise(folder, "llama31_8b")
        results = {row["hypothesis"]: row for row in summary["confirmatory"]}
        self.assertAlmostEqual(results["H1"]["mean"], 0.4, delta=0.1)
        self.assertAlmostEqual(results["H2"]["mean"], -0.2, delta=0.1)              # the instruction is warmer
        self.assertLess(results["H6"]["p"], 0.05)                                    # quality unchanged: non-inferior
        self.assertEqual(summary["n_primary"], n)
        cf.verdict(argparse.Namespace())
        report = json.loads((cf.ROOT / "report.json").read_text())
        rows = {row["id"]: row for row in report["hypotheses"]}
        self.assertTrue(rows["llama31_8b:H1"]["confirmed"])
        self.assertEqual(rows["llama31_8b:H2"]["direction"], "warm higher")
        self.assertFalse(rows["sarvam_30b:H1"]["ran"] or rows["sarvam_30b:H1"]["confirmed"])
        self.assertEqual(sorted(report["models_not_run"]), ["emollama_chat_7b", "llama2_13b_chat", "sarvam_30b"])
        self.assertIn("not run", (cf.ROOT / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
