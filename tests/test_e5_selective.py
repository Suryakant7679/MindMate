"""E5 (quality-constrained adaptive selective steering): risk screen, decision logic, run-time service, calibration
procedure, held-out runner, judge cache and report - all on fakes (no model, no GPU)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backend.emotion_service import STATE_ALPHA_RANGE
from backend.selective_service import SelectiveSteeringService, hook_summary, panel_fingerprint
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
from chatbot_feature.emotion_reasoner.risk_screen import screen
from chatbot_feature.emotion_reasoner.selective import (PolicyCell, SelectivePolicy, accept, choose, deficit_decision,
                                                        language_of, markers, pre_decision)
from generation_analysis import e5_fit
from generation_analysis import e5_selective as e5
from tests.test_always_steering import PANELS
from tests.test_emotion_service import CRISIS, HINGLISH, PLAIN, POLICY, SAD, STEERED, Runtime, Scorer

TAG = "llama31_8b"


def policy(cells=None, **deficit) -> SelectivePolicy:
    out = SelectivePolicy(model=TAG, alpha_range=tuple(STATE_ALPHA_RANGE["default"]),
                          cells=cells if cells is not None else {"*|english": PolicyCell((1.0,), "test"),
                                                                 "*|hinglish": PolicyCell((0.5,), "test")})
    if deficit:
        out.deficit.update(deficit)
    return out


def selective_service(selective=None, love_plain=0.05, love_steered=0.40, steered_text=STEERED):
    runtime = Runtime(steered_text=steered_text)
    runtime._emotion_panels = PANELS
    svc = SelectiveSteeringService(runtime, selective or policy(), agent=EmotionReasonerAgent(["love"], policy=POLICY),
                                   scorer=Scorer(love_plain, love_steered), evidence={})
    return svc, runtime


def user(text):
    return [{"role": "user", "content": text}]


class RiskScreenTests(unittest.TestCase):
    def category(self, text, history=()):
        return screen(text, history)["category"]

    def test_disclosure_mention_and_figurative_are_told_apart(self):
        self.assertEqual(self.category("I want to die, nothing helps anymore"), "disclosure")
        self.assertEqual(self.category("I don't want to die, I just want the pain to stop"), "mention")
        self.assertEqual(self.category("My friend told me she wants to kill herself"), "mention")
        self.assertEqual(self.category("My friend left and now I want to die"), "disclosure")
        self.assertEqual(self.category("This exam is killing me lol"), "figurative")
        self.assertTrue(screen("This exam is killing me lol")["steering_allowed"])

    def test_other_risks_and_languages_disable_steering(self):
        for text, expected in (("He hits me when he drinks", "abuse_or_danger"),
                               ("I'm going to make them pay, I could kill him", "harm_to_others"),
                               ("I have chest pain and can't breathe", "medical_emergency"),
                               ("ab jeene ka koi matlab nahi lagta yaar", "indirect"),
                               ("मैं अब जीना नहीं चाहता", "disclosure"),
                               ("Honestly I'm a burden to everyone around me", "indirect")):
            found = screen(text)
            self.assertEqual(found["category"], expected, text)
            self.assertFalse(found["steering_allowed"])

    def test_no_cue_is_allowed_and_history_carries_risk(self):
        self.assertEqual(self.category("Got the job today, so happy!"), "none")
        found = screen("thanks for listening", [{"role": "user", "content": CRISIS}, {"role": "assistant", "content": "..."}])
        self.assertEqual((found["category"], found["source"]), ("disclosure", "history"))


class DecisionLogicTests(unittest.TestCase):
    def test_language_and_markers(self):
        self.assertEqual(language_of("मैं बहुत परेशान हूँ"), "hindi")
        self.assertEqual(language_of(HINGLISH), "hinglish")
        self.assertEqual(language_of(SAD), "english")
        found = markers("Oh sweetie, I'll always be here for you. Everything will be fine, I promise.")
        self.assertEqual(set(found) - {"validation"}, {"familiarity", "dependency", "false_reassurance"})
        self.assertEqual(markers("I love your idea, as you have mentioned"), {})

    def test_policy_round_trip_lookup_and_strength(self):
        cells = {"strategy:compassionate_comfort|english": PolicyCell((0.5,), "own cell"),
                 "love|english": PolicyCell((1.0,), "emotion cell"), "*|english": PolicyCell((), "residual")}
        original = policy(cells, enabled=True, threshold=0.3)
        with tempfile.TemporaryDirectory() as folder:
            original.save(Path(folder) / "p.json")
            loaded = SelectivePolicy.load(Path(folder) / "p.json")
        self.assertEqual(loaded.fingerprint(), original.fingerprint())
        self.assertEqual(loaded.cell_for("compassionate_comfort", "english", "love")[0], "strategy:compassionate_comfort|english")
        self.assertEqual(loaded.cell_for("warm_connection", "english", "love")[0], "love|english")
        self.assertEqual(loaded.cell_for("celebratory_sharing", "english", "joy")[0], "*|english")
        self.assertEqual(loaded.cell_for("celebratory_sharing", "hindi", "joy"), (None, None))
        self.assertAlmostEqual(loaded.alpha_for(0.5), 0.5)
        self.assertAlmostEqual(loaded.alpha_for(0.767), 0.63)                 # rounded as E4 rounds it
        self.assertAlmostEqual(loaded.alpha_for(2.0), 0.75)

    def test_pre_decision_order_safety_panel_context(self):
        plan = {"strategy": "compassionate_comfort", "response_emotion": "love", "state_level": 0.5, "safety_flag": False}
        none = {"category": "none", "cues": []}
        crisis = pre_decision(policy(), {**plan, "safety_flag": True}, SAD, none, panel="love")
        self.assertEqual((crisis["eligible"], crisis["stage"]), (False, "safety"))
        risky = pre_decision(policy(), plan, SAD, {"category": "indirect", "cues": ["x"], "reason": "r"}, panel="love")
        self.assertEqual(risky["stage"], "safety")
        self.assertEqual(pre_decision(policy(), plan, SAD, none, panel=None)["stage"], "panel")
        empty = pre_decision(policy({"*|english": PolicyCell((), "not steered: harmful")}), plan, SAD, none, panel="love")
        self.assertEqual((empty["stage"], empty["reason"]), ("context", "not steered: harmful"))
        unknown = pre_decision(policy({}), plan, SAD, none, panel="love")
        self.assertIn("no development evidence", unknown["reason"])
        ok = pre_decision(policy({"*|english": PolicyCell((1.0, 0.5), "x")}), plan, SAD, none, panel="love")
        self.assertTrue(ok["eligible"])
        self.assertEqual(ok["alphas"], [0.5, 0.25])

    def test_deficit_gate_only_in_its_languages(self):
        gated = policy(enabled=True, threshold=0.3)
        self.assertFalse(deficit_decision(gated, "love", PLAIN, {"love": 0.45}, "english")["steer"])
        self.assertTrue(deficit_decision(gated, "love", PLAIN, {"love": 0.10}, "english")["steer"])
        hinglish = deficit_decision(gated, "love", PLAIN, {"love": 0.90}, "hinglish")
        self.assertTrue(hinglish["steer"])
        self.assertIn("not used for hinglish", hinglish["reason"])
        self.assertTrue(deficit_decision(policy(), "love", PLAIN, {"love": 0.90}, "english")["steer"])

    def test_acceptance_checks_and_choice(self):
        base = "That sounds really hard. What happened today that made it feel this way?"
        checks = {
            "0.9": ("0.9", False, "not a conversational answer"),
            "repetitive": ("so sad so sad so sad so sad so sad so sad", False, "more repetitive"),
            "short": ("Oh no, that is sad.", False, "much shorter"),
            "familiar": ("Oh sweetie, that sounds really hard. What happened today that made it feel this way?", False,
                         "added risky phrasing: familiarity"),
            "fine": ("That sounds really hard, and I'm so sorry. What happened today that made it feel like this?", True, "passed"),
        }
        for name, (candidate, ok, reason) in checks.items():
            accepted, why, _ = accept(policy(), base, candidate)
            self.assertEqual(accepted, ok, name)
            self.assertIn(reason, why, name)
        hinglish = accept(policy(), "yaar bahut bura laga hoga, kya hua?", "That must have felt really bad, what happened?")
        self.assertFalse(hinglish[0])
        capped = policy()
        capped.acceptance["max_word_ratio"] = 1.5
        self.assertFalse(accept(capped, base, base + " " + base)[0])
        choice = choose(policy(), base, [(0.6, "0.9"), (0.3, checks["fine"][0])])
        self.assertEqual((choice["sent"], choice["alpha"]), ("steered", 0.3))
        self.assertEqual([t["accepted"] for t in choice["tried"]], [False, True])
        self.assertEqual(choose(policy(), base, [(0.6, "0.9")])["sent"], "unsteered")


class ServiceTests(unittest.TestCase):
    def test_eligible_turn_generates_candidates_and_logs_everything(self):
        svc, runtime = selective_service()
        analysis = svc.respond_selective(user(SAD), 64)
        log = analysis["e5"]
        self.assertEqual(analysis["reply"], STEERED)
        self.assertEqual((log["status"], log["attempted"], log["retained"]), ("applied", True, True))
        self.assertEqual(runtime.calls[-1][0], 0.0)                    # unsteered row decoded in the same batch
        self.assertAlmostEqual(log["applied_alpha"], log["alpha_e4"])
        self.assertAlmostEqual(log["alpha_e4"], log["e4_proposal"]["alpha"])   # E5 reuses E4's state-scaled strength
        self.assertTrue(log["hooks"]["hooks_executed"])
        self.assertEqual(log["panel"], "love")
        self.assertEqual(log["panel_fingerprint"], panel_fingerprint(PANELS["love"]))
        self.assertEqual(analysis["decision"]["applied_alpha"], log["applied_alpha"])

    def test_deficit_gate_keeps_the_unsteered_reply(self):
        svc, _ = selective_service(policy(enabled=True, threshold=0.3), love_plain=0.5)
        analysis = svc.respond_selective(user(SAD), 64)
        self.assertEqual((analysis["reply"], analysis["e5"]["status"]), (PLAIN, "skipped"))
        self.assertTrue(analysis["e5"]["attempted"])
        self.assertFalse(analysis["e5"]["retained"])

    def test_rejected_candidate_falls_back(self):
        svc, _ = selective_service(steered_text="0.5")
        analysis = svc.respond_selective(user(SAD), 64)
        self.assertEqual((analysis["reply"], analysis["e5"]["status"]), (PLAIN, "rejected"))
        self.assertIn("not a conversational answer", analysis["e5"]["decision_reason"])

    def test_crisis_and_unsteered_contexts_never_generate_candidates(self):
        svc, runtime = selective_service()
        analysis = svc.respond_selective(user(CRISIS), 64)
        self.assertEqual((analysis["e5"]["status"], analysis["e5"]["stage"]), ("not_eligible", "safety"))
        self.assertFalse(any(isinstance(call, list) for call in runtime.calls))   # no candidate batch
        svc, runtime = selective_service(policy({"*|english": PolicyCell((), "not steered")}))
        analysis = svc.respond_selective(user(SAD), 64)
        self.assertEqual(analysis["e5"]["stage"], "context")
        self.assertEqual(analysis["reply"].strip(), PLAIN)

    def test_existing_modes_are_unchanged_and_policies_are_model_specific(self):
        svc, _ = selective_service()
        turn_id, plan = svc.begin_turn(user(SAD), steering_mode="always")
        self.assertEqual(svc.respond(turn_id, 64)["decision"]["status"], "applied")
        with self.assertRaises(ValueError):
            selective_service(SelectivePolicy(model="sarvam_30b"))

    def test_hook_summary(self):
        self.assertFalse(hook_summary(None)["hooks_executed"])
        summary = hook_summary({"alpha": 0.5, "layers": [{"layer": 8, "neurons": [1, 2], "decode_positions": 3,
                                                           "applied_shift": [0.1, 0.0]}]})
        self.assertTrue(summary["hooks_executed"] and summary["shift_applied"])
        self.assertEqual(summary["neurons"], 2)


def calibration_frame(seed=0) -> pd.DataFrame:
    """Synthetic development table: a helpful cell, a harmful one, a weak one and a tiny one."""
    rng = np.random.default_rng(seed)
    rows = []
    plan = {"helpful": (0.5, 0.0, 0.0, 30), "harmful": (0.6, -0.05, -0.3, 30), "weak": (0.05, 0.0, 0.0, 30),
            "tiny": (0.8, 0.0, 0.0, 4)}
    for strategy, (tone, quality, safety, n) in plan.items():
        for i in range(n):
            row = {"text": f"{strategy} message {i}", "strategy": strategy, "lang3": "english", "eligible": True,
                   "s0_batch": 0.05 if i % 2 else 0.6, "kg_text": "plain reply that listens to you today",
                   "c100_text": "steered reply with warmth and care for you today", "b100_text": "plain reply that listens to you today"}
            for metric in ("judge_emotion", *e5_fit.QUALITY, "safety", "validation"):
                row[f"kg_{metric}"] = 3.0
                shift = tone if metric == "judge_emotion" else safety if metric == "safety" else quality
                row[f"c100_{metric}"] = 3.0 + shift + rng.normal(0, 0.05)
                row[f"b100_{metric}"] = 3.0
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index = [f"k{i}" for i in range(len(frame))]
    return frame


class CalibrationTests(unittest.TestCase):
    def test_cells_follow_the_constraints(self):
        table = calibration_frame()
        chosen, evidence = e5_fit.fit_cells(table, {"deficit_threshold": None, "max_word_ratio": None}, TAG)
        self.assertEqual(chosen["strategy:helpful|english"], (1.0,))
        self.assertEqual(chosen["strategy:harmful|english"], ())
        self.assertEqual(chosen["strategy:weak|english"], ())
        self.assertIn("*|english", chosen)                             # the tiny strategy pools into the residual cell
        self.assertTrue(evidence["population"]["passes"])
        reasons = evidence["cells"]["strategy:harmful|english"]["1.0"]["reasons"]
        self.assertTrue(any("safety" in reason for reason in reasons))

    def test_simulation_applies_deficit_gate_and_acceptance(self):
        table = calibration_frame()
        effects = e5_fit.simulate(table, (1.0,), {"deficit_threshold": 0.3}, TAG)
        blocked = table.s0_batch >= 0.3
        self.assertFalse(effects.loc[blocked, "steered"].any())
        self.assertTrue(effects.loc[~blocked, "steered"].all())
        table.loc[table.index[:5], "c100_text"] = "0.7"                # degenerate candidates are rejected
        effects = e5_fit.simulate(table, (1.0,), {}, TAG)
        self.assertFalse(effects.loc[table.index[:5], "steered"].any())
        self.assertTrue((effects.loc[table.index[:5], "d_judge_emotion"] == 0).all())

    def test_fit_is_deterministic_and_freezes_a_policy(self):
        table = calibration_frame()
        first, report = e5_fit.fit(table, TAG, (0.25, 0.75), grid={"deficit_threshold": (None, 0.3), "max_word_ratio": (None,)})
        second, _ = e5_fit.fit(table, TAG, (0.25, 0.75), grid={"deficit_threshold": (None, 0.3), "max_word_ratio": (None,)})
        first.provenance.pop("fitted_at_utc"), second.provenance.pop("fitted_at_utc")
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertTrue(first.frozen)
        self.assertEqual(first.cells["strategy:helpful|english"].multipliers, (1.0,))
        self.assertEqual(len(report["grid"]), 2)

    def test_tiny_development_sets_steer_nothing_without_crashing(self):
        policy, report = e5_fit.fit(calibration_frame().head(3), TAG, (0.25, 0.75),
                                    grid={"deficit_threshold": (None,), "max_word_ratio": (None,)})
        self.assertEqual(report["steered_cells"], {})
        self.assertTrue(all(not cell.multipliers for cell in policy.cells.values()))

    def test_population_check_fails_on_a_safety_loss(self):
        effects = pd.DataFrame({"d_judge_emotion": [0.5] * 40, "d_safety": [-0.2] * 40, "d_appropriateness": [0.0] * 40,
                                "steered": [True] * 40, "attempted": [True] * 40})
        check = e5_fit.population_check(effects)
        self.assertFalse(check["passes"])
        self.assertFalse(check["bounds"]["safety"]["passes"])


class FakeJudge:
    def __init__(self):
        self.calls = 0

    def rate(self, contexts, replies, rubrics):
        self.calls += len(replies)
        return np.array([4.5 if reply.startswith("steered") and "tone" in rubric.lower() else 4.0 for reply, rubric
                         in zip(replies, rubrics)])

    def prefer(self, contexts, first, second):
        self.calls += len(first)
        return np.full(len(first), 0.6)

    def close(self):
        pass


def items(rows):
    frame = pd.DataFrame(rows)
    frame["history"] = [[] for _ in range(len(frame))]
    for column, value in (("group", "english"), ("category", "test"), ("turns", "single"), ("language", "english")):
        if column not in frame:
            frame[column] = value
    return frame


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.base = Path(self.folder.name)
        self.patches = [patch.object(e5, "PROGRESS", self.base / "progress"), patch.object(e5, "LOGS", self.base / "logs")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.folder.cleanup()

    def generate(self, frame, selective=None):
        svc, runtime = selective_service(selective)
        path = self.base / "generations.jsonl"
        e5.run_heldout(svc, frame, path, e5.Progress("test", len(frame)), retry_failed=False)
        return {(r["item_id"], r["condition"]): r for r in map(json.loads, path.read_text().splitlines())}, runtime, path

    def test_runner_writes_every_condition_with_reuse_rules(self):
        frame = items([{"item_id": "e1", "text": SAD}, {"item_id": "c1", "text": CRISIS}])
        records, _, path = self.generate(frame)
        self.assertEqual(len(records), 12)
        self.assertEqual(records[("e1", "E5")]["e5"]["status"], "applied")
        self.assertEqual(records[("c1", "E5")]["reused_from"], "E3")    # safety gate: the KG-only reply
        self.assertEqual(records[("c1", "E5")]["e5"]["stage"], "safety")
        self.assertEqual(records[("c1", "E4")]["status"], "applied")    # E4 (mode always) is unchanged: it steers crisis
        self.assertEqual(e5.plan_mismatches(records.values()), [])
        before = path.read_text()
        svc, _ = selective_service()
        e5.run_heldout(svc, frame, path, e5.Progress("test", len(frame)), retry_failed=False)   # resume: nothing to do
        self.assertEqual(path.read_text(), before)

    def test_judge_cache_resumes_without_recomputing(self):
        judge = e5.CachedJudge("judge1", "fake", False, self.base / "cache.jsonl")
        judge._judge = FakeJudge()
        first = judge.rate(["c"] * 3, ["a", "b", "a"], ["r"] * 3)
        self.assertEqual(judge._judge.calls, 2)
        again = e5.CachedJudge("judge1", "fake", False, self.base / "cache.jsonl")
        again._judge = FakeJudge()
        np.testing.assert_array_equal(again.rate(["c"] * 3, ["a", "b", "a"], ["r"] * 3), first)
        self.assertEqual(again._judge.calls, 0)
        self.assertEqual(list(again.prefer(["c", "c"], ["x", "y"], ["x", "z"])), [0.5, 0.6])

    def test_report_and_verdict_run_end_to_end(self):
        frame = items([{"item_id": f"e{i}", "text": SAD if i % 2 else "I passed my exam and I feel proud and happy today"}
                       for i in range(8)] + [{"item_id": "c1", "text": CRISIS}])
        records, _, path = self.generate(frame)
        folder = self.base / "heldout" / TAG
        folder.mkdir(parents=True)
        (folder / "generations.jsonl").write_text(path.read_text())
        with patch.object(e5, "heldout_items", lambda: frame), patch.object(e5, "heldout_dir", lambda tag: folder):
            table = e5.long_table(TAG)
        self.assertIn("E4_batch0", set(table.condition))
        table["lang3"] = table.text.map(language_of)
        table["language_match"], table["indep_target"] = 1.0, 0.1
        table["reply_words"], table["degenerate"] = table.reply.str.split().str.len(), False
        table.drop(columns=["history"]).to_csv(folder / "replies.csv", index=False)
        conditions = table[table.condition.isin(e5.CONDITIONS)]
        wide = conditions.pivot(index="item_id", columns="condition", values="reply")
        for name in ("judge1", "judge2"):
            judge = e5.CachedJudge(name, "fake", False, folder / f"cache_{name}.jsonl")
            judge._judge = FakeJudge()
            e5.rate_rows(judge, table, "t").to_csv(folder / f"ratings_{name}.csv", index=False)
            preferences = pd.DataFrame(index=wide.index)
            for first, second in (*e5.PRIMARY_PAIRS, *e5.SECONDARY_PAIRS):
                preferences[f"{first}_over_{second}"] = judge.prefer(["c"] * len(wide), wide[first].tolist(), wide[second].tolist())
            preferences.to_csv(folder / f"preferences_{name}.csv")
        from generation_analysis import e5_report
        summary = e5_report.report(TAG, self.base)
        self.assertEqual(len(summary["registered"]), 8)
        h1 = next(row for row in summary["registered"] if row["hypothesis"] == "H1")
        self.assertGreater(h1["mean"], 0)                              # steered replies were rated warmer
        self.assertGreater(summary["coverage"]["E5"]["retained_rate"], 0)
        self.assertEqual(summary["coverage"]["E5"]["stages"].get("safety"), 1)
        out = e5_report.verdict(self.base)
        self.assertIn(TAG, out["verdicts"])
        self.assertTrue((self.base / "report.md").exists())
        e5_report.examples(self.base)
        self.assertIn("successes", (self.base / "examples.md").read_text())


class AuthoredDataTests(unittest.TestCase):
    def test_authored_messages_are_well_formed(self):
        path = Path("data/e5_authored_messages.csv")
        if not path.exists():
            self.skipTest("authored messages not written yet")
        frame = pd.read_csv(path)
        self.assertTrue({"text", "language", "category", "label", "split"} <= set(frame.columns))
        self.assertFalse(frame.text.str.lower().str.strip().duplicated().any())
        self.assertTrue(frame.split.isin(["dev", "heldout"]).all())
        self.assertTrue(frame.language.isin(["english", "hinglish", "hindi"]).all())
        hindi = frame[frame.language == "hindi"]
        self.assertTrue(hindi.text.map(language_of).eq("hindi").all(), hindi.text[hindi.text.map(language_of) != "hindi"].tolist())


if __name__ == "__main__":
    unittest.main()
