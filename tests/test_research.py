"""Emotion trajectory, steered/unsteered comparison, per-neuron records, research mode and feedback."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from backend.emotion_service import EmotionSteeringService
from backend.research import ResearchStore, load_cards
from backend.server import RequestHandler
from chatbot_feature.emotion_reasoner import EmotionReasonerAgent
from chatbot_feature.emotion_reasoner import comparison as cmp
from chatbot_feature.emotion_reasoner import trajectory as tr
from chatbot_feature.emotion_reasoner.intervention import summarize_intervention

LABELS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "neutral", "sadness", "surprise")
SWEEP = [{"alpha": 0.75, "n": 8, "target_gain": 0.3, "safe": True}]
POLICY = {"rule": {}, "ceiling": {"threshold": 0.3},
          "cells": {"love|english": {"alpha": 0.75, "reason": "safe gain", "sweep": SWEEP},
                    "love|hinglish": {"alpha": 0.0, "reason": "not steered: quality fell", "sweep": SWEEP}}}
PLAIN = "plain reply that listens to you today"
STEERED = "steered reply with warmth and care for you today"
SAD = "Everyone forgot about me today and it hurts"
HAPPY = "thank you, that helped and I feel happy now"
HINGLISH = "yaar aaj bahut akela feel ho raha hai, koi nahi hai mere saath"
CRISIS = "I want to end my life"


def trace(emotion, alpha):
    return {"emotion": emotion, "alpha": alpha, "gain": 1.75, "location": "test site", "layers": [
        {"layer": 8, "neurons": [1, 2, 3], "decode_positions": 4, "applied_shift": [0.2 * alpha, -0.1 * alpha, 0.0],
         "mean_activation_before_shift": [0.1 + 0.05 * bool(alpha), 0.3, -0.2],
         **({"write_norm": [2.0, 1.0, 1.0], "residual_update_norm": 0.5 * alpha} if alpha else {})},
        {"layer": 9, "neurons": [7], "decode_positions": 4, "applied_shift": [0.05 * alpha],
         "mean_activation_before_shift": [0.4],
         **({"write_norm": [1.0], "residual_update_norm": 0.05 * alpha} if alpha else {})}]}


class Runtime:
    _emotion_panels = {"love": {8: ([1, 2, 3], [0.2, -0.1, 0.0])}}
    model_id = "test/model"

    def __init__(self):
        self.calls = []

    def generate(self, messages, max_new_tokens, deterministic, temperature, system_prompt=None, **_):
        self.calls.append(0.0)
        return {"response": PLAIN}

    def generate_candidates(self, messages, max_new_tokens, system_prompt, steering_emotion, alphas, panel=None):
        self.calls.append(list(alphas))
        return [{"response": STEERED if alpha else PLAIN, "trace": trace(steering_emotion, alpha)} for alpha in alphas]


class Scorer:
    model_id, method, uses_classifier = "fake", "fake classifier", True

    def score(self, text):
        scores = dict.fromkeys(LABELS, 0.01)
        if text.startswith("plain"):
            scores["love"] = 0.05
        elif text.startswith("steered"):
            scores["love"] = 0.40
        elif "happy" in text:
            scores["joy"] = 0.9
        else:
            scores["sadness"] = 0.9
        return scores


def service(**options):
    runtime = Runtime()
    return EmotionSteeringService(runtime, EmotionReasonerAgent(["love"], policy=POLICY), Scorer(), evidence={},
                                  **options), runtime


def chat(svc, texts, conversation="conv-1"):
    """Send ``texts`` as consecutive user turns of one conversation; returns the analyses."""
    messages, analyses = [], []
    for text in texts:
        messages.append({"role": "user", "content": text})
        turn_id, _ = svc.begin_turn(list(messages), conversation_id=conversation)
        analysis = svc.respond(turn_id, 64)
        analyses.append(analysis)
        messages.append({"role": "assistant", "content": analysis["reply"]})
    return analyses


class TrajectoryTests(unittest.TestCase):
    def state(self, emotion, probability, text="hello"):
        rest = (1 - probability) / 9
        fused = {label: (probability if label == emotion else rest) for label in LABELS}
        return tr.turn_state(text, {"fused": fused, "emotion": emotion, "confidence": probability, "intensity": 0.6,
                                    "valence": tr.EMOTIONS[emotion][0]}, fused)

    def test_languages(self):
        self.assertEqual(tr.language_of("मुझे बहुत दुख हो रहा है"), "hindi")
        self.assertEqual(tr.language_of(HINGLISH), "hinglish")
        self.assertEqual(tr.language_of(SAD), "english")

    def test_expected_valence_uses_the_whole_distribution(self):
        sure, unsure = self.state("sadness", 0.91), self.state("sadness", 0.30)
        self.assertLess(sure["expected_valence"], unsure["expected_valence"])     # an unsure turn sits nearer the middle
        self.assertLess(unsure["expected_valence"], 0)
        self.assertFalse(sure["uncertain"])
        self.assertTrue(unsure["uncertain"])
        self.assertGreater(unsure["entropy"], sure["entropy"])
        self.assertAlmostEqual(sum(sure["distribution"].values()), 1.0, places=2)

    def test_summary_tracks_changes_frequency_and_trend(self):
        turns = [self.state("sadness", 0.8), self.state("sadness", 0.7), self.state("neutral", 0.6), self.state("joy", 0.8)]
        summary = tr.summarize(turns)
        self.assertEqual(summary["frequency"], {"sadness": 2, "neutral": 1, "joy": 1})
        self.assertEqual((summary["dominant_first_half"], summary["dominant_second_half"]), ("sadness", "joy"))
        self.assertGreater(summary["mood"]["trend_per_turn"], 0)
        self.assertFalse(summary["mood"]["trend_interpretable"])                   # four turns are too few for a trend
        self.assertIsNone(summary["turns"][0]["delta_valence"])
        self.assertGreater(summary["turns"][3]["delta_valence"], 0)
        self.assertTrue(0 < summary["turns"][3]["js_divergence"] <= 1)
        self.assertAlmostEqual(sum(summary["mean_distribution"].values()), 1.0, places=2)
        self.assertAlmostEqual(summary["mood"]["label_switch_rate"], 2 / 3, places=3)
        self.assertTrue(any("not calibrated" in caveat for caveat in summary["caveats"]))

    def test_self_ratings_are_compared_with_the_estimate(self):
        turns = [self.state("sadness", 0.8), self.state("neutral", 0.6), self.state("joy", 0.8), self.state("love", 0.8)]
        report = tr.summarize(turns, {0: 1, 1: 3, 2: 4, 3: 5})["self_report"]
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["pairs"][0]["reported_valence"], -1.0)
        self.assertGreater(report["spearman"], 0.9)
        self.assertEqual(tr.summarize(turns, {0: 1})["self_report"]["spearman"], None)   # too few for a correlation

    def test_after_response_is_grouped_but_flagged_observational(self):
        turns = [self.state("sadness", 0.8), self.state("neutral", 0.6), self.state("joy", 0.8)]
        after = tr.after_response(turns, [{"delivered": "steered", "emotion": "love", "expected_valence": 0.5},
                                          {"delivered": "unsteered", "emotion": "neutral", "expected_valence": 0.0}])
        self.assertEqual(after["n"], 2)
        self.assertEqual(after["by_delivery"]["steered"]["n"], 1)
        self.assertEqual(after["by_delivery"]["unsteered"]["n"], 1)
        self.assertIn("not the effect", after["note"])


class ComparisonTests(unittest.TestCase):
    def compare(self, steered, love=0.4, user=SAD, unsteered="I hear you. That sounds really hard today."):
        return cmp.compare_replies(user, unsteered, steered, {"love": 0.1, "joy": 0.0}, {"love": love, "joy": 0.0}, "love",
                                   delivered="unsteered")

    def test_verdicts(self):
        self.assertEqual(self.compare("I hear you, and I care about you. That sounds really hard today.")["verdict"]["label"], "improved")
        self.assertEqual(self.compare("I hear you. That sounds really hard today.")["verdict"]["label"], "no_meaningful_change")
        self.assertEqual(self.compare("I hear you, that does sound hard for you today.", love=0.12)["verdict"]["label"],
                         "no_meaningful_change")
        self.assertEqual(self.compare("so so so so so so so so so so so much love", love=0.9)["verdict"]["label"], "harmed")
        self.assertEqual(self.compare("ok fine", love=0.9)["verdict"]["label"], "harmed")
        self.assertEqual(self.compare("I hear you. That sounds really hard today, friend.", love=0.01)["verdict"]["label"], "harmed")

    def test_hinglish_reply_that_turns_english_is_harm(self):
        result = self.compare("I am here for you and I care about how you feel today.", user=HINGLISH,
                              unsteered="Yaar main samajh sakta hoon, aaj bahut akela lag raha hai tumhe.")
        self.assertEqual(result["verdict"]["label"], "harmed")
        self.assertFalse(result["automatic"]["hinglish_kept"])

    def test_no_candidate_is_not_compared(self):
        result = cmp.compare_replies(SAD, PLAIN, None, {"love": 0.1}, None, "love", delivered="unsteered")
        self.assertFalse(result["available"])
        self.assertEqual(result["verdict"]["label"], "not_compared")

    def test_judged_results_take_precedence_and_keep_the_automatic_verdict(self):
        base = self.compare("I hear you, and I care about you. That sounds really hard today.")
        judged = {"difference": {"judge1": {"emotion": 0.6}, "judge2": {"emotion": 0.3},
                                 "both": {"emotion": 0.45, "empathy": 0.1, "relevance": -0.9, "fluency": 0.0}},
                  "preference": {"both": 0.6}}
        merged = cmp.with_judged(base, judged)
        self.assertEqual((merged["verdict"]["label"], merged["verdict"]["basis"]), ("harmed", "judged"))
        self.assertEqual(merged["automatic_verdict"]["label"], "improved")
        judged["difference"]["both"]["relevance"] = -0.1
        self.assertEqual(cmp.with_judged(base, judged)["verdict"]["label"], "improved")
        judged["difference"]["judge2"]["emotion"] = -0.2          # the judges disagree on the gain
        self.assertEqual(cmp.with_judged(base, judged)["verdict"]["label"], "no_meaningful_change")

    def test_lexical_overlap(self):
        self.assertEqual(cmp.lexical_overlap("a b c d", "a b c d"), 1.0)
        self.assertEqual(cmp.lexical_overlap("a b c d", "w x y z"), 0.0)
        self.assertAlmostEqual(cmp.lexical_overlap("a b c d", "a b x d"), 0.75)


class InterventionRecordTests(unittest.TestCase):
    def test_per_neuron_records_and_contributions(self):
        cards = {(8, 1): {"attribution_rank": 1}}
        summary = summarize_intervention(trace("love", 1.0), trace("love", 0.0), cards=cards, detail=True)
        first = summary["neurons"][0]
        self.assertEqual((first["layer"], first["neuron"], first["rank"]), (8, 1, 1))     # |0.2| x write norm 2 is the largest
        self.assertEqual(first["evidence"], {"attribution_rank": 1})
        self.assertAlmostEqual(first["steered_activation"], 0.35)                          # 0.15 before + 0.2 shift
        self.assertAlmostEqual(first["unsteered_activation"], 0.10)
        self.assertAlmostEqual(first["measured_change"], 0.25)
        self.assertAlmostEqual(first["upstream_change"], 0.05)                             # moved before its own shift
        self.assertEqual(first["direction"], "raised")
        self.assertEqual(summary["neurons"][1]["direction"], "lowered")
        self.assertAlmostEqual(sum(entry["contribution_share"] for entry in summary["neurons"]), 1.0, places=3)
        layer = summary["layers"][0]
        self.assertEqual((layer["neurons_raised"], layer["neurons_lowered"]), (1, 1))
        self.assertAlmostEqual(layer["residual_update_norm"], 0.5)
        self.assertIsNotNone(layer["normalized_shift"])
        self.assertAlmostEqual(sum(row["contribution_share"] for row in summary["layers"]), 1.0, places=3)
        self.assertEqual(len(summary["heatmap"]["rows"]), 2)
        self.assertEqual(summary["heatmap"]["rows"][0]["cells"][0], {"neuron": 1, "shift": 0.2, "change": 0.25})
        self.assertIn("write vector", summary["contribution_basis"])
        self.assertNotIn("neurons", summarize_intervention(trace("love", 1.0), trace("love", 0.0)))

    def test_cards_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cards.json"
            path.write_text(json.dumps({"panels": {"love": {"neurons": {"8:1": {"attribution_rank": 3}}}}}))
            self.assertEqual(load_cards(path), {"love": {(8, 1): {"attribution_rank": 3}}})
            self.assertEqual(load_cards(Path(folder) / "missing.json"), {})


class ConversationTests(unittest.TestCase):
    def test_every_turn_carries_the_conversation_trajectory(self):
        svc, _ = service()
        first, second = chat(svc, [SAD, HAPPY])
        self.assertEqual(len(first["trajectory"]["turns"]), 1)
        self.assertEqual(len(second["trajectory"]["turns"]), 2)
        self.assertEqual([turn["emotion"] for turn in second["trajectory"]["turns"]], ["sadness", "joy"])
        self.assertGreater(second["trajectory"]["turns"][1]["delta_valence"], 0)
        after = second["trajectory"]["after_response"]
        self.assertEqual(after["n"], 1)
        self.assertEqual(after["rows"][0]["delivered"], "steered")
        self.assertEqual(second["conversation_id"], "conv-1")
        self.assertEqual(second["turn_index"], 1)
        responses = second["trajectory"]["responses"]
        self.assertEqual(responses[0]["status"], "applied")
        self.assertEqual(responses[0]["verdict"], "improved")
        self.assertAlmostEqual(responses[0]["target_gain"], 0.35)

    def test_summary_counts_decisions_and_verdicts(self):
        svc, _ = service()
        analyses = chat(svc, [SAD, HINGLISH, HAPPY])
        summary = analyses[-1]["summary"]
        self.assertEqual(summary["turns"], 3)
        self.assertEqual(summary["decisions"]["applied"], 1)
        self.assertEqual(summary["comparison"]["compared"], 1)                 # only the steered turn has two replies
        self.assertEqual(summary["comparison"]["counts"]["improved"], 1)
        self.assertEqual(summary["by_language"]["hinglish"]["turns"], 1)
        self.assertEqual(summary["by_language"]["hinglish"]["steered_delivered"], 0)
        self.assertEqual(svc.conversation_view("conv-1")["summary"], summary)

    def test_research_mode_compares_unsteered_turns_without_sending_the_candidate(self):
        svc, runtime = service(research_compare=True)
        analysis = chat(svc, [HINGLISH])[0]
        self.assertEqual(analysis["decision"]["status"], "not_planned")
        self.assertEqual(analysis["decision"]["disabled_by"], "policy")
        self.assertEqual(analysis["decision"]["shadow_alpha"], svc.shadow_alpha)
        self.assertEqual(analysis["reply"], PLAIN)                             # the shadow candidate is never sent
        self.assertEqual(analysis["effect"]["steered_reply"], STEERED)
        self.assertTrue(analysis["effect"]["shadow"])
        self.assertEqual(analysis["effect"]["sent"], "unsteered")
        self.assertTrue(analysis["comparison"]["available"])
        self.assertTrue(analysis["comparison"]["shadow"])
        self.assertTrue(analysis["steering"]["applied"])
        self.assertFalse(analysis["steering"]["sent"])
        self.assertEqual(runtime.calls, [[0.0, svc.shadow_alpha]])

    def test_research_mode_never_generates_a_candidate_on_a_crisis_turn(self):
        svc, runtime = service(research_compare=True)
        analysis = chat(svc, [CRISIS])[0]
        self.assertEqual(analysis["decision"]["disabled_by"], "safety")
        self.assertIsNone(analysis["effect"]["steered_reply"])
        self.assertEqual(runtime.calls, [0.0])

    def test_without_research_mode_an_unsteered_turn_is_not_compared(self):
        svc, runtime = service()
        analysis = chat(svc, [HINGLISH])[0]
        self.assertFalse(analysis["comparison"]["available"])
        self.assertEqual(runtime.calls, [0.0])

    def test_feedback_enters_the_trajectory(self):
        svc, _ = service()
        chat(svc, [SAD, HAPPY])
        view = svc.add_feedback("conv-1", 0, mood_rating=1, helpfulness=4)
        self.assertEqual(view["trajectory"]["turns"][0]["self_rating"], 1)
        self.assertEqual(view["trajectory"]["self_report"]["n"], 1)
        self.assertEqual(view["summary"]["helpfulness"], {"n": 1, "mean": 4.0})
        self.assertEqual(view["trajectory"]["responses"][0]["helpfulness"], 4)

    def test_a_regenerated_turn_replaces_the_later_ones(self):
        svc, _ = service()
        chat(svc, [SAD, HAPPY])
        analysis = chat(svc, [HINGLISH])[0]                                    # same conversation id, restarted
        self.assertEqual(len(analysis["trajectory"]["turns"]), 1)

    def test_unknown_earlier_turns_are_reconstructed_from_the_messages(self):
        svc, _ = service()
        messages = [{"role": "user", "content": SAD}, {"role": "assistant", "content": PLAIN},
                    {"role": "user", "content": HAPPY}]
        turn_id, _ = svc.begin_turn(messages)                                   # no conversation id
        analysis = svc.respond(turn_id, 64)
        self.assertEqual(len(analysis["trajectory"]["turns"]), 2)
        self.assertIsNone(analysis["trajectory"]["responses"][0]["status"])     # not tracked: no decision is invented
        self.assertEqual(analysis["summary"]["tracked_turns"], 1)

    def test_neuron_record_and_telemetry_files(self):
        with tempfile.TemporaryDirectory() as folder:
            svc, _ = service(research=ResearchStore("test", directory=folder, persist=True))
            analysis = chat(svc, [SAD])[0]
            record = svc.neuron_record(analysis["turn_id"])
            self.assertEqual(len(record["neurons"]), 4)
            self.assertEqual(record["panel"], "love")
            self.assertTrue(record["delivered"])
            lines = (Path(folder) / "turns.jsonl").read_text().splitlines()
            saved = json.loads(lines[0])
            self.assertEqual(saved["replies"]["delivered"], "steered")
            self.assertEqual(saved["user"]["emotion"], "sadness")
            self.assertNotIn("neuron_ids", saved["steering"]["layers"][0])
            self.assertTrue((Path(folder) / "neurons" / f"{analysis['turn_id']}.json").exists())
            # Offline judge results written later are picked up for that turn.
            (Path(folder) / "judged.jsonl").write_text(json.dumps({
                "turn_id": analysis["turn_id"], "preference": {"both": 0.3},
                "difference": {"judge1": {"emotion": 0.5}, "judge2": {"emotion": 0.4},
                               "both": {"emotion": 0.45, "fluency": -1.0}}}) + "\n")
            view = svc.trace(analysis["turn_id"])
            self.assertEqual((view["comparison"]["verdict"]["label"], view["comparison"]["verdict"]["basis"]), ("harmed", "judged"))
            self.assertEqual(view["summary"]["comparison"]["counts"]["harmed"], 1)

    def test_agent_turns_are_tracked_like_direct_turns(self):
        svc, _ = service()
        messages = [{"role": "user", "content": SAD}]
        plan = svc.agent.plan(SAD, svc.probe_scores(SAD)).to_dict()
        analysis = svc.respond_with_plan(messages, "prompt", {**plan, "user_text": SAD}, 64)
        result = {"response": analysis["reply"], "emotion_plan": plan,
                  "steering": {**analysis["steering"], "decision": analysis["decision"], "effect": analysis["effect"],
                               "comparison": analysis["comparison"], "turn_id": analysis["turn_id"], "run_id": "r"}}
        tracked = svc.track_agent_turn(messages, result, "agent:u:s")
        self.assertEqual(tracked["trajectory"]["responses"][0]["mode"], "agent")
        self.assertEqual(tracked["trajectory"]["responses"][0]["status"], "applied")
        tool = svc.track_agent_turn(messages, {"response": "The weather is fine.", "emotion_plan": None}, "agent:u:t")
        self.assertEqual(tool["decision"]["disabled_by"], "agent")
        self.assertFalse(tool["comparison"]["available"])


class EndpointTests(unittest.TestCase):
    OWNER = "a" * 64          # the signed session cookie's token (Agent 1 scopes records to their owner)

    def handler(self, svc, path, payload=None):
        import hashlib
        import hmac
        from chatbot_feature.session import SessionStore
        handler = RequestHandler.__new__(RequestHandler)
        handler.path = path
        if not hasattr(self, "server"):
            self.server = Mock()
            self.server.session_secret = b"0" * 32
            self.server.resource_owners = SessionStore()
            self.server.resource_owners.put("conv-1", self.OWNER)
        handler.server = self.server
        signature = hmac.new(self.server.session_secret, self.OWNER.encode(), hashlib.sha256).hexdigest()
        handler.headers = {"Cookie": f"mindmate_session={self.OWNER}.{signature}"}
        handler._emotion_service = lambda: svc
        handler.send_json = Mock()
        if payload is not None:
            body = json.dumps(payload).encode()
            handler.headers = {**handler.headers, "Content-Length": str(len(body))}
            handler.rfile = io.BytesIO(body)
            handler.do_POST()
        else:
            handler.do_GET()
        return handler.send_json.call_args.args

    def test_conversation_feedback_settings_and_overview(self):
        svc, _ = service()
        analysis = chat(svc, [SAD, HAPPY])[-1]
        status, view = self.handler(svc, "/api/emotion/conversation/conv-1")
        self.assertEqual((status, len(view["trajectory"]["turns"])), (200, 2))
        self.assertEqual(self.handler(svc, "/api/emotion/conversation/unknown")[0], 404)
        status, view = self.handler(svc, "/api/emotion/feedback", {"conversation_id": "conv-1", "turn_index": 1, "mood_rating": 4})
        self.assertEqual((status, view["trajectory"]["turns"][1]["self_rating"]), (200, 4))
        for bad in ({"conversation_id": "conv-1", "turn_index": 1, "mood_rating": 9},
                    {"conversation_id": "../x", "turn_index": 1, "mood_rating": 3},
                    {"conversation_id": "conv-1", "turn_index": 1}):
            self.assertEqual(self.handler(svc, "/api/emotion/feedback", bad)[0], 400)
        self.assertEqual(self.handler(svc, "/api/emotion/feedback", {"conversation_id": "conv-1", "turn_index": 7, "helpfulness": 3})[0], 404)
        status, settings = self.handler(svc, "/api/emotion/settings", {"research_compare": True})
        self.assertEqual((status, settings["research_compare"], svc.research_compare), (200, True, True))
        self.assertEqual(self.handler(svc, "/api/emotion/settings", {"research_compare": "yes"})[0], 400)
        status, overview = self.handler(svc, "/api/emotion/research")
        self.assertEqual(status, 200)
        self.assertFalse(overview["settings"]["allow_overrides"])
        self.assertEqual(overview["policy"]["cells"][0]["context"], "love|english")
        self.server.resource_owners.put(analysis["turn_id"], self.OWNER)      # as the chat endpoint records it
        status, trace_view = self.handler(svc, f"/api/emotion/trace/{analysis['turn_id']}")
        self.assertEqual((status, trace_view["turn_index"]), (200, 1))
        other = chat(svc, [SAD], "conv-2")[0]["turn_id"]
        self.assertEqual(self.handler(svc, f"/api/emotion/neurons/{other}")[0], 404)      # not this owner's turn
        self.server.resource_owners.put(other, self.OWNER)
        status, record = self.handler(svc, f"/api/emotion/neurons/{other}")
        self.assertEqual((status, record["panel"]), (200, "love"))


if __name__ == "__main__":
    unittest.main()
