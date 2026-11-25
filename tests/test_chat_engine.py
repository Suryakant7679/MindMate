"""One stored-conversation turn: streaming, steering record, unsteered baseline, cancellation and titles."""
import threading
import unittest

from backend.chat_engine import TurnEngine, clean_model_title, fallback_title, make_title
from backend.config import MODEL_REGISTRY
from backend.generation_control import GenerationControl, current, use
from chatbot_feature.session import SessionStore
from tests.test_emotion_service import CRISIS, HINGLISH, PLAIN, SAD, STEERED, service

SPEC = MODEL_REGISTRY["llama"]
PANELS = {"love": {8: ([1, 2], [0.1, 0.2]), 9: ([3], [0.3])}}


def run(svc, runtime, text=SAD, steering="policy", control=None, events=None):
    events = events if events is not None else []
    return TurnEngine(SessionStore()).run(
        runtime=runtime, service=svc, spec=SPEC, owner="owner", conversation_key="conv",
        messages=[{"role": "user", "content": text}], steering_mode=steering, control=control or GenerationControl(),
        emit=lambda event, data: events.append((event, data)))


def deltas(events):
    return [data["text"] for event, data in events if event == "delta"]


class EngineTests(unittest.TestCase):
    def test_steered_reply_is_recorded_with_panel_size_and_layers(self):
        svc, runtime = service(0.05, 0.40)
        runtime._emotion_panels = PANELS
        runtime.steering_location = "input of model.layers[L].mlp.down_proj, decode steps only"
        events = []
        result = run(svc, runtime, events=events)
        self.assertEqual(result["reply"], STEERED)
        self.assertEqual("".join(deltas(events)), STEERED)         # the gate's choice, sent after the decision
        steering = result["steering"]
        self.assertEqual((steering["status"], steering["applied"], steering["applied_alpha"]), ("applied", True, 0.75))
        self.assertEqual((steering["panel"], steering["neurons"], steering["layers"]), ("love", 3, [8, 9]))
        self.assertFalse(result["cancelled"])

    def test_unsteered_reply_streams_token_by_token(self):
        svc, runtime = service(0.05, 0.90)                      # Hinglish: the policy does not steer
        events = []
        result = run(svc, runtime, text=HINGLISH, events=events)
        self.assertEqual(runtime.calls, ["stream"])
        self.assertEqual(len(deltas(events)), 7)                  # one event per generated token
        self.assertEqual("".join(deltas(events)), PLAIN)
        self.assertEqual(result["reply"], PLAIN)
        self.assertIsNotNone(svc.analysis(result["turn_id"]))

    def test_baseline_switch_plans_the_same_strategy_without_steering(self):
        svc, runtime = service(0.05, 0.40)
        result = run(svc, runtime, steering="off")
        self.assertEqual(result["reply"], PLAIN)
        self.assertEqual(result["steering"]["planned_alpha"], 0.0)
        self.assertEqual(result["steering"]["disabled_by"], "baseline")
        self.assertEqual(result["plan"]["steering_mode"], "off")
        self.assertIsNotNone(result["plan"]["strategy"])             # the plan itself is unchanged
        self.assertEqual(runtime.calls, ["stream"])                  # streamed, no steered candidate

    def test_crisis_turn_is_planned_but_never_steered(self):
        svc, runtime = service(0.05, 0.40)
        result = run(svc, runtime, text=CRISIS)
        self.assertTrue(result["plan"]["safety_flag"])
        self.assertEqual(result["steering"]["applied_alpha"], 0.0)
        self.assertEqual(result["steering"]["disabled_by"], "safety")
        self.assertNotIn([0.0, 0.75], runtime.calls)

    def test_stop_keeps_only_the_text_already_shown(self):
        svc, runtime = service(0.05, 0.90)
        control = GenerationControl()
        events = []

        def emit(event, data):
            events.append((event, data))
            if len(deltas(events)) == 3:
                control.cancel()                               # the user presses Stop after three tokens
        result = TurnEngine(SessionStore()).run(
            runtime=runtime, service=svc, spec=SPEC, owner="o", conversation_key="conv",
            messages=[{"role": "user", "content": HINGLISH}], steering_mode="policy", control=control, emit=emit)
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["reply"], "plain reply that")
        self.assertEqual(len(deltas(events)), 3)
        self.assertIsNone(result["turn_id"])
        self.assertEqual(svc.research.turns("conv"), [])           # no research record of a stopped reply

    def test_stop_during_batched_candidates_releases_nothing(self):
        svc, runtime = service(0.05, 0.40)
        control = GenerationControl()
        original = runtime.generate_candidates

        def cancelling(*args, **kwargs):
            self.assertIs(current(), control)                     # the request's control reaches the runtime
            control.cancel()
            return original(*args, **kwargs)
        runtime.generate_candidates = cancelling
        events = []
        result = run(svc, runtime, control=control, events=events)
        self.assertTrue(result["cancelled"])
        self.assertEqual((result["reply"], deltas(events)), ("", []))

    def test_model_without_emotion_reasoner_streams_plainly(self):
        class Plain:
            def stream(self, messages, **options):
                self.options = options
                yield from ("plain", " answer")
        runtime = Plain()
        events = []
        result = run(None, runtime, events=events)
        self.assertEqual((result["reply"], deltas(events)), ("plain answer", ["plain", " answer"]))
        self.assertEqual(result["steering"]["status"], "not_run")
        self.assertTrue(runtime.options["deterministic"])


class TitleTests(unittest.TestCase):
    def test_model_title_is_cleaned_or_replaced(self):
        self.assertEqual(clean_model_title('Title: "coping with exam stress."'), "Coping with exam stress")
        self.assertIsNone(clean_model_title("I'm sorry, I cannot help with that"))
        self.assertIsNone(clean_model_title(""))
        self.assertIsNone(clean_model_title("word " * 30))
        self.assertEqual(fallback_title("I keep lying awake at night worrying. Help?"), "I keep lying awake at night…")

    def test_crisis_first_message_gets_a_neutral_title(self):
        class Runtime:
            def generate(self, *args, **kwargs):
                raise AssertionError("no model call for a sensitive first message")
        self.assertEqual(make_title(Runtime(), "something", sensitive=True), ("Support conversation", "fallback"))


class ControlTests(unittest.TestCase):
    def test_control_is_thread_local_to_the_request(self):
        control, seen = GenerationControl(), []
        with use(control):
            worker = threading.Thread(target=lambda: seen.append(current()))
            worker.start()
            worker.join()
            self.assertIs(current(), control)
        self.assertEqual(seen, [None])          # other threads do not inherit it unless copied explicitly
        self.assertIsNone(current())


if __name__ == "__main__":
    unittest.main()
