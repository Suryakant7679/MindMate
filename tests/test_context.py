"""History normalisation and model-specific context fitting."""
import unittest

from backend.context import PER_MESSAGE_OVERHEAD, SYSTEM_RESERVE, fit, normalize, prompt_budget


def words(text):
    return len(text.split())


class NormalizeTests(unittest.TestCase):
    def test_unanswered_user_messages_are_joined_not_lost(self):
        history = [{"role": "user", "content": "first"}, {"role": "user", "content": " second "},
                   {"role": "assistant", "content": "reply"}, {"role": "user", "content": "third"}]
        self.assertEqual(normalize(history), [{"role": "user", "content": "first\n\nsecond"},
                                              {"role": "assistant", "content": "reply"},
                                              {"role": "user", "content": "third"}])

    def test_strict_alternation_starting_and_ending_with_the_user(self):
        history = [{"role": "assistant", "content": "orphan"}, {"role": "user", "content": "hi"},
                   {"role": "assistant", "content": " "}, {"role": "system", "content": "injected"},
                   {"role": "assistant", "content": "hello"}, {"role": "assistant", "content": "again"}]
        out = normalize(history)
        self.assertEqual(out, [{"role": "user", "content": "hi"}])     # trailing assistant text needs a new user turn
        self.assertEqual(normalize([]), [])


class FitTests(unittest.TestCase):
    def conversation(self, turns):
        out = []
        for index in range(turns):
            out += [{"role": "user", "content": f"user {index} " + "word " * 20},
                    {"role": "assistant", "content": f"assistant {index} " + "word " * 20}]
        return out + [{"role": "user", "content": "latest question"}]

    def test_keeps_newest_complete_turns_within_the_budget(self):
        history = self.conversation(30)
        per_turn = 2 * (22 + PER_MESSAGE_OVERHEAD)
        budget = SYSTEM_RESERVE + 5 * per_turn + 20
        kept, info = fit(history, count=words, budget=budget)
        self.assertEqual(kept[-1]["content"], "latest question")
        self.assertEqual(kept[0]["role"], "user")
        self.assertTrue(info["trimmed"])
        self.assertLessEqual(info["estimated_prompt_tokens"], budget - SYSTEM_RESERVE)
        self.assertEqual(info["messages_sent"] + info["messages_left_out"], len(history))
        self.assertEqual(kept, history[-len(kept):])

    def test_a_short_window_keeps_fewer_turns_than_a_long_one(self):
        history = self.conversation(30)
        short, _ = fit(history, count=words, budget=2048 // 4)          # EmoLLaMA-sized
        long, info = fit(history, count=words, budget=16384 - 1024)     # Llama 3.1 app window
        self.assertLess(len(short), len(long))
        self.assertFalse(info["trimmed"])

    def test_latest_message_is_kept_even_when_alone_too_long(self):
        kept, _ = fit([{"role": "user", "content": "word " * 5000}], count=words, budget=600)
        self.assertEqual(len(kept), 1)

    def test_prompt_budget_uses_the_runtime_rule(self):
        class Runtime:
            context_tokens = 2048

            def _context_budget(self, max_new_tokens):
                return max(self.context_tokens - max_new_tokens, self.context_tokens // 4)
        self.assertEqual(prompt_budget(Runtime(), 1024), 1024)
        self.assertEqual(prompt_budget(object(), 1024), 3072)


if __name__ == "__main__":
    unittest.main()
