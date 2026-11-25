"""Real Hugging Face generation on the CPU with a tiny random Llama and a cached real tokenizer.

Checks what mocks cannot: steering hooks are registered only under the runtime lock (so concurrent requests never
share steering), cancellation stops ``model.generate`` mid-reply in every code path, progress counts decode steps,
an unused control leaves greedy output unchanged, and context fitting holds with real chat templates.

Skipped when transformers or a cached Llama-family tokenizer is unavailable.
"""
import threading
import unittest

from backend.config import EMOLLAMA_CHAT_TEMPLATE, MODEL_REGISTRY
from backend.context import fit, normalize, prompt_budget, token_counter
from backend.generation_control import GenerationControl, use
from backend.models import is_downloaded

TOKENIZER_IDS = ("meta-llama/Llama-2-13b-chat-hf", "lzw1008/Emollama-chat-7b", "meta-llama/Llama-3.1-8B-Instruct")


def tiny_runtime():
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    from backend.runtime import HuggingFaceRuntime

    tokenizer_id = next(model_id for model_id in TOKENIZER_IDS if is_downloaded(model_id))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = EMOLLAMA_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=512,
                         bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = None          # random weights: never stop early, so lengths are exact
    runtime = HuggingFaceRuntime.__new__(HuggingFaceRuntime)
    runtime.torch, runtime.tokenizer, runtime.model = torch, tokenizer, model
    runtime.model_id, runtime.results_tag, runtime.enable_thinking = "tiny-random-llama", None, None
    runtime.lock = threading.Lock()
    runtime.context_tokens = 512
    # A synthetic panel with large shifts, so a steered reply visibly differs from the unsteered one.
    runtime._emotion_panels = {"love": {0: ([1, 5, 9, 40], [30.0, -30.0, 30.0, 30.0]),
                                        1: ([2, 3, 64, 100], [30.0, 30.0, -30.0, 30.0])}}
    return runtime


MESSAGES = [{"role": "user", "content": "I had a long and difficult day at work."}]


@unittest.skipUnless(any(is_downloaded(model_id) for model_id in TOKENIZER_IDS), "no cached Llama tokenizer")
class RealGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.runtime = tiny_runtime()
        except ImportError as exc:          # pragma: no cover
            raise unittest.SkipTest(str(exc))

    def generate(self, **options):
        return self.runtime.generate(MESSAGES, max_new_tokens=12, deterministic=True, temperature=0.7,
                                     system_prompt=None, **options)

    def test_unused_control_leaves_greedy_output_unchanged_and_counts_steps(self):
        baseline = self.generate()["response"]
        control = GenerationControl()
        with use(control):
            controlled = self.generate()
        self.assertEqual(controlled["response"], baseline)
        self.assertEqual(controlled["generated_tokens"], 12)
        self.assertEqual(control.steps, 12)

    def test_cancellation_stops_generate_mid_reply(self):
        control = GenerationControl(on_progress=lambda steps: control.cancel() if steps >= 3 else None)
        with use(control):
            result = self.runtime.generate(MESSAGES, max_new_tokens=200, deterministic=True, temperature=0.7,
                                           system_prompt=None)
        self.assertTrue(control.cancelled)
        self.assertLessEqual(result["generated_tokens"], 4)

    def test_cancellation_reaches_the_stream_worker_thread(self):
        control = GenerationControl(on_progress=lambda steps: control.cancel() if steps >= 2 else None)
        with use(control):
            pieces = list(self.runtime.stream(MESSAGES, max_new_tokens=200, deterministic=True, temperature=0.7,
                                              system_prompt=None))
        self.assertTrue(control.cancelled)
        self.assertLessEqual(control.steps, 3)
        self.assertLess(len("".join(pieces)), 200)

    def test_cancellation_stops_batched_candidates(self):
        control = GenerationControl(on_progress=lambda steps: control.cancel() if steps >= 2 else None)
        with use(control):
            rows = self.runtime.generate_candidates(MESSAGES, max_new_tokens=200, system_prompt=None,
                                                    steering_emotion="love", alphas=[0.0, 1.0])
        self.assertTrue(control.cancelled)
        self.assertTrue(all(row["generated_tokens"] <= 3 for row in rows))

    def test_steering_hooks_are_registered_only_under_the_runtime_lock(self):
        runtime = self.runtime
        unlocked = []
        original = runtime.steering_module

        def watched(layer):
            module = original(layer)
            register = module.register_forward_pre_hook

            def checked(hook, *args, **kwargs):
                if not runtime.lock.locked():
                    unlocked.append(layer)
                return register(hook, *args, **kwargs)
            module.register_forward_pre_hook = checked
            return module
        runtime.steering_module = watched
        try:
            runtime.generate_candidates(MESSAGES, max_new_tokens=4, system_prompt=None, steering_emotion="love",
                                        alphas=[0.0, 1.0])
            runtime.generate(MESSAGES, max_new_tokens=4, deterministic=True, temperature=0.7, system_prompt=None,
                             steering_alpha=1.0, steering_emotion="love")
        finally:
            runtime.steering_module = original
            for layer in (0, 1):
                original(layer).__dict__.pop("register_forward_pre_hook", None)
        self.assertEqual(unlocked, [])
        for layer in (0, 1):
            self.assertEqual(len(original(layer)._forward_pre_hooks), 0)      # every hook removed

    def test_concurrent_steered_and_unsteered_requests_do_not_interfere(self):
        baseline = self.generate()["response"]
        steered = self.generate(steering_alpha=1.0, steering_emotion="love")["response"]
        self.assertNotEqual(steered, baseline)          # the panel visibly changes the output
        plain_outputs, errors = [], []

        def steering_user():
            try:
                for _ in range(15):
                    self.runtime.generate_candidates(MESSAGES, max_new_tokens=12, system_prompt=None,
                                                     steering_emotion="love", alphas=[1.0])
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        def plain_user():
            try:
                for _ in range(15):
                    plain_outputs.append(self.generate()["response"])
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=steering_user), threading.Thread(target=plain_user)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        self.assertEqual(errors, [])
        self.assertEqual(set(plain_outputs), {baseline})

    def test_context_fit_with_a_real_tokenizer_and_template(self):
        history = []
        for index in range(60):
            history += [{"role": "user", "content": f"Message {index}: " + "I keep thinking about it. " * 6},
                        {"role": "assistant", "content": f"Reply {index}: " + "That sounds hard to carry. " * 6}]
        history.append({"role": "user", "content": "What do you remember?"})
        budget = prompt_budget(self.runtime, 128)
        kept, info = fit(normalize(history), count=token_counter(self.runtime), budget=budget)
        self.assertTrue(info["trimmed"])
        self.assertEqual(kept[-1]["content"], "What do you remember?")
        templated = self.runtime.tokenizer.apply_chat_template(kept, tokenize=True, add_generation_prompt=True)
        self.assertLessEqual(len(templated), budget)
        result = self.runtime.generate(kept, max_new_tokens=4, deterministic=True, temperature=0.7, system_prompt=None)
        self.assertEqual(result["generated_tokens"], 4)

    def test_registry_chat_templates_render_alternating_history(self):
        from transformers import AutoTokenizer
        history = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "hi"},
                   {"role": "assistant", "content": "hello"}, {"role": "user", "content": "how are you?"}]
        for key in ("llama2", "emollama", "llama"):
            spec = MODEL_REGISTRY[key]
            if not is_downloaded(spec.config.model_id):
                continue
            with self.subTest(model=key):
                tokenizer = AutoTokenizer.from_pretrained(spec.config.model_id, local_files_only=True)
                if spec.config.chat_template and not getattr(tokenizer, "chat_template", None):
                    tokenizer.chat_template = spec.config.chat_template
                text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                self.assertIn("how are you?", text)
                self.assertIn("hello", text)


if __name__ == "__main__":
    unittest.main()
