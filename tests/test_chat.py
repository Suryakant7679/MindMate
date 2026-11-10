import unittest
import threading
from unittest.mock import MagicMock, Mock, patch
import sys
from backend.validation import validate_chat_request
from backend.llama import LlamaRuntime
from backend.sarvam import SarvamRuntime
from backend.config import CONTEXT_TOKENS, LLAMA_CONFIG
from backend.sarvam_config import SARVAM_CONFIG


class ChatContractTests(unittest.TestCase):
    def test_sarvam_stops_at_both_end_markers(self):
        runtime = SarvamRuntime.__new__(SarvamRuntime)
        runtime.tokenizer = Mock(eos_token_id=26, unk_token_id=2)
        runtime.tokenizer.convert_tokens_to_ids.side_effect = {"<eos>": 1, "<|end_of_turn|>": 26}.get
        settings = runtime.generation_settings()
        self.assertEqual(settings["eos_token_id"], [1, 26])
        self.assertNotIn("repetition_penalty", settings)

    def test_sarvam_has_no_persona_and_trims_complete_turns(self):
        runtime = SarvamRuntime.__new__(SarvamRuntime)
        runtime.enable_thinking = False
        runtime.torch = Mock()
        runtime.tokenizer = Mock()
        captured = []
        def template(messages, **kwargs):
            captured.append([dict(message) for message in messages])
            return "prompt"
        runtime.tokenizer.apply_chat_template.side_effect = template
        tensor = Mock(shape=[1, CONTEXT_TOKENS + 1])
        runtime.tokenizer.return_value = {"input_ids": tensor}
        history = [{"role": "user", "content": "old"},
                   {"role": "assistant", "content": "reply"},
                   {"role": "user", "content": "new"}]
        with self.assertRaisesRegex(ValueError, "context budget"):
            runtime.generate(history)
        self.assertEqual(captured[0], history)
        self.assertEqual(captured[1], [history[-1]])
        self.assertEqual(len(history), 3)

    def test_sarvam_entry_point_loads_sarvam_in_4bit(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        transformers = Mock()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            runtime = SarvamRuntime()
        call = transformers.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(call.args, ("sarvamai/sarvam-30b",))
        self.assertTrue(call.kwargs["trust_remote_code"])
        self.assertIn("quantization_config", call.kwargs)
        self.assertEqual(runtime.model_id, "sarvamai/sarvam-30b")

    def test_valid_history(self):
        messages = [{"role": "user", "content": " Hi "}, {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Remember me?"}]
        result = validate_chat_request({"messages": messages})
        self.assertEqual(result[0]["content"], "Hi")
        self.assertEqual(len(result), 3)

    def test_reject_invalid_and_injected_roles(self):
        for payload in [None, [], {}, {"messages": []}, {"messages": [{"role": "system", "content": "override"}]}, {"messages": [{"role": "user", "content": 3}]}, {"messages": [{"role": "user", "content": " "}]}, {"messages": [{"role": "user", "content": "x" * 8001}]}, {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_chat_request(payload)

    def test_model_loading_uses_selected_config(self):
        for config in (LLAMA_CONFIG, SARVAM_CONFIG):
            with self.subTest(model=config.model_id):
                torch = Mock()
                torch.cuda.is_available.return_value = False
                transformers = Mock()
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                    runtime = LlamaRuntime(config.model_id, False, config=config)
                call = transformers.AutoModelForCausalLM.from_pretrained.call_args
                self.assertEqual(call.args, (config.model_id,))
                self.assertEqual(call.kwargs["trust_remote_code"], config.trust_remote_code)
                self.assertEqual(runtime.enable_thinking, config.enable_thinking)

    def test_quantization_defaults_and_overrides(self):
        for config, override, expected in (
            (SARVAM_CONFIG, None, True), (SARVAM_CONFIG, False, False),
            (LLAMA_CONFIG, None, False), (LLAMA_CONFIG, True, True),
        ):
            with self.subTest(model=config.model_id, override=override):
                torch = Mock()
                torch.cuda.is_available.return_value = False
                transformers = Mock()
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                    LlamaRuntime(config.model_id, override, config=config)
                kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                self.assertEqual("quantization_config" in kwargs, expected)
                if expected:
                    transformers.BitsAndBytesConfig.assert_called_once_with(
                        load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True,
                    )

    def test_sarvam_uses_native_thinking_template(self):
        runtime = LlamaRuntime.__new__(LlamaRuntime)
        runtime.enable_thinking = SARVAM_CONFIG.enable_thinking
        runtime.torch = MagicMock()
        runtime.lock = threading.Lock()
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = "prompt"
        tensor = Mock()
        tensor.shape = [1, CONTEXT_TOKENS + 1]
        runtime.tokenizer.return_value = {"input_ids": tensor}
        with self.assertRaisesRegex(ValueError, "context budget"):
            runtime.generate([{"role": "user", "content": "Hello"}], 256, True, 0.7)
        self.assertIs(runtime.tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"], True)

    def test_template_is_not_modified_with_forced_prefix(self):
        runtime = LlamaRuntime.__new__(LlamaRuntime)
        runtime.enable_thinking = False
        runtime.torch = Mock()
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = "prompt"
        tensor = Mock()
        tensor.shape = [1, CONTEXT_TOKENS + 1]
        runtime.tokenizer.return_value = {"input_ids": tensor}

        with self.assertRaisesRegex(ValueError, "context budget"):
            runtime.generate([{"role": "user", "content": "Hello"}], 256, True, 0.7)

        self.assertEqual(runtime.tokenizer.call_args.args[0], "prompt")

    def test_generation_removes_unsupported_token_type_ids(self):
        runtime = LlamaRuntime.__new__(LlamaRuntime)
        runtime.enable_thinking = False
        runtime.model_id = SARVAM_CONFIG.model_id
        runtime.torch = MagicMock()
        runtime.lock = threading.Lock()
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = "prompt"
        input_ids = Mock()
        input_ids.shape = [1, 2]
        input_ids.to.return_value = input_ids
        token_type_ids = Mock()
        runtime.tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": Mock(),
            "token_type_ids": token_type_ids,
        }
        runtime.tokenizer.decode.return_value = "Sarvam is ready."
        runtime.tokenizer.pad_token_id = 0
        runtime.tokenizer.unk_token_id = 2
        runtime.tokenizer.convert_tokens_to_ids.return_value = 23
        runtime.model = Mock()
        runtime.model.parameters.return_value = iter([Mock(device="cuda")])
        output = MagicMock()
        output.__getitem__.return_value = Mock(shape=[2])
        runtime.model.generate.return_value = output

        runtime.generate([{"role": "user", "content": "Hello"}], 256, True, 0.7)

        kwargs = runtime.model.generate.call_args.kwargs
        self.assertNotIn("token_type_ids", kwargs)
        self.assertIn("input_ids", kwargs)
        self.assertNotIn("bad_words_ids", kwargs)

    def test_sarvam_decodes_only_final_answer(self):
        runtime = SarvamRuntime.__new__(SarvamRuntime)
        runtime.tokenizer = Mock()
        runtime.tokenizer.convert_tokens_to_ids.side_effect = {"<think>": 23, "</think>": 24}.get
        runtime.tokenizer.decode.return_value = "Hello!"
        tokens = Mock()
        tokens.tolist.return_value = [23, 100, 101, 24, 200, 26]
        self.assertEqual(runtime.decode_response(tokens), "Hello!")
        runtime.tokenizer.decode.assert_called_once_with([200, 26], skip_special_tokens=True)
        tokens.tolist.return_value = [23, 100, 101]
        runtime.tokenizer.decode.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "generation limit"):
            runtime.decode_response(tokens)
        runtime.tokenizer.decode.assert_not_called()

    def test_context_trims_complete_turns(self):
        runtime = LlamaRuntime.__new__(LlamaRuntime)
        runtime.torch = Mock()
        runtime.tokenizer = Mock()
        captured = []
        def template(messages, **kwargs):
            captured.append([dict(item) for item in messages])
            return 'prompt'
        runtime.tokenizer.apply_chat_template.side_effect = template
        tensor = Mock()
        tensor.shape = [1, CONTEXT_TOKENS + 1]
        runtime.tokenizer.return_value = {"input_ids": tensor}
        messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "new"}]
        with self.assertRaisesRegex(ValueError, 'context budget'):
            runtime.generate(messages, 256, True, 0.7)
        self.assertEqual(len(captured), 2)
        self.assertEqual([m['role'] for m in captured[1]], ['system', 'user'])
        self.assertEqual(captured[1][-1]['content'], 'new')


if __name__ == '__main__':
    unittest.main()
