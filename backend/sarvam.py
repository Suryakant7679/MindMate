"""Load Sarvam 30B in 4-bit and generate replies without a persona prompt."""
from .runtime import HuggingFaceRuntime
from .sarvam_config import SARVAM_CONFIG


class SarvamRuntime(HuggingFaceRuntime):
    def __init__(self, model_id=SARVAM_CONFIG.model_id, load_in_4bit=None, *, local_files_only=False):
        super().__init__(model_id, load_in_4bit, config=SARVAM_CONFIG, local_files_only=local_files_only)
        self._accept_output_dtype()

    def _accept_output_dtype(self):
        """Let 4-bit layers take the ``output_dtype`` argument of Sarvam's own down-projection.

        The checkpoint's MoE code calls ``down_proj(x, output_dtype=float32)`` on its Linear subclass.
        Quantisation replaces that layer by a bitsandbytes ``Linear4bit``, which does not know the
        argument; the result is cast instead.
        """
        import functools
        import inspect
        try:
            modules = iter(self.model.modules())
        except TypeError:
            return
        for module in modules:
            if type(module).__name__ != "Linear4bit" or "output_dtype" in inspect.signature(module.forward).parameters:
                continue
            original = module.forward

            @functools.wraps(original)
            def forward(hidden, output_dtype=None, _original=original):
                output = _original(hidden)
                return output if output_dtype is None else output.to(output_dtype)
            module.forward = forward

    def generation_settings(self):
        # The checkpoint config and tokenizer define different end markers.
        # Stop on either; otherwise an end-of-sequence can be decoded away while
        # generation continues into another response.
        stop_ids = {self.tokenizer.eos_token_id}
        for token in ("<eos>", "<|end_of_turn|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id != self.tokenizer.unk_token_id:
                stop_ids.add(token_id)
        return {
            "eos_token_id": sorted(i for i in stop_ids if isinstance(i, int)),
        }

    def decode_response(self, new_tokens):
        # Parse token IDs before skip_special_tokens can remove delimiters.
        # Reasoning never enters the browser or subsequent conversation history.
        ids = new_tokens.tolist()
        opening = self.tokenizer.convert_tokens_to_ids("<think>")
        closing = self.tokenizer.convert_tokens_to_ids("</think>")
        depth = 0
        answer = []
        for token_id in ids:
            if token_id == opening:
                depth += 1
            elif token_id == closing:
                if depth == 0:
                    raise RuntimeError("Malformed reasoning boundary in model response.")
                depth -= 1
            elif depth == 0:
                answer.append(token_id)
        if depth:
            raise RuntimeError("Sarvam reached the generation limit before completing its answer.")
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()

    def generate(self, messages, max_new_tokens=4096, deterministic=True,
                 temperature=0.7, system_prompt=None):
        # Only the workspace endpoint supplies an explicit system prompt.
        return super().generate(messages, max_new_tokens, deterministic,
                                temperature, system_prompt=system_prompt)
