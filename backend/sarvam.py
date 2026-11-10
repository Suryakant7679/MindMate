"""Load Sarvam 30B in 4-bit and generate replies without a persona prompt."""
from .runtime import HuggingFaceRuntime
from .sarvam_config import SARVAM_CONFIG


class SarvamRuntime(HuggingFaceRuntime):
    def __init__(self, model_id=SARVAM_CONFIG.model_id, load_in_4bit=None):
        super().__init__(model_id, load_in_4bit, config=SARVAM_CONFIG)

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
