"""Llama entry point; shared inference mechanics live in runtime.py."""
from .config import LLAMA_CONFIG
from .runtime import HuggingFaceRuntime


class LlamaRuntime(HuggingFaceRuntime):
    def __init__(self, model_id=LLAMA_CONFIG.model_id, load_in_4bit=None, *, config=LLAMA_CONFIG,
                 local_files_only=False):
        super().__init__(model_id, load_in_4bit, config=config, local_files_only=local_files_only)
