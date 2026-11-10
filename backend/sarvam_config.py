"""Hugging Face Sarvam 30B settings for direct conversational responses."""
from .config import ModelConfig

SARVAM_CONFIG = ModelConfig(
    model_id="sarvamai/sarvam-30b",
    load_in_4bit=True,
    trust_remote_code=True,
    enable_thinking=True,
)
