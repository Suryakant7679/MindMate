"""Baseline settings: change here, not through patient-facing controls."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    enable_thinking: bool | None = None


DEFAULT_MODEL = "llama"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
CONTEXT_TOKENS = 16384
SYSTEM_PROMPT = (
    "You are MindMate, a supportive conversational assistant. "
    "Listen carefully, acknowledge feelings without making assumptions, and respond warmly and concisely. "
    "Return only the answer for the user. Never reveal chain-of-thought, analysis, or <think> content. "
    "Use the user's language. Ask a gentle follow-up question when helpful. "
    "Do not claim to be a therapist, diagnose conditions, or prescribe treatment. "
    "If someone describes immediate danger or intent to hurt themselves or others, "
    "encourage immediate help from local emergency services and a trusted person nearby."
)

LLAMA_CONFIG = ModelConfig(model_id="meta-llama/Llama-3.1-8B-Instruct")

# Emotion-neuron steering (Llama only).  A request's steering_alpha is multiplied
# by STEERING_GAIN before it scales each panel neuron's emotion-minus-neutral
# activation shift.  Calibrated on the MindMate prompt: alpha 1 gives a clearly
# more emotional but fluent reply, alpha 2 is strong and starts to cost fluency.
STEERING_GAIN = 1.75
# Response emotion steered for each detected user emotion when the request does
# not name one.  Negative user emotions are answered with warmth rather than
# mirrored: steering sadness, fear or anger into replies to a distressed user
# produced discouraging or alarming text in testing.
DEFAULT_STEERING_TARGET = "love"
SUPPORTIVE_TARGET = {
    "admiration": "admiration",
    "joy": "joy",
    "love": "love",
    "surprise": "surprise",
    "neutral": "love",
    "sadness": "love",
    "fear": "love",
    "anger": "love",
    "disgust": "love",
    "disapproval": "love",
}
