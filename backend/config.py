"""Baseline settings: change here, not through patient-facing controls."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    enable_thinking: bool | None = None
    # Used only when the checkpoint's tokenizer ships without a chat template.
    chat_template: str | None = None
    # Prefix of this model's result folders: results/<tag>_emotions_full, <tag>_neuron_analysis,
    # <tag>_generation_emotion.  Steering panels, probe, policy and evidence are read from there and
    # are only used when they were produced with this exact checkpoint.
    results_tag: str | None = None
    # Panel file inside results/<tag>_generation_emotion/steering_panels/.
    panel_file: str = "panels.json"


# Model loaded at start (Sarvam-30B since 2026-11-21; best Hinglish replies in the v3 study).  Every other registry
# model can be selected per message; Llama 3.1-8B remains the main emotion-steering research system.
DEFAULT_MODEL = "sarvam"
DEFAULT_MODEL_ID = "sarvamai/sarvam-30b"
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

LLAMA_CONFIG = ModelConfig(model_id="meta-llama/Llama-3.1-8B-Instruct", results_tag="llama31_8b",
                           panel_file="panels_causal.json")

# Emotion-neuron steering (Llama only).  A request's steering_alpha is multiplied
# by STEERING_GAIN before it scales each panel neuron's emotion-minus-neutral
# activation shift.  Calibrated on the MindMate prompt: alpha 1 gives a clearly
# more emotional but fluent reply, alpha 2 is strong and starts to cost fluency.
STEERING_GAIN = 1.75
# Panel file of the main model (see ModelConfig.panel_file).  "panels.json" is the attribution-selected
# panel from build_steering_panels; a causally validated subset is only named after it beat that
# panel on held-out messages (see generation_analysis/README.md).
STEERING_PANEL_FILE = LLAMA_CONFIG.panel_file
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


# ----------------------------------------------------------------------------- model registry
# Every chat model the app can serve.  The chat composer lists these; the backend loads the selected
# one on demand (one model on the GPU at a time) and never downloads weights unless started with
# --allow-download.  The experiment pipeline is configured per model through the same entries
# (MINDMATE_MODEL=<key>, see run_model_pipeline.sh).

# EmoLLaMA ships no chat template.  Its model card uses "Human: ... Assistant:" turns, so chat uses
# that format; a system prompt is placed before the first turn.
EMOLLAMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'user' %}Human: \n{{ message['content'] }}\n\n"
    "{% else %}Assistant:\n{{ message['content'] }}{{ eos_token }}\n\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}Assistant:\n{% endif %}"
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    purpose: str
    config: ModelConfig
    runtime: str = "generic"          # "llama" (main system), "sarvam", or "generic"
    persona: bool = True              # False: no MindMate system prompt (Sarvam's existing behaviour)
    max_new_tokens: int = 1024
    gpu_gib: float = 0.0              # approximate GPU memory when loaded (used before the first measured load)
    # Chat generation settings.  Greedy decoding is what every calibration and held-out comparison
    # used (generate_candidates always decodes greedily), so it stays the default for every model; temperature and
    # top_p only apply when ``deterministic`` is False.
    deterministic: bool = True
    temperature: float = 0.7
    top_p: float = 0.9


MODEL_REGISTRY: dict[str, ModelSpec] = {spec.key: spec for spec in (
    ModelSpec("llama", "Llama 3.1-8B", "Main experimental system (emotion reasoner + neuron steering)",
              LLAMA_CONFIG, runtime="llama", gpu_gib=16),
    ModelSpec("llama2", "Llama-2-13B-chat", "General-purpose Llama baseline",
              ModelConfig(model_id="meta-llama/Llama-2-13b-chat-hf", results_tag="llama2_13b_chat"), gpu_gib=26),
    ModelSpec("emollama", "EmoLLaMA-chat-7B", "Emotion-specialized baseline",
              ModelConfig(model_id="lzw1008/Emollama-chat-7b", chat_template=EMOLLAMA_CHAT_TEMPLATE,
                          results_tag="emollama_chat_7b"), gpu_gib=14),
    ModelSpec("sarvam", "Sarvam-30B", "Multilingual and Hinglish conversational baseline",
              ModelConfig(model_id="sarvamai/sarvam-30b", load_in_4bit=True, trust_remote_code=True,
                          enable_thinking=False, results_tag="sarvam_30b"),
              # Reasoning off: the empty reasoning block is pre-filled (the checkpoint ignores its own flag), so a
              # reply takes ~1 s instead of ~50 s and the Emotion Reasoner's measurements cover the answer tokens.
              # This is the configuration every Sarvam calibration and held-out result was produced with.
              runtime="sarvam", persona=False, max_new_tokens=1024, gpu_gib=20),
)}
