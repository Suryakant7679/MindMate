"""Validate browser input before it reaches the model."""


def validate_steering_alpha(payload: object) -> float:
    """Return the optional Llama emotion-steering intensity from a chat request."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    alpha = payload.get("steering_alpha", 0.0)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("steering_alpha must be a number.")
    alpha = float(alpha)
    if not -2.0 <= alpha <= 2.0:
        raise ValueError("steering_alpha must be between -2 and 2.")
    return alpha


STEERING_EMOTIONS = ("admiration", "anger", "disapproval", "disgust", "fear", "joy", "love", "sadness", "surprise")


def validate_steering_emotion(payload: object) -> str:
    """Return the optional steering target: "auto", "mirror" or an emotion label."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    emotion = payload.get("steering_emotion", "auto")
    if emotion is None:
        return "auto"
    if not isinstance(emotion, str) or emotion not in ("auto", "mirror", *STEERING_EMOTIONS):
        raise ValueError("steering_emotion must be auto, mirror or one of: " + ", ".join(STEERING_EMOTIONS) + ".")
    return emotion


def validate_chat_request(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 41:
        raise ValueError("Send between 1 and 41 conversation messages.")
    clean = []
    for index, item in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if not isinstance(item, dict) or item.get("role") != expected:
            raise ValueError("Conversation must alternate user and assistant messages.")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip() or len(content) > 8000:
            raise ValueError("Each message must contain between 1 and 8000 characters.")
        clean.append({"role": expected, "content": content.strip()})
    if clean[-1]["role"] != "user":
        raise ValueError("The last message must be from the user.")
    return clean
