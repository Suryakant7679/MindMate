"""Shared, dependency-free data types used by chatbot features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    """One normalized message in a conversation."""

    role: Role
    content: str
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("ChatMessage content cannot be empty.")


@dataclass
class ChatContext:
    """Mutable request context enriched by features before generation."""

    session_id: str
    messages: list[ChatMessage]
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty.")
        if not self.messages:
            raise ValueError("A chat request must contain at least one message.")


@dataclass
class ChatResponse:
    """Model response plus feature-generated metadata."""

    message: ChatMessage
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedDocument:
    """A chunk returned by a RAG or memory search."""

    document_id: str
    text: str
    score: float
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SafetyDecision:
    """Result returned by an input or output safety guard."""

    allowed: bool
    category: str | None = None
    reason: str | None = None
    replacement_text: str | None = None
