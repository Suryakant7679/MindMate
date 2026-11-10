"""Contracts that concrete chatbot feature implementations should satisfy."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .types import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    RetrievedDocument,
    SafetyDecision,
)


@runtime_checkable
class ChatFeature(Protocol):
    """A hook that can enrich requests and responses."""

    name: str

    async def before_generation(self, context: ChatContext) -> None:
        """Inspect or enrich a request before the model is called."""

    async def after_generation(
        self, context: ChatContext, response: ChatResponse
    ) -> None:
        """Inspect or enrich a response after the model is called."""


@runtime_checkable
class ChatModel(Protocol):
    """Adapter contract for a local or hosted language model."""

    async def generate(
        self, messages: Sequence[ChatMessage], metadata: Mapping[str, Any]
    ) -> ChatMessage:
        """Generate one assistant message."""

    def stream(
        self, messages: Sequence[ChatMessage], metadata: Mapping[str, Any]
    ) -> AsyncIterator[str]:
        """Stream assistant text chunks."""


@runtime_checkable
class MemoryStore(Protocol):
    """Short- or long-term conversation memory backend."""

    async def add(self, session_id: str, message: ChatMessage) -> None: ...

    async def search(
        self, query: str, *, session_id: str, limit: int = 5
    ) -> Sequence[RetrievedDocument]: ...

    async def clear(self, session_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """Knowledge-base retrieval backend used by RAG."""

    async def search(
        self, query: str, *, limit: int = 5, filters: Mapping[str, Any] | None = None
    ) -> Sequence[RetrievedDocument]: ...


@runtime_checkable
class MCPConnector(Protocol):
    """Transport-neutral interface to one Model Context Protocol server."""

    async def connect(self) -> None: ...

    async def list_tools(self) -> Sequence[Mapping[str, Any]]: ...

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any: ...

    async def close(self) -> None: ...


@runtime_checkable
class SafetyGuard(Protocol):
    """Safety policy applied to user input or generated output."""

    async def inspect(self, text: str, *, direction: str) -> SafetyDecision: ...
