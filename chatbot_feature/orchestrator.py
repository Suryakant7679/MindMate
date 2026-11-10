"""Small orchestration boundary joining a model and optional features."""

from __future__ import annotations

from .protocols import ChatModel
from .registry import FeatureRegistry
from .types import ChatContext, ChatResponse


class ChatOrchestrator:
    """Run registered feature hooks around one model generation."""

    def __init__(self, model: ChatModel, registry: FeatureRegistry | None = None) -> None:
        self.model = model
        self.registry = registry or FeatureRegistry()

    async def generate(self, context: ChatContext) -> ChatResponse:
        await self.registry.before_generation(context)
        message = await self.model.generate(context.messages, context.metadata)
        if message.role != "assistant":
            raise ValueError("Chat models must return an assistant message.")
        response = ChatResponse(message=message)
        await self.registry.after_generation(context, response)
        return response
