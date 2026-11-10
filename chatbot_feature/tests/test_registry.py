"""Unit tests for the dependency-free feature orchestration scaffold."""

from __future__ import annotations

import asyncio
import unittest

from chatbot_feature import (
    ChatContext,
    ChatMessage,
    ChatOrchestrator,
    FeatureRegistry,
)


class RecordingFeature:
    name = "recording"

    async def before_generation(self, context: ChatContext) -> None:
        context.metadata.setdefault("events", []).append("before")

    async def after_generation(self, context, response) -> None:
        context.metadata["events"].append("after")
        response.metadata["feature_ran"] = True


class FakeModel:
    async def generate(self, messages, metadata):
        metadata["events"].append("model")
        return ChatMessage(role="assistant", content="Hello")

    async def stream(self, messages, metadata):
        yield "Hello"


class FeatureRegistryTests(unittest.TestCase):
    def test_orchestrator_runs_hooks_around_model(self) -> None:
        context = ChatContext(
            session_id="session-1",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        registry = FeatureRegistry([RecordingFeature()])
        response = asyncio.run(ChatOrchestrator(FakeModel(), registry).generate(context))

        self.assertEqual(context.metadata["events"], ["before", "model", "after"])
        self.assertEqual(response.message.content, "Hello")
        self.assertTrue(response.metadata["feature_ran"])

    def test_duplicate_feature_names_are_rejected(self) -> None:
        registry = FeatureRegistry([RecordingFeature()])
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(RecordingFeature())


if __name__ == "__main__":
    unittest.main()
