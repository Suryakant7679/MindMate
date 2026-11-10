"""Composable feature layer for the MindMate chatbot.

This package is intentionally independent from ``backend`` until individual
features are ready to be connected to the production chat endpoints.
"""

from .orchestrator import ChatOrchestrator
from .registry import FeatureRegistry
from .types import ChatContext, ChatMessage, ChatResponse

__all__ = [
    "ChatContext",
    "ChatMessage",
    "ChatOrchestrator",
    "ChatResponse",
    "FeatureRegistry",
]
