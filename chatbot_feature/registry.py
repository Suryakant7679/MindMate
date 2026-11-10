"""Ordered registry for optional chatbot capabilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .protocols import ChatFeature
from .types import ChatContext, ChatResponse


class FeatureRegistry:
    """Register and run feature hooks in deterministic order."""

    def __init__(self, features: Iterable[ChatFeature] = ()) -> None:
        self._features: dict[str, ChatFeature] = {}
        for feature in features:
            self.register(feature)

    def register(self, feature: ChatFeature) -> None:
        name = feature.name.strip()
        if not name:
            raise ValueError("Feature name cannot be empty.")
        if name in self._features:
            raise ValueError(f"Feature {name!r} is already registered.")
        self._features[name] = feature

    def unregister(self, name: str) -> ChatFeature:
        try:
            return self._features.pop(name)
        except KeyError as exc:
            raise KeyError(f"Unknown chatbot feature: {name}") from exc

    def get(self, name: str) -> ChatFeature:
        try:
            return self._features[name]
        except KeyError as exc:
            raise KeyError(f"Unknown chatbot feature: {name}") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._features

    def __iter__(self) -> Iterator[ChatFeature]:
        return iter(self._features.values())

    def __len__(self) -> int:
        return len(self._features)

    async def before_generation(self, context: ChatContext) -> None:
        for feature in self._features.values():
            await feature.before_generation(context)

    async def after_generation(
        self, context: ChatContext, response: ChatResponse
    ) -> None:
        # Unwind in reverse order, like middleware.
        for feature in reversed(self._features.values()):
            await feature.after_generation(context, response)
