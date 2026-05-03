from __future__ import annotations

from typing import Any

from classification_agent.ports.event_publisher_port import EventPublisherPort


class StubEventPublisherAdapter(EventPublisherPort):
    """In-memory stub for Pub/Sub event publisher.

    Records all published events for test assertions.
    """

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        """All recorded events."""
        return self._events

    def get_events_for_topic(self, topic: str) -> list[dict[str, Any]]:
        """Filter events by topic name."""
        return [e for e in self._events if e["topic"] == topic]

    # -----------------------------------------------------------------------
    # Port implementation
    # -----------------------------------------------------------------------

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self._events.append({"topic": topic, "payload": payload})
