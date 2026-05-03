from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EventPublisherPort(ABC):
    """Port for publishing Pub/Sub events back to the Orchestrator.

    Classification Agent publishes to:
      - Topic #5 (assessorflow.classification.complete) on success
      - Topic #6 (assessorflow.classification.insufficient) on insufficient material
    """

    @abstractmethod
    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish an event to a Pub/Sub topic.

        Args:
            topic: Pub/Sub topic name (e.g. 'assessorflow.classification.complete').
            payload: Event payload dictionary.
        """
