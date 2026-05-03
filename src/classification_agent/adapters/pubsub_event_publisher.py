"""Real Pub/Sub event publisher for the Classification Agent.

Publishes to assessorflow.classification.complete or
assessorflow.classification.insufficient. Subscribes to
assessorflow.classification.trigger via polling.

Inherits all publish/subscribe/envelope logic from the shared
AgentPubSubSubscriber base class.
"""

from __future__ import annotations

from typing import Any

from af_shared.pubsub.agent_subscriber import AgentPubSubSubscriber
from classification_agent.ports.event_publisher_port import EventPublisherPort


class PubSubEventPublisherAdapter(AgentPubSubSubscriber, EventPublisherPort):
    """Real Pub/Sub adapter for the Classification Agent.

    Inherits publish(), subscribe_and_process(), envelope wrapping, and
    poll-based subscription from AgentPubSubSubscriber. Only the agent
    name differs.
    """

    def __init__(
        self,
        project_id: str | None = None,
        emulator_host: str | None = None,
    ) -> None:
        super().__init__(
            agent_name="classification-agent",
            project_id=project_id,
            emulator_host=emulator_host,
        )

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        """Publish with positional args matching the Classification port interface."""
        await super().publish(topic, payload)
