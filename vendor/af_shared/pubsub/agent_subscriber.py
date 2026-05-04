"""Shared Pub/Sub publish + subscribe base class for stateless agents.

Vendored shim of upstream ``af_shared.pubsub.agent_subscriber``. Provides
the publish-with-envelope pattern + poll-based subscribe loop. Agents
inherit and add their own port-conforming adapter wrapper.

Envelope shape (matches tests/test_pubsub_adapter.py expectations):

    {
        "event_id":      "<uuid>",
        "timestamp":     "<iso8601 utc>",
        "correlation_id":"<uuid or workflow_id>",
        "workflow_id":   "<from payload>",
        "event_type":    "topic",
        "source_agent":  "<agent_name>",
        "payload":       <inner dict>
    }

The ``_topic_prefix`` attribute lets the golden namespace prepend
``golden.`` to topics without touching production callers.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)


# Stale-message age cap (ack-and-drop messages older than this).
# Matches the upstream default; can be overridden via env.
_STALE_AGE_SECONDS = int(os.environ.get("PUBSUB_STALE_AGE_SECONDS", "600"))


class AgentPubSubSubscriber:
    """Shared base for agent Pub/Sub publish + subscribe.

    Constructor lazily creates google-cloud-pubsub clients. Tests can
    bypass __init__ via ``__new__`` and inject ``_publisher`` / ``_subscriber``
    mocks (see tests/test_pubsub_adapter.py:_make_adapter).
    """

    def __init__(
        self,
        agent_name: str,
        project_id: str | None = None,
        emulator_host: str | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._project_id = project_id or os.environ.get("PUBSUB_PROJECT_ID", "")
        self._emulator_host = emulator_host or os.environ.get("PUBSUB_EMULATOR_HOST", "")
        self._poll_tasks: list[asyncio.Task[Any]] = []
        # Set by the golden-namespace wrapper if needed; defaults to "" (prod).
        self._topic_prefix: str = os.environ.get("PUBSUB_TOPIC_PREFIX", "")

        # Lazily initialise google-cloud-pubsub clients. If the package isn't
        # installed (test environments using mocks), let the tests inject
        # mocks directly into _publisher / _subscriber.
        self._publisher: Any = None
        self._subscriber: Any = None
        try:
            from google.cloud import pubsub_v1  # type: ignore[import-not-found]

            self._publisher = pubsub_v1.PublisherClient()
            self._subscriber = pubsub_v1.SubscriberClient()
        except ImportError:
            logger.info("pubsub_clients_unavailable_using_mocks_or_skipping")

    # ------------------------------------------------------------------ publish

    async def publish(self, topic: str, payload: dict[str, Any]) -> str:
        """Wrap payload in envelope, publish to <prefix><topic>, return msg-id."""
        topic_with_prefix = f"{self._topic_prefix}{topic}"
        topic_path = self._publisher.topic_path(self._project_id, topic_with_prefix)

        envelope = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": payload.get("workflow_id", str(uuid.uuid4())),
            "workflow_id": payload.get("workflow_id", ""),
            "event_type": "topic",
            "source_agent": self._agent_name,
            "payload": payload,
        }
        data = json.dumps(envelope).encode("utf-8")

        future = self._publisher.publish(
            topic_path,
            data=data,
            workflow_id=str(payload.get("workflow_id", "")),
            source_agent=self._agent_name,
            event_type="topic",
        )
        return await asyncio.to_thread(future.result)

    # ---------------------------------------------------------------- subscribe

    async def subscribe_and_process(
        self,
        subscription: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Spawn a background poll loop. Each message is ACK'd before handler
        runs (at-most-once delivery — handler errors do NOT trigger redelivery).
        """
        subscription_with_prefix = f"{self._topic_prefix}{subscription}"
        subscription_path = self._subscriber.subscription_path(
            self._project_id, subscription_with_prefix
        )
        task = asyncio.create_task(self._poll_loop(subscription_path, handler))
        self._poll_tasks.append(task)

    async def _poll_loop(
        self,
        subscription_path: str,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        while True:
            try:
                response = await asyncio.to_thread(
                    self._subscriber.pull,
                    request={"subscription": subscription_path, "max_messages": 10},
                    timeout=10,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("pubsub_pull_failed", error=str(exc))
                await asyncio.sleep(1)
                continue

            for received in response.received_messages:
                ack_id = received.ack_id

                # ACK first (at-most-once). A long-running or failing handler
                # must NOT trigger redelivery.
                try:
                    await asyncio.to_thread(
                        self._subscriber.acknowledge,
                        request={
                            "subscription": subscription_path,
                            "ack_ids": [ack_id],
                        },
                    )
                except Exception as exc:
                    logger.warning("pubsub_ack_failed", error=str(exc))

                # Drop messages older than the stale cap.
                publish_time = received.message.publish_time
                if publish_time is not None:
                    try:
                        age = (datetime.now(timezone.utc) - publish_time).total_seconds()
                        if age > _STALE_AGE_SECONDS:
                            logger.info("pubsub_stale_message_dropped", age_s=age)
                            continue
                    except (TypeError, ValueError):
                        pass

                try:
                    payload = json.loads(received.message.data.decode("utf-8"))
                    # Unwrap envelope if present (back-compat: tests pass
                    # bare payload, prod sends envelopes).
                    if isinstance(payload, dict) and "payload" in payload \
                            and isinstance(payload["payload"], dict):
                        inner = payload["payload"]
                    else:
                        inner = payload
                    await handler(inner)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("pubsub_handler_failed", error=str(exc))
