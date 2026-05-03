"""Tests for main.create_app adapter-wiring branches and HTTP /trigger.

Directly patches the module-level ``settings`` object so each branch
in ``_build_service`` exercises its real adapter-import path without
depending on module reload semantics.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from classification_agent import main as main_mod


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "knowledge_service_adapter": "stub",
        "assessment_config_adapter": "stub",
        "model_broker_adapter": "stub",
        "event_publisher_adapter": "stub",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestBuildServiceAdapterBranches:
    def test_http_knowledge_service_branch(self) -> None:
        with patch.object(
            main_mod, "settings", _settings(knowledge_service_adapter="http"),
        ):
            with patch(
                "classification_agent.adapters.knowledge_service_http.KnowledgeServiceHttpAdapter"
            ) as MockKS:
                MockKS.return_value = MagicMock()
                service, _ = main_mod._build_service()
        assert service is not None
        MockKS.assert_called_once()

    def test_grpc_assessment_config_branch(self) -> None:
        with patch.object(
            main_mod, "settings", _settings(assessment_config_adapter="grpc"),
        ):
            with patch(
                "classification_agent.adapters.assessment_config_grpc.GrpcAssessmentConfigAdapter"
            ) as MockAC:
                MockAC.return_value = MagicMock()
                service, _ = main_mod._build_service()
        MockAC.assert_called_once()
        assert service is not None

    def test_http_model_broker_branch(self) -> None:
        with patch.object(
            main_mod, "settings", _settings(model_broker_adapter="http"),
        ):
            with patch(
                "classification_agent.adapters.model_broker_http.ModelBrokerHttpAdapter"
            ) as MockMB:
                MockMB.return_value = MagicMock()
                service, _ = main_mod._build_service()
        MockMB.assert_called_once()

    def test_pubsub_event_publisher_branch(self) -> None:
        with patch.object(
            main_mod, "settings", _settings(event_publisher_adapter="pubsub"),
        ):
            with patch(
                "classification_agent.adapters.pubsub_event_publisher.PubSubEventPublisherAdapter"
            ) as MockPub:
                MockPub.return_value = MagicMock()
                _, publisher = main_mod._build_service()
        MockPub.assert_called_once()
        assert publisher is MockPub.return_value


class TestTriggerEndpoint:
    async def test_trigger_invokes_classify_and_returns_payload(self) -> None:
        app = main_mod.create_app()

        # Replace the wired service with a mock that returns a known response
        fake_response = MagicMock()
        fake_response.model_dump = MagicMock(
            return_value={
                "workflow_id": "wf-t",
                "assessment_id": "a-t",
                "sufficient": True,
                "topics": None,
                "gap_analysis": [],
            }
        )
        app.state.classification_service.classify = AsyncMock(
            return_value=fake_response,
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/trigger",
                json={
                    "workflow_id": "wf-t",
                    "assessment_id": "a-t",
                    "classification_type": "sufficiency_and_topics",
                    "chunks": [],
                },
            )
        assert resp.status_code == 200
        assert resp.json()["workflow_id"] == "wf-t"
        app.state.classification_service.classify.assert_awaited_once()

    async def test_trigger_coerces_invalid_classification_type_to_default(self) -> None:
        app = main_mod.create_app()

        captured: dict = {}

        async def capture(req) -> MagicMock:
            captured["req"] = req
            m = MagicMock()
            m.model_dump = MagicMock(return_value={"sufficient": True})
            return m

        app.state.classification_service.classify = capture

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/trigger",
                json={
                    "workflow_id": "wf-x",
                    "assessment_id": "a-x",
                    "classification_type": "garbage_value",
                },
            )
        assert resp.status_code == 200
        # Invalid type should have been coerced to the default enum
        from classification_agent.api.schemas import ClassificationType
        assert captured["req"].classification_type == ClassificationType.SUFFICIENCY_AND_TOPICS

    async def test_trigger_fills_unknown_workflow_when_missing(self) -> None:
        app = main_mod.create_app()

        captured: dict = {}

        async def capture(req) -> MagicMock:
            captured["req"] = req
            m = MagicMock()
            m.model_dump = MagicMock(return_value={"sufficient": True})
            return m

        app.state.classification_service.classify = capture

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/trigger", json={})
        assert resp.status_code == 200
        assert captured["req"].workflow_id == "unknown"
        assert captured["req"].assessment_id == "unknown"


class TestLifespanSubscription:
    async def test_lifespan_subscribes_when_pubsub_publisher(self) -> None:
        """Exercise the lifespan's subscribe_and_process branch.

        We patch settings + construct a PubSubEventPublisherAdapter instance
        (bypassing __init__ so no real client is built). The lifespan's
        ``isinstance`` check must succeed, so we return the REAL class
        instance rather than a MagicMock.
        """
        from classification_agent.adapters.pubsub_event_publisher import (
            PubSubEventPublisherAdapter,
        )

        fake_publisher = PubSubEventPublisherAdapter.__new__(
            PubSubEventPublisherAdapter,
        )
        fake_publisher.subscribe_and_process = AsyncMock()

        with patch.object(
            main_mod, "settings", _settings(event_publisher_adapter="pubsub"),
        ):
            # Patch the class only during create_app so _build_service uses
            # our fake instance. Remove the patch before the lifespan runs so
            # the lifespan's fresh import gets the real class (required for
            # isinstance()). Our fake_publisher IS a real class instance so
            # the isinstance check still passes.
            with patch(
                "classification_agent.adapters.pubsub_event_publisher.PubSubEventPublisherAdapter",
                return_value=fake_publisher,
            ):
                app = main_mod.create_app()

            # Lifespan now imports the real class — isinstance(fake_publisher)
            # still succeeds because fake_publisher was built via __new__.
            async with app.router.lifespan_context(app):
                pass

        fake_publisher.subscribe_and_process.assert_awaited()
