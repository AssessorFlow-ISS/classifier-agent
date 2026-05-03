"""Tests for main.py — handler dispatch and config-driven adapter swap."""
from __future__ import annotations

import os
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient

from classification_agent.main import create_app


class TestCreateApp:
    """Tests for app factory and config-driven adapter selection."""

    def test_stub_adapters_by_default(self) -> None:
        """Default config uses all stub adapters."""
        with patch.dict(os.environ, {}, clear=False):
            app = create_app()
            assert app.title == "Classification Agent (#4)"

    def test_real_model_broker_when_http(self) -> None:
        """MODEL_BROKER_ADAPTER=http selects ModelBrokerHttpAdapter."""
        env = {
            "MODEL_BROKER_ADAPTER": "http",
            "MODEL_BROKER_URL": "http://localhost:8010",
            "EVENT_PUBLISHER_ADAPTER": "stub",
        }
        with patch.dict(os.environ, env, clear=False):
            app = create_app()
            # App should be created without error
            assert app is not None


class TestHealthEndpoints:
    """Smoke tests for health and readiness probes."""

    async def test_health(self) -> None:
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    async def test_ready(self) -> None:
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/ready")
            assert resp.status_code == 200


class TestInvokeEndpoint:
    """Tests for POST /invoke with stub adapters."""

    async def test_invoke_sufficient_material(self) -> None:
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/invoke", json={
                "workflow_id": "wf-test",
                "assessment_id": "a-test",
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["sufficient"] is True
            assert data["workflow_id"] == "wf-test"
