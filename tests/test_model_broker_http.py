"""Tests for the ModelBrokerHttpAdapter.

Uses httpx mock transport to test HTTP request/response handling
without a real Model Broker service.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from classification_agent.adapters.model_broker_http import ModelBrokerHttpAdapter


_BROKER_URL = "http://test-broker:8010"
_GENERATE_URL = f"{_BROKER_URL}/api/v1/generate"


def _mock_request() -> httpx.Request:
    return httpx.Request("POST", _GENERATE_URL)


@pytest.fixture
def success_response() -> dict:
    return {
        "content": '{"topics": [{"name": "OOP", "subtopics": []}]}',
        "model_used": "gemini-2.5-flash-lite",
        "model_tier": "CHEAP",
    }


@pytest.fixture
def adapter() -> ModelBrokerHttpAdapter:
    return ModelBrokerHttpAdapter(base_url=_BROKER_URL)


class TestModelBrokerHttpAdapter:
    """Tests for HTTP adapter bridging to Model Broker."""

    async def test_invoke_sends_correct_request(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.invoke(
                "classification.topic_extraction",
                "Extract topics from these chunks",
                workflow_id="wf-test-001",
            )

            adapter._client.post.assert_called_once()
            call_args = adapter._client.post.call_args
            assert call_args[0][0] == "/api/v1/generate"
            body = call_args[1]["json"]
            assert body["task_key"] == "classification.topic_extraction"
            assert body["prompt"] == "Extract topics from these chunks"
            assert body["agent_id"] == "classification-agent"
            assert body["session_id"] == "wf-test-001"
            # Aligned with the adapter's current max_tokens (bumped for richer
            # structured JSON payloads -- rubric + topics).
            assert body["max_tokens"] == 65536
            assert body["temperature"] == 0.3

    async def test_invoke_returns_parsed_content(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.invoke(
                "classification.topic_extraction", "prompt"
            )
            # Adapter parses JSON content string into dict, so result has
            # the parsed keys (e.g., "topics") not the raw "content" key
            assert result["model_used"] == "gemini-2.5-flash-lite"
            assert result["topics"] == [{"name": "OOP", "subtopics": []}]

    async def test_invoke_with_keyword_args(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        """Verify invoke accepts both positional and keyword args."""
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await adapter.invoke(
                task_key="classification.sufficiency_check",
                prompt="Check sufficiency",
                workflow_id="wf-kw",
            )
            # Parsed JSON — topics key present from parsed content
            assert result["model_used"] == "gemini-2.5-flash-lite"
            assert "topics" in result

    async def test_invoke_raises_on_http_error(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        mock_response = httpx.Response(
            500,
            json={"detail": "Internal Server Error"},
            request=httpx.Request("POST", "http://test-broker:8010/api/v1/generate"),
        )
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.invoke("classification.topic_extraction", "prompt")

    async def test_invoke_raises_on_connection_error(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                await adapter.invoke("classification.topic_extraction", "prompt")

    async def test_prompt_version_format(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.invoke("classification.topic_extraction", "prompt")
            body = adapter._client.post.call_args[1]["json"]
            assert body["prompt_version"] == "classification/topic_extraction@v1"

    async def test_default_workflow_id(
        self, adapter: ModelBrokerHttpAdapter, success_response: dict
    ) -> None:
        mock_response = httpx.Response(200, json=success_response, request=_mock_request())
        with patch.object(adapter._client, "post", new_callable=AsyncMock, return_value=mock_response):
            await adapter.invoke("classification.topic_extraction", "prompt")
            body = adapter._client.post.call_args[1]["json"]
            assert body["session_id"] == "unknown"

    async def test_close_closes_client(self, adapter: ModelBrokerHttpAdapter) -> None:
        with patch.object(adapter._client, "aclose", new_callable=AsyncMock) as mock_close:
            await adapter.close()
            mock_close.assert_called_once()

    def test_missing_base_url_raises(self, monkeypatch) -> None:
        # PROD-safety: no source-code localhost default. Missing env must raise.
        import pytest as _pytest
        monkeypatch.delenv("MODEL_BROKER_URL", raising=False)
        with _pytest.raises(RuntimeError, match="MODEL_BROKER_URL env var is required"):
            ModelBrokerHttpAdapter()
