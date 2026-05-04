"""Extra tests for ModelBrokerHttpAdapter covering branches not exercised
by the base suite: 422 guardrail-retry, tools path, env-var defaulting,
markdown-fence extraction, and the various JSON-parse fallback attempts.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from classification_agent.adapters.model_broker_http import ModelBrokerHttpAdapter


_BROKER_URL = "http://test-broker:8010"
_GENERATE_URL = f"{_BROKER_URL}/api/v1/generate"
_TOOLS_URL = f"{_BROKER_URL}/api/v1/generate-with-tools"


def _mock_request(url: str = _GENERATE_URL) -> httpx.Request:
    return httpx.Request("POST", url)


@pytest.fixture
def adapter() -> ModelBrokerHttpAdapter:
    return ModelBrokerHttpAdapter(base_url=_BROKER_URL)


# ---------------------------------------------------------------------------
# Guardrail (422) retry path
# ---------------------------------------------------------------------------


class TestGuardrail422Retry:
    async def test_retry_succeeds_on_second_attempt(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        """First 422 triggers prompt-mutation retry; second response succeeds."""
        success_payload = {
            "content": '{"ok": true, "topics": []}',
            "model_used": "gemini-flash",
            "model_tier": "CHEAP",
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "cost_usd": 0.001,
        }
        responses = [
            httpx.Response(422, text="PII detected", request=_mock_request()),
            httpx.Response(200, json=success_payload, request=_mock_request()),
        ]

        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock, side_effect=responses,
        ) as mock_post:
            result = await adapter.invoke(
                "classification.sufficiency_check",
                "Original prompt",
                workflow_id="wf-guard",
                prompt_version="classification/sufficiency_check@v1",
            )
            assert mock_post.await_count == 2
            second_body = mock_post.await_args_list[1].kwargs["json"]
            # Retry adds the anti-PII suffix
            assert "Do not include any phone numbers" in second_body["prompt"]
            assert result["ok"] is True
            assert result["model_used"] == "gemini-flash"
            # Token stats accumulated from the success response
            assert adapter.total_tokens == 15
            assert adapter.total_cost_usd == pytest.approx(0.001)

    async def test_final_422_returns_guardrail_blocked(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        """Two 422s in a row => adapter returns the BLOCKED_BY_GUARDRAIL sentinel."""
        responses = [
            httpx.Response(422, text="PII", request=_mock_request()),
            httpx.Response(422, text="still PII", request=_mock_request()),
        ]
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock, side_effect=responses,
        ):
            result = await adapter.invoke(
                "classification.sufficiency_check", "prompt",
                prompt_version="classification/sufficiency_check@v1",
            )
            assert result["guardrail_blocked"] is True
            assert result["sufficient"] is False
            assert result["model_used"] == "guardrail"


# ---------------------------------------------------------------------------
# JSON parse fallback attempts
# ---------------------------------------------------------------------------


class TestJsonParseFallbacks:
    async def test_parse_direct_json(self, adapter: ModelBrokerHttpAdapter) -> None:
        resp_body = {
            "content": '{"topics": [{"name": "A", "subtopics": []}]}',
            "model_used": "m",
            "token_usage": {"total_tokens": 3},
        }
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request()),
        ):
            result = await adapter.invoke("t", "p", prompt_version="classification/t@v1")
            assert result["topics"][0]["name"] == "A"

    async def test_parse_markdown_fenced_json(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        """Gemini flash/pro often wraps structured output in ```json fences."""
        resp_body = {
            "content": "Here is your answer:\n```json\n{\"topics\": [\"B\"]}\n```",
            "model_used": "gemini-pro",
        }
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request()),
        ):
            result = await adapter.invoke("t", "p", prompt_version="classification/t@v1")
            assert result["topics"] == ["B"]

    async def test_parse_first_brace_block(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        """Preamble before a bare JSON object still resolves via brace-match fallback."""
        resp_body = {
            "content": "Sorry, here: {\"decision\": \"sufficient\"} trailing noise",
            "model_used": "gemini-flash-lite",
        }
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request()),
        ):
            result = await adapter.invoke("t", "p", prompt_version="classification/t@v1")
            assert result["decision"] == "sufficient"

    async def test_parse_dict_content_passes_through(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        """When the broker already returns a dict, it flows through unchanged."""
        resp_body = {
            "content": {"topics": ["X"]},
            "model_used": "openai",
        }
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request()),
        ):
            result = await adapter.invoke("t", "p", prompt_version="classification/t@v1")
            assert result["topics"] == ["X"]

    async def test_parse_unparseable_returns_raw(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        resp_body = {
            "content": "This is not JSON at all, sorry!",
            "model_used": "unknown",
        }
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request()),
        ):
            result = await adapter.invoke("t", "p", prompt_version="classification/t@v1")
            # When all three parse attempts fail, adapter returns {"content": ..., "model_used": ...}
            assert "content" in result
            assert result["model_used"] == "unknown"


# ---------------------------------------------------------------------------
# Env-var base URL + optional body params
# ---------------------------------------------------------------------------


class TestAdapterEnvConfig:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_BROKER_URL", "http://broker-env:9000")
        env_adapter = ModelBrokerHttpAdapter()
        assert env_adapter._base_url == "http://broker-env:9000"

    async def test_response_format_and_schema_forwarded(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        resp_body = {"content": "{}", "model_used": "m"}
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request()),
        ) as mock_post:
            await adapter.invoke(
                "t", "p",
                response_format="json",
                response_schema={"type": "object"},
                prompt_version="classification/t@v1",
            )
            body = mock_post.call_args[1]["json"]
            assert body["response_format"] == "json"
            assert body["response_schema"] == {"type": "object"}


# ---------------------------------------------------------------------------
# invoke_with_tools path
# ---------------------------------------------------------------------------


class TestInvokeWithTools:
    async def test_returns_tool_calls(self, adapter: ModelBrokerHttpAdapter) -> None:
        resp_body = {
            "tool_calls": [{"function": {"name": "similarity_search", "arguments": "{}"}}],
            "content": None,
            "model_used": "openai-gpt4",
            "model_tier": "SMART",
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
            "cost_usd": 0.02,
        }
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request(_TOOLS_URL)),
        ) as mock_post:
            result = await adapter.invoke_with_tools(
                "classification.react_sufficiency",
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
                workflow_id="wf-tools",
                prompt_version="classification/react_sufficiency@v6",
            )
            # Correct endpoint
            assert mock_post.call_args[0][0] == "/api/v1/generate-with-tools"
            assert len(result["tool_calls"]) == 1
            assert result["model_used"] == "openai-gpt4"
            # Accumulated token stats
            assert adapter.total_tokens == 27
            assert adapter.last_model_used == "openai-gpt4"
            assert adapter.last_model_tier == "SMART"

    async def test_defaults_tool_workflow_id(
        self, adapter: ModelBrokerHttpAdapter
    ) -> None:
        resp_body = {"tool_calls": [], "content": "", "model_used": "m"}
        with patch.object(
            adapter._client, "post",
            new_callable=AsyncMock,
            return_value=httpx.Response(200, json=resp_body, request=_mock_request(_TOOLS_URL)),
        ) as mock_post:
            await adapter.invoke_with_tools(
                "t", messages=[], tools=[],
                prompt_version="classification/t@v1",
            )
            body = mock_post.call_args[1]["json"]
            assert body["session_id"] == "unknown"
            assert body["agent_id"] == "classification-agent"
