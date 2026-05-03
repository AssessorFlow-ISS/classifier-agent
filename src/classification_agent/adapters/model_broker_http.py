"""Real Model Broker HTTP client for the Classification Agent.

Bridges the Classification Agent's ModelBrokerPort (invoke(task_key, prompt))
to the real Model Broker FastAPI service (POST /api/v1/generate).
"""
from __future__ import annotations

import json as _json
import os
import re
from typing import Any

import httpx
import structlog

from classification_agent.ports.model_broker_port import ModelBrokerPort

logger = structlog.get_logger(__name__)


class ModelBrokerHttpAdapter(ModelBrokerPort):
    """HTTP client adapter calling the real Model Broker service.

    Tracks cumulative token usage and cost across all requests for
    test reporting. Access via `total_tokens`, `total_cost_usd`,
    and `request_count` after tests complete.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 300.0) -> None:
        self._base_url = base_url or os.environ.get("MODEL_BROKER_URL", "http://localhost:8010")
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)
        self.request_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        # Per-model breakdown for O+ dashboard model segregation
        self._per_model: dict[str, dict] = {}
        self.last_model_used: str = "unknown"
        self.last_model_tier: str = "unknown"

    async def invoke(
        self,
        task_key: str,
        prompt: str,
        *,
        workflow_id: str | None = None,
        experiment_id: str | None = None,
        response_format: str | None = None,
        response_schema: dict | None = None,
    ) -> dict[str, Any]:
        """Call Model Broker and return the full response as a dict."""
        request_body: dict[str, Any] = {
            "task_key": task_key,
            "prompt": prompt,
            "max_tokens": 65536,
            "temperature": 0.3,
            "session_id": workflow_id or "unknown",
            "agent_id": "classification-agent",
            "prompt_version": f"classification/{task_key.split('.')[-1]}@v1",
        }
        if response_format:
            request_body["response_format"] = response_format
        if response_schema:
            request_body["response_schema"] = response_schema

        logger.info("model_broker_request", task_key=task_key, workflow_id=workflow_id)

        response = await self._client.post("/api/v1/generate", json=request_body)

        # Handle 422 GuardrailViolationError — L-10 output scan blocked the response.
        # Retry once with a modified prompt asking the LLM to avoid PII-like patterns.
        if response.status_code == 422:
            logger.warning(
                "model_broker_guardrail_blocked",
                task_key=task_key,
                status=422,
                detail=response.text[:200],
                workflow_id=workflow_id,
            )
            # Retry with explicit instruction to avoid phone/email/ID patterns
            request_body["prompt"] = (
                request_body["prompt"]
                + "\n\nIMPORTANT: Do not include any phone numbers, email addresses, "
                "NRIC numbers, credit card numbers, or other personally identifiable "
                "information in your response. Use placeholder text if referencing such data."
            )
            response = await self._client.post("/api/v1/generate", json=request_body)
            if response.status_code == 422:
                logger.warning(
                    "model_broker_guardrail_blocked_final",
                    task_key=task_key,
                    detail=response.text[:200],
                )
                return {
                    "content": "BLOCKED_BY_GUARDRAIL",
                    "guardrail_blocked": True,
                    "sufficient": False,
                    "model_used": "guardrail",
                }

        response.raise_for_status()
        data = response.json()

        logger.info("model_broker_response", model=data.get("model_used"), tier=data.get("model_tier"))

        # Accumulate token/cost stats from Model Broker response
        self.request_count += 1
        usage = data.get("token_usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        t_tok = usage.get("total_tokens", 0)
        cost = data.get("cost_usd", 0.0)
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok
        self.total_tokens += t_tok
        self.total_cost_usd += cost

        # Per-model breakdown
        model_id = data.get("model_used", "unknown")
        self.last_model_used = model_id
        self.last_model_tier = data.get("model_tier", "unknown")
        pm = self._per_model.setdefault(model_id, {
            "request_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0, "tests_passed": 0, "tests_failed": 0,
        })
        pm["request_count"] += 1
        pm["prompt_tokens"] += p_tok
        pm["completion_tokens"] += c_tok
        pm["total_tokens"] += t_tok
        pm["cost_usd"] += cost

        # Parse LLM content as JSON — domain code expects a dict with
        # task-specific keys (e.g., "topics", "sufficient"), not raw text.
        # Different models (Gemini, OpenAI) format JSON differently:
        #   - Gemini flash/pro: often wraps in ```json ... ```
        #   - Gemini flash-lite: sometimes returns plain text or malformed JSON
        #   - OpenAI gpt-4: usually returns clean JSON
        content = data["content"]
        model_used = data.get("model_used", "unknown")

        logger.info("model_broker_raw_content",
                     content_preview=str(content)[:300],
                     content_type=type(content).__name__,
                     model=model_used)

        def _try_parse(text: str) -> dict | None:
            try:
                parsed = _json.loads(text)
                if isinstance(parsed, dict):
                    parsed["model_used"] = model_used
                    return parsed
            except (_json.JSONDecodeError, TypeError, ValueError):
                return None

        # Attempt 1: direct JSON parse
        if isinstance(content, str):
            result = _try_parse(content)
            if result:
                return result

            # Attempt 2: strip markdown code fences (Gemini flash/pro pattern)
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if match:
                result = _try_parse(match.group(1).strip())
                if result:
                    return result

            # Attempt 3: find first { ... } block (handles preamble text)
            brace_match = re.search(r"\{[\s\S]*\}", content)
            if brace_match:
                result = _try_parse(brace_match.group(0))
                if result:
                    return result

        elif isinstance(content, dict):
            content["model_used"] = model_used
            return content

        logger.warning("model_broker_json_parse_failed",
                       content_preview=str(content)[:200], model=model_used)
        return {"content": content, "model_used": model_used}

    async def invoke_with_tools(
        self,
        task_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        """Call Model Broker with tool definitions for ReAct reasoning."""
        request_body = {
            "task_key": task_key,
            "messages": messages,
            "tools": tools,
            "max_tokens": 65536,
            "temperature": 0.3,
            "session_id": workflow_id or "unknown",
            "agent_id": "classification-agent",
            "prompt_version": f"classification/{task_key.split('.')[-1]}@v1",
        }

        logger.info("model_broker_tool_request", task_key=task_key, workflow_id=workflow_id)

        response = await self._client.post("/api/v1/generate-with-tools", json=request_body)
        response.raise_for_status()
        data = response.json()

        # Accumulate token/cost stats
        self.request_count += 1
        usage = data.get("token_usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        t_tok = usage.get("total_tokens", 0)
        cost = data.get("cost_usd", 0.0)
        self.total_prompt_tokens += p_tok
        self.total_completion_tokens += c_tok
        self.total_tokens += t_tok
        self.total_cost_usd += cost

        # Per-model breakdown
        model_id = data.get("model_used", "unknown")
        self.last_model_used = model_id
        self.last_model_tier = data.get("model_tier", "unknown")
        pm = self._per_model.setdefault(model_id, {
            "request_count": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0, "tests_passed": 0, "tests_failed": 0,
        })
        pm["request_count"] += 1
        pm["prompt_tokens"] += p_tok
        pm["completion_tokens"] += c_tok
        pm["total_tokens"] += t_tok
        pm["cost_usd"] += cost

        # Parse the final response content into a dict when no tool calls were
        # returned. Gemini routinely returns the final verdict as a text string
        # (sometimes wrapped in ```json fences or prose preamble). The prober's
        # _parse_final_response expects content.get(...), so a bare string
        # crashes downstream with AttributeError. Apply the same waterfall used
        # by invoke(): direct JSON parse -> markdown-fenced -> first {...} block.
        raw_content = data.get("content")
        tool_calls = data.get("tool_calls", []) or []
        parsed_content: Any = raw_content
        if not tool_calls and isinstance(raw_content, str):
            parsed_content = self._parse_json_content(raw_content) or raw_content

        return {
            "tool_calls": tool_calls,
            "content": parsed_content,
            "model_used": data.get("model_used", "unknown"),
        }

    @staticmethod
    def _parse_json_content(text: str) -> dict | None:
        """Best-effort JSON extraction from an LLM text response.

        Matches invoke()'s three-stage fallback: direct parse, strip markdown
        code fences, extract first ``{ ... }`` block. Returns None if all
        attempts fail so the caller can choose whether to keep the raw string.
        """

        def _try(text_candidate: str) -> dict | None:
            try:
                parsed = _json.loads(text_candidate)
                return parsed if isinstance(parsed, dict) else None
            except (_json.JSONDecodeError, TypeError, ValueError):
                return None

        direct = _try(text)
        if direct is not None:
            return direct
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            fenced = _try(fence_match.group(1).strip())
            if fenced is not None:
                return fenced
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            return _try(brace_match.group(0))
        return None

    async def close(self) -> None:
        await self._client.aclose()
