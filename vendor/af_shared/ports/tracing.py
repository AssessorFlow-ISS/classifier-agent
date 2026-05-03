"""Vendored shim — TracingPort ABC. See vendor/af_shared/__init__.py."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from af_shared.models.domain import (
    DecisionLogEntry,
    EvaluationAuditEntry,
    ModelResponse,
    TokenUsageEntry,
)


class TracingPort(ABC):
    @abstractmethod
    async def trace_decision(self, entry: DecisionLogEntry) -> None: ...

    @abstractmethod
    async def trace_token_usage(self, entry: TokenUsageEntry) -> None: ...

    @abstractmethod
    async def trace_evaluation(self, entry: EvaluationAuditEntry) -> None: ...

    @abstractmethod
    async def trace_llm_call(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        task_key: str,
        prompt_version: str,
        model_response: ModelResponse,
        prompt_text: str = "",
        retrieval_context: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def trace_tool_call(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        tool_name: str,
        input_params: dict[str, Any],
        output_summary: dict[str, Any],
        latency_ms: float,
    ) -> None: ...
