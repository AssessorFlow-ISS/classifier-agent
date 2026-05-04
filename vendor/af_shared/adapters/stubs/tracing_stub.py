"""In-memory stub TracingPort — captures every trace_* call.

Used by ``tests/test_tracing_wiring.py`` to assert that the
ClassificationService emits the right trace events for each LLM call,
tool call, and decision.
"""

from __future__ import annotations

from typing import Any

from af_shared.models.domain import (
    DecisionLogEntry,
    EvaluationAuditEntry,
    ModelResponse,
    TokenUsageEntry,
)
from af_shared.ports.tracing import TracingPort


class StubTracingAdapter(TracingPort):
    """Records every trace_* call into typed lists for test assertions."""

    def __init__(self) -> None:
        self.decisions: list[DecisionLogEntry] = []
        self.token_usages: list[TokenUsageEntry] = []
        self.evaluations: list[EvaluationAuditEntry] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    async def trace_decision(self, entry: DecisionLogEntry) -> None:
        self.decisions.append(entry)

    async def trace_token_usage(self, entry: TokenUsageEntry) -> None:
        self.token_usages.append(entry)

    async def trace_evaluation(self, entry: EvaluationAuditEntry) -> None:
        self.evaluations.append(entry)

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
    ) -> None:
        self.llm_calls.append(
            {
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "task_key": task_key,
                "prompt_version": prompt_version,
                "model_response": model_response,
                "prompt_text": prompt_text,
                "retrieval_context": retrieval_context,
            }
        )

    async def trace_tool_call(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        tool_name: str,
        input_params: dict[str, Any],
        output_summary: dict[str, Any],
        latency_ms: float,
    ) -> None:
        self.tool_calls.append(
            {
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "tool_name": tool_name,
                "input_params": input_params,
                "output_summary": output_summary,
                "latency_ms": latency_ms,
            }
        )
