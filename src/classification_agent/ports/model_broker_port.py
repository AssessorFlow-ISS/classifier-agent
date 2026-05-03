from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelBrokerPort(ABC):
    """Port for LLM inference via Model Broker (L-09).

    All LLM calls go through the Model Broker -- never call providers directly
    (Invariant #6). The Classification Agent uses CHEAP tier for both
    sufficiency checking and topic extraction.
    """

    @abstractmethod
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
        """Send a prompt to the Model Broker and return parsed JSON response.

        Args:
            task_key: Routing key for model tier selection
                      (e.g. 'classification.sufficiency_check').
            prompt: Rendered prompt string.
            workflow_id: Workflow ID for tracing.
            experiment_id: Langfuse experiment ID for tracing.
            response_format: Set to "json" to enforce structured JSON output.
            response_schema: JSON Schema for the expected response structure.
        """

    @abstractmethod
    async def invoke_with_tools(
        self,
        task_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        """Send messages with tool definitions and return response with optional tool_calls.

        Used for ReAct-style reasoning where the LLM decides which tools
        to invoke. The response contains either ``tool_calls`` (LLM wants
        to call a tool) or ``content`` (final answer).

        Args:
            task_key: Routing key for model tier selection.
            messages: Conversation messages in OpenAI chat format.
            tools: Tool definitions in OpenAI function-calling format.
            workflow_id: Workflow ID for tracing.

        Returns:
            Dict with ``tool_calls`` (list) and ``content`` (dict | None).
        """
