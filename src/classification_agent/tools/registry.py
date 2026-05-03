"""Tool registry for Classification Agent LLM tool-calling (AF-137).

Collects all tool definitions and dispatches tool calls by name.
Also provides a factory for creating ReactSufficiencyProber instances
wired with the tool registry -- this keeps the domain/ layer free of
tools/ imports (hexagonal architecture boundary).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from classification_agent.domain.sufficiency import ReactSufficiencyProber
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort
from classification_agent.ports.model_broker_port import ModelBrokerPort
from classification_agent.tools.search_policies_tool import SearchPoliciesTool
from classification_agent.tools.similarity_search_tool import SimilaritySearchTool


class ToolRegistry:
    """Registry of LLM-callable tools with dispatch capability."""

    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools
        self._tool_map = {t.name: t for t in tools}

    def to_openai_functions(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI function-calling format."""
        return [t.to_openai_function() for t in self.tools]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call by name."""
        tool = self._tool_map.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await tool.execute(**arguments)


def build_tool_registry(
    knowledge_service: KnowledgeServicePort,
    workflow_id: str,
    assessment_id: str,
) -> ToolRegistry:
    """Create a ToolRegistry with all Classification Agent tools."""
    tools = [
        SimilaritySearchTool(
            knowledge_service=knowledge_service,
            workflow_id=workflow_id,
        ),
        SearchPoliciesTool(
            knowledge_service=knowledge_service,
            assessment_id=assessment_id,
        ),
    ]
    return ToolRegistry(tools=tools)


def build_react_prober_factory(
    model_broker: ModelBrokerPort,
    knowledge_service: KnowledgeServicePort,
) -> Callable[[str, str], ReactSufficiencyProber]:
    """Create a factory that builds ReactSufficiencyProber instances per request.

    The factory is injected into ClassificationService so the domain layer
    does not import from tools/ (hexagonal architecture boundary).

    Args:
        model_broker: Model Broker port for LLM calls.
        knowledge_service: Knowledge Service port for tool execution.

    Returns:
        A callable (workflow_id, assessment_id) -> ReactSufficiencyProber.
    """

    def factory(workflow_id: str, assessment_id: str) -> ReactSufficiencyProber:
        registry = build_tool_registry(
            knowledge_service=knowledge_service,
            workflow_id=workflow_id,
            assessment_id=assessment_id,
        )
        return ReactSufficiencyProber(
            model_broker=model_broker,
            workflow_id=workflow_id,
            tool_schemas=registry.to_openai_functions(),
            tool_executor=registry.execute,
        )

    return factory
