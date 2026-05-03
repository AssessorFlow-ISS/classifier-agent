"""SimilaritySearch tool for LLM function-calling (AF-137).

Wraps KnowledgeServicePort.similarity_search as an OpenAI-compatible
function-calling tool so the LLM can invoke semantic search during
ReAct sufficiency probing.
"""
from __future__ import annotations

from typing import Any

from classification_agent.ports.knowledge_service_port import KnowledgeServicePort


class SimilaritySearchTool:
    """Semantic similarity search tool for ReAct probing.

    Searches a knowledge base for chunks semantically similar to the query.
    Used by the Classification Agent's ReAct sufficiency probing loop to
    formulate depth queries per borderline topic at specific difficulty levels.
    """

    def __init__(
        self,
        knowledge_service: KnowledgeServicePort,
        workflow_id: str,
    ) -> None:
        self.name = "similarity_search"
        self._knowledge_service = knowledge_service
        self._workflow_id = workflow_id

    def to_openai_function(self) -> dict[str, Any]:
        """Return OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search for semantically similar document chunks in a knowledge base. "
                    "Use this to probe the depth of available material on a specific topic "
                    "or subtopic at a given difficulty level."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The semantic search query to find similar content.",
                        },
                        "knowledge_base_target": {
                            "type": "string",
                            "enum": ["document", "policy", "enriched"],
                            "description": "Which knowledge base to search: document, policy, or enriched.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default 5).",
                            "default": 5,
                        },
                    },
                    "required": ["query", "knowledge_base_target"],
                },
            },
        }

    async def execute(
        self,
        query: str,
        knowledge_base_target: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Execute the similarity search and return serializable dicts."""
        results = await self._knowledge_service.similarity_search(
            query=query,
            knowledge_base_target=knowledge_base_target,
            workflow_id=self._workflow_id,
            top_k=top_k,
        )
        return [r.model_dump() for r in results]
