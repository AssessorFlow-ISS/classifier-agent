"""SearchPolicies tool for LLM function-calling (AF-137).

Wraps KnowledgeServicePort.search_policies as an OpenAI-compatible
function-calling tool so the LLM can query rubrics and policies during
ReAct sufficiency probing and rubric fitness assessment.
"""
from __future__ import annotations

from typing import Any

from classification_agent.ports.knowledge_service_port import KnowledgeServicePort


class SearchPoliciesTool:
    """Policy search tool for rubric fitness and domain policy checks.

    Searches the Policy Knowledge Base for rubrics and grading policies.
    Used by the Classification Agent to check domain web research policies
    and assess rubric fitness during ReAct reasoning.
    """

    def __init__(
        self,
        knowledge_service: KnowledgeServicePort,
        assessment_id: str,
    ) -> None:
        self.name = "search_policies"
        self._knowledge_service = knowledge_service
        self._assessment_id = assessment_id

    def to_openai_function(self) -> dict[str, Any]:
        """Return OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the policy knowledge base for rubrics and grading policies. "
                    "Use this to find assessor rubrics, system default policies, or "
                    "domain-specific web research policies."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for finding relevant policies or rubrics.",
                        },
                        "policy_type": {
                            "type": "string",
                            "enum": ["system_default", "assessor_rubric"],
                            "description": "Filter by policy type. Omit to search all types.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(
        self,
        query: str,
        policy_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the policy search and return serializable dicts."""
        results = await self._knowledge_service.search_policies(
            query=query,
            policy_type=policy_type,
            assessment_id=self._assessment_id,
        )
        return [r.model_dump() for r in results]
