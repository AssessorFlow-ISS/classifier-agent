from __future__ import annotations

from typing import Any

from classification_agent.ports.model_broker_port import ModelBrokerPort

_DEFAULT_SUFFICIENCY_RESPONSE: dict[str, Any] = {
    "sufficient": True,
    "reason": "Material covers all required topics with adequate depth",
    "gap_analysis": [],
}

_DEFAULT_TOPIC_RESPONSE: dict[str, Any] = {
    "topics": [
        {
            "topic_id": "t-001",
            "name": "Object-Oriented Programming",
            "subtopics": [
                {"topic_id": "t-001-1", "name": "Encapsulation"},
                {"topic_id": "t-001-2", "name": "Polymorphism"},
            ],
        },
        {
            "topic_id": "t-002",
            "name": "Data Structures",
            "subtopics": [
                {"topic_id": "t-002-1", "name": "Arrays and Lists"},
                {"topic_id": "t-002-2", "name": "Hash Maps"},
            ],
        },
        {
            "topic_id": "t-003",
            "name": "Algorithms",
            "subtopics": [
                {"topic_id": "t-003-1", "name": "Sorting"},
                {"topic_id": "t-003-2", "name": "Searching"},
            ],
        },
    ],
}

_DEFAULT_INSUFFICIENCY_RESPONSE: dict[str, Any] = {
    "sufficient": False,
    "reason": "Insufficient material: too few chunks to cover required question count",
    "gap_analysis": [
        {
            "topic": "Data Structures",
            "current_depth": "surface",
            "required_depth": "moderate",
            "gap_description": "Need more content on data structures",
            "fillable_by_web": True,
            "confidence": 0.8,
        },
        {
            "topic": "Algorithms",
            "current_depth": "surface",
            "required_depth": "deep",
            "gap_description": "No content on algorithms",
            "fillable_by_web": True,
            "confidence": 0.75,
        },
    ],
}

_DEFAULT_RUBRIC_FITNESS_RESPONSE: dict[str, Any] = {
    "is_aligned": True,
    "alignment_score": 0.85,
    "gap_description": None,
    "recommendation": "use_as_is",
}


class StubModelBrokerAdapter(ModelBrokerPort):
    """In-memory stub for Model Broker.

    Returns configurable canned responses per task_key.
    Supports both simple invoke() and tool-calling invoke_with_tools().
    """

    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {
            "classification.sufficiency_check": _DEFAULT_INSUFFICIENCY_RESPONSE,
            "classification.topic_extraction": _DEFAULT_TOPIC_RESPONSE,
            "classification.rubric_fitness": _DEFAULT_RUBRIC_FITNESS_RESPONSE,
        }
        # Default tool-call response for the unified ReAct prober so that
        # smoke tests using create_app() without explicit configuration
        # get a passing (sufficient=True) result out of the box.
        self._tool_call_responses: dict[str, list[dict[str, Any]]] = {
            "classification.react_sufficiency": [
                {
                    "tool_calls": [],
                    "content": {
                        "sufficient": True,
                        "reason": "Material sufficient (stub default)",
                        "gap_analysis": [],
                        "search_queries": [],
                        "autonomy_exercised": False,
                        "rubric_fitness": "NO_RUBRIC",
                        "rubric_reasoning": "",
                        "rubric_source": "none",
                    },
                }
            ],
        }
        self._invocations: list[dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    def set_response(self, task_key: str, response: dict[str, Any]) -> None:
        """Configure the canned response for a given task_key."""
        self._responses[task_key] = response

    def set_tool_call_responses(
        self, task_key: str, responses: list[dict[str, Any]],
    ) -> None:
        """Configure a sequence of tool-call responses for invoke_with_tools.

        Each call to invoke_with_tools pops the next response from the
        sequence. This enables testing multi-turn ReAct loops where the
        LLM first requests tool calls, then returns a final answer.
        """
        self._tool_call_responses[task_key] = list(responses)

    @property
    def invocations(self) -> list[dict[str, Any]]:
        """All recorded invocations (for test assertions)."""
        return self._invocations

    # -----------------------------------------------------------------------
    # Port implementation
    # -----------------------------------------------------------------------

    async def invoke(
        self,
        task_key: str,
        prompt: str,
        *,
        workflow_id: str | None = None,
        experiment_id: str | None = None,
        response_format: str | None = None,
        response_schema: dict | None = None,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        self._invocations.append(
            {
                "task_key": task_key,
                "prompt": prompt,
                "workflow_id": workflow_id,
                "experiment_id": experiment_id,
                "prompt_version": prompt_version,
            }
        )
        response = self._responses.get(task_key, {})
        if "model_used" not in response:
            response = {**response, "model_used": "stub"}
        return response

    async def invoke_with_tools(
        self,
        task_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        workflow_id: str | None = None,
        prompt_version: str | None = None,
    ) -> dict[str, Any]:
        self._invocations.append(
            {
                "task_key": task_key,
                "messages": messages,
                "tools": tools,
                "workflow_id": workflow_id,
                "prompt_version": prompt_version,
            }
        )
        sequence = self._tool_call_responses.get(task_key, [])
        if sequence:
            return sequence.pop(0)
        return {"tool_calls": [], "content": {}, "model_used": "stub"}
