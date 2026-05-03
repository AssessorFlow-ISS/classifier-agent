from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from af_shared.utils.prompt_loader import get_prompt_version, load_prompt

from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    GapAnalysisEntry,
    WebResearchMode,
)

# Type alias for an async tool executor callable.
# Accepts (tool_name, arguments) and returns Any.
ToolExecutor = Callable[[str, dict[str, Any]], Any]

logger = structlog.get_logger(__name__)

_REACT_TASK_KEY = "classification.react_sufficiency"
_REACT_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "react_sufficiency.yaml"
_REACT_FRONTMATTER, _REACT_USER_TEMPLATE = load_prompt(_REACT_PROMPT_PATH)
# v5+ splits the persona/rules/output contract into frontmatter.system_prompt
# and keeps the per-workflow task context in the body.
_REACT_SYSTEM_PROMPT: str = (_REACT_FRONTMATTER.get("system_prompt") or "").strip()


# ---------------------------------------------------------------------------
# Unified ReAct result model
# ---------------------------------------------------------------------------


class ReactSufficiencyResult:
    """Result of unified ReAct sufficiency + rubric fitness probing."""

    def __init__(
        self,
        *,
        sufficient: bool,
        reason: str,
        gap_analysis: list[GapAnalysisEntry] | None = None,
        search_queries: list[str] | None = None,
        autonomy_exercised: bool = False,
        rubric_fitness: str = "NO_RUBRIC",   # "ALIGNED" | "MISALIGNED" | "NO_RUBRIC"
        rubric_reasoning: str = "",
        rubric_source: str = "none",
    ) -> None:
        self.sufficient = sufficient
        self.reason = reason
        self.gap_analysis = gap_analysis or []
        self.search_queries = search_queries or []
        self.autonomy_exercised = autonomy_exercised
        self.rubric_fitness = rubric_fitness
        self.rubric_reasoning = rubric_reasoning
        self.rubric_source = rubric_source


# ---------------------------------------------------------------------------
# Unified ReAct sufficiency + rubric prober
# ---------------------------------------------------------------------------


class ReactSufficiencyProber:
    """Unified ReAct probe: material sufficiency + rubric fitness.

    A single tool-calling session covers both TASK 1 (chunk depth probing
    via SimilaritySearch) and TASK 2 (rubric fitness via SearchPolicies).

    The rubric_block argument provides the assessor rubric context
    (markdown text or "NO_RUBRIC") injected into the prompt so the LLM
    can reason about alignment without a separate LLM call.

    Max 10 tool calls enforced as circuit breaker.

    Tool schemas and execution are injected via constructor to maintain
    hexagonal architecture boundaries (domain/ must not import from tools/).
    """

    MAX_TOOL_CALLS = 10

    def __init__(
        self,
        model_broker,
        workflow_id: str,
        tool_schemas: list[dict[str, Any]],
        tool_executor: ToolExecutor,
    ) -> None:
        self._model_broker = model_broker
        self._workflow_id = workflow_id
        self._tool_schemas = tool_schemas
        self._tool_executor = tool_executor
        self._prompt_version = get_prompt_version(_REACT_PROMPT_PATH)
        self.last_model_used: str = "unknown"

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    async def probe(
        self,
        chunks: list[ChunkData],
        config: AssessmentConfig,
        rubric_block: str = "NO_RUBRIC",
    ) -> ReactSufficiencyResult:
        """Run the unified ReAct sufficiency + rubric fitness probing loop."""
        tools = self._tool_schemas

        # Build initial messages
        chunk_summaries = "\n".join(
            f"- [{c.source_type.value}] {c.content[:120]}..." for c in chunks
        )

        web_mode = getattr(config, "web_research_mode", None)
        web_mode_str = web_mode.value if hasattr(web_mode, "value") else str(web_mode or "manual")

        user_prompt = _REACT_USER_TEMPLATE.format(
            assessment_title=config.assessment_title,
            structured_count=config.structured_question_count,
            non_structured_count=config.non_structured_question_count,
            difficulty_level=config.difficulty_level.value,
            web_research_mode=web_mode_str,
            chunk_count=len(chunks),
            chunk_summaries=chunk_summaries,
            max_tool_calls=self.MAX_TOOL_CALLS,
            rubric_block=rubric_block,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        tool_call_count = 0

        while True:
            response = await self._model_broker.invoke_with_tools(
                _REACT_TASK_KEY,
                messages=messages,
                tools=tools,
                workflow_id=self._workflow_id,
            )
            # Capture model_used from response (if available)
            if response.get("model_used"):
                self.last_model_used = response["model_used"]

            tool_calls = response.get("tool_calls", [])

            # If no tool calls or we hit the limit, parse the final response
            if not tool_calls or tool_call_count >= self.MAX_TOOL_CALLS:
                return self._parse_final_response(response, config)

            # Execute each tool call
            for tc in tool_calls:
                if tool_call_count >= self.MAX_TOOL_CALLS:
                    break

                func = tc.get("function", {})
                tool_name = func.get("name", "")
                raw_args = func.get("arguments", "{}")
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                tool_call_id = tc.get("id", "")

                try:
                    result = await self._tool_executor(tool_name, arguments)
                except Exception as exc:
                    result = {"error": str(exc)}

                # Append tool call + result to messages for context
                messages.append({
                    "role": "assistant",
                    "tool_calls": [tc],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

                tool_call_count += 1

            logger.info(
                "react_tool_calls_executed",
                workflow_id=self._workflow_id,
                tool_call_count=tool_call_count,
            )

    def _parse_final_response(
        self,
        response: dict[str, Any],
        config: AssessmentConfig,
    ) -> ReactSufficiencyResult:
        """Parse the LLM's final response into a ReactSufficiencyResult."""
        content = response.get("content", {})
        if content is None:
            content = {}
        # Some adapters may hand us a raw string if the model broker did not
        # parse the final JSON itself. Attempt a best-effort parse and fall
        # back to an empty dict so downstream .get() calls don't crash.
        if isinstance(content, str):
            import json as _json
            import re as _re
            parsed: dict[str, Any] | None = None
            try:
                maybe = _json.loads(content)
                if isinstance(maybe, dict):
                    parsed = maybe
            except (_json.JSONDecodeError, ValueError):
                parsed = None
            if parsed is None:
                fence = _re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
                if fence:
                    try:
                        maybe = _json.loads(fence.group(1).strip())
                        if isinstance(maybe, dict):
                            parsed = maybe
                    except (_json.JSONDecodeError, ValueError):
                        parsed = None
            if parsed is None:
                brace = _re.search(r"\{[\s\S]*\}", content)
                if brace:
                    try:
                        maybe = _json.loads(brace.group(0))
                        if isinstance(maybe, dict):
                            parsed = maybe
                    except (_json.JSONDecodeError, ValueError):
                        parsed = None
            content = parsed or {}

        sufficient = content.get("sufficient", False)
        reason = content.get("reason", "Unknown")

        # Parse gap analysis entries
        raw_gaps = content.get("gap_analysis", [])
        gap_analysis: list[GapAnalysisEntry] = []
        for item in raw_gaps:
            if isinstance(item, dict):
                gap_analysis.append(GapAnalysisEntry(**item))

        search_queries = content.get("search_queries", [])
        autonomy_exercised = False

        web_mode = getattr(config, "web_research_mode", WebResearchMode.MANUAL)
        if web_mode == WebResearchMode.AUTO:
            autonomy_exercised = content.get("autonomy_exercised", False)

        # Rubric fitness fields
        rubric_fitness = content.get("rubric_fitness", "NO_RUBRIC")
        rubric_reasoning = content.get("rubric_reasoning", "")
        rubric_source = content.get("rubric_source", "none")

        return ReactSufficiencyResult(
            sufficient=sufficient,
            reason=reason,
            gap_analysis=gap_analysis,
            search_queries=search_queries,
            autonomy_exercised=autonomy_exercised,
            rubric_fitness=rubric_fitness,
            rubric_reasoning=rubric_reasoning,
            rubric_source=rubric_source,
        )
