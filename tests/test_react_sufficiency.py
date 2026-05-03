"""Tests for unified ReAct sufficiency + rubric fitness probe.

The ReactSufficiencyProber runs a single tool-calling session that covers:
  TASK 1 — material sufficiency (SimilaritySearch tool)
  TASK 2 — rubric fitness (SearchPolicies tool)

Both verdicts are returned in one structured payload.
Max 10 tool calls enforced as circuit breaker.
"""
from __future__ import annotations

from typing import Any


from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    DifficultyLevel,
    GapAnalysisEntry,
    SourceType,
    WebResearchMode,
)
from classification_agent.tools.registry import build_react_prober_factory


def _build_prober(
    mb: StubModelBrokerAdapter,
    ks: StubKnowledgeServiceAdapter,
    workflow_id: str = "wf-react",
    assessment_id: str = "assess-react",
):
    """Build a ReactSufficiencyProber using the factory (test helper)."""
    factory = build_react_prober_factory(
        model_broker=mb,
        knowledge_service=ks,
    )
    return factory(workflow_id, assessment_id)


def _make_chunks(n: int, workflow_id: str = "wf-react") -> list[ChunkData]:
    """Create n direct_text chunks for testing."""
    return [
        ChunkData(
            chunk_id=f"chunk-{i:03d}",
            workflow_id=workflow_id,
            content=f"Content about topic area {i} with moderate depth.",
            source_type=SourceType.DIRECT_TEXT,
        )
        for i in range(n)
    ]


def _make_config(
    *,
    mcq: int = 10,
    oe: int = 5,
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    web_research_mode: WebResearchMode = WebResearchMode.MANUAL,
) -> AssessmentConfig:
    return AssessmentConfig(
        assessment_id="assess-react",
        assessment_title="Computer Science Fundamentals",
        structured_question_count=mcq,
        non_structured_question_count=oe,
        difficulty_level=difficulty,
        web_research_mode=web_research_mode,
    )


def _react_response_with_tool_calls(tool_calls: list[dict]) -> dict[str, Any]:
    """Build a model broker response that includes tool_calls for ReAct."""
    return {
        "tool_calls": tool_calls,
        "content": None,
    }


def _react_final_response(
    sufficient: bool,
    gap_analysis: list[dict] | None = None,
    search_queries: list[str] | None = None,
    autonomy_exercised: bool = False,
    rubric_fitness: str = "NO_RUBRIC",
    rubric_reasoning: str = "",
    rubric_source: str = "none",
) -> dict[str, Any]:
    """Build a model broker final response with full unified payload."""
    return {
        "tool_calls": [],
        "content": {
            "sufficient": sufficient,
            "reason": "Sufficient material" if sufficient else "Insufficient depth",
            "gap_analysis": gap_analysis or [],
            "search_queries": search_queries or [],
            "autonomy_exercised": autonomy_exercised,
            "rubric_fitness": rubric_fitness,
            "rubric_reasoning": rubric_reasoning,
            "rubric_source": rubric_source,
        },
    }


class TestReActSufficiencyProbing:
    """Tests for ReAct sufficiency probing loop (unified probe)."""

    async def test_react_probing_uses_tool_calls(self) -> None:
        """AC-1: Prober uses LLM tool-calling for depth probing."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_response_with_tool_calls([
                {
                    "id": "call_001",
                    "function": {
                        "name": "similarity_search",
                        "arguments": '{"query": "advanced OOP design patterns", "knowledge_base_target": "document", "top_k": 5}',
                    },
                },
            ]),
            _react_final_response(sufficient=True, rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.sufficient is True
        assert len(mb.invocations) >= 2

    async def test_react_probing_produces_structured_gap_analysis(self) -> None:
        """AC-2: Gap analysis output uses GapAnalysisEntry with confidence scores."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=False,
                gap_analysis=[
                    {
                        "topic": "Data Structures",
                        "current_depth": "surface",
                        "required_depth": "deep",
                        "gap_description": "Only introductory content on trees and graphs",
                        "fillable_by_web": True,
                        "confidence": 0.35,
                    },
                ],
                rubric_fitness="ALIGNED",
                rubric_source="admin_seeded",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(5)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.sufficient is False
        assert len(result.gap_analysis) == 1
        gap = result.gap_analysis[0]
        assert isinstance(gap, GapAnalysisEntry)
        assert gap.topic == "Data Structures"
        assert gap.confidence == 0.35
        assert gap.fillable_by_web is True

    async def test_react_probing_stops_at_max_tool_calls(self) -> None:
        """AC-3: ReAct loop is bounded at max 10 tool calls (circuit breaker)."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        endless_tool_calls = [
            _react_response_with_tool_calls([
                {
                    "id": f"call_{i:03d}",
                    "function": {
                        "name": "similarity_search",
                        "arguments": f'{{"query": "probe query {i}", "knowledge_base_target": "document", "top_k": 5}}',
                    },
                },
            ])
            for i in range(15)
        ]
        endless_tool_calls.append(
            _react_final_response(sufficient=False, gap_analysis=[], rubric_fitness="NO_RUBRIC")
        )
        mb.set_tool_call_responses("classification.react_sufficiency", endless_tool_calls)

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        await prober.probe(chunks, config)

        tool_execution_count = sum(
            1 for inv in mb.invocations
            if inv.get("task_key") == "classification.react_sufficiency"
        )
        assert tool_execution_count <= 11

    async def test_react_probing_with_multiple_tool_calls_per_turn(self) -> None:
        """AC-4: LLM can request multiple tool calls in a single turn."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_response_with_tool_calls([
                {
                    "id": "call_001",
                    "function": {
                        "name": "similarity_search",
                        "arguments": '{"query": "OOP patterns", "knowledge_base_target": "document"}',
                    },
                },
                {
                    "id": "call_002",
                    "function": {
                        "name": "similarity_search",
                        "arguments": '{"query": "data structure depth", "knowledge_base_target": "document"}',
                    },
                },
            ]),
            _react_final_response(sufficient=True, rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.sufficient is True

    async def test_react_probing_with_search_policies_tool(self) -> None:
        """AC-5: ReAct loop invokes SearchPolicies tool for rubric fitness."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_response_with_tool_calls([
                {
                    "id": "call_001",
                    "function": {
                        "name": "search_policies",
                        "arguments": '{"query": "grading criteria for computer science", "policy_type": "assessor_rubric"}',
                    },
                },
            ]),
            _react_final_response(
                sufficient=False,
                gap_analysis=[
                    {
                        "topic": "Algorithms",
                        "current_depth": "surface",
                        "required_depth": "deep",
                        "gap_description": "No content on algorithms",
                        "fillable_by_web": True,
                        "confidence": 0.25,
                    },
                ],
                search_queries=["advanced sorting algorithms complexity analysis"],
                rubric_fitness="ALIGNED",
                rubric_source="assessor_upload",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(5)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.sufficient is False
        assert len(result.search_queries) > 0

    async def test_react_probe_returns_rubric_fitness_aligned(self) -> None:
        """AC-6: Prober returns rubric_fitness=ALIGNED when rubric matches material."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=True,
                rubric_fitness="ALIGNED",
                rubric_reasoning="Rubric covers OOP and data structures which match material",
                rubric_source="assessor_upload",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.sufficient is True
        assert result.rubric_fitness == "ALIGNED"
        assert result.rubric_source == "assessor_upload"
        assert "OOP" in result.rubric_reasoning or result.rubric_reasoning != ""

    async def test_react_probe_returns_rubric_fitness_misaligned(self) -> None:
        """AC-7: Prober returns rubric_fitness=MISALIGNED when rubric doesn't match."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=True,
                rubric_fitness="MISALIGNED",
                rubric_reasoning="Rubric focuses on grammar; material is about algorithms",
                rubric_source="system_default",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.sufficient is True
        assert result.rubric_fitness == "MISALIGNED"
        assert result.rubric_source == "system_default"

    async def test_react_probe_returns_no_rubric_when_absent(self) -> None:
        """AC-8: Prober returns rubric_fitness=NO_RUBRIC when no rubric exists."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=True,
                rubric_fitness="NO_RUBRIC",
                rubric_reasoning="",
                rubric_source="none",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert result.rubric_fitness == "NO_RUBRIC"
        assert result.rubric_source == "none"

    async def test_rubric_block_injected_into_prompt(self) -> None:
        """AC-9: rubric_block is passed through on the user turn (v5+ system/user split)."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(sufficient=True, rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(10)
        config = _make_config()
        rubric_block = "RUBRIC SOURCE: admin_seeded\nRUBRIC TYPE: system_default\n\nGrade on clarity and accuracy."
        await prober.probe(chunks, config, rubric_block=rubric_block)

        # v5+ splits persona (system) from task context (user). The rubric
        # block is dynamic per-workflow content and belongs on the user turn.
        first_call = mb.invocations[0]
        messages = first_call["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Grade on clarity and accuracy" in messages[1]["content"]


class TestReActAutonomyReasoning:
    """Tests for autonomy reasoning with web_research_mode."""

    async def test_auto_mode_produces_search_queries(self) -> None:
        """AC-10: When web_research_mode=auto, LLM can produce search queries."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=False,
                gap_analysis=[
                    {
                        "topic": "Machine Learning",
                        "current_depth": "surface",
                        "required_depth": "deep",
                        "gap_description": "No ML content",
                        "fillable_by_web": True,
                        "confidence": 0.20,
                    },
                ],
                search_queries=[
                    "machine learning fundamentals supervised learning",
                    "neural network architectures assessment",
                ],
                autonomy_exercised=True,
                rubric_fitness="NO_RUBRIC",
                rubric_source="none",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(5)
        config = _make_config(web_research_mode=WebResearchMode.AUTO)
        result = await prober.probe(chunks, config)

        assert result.sufficient is False
        assert result.autonomy_exercised is True
        assert len(result.search_queries) == 2

    async def test_manual_mode_no_autonomy(self) -> None:
        """AC-11: When web_research_mode=manual, autonomy is not exercised."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=False,
                gap_analysis=[],
                autonomy_exercised=False,
                rubric_fitness="ALIGNED",
                rubric_source="admin_seeded",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(5)
        config = _make_config(web_research_mode=WebResearchMode.MANUAL)
        result = await prober.probe(chunks, config)

        assert result.autonomy_exercised is False

    async def test_react_prompt_version_format(self) -> None:
        """AC-12: ReactSufficiencyProber has correct prompt version format (v5)."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        prober = _build_prober(mb, ks)

        assert prober.prompt_version.startswith("classifier-agent/")
        assert "@v" in prober.prompt_version
        assert "@v5" in prober.prompt_version


class TestReActResultModel:
    """Tests for the unified ReAct result model."""

    async def test_result_includes_search_queries(self) -> None:
        """Unified result includes search_queries for web research dispatch."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=False,
                search_queries=["query1", "query2"],
                rubric_fitness="NO_RUBRIC",
                rubric_source="none",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(5)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert hasattr(result, "search_queries")
        assert result.search_queries == ["query1", "query2"]

    async def test_result_includes_rubric_fields(self) -> None:
        """Unified result always includes rubric_fitness, rubric_reasoning, rubric_source."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(
                sufficient=True,
                rubric_fitness="ALIGNED",
                rubric_reasoning="Topics align well with rubric criteria",
                rubric_source="assessor_upload",
            ),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert hasattr(result, "rubric_fitness")
        assert hasattr(result, "rubric_reasoning")
        assert hasattr(result, "rubric_source")
        assert result.rubric_fitness == "ALIGNED"
        assert result.rubric_source == "assessor_upload"

    async def test_result_includes_autonomy_flag(self) -> None:
        """Unified result includes autonomy_exercised flag."""
        ks = StubKnowledgeServiceAdapter()
        mb = StubModelBrokerAdapter()

        mb.set_tool_call_responses("classification.react_sufficiency", [
            _react_final_response(sufficient=True, rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        prober = _build_prober(mb, ks)

        chunks = _make_chunks(20)
        config = _make_config()
        result = await prober.probe(chunks, config)

        assert hasattr(result, "autonomy_exercised")
