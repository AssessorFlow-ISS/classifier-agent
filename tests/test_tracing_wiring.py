"""Tests for TracingPort wiring in the Classification Agent.

Verifies that the ClassificationService emits trace calls to the TracingPort
for all LLM calls, tool calls, and decision audit entries (dual-sink ADR-40).

Uses StubTracingAdapter from af_shared for test assertions.
"""
from __future__ import annotations

import pytest

from af_shared.adapters.stubs.tracing_stub import StubTracingAdapter
from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    ClassificationRequest,
    DifficultyLevel,
    SourceType,
)
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.tools.registry import build_react_prober_factory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(count: int, workflow_id: str = "wf-001") -> list[ChunkData]:
    """Create N chunks with unique IDs and content."""
    return [
        ChunkData(
            chunk_id=f"chunk-{i:03d}",
            workflow_id=workflow_id,
            content=f"Content about topic {i} covering important concepts in detail. " * 3,
            source_type=SourceType.DIRECT_TEXT,
        )
        for i in range(1, count + 1)
    ]


def _make_config(
    assessment_id: str = "assess-001",
    structured: int = 5,
    non_structured: int = 3,
) -> AssessmentConfig:
    return AssessmentConfig(
        assessment_id=assessment_id,
        assessment_title="Computer Science Fundamentals",
        structured_question_count=structured,
        non_structured_question_count=non_structured,
        difficulty_level=DifficultyLevel.MEDIUM,
    )


def _react_sufficient() -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": True,
            "reason": "Material sufficient",
            "gap_analysis": [],
            "search_queries": [],
            "autonomy_exercised": False,
            "rubric_fitness": "NO_RUBRIC",
            "rubric_reasoning": "",
            "rubric_source": "none",
        },
    }


def _react_insufficient() -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": False,
            "reason": "Insufficient material",
            "gap_analysis": [],
            "search_queries": [],
            "autonomy_exercised": False,
            "rubric_fitness": "NO_RUBRIC",
            "rubric_reasoning": "",
            "rubric_source": "none",
        },
    }


def _build_service(
    *,
    knowledge_service: StubKnowledgeServiceAdapter,
    assessment_config: StubAssessmentConfigAdapter,
    model_broker: StubModelBrokerAdapter,
    decision_audit: StubDecisionAuditAdapter,
    event_publisher: StubEventPublisherAdapter,
    tracing: StubTracingAdapter | None = None,
) -> ClassificationService:
    topic_extractor = TopicExtractor(model_broker=model_broker)
    react_prober_factory = build_react_prober_factory(
        model_broker=model_broker,
        knowledge_service=knowledge_service,
    )
    return ClassificationService(
        knowledge_service=knowledge_service,
        assessment_config=assessment_config,
        topic_extractor=topic_extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        react_prober_factory=react_prober_factory,
        tracing=tracing,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTracingWiring:
    """Verify TracingPort integration in ClassificationService."""

    @pytest.fixture()
    def stubs(self):
        """Create all stub adapters."""
        return {
            "knowledge_service": StubKnowledgeServiceAdapter(),
            "assessment_config": StubAssessmentConfigAdapter(),
            "model_broker": StubModelBrokerAdapter(),
            "decision_audit": StubDecisionAuditAdapter(),
            "event_publisher": StubEventPublisherAdapter(),
            "tracing": StubTracingAdapter(),
        }

    # -- Constructor tests --------------------------------------------------

    async def test_service_accepts_tracing_port(self, stubs):
        """ClassificationService constructor accepts optional tracing parameter."""
        service = _build_service(**stubs)
        assert service is not None

    async def test_service_works_without_tracing(self, stubs):
        """Service works when tracing is None (backward compatible)."""
        stubs_no_trace = {k: v for k, v in stubs.items() if k != "tracing"}
        service = _build_service(**stubs_no_trace, tracing=None)

        config = _make_config()
        chunks = _make_chunks(20)
        stubs["knowledge_service"].add_chunks("wf-001", chunks)
        stubs["assessment_config"].set_config("assess-001", config)
        stubs["model_broker"].set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(),
        ])

        service._knowledge_service = stubs["knowledge_service"]
        service._assessment_config = stubs["assessment_config"]

        request = ClassificationRequest(
            workflow_id="wf-001",
            assessment_id="assess-001",
        )
        response = await service.classify(request)
        assert response.sufficient is True

    # -- Decision tracing (sufficient path) ---------------------------------

    async def test_classification_traces_decision_on_sufficient(self, stubs):
        """Successful classification traces decision to Langfuse (Sink 2)."""
        config = _make_config()
        chunks = _make_chunks(20)
        stubs["knowledge_service"].add_chunks("wf-001", chunks)
        stubs["assessment_config"].set_config("assess-001", config)
        stubs["model_broker"].set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(),
        ])

        service = _build_service(**stubs)
        request = ClassificationRequest(
            workflow_id="wf-001",
            assessment_id="assess-001",
        )
        await service.classify(request)

        tracing: StubTracingAdapter = stubs["tracing"]
        assert len(tracing.decisions) == 1
        decision = tracing.decisions[0]
        assert decision.agent_name == "classification-agent"
        assert decision.decision_type == "classification_governance"
        assert decision.workflow_id == "wf-001"
        assert decision.prompt_version is not None

    # -- Decision tracing (insufficient path) -------------------------------

    async def test_insufficient_traces_decision(self, stubs):
        """Insufficient classification traces decision to Langfuse."""
        config = _make_config(structured=10, non_structured=5)
        chunks = _make_chunks(3, workflow_id="wf-sparse")
        stubs["knowledge_service"].add_chunks("wf-sparse", chunks)
        stubs["assessment_config"].set_config("assess-001", config)
        stubs["model_broker"].set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient(),
        ])

        service = _build_service(**stubs)
        request = ClassificationRequest(
            workflow_id="wf-sparse",
            assessment_id="assess-001",
        )
        await service.classify(request)

        tracing: StubTracingAdapter = stubs["tracing"]
        assert len(tracing.decisions) == 1
        decision = tracing.decisions[0]
        assert decision.agent_name == "classification-agent"
        assert decision.decision_type == "classification_governance"
        assert decision.workflow_id == "wf-sparse"

    # -- LLM call tracing (topic extraction) --------------------------------

    async def test_topic_extraction_traces_llm_call(self, stubs):
        """Topic extraction LLM call is traced to Langfuse."""
        config = _make_config()
        chunks = _make_chunks(20)
        stubs["knowledge_service"].add_chunks("wf-001", chunks)
        stubs["assessment_config"].set_config("assess-001", config)
        stubs["model_broker"].set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(),
        ])

        service = _build_service(**stubs)
        request = ClassificationRequest(
            workflow_id="wf-001",
            assessment_id="assess-001",
        )
        await service.classify(request)

        tracing: StubTracingAdapter = stubs["tracing"]
        topic_calls = [
            c for c in tracing.llm_calls
            if c["task_key"] == "classification.topic_extraction"
        ]
        assert len(topic_calls) >= 1
        assert topic_calls[0]["agent_name"] == "classification-agent"
        assert "prompt_version" in topic_calls[0]

    # -- Tool call tracing --------------------------------------------------

    async def test_knowledge_service_tool_calls_traced(self, stubs):
        """Knowledge Service tool calls (get_chunks, store_topics) are traced."""
        config = _make_config()
        chunks = _make_chunks(20)
        stubs["knowledge_service"].add_chunks("wf-001", chunks)
        stubs["assessment_config"].set_config("assess-001", config)
        stubs["model_broker"].set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(),
        ])

        service = _build_service(**stubs)
        request = ClassificationRequest(
            workflow_id="wf-001",
            assessment_id="assess-001",
        )
        await service.classify(request)

        tracing: StubTracingAdapter = stubs["tracing"]
        get_chunk_calls = [
            c for c in tracing.tool_calls
            if c["tool_name"] == "ks-get-chunks-by-workflow"
        ]
        assert len(get_chunk_calls) >= 1
        assert get_chunk_calls[0]["agent_name"] == "classification-agent"
        assert get_chunk_calls[0]["workflow_id"] == "wf-001"

    # -- Dual-sink consistency ----------------------------------------------

    async def test_dual_sink_consistency(self, stubs):
        """Decision audit (Sink 1) and tracing (Sink 2) receive same data."""
        config = _make_config()
        chunks = _make_chunks(20)
        stubs["knowledge_service"].add_chunks("wf-001", chunks)
        stubs["assessment_config"].set_config("assess-001", config)
        stubs["model_broker"].set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(),
        ])

        service = _build_service(**stubs)
        request = ClassificationRequest(
            workflow_id="wf-001",
            assessment_id="assess-001",
        )
        await service.classify(request)

        audit: StubDecisionAuditAdapter = stubs["decision_audit"]
        tracing: StubTracingAdapter = stubs["tracing"]

        assert len(audit.decisions) == 1
        assert len(tracing.decisions) == 1

        assert audit.decisions[0].workflow_id == tracing.decisions[0].workflow_id
        assert audit.decisions[0].agent_name == tracing.decisions[0].agent_name
        assert audit.decisions[0].decision_type == tracing.decisions[0].decision_type
