"""Targeted tests for TopicExtractionRunner — covers happy path + guardrail terminal."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    ClassificationRequest,
    ClassificationType,
    DifficultyLevel,
)
from classification_agent.domain.topic_extractor import GuardrailBlockedError, TopicExtractor
from classification_agent.domain.use_cases.topic_extraction_runner import (
    TopicExtractionRunner,
)


def _make_request() -> ClassificationRequest:
    return ClassificationRequest(
        workflow_id="wf-test",
        assessment_id="assess-test",
        assessor_id="assessor-test",
        classification_type=ClassificationType.SUFFICIENCY_AND_TOPICS,
    )


def _make_config() -> AssessmentConfig:
    return AssessmentConfig(
        assessment_id="assess-test",
        assessment_title="Test Assessment",
        structured_question_count=5,
        non_structured_question_count=2,
        difficulty_level=DifficultyLevel.MEDIUM,
    )


def _make_chunks(n: int = 3) -> list[ChunkData]:
    return [
        ChunkData(
            chunk_id=f"chunk-{i}",
            workflow_id="wf-test",
            content=f"Sample content for chunk {i}",
            source_type="direct_text",
        )
        for i in range(n)
    ]


def _make_runner(
    *,
    topic_extractor: TopicExtractor,
    decision_audit: StubDecisionAuditAdapter,
    event_publisher: StubEventPublisherAdapter,
    tracing=None,
) -> TopicExtractionRunner:
    return TopicExtractionRunner(
        topic_extractor=topic_extractor,
        knowledge_service=StubKnowledgeServiceAdapter(),
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        tracing=tracing,
        sufficiency_confidence_provider=lambda: 0.85,
    )


@pytest.mark.asyncio
async def test_topic_extraction_happy_path(model_broker_stub):
    extractor = TopicExtractor(model_broker=model_broker_stub)
    decision_audit = StubDecisionAuditAdapter()
    event_publisher = StubEventPublisherAdapter()

    runner = _make_runner(
        topic_extractor=extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
    )

    outcome = await runner.run(_make_chunks(), _make_config(), _make_request())

    assert outcome.terminal_response is None
    assert outcome.topics is not None
    assert outcome.extract_latency_ms >= 0
    assert outcome.store_latency_ms >= 0


@pytest.mark.asyncio
async def test_topic_extraction_guardrail_publishes_terminal(model_broker_stub):
    extractor = TopicExtractor(model_broker=model_broker_stub)
    extractor.extract = AsyncMock(side_effect=GuardrailBlockedError("blocked by output PII filter"))

    decision_audit = StubDecisionAuditAdapter()
    event_publisher = StubEventPublisherAdapter()

    runner = _make_runner(
        topic_extractor=extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
    )

    outcome = await runner.run(_make_chunks(), _make_config(), _make_request())

    assert outcome.terminal_response is not None
    response = outcome.terminal_response
    assert response.sufficient is False
    assert response.topics is None
    assert "blocked by output PII filter" in response.reason

    # Decision audit was written
    assert len(decision_audit.entries) == 1
    entry = decision_audit.entries[0]
    assert entry.output_summary["terminal_signal"]["status"] == "TERMINATE"
    assert entry.output_summary["terminal_signal"]["reason_code"] == "GUARDRAIL_BLOCKED_TOPIC_EXTRACTION"

    # Completion event was published with terminal payload
    published = event_publisher.events
    assert len(published) == 1
    assert published[0]["topic"] == "assessorflow.classification.complete"
    payload = published[0]["payload"]
    assert payload["reason_code"] == "GUARDRAIL_BLOCKED_TOPIC_EXTRACTION"
    assert payload["terminal_signal"]["status"] == "TERMINATE"


@pytest.mark.asyncio
async def test_topic_extraction_traces_when_tracing_provided(model_broker_stub):
    extractor = TopicExtractor(model_broker=model_broker_stub)
    decision_audit = StubDecisionAuditAdapter()
    event_publisher = StubEventPublisherAdapter()
    tracing = AsyncMock()

    runner = _make_runner(
        topic_extractor=extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        tracing=tracing,
    )

    await runner.run(_make_chunks(), _make_config(), _make_request())

    # Both LLM call and store_topics tool call were traced
    assert tracing.trace_llm_call.await_count == 1
    assert tracing.trace_tool_call.await_count == 1


@pytest.mark.asyncio
async def test_topic_extraction_terminal_traces_decision_when_tracing(model_broker_stub):
    extractor = TopicExtractor(model_broker=model_broker_stub)
    extractor.extract = AsyncMock(side_effect=GuardrailBlockedError("blocked"))

    decision_audit = StubDecisionAuditAdapter()
    event_publisher = StubEventPublisherAdapter()
    tracing = AsyncMock()

    runner = _make_runner(
        topic_extractor=extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        tracing=tracing,
    )

    outcome = await runner.run(_make_chunks(), _make_config(), _make_request())

    assert outcome.terminal_response is not None
    assert tracing.trace_decision.await_count == 1


@pytest.mark.asyncio
async def test_topic_extraction_swallows_tracing_failures(model_broker_stub):
    """A tracing failure must not break the happy path."""
    extractor = TopicExtractor(model_broker=model_broker_stub)
    decision_audit = StubDecisionAuditAdapter()
    event_publisher = StubEventPublisherAdapter()
    tracing = AsyncMock()
    tracing.trace_llm_call.side_effect = RuntimeError("langfuse offline")
    tracing.trace_tool_call.side_effect = RuntimeError("langfuse offline")

    runner = _make_runner(
        topic_extractor=extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        tracing=tracing,
    )

    outcome = await runner.run(_make_chunks(), _make_config(), _make_request())
    assert outcome.terminal_response is None
    assert outcome.topics is not None


@pytest.mark.asyncio
async def test_topic_extraction_terminal_swallows_tracing_failure(model_broker_stub):
    """A tracing failure on the terminal path must not mask the terminal signal."""
    extractor = TopicExtractor(model_broker=model_broker_stub)
    extractor.extract = AsyncMock(side_effect=GuardrailBlockedError("blocked"))

    decision_audit = StubDecisionAuditAdapter()
    event_publisher = StubEventPublisherAdapter()
    tracing = AsyncMock()
    tracing.trace_decision.side_effect = RuntimeError("langfuse offline")

    runner = _make_runner(
        topic_extractor=extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        tracing=tracing,
    )

    outcome = await runner.run(_make_chunks(), _make_config(), _make_request())

    assert outcome.terminal_response is not None
    assert outcome.terminal_response.sufficient is False
