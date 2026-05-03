from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    DifficultyLevel,
)
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.tools.registry import build_react_prober_factory

_FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Asyncpg isolation
#
# ``ClassificationService._write_progress_event`` opens a real asyncpg
# connection to the Orchestrator DB. In unit tests there is no DB running,
# so we patch ``asyncpg.connect`` to raise and let the service's own
# ``try/except`` swallow it. This keeps tests hermetic and silent.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_asyncpg_connect():
    """Prevent _write_progress_event from making real DB connections."""
    with patch(
        "asyncpg.connect",
        new=AsyncMock(side_effect=ConnectionError("asyncpg disabled in unit tests")),
    ):
        yield


def _load_chunks(filename: str) -> list[ChunkData]:
    with open(_FIXTURES / filename) as f:
        raw = json.load(f)
    return [ChunkData(**item) for item in raw]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sufficient_chunks() -> list[ChunkData]:
    return _load_chunks("sufficient_chunks.json")


@pytest.fixture()
def insufficient_chunks() -> list[ChunkData]:
    return _load_chunks("insufficient_chunks.json")


@pytest.fixture()
def default_config() -> AssessmentConfig:
    return AssessmentConfig(
        assessment_id="assess-001",
        assessment_title="Computer Science Fundamentals",
        structured_question_count=10,
        non_structured_question_count=5,
        difficulty_level=DifficultyLevel.MEDIUM,
    )


@pytest.fixture()
def mcq_only_config() -> AssessmentConfig:
    return AssessmentConfig(
        assessment_id="assess-mcq",
        assessment_title="MCQ-Only Assessment",
        structured_question_count=8,
        non_structured_question_count=0,
        difficulty_level=DifficultyLevel.EASY,
    )


@pytest.fixture()
def heavy_open_ended_config() -> AssessmentConfig:
    return AssessmentConfig(
        assessment_id="assess-open",
        assessment_title="Open-Ended Heavy Assessment",
        structured_question_count=2,
        non_structured_question_count=10,
        difficulty_level=DifficultyLevel.HARD,
    )


# ---------------------------------------------------------------------------
# Adapter stubs
# ---------------------------------------------------------------------------

@pytest.fixture()
def knowledge_service_stub() -> StubKnowledgeServiceAdapter:
    return StubKnowledgeServiceAdapter()


@pytest.fixture()
def assessment_config_stub() -> StubAssessmentConfigAdapter:
    return StubAssessmentConfigAdapter()


@pytest.fixture()
def model_broker_stub() -> StubModelBrokerAdapter:
    return StubModelBrokerAdapter()


@pytest.fixture()
def decision_audit_stub() -> StubDecisionAuditAdapter:
    return StubDecisionAuditAdapter()


@pytest.fixture()
def event_publisher_stub() -> StubEventPublisherAdapter:
    return StubEventPublisherAdapter()


# ---------------------------------------------------------------------------
# Domain components
# ---------------------------------------------------------------------------

@pytest.fixture()
def topic_extractor(model_broker_stub: StubModelBrokerAdapter) -> TopicExtractor:
    return TopicExtractor(model_broker=model_broker_stub)


# ---------------------------------------------------------------------------
# Full service
# ---------------------------------------------------------------------------

@pytest.fixture()
def classification_service(
    knowledge_service_stub: StubKnowledgeServiceAdapter,
    assessment_config_stub: StubAssessmentConfigAdapter,
    model_broker_stub: StubModelBrokerAdapter,
    topic_extractor: TopicExtractor,
    decision_audit_stub: StubDecisionAuditAdapter,
    event_publisher_stub: StubEventPublisherAdapter,
) -> ClassificationService:
    react_prober_factory = build_react_prober_factory(
        model_broker=model_broker_stub,
        knowledge_service=knowledge_service_stub,
    )
    return ClassificationService(
        knowledge_service=knowledge_service_stub,
        assessment_config=assessment_config_stub,
        topic_extractor=topic_extractor,
        decision_audit=decision_audit_stub,
        event_publisher=event_publisher_stub,
        react_prober_factory=react_prober_factory,
    )
