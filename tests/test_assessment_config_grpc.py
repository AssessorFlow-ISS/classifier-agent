"""Tests for GrpcAssessmentConfigAdapter — proto-to-domain mapping.

Covers the happy path, the fallback branch triggered by client exceptions,
and the case-insensitive difficulty / web_research_mode handling with
MEDIUM / MANUAL fallbacks for unknown enum values.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from classification_agent._grpc import submission_pb2
from classification_agent.adapters.assessment_config_grpc import (
    GrpcAssessmentConfigAdapter,
)
from classification_agent.api.schemas import DifficultyLevel, WebResearchMode


def _response(**kwargs) -> submission_pb2.GetAssessmentConfigResponse:
    config = submission_pb2.AssessmentConfig(**kwargs)
    return submission_pb2.GetAssessmentConfigResponse(config=config)


def _adapter_with_response(resp) -> tuple[GrpcAssessmentConfigAdapter, MagicMock]:
    client = MagicMock()
    client.get_assessment_config = AsyncMock(return_value=resp)
    client.close = AsyncMock()
    adapter = GrpcAssessmentConfigAdapter(client=client)
    return adapter, client


def _adapter_with_error(exc: Exception) -> tuple[GrpcAssessmentConfigAdapter, MagicMock]:
    client = MagicMock()
    client.get_assessment_config = AsyncMock(side_effect=exc)
    client.close = AsyncMock()
    return GrpcAssessmentConfigAdapter(client=client), client


class TestGrpcAdapterHappyPath:
    async def test_maps_proto_to_domain(self) -> None:
        adapter, client = _adapter_with_response(
            _response(
                assessment_id="a-1",
                assessment_title="CS 101",
                structured_question_count=12,
                non_structured_question_count=4,
                difficulty_level="HARD",
                web_research_mode="AUTO",
            )
        )
        result = await adapter.get_assessment_config("a-1")

        assert result.assessment_id == "a-1"
        assert result.assessment_title == "CS 101"
        assert result.structured_question_count == 12
        assert result.non_structured_question_count == 4
        assert result.difficulty_level == DifficultyLevel.HARD
        assert result.web_research_mode == WebResearchMode.AUTO
        client.get_assessment_config.assert_awaited_once_with(assessment_id="a-1")

    async def test_close_delegates_to_client(self) -> None:
        adapter, client = _adapter_with_response(_response(assessment_id="a-c"))
        await adapter.close()
        client.close.assert_awaited_once()


class TestGrpcAdapterFallbacks:
    async def test_client_exception_returns_fallback_defaults(self) -> None:
        adapter, _ = _adapter_with_error(RuntimeError("submission down"))

        result = await adapter.get_assessment_config("a-broken")
        assert result.assessment_id == "a-broken"
        assert result.assessment_title == "Untitled Assessment"
        assert result.difficulty_level == DifficultyLevel.MEDIUM
        assert result.web_research_mode == WebResearchMode.MANUAL

    async def test_unknown_difficulty_falls_back_to_medium(self) -> None:
        adapter, _ = _adapter_with_response(
            _response(
                assessment_id="a-2",
                assessment_title="T",
                structured_question_count=1,
                non_structured_question_count=1,
                difficulty_level="nightmare",
                web_research_mode="manual",
            )
        )
        result = await adapter.get_assessment_config("a-2")
        assert result.difficulty_level == DifficultyLevel.MEDIUM

    async def test_unknown_web_research_mode_falls_back_to_manual(self) -> None:
        adapter, _ = _adapter_with_response(
            _response(
                assessment_id="a-3",
                assessment_title="T",
                structured_question_count=1,
                non_structured_question_count=1,
                difficulty_level="easy",
                web_research_mode="chaotic",
            )
        )
        result = await adapter.get_assessment_config("a-3")
        assert result.web_research_mode == WebResearchMode.MANUAL
        assert result.difficulty_level == DifficultyLevel.EASY

    async def test_missing_fields_use_fallback_counts_and_title(self) -> None:
        # Empty config — zero ints + empty strings in proto
        adapter, _ = _adapter_with_response(_response())
        result = await adapter.get_assessment_config("a-empty")

        assert result.assessment_id == "a-empty"
        assert result.assessment_title == "Untitled Assessment"
        # Fallback values (env-defaulted) are >0
        assert result.structured_question_count > 0
        assert result.non_structured_question_count > 0
        assert result.difficulty_level == DifficultyLevel.MEDIUM
        # empty web_research_mode → "manual" fallback
        assert result.web_research_mode == WebResearchMode.MANUAL


class TestGrpcAdapterConstruction:
    def test_default_client_created_when_none_passed(self) -> None:
        # Just construct without a client — it should build a SubmissionClient
        adapter = GrpcAssessmentConfigAdapter()
        assert adapter._client is not None
