"""GrpcAssessmentConfigAdapter -- Submission Service gRPC (Phase 6C).

Thin mapping layer between the agent's domain-oriented
:class:`AssessmentConfigPort` interface and the canonical
:class:`SubmissionClient`. Every cross-cutting concern (retry, timeouts,
channel lifecycle, logging) lives in the client so this file stays
focused on translation: proto messages -> agent pydantic schemas.

Environment variables:
    SUBMISSION_SERVICE_GRPC_URL: host:port of the submission service gRPC
        endpoint. REQUIRED in every environment — no source-code default.
        In-cluster value: ``submission-service.af-submission.svc.cluster.local:9001``.
"""
from __future__ import annotations

import os

import structlog

from classification_agent.api.schemas import (
    AssessmentConfig,
    DifficultyLevel,
    WebResearchMode,
)
from classification_agent.clients.submission_client import SubmissionClient
from classification_agent.ports.assessment_config_port import AssessmentConfigPort

logger = structlog.get_logger(__name__)


# Fallback defaults when the row is not found or the proto comes back empty.
# Matches the defaults used by the stub adapter so local dev stays stable.
_FALLBACK_TITLE = "Untitled Assessment"
_FALLBACK_MCQ = int(os.getenv("DEFAULT_MCQ_COUNT", "10"))
_FALLBACK_OE = int(os.getenv("DEFAULT_OE_COUNT", "5"))


class GrpcAssessmentConfigAdapter(AssessmentConfigPort):
    """Submission Service gRPC adapter for assessment config reads."""

    def __init__(
        self,
        *,
        client: SubmissionClient | None = None,
    ) -> None:
        # Allow tests to inject a preconfigured client; otherwise we
        # build the default one that reads SUBMISSION_SERVICE_GRPC_URL.
        self._client = client or SubmissionClient()

    async def close(self) -> None:
        await self._client.close()

    # -- AssessmentConfigPort --------------------------------------------

    async def get_assessment_config(self, assessment_id: str) -> AssessmentConfig:
        try:
            response = await self._client.get_assessment_config(
                assessment_id=assessment_id,
            )
        except Exception:
            logger.warning(
                "assessment_config_fetch_failed",
                assessment_id=assessment_id,
                exc_info=True,
            )
            return AssessmentConfig(
                assessment_id=assessment_id,
                assessment_title=_FALLBACK_TITLE,
                structured_question_count=_FALLBACK_MCQ,
                non_structured_question_count=_FALLBACK_OE,
                difficulty_level=DifficultyLevel.MEDIUM,
                web_research_mode=WebResearchMode.MANUAL,
            )

        config = response.config

        # Difficulty: case-insensitive mapping with MEDIUM fallback.
        raw_difficulty = (config.difficulty_level or "medium").lower()
        try:
            difficulty = DifficultyLevel(raw_difficulty)
        except ValueError:
            logger.warning(
                "assessment_config_unknown_difficulty",
                assessment_id=assessment_id,
                raw=raw_difficulty,
            )
            difficulty = DifficultyLevel.MEDIUM

        # Web research mode: MANUAL fallback keeps the classifier on the
        # basic (non-ReAct) sufficiency path when the config omits it.
        raw_mode = (config.web_research_mode or "manual").lower()
        try:
            web_mode = WebResearchMode(raw_mode)
        except ValueError:
            logger.warning(
                "assessment_config_unknown_web_research_mode",
                assessment_id=assessment_id,
                raw=raw_mode,
            )
            web_mode = WebResearchMode.MANUAL

        result = AssessmentConfig(
            assessment_id=config.assessment_id or assessment_id,
            assessment_title=config.assessment_title or _FALLBACK_TITLE,
            structured_question_count=(
                config.structured_question_count or _FALLBACK_MCQ
            ),
            non_structured_question_count=(
                config.non_structured_question_count or _FALLBACK_OE
            ),
            difficulty_level=difficulty,
            web_research_mode=web_mode,
        )
        logger.info(
            "assessment_config_loaded",
            assessment_id=result.assessment_id,
            mcq=result.structured_question_count,
            oe=result.non_structured_question_count,
            difficulty=result.difficulty_level.value,
            web_research_mode=result.web_research_mode.value,
        )
        return result
