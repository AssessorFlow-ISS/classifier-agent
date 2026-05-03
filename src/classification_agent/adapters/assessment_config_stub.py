from __future__ import annotations

import os

from classification_agent.api.schemas import AssessmentConfig, DifficultyLevel
from classification_agent.ports.assessment_config_port import AssessmentConfigPort

_DEFAULT_CONFIG = AssessmentConfig(
    assessment_id="default",
    assessment_title="Untitled Assessment",
    structured_question_count=int(os.getenv("DEFAULT_MCQ_COUNT", "10")),
    non_structured_question_count=int(os.getenv("DEFAULT_OE_COUNT", "5")),
    difficulty_level=DifficultyLevel.MEDIUM,
)


class StubAssessmentConfigAdapter(AssessmentConfigPort):
    """In-memory stub for Assessment Submission Service config endpoint.

    Returns a configurable AssessmentConfig (default: 10 MCQ + 5 open-ended, medium).
    """

    def __init__(self) -> None:
        self._configs: dict[str, AssessmentConfig] = {}

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    def set_config(self, assessment_id: str, config: AssessmentConfig) -> None:
        """Pre-set a config for a given assessment (test setup)."""
        self._configs[assessment_id] = config

    # -----------------------------------------------------------------------
    # Port implementation
    # -----------------------------------------------------------------------

    async def get_assessment_config(self, assessment_id: str) -> AssessmentConfig:
        if assessment_id in self._configs:
            return self._configs[assessment_id]
        # Return default config with the requested assessment_id
        return AssessmentConfig(
            assessment_id=assessment_id,
            assessment_title=_DEFAULT_CONFIG.assessment_title,
            structured_question_count=_DEFAULT_CONFIG.structured_question_count,
            non_structured_question_count=_DEFAULT_CONFIG.non_structured_question_count,
            difficulty_level=_DEFAULT_CONFIG.difficulty_level,
        )
