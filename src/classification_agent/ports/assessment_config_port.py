from __future__ import annotations

from abc import ABC, abstractmethod

from classification_agent.api.schemas import AssessmentConfig


class AssessmentConfigPort(ABC):
    """Port for reading assessment configuration via Assessment Submission Service.

    Provides assessment parameters (question counts, difficulty) needed for
    sufficiency checking.
    """

    @abstractmethod
    async def get_assessment_config(self, assessment_id: str) -> AssessmentConfig:
        """Retrieve assessment configuration (gRPC GetAssessmentConfig 2.2.1)."""
