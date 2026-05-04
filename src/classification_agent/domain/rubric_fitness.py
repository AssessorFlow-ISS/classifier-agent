"""Rubric fitness assessment (AF-139).

The RubricFitnessAssessor searches for assessor rubric via SearchPolicies,
falls back to system defaults, and reasons about semantic alignment between
rubric and material topics.  It signals misalignment to the Orchestrator
but does NOT write to Policy KB — rubric synthesis flows through
Web Research → Validator → Knowledge Service (Thet Q-3).
"""
from __future__ import annotations

from pathlib import Path

import structlog

from af_shared.utils.prompt_loader import get_prompt_version, load_prompt

from classification_agent.api.schemas import (
    RubricFitnessResult,
    TopicHierarchy,
)
from classification_agent.domain.response_models import RubricFitnessResponse
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort
from classification_agent.ports.model_broker_port import ModelBrokerPort

logger = structlog.get_logger(__name__)

_RUBRIC_FITNESS_TASK_KEY = "classification.rubric_fitness"
_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "rubric_fitness"
_, _RUBRIC_FITNESS_TEMPLATE = load_prompt(_PROMPT_PATH)


class RubricFitnessAssessor:
    """Assesses rubric-material alignment and signals misalignment.

    Workflow:
    1. Search for assessor rubric via SearchPolicies (assessor_rubric type)
    2. If no assessor rubric, search system defaults (system_default type)
    3. If no rubric at all, return NO_RUBRIC result
    4. Call LLM to assess alignment between rubric and topics
    5. Return result with recommendation — Orchestrator decides next action
    """

    def __init__(
        self,
        model_broker: ModelBrokerPort,
        knowledge_service: KnowledgeServicePort,
    ) -> None:
        self._model_broker = model_broker
        self._knowledge_service = knowledge_service
        self._prompt_version = get_prompt_version(_PROMPT_PATH)
        self.last_model_used: str = "unknown"

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    async def assess(
        self,
        topics: TopicHierarchy,
        assessment_id: str,
        workflow_id: str,
    ) -> RubricFitnessResult:
        """Assess rubric fitness against extracted topics."""
        topic_names = [t.name for t in topics.topics]

        # 1. Search for assessor rubric
        assessor_rubrics = await self._knowledge_service.search_policies(
            query="rubric grading criteria",
            policy_type="assessor_rubric",
            assessment_id=assessment_id,
        )

        if assessor_rubrics:
            rubric_chunks = assessor_rubrics
            rubric_source = assessor_rubrics[0].source
        else:
            # 2. Fall back to system defaults
            system_defaults = await self._knowledge_service.search_policies(
                query="rubric grading criteria",
                policy_type="system_default",
            )
            if system_defaults:
                rubric_chunks = system_defaults
                rubric_source = "system_default"
            else:
                # 3. No rubric at all
                logger.info(
                    "rubric_fitness_no_rubric",
                    workflow_id=workflow_id,
                    assessment_id=assessment_id,
                )
                return RubricFitnessResult(
                    is_aligned=False,
                    rubric_source="none",
                    alignment_score=0.0,
                    gap_description="No rubric found in Policy KB",
                    recommendation="synthesize_new",
                )

        # 4. Call LLM to assess alignment
        rubric_content = "\n".join(c.content for c in rubric_chunks)
        topic_summary = ", ".join(topic_names)

        prompt = _RUBRIC_FITNESS_TEMPLATE.format(
            rubric_content=rubric_content,
            topic_summary=topic_summary,
        )

        from classification_agent.domain.schemas_json import RUBRIC_FITNESS_SCHEMA

        llm_result = await self._model_broker.invoke(
            _RUBRIC_FITNESS_TASK_KEY,
            prompt,
            workflow_id=workflow_id,
            response_format="json",
            response_schema=RUBRIC_FITNESS_SCHEMA,
            prompt_version=self._prompt_version,
        )
        self.last_model_used = llm_result.get("model_used", "unknown")

        # Validate LLM response via Pydantic model
        parsed = RubricFitnessResponse.model_validate(llm_result)

        is_aligned = parsed.is_aligned
        alignment_score = parsed.alignment_score
        gap_description = parsed.gap_description
        recommendation = parsed.recommendation

        # 5. Signal misalignment via result — Orchestrator dispatches
        # Web Research Agent if recommendation is synthesize_new (Thet Q-3).
        # Classification Agent does NOT write to Policy KB.

        logger.info(
            "rubric_fitness_assessed",
            workflow_id=workflow_id,
            is_aligned=is_aligned,
            rubric_source=rubric_source,
            alignment_score=alignment_score,
        )

        return RubricFitnessResult(
            is_aligned=is_aligned,
            rubric_source=rubric_source,
            alignment_score=alignment_score,
            gap_description=gap_description,
            recommendation=recommendation,
        )
