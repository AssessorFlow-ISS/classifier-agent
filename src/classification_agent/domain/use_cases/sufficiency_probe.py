from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from af_shared.models.domain import DecisionLogEntry
from af_shared.ports.tracing import TracingPort
from classification_agent.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    RubricFitnessResult,
)
from classification_agent.domain.sufficiency import (
    ReactSufficiencyProber,
    ReactSufficiencyResult,
)
from classification_agent.ports.decision_audit_port import DecisionAuditPort
from classification_agent.ports.event_publisher_port import EventPublisherPort
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

ReactProberFactory = Callable[[str, str], ReactSufficiencyProber]

_AGENT_NAME = "classification-agent"
_PHASE_SUFFICIENCY = "Phase 4: Material Sufficiency Check"
_TOPIC_INSUFFICIENT = "assessorflow.classification.insufficient"

_RUBRIC_ALIGNED = "ALIGNED"
_RUBRIC_MISALIGNED = "MISALIGNED"
_RUBRIC_NO_RUBRIC = "NO_RUBRIC"


def react_result_to_rubric_fitness(react_result: ReactSufficiencyResult) -> RubricFitnessResult:
    """Convert unified probe rubric fields to a RubricFitnessResult.

    Maps the three-value ``rubric_fitness`` string from the probe to
    ``is_aligned`` / ``alignment_score`` fields used by the existing
    orchestrator payload schema and frontend trace UI.
    """
    fitness = react_result.rubric_fitness
    if fitness == _RUBRIC_ALIGNED:
        return RubricFitnessResult(
            is_aligned=True,
            rubric_source=react_result.rubric_source,
            alignment_score=1.0,
            gap_description=None,
            recommendation="use_as_is",
        )
    elif fitness == _RUBRIC_MISALIGNED:
        return RubricFitnessResult(
            is_aligned=False,
            rubric_source=react_result.rubric_source,
            alignment_score=0.0,
            gap_description=react_result.rubric_reasoning or "Rubric misaligned with material topics",
            recommendation="synthesize_new",
        )
    else:  # NO_RUBRIC
        return RubricFitnessResult(
            is_aligned=False,
            rubric_source="none",
            alignment_score=0.0,
            gap_description=None,
            recommendation=None,
        )


def build_rubric_fetcher(knowledge_service: KnowledgeServicePort) -> Callable:
    """Return an async helper that fetches the rubric block for a given assessment."""

    async def fetch(assessment_id: str, workflow_id: str) -> str:
        try:
            assessor_rubrics = await knowledge_service.search_policies(
                query="rubric grading criteria",
                policy_type="assessor_rubric",
                assessment_id=assessment_id,
            )
            if assessor_rubrics:
                content = "\n".join(c.content for c in assessor_rubrics)
                source = assessor_rubrics[0].source
                return (
                    f"RUBRIC SOURCE: {source}\n"
                    f"RUBRIC TYPE: assessor_uploaded\n\n"
                    f"{content}"
                )

            system_defaults = await knowledge_service.search_policies(
                query="rubric grading criteria",
                policy_type="system_default",
            )
            if system_defaults:
                content = "\n".join(c.content for c in system_defaults)
                return (
                    f"RUBRIC SOURCE: admin_seeded\n"
                    f"RUBRIC TYPE: system_default\n\n"
                    f"{content}"
                )
        except Exception:
            logger.warning(
                "rubric_fetch_failed",
                workflow_id=workflow_id,
                assessment_id=assessment_id,
            )

        return "NO_RUBRIC"

    return fetch


@dataclass
class SufficiencyProbeOutcome:
    """Carries the post-probe state forward to the rest of ``classify``.

    ``insufficient_response`` is non-None when the probe concluded the
    workflow should terminate at sufficiency — caller returns it directly.
    Otherwise ``react_result`` holds the probe data for downstream decision
    log composition.
    """

    insufficient_response: ClassificationResponse | None
    react_result: ReactSufficiencyResult | None
    sufficiency_confidence: float


class SufficiencyProbeRunner:
    """Runs the unified ReAct sufficiency + rubric probe and handles its failure path.

    On a failure outcome, also writes a decision log, traces the decision, and
    publishes the ``classification.insufficient`` event before returning a
    pre-built ``ClassificationResponse`` for the orchestrator to forward.
    """

    def __init__(
        self,
        *,
        react_prober_factory: ReactProberFactory | None,
        knowledge_service: KnowledgeServicePort,
        decision_audit: DecisionAuditPort,
        event_publisher: EventPublisherPort,
        tracing: TracingPort | None,
    ) -> None:
        self._react_prober_factory = react_prober_factory
        self._decision_audit = decision_audit
        self._event_publisher = event_publisher
        self._tracing = tracing
        self._fetch_rubric_block = build_rubric_fetcher(knowledge_service)

    async def run(
        self,
        chunks: list,
        config,
        request: ClassificationRequest,
    ) -> SufficiencyProbeOutcome:
        if self._react_prober_factory is None:
            return SufficiencyProbeOutcome(
                insufficient_response=None,
                react_result=None,
                sufficiency_confidence=0.90,
            )

        prober = self._react_prober_factory(
            request.workflow_id, request.assessment_id,
        )

        rubric_block = await self._fetch_rubric_block(
            request.assessment_id, request.workflow_id,
        )

        react_result = await prober.probe(chunks, config, rubric_block=rubric_block)

        probe_failed = (
            not react_result.sufficient
            or react_result.rubric_fitness == _RUBRIC_MISALIGNED
        )

        if not probe_failed:
            return SufficiencyProbeOutcome(
                insufficient_response=None,
                react_result=react_result,
                sufficiency_confidence=0.90,
            )

        if not react_result.sufficient:
            reason_code = "MATERIAL_INSUFFICIENT"
            reason_msg = react_result.reason
        else:
            reason_code = "MATERIAL_INSUFFICIENT"
            reason_msg = f"Rubric misaligned: {react_result.rubric_reasoning}"

        entry = DecisionLogEntry(
            workflow_id=request.workflow_id,
            agent_name=_AGENT_NAME,
            decision_type="classification_governance",
            assessor_id=request.assessor_id,
            input_summary={
                "phase": _PHASE_SUFFICIENCY,
                "chunk_count": len(chunks),
                "assessment_id": request.assessment_id,
                "tools_used": ["ks-get-chunks-by-workflow", "react-unified-probe"],
            },
            output_summary={
                "sufficient": False,
                "reason": reason_msg,
                "gap_analysis": [g.model_dump() for g in react_result.gap_analysis],
                "rubric_fitness": react_result.rubric_fitness,
                "autonomy_exercised": react_result.autonomy_exercised,
                "search_queries": react_result.search_queries,
            },
            reasoning_steps=[
                {
                    "step": 1,
                    "action": f"Retrieved {len(chunks)} chunks for workflow",
                },
                {
                    "step": 2,
                    "action": (
                        f"Unified ReAct probe FAILED: sufficient={react_result.sufficient}, "
                        f"rubric_fitness={react_result.rubric_fitness}"
                    ),
                },
            ],
            confidence_score=round(len(chunks) / max(len(chunks) + 1, 1), 4),
            prompt_version=prober.prompt_version,
            model_id=prober.last_model_used,
            grounding_sources=[c.chunk_id for c in chunks],
        )

        await self._decision_audit.log_decision(entry)

        if self._tracing:
            try:
                await self._tracing.trace_decision(entry)
            except Exception:
                logger.warning(
                    "tracing_decision_failed",
                    decision_type="classification_governance",
                )

        rubric_fitness_result = react_result_to_rubric_fitness(react_result)

        insuff_payload = {
            "workflow_id": request.workflow_id,
            "assessment_id": request.assessment_id,
            "source_agent": "classification-agent",
            "reason_code": reason_code,
            "message": reason_msg,
            "assessment_topic": config.assessment_title,
            "search_focus": config.assessment_title,
            "gap_areas": [g.model_dump() for g in react_result.gap_analysis],
            "identified_topics": [c.content[:80] for c in chunks[:5]],
            "chunk_count": len(chunks),
            "difficulty_level": config.difficulty_level,
            "total_questions": (
                config.structured_question_count
                + config.non_structured_question_count
            ),
            "gap_analysis": [g.model_dump() for g in react_result.gap_analysis],
            "search_queries": react_result.search_queries,
            "autonomy_exercised": react_result.autonomy_exercised,
            "rubric_fitness": rubric_fitness_result.model_dump(),
            "rubric_source": react_result.rubric_source,
        }

        await self._event_publisher.publish(_TOPIC_INSUFFICIENT, insuff_payload)

        response = ClassificationResponse(
            workflow_id=request.workflow_id,
            sufficient=False,
            reason=reason_msg,
            gap_analysis=react_result.gap_analysis,
            topics=None,
            rubric_fitness=rubric_fitness_result,
            rubric_source=react_result.rubric_source,
            autonomy_exercised=react_result.autonomy_exercised,
            search_queries=react_result.search_queries,
        )

        return SufficiencyProbeOutcome(
            insufficient_response=response,
            react_result=react_result,
            sufficiency_confidence=0.90,
        )
