from __future__ import annotations

import structlog

from af_shared.models.domain import DecisionLogEntry
from af_shared.ports.tracing import TracingPort
from classification_agent.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    RubricFitnessResult,
)
from classification_agent.domain.sufficiency import ReactSufficiencyResult
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.ports.decision_audit_port import DecisionAuditPort
from classification_agent.ports.event_publisher_port import EventPublisherPort

logger = structlog.get_logger(__name__)

_AGENT_NAME = "classification-agent"
_PHASE_SUFFICIENCY = "Phase 4: Material Sufficiency Check"
_PHASE_TOPICS = "Phase 4: Topic Extraction"
_TOPIC_COMPLETE = "assessorflow.classification.complete"


class DecisionRecorder:
    """Composes the success-path decision log + completion event + response.

    Splits out the ~120 lines of payload composition that previously sat at
    the bottom of ``ClassificationService.classify``.
    """

    def __init__(
        self,
        *,
        topic_extractor: TopicExtractor,
        decision_audit: DecisionAuditPort,
        event_publisher: EventPublisherPort,
        tracing: TracingPort | None,
    ) -> None:
        self._topic_extractor = topic_extractor
        self._decision_audit = decision_audit
        self._event_publisher = event_publisher
        self._tracing = tracing

    async def record_success(
        self,
        *,
        request: ClassificationRequest,
        chunks: list,
        topics: object | None,
        react_probe_result: ReactSufficiencyResult | None,
        rubric_fitness_result: RubricFitnessResult | None,
        rubric_source: str | None,
        sufficiency_confidence: float,
    ) -> ClassificationResponse:
        topic_names = [t.name for t in topics.topics] if topics else []
        reasoning_steps = [
            {
                "step": 1,
                "action": (
                    f"Retrieved {len(chunks)} chunks from Knowledge Service for workflow "
                    f"{request.workflow_id}"
                ),
                "component": "knowledge_service",
            },
            {
                "step": 2,
                "action": (
                    f"Unified ReAct probe: sufficiency PASS ({len(chunks)} chunks), "
                    f"rubric_fitness="
                    f"{react_probe_result.rubric_fitness if react_probe_result else 'NO_RUBRIC'}"
                ),
                "component": "sufficiency_check",
            },
        ]
        if topics is not None:
            te_conf = round(float(getattr(topics, "confidence", 0.0)), 4)
            reasoning_steps.append({
                "step": 3,
                "action": f"Extracted {len(topics.topics)} topics: {', '.join(topic_names[:8])}",
                "component": "topic_extraction",
                "confidence": te_conf,
            })
            reasoning_steps.append({
                "step": 4,
                "action": f"Stored {len(topics.topics)} topics to Knowledge Service",
                "tool": "ks-store-topics",
                "component": "topic_extraction",
                "confidence": round(sufficiency_confidence, 4),
            })
        if rubric_fitness_result is not None:
            reasoning_steps.append({
                "step": len(reasoning_steps) + 1,
                "action": (
                    f"Rubric fitness: "
                    f"{react_probe_result.rubric_fitness if react_probe_result else 'NO_RUBRIC'}, "
                    f"source={rubric_source}"
                ),
                "component": "rubric_fitness",
                "confidence": round(float(rubric_fitness_result.alignment_score), 4),
            })

        entry = DecisionLogEntry(
            workflow_id=request.workflow_id,
            agent_name=_AGENT_NAME,
            decision_type="classification_governance",
            assessor_id=request.assessor_id,
            input_summary={
                "phase": _PHASE_TOPICS if topics else _PHASE_SUFFICIENCY,
                "chunk_count": len(chunks),
                "assessment_id": request.assessment_id,
                "tools_used": [
                    "ks-get-chunks-by-workflow",
                    "react-unified-probe",
                    "topic-extraction-llm",
                    "ks-store-topics",
                ],
            },
            output_summary={
                "sufficient": True,
                "topic_count": len(topics.topics) if topics else 0,
                **({"rubric_fitness": {
                    "alignment_score": rubric_fitness_result.alignment_score,
                    "is_aligned": rubric_fitness_result.is_aligned,
                    "rubric_source": rubric_fitness_result.rubric_source,
                    "gap_description": rubric_fitness_result.gap_description,
                }} if rubric_fitness_result is not None else {}),
                "terminal_signal": {
                    "status": "PROCEED",
                    "reason_code": "CLASSIFICATION_PASS",
                    "message": (
                        f"Classification PASS — "
                        f"{len(topics.topics) if topics else 0} topic(s) extracted, "
                        f"rubric "
                        f"{react_probe_result.rubric_fitness.lower() if react_probe_result else 'no_rubric'}."
                    ),
                },
            },
            reasoning_steps=reasoning_steps,
            confidence_score=sufficiency_confidence,
            prompt_version=self._topic_extractor.prompt_version,
            model_id=self._topic_extractor.last_model_used,
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

        completion_payload = {
            "workflow_id": request.workflow_id,
            "assessment_id": request.assessment_id,
            "source_agent": "classification-agent",
            "reason_code": "CLASSIFICATION_COMPLETE",
            "message": "Classification completed successfully",
            "topic_count": len(topics.topics) if topics else 0,
        }
        if rubric_fitness_result is not None:
            completion_payload["rubric_fitness"] = rubric_fitness_result.model_dump()
            completion_payload["rubric_source"] = rubric_source

        await self._event_publisher.publish(_TOPIC_COMPLETE, completion_payload)

        logger.info(
            "classification_complete",
            workflow_id=request.workflow_id,
            sufficient=True,
            topic_count=len(topics.topics) if topics else 0,
        )

        return ClassificationResponse(
            workflow_id=request.workflow_id,
            sufficient=True,
            reason="Material sufficient for assessment requirements",
            gap_analysis=[],
            topics=topics,
            rubric_fitness=rubric_fitness_result,
            rubric_source=rubric_source,
            autonomy_exercised=react_probe_result.autonomy_exercised if react_probe_result else False,
            search_queries=react_probe_result.search_queries if react_probe_result else [],
        )
