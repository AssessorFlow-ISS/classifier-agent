"""Thin orchestrator for the Phase 4 classification pipeline.

The bulk of the work lives in ``classification_agent.domain.use_cases``:

- ``SufficiencyProbeRunner`` runs the unified ReAct material + rubric probe
- ``TopicExtractionRunner`` extracts topics + handles the guardrail-terminal path
- ``DecisionRecorder`` composes the success-path decision log + completion event
- ``ProgressEmitter`` writes per-stage workflow_events rows for live UI sub-cards

``ClassificationService`` wires these together and runs ``classify`` step by step.
"""

from __future__ import annotations

import time

import structlog

from af_shared.ports.tracing import TracingPort
from classification_agent.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ClassificationType,
    RubricFitnessResult,
    WebResearchMode,
)
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.domain.use_cases.decision_recorder import DecisionRecorder
from classification_agent.domain.use_cases.progress_emitter import ProgressEmitter
from classification_agent.domain.use_cases.sufficiency_probe import (
    ReactProberFactory,
    SufficiencyProbeRunner,
    react_result_to_rubric_fitness,
)
from classification_agent.domain.use_cases.topic_extraction_runner import (
    TopicExtractionRunner,
)
from classification_agent.ports.assessment_config_port import AssessmentConfigPort
from classification_agent.ports.decision_audit_port import DecisionAuditPort
from classification_agent.ports.event_publisher_port import EventPublisherPort
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort

# Re-export for legacy imports (none currently external — kept for safety).
__all__ = ["ClassificationService", "ReactProberFactory"]
_react_result_to_rubric_fitness = react_result_to_rubric_fitness  # legacy private alias

logger = structlog.get_logger(__name__)

_AGENT_NAME = "classification-agent"


class ClassificationService:
    """Phase 4 classification orchestrator.

    Stateless — all collaborators are injected and all state lives in ports.
    """

    def __init__(
        self,
        *,
        knowledge_service: KnowledgeServicePort,
        assessment_config: AssessmentConfigPort,
        topic_extractor: TopicExtractor,
        decision_audit: DecisionAuditPort,
        event_publisher: EventPublisherPort,
        react_prober_factory: ReactProberFactory | None = None,
        tracing: TracingPort | None = None,
    ) -> None:
        self._knowledge_service = knowledge_service
        self._assessment_config = assessment_config
        self._tracing = tracing

        self._last_sufficiency_confidence = 0.90

        self._sufficiency_runner = SufficiencyProbeRunner(
            react_prober_factory=react_prober_factory,
            knowledge_service=knowledge_service,
            decision_audit=decision_audit,
            event_publisher=event_publisher,
            tracing=tracing,
        )
        self._topic_runner = TopicExtractionRunner(
            topic_extractor=topic_extractor,
            knowledge_service=knowledge_service,
            decision_audit=decision_audit,
            event_publisher=event_publisher,
            tracing=tracing,
            sufficiency_confidence_provider=lambda: self._last_sufficiency_confidence,
        )
        self._decision_recorder = DecisionRecorder(
            topic_extractor=topic_extractor,
            decision_audit=decision_audit,
            event_publisher=event_publisher,
            tracing=tracing,
        )
        self._progress_emitter = ProgressEmitter()

    async def classify(self, request: ClassificationRequest) -> ClassificationResponse:
        logger.info(
            "classification_start",
            workflow_id=request.workflow_id,
            assessment_id=request.assessment_id,
            classification_type=request.classification_type.value,
        )

        config = await self._assessment_config.get_assessment_config(request.assessment_id)

        chunks, ks_latency = await self._fetch_chunks(request)

        react_probe_result = None
        rubric_fitness_result: RubricFitnessResult | None = None

        if request.classification_type != ClassificationType.TOPICS_ONLY:
            outcome = await self._sufficiency_runner.run(chunks, config, request)
            self._last_sufficiency_confidence = outcome.sufficiency_confidence
            if outcome.insufficient_response is not None:
                return outcome.insufficient_response
            react_probe_result = outcome.react_result

            await self._write_progress_event(
                request.workflow_id,
                "assessorflow.classification.sufficiency-complete",
                f"Material sufficiency: PASS ({len(chunks)} chunks available)",
            )
            if self._tracing:
                try:
                    web_mode = getattr(config, "web_research_mode", WebResearchMode.MANUAL)  # noqa: F841
                    await self._tracing.trace_tool_call(
                        workflow_id=request.workflow_id,
                        agent_name=_AGENT_NAME,
                        tool_name="sufficiency-check",
                        input_params={"chunk_count": len(chunks), "mode": "react"},
                        output_summary={
                            "sufficient": True,
                            "confidence": round(self._last_sufficiency_confidence, 4),
                        },
                        latency_ms=0,
                    )
                except Exception:
                    pass

        topics = None
        if request.classification_type != ClassificationType.SUFFICIENCY_ONLY:
            topic_outcome = await self._topic_runner.run(chunks, config, request)
            if topic_outcome.terminal_response is not None:
                return topic_outcome.terminal_response
            topics = topic_outcome.topics

            if request.classification_type != ClassificationType.TOPICS_ONLY:
                topic_names = [t.name for t in topics.topics]
                await self._write_progress_event(
                    request.workflow_id,
                    "assessorflow.classification.topic-extraction-complete",
                    f"Extracted {len(topics.topics)} subtopics: {', '.join(topic_names[:6])}",
                )

        rubric_source = None
        if react_probe_result is not None:
            rubric_fitness_result = react_result_to_rubric_fitness(react_probe_result)
            rubric_source = react_probe_result.rubric_source

            if request.classification_type != ClassificationType.TOPICS_ONLY:
                await self._write_progress_event(
                    request.workflow_id,
                    "assessorflow.classification.rubric-fitness-complete",
                    f"Rubric fitness: {react_probe_result.rubric_fitness}, source={rubric_source}",
                )

        return await self._decision_recorder.record_success(
            request=request,
            chunks=chunks,
            topics=topics,
            react_probe_result=react_probe_result,
            rubric_fitness_result=rubric_fitness_result,
            rubric_source=rubric_source,
            sufficiency_confidence=self._last_sufficiency_confidence,
        )

    async def _fetch_chunks(self, request: ClassificationRequest) -> tuple[list, float]:
        """Resolve the chunk corpus from the request override or Knowledge Service."""
        if request.chunks is not None:
            from classification_agent.api.schemas import ChunkData
            chunks = [
                ChunkData(
                    chunk_id=c.get("chunk_id") or c.get("id") or f"override-{i}",
                    workflow_id=c.get("workflow_id", request.workflow_id),
                    content=c.get("content", ""),
                    source_type=c.get("source_type", "direct_text"),
                    metadata=c.get("metadata"),
                )
                for i, c in enumerate(request.chunks)
            ]
            logger.info(
                "chunks_override_used",
                workflow_id=request.workflow_id,
                chunk_count=len(chunks),
            )
            return chunks, 0.0

        t0 = time.monotonic()
        chunks = await self._knowledge_service.get_chunks_by_workflow(request.workflow_id)
        ks_latency = (time.monotonic() - t0) * 1000

        logger.info(
            "chunks_retrieved",
            workflow_id=request.workflow_id,
            chunk_count=len(chunks),
        )

        if self._tracing:
            try:
                await self._tracing.trace_tool_call(
                    workflow_id=request.workflow_id,
                    agent_name=_AGENT_NAME,
                    tool_name="ks-get-chunks-by-workflow",
                    input_params={"workflow_id": request.workflow_id},
                    output_summary={"chunk_count": len(chunks)},
                    latency_ms=ks_latency,
                )
            except Exception:
                logger.warning(
                    "tracing_tool_call_failed",
                    tool_name="ks-get-chunks-by-workflow",
                )

        return chunks, ks_latency

    async def _write_progress_event(
        self,
        workflow_id: str,
        event_type: str,
        summary: str,
    ) -> None:
        """Delegate to ProgressEmitter — preserved as a method for legacy patch hooks."""
        await self._progress_emitter.emit(workflow_id, event_type, summary)
