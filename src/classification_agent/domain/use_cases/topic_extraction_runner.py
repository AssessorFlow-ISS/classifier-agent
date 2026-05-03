from __future__ import annotations

import time
from dataclasses import dataclass

import structlog

from af_shared.models.domain import DecisionLogEntry, ModelResponse
from af_shared.ports.tracing import TracingPort
from classification_agent.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
)
from classification_agent.domain.topic_extractor import (
    GuardrailBlockedError,
    TopicExtractor,
)
from classification_agent.ports.decision_audit_port import DecisionAuditPort
from classification_agent.ports.event_publisher_port import EventPublisherPort
from classification_agent.ports.knowledge_service_port import KnowledgeServicePort

logger = structlog.get_logger(__name__)

_AGENT_NAME = "classification-agent"
_PHASE_TOPICS = "Phase 4: Topic Extraction"
_TOPIC_COMPLETE = "assessorflow.classification.complete"


@dataclass
class TopicExtractionOutcome:
    """Either the extracted topics + storage latency, or a terminal response.

    When ``terminal_response`` is non-None the topic extraction was guardrail-blocked
    and the caller should return it immediately.
    """

    topics: object | None
    extract_latency_ms: float
    store_latency_ms: float
    terminal_response: ClassificationResponse | None


class TopicExtractionRunner:
    """Wraps the topic extraction LLM call, KS storage, and guardrail-terminal handling."""

    def __init__(
        self,
        *,
        topic_extractor: TopicExtractor,
        knowledge_service: KnowledgeServicePort,
        decision_audit: DecisionAuditPort,
        event_publisher: EventPublisherPort,
        tracing: TracingPort | None,
        sufficiency_confidence_provider,
    ) -> None:
        self._topic_extractor = topic_extractor
        self._knowledge_service = knowledge_service
        self._decision_audit = decision_audit
        self._event_publisher = event_publisher
        self._tracing = tracing
        self._sufficiency_confidence_provider = sufficiency_confidence_provider

    async def run(
        self,
        chunks: list,
        config,
        request: ClassificationRequest,
    ) -> TopicExtractionOutcome:
        t0 = time.monotonic()
        try:
            topics = await self._topic_extractor.extract(
                chunks,
                config,
                workflow_id=request.workflow_id,
            )
        except GuardrailBlockedError as exc:
            terminal = await self._publish_terminal(
                request=request,
                chunks=chunks,
                reason_msg=str(exc),
            )
            return TopicExtractionOutcome(
                topics=None,
                extract_latency_ms=(time.monotonic() - t0) * 1000,
                store_latency_ms=0.0,
                terminal_response=terminal,
            )
        extract_latency = (time.monotonic() - t0) * 1000

        if self._tracing:
            try:
                await self._tracing.trace_llm_call(
                    workflow_id=request.workflow_id,
                    agent_name=_AGENT_NAME,
                    task_key="classification.topic_extraction",
                    prompt_version=self._topic_extractor.prompt_version,
                    model_response=ModelResponse(
                        content="",
                        model_used="cheap-tier",
                        model_tier="CHEAP",
                        tokens_input=0,
                        tokens_output=0,
                        cost_usd=0.0,
                        latency_ms=extract_latency,
                    ),
                )
            except Exception:
                logger.warning(
                    "tracing_llm_call_failed",
                    task_key="classification.topic_extraction",
                )

        t0 = time.monotonic()
        await self._knowledge_service.store_topics(request.workflow_id, topics)
        store_latency = (time.monotonic() - t0) * 1000

        if self._tracing:
            try:
                await self._tracing.trace_tool_call(
                    workflow_id=request.workflow_id,
                    agent_name=_AGENT_NAME,
                    tool_name="ks-store-topics",
                    input_params={
                        "workflow_id": request.workflow_id,
                        "topic_count": len(topics.topics),
                    },
                    output_summary={"stored": True},
                    latency_ms=store_latency,
                )
            except Exception:
                logger.warning("tracing_tool_call_failed", tool_name="ks-store-topics")

        return TopicExtractionOutcome(
            topics=topics,
            extract_latency_ms=extract_latency,
            store_latency_ms=store_latency,
            terminal_response=None,
        )

    async def _publish_terminal(
        self,
        request: ClassificationRequest,
        chunks: list,
        reason_msg: str,
    ) -> ClassificationResponse:
        reason_code = "GUARDRAIL_BLOCKED_TOPIC_EXTRACTION"
        terminal_signal = {
            "status": "TERMINATE",
            "reason_code": reason_code,
            "message": reason_msg,
        }

        entry = DecisionLogEntry(
            workflow_id=request.workflow_id,
            agent_name=_AGENT_NAME,
            decision_type="classification_governance",
            assessor_id=request.assessor_id,
            input_summary={
                "phase": _PHASE_TOPICS,
                "chunk_count": len(chunks),
                "assessment_id": request.assessment_id,
                "tools_used": ["ks-get-chunks-by-workflow", "topic-extraction-llm"],
            },
            output_summary={
                "sufficient": False,
                "topic_count": 0,
                "terminal_signal": terminal_signal,
            },
            reasoning_steps=[
                {
                    "step": 1,
                    "action": (
                        "Topic extraction LLM output blocked by Model Broker output "
                        "guardrail (PII pattern matched in extracted-topic JSON, retry "
                        "with PII-avoidance hint also blocked). Cannot return a valid "
                        "topic hierarchy; terminating workflow at classification phase."
                    ),
                    "component": "topic_extraction",
                },
            ],
            confidence_score=self._sufficiency_confidence_provider(),
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
            "reason_code": reason_code,
            "message": reason_msg,
            "topic_count": 0,
            "terminal_signal": terminal_signal,
        }
        await self._event_publisher.publish(_TOPIC_COMPLETE, completion_payload)

        logger.warning(
            "classification_terminated_guardrail_blocked",
            workflow_id=request.workflow_id,
            reason_code=reason_code,
        )

        return ClassificationResponse(
            workflow_id=request.workflow_id,
            sufficient=False,
            reason=reason_msg,
            gap_analysis=[],
            topics=None,
            rubric_fitness=None,
            rubric_source=None,
        )
