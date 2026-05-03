from __future__ import annotations

import uuid
from pathlib import Path

import structlog

from af_shared.utils.prompt_loader import get_prompt_version, load_prompt

from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    SubTopic,
    Topic,
    TopicHierarchy,
)
from classification_agent.domain.response_models import TopicExtractionResponse
from classification_agent.ports.model_broker_port import ModelBrokerPort

logger = structlog.get_logger(__name__)


class GuardrailBlockedError(RuntimeError):
    """Raised when Model Broker's output guardrail blocks topic extraction.

    The model_broker_http adapter retries once with a PII-avoidance hint;
    if that also fails it returns a sentinel dict with
    ``guardrail_blocked=True``. Treating that as "0 topics extracted"
    silently masks a real failure and lets downstream agents proceed
    with empty input — see WF-05CF28 (2026-04-24) where qna-generation
    crashed with INVALID_ARGUMENT after receiving 0 topics. Raise
    instead so services.py can publish a terminal classification.complete
    and the orchestrator can route to TERMINATE.
    """


_TASK_KEY = "classification.topic_extraction"
_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "topic_extraction.yaml"
_, _TOPIC_EXTRACTION_TEMPLATE = load_prompt(_PROMPT_PATH)


def _generate_topic_id(prefix: str = "t") -> str:
    """Generate a short unique topic ID."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _parse_topics_from_llm(raw: dict, workflow_id: str) -> TopicHierarchy:
    """Convert the raw LLM JSON output into a validated TopicHierarchy.

    Validates the raw dict via Pydantic TopicExtractionResponse model,
    then converts to the domain TopicHierarchy with generated IDs for
    any missing topic_id fields.
    """
    parsed = TopicExtractionResponse.model_validate(raw)

    topics: list[Topic] = []
    for idx, parsed_topic in enumerate(parsed.topics, start=1):
        subtopics: list[SubTopic] = []
        for sub_idx, parsed_sub in enumerate(parsed_topic.subtopics, start=1):
            subtopics.append(
                SubTopic(
                    topic_id=parsed_sub.topic_id or _generate_topic_id(f"t-{idx:03d}-{sub_idx}"),
                    name=parsed_sub.name,
                    subtopics=[],
                )
            )
        topics.append(
            Topic(
                topic_id=parsed_topic.topic_id or _generate_topic_id(f"t-{idx:03d}"),
                name=parsed_topic.name,
                subtopics=subtopics,
            )
        )
    return TopicHierarchy(workflow_id=workflow_id, topics=topics, confidence=float(getattr(parsed, "confidence", 0.0)))


class TopicExtractor:
    """Extracts hierarchical topics from document chunks.

    Uses Model Broker (CHEAP tier) for LLM extraction in real mode.
    In stub mode the Model Broker returns a canned topic hierarchy.
    """

    def __init__(self, model_broker: ModelBrokerPort) -> None:
        self._model_broker = model_broker
        self._prompt_version = get_prompt_version(_PROMPT_PATH)
        self.last_model_used: str = "unknown"

    async def extract(
        self,
        chunks: list[ChunkData],
        config: AssessmentConfig,
        *,
        workflow_id: str | None = None,
    ) -> TopicHierarchy:
        if not chunks:
            logger.info("topic_extraction_skip_empty", workflow_id=workflow_id)
            return TopicHierarchy(workflow_id=workflow_id or "", topics=[])

        chunk_text = "\n---\n".join(
            f"[{c.chunk_id}] {c.content}" for c in chunks
        )

        prompt = _TOPIC_EXTRACTION_TEMPLATE.format(
            chunks=chunk_text,
        )

        from classification_agent.domain.schemas_json import TOPIC_EXTRACTION_SCHEMA

        llm_result = await self._model_broker.invoke(
            _TASK_KEY,
            prompt,
            workflow_id=workflow_id,
            response_format="json",
            response_schema=TOPIC_EXTRACTION_SCHEMA,
        )
        self.last_model_used = llm_result.get("model_used", "unknown")

        if llm_result.get("guardrail_blocked"):
            logger.warning(
                "topic_extraction_guardrail_blocked",
                workflow_id=workflow_id,
                detail=str(llm_result.get("content", ""))[:200],
            )
            raise GuardrailBlockedError(
                "topic_extraction LLM output blocked by model-broker output guardrail "
                "after retry; cannot return a valid topic hierarchy"
            )

        wf = workflow_id or ""
        hierarchy = _parse_topics_from_llm(llm_result, wf)

        logger.info(
            "topic_extraction_complete",
            workflow_id=workflow_id,
            topic_count=len(hierarchy.topics),
        )

        return hierarchy

    @property
    def prompt_version(self) -> str:
        return self._prompt_version
