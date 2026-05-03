from __future__ import annotations


import pytest

from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
)
from classification_agent.domain.topic_extractor import (
    GuardrailBlockedError,
    TopicExtractor,
)


class TestTopicExtractor:
    """Tests for the TopicExtractor domain component."""

    async def test_extract_topics_from_diverse_chunks(
        self,
        topic_extractor: TopicExtractor,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        result = await topic_extractor.extract(
            sufficient_chunks, default_config, workflow_id="wf-test"
        )
        assert result.workflow_id == "wf-test"
        assert len(result.topics) > 0

    async def test_each_topic_has_name_and_subtopics(
        self,
        topic_extractor: TopicExtractor,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        result = await topic_extractor.extract(
            sufficient_chunks, default_config, workflow_id="wf-test"
        )
        for topic in result.topics:
            assert topic.name
            assert topic.topic_id
            assert isinstance(topic.subtopics, list)
            for sub in topic.subtopics:
                assert sub.name
                assert sub.topic_id

    async def test_topic_ids_are_generated(
        self,
        topic_extractor: TopicExtractor,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        result = await topic_extractor.extract(
            sufficient_chunks, default_config, workflow_id="wf-test"
        )
        topic_ids = [t.topic_id for t in result.topics]
        # All topic IDs should be non-empty and unique
        assert all(tid for tid in topic_ids)
        assert len(topic_ids) == len(set(topic_ids))

    async def test_empty_chunks_returns_empty_topics(
        self,
        topic_extractor: TopicExtractor,
        default_config: AssessmentConfig,
    ) -> None:
        result = await topic_extractor.extract(
            [], default_config, workflow_id="wf-empty"
        )
        assert result.workflow_id == "wf-empty"
        assert result.topics == []

    async def test_model_broker_invoked_with_correct_task_key(
        self,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        extractor = TopicExtractor(model_broker=model_broker_stub)
        await extractor.extract(
            sufficient_chunks, default_config, workflow_id="wf-test"
        )
        assert len(model_broker_stub.invocations) == 1
        assert model_broker_stub.invocations[0]["task_key"] == "classification.topic_extraction"
        assert model_broker_stub.invocations[0]["workflow_id"] == "wf-test"

    async def test_model_broker_not_invoked_for_empty_chunks(
        self,
        model_broker_stub: StubModelBrokerAdapter,
        default_config: AssessmentConfig,
    ) -> None:
        extractor = TopicExtractor(model_broker=model_broker_stub)
        await extractor.extract([], default_config, workflow_id="wf-empty")
        assert len(model_broker_stub.invocations) == 0

    async def test_guardrail_blocked_sentinel_raises(
        self,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """When model_broker_http exhausts its retry and returns the
        BLOCKED_BY_GUARDRAIL sentinel, the extractor must raise rather
        than silently parse it as 0 topics. WF-05CF28 (2026-04-24)
        regression: silent 0-topics caused qna-generation to crash.
        """
        model_broker_stub.set_response(
            "classification.topic_extraction",
            {
                "content": "BLOCKED_BY_GUARDRAIL",
                "guardrail_blocked": True,
                "sufficient": False,
                "model_used": "guardrail",
            },
        )
        extractor = TopicExtractor(model_broker=model_broker_stub)
        with pytest.raises(GuardrailBlockedError):
            await extractor.extract(
                sufficient_chunks, default_config, workflow_id="wf-blocked"
            )

    async def test_prompt_version_format(
        self,
        topic_extractor: TopicExtractor,
    ) -> None:
        """Prompt version follows ADR-39 format."""
        assert topic_extractor.prompt_version == "classification-agent/topic_extraction@v1"

    async def test_custom_model_broker_response(
        self,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """Verify the extractor parses a custom LLM response correctly."""
        model_broker_stub.set_response(
            "classification.topic_extraction",
            {
                "topics": [
                    {
                        "name": "Custom Topic",
                        "subtopics": [{"name": "Custom Sub A"}, {"name": "Custom Sub B"}],
                    }
                ]
            },
        )
        extractor = TopicExtractor(model_broker=model_broker_stub)
        result = await extractor.extract(
            sufficient_chunks, default_config, workflow_id="wf-custom"
        )
        assert len(result.topics) == 1
        assert result.topics[0].name == "Custom Topic"
        assert len(result.topics[0].subtopics) == 2
        assert result.topics[0].subtopics[0].name == "Custom Sub A"
