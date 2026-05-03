from __future__ import annotations

from af_shared.models.domain import DecisionLogEntry
from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    DifficultyLevel,
    PolicyChunk,
    SimilarityResult,
    SourceType,
    Topic,
    TopicHierarchy,
)


class TestStubKnowledgeServiceAdapter:
    """Tests for the Knowledge Service stub."""

    async def test_returns_chunks_for_workflow(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        chunks = [
            ChunkData(
                chunk_id="c1",
                workflow_id="wf-1",
                content="Content 1",
                source_type=SourceType.DIRECT_TEXT,
            ),
            ChunkData(
                chunk_id="c2",
                workflow_id="wf-1",
                content="Content 2",
                source_type=SourceType.DIRECT_TEXT,
            ),
        ]
        knowledge_service_stub.add_chunks("wf-1", chunks)

        result = await knowledge_service_stub.get_chunks_by_workflow("wf-1")
        assert len(result) == 2
        assert result[0].chunk_id == "c1"

    async def test_returns_default_chunks_for_unknown_workflow(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """Unknown workflow IDs return 22 default chunks for E2E testing."""
        result = await knowledge_service_stub.get_chunks_by_workflow("wf-unknown")
        assert len(result) == 22
        assert all(c.source_type.value == "direct_text" for c in result)
        assert result[0].chunk_id == "default-chunk-001"

    async def test_explicit_empty_overrides_defaults(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """Explicitly adding empty list overrides default chunks."""
        knowledge_service_stub.add_chunks("wf-empty", [])
        result = await knowledge_service_stub.get_chunks_by_workflow("wf-empty")
        assert result == []

    async def test_stores_topics(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        topics = TopicHierarchy(
            workflow_id="wf-1",
            topics=[
                Topic(topic_id="t-1", name="Topic A", subtopics=[]),
            ],
        )
        await knowledge_service_stub.store_topics("wf-1", topics)

        stored = knowledge_service_stub.get_stored_topics("wf-1")
        assert stored is not None
        assert stored.topics[0].name == "Topic A"

    async def test_no_stored_topics_initially(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        assert knowledge_service_stub.get_stored_topics("wf-1") is None

    async def test_separate_workflows_isolated(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        chunks_a = [
            ChunkData(chunk_id="a1", workflow_id="wf-a", content="A", source_type=SourceType.DIRECT_TEXT)
        ]
        chunks_b = [
            ChunkData(chunk_id="b1", workflow_id="wf-b", content="B", source_type=SourceType.DIRECT_TEXT),
            ChunkData(chunk_id="b2", workflow_id="wf-b", content="B2", source_type=SourceType.DIRECT_TEXT),
        ]
        knowledge_service_stub.add_chunks("wf-a", chunks_a)
        knowledge_service_stub.add_chunks("wf-b", chunks_b)

        assert len(await knowledge_service_stub.get_chunks_by_workflow("wf-a")) == 1
        assert len(await knowledge_service_stub.get_chunks_by_workflow("wf-b")) == 2


class TestStubKnowledgeServiceSimilaritySearch:
    """Tests for new stub similarity_search method (AF-136)."""

    async def test_similarity_search_returns_default_results(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """AF-136 AC-1: Stub returns default similarity results for unknown KB."""
        results = await knowledge_service_stub.similarity_search(
            query="test query",
            knowledge_base_target="document",
            workflow_id="wf-test",
        )
        assert len(results) > 0
        assert all(isinstance(r, SimilarityResult) for r in results)
        # Default results should have decreasing scores
        assert results[0].similarity_score >= results[-1].similarity_score

    async def test_similarity_search_respects_top_k(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """AF-136 AC-1: Stub respects top_k parameter."""
        results = await knowledge_service_stub.similarity_search(
            query="test",
            knowledge_base_target="document",
            workflow_id="wf-test",
            top_k=1,
        )
        assert len(results) <= 1

    async def test_similarity_search_configurable_fixtures(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """AF-136 AC-2: Stub accepts pre-configured fixture responses."""
        custom_results = [
            SimilarityResult(
                chunk_id="custom-001",
                content="Custom result",
                similarity_score=0.99,
                source_document="custom.pdf",
            ),
        ]
        knowledge_service_stub.set_similarity_results("document", custom_results)

        results = await knowledge_service_stub.similarity_search(
            query="any",
            knowledge_base_target="document",
            workflow_id="wf-test",
        )
        assert len(results) == 1
        assert results[0].chunk_id == "custom-001"
        assert results[0].similarity_score == 0.99


class TestStubKnowledgeServiceSearchPolicies:
    """Tests for new stub search_policies method (AF-136)."""

    async def test_search_policies_returns_default_policy(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """AF-136 AC-3: Stub returns default policy chunks."""
        results = await knowledge_service_stub.search_policies(
            query="grading rubric",
        )
        assert len(results) > 0
        assert all(isinstance(r, PolicyChunk) for r in results)

    async def test_search_policies_filters_by_type(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """AF-136 AC-3: Stub filters by policy_type."""
        knowledge_service_stub.set_policy_chunks([
            PolicyChunk(
                chunk_id="sys-001",
                content="System default",
                policy_type="system_default",
                source="admin_seeded",
            ),
            PolicyChunk(
                chunk_id="asr-001",
                content="Assessor rubric",
                policy_type="assessor_rubric",
                source="assessor_upload",
                assessment_id="assess-001",
            ),
        ])

        system_only = await knowledge_service_stub.search_policies(
            query="rubric",
            policy_type="system_default",
        )
        assert all(r.policy_type == "system_default" for r in system_only)

        assessor_only = await knowledge_service_stub.search_policies(
            query="rubric",
            policy_type="assessor_rubric",
        )
        assert all(r.policy_type == "assessor_rubric" for r in assessor_only)

    async def test_search_policies_configurable_fixtures(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
    ) -> None:
        """AF-136 AC-4: Stub accepts pre-configured policy chunks."""
        custom_policies = [
            PolicyChunk(
                chunk_id="custom-pol",
                content="Custom policy",
                policy_type="assessor_rubric",
                source="assessor_upload",
                assessment_id="assess-custom",
                similarity_score=0.95,
            ),
        ]
        knowledge_service_stub.set_policy_chunks(custom_policies)

        results = await knowledge_service_stub.search_policies(query="custom")
        assert len(results) == 1
        assert results[0].chunk_id == "custom-pol"


class TestStubModelBrokerToolCallSupport:
    """Tests for Model Broker stub tool-calling support (AF-136)."""

    async def test_set_tool_call_responses(
        self,
        model_broker_stub: StubModelBrokerAdapter,
    ) -> None:
        """AF-136 AC-6: Stub supports tool_call response sequences."""
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            {
                "tool_calls": [
                    {
                        "id": "call_001",
                        "function": {
                            "name": "similarity_search",
                            "arguments": '{"query": "test", "knowledge_base_target": "document"}',
                        },
                    },
                ],
                "content": None,
            },
            {
                "tool_calls": [],
                "content": {
                    "sufficient": True,
                    "reason": "Sufficient",
                    "gap_analysis": [],
                },
            },
        ])

        # First call returns tool_calls
        result1 = await model_broker_stub.invoke_with_tools(
            "classification.react_sufficiency",
            messages=[],
            tools=[],
            workflow_id="wf-test",
        )
        assert len(result1["tool_calls"]) == 1

        # Second call returns final content
        result2 = await model_broker_stub.invoke_with_tools(
            "classification.react_sufficiency",
            messages=[],
            tools=[],
            workflow_id="wf-test",
        )
        assert result2["tool_calls"] == []
        assert result2["content"]["sufficient"] is True


class TestStubAssessmentConfigAdapter:
    """Tests for the Assessment Config stub."""

    async def test_returns_default_config(
        self,
        assessment_config_stub: StubAssessmentConfigAdapter,
    ) -> None:
        config = await assessment_config_stub.get_assessment_config("any-id")
        assert config.assessment_id == "any-id"
        assert config.structured_question_count == 10
        assert config.non_structured_question_count == 5

    async def test_returns_custom_config(
        self,
        assessment_config_stub: StubAssessmentConfigAdapter,
    ) -> None:
        custom = AssessmentConfig(
            assessment_id="custom",
            assessment_title="Custom Assessment",
            structured_question_count=20,
            non_structured_question_count=0,
            difficulty_level=DifficultyLevel.HARD,
        )
        assessment_config_stub.set_config("custom", custom)

        config = await assessment_config_stub.get_assessment_config("custom")
        assert config.structured_question_count == 20
        assert config.non_structured_question_count == 0
        assert config.difficulty_level == DifficultyLevel.HARD


class TestStubModelBrokerAdapter:
    """Tests for the Model Broker stub."""

    async def test_returns_default_sufficiency_response(
        self,
        model_broker_stub: StubModelBrokerAdapter,
    ) -> None:
        result = await model_broker_stub.invoke(
            "classification.sufficiency_check",
            "test prompt",
        )
        assert "sufficient" in result
        assert "gap_analysis" in result

    async def test_returns_default_topic_response(
        self,
        model_broker_stub: StubModelBrokerAdapter,
    ) -> None:
        result = await model_broker_stub.invoke(
            "classification.topic_extraction",
            "test prompt",
        )
        assert "topics" in result
        assert len(result["topics"]) > 0

    async def test_custom_response(
        self,
        model_broker_stub: StubModelBrokerAdapter,
    ) -> None:
        model_broker_stub.set_response(
            "classification.sufficiency_check",
            {"sufficient": True, "reason": "Custom reason", "gap_analysis": []},
        )
        result = await model_broker_stub.invoke(
            "classification.sufficiency_check",
            "test prompt",
        )
        assert result["reason"] == "Custom reason"

    async def test_invocations_tracked(
        self,
        model_broker_stub: StubModelBrokerAdapter,
    ) -> None:
        await model_broker_stub.invoke(
            "classification.topic_extraction",
            "prompt text",
            workflow_id="wf-track",
        )
        assert len(model_broker_stub.invocations) == 1
        assert model_broker_stub.invocations[0]["task_key"] == "classification.topic_extraction"
        assert model_broker_stub.invocations[0]["workflow_id"] == "wf-track"


class TestStubDecisionAuditAdapter:
    """Tests for the Decision Audit stub."""

    async def test_logs_decision(
        self,
        decision_audit_stub: StubDecisionAuditAdapter,
    ) -> None:
        entry = DecisionLogEntry(
            workflow_id="wf-1",
            agent_name="classification-agent",
            decision_type="test",
            input={"phase": "Phase 4"},
            output={"test": True},
            model_id="cheap-tier",
            confidence_score=0.9,
            prompt_version="classification/test@v1",
            reasoning_steps=[{"step": 1, "action": "test action"}],
            grounding_sources=["chunk-1"],
        )
        await decision_audit_stub.log_decision(entry)
        assert len(decision_audit_stub.decisions) == 1
        assert decision_audit_stub.decisions[0].workflow_id == "wf-1"

    async def test_filter_by_workflow(
        self,
        decision_audit_stub: StubDecisionAuditAdapter,
    ) -> None:
        for wf_id in ["wf-1", "wf-2", "wf-1"]:
            entry = DecisionLogEntry(
                workflow_id=wf_id,
                agent_name="classification-agent",
                decision_type="test",
                input={"phase": "Phase 4"},
                output={},
                model_id="cheap-tier",
                confidence_score=None,
                prompt_version="classification/test@v1",
                reasoning_steps=[],
                grounding_sources=[],
            )
            await decision_audit_stub.log_decision(entry)
        assert len(decision_audit_stub.get_decisions_for_workflow("wf-1")) == 2
        assert len(decision_audit_stub.get_decisions_for_workflow("wf-2")) == 1


class TestStubEventPublisherAdapter:
    """Tests for the Event Publisher stub."""

    async def test_publishes_event(
        self,
        event_publisher_stub: StubEventPublisherAdapter,
    ) -> None:
        await event_publisher_stub.publish(
            "assessorflow.classification.complete",
            {"workflow_id": "wf-1"},
        )
        assert len(event_publisher_stub.events) == 1
        assert event_publisher_stub.events[0]["topic"] == "assessorflow.classification.complete"

    async def test_filter_by_topic(
        self,
        event_publisher_stub: StubEventPublisherAdapter,
    ) -> None:
        await event_publisher_stub.publish("topic.a", {"id": 1})
        await event_publisher_stub.publish("topic.b", {"id": 2})
        await event_publisher_stub.publish("topic.a", {"id": 3})

        a_events = event_publisher_stub.get_events_for_topic("topic.a")
        assert len(a_events) == 2
        assert a_events[0]["payload"]["id"] == 1
        assert a_events[1]["payload"]["id"] == 3
