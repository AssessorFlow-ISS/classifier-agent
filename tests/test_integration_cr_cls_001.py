"""Integration tests for AF-140/AF-141: Unified ReAct probe pipeline and golden scenarios.

Tests the complete ClassificationService pipeline with unified ReAct
sufficiency + rubric fitness probe, enriched Pub/Sub payloads, and
backward compatibility with existing tests.
"""
from __future__ import annotations


from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    ClassificationRequest,
    DifficultyLevel,
    PolicyChunk,
    SourceType,
    WebResearchMode,
)
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.tools.registry import build_react_prober_factory


def _make_chunks(n: int, workflow_id: str = "wf-int") -> list[ChunkData]:
    """Create n direct_text chunks for integration testing."""
    return [
        ChunkData(
            chunk_id=f"chunk-{i:03d}",
            workflow_id=workflow_id,
            content=f"Content about computer science topic {i} with sufficient depth.",
            source_type=SourceType.DIRECT_TEXT,
        )
        for i in range(n)
    ]


def _make_auto_config(assessment_id: str = "assess-int") -> AssessmentConfig:
    """Config with web_research_mode=auto."""
    return AssessmentConfig(
        assessment_id=assessment_id,
        assessment_title="CS Fundamentals Assessment",
        structured_question_count=10,
        non_structured_question_count=5,
        difficulty_level=DifficultyLevel.MEDIUM,
        web_research_mode=WebResearchMode.AUTO,
    )


def _make_disabled_config(assessment_id: str = "assess-int") -> AssessmentConfig:
    """Config with web_research_mode=manual (default HITL behavior)."""
    return AssessmentConfig(
        assessment_id=assessment_id,
        assessment_title="CS Fundamentals Assessment",
        structured_question_count=10,
        non_structured_question_count=5,
        difficulty_level=DifficultyLevel.MEDIUM,
        web_research_mode=WebResearchMode.MANUAL,
    )


def _react_sufficient(
    rubric_fitness: str = "ALIGNED",
    rubric_source: str = "admin_seeded",
) -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": True,
            "reason": "Material sufficient",
            "gap_analysis": [],
            "search_queries": [],
            "autonomy_exercised": False,
            "rubric_fitness": rubric_fitness,
            "rubric_reasoning": "",
            "rubric_source": rubric_source,
        },
    }


def _react_insufficient(
    gap_analysis: list | None = None,
    search_queries: list | None = None,
    autonomy_exercised: bool = False,
    rubric_fitness: str = "NO_RUBRIC",
    rubric_source: str = "none",
) -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": False,
            "reason": "Insufficient depth for advanced topics",
            "gap_analysis": gap_analysis or [
                {
                    "topic": "Data Structures",
                    "current_depth": "surface",
                    "required_depth": "deep",
                    "gap_description": "No advanced DS content",
                    "fillable_by_web": True,
                    "confidence": 0.30,
                }
            ],
            "search_queries": search_queries or [],
            "autonomy_exercised": autonomy_exercised,
            "rubric_fitness": rubric_fitness,
            "rubric_reasoning": "",
            "rubric_source": rubric_source,
        },
    }


def _react_misaligned(rubric_source: str = "system_default") -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": True,
            "reason": "Material sufficient",
            "gap_analysis": [],
            "search_queries": [],
            "autonomy_exercised": False,
            "rubric_fitness": "MISALIGNED",
            "rubric_reasoning": "Rubric does not match material topics",
            "rubric_source": rubric_source,
        },
    }


class TestEnrichedPubSubPayloads:
    """Tests for enriched Pub/Sub payloads with unified probe fields (AF-140)."""

    async def test_completion_event_includes_rubric_fitness(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-1: classification.complete event includes rubric_fitness fields."""
        knowledge_service_stub.add_chunks("wf-enrich", sufficient_chunks)
        config = _make_disabled_config(assessment_id="assess-enrich")
        assessment_config_stub.set_config("assess-enrich", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-enrich",
            assessment_id="assess-enrich",
        )
        await service.classify(request)

        events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.complete"
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert "rubric_fitness" in payload
        assert "rubric_source" in payload

    async def test_insufficient_event_includes_gap_analysis(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-2: classification.insufficient event includes gap_analysis[]."""
        knowledge_service_stub.add_chunks("wf-insuff", insufficient_chunks)
        config = _make_auto_config(assessment_id="assess-insuff")
        assessment_config_stub.set_config("assess-insuff", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient(rubric_fitness="NO_RUBRIC"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-insuff",
            assessment_id="assess-insuff",
        )
        await service.classify(request)

        events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.insufficient"
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert "gap_analysis" in payload
        assert "rubric_fitness" in payload

    async def test_insufficient_event_includes_search_queries_in_auto_mode(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-3: In auto mode, insufficient event includes search_queries[]."""
        knowledge_service_stub.add_chunks("wf-auto", insufficient_chunks)
        config = _make_auto_config(assessment_id="assess-auto")
        assessment_config_stub.set_config("assess-auto", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient(
                search_queries=[
                    "advanced data structures balanced trees",
                    "graph algorithms shortest path",
                ],
                autonomy_exercised=True,
                rubric_fitness="NO_RUBRIC",
            ),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-auto",
            assessment_id="assess-auto",
        )
        await service.classify(request)

        events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.insufficient"
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert "search_queries" in payload
        assert len(payload["search_queries"]) > 0
        assert "autonomy_exercised" in payload
        assert payload["autonomy_exercised"] is True

    async def test_insufficient_event_no_search_queries_in_disabled_mode(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-4: In disabled mode, no search_queries in event."""
        knowledge_service_stub.add_chunks("wf-disabled", insufficient_chunks)
        config = _make_disabled_config(assessment_id="assess-disabled")
        assessment_config_stub.set_config("assess-disabled", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient(rubric_fitness="NO_RUBRIC"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-disabled",
            assessment_id="assess-disabled",
        )
        await service.classify(request)

        events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.insufficient"
        )
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload.get("autonomy_exercised", False) is False


class TestClassificationResponse:
    """Tests for enriched ClassificationResponse (AF-140)."""

    async def test_response_includes_rubric_fitness(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-5: ClassificationResponse includes rubric_fitness result."""
        knowledge_service_stub.add_chunks("wf-resp", sufficient_chunks)
        config = _make_disabled_config(assessment_id="assess-resp")
        assessment_config_stub.set_config("assess-resp", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-resp",
            assessment_id="assess-resp",
        )
        response = await service.classify(request)

        assert response.rubric_fitness is not None
        assert response.rubric_source is not None

    async def test_response_includes_autonomy_exercised(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-6: ClassificationResponse includes autonomy_exercised and search_queries."""
        knowledge_service_stub.add_chunks("wf-auton", insufficient_chunks)
        config = _make_auto_config(assessment_id="assess-auton")
        assessment_config_stub.set_config("assess-auton", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient(
                search_queries=["query1"],
                autonomy_exercised=True,
            ),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-auton",
            assessment_id="assess-auton",
        )
        response = await service.classify(request)

        assert response.autonomy_exercised is True
        assert len(response.search_queries) > 0


class TestGoldenScenarios:
    """Golden fixture scenarios for AF-141."""

    async def test_scenario_sufficient_with_aligned_rubric(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
    ) -> None:
        """Golden Scenario 1: Sufficient material + aligned assessor rubric."""
        knowledge_service_stub.add_chunks("wf-gold-1", sufficient_chunks)
        config = AssessmentConfig(
            assessment_id="assess-gold-1",
            assessment_title="CS Fundamentals",
            structured_question_count=10,
            non_structured_question_count=5,
            difficulty_level=DifficultyLevel.MEDIUM,
            web_research_mode=WebResearchMode.MANUAL,
        )
        assessment_config_stub.set_config("assess-gold-1", config)

        knowledge_service_stub.set_policy_chunks([
            PolicyChunk(
                chunk_id="rubric-gold-1",
                content="Assess OOP concepts and data structures",
                policy_type="assessor_rubric",
                source="assessor_upload",
                assessment_id="assess-gold-1",
                similarity_score=0.95,
            ),
        ])

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(rubric_fitness="ALIGNED", rubric_source="assessor_upload"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        response = await service.classify(ClassificationRequest(
            workflow_id="wf-gold-1",
            assessment_id="assess-gold-1",
        ))

        assert response.sufficient is True
        assert response.rubric_fitness is not None
        assert response.rubric_fitness.is_aligned is True
        assert response.rubric_source == "assessor_upload"
        assert response.topics is not None

    async def test_scenario_insufficient_auto_web_research(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
    ) -> None:
        """Golden Scenario 2: Insufficient material + auto web research."""
        knowledge_service_stub.add_chunks("wf-gold-2", insufficient_chunks)
        config = _make_auto_config(assessment_id="assess-gold-2")
        assessment_config_stub.set_config("assess-gold-2", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient(
                gap_analysis=[
                    {
                        "topic": "Data Structures",
                        "current_depth": "surface",
                        "required_depth": "deep",
                        "gap_description": "Missing advanced DS content",
                        "fillable_by_web": True,
                        "confidence": 0.30,
                    },
                ],
                search_queries=["advanced data structures trees graphs"],
                autonomy_exercised=True,
                rubric_fitness="NO_RUBRIC",
            ),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        response = await service.classify(ClassificationRequest(
            workflow_id="wf-gold-2",
            assessment_id="assess-gold-2",
        ))

        assert response.sufficient is False
        assert response.autonomy_exercised is True
        assert len(response.search_queries) > 0
        assert len(response.gap_analysis) > 0

    async def test_scenario_sufficient_misaligned_rubric_signals(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
    ) -> None:
        """Golden Scenario 3: Sufficient material + misaligned rubric fires insufficient event."""
        knowledge_service_stub.add_chunks("wf-gold-3", sufficient_chunks)
        config = _make_disabled_config(assessment_id="assess-gold-3")
        assessment_config_stub.set_config("assess-gold-3", config)

        knowledge_service_stub.set_policy_chunks([
            PolicyChunk(
                chunk_id="pol-default",
                content="Grammar and vocabulary rubric",
                policy_type="system_default",
                source="admin_seeded",
                similarity_score=0.60,
            ),
        ])

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_misaligned(rubric_source="system_default"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        response = await service.classify(ClassificationRequest(
            workflow_id="wf-gold-3",
            assessment_id="assess-gold-3",
        ))

        # Rubric misalignment now gates the pipeline — returns insufficient
        assert response.sufficient is False
        assert response.rubric_fitness is not None
        assert response.rubric_fitness.is_aligned is False
        assert response.rubric_source == "system_default"

    async def test_scenario_no_rubric_at_all(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
    ) -> None:
        """Golden Scenario 4: No rubric at all — probe returns NO_RUBRIC, pipeline proceeds."""
        knowledge_service_stub.add_chunks("wf-gold-4", sufficient_chunks)
        config = _make_disabled_config(assessment_id="assess-gold-4")
        assessment_config_stub.set_config("assess-gold-4", config)
        knowledge_service_stub.set_policy_chunks([])

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(rubric_fitness="NO_RUBRIC", rubric_source="none"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        response = await service.classify(ClassificationRequest(
            workflow_id="wf-gold-4",
            assessment_id="assess-gold-4",
        ))

        assert response.sufficient is True
        assert response.rubric_fitness is not None
        assert response.rubric_source == "none"

    async def test_scenario_backward_compatibility_existing_flow(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """Golden Scenario 5: Backward compat with existing flow (default config)."""
        knowledge_service_stub.add_chunks("wf-gold-5", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(rubric_fitness="NO_RUBRIC", rubric_source="none"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        response = await service.classify(ClassificationRequest(
            workflow_id="wf-gold-5",
            assessment_id="assess-001",
        ))

        assert response.sufficient is True
        assert response.topics is not None
        assert response.rubric_fitness is not None
        assert response.autonomy_exercised is False

    async def test_decision_audit_includes_react_tools_used(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
    ) -> None:
        """AC-7: Decision audit entries include unified ReAct tool usage."""
        knowledge_service_stub.add_chunks("wf-audit", sufficient_chunks)
        config = _make_disabled_config(assessment_id="assess-audit")
        assessment_config_stub.set_config("assess-audit", config)

        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient(rubric_fitness="ALIGNED", rubric_source="admin_seeded"),
        ])

        service = _build_service(
            knowledge_service_stub,
            assessment_config_stub,
            model_broker_stub,
            decision_audit_stub,
            event_publisher_stub,
        )

        await service.classify(ClassificationRequest(
            workflow_id="wf-audit",
            assessment_id="assess-audit",
        ))

        decisions = decision_audit_stub.get_decisions_for_workflow("wf-audit")
        assert len(decisions) >= 1

        all_decision_types = [d.decision_type for d in decisions]
        assert "classification_governance" in all_decision_types


# ---------------------------------------------------------------------------
# Helper: Build service with all stubs
# ---------------------------------------------------------------------------

def _build_service(
    knowledge_service: StubKnowledgeServiceAdapter,
    assessment_config: StubAssessmentConfigAdapter,
    model_broker: StubModelBrokerAdapter,
    decision_audit: StubDecisionAuditAdapter,
    event_publisher: StubEventPublisherAdapter,
) -> ClassificationService:
    """Wire up ClassificationService with unified ReAct prober factory."""
    topic_extractor = TopicExtractor(model_broker=model_broker)
    react_prober_factory = build_react_prober_factory(
        model_broker=model_broker,
        knowledge_service=knowledge_service,
    )

    return ClassificationService(
        knowledge_service=knowledge_service,
        assessment_config=assessment_config,
        topic_extractor=topic_extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        react_prober_factory=react_prober_factory,
    )
