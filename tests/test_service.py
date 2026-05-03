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
    ClassificationType,
)
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.tools.registry import build_react_prober_factory


def _make_service(
    knowledge_service_stub: StubKnowledgeServiceAdapter,
    assessment_config_stub: StubAssessmentConfigAdapter,
    model_broker_stub: StubModelBrokerAdapter,
    decision_audit_stub: StubDecisionAuditAdapter,
    event_publisher_stub: StubEventPublisherAdapter,
) -> ClassificationService:
    """Build ClassificationService wired with stub adapters."""
    topic_extractor = TopicExtractor(model_broker=model_broker_stub)
    react_prober_factory = build_react_prober_factory(
        model_broker=model_broker_stub,
        knowledge_service=knowledge_service_stub,
    )
    return ClassificationService(
        knowledge_service=knowledge_service_stub,
        assessment_config=assessment_config_stub,
        topic_extractor=topic_extractor,
        decision_audit=decision_audit_stub,
        event_publisher=event_publisher_stub,
        react_prober_factory=react_prober_factory,
    )


def _react_sufficient_response(rubric_fitness: str = "NO_RUBRIC", rubric_source: str = "none") -> dict:
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


def _react_insufficient_response(
    gap_analysis: list | None = None,
    rubric_fitness: str = "NO_RUBRIC",
    rubric_source: str = "none",
) -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": False,
            "reason": "Insufficient material: too few chunks",
            "gap_analysis": gap_analysis or [
                {
                    "topic": "Data Structures",
                    "current_depth": "surface",
                    "required_depth": "moderate",
                    "gap_description": "Need more content on data structures",
                    "fillable_by_web": True,
                    "confidence": 0.8,
                },
            ],
            "search_queries": [],
            "autonomy_exercised": False,
            "rubric_fitness": rubric_fitness,
            "rubric_reasoning": "",
            "rubric_source": rubric_source,
        },
    }


def _react_misaligned_response() -> dict:
    return {
        "tool_calls": [],
        "content": {
            "sufficient": True,
            "reason": "Material sufficient",
            "gap_analysis": [],
            "search_queries": [],
            "autonomy_exercised": False,
            "rubric_fitness": "MISALIGNED",
            "rubric_reasoning": "Rubric focuses on grammar; material is about algorithms",
            "rubric_source": "system_default",
        },
    }


class TestClassificationService:
    """Tests for the full classification pipeline."""

    async def test_sufficient_material_extracts_topics(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-test", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient_response(rubric_fitness="NO_RUBRIC"),
        ])

        request = ClassificationRequest(
            workflow_id="wf-test",
            assessment_id="assess-001",
        )
        response = await classification_service.classify(request)

        assert response.sufficient is True
        assert response.topics is not None
        assert len(response.topics.topics) > 0
        assert response.gap_analysis == []

    async def test_insufficient_material_returns_early(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        insufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-sparse", insufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-sparse",
            assessment_id="assess-001",
        )
        response = await classification_service.classify(request)

        assert response.sufficient is False
        assert response.topics is None
        assert len(response.gap_analysis) > 0

    async def test_topics_stored_in_knowledge_service_on_success(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-test", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-test",
            assessment_id="assess-001",
        )
        await classification_service.classify(request)

        stored = knowledge_service_stub.get_stored_topics("wf-test")
        assert stored is not None
        assert stored.workflow_id == "wf-test"
        assert len(stored.topics) > 0

    async def test_topics_not_stored_when_insufficient(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        insufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-sparse", insufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-sparse",
            assessment_id="assess-001",
        )
        await classification_service.classify(request)

        stored = knowledge_service_stub.get_stored_topics("wf-sparse")
        assert stored is None

    async def test_decision_audit_logged_on_sufficient(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-test", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-test",
            assessment_id="assess-001",
        )
        await classification_service.classify(request)

        decisions = decision_audit_stub.get_decisions_for_workflow("wf-test")
        assert len(decisions) == 1
        assert decisions[0].agent_name == "classification-agent"
        assert decisions[0].decision_type == "classification_governance"
        assert decisions[0].output_summary["sufficient"] is True
        assert decisions[0].prompt_version is not None

    async def test_decision_audit_logged_on_insufficient(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        insufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-sparse", insufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-sparse",
            assessment_id="assess-001",
        )
        await classification_service.classify(request)

        decisions = decision_audit_stub.get_decisions_for_workflow("wf-sparse")
        assert len(decisions) == 1
        assert decisions[0].decision_type == "classification_governance"
        assert decisions[0].output_summary["sufficient"] is False

    async def test_completion_event_published_on_success(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-test", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-test",
            assessment_id="assess-001",
        )
        await classification_service.classify(request)

        complete_events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.complete"
        )
        assert len(complete_events) == 1
        assert complete_events[0]["payload"]["workflow_id"] == "wf-test"
        assert complete_events[0]["payload"]["reason_code"] == "CLASSIFICATION_COMPLETE"

    async def test_insufficient_event_published_on_failure(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        knowledge_service_stub.add_chunks("wf-sparse", insufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-sparse",
            assessment_id="assess-001",
        )
        await classification_service.classify(request)

        insuff_events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.insufficient"
        )
        assert len(insuff_events) == 1
        assert insuff_events[0]["payload"]["reason_code"] == "MATERIAL_INSUFFICIENT"

    async def test_sufficiency_only_mode(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """SUFFICIENCY_ONLY skips topic extraction."""
        knowledge_service_stub.add_chunks("wf-suff", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient_response(),
        ])

        request = ClassificationRequest(
            workflow_id="wf-suff",
            assessment_id="assess-001",
            classification_type=ClassificationType.SUFFICIENCY_ONLY,
        )
        response = await classification_service.classify(request)

        assert response.sufficient is True
        assert response.topics is None
        assert knowledge_service_stub.get_stored_topics("wf-suff") is None

    async def test_topics_only_mode(
        self,
        classification_service: ClassificationService,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """TOPICS_ONLY skips sufficiency check."""
        knowledge_service_stub.add_chunks("wf-topics", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)

        request = ClassificationRequest(
            workflow_id="wf-topics",
            assessment_id="assess-001",
            classification_type=ClassificationType.TOPICS_ONLY,
        )
        response = await classification_service.classify(request)

        assert response.sufficient is True
        assert response.topics is not None
        assert len(response.topics.topics) > 0

    # -----------------------------------------------------------------------
    # New integration tests: unified rubric gate behaviour
    # -----------------------------------------------------------------------

    async def test_rubric_misaligned_fires_insufficient_event(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """When probe returns sufficient=True but rubric_fitness=MISALIGNED,
        the service fires the insufficient event (rubric gate)."""
        knowledge_service_stub.add_chunks("wf-misalign", sufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_misaligned_response(),
        ])

        service = _make_service(
            knowledge_service_stub, assessment_config_stub,
            model_broker_stub, decision_audit_stub, event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-misalign",
            assessment_id="assess-001",
        )
        response = await service.classify(request)

        # Rubric gate: overall result is insufficient
        assert response.sufficient is False

        # The insufficient event MUST be published
        insuff_events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.insufficient"
        )
        assert len(insuff_events) == 1
        payload = insuff_events[0]["payload"]
        assert payload["reason_code"] == "MATERIAL_INSUFFICIENT"
        assert "rubric_fitness" in payload

    async def test_material_insufficient_gap_analysis_preserved(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        insufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """When probe returns sufficient=False with gap_analysis,
        the event payload preserves the gap_analysis entries."""
        knowledge_service_stub.add_chunks("wf-gaps", insufficient_chunks)
        assessment_config_stub.set_config("assess-001", default_config)
        gaps = [
            {
                "topic": "Algorithms",
                "current_depth": "surface",
                "required_depth": "deep",
                "gap_description": "Missing algorithm complexity analysis",
                "fillable_by_web": True,
                "confidence": 0.75,
            }
        ]
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_insufficient_response(gap_analysis=gaps),
        ])

        service = _make_service(
            knowledge_service_stub, assessment_config_stub,
            model_broker_stub, decision_audit_stub, event_publisher_stub,
        )

        request = ClassificationRequest(
            workflow_id="wf-gaps",
            assessment_id="assess-001",
        )
        response = await service.classify(request)

        assert response.sufficient is False
        assert len(response.gap_analysis) == 1
        assert response.gap_analysis[0].topic == "Algorithms"

        insuff_events = event_publisher_stub.get_events_for_topic(
            "assessorflow.classification.insufficient"
        )
        assert len(insuff_events) == 1
        payload = insuff_events[0]["payload"]
        assert len(payload["gap_analysis"]) == 1
        assert payload["gap_analysis"][0]["topic"] == "Algorithms"

    async def test_multiple_assessments_isolated(
        self,
        knowledge_service_stub: StubKnowledgeServiceAdapter,
        assessment_config_stub: StubAssessmentConfigAdapter,
        model_broker_stub: StubModelBrokerAdapter,
        decision_audit_stub: StubDecisionAuditAdapter,
        event_publisher_stub: StubEventPublisherAdapter,
        sufficient_chunks: list[ChunkData],
        insufficient_chunks: list[ChunkData],
        default_config: AssessmentConfig,
    ) -> None:
        """Two workflows should not interfere with each other."""
        knowledge_service_stub.add_chunks("wf-A", sufficient_chunks)
        knowledge_service_stub.add_chunks("wf-B", insufficient_chunks)
        assessment_config_stub.set_config("assess-A", default_config)
        assessment_config_stub.set_config("assess-B", default_config)

        # Prober factory is stateless — configure per-task_key sequences
        # Both requests share the same StubModelBrokerAdapter; each
        # invoke_with_tools call pops from the sequence.
        model_broker_stub.set_tool_call_responses("classification.react_sufficiency", [
            _react_sufficient_response(),
            _react_insufficient_response(),
        ])

        service = _make_service(
            knowledge_service_stub, assessment_config_stub,
            model_broker_stub, decision_audit_stub, event_publisher_stub,
        )

        resp_a = await service.classify(
            ClassificationRequest(workflow_id="wf-A", assessment_id="assess-A")
        )
        resp_b = await service.classify(
            ClassificationRequest(workflow_id="wf-B", assessment_id="assess-B")
        )

        assert resp_a.sufficient is True
        assert resp_b.sufficient is False
        assert knowledge_service_stub.get_stored_topics("wf-A") is not None
        assert knowledge_service_stub.get_stored_topics("wf-B") is None
