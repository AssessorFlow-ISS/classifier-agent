from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.adapters.decision_audit_stub import StubDecisionAuditAdapter
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.schemas import (
    AssessmentConfig,
    ChunkData,
    DifficultyLevel,
    SourceType,
)
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor
from classification_agent.main import create_app
from classification_agent.tools.registry import build_react_prober_factory


def _make_app_with_stubs(
    knowledge_stub: StubKnowledgeServiceAdapter,
    config_stub: StubAssessmentConfigAdapter,
    model_stub: StubModelBrokerAdapter,
    audit_stub: StubDecisionAuditAdapter,
    event_stub: StubEventPublisherAdapter,
):
    """Build a FastAPI app wired with the given stubs."""
    from fastapi import FastAPI
    from classification_agent.api.routes import router

    react_prober_factory = build_react_prober_factory(
        model_broker=model_stub,
        knowledge_service=knowledge_stub,
    )
    app = FastAPI()
    app.state.classification_service = ClassificationService(
        knowledge_service=knowledge_stub,
        assessment_config=config_stub,
        topic_extractor=TopicExtractor(model_broker=model_stub),
        decision_audit=audit_stub,
        event_publisher=event_stub,
        react_prober_factory=react_prober_factory,
    )
    app.include_router(router)
    return app


def _sufficient_chunks(workflow_id: str = "wf-test") -> list[ChunkData]:
    return [
        ChunkData(
            chunk_id=f"chunk-{i:03d}",
            workflow_id=workflow_id,
            content=f"Detailed content about computer science topic {i}",
            source_type=SourceType.DIRECT_TEXT,
        )
        for i in range(25)
    ]


def _insufficient_chunks(workflow_id: str = "wf-sparse") -> list[ChunkData]:
    return [
        ChunkData(
            chunk_id="chunk-101",
            workflow_id=workflow_id,
            content="A brief introduction to programming.",
            source_type=SourceType.DIRECT_TEXT,
        ),
    ]


class TestClassificationAPI:
    """Tests for the FastAPI endpoints."""

    async def test_invoke_sufficient(self) -> None:
        ks = StubKnowledgeServiceAdapter()
        cs = StubAssessmentConfigAdapter()
        mb = StubModelBrokerAdapter()
        da = StubDecisionAuditAdapter()
        ep = StubEventPublisherAdapter()

        ks.add_chunks("wf-test", _sufficient_chunks())
        cs.set_config(
            "assess-001",
            AssessmentConfig(
                assessment_id="assess-001",
                assessment_title="CS 101",
                structured_question_count=10,
                non_structured_question_count=5,
                difficulty_level=DifficultyLevel.MEDIUM,
            ),
        )
        mb.set_tool_call_responses("classification.react_sufficiency", [
            {
                "tool_calls": [],
                "content": {
                    "sufficient": True,
                    "reason": "Material sufficient",
                    "gap_analysis": [],
                    "search_queries": [],
                    "autonomy_exercised": False,
                    "rubric_fitness": "NO_RUBRIC",
                    "rubric_reasoning": "",
                    "rubric_source": "none",
                },
            }
        ])

        app = _make_app_with_stubs(ks, cs, mb, da, ep)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/invoke",
                json={
                    "workflow_id": "wf-test",
                    "assessment_id": "assess-001",
                    "classification_type": "sufficiency_and_topics",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["sufficient"] is True
        assert body["topics"] is not None
        assert body["gap_analysis"] == []

    async def test_invoke_insufficient(self) -> None:
        ks = StubKnowledgeServiceAdapter()
        cs = StubAssessmentConfigAdapter()
        mb = StubModelBrokerAdapter()
        da = StubDecisionAuditAdapter()
        ep = StubEventPublisherAdapter()

        ks.add_chunks("wf-sparse", _insufficient_chunks())
        mb.set_tool_call_responses("classification.react_sufficiency", [
            {
                "tool_calls": [],
                "content": {
                    "sufficient": False,
                    "reason": "Insufficient material",
                    "gap_analysis": [
                        {
                            "topic": "Data Structures",
                            "current_depth": "surface",
                            "required_depth": "moderate",
                            "gap_description": "Need more content",
                            "fillable_by_web": True,
                            "confidence": 0.8,
                        }
                    ],
                    "search_queries": [],
                    "autonomy_exercised": False,
                    "rubric_fitness": "NO_RUBRIC",
                    "rubric_reasoning": "",
                    "rubric_source": "none",
                },
            }
        ])

        app = _make_app_with_stubs(ks, cs, mb, da, ep)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/invoke",
                json={
                    "workflow_id": "wf-sparse",
                    "assessment_id": "assess-002",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["sufficient"] is False
        assert body["topics"] is None
        assert len(body["gap_analysis"]) > 0

    async def test_invoke_missing_fields_returns_422(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/invoke", json={})

        assert resp.status_code == 422

    async def test_invoke_missing_assessment_id_returns_422(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/invoke",
                json={"workflow_id": "wf-test"},
            )

        assert resp.status_code == 422

    async def test_health_endpoint(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "classification-agent"

    async def test_ready_endpoint(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ready")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "classification-agent"

    async def test_invoke_invalid_classification_type_returns_422(self) -> None:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/invoke",
                json={
                    "workflow_id": "wf-test",
                    "assessment_id": "assess-001",
                    "classification_type": "invalid_type",
                },
            )

        assert resp.status_code == 422
