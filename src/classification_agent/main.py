"""FastAPI entry point for the Classification Agent (#4).

Wires adapters based on environment configuration. In event-driven mode,
subscribes to assessorflow.classification.trigger on startup.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from af_shared.adapters.factory import get_decision_audit, get_tracing
from classification_agent.adapters.assessment_config_stub import StubAssessmentConfigAdapter
from classification_agent.ports.assessment_config_port import AssessmentConfigPort
from classification_agent.adapters.event_publisher_stub import StubEventPublisherAdapter
from classification_agent.adapters.knowledge_service_stub import StubKnowledgeServiceAdapter
from classification_agent.adapters.model_broker_stub import StubModelBrokerAdapter
from classification_agent.api.routes import router
from classification_agent.api.schemas import ClassificationRequest
from classification_agent.config import settings
from classification_agent.domain.services import ClassificationService
from classification_agent.domain.topic_extractor import TopicExtractor

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger(__name__)


_PROD_ENVS = {"prod", "production", "smoke", "staging"}


def _guard_prod_against_stubs() -> None:
    """Refuse to start with stub adapters when ENV is prod/smoke/staging.

    Hexagonal swap is for local dev. A misconfigured PROD deploy that
    silently runs on stubs is worse than a loud boot-time crash.
    """
    env = os.environ.get("ENV", "dev").lower()
    if env not in _PROD_ENVS:
        return
    stub_adapters: list[str] = []
    if settings.knowledge_service_adapter not in ("http", "real"):
        stub_adapters.append("KNOWLEDGE_SERVICE_ADAPTER")
    if settings.assessment_config_adapter not in ("grpc", "real"):
        stub_adapters.append("ASSESSMENT_CONFIG_ADAPTER")
    if settings.model_broker_adapter not in ("http", "google_ai_studio", "vertex_ai"):
        stub_adapters.append("MODEL_BROKER_ADAPTER")
    if settings.event_publisher_adapter not in ("pubsub", "real", "emulator"):
        stub_adapters.append("EVENT_PUBLISHER_ADAPTER")
    if stub_adapters:
        raise RuntimeError(
            f"Refusing to boot in {env} with stub adapters: {', '.join(stub_adapters)}. "
            "Set each to a non-stub value or unset ENV to dev."
        )


def _build_service() -> tuple[ClassificationService, Any]:
    """Wire up the ClassificationService. Returns (service, event_publisher)."""
    _guard_prod_against_stubs()

    if settings.knowledge_service_adapter in ("http", "real"):
        from classification_agent.adapters.knowledge_service_http import KnowledgeServiceHttpAdapter
        knowledge_service = KnowledgeServiceHttpAdapter()
        logger.info("using_http_knowledge_service")
    else:
        knowledge_service = StubKnowledgeServiceAdapter()

    # Assessment Config — default: grpc (talks to Submission Service), fallback: stub for tests
    assessment_config: AssessmentConfigPort
    if settings.assessment_config_adapter in ("grpc", "real"):
        from classification_agent.adapters.assessment_config_grpc import GrpcAssessmentConfigAdapter
        assessment_config = GrpcAssessmentConfigAdapter()
        logger.info("using_grpc_assessment_config")
    else:
        assessment_config = StubAssessmentConfigAdapter()
        logger.info("using_stub_assessment_config")

    decision_audit = get_decision_audit()

    # Model Broker
    if settings.model_broker_adapter in ("http", "google_ai_studio", "vertex_ai"):
        from classification_agent.adapters.model_broker_http import ModelBrokerHttpAdapter
        model_broker = ModelBrokerHttpAdapter()
        logger.info("using_real_model_broker")
    else:
        model_broker = StubModelBrokerAdapter()

    # Event Publisher
    if settings.event_publisher_adapter in ("pubsub", "real", "emulator"):
        from classification_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter
        event_publisher = PubSubEventPublisherAdapter()
        logger.info("using_real_pubsub_adapter")
    else:
        event_publisher = StubEventPublisherAdapter()

    topic_extractor = TopicExtractor(model_broker=model_broker)

    # Unified ReAct prober factory (Phase 4 — sufficiency + rubric in one session)
    from classification_agent.tools.registry import build_react_prober_factory

    react_prober_factory = build_react_prober_factory(
        model_broker=model_broker,
        knowledge_service=knowledge_service,
    )

    # Tracing adapter (Langfuse -- Walfa implements real adapter)
    tracing = get_tracing()

    service = ClassificationService(
        knowledge_service=knowledge_service,
        assessment_config=assessment_config,
        topic_extractor=topic_extractor,
        decision_audit=decision_audit,
        event_publisher=event_publisher,
        react_prober_factory=react_prober_factory,
        tracing=tracing,
    )
    return service, event_publisher


def create_app() -> FastAPI:
    service, event_publisher = _build_service()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        if settings.event_publisher_adapter in ("pubsub", "real", "emulator"):
            from classification_agent.adapters.pubsub_event_publisher import PubSubEventPublisherAdapter
            if isinstance(event_publisher, PubSubEventPublisherAdapter):
                async def handle_trigger(payload: dict) -> None:
                    logger.info("classification_trigger_received", workflow_id=payload.get("workflow_id"))
                    os.environ["CURRENT_WORKFLOW_ID"] = payload.get("workflow_id", "unknown")
                    from classification_agent.api.schemas import ClassificationType
                    ct_raw = payload.get("classification_type", "sufficiency_and_topics")
                    try:
                        ct = ClassificationType(ct_raw)
                    except ValueError:
                        ct = ClassificationType.SUFFICIENCY_AND_TOPICS
                    request = ClassificationRequest(
                        workflow_id=payload.get("workflow_id", "unknown"),
                        assessment_id=payload.get("assessment_id", "unknown"),
                        assessor_id=payload.get("assessor_id"),
                        classification_type=ct,
                        chunks=payload.get("chunks"),
                    )
                    # Service handles classification + internal Pub/Sub publish
                    response = await service.classify(request)
                    logger.info("classification_complete_published", workflow_id=request.workflow_id, sufficient=response.sufficient)

                await event_publisher.subscribe_and_process("assessorflow.classification.trigger.sub", handle_trigger)
                logger.info("classification_listening", topic="assessorflow.classification.trigger")
        yield

    application = FastAPI(
        title="Classification Agent (#4)",
        description="Material sufficiency + topic extraction",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    application.state.classification_service = service
    application.include_router(router)

    @application.post("/trigger")
    async def trigger(body: dict) -> dict:
        """HTTP trigger — same as Pub/Sub handler, returns completion payload."""
        from classification_agent.api.schemas import ClassificationType

        os.environ["CURRENT_WORKFLOW_ID"] = body.get("workflow_id", "unknown")
        ct_raw = body.get("classification_type", "sufficiency_and_topics")
        try:
            ct = ClassificationType(ct_raw)
        except ValueError:
            ct = ClassificationType.SUFFICIENCY_AND_TOPICS
        request = ClassificationRequest(
            workflow_id=body.get("workflow_id", "unknown"),
            assessment_id=body.get("assessment_id", "unknown"),
            assessor_id=body.get("assessor_id"),
            classification_type=ct,
            chunks=body.get("chunks"),
        )
        response = await service.classify(request)
        return response.model_dump()

    return application


app = create_app()
