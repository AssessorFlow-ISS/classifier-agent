from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)


class Settings:
    """Application settings, loaded from environment variables.

    Config-driven adapter swap per ADR-42:
      KNOWLEDGE_SERVICE_ADAPTER = stub | grpc
      ASSESSMENT_CONFIG_ADAPTER = grpc | stub
      MODEL_BROKER_ADAPTER = stub | google_ai_studio | vertex_ai
      AUDIT_ADAPTER = stub | postgres  (read by af_shared factory)
      EVENT_PUBLISHER_ADAPTER = stub | pubsub
    """

    def __init__(self) -> None:
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))

        # Adapter selection (default: stub for local dev)
        self.knowledge_service_adapter: str = os.getenv(
            "KNOWLEDGE_SERVICE_ADAPTER", "stub"
        )
        self.assessment_config_adapter: str = os.getenv(
            "ASSESSMENT_CONFIG_ADAPTER", "grpc"
        )
        self.model_broker_adapter: str = os.getenv(
            "MODEL_BROKER_ADAPTER", "stub"
        )
        # AUDIT_ADAPTER is read by af_shared.adapters.factory.get_decision_audit()
        # Not read here — kept for documentation only
        self.event_publisher_adapter: str = os.getenv(
            "EVENT_PUBLISHER_ADAPTER", "stub"
        )

        # Service endpoints (used by real adapters)
        self.knowledge_service_url: str = os.getenv(
            "KNOWLEDGE_SERVICE_URL", "localhost:50051"
        )
        self.assessment_service_url: str = os.getenv(
            "ASSESSMENT_SERVICE_URL", "localhost:50052"
        )
        self.decision_audit_url: str = os.getenv(
            "DECISION_AUDIT_URL", "localhost:50053"
        )
        self.model_broker_url: str = os.getenv(
            "MODEL_BROKER_URL", "localhost:50054"
        )
        self.pubsub_project_id: str = os.getenv(
            "PUBSUB_PROJECT_ID", "assessorflow-local"
        )

        # Submission Service gRPC endpoint (Phase 6C — used by GrpcAssessmentConfigAdapter)
        self.submission_service_grpc_url: str = os.getenv(
            "SUBMISSION_SERVICE_GRPC_URL", "localhost:9001"
        )


settings = Settings()
