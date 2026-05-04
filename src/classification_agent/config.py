from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)


def _require(name: str) -> str:
    """Return env var; raise if missing or empty.

    No localhost defaults anywhere — caller MUST set the env in every
    environment (dev / smoke / staging / prod). Local-dev convenience
    lives in `.env` files or shell exports, NOT in source code defaults.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} env var is required (no source-code default)")
    return value


class Settings:
    """Application settings, loaded from environment variables.

    Config-driven adapter swap per ADR-42:
      KNOWLEDGE_SERVICE_ADAPTER = stub | grpc
      ASSESSMENT_CONFIG_ADAPTER = grpc | stub
      MODEL_BROKER_ADAPTER = stub | google_ai_studio | vertex_ai
      AUDIT_ADAPTER = stub | postgres  (read by af_shared factory)
      EVENT_PUBLISHER_ADAPTER = stub | pubsub

    All service URLs are REQUIRED env vars. There are no source-code defaults
    pointing at localhost — a misconfigured deploy raises at boot rather than
    silently routing traffic to a non-existent local socket.
    """

    def __init__(self) -> None:
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))

        # Adapter selection — defaults to stub but main.py refuses to boot
        # in PROD if any of these are stub (see _guard_prod_against_stubs).
        self.knowledge_service_adapter: str = os.getenv("KNOWLEDGE_SERVICE_ADAPTER", "stub")
        self.assessment_config_adapter: str = os.getenv("ASSESSMENT_CONFIG_ADAPTER", "grpc")
        self.model_broker_adapter: str = os.getenv("MODEL_BROKER_ADAPTER", "stub")
        # AUDIT_ADAPTER is read by af_shared.adapters.factory.get_decision_audit().
        self.event_publisher_adapter: str = os.getenv("EVENT_PUBLISHER_ADAPTER", "stub")

        # Service endpoints — REQUIRED. Stub adapters don't read these so
        # local dev with all adapters set to "stub" doesn't need to set them.
        # Real adapters (http / grpc / pubsub) require them.
        self._service_urls_loaded: dict[str, str] = {}

    def _service_url(self, env_name: str) -> str:
        """Lazy-load a service URL — required env, no default."""
        if env_name not in self._service_urls_loaded:
            self._service_urls_loaded[env_name] = _require(env_name)
        return self._service_urls_loaded[env_name]

    @property
    def knowledge_service_url(self) -> str:
        return self._service_url("KNOWLEDGE_SERVICE_URL")

    @property
    def assessment_service_url(self) -> str:
        return self._service_url("ASSESSMENT_SERVICE_URL")

    @property
    def decision_audit_url(self) -> str:
        return self._service_url("DECISION_AUDIT_URL")

    @property
    def model_broker_url(self) -> str:
        return self._service_url("MODEL_BROKER_URL")

    @property
    def submission_service_grpc_url(self) -> str:
        return self._service_url("SUBMISSION_SERVICE_GRPC_URL")

    @property
    def pubsub_project_id(self) -> str:
        return _require("PUBSUB_PROJECT_ID")


settings = Settings()
