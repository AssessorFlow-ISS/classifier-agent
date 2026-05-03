from __future__ import annotations

import json as _json
import os

import structlog

logger = structlog.get_logger(__name__)

_STAGE_MAP = {
    "assessorflow.classification.sufficiency-complete": 1,
    "assessorflow.classification.topic-extraction-complete": 2,
    "assessorflow.classification.rubric-fitness-complete": 3,
}


class ProgressEmitter:
    """Writes per-stage progress rows to ``workflow_events`` for live UI sub-cards.

    Lives outside ``ClassificationService`` so the orchestrator file does not
    own a side-channel write to the Orchestrator DB. Conftest hermeticity is
    preserved by patching ``asyncpg.connect`` upstream — this class still
    routes through the same module-level ``asyncpg`` import.
    """

    AGENT_NAME = "classification-agent"

    async def emit(
        self,
        workflow_id: str,
        event_type: str,
        summary: str,
    ) -> None:
        stage = _STAGE_MAP.get(event_type, 0)
        payload = _json.dumps({"pipeline_group": "classification-pipeline", "stage": stage})
        try:
            import asyncpg
            host = os.getenv("ORCHESTRATOR_DB_HOST", "localhost")
            port = os.getenv("ORCHESTRATOR_DB_PORT", "15432")
            name = os.getenv("ORCHESTRATOR_DB_NAME", "af_orchestrator")
            user = os.getenv("ORCHESTRATOR_DB_USER", "assessorflow")
            password = os.getenv("ORCHESTRATOR_DB_PASSWORD", "dev_password")
            dsn = f"postgresql://{user}:{password}@{host}:{port}/{name}"
            conn = await asyncpg.connect(dsn)
            try:
                await conn.execute(
                    """INSERT INTO workflow_events (workflow_id, event_type, source_agent, summary, payload)
                       VALUES ($1, $2, 'classification-agent', $3, $4::jsonb)""",
                    workflow_id, event_type, summary, payload,
                )
            finally:
                await conn.close()
        except Exception:
            logger.warning(
                "progress_event_write_failed",
                workflow_id=workflow_id,
                event_type=event_type,
                exc_info=True,
            )
