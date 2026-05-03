from __future__ import annotations

import structlog

from af_shared.models.domain import DecisionLogEntry
from classification_agent.ports.decision_audit_port import DecisionAuditPort

logger = structlog.get_logger(__name__)


class StubDecisionAuditAdapter(DecisionAuditPort):
    """In-memory stub for Decision Audit Service.

    Records all logged decisions for test assertions.
    """

    def __init__(self) -> None:
        self.entries: list[DecisionLogEntry] = []

    # -----------------------------------------------------------------------
    # Test helpers
    # -----------------------------------------------------------------------

    @property
    def decisions(self) -> list[DecisionLogEntry]:
        """All recorded audit decisions."""
        return self.entries

    def get_decisions_for_workflow(self, workflow_id: str) -> list[DecisionLogEntry]:
        """Filter decisions by workflow_id."""
        return [e for e in self.entries if e.workflow_id == workflow_id]

    # -----------------------------------------------------------------------
    # Port implementation
    # -----------------------------------------------------------------------

    async def log_decision(self, entry: DecisionLogEntry) -> None:
        self.entries.append(entry)
        logger.info(
            "stub_audit_decision",
            workflow_id=entry.workflow_id,
            agent_name=entry.agent_name,
            decision_type=entry.decision_type,
        )
