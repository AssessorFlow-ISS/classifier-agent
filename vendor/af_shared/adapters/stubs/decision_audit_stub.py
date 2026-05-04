"""In-memory stub for the Decision Audit Service (L-11 gRPC, Section 4)."""

from __future__ import annotations

from af_shared.models.domain import DecisionLogEntry


class StubDecisionAuditAdapter:
    """Records all logged decisions for test assertions.

    Duck-types ``classification_agent.ports.decision_audit_port.DecisionAuditPort``
    via the matching ``async log_decision`` signature. The classifier port
    cannot be subclassed from this shim without a circular import.
    """

    def __init__(self) -> None:
        self.entries: list[DecisionLogEntry] = []

    @property
    def decisions(self) -> list[DecisionLogEntry]:
        """Alias for ``entries`` — matches the classifier-side stub API."""
        return self.entries

    def get_decisions_for_workflow(self, workflow_id: str) -> list[DecisionLogEntry]:
        return [e for e in self.entries if e.workflow_id == workflow_id]

    async def log_decision(self, entry: DecisionLogEntry) -> None:
        self.entries.append(entry)
