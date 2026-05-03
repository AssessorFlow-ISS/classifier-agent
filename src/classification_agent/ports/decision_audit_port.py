from __future__ import annotations

from abc import ABC, abstractmethod

from af_shared.models.domain import DecisionLogEntry


class DecisionAuditPort(ABC):
    """Port for logging agent decisions via Decision Audit Service (L-11, gRPC Section 4).

    All decisions are append-only (Invariant #5). Each entry MUST include
    prompt_version in the format {agent}/{template}@v{version} (ADR-39).
    """

    @abstractmethod
    async def log_decision(self, entry: DecisionLogEntry) -> None:
        """Log an agent decision to the Decision Audit Service (gRPC LogDecision 4.1).

        Fire-and-forget -- failures should be logged but not block the pipeline.
        """
